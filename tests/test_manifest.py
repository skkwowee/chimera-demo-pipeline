"""Manifest safety: loud load(), shrink-proof push(), tolerant from_json."""

import json

import pytest
from huggingface_hub.utils import EntryNotFoundError

from pipeline.manifest import (
    Manifest,
    ManifestEntry,
    TickSequenceManifestEntry,
    FailureEntry,
    record_failure,
    should_skip_failed,
    MAX_FAILURE_ATTEMPTS,
)


def make_entry(match_id=1, **kw) -> ManifestEntry:
    base = dict(match_id=match_id, slug=f"s{match_id}", team1="a", team2="b",
                event="ev", score="2 - 0", stars=3, date_unix=None,
                demo_files=[f"demos/{match_id}/x.dem"], bytes_total=10)
    base.update(kw)
    return ManifestEntry(**base)


class FakeApi:
    """Stub HfApi: hf_hub_download serves a scripted manifest body (or raises);
    upload_file records pushes."""

    def __init__(self, body: str | None = None, error: Exception | None = None):
        self.body = body
        self.error = error
        self.uploaded: list[bytes] = []
        self._tmp = None

    def hf_hub_download(self, repo_id, filename, repo_type, **kw):
        if self.error is not None:
            raise self.error
        import tempfile
        f = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
        f.write(self.body or "")
        f.close()
        self._tmp = f.name
        return f.name

    def upload_file(self, path_or_fileobj, path_in_repo, repo_id, repo_type,
                    commit_message=None):
        self.uploaded.append(path_or_fileobj)


class TestLoad:
    def test_missing_manifest_is_fresh_start(self):
        api = FakeApi(error=EntryNotFoundError("no such file"))
        mf = Manifest(api, "repo")
        mf.load()
        assert mf.entries == []
        assert mf.loaded_ok

    def test_transient_error_raises_not_fresh_start(self):
        api = FakeApi(error=ConnectionError("network down"))
        mf = Manifest(api, "repo")
        with pytest.raises(ConnectionError):
            mf.load()
        assert not mf.loaded_ok

    def test_loads_existing_entries(self):
        body = "\n".join(make_entry(i).to_json() for i in range(3)) + "\n"
        mf = Manifest(FakeApi(body=body), "repo")
        mf.load()
        assert len(mf.entries) == 3
        assert mf.has(2) and not mf.has(9)


class TestPushShrinkGuard:
    def test_refuses_to_shrink_remote(self):
        remote = "\n".join(make_entry(i).to_json() for i in range(30)) + "\n"
        api = FakeApi(body=remote)
        mf = Manifest(api, "repo")
        mf.add(make_entry(100))
        mf.add(make_entry(101))
        with pytest.raises(RuntimeError, match="refusing to shrink"):
            mf.push()
        assert api.uploaded == []

    def test_force_overrides(self):
        remote = "\n".join(make_entry(i).to_json() for i in range(30)) + "\n"
        api = FakeApi(body=remote)
        mf = Manifest(api, "repo")
        mf.add(make_entry(100))
        mf.push(force=True)
        assert len(api.uploaded) == 1

    def test_push_ok_when_growing(self):
        remote = make_entry(1).to_json() + "\n"
        api = FakeApi(body=remote)
        mf = Manifest(api, "repo")
        mf.add(make_entry(1))
        mf.add(make_entry(2))
        mf.push()
        assert len(api.uploaded) == 1

    def test_push_ok_when_no_remote(self):
        api = FakeApi(error=EntryNotFoundError("none"))
        mf = Manifest(api, "repo")
        mf.add(make_entry(1))
        mf.push()
        assert len(api.uploaded) == 1


class TestFromJsonCompat:
    def test_old_line_without_new_fields(self):
        old = {"match_id": 5, "slug": "s", "team1": "a", "team2": "b",
               "event": "e", "score": "2 - 1", "stars": 3, "date_unix": None,
               "demo_files": ["demos/x.dem"], "bytes_total": 1,
               "processed_at_unix": 1}
        e = ManifestEntry.from_json(json.dumps(old))
        assert e.demo_url is None and e.demo_id is None and e.hltv_maps == []

    def test_unknown_keys_dropped(self):
        d = make_entry(7).__dict__ | {"future_field": 42}
        e = ManifestEntry.from_json(json.dumps(d))
        assert e.match_id == 7

    def test_old_ts_line_without_provenance(self):
        old = {"match_id": 5, "slug": "s", "team1": "a", "team2": "b",
               "event": "e", "pt_files": ["tick_sequences/5/train.pt"],
               "feature_schema_path": None, "n_rounds": 20,
               "total_ticks": 100, "feature_dim": 597, "downsample": 8,
               "bytes_total": 1, "source_demo_files": [],
               "processed_at_unix": 1}
        e = TickSequenceManifestEntry.from_json(json.dumps(old))
        assert e.schema_version is None
        assert e.parsed_files == [] and e.parsed_bytes == 0

    def test_roundtrip_with_provenance(self):
        e = TickSequenceManifestEntry(
            match_id=1, slug="s", team1="a", team2="b", event="e",
            pt_files=[], feature_schema_path=None, n_rounds=1, total_ticks=1,
            feature_dim=597, downsample=8, bytes_total=1,
            source_demo_files=[], schema_version="feature_schema_v2.1",
            builder_commit="abc1234", builder_dirty=False,
            pipeline_commit="def5678", awpy_version="2.0.2",
            demoparser2_version="0.41.3",
            parsed_files=["parsed/1/x_ticks.parquet"], parsed_bytes=99)
        e2 = TickSequenceManifestEntry.from_json(e.to_json())
        assert e2 == e


class TestFailures:
    def test_record_and_skip_after_max_attempts(self):
        api = FakeApi(error=EntryNotFoundError("none"))
        mf = Manifest(api, "repo", path="failures.jsonl",
                      entry_cls=FailureEntry)
        mf.load()
        for _ in range(MAX_FAILURE_ATTEMPTS):
            record_failure(mf, 42, "parse", "boom")
        assert len(mf.entries) == 1
        assert mf.entries[0].attempts == MAX_FAILURE_ATTEMPTS
        assert should_skip_failed(mf, 42, retry_failed=False)
        assert not should_skip_failed(mf, 42, retry_failed=True)
        assert not should_skip_failed(mf, 43, retry_failed=False)

    def test_below_threshold_not_skipped(self):
        api = FakeApi(error=EntryNotFoundError("none"))
        mf = Manifest(api, "repo", path="failures.jsonl",
                      entry_cls=FailureEntry)
        mf.load()
        record_failure(mf, 42, "download", "boom")
        assert not should_skip_failed(mf, 42, retry_failed=False)
