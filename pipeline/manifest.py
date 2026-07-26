"""Processed-match tracking, stored as JSONL on the HF dataset itself.

Why on HF and not local: the pipeline must be runnable from a fresh RunPod
worker without prior state. Manifest-on-HF makes the *latest commit* the
single source of truth — any worker can resume by downloading the manifest,
filtering, and proceeding.

Layout on HF:
    demos/<match_id>/<t1>-vs-<t2>-m<N>-<map>.dem   # demo files (new, namespaced)
    demos/<t1>-vs-<t2>-m<N>-<map>.dem              # demo files (legacy, flat)
    parsed/<match_id>/*.parquet + *.json           # archived parse intermediates
    processed_manifest.jsonl                       # one JSON line per match
    processed_tick_sequences_manifest.jsonl        # one line per baked match
    failures.jsonl                                 # persistent failure record
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, fields
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError


MANIFEST_PATH = "processed_manifest.jsonl"
TICK_SEQUENCES_MANIFEST_PATH = "processed_tick_sequences_manifest.jsonl"
FAILURES_MANIFEST_PATH = "failures.jsonl"

# Matches with this many recorded failures are skipped unless --retry-failed
MAX_FAILURE_ATTEMPTS = 3


def _from_json_tolerant(cls, s: str):
    """Back/forward-compatible JSONL row -> dataclass: unknown keys are
    dropped, missing keys fall back to field defaults (so old manifest lines
    survive new fields and vice versa)."""
    d = json.loads(s)
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in d.items() if k in names})


@dataclass
class ManifestEntry:
    match_id: int
    slug: str
    team1: str
    team2: str
    event: str
    score: str
    stars: int
    date_unix: int | None
    demo_files: list[str]   # paths in repo, e.g. ["demos/2394174/foo-m1-mirage.dem"]
    bytes_total: int
    processed_at_unix: int = field(default_factory=lambda: int(time.time()))
    # Provenance (added post-2026-07-18 reset; optional for old lines)
    demo_url: str | None = None       # https://www.hltv.org/download/demo/<id>
    demo_id: int | None = None
    hltv_maps: list[str] = field(default_factory=list)  # match-page series order

    def to_json(self) -> str:
        return json.dumps(self.__dict__, separators=(",", ":"))

    @classmethod
    def from_json(cls, s: str) -> "ManifestEntry":
        return _from_json_tolerant(cls, s)


@dataclass
class TickSequenceManifestEntry:
    """One match's tick-sequence outputs uploaded to HF.

    `pt_files` lists the path-in-repo for each `.pt` chunk (one per demo).
    `feature_schema_path` points to the schema JSON we shipped alongside,
    so downstream consumers can verify dim/layout.
    """
    match_id: int
    slug: str
    team1: str
    team2: str
    event: str
    pt_files: list[str]
    feature_schema_path: str | None
    n_rounds: int
    total_ticks: int
    feature_dim: int
    downsample: int
    bytes_total: int
    source_demo_files: list[str]
    processed_at_unix: int = field(default_factory=lambda: int(time.time()))
    # Provenance (added post-2026-07-18 reset; optional for old lines).
    # schema_version is the re-bake selector: run_process re-bakes any match
    # whose recorded version differs from the current builder's version.
    schema_version: str | None = None       # e.g. "feature_schema_v2.1"
    builder_commit: str | None = None       # chimera repo HEAD at bake time
    builder_dirty: bool | None = None       # uncommitted edits to the scripts?
    pipeline_commit: str | None = None      # this repo's HEAD at bake time
    awpy_version: str | None = None
    demoparser2_version: str | None = None
    # Archived parse intermediates at parsed/<match_id>/ (re-bakes skip the
    # expensive awpy parse entirely when these exist)
    parsed_files: list[str] = field(default_factory=list)
    parsed_bytes: int = 0

    def to_json(self) -> str:
        return json.dumps(self.__dict__, separators=(",", ":"))

    @classmethod
    def from_json(cls, s: str) -> "TickSequenceManifestEntry":
        return _from_json_tolerant(cls, s)


@dataclass
class FailureEntry:
    """One match's persistent failure record (failures.jsonl on HF).

    Keeps known-bad matches from being re-downloaded/re-fetched forever:
    runs skip matches with attempts >= MAX_FAILURE_ATTEMPTS unless
    --retry-failed is passed.
    """
    match_id: int
    stage: str      # scrape | download | extract | parse | bake | upload
    reason: str
    attempts: int = 1
    last_attempt_unix: int = field(default_factory=lambda: int(time.time()))

    def to_json(self) -> str:
        return json.dumps(self.__dict__, separators=(",", ":"))

    @classmethod
    def from_json(cls, s: str) -> "FailureEntry":
        return _from_json_tolerant(cls, s)


class Manifest:
    """Read-modify-append the JSONL manifest on the HF dataset.

    Usage: `load()` once at batch start, then `add()` + `push()` after each
    confirmed upload. A manifest line is only added once its artifacts are
    on HF, so a kill mid-batch leaves finished matches consistent.

    `path` keeps multiple parallel manifests on the same repo
    (e.g. processed_manifest.jsonl for demos vs.
    processed_tick_sequences_manifest.jsonl for derived `.pt` chunks).
    """

    def __init__(self, api: HfApi, repo_id: str, repo_type: str = "dataset",
                 path: str = MANIFEST_PATH,
                 entry_cls: type = ManifestEntry):
        self.api = api
        self.repo_id = repo_id
        self.repo_type = repo_type
        self.path = path
        self.entry_cls = entry_cls
        self.entries: list = []
        self.processed_ids: set[int] = set()
        self.loaded_ok: bool = False

    def load(self) -> None:
        """Download manifest from HF if it exists. Populates entries + processed_ids.

        ONLY a genuinely-missing manifest (or repo) is a fresh start. Any
        other error (network, auth, ...) re-raises: swallowing it here used
        to make the next push() silently overwrite the on-HF manifest with
        just this run's entries, destroying the processed-matches record.
        """
        try:
            local = self.api.hf_hub_download(
                repo_id=self.repo_id,
                filename=self.path,
                repo_type=self.repo_type,
            )
        except (EntryNotFoundError, RepositoryNotFoundError):
            # No manifest yet — fresh start
            self.loaded_ok = True
            return
        for line in Path(local).read_text().splitlines():
            if line.strip():
                e = self.entry_cls.from_json(line)
                self.entries.append(e)
                self.processed_ids.add(e.match_id)
        self.loaded_ok = True

    def has(self, match_id: int) -> bool:
        return match_id in self.processed_ids

    def get(self, match_id: int):
        return next((e for e in self.entries if e.match_id == match_id), None)

    def add(self, entry) -> None:
        self.entries.append(entry)
        self.processed_ids.add(entry.match_id)

    def _remote_line_count(self) -> int | None:
        """Line count of the manifest currently on HF (None if absent)."""
        try:
            local = self.api.hf_hub_download(
                repo_id=self.repo_id,
                filename=self.path,
                repo_type=self.repo_type,
                force_download=True,   # the cached copy may be stale
            )
        except (EntryNotFoundError, RepositoryNotFoundError):
            return None
        return sum(1 for line in Path(local).read_text().splitlines()
                   if line.strip())

    def push(self, commit_message: str | None = None,
             force: bool = False) -> None:
        """Upload the FULL manifest (all entries) to HF in one commit.

        Shrink-proof: refuses to overwrite a remote manifest with MORE lines
        than we hold locally (that means load() never saw them — pushing
        would destroy the record). Pass force=True only when a shrink is
        intentional (e.g. deliberately pruning entries).
        """
        if not force:
            remote_n = self._remote_line_count()
            if remote_n is not None and remote_n > len(self.entries):
                raise RuntimeError(
                    f"refusing to shrink {self.path} on {self.repo_id}: "
                    f"remote has {remote_n} lines, local has "
                    f"{len(self.entries)} — load() missed entries? "
                    f"Pass force=True only for an intentional prune.")
        body = "\n".join(e.to_json() for e in self.entries) + "\n"
        self.api.upload_file(
            path_or_fileobj=body.encode("utf-8"),
            path_in_repo=self.path,
            repo_id=self.repo_id,
            repo_type=self.repo_type,
            commit_message=commit_message or f"manifest: {len(self.entries)} entries",
        )


def record_failure(fail_mf: Manifest, match_id: int, stage: str,
                   reason: str) -> FailureEntry:
    """Append/update the failure record for a match (caller pushes)."""
    e = fail_mf.get(match_id)
    if e is not None:
        e.stage = stage
        e.reason = reason
        e.attempts += 1
        e.last_attempt_unix = int(time.time())
        return e
    e = FailureEntry(match_id=match_id, stage=stage, reason=reason)
    fail_mf.add(e)
    return e


def should_skip_failed(fail_mf: Manifest, match_id: int,
                       retry_failed: bool) -> bool:
    """True if this match has exhausted its failure attempts."""
    if retry_failed:
        return False
    e = fail_mf.get(match_id)
    return e is not None and e.attempts >= MAX_FAILURE_ATTEMPTS
