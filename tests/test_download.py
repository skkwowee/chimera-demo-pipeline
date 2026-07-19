"""download.py: naming, series ordering, download integrity."""

from pathlib import Path

import pytest

from pipeline.download import (
    detect_map_name,
    download_rar,
    normalize_demo_name,
    order_dems_for_series,
)


class TestDetectMapName:
    def test_canonical_new_format(self):
        assert detect_map_name("furia-vs-vitality-m1-mirage") == "mirage"

    def test_old_id_de_format(self):
        assert detect_map_name("87421-de_mirage") == "mirage"

    def test_event_underscore_format(self):
        assert detect_map_name("blast_de_inferno_gotv") == "inferno"

    def test_bare_map_with_boundaries(self):
        assert detect_map_name("87421-dust2") == "dust2"

    def test_unknown(self):
        assert detect_map_name("some-showmatch-thing") == "unknown"


class TestNormalizeDemoName:
    def test_builds_canonical_name(self):
        name = normalize_demo_name(Path("87421-de_nuke.dem"),
                                   "Natus Vincere", "Vitality", 2394174, 2)
        assert name == "natus-vincere-vs-vitality-m2-nuke.dem"


class TestOrderDemsForSeries:
    def test_old_format_ordered_by_match_page_maps(self):
        dems = [Path("87421-de_inferno.dem"),   # sorted-name order != series
                Path("87421-de_mirage.dem"),
                Path("87421-de_nuke.dem")]
        ordered = order_dems_for_series(dems, ["nuke", "mirage", "inferno"])
        assert [detect_map_name(d.stem) for d in ordered] == \
            ["nuke", "mirage", "inferno"]

    def test_new_format_with_mn_tokens_untouched(self):
        dems = [Path("a-vs-b-m1-inferno.dem"), Path("a-vs-b-m2-nuke.dem")]
        assert order_dems_for_series(dems, ["nuke", "inferno"]) == dems

    def test_no_map_list_untouched(self):
        dems = [Path("87421-de_inferno.dem"), Path("87421-de_mirage.dem")]
        assert order_dems_for_series(dems, []) == dems

    def test_unknown_maps_sort_last(self):
        dems = [Path("87421-weird.dem"), Path("87421-de_mirage.dem")]
        ordered = order_dems_for_series(dems, ["mirage"])
        assert detect_map_name(ordered[0].stem) == "mirage"


class FakeResponse:
    def __init__(self, payload: bytes, content_length: int | None):
        self.payload = payload
        self.headers = ({"Content-Length": str(content_length)}
                        if content_length is not None else {})

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        yield self.payload

    def close(self):
        pass


class FakeSession:
    def __init__(self, payload: bytes, content_length: int | None):
        self.resp = FakeResponse(payload, content_length)

    def get(self, url, stream=True, timeout=None):
        return self.resp


class TestDownloadIntegrity:
    def test_short_read_raises_and_removes_part(self, tmp_path, monkeypatch):
        monkeypatch.setattr("pipeline.download.time.sleep", lambda s: None)
        dest = tmp_path / "x.rar"
        s = FakeSession(b"12345", content_length=100)   # truncated
        with pytest.raises(RuntimeError, match="short read"):
            download_rar("http://example/demo", dest, scraper=s, max_retries=1)
        assert not dest.exists()
        assert not dest.with_suffix(".rar.part").exists()

    def test_full_read_succeeds(self, tmp_path):
        dest = tmp_path / "x.rar"
        s = FakeSession(b"12345", content_length=5)
        out = download_rar("http://example/demo", dest, scraper=s)
        assert out == dest and dest.read_bytes() == b"12345"

    def test_no_content_length_accepts_any_size(self, tmp_path):
        dest = tmp_path / "x.rar"
        s = FakeSession(b"12345", content_length=None)
        assert download_rar("http://example/demo", dest, scraper=s) == dest
