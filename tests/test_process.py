"""process.py preflight guards + provenance helpers (no network, no HF)."""

import json
from pathlib import Path

import pytest

from pipeline.process import (
    _builder_schema_version,
    _collect_parse_bundle,
    _read_schema_version,
    _vtuple,
    MIN_DEMOPARSER2,
    BUILDER_FIX_MARKER,
)


class TestVtuple:
    def test_parses_versions(self):
        assert _vtuple("0.41.3") == (0, 41, 3)
        assert _vtuple("2.0.2") == (2, 0, 2)

    def test_comparison_against_floor(self):
        assert _vtuple("0.41.3") >= MIN_DEMOPARSER2
        assert _vtuple("0.42.0") >= MIN_DEMOPARSER2
        assert _vtuple("0.40.9") < MIN_DEMOPARSER2
        assert _vtuple("0.38.0") < MIN_DEMOPARSER2


class TestBuilderGuard:
    def _write_builder(self, tmp_path: Path, text: str) -> Path:
        d = tmp_path / "chimera"
        (d / "scripts").mkdir(parents=True)
        (d / "scripts" / "build_tick_sequences.py").write_text(text)
        return d

    def test_prefix_builder_rejected(self, tmp_path):
        d = self._write_builder(
            tmp_path,
            'SCHEMA_VERSION = "feature_schema_v2"\n'
            'site_str = "planted_b" if (bomb_site or "").lower().endswith("b") else "planted_a"\n')
        with pytest.raises(SystemExit, match="site-from-plant-position"):
            _builder_schema_version(d)

    def test_fixed_builder_returns_version(self, tmp_path):
        d = self._write_builder(
            tmp_path,
            'SCHEMA_VERSION = "feature_schema_v2.1"\n'
            f'def {BUILDER_FIX_MARKER}(map_name, x, y, z, s=None):\n'
            '    return "a"\n')
        assert _builder_schema_version(d) == "feature_schema_v2.1"

    def test_real_chimera_builder_passes(self):
        d = Path("/home/soone/chimera")
        if not (d / "scripts" / "build_tick_sequences.py").exists():
            pytest.skip("chimera repo not present")
        assert _builder_schema_version(d) == "feature_schema_v2.1"


class TestSchemaVersionRead:
    def test_reads_canonical_then_legacy_name(self, tmp_path):
        (tmp_path / "feature_schema.json").write_text(
            json.dumps({"version": "feature_schema_v2.1"}))
        assert _read_schema_version(tmp_path) == "feature_schema_v2.1"

    def test_legacy_only(self, tmp_path):
        (tmp_path / "feature_schema_v1.json").write_text(
            json.dumps({"version": "feature_schema_v2"}))
        assert _read_schema_version(tmp_path) == "feature_schema_v2"

    def test_missing(self, tmp_path):
        assert _read_schema_version(tmp_path) is None


class TestCollectParseBundle:
    def test_collects_parquet_and_event_jsons(self, tmp_path):
        stems = ["a-vs-b-m1-mirage", "a-vs-b-m2-nuke"]
        for s in stems:
            (tmp_path / f"{s}_ticks.parquet").write_bytes(b"x")
            for suf in ("kills", "bomb", "damages", "rounds", "header"):
                (tmp_path / f"{s}_{suf}.json").write_text("{}")
        (tmp_path / "unrelated.txt").write_text("no")
        bundle = _collect_parse_bundle(tmp_path)
        assert len(bundle) == 2 * 6
        assert all(p.name != "unrelated.txt" for p in bundle)
        # parquets first (build inputs), then JSONs
        assert bundle[0].name.endswith("_ticks.parquet")
