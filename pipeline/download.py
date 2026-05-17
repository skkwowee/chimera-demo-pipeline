"""Demo fetch + extract: HLTV demo URL → .rar → .dem files in /tmp.

The .rar contains 1 .dem per played map (best-of-3 → up to 3 .dem files).
This module ONLY handles the binary transport; naming + manifest are in
upload.py / manifest.py.

Storage discipline (chimera-demo-pipeline goal: ZERO persistent local):
    All downloads + extracts go to a caller-supplied tmpdir. Caller is
    responsible for cleanup (typically `tempfile.TemporaryDirectory`).
"""

from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

import rarfile
from curl_cffi import requests as cffi_requests

from .hltv import DEFAULT_HEADERS, DEFAULT_IMPERSONATE

# Bytes per network read — 1 MB keeps memory small for big multi-map .rar files
CHUNK = 1 << 20

# Map common HLTV demo aliases to canonical map names used in chimera
MAP_NORM = {
    "de_mirage": "mirage",
    "de_inferno": "inferno",
    "de_nuke": "nuke",
    "de_ancient": "ancient",
    "de_anubis": "anubis",
    "de_vertigo": "vertigo",
    "de_dust2": "dust2",
    "de_overpass": "overpass",
    "de_train": "train",
}


def _new_scraper() -> cffi_requests.Session:
    s = cffi_requests.Session(impersonate=DEFAULT_IMPERSONATE)
    s.headers.update(DEFAULT_HEADERS)
    return s


def download_rar(demo_url: str, dest: Path,
                 scraper: cffi_requests.Session | None = None,
                 max_retries: int = 3) -> Path:
    """Stream-download .rar to `dest`. Returns dest on success."""
    s = scraper or _new_scraper()
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            # curl_cffi's stream mode: pass stream=True, iterate with iter_content
            r = s.get(demo_url, stream=True, timeout=300)
            try:
                r.raise_for_status()
                size = int(r.headers.get("Content-Length", 0))
                tmp = dest.with_suffix(dest.suffix + ".part")
                got = 0
                t0 = time.time()
                with tmp.open("wb") as f:
                    for chunk in r.iter_content(chunk_size=CHUNK):
                        if chunk:
                            f.write(chunk)
                            got += len(chunk)
                tmp.rename(dest)
                dt = time.time() - t0
                rate = got / max(dt, 1e-6) / 1e6
                print(f"  downloaded {got/1e6:.0f} MB in {dt:.0f}s ({rate:.1f} MB/s) "
                      f"{'(expected {:.0f} MB)'.format(size/1e6) if size else ''}",
                      flush=True)
                return dest
            finally:
                r.close()
        except Exception as e:
            last_err = e
            print(f"  download error (attempt {attempt+1}): {e}", flush=True)
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"download_rar failed: {demo_url} ({last_err})")


# RAR extractor backends, in preference order:
#   unar    — GPL'd The Unarchiver; full RAR3 + RAR5; cleanest default
#   7z      — p7zip-full; handles RAR5 via the rar codec; often preinstalled
#   unrar   — RARLAB official (Debian non-free); fastest but proprietary
# We do NOT prefer `unrar-free` — it has incomplete RAR5 support and silently
# truncates many modern WinRAR archives, including some HLTV demo .rars.
RAR_TOOLS_PREFERENCE = [
    ("unar", "unar"),
    ("7z", "7zip"),
    ("unrar", "unrar"),
]


def _detect_rar_tool() -> str:
    """Return the rarfile UNRAR_TOOL name for the first available extractor."""
    for binary, _ in RAR_TOOLS_PREFERENCE:
        if shutil.which(binary):
            return binary
    raise RuntimeError(
        "No RAR extractor found. Install one (in order of preference):\n"
        "  apt install unar          # Linux, recommended (GPL, full RAR5)\n"
        "  apt install p7zip-full    # Linux, alternative (also handles RAR5)\n"
        "  apt install unrar         # Linux non-free (RARLAB, fastest)\n"
        "  brew install unar         # macOS\n"
        "Skip `unrar-free` — incomplete RAR5 support."
    )


def extract_dems(rar_path: Path, dest_dir: Path) -> list[Path]:
    """Extract all .dem from rar_path into dest_dir. Returns list of .dem paths."""
    tool = _detect_rar_tool()
    rarfile.UNRAR_TOOL = tool
    dest_dir.mkdir(parents=True, exist_ok=True)
    with rarfile.RarFile(str(rar_path)) as rf:
        members = [m for m in rf.namelist() if m.lower().endswith(".dem")]
        if not members:
            raise RuntimeError(f"no .dem files in {rar_path}")
        rf.extractall(path=str(dest_dir), members=members)
    return sorted(dest_dir.glob("*.dem"))


def normalize_demo_name(orig_path: Path, team1: str, team2: str,
                         match_id: int, map_index: int) -> str:
    """Build the canonical name used in chimera/data/demos/*.dem.

    Examples:
        furia-vs-vitality-m1-mirage.dem
        spirit-vs-natus-vincere-m2-dust2.dem

    HLTV often packs demos as `<id>-de_mirage.dem` etc. We extract the map
    name and rebuild around it. If a map name is not recognizable, we keep
    the original stem with `_unknown` suffix.
    """
    stem = orig_path.stem.lower()
    map_name = "unknown"
    for token, canon in MAP_NORM.items():
        if token in stem:
            map_name = canon
            break
    t1 = _slug(team1)
    t2 = _slug(team2)
    return f"{t1}-vs-{t2}-m{map_index}-{map_name}.dem"


def _slug(s: str) -> str:
    return (s.lower()
              .replace(" ", "-")
              .replace(".", "")
              .replace("'", "")
              .replace("/", "-"))
