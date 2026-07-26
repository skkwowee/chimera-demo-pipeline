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
                if size and got != size:
                    # Truncated read — renaming .part into place would hand
                    # a corrupt rar to the extractor. Raise so we retry.
                    tmp.unlink(missing_ok=True)
                    raise IOError(
                        f"short read: got {got} of {size} bytes")
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


# Shell out to a RAR extractor. The `rarfile` Python lib fails on many
# HLTV RAR5 archives with "BadRarFile: Failed the read enough data:
# req=65536 got=176"; real extractors have complete RAR5 support.
# Preference order:
#   7z      — p7zip-full; clean RAR5 support; usually pre-installed
#   unar    — The Unarchiver; GPL; full RAR3 + RAR5; lsar to list
#   unrar   — RARLAB official (Debian non-free); fastest
# We do NOT prefer `unrar-free` — incomplete RAR5 support; will silently
# fail on most HLTV demos.
RAR_TOOLS_PREFERENCE = ["7z", "unar", "unrar"]


def _detect_rar_tool() -> str:
    """Return the path of the first available RAR extractor binary."""
    for binary in RAR_TOOLS_PREFERENCE:
        path = shutil.which(binary)
        if not path:
            # RunPod/cron PATHs often omit ~/.local/bin — probe it explicitly.
            local = Path.home() / ".local" / "bin" / binary
            if local.is_file():
                path = str(local)
        if path:
            # Exclude unrar-free, which presents as `unrar` but has broken RAR5.
            if binary == "unrar":
                try:
                    out = subprocess.run([path, "--help"], capture_output=True,
                                         text=True, timeout=5).stdout.lower() \
                          + subprocess.run([path, "--help"], capture_output=True,
                                           text=True, timeout=5).stderr.lower()
                    if "unrar-free" in out:
                        continue   # broken RAR5; skip
                except Exception:
                    pass
            return path
    raise RuntimeError(
        "No working RAR extractor found. Install one:\n"
        "  apt install p7zip-full    # Linux, recommended (clean RAR5)\n"
        "  apt install unar          # Linux/macOS (GPL, full RAR5)\n"
        "  apt install unrar         # Linux non-free (RARLAB)\n"
        "AVOID `unrar-free` — it has incomplete RAR5 support."
    )


def extract_dems(rar_path: Path, dest_dir: Path) -> list[Path]:
    """Extract all .dem from rar_path into dest_dir. Returns list of .dem paths.

    Shells out to 7z / unar / unrar, whichever is found first.
    """
    tool = _detect_rar_tool()
    dest_dir.mkdir(parents=True, exist_ok=True)
    binary = Path(tool).name
    if binary == "7z":
        cmd = [tool, "x", "-y", str(rar_path), f"-o{dest_dir}", "*.dem", "-r"]
    elif binary == "unar":
        cmd = [tool, "-o", str(dest_dir), "-f", str(rar_path)]
    else:  # unrar
        cmd = [tool, "x", "-y", str(rar_path), str(dest_dir) + "/"]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        raise RuntimeError(
            f"{binary} extract failed (rc={proc.returncode}):\n"
            f"  stdout: {proc.stdout[-500:]}\n  stderr: {proc.stderr[-500:]}"
        )
    # Some tools preserve directories — flatten .dem to dest_dir root
    dems = list(dest_dir.rglob("*.dem"))
    if not dems:
        raise RuntimeError(f"no .dem files extracted from {rar_path}")
    flat: list[Path] = []
    for d in dems:
        if d.parent != dest_dir:
            target = dest_dir / d.name
            d.rename(target)
            flat.append(target)
        else:
            flat.append(d)
    return sorted(flat)


def detect_map_name(stem: str) -> str:
    """Map name from a demo file stem, or 'unknown'.

    Source filenames vary:
      - HLTV new format: `team1-vs-team2-m1-dust2.dem` (already canonical)
      - HLTV older:      `<id>-de_mirage.dem` or `<id>-mirage.dem`
      - Some packs:      `<event>_de_inferno_<demo>.dem`
    We scan for both `de_<map>` and bare `<map>` tokens (with word
    boundaries to avoid false matches like "anubis" inside an event name).
    """
    import re as _re
    stem = stem.lower()
    # Prefer de_* match (most specific) over bare name
    for token, canon in MAP_NORM.items():
        if token in stem:
            return canon
    # No de_* match — try bare canonical names with word boundaries.
    # Use longest names first to match `dust2` before `dust`.
    canonical = sorted(set(MAP_NORM.values()), key=len, reverse=True)
    # Use custom boundary — Python's \b treats `_` as a word char so
    # `\binferno\b` won't match `event_inferno_demo`. Match on -, _, or
    # start/end of string.
    for name in canonical:
        if _re.search(rf"(?:^|[-_.]){name}(?:[-_.]|$)", stem):
            return name
    return "unknown"


def order_dems_for_series(dems: list[Path], hltv_maps: list[str]) -> list[Path]:
    """Order extracted .dem files so map_index reflects the SERIES order.

    Old-format archive names (`<id>-de_mirage.dem`) carry no mN token, so
    sorted-filename order can misstate m1/m2/m3. When no filename has an mN
    token and the match page gave us the map order, sort by each demo's map
    position in `hltv_maps` (unknown maps last, original order preserved as
    tie-break). Filenames WITH mN tokens already sort correctly.
    """
    import re as _re
    if not hltv_maps or any(_re.search(r"(?:^|-)m\d+-", d.stem.lower())
                            for d in dems):
        return dems
    order = {m.lower(): i for i, m in enumerate(hltv_maps)}
    return sorted(dems, key=lambda d: (
        order.get(detect_map_name(d.stem), len(order)),))


def normalize_demo_name(orig_path: Path, team1: str, team2: str,
                         match_id: int, map_index: int) -> str:
    """Build the canonical name used in chimera/data/demos/*.dem.

    Examples:
        furia-vs-vitality-m1-mirage.dem
        spirit-vs-natus-vincere-m2-dust2.dem

    Falls back to "unknown" only if no map name appears (see detect_map_name).
    """
    map_name = detect_map_name(orig_path.stem)
    t1 = _slug(team1)
    t2 = _slug(team2)
    return f"{t1}-vs-{t2}-m{map_index}-{map_name}.dem"


def _slug(s: str) -> str:
    return (s.lower()
              .replace(" ", "-")
              .replace(".", "")
              .replace("'", "")
              .replace("/", "-"))
