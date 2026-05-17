# chimera-demo-pipeline

Auto-ingest CS2 pro demos from HLTV into the `skkwowee/chimera-cs2` HuggingFace dataset.

**Goal**: scale demo collection without local persistence. Each match is downloaded to a tempdir, extracted, uploaded to HF, and cleaned up. Resumable across crashes/restarts via on-HF manifest.

## Install

```bash
git clone <this repo>
cd chimera-demo-pipeline
python3 -m venv .venv && source .venv/bin/activate
pip install -e .

# System dep — needed for extracting .rar archives from HLTV.
# `unar` is recommended: GPL, full RAR5 support, in standard repos.
# (Don't use `unrar-free` — incomplete RAR5 support, may silently fail.)
sudo apt install unar         # Linux (preferred)
# or:
sudo apt install p7zip-full   # Linux (alternative; auto-detected as fallback)
brew install unar             # macOS

# HF auth (needs write access to the target dataset repo)
huggingface-cli login
```

## Usage

```bash
# Scrape (dry, prints listing only)
chimera-demo scrape --stars 5 --max-matches 10

# Get one match's demo URL
chimera-demo fetch-match 2394156 spirit-vs-falcons-pgl-astana-2026

# Full pipeline — scrape + download + extract + upload + manifest
chimera-demo run --stars 3 --max-matches 50 --repo skkwowee/chimera-cs2

# Resume — auto-skips already-processed matches (manifest on HF is truth source)
chimera-demo run --stars 3 --max-matches 50

# Inspect what's already been processed
chimera-demo manifest --show-last 20
```

`stars` is HLTV's match-tier filter (1 = all, 5 = LAN majors only). For training data, stars=3 strikes a good balance between volume and quality.

## How it works

1. **Scrape** HLTV results listings (`/results?stars=N&offset=K`) — paginated, 100 matches per page.
2. **Skip** any match whose `match_id` already appears in `processed_manifest.jsonl` on HF.
3. **Fetch match page** to extract the `/download/demo/<id>` URL.
4. **Stream-download** the `.rar` (multi-map series → one rar containing N `.dem` files) to `tempfile.TemporaryDirectory()`.
5. **Extract** `.dem` files (`rarfile` + system `unrar`).
6. **Rename** to canonical `team1-vs-team2-mN-map.dem` form (matches existing demos in the dataset).
7. **Upload** all `.dem`s for the match in a single atomic HF commit.
8. **Append** one `ManifestEntry` JSON line to `processed_manifest.jsonl` (per-match commit) so reruns skip this match.
9. **Cleanup**: `TemporaryDirectory` exit removes everything from local disk.

Peak local storage during a run = **1 match worth of files** (~500 MB to 2 GB). No persistent state.

## Why curl_cffi instead of cloudscraper

HLTV uses Cloudflare's WAF which fingerprints the TLS handshake (JA3 / JA4), not just JS challenges. `cloudscraper` uses Python's `ssl` module → distinctive fingerprint → 403. `curl_cffi` ships pre-compiled Chrome/Firefox curl binaries with their real TLS fingerprints → indistinguishable from a real browser → 200.

## Architecture for cloud-only runs (RunPod, GCE, etc.)

Same `chimera-demo run` works on any worker with HF auth. To run on RunPod:

```bash
# pod startup
apt update && apt install -y unar git python3-venv
git clone <this repo> && cd chimera-demo-pipeline
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
export HF_TOKEN=<token>
chimera-demo run --stars 3 --max-matches 200
```

Add a cron job or systemd timer to run hourly / daily for continuous ingest.

## Rate limiting + politeness

Default 5s between HLTV requests. HLTV doesn't publish a robots/scraping policy but heavy scrapers do get IP-banned — keep `--max-matches` modest per run (50-200) and run from rotating IPs if scaling beyond ~1k/day.

## Files

```
pipeline/
├── hltv.py         # scrape match listings + match pages
├── download.py     # fetch + extract .rar → .dem files
├── upload.py       # push to HF dataset (atomic per-match commit)
├── manifest.py     # processed-match tracking on HF
└── cli.py          # `chimera-demo` entry point
```
