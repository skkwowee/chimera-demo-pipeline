# chimera-demo-pipeline

The data pipeline for the [chimera](https://github.com/skkwowee/chimera) CS2 world model: HLTV → `.dem` → HuggingFace → tick sequences.

This repo ingests pro CS2 demos from HLTV into the `skkwowee/chimera-cs2` HuggingFace dataset and processes them into the per-frame game-state sequences the world model trains on (the k=4 / 500 ms rollout-native distributional world model, 19M params; v2 = 597-d and v3 = 687-d state frames — see chimera `docs/retrain-recipe.md`). Its only job is producing that training data, to the standard set by chimera's `docs/datasheet.md` defect registry.

**Goal**: scale demo collection and state-sequence extraction without local persistence. Each match is downloaded to a tempdir, extracted, uploaded to HF, processed into tick sequences, and cleaned up. Resumable across crashes/restarts via on-HF manifests.

> **Parked: commentary (phase 2).** The caster-commentary grounding effort (VOD scrape, ASR transcription, demo↔caption alignment) is dormant until the later language phase, when a frozen LLM gets bridged into the world-model latents. Those modules and findings now live in [`parked/commentary/`](parked/commentary/README.md) and are *not* part of the main ingest flow.

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

There are two stages. **`run`** ingests demos (HLTV → `.dem` → HF). **`process`** turns those demos into world-model tick sequences (HF `.dem` → HF tick sequences).

```bash
# Scrape (dry, prints listing only)
chimera-demo scrape --stars 5 --max-matches 10

# Get one match's demo URL
chimera-demo fetch-match 2394156 spirit-vs-falcons-pgl-astana-2026

# Stage 1 — ingest: scrape + download + extract + upload + manifest
chimera-demo run --stars 3 --max-matches 50 --repo skkwowee/chimera-cs2

# Resume — auto-skips already-processed matches (manifest on HF is truth source)
chimera-demo run --stars 3 --max-matches 50

# Stage 2 — process ingested demos into tick sequences (the world-model training data)
chimera-demo process --max-matches 50

# Inspect what's already been ingested
chimera-demo manifest --show-last 20
```

`stars` is HLTV's match-tier filter (1 = all, 5 = LAN majors only). For training data, stars=3 strikes a good balance between volume and quality.

## How it works

### Stage 1 — `run` (ingest)

1. **Scrape** HLTV results listings (`/results?stars=N&offset=K`) — paginated, 100 matches per page.
2. **Skip** any match whose `match_id` already appears in `processed_manifest.jsonl` on HF.
3. **Fetch match page** to extract the `/download/demo/<id>` URL.
4. **Stream-download** the `.rar` (multi-map series → one rar containing N `.dem` files) to `tempfile.TemporaryDirectory()`.
5. **Extract** `.dem` files (`rarfile` + system `unrar`).
6. **Rename** to canonical `team1-vs-team2-mN-map.dem` form (matches existing demos in the dataset).
7. **Upload** all `.dem`s for the match in a single atomic HF commit.
8. **Append** one `ManifestEntry` JSON line to `processed_manifest.jsonl` (per-match commit) so reruns skip this match.
9. **Cleanup**: `TemporaryDirectory` exit removes everything from local disk.

### Stage 2 — `process` (tick sequences)

For each match in `processed_manifest.jsonl` not yet in `processed_tick_sequences_manifest.jsonl`:

1. **Download** the match's `.dem` files into a tempdir.
2. **Parse** (chimera's `parse_demos.py`) → per-tick parquet + per-event JSON.
3. **Build** (chimera's `build_tick_sequences.py`) → per-round `.pt` tensors of game-state frames (the world model's training input), with a `feature_schema` and `manifest.json`. Default downsample 8 (64 Hz → 8 Hz).
4. **Upload** artifacts under `tick_sequences/<match_id>/` on HF.
5. **Append** a line to `processed_tick_sequences_manifest.jsonl`.

Both stages are **match-atomic** (per-match tempdir + per-match commit + per-match manifest push), so a crash mid-batch never loses finished matches.

Peak local storage during a run = **1 match worth of files** (~500 MB to 2 GB). No persistent state.

## Provenance & archiving

- **`process` executes the LIVE `--chimera-dir` scripts** (copied into a sandbox at run time). To keep bakes reproducible it refuses to run when those scripts have uncommitted edits (`--allow-dirty` to override), verifies the builder contains the site-from-plant-position fix, and aborts if chimera's venv has `demoparser2 < 0.41.3` (EntityNotFound on Major demos).
- **Every tick-sequence manifest line records** `schema_version`, `builder_commit`(+dirty), `pipeline_commit`, and `awpy`/`demoparser2` versions. When the builder's `SCHEMA_VERSION` bumps, `process` automatically re-bakes every match recorded under an older version — the corpus-rebuild mechanism, no manifest surgery needed.
- **Parse intermediates are archived** to `parsed/<match_id>/` (`--archive-parsed`, default on): per-demo `*_ticks.parquet` + 5 event/header JSONs, measured ~12.8 MB/map ≈ 32 MB/match (92-match corpus ≈ 1.2 GB; 1000 matches ≈ 13 GB — 1-3% of the `.dem` bytes already stored). Re-bakes then use `--from-parsed` and skip the 3-6 min/demo awpy parse entirely.
- **Demos are namespaced** at `demos/<match_id>/<name>.dem` so a rematch of the same teams on the same map slot can't overwrite an earlier match (legacy flat paths still resolve via the manifest).
- **Failures persist** in `failures.jsonl` on HF; matches with 3+ recorded failures are skipped (not re-downloaded forever) unless `--retry-failed`. Inspect with `chimera-demo failures`.
- **Legacy local-only demos** (the ~54 stems not on HF) can be pushed with `chimera-demo backfill --demos-dir <dir> [--dry-run]`, which synthesizes manifest entries under a deterministic negative match-id namespace.

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
├── manifest.py     # processed-match tracking on HF (both stages)
├── process.py      # .dem → tick-sequence tensors (stage 2)
└── cli.py          # `chimera-demo` entry point

parked/
└── commentary/     # phase-2 caster-commentary tooling + findings (dormant)
```
