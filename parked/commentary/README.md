# Parked: commentary grounding

> **Status update (2026-06-05): the commentary pilot was DROPPED, not merely
> parked** — commentary-as-events is dead per the pilot-findings memo.
> Validated on the way out: tactical density is actually **~60–72%** (the
> lexical gate undercounted at 46% recall — the "8–31%" figure below is
> WRONG); the real blockers are per-event alignment (~25%, ASR-capped) and
> temporal resolution, not sparsity. Event/strategy label creation will be
> solved another way. Code kept only for reusable parts (e.g.
> `extract_round_frames`).

This is the caster-commentary → tick alignment work from the world-model
(next-state prediction) phase's earlier plan.

## What's here
- `vod.py` — YouTube/HLTV VOD discovery + title scoring (yt-dlp).
- `transcribe.py` — VOD audio → timestamped transcript (faster-whisper). Pod-only.
- `commentary_pilot.py` — VTT parse, 15s windowing, lexical relevance gate,
  kills↔name-mention cross-correlation, asymmetric lead/lag alignment.
- `demo_anchors.py` — awpy demo parse → kill/round anchors + roster + per-tick
  frames (`extract_round_frames`). NOTE: `extract_round_frames` is the bit the
  cs2-demo-viewer will reuse for rollout visualization — lift it out when needed.
- `build_aligned_example.py` — end-to-end aligned-example builder (the 4.6σ
  Spirit-vs-Falcons dust2 demo).
- `pilot_minimum.py` — hand-labeled caption pilot.
- `commentary_pilot_k9E4wwLKXE0.jsonl`, `pilot_labels.template.jsonl` — pilot data.
- `commentary-grounding.md` — the full pipeline design doc (stages 1–5).

## Key findings (so they aren't re-derived)
- Global VOD↔demo alignment locks at **4.6σ** via kills↔name-mentions, but
  per-event anchoring is only **~25%** — capped by auto-caption ASR name recall
  (NiKo/TeSeS dropped; phonetic skeleton matching helps). Frame-exact alignment
  needs Whisper-grade ASR + multi-signal anchors (round-end bursts, scoreline).
- ~~Tactical-commentary density ~[8–31%]~~ — **superseded 2026-06-05**: the
  lexical gate had 46% recall; actual density is **~60–72%**. Silence remains
  a non-issue (casters never stop).

## To restore (only if commentary is ever revived — see DROP note above)
```
git mv parked/commentary/vod.py             pipeline/vod.py
git mv parked/commentary/transcribe.py      pipeline/transcribe.py
git mv parked/commentary/commentary_pilot.py pipeline/commentary_pilot.py
git mv parked/commentary/demo_anchors.py    pipeline/demo_anchors.py
git mv parked/commentary/build_aligned_example.py scripts/
git mv parked/commentary/pilot_minimum.py   scripts/
```
Then re-add to `pipeline/cli.py`:
```python
from .vod import find_vods_for_match
from .transcribe import transcribe_vod
```
and restore the `find_vods` / `transcribe` commands (see git history of cli.py).
