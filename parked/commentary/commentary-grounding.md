# Commentary grounding: VOD → demo alignment

Goal: attach real caster commentary (Twitch/YouTube) to demo ticks, so we can
ground the L3 verbalization layer in authentic human tactical language instead
of model-generated captions (which are circular — see chimera's
`data/captions/CROSSVAL.md`: Claude captions only paraphrase the structured
features they were given, so they can't carry signal those features lack).

Full pipeline:

```
manifest match ──► find-vods ──► download audio ──► ASR ──► align ──► tick-tagged transcript
   (have)          (DONE)         (needs ffmpeg)   (whisper)  (hard)      (the product)
```

## Stage 1 — VOD discovery  ✅ DONE  (`pipeline/vod.py`, `chimera-demo find-vods`)

Per-map demos self-label the map (`spirit-vs-falcons-m1-dust2.dem`), and
official broadcast VODs self-label in the title ("... Mapa 1 - Dust 2 ..."),
so matching is a title-parse + score, not a fuzzy guess.

- Two-pass search: generic series query, then a targeted per-map query
  (`"{t1} vs {t2} {event} map {i} {mapname}"`) when the generic pool misses
  map 2/3 (official channels upload one VOD per map; the top-N of the generic
  search often only surfaces map 1).
- Score = teams (1 each) + event (1) + map name (2) + map index (1); penalize
  highlight reels and sub-12-min clips. A correct per-map VOD scores ~6.
- **Verified live**: Spirit vs Falcons (PGL Astana 2026) → all 3 maps matched
  to the correct per-map official PGL VODs at +6.0.
- Gotcha found & fixed: `yt-dlp --flat-playlist` TRUNCATES some titles (drops
  the trailing date), silently dropping event/team token hits and tanking the
  score. Use full `--skip-download` extraction for complete titles + real
  durations (a few seconds slower per search; worth it).

## Stages 2 + 3 — audio extraction + ASR  ✅ CODED (`pipeline/transcribe.py`, `chimera-demo transcribe`)

Grouped into ONE streaming pass, because they share a machine (ffmpeg + GPU =
the RunPod worker) and the audio between them is large + transient — no reason
to persist a separate WAV:

```
VOD --(yt-dlp + ffmpeg)--> 16kHz mono wav in tempdir
    --(faster-whisper)--> [(word, t_start, t_end, prob), ...]   (VOD time)
    --> discard wav, keep only the transcript
```

- `extract_audio`: `yt-dlp -f bestaudio -x` with `ffmpeg:-ac 1 -ar 16000` so
  the wav is born in exactly the format whisper wants (no second resample).
- `transcribe_audio`: faster-whisper `large-v3`, `word_timestamps=True`,
  `vad_filter=True` (drops desk/pause silences → fewer spurious words to align).
  `language=None` auto-detects (broadcasts are EN official + RU/PT/TR community
  casts).
- `transcribe_vod`: the two in one `TemporaryDirectory` pass — wav deleted on
  exit (success or error), only the ~tens-of-KB transcript kept.
- Output `Transcript.to_dict()`: segments → word list with `(t, s, e, p)` in
  VOD seconds. **These VOD-time stamps are exactly what stage 4 aligns to ticks.**

**Runs on the pod, not WSL**: needs `ffmpeg` (apt) + `faster-whisper` (pip) +
CUDA. The module imports fine without them — faster-whisper is lazy-imported
inside the function, and the CLI command registers — but actual transcription
must run where ffmpeg + a GPU exist.

## Stage 4 — alignment  ⚠️ THE HARD PART

VOD time ≠ demo time. Three offsets to solve:

1. **Pre-roll**: the VOD starts before the map (analyst desk, pauses, tech
   timeouts). 0–20 min, not constant across VODs.
2. **Caster reaction latency**: casters react ~0.5–3 s AFTER an event. Roughly
   constant within a VOD, estimable.
3. **Stoppages**: tac-/tech-pauses stretch VOD time vs demo time mid-map, so a
   single global offset is insufficient.

**Anchor signal — round transitions + kills.** The demo gives the exact
wall-clock sequence of round starts/ends and kills (tick / tickrate), with
player names per kill. Casters produce a reliable burst of speech at round
ends ("and Spirit take it") and say player names on kills. So:

- Build the demo event timeline: round-end times, kill times, player names.
- From the ASR transcript, detect the same landmarks: player-name mentions
  (kills) and round-result phrases.
- **Cross-correlate** the two event trains to recover global offset + drift;
  refine per-round with a local search around each round boundary. Player-name
  mentions are the densest anchor — every kill is a name the demo knows
  exactly.
- Robustness: RANSAC-style fit (most rounds agree on one offset; pauses are
  outliers we re-anchor after). Per-round confidence = anchor density; drop
  low-confidence rounds rather than emit a bad alignment.

Tractable but not trivial: anchors are real and dense (10–30 kills/map, each a
name), but ASR mis-hears names, casters discuss past rounds, and multi-language
casts exist. Expect to keep only high-confidence windows — we want *clean*
grounded captions, not full coverage.

## Stage 5 — product

Per high-confidence window: `(demo_stem, round, tick_range, commentary_text)`.
This replaces model captions in the discriminative check. The real,
non-circular test: **do transcript-derived captions beat the structured
ceiling?** If yes, human commentary carries tactical signal the features
don't — the actual green light for the encoder→language bridge.

## Storage discipline

Same as the rest of the pipeline: VOD audio is large and transient — stream
through a tempdir, transcribe, keep only the (small) tick-tagged transcript on
HF. Track processed VODs in a `processed_commentary_manifest.jsonl` parallel to
the demo + tick-sequence manifests.
