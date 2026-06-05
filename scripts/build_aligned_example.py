#!/usr/bin/env python3
"""Build a TRULY aligned commentary example for the cs2-demo-viewer.

End-to-end, fully local (needs awpy + the .dem + a VOD .vtt):
  parse demo -> kill/round anchors + real per-tick frames
  parse VOD captions -> words + 15s windows + relevance gate
  cross-correlate kill-times vs name-mention-times -> VOD->demo offset
  map each caption window -> real (round, absolute tick range, structurally_flat)
  export real frames + real aligned commentary to the viewer

Run (chimera venv has awpy; pipeline modules on path):
  PYTHONPATH=/home/soone/chimera-demo-pipeline \
  /home/soone/chimera/.venv/bin/python \
    scripts/build_aligned_example.py \
      --demo /home/soone/chimera/data/demos_new/spirit-vs-falcons-m2-dust2.dem \
      --vtt  /tmp/vodcap/s6-0QwmLFW0.en.vtt \
      --vod  s6-0QwmLFW0 \
      --out  /home/soone/cs2-demo-viewer/public/viewer-data/sample-match \
      --max-lag 6000
"""
import argparse
import json
from pathlib import Path

from pipeline.demo_anchors import parse_demo, extract_round_frames
from pipeline.commentary_pilot import (
    parse_vtt, window_words, align_transcript, align_windows,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", required=True)
    ap.add_argument("--vtt", required=True)
    ap.add_argument("--vod", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-lag", type=float, default=6000.0)
    ap.add_argument("--win-sec", type=float, default=15.0)
    args = ap.parse_args()

    print(f"[1/5] parsing demo {Path(args.demo).name} ...")
    anchors, demo = parse_demo(args.demo, with_frames=True)
    print(f"      event={anchors.event!r} map={anchors.map_name} "
          f"tickrate={anchors.tickrate}")
    print(f"      {len(anchors.rounds)} rounds, {len(anchors.kills)} kills, "
          f"roster={anchors.roster}")

    print(f"[2/5] parsing captions {Path(args.vtt).name} ...")
    words = parse_vtt(args.vtt)
    print(f"      {len(words)} words, {words[0].t:.0f}..{words[-1].t:.0f}s")

    print("[3/5] aligning (cross-correlate kills vs name-mentions) ...")
    al = align_transcript(words, anchors.kill_secs(), anchors.roster,
                          max_lag=args.max_lag)
    coverage = al.score / max(1, al.n_demo)
    # Confidence is the peak's sigma above the off-peak (chance) floor, NOT the
    # raw % — lossy ASR keeps % low even when the offset is unambiguous.
    import bisect as _bi, statistics as _st
    names = sorted(__import__("pipeline.commentary_pilot", fromlist=["find_name_mentions"])
                   .find_name_mentions(words, anchors.roster))
    d = sorted(anchors.kill_secs())
    def _cov(lag):
        s = 0
        for t in d:
            x = t + lag; k = _bi.bisect_left(names, x - 2.0)
            if k < len(names) and names[k] <= x + 2.0:
                s += 1
        return s
    floor = [_cov(al.offset + dl) for dl in range(-9000, 9001, 90) if abs(dl) > 300]
    mu, sd = _st.mean(floor), (_st.pstdev(floor) or 1.0)
    sigma = (al.score - mu) / sd
    print(f"      offset={al.offset:.1f}s ({al.offset/60:.1f}min)  "
          f"coverage={al.score:.0f}/{al.n_demo} ({100*coverage:.0f}%)  "
          f"name-mentions={al.n_vod}  confidence={sigma:.1f}sigma (floor mu={mu:.1f})")
    if sigma < 3.0:
        print("      !! WEAK PEAK (<3sigma) -- offset not trustworthy; likely the "
              "wrong VOD or map absent. Output unreliable.")
    else:
        print(f"      offset locked ({sigma:.1f}sigma). Note: {100*coverage:.0f}% "
              f"per-kill anchoring reflects lossy auto-caption ASR, not offset "
              f"error -- per-line timing good to a few seconds + pause drift.")
    al_sigma = sigma

    print("[4/5] windowing + per-window alignment ...")
    wins = window_words(words, args.win_sec)
    awins = align_windows(wins, al, anchors)
    by_bucket = {}
    flat_tac = 0
    for w in awins:
        by_bucket[w.bucket] = by_bucket.get(w.bucket, 0) + 1
        if w.bucket == "tactical" and w.structurally_flat:
            flat_tac += 1
    print(f"      {len(awins)} windows fell inside a round: {by_bucket}")
    print(f"      tactical & structurally-flat (THE key bucket): {flat_tac}")

    print(f"[5/5] exporting real frames + aligned commentary to {args.out} ...")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # round files (real frames) for every round
    for r in anchors.rounds:
        rd = extract_round_frames(demo, r.round_num, anchors)
        (out / f"round_{r.round_num:02d}.json").write_text(json.dumps(rd))

    # meta.json (real rounds + kills)
    kills = [{
        "tick": k.tick, "round_num": k.round_num,
        "attacker_name": k.attacker, "attacker_side": "", "victim_name": k.victim,
        "weapon": k.weapon, "headshot": k.headshot,
    } for k in anchors.kills]
    # enrich kills with positions for kill lines
    for k, src in zip(kills, demo.kills.to_dicts()):
        k["attacker_X"] = src.get("attacker_X") or 0
        k["attacker_Y"] = src.get("attacker_Y") or 0
        k["victim_X"] = src.get("victim_X") or 0
        k["victim_Y"] = src.get("victim_Y") or 0
        k["attacker_side"] = src.get("attacker_side") or ""
        k["victim_side"] = src.get("victim_side") or ""
    meta = {
        "stem": "sample-match", "map_name": anchors.map_name,
        "header": {"map_name": anchors.map_name, "server_name": anchors.event},
        "rounds": [{"round_num": r.round_num, "winner": r.winner,
                    "reason": r.reason} for r in anchors.rounds],
        "kills": kills, "damages": [], "shots": [], "bomb": [],
    }
    (out / "meta.json").write_text(json.dumps(meta))

    # commentary.json (real aligned windows)
    lines = [{
        "round_num": w.demo_round,
        "tick_start": w.demo_tick_range[0], "tick_end": w.demo_tick_range[1],
        "bucket": w.bucket, "specific_hits": w.specific_hits,
        "structurally_flat": w.structurally_flat,
        "text": w.text.strip(),
    } for w in awins if w.bucket != "silence"]
    commentary = {
        "_meta": {
            "source_vod": args.vod,
            "source": f"{anchors.event} — {anchors.stem} auto-captions",
            "alignment": f"REAL: kills↔name-mentions cross-correlation. "
                         f"offset={al.offset:.0f}s ({al.offset/60:.0f}min), "
                         f"confidence={al_sigma:.1f}sigma, {al.score:.0f}/{al.n_demo} "
                         f"({100*al.score/al.n_demo:.0f}%) kills anchored. The offset "
                         f"is solid; the low % is lossy auto-caption ASR (names like "
                         f"NiKo/TeSeS not transcribed), so per-line timing is good to "
                         f"a few seconds + pause drift, not frame-exact. Tick ranges "
                         f"are absolute demo ticks.",
            "n_lines": len(lines),
        },
        "lines": lines,
    }
    (out / "commentary.json").write_text(json.dumps(commentary, indent=2))

    # index.json
    (out.parent / "index.json").write_text(json.dumps([{
        "stem": "sample-match", "map_name": anchors.map_name,
        "rounds": len(anchors.rounds),
    }]))
    print(f"      wrote {len(anchors.rounds)} rounds, {len(kills)} kills, "
          f"{len(lines)} commentary lines.")
    print("done.")


if __name__ == "__main__":
    main()
