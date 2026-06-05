#!/usr/bin/env python3
"""Validate the lexical tactical-gate against an LLM judge.

The keyword gate has known precision issues (homographs) but UNKNOWN RECALL --
it can't catch paraphrased tactics ("stacking that corner", "playing for picks").
This measures precision + recall + a corrected tactical density by having Claude
judge a stratified sample of windows, then reweighting by the true bucket sizes.

Claude here is a MEASUREMENT INSTRUMENT on real human commentary (not a label
generator for training) -- so this is not the circular-caption problem.
"""
import os, sys, glob, random, re

import anthropic

sys.path.insert(0, "/home/soone/chimera-demo-pipeline/parked/commentary")
from commentary_pilot import parse_vtt, window_words

KEY = None
for line in open("/home/soone/perplexity-spike/.env"):
    if line.startswith("ANTHROPIC_API_KEY"):
        KEY = line.split("=", 1)[1].strip().strip('"').strip("'")
client = anthropic.Anthropic(api_key=KEY)
MODEL = "claude-sonnet-4-6"

VODS = {"Fdp3xEqDNtM": "IEM G2-FaZe", "k9E4wwLKXE0": "PGL Spirit-Falcons"}
N_PER_BUCKET = 40
random.seed(0)

# gather all windows across the VODs, tagged by gate bucket
allw = []
for vid in VODS:
    w = parse_vtt(f"/tmp/vodcap/{vid}.en.vtt")
    for win in window_words(w, 15.0):
        if win.bucket != "silence":
            allw.append(win)
from collections import Counter
counts = Counter(w.bucket for w in allw)
total = sum(counts.values())
props = {b: counts[b] / total for b in counts}
print(f"pooled windows: {total}  gate props: " +
      " ".join(f"{b}={props.get(b,0):.2f}" for b in ("tactical", "vague", "offtopic")))

# stratified sample
sample = {}
for b in ("tactical", "vague", "offtopic"):
    pool = [w for w in allw if w.bucket == b]
    random.shuffle(pool)
    sample[b] = pool[:N_PER_BUCKET]

SYS = ("You judge whether a 15-second window of a CS2 (Counter-Strike 2) esports "
       "broadcast transcript (noisy auto-captions) contains SPECIFIC, ACTIONABLE "
       "TACTICAL content about the LIVE game: callouts/map positions, what a team or "
       "player is doing or about to do, utility usage (smoke/flash/molotov), economy/buy "
       "reads, rotations, executes/fakes, who holds an advantage, clutch situations. "
       "Mark NOT if it is hype, crowd noise, casual banter, player biography/history, "
       "caster chit-chat, sponsor/segment talk, generic praise ('nice', 'huge'), or "
       "off-topic. Captions are noisy; judge the gist. Reply ONLY with lines like "
       "'<n>: TACTICAL' or '<n>: NOT'.")


def judge(batch):
    lines = [f"{i+1}: {w.text.strip()[:300]}" for i, w in enumerate(batch)]
    msg = client.messages.create(
        model=MODEL, max_tokens=1000, system=SYS,
        messages=[{"role": "user", "content": "\n\n".join(lines)}])
    txt = msg.content[0].text
    out = {}
    for m in re.finditer(r"(\d+)\s*:\s*(TACTICAL|NOT)", txt, re.I):
        out[int(m.group(1))] = m.group(2).upper() == "TACTICAL"
    return out


rates = {}
for b, batch in sample.items():
    labels = {}
    for i in range(0, len(batch), 15):
        chunk = batch[i:i+15]
        res = judge(chunk)
        for j, w in enumerate(chunk):
            labels[i + j] = res.get(j + 1, False)
    r = sum(labels.values()) / len(labels)
    rates[b] = r
    print(f"  gate={b:9s} n={len(batch)}  Claude-tactical rate = {r:.2f}")

p = props
corrected = sum(p.get(b, 0) * rates[b] for b in rates)
precision = rates["tactical"]
recall = (p.get("tactical", 0) * rates["tactical"]) / corrected if corrected else 0
print(f"\n--- RESULTS ---")
print(f"gate tactical density (hardened):   {p.get('tactical',0)*100:.0f}%")
print(f"CORRECTED tactical density (Claude): {corrected*100:.0f}%")
print(f"gate PRECISION (gate-tac that are really tac): {precision*100:.0f}%")
print(f"gate RECALL (real-tac the gate caught):        {recall*100:.0f}%")
print(f"  -> Claude calls tactical: vague-bucket {rates['vague']*100:.0f}%, "
      f"offtopic-bucket {rates['offtopic']*100:.0f}%  (these are the gate's MISSES)")
