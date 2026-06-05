#!/usr/bin/env python3
"""pilot_minimum — the cheapest possible decisive test of expert-corpus methodology.

After 33 iterations of design (see docs/loop-distillation.md for the journey),
this is the SINGLE FIRST experiment to run. ~4 hours, $0 in compute.

Goal: decide whether expert-corpus grounding of structured-state features can
produce useful retrieval-based explanations. If yes, build v0. If no, rethink.

Pipeline:
  1. Load hand-written labels from data/pilot_labels.jsonl
  2. Pull cached tick_sequences from HF for the labeled (demo, round, tick) triples
  3. Compute window-aggregated features (mean/std/min/max over 5s causal window)
  4. Embed captions with frozen MiniLM
  5. Pearson correlate cosine sim matrices in both spaces
  6. Inspect top-1 retrieval per held-out query for sanity
  7. Decision rule: GO / PROCEED-CAUTIOUSLY / STOP

NO Whisper. NO Qwen. NO encoder. NO oracle. Just hand-labeled captions +
aggregated features + frozen sentence embedder + retrieval over a 30-row corpus.

If even THIS doesn't show correlation, the architecture has a deeper problem.

Usage:
    # First, write 30 labels (1 hour by hand, see data/pilot_labels.template.jsonl)
    python scripts/pilot_minimum.py --labels data/pilot_labels.jsonl

Output:
    Pearson r + decision + retrieval inspection printed to stdout
    Optional JSON dump via --output
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = "skkwowee/chimera-cs2"
WINDOW_TICKS = 40  # 5 seconds at 8 Hz downsampled


@dataclass
class Label:
    demo_stem: str
    round_num: int
    tick: int
    caption: str
    match_id: int | None = None  # filled in during HF lookup if known

    @classmethod
    def from_row(cls, row: dict) -> "Label":
        return cls(
            demo_stem=row["demo_stem"],
            round_num=int(row["round_num"]),
            tick=int(row["tick"]),
            caption=row["caption"],
            match_id=row.get("match_id"),
        )


def load_labels(path: Path) -> list[Label]:
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out.append(Label.from_row(json.loads(line)))
    print(f"Loaded {len(out)} labels", flush=True)
    return out


def find_tick_sequence_files() -> dict[str, list[str]]:
    """Map demo_stem → list of HF file paths that might contain that demo's data.

    tick_sequences/<match_id>/{train,val}.pt — match_id is HLTV's numeric ID.
    We don't know which match_id corresponds to which demo_stem without loading,
    so we scan a sample of files and build a lookup.
    """
    from huggingface_hub import HfApi
    files = HfApi().list_repo_files(REPO, repo_type="dataset")
    pt_files = [f for f in files
                  if f.startswith("tick_sequences/") and f.endswith(".pt")]
    # Group by match_id directory
    by_match: dict[str, list[str]] = defaultdict(list)
    for f in pt_files:
        parts = f.split("/")
        if len(parts) >= 3:
            by_match[parts[1]].append(f)
    return dict(by_match)


def extract_window_features(labels: list[Label]) -> tuple[np.ndarray, list[int]]:
    """Pull tick features from HF for each labeled (demo, round, tick) tuple.

    For each label, compute window aggregation: concat([mean, std, min, max])
    over the 5s causal window ending at the labeled tick.

    Returns (X, kept_indices) — X is (N_kept, 4 * feature_dim), kept_indices are
    the indices into `labels` for the rows we successfully extracted.
    """
    import torch
    from huggingface_hub import hf_hub_download

    pt_files_by_match = find_tick_sequence_files()
    print(f"Found {len(pt_files_by_match)} matches on HF", flush=True)

    # Need to find which match_id contains which demo_stem.
    # Build (demo_stem, round_num, tick) → tensor lookup.
    needed = {(l.demo_stem, l.round_num) for l in labels}
    found: dict[tuple, np.ndarray] = {}    # (demo_stem, round_num) → (T, F) tensor as numpy
    scanned = 0
    for match_id, files in pt_files_by_match.items():
        if all(k in found for k in needed):
            break
        for pt_path in files:
            try:
                local = hf_hub_download(REPO, pt_path, repo_type="dataset")
                blob = torch.load(local, weights_only=False, map_location="cpu")
            except Exception as e:
                print(f"  skip {pt_path}: {e}", flush=True)
                continue
            scanned += 1
            for tensor, meta in zip(blob["tensors"], blob["metas"]):
                key = (meta["demo_stem"], meta["round_num"])
                if key in needed and key not in found:
                    found[key] = tensor.numpy().astype(np.float32)
                    print(f"  matched {key[0]} round {key[1]}", flush=True)
                    if all(k in found for k in needed):
                        break

    print(f"Scanned {scanned} .pt files; matched {len(found)}/{len(needed)} "
          f"unique (demo, round) pairs", flush=True)

    X = []
    kept = []
    for i, l in enumerate(labels):
        key = (l.demo_stem, l.round_num)
        if key not in found:
            continue
        tensor = found[key]  # (T, F)
        T = tensor.shape[0]
        if not (0 <= l.tick < T):
            print(f"  skip label {i}: tick {l.tick} out of [0, {T})", flush=True)
            continue
        start = max(0, l.tick - WINDOW_TICKS)
        window = tensor[start: l.tick + 1]  # (W, F), causal
        agg = np.concatenate([
            window.mean(axis=0),
            window.std(axis=0),
            window.min(axis=0),
            window.max(axis=0),
        ])  # (4*F,) = (2388,)
        X.append(agg)
        kept.append(i)

    if not X:
        sys.exit("ERROR: no labels resolved to embeddings; check data/pilot_labels.jsonl "
                  "(do demo_stem + round_num + tick all match what's on HF?)")
    print(f"Extracted features for {len(X)}/{len(labels)} labels", flush=True)
    return np.stack(X), kept


def embed_captions(captions: list[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return model.encode(captions, normalize_embeddings=True, convert_to_numpy=True)


def cosine_sim_matrix(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    xn = x / np.maximum(n, 1e-8)
    return xn @ xn.T


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--labels", type=Path,
                     default=Path("data/pilot_labels.jsonl"))
    ap.add_argument("--output", type=Path, default=None,
                     help="optional JSON output path")
    args = ap.parse_args()

    labels_all = load_labels(args.labels)
    X, kept = extract_window_features(labels_all)
    labels = [labels_all[i] for i in kept]
    C = embed_captions([l.caption for l in labels])
    print(f"Features {X.shape}, captions {C.shape}", flush=True)

    from scipy.stats import pearsonr, spearmanr
    S_X = cosine_sim_matrix(X)
    S_C = cosine_sim_matrix(C)
    N = len(labels)
    iu = np.triu_indices(N, k=1)
    r, p = pearsonr(S_X[iu], S_C[iu])
    rs, ps = spearmanr(S_X[iu], S_C[iu])

    print("\n" + "=" * 64)
    print(f"Pearson  r: {r:+.4f}  (p={p:.4f})  N_pairs={len(iu[0])}")
    print(f"Spearman r: {rs:+.4f}  (p={ps:.4f})")
    print("=" * 64)
    if r > 0.20:
        decision = "GO — proceed with full v0 build"
    elif r > 0.05:
        decision = "PROCEED-CAUTIOUSLY — run G0 (with Whisper) before committing v0"
    else:
        decision = "STOP — features don't capture caption-level structure; rethink encoder"
    print(f"DECISION: {decision}\n")

    print("Top-1 retrieval inspection:")
    print("(query → nearest neighbor in feature space)\n")
    for i in range(min(N, 30)):
        order = np.argsort(-S_X[i])
        nearest = order[1] if order[0] == i else order[0]
        flag = "✓" if S_C[i, nearest] > 0.5 else " "
        print(f"  [{i:2d}] {flag} q: {labels[i].caption[:60]}")
        print(f"          n: {labels[nearest].caption[:60]} "
               f"(feat_cos={S_X[i, nearest]:.2f}, cap_cos={S_C[i, nearest]:.2f})")

    if args.output:
        args.output.write_text(json.dumps({
            "n_labels": N,
            "pearson_r": float(r), "pearson_p": float(p),
            "spearman_r": float(rs), "spearman_p": float(ps),
            "decision": decision,
        }, indent=2))
        print(f"\nWrote {args.output}")

    return 0 if r > 0.05 else 1


if __name__ == "__main__":
    sys.exit(main())
