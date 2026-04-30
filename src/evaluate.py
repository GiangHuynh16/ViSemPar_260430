#!/usr/bin/env python3
"""
Evaluation for ViSemPar_260430.

Computes: Smatch F1, parsable_rate, avg_output_len, root_acc
Saves scores.json to the results/{group}/{method}/ directory.

Usage:
    python src/evaluate.py \
        --pred  results/baselines/B1/s42_pred.txt \
        --gold  data/raw/public_test_ground_truth.txt \
        --method B1 --seed 42 --group baselines
"""

import re
import sys
import json
import argparse
from pathlib import Path

try:
    import penman
    HAS_PENMAN = True
except ImportError:
    HAS_PENMAN = False

try:
    import smatch
    HAS_SMATCH = True
except ImportError:
    HAS_SMATCH = False


# ─────────────────────────────────────────────────────────────────────────────
# Parsers
# ─────────────────────────────────────────────────────────────────────────────

def read_predictions(path: str) -> list:
    """One AMR per line (flat single-line format)."""
    with open(path, encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def read_gold(path: str) -> list:
    """Parse #::snt + multi-line AMR blocks."""
    with open(path, encoding='utf-8') as f:
        content = f.read()
    graphs = []
    for block in content.strip().split('\n\n'):
        block = block.strip()
        if not block:
            continue
        amr_lines = [l for l in block.split('\n') if not l.startswith('#')]
        amr = ' '.join(amr_lines)
        amr = re.sub(r'\s+', ' ', amr).strip()
        if amr:
            graphs.append(amr)
    return graphs


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def is_parsable(amr: str) -> bool:
    """Check if AMR is valid PENMAN."""
    if HAS_PENMAN:
        try:
            penman.decode(amr)
            return True
        except Exception:
            return False
    # Fallback: basic structural check
    s = amr.strip()
    if not s.startswith('(') or '/' not in s:
        return False
    return s.count('(') == s.count(')')


def get_root_concept(amr: str) -> str:
    """Extract root concept from AMR."""
    m = re.search(r'\(\s*\w+\s*/\s*([^\s:()]+)', amr)
    return m.group(1) if m else ''


def compute_smatch_f1(predictions: list, gold: list) -> dict:
    """Compute Smatch P/R/F1 using official smatch library."""
    if not HAS_SMATCH:
        print("WARNING: smatch not installed. Run: pip install smatch")
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

    total_match = total_pred = total_gold = 0
    errors = 0
    n = min(len(predictions), len(gold))

    for pred, g in zip(predictions[:n], gold[:n]):
        try:
            match, pred_n, gold_n = smatch.get_amr_match(pred, g)
            total_match += match
            total_pred  += pred_n
            total_gold  += gold_n
        except Exception:
            errors += 1

    if errors:
        print(f"  [Smatch] {errors}/{n} pairs failed to compute")

    p = total_match / total_pred  if total_pred  > 0 else 0.0
    r = total_match / total_gold  if total_gold  > 0 else 0.0
    f1 = 2 * p * r / (p + r)     if (p + r) > 0 else 0.0
    return {'precision': round(p, 4), 'recall': round(r, 4), 'f1': round(f1, 4)}


def compute_metrics(predictions: list, gold: list) -> dict:
    n = min(len(predictions), len(gold))
    preds = predictions[:n]
    golds = gold[:n]

    # Parsable rate
    parsable = sum(1 for p in preds if is_parsable(p))
    parsable_rate = round(parsable / n, 4) if n > 0 else 0.0

    # Avg output length (chars)
    avg_len = round(sum(len(p) for p in preds) / n, 1) if n > 0 else 0.0

    # Root accuracy
    root_correct = sum(
        1 for p, g in zip(preds, golds)
        if get_root_concept(p) == get_root_concept(g)
    )
    root_acc = round(root_correct / n, 4) if n > 0 else 0.0

    # Smatch
    smatch_scores = compute_smatch_f1(preds, golds)

    return {
        'n_samples': n,
        'parsable': parsable,
        'parsable_rate': parsable_rate,
        'avg_output_len': avg_len,
        'root_acc': root_acc,
        **smatch_scores,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred',   required=True)
    parser.add_argument('--gold',   required=True)
    parser.add_argument('--method', required=True)
    parser.add_argument('--seed',   type=int, required=True)
    parser.add_argument('--group',  required=True)
    args = parser.parse_args()

    if not Path(args.pred).exists():
        print(f"ERROR: pred file not found: {args.pred}")
        sys.exit(1)

    preds = read_predictions(args.pred)
    gold  = read_gold(args.gold)

    if len(preds) != len(gold):
        print(f"WARNING: pred count ({len(preds)}) != gold count ({len(gold)})")

    metrics = compute_metrics(preds, gold)

    scores = {
        'method': args.method,
        'seed':   args.seed,
        'group':  args.group,
        **metrics,
    }

    # Print summary
    print(f"\n{'='*50}")
    print(f"  {args.method} seed={args.seed}")
    print(f"{'='*50}")
    print(f"  Smatch F1:      {scores['f1']:.4f}")
    print(f"  Precision:      {scores['precision']:.4f}")
    print(f"  Recall:         {scores['recall']:.4f}")
    print(f"  Parsable rate:  {scores['parsable_rate']:.4f}  ({scores['parsable']}/{scores['n_samples']})")
    print(f"  Root accuracy:  {scores['root_acc']:.4f}")
    print(f"  Avg output len: {scores['avg_output_len']:.1f} chars")
    print(f"{'='*50}\n")

    # Save scores.json next to pred file
    out_dir = Path(args.pred).parent
    scores_path = out_dir / 'scores.json'

    # Load existing and merge (keep all seeds)
    existing = {}
    if scores_path.exists():
        with open(scores_path) as f:
            existing = json.load(f)
        if not isinstance(existing, dict):
            existing = {}

    existing[f"s{args.seed}"] = scores

    with open(scores_path, 'w') as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)

    print(f"Scores saved → {scores_path}")


if __name__ == '__main__':
    main()
