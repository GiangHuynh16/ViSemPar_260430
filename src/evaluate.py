#!/usr/bin/env python3
"""Strict PENMAN/Smatch evaluation for Vietnamese AMR."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import penman
import smatch

from amr_utils import canonicalize_amr, graph_components
from data_pipeline import parse_amr_file


def read_predictions(path: str) -> list[str]:
    """Read exactly one prediction record per input line, including blanks."""
    with open(path, encoding="utf-8") as handle:
        return [line.rstrip("\n") for line in handle]


def read_gold_records(path: str) -> list[dict[str, str]]:
    """Read sentence/graph records and canonicalize source annotations."""
    records = []
    for record in parse_amr_file(path):
        graph = canonicalize_amr(record["amr"])
        # Gold data is a contract: fail immediately if it is not valid.
        penman.decode(graph)
        records.append({"sentence": record["sentence"].strip(), "graph": graph})
    return records


def read_gold(path: str) -> list[str]:
    return [record["graph"] for record in read_gold_records(path)]


def parse_prediction(amr: str):
    """Parse model output without repairing or canonicalizing it."""
    text = amr.strip()
    if not text:
        return None
    try:
        return penman.decode(text)
    except Exception:
        return None


def is_parsable(amr: str) -> bool:
    return parse_prediction(amr) is not None


def _f1(match: int, predicted: int, gold: int) -> dict[str, float]:
    precision = match / predicted if predicted else 0.0
    recall = match / gold if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


def _counter_score(pred: Counter, gold: Counter) -> dict[str, float]:
    return _f1(sum((pred & gold).values()), sum(pred.values()), sum(gold.values()))


def _deterministic_random_init(candidate_mapping, rng: random.Random):
    """Equivalent to smatch's restart initializer without time-based seeding."""
    matched = set()
    result = []
    for candidates in candidate_mapping:
        remaining = list(candidates)
        while remaining:
            candidate = remaining.pop(rng.randrange(len(remaining)))
            if candidate not in matched:
                matched.add(candidate)
                result.append(candidate)
                break
        else:
            result.append(-1)
    return result


def compute_smatch_f1(
    predictions: list[str], gold: list[str], smatch_seed: int = 0
) -> dict[str, float]:
    """Compute Smatch with a reproducible seed for smatch's hill climbing."""
    match = predicted = gold_count = 0
    previous_initializer = smatch.random_init_mapping
    rng = random.Random(smatch_seed)
    smatch.random_init_mapping = lambda candidates: _deterministic_random_init(candidates, rng)
    try:
        for prediction, reference in zip(predictions, gold):
            prediction_graph = parse_prediction(prediction)
            if prediction_graph is None:
                try:
                    _, _, reference_count = smatch.get_amr_match("(x / invalid)", reference)
                except Exception:
                    reference_count = 0
                gold_count += reference_count
                continue
            try:
                current_match, current_predicted, current_gold = smatch.get_amr_match(
                    prediction, reference
                )
            except Exception:
                current_match = current_predicted = 0
                try:
                    _, _, current_gold = smatch.get_amr_match("(x / invalid)", reference)
                except Exception:
                    current_gold = 0
            match += current_match
            predicted += current_predicted
            gold_count += current_gold
    finally:
        smatch.random_init_mapping = previous_initializer
    return _f1(match, predicted, gold_count)


def compute_metrics(predictions: list[str], gold: list[str], smatch_seed: int = 0) -> dict:
    if len(predictions) != len(gold):
        raise ValueError(f"Prediction/gold count mismatch: {len(predictions)} != {len(gold)}")

    valid_graphs = 0
    root_correct = 0
    concepts_pred = Counter()
    concepts_gold = Counter()
    relations_pred = Counter()
    relations_gold = Counter()
    attributes_pred = Counter()
    attributes_gold = Counter()

    for prediction, reference in zip(predictions, gold):
        reference_graph = penman.decode(reference)
        ref_components = graph_components(reference_graph)
        concepts_gold += ref_components[0]
        relations_gold += ref_components[1]
        attributes_gold += ref_components[2]

        prediction_graph = parse_prediction(prediction)
        if prediction_graph is None:
            continue
        valid_graphs += 1
        pred_components = graph_components(prediction_graph)
        concepts_pred += pred_components[0]
        relations_pred += pred_components[1]
        attributes_pred += pred_components[2]
        if pred_components[3] == ref_components[3]:
            root_correct += 1

    n = len(gold)
    smatch_scores = compute_smatch_f1(predictions, gold, smatch_seed=smatch_seed)
    concept_scores = _counter_score(concepts_pred, concepts_gold)
    relation_scores = _counter_score(relations_pred, relations_gold)
    attribute_scores = _counter_score(attributes_pred, attributes_gold)
    return {
        "n_samples": n,
        "valid_graphs": valid_graphs,
        "valid_graph_rate": round(valid_graphs / n, 4) if n else 0.0,
        "parsable": valid_graphs,
        "parsable_rate": round(valid_graphs / n, 4) if n else 0.0,
        "avg_output_len": round(sum(len(p) for p in predictions) / n, 1) if n else 0.0,
        "root_acc": round(root_correct / n, 4) if n else 0.0,
        "precision": smatch_scores["precision"],
        "recall": smatch_scores["recall"],
        "f1": smatch_scores["f1"],
        "smatch": smatch_scores,
        "concept": concept_scores,
        "relation": relation_scores,
        "attribute": attribute_scores,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--group", required=True)
    args = parser.parse_args()

    if not Path(args.pred).exists():
        print(f"ERROR: pred file not found: {args.pred}")
        sys.exit(1)

    predictions = read_predictions(args.pred)
    gold = read_gold(args.gold)
    metrics = compute_metrics(predictions, gold)
    scores = {"method": args.method, "seed": args.seed, "group": args.group, **metrics}

    print(f"\n{args.method} seed={args.seed}")
    print(f"  Smatch F1:       {metrics['smatch']['f1']:.4f}")
    print(f"  Valid graph rate:{metrics['valid_graph_rate']:.4f}")
    print(f"  Concept F1:      {metrics['concept']['f1']:.4f}")
    print(f"  Relation F1:     {metrics['relation']['f1']:.4f}")
    print(f"  Attribute F1:    {metrics['attribute']['f1']:.4f}")
    print(f"  Root accuracy:   {metrics['root_acc']:.4f}")

    output = Path(args.pred).parent / "scores.json"
    existing = json.loads(output.read_text()) if output.exists() else {}
    if not isinstance(existing, dict):
        existing = {}
    existing[f"s{args.seed}"] = scores
    output.write_text(json.dumps(existing, indent=2, ensure_ascii=False))
    print(f"Scores saved → {output}")


if __name__ == "__main__":
    main()
