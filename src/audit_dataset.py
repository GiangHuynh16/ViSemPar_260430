#!/usr/bin/env python3
"""Validate active Vietnamese manifests and their PENMAN gold graphs."""

from __future__ import annotations

import argparse
from pathlib import Path

from amr_utils import validate_amr
from data_pipeline import parse_amr_file
from evaluate import read_gold


def sentence_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def check_pair(sentence_path: Path, gold_path: Path) -> int:
    sentences = sentence_lines(sentence_path)
    records = parse_amr_file(str(gold_path))
    graphs = read_gold(str(gold_path))
    if len(sentences) != len(records) or len(records) != len(graphs):
        raise ValueError(
            f"{sentence_path}: sentence/gold mismatch "
            f"{len(sentences)}/{len(records)}/{len(graphs)}"
        )
    if [record["sentence"].strip() for record in records] != sentences:
        raise ValueError(f"{sentence_path}: sentence ordering does not match gold ordering")
    if not all(validate_amr(graph) for graph in graphs):
        raise ValueError(f"{gold_path}: at least one graph failed PENMAN validation")
    return len(sentences)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formatted-dir", default="data/formatted/vietnamese_v2")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    directory = root / args.formatted_dir
    dev_count = check_pair(directory / "dev.txt", directory / "dev_ground_truth.txt")
    test_count = check_pair(directory / "test.txt", directory / "test_ground_truth.txt")
    stats = directory / "dataset_stats.json"
    print(f"Validated dev={dev_count}, test={test_count}; all gold graphs parse")
    if not stats.exists():
        raise ValueError(f"Missing dataset statistics: {stats}")


if __name__ == "__main__":
    main()
