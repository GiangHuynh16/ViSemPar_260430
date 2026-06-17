#!/usr/bin/env python3
from __future__ import annotations

"""Prepare the public Little Prince AMR corpus for English experiments."""

import argparse
import re
from pathlib import Path


def read_blocks(path: Path) -> list[str]:
    content = path.read_text(encoding="utf-8")
    return [
        block.strip()
        for block in content.split("\n\n")
        if re.search(r"^# ::snt\s+", block, flags=re.MULTILINE)
    ]


def block_id(block: str) -> str | None:
    match = re.search(r"^# ::id\s+(\S+)", block, flags=re.MULTILINE)
    return match.group(1) if match else None


def write_blocks(path: Path, blocks: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")


def write_sentences(path: Path, blocks: list[str]) -> None:
    sentences = []
    for block in blocks:
        match = re.search(r"^# ::snt\s+(.+)$", block, flags=re.MULTILINE)
        if match:
            sentences.append(match.group(1).strip())
    path.write_text("\n".join(sentences) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--out_dir", default="data/english/little_prince/raw")
    args = parser.parse_args()

    full_blocks = read_blocks(Path(args.full))
    train_blocks = read_blocks(Path(args.train))
    test_blocks = read_blocks(Path(args.test))

    train_ids = {block_id(block) for block in train_blocks}
    test_ids = {block_id(block) for block in test_blocks}
    train_ids.discard(None)
    test_ids.discard(None)

    dev_blocks = [
        block for block in full_blocks
        if block_id(block) not in train_ids and block_id(block) not in test_ids
    ]

    out_dir = Path(args.out_dir)
    write_blocks(out_dir / "train_amr_1.txt", train_blocks)
    write_blocks(out_dir / "train_amr_2.txt", [])
    write_blocks(out_dir / "dev_ground_truth.txt", dev_blocks)
    write_blocks(out_dir / "test_ground_truth.txt", test_blocks)
    write_sentences(out_dir / "dev.txt", dev_blocks)
    write_sentences(out_dir / "test.txt", test_blocks)

    print(f"Saved Little Prince AMR to {out_dir}")
    print(f"  train: {len(train_blocks)}")
    print(f"  dev:   {len(dev_blocks)}")
    print(f"  test:  {len(test_blocks)}")


if __name__ == "__main__":
    main()
