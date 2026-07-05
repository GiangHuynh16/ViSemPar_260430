#!/usr/bin/env python3
from __future__ import annotations

"""Profile token lengths and expected training work without using a GPU."""

import argparse
import math
import statistics
import sys
import time
from pathlib import Path

from transformers import AutoTokenizer

from config_utils import load_yaml_config


def read_chat_examples(path: Path) -> list[str]:
    with open(path, encoding="utf-8") as f:
        content = f.read()

    examples = []
    for raw in content.strip().split("\n\n<|im_start|>"):
        text = raw.strip()
        if not text:
            continue
        if not text.startswith("<|im_start|>"):
            text = "<|im_start|>" + text
        if not text.endswith("<|im_end|>"):
            text += "\n<|im_end|>"
        examples.append(text)
    return examples


def percentile(values: list[int], pct: float) -> int:
    if not values:
        return 0
    index = math.ceil((pct / 100.0) * len(values)) - 1
    index = min(max(index, 0), len(values) - 1)
    return sorted(values)[index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/vietnamese_v2/MTUP2.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-per-device", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    cfg = load_yaml_config(args.config)
    train_path = root / "data" / "formatted" / cfg["data"]["train"]
    tr_cfg = cfg["training"]
    max_length = int(tr_cfg.get("max_length", 2048))
    batch_size = int(args.batch_per_device or tr_cfg.get("batch_per_device", 1))
    grad_accum = int(args.grad_accum or tr_cfg.get("grad_accum", 16))
    epochs = int(tr_cfg.get("epochs", 30))
    base_batch_size = int(tr_cfg.get("batch_per_device", 1))
    base_grad_accum = int(tr_cfg.get("grad_accum", 16))

    print(f"Config: {args.config}")
    print(f"Train file: {train_path}")
    print(f"Model: {cfg['model']['name']}")

    start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"], trust_remote_code=True)
    examples = read_chat_examples(train_path)
    lengths = []
    for text in examples:
        tokens = tokenizer(
            text,
            truncation=False,
            padding=False,
            add_special_tokens=False,
        )
        lengths.append(len(tokens["input_ids"]))
    elapsed = time.perf_counter() - start

    truncated = sum(1 for length in lengths if length > max_length)
    micro_batches = math.ceil(len(examples) / batch_size)
    optimizer_steps = math.ceil(micro_batches / grad_accum)
    total_micro_batches = micro_batches * epochs
    total_optimizer_steps = optimizer_steps * epochs

    print()
    print(f"Examples: {len(examples)}")
    print(f"Tokenization seconds: {elapsed:.2f}")
    print(f"max_length: {max_length}")
    print(f"min/p50/p90/p95/p99/max tokens: "
          f"{min(lengths)} / {percentile(lengths, 50)} / "
          f"{percentile(lengths, 90)} / {percentile(lengths, 95)} / "
          f"{percentile(lengths, 99)} / {max(lengths)}")
    print(f"mean tokens: {statistics.mean(lengths):.1f}")
    print(f"truncated examples: {truncated}")
    print()
    print(f"batch_per_device: {batch_size}")
    print(f"grad_accum: {grad_accum}")
    print(f"effective batch: {batch_size * grad_accum}")
    if (batch_size, grad_accum) != (base_batch_size, base_grad_accum):
        print(
            "base config effective batch: "
            f"{base_batch_size * base_grad_accum} "
            f"({base_batch_size} x {base_grad_accum})"
        )
    print(f"micro-batches/epoch: {micro_batches}")
    print(f"optimizer steps/epoch: {optimizer_steps}")
    print(f"total micro-batches for {epochs} epochs: {total_micro_batches}")
    print(f"total optimizer steps for {epochs} epochs: {total_optimizer_steps}")
    print()
    if cfg.get("model", {}).get("flash_attn"):
        try:
            import flash_attn  # noqa: F401
        except ImportError:
            print("WARNING: config requests FlashAttention, but package flash_attn is not installed.")
        else:
            print("FlashAttention package is installed.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
