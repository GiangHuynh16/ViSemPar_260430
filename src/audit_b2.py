#!/usr/bin/env python3
"""Audit the clean B2 baseline against MTUP2 step 1.

Checks:
  1. B2 train/dev examples have the same sentence order as MTUP2.
  2. B2 targets are exactly MTUP2 [STEP 1: AMR_NO_VAR].
  3. B2 examples do not reuse MTUP2 labels, prompts, or final AMR output.
  4. No-var targets do not contain AMR declarations like ``(x / concept)``.
  5. No-var targets do not keep bare variables from the paired full AMR.
"""

import argparse
import re
from pathlib import Path


EXAMPLE_RE = re.compile(
    r"\A<\|im_start\|>system\n(?P<system>.*?)\n<\|im_end\|>\n"
    r"<\|im_start\|>user\n(?P<user>.*?)\n<\|im_end\|>\n"
    r"<\|im_start\|>assistant\n(?P<assistant>.*?)\n<\|im_end\|>\Z",
    re.DOTALL,
)


def read_examples(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        content = f.read().strip()
    if not content:
        return []
    examples = []
    for block in content.split("\n\n<|im_start|>"):
        block = block.strip()
        if not block:
            continue
        if not block.startswith("<|im_start|>"):
            block = "<|im_start|>" + block
        match = EXAMPLE_RE.match(block)
        if not match:
            raise ValueError(f"Bad chat template block in {path}: {block[:200]!r}")
        examples.append(match.groupdict())
    return examples


def extract_b2_no_var(example: dict) -> str:
    text = example["assistant"].strip()
    marker = "AMR_NO_VAR:"
    if marker not in text:
        raise ValueError("Missing AMR_NO_VAR marker in B2 example")
    return text.split(marker, 1)[1].strip()


def extract_mtup2_parts(example: dict) -> tuple[str, str]:
    text = example["assistant"].strip()
    step1 = "[STEP 1: AMR_NO_VAR]"
    step2 = "[STEP 2: AMR_WITH_VAR]"
    if step1 not in text or step2 not in text:
        raise ValueError("Missing MTUP2 step markers")
    no_var = text.split(step1, 1)[1].split(step2, 1)[0].strip()
    full_amr = text.split(step2, 1)[1].strip()
    return no_var, full_amr


def declared_vars(full_amr: str) -> set[str]:
    return set(
        re.findall(
            r"\(\s*([A-Za-zÀ-ỹ][A-Za-zÀ-ỹ0-9_-]*)\s*/\s*[^\s:()]+",
            full_amr,
        )
    )


def declaration_spans(no_var: str) -> list[str]:
    return re.findall(
        r"\(\s*[A-Za-zÀ-ỹ][A-Za-zÀ-ỹ0-9_-]*\s*/\s*[^\s:()]+",
        no_var,
    )


def bare_var_uses(no_var: str, vars_from_full: set[str]) -> list[str]:
    hits = []
    for var in sorted(vars_from_full, key=len, reverse=True):
        if re.search(r"(:[A-Za-z0-9_-]+\s+)" + re.escape(var) + r"(?=\s|\))", no_var):
            hits.append(var)
    return hits


def audit_pair(b2_path: Path, mtup2_path: Path, split_name: str, max_examples: int) -> bool:
    b2_examples = read_examples(b2_path)
    mtup2_examples = read_examples(mtup2_path)

    count_mismatch = len(b2_examples) != len(mtup2_examples)
    order_mismatches = []
    target_mismatches = []
    declarations = []
    bare_refs = []
    b2_label_leaks = []
    prompt_reuse = []
    final_reuse = []

    if not count_mismatch:
        for idx, (b2_ex, mtup2_ex) in enumerate(zip(b2_examples, mtup2_examples), start=1):
            if b2_ex["user"].strip() != mtup2_ex["user"].strip():
                order_mismatches.append((idx, b2_ex["user"].strip(), mtup2_ex["user"].strip()))
                continue

            b2_no_var = extract_b2_no_var(b2_ex)
            mtup2_no_var, full_amr = extract_mtup2_parts(mtup2_ex)

            if b2_no_var != mtup2_no_var:
                target_mismatches.append(idx)

            if "[STEP " in b2_ex["assistant"] or "AMR_WITH_VAR" in b2_ex["assistant"]:
                b2_label_leaks.append(idx)

            if b2_ex["system"] == mtup2_ex["system"]:
                prompt_reuse.append(idx)

            if full_amr in b2_ex["assistant"]:
                final_reuse.append(idx)

            decls = declaration_spans(b2_no_var)
            if decls:
                declarations.append((idx, decls[:3]))

            leaked_vars = bare_var_uses(b2_no_var, declared_vars(full_amr))
            if leaked_vars:
                bare_refs.append((idx, leaked_vars[:10]))

    print(f"\n[{split_name}]")
    print(f"B2 examples:    {len(b2_examples)}")
    print(f"MTUP2 examples: {len(mtup2_examples)}")
    print(f"Example-count mismatch:       {int(count_mismatch)}")
    print(f"Sentence-order mismatches:     {len(order_mismatches)}")
    print(f"B2 == MTUP2 step 1 mismatches: {len(target_mismatches)}")
    print(f"B2 MTUP label leaks:           {len(b2_label_leaks)}")
    print(f"B2 reused MTUP2 system prompt: {len(prompt_reuse)}")
    print(f"B2 reused final full AMR:      {len(final_reuse)}")
    print(f"No-var declaration violations: {len(declarations)}")
    print(f"Bare full-AMR variable refs:   {len(bare_refs)}")

    samples = [
        ("order", order_mismatches),
        ("target", target_mismatches),
        ("label-leak", b2_label_leaks),
        ("prompt-reuse", prompt_reuse),
        ("final-reuse", final_reuse),
        ("declaration", declarations),
        ("bare-ref", bare_refs),
    ]
    for label, rows in samples:
        if not rows:
            continue
        print(f"\nFirst {label} examples:")
        for row in rows[:max_examples]:
            print(f"  {row}")

    return not any([
        count_mismatch,
        order_mismatches,
        target_mismatches,
        b2_label_leaks,
        prompt_reuse,
        final_reuse,
        declarations,
        bare_refs,
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--b2", default="data/formatted/vietnamese_v2/B2_train.txt")
    parser.add_argument("--mtup2", default="data/formatted/vietnamese_v2/MTUP2_train.txt")
    parser.add_argument("--b2_dev", default=None)
    parser.add_argument("--mtup2_dev", default=None)
    parser.add_argument("--max_examples", type=int, default=5)
    args = parser.parse_args()

    ok = audit_pair(Path(args.b2), Path(args.mtup2), "train", args.max_examples)
    if args.b2_dev or args.mtup2_dev:
        if not args.b2_dev or not args.mtup2_dev:
            raise SystemExit("--b2_dev and --mtup2_dev must be provided together")
        ok = audit_pair(Path(args.b2_dev), Path(args.mtup2_dev), "dev", args.max_examples) and ok

    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
