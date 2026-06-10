#!/usr/bin/env python3
"""Audit the clean B2 baseline against MTUP2 step 1.

Checks:
  1. B2 targets are exactly MTUP2 [STEP 1: AMR_NO_VAR] for the same examples.
  2. No-var targets do not contain AMR variable declarations like ``(x / concept)``.
  3. No-var targets do not keep bare variables from the paired full AMR.
"""

import argparse
import re
from pathlib import Path


ASSISTANT_RE = re.compile(
    r"<\|im_start\|>assistant\n(?P<assistant>.*?)\n<\|im_end\|>",
    re.DOTALL,
)


def read_examples(path: Path) -> list[str]:
    with open(path, encoding="utf-8") as f:
        content = f.read().strip()
    if not content:
        return []
    return [
        block if block.startswith("<|im_start|>") else "<|im_start|>" + block
        for block in content.split("\n\n<|im_start|>")
        if block.strip()
    ]


def assistant_text(example: str) -> str:
    match = ASSISTANT_RE.search(example)
    if not match:
        raise ValueError("Missing assistant block")
    return match.group("assistant").strip()


def extract_b2_no_var(example: str) -> str:
    text = assistant_text(example)
    marker = "AMR_NO_VAR:"
    if marker not in text:
        raise ValueError("Missing AMR_NO_VAR marker in B2 example")
    return text.split(marker, 1)[1].strip()


def extract_mtup2_parts(example: str) -> tuple[str, str]:
    text = assistant_text(example)
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--b2", default="data/formatted/vietnamese_v2/B2_train.txt")
    parser.add_argument("--mtup2", default="data/formatted/vietnamese_v2/MTUP2_train.txt")
    parser.add_argument("--max_examples", type=int, default=5)
    args = parser.parse_args()

    b2_path = Path(args.b2)
    mtup2_path = Path(args.mtup2)
    b2_examples = read_examples(b2_path)
    mtup2_examples = read_examples(mtup2_path)

    if len(b2_examples) != len(mtup2_examples):
        raise SystemExit(
            f"Example-count mismatch: B2={len(b2_examples)} MTUP2={len(mtup2_examples)}"
        )

    mismatches = []
    declarations = []
    bare_refs = []

    for idx, (b2_ex, mtup2_ex) in enumerate(zip(b2_examples, mtup2_examples), start=1):
        b2_no_var = extract_b2_no_var(b2_ex)
        mtup2_no_var, full_amr = extract_mtup2_parts(mtup2_ex)

        if b2_no_var != mtup2_no_var:
            mismatches.append(idx)

        decls = declaration_spans(b2_no_var)
        if decls:
            declarations.append((idx, decls[:3]))

        leaked_vars = bare_var_uses(b2_no_var, declared_vars(full_amr))
        if leaked_vars:
            bare_refs.append((idx, leaked_vars[:10]))

    print(f"B2 examples:    {len(b2_examples)}")
    print(f"MTUP2 examples: {len(mtup2_examples)}")
    print(f"B2 == MTUP2 step 1 mismatches: {len(mismatches)}")
    print(f"No-var declaration violations: {len(declarations)}")
    print(f"Bare full-AMR variable refs:   {len(bare_refs)}")

    for label, rows in [
        ("mismatch", mismatches),
        ("declaration", declarations),
        ("bare-ref", bare_refs),
    ]:
        if not rows:
            continue
        print(f"\nFirst {label} examples:")
        for row in rows[: args.max_examples]:
            print(f"  {row}")

    if mismatches or declarations or bare_refs:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
