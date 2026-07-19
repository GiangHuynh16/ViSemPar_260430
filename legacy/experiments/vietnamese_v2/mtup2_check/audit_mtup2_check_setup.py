#!/usr/bin/env python3
"""Pre-flight audit for the MTUP2_CHECK Vietnamese setup."""

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config_utils import load_yaml_config
from data_pipeline import make_amr_check, parse_amr_file, strip_variables
from postprocess import postprocess_amr
from prompts import get_system_prompt


EXAMPLE_RE = re.compile(
    r"\A<\|im_start\|>system\n(?P<system>.*?)\n<\|im_end\|>\n"
    r"<\|im_start\|>user\n(?P<user>.*?)\n<\|im_end\|>\n"
    r"<\|im_start\|>assistant\n(?P<assistant>.*?)\n<\|im_end\|>\Z",
    re.DOTALL,
)

STEP1 = "[STEP 1: AMR_NO_VAR]"
STEP2 = "[STEP 2: CHECK]"
STEP3 = "[STEP 3: AMR_WITH_VAR]"
CHECK_KEYS = ["ROOT:", "CONCEPTS:", "RELATIONS:", "REENTRANCIES:", "VARIABLE_CHECK:"]


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


def read_sentences(path: Path) -> set[str]:
    sentences = set()
    if not path.exists():
        return sentences
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                sentences.add(line)
    return sentences


def read_raw_amr_sentences(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {ex["sentence"].strip() for ex in parse_amr_file(str(path))}


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def extract_steps(assistant: str) -> tuple[str, str, str]:
    for marker in [STEP1, STEP2, STEP3]:
        if assistant.count(marker) != 1:
            raise ValueError(f"Assistant output must contain exactly one {marker}")
    if not (assistant.index(STEP1) < assistant.index(STEP2) < assistant.index(STEP3)):
        raise ValueError("MTUP2_CHECK steps must appear in STEP 1, STEP 2, STEP 3 order")

    no_var = assistant.split(STEP1, 1)[1].split(STEP2, 1)[0].strip()
    check = assistant.split(STEP2, 1)[1].split(STEP3, 1)[0].strip()
    full = assistant.split(STEP3, 1)[1].strip()
    return no_var, check, full


def declared_vars(full_amr: str) -> set[str]:
    return set(
        re.findall(
            r"\(\s*([A-Za-zÀ-ỹ][A-Za-zÀ-ỹ0-9_-]*)\s*/\s*[^\s:()]+",
            full_amr,
        )
    )


def no_var_declarations(no_var: str) -> list[str]:
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


def check_has_required_keys(check: str) -> bool:
    lines = [line.strip() for line in check.splitlines() if line.strip()]
    if len(lines) != len(CHECK_KEYS):
        return False
    return all(line.startswith(key) for line, key in zip(lines, CHECK_KEYS))


def audit_file(path: Path, expected_prompt: str, no_var_mode: str) -> dict:
    examples = read_examples(path)
    stats = {
        "path": str(path),
        "examples": len(examples),
        "prompt_mismatch": 0,
        "step_mismatch": 0,
        "check_format_mismatch": 0,
        "check_value_mismatch": 0,
        "no_var_declarations": 0,
        "bare_var_refs": 0,
        "bad_full_amr": 0,
        "postprocess_changes": 0,
        "stale_headers": 0,
        "sentences": set(),
    }

    first_errors = []
    for idx, ex in enumerate(examples, start=1):
        stats["sentences"].add(ex["user"].strip())
        if ex["system"] != expected_prompt:
            stats["prompt_mismatch"] += 1
            first_errors.append((idx, "prompt_mismatch"))

        if "Task 1:" in ex["assistant"] or "Task 2:" in ex["assistant"]:
            stats["stale_headers"] += 1
            first_errors.append((idx, "stale_task_header"))

        no_var, check, full = extract_steps(ex["assistant"])
        expected_no_var = strip_variables(full, mode=no_var_mode)
        if normalize_space(no_var) != normalize_space(expected_no_var):
            stats["step_mismatch"] += 1
            first_errors.append((idx, "step1_not_strip_variables_step3"))

        if not check_has_required_keys(check):
            stats["check_format_mismatch"] += 1
            first_errors.append((idx, "bad_check_format"))

        expected_check = make_amr_check(full)
        if normalize_space(check) != normalize_space(expected_check):
            stats["check_value_mismatch"] += 1
            first_errors.append((idx, "check_not_deterministic_from_step3"))

        decls = no_var_declarations(no_var)
        if decls:
            stats["no_var_declarations"] += 1
            first_errors.append((idx, f"no_var_declaration={decls[:2]}"))

        leaked = bare_var_uses(no_var, declared_vars(full))
        if leaked:
            stats["bare_var_refs"] += 1
            first_errors.append((idx, f"bare_var_refs={leaked[:5]}"))

        if not full.startswith("(") or "/" not in full or full.count("(") != full.count(")"):
            stats["bad_full_amr"] += 1
            first_errors.append((idx, "bad_full_amr"))

        if normalize_space(postprocess_amr(full)) != normalize_space(full):
            stats["postprocess_changes"] += 1

    stats["first_errors"] = first_errors[:10]
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/vietnamese_v2/mtup2_check/MTUP2_CHECK.yaml")
    args = parser.parse_args()

    root = Path(__file__).parent.parent
    cfg = load_yaml_config(args.config)
    if cfg["method"] != "MTUP2_CHECK":
        raise SystemExit("This audit is for MTUP2_CHECK only.")

    expected_prompt = get_system_prompt(cfg.get("language", "vi"), cfg["method"])
    train_path = root / "data" / "formatted" / cfg["data"]["train"]
    dev_path = root / "data" / "formatted" / cfg["data"]["dev_formatted"]
    test_path = root / cfg["data"]["test"]
    raw_dir = root / cfg.get("data", {}).get("raw_dir", "data/raw")
    no_var_mode = cfg.get("no_var_mode", "clean")

    train_stats = audit_file(train_path, expected_prompt, no_var_mode)
    dev_stats = audit_file(dev_path, expected_prompt, no_var_mode)
    public_test_sentences = read_sentences(test_path)
    raw_train_sentences = (
        read_raw_amr_sentences(raw_dir / "train_amr_1.txt")
        | read_raw_amr_sentences(raw_dir / "train_amr_2.txt")
    )
    train_dev_overlaps = train_stats["sentences"] & dev_stats["sentences"]
    train_test_overlaps = train_stats["sentences"] & public_test_sentences
    dev_test_overlaps = dev_stats["sentences"] & public_test_sentences
    raw_test_overlaps = raw_train_sentences & public_test_sentences

    for stats in [train_stats, dev_stats]:
        print(f"\n{stats['path']}")
        for key in [
            "examples",
            "prompt_mismatch",
            "step_mismatch",
            "check_format_mismatch",
            "check_value_mismatch",
            "no_var_declarations",
            "bare_var_refs",
            "bad_full_amr",
            "postprocess_changes",
            "stale_headers",
        ]:
            print(f"  {key}: {stats[key]}")
        if stats["first_errors"]:
            print(f"  first_errors: {stats['first_errors']}")

    print(f"\ntrain_dev_exact_sentence_overlaps: {len(train_dev_overlaps)}")
    print(f"train_public_test_exact_sentence_overlaps: {len(train_test_overlaps)}")
    print(f"dev_public_test_exact_sentence_overlaps: {len(dev_test_overlaps)}")
    print(f"raw_train_public_test_exact_sentence_overlaps: {len(raw_test_overlaps)}")

    hard_fail = (
        train_stats["prompt_mismatch"]
        or dev_stats["prompt_mismatch"]
        or train_stats["step_mismatch"]
        or dev_stats["step_mismatch"]
        or train_stats["check_format_mismatch"]
        or dev_stats["check_format_mismatch"]
        or train_stats["check_value_mismatch"]
        or dev_stats["check_value_mismatch"]
        or train_stats["no_var_declarations"]
        or dev_stats["no_var_declarations"]
        or train_stats["bare_var_refs"]
        or dev_stats["bare_var_refs"]
        or train_stats["bad_full_amr"]
        or dev_stats["bad_full_amr"]
        or train_stats["postprocess_changes"]
        or dev_stats["postprocess_changes"]
        or train_stats["stale_headers"]
        or dev_stats["stale_headers"]
        or train_dev_overlaps
        or train_test_overlaps
        or dev_test_overlaps
        or raw_test_overlaps
    )
    if hard_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
