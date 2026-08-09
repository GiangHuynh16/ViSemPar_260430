#!/usr/bin/env python3
"""
Data pipeline for ViSemPar_260430.

Steps:
  1. Load train_amr_1 + train_amr_2, deduplicate by sentence
  2. Build formatted chat examples for B1, B2, MTUP2, MTUP3, and MTUP4
  3. Save formatted/{method}_train.txt (ready to feed into train.py)

Methods: B1, B2, MTUP2, MTUP3, MTUP4
  B1    – direct: sentence → AMR
  B2    – no-reentrancy skeleton baseline
  MTUP2 – 2 steps: no-var skeleton → AMR
  MTUP3 – 3 steps: concepts → no-var skeleton → AMR
  MTUP4 – 4 steps: concepts → relations → no-var skeleton → AMR

Usage:
    python src/data_pipeline.py [--verify] [--amr1_weight 1] [--amr2_weight 1]
"""

import re
import json
import argparse
import random
from pathlib import Path

import penman

from amr_utils import canonicalize_amr, validate_amr
from prompts import get_system_prompt


# ─────────────────────────────────────────────────────────────────────────────
# Frozen system prompts (identical for ALL methods per plan)
# ─────────────────────────────────────────────────────────────────────────────
# Prompt text is imported from src/prompts.py so training data and inference
# cannot drift apart.

# ─────────────────────────────────────────────────────────────────────────────
# AMR processing helpers
# ─────────────────────────────────────────────────────────────────────────────

def fix_polarity_format(text: str) -> str:
    return re.sub(r':polarity\s*\([^)]*\bno\b[^)]*\)', ':polarity -', text)


def _extract_var_to_concept(amr: str) -> dict:
    """Build variable -> concept map from AMR node definitions."""
    var_to_concept = {}
    for var, concept in re.findall(
        r'\(\s*([A-Za-zÀ-ỹ][A-Za-zÀ-ỹ0-9_-]*)\s*/\s*([^():]+?)(?=\s+:[A-Za-z0-9_-]+|\s*\))',
        amr,
    ):
        var_to_concept[var] = concept.strip()
    return var_to_concept


def strip_variables(amr: str, mode: str = 'clean') -> str:
    """Convert full AMR to a skeleton; reentrancies are intentionally lost."""
    if mode != 'clean':
        raise ValueError(f"Unsupported no-var mode: {mode}")

    var_to_concept = _extract_var_to_concept(amr)
    result = re.sub(r'\(\s*\w+\s*/\s*', '(', amr)

    for var, concept in sorted(var_to_concept.items(), key=lambda x: len(x[0]), reverse=True):
        result = re.sub(
            r'(:[A-Za-z0-9_-]+\s+)' + re.escape(var) + r'(?=\s|\))',
            rf'\1({concept})',
            result,
        )
    result = re.sub(
        r'(:[A-Za-z0-9_-]+\s+)([A-Za-zÀ-ỹ][A-Za-zÀ-ỹ0-9_-]{0,2})(?=\s|\))',
        r'\1(amr-unknown)',
        result,
    )

    return result


def concept_list(amr: str) -> str:
    """Render graph concepts as a deterministic intermediate target.

    This is not an AMR and is never evaluated as one.  Concepts remain in
    their canonical PENMAN spelling, including underscores used by the source
    corpus for multiword concepts.
    """
    graph = penman.decode(canonicalize_amr(amr))
    concepts = [target for _source, role, target in graph.triples
                if role == ':instance']
    return '\n'.join(f'concept_{index} | {concept}'
                     for index, concept in enumerate(concepts, start=1))


def relation_list(amr: str) -> str:
    """Render graph edges in concept space as a deterministic intermediate target."""
    graph = penman.decode(canonicalize_amr(amr))
    instance_map = {source: target for source, role, target in graph.triples
                    if role == ':instance'}
    lines = []
    for source, role, target in graph.triples:
        if role == ':instance':
            continue
        source_label = instance_map.get(source, source)
        target_label = instance_map.get(target, target)
        lines.append(f'{source_label} | {role} | {target_label}')
    return '\n'.join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Chat example builders per method
# ─────────────────────────────────────────────────────────────────────────────

def _wrap(system: str, user: str, assistant: str) -> str:
    return (f"<|im_start|>system\n{system}\n<|im_end|>\n"
            f"<|im_start|>user\n{user}\n<|im_end|>\n"
            f"<|im_start|>assistant\n{assistant}\n<|im_end|>")


def make_B1(sentence: str, amr: str, no_var_mode: str = 'clean',
            language: str = 'vi') -> str:
    """B1: direct sentence → AMR."""
    return _wrap(get_system_prompt(language, 'B1'), sentence, f"AMR: {amr}")


def make_B2(sentence: str, amr: str, no_var_mode: str = 'clean',
            language: str = 'vi') -> str:
    """B2: sentence → no-reentrancy AMR skeleton."""
    no_var = strip_variables(amr, mode=no_var_mode)
    return _wrap(get_system_prompt(language, 'B2'), sentence, f"AMR_NO_VAR: {no_var}")


def make_MTUP2(sentence: str, amr: str, no_var_mode: str = 'clean',
               language: str = 'vi') -> str:
    """MTUP2: no-var → full AMR (closer intermediate, easier for model to add variables)."""
    no_var = strip_variables(amr, mode=no_var_mode)
    assistant = f"[STEP 1: AMR_NO_VAR]\n{no_var}\n\n[STEP 2: AMR_WITH_VAR]\n{amr}"
    return _wrap(get_system_prompt(language, 'MTUP2'), sentence, assistant)


def make_MTUP3(sentence: str, amr: str, no_var_mode: str = 'clean',
               language: str = 'vi') -> str:
    """MTUP3A: concepts → no-var skeleton → full AMR."""
    concepts = concept_list(amr)
    no_var = strip_variables(amr, mode=no_var_mode)
    assistant = (
        f'[STEP 1: CONCEPTS]\n{concepts}\n\n'
        f'[STEP 2: AMR_NO_VAR]\n{no_var}\n\n'
        f'[STEP 3: AMR_WITH_VAR]\n{amr}'
    )
    return _wrap(get_system_prompt(language, 'MTUP3'), sentence, assistant)


def make_MTUP4(sentence: str, amr: str, no_var_mode: str = 'clean',
               language: str = 'vi') -> str:
    """MTUP4: concepts → relations → no-var skeleton → full AMR."""
    concepts = concept_list(amr)
    relations = relation_list(amr)
    no_var = strip_variables(amr, mode=no_var_mode)
    assistant = (
        f'[STEP 1: CONCEPTS]\n{concepts}\n\n'
        f'[STEP 2: RELATIONS]\n{relations}\n\n'
        f'[STEP 3: AMR_NO_VAR]\n{no_var}\n\n'
        f'[STEP 4: AMR_WITH_VAR]\n{amr}'
    )
    return _wrap(get_system_prompt(language, 'MTUP4'), sentence, assistant)


BUILDERS = {
    'B1': make_B1,
    'B2': make_B2,
    'MTUP2': make_MTUP2,
    'MTUP3': make_MTUP3,
    'MTUP4': make_MTUP4,
}


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def parse_amr_file(path: str):
    with open(path, encoding='utf-8') as f:
        content = f.read()
    examples = []
    for block in re.split(r'(?m)(?=^#\s*::snt\s+)', content):
        block = block.strip()
        if not block:
            continue
        sentence = None
        header_match = re.match(r'#\s*::snt\s+(.+)', block)
        if header_match:
            sentence = header_match.group(1).strip()
        graph_text = '\n'.join(line for line in block.split('\n')[1:] if not line.startswith('#'))
        graphs = []
        start = None
        depth = 0
        quoted = False
        for index, char in enumerate(graph_text):
            if char == '"':
                quoted = not quoted
            if quoted:
                continue
            if char == '(':
                if depth == 0:
                    start = index
                depth += 1
            elif char == ')' and depth:
                depth -= 1
                if depth == 0 and start is not None:
                    graphs.append(graph_text[start:index + 1])
                    start = None
        # A malformed source block can contain a stray preceding graph. The
        # final top-level graph is the one paired with its sentence header.
        if sentence and graphs:
            examples.append({'sentence': sentence, 'amr': graphs[-1]})
    return examples


def dedup(examples):
    """Deduplicate by sentence (keep first occurrence)."""
    seen = set()
    out = []
    for ex in examples:
        key = ex['sentence'].strip()
        if key not in seen:
            seen.add(key)
            out.append(ex)
    return out


def split_train_dev(examples, dev_split: float, seed: int):
    """Deterministically split raw examples before weighting/formatting."""
    if dev_split <= 0:
        return examples, []
    if dev_split >= 1:
        raise ValueError("--dev_split must be in [0, 1)")
    rng = random.Random(seed)
    indices = list(range(len(examples)))
    rng.shuffle(indices)
    n_dev = max(1, int(round(len(examples) * dev_split)))
    dev_idx = set(indices[:n_dev])
    train = [ex for i, ex in enumerate(examples) if i not in dev_idx]
    dev = [ex for i, ex in enumerate(examples) if i in dev_idx]
    return train, dev


def write_amr_blocks(path: Path, examples: list):
    """Write raw sentence/gold AMR blocks for dev Smatch evaluation."""
    blocks = []
    for ex in examples:
        amr = canonicalize_amr(fix_polarity_format(ex['amr']))
        if not validate_amr(amr):
            raise ValueError(f"Invalid canonical AMR: {ex['sentence']}")
        blocks.append(f"# ::snt {ex['sentence']}\n{amr}")
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(blocks))


def write_sentence_lines(path: Path, examples: list):
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(ex['sentence'] for ex in examples))


def write_aligned_public_test(raw_sentence_path: Path, raw_gold_path: Path,
                              sentence_path: Path, gold_path: Path) -> list[str]:
    """Write the maximal ordered test/gold intersection and return dropped text."""
    sentences = [line.strip() for line in raw_sentence_path.read_text(encoding='utf-8').splitlines()
                 if line.strip() and not line.startswith('#')]
    gold = parse_amr_file(str(raw_gold_path))
    kept_sentences = []
    kept_gold = []
    dropped = []
    gold_index = 0
    for sentence in sentences:
        if gold_index < len(gold) and sentence == gold[gold_index]['sentence'].strip():
            kept_sentences.append(sentence)
            kept_gold.append(gold[gold_index])
            gold_index += 1
        else:
            dropped.append(sentence)
    if gold_index != len(gold):
        raise ValueError(f"Unmatched gold entries remain: {len(gold) - gold_index}")
    write_sentence_lines(sentence_path, [{'sentence': s} for s in kept_sentences])
    write_amr_blocks(gold_path, kept_gold)
    return dropped


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset(method: str, examples_1, examples_2,
                  amr1_weight: int, amr2_weight: int,
                  no_var_mode: str = 'clean',
                  language: str = 'vi') -> list:
    builder = BUILDERS[method]
    chat_examples = []

    def add(exs, weight):
        for _ in range(weight):
            for ex in exs:
                amr = canonicalize_amr(fix_polarity_format(ex['amr']))
                if not validate_amr(amr):
                    raise ValueError(f"Invalid canonical AMR: {ex['sentence']}")
                result = builder(ex['sentence'], amr, no_var_mode=no_var_mode,
                                 language=language)
                if result:
                    chat_examples.append(result)

    add(examples_1, amr1_weight)
    add(examples_2, amr2_weight)
    return chat_examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--amr1_weight', type=int, default=1)
    parser.add_argument('--amr2_weight', type=int, default=1)
    parser.add_argument('--methods', nargs='+',
                        default=['B1', 'B2', 'MTUP2', 'MTUP3', 'MTUP4'],
                        choices=sorted(BUILDERS))
    parser.add_argument('--no_var_mode', choices=['clean'], default='clean',
                        help='Active no-reentrancy skeleton representation.')
    parser.add_argument('--raw_dir', default='data/raw',
                        help='Directory containing train_amr_1.txt and train_amr_2.txt')
    parser.add_argument('--formatted_dir', default='data/formatted',
                        help='Output directory for formatted training files')
    parser.add_argument('--no_dedup', action='store_true',
                        help='Keep all AMR blocks instead of deduplicating by sentence')
    parser.add_argument('--language', choices=['vi'], default='vi')
    parser.add_argument('--dev_split', type=float, default=0.0,
                        help='Fraction of examples to hold out as dev set (e.g. 0.1). '
                             'Written to {method}_dev.txt. 0 = disabled.')
    parser.add_argument('--dev_seed', type=int, default=42,
                        help='Random seed for deterministic train/dev split.')
    parser.add_argument('--verify', action='store_true',
                        help='Print 2 examples per method and exit')
    args = parser.parse_args()

    root = Path(__file__).parent.parent
    raw_dir = root / args.raw_dir
    fmt_dir = root / args.formatted_dir
    fmt_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    ex1_raw = parse_amr_file(str(raw_dir / 'train_amr_1.txt'))
    ex2_raw = parse_amr_file(str(raw_dir / 'train_amr_2.txt'))

    all_raw = ex1_raw + ex2_raw
    examples = all_raw if args.no_dedup else dedup(all_raw)

    print(f"  train_amr_1: {len(ex1_raw)} raw")
    print(f"  train_amr_2: {len(ex2_raw)} raw")
    print(f"  unified:     {len(examples)} after dedup")
    if args.amr1_weight != 1 or args.amr2_weight != 1:
        raise ValueError("Source-specific weighting is disabled for the unified manifest; use weight 1.")
    dropped_test = []
    if args.dev_split > 0:
        print(f"  dev split:   {args.dev_split:.3f} with seed {args.dev_seed}")
    print()

    if args.dev_split > 0:
        train_examples_raw, dev_raw_examples = split_train_dev(
            examples, args.dev_split, args.dev_seed)
        write_sentence_lines(fmt_dir / 'dev.txt', dev_raw_examples)
        write_amr_blocks(fmt_dir / 'dev_ground_truth.txt', dev_raw_examples)
        write_sentence_lines(fmt_dir / 'train.txt', train_examples_raw)
        print(f"  train raw:   {len(train_examples_raw)} sentences")
        print(f"  dev raw:     {len(dev_raw_examples)} sentences")
    else:
        train_examples_raw, dev_raw_examples = examples, []

    dropped_test = write_aligned_public_test(
        raw_dir / 'public_test.txt', raw_dir / 'public_test_ground_truth.txt',
        fmt_dir / 'test.txt', fmt_dir / 'test_ground_truth.txt')
    print(f"  test aligned: {len(parse_amr_file(str(raw_dir / 'public_test_ground_truth.txt')))}; dropped={len(dropped_test)}")

    stats = {}
    for method in args.methods:
        if args.dev_split > 0:
            train_examples = build_dataset(method, train_examples_raw, [], 1, 0,
                                           no_var_mode=args.no_var_mode,
                                           language=args.language)
            dev_examples = build_dataset(method, dev_raw_examples, [], 1, 0,
                                         no_var_mode=args.no_var_mode,
                                         language=args.language)
            dev_path = fmt_dir / f"{method}_dev.txt"
            with open(dev_path, 'w', encoding='utf-8') as f:
                f.write('\n\n'.join(dev_examples))
            print(f"  {method}: {len(train_examples)} train + {len(dev_examples)} dev -> {fmt_dir}")
        else:
            train_examples = build_dataset(method, train_examples_raw, [], 1, 0,
                                           no_var_mode=args.no_var_mode,
                                           language=args.language)

        out_path = fmt_dir / f"{method}_train.txt"
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write('\n\n'.join(train_examples))
        stats[method] = len(train_examples)

        if args.verify:
            print(f"\n--- {method} sample 1 ---")
            print(train_examples[0][:600])
            print(f"\n--- {method} sample 2 ---")
            print(train_examples[-1][:600])
            print()
        elif args.dev_split == 0:
            print(f"  {method}: {len(train_examples)} examples -> {out_path}")

    # Save stats
    stats_path = fmt_dir / 'dataset_stats.json'
    with open(stats_path, 'w') as f:
        json.dump({
            'raw_examples': len(all_raw),
            'dedup_examples': len(examples),
            'weights': {'amr1': args.amr1_weight, 'amr2': args.amr2_weight},
            'no_var_mode': args.no_var_mode,
            'dedup': not args.no_dedup,
            'language': args.language,
            'dev_split': args.dev_split,
            'dev_seed': args.dev_seed if args.dev_split > 0 else None,
            'methods': stats,
            'train_examples': len(train_examples_raw),
            'dev_examples': len(dev_raw_examples),
            'test_examples': len(fmt_dir.joinpath('test.txt').read_text(encoding='utf-8').splitlines()) if (fmt_dir / 'test.txt').exists() else 0,
            'test_dropped_sentences': dropped_test,
        }, f, indent=2)
    print(f"\nStats saved to {stats_path}")


if __name__ == '__main__':
    main()
