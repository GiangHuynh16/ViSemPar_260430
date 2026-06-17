#!/usr/bin/env python3
from __future__ import annotations

"""Normalise raw Brazilian Portuguese AMR corpora and split into train/dev/test sets."""

import re
from pathlib import Path

ALIGN_RE = re.compile(r'~e\.\d+(?:,\d+)*')


def strip_alignment_markers(amr_line: str) -> str:
    """Remove ~e.N,M alignment markers from an AMR token."""
    return ALIGN_RE.sub('', amr_line)


def normalise_block(raw_block: str) -> dict | None:
    """
    Parse one AMR block (separated by double newlines) into {'sentence': ..., 'amr': ...}.

    For LPP blocks: reads `# ::snt <text>` as the sentence.
    For AMRNews blocks: reads `# ::tok-pt <text>` as the sentence.
    Strips ::tok-en, ::alignments-*, and other # comment lines.
    Strips ~e.N alignment markers from AMR content lines.
    Returns None if no sentence or AMR found.
    """
    sentence = None
    amr_lines = []
    for line in raw_block.strip().split('\n'):
        line = line.rstrip()
        if not line:
            continue
        snt_match = re.match(r'#\s*::snt\s+(.+)', line)
        tok_match = re.match(r'#\s*::tok-pt\s+(.+)', line)
        if snt_match:
            sentence = snt_match.group(1).strip()
        elif tok_match:
            sentence = tok_match.group(1).strip()
        elif line.startswith('#'):
            continue  # drop ::id, ::tok-en, ::alignments-*, etc.
        else:
            amr_lines.append(strip_alignment_markers(line))
    amr = '\n'.join(amr_lines).strip()
    if sentence and amr:
        return {'sentence': sentence, 'amr': amr}
    return None


def parse_corpus(path: str) -> list[dict]:
    """Read a raw AMR file, return list of {'sentence', 'amr'} dicts."""
    text = Path(path).read_text(encoding='utf-8')
    blocks = text.strip().split('\n\n')
    examples = []
    for block in blocks:
        result = normalise_block(block)
        if result:
            examples.append(result)
    return examples


def write_amr_file(examples: list[dict], path: Path) -> None:
    """Write examples in AMR block format with # ::snt headers."""
    blocks = []
    for ex in examples:
        blocks.append(f"# ::snt {ex['sentence']}\n{ex['amr']}")
    path.write_text('\n\n'.join(blocks), encoding='utf-8')


def write_sentences_file(examples: list[dict], path: Path) -> None:
    """Write one raw sentence per line (for test.txt used by inference.py)."""
    path.write_text('\n'.join(ex['sentence'] for ex in examples), encoding='utf-8')


def split_lpp(examples: list[dict], train_ratio: float = 0.8, dev_ratio: float = 0.1) -> tuple:
    """
    Deterministic 80/10/10 split preserving narrative order (no shuffle).
    Returns (train, dev, test) as three lists.
    """
    n = len(examples)
    n_train = int(n * train_ratio)
    n_dev = int(n * dev_ratio)
    train = examples[:n_train]
    dev = examples[n_train:n_train + n_dev]
    test = examples[n_train + n_dev:]
    return train, dev, test


def main():
    root = Path(__file__).parent.parent
    lpp_raw = root / 'data/portuguese/little_prince/raw/amr_br_raw.txt'
    news_raw = root / 'data/portuguese/news/raw/amrnews_raw.txt'
    lpp_out = root / 'data/portuguese/little_prince/raw'
    news_out = root / 'data/portuguese/news/raw'

    # ── LPP ──────────────────────────────────────────────────
    lpp = parse_corpus(str(lpp_raw))
    print(f"LPP: {len(lpp)} examples parsed")

    train, dev, test = split_lpp(lpp)
    print(f"  train={len(train)}, dev={len(dev)}, test={len(test)}")

    # Split train into two halves for amr1_weight/amr2_weight compatibility
    mid = len(train) // 2
    train1, train2 = train[:mid], train[mid:]

    write_amr_file(train1, lpp_out / 'train_amr_1.txt')
    write_amr_file(train2, lpp_out / 'train_amr_2.txt')
    write_amr_file(dev, lpp_out / 'dev.txt')
    write_amr_file(dev, lpp_out / 'dev_ground_truth.txt')
    write_sentences_file(test, lpp_out / 'test.txt')
    write_amr_file(test, lpp_out / 'test_ground_truth.txt')
    print(f"  Written to {lpp_out}")

    # ── AMRNews (test subset only — last 20%) ─────────────────
    news = parse_corpus(str(news_raw))
    print(f"\nAMRNews: {len(news)} examples parsed")

    n_test_news = max(1, len(news) // 5)
    news_test = news[-n_test_news:]
    print(f"  news test subset: {len(news_test)} examples")

    write_sentences_file(news_test, news_out / 'test.txt')
    write_amr_file(news_test, news_out / 'test_ground_truth.txt')
    print(f"  Written to {news_out}")


if __name__ == '__main__':
    main()
