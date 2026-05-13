#!/usr/bin/env python3
"""
Data pipeline for ViSemPar_260430.

Steps:
  1. Load train_amr_1 + train_amr_2, deduplicate by sentence
  2. Build formatted chat examples for each method
  3. Save formatted/{method}_train.txt (ready to feed into train.py)

Methods: B1, B2, MTUP2, MTUP3A, MTUP3B, MTUP4
  B1     – direct: sentence → AMR
  B2     – no-var:  sentence → AMR-no-var  (intermediate only)
  MTUP2  – 2 steps: no-var → AMR
  MTUP3A – 3 steps: concepts → no-var → AMR
  MTUP3B – 3 steps: concepts → relations → AMR
  MTUP4  – 4 steps: concepts → relations → no-var → AMR

Usage:
    python src/data_pipeline.py [--verify] [--amr1_weight 1] [--amr2_weight 3] [--syn_weight 10]
"""

import re
import json
import argparse
from pathlib import Path
from collections import OrderedDict


# ─────────────────────────────────────────────────────────────────────────────
# Frozen system prompts (identical for ALL methods per plan)
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT_MULTISTEP = """Bạn là chuyên gia phân tích AMR (Abstract Meaning Representation) cho tiếng Việt.

QUY TẮC QUAN TRỌNG:
1. Chỉ dùng từ tiếng Việt trong câu gốc làm concept. KHÔNG tự tạo từ mới.
2. Concept AMR chuẩn được phép dùng (tiếng Anh): person, name, now, after, before, and, or, contrast-01, obligate-01, possible-01, rate-entity-91, temporal-quantity, percentage-entity, multi-sentence, have-rel-role-91, have-org-role-91, amr-unknown, province, city, mountain, country, company, ethnic-group, date-entity, date-interval, monetary-quantity, about
3. :classifier cho lượng từ: người, anh, chị, cái, con, chiếc, đứa, ngôi, căn, cùi, ngọn, chú, cánh
4. :polarity - (dấu trừ literal) cho phủ định
5. :modality + cho được, :modality - cho bị
6. multi-sentence cho câu có nhiều mệnh đề độc lập (phân tách bởi dấu phẩy)
7. :wiki - cho tên riêng người/địa danh (dấu trừ literal)
8. :pivot cho trạng thái/tính từ, :agent cho hành động
9. "hiện nay", "đến nay", "nay" → (n / now)
10. Từ ghép giữ nguyên: "làm việc", "chịu thua", "thoát nghèo", "vận động"
11. Đảm bảo dấu ngoặc mở và đóng cân bằng
12. Mỗi biến (chữ cái + số tùy chọn) phải DUY NHẤT trong toàn bộ AMR
13. :prep cho giới từ: về, từ, ở, vào, trong, cho, tại, bởi, vì, để, với
14. "theo X, ..." → root là (t / theo :source(X) :topic(mệnh đề chính))
15. "thưa X, ..." → thêm :polite+(wz / sir) vào root verb của câu chính
16. Động từ chuyển động sau main verb (đi, về, lại, tiếp, ra, vào) → :manner(đ / đi), KHÔNG dùng :compound
17. :domain cho tính từ/trạng thái khi subject không chủ động hành động
18. "X nói/khẳng định/cho biết: ..." → (s / say-01 :agent X :topic(mệnh đề))"""

SYSTEM_PROMPT_B1 = SYSTEM_PROMPT_MULTISTEP  # identical per plan

# ─────────────────────────────────────────────────────────────────────────────
# Synthetic examples (frozen – never change after first run)
# ─────────────────────────────────────────────────────────────────────────────

# Synthetic examples removed — all 15 original examples were found in the test set
# (data leakage). Training relies solely on train_amr_1.txt and train_amr_2.txt.
SYNTHETIC_EXAMPLES = []


# ─────────────────────────────────────────────────────────────────────────────
# AMR processing helpers
# ─────────────────────────────────────────────────────────────────────────────

def convert_underscore_to_space(text: str) -> str:
    """Replace underscores with spaces except inside :wiki(...)."""
    lines = text.split('\n')
    result = []
    for line in lines:
        if ':wiki(' in line or ':wiki ' in line:
            result.append(line)
        else:
            processed = re.sub(r'/_+', '/ ', line)
            processed = re.sub(r'_+', '_', processed)
            while True:
                old = processed
                processed = re.sub(r'([a-zA-ZÀ-ỹ]+)_([a-zA-ZÀ-ỹ]+)', r'\1 \2', processed)
                if processed == old:
                    break
            result.append(processed)
    return '\n'.join(result)


def fix_polarity_format(text: str) -> str:
    return re.sub(r':polarity\s*\([^)]*\bno\b[^)]*\)', ':polarity -', text)


def validate_amr(amr: str) -> bool:
    if not amr or not amr.strip():
        return False
    s = amr.strip()
    if not s.startswith('('):
        return False
    if s.count('(') != s.count(')'):
        return False
    if '/' not in s:
        return False
    return True


def strip_variables(amr: str) -> str:
    """Convert full AMR to no-var form: (x / concept ...) → (concept ...)."""
    result = re.sub(r'\(\s*\w+\s*/\s*', '(', amr)
    return result


def parse_amr_tree(amr_text: str):
    """Return (root_concept, [(parent, relation, child), ...])."""
    triples = []
    concept_stack = []
    current_relation = None
    root_concept = None
    i = 0
    text = amr_text

    while i < len(text):
        if text[i] in ' \t\n':
            i += 1
            continue

        if text[i] == '(':
            i += 1
            while i < len(text) and text[i] in ' \t\n':
                i += 1
            var_start = i
            while i < len(text) and text[i] not in ' \t\n/()':
                i += 1
            while i < len(text) and text[i] in ' \t\n':
                i += 1
            if i < len(text) and text[i] == '/':
                i += 1
                while i < len(text) and text[i] in ' \t\n':
                    i += 1
                concept_start = i
                while i < len(text):
                    if text[i] in '()':
                        break
                    if text[i] == ':':
                        break
                    if text[i] == '\n':
                        j = i + 1
                        while j < len(text) and text[j] in ' \t':
                            j += 1
                        if j < len(text) and text[j] in ':)(': break
                    i += 1
                concept = text[concept_start:i].strip()
                if not root_concept:
                    root_concept = concept
                if concept_stack and current_relation:
                    triples.append((concept_stack[-1][1], current_relation, concept))
                    current_relation = None
                var = text[var_start:var_start + 10].split()[0] if var_start < len(text) else 'x'
                concept_stack.append((var, concept))
            else:
                i = var_start
                while i < len(text) and text[i] not in ' \t\n)(':
                    i += 1
        elif text[i] == ')':
            if concept_stack:
                concept_stack.pop()
            i += 1
        elif text[i] == ':':
            rel_start = i
            i += 1
            while i < len(text) and (text[i].isalnum() or text[i] in '-_+'):
                i += 1
            relation = text[rel_start:i].strip()
            while i < len(text) and text[i] in ' \t\n':
                i += 1
            if i < len(text) and text[i] != '(':
                val_start = i
                while i < len(text) and text[i] not in '\t\n)(':
                    if text[i] == ':':
                        break
                    if text[i] == ' ':
                        j = i + 1
                        while j < len(text) and text[j] == ' ':
                            j += 1
                        if j >= len(text) or text[j] in ':)(\n':
                            break
                        break
                    i += 1
                value = text[val_start:i].strip()
                if value and concept_stack:
                    triples.append((concept_stack[-1][1], relation, value))
            else:
                current_relation = relation
        else:
            i += 1

    return root_concept, triples


def build_concept_list(amr: str) -> str:
    """Flat concept list: 'concept_1 | want-01'  (one per line)."""
    root, triples = parse_amr_tree(amr)
    if not root:
        return None
    seen = OrderedDict()
    seen[root] = True
    for _, _, child in triples:
        if re.match(r'^[a-zA-ZÀ-ỹ]', child):
            seen[child] = True
    return '\n'.join(f"concept_{i+1} | {c}" for i, c in enumerate(seen))


def build_relation_list(amr: str) -> str:
    """Flat relation list: 'parent | :role | child'  (one per line)."""
    _, triples = parse_amr_tree(amr)
    if not triples:
        return None
    return '\n'.join(f"{p} | {r} | {c}" for p, r, c in triples)


def build_task1_list(amr: str) -> str:
    """Task-1 triples for MTUP2/v13 style: 'root: X\\nX -> :role -> Y'."""
    root, triples = parse_amr_tree(amr)
    if not root:
        return None
    lines = [f"root: {root}"]
    lines += [f"{p} -> {r} -> {c}" for p, r, c in triples]
    return '\n'.join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Chat example builders per method
# ─────────────────────────────────────────────────────────────────────────────

def _wrap(system: str, user: str, assistant: str) -> str:
    return (f"<|im_start|>system\n{system}\n<|im_end|>\n"
            f"<|im_start|>user\n{user}\n<|im_end|>\n"
            f"<|im_start|>assistant\n{assistant}\n<|im_end|>")


def make_B1(sentence: str, amr: str) -> str:
    """B1: direct sentence → AMR."""
    return _wrap(SYSTEM_PROMPT_B1, sentence, f"AMR: {amr}")


def make_B2(sentence: str, amr: str) -> str:
    """B2: sentence → AMR-no-var."""
    no_var = strip_variables(amr)
    return _wrap(SYSTEM_PROMPT_MULTISTEP, sentence, f"AMR_NO_VAR: {no_var}")


def make_MTUP2(sentence: str, amr: str) -> str:
    """MTUP2: no-var → full AMR (closer intermediate, easier for model to add variables)."""
    no_var = strip_variables(amr)
    assistant = f"[STEP 1: AMR_NO_VAR]\n{no_var}\n\n[STEP 2: AMR_WITH_VAR]\n{amr}"
    return _wrap(SYSTEM_PROMPT_MULTISTEP, sentence, assistant)


def make_MTUP3A(sentence: str, amr: str) -> str:
    """MTUP3A: concepts → no-var → AMR."""
    concepts = build_concept_list(amr)
    no_var = strip_variables(amr)
    if concepts is None:
        return None
    assistant = (f"[STEP 1: CONCEPTS]\n{concepts}\n\n"
                 f"[STEP 2: AMR_NO_VAR]\n{no_var}\n\n"
                 f"[STEP 3: AMR_WITH_VAR]\n{amr}")
    return _wrap(SYSTEM_PROMPT_MULTISTEP, sentence, assistant)


def make_MTUP3B(sentence: str, amr: str) -> str:
    """MTUP3B: concepts → relations → AMR."""
    concepts = build_concept_list(amr)
    relations = build_relation_list(amr)
    if concepts is None or relations is None:
        return None
    assistant = (f"[STEP 1: CONCEPTS]\n{concepts}\n\n"
                 f"[STEP 2: RELATIONS]\n{relations}\n\n"
                 f"[STEP 3: AMR_WITH_VAR]\n{amr}")
    return _wrap(SYSTEM_PROMPT_MULTISTEP, sentence, assistant)


def make_MTUP4(sentence: str, amr: str) -> str:
    """MTUP4: concepts → relations → no-var → AMR."""
    concepts = build_concept_list(amr)
    relations = build_relation_list(amr)
    no_var = strip_variables(amr)
    if concepts is None or relations is None:
        return None
    assistant = (f"[STEP 1: CONCEPTS]\n{concepts}\n\n"
                 f"[STEP 2: RELATIONS]\n{relations}\n\n"
                 f"[STEP 3: AMR_NO_VAR]\n{no_var}\n\n"
                 f"[STEP 4: AMR_WITH_VAR]\n{amr}")
    return _wrap(SYSTEM_PROMPT_MULTISTEP, sentence, assistant)


BUILDERS = {
    'B1': make_B1,
    'B2': make_B2,
    'MTUP2': make_MTUP2,
    'MTUP3A': make_MTUP3A,
    'MTUP3B': make_MTUP3B,
    'MTUP4': make_MTUP4,
}


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def parse_amr_file(path: str):
    with open(path, encoding='utf-8') as f:
        content = f.read()
    examples = []
    for block in content.strip().split('\n\n'):
        block = block.strip()
        if not block:
            continue
        sentence = None
        amr_lines = []
        for line in block.split('\n'):
            if line.startswith('#::snt'):
                sentence = line.replace('#::snt', '').strip()
            elif not line.startswith('#'):
                amr_lines.append(line)
        if sentence and amr_lines:
            examples.append({'sentence': sentence, 'amr': '\n'.join(amr_lines)})
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


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset(method: str, examples_1, examples_2, syn_weight: int,
                  amr1_weight: int, amr2_weight: int) -> list:
    builder = BUILDERS[method]
    chat_examples = []

    def add(exs, weight):
        for _ in range(weight):
            for ex in exs:
                amr = convert_underscore_to_space(ex['amr'])
                amr = fix_polarity_format(amr)
                if not validate_amr(amr):
                    continue
                result = builder(ex['sentence'], amr)
                if result:
                    chat_examples.append(result)

    add(examples_1, amr1_weight)
    add(examples_2, amr2_weight)

    for _ in range(syn_weight):
        for syn in SYNTHETIC_EXAMPLES:
            amr = convert_underscore_to_space(syn['amr'])
            amr = fix_polarity_format(amr)
            if not validate_amr(amr):
                continue
            result = builder(syn['sentence'], amr)
            if result:
                chat_examples.append(result)

    return chat_examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--amr1_weight', type=int, default=1)
    parser.add_argument('--amr2_weight', type=int, default=3)
    parser.add_argument('--syn_weight',  type=int, default=10)
    parser.add_argument('--methods', nargs='+',
                        default=['B1', 'B2', 'MTUP2', 'MTUP3A', 'MTUP3B', 'MTUP4'])
    parser.add_argument('--verify', action='store_true',
                        help='Print 2 examples per method and exit')
    args = parser.parse_args()

    root = Path(__file__).parent.parent
    raw_dir = root / 'data' / 'raw'
    fmt_dir = root / 'data' / 'formatted'
    fmt_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    ex1_raw = parse_amr_file(str(raw_dir / 'train_amr_1.txt'))
    ex2_raw = parse_amr_file(str(raw_dir / 'train_amr_2.txt'))

    # Deduplicate each file independently (cross-file overlap already 0)
    ex1 = dedup(ex1_raw)
    ex2 = dedup(ex2_raw)

    print(f"  train_amr_1: {len(ex1_raw)} raw → {len(ex1)} after dedup")
    print(f"  train_amr_2: {len(ex2_raw)} raw → {len(ex2)} after dedup")
    print(f"  synthetic:   {len(SYNTHETIC_EXAMPLES)} examples × {args.syn_weight}")
    print()

    stats = {}
    for method in args.method if hasattr(args, 'method') else args.methods:
        examples = build_dataset(method, ex1, ex2, args.syn_weight,
                                 args.amr1_weight, args.amr2_weight)
        out_path = fmt_dir / f"{method}_train.txt"
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write('\n\n'.join(examples))
        stats[method] = len(examples)
        print(f"  {method}: {len(examples)} examples → {out_path}")

        if args.verify:
            print(f"\n--- {method} sample 1 ---")
            print(examples[0][:600])
            print(f"\n--- {method} sample 2 (synthetic) ---")
            print(examples[-1][:600])
            print()

    # Save stats
    stats_path = fmt_dir / 'dataset_stats.json'
    with open(stats_path, 'w') as f:
        json.dump({
            'amr1_examples': len(ex1),
            'amr2_examples': len(ex2),
            'synthetic_examples': len(SYNTHETIC_EXAMPLES),
            'weights': {'amr1': args.amr1_weight, 'amr2': args.amr2_weight, 'syn': args.syn_weight},
            'methods': stats,
        }, f, indent=2)
    print(f"\nStats saved to {stats_path}")


if __name__ == '__main__':
    main()
