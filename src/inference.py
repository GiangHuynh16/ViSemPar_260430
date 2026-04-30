#!/usr/bin/env python3
"""
Inference for ViSemPar_260430.

Loads the final LoRA adapter, runs greedy decoding on a test file,
extracts the final AMR output (last step for MTUP variants),
and saves predictions.

Usage:
    python src/inference.py --config configs/MTUP2.yaml --seed 42 --split test
    python src/inference.py --config configs/B1.yaml    --seed 42 --split test

Outputs:
    results/{group}/{method}/s{seed}_pred.txt       – one AMR per line (for eval)
    results/{group}/{method}/s{seed}_full.txt       – full model output (for debug)
"""

import re
import sys
import yaml
import json
import argparse
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


# ─────────────────────────────────────────────────────────────────────────────
# Extraction logic per method
# ─────────────────────────────────────────────────────────────────────────────

def extract_balanced_amr(text: str, start_pos: int) -> str:
    """Extract first balanced AMR starting from start_pos."""
    paren_start = text.find('(', start_pos)
    if paren_start == -1:
        return None
    depth = 0
    for i in range(paren_start, len(text)):
        if text[i] == '(':
            depth += 1
        elif text[i] == ')':
            depth -= 1
            if depth == 0:
                return text[paren_start:i + 1]
    # Unbalanced – return up to last )
    last_close = text.rfind(')', paren_start)
    if last_close != -1:
        fragment = text[paren_start:last_close + 1]
        # Balance by appending missing )
        opens = fragment.count('(')
        closes = fragment.count(')')
        if opens > closes:
            fragment += ')' * (opens - closes)
        return fragment
    return None


def extract_amr_B1(response: str) -> str:
    marker = response.find('AMR:')
    if marker == -1:
        return '(a / amr-unknown)'
    amr = extract_balanced_amr(response, marker + 4)
    return amr or '(a / amr-unknown)'


def extract_amr_B2(response: str) -> str:
    marker = response.find('AMR_NO_VAR:')
    if marker == -1:
        return '(amr-unknown)'
    pos = marker + len('AMR_NO_VAR:')
    pstart = response.find('(', pos)
    if pstart == -1:
        return '(amr-unknown)'
    amr = extract_balanced_amr(response, pstart)
    return amr or '(amr-unknown)'


def extract_amr_MTUP2(response: str) -> str:
    """Extract Task 2 (final AMR with variables)."""
    marker = response.rfind('Task 2:')
    if marker == -1:
        # Fallback: try Task 1 as rough output
        marker = response.rfind('Task 1:')
    if marker == -1:
        return '(a / amr-unknown)'
    amr = extract_balanced_amr(response, marker)
    return amr or '(a / amr-unknown)'


def extract_amr_MTUP3A(response: str) -> str:
    marker = response.rfind('[STEP 3: AMR_WITH_VAR]')
    if marker == -1:
        # fallback to step 2 no-var
        marker = response.rfind('[STEP 2: AMR_NO_VAR]')
    if marker == -1:
        return '(a / amr-unknown)'
    amr = extract_balanced_amr(response, marker)
    return amr or '(a / amr-unknown)'


def extract_amr_MTUP3B(response: str) -> str:
    marker = response.rfind('[STEP 3: AMR_WITH_VAR]')
    if marker == -1:
        return '(a / amr-unknown)'
    amr = extract_balanced_amr(response, marker)
    return amr or '(a / amr-unknown)'


def extract_amr_MTUP4(response: str) -> str:
    marker = response.rfind('[STEP 4: AMR_WITH_VAR]')
    if marker == -1:
        marker = response.rfind('[STEP 3: AMR_NO_VAR]')
    if marker == -1:
        return '(a / amr-unknown)'
    amr = extract_balanced_amr(response, marker)
    return amr or '(a / amr-unknown)'


EXTRACTORS = {
    'B1': extract_amr_B1,
    'B2': extract_amr_B2,
    'MTUP2': extract_amr_MTUP2,
    'MTUP3A': extract_amr_MTUP3A,
    'MTUP3B': extract_amr_MTUP3B,
    'MTUP4': extract_amr_MTUP4,
}


# ─────────────────────────────────────────────────────────────────────────────
# Model helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_model(base_model: str, adapter_path: str):
    tokenizer = AutoTokenizer.from_pretrained(
        base_model, trust_remote_code=True, padding_side='left')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map='auto',
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


def load_sentences(path: str) -> list:
    sentences = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                sentences.append(line)
    return sentences


def build_prompt(tokenizer, system_prompt: str, sentence: str) -> torch.Tensor:
    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': sentence},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    return tokenizer(text, return_tensors='pt', add_special_tokens=False)


SYSTEM_PROMPT = """Bạn là chuyên gia phân tích AMR (Abstract Meaning Representation) cho tiếng Việt.

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',   required=True)
    parser.add_argument('--seed',     type=int, required=True)
    parser.add_argument('--split',    default='test',
                        choices=['test', 'val'],
                        help='Which split to run inference on')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Load base config
    if 'base' in cfg:
        base_path = Path(args.config).parent / cfg['base']
        with open(base_path) as f:
            base = yaml.safe_load(f)
        base.update(cfg)
        cfg = base

    method = cfg['method']
    group = cfg['group']
    root = Path(__file__).parent.parent

    adapter_path = root / 'checkpoints' / group / method / f"s{args.seed}" / 'final'
    if not adapter_path.exists():
        print(f"ERROR: adapter not found at {adapter_path}")
        sys.exit(1)

    # Select test file
    if args.split == 'test':
        test_file = root / 'data' / 'raw' / 'public_test.txt'
    else:
        test_file = root / 'data' / 'raw' / cfg['data'].get('val', 'public_test.txt')

    out_dir = root / 'results' / group / method
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / f"s{args.seed}_pred.txt"
    full_path = out_dir / f"s{args.seed}_full.txt"

    print(f"Method: {method} | Seed: {args.seed} | Split: {args.split}")
    print(f"Adapter: {adapter_path}")
    print(f"Test file: {test_file}")

    model, tokenizer = load_model(cfg['model']['name'], str(adapter_path))
    sentences = load_sentences(str(test_file))
    extractor = EXTRACTORS[method]
    max_new_tokens = cfg.get('decoding', {}).get('max_new_tokens', 1024)

    predictions = []
    full_outputs = []

    for sentence in tqdm(sentences, desc=f"Inference {method} s{args.seed}"):
        inputs = build_prompt(tokenizer, SYSTEM_PROMPT, sentence)
        input_ids = inputs['input_ids'].to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        new_ids = output_ids[0][input_ids.shape[1]:]
        response = tokenizer.decode(new_ids, skip_special_tokens=True)
        full_outputs.append(response)

        amr = extractor(response)
        # Flatten to single line for eval
        amr_flat = re.sub(r'\s+', ' ', amr).strip()
        predictions.append(amr_flat)

    with open(pred_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(predictions))

    with open(full_path, 'w', encoding='utf-8') as f:
        for sent, full in zip(sentences, full_outputs):
            f.write(f"#::snt {sent}\n{full}\n\n")

    print(f"Saved {len(predictions)} predictions → {pred_path}")
    print(f"Full outputs → {full_path}")


if __name__ == '__main__':
    main()
