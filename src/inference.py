#!/usr/bin/env python3
"""
Inference for ViSemPar_260430.

Loads the selected LoRA adapter, runs greedy decoding on a test file,
extracts the final AMR output (last step for MTUP variants),
and saves predictions.

Usage:
    python src/inference.py --config configs/vietnamese_v2/MTUP2.yaml --seed 42 --split test
    python src/inference.py --config configs/vietnamese_v2/B1.yaml    --seed 42 --split test

Outputs:
    results/{group}/{method}/s{seed}_pred.txt       – one AMR per line (for eval)
    results/{group}/{method}/s{seed}_full.txt       – full model output (for debug)
"""

import re
import sys
import json
import argparse
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.insert(0, str(Path(__file__).parent))
from config_utils import load_yaml_config
from prompts import get_system_prompt as prompt_for
from postprocess import postprocess_amr
from var_binder import bind_no_var_amr


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
    # Do not repair malformed model output. Evaluation will count it invalid.
    return None


def extract_amr_B1(response: str) -> str:
    marker = response.find('AMR:')
    if marker == -1:
        return ''
    amr = extract_balanced_amr(response, marker + 4)
    return amr or ''


def extract_amr_B2(response: str) -> str:
    marker = response.find('AMR_NO_VAR:')
    if marker == -1:
        return ''
    pos = marker + len('AMR_NO_VAR:')
    pstart = response.find('(', pos)
    if pstart == -1:
        return ''
    amr = extract_balanced_amr(response, pstart)
    return amr or ''


def extract_amr_MTUP2(response: str) -> str:
    """Extract Step 2 (final AMR with variables)."""
    marker = response.rfind('[STEP 2: AMR_WITH_VAR]')
    if marker == -1:
        return ''
    amr = extract_balanced_amr(response, marker)
    return amr or ''


def extract_amr_MTUP3(response: str) -> str:
    """Extract only MTUP3's final AMR; earlier stages are never a fallback."""
    marker = response.rfind('[STEP 3: AMR_WITH_VAR]')
    if marker == -1:
        return ''
    return extract_balanced_amr(response, marker) or ''


def extract_amr_MTUP4(response: str) -> str:
    """Extract only MTUP4's final AMR; earlier stages are never a fallback."""
    marker = response.rfind('[STEP 4: AMR_WITH_VAR]')
    if marker == -1:
        return ''
    return extract_balanced_amr(response, marker) or ''


EXTRACTORS = {
    'B1': extract_amr_B1,
    'B2': extract_amr_B2,
    'MTUP2': extract_amr_MTUP2,
    'MTUP3': extract_amr_MTUP3,
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


def get_system_prompt(cfg: dict) -> str:
    lang = cfg.get('language') or cfg.get('data', {}).get('language', 'vi')
    method = cfg.get('method')
    return prompt_for(lang, method)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',   required=True)
    parser.add_argument('--seed',     type=int, required=True)
    parser.add_argument('--split',    default='test',
                        choices=['test', 'val'],
                        help='Which split to run inference on')
    parser.add_argument('--adapter_path',
                        help='Override adapter path, e.g. an epoch checkpoint.')
    parser.add_argument('--output_dir',
                        help='Override output directory for predictions.')
    parser.add_argument('--output_prefix',
                        help='Override output filename prefix before _pred/_full.')
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)

    method = cfg['method']
    run_name = cfg.get('run_name', method)
    group = cfg['group']
    root = Path(__file__).parent.parent

    configured_adapter = args.adapter_path or cfg.get('adapter_path')
    if configured_adapter:
        adapter_path = Path(configured_adapter)
        if not adapter_path.is_absolute():
            adapter_path = root / adapter_path
    else:
        adapter_path = root / 'checkpoints' / group / run_name / f"s{args.seed}" / 'final'
    if not adapter_path.exists():
        print(f"ERROR: adapter not found at {adapter_path}")
        sys.exit(1)

    # Select test file
    if args.split == 'test':
        configured_test = cfg.get('data', {}).get('test')
        test_file = root / configured_test if configured_test else root / 'data' / 'raw' / 'public_test.txt'
    else:
        configured_val = cfg.get('data', {}).get('val')
        if not configured_val:
            raise SystemExit(
                "Config has no data.val for --split val. Generate a dev split with "
                "src/data_pipeline.py --dev_split ... and set data.val to the dev sentence file."
            )
        test_file = root / configured_val

    out_dir = Path(args.output_dir) if args.output_dir else root / 'results' / group / run_name
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = args.output_prefix or f"s{args.seed}"
    pred_path = out_dir / f"{output_prefix}_pred.txt"
    full_path = out_dir / f"{output_prefix}_full.txt"
    meta_path = out_dir / f"{output_prefix}_{args.split}_inference_meta.json"

    print(f"Method: {method} | Run: {run_name} | Seed: {args.seed} | Split: {args.split}")
    print(f"Adapter: {adapter_path}")
    print(f"Test file: {test_file}")

    model, tokenizer = load_model(cfg['model']['name'], str(adapter_path))
    sentences = load_sentences(str(test_file))
    extractor = EXTRACTORS[method]
    max_new_tokens = cfg.get('decoding', {}).get('max_new_tokens', 1024)
    system_prompt = get_system_prompt(cfg)

    predictions = []
    full_outputs = []
    final_failures = 0

    for sentence in tqdm(sentences, desc=f"Inference {method} s{args.seed}"):
        inputs = build_prompt(tokenizer, system_prompt, sentence)
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
        if not amr:
            final_failures += 1
        if method == 'B2':
            amr = bind_no_var_amr(amr)
        amr = postprocess_amr(amr)
        # Flatten to single line for eval
        amr_flat = re.sub(r'\s+', ' ', amr).strip()
        predictions.append(amr_flat)

    with open(pred_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(predictions))

    with open(full_path, 'w', encoding='utf-8') as f:
        for sent, full in zip(sentences, full_outputs):
            f.write(f"#::snt {sent}\n{full}\n\n")

    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump({
            'method': method,
            'run_name': run_name,
            'group': group,
            'seed': args.seed,
            'split': args.split,
            'config': args.config,
            'adapter_path': str(adapter_path),
            'input_file': str(test_file),
            'pred_path': str(pred_path),
            'full_path': str(full_path),
            'num_sentences': len(sentences),
            'final_extraction_failures': final_failures,
            'max_new_tokens': max_new_tokens,
        }, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(predictions)} predictions → {pred_path}")
    print(f"Full outputs → {full_path}")
    print(f"Inference metadata → {meta_path}")


if __name__ == '__main__':
    main()
