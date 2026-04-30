#!/usr/bin/env python3
"""
LoRA SFT training for ViSemPar_260430.

Usage:
    python src/train.py --config configs/B1.yaml --seed 42

Saves best checkpoint (by val Smatch) to:
    checkpoints/{group}/{method}/s{seed}/best/
    checkpoints/{group}/{method}/s{seed}/final/
"""

import os
import re
import sys
import json
import yaml
import random
import argparse
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, TaskType


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(config_path: str, seed: int) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Load base config if specified
    if 'base' in cfg:
        base_path = Path(config_path).parent / cfg['base']
        with open(base_path) as f:
            base = yaml.safe_load(f)
        base.update(cfg)
        cfg = base

    cfg['seed'] = seed
    return cfg


class ChatDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = self._load(data_path)

    def _load(self, path: str):
        with open(path, encoding='utf-8') as f:
            content = f.read()

        # Split on double newline before <|im_start|> (our separator)
        raw = re.split(r'\n\n(?=<\|im_start\|>)', content.strip())
        examples = []
        skipped = 0

        for text in raw:
            text = text.strip()
            if not text:
                continue
            if not text.endswith('<|im_end|>'):
                text = text + '<|im_end|>'

            encoded = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                add_special_tokens=False,
                return_tensors=None,
            )
            input_ids = encoded['input_ids']

            # Build labels: mask everything up to and including <|im_start|>assistant\n
            labels = [-100] * len(input_ids)
            assistant_marker = '<|im_start|>assistant\n'
            marker_ids = self.tokenizer.encode(
                assistant_marker, add_special_tokens=False)

            # Find last occurrence of marker in input_ids
            marker_pos = -1
            for i in range(len(input_ids) - len(marker_ids), -1, -1):
                if input_ids[i:i + len(marker_ids)] == marker_ids:
                    marker_pos = i
                    break

            if marker_pos == -1:
                skipped += 1
                continue

            # Only compute loss on tokens after the marker
            for j in range(marker_pos + len(marker_ids), len(input_ids)):
                labels[j] = input_ids[j]

            if all(l == -100 for l in labels):
                skipped += 1
                continue

            examples.append({
                'input_ids': input_ids,
                'labels': labels,
            })

        if skipped:
            print(f"  [Dataset] Skipped {skipped} examples (no assistant marker / truncated)")
        return examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return {k: torch.tensor(v) for k, v in self.examples[idx].items()}


def build_model(cfg: dict):
    model_name = cfg['model']['name']
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, padding_side='right')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map='auto',
        trust_remote_code=True,
        attn_implementation='flash_attention_2' if cfg['model'].get('flash_attn') else 'eager',
    )

    lora_cfg = cfg['lora']
    target_modules = lora_cfg.get('target_modules',
        ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'])

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_cfg['r'],
        lora_alpha=lora_cfg['alpha'],
        lora_dropout=lora_cfg.get('dropout', 0.05),
        target_modules=target_modules,
        bias=lora_cfg.get('bias', 'none'),
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config, args.seed)
    set_seed(args.seed)

    method = cfg['method']
    group = cfg['group']
    root = Path(__file__).parent.parent

    train_path = root / 'data' / 'formatted' / cfg['data']['train']
    output_dir = root / 'checkpoints' / group / method / f"s{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Method: {method} | Seed: {args.seed}")
    print(f"Train: {train_path}")
    print(f"Output: {output_dir}")

    model, tokenizer = build_model(cfg)

    tr_cfg = cfg['training']
    dataset = ChatDataset(str(train_path), tokenizer,
                          max_length=tr_cfg.get('max_length', 2048))
    print(f"Train examples: {len(dataset)}")

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=tr_cfg['epochs'],
        per_device_train_batch_size=tr_cfg.get('batch_per_device', 1),
        gradient_accumulation_steps=tr_cfg.get('grad_accum', 16),
        learning_rate=tr_cfg['lr'],
        lr_scheduler_type=tr_cfg.get('scheduler', 'cosine'),
        warmup_ratio=tr_cfg.get('warmup_ratio', 0.05),
        weight_decay=tr_cfg.get('weight_decay', 0.01),
        max_grad_norm=tr_cfg.get('max_grad_norm', 1.0),
        optim=tr_cfg.get('optimizer', 'adamw_torch'),
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        save_strategy='epoch',
        save_total_limit=3,
        seed=args.seed,
        report_to='none',
        dataloader_num_workers=0,
    )

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,
        label_pad_token_id=-100,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )

    trainer.train()

    # Save final adapter
    final_dir = output_dir / 'final'
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"Saved final adapter → {final_dir}")

    # Save training metadata
    meta = {
        'method': method, 'seed': args.seed, 'group': group,
        'config': args.config,
        'train_examples': len(dataset),
        'epochs': tr_cfg['epochs'],
        'lora_r': cfg['lora']['r'],
        'lr': tr_cfg['lr'],
    }
    with open(output_dir / 'train_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)


if __name__ == '__main__':
    main()
