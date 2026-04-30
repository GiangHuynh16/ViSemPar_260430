#!/usr/bin/env python3
"""
LoRA SFT training for ViSemPar_260430.

Usage:
    python src/train.py --config configs/B1.yaml --seed 42

Saves adapter to:
    checkpoints/{group}/{method}/s{seed}/final/
"""

import re
import json
import yaml
import random
import argparse
import numpy as np
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
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
    if 'base' in cfg:
        base_path = Path(config_path).parent / cfg['base']
        with open(base_path) as f:
            base = yaml.safe_load(f)
        base.update(cfg)
        cfg = base
    cfg['seed'] = seed
    return cfg


def load_chat_dataset(data_path: str, tokenizer, max_length: int = 2048):
    """
    Load chat-format training data and build loss-masked dataset.
    Pattern: proven from train_mtup_chat.py (v13 which reached 0.49).
    """
    print(f"Loading dataset from: {data_path}")
    with open(data_path, encoding='utf-8') as f:
        content = f.read()

    # Split on the separator between examples
    raw = content.strip().split('\n\n<|im_start|>')

    # Restore the <|im_start|> prefix removed by split
    examples_text = []
    for i, ex in enumerate(raw):
        ex = ex.strip()
        if not ex:
            continue
        if not ex.startswith('<|im_start|>'):
            ex = '<|im_start|>' + ex
        if not ex.endswith('<|im_end|>'):
            ex = ex + '\n<|im_end|>'
        examples_text.append(ex)

    print(f"Found {len(examples_text)} examples")

    tokenized = []
    skipped = 0

    for ex in examples_text:
        tokens = tokenizer(
            ex,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors=None,
            add_special_tokens=False,
        )
        input_ids = tokens['input_ids']
        labels = input_ids.copy()

        # Find assistant block start by decoding prefix incrementally
        # (proven approach from train_mtup_chat.py)
        assistant_start = -1
        for i in range(len(input_ids)):
            decoded = tokenizer.decode(input_ids[:i + 1])
            if '<|im_start|>assistant' in decoded:
                assistant_start = i
                break

        if assistant_start == -1:
            skipped += 1
            continue

        # Skip past the "assistant\n" header token(s)
        for j in range(assistant_start, len(input_ids)):
            decoded = tokenizer.decode([input_ids[j]])
            if '\n' in decoded:
                assistant_start = j + 1
                break

        # Mask system + user + assistant header
        for i in range(min(assistant_start, len(labels))):
            labels[i] = -100

        # Must have at least some response tokens
        if all(l == -100 for l in labels):
            skipped += 1
            continue

        tokenized.append({'input_ids': input_ids, 'labels': labels})

    if skipped:
        print(f"  Skipped {skipped} examples (no assistant marker or all masked)")

    if tokenized:
        s = tokenized[0]
        n_resp = sum(1 for l in s['labels'] if l != -100)
        print(f"  Loaded {len(tokenized)} examples. "
              f"First: {len(s['input_ids'])} tokens, {n_resp} response tokens.")

    return Dataset.from_list(tokenized)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config, args.seed)
    set_seed(args.seed)

    method = cfg['method']
    group  = cfg['group']
    root   = Path(__file__).parent.parent

    train_path = root / 'data' / 'formatted' / cfg['data']['train']
    output_dir = root / 'checkpoints' / group / method / f"s{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Method: {method} | Seed: {args.seed} | Group: {group}")
    print(f"Train data: {train_path}")
    print(f"Output dir: {output_dir}")

    # ── Tokenizer ──────────────────────────────────────────────────────────
    model_name = cfg['model']['name']
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Dataset ────────────────────────────────────────────────────────────
    tr_cfg = cfg['training']
    dataset = load_chat_dataset(
        str(train_path), tokenizer,
        max_length=tr_cfg.get('max_length', 2048))
    print(f"Train examples: {len(dataset)}")

    # ── Model ──────────────────────────────────────────────────────────────
    print(f"\nLoading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map='auto',
        trust_remote_code=True,
    )

    lora_cfg = cfg['lora']
    target_modules = lora_cfg.get('target_modules',
        ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'])

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_cfg['r'],
        lora_alpha=lora_cfg['alpha'],
        lora_dropout=lora_cfg.get('dropout', 0.05),
        target_modules=target_modules,
        bias=lora_cfg.get('bias', 'none'),
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Critical: must call in this exact order ────────────────────────────
    # enable_input_require_grads() BEFORE gradient_checkpointing_enable()
    # so that the hook set by enable_input_require_grads is not overwritten.
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    # ── Training args ──────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=tr_cfg['epochs'],
        per_device_train_batch_size=tr_cfg.get('batch_per_device', 1),
        gradient_accumulation_steps=tr_cfg.get('grad_accum', 16),
        learning_rate=tr_cfg['lr'],
        lr_scheduler_type=tr_cfg.get('scheduler', 'cosine'),
        warmup_ratio=tr_cfg.get('warmup_ratio', 0.1),
        weight_decay=tr_cfg.get('weight_decay', 0.01),
        max_grad_norm=tr_cfg.get('max_grad_norm', 1.0),
        optim=tr_cfg.get('optimizer', 'adamw_torch'),
        bf16=True,
        # Do NOT set gradient_checkpointing=True here – already enabled above
        gradient_checkpointing=False,
        logging_steps=10,
        save_strategy='epoch',
        save_total_limit=3,
        seed=args.seed,
        report_to='none',
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        return_tensors='pt',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    trainer.train()

    # ── Save ───────────────────────────────────────────────────────────────
    final_dir = output_dir / 'final'
    trainer.model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"\nSaved adapter → {final_dir}")

    meta = {
        'method': method, 'seed': args.seed, 'group': group,
        'config': args.config,
        'train_examples': len(dataset),
        'epochs': tr_cfg['epochs'],
        'lora_r': lora_cfg['r'],
        'lr': tr_cfg['lr'],
    }
    with open(output_dir / 'train_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)


if __name__ == '__main__':
    main()
