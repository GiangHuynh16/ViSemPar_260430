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

try:
    from sklearn.model_selection import train_test_split
except ImportError:
    train_test_split = None


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
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


def split_dataset_for_dev(dataset: Dataset, dev_split: float, seed: int):
    """Split a tokenized dataset with sklearn so model selection is not test-based."""
    if dev_split <= 0:
        return dataset, None
    if train_test_split is None:
        raise RuntimeError(
            "scikit-learn is required for automatic train/dev splitting. "
            "Install scikit-learn or generate *_dev.txt with src/data_pipeline.py."
        )
    indices = list(range(len(dataset)))
    train_idx, dev_idx = train_test_split(
        indices,
        test_size=dev_split,
        random_state=seed,
        shuffle=True,
    )
    return dataset.select(train_idx), dataset.select(dev_idx)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--auto_dev_split', type=float, default=None,
                        help='If no data.dev_formatted file exists, split this fraction '
                             'from train with sklearn train_test_split.')
    args = parser.parse_args()

    cfg = load_config(args.config, args.seed)
    set_seed(args.seed)

    method = cfg['method']
    group  = cfg['group']
    root   = Path(__file__).parent.parent

    train_path = root / 'data' / 'formatted' / cfg['data']['train']
    output_dir = root / 'checkpoints' / group / method / f"s{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Dev dataset: optional, derived from data.dev key in config
    dev_key = cfg.get('data', {}).get('dev_formatted')
    dev_path = (root / 'data' / 'formatted' / dev_key) if dev_key else None

    print(f"Method: {method} | Seed: {args.seed} | Group: {group}")
    print(f"Train data: {train_path}")
    if dev_path:
        print(f"Dev data:   {dev_path}")
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

    eval_dataset = None
    if dev_path and dev_path.exists():
        eval_dataset = load_chat_dataset(
            str(dev_path), tokenizer,
            max_length=tr_cfg.get('max_length', 2048))
        print(f"Dev examples:  {len(eval_dataset)}")
    else:
        auto_dev_split = args.auto_dev_split
        if auto_dev_split is None:
            auto_dev_split = cfg.get('data', {}).get('auto_dev_split', 0.0)
        if auto_dev_split and auto_dev_split > 0:
            dataset, eval_dataset = split_dataset_for_dev(
                dataset, auto_dev_split, cfg.get('data', {}).get('dev_seed', args.seed))
            print(f"Auto dev split: {auto_dev_split:.3f}")
            print(f"Train examples after split: {len(dataset)}")
            print(f"Dev examples after split:   {len(eval_dataset)}")

    # ── Model ──────────────────────────────────────────────────────────────
    print(f"\nLoading model: {model_name}")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is not available. This training setup requires a GPU; "
            "check nvidia-smi, CUDA_VISIBLE_DEVICES, and the PyTorch CUDA build."
        )

    use_bf16 = torch.cuda.is_bf16_supported()
    model_dtype = torch.bfloat16 if use_bf16 else torch.float16
    print(f"Precision: {'bf16' if use_bf16 else 'fp16'}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=model_dtype,
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
        bf16=use_bf16,
        fp16=not use_bf16,
        # Do NOT set gradient_checkpointing=True here – already enabled above
        gradient_checkpointing=False,
        logging_strategy=tr_cfg.get('logging_strategy', 'epoch'),
        logging_steps=tr_cfg.get('logging_steps', 10),
        save_strategy='epoch',
        save_total_limit=tr_cfg.get('save_total_limit', 3),
        seed=args.seed,
        report_to='none',
        remove_unused_columns=False,
        dataloader_num_workers=0,
        # Eval settings — only active when eval_dataset is provided
        eval_strategy='epoch' if eval_dataset is not None else 'no',
        load_best_model_at_end=eval_dataset is not None,
        metric_for_best_model='eval_loss' if eval_dataset is not None else None,
        greater_is_better=False if eval_dataset is not None else None,
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
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    trainer.train()

    log_history = trainer.state.log_history
    with open(output_dir / 'trainer_log_history.json', 'w') as f:
        json.dump(log_history, f, indent=2)

    if eval_dataset is not None:
        try:
            import matplotlib.pyplot as plt

            train_epochs = [x['epoch'] for x in log_history if 'loss' in x and 'epoch' in x]
            train_losses = [x['loss'] for x in log_history if 'loss' in x and 'epoch' in x]
            eval_epochs = [x['epoch'] for x in log_history if 'eval_loss' in x and 'epoch' in x]
            eval_losses = [x['eval_loss'] for x in log_history if 'eval_loss' in x and 'epoch' in x]

            if train_losses or eval_losses:
                plt.figure(figsize=(7, 4))
                if train_losses:
                    plt.plot(train_epochs, train_losses, marker='o', label='train loss')
                if eval_losses:
                    plt.plot(eval_epochs, eval_losses, marker='o', label='dev loss')
                plt.xlabel('epoch')
                plt.ylabel('loss')
                plt.title(f'{method} seed={args.seed}')
                plt.legend()
                plt.tight_layout()
                plt.savefig(output_dir / 'loss_curve.png', dpi=160)
                plt.close()
                print(f"Loss curve → {output_dir / 'loss_curve.png'}")
        except Exception as exc:
            print(f"WARNING: could not write loss curve: {exc}")

    # ── Save ───────────────────────────────────────────────────────────────
    final_dir = output_dir / 'final'
    trainer.model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"\nSaved adapter → {final_dir}")

    meta = {
        'method': method, 'seed': args.seed, 'group': group,
        'config': args.config,
        'train_examples': len(dataset),
        'dev_examples': len(eval_dataset) if eval_dataset is not None else 0,
        'epochs': tr_cfg['epochs'],
        'lora_r': lora_cfg['r'],
        'lr': tr_cfg['lr'],
        'best_checkpoint': trainer.state.best_model_checkpoint,
        'best_eval_loss': trainer.state.best_metric,
    }
    with open(output_dir / 'train_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)


if __name__ == '__main__':
    main()
