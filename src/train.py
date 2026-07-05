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
import os
import random
import argparse
import math
import time
import importlib.util
import numpy as np
from pathlib import Path

# Reduce CUDA allocator fragmentation on long train/eval transitions.
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, TaskType
from config_utils import load_yaml_config

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
    cfg = load_yaml_config(config_path)
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


class TimingCallback(TrainerCallback):
    """Record coarse optimizer-step and epoch timing for GPU-cost diagnosis."""

    def __init__(self, log_steps: int = 10):
        self.log_steps = max(1, int(log_steps))
        self.rows = []
        self._train_start = None
        self._epoch_start = None
        self._step_start = None

    def on_train_begin(self, args, state, control, **kwargs):
        self._train_start = time.perf_counter()

    def on_epoch_begin(self, args, state, control, **kwargs):
        self._epoch_start = time.perf_counter()

    def on_step_begin(self, args, state, control, **kwargs):
        self._step_start = time.perf_counter()

    def on_step_end(self, args, state, control, **kwargs):
        if self._step_start is None:
            return
        elapsed = time.perf_counter() - self._step_start
        if state.global_step and state.global_step % self.log_steps == 0:
            row = {
                'event': 'optimizer_step',
                'global_step': state.global_step,
                'epoch': state.epoch,
                'seconds': round(elapsed, 4),
            }
            self.rows.append(row)
            print(
                f"[timing] step={state.global_step} "
                f"epoch={state.epoch:.4f} seconds={elapsed:.2f}",
                flush=True,
            )

    def on_epoch_end(self, args, state, control, **kwargs):
        if self._epoch_start is None:
            return
        elapsed = time.perf_counter() - self._epoch_start
        row = {
            'event': 'epoch',
            'global_step': state.global_step,
            'epoch': state.epoch,
            'seconds': round(elapsed, 4),
        }
        self.rows.append(row)
        print(
            f"[timing] epoch={state.epoch:.4f} seconds={elapsed:.2f}",
            flush=True,
        )

    def on_train_end(self, args, state, control, **kwargs):
        if self._train_start is None:
            return
        elapsed = time.perf_counter() - self._train_start
        self.rows.append({
            'event': 'train',
            'global_step': state.global_step,
            'epoch': state.epoch,
            'seconds': round(elapsed, 4),
        })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--auto_dev_split', type=float, default=None,
                        help='If no data.dev_formatted file exists, split this fraction '
                             'from train with sklearn train_test_split.')
    parser.add_argument(
        '--resume_from_checkpoint',
        nargs='?',
        const='auto',
        default=None,
        help='Resume from the latest checkpoint, or from an explicit checkpoint path.',
    )
    parser.add_argument('--max_steps', type=int, default=-1,
                        help='Diagnostic mode: stop after this many optimizer steps.')
    args = parser.parse_args()

    cfg = load_config(args.config, args.seed)
    set_seed(args.seed)

    method = cfg['method']
    run_name = cfg.get('run_name', method)
    group  = cfg['group']
    root   = Path(__file__).parent.parent

    train_path = root / 'data' / 'formatted' / cfg['data']['train']
    output_dir = root / 'checkpoints' / group / run_name / f"s{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Dev dataset: optional, derived from data.dev key in config
    dev_key = cfg.get('data', {}).get('dev_formatted')
    dev_path = (root / 'data' / 'formatted' / dev_key) if dev_key else None

    print(f"Method: {method} | Run: {run_name} | Seed: {args.seed} | Group: {group}")
    print(f"Train data: {train_path}")
    if dev_path:
        print(f"Dev data:   {dev_path}")
    print(f"Output dir: {output_dir}")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is not available. This training setup requires a GPU; "
            "check nvidia-smi, CUDA_VISIBLE_DEVICES, and the PyTorch CUDA build."
        )

    # ── Tokenizer ──────────────────────────────────────────────────────────
    model_name = cfg['model']['name']
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tr_cfg = cfg['training']
    use_eval_loss_selection = bool(tr_cfg.get('load_best_model_at_end', False))
    use_eval_loss_logging = bool(
        tr_cfg.get('eval_loss_during_training', use_eval_loss_selection)
    )

    # ── Dataset ────────────────────────────────────────────────────────────
    dataset = load_chat_dataset(
        str(train_path), tokenizer,
        max_length=tr_cfg.get('max_length', 2048))
    print(f"Train examples: {len(dataset)}")

    eval_dataset = None
    if use_eval_loss_logging and dev_path and dev_path.exists():
        eval_dataset = load_chat_dataset(
            str(dev_path), tokenizer,
            max_length=tr_cfg.get('max_length', 2048))
        print(f"Dev examples:  {len(eval_dataset)}")
    elif use_eval_loss_logging:
        auto_dev_split = args.auto_dev_split
        if auto_dev_split is None:
            auto_dev_split = cfg.get('data', {}).get('auto_dev_split', 0.0)
        if auto_dev_split and auto_dev_split > 0:
            dataset, eval_dataset = split_dataset_for_dev(
                dataset, auto_dev_split, cfg.get('data', {}).get('dev_seed', args.seed))
            print(f"Auto dev split: {auto_dev_split:.3f}")
            print(f"Train examples after split: {len(dataset)}")
            print(f"Dev examples after split:   {len(eval_dataset)}")
    elif dev_path and dev_path.exists():
        print("Dev loss eval disabled; external dev-Smatch selection will use raw dev inference.")

    train_batch_size = tr_cfg.get('batch_per_device', 1)
    grad_accum = tr_cfg.get('grad_accum', 16)
    micro_batches_per_epoch = math.ceil(len(dataset) / train_batch_size)
    optimizer_steps_per_epoch = math.ceil(micro_batches_per_epoch / grad_accum)
    print(
        "Training cost estimate: "
        f"{micro_batches_per_epoch} micro-batches/epoch, "
        f"{optimizer_steps_per_epoch} optimizer steps/epoch, "
        f"grad_accum={grad_accum}"
    )

    # ── Model ──────────────────────────────────────────────────────────────
    print(f"\nLoading model: {model_name}")
    use_bf16 = torch.cuda.is_bf16_supported()
    model_dtype = torch.bfloat16 if use_bf16 else torch.float16
    print(f"Precision: {'bf16' if use_bf16 else 'fp16'}")

    device_map_cfg = cfg.get('model', {}).get('device_map', 'single_cuda')
    if device_map_cfg in {'single_cuda', 'cuda'}:
        device_map = {'': torch.cuda.current_device()}
    elif device_map_cfg in {'none', None}:
        device_map = None
    elif device_map_cfg == 'auto':
        device_map = 'auto'
        print("WARNING: model.device_map=auto may offload parameters and break training.")
    else:
        raise ValueError(f"Unsupported model.device_map for training: {device_map_cfg}")
    print(f"Model device_map: {device_map_cfg}")

    model_kwargs = {
        'torch_dtype': model_dtype,
        'trust_remote_code': True,
    }
    if cfg.get('model', {}).get('flash_attn') and importlib.util.find_spec('flash_attn'):
        model_kwargs['attn_implementation'] = 'flash_attention_2'
    elif cfg.get('model', {}).get('flash_attn'):
        print("WARNING: model.flash_attn=true but package 'flash_attn' is not installed; using default attention.")
    elif cfg.get('model', {}).get('attn_implementation'):
        model_kwargs['attn_implementation'] = cfg['model']['attn_implementation']
    if device_map is not None:
        model_kwargs['device_map'] = device_map
    print(f"Attention implementation: {model_kwargs.get('attn_implementation', 'transformers default')}")

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

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
    use_gradient_checkpointing = bool(tr_cfg.get('gradient_checkpointing', True))
    if use_gradient_checkpointing:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
    print(f"Gradient checkpointing: {use_gradient_checkpointing}")

    # ── Training args ──────────────────────────────────────────────────────
    eval_batch_size = tr_cfg.get('eval_batch_per_device', train_batch_size)
    print(
        f"Batch sizes: train={train_batch_size}, eval={eval_batch_size}, "
        f"grad_accum={tr_cfg.get('grad_accum', 16)}"
    )

    use_eval_loss_selection = use_eval_loss_selection and eval_dataset is not None

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=tr_cfg['epochs'],
        max_steps=args.max_steps,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
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
        eval_accumulation_steps=1 if eval_dataset is not None else None,
        prediction_loss_only=eval_dataset is not None,
        load_best_model_at_end=use_eval_loss_selection,
        metric_for_best_model='eval_loss' if use_eval_loss_selection else None,
        greater_is_better=False if use_eval_loss_selection else None,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        return_tensors='pt',
    )

    callbacks = []
    timing_callback = TimingCallback(log_steps=tr_cfg.get('timing_log_steps', 10))
    callbacks.append(timing_callback)
    early_stopping_patience = tr_cfg.get('early_stopping_patience')
    if use_eval_loss_selection and early_stopping_patience:
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=int(early_stopping_patience),
            early_stopping_threshold=float(tr_cfg.get('early_stopping_threshold', 0.0)),
        ))
        print(
            "Early stopping: "
            f"patience={early_stopping_patience}, "
            f"threshold={tr_cfg.get('early_stopping_threshold', 0.0)}"
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    resume_from_checkpoint = None
    if args.resume_from_checkpoint == 'auto':
        resume_from_checkpoint = True
    elif args.resume_from_checkpoint:
        resume_from_checkpoint = args.resume_from_checkpoint
    if resume_from_checkpoint:
        print(f"Resuming from checkpoint: {resume_from_checkpoint}")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    log_history = trainer.state.log_history
    with open(output_dir / 'trainer_log_history.json', 'w') as f:
        json.dump(log_history, f, indent=2)
    with open(output_dir / 'timing_profile.json', 'w') as f:
        json.dump(timing_callback.rows, f, indent=2)
    print(f"Timing profile → {output_dir / 'timing_profile.json'}")

    if eval_dataset is not None:
        try:
            import matplotlib.pyplot as plt

            train_epochs = [float(x['epoch']) for x in log_history if 'loss' in x and 'epoch' in x]
            train_losses = [float(x['loss']) for x in log_history if 'loss' in x and 'epoch' in x]
            eval_epochs = [float(x['epoch']) for x in log_history if 'eval_loss' in x and 'epoch' in x]
            eval_losses = [float(x['eval_loss']) for x in log_history if 'eval_loss' in x and 'epoch' in x]

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
        'method': method, 'run_name': run_name, 'seed': args.seed, 'group': group,
        'config': args.config,
        'train_examples': len(dataset),
        'dev_examples': len(eval_dataset) if eval_dataset is not None else 0,
        'dev_formatted_path': str(dev_path) if dev_path else None,
        'eval_loss_during_training': use_eval_loss_logging,
        'train_batch_per_device': train_batch_size,
        'eval_batch_per_device': eval_batch_size,
        'grad_accum': grad_accum,
        'effective_batch_size': train_batch_size * grad_accum,
        'epochs': tr_cfg['epochs'],
        'max_steps': args.max_steps,
        'last_epoch': trainer.state.epoch,
        'global_step': trainer.state.global_step,
        'lora_r': lora_cfg['r'],
        'lr': tr_cfg['lr'],
        'gradient_checkpointing': use_gradient_checkpointing,
        'attention_implementation': model_kwargs.get('attn_implementation'),
        'micro_batches_per_epoch': micro_batches_per_epoch,
        'optimizer_steps_per_epoch': optimizer_steps_per_epoch,
        'early_stopping_patience': early_stopping_patience,
        'early_stopping_threshold': tr_cfg.get('early_stopping_threshold', 0.0),
        'selection_metric': (
            'eval_loss' if use_eval_loss_selection
            else tr_cfg.get('selection_metric')
        ),
        'final_adapter_path': str(final_dir),
        'best_checkpoint': trainer.state.best_model_checkpoint,
        'best_eval_loss': (
            trainer.state.best_metric if use_eval_loss_selection else None
        ),
    }
    with open(output_dir / 'train_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)


if __name__ == '__main__':
    main()
