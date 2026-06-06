#!/usr/bin/env python3
"""Generate small RandomizedSearchCV-style learning-rate runs.

This helper keeps expensive LoRA training outside sklearn's estimator API, but
uses sklearn's random search sampler so the search space and seed are explicit.

Example:
    python src/tune_lr.py --config configs/vietnamese_v2/MTUP2.yaml --seed 42

Then run each printed command. Pick the run with the lowest dev loss from
checkpoints/.../trainer_log_history.json before running public-test inference.
"""

import argparse
import copy
import json
from pathlib import Path

import yaml

try:
    from scipy.stats import loguniform
    from sklearn.model_selection import ParameterSampler
except ImportError as exc:
    raise SystemExit(
        "Missing tuning dependencies. Install scikit-learn and scipy first."
    ) from exc


def load_config(path: Path) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if 'base' in cfg:
        base_path = path.parent / cfg['base']
        with open(base_path) as f:
            base = yaml.safe_load(f)
        base.update(cfg)
        cfg = base
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_iter', type=int, default=5)
    parser.add_argument('--lr_min', type=float, default=1e-6)
    parser.add_argument('--lr_max', type=float, default=5e-5)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--out_dir', default='configs/tuning')
    args = parser.parse_args()

    config_path = Path(args.config)
    cfg = load_config(config_path)
    if not cfg.get('data', {}).get('dev_formatted'):
        raise SystemExit(
            "Config must define data.dev_formatted. Generate data with "
            "src/data_pipeline.py --dev_split 0.1 first."
        )

    sampler = ParameterSampler(
        {'lr': loguniform(args.lr_min, args.lr_max)},
        n_iter=args.n_iter,
        random_state=args.seed,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    commands = []

    for i, params in enumerate(sampler, start=1):
        tuned = copy.deepcopy(cfg)
        tuned.pop('base', None)
        tuned['group'] = f"{cfg['group']}_lrsearch"
        tuned['training']['lr'] = float(params['lr'])
        tuned['training']['epochs'] = args.epochs
        tuned['training']['batch_per_device'] = tuned['training'].get('batch_per_device', 1)
        tuned['training']['grad_accum'] = tuned['training'].get('grad_accum', 16)
        tuned_path = out_dir / f"{config_path.stem}_lr{i}.yaml"
        with open(tuned_path, 'w') as f:
            yaml.safe_dump(tuned, f, sort_keys=False)
        commands.append(f"python src/train.py --config {tuned_path} --seed {args.seed}")

    manifest = {
        'source_config': str(config_path),
        'seed': args.seed,
        'n_iter': args.n_iter,
        'lr_min': args.lr_min,
        'lr_max': args.lr_max,
        'epochs': args.epochs,
        'commands': commands,
    }
    manifest_path = out_dir / f"{config_path.stem}_lrsearch_manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote tuning configs to {out_dir}")
    print(f"Manifest: {manifest_path}")
    print("\nCommands:")
    for command in commands:
        print(command)


if __name__ == '__main__':
    main()
