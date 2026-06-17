#!/usr/bin/env python3
"""Generate and summarize RandomizedSearchCV-style learning-rate runs.

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
from config_utils import load_yaml_config

try:
    from scipy.stats import loguniform
    from sklearn.model_selection import ParameterSampler
except ImportError as exc:
    raise SystemExit(
        "Missing tuning dependencies. Install scikit-learn and scipy first."
    ) from exc


def load_config(path: Path) -> dict:
    return load_yaml_config(path)


def to_float(value) -> float:
    return float(value)


def best_eval_from_log(log_path: Path) -> dict | None:
    if not log_path.exists():
        return None
    with open(log_path) as f:
        rows = json.load(f)
    eval_rows = [row for row in rows if "eval_loss" in row and "epoch" in row]
    if not eval_rows:
        return None
    best = min(eval_rows, key=lambda row: to_float(row["eval_loss"]))
    train_rows = [row for row in rows if "loss" in row and "epoch" in row]
    epoch_rows = [row for row in rows if "epoch" in row]
    last_epoch = max((to_float(row["epoch"]) for row in epoch_rows), default=None)
    train_at_epoch = None
    same_epoch = [
        row for row in train_rows
        if round(to_float(row["epoch"]), 4) == round(to_float(best["epoch"]), 4)
    ]
    if same_epoch:
        train_at_epoch = to_float(same_epoch[-1]["loss"])
    return {
        "best_epoch": to_float(best["epoch"]),
        "last_epoch": last_epoch,
        "best_eval_loss": to_float(best["eval_loss"]),
        "train_loss_at_best_epoch": train_at_epoch,
    }


def summarize_runs(manifest_path: Path, seed: int) -> None:
    with open(manifest_path) as f:
        manifest = json.load(f)

    rows = []
    for config_name in manifest["configs"]:
        config_path = Path(config_name)
        cfg = load_config(config_path)
        run_name = cfg.get("run_name", cfg["method"])
        ckpt_dir = (
            Path("checkpoints")
            / cfg["group"]
            / run_name
            / f"s{seed}"
        )
        best = best_eval_from_log(ckpt_dir / "trainer_log_history.json")
        rows.append({
            "config": str(config_path),
            "run_name": run_name,
            "lr": cfg["training"]["lr"],
            "max_epochs": cfg["training"]["epochs"],
            "checkpoint_dir": str(ckpt_dir),
            **(best or {
                "best_epoch": None,
                "last_epoch": None,
                "best_eval_loss": None,
                "train_loss_at_best_epoch": None,
            }),
        })

    complete = [row for row in rows if row["best_eval_loss"] is not None]
    if complete:
        selected = min(complete, key=lambda row: row["best_eval_loss"])
    else:
        selected = None

    summary = {
        "manifest": str(manifest_path),
        "selection_metric": "lowest eval_loss; public test is not used",
        "selected": selected,
        "runs": rows,
    }
    summary_path = manifest_path.with_name(
        manifest_path.stem.replace("_manifest", "_summary") + ".json"
    )
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Summary: {summary_path}")
    if selected:
        print(
            "Selected "
            f"lr={selected['lr']:.6g} epoch={selected['best_epoch']} "
            f"last_epoch={selected['last_epoch']} "
            f"eval_loss={selected['best_eval_loss']:.6f}"
        )
    else:
        print("No completed runs with eval_loss were found yet.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_iter', type=int, default=5)
    parser.add_argument('--lr_min', type=float, default=1e-6)
    parser.add_argument('--lr_max', type=float, default=5e-5)
    parser.add_argument('--epochs', type=int, default=50,
                        help='Train each sampled LR up to this many epochs; choose best epoch by dev loss.')
    parser.add_argument('--out_dir', default='configs/tuning')
    parser.add_argument('--summarize', metavar='MANIFEST',
                        help='Summarize completed runs from a manifest JSON instead of creating configs.')
    args = parser.parse_args()

    if args.summarize:
        summarize_runs(Path(args.summarize), args.seed)
        return

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
    config_paths = []

    for i, params in enumerate(sampler, start=1):
        tuned = copy.deepcopy(cfg)
        tuned.pop('base', None)
        tuned['group'] = f"{cfg['group']}_lrsearch"
        tuned['run_name'] = f"{cfg['method']}_lr{i}"
        tuned['training']['lr'] = float(params['lr'])
        tuned['training']['epochs'] = args.epochs
        tuned['training']['batch_per_device'] = tuned['training'].get('batch_per_device', 1)
        tuned['training']['grad_accum'] = tuned['training'].get('grad_accum', 16)
        tuned_path = out_dir / f"{config_path.stem}_lr{i}.yaml"
        with open(tuned_path, 'w') as f:
            yaml.safe_dump(tuned, f, sort_keys=False)
        config_paths.append(str(tuned_path))
        commands.append(f"python3 src/train.py --config {tuned_path} --seed {args.seed}")

    manifest = {
        'source_config': str(config_path),
        'seed': args.seed,
        'n_iter': args.n_iter,
        'lr_min': args.lr_min,
        'lr_max': args.lr_max,
        'epochs': args.epochs,
        'selection_metric': 'lowest dev eval_loss; do not inspect public-test score',
        'configs': config_paths,
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
