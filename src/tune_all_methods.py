#!/usr/bin/env python3
from __future__ import annotations

"""Generate and summarize LR-search runs for all Vietnamese v2 methods.

The same sampled learning-rate list is used for every method so the search is
comparable across B1, B2, and MTUP variants. Public-test scores are not used.
"""

import argparse
import copy
import json
import stat
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


DEFAULT_CONFIGS = [
    "configs/vietnamese_v2/B1.yaml",
    "configs/vietnamese_v2/B2.yaml",
    "configs/vietnamese_v2/MTUP2.yaml",
    "configs/vietnamese_v2/MTUP3A.yaml",
    "configs/vietnamese_v2/MTUP3B.yaml",
    "configs/vietnamese_v2/MTUP4.yaml",
]


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


def generate(args: argparse.Namespace) -> None:
    config_paths = [Path(path) for path in args.configs]
    samples = list(ParameterSampler(
        {"lr": loguniform(args.lr_min, args.lr_max)},
        n_iter=args.n_iter,
        random_state=args.seed,
    ))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_commands = []
    methods = {}

    for config_path in config_paths:
        cfg = load_config(config_path)
        method = cfg["method"]
        if not cfg.get("data", {}).get("dev_formatted"):
            raise SystemExit(
                f"{config_path} must define data.dev_formatted. "
                "Generate data with src/data_pipeline.py --dev_split 0.1 first."
            )

        method_configs = []
        method_commands = []

        for i, params in enumerate(samples, start=1):
            tuned = copy.deepcopy(cfg)
            tuned.pop("base", None)
            tuned["group"] = f"{cfg['group']}_lrsearch"
            tuned["run_name"] = f"{method}_lr{i}"
            tuned["training"]["lr"] = float(params["lr"])
            tuned["training"]["epochs"] = args.epochs
            tuned["training"]["batch_per_device"] = tuned["training"].get("batch_per_device", 1)
            tuned["training"]["grad_accum"] = tuned["training"].get("grad_accum", 16)

            tuned_path = out_dir / f"{method}_lr{i}.yaml"
            with open(tuned_path, "w") as f:
                yaml.safe_dump(tuned, f, sort_keys=False)

            command = f"python3 src/train.py --config {tuned_path} --seed {args.seed}"
            method_configs.append(str(tuned_path))
            method_commands.append(command)
            all_commands.append(command)

        methods[method] = {
            "source_config": str(config_path),
            "group": cfg["group"],
            "search_group": f"{cfg['group']}_lrsearch",
            "configs": method_configs,
            "commands": method_commands,
        }

    manifest = {
        "seed": args.seed,
        "n_iter": args.n_iter,
        "lr_min": args.lr_min,
        "lr_max": args.lr_max,
        "epochs": args.epochs,
        "lr_samples": [float(sample["lr"]) for sample in samples],
        "selection_metric": "lowest dev eval_loss; public test is not used",
        "methods": methods,
        "commands": all_commands,
    }
    manifest_path = out_dir / "vietnamese_v2_all_lrsearch_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    commands_path = out_dir / "run_all_lrsearch.sh"
    with open(commands_path, "w") as f:
        f.write("#!/usr/bin/env bash\nset -e\n\n")
        for command in all_commands:
            f.write(command + "\n")
    commands_path.chmod(commands_path.stat().st_mode | stat.S_IXUSR)

    print(f"Wrote tuning configs to {out_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"Run script: {commands_path}")
    print("\nCommands:")
    for command in all_commands:
        print(command)


def summarize(args: argparse.Namespace) -> None:
    manifest_path = Path(args.summarize)
    with open(manifest_path) as f:
        manifest = json.load(f)

    results = {}
    for method, method_info in manifest["methods"].items():
        rows = []
        for config_name in method_info["configs"]:
            config_path = Path(config_name)
            cfg = load_config(config_path)
            run_name = cfg.get("run_name", cfg["method"])
            ckpt_dir = (
                Path("checkpoints")
                / cfg["group"]
                / run_name
                / f"s{manifest['seed']}"
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
        selected = min(complete, key=lambda row: row["best_eval_loss"]) if complete else None
        results[method] = {
            "source_config": method_info["source_config"],
            "selected": selected,
            "runs": rows,
        }

    summary = {
        "manifest": str(manifest_path),
        "selection_metric": manifest["selection_metric"],
        "methods": results,
    }
    summary_path = manifest_path.with_name("vietnamese_v2_all_lrsearch_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Summary: {summary_path}")
    for method, info in results.items():
        selected = info["selected"]
        if selected:
            print(
                f"{method}: lr={selected['lr']:.6g} "
                f"epoch={selected['best_epoch']} "
                f"last_epoch={selected['last_epoch']} "
                f"eval_loss={selected['best_eval_loss']:.6f}"
            )
        else:
            print(f"{method}: no completed LR-search run found")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_iter", type=int, default=5)
    parser.add_argument("--lr_min", type=float, default=1e-6)
    parser.add_argument("--lr_max", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--out_dir", default="configs/tuning/vietnamese_v2_all")
    parser.add_argument("--summarize", metavar="MANIFEST")
    args = parser.parse_args()

    if args.summarize:
        summarize(args)
    else:
        generate(args)


if __name__ == "__main__":
    main()
