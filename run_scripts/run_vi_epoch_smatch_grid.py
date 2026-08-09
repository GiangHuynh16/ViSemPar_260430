#!/usr/bin/env python3
from __future__ import annotations

"""Run Vietnamese v2 checkpoint selection by final-AMR dev Smatch.

This runner trains each run for a fixed number of epochs, keeps every epoch
checkpoint, scores each checkpoint on the held-out dev set, selects the best
checkpoint by final-AMR Smatch F1, and only then evaluates that checkpoint on
the public test set.
"""

import argparse
import copy
import csv
import json
import shlex
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from config_utils import load_yaml_config, yaml  # noqa: E402
from evaluate import compute_metrics, read_gold, read_predictions  # noqa: E402


DEFAULT_CONFIGS = {
    "B1": "configs/vietnamese_v2/B1.yaml",
    "B2": "configs/vietnamese_v2/B2.yaml",
    "MTUP2": "configs/vietnamese_v2/MTUP2.yaml",
    "MTUP3": "configs/vietnamese_v2/MTUP3.yaml",
    "MTUP4": "configs/vietnamese_v2/MTUP4.yaml",
}
DEFAULT_MTUP2_LRS = [2e-4, 1e-4, 5e-5]
DEFAULT_MTUP3_LRS = [5e-5]
DEFAULT_MTUP4_LRS = [5e-5]
DEFAULT_BASELINE_LRS = [2e-4, 1e-4, 5e-5, 1e-5]


def run_command(cmd: list[str], dry_run: bool = False, log_path: Path | None = None) -> None:
    print("$ " + " ".join(shlex.quote(part) for part in cmd), flush=True)
    if dry_run:
        return
    if log_path is None:
        subprocess.run(cmd, cwd=ROOT, check=True)
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            cmd,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)


def count_sentence_lines(path: Path) -> int:
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip() and not line.startswith("#"))


def preflight(args: argparse.Namespace, dev_gold: Path, test_gold: Path) -> None:
    dev_sentences = count_sentence_lines(ROOT / "data/formatted/vietnamese_v2/dev.txt")
    dev_gold_count = len(read_gold(str(dev_gold)))
    if dev_sentences != dev_gold_count:
        raise RuntimeError(
            "Dev sentence/gold count mismatch: "
            f"{dev_sentences} sentences vs {dev_gold_count} gold AMRs"
        )

    test_sentences = count_sentence_lines(ROOT / "data/formatted/vietnamese_v2/test.txt")
    test_gold_count = len(read_gold(str(test_gold)))
    if test_sentences != test_gold_count:
        raise RuntimeError(
            "Test sentence/gold count mismatch: "
            f"{test_sentences} sentences vs {test_gold_count} gold AMRs"
        )

    print(
        f"Preflight counts: dev={dev_sentences}/{dev_gold_count}, "
        f"test={test_sentences}/{test_gold_count}"
    )

    methods = set(args.methods)
    if {"B2", "MTUP2"}.issubset(methods):
        run_command([
            sys.executable,
            "src/audit_b2.py",
            "--b2",
            "data/formatted/vietnamese_v2/B2_train.txt",
            "--mtup2",
            "data/formatted/vietnamese_v2/MTUP2_train.txt",
            "--b2_dev",
            "data/formatted/vietnamese_v2/B2_dev.txt",
            "--mtup2_dev",
            "data/formatted/vietnamese_v2/MTUP2_dev.txt",
        ], dry_run=args.dry_run)

    if "MTUP2" in methods:
        run_command([
            sys.executable,
            "src/audit_mtup2_setup.py",
            "--config",
            "configs/vietnamese_v2/MTUP2.yaml",
        ], dry_run=args.dry_run)


def lr_label(lr: float) -> str:
    return f"{lr:.0e}".replace("+", "")


def checkpoint_step(path: Path) -> int:
    return int(path.name.replace("checkpoint-", ""))


def list_checkpoints(run_dir: Path) -> list[Path]:
    checkpoints = [
        path for path in run_dir.glob("checkpoint-*")
        if path.is_dir() and path.name.replace("checkpoint-", "").isdigit()
    ]
    return sorted(checkpoints, key=checkpoint_step)


def checkpoints_to_score(checkpoints: list[Path], interval: int) -> list[Path]:
    if interval <= 1:
        return checkpoints
    selected = []
    for index, checkpoint in enumerate(checkpoints, start=1):
        if index % interval == 0 or index == len(checkpoints):
            selected.append(checkpoint)
    return selected


def write_config(
    source_config: Path,
    out_dir: Path,
    run_name: str,
    group: str,
    epochs: int,
    lr: float | None,
    batch_per_device: int | None,
    grad_accum: int | None,
    gradient_checkpointing: bool | None,
    flash_attn: bool | None,
    dry_run: bool = False,
) -> Path:
    cfg = copy.deepcopy(load_yaml_config(source_config))
    cfg.pop("base", None)
    cfg["group"] = group
    cfg["run_name"] = run_name
    if flash_attn is not None:
        model = cfg.setdefault("model", {})
        model["flash_attn"] = bool(flash_attn)
    training = cfg.setdefault("training", {})
    training["epochs"] = epochs
    training["save_total_limit"] = epochs
    training["early_stopping_patience"] = None
    training["load_best_model_at_end"] = False
    training["eval_loss_during_training"] = False
    training["selection_metric"] = "dev_smatch_final_amr"
    if lr is not None:
        training["lr"] = float(lr)
    if batch_per_device is not None:
        training["batch_per_device"] = int(batch_per_device)
    if grad_accum is not None:
        training["grad_accum"] = int(grad_accum)
    if gradient_checkpointing is not None:
        training["gradient_checkpointing"] = bool(gradient_checkpointing)

    path = out_dir / f"{run_name}.yaml"
    if dry_run:
        return path

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return path


def run_name_suffix(args: argparse.Namespace) -> str:
    parts = []
    if args.batch_per_device is not None or args.grad_accum is not None:
        batch = args.batch_per_device if args.batch_per_device is not None else "cfg"
        accum = args.grad_accum if args.grad_accum is not None else "cfg"
        parts.append(f"b{batch}_ga{accum}")
    if args.gradient_checkpointing != "config":
        parts.append("gc" if args.gradient_checkpointing == "on" else "nogc")
    if args.timing_probe_steps > 0:
        parts.append(f"probe{args.timing_probe_steps}")
    return "_" + "_".join(parts) if parts else ""


def gradient_checkpointing_override(value: str) -> bool | None:
    if value == "config":
        return None
    return value == "on"


def flash_attn_override(value: str) -> bool | None:
    if value == "config":
        return None
    return value == "on"


def warn_training_override(
    method: str,
    base_cfg: dict,
    args: argparse.Namespace,
) -> None:
    training = base_cfg["training"]
    base_batch = int(training.get("batch_per_device", 1))
    base_accum = int(training.get("grad_accum", 16))
    batch = int(args.batch_per_device or base_batch)
    accum = int(args.grad_accum or base_accum)
    base_effective = base_batch * base_accum
    effective = batch * accum

    if args.batch_per_device is not None or args.grad_accum is not None:
        print(
            f"Training batch override for {method}: "
            f"batch_per_device={batch}, grad_accum={accum}, "
            f"effective_batch={effective}."
        )
        if effective != base_effective:
            print(
                "WARNING: effective batch changed from "
                f"{base_effective} to {effective}. This is still applied "
                "uniformly by this runner, but it changes the optimization "
                "protocol and should be reported."
            )

    if args.gradient_checkpointing != "config":
        print(
            f"Gradient checkpointing override for {method}: "
            f"{args.gradient_checkpointing}. Apply the same setting to all "
            "methods for fair timing/comparison."
        )


def checkpoint_epoch_map(run_dir: Path) -> dict[int, float]:
    log_path = run_dir / "trainer_log_history.json"
    if not log_path.exists():
        return {}
    with open(log_path, encoding="utf-8") as f:
        rows = json.load(f)
    out = {}
    for row in rows:
        if "step" in row and "epoch" in row and "eval_loss" in row:
            out[int(row["step"])] = float(row["epoch"])
    return out


def score_file(pred_path: Path, gold_path: Path) -> dict:
    preds = read_predictions(str(pred_path))
    gold = read_gold(str(gold_path))
    if len(preds) != len(gold):
        raise RuntimeError(
            f"Prediction/gold count mismatch for {pred_path}: "
            f"{len(preds)} predictions vs {len(gold)} gold AMRs"
        )
    return compute_metrics(preds, gold)


def maybe_train(config_path: Path, seed: int, run_dir: Path, epochs: int, args) -> None:
    checkpoints = list_checkpoints(run_dir)
    if len(checkpoints) >= epochs and not args.force_train:
        print(f"Found {len(checkpoints)} checkpoints in {run_dir}; skipping train.")
        return

    cmd = [
        sys.executable,
        "src/train.py",
        "--config",
        str(config_path),
        "--seed",
        str(seed),
    ]
    if checkpoints and not args.force_train:
        cmd.extend(["--resume_from_checkpoint", "auto"])
    if args.timing_probe_steps > 0:
        cmd.extend(["--max_steps", str(args.timing_probe_steps)])
    run_command(cmd, dry_run=args.dry_run, log_path=run_dir / "train_command.log")


def score_dev_checkpoints(
    config_path: Path,
    method: str,
    run_name: str,
    group: str,
    seed: int,
    run_dir: Path,
    dev_gold: Path,
    args,
) -> tuple[dict, list[dict]]:
    score_dir = ROOT / "results" / group / run_name / "dev_epoch_scores" / f"s{seed}"
    score_dir.mkdir(parents=True, exist_ok=True)
    score_path = score_dir / "dev_epoch_scores.json"

    if score_path.exists() and args.reuse_dev_scores:
        with open(score_path, encoding="utf-8") as f:
            records = json.load(f)
        best = max(records, key=lambda row: (row["f1"], -row["epoch_sort"]))
        return best, records

    epoch_by_step = checkpoint_epoch_map(run_dir)
    records = []
    checkpoints = list_checkpoints(run_dir)
    for checkpoint in checkpoints_to_score(checkpoints, args.dev_eval_interval):
        index = checkpoints.index(checkpoint) + 1
        step = checkpoint_step(checkpoint)
        epoch = epoch_by_step.get(step, float(index))
        prefix = f"s{seed}_{checkpoint.name}_dev"
        pred_path = score_dir / f"{prefix}_pred.txt"
        if not pred_path.exists() or not args.reuse_predictions:
            run_command([
                sys.executable,
                "src/inference.py",
                "--config",
                str(config_path),
                "--seed",
                str(seed),
                "--split",
                "val",
                "--adapter_path",
                str(checkpoint),
                "--output_dir",
                str(score_dir),
                "--output_prefix",
                prefix,
            ], dry_run=args.dry_run)
        if args.dry_run:
            continue

        metrics = score_file(pred_path, dev_gold)
        record = {
            "method": method,
            "run_name": run_name,
            "group": group,
            "seed": seed,
            "checkpoint": str(checkpoint),
            "checkpoint_step": step,
            "epoch": epoch,
            "epoch_sort": float(epoch),
            "pred_path": str(pred_path),
            "dev_eval_interval": args.dev_eval_interval,
            **metrics,
        }
        records.append(record)
        print(
            f"DEV {run_name} s{seed} epoch={epoch:g} "
            f"f1={metrics['f1']:.4f} parsable={metrics['parsable_rate']:.4f}",
            flush=True,
        )

    if args.dry_run:
        return {}, []
    if not records:
        raise RuntimeError(f"No dev scores produced for {run_name} seed {seed}")

    with open(score_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    best = max(records, key=lambda row: (row["f1"], -row["epoch_sort"]))
    with open(score_dir / "best_dev_checkpoint.json", "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2, ensure_ascii=False)
    return best, records


def evaluate_best_on_test(
    config_path: Path,
    method: str,
    run_name: str,
    group: str,
    seed: int,
    best: dict,
    test_gold: Path,
    args,
) -> dict:
    out_dir = ROOT / "results" / group / run_name / "selected_test"
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"s{seed}_best_test"
    pred_path = out_dir / f"{prefix}_pred.txt"

    if not pred_path.exists() or not args.reuse_predictions:
        run_command([
            sys.executable,
            "src/inference.py",
            "--config",
            str(config_path),
            "--seed",
            str(seed),
            "--split",
            "test",
            "--adapter_path",
            best["checkpoint"],
            "--output_dir",
            str(out_dir),
            "--output_prefix",
            prefix,
        ], dry_run=args.dry_run)
    if args.dry_run:
        return {}

    metrics = score_file(pred_path, test_gold)
    record = {
        "method": method,
        "run_name": run_name,
        "group": group,
        "seed": seed,
        "selection_metric": "dev_smatch_final_amr",
        "selected_checkpoint": best["checkpoint"],
        "selected_epoch": best["epoch"],
        "dev_f1": best["f1"],
        "test_pred_path": str(pred_path),
        **metrics,
    }
    score_path = out_dir / f"s{seed}_test_score.json"
    with open(score_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    print(
        f"TEST {run_name} s{seed} selected_epoch={best['epoch']:g} "
        f"dev_f1={best['f1']:.4f} test_f1={metrics['f1']:.4f}",
        flush=True,
    )
    return record


def summarize(group: str, records: list[dict]) -> None:
    if not records:
        return
    out_dir = ROOT / "results" / group
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "epoch_smatch_test_records.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    rows = []
    for run_name in sorted({row["run_name"] for row in records}):
        run_rows = [row for row in records if row["run_name"] == run_name]
        f1s = [float(row["f1"]) for row in run_rows]
        rows.append({
            "method": run_rows[0]["method"],
            "run_name": run_name,
            "seeds": " ".join(str(row["seed"]) for row in run_rows),
            "n": len(run_rows),
            "test_f1_mean": statistics.mean(f1s),
            "test_f1_std": statistics.stdev(f1s) if len(f1s) > 1 else 0.0,
            "test_f1_values": " ".join(f"{value:.4f}" for value in f1s),
            "dev_f1_values": " ".join(f"{row['dev_f1']:.4f}" for row in run_rows),
            "selected_epochs": " ".join(f"{row['selected_epoch']:g}" for row in run_rows),
        })

    csv_path = out_dir / "epoch_smatch_test_summary.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Summary CSV: {csv_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", default=["MTUP2"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--dev-eval-interval", type=int, default=5,
                        help="Score every Nth epoch checkpoint on dev; last checkpoint is always scored.")
    parser.add_argument("--score-all-checkpoints", action="store_true",
                        help="Equivalent to --dev-eval-interval 1.")
    parser.add_argument("--group", default="vietnamese_v2_epoch_smatch")
    parser.add_argument("--config-dir", default="configs/generated/vi_epoch_smatch")
    parser.add_argument("--mtup2-lrs", nargs="+", type=float, default=DEFAULT_MTUP2_LRS)
    parser.add_argument("--mtup3-lrs", nargs="+", type=float, default=DEFAULT_MTUP3_LRS,
                        help="Learning rates for the three-subtask MTUP3 method.")
    parser.add_argument("--mtup4-lrs", nargs="+", type=float, default=DEFAULT_MTUP4_LRS,
                        help="Learning rates for the four-subtask MTUP4 method.")
    parser.add_argument("--baseline-lrs", nargs="+", type=float, default=DEFAULT_BASELINE_LRS)
    parser.add_argument("--tune-baselines", action="store_true",
                        help="Run B1/B2 over --baseline-lrs instead of only their config LR.")
    parser.add_argument("--batch-per-device", type=int, default=1,
                        help="Override training.batch_per_device for every generated run.")
    parser.add_argument("--grad-accum", type=int, default=16,
                        help="Override training.grad_accum for every generated run.")
    parser.add_argument("--gradient-checkpointing", choices=["config", "on", "off"], default="config",
                        help="Override gradient checkpointing for every generated run.")
    parser.add_argument("--flash-attn", choices=["config", "on", "off"], default="config",
                        help="Override model.flash_attn for every generated run.")
    parser.add_argument("--timing-probe-steps", type=int, default=-1,
                        help="Run train.py for only N optimizer steps and skip dev/test scoring.")
    parser.add_argument("--dev-gold", default="data/formatted/vietnamese_v2/dev_ground_truth.txt")
    parser.add_argument("--test-gold", default="data/formatted/vietnamese_v2/test_ground_truth.txt")
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--reuse-predictions", action="store_true")
    parser.add_argument("--reuse-dev-scores", action="store_true")
    parser.add_argument("--skip-audits", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.score_all_checkpoints:
        args.dev_eval_interval = 1
    if args.dev_eval_interval < 1:
        parser.error("--dev-eval-interval must be >= 1")
    if args.batch_per_device is not None and args.batch_per_device < 1:
        parser.error("--batch-per-device must be >= 1")
    if args.grad_accum is not None and args.grad_accum < 1:
        parser.error("--grad-accum must be >= 1")
    if args.timing_probe_steps == 0 or args.timing_probe_steps < -1:
        parser.error("--timing-probe-steps must be positive, or omitted")
    return args


def main() -> None:
    args = parse_args()
    config_dir = ROOT / args.config_dir
    dev_gold = ROOT / args.dev_gold
    test_gold = ROOT / args.test_gold
    if not dev_gold.exists():
        raise SystemExit(f"Missing dev gold file: {dev_gold}")
    if not test_gold.exists():
        raise SystemExit(f"Missing test gold file: {test_gold}")
    if not args.skip_audits:
        preflight(args, dev_gold, test_gold)

    test_records = []
    for method in args.methods:
        if method not in DEFAULT_CONFIGS:
            raise SystemExit(f"Unsupported method {method}; choose from {sorted(DEFAULT_CONFIGS)}")
        source_config = ROOT / DEFAULT_CONFIGS[method]
        base_cfg = load_yaml_config(source_config)
        if method == "MTUP2":
            lrs = args.mtup2_lrs
        elif method == "MTUP3":
            lrs = args.mtup3_lrs
        elif method == "MTUP4":
            lrs = args.mtup4_lrs
        elif args.tune_baselines:
            lrs = args.baseline_lrs
        else:
            lrs = [float(base_cfg["training"]["lr"])]
        warn_training_override(method, base_cfg, args)

        for lr in lrs:
            run_name = f"{method}_lr{lr_label(lr)}{run_name_suffix(args)}"
            config_path = write_config(
                source_config=source_config,
                out_dir=config_dir,
                run_name=run_name,
                group=args.group,
                epochs=args.epochs,
                lr=lr,
                batch_per_device=args.batch_per_device,
                grad_accum=args.grad_accum,
                gradient_checkpointing=gradient_checkpointing_override(args.gradient_checkpointing),
                flash_attn=flash_attn_override(args.flash_attn),
                dry_run=args.dry_run,
            )
            for seed in args.seeds:
                run_dir = ROOT / "checkpoints" / args.group / run_name / f"s{seed}"
                maybe_train(config_path, seed, run_dir, args.epochs, args)
                if args.timing_probe_steps > 0:
                    continue
                best, _records = score_dev_checkpoints(
                    config_path=config_path,
                    method=method,
                    run_name=run_name,
                    group=args.group,
                    seed=seed,
                    run_dir=run_dir,
                    dev_gold=dev_gold,
                    args=args,
                )
                if args.dry_run:
                    continue
                test_record = evaluate_best_on_test(
                    config_path=config_path,
                    method=method,
                    run_name=run_name,
                    group=args.group,
                    seed=seed,
                    best=best,
                    test_gold=test_gold,
                    args=args,
                )
                if test_record:
                    test_records.append(test_record)

    if not args.dry_run:
        summarize(args.group, test_records)


if __name__ == "__main__":
    main()
