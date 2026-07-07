#!/usr/bin/env python3
from __future__ import annotations

"""Delete non-selected checkpoints after dev-Smatch selection.

Default mode is dry-run. The script reads
results/<group>/<run_name>/dev_epoch_scores/s<seed>/best_dev_checkpoint.json
and keeps only the selected checkpoint for that run/seed.
"""

import argparse
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def checkpoint_step(path: Path) -> int:
    return int(path.name.replace("checkpoint-", ""))


def list_checkpoints(run_dir: Path) -> list[Path]:
    return sorted(
        [
            path for path in run_dir.glob("checkpoint-*")
            if path.is_dir() and path.name.replace("checkpoint-", "").isdigit()
        ],
        key=checkpoint_step,
    )


def resolve_checkpoint(path_text: str) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def iter_best_files(group: str, run_name: str | None, seed: int | None):
    results_group = ROOT / "results" / group
    run_dirs = [results_group / run_name] if run_name else sorted(results_group.glob("*"))
    for run_dir in run_dirs:
        if not run_dir.is_dir():
            continue
        seed_dirs = sorted((run_dir / "dev_epoch_scores").glob("s*"))
        for seed_dir in seed_dirs:
            if seed is not None and seed_dir.name != f"s{seed}":
                continue
            best_file = seed_dir / "best_dev_checkpoint.json"
            if best_file.exists():
                yield run_dir.name, seed_dir.name, best_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", default="vietnamese_v2_epoch_smatch")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--execute", action="store_true",
                        help="Actually delete checkpoints. Default is dry-run.")
    args = parser.parse_args()

    plans = []
    for run_name, seed_name, best_file in iter_best_files(args.group, args.run_name, args.seed):
        with open(best_file, encoding="utf-8") as f:
            best = json.load(f)
        selected = resolve_checkpoint(best["checkpoint"])
        run_dir = ROOT / "checkpoints" / args.group / run_name / seed_name
        checkpoints = list_checkpoints(run_dir)
        if not checkpoints:
            print(f"SKIP no checkpoints: {run_dir}")
            continue

        selected_matches = [path for path in checkpoints if path.resolve() == selected]
        if not selected_matches:
            print(f"SKIP selected checkpoint is missing for {run_name}/{seed_name}: {selected}")
            continue

        delete_paths = [path for path in checkpoints if path.resolve() != selected]
        plans.append({
            "run_name": run_name,
            "seed": seed_name,
            "selected": str(selected_matches[0]),
            "delete_count": len(delete_paths),
            "delete_paths": [str(path) for path in delete_paths],
        })

        print(f"{run_name}/{seed_name}")
        print(f"  keep:   {selected_matches[0]}")
        print(f"  delete: {len(delete_paths)} checkpoints")
        for path in delete_paths:
            print(f"    {path}")

        if args.execute:
            for path in delete_paths:
                shutil.rmtree(path)

    out_path = ROOT / "results" / args.group / "checkpoint_prune_plan.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(plans, f, indent=2, ensure_ascii=False)

    mode = "EXECUTED" if args.execute else "DRY-RUN"
    print(f"{mode}: wrote plan -> {out_path}")
    if not args.execute:
        print("Run again with --execute to delete the listed checkpoints.")


if __name__ == "__main__":
    main()
