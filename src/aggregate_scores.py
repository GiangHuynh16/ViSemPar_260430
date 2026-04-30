#!/usr/bin/env python3
"""
Scan all results/**/scores.json and build results/summary/all_scores.csv.

Usage:
    python src/aggregate_scores.py
"""

import json
import csv
from pathlib import Path
from statistics import mean, stdev


SEEDS = [42, 123, 456]
METHODS_ORDER = ['B1', 'B2', 'MTUP2', 'MTUP3A', 'MTUP3B', 'MTUP4']


def main():
    root = Path(__file__).parent.parent
    results_dir = root / 'results'
    summary_dir = root / 'results' / 'summary'
    summary_dir.mkdir(parents=True, exist_ok=True)

    # Collect all scores.json
    all_data = {}  # method -> {seed -> scores_dict}

    for scores_file in sorted(results_dir.rglob('scores.json')):
        with open(scores_file) as f:
            data = json.load(f)

        for seed_key, scores in data.items():
            method = scores.get('method', '?')
            seed   = scores.get('seed', 0)
            if method not in all_data:
                all_data[method] = {}
            all_data[method][seed] = scores

    if not all_data:
        print("No scores.json found under results/")
        return

    # Build CSV rows
    rows = []
    headers = [
        'method', 'group',
        'smatch_s42', 'smatch_s123', 'smatch_s456',
        'smatch_mean', 'smatch_std',
        'parsable_s42', 'parsable_s123', 'parsable_s456', 'parsable_mean',
        'root_acc_mean', 'avg_len_mean',
    ]

    methods = [m for m in METHODS_ORDER if m in all_data]
    methods += [m for m in sorted(all_data) if m not in methods]

    for method in methods:
        seed_data = all_data[method]
        row = {'method': method, 'group': next(iter(seed_data.values())).get('group', '')}

        f1s = []
        parsables = []
        root_accs = []
        avg_lens = []

        for seed in SEEDS:
            s = seed_data.get(seed, {})
            f1 = s.get('f1', None)
            row[f'smatch_s{seed}']   = f"{f1:.4f}" if f1 is not None else ''
            row[f'parsable_s{seed}'] = f"{s.get('parsable_rate', ''):.4f}" if s.get('parsable_rate') is not None else ''
            if f1 is not None:
                f1s.append(f1)
            if s.get('parsable_rate') is not None:
                parsables.append(s['parsable_rate'])
            if s.get('root_acc') is not None:
                root_accs.append(s['root_acc'])
            if s.get('avg_output_len') is not None:
                avg_lens.append(s['avg_output_len'])

        row['smatch_mean']    = f"{mean(f1s):.4f}"    if f1s        else ''
        row['smatch_std']     = f"{stdev(f1s):.4f}"   if len(f1s) > 1 else '0.0000'
        row['parsable_mean']  = f"{mean(parsables):.4f}" if parsables else ''
        row['root_acc_mean']  = f"{mean(root_accs):.4f}" if root_accs  else ''
        row['avg_len_mean']   = f"{mean(avg_lens):.1f}"  if avg_lens   else ''

        rows.append(row)

    # Write CSV
    out_path = summary_dir / 'all_scores.csv'
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved → {out_path}\n")

    # Print table
    col_w = 12
    header_line = f"{'Method':<10} {'Group':<15} {'s42':>8} {'s123':>8} {'s456':>8} {'mean':>8} {'std':>8} {'parsable':>10} {'root_acc':>10}"
    print(header_line)
    print('-' * len(header_line))
    for row in rows:
        print(
            f"{row['method']:<10} {row['group']:<15}"
            f" {row.get('smatch_s42',''):>8}"
            f" {row.get('smatch_s123',''):>8}"
            f" {row.get('smatch_s456',''):>8}"
            f" {row.get('smatch_mean',''):>8}"
            f" {row.get('smatch_std',''):>8}"
            f" {row.get('parsable_mean',''):>10}"
            f" {row.get('root_acc_mean',''):>10}"
        )


if __name__ == '__main__':
    main()
