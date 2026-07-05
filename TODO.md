# TODO

## Current Protocol

- [x] Use clean B2 no-var targets shared with MTUP2 step 1.
- [x] Use deterministic Vietnamese v2 train/dev split.
- [x] Train for 30 epochs and keep epoch checkpoints.
- [x] Select checkpoints by dev Smatch on final AMR, not dev loss.
- [x] Add preflight audits for B2/MTUP2 and sentence/gold counts.
- [x] Add MTUP2 LR grid around LoRA-relevant values.
- [x] Add optional fair LR grid for B1/B2 via `--tune-baselines`.
- [x] Make the default runner a low-cost quick MTUP2 signal instead of a full comparison.
- [x] Disable dev-loss evaluation during training by default.
- [x] Score dev Smatch every 5 epochs by default; keep `--score-all-checkpoints` for final runs.
- [x] Add fair runtime overrides for `batch_per_device`, `grad_accum`, and gradient checkpointing.
- [x] Add run-name suffixes for runtime overrides to avoid reusing incompatible checkpoints.
- [x] Add profiler overrides to estimate micro-batch savings before GPU runs.
- [x] Keep `batch_per_device=1, grad_accum=16` as the RTX 6000/Turing default after `b2_ga8` probed slower.
- [x] Disable FlashAttention by default for RTX 6000/Turing compatibility.

## Next Runs

- [ ] Run a 20-step timing probe with the default config.
- [ ] Run `python3 run_scripts/run_vi_epoch_smatch_grid.py` for the quick MTUP2 signal.
- [ ] Run `python3 run_scripts/run_vi_epoch_smatch_grid.py --methods B1 B2 MTUP2 --seeds 42 123 456` for a broader comparison.
- [ ] Run the final exhaustive protocol only if the quick/broader runs justify the compute.
- [ ] Inspect `results/vietnamese_v2_epoch_smatch/epoch_smatch_test_summary.csv`.
- [ ] Record command, git commit, seed, LR, selected checkpoint, dev Smatch, and test Smatch for every final run.

## Reporting

- [ ] Conclude MTUP2 is promising only if its dev-selected test scores beat tuned B1/B2 consistently across seeds.
- [ ] Report selected epochs and dev/test Smatch together so the comparison is auditable.
