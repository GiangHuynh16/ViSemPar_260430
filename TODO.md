# TODO

## Priority 1: Clean B2 Baseline

- [x] Switch default no-var generation to `clean` in `src/data_pipeline.py`.
- [x] Ensure `B2` and `MTUP2` step 1 share the same no-var generation path.
- [x] Ensure the Vietnamese prompt tells the model not to leave bare variable references in no-var output.
- [ ] Regenerate Vietnamese v2 formatted data and inspect `data/formatted/vietnamese_v2/B2_train.txt`.
- [ ] Verify B2 no-var targets contain no AMR declarations like `(x / concept)` and no bare reentrancy variables.
- [ ] Retrain B2 after regenerated data; old B2 checkpoints are not enough for final comparison.

## Priority 2: Prompt Leakage Audit

- [x] Replace the detailed Vietnamese prompt with a generic prompt in `src/data_pipeline.py`.
- [x] Replace the detailed Vietnamese inference prompt with the same generic prompt in `src/inference.py`.
- [x] Keep English and Portuguese prompts generic.
- [ ] Audit any future prompt edits against public-test gold data. Do not add public-test-derived rules.
- [ ] Document every non-generic prompt rule as one of: annotation-guideline-derived, train-data-derived, or test-derived. Test-derived rules must be removed.

## Priority 3: Train/Dev Split and Model Selection

- [x] Update `src/data_pipeline.py` so `--dev_split` randomly splits raw examples before weighting/formatting.
- [x] Add `--dev_seed` for deterministic splits.
- [x] Save `dev_split` and `dev_seed` in `dataset_stats.json`.
- [x] Update `src/train.py` to save `trainer_log_history.json`.
- [x] Update `src/train.py` to save `loss_curve.png` when dev data exists.
- [ ] Decide final model-selection metric: dev loss is implemented; validation Smatch still needs an inference-on-dev callback or separate script.
- [ ] Never tune by public-test score. Use public test only after LR/epoch/checkpoint are fixed from dev.

## Priority 4: Learning-Rate Tuning

- [x] Add `src/tune_lr.py` to generate sklearn random-search style LR configs.
- [ ] Run LR search per method on a dev split. Suggested starting range: `1e-6` to `5e-5`.
- [ ] Keep effective batch size around 16 unless GPU memory allows larger.
- [ ] Compare train/dev curves after each run.
- [ ] Pick best LR from dev behavior, then rerun final three seeds.
- [ ] Do not keep 25 epochs by default. Use v2 default `10` first, then adjust from curves.

## Priority 5: Reproducibility Package

- [ ] Add a pinned dependency file, e.g. `requirements.txt` or `environment.yml`.
- [ ] Save commit hash, command, config, Python version, package versions, CUDA version, GPU model, seed, checkpoint path, prediction path, and score path for every final run.
- [ ] Add generated run logs under a dedicated logs directory or external artifact store.
- [ ] Rerun all final methods with seeds `42`, `123`, and `456`.
- [ ] Regenerate `results/summary/all_scores.csv`.

## Priority 6: Answer Teacher Notes

- [x] Keep `references/2026-06-01-current-repo-questions-answered.md`.
- [ ] Update that file after final reruns with exact command history and environment.
- [ ] Clearly separate historical scores from final clean scores in the report.
