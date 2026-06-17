# MTUP2 Vietnamese Clean Setup Audit

Date: 2026-06-17

## Setup Audit

Correct:

- `B2_train` and `MTUP2_train` now both contain 1623 examples from the same regenerated split.
- `B2_dev` and `MTUP2_dev` both contain 181 examples.
- B2 targets exactly match `MTUP2` `[STEP 1: AMR_NO_VAR]` for train and dev.
- B2 no-var targets contain no AMR declarations and no bare variables from the paired full AMR.
- MTUP2 train/dev prompts, step markers, target construction, post-processing stability, and stale-header checks all pass.
- Exact sentence-overlap checks pass for train/dev, train/public-test, dev/public-test, and raw-train/public-test.

Wrong before this pass:

- `B2_train` had 1626 examples while `MTUP2_train` had 1623, with three B2-only sentences.
- Inference ignored `run_name`, so LR-search configs could point training and inference at different checkpoint locations.

Risky:

- Old helper scripts still exist for historical workflows; use `run_scripts/run_mtup2_vi_clean.sh` for this protocol.
- Existing `results/**` files remain historical and should not be used as final evidence.

Must be fixed before public-test reporting:

- Run the LR-search training on a CUDA machine.
- Select the LR/checkpoint by dev loss only.
- Run one public-test inference/evaluation after selection is fixed.

## Teacher Note Answers

1. Current old `results/**` files are not final evidence; regenerate with the clean protocol.
2. New inference writes `full`, `pred`, and metadata from the same loop.
3. The clean protocol is encoded in `run_scripts/run_mtup2_vi_clean.sh`.
4. A deterministic dev split is used; LR and epoch/checkpoint selection are by dev loss, never public test.
5. B2 now uses the same clean no-var target as MTUP2 step 1 and is audited against it.
6. This pass intentionally runs one representative MTUP2 Vietnamese seed, not all methods/seeds/datasets.
7. No prediction files were manually edited in this pass.
8. Dependencies were installed locally, but local training is blocked because CUDA is unavailable.
9. The Vietnamese prompt is generic and explicitly forbids test-derived examples, labels, and rules.
10. The English prompt is generic but now method-aware for MTUP2 output sections; it contains no dataset-specific answers.

## Changed Files And Why

- `src/audit_b2.py`: strengthened train/dev B2-vs-MTUP2 checks, prompt separation, label leakage, final-output reuse, declarations, and bare variables.
- `src/audit_mtup2_setup.py`: added stale-header and split-overlap checks.
- `src/prompts.py`: added method-aware English MTUP2 formatting instructions without answer leakage.
- `src/inference.py`: uses `run_name` or explicit `adapter_path` and writes inference metadata.
- `src/train.py`: records selected adapter metadata and fails early when CUDA is unavailable.
- `src/tune_lr.py`: uses shared config YAML handling.
- `src/config_utils.py`: adds a small YAML fallback for simple repo configs when PyYAML is unavailable.
- `run_scripts/run_mtup2_vi_clean.sh`: one clean B2/MTUP2 audit, LR tuning, and optional CUDA training/evaluation protocol.
- `data/formatted/vietnamese_v2/B2_train.txt` and `dataset_stats.json`: regenerated with the clean B2/MTUP2-only split.

## Local Verification

- `python3 src/audit_b2.py --b2 data/formatted/vietnamese_v2/B2_train.txt --mtup2 data/formatted/vietnamese_v2/MTUP2_train.txt --b2_dev data/formatted/vietnamese_v2/B2_dev.txt --mtup2_dev data/formatted/vietnamese_v2/MTUP2_dev.txt`
- `python3 src/audit_mtup2_setup.py --config configs/vietnamese_v2/MTUP2.yaml`
- `bash run_scripts/run_mtup2_vi_clean.sh`

Training was not run locally because `torch.cuda.is_available()` is `False`.
