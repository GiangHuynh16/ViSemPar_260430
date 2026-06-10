# Current Repo Answers

Date: 2026-06-10

This answers the note questions in `references/2026-06-01-current-repo-questions.md`
after the cleanup pass.

1. **Current result reproducibility**
   Existing `results/**/scores.json` should be treated as preliminary unless rerun from the
   current code. The source now has cleaner v2 configs and regenerated Vietnamese v2 data.

2. **Full/pred alignment**
   `src/inference.py` writes `s{seed}_full.txt` and `s{seed}_pred.txt` from the same loop.
   Old result files may still come from older scripts and should be regenerated for final reporting.

3. **Exact commands**
   Vietnamese v2 data was regenerated with:
   `python3 src/data_pipeline.py --raw_dir data/raw --formatted_dir data/formatted/vietnamese_v2 --language vi --no_var_mode clean --amr1_weight 1 --amr2_weight 1 --syn_weight 0 --dev_split 0.1 --dev_seed 42 --methods B1 B2 MTUP2 MTUP3A MTUP3B MTUP4`

4. **Dev split and model selection**
   A deterministic train/dev split is now generated before weighting/formatting. Training saves
   `trainer_log_history.json` and `loss_curve.png`. Final LR/epoch selection should use lowest
   dev loss, not public-test score.

5. **B2 correctness**
   B2 now uses the same `strip_variables(..., mode="clean")` path as MTUP2 step 1.
   `src/audit_b2.py` verified train and dev:
   B2 equals MTUP2 step 1, declaration violations = 0, bare full-AMR variable refs = 0.
   B2 inference still binds variables deterministically with `src/var_binder.py` before Smatch.

6. **Three-seed protocol**
   Final results still need reruns for seeds `42`, `123`, and `456` after LR/epoch selection.

7. **Manual edits / old scripts**
   No manual prediction editing was done in this cleanup. Old predictions should not be used as
   final evidence without regeneration.

8. **Environment**
   Not solved yet. A pinned dependency/environment file is still needed.

9. **Vietnamese prompt leakage**
   The Vietnamese system prompt is generic and explicitly forbids examples, gold labels, or rules
   derived from the test set. Synthetic examples remain removed because prior examples were found
   in the test set.

10. **English prompt**
    The English prompt is generic: format, standard AMR concepts/variables/relations, balanced
    parentheses, unique variables, reentrancy reuse, and no extra output. No dataset-specific
    English rules were added.

Changed code:

- `src/data_pipeline.py`: clean no-var path, deterministic dev split, dev sentence/gold export.
- `src/audit_b2.py`: new B2/MTUP2 step-1 and no-var leakage audit.
- `src/train.py`: optional sklearn auto split, epoch-level logging, dev loss metadata.
- `src/tune_lr.py`: sklearn random LR sampler, 1e-6 to 5e-5 default range, 50-epoch search option,
  and summary selection by dev loss.
- `src/inference.py`: `--split val` no longer falls back to public test.
- `configs/vietnamese_v2/*.yaml`: dev sentence path added.
- `configs/base_v2.yaml`: v2 logging and dev seed defaults.
