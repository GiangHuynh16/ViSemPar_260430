# MTUP2_CHECK

MTUP2_CHECK is the MTUP2 trajectory with one extra compact graph-check step:

```text
[STEP 1: AMR_NO_VAR]
...

[STEP 2: CHECK]
ROOT: ...
CONCEPTS: ...
RELATIONS: ...
REENTRANCIES: ...
VARIABLE_CHECK: ...

[STEP 3: AMR_WITH_VAR]
...
```

The final AMR used for dev/test Smatch is always `[STEP 3: AMR_WITH_VAR]`.

## Validate Format

```bash
python3 src/audit_mtup2_check_setup.py \
  --config configs/vietnamese_v2/mtup2_check/MTUP2_CHECK.yaml
```

## Run

Quick LR grid, seed 42:

```bash
python3 run_scripts/run_vi_epoch_smatch_grid.py \
  --methods MTUP2_CHECK \
  --seeds 42
```

Final style, all checkpoints scored:

```bash
python3 run_scripts/run_vi_epoch_smatch_grid.py \
  --methods MTUP2_CHECK \
  --seeds 42 123 456 \
  --score-all-checkpoints
```
