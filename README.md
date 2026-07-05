# ViSemPar_260430

LoRA SFT experiments for Vietnamese AMR parsing with B1, B2, and MTUP2.

## Current Goal

The active protocol answers one question: whether MTUP2 is better than the B1/B2 baselines under an objective Vietnamese v2 setup.

Model selection is based on held-out dev Smatch for the final AMR only. Dev loss is logged for diagnostics, but it is not used to choose MTUP2 checkpoints.

## Active Files

- `src/data_pipeline.py`: builds clean chat-format train/dev data.
- `src/train.py`: trains Qwen2.5 LoRA adapters and saves one checkpoint per epoch.
- `src/inference.py`: runs inference from either `final` or an explicit `checkpoint-*`.
- `src/evaluate.py`: computes Smatch and structural diagnostics.
- `src/audit_b2.py`: verifies B2 target equals MTUP2 step 1 without final-AMR leakage.
- `src/audit_mtup2_setup.py`: verifies MTUP2 prompt/template/split cleanliness.
- `run_scripts/run_vi_epoch_smatch_grid.py`: current end-to-end experiment runner.
- `run_scripts/run_vietnamese_v2.sh`: compatibility wrapper around the current runner.
- `configs/vietnamese_v2/`: source configs for B1, B2, and MTUP2.

## Environment

```bash
bash run_scripts/setup_conda_env.sh
conda activate visempar-mtup2
```

The setup script installs CUDA PyTorch and `requirements.txt`. Training requires a CUDA GPU.

## Data

Regenerate Vietnamese v2 formatted data when raw data or prompt formatting changes:

```bash
python3 src/data_pipeline.py \
  --language vi \
  --no_var_mode clean \
  --raw_dir data/raw \
  --formatted_dir data/formatted/vietnamese_v2 \
  --amr1_weight 1 \
  --amr2_weight 1 \
  --dev_split 0.1 \
  --dev_seed 42 \
  --methods B1 B2 MTUP2
```

## Run The Current Experiment

Before spending GPU time, inspect expected cost:

```bash
python3 src/profile_training_cost.py --config configs/vietnamese_v2/MTUP2.yaml
```

On the GPU server, run a short timing probe before a full run:

```bash
python3 src/train.py \
  --config configs/vietnamese_v2/MTUP2.yaml \
  --seed 42 \
  --max_steps 20
```

Then inspect:

```text
checkpoints/<group>/<run_name>/s42/timing_profile.json
```

FlashAttention is disabled by default because Quadro RTX 6000/Turing does not
support FlashAttention 2. If you later run on an Ampere-or-newer GPU such as
RTX A6000/A100, you can explicitly enable it:

```bash
bash run_scripts/install_flash_attn.sh
python3 run_scripts/run_vi_epoch_smatch_grid.py --flash-attn on
```

On unsupported GPUs, keep the default `model.flash_attn: false`.

Quick protocol: MTUP2 only, seed `42`, three LR values, `batch=2`,
`grad_accum=8`, and dev Smatch every 5 epochs.

```bash
python3 run_scripts/run_vi_epoch_smatch_grid.py
```

Comparison protocol: add B1/B2 and all three seeds after the quick MTUP2 signal looks worth the compute.

```bash
python3 run_scripts/run_vi_epoch_smatch_grid.py \
  --methods B1 B2 MTUP2 \
  --seeds 42 123 456
```

Final exhaustive protocol: tune B1/B2 too and score every epoch checkpoint. This is expensive and should be run only for final evidence.

```bash
python3 run_scripts/run_vi_epoch_smatch_grid.py \
  --methods B1 B2 MTUP2 \
  --seeds 42 123 456 \
  --tune-baselines \
  --score-all-checkpoints
```

## Runtime Optimization Without Biasing The Comparison

Safe optimizations already used by default:

- Dev-loss evaluation during training is disabled because checkpoint selection is done by external dev Smatch.
- FlashAttention is disabled for RTX 6000/Turing compatibility.
- The default runner is MTUP2-only/seed-42/dev-every-5-epochs to avoid accidentally launching the full expensive grid.
- Every final run still trains for 30 epochs and keeps epoch checkpoints for dev-Smatch selection.

The default runtime is now `batch_per_device=2` and `grad_accum=8`, preserving
the same effective batch size of `16` while cutting the number of micro-batches
roughly in half compared with the old `batch=1`, `grad_accum=16` setup. This
changes GPU memory use and can slightly change runtime numerics, so apply it
uniformly to all compared methods and report it with the results.

Estimate the cost before running:

```bash
python3 src/profile_training_cost.py \
  --config configs/vietnamese_v2/MTUP2.yaml \
  --batch-per-device 2 \
  --grad-accum 8
```

Run the grid. The same fair batch setting is applied to every method/LR/seed:

```bash
python3 run_scripts/run_vi_epoch_smatch_grid.py \
  --methods B1 B2 MTUP2 \
  --seeds 42 123 456 \
  --batch-per-device 2 \
  --grad-accum 8
```

The runner appends the batch setting to the run name, for example
`MTUP2_lr1e-04_b2_ga8`, so these checkpoints cannot be confused with the
old `batch=1, grad_accum=16` checkpoints.

If RTX 6000 still runs out of memory, fall back to the old lower-memory runtime:

```bash
python3 run_scripts/run_vi_epoch_smatch_grid.py \
  --batch-per-device 1 \
  --grad-accum 16
```

Only disable gradient checkpointing after a short probe confirms RTX 6000 has
enough VRAM:

```bash
python3 run_scripts/run_vi_epoch_smatch_grid.py \
  --methods MTUP2 \
  --seeds 42 \
  --mtup2-lrs 1e-4 \
  --batch-per-device 2 \
  --grad-accum 8 \
  --gradient-checkpointing off
```

Do not silently change LoRA rank, target modules, `max_length`, epochs, or LR
grid for only one method. Those are model/protocol changes, not just runtime
optimizations.

Default quick MTUP2 LR grid:

```text
2e-4, 1e-4, 5e-5
```

Default baseline LR grid when `--tune-baselines` is used:

```text
2e-4, 1e-4, 5e-5, 1e-5
```

## Selection Rule

For every run and seed:

1. Train for 30 epochs.
2. Keep each epoch checkpoint.
3. Run dev inference every `--dev-eval-interval` checkpoints, always including the last checkpoint.
4. Compute dev Smatch on the extracted final AMR.
5. Select the checkpoint with the best dev Smatch.
6. Evaluate that selected checkpoint on the public test set once.

Outputs are written under:

```text
results/vietnamese_v2_epoch_smatch/
```

The main summary is:

```text
results/vietnamese_v2_epoch_smatch/epoch_smatch_test_summary.csv
```

## Notes

- Public test is never used for checkpoint or LR selection.
- `data/raw/public_test.txt` has 150 sentences while `public_test_ground_truth.txt` has 149 gold AMRs; the evaluator handles this known missing-gold case for test only.
- Generated configs are written under `configs/generated/vi_epoch_smatch/` when the runner executes.
- Historical scripts and probe/dev-loss tuning helpers were removed to keep the repo focused on the active protocol.
- Training dev-loss evaluation is disabled by default to avoid spending GPU on a metric that is not used for selection.
