# ViSemPar_260430

LoRA SFT experiments for Vietnamese AMR parsing and MTUP baselines.

## What This Repo Contains

- `src/data_pipeline.py`: builds chat-format training files for `B1`, `B2`, `MTUP2`, `MTUP3A`, `MTUP3B`, and `MTUP4`.
- `src/train.py`: trains a Qwen2.5 LoRA adapter from a config.
- `src/inference.py`: runs greedy inference and writes `*_full.txt` plus extracted `*_pred.txt`.
- `src/evaluate.py`: computes Smatch, parsable rate, output length, and root accuracy.
- `src/aggregate_scores.py`: aggregates all `results/**/scores.json` into `results/summary/all_scores.csv`.
- `src/tune_lr.py`: creates sklearn random-search style learning-rate tuning configs.
- `configs/`: method configs. The cleaner Vietnamese setup is under `configs/vietnamese_v2/`.
- `run_scripts/`: end-to-end scripts for Vietnamese, English Little Prince, and Portuguese Little Prince.
- `references/`: repo audit notes, current questions, and experimental-plan drafts.

## Methods

- `B1`: sentence to full AMR.
- `B2`: sentence to no-var AMR. At inference time, `src/var_binder.py` deterministically adds variables so Smatch can run.
- `MTUP2`: step 1 no-var AMR, step 2 full AMR.
- `MTUP3A`: concepts, no-var AMR, full AMR.
- `MTUP3B`: concepts, relations, full AMR.
- `MTUP4`: concepts, relations, no-var AMR, full AMR.

Important current convention: `B2` no-var output and `MTUP2` step 1 are produced by the same `strip_variables(..., mode="clean")` path when the same formatted data command is used. `clean` mode removes declarations and expands bare reentrancy variables, so B2 is actually variable-free.

## Environment

Create one named Conda environment on the CUDA server, then activate it whenever
running audits, training, inference, or evaluation.

Recommended one-time setup:

```bash
conda create -y -n visempar-mtup2 python=3.10
conda activate visempar-mtup2
python --version
pip install --upgrade pip
```

Install CUDA PyTorch and repo dependencies. For CUDA 12.1:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Or use the helper script for the same setup:

```bash
bash run_scripts/setup_conda_env.sh
conda activate visempar-mtup2
```

If the server uses a different CUDA version, override the PyTorch wheel index,
for example:

```bash
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 bash run_scripts/setup_conda_env.sh
```

Verify CUDA before training:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda version:", torch.version.cuda)
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
```

Current code expects:

- Python 3.10+ recommended
- CUDA GPU for training
- `torch`
- `transformers`
- `datasets`
- `peft`
- `pyyaml`
- `tqdm`
- `smatch`
- `penman` optional
- `matplotlib` optional, for `loss_curve.png`
- `scikit-learn` and `scipy` for `src/tune_lr.py`

Before final paper runs, save `python --version`, `pip freeze`, CUDA version,
GPU model, command, git commit, config path, and checkpoint path for every run.

## Recommended Vietnamese V2 Workflow

Build clean formatted data with a deterministic dev split:

```bash
python src/data_pipeline.py \
  --language vi \
  --no_var_mode clean \
  --raw_dir data/raw \
  --formatted_dir data/formatted/vietnamese_v2 \
  --amr1_weight 1 \
  --amr2_weight 1 \
  --dev_split 0.1 \
  --dev_seed 42
```

Train, infer, and evaluate all Vietnamese v2 methods:

```bash
bash run_scripts/run_vietnamese_v2.sh
```

Run one method manually:

```bash
python src/train.py --config configs/vietnamese_v2/MTUP2.yaml --seed 42
python src/inference.py --config configs/vietnamese_v2/MTUP2.yaml --seed 42 --split test
python src/evaluate.py \
  --pred results/vietnamese_v2_mtup_2tasks/MTUP2/s42_pred.txt \
  --gold data/raw/public_test_ground_truth.txt \
  --method MTUP2 \
  --seed 42 \
  --group vietnamese_v2_mtup_2tasks
python src/aggregate_scores.py
```

Training with a dev split writes:

- `checkpoints/<group>/<method>/s<seed>/trainer_log_history.json`
- `checkpoints/<group>/<method>/s<seed>/loss_curve.png`
- `checkpoints/<group>/<method>/s<seed>/train_meta.json`

Use the dev loss curve to detect overfitting and choose a defensible epoch/checkpoint before public-test inference.
For validation inference, v2 configs use `data/formatted/vietnamese_v2/dev.txt`; public test is never the fallback for `--split val`.

If training succeeds but validation fails with CUDA OOM exactly at an epoch
boundary, verify that the updated code prints:

```text
Batch sizes: train=1, eval=1, grad_accum=16
```

Older code left Hugging Face's default evaluation batch size at `8`, even
though training used batch size `1`. Pull the latest code before restarting.
To resume when a `checkpoint-*` directory exists:

```bash
python src/train.py --config <config.yaml> --seed 42 --resume_from_checkpoint
```

## MTUP2-Only Cascade Tuning

Current decision protocol: do not rerun every old method/dataset. Use one representative method, `MTUP2`, on Vietnamese v2 to get an early signal about whether the idea works.

On the CUDA server, activate the environment first:

```bash
conda activate visempar-mtup2
```

Run the agreed short MTUP2 probe first: one middle learning rate and five epochs.

```bash
bash run_scripts/run_vi_probe5.sh MTUP2
```

This uses:

```text
config: configs/vietnamese_v2/MTUP2_probe5.yaml
learning rate: 1.752410111812816e-5
maximum epochs: 5
train batch: 1
validation batch: 1
effective batch: 16
```

The progress bar should show approximately `510` optimizer steps, not `3060`.
After this run, inspect its train/dev loss curve before deciding whether to
run additional learning rates.

The first MTUP2 probe result was:

```text
MTUP2 mid-LR probe5, seed 42:
best dev loss = 0.1367 at epoch 3
public-test Smatch F1 = 0.5236
parsable rate = 1.0
```

Before spending more time on MTUP2 learning-rate variants, run one clean direct
baseline probe with the same model, LoRA size, LR, batch size, split, and epoch
budget:

```bash
bash run_scripts/run_vi_probe5.sh B1
```

Decision rule:

- If B1 is clearly below MTUP2 `0.5236`, MTUP2 has a positive early signal.
- If B1 is similar or higher, run `bash run_scripts/run_vi_probe5.sh MTUP2_LOW`
  once to test whether a lower LR helps MTUP2.
- Run `bash run_scripts/run_vi_probe5.sh B2` only if the B1/MTUP2 comparison is
  inconclusive or the report needs the cleaned B2 baseline number.

Preferred one-command protocol:

```bash
RUN_TRAIN=1 bash run_scripts/run_mtup2_vi_clean.sh
```

This regenerates only `B2` and `MTUP2` Vietnamese v2 data, runs the setup
audits, trains only the three MTUP2 LR candidates, selects by dev loss, and then
runs one public-test inference/evaluation for seed `42`.

Use hierarchical cascade tuning, not full grid search:

1. Fix defaults: MTUP2 uses LoRA `r=64`, `alpha=128`, `dropout=0.05`, effective batch size 16, scheduler, warmup, weight decay.
2. Tune learning rate first on dev.
3. Fix the selected learning rate and best epoch.
4. Only if the dev curve is clearly unstable, tune the next hyperparameter in a narrow range.

Run the MTUP2 pre-flight audit before GPU training:

```bash
python src/audit_mtup2_setup.py --config configs/vietnamese_v2/MTUP2.yaml
```

Generate MTUP2-only learning-rate configs:

```bash
python src/tune_lr.py \
  --config configs/vietnamese_v2/MTUP2.yaml \
  --seed 42 \
  --n_iter 3 \
  --lr_min 1e-6 \
  --lr_max 5e-5 \
  --epochs 30 \
  --out_dir configs/tuning/mtup2_vi_lr
```

Run the printed train commands. Then summarize completed runs:

```bash
python src/tune_lr.py \
  --config configs/vietnamese_v2/MTUP2.yaml \
  --seed 42 \
  --summarize configs/tuning/mtup2_vi_lr/MTUP2_lrsearch_manifest.json
```

Pick `selected.lr` and `selected.best_epoch` from the summary. Public test is used only after this selection is fixed.

Final MTUP2 rerun uses three seeds:

```bash
python src/train.py --config configs/vietnamese_v2/MTUP2.yaml --seed 42
python src/train.py --config configs/vietnamese_v2/MTUP2.yaml --seed 123
python src/train.py --config configs/vietnamese_v2/MTUP2.yaml --seed 456
```

## Prompt Leakage Policy

The Vietnamese prompt has been simplified to avoid dataset-specific lexical rules. It now explicitly says not to use examples, gold labels, or rules derived from the test set.

Before final reporting:

- Do not add synthetic examples from public test.
- Do not add prompt rules discovered by inspecting public-test gold AMRs.
- If a rule comes from annotation guidelines or training data only, document that source.
- Keep Vietnamese, English, and Portuguese prompts general unless the paper explicitly claims a dataset-specific prompt ablation.

## Caveats About Existing Results

Existing `results/**` files are useful as preliminary evidence, but not all are cleanly reproducible from current code. In particular, old Vietnamese `MTUP2` outputs used older `Task 1:` / `Task 2:` headers while current code expects `[STEP ...]` headers.

Treat current saved scores as historical until the v2 workflow is rerun with recorded commands, fixed dev split, clean no-var mode, and three seeds.
