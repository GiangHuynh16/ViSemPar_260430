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

The exact historical environment is not recorded. Current code expects:

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

Before final paper runs, add a pinned environment file and save `python --version`, `pip freeze`, CUDA version, GPU model, command, git commit, config path, and checkpoint path for every run.

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

## Learning-Rate Tuning

Do not keep `25` epochs as the default decision for final runs. For v2, start with batch effective size `16` (`batch_per_device: 1`, `grad_accum: 16`) and tune learning rate.

Generate random-search learning-rate configs:

```bash
python src/tune_lr.py \
  --config configs/vietnamese_v2/MTUP2.yaml \
  --seed 42 \
  --n_iter 5 \
  --lr_min 1e-6 \
  --lr_max 5e-5 \
  --epochs 10
```

Run the printed train commands. Pick the learning rate with the lowest dev loss from `trainer_log_history.json` and the healthiest train/dev curve. After selecting LR, run all seeds for the method.

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
