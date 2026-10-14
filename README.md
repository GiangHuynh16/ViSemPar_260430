# Vietnamese AMR Parsing with MTUP2

This repository contains Vietnamese AMR parsing experiments using Qwen2.5-7B-Instruct, LoRA, and the B1, B2, and MTUP2 output formulations. The results reported here reflect the artifacts currently available in the repository. Some results remain provisional because they must be rerun after the Smatch evaluator was corrected.

## 1. Research Background

This research started from Vietnamese AMR parsing experiments in VLSP 2025 and was extended as part of a thesis project.

- The VLSP 2025 work primarily used encoder–decoder approaches for direct AMR generation, with backbones such as BARTPho, ViT5, and related models.
- The thesis extends this direction by investigating supervised intermediate task decomposition for AMR parsing.
- MTUP2 has so far been evaluated in only one main experimental setting and one training seed. Therefore, the current results are exploratory evidence rather than sufficient evidence that MTUP2 consistently outperforms the baselines.

The current repository uses a decoder-only Qwen2.5-7B-Instruct model with LoRA so that B1, B2, and MTUP2 can be compared with the same backbone. MTUP3 and MTUP4 pipelines and configurations are available, but complete results are not yet included in the current result group.

## 2. Research Question

**Does MTUP2 improve Vietnamese AMR parsing quality compared with the B1 and B2 baselines?**

The compared methods are:

- **B1**: direct generation, Vietnamese sentence → full AMR.
- **B2**: Vietnamese sentence → variable-free AMR skeleton (`AMR_NO_VAR`).
- **MTUP2**: Vietnamese sentence → `AMR_NO_VAR` → `AMR_WITH_VAR`.

For MTUP2, only the final `AMR_WITH_VAR` output is used to calculate Smatch.

## 3. Methodology

### 3.1. Methods

| Method | Training output sequence | Role |
|---|---|---|
| B1 | `sentence → AMR` | Direct-generation baseline |
| B2 | `sentence → AMR_NO_VAR` | Skeleton baseline; reentrancies are not represented directly |
| MTUP2 | `sentence → AMR_NO_VAR → AMR_WITH_VAR` | Two-task decomposition method |

`AMR_NO_VAR` is a graph-like intermediate representation without variable declarations; it is not equivalent to a complete AMR graph. For B2, variables are bound in post-processing before evaluation. For MTUP2, the model must generate the complete AMR in the final step; the intermediate output is not used as a fallback when the final step fails.

MTUP2 is **structured intermediate supervision**, not only Chain-of-Thought prompting: the intermediate sections have gold supervision during training and are generated sequentially by the same model.

### 3.2. Training changes

- The same `Qwen/Qwen2.5-7B-Instruct` backbone and LoRA configuration are used across methods.
- LoRA uses `r=128`, `alpha=256`, `dropout=0.05`, and the target modules `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj`.
- The data is deduplicated with `amr1_weight=1`, `amr2_weight=1`, and `no_var_mode=clean`.
- Checkpoints are selected using **dev Smatch on the final AMR**, not dev loss.
- MTUP2 was tested with learning rates `5e-5`, `1e-4`, and `2e-4`; the current baseline results use `5e-5`.
- During inference, only the final AMR is extracted and evaluated; intermediate outputs are not treated as final predictions.

## 4. Dataset & Evaluation

### 4.1. Data and splits

The data source is the Vietnamese AMR corpus in [`data/raw/`](data/raw/). The pipeline reads `train_amr_1.txt` and `train_amr_2.txt`, deduplicates examples by sentence, and creates chat-formatted training data for each method.

| Split or stage | Number of examples |
|---|---:|
| Raw examples | 1,842 |
| After deduplication | 1,808 |
| Train | 1,627 |
| Dev | 181 |
| Public test | 150 |

The dev split uses `dev_split=0.1` and `dev_seed=42`. The public test set is a fixed manifest containing 150 sentences and 150 order-aligned gold graphs; no test sentences were dropped. Details are available in [`dataset_stats.json`](data/formatted/vietnamese_v2/dataset_stats.json).

The seeds should be interpreted separately:

- `42`: dev-split seed and training seed for the currently saved results.
- `123`, `456`: planned additional training seeds; complete results are not yet available in the current result group.
- `0`: Smatch evaluation seed used in the recheck; it is not a training seed.

### 4.2. Smatch computation

The evaluator in [`src/evaluate.py`](src/evaluate.py) follows these steps:

1. Decode the gold and predicted graphs using PENMAN.
2. For each graph pair, find the optimal variable mapping under Smatch and count matched triples, predicted triples, and gold triples.
3. Aggregate the counts over the full corpus and calculate:

   ```text
   precision = matched / predicted
   recall    = matched / gold
   F1        = 2 * precision * recall / (precision + recall)
   ```

4. If a prediction cannot be parsed, it contributes zero matched and zero predicted triples, while all gold triples remain in the denominator. If a gold graph cannot be parsed, evaluation stops with an error.

The evaluator now clears the mapping cache between graph pairs and seeds both smart and random initialization. The recheck used `smatch=1.0.4`, `penman=1.3.1`, evaluation seed `0`, and five iterations. Concept, relation, attribute, and root scores are supplementary diagnostics; **Smatch F1 is the primary metric**.

## 5. Experiments & Results

### 5.1. Current results

The table below preserves all metrics available in the saved score files. Test scores are saved results on 150 examples with training seed 42. Rows marked with `*` should not be used as final publication evidence until they are rerun with the corrected evaluator.

| Method | LR | Test artifact epoch | Rechecked dev Smatch | Test P | Test R | Test F1 | Valid graphs | Concept F1 | Relation F1 | Attribute F1 | Root accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B1* | `5e-5` | 15 | 71.81 (epoch 30) | 62.22 | 50.53 | 55.77 | 96.00% | 76.27 | 53.22 | 54.31 | 56.67% |
| B2 | `5e-5` | 25 | 72.17 (epoch 25) | 62.52 | 49.90 | 55.50 | 91.33% | 73.93 | 44.43 | 60.83 | 55.33% |
| MTUP2 | `5e-5` | 30 | 69.45 (epoch 30) | 65.69 | 55.42 | **60.12** | 96.00% | 77.40 | 53.63 | 67.78 | 57.33% |
| MTUP2 | `1e-4` | 30 | **71.64 (epoch 30)** | 66.46 | 52.30 | 58.54 | 96.67% | **77.79** | **55.04** | 66.95 | 56.67% |
| MTUP2 | `2e-4` | 30 | 70.33 (epoch 30) | 65.26 | 44.27 | 52.75 | 90.67% | 71.03 | 52.28 | **69.46** | 49.33% |

Machine-readable artifacts:

- [Summary CSV](results/vietnamese_v2_epoch_smatch/epoch_smatch_test_summary.csv)
- [Rechecked MTUP2 dev scores](results/vietnamese_v2_epoch_smatch/rechecks/mtup2_dev_checkpoint_recheck_2026-10-07.md)
- Score JSON files and predictions are available under [`results/vietnamese_v2_epoch_smatch/`](results/vietnamese_v2_epoch_smatch/).

The results should be interpreted cautiously:

- In the currently saved test artifacts, MTUP2 with `5e-5` reaches 60.12 Smatch F1, compared with 55.77 for the saved B1 artifact and 55.50 for B2. This supports further investigation, but it is not evidence of stability across training seeds.
- The recheck changed the best B1 dev checkpoint from epoch 15 to epoch 30 (`71.81`). The repository does not currently contain test predictions for B1 epoch 30, so the B1–MTUP2 test comparison remains provisional.
- MTUP2 with `1e-4` has the highest rechecked dev score among the tested MTUP2 configurations, while the highest saved test artifact is from `5e-5`. The test set must not be used to select the final learning rate.
- MTUP3 and MTUP4 do not yet have complete results in the current artifact group.

### 5.2. Training configuration

The results above use the following shared configuration:

| Parameter | Value |
|---|---|
| Backbone | `Qwen/Qwen2.5-7B-Instruct` |
| Optimizer | `adamw_torch` |
| Scheduler | cosine, warmup ratio `0.05` |
| Epochs | 30; save one checkpoint per epoch |
| Batch / gradient accumulation | 1 / 16 (effective batch size 16) |
| Weight decay / max gradient norm | 0.01 / 1.0 |
| Maximum input length | 2048 tokens |
| Gradient checkpointing | enabled |
| FlashAttention | disabled by default for Quadro RTX 6000/Turing compatibility |
| Dev evaluation | every 5 epochs, always including the final checkpoint |
| Generation limit | B1/B2: 512; MTUP2: 1536 tokens |

## 6. Reproducibility

### 6.1. Environment setup

Training requires a CUDA GPU. The setup script creates the Conda environment and installs CUDA PyTorch plus the packages listed in [`requirements.txt`](requirements.txt):

```bash
bash run_scripts/setup_conda_env.sh
conda activate visempar-mtup2
```

The main dependencies are PyTorch, Transformers, PEFT, Datasets, Accelerate, PyYAML, Penman, and Smatch. The evaluator versions used in the recheck were `smatch==1.0.4` and `penman==1.3.1`.

### 6.2. Data generation and validation

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
  --methods B1 B2 MTUP2 MTUP3 MTUP4

PYTHONPATH=src python3 src/audit_dataset.py
PYTHONPATH=src python3 src/audit_b2.py \
  --b2 data/formatted/vietnamese_v2/B2_train.txt \
  --mtup2 data/formatted/vietnamese_v2/MTUP2_train.txt \
  --b2_dev data/formatted/vietnamese_v2/B2_dev.txt \
  --mtup2_dev data/formatted/vietnamese_v2/MTUP2_dev.txt
PYTHONPATH=src python3 src/audit_mtup2_setup.py \
  --config configs/vietnamese_v2/MTUP2.yaml
```

### 6.3. Training, checkpoint selection, and test evaluation

Use a dry run to inspect the generated commands:

```bash
PYTHONPATH=src python3 run_scripts/run_vi_epoch_smatch_grid.py \
  --methods B1 B2 MTUP2 \
  --seeds 42 \
  --dry-run
```

Run the current protocol on a GPU:

```bash
PYTHONPATH=src python3 run_scripts/run_vi_epoch_smatch_grid.py \
  --methods B1 B2 MTUP2 \
  --seeds 42 \
  --mtup2-lrs 5e-5 1e-4 2e-4 \
  --batch-per-device 1 \
  --grad-accum 16 \
  --gradient-checkpointing on \
  --flash-attn off \
  --dev-eval-interval 5
```

The runner:

1. trains for 30 epochs and saves epoch checkpoints;
2. runs inference on the dev set;
3. selects the checkpoint with the highest dev Smatch F1;
4. only then runs inference and scoring on the public test set.

To evaluate an existing prediction file:

```bash
PYTHONPATH=src python3 src/evaluate.py \
  --pred results/vietnamese_v2_epoch_smatch/MTUP2_lr5e-05_b1_ga16/selected_test/s42_best_test_pred.txt \
  --gold data/formatted/vietnamese_v2/test_ground_truth.txt \
  --method MTUP2 \
  --seed 42 \
  --group vietnamese_v2_epoch_smatch
```

Main scripts:

- [`src/data_pipeline.py`](src/data_pipeline.py): generate formatted datasets.
- [`src/train.py`](src/train.py): LoRA fine-tuning and checkpoint saving.
- [`src/inference.py`](src/inference.py): inference and final-AMR extraction.
- [`src/evaluate.py`](src/evaluate.py): Smatch and diagnostic metrics.
- [`run_scripts/run_vi_epoch_smatch_grid.py`](run_scripts/run_vi_epoch_smatch_grid.py): end-to-end experiment runner.

### 6.4. Checkpoints and artifacts

The repository currently contains **no adapter checkpoints** in [`checkpoints/`](checkpoints/), so inference cannot be reproduced from a clean clone alone. Saved predictions, score JSON files, metadata, and summary CSV files are available under [`results/`](results/). The `/mnt/...` paths in metadata refer to the training server and are not portable paths.

To share a checkpoint, provide the adapter corresponding to its `run_name`, seed, and epoch. Sharing only `final` is insufficient if the checkpoint-selection procedure also needs to be audited.

## 7. Limitations & Future Work

### Limitations

- There is no independent private benchmark yet. The 150-example public test set may not represent out-of-distribution data sufficiently.
- The current results primarily use one training seed (`42`), so variation across independent training runs has not been measured.
- B1 must be rerun at epoch 30 after the evaluator recheck; the current test comparisons are therefore not final publication results.
- Complete MTUP3 and MTUP4 results are not available, so the optimal number of intermediate tasks cannot yet be determined.
- MTUP2 produces longer outputs and may propagate errors from the skeleton to the final AMR; training and inference costs have not yet been fully reported by time and token count.
- B2 removes reentrancy information from its intermediate target and is therefore a structural baseline, not a fully equivalent AMR representation.

### Future Work

1. Rerun B1 at epoch 30 and rescore the complete test prediction with the corrected evaluator.
2. Repeat B1, B2, and MTUP2 with training seeds `123` and `456`; report mean and standard deviation rather than a single run.
3. Run MTUP3 and MTUP4 with the same split, checkpoint-selection rule, and comparison budget.
4. Maintain an independent private benchmark for final evaluation and never use it to select the learning rate or checkpoint.
5. Report computational cost, output length, valid-graph rate, and subgroup analyses by sentence length, graph size, and reentrancy.
6. Use bootstrap or paired randomization tests on locked predictions to assess whether MTUP2–baseline differences are stable.

Further details about the rechecked protocol and the next experiment cycle are available in [`references/RECHECKED_EXPERIMENT_PROTOCOL.md`](references/RECHECKED_EXPERIMENT_PROTOCOL.md).
