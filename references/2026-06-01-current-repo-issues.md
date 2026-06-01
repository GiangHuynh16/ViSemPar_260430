# Current Repo Issues

Date: 2026-06-01

Purpose: priority-ordered risks to resolve or document before building the next MTUP experiment.

## P0 - Vietnamese MTUP2 Output/Extractor Drift

**Issue:** Saved Vietnamese `MTUP2` full outputs use `Task 1:` / `Task 2:`, while the current extractor expects `[STEP 1: AMR_NO_VAR]` / `[STEP 2: AMR_WITH_VAR]`.
**Note:** Current code would not extract these saved outputs correctly, so the reported MTUP2 score may not be reproducible from the checked-in inference script.
**Suggested action:** Re-run inference with one frozen extractor, or add explicit extractor support for all historical headers and record which extractor produced each result.

## P0 - No Fixed Dev Split Or Validation-Smatch Selection

**Issue:** Main results do not clearly use a fixed validation split or checkpoint selection by validation Smatch.
**Note:** The original design required this for fair model selection; otherwise, test-set-driven decisions may contaminate the comparison.
**Suggested action:** Define a fixed dev split, select checkpoints by validation Smatch, and reserve the public test set for final reporting only.

## P0 - Prompt Format Differs From Original MTUP Design

**Issue:** Current prompts use detailed language-specific system prompts and raw sentence user prompts instead of the original fixed `### TASK` / `### INPUT` / `### OUTPUT` format.
**Note:** Current results test a modified MTUP implementation, not the exact original design.
**Suggested action:** Freeze and document the final prompt format before rerunning or extending experiments.

## P0 - Detailed System Prompt May Leak Test-Specific Rules

**Issue:** The Vietnamese system prompt contains many detailed annotation rules that may have been derived from observing public test examples.
**Note:** If prompt rules were tuned using test data, the reported scores may overestimate true generalization.
**Suggested action:** Audit each rule's source and separate general annotation guidelines from test-derived fixes.

## P1 - Detailed System Prompt May Reduce Generalizability

**Issue:** Some prompt rules are tightly tied to the current Vietnamese dataset's annotation style.
**Note:** Dataset-specific rules can make B1/B2 very strong and reduce the apparent benefit of MTUP on broader AMR parsing.
**Suggested action:** Report whether rules are dataset-specific, and consider a general-prompt ablation.

## P1 - Legacy No-Var Mode May Not Be Truly Variable-Free

**Issue:** `legacy` no-var conversion removes declarations but can preserve bare variable references.
**Note:** B2 is intended to test variable-free AMR plus deterministic binding; leftover variables weaken that interpretation.
**Suggested action:** Use `clean` no-var mode for new experiments and document the old mode as historical.

## P1 - Equal Token Loss Under-Trains Final AMR

**Issue:** All assistant tokens are trained equally, so MTUP variants spend substantial loss on subtasks instead of the final AMR.
**Note:** The metric evaluates final AMR, not subtask quality; equal weighting can make longer MTUP trajectories less competitive.
**Suggested action:** Use fixed section weights: draft AMR `1.0`, checks/subtasks `0.5`, final AMR `2.0`.

## P1 - Gold-Intermediate Teacher Forcing Creates Exposure Bias

**Issue:** During training, final AMR is conditioned on gold intermediate outputs; during inference, it is conditioned on model-generated intermediates.
**Note:** Final AMR generation can degrade when earlier generated steps contain errors.
**Suggested action:** Include corrupted-gold and B1/B2 draft examples so the model learns to recover from realistic draft errors.

## P1 - Different Max Generation Budgets Need A Policy

**Issue:** Baselines and MTUP variants use different `max_new_tokens` values.
**Note:** Different budgets can bias results unless they are justified as a fixed length-scaled policy.
**Suggested action:** Define one decoding-budget policy and apply it consistently across all methods.

## P2 - Missing Error-Type Metrics

**Issue:** Current evaluation reports Smatch, parsability, root accuracy, and length, but not concept F1, relation F1, reentrancy accuracy, or targeted error counts.
**Note:** The MTUP story depends on reducing semantic and graph-specific errors, not only improving aggregate Smatch.
**Suggested action:** Add targeted metrics for root, concepts, relations, `:wiki`, `:op1/:op2`, `:manner`, variables, and reentrancies.

## P2 - Reproducibility Artifacts Are Missing

**Issue:** Formatted data, checkpoints, dependency files, and full run metadata are not tracked in a complete reproducibility package.
**Note:** The team cannot easily verify or rerun reported results from a clean checkout.
**Suggested action:** Add run manifests, dependency specs, generated-data recipes, and immutable result metadata for future experiments.
