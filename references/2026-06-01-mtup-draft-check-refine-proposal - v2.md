# MTUP Draft-Check-Refine Proposal

Date: 2026-06-01

Purpose: define the next MTUP direction as a trajectory-conditioned unified prompting framework for AMR parsing.

## Core Idea

1. MTUP should train one model to follow different explicit output trajectories. The user prompt tells the model which trajectory to generate, and the output headers make extraction deterministic.
2. The strongest MTUP trajectory is likely: `draft AMR -> graph check / subtasks -> final AMR`. The model first produces an AMR draft, then exposes semantic/graph checks and optional subtasks, then produces a corrected final AMR. Explicit subtasks help most when they supervise semantic error detection and graph repair, not just simply prepended before the final graph. Final AMR improves because the model is trained to repair imperfect drafts using supervised semantic checks.

The research will focus on this trajectory:

```text
[AMR_DRAFT]
...

[CHECK]
...

[SUBTASKS]
...

[AMR_FINAL]
...
```

## Trajectories

| Trajectory | User Prompt Hint | Expected Output | Notes |
|---|---|---|---|
| `direct-amr` | `Generate one final AMR graph directly from the sentence.` | `[AMR]` | Like the current B1 direct AMR baseline. |
| `novar-then-amr` | `First generate a variable-free AMR, then generate the final AMR with variables.` | `[AMR_NO_VAR]`, `[AMR_FINAL]` | Like old MTUP2: generate no-var AMR first, then final AMR. |
| `draft-check-final` | `First generate a draft AMR, then check semantic/graph errors, then generate a corrected final AMR.` | `[AMR_DRAFT]`, `[CHECK]`, `[AMR_FINAL]` | Based on draft-feedback-refine/self-correction, e.g. [Self-Refine](https://arxiv.org/abs/2303.17651) and critique/revision-style self-improvement in [Constitutional AI](https://arxiv.org/abs/2212.08073). |
| `draft-subtasks-final` | `First generate a draft AMR, then generate AMR subtasks, then generate a corrected final AMR.` | `[AMR_DRAFT]`, `[SUBTASKS]`, `[AMR_FINAL]` | Based on AMR decomposition into explicit graph subtasks: concepts, relations, and no-var structure; related to structure-aware AMR modeling such as [Structure-aware AMR parsing](https://arxiv.org/abs/2110.15534) and [Graph Pre-training for AMR](https://arxiv.org/abs/2203.07836). |
| `draft-check-subtasks-final` | `First generate a draft AMR, then check errors, then generate AMR subtasks, then generate a corrected final AMR.` | `[AMR_DRAFT]`, `[CHECK]`, `[SUBTASKS]`, `[AMR_FINAL]` | Main proposed MTUP trajectory; combines self-correction with AMR graph/subtask supervision. |

Prompt hint alone is not enough. The user prompt should include both `### TRAJECTORY` and an explicit `### OUTPUT TRAJECTORY` block listing the exact required section headers. The explicit section headers are what make generation steerable and extraction deterministic.

Example user prompt shape:

```text
### TASK
Parse the sentence into AMR using the requested trajectory.

### INPUT
Sentence: {SENTENCE}

### TRAJECTORY
draft-check-subtasks-final

### OUTPUT TRAJECTORY
[AMR_DRAFT]
[CHECK]
[SUBTASKS]
[AMR_FINAL]
```

## Training Setup

- Draft AMR mixing:
  - gold AMR draft
  - corrupted-gold AMR draft
  - B1/B2 prediction draft
- Checks and subtasks mixing.
- Final AMR is always the gold AMR.
- Loss weights to focus learning:
  - `AMR_DRAFT`: `1.0`
  - `CHECK` / `SUBTASKS`: `0.5`
  - `AMR_FINAL`: `2.0`

The draft should remain competitive with direct AMR generation, while the final AMR learns to repair realistic draft errors.

## Checks And Subtasks

Old MTUP subtasks:

- no-var AMR
- concept list
- relation list

Oracle checks:

- wrong root
- missing concepts
- wrong or missing relations
- missing `:wiki`, `:op1/:op2`, `:manner`
- variable or reentrancy errors

The checks should be compact. Their purpose is to guide final AMR repair, not to become the main output.

## Evaluation

- Compare `AMR_DRAFT` against the `direct-amr` baseline.
- Compare `AMR_FINAL` against draft, B1, B2, and old MTUP2.
- Report Smatch, root accuracy, parsability, and targeted repair counts.
- Track whether final AMR actually changes and repairs the draft, instead of only copying it.

Expected outcome:

- Vietnamese: `MTUP_FINAL > old MTUP2 ~= B2 > B1`.
- English: smaller gain; target is `MTUP_FINAL >= B1`.

## Optional Future Work

Iterative MTUP self-improve: use trained MTUP to generate improved drafts for a later training round, then continue training to fix its own residual errors. Treat this as optional or future work unless the first MTUP draft-check-final experiment is not strong enough.
