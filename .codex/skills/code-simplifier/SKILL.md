---
name: code-simplifier
description: Simplify and refine existing code for clarity, consistency, and maintainability while preserving exact behavior. Use when the user asks to simplify, clean up, de-complexify, or make code easier to read without changing outputs, features, or contracts. Best for recently modified code, a narrow file set, or a concrete function/component the user wants rewritten more simply. Prefer this skill over broad review when the task is to implement a surgical simplification, not just report findings.
---

# Code Simplifier

Simplify code without changing what it does. Optimize for lower reading cost, clearer control flow, and fewer unnecessary moving parts.

## Execution Model

Use this skill as rewrite policy, not review-only policy.

- The main conversation owns scope, edits, verification, and final explanation.
- Default to the smallest possible scope: the current diff, the files just touched, or the exact files the user named.
- If the user did not name a scope, simplify only the code modified in the current session unless local context proves a slightly wider change is required.
- Do not silently broaden into general cleanup.
- If the task is primarily inspection and recommendations, use `code-review` instead of this skill.

## Core Goal

Preserve exact behavior while making the code easier to trust and maintain.

Prefer:
- explicit control flow over dense or clever expressions
- small local rewrites over architectural refactors
- removing redundant branches, wrappers, and helpers
- direct use of documented library or SDK behavior when the contract is known
- consistent naming and structure that matches the surrounding codebase

Avoid:
- feature additions
- speculative abstractions
- style churn outside the requested scope
- reducing line count at the cost of clarity
- replacing clear abstractions with inlined duplication

Read [references/simplification-checks.md](references/simplification-checks.md) before editing when the code contains fallback-heavy logic, nested conditionals, or dynamic shape handling.

## Workflow

1. Define the exact scope and the non-negotiable behavior that must not change.
2. Inspect local repo instructions such as `AGENTS.md`, `CLAUDE.md`, or project conventions before editing.
3. Identify the specific complexity drivers:
   - deep nesting
   - boolean logic that hides intent
   - fallback chains
   - one-off abstractions with no reuse value
   - mixed responsibilities inside one function
   - custom parsing that duplicates library behavior
4. Choose the least disruptive rewrite that makes the contract more explicit.
5. Implement the simplification directly instead of only describing it, unless the user asked for review only.
6. Verify behavior with the narrowest reliable check:
   - existing tests for the touched area
   - a new focused regression test when a bug or fragile path is involved
   - lint, typecheck, or a local smoke path when tests are unavailable
7. Remove only the imports, helpers, or branches made obsolete by the simplification.

## Scope Rules

- Keep edits surgical. Every changed line should trace to the simplification goal.
- Prefer rewriting one function or component cleanly over partially rewriting many files.
- Do not refactor adjacent code just because it also looks messy.
- If a wider cleanup is justified, state why before doing it.
- If exact behavior cannot be preserved confidently without more information, stop and surface the uncertainty.

## Preferred Simplifications

Treat these as high-signal targets:

### Nested Control Flow

- Flatten nested `if/else`, `try/except`, or guard stacks when early returns make the path clearer.
- Separate validation from transformation when both are mixed together.

### Fallback-Heavy Data Access

- Replace `a or b or c` data selection with explicit ordered checks when falsy values may be valid.
- Remove fallback branches entirely when local code or verified docs show a canonical field or shape.

### Dynamic Or Defensive Access

- Remove `getattr`, `isinstance`, or dict-shape guessing when the real contract is known.
- Keep a small adapter only when multiple shapes genuinely exist.

### Redundant Abstractions

- Inline single-use wrappers that hide simple behavior.
- Delete helpers introduced only to save a few lines when they increase indirection.

### Dense Expressions

- Replace nested ternaries, compact boolean tricks, or heavily chained expressions with explicit branches.
- Prefer clarity over brevity.

## Verification Standard

Do not claim "no behavior change" without a concrete check.

- Run the most relevant existing verification for the touched code.
- Add a focused regression test when a simplification could plausibly alter branching or data handling.
- If no automated check is available, say exactly what was and was not verified.
- When simplifying SDK or library interactions, confirm the replacement behavior from local code, types, or official references before deleting defensive logic.

## Output Contract

After completing the work, report:
- what scope was simplified
- what complexity was removed
- what verification was run
- any residual risk if full behavioral equivalence could not be proven

Keep the explanation brief. The main artifact is the simplified code, not a long rationale.
