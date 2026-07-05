---
name: code-review
description: Maintainability-focused code review for complex or brittle logic. Use when Codex needs to scan a codebase or module for custom logic that should be simplified, high-complexity flows, nested try/except blocks, fallback-heavy code, ambiguous dict access, use of or/getattr/isinstance, or places where SDK-native methods and attributes should replace custom handling with minimal flow changes. Use for targeted module review or for broad or multi-feature review, and spawn sub-agents only when the scope spans the overall system or n+1 features, delegation is allowed, and the user did not restrict the review to one small feature.
---

# Code Review

Review for maintainability, confidence, and unnecessary custom logic. Prefer surgical refactors that preserve behavior and reduce future reading cost.

## Execution Model

Use this skill as the stable review policy.

- The skill defines what to inspect, what evidence is required, and how to report findings.
- The main conversation owns scope selection, orchestration, and final synthesis.
- Runtime sub-agents are an execution strategy for broad or multi-feature review, not the primary artifact.

Do not treat this skill as a persistent standalone agent. Treat it as:
- `skill` for review standards and output contract
- `main conversation` for final judgment and prioritization
- `runtime sub-agents` for parallel scanning when the scope is large enough

## Review Goals

Find code that is hard to trust, hard to read, or too expensive to maintain.

Prioritize:
- nested `try/except`
- fallback chains
- boolean defaulting with `or`
- `getattr`
- `isinstance`
- ambiguous or weakly justified dict access
- custom logic that duplicates SDK capabilities
- long methods with branching, shape guessing, or mixed responsibilities

## Scope Decision

Choose the review mode before scanning.

### Targeted Review

Use this when the user points to one small module, file, feature, route, service, or bug area.

Do not spawn sub-agents.
Review locally and go deep.

### Broad Or Multi-Feature Review

Use this when the user asks to review the codebase, a large package, or n+1 features.

If the runtime allows delegation and the user did not narrow the scope, split the review by module or feature and spawn sub-agents in parallel.

Suggested split:
- API/domain modules
- core orchestration or business logic
- infra/integration adapters
- frontend service/state layers

Each sub-agent should own one slice and return:
- top maintainability findings
- candidate SDK-native replacements
- minimal-change refactor ideas
- specific methods, attributes, or docs that justify the proposal

Main conversation responsibilities:
- choose the split
- avoid overlapping ownership
- merge findings
- remove duplicates
- rank severity across modules
- produce one final recommendation set

## Workflow

1. Identify the review scope and whether it is targeted or broad.
2. Search the source for likely problem patterns before reading deeply.
3. Read the highest-signal files and isolate the exact methods causing complexity.
4. Check current SDK docs, source, type hints, or official references before proposing replacements, and browse the internet when the behavior is not fully verifiable from local code alone.
5. Prefer existing SDK methods and typed attributes over custom parsing.
6. Produce findings ordered by severity and maintenance impact.
7. Propose the smallest refactor that materially improves confidence and readability.

## Search Heuristics

Start with fast pattern search. Prefer `rg`.

Useful patterns:
- `try:`
- `except`
- ` or `
- `getattr(`
- `isinstance(`
- `.get(`
- `dict[`
- manual normalization or shape conversion helpers
- long controller/service/repository methods
- response parsing around SDK objects

Use search results to rank review targets. Do not report raw pattern counts as findings without reading the code.

## Evidence Standard

Do not guess about SDK behavior.

Before recommending a change:
- inspect the local code that uses the SDK
- inspect SDK docs, source, stubs, models, or official examples when the behavior may be version-sensitive
- name the exact method, property, field, or response model that should be used
- explain why it is safer than the current custom logic

If you cannot confirm the SDK surface confidently, say so and keep the recommendation conditional.

Treat the following patterns as mandatory verification targets, not optional review points:
- fallback chains
- `or` used for data selection or defaulting
- `getattr`
- code paths that feel confusing, shape-uncertain, or weakly justified

For every such case, explicitly trace the real contract by checking the SDK, official docs, source, stubs, examples, or current internet references until the expected shape and access pattern are clear enough to judge whether the code is correct.
Do not skip these cases just because the current code appears to work.

## Preferred Refactor Direction

Default to the least disruptive fix that improves confidence.

Prefer:
- explicit ordered field lookup over broad fallback chains
- direct attribute access when the SDK contract is known
- typed response models or documented fields over dynamic shape guessing
- small adapter functions with one responsibility over large normalization methods
- early validation and narrow branching over nested exception flow

Avoid recommending:
- broad rewrites when a local refactor is enough
- defensive code for unsupported shapes without evidence
- new abstractions for one-off logic

## Specific Anti-Patterns

Treat these as high-signal review targets:

### Nested `try/except`

Usually indicates hidden control flow or weak understanding of the actual response contract.

Look for:
- parse attempts layered by fallback
- exception-driven branching
- mixed transport, parsing, and business logic in one block

Recommend:
- validate shape once
- branch explicitly
- move the narrow risky call behind a small helper if needed

### `or` For Fallback Data Selection

`or` hides meaning and breaks when valid falsy values exist.

Prefer:
- explicit priority order
- explicit `is not None` checks
- exact field mapping based on the real schema

Always verify the true response shape before changing this logic. Check whether the underlying SDK or API already guarantees one canonical field so the fallback can be removed entirely.

### `getattr`

`getattr` often hides uncertainty about the SDK object model.

Prefer:
- direct documented attributes
- a single adapter layer if multiple versions genuinely exist

Do not leave `getattr` in place without proving why direct attribute access is unsafe. Read the SDK contract first.

### `isinstance`

Use only when the type split is intentional and proven.

If it exists because the code does not know the contract, look for:
- a stronger common interface
- a documented SDK model
- a preprocessing boundary that normalizes once

## Mandatory Deep Checks

Do not miss review hotspots just because they look routine.

Whenever the code contains fallback-heavy access, dynamic attribute lookup, or confusing data-shape handling:
1. inspect the surrounding function and callers
2. identify the upstream SDK, library, or API source
3. verify the real response/model shape from official docs or source
4. decide whether the current code is correct, over-defensive, or wrong
5. propose the minimal refactor that makes the contract explicit

If the code uses a dict where the upstream source is actually a typed SDK object, call that out explicitly and recommend moving to the documented attributes where feasible.

## SDK Replacement Review

When custom logic wraps an SDK, actively check whether the SDK already provides:
- canonical accessors
- pagination helpers
- retry helpers
- serialization or parsing helpers
- streaming/event iterators
- typed submodels
- utility methods for filtering, projection, or transformation

If a suitable SDK method exists and switching to it keeps the flow mostly intact, recommend using it directly.

State:
- exact SDK method or attribute
- current custom logic it replaces
- minimal surrounding changes needed to adapt callers
- expected maintenance win

## Output Contract

Report findings first. Keep summaries brief.

For each finding include:
- severity
- file and method
- why the current code is hard to maintain
- exact risky pattern
- SDK/doc/source evidence if relevant
- minimal refactor path
- residual risk or open question

After findings, include:

### Refactor Opportunities

List the best candidates where a small change yields a large maintenance win.

### SDK-Native Replacements

List the exact SDK methods, attributes, or models that should replace custom logic when confirmed.

### Review Coverage

State what was reviewed and what was not.

## Delegation Rules

Only use sub-agents for broad or multi-feature reviews when the environment and current instructions allow delegation.

When delegating:
- split by disjoint module or feature
- keep the immediate blocking narrow review in the main conversation
- tell each sub-agent not to refactor, only to inspect and report unless the user asked for fixes
- require file references and concrete method names
- require SDK evidence, not speculation
- merge the results into one ranked report

Recommended delegation pattern:
1. Main conversation decides whether the scope is small or broad.
2. If small, review locally and finish in the main conversation.
3. If broad or n+1 features, assign one module or feature slice per sub-agent.
4. Keep the prompt narrow: scan, verify SDK shape, report findings, do not speculate.
5. Main conversation synthesizes the combined report and makes the final call.

Do not spawn sub-agents when:
- the user asks about one small module
- the user asks about one small feature
- the next step is blocked on your own direct review of a narrow area
- the runtime or instructions do not permit delegation

Example sub-agent task shape:
- own one module or one feature slice
- scan for nested `try/except`, fallback chains, `or`, `getattr`, `isinstance`, and confusing SDK handling
- verify questionable shapes against SDK docs, source, stubs, or official references
- return only findings, evidence, and minimal-change refactor options

## Response Style

- Be strict about evidence.
- Be explicit about confidence level.
- Prefer concrete code paths over general style advice.
- Recommend fewer, stronger changes rather than broad cleanup themes.
