---
name: data-operation-review
description: Data-operation review for database, cache, Redis, and application-side data flows. Use when Codex needs to inspect how keys, values, records, documents, or payloads are shaped; whether CRUD flows update, invalidate, sync, or propagate data correctly; whether database, cache, search index, or source-of-truth data stay consistent; whether read/write paths use the right SDK methods; and whether broad or multi-feature data review should be split across sub-agents while narrow feature review stays in the main conversation.
---

# Data Operation Review

Review data flows for correctness, consistency, and maintainability. Focus on whether the code has a clear source of truth and whether every read/write/sync path preserves that contract across database, cache, index, and derived stores.

## Execution Model

Use this skill as the stable review policy.

- The skill defines what to inspect, what evidence is required, and how to report findings.
- The main conversation owns scope selection, orchestration, and final synthesis.
- Runtime sub-agents are an execution strategy for broad or multi-feature review, not the primary artifact.

Do not treat this skill as a persistent standalone agent. Treat it as:
- `skill` for data review standards and output contract
- `main conversation` for final judgment and prioritization
- `runtime sub-agents` for parallel scanning when the scope is large enough

## Scope Decision

Choose the review mode first.

### Focused Review

Use this when the user asks about one small feature, one module, one repository, one table or collection flow, one cache key family, or one CRUD path.

Do not spawn sub-agents.
Review locally for speed.

### Broad Or Multi-Feature Review

Use this when the user asks to review the overall data flow design, multiple modules, or n+1 features.

If delegation is allowed, split the work by module or feature and spawn sub-agents in parallel.

Suggested split:
- API and controller cache usage
- repository or service cache coordination
- Redis adapter or cache client layer
- background jobs, invalidation hooks, or event consumers

Each sub-agent should return:
- data consistency findings
- missing update, invalidation, sync, or stale-write risks
- key/value/record shape concerns
- SDK-native methods or patterns that should replace custom handling
- minimal-change refactor suggestions

Main conversation responsibilities:
- choose the split
- avoid overlapping ownership
- merge findings
- remove duplicates
- rank severity across modules
- produce one final recommendation set

## Workflow

1. Identify the source of truth for the reviewed data.
2. Map every create, read, update, delete, sync, invalidate, refresh, and propagation path.
3. Check whether CRUD and background operations keep database, cache, index, and source of truth consistent.
4. Check whether key construction, identifiers, and value or record serialization are explicit and stable.
5. Verify the database, Redis, cache, or data SDK usage before judging custom logic.
6. Report correctness and maintenance findings in severity order.
7. Recommend the smallest refactor that improves confidence without changing the flow unnecessarily.

## What To Check

Review these areas explicitly:
- table, collection, document, or entity ownership
- primary key, partition key, Redis key, and foreign-key shaping
- record, document, and payload shape consistency across writers and readers
- cache key naming, namespacing, and versioning
- key construction consistency across modules
- value shape consistency across writers and readers
- TTL choice and TTL consistency
- database write path and follow-on cache/index updates
- upsert, patch, merge, and delete semantics
- transaction, batch, or partial-failure handling
- write-through, write-behind, cache-aside, or mixed patterns
- invalidation on create, update, delete, bulk update, and background mutation
- list/detail cache coordination
- search index or materialized-view refresh behavior
- outbox/event-driven propagation when applicable
- fan-out invalidation across related keys
- stale reads after write
- double writes or forgotten writes
- memory cache and Redis cache divergence
- fallback behavior when cache misses or Redis errors happen

## Evidence Standard

Do not assume data correctness from happy-path code.

Before recommending changes:
- inspect the full read/write/invalidate path
- identify the database, cache, search, or data SDK in use
- verify the exact SDK methods, return shapes, and semantics from docs, source, stubs, or official examples
- browse current official docs or source when local code is not enough to confirm behavior
- name the exact method, parameter, or return contract that the code should rely on

If you cannot confirm the behavior confidently, mark the recommendation conditional.

Treat these as mandatory deep-check targets:
- custom key-building helpers
- manual serialization or deserialization logic
- manual DTO or record-shape reshaping around SDK objects
- fallback reads across multiple keys
- read-after-write logic
- invalidate-on-update or invalidate-on-delete paths
- update-then-index or update-then-publish flows
- cache refresh code in background jobs
- confusing code that mixes dicts, SDK objects, and custom wrappers

Do not skip these cases because the code appears to work.

## Preferred Refactor Direction

Prefer:
- one explicit source of truth per data flow
- one canonical key-building path per key family
- one canonical value shape per cache record
- one canonical record or document shape per entity flow
- explicit propagation steps after writes when derived stores exist
- explicit invalidation or refresh behavior on every mutation path
- SDK-native methods over custom wrappers when they fit with small adaptations
- narrow helper functions with one data responsibility

Avoid:
- layered fallback chains for cache lookups
- layered fallback chains for record or document field selection
- silent cache writes inside unrelated business logic
- mixed value shapes for the same key family
- mixed record shapes for the same logical entity
- cache repair logic that hides incorrect invalidation
- data repair logic that hides broken write propagation
- broad rewrites when a targeted fix is enough

## Database And Cache SDK Review

When a database, Redis, cache, search, or persistence SDK is involved, actively check whether the SDK already provides:
- canonical get/set/delete helpers
- bulk operations
- hash helpers
- pipelines or transactions
- TTL-aware setters
- atomic update primitives
- partial update or patch helpers
- optimistic concurrency or etag/version helpers
- batch read or write helpers
- pub/sub or event hooks where relevant
- scan or key-iteration helpers
- typed response models or record wrappers

If the code manually reimplements behavior the SDK already provides, recommend using the SDK method directly when the surrounding flow only needs small adaptations.

State:
- the exact SDK method or command surface
- the custom logic it can replace
- the minimal surrounding code changes needed
- the consistency or maintenance improvement gained

## High-Signal Findings

Treat these as likely bugs or maintenance risks:
- create or update path writes database state but does not update dependent stores
- update path writes source data but leaves stale cache
- delete path removes source data but not related cache entries
- source-of-truth write succeeds but index or derived projection stays stale
- partial failure can leave database and cache or index diverged
- list cache never invalidates when detail records change
- one module writes one value shape and another reads a different shape
- one module writes one document or record shape and another expects a different shape
- different modules build the same logical key differently
- cache fallback uses `or` and masks valid falsy values
- code guesses dict fields even though the SDK provides typed attributes or command results

## Output Contract

Report findings first.

For each finding include:
- severity
- file and method
- data flow being reviewed
- why the logic may become stale, inconsistent, or hard to maintain
- exact risky behavior
- SDK/doc/source evidence when relevant
- minimal refactor path
- residual risk or remaining unknown

After findings, include:

### Consistency Gaps

List missing or weak read/write/update/delete/sync/invalidate guarantees.

### SDK-Native Replacements

List exact database, Redis, cache, search, or data SDK methods, commands, or typed surfaces that should replace custom logic when confirmed.

### Review Coverage

State what modules and data flows were reviewed and what was not covered.

## Delegation Rules

Only use sub-agents when the review is broad or spans multiple features and the runtime allows delegation.

When delegating:
- split by disjoint module or feature
- keep the immediate blocking narrow review in the main conversation
- keep one ownership slice per sub-agent
- require concrete file references and data paths
- require SDK evidence instead of speculation
- merge the results into one ranked report

Recommended delegation pattern:
1. Main conversation decides whether the scope is small or broad.
2. If small, review locally and finish in the main conversation.
3. If broad or n+1 features, assign one module or feature slice per sub-agent.
4. Keep the prompt narrow: trace data flow, verify SDK shape, report findings, do not speculate.
5. Main conversation synthesizes the combined report and makes the final call.

Do not spawn sub-agents when:
- the user asks about one small feature
- the user names one narrow module or one CRUD path
- the next critical step is a local deep read of one specific area
- the runtime or current instructions do not permit delegation

Example sub-agent task shape:
- own one module or one feature slice
- trace create/read/update/delete/sync/invalidate paths
- verify questionable shapes, commands, and return contracts against SDK docs, source, stubs, or official references
- return only findings, evidence, and minimal-change refactor options

## Response Style

- Be strict about data correctness.
- Prefer concrete data-flow analysis over generic advice.
- Call out inconsistent key/value/record shape directly.
- Recommend fewer, stronger fixes that reduce maintenance cost.
