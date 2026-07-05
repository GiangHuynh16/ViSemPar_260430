# Simplification Checks

Use this checklist when deciding whether a rewrite is genuinely simpler or just shorter.

## Keep

- behavior, outputs, public contracts, and side effects
- repository conventions and local instruction files
- useful abstractions with real reuse or boundary value
- explicit error handling that reflects a real failure mode

## Remove Or Reduce

- single-use wrappers that hide trivial calls
- fallback chains with unclear priority
- nested ternaries
- branch duplication
- shape guessing around known SDK or model types
- comments that only restate obvious code after the rewrite

## Prefer

- early returns
- explicit `is None` or `is not None` checks when falsy values matter
- one responsibility per helper
- direct documented attributes or methods over generic probing
- narrow tests that lock down the touched behavior

## Stop And Ask

- the "simpler" rewrite changes ordering, retries, exception behavior, or logging semantics
- the current abstraction may be used by callers you have not checked
- the code looks messy because it is compensating for version-specific library behavior you have not verified
- proving equivalence would require domain knowledge or test coverage you do not have
