# Artifact Formats

Use these artifacts to keep role handoffs explicit and auditable.

## BA Deliverables

Produce these before Dev starts:
- `Clarifications Needed`: question, why it matters, options, chosen answer, open owner
- `Chosen Options`: architecture or tooling decisions that were explicitly selected
- `User Stories`: `US-<n>`, story statement, acceptance checks
- `Acceptance Test Cases`: `TC-<n>`, linked story, request or input, expected result, pass rule

For AI quality targets:
- tie each quality claim to an eval set, labeled sample, or approved proxy
- if no eval surface exists, mark the target as unverified instead of green

## DevOps Deliverables

Produce:
- `Environment Checklist`: runtime, package manager, framework, ports, services, env vars, credentials, hardware constraints
- `Missing Prerequisites`: what the user must provide before implementation can be trusted
- `Verification Step`: exact command or observable check proving the environment is ready

Prefer local-first setup when allowed:
- SQLite for simple SQL-backed work
- mocked or local services only when they still exercise the intended contract

## Dev Deliverables

Produce:
- `Implementation Status`: what was implemented and what was deferred
- `Unit Test Status`: exact command and result
- `Local API Contract`: routes, payload shape, field descriptions, example request and response
- `Run Instructions`: minimal commands needed for Tester to hit the backend

After the full loop is green, also produce:
- `OpenAPI Export Status`: path or endpoint for the FastAPI OpenAPI JSON export

## Tester Deliverables

Tester must stay black-box. Do not read Python source.

Produce:
- `Test Harness`: scripts or requests used to hit the API
- `Test Results`: one line per case with status
- `Defect Report`: case id, repro steps, payload, expected result, actual result

Tester may read only:
- API routes
- payload structures
- field descriptions
- run instructions
- user stories and test cases

## Code Reviewer Deliverables

Produce:
- `Review Findings`: severity, file or module, risk, why it is brittle, requested fix
- `Hardcoding Check`: explicit yes or no verdict plus evidence
- `Library-Native Opportunities`: SDK or library features that can replace custom logic

High-signal review targets:
- prompt branches that inject expected answers
- rule trees that mirror known test cases
- string matching added only to force pass behavior
- unnecessary fallback chains when the library already provides a stable contract

## BA Final Pass

After Reviewer and Tester both pass:
- generate 2 to 5 extra cases that were not used during implementation
- prefer edge cases, adversarial phrasing, empty or partial inputs, and realistic variations
- send those cases back through Tester

If those cases fail:
- reopen the Dev -> Reviewer -> Tester loop
- do not mark the story done until the extra cases are green
