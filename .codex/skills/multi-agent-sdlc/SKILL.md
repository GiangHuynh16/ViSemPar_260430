---
name: multi-agent-sdlc
description: Orchestrate a multi-agent SDLC delivery loop from technical requirements and measurable acceptance criteria. Use when Codex needs to turn a product or technical brief into a workflow with requirement clarification, environment setup, backend implementation, black-box API testing, anti-hardcoding review, OpenAPI export, and iterative fixes before handoff. Best for Python, FastAPI, backend, or AI-agent systems where BA, DevOps, Dev, Tester, and Reviewer roles must stay separate and assumptions must be surfaced instead of hidden.
---

# Multi-Agent SDLC

Coordinate a delivery team, not a single linear implementation. Drive the work from explicit requirements, measurable acceptance criteria, and role-separated quality gates.

## Execution Model

Use this skill as team orchestration policy.

- The main conversation owns plan, sequencing, integration, and final synthesis.
- Runtime sub-agents own role-specific execution when delegation is available and the user asked for agent-style work.
- If runtime sub-agents are unavailable, simulate the same roles sequentially in the main conversation and keep artifacts separate by role.
- Do not pretend a role has completed work unless that role's artifact or verdict is actually produced.

## Input Contract

Require these inputs before implementation starts:
- technical requirement
- measurable acceptance criteria
- known constraints: language, framework, deployment target, latency or cost limits, AI quality targets, output format, external services
- required integrations and credentials the user must provide
- testable output surface: API, CLI, file output, HTML, JSON, or another observable contract

If acceptance criteria are vague, BA must stop and convert them into measurable checks such as:
- accuracy target
- allowed failure rate
- output schema or HTML requirements
- latency budget
- must-pass user stories
- must-pass black-box test cases

If the user asks for statistical quality targets such as 90 percent accuracy, require an evaluation set, a labeled sample, or an explicit proxy metric before claiming the target is met.

Read [references/artifacts.md](references/artifacts.md) for artifact formats and [references/quality-gates.md](references/quality-gates.md) for stop conditions.

## Team Roles

### BA

- Act as the tech lead facing the user.
- Ask only the questions that materially change design, scope, or verification.
- Offer concrete options when multiple interpretations exist.
- Freeze clarified scope into user stories and acceptance test cases before Dev starts.
- After the team reports green, generate a small second-pass set of additional tests and send them back through Tester and Reviewer.

### DevOps

- Build the setup checklist before coding.
- Identify required libraries, local services, env vars, model or provider keys, GPU or CUDA constraints, and external dependencies.
- Ask the user to provide missing prerequisites early instead of letting Dev discover them late.
- Prefer simple local infrastructure for development when acceptable, such as SQLite for SQL-backed prototypes.
- Do not claim environment readiness without a runnable verification step.

### Dev

- Implement the backend or core behavior.
- Keep changes minimal and requirement-driven.
- Add unit tests for core logic and make them pass before handing off.
- Expose a local backend API for black-box testing.
- Export FastAPI OpenAPI JSON after the delivery loop is green.

### Tester

- Test only through the public surface: API, payload schema, field descriptions, scripts, and observable outputs.
- Do not read the Python implementation.
- Validate user stories and test cases with black-box scripts.
- Return a defect report with reproduction steps, payloads, and expected versus actual behavior.

### Code Reviewer

- Review the backend after each Dev handoff and before Tester sign-off is accepted as final.
- Reject hard-coded logic, forced prompt outputs, hidden test-case matching, or brittle custom rules added only to satisfy known examples.
- Prefer library-native capabilities and simple adapters over custom logic.
- Allow output restructuring or normalization when it stays faithful to library results.
- Limit review to one focused pass per Dev handoff, then send concrete fixes back to Dev.

## Workflow

1. Parse the requirement and extract missing decisions, blockers, and measurable success criteria.
2. Run BA clarification with the user before any implementation work that depends on unclear assumptions.
3. Produce the first artifact set:
   - clarified scope
   - chosen options
   - user stories
   - acceptance test cases
4. Run DevOps setup and stop if credentials, services, hardware, or local dependencies are missing.
5. Run Dev implementation and unit tests until the backend API is ready for black-box testing.
6. Run Code Review on the delivered backend before trusting the result.
7. Run Tester against the public API surface only.
8. If Reviewer or Tester fails, route concrete feedback to Dev and repeat steps 5 through 7.
9. When both Reviewer and Tester pass, require Dev to export FastAPI OpenAPI JSON for frontend consumers.
10. Send the result back to BA for final inspection and a small batch of new edge cases.
11. Run the BA-generated cases back through Tester, then Reviewer if fixes were needed.
12. Finish only when all stories, original tests, BA extra tests, and review gates are green.

## Role Order And Loop Rules

Follow this strict order:
- BA -> DevOps -> Dev -> Code Reviewer -> Tester
- If Dev changes code after Reviewer or Tester feedback, run Code Reviewer again before accepting the next Tester pass.
- If BA adds extra cases at the end and they fail, route the defect to Dev, then Reviewer, then Tester again.
- Do not skip Reviewer just because Tester passed.
- Do not skip Tester just because unit tests passed.

## Delegation Rules

Use real sub-agents only when the runtime allows it and the user explicitly wants agent-style collaboration. Otherwise emulate the roles locally.

When spawning runtime sub-agents for this workflow, set the model override to `gpt-5.4-mini` for BA, DevOps, Dev, and Tester.
Set the Code Reviewer model override to `gpt-5.4`.

When delegating:
- assign one owner per role
- give each role only the context it should legitimately see
- keep Tester blind to source code
- keep Reviewer focused on source and anti-hardcoding checks
- merge results in the main conversation
- avoid overlapping write ownership across concurrent workers

Recommended split:
- main conversation: BA and final orchestration
- sub-agent or local phase: DevOps
- worker: Dev
- worker or explorer with a strict black-box brief: Tester
- explorer or worker with a review brief: Code Reviewer

## Non-Negotiable Standards

- Surface unclear requirements instead of guessing.
- Use the simplest implementation that satisfies the requirement.
- Avoid speculative features and premature abstraction.
- Prefer official or library-native behavior over custom emulation.
- Treat acceptance criteria as executable checks, not prose.
- Keep every changed line traceable to a user story or failing test.
- For AI systems, optimize for general behavior that still passes tests. Never hard-code prompts, outputs, or branch rules to match known cases only.

## Output Contract

Default to these sections in the main conversation:
- Clarifications Needed
- Chosen Options
- User Stories
- Acceptance Test Cases
- Environment Checklist
- Implementation Status
- Review Findings
- Test Results
- OpenAPI Export Status
- Remaining Risks

Use the artifact formats in [references/artifacts.md](references/artifacts.md) and the exit criteria in [references/quality-gates.md](references/quality-gates.md).
