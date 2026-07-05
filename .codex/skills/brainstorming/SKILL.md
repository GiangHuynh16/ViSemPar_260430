---
name: brainstorming
description: Research-backed solution discovery and option analysis for technical decisions. Use when Codex needs to brainstorm architecture, platform, SDK, library, service, open-source, or implementation options; compare trade-offs for AI applications such as RAG, agents, multi-agent systems, GenAI, AI platforms, backend, frontend, DevOps, or Azure solutions; clarify unclear requirements before proposing a design; and recommend the strongest option for the current context instead of guessing.
---

# Brainstorming

Provide high-signal technical option analysis. Optimize for decision quality, not idea volume.

## Workflow

1. Parse the request and identify the actual decision to make.
2. Check whether the requirement is clear enough to compare options.
3. If key constraints are missing, ask a small number of targeted questions before proposing a solution.
4. Research the current solution space before inventing custom designs.
5. Produce up to 3 viable options, with the recommended option first.
6. Explain why the recommended option is best for the current context and what trade-offs remain.

## Clarify First When Needed

Do not silently fill in major gaps. If the request is vague, stop and ask only the questions that materially change the recommendation.

Prefer questions about:
- scale, traffic, latency, and cost targets
- Azure or cloud constraints
- build vs buy expectations
- team strength, delivery speed, and ops tolerance
- compliance, data residency, security, and lock-in limits
- whether the user wants an existing product/open-source path or a custom design

Keep clarification tight:
- Ask 1-4 questions, not a long questionnaire.
- Explain why each missing detail matters when the trade-off is non-obvious.
- If enough is already known, do not ask questions just to be formal.

## Research First

Start with current external research unless the user explicitly says not to browse or the problem is purely internal to the local codebase.

Research priority:
1. Official product docs, SDK docs, API references, pricing pages, and vendor architecture guidance.
2. Primary sources such as GitHub repositories, RFCs, release notes, benchmarks, and technical design docs.
3. Reputable secondary analysis only when it adds useful synthesis.

Prefer proven solutions before custom invention:
- managed services over bespoke infrastructure when they fit the constraints
- established SDKs and libraries over reimplementing primitives
- mature open-source projects over greenfield components when maintenance cost matters

Only recommend a custom design first when:
- the user explicitly asks for a bespoke design
- existing tools do not fit the constraints
- the trade-off clearly favors control or specialization over speed

## Decision Standards

Assume the user is technical. Skip basic explanations and focus on decision quality.

Evaluate options using the dimensions that matter for the prompt. Common dimensions:
- implementation speed
- operational complexity
- cost now and at scale
- latency and throughput
- reliability and failure modes
- observability and debuggability
- security, compliance, and data boundaries
- vendor lock-in and portability
- extensibility for future product needs

For AI application topics, also check:
- retrieval quality and grounding
- agent controllability and evalability
- orchestration complexity
- tool integration surface
- memory/state management
- safety and guardrail support

## Output Contract

Default to this structure:

### Recommended

State the best option first. Make the recommendation explicit in the first sentence.

Include:
- what the option is
- why it is the best fit now
- where it wins
- its main trade-offs
- when it stops being the best choice

### Option 2

Present the strongest alternative when the user values a different trade-off.

### Option 3

Present the third viable choice only if it is meaningfully distinct. Do not force a third option if only two are credible.

### Gaps To Confirm

List only the unresolved questions that could still flip the recommendation.

## Option Format

Use concise, decision-oriented formatting. For each option, cover:
- approach
- best for
- strengths
- trade-offs
- operational impact
- implementation shape

When helpful, include a compact comparison table.

## Recommendation Rules

- Put the recommended option first, always.
- Keep the shortlist to at most 3 options.
- Do not present weak options just to make the list longer.
- Do not hedge unnecessarily. Recommend one path unless the decision genuinely depends on unresolved constraints.
- If the answer depends on missing information, say exactly what would change the recommendation.

## When The User Wants A Custom Design

If the user explicitly asks for a design optimized for their situation:
1. Research existing patterns, services, SDKs, and open-source implementations first.
2. Reuse proven building blocks where they reduce risk.
3. Customize only the parts that benefit from a bespoke design.
4. Explain why the tailored design beats off-the-shelf approaches for that context.

## Response Style

- Be direct and technical.
- Prefer concrete services, libraries, repos, and protocols over abstract advice.
- Name trade-offs explicitly.
- Call out assumptions instead of hiding them.
- Avoid long brainstorming dumps. Curate the best options.
