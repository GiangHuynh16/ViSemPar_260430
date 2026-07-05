# Quality Gates

Use these gates as stop conditions. Do not advance on partial confidence.

## BA Gate

Pass only when:
- the requirement is clear enough to implement
- assumptions are either confirmed or explicitly deferred
- user stories and acceptance test cases exist
- measurable success criteria exist for every critical story

Fail when:
- success is described only in prose
- multiple interpretations exist and none was chosen
- an accuracy target exists without an eval surface or approved proxy

## DevOps Gate

Pass only when:
- required dependencies and services are listed
- required credentials or user-provided assets are identified
- environment verification can be run or observed

Fail when:
- Dev would be blocked by missing keys, services, hardware, or local data
- the setup depends on a hidden assumption

## Dev Gate

Pass only when:
- core behavior is implemented
- unit tests for the core path pass
- a public local API surface exists for black-box testing

Fail when:
- behavior exists only in notebooks, scripts, or internal functions
- unit tests are skipped without an explicit reason

## Review Gate

Pass only when:
- no intentional hardcoding or test-case forcing remains
- the implementation is general enough for nearby inputs
- custom logic is justified and not replacing an available library feature without reason

Fail on:
- expected outputs embedded in prompts or branching rules
- branches that match exact test phrases without a general rule
- brittle custom parsing where the upstream library already provides a stable structure

## Tester Gate

Pass only when:
- all required stories pass through the public contract
- all acceptance test cases pass
- failures are reproducible from requests and responses alone

Fail when:
- the backend passes only under white-box help
- payloads or field meanings are unclear
- expected versus actual behavior is not recorded

## OpenAPI Gate

Pass only when:
- the FastAPI OpenAPI JSON is exported after the delivery loop is green
- frontend consumers can discover payload shapes from that export

## Final BA Gate

Pass only when:
- BA extra cases pass after the main test suite is green
- any new fixes go back through Review before the next Tester verdict is accepted

## Exit Rule

Declare completion only when every gate above is green.

If a gate fails, route the work as follows:
- BA failure -> return to user clarification
- DevOps failure -> block Dev and request prerequisites
- Dev failure -> return to Dev
- Review failure -> return to Dev, then rerun Review
- Tester failure -> return to Dev, then rerun Review, then rerun Tester
- Final BA failure -> return to Dev, rerun Review, rerun Tester, then rerun BA extra cases
