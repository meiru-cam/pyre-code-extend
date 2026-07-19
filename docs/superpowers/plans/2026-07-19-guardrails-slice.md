# Agent Guardrails and Security Slice (Plan 5 of 6)

**Goal:** Add a separate executable guardrails path covering policy composition, human approval, and a guarded runtime with provenance, redaction, audit, and cancellation.

**Base:** New isolated `feature/guardrails-slice` worktree from merged Plan 4 `origin/main`. All adversarial fixtures are local and deterministic.

## Task 1: Guardrail fixture harness

**Files:** extend the agent scenario/data harness with adversarial fixture validation; create `tests/fixtures/guardrails/*.json` and `tests/test_guardrail_fixtures.py`.

- Fixture classes: legitimate request, direct injection, indirect tool-output injection, secret/PII egress, privilege escalation, poisoned memory, recursive delegation, and cancellation.
- Every rejection case has a paired legitimate case to measure false-positive behavior. Invalid fixture provenance or ambiguous expected action is a harness error.

## Task 2: `policy_engine`

**Contract:** `PolicyEngine(policies).evaluate(context) -> PolicyDecision`, where decision is typed `allow`, `deny`, or `transform`; policy order is deterministic; deny wins; transformations compose in order; exceptions fail closed with a recorded reason.

- Mutations: first-policy-wins, allow-overrides-deny, unordered set, fail-open exception, transform-after-deny, context mutation.
- Teach composition, false positives/negatives, latency/usability, and fail-open/fail-closed tradeoffs.

## Task 3: `approval_gate`

**Contract:** `ApprovalGate.classify(action, capabilities, provenance) -> ApprovalRequest | ApprovedAction | DeniedAction`; deny unauthorized capabilities, require approval for configured risk, bind an approval to action digest/arguments/identity/expiry, and prevent replay or changed-argument reuse.

- Mutations: approve-by-tool-name only, ignore arguments, reusable token, no expiry, privilege inheritance, untrusted provenance treated as trusted.
- Exercise least privilege, human-in-the-loop races, idempotency, and auditable API design.

## Task 4: `guarded_runtime`

**Contract:** `guarded_runtime(runtime, policy_engine, approval_gate, redactor, audit_sink, request) -> GuardedResult`; preserve trusted/untrusted boundaries, inspect model/tool inputs and outputs, redact likely secrets/PII before egress/logging, enforce delegation depth/budgets, support emergency cancellation, and emit tamper-evident ordered audit records without raw secrets.

- Failure injection covers indirect injection through tools, poisoned memory provenance, partial tool failure, cancellation during approval, and audit failure.
- Reject sanitize-input-only, redact-after-log, uncapped delegation, dropped provenance, audit-secrets, approval bypass, and cancel-without-propagation.
- Pin exact implementation locations in NeMo Guardrails, OpenAI Agents SDK guardrails, Meta Purple Llama, OpenClaw/Hermes, plus precise OWASP agentic guidance; never require a live moderation API.

## Task 5: Path and merge

- Add `agent-guardrails-security` path, two independent hint levels, pros/cons, precise optional references, starters, and catalogs.
- Require attack rejection and legitimate acceptance, every named mutation rejected, three deterministic runs, full repository verification, code review/fixes, PR, and merge.
