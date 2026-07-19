# Agent Runtime and Multi-Agent System Design Slice (Plan 4 of 6)

**Goal:** Build a reusable deterministic agent/data harness and a separate path covering tool APIs, a budgeted retrying runtime, and supervisor-worker communication with partial failures.

**Base:** New isolated `feature/agent-runtime-slice` worktree from merged Plan 3 `origin/main`; grading must never call live models/tools.

## Task 1: Agent/data harness

**Files:** create `torch_judge/harness/agents/{__init__,protocol,scripted_model,fake_tools,virtual_clock,scenarios,trace,assertions}.py`; test each in `tests/test_agent_harness.py`.

- `MessageEnvelope(message_id, correlation_id, parent_run_id, sender, recipient, deadline, attempt, idempotency_key, payload)` validates required strings, non-negative attempts, and JSON-like payloads.
- `ScriptedModel` consumes ordered responses/failures and records calls; exhaustion is a harness error.
- `FakeTool` supports deterministic results, retryable/permanent failures, side-effect counters, and idempotency keys.
- `VirtualClock.now()`/`sleep(delay)` advances virtual time only.
- A versioned `Scenario` loader validates unknown tools, malformed messages, failure schedules, budgets, and expected traces. This fixture loader is the plan's data harness.
- Typed `TraceEvent` and assertions cover ordering, retry delays, no duplicated effects, budgets, concurrency, final state, and correlation chains.

## Task 2: `tool_registry`

**Contract:** `ToolRegistry.register(tool)`, `.describe()`, and `.invoke(name, arguments, idempotency_key=None)`. Tools expose name, description, JSON-like argument schema, and callable handler. Reject duplicate names, unknown tools, missing/extra/type-invalid arguments before invoking a handler.

- Visible/unshown scenarios and mutations cover discovery order, validation, error taxonomy, idempotency-key forwarding, and least data returned.
- Connect exact tool abstractions in OpenClaw and Hermes; analyze centralized registry pros/cons and provider-neutral API boundaries.

## Task 3: `budgeted_agent_loop`

**Contract:** `budgeted_agent_loop(model, registry, messages, limits, retry_policy, clock, trace) -> AgentResult`. Iterate scripted model responses; execute requested tools; classify retryable versus permanent failures; honor retry-after/backoff with virtual time; deduplicate side effects; enforce iterations/tool calls/tokens/time/cost; support cancellation; emit ordered lifecycle events.

- Reject retry-all, no retry, real sleep, duplicate effect, off-by-one budgets, missing cancellation, malformed tool response, and swallowed permanent failure.
- Design note rubric: runtime state ownership, adapter boundaries, retry policy, backpressure, observability, and tradeoffs.

## Task 4: `supervisor_orchestration`

**Contract:** `supervisor_orchestration(request, workers, scheduler, bus, limits, checkpoint_store, trace) -> OrchestrationResult`. Supervisor emits correlated envelopes, routes by declared capabilities, respects global concurrency, fans out/fans in, returns successful partial results plus typed failures, checkpoints completed work, and resumes without repeating completed side effects.

- Scenarios: message ordering, fair routing, worker/API failure, timeout, retry exhaustion, queue pressure, cancellation propagation, checkpoint resume, dead-letter outcome, parent/correlation IDs.
- Reject unbounded fan-out, shared mutable message, retry without attempt increment, lost partial success, duplicate resumed work, missing backpressure, and failure-as-success.
- Connect OpenClaw, Hermes, and Microsoft Agent Framework exact classes/APIs; discuss centralized supervisor versus peer-to-peer topology.

## Task 5: Path and merge

- Add `agent-runtime-system-design` path and starters/catalogs. Each exercise has two hint levels, code provenance, application connections, pros/cons, deterministic statuses, and a design-note rubric marker for Plan 6.
- Run reference, mutation, failure-injection, and repeated deterministic suites; then full repository verification, review/fix, PR, and merge.
