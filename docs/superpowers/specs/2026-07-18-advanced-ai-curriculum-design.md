# Advanced AI Systems Curriculum Extension

Date: 2026-07-18
Status: Draft for review
Repository: Pyre Code extension

## Summary

Extend Pyre Code with four separate, reusable learning paths:

1. Advanced Attention in Frontier Models
2. MoE Architecture and Training
3. Agent Runtime and Multi-Agent System Design
4. Agent Guardrails and Security

The paths retain Pyre Code's existing implementation-first, LeetCode-like workflow: learners implement small contracts, run deterministic local evaluator cases, inspect behavior-level failures, reveal hints only when wanted, and compare with a minimal reference implementation.

The extension emphasizes real source code over architecture-name recognition. Exercises connect minimal implementations to pinned code and precise optional paper sections from frontier models and production agent systems. No GPU, external API, or network connection is required for grading.

DeepSeek V4 Pro is the recommended optional AI provider for free-form help and system-design feedback. AI feedback never determines whether an exercise is solved.

## Goals

- Teach the implementation details and tradeoffs of modern attention mechanisms used by Kimi, GLM, Qwen, Llama, Gemma, and Mistral families.
- Expand MoE coverage from basic routing and load balancing into dispatch, capacity, router stability, shared experts, expert parallelism, and complete training steps.
- Teach actual agent-runtime implementation: tool APIs, model adapters, event loops, communication, scheduling, budgets, retries, idempotency, checkpointing, and multi-agent orchestration.
- Teach guardrails as executable system behavior: policy composition, least privilege, injection defense, approval gates, provenance, redaction, audit, and cancellation.
- Embed system design into agent exercises through executable interfaces and short structured design notes.
- Make every new exercise deterministic, offline, mutation-tested, and connected to pinned primary sources.
- Preserve current exercise IDs and learner progress.

## Non-goals

- Reproducing or pretraining full frontier-scale models.
- Copying substantial modules from production repositories.
- Requiring learners to read full papers before coding.
- Requiring live model-provider or tool APIs for grading.
- Automatically assigning an authoritative score to free-form architectural reasoning.
- Teaching introductory Python or all of PyTorch inside the advanced paths.
- Replacing the current paths; the existing LLM Frontier Architectures path remains a shorter overview.

## Learner Model

The new paths assume minimum PyTorch knowledge:

- Basic tensor creation, indexing, broadcasting, reshaping, transposition, and matrix multiplication
- Small `torch.nn.Module` implementations
- Losses, parameters, gradients, and optimizers
- Recognition of ordinary multi-head attention

Knowledge of frontier attention, MoE training, distributed expert execution, agent runtimes, or guardrails is not assumed.

Prerequisites are advisory. Paths and exercises are never locked. A readiness check may direct learners to existing Pyre Code exercises without blocking access.

## Curriculum Architecture

Each path follows a three-layer progression:

1. Primitive: implement one mechanism with a small contract.
2. Subsystem: compose mechanisms and handle realistic edge cases.
3. Integrative exercise: combine prior capabilities under production-like constraints.

The first delivery is a vertical slice that validates the full authoring, grading, hint, provenance, UI, and persistence workflow. Later pull requests expand each path to its full sequence.

### First vertical slice

| Path | Primitive | Subsystem | Integrative exercise |
|---|---|---|---|
| Advanced Attention | QK normalization | Hybrid local/global attention schedule | Configurable `FrontierAttentionBlock` |
| MoE | Top-k router | Capacity-aware dispatch and gather | Differentiable TinyMoE training step |
| Agent Runtime | Tool contract and registry | Budgeted agent loop with retries and idempotency | Supervisor-worker orchestration with partial failures |
| Guardrails | Composable policy engine | Tool risk and approval gate | Guarded agent runtime with redaction and audit trace |

Existing GQA, sliding-window attention, MLA, basic MoE, and MoE load-balancing exercises become advisory prerequisites where relevant.

### Full-path targets

- Advanced Attention: approximately 12 exercises
- MoE Architecture and Training: approximately 13 exercises
- Agent Runtime and Multi-Agent System Design: approximately 15 exercises
- Agent Guardrails and Security: approximately 10 exercises

## Path 1: Advanced Attention in Frontier Models

The path teaches attention as a configurable design space rather than a list of unrelated mechanisms.

Proposed full sequence:

1. Multi-Query Attention
2. Grouped-Query Attention, reusing and strengthening the current exercise
3. Sliding-window attention with Mistral connection
4. Local/global layer schedules with Gemma connection
5. QK normalization and logit stability
6. Interleaved RoPE/no-position layers, including iRoPE-style schedules
7. Differential attention, reusing and strengthening the current exercise
8. Multi-head Latent Attention, reusing current MLA foundations
9. Compressed MLA KV cache
10. Attention temperature scaling and QK clipping
11. Configurable hybrid attention dispatcher
12. `FrontierAttentionBlock` capstone

The capstone uses a small `AttentionProfile` configuration to represent architectural choices and connect them to Kimi, GLM, Qwen, Llama, Gemma, and Mistral variants. It does not claim that all models share one implementation.

Each model connection explains:

- What mechanism is used
- Where it appears in pinned source code
- Why the model may have selected it
- Memory, compute, quality, stability, and implementation tradeoffs
- What the exercise simplifies relative to production code

## Path 2: MoE Architecture and Training

The MoE path must go beyond a forward-only router. It teaches both sparse execution and how routing is trained.

Proposed full sequence:

1. Dense FFN versus sparse expert execution
2. Router probabilities and top-k selection
3. Dispatch, expert execution, and gather
4. Capacity, overflow, token dropping, and padding
5. Load-balancing loss, reusing and strengthening the current exercise
6. Router z-loss and logit stability
7. Noisy routing and exploration
8. Shared and routed experts
9. Fine-grained expert segmentation and specialization
10. Auxiliary-loss-free dynamic routing bias
11. Complete MoE training step with task and router losses
12. Expert-parallel all-to-all simulation
13. `TinyMoETransformer` training capstone on deterministic synthetic domains

MoE evaluator cases cover:

- Dispatch equivalence with a dense oracle
- Token conservation and absence of duplication
- Top-k selection and normalization
- Capacity and overflow behavior
- Sparse expert invocation counts
- Selected expert gradients
- Router gradients
- Load distribution metrics
- Expert permutation invariance
- Deterministic training behavior

The training capstone must verify more than decreasing loss. It checks gradient reachability, routing behavior, expert specialization signals, and stable behavior on seeded synthetic domains.

## Path 3: Agent Runtime and Multi-Agent System Design

System design is a first-class part of this path, not a separate fifth path. Every subsystem exercise connects code behavior to API boundaries, failure policy, concurrency, state, and observability.

Proposed full sequence:

1. Tool schemas and argument validation
2. Tool registry and capability discovery
3. Provider-neutral `ModelClient`
4. Deterministic agent-loop state machine
5. Lifecycle events and streaming traces
6. Session transcripts, context, and message ordering
7. Failure classification, retries, and virtual time
8. Idempotency and duplicate side-effect prevention
9. Per-session queues, global concurrency, and backpressure
10. Iteration, token, time, and cost budgets with cancellation
11. Mailboxes and typed message envelopes
12. Capability-based workload routing and fairness
13. Supervisor-worker fan-out/fan-in with partial failure
14. Checkpoint, resume, replay, and dead-letter handling
15. Gateway protocol and durable multi-agent capstone

The shared API vocabulary stays intentionally small:

- `ModelClient`
- `Tool`
- `Agent`
- `AgentRuntime`
- `MessageBus`
- `Scheduler`
- `StateStore`
- `CheckpointStore`
- `Guardrail`
- `TraceSink`

A typed `MessageEnvelope` includes fields such as:

- `message_id`
- `correlation_id`
- `parent_run_id`
- `sender`
- `recipient`
- `deadline`
- `attempt`
- `idempotency_key`
- `payload`

The capstone coordinates specialized agents, enforces concurrency and budgets, injects model and tool failures, resumes from checkpoints, and emits an auditable event trace.

Implementation connections include pinned, precise locations in OpenClaw, Hermes Agent, and Microsoft Agent Framework. The curriculum focuses on their runtime and system-design patterns rather than reproducing their code.

## Path 4: Agent Guardrails and Security

Proposed full sequence:

1. Structured input and output validation
2. Policy composition: allow, deny, transform, and fail closed
3. Tool capabilities and least-privilege permissions
4. Prompt-injection defense and trusted/untrusted data boundaries
5. Secret and PII detection and redaction
6. Action risk scoring and human approval gates
7. Memory-poisoning defense with provenance tracking
8. Delegation boundaries, recursion limits, and budget propagation
9. Audit traces, anomaly detection, and emergency cancellation
10. Guarded runtime capstone

Guardrail exercises use deterministic adversarial fixtures, not live moderation services. Tests cover both attack rejection and legitimate requests that must remain allowed. Descriptions discuss false positives, false negatives, latency, usability, and fail-open/fail-closed tradeoffs.

Implementation connections include NeMo Guardrails, OpenAI Agents SDK guardrails, Meta Purple Llama, OpenClaw, Hermes, and OWASP agentic-security guidance.

## Exercise Contract

New exercises extend the existing `TASK` dictionary while preserving legacy fields.

Conceptual schema:

```python
TASK = {
    "id": "moe_topk_router",
    "version": 1,
    "title": "Implement a Top-k Expert Router",
    "difficulty": "medium",
    "description_en": "...",
    "function_name": "topk_route",
    "advisory_prerequisites": ["softmax", "moe"],
    "hints": [
        {
            "level": 1,
            "kind": "questions",
            "content": "Which dimension represents experts? What must sum to one?",
        },
        {
            "level": 2,
            "kind": "analysis",
            "content": "Normalize router logits per token before selecting experts...",
        },
    ],
    "model_connections": ["..."],
    "pro_con_analysis": {"pros": ["..."], "cons": ["..."]},
    "sources": ["..."],
    "tests": ["..."],
    "solution": "...",
}
```

Legacy `hint` and `hint_zh` remain accepted. Existing exercises do not require immediate migration. Reused exercises added to the new paths receive the two-level hint structure when strengthened.

### Two independent hint levels

- Level 1 contains guiding questions only.
- Level 2 contains analytical guidance, equations, tensor-shape reasoning, or algorithm ordering.
- Level 2 never automatically opens when Level 1 is revealed.
- Neither level contains a complete submit-ready solution.
- English-only hints are acceptable for the new paths.
- The UI exposes separate controls for each level.
- Notebook/API helpers accept an explicit level, for example `hint("moe_dispatch", level=1)`.

### Optional precise references

Exercises are attemptable without opening external sources. An optional reference section points learners directly to:

- A named paper/report section
- Specific equations, figures, and pages
- A commit-pinned repository file
- A class, method, or exact line anchor

References are additional information, not prerequisites.

## Evaluation Model

### Visible and unshown cases

The evaluator uses a hybrid model:

- Visible examples clarify the contract.
- Unshown evaluator cases require a general implementation.
- Because the repository is open source, unshown cases are not described as secret or hidden.
- Failed cases report a behavior category without exposing the complete input or implementation.

Conceptual evaluator metadata:

```python
TestCase(
    name="router_probabilities_sum_to_one",
    behavior="routing.normalization",
    visibility="unshown",
    run=test_router_probabilities_sum_to_one,
    failure_message=(
        "Routing normalization failed: expert probabilities must "
        "sum to one for each token."
    ),
)
```

Shared behavior categories include:

- `contract.signature`
- `tensor.shape`
- `tensor.dtype_device`
- `numerics.stability`
- `edge.empty_or_boundary`
- `state.invariant`
- `gradient.flow`
- `attention.masking`
- `attention.cache`
- `routing.selection`
- `routing.normalization`
- `dispatch.token_conservation`
- `capacity.overflow`
- `experts.sparsity`
- `training.router_gradient`
- `training.load_balance`
- `protocol.validation`
- `events.ordering`
- `retry.classification`
- `retry.backoff`
- `effects.idempotency`
- `scheduler.concurrency`
- `budget.enforcement`
- `checkpoint.recovery`
- `security.permission`
- `security.injection`
- `security.redaction`

Each evaluator case has one primary behavior category assigned by the author. The grader never infers categories from tracebacks.

### Solved status

Version one uses deterministic evaluation as the only source of `Solved` status.

- All required deterministic evaluator cases must pass.
- Structural properties are checked deterministically. For example, sparse execution uses invocation counts rather than timing benchmarks.
- Hardware-speed-dependent or flaky benchmarks do not gate completion.
- AI feedback never changes solved status.

System-design exercises display separate outcomes:

- Implementation: `Solved` or not solved
- Design note: present or absent
- AI review: `Not requested` or `Reviewed`

## Shared Harnesses

The tensor and agent paths use reusable evaluator infrastructure.

```text
torch_judge/harness/
├── tensor/
│   ├── assertions.py
│   ├── differential.py
│   └── gradients.py
└── agents/
    ├── protocol.py
    ├── scripted_model.py
    ├── fake_tools.py
    ├── virtual_clock.py
    ├── scenarios.py
    ├── trace.py
    └── assertions.py
```

### Harness terminology

- Agent harness: loop, tools, budgets, retries, orchestration, and state surrounding the learner implementation.
- Evaluation harness: deterministic scenarios and assertions used to grade behavior.
- Data harness: versioned fixture loading and validation inside the evaluation harness.

### Agent scenario behavior

An offline scenario may provide:

- Scripted model responses
- Fake tool responses and failures
- Virtual time
- Retry-after delays
- Cancellation signals
- Concurrency and budget limits
- Expected state and event traces

Evaluator assertions cover:

- Correct tool arguments
- Retry only for retryable failures
- Respect for retry-after using virtual time
- No duplicated side effects
- Event and message ordering
- Budget and concurrency enforcement
- Cancellation and checkpoint behavior
- Correct final state and trace

No agent evaluator calls a live model or tool API.

## Author-Quality Gate

An exercise is mergeable only when its evaluator has independent evidence of correctness.

Required evidence:

1. Hand-calculated examples for basic semantics
2. An independent oracle or PyTorch comparison
3. Invariant and metamorphic tests
4. Seeded randomized cases across valid shapes and boundaries
5. Gradient checks where applicable
6. Failure injection for agent and guardrail exercises
7. Named deliberately broken implementations
8. Rejection of every required mutation for the intended behavior category
9. Repeated deterministic offline runs
10. Pinned source provenance and adaptation notes

The evaluator must never import or call the exercise's reference implementation.

Representative MoE mutations include:

- Normalizing over the wrong dimension
- Losing or duplicating tokens during dispatch
- Applying capacity limits incorrectly
- Detaching router gradients
- Executing every expert despite sparse routing
- Counting shared experts as routed experts

## Provenance and Exercise Revisions

Every model connection stores:

- Repository URL
- Commit hash
- File path
- Class or function
- Optional line anchor
- Paper/report section, equation, figure, or page
- License
- What was adapted versus independently implemented
- Simplifications relative to production code

Source pins are immutable. Upstream changes do not silently alter an exercise.

Exercises have a stable ID and integer contract version:

- Wording, citations, hints, or compatible test strengthening do not require a version bump.
- Signature, semantics, required invariants, or architectural behavior changes increment the version.
- Learner attempts remain associated with the exercise version used.
- Prior completion remains visible when a newer version is available.

## System-Design Notes and AI Feedback

Executable system-design decisions receive deterministic checks. Examples include APIs, schemas, state transitions, retry behavior, idempotency, concurrency, and recovery.

Free-form reasoning uses a short structured design note covering:

- API boundaries and class responsibilities
- State and ownership
- Failure behavior and recovery
- Backpressure and concurrency
- Durability and idempotency
- Observability
- Security
- Explicit tradeoffs

A visible rubric and expert example support self-review. DeepSeek V4 Pro may provide optional feedback against the rubric. The provider interface remains OpenAI-compatible and provider-neutral.

The first release follows existing persistence conventions:

- Code and design-note drafts: browser local storage
- Submitted code and design notes: SQLite submission history
- Deterministic progress: existing progress model
- AI Help and design-review output: transient UI state
- API keys: environment only, never database storage

### AI data boundary

All external AI use is explicit and optional.

AI Help has two modes:

- Ask a question: send the learner query and exercise context; current code is optional.
- Review my code: additionally send current code after explicit learner choice.

Design review sends only the exercise context, rubric, and design note. It does not send repository files, `.env` contents, API keys, unrelated code, or the reference implementation.

The UI states clearly that deterministic grading is local while optional AI feedback leaves the machine. A local secret-pattern check blocks likely credentials from being included in an external request.

## UI Changes

The exercise workspace adds:

- Independent Level 1 and Level 2 hint controls
- Optional primary-source links with precise anchors
- Model connection and pros/cons sections
- Behavior-category failure summaries
- Structured system-design note editor where applicable
- Separate implementation, design-note, and AI-review statuses
- Explicit AI Help modes and code-sharing choice
- Optional `Review my design` action

Legacy exercises retain the current single-hint display until migrated.

## Compatibility and Migration

- Existing task IDs and path IDs remain unchanged.
- Existing progress records remain valid.
- Legacy `hint` and `hint_zh` fields remain readable.
- New `hints` metadata is additive.
- The existing LLM Frontier Architectures path remains available as an overview.
- New paths may reuse existing exercises through advisory prerequisites without duplicating IDs.
- Version-aware progress requires a backward-compatible database migration with existing records treated as contract version 1.

## Error Handling

- Invalid task metadata fails export/build with a precise field error.
- Invalid hint levels or duplicate levels fail validation.
- Agent scenarios reject unknown tools, malformed messages, invalid failure schedules, and non-deterministic clocks.
- Evaluator failures distinguish learner failures from harness failures.
- Harness failures do not mark a learner submission incorrect.
- AI endpoint errors remain optional UI errors and never affect deterministic status.
- Rate limits, authentication failures, malformed AI responses, and unavailable providers receive distinct messages.

## Verification Strategy

The vertical slice is verified through:

- Python unit tests for metadata, tensor harness, agent harness, and database migration
- Reference-solution evaluator runs for all 12 exercises
- Mutation suites for all 12 exercises
- Repeated seeded runs to detect nondeterminism
- TypeScript tests for metadata conversion and UI state
- API tests for hint retrieval, behavior results, design-note submission, and AI request minimization
- Frontend interaction tests for independent hint disclosure and separate statuses
- Build/export verification for generated problem and solution data
- Manual browser smoke test on the configured frontend and backend ports

## Git and Repository Workflow

The clone currently points `origin` at the original project. Before implementation:

1. Rename the current remote to `upstream`.
2. Add `https://github.com/meiru-cam/pyre-code-extend.git` as `origin`.
3. Fetch both remotes and verify `main` tracks the intended base.
4. Create a feature branch for the vertical slice.
5. Keep unrelated local changes, including existing `.gitignore` and PM2 configuration work, out of curriculum commits unless explicitly included.
6. Commit in reviewable units: schema/harness, UI, each path slice, AI review, and documentation.
7. Push feature branches to the user's `origin` and open pull requests against the desired repository.
8. Periodically fetch and rebase or merge from `upstream` with explicit conflict review.

No commit message includes `Co-Authored-By` lines.

Local environment files remain ignored and protected by the repository-managed pre-commit guard. The real `web/.env` is never staged or pushed.

## Delivery Sequence

1. Repository remote and feature-branch setup
2. Task schema, validation, versioning, and export compatibility
3. Two-level hint UI and APIs
4. Behavior-aware evaluator result model
5. Tensor harness and author-quality test helpers
6. Agent harness with scripted dependencies and virtual time
7. Four primitive exercises
8. Four subsystem exercises
9. Four integrative exercises
10. Design-note UI and persistence following current conventions
11. Optional DeepSeek design review and AI data-boundary hardening
12. Full mutation, integration, export, and browser verification
13. Subsequent pull requests expanding each path to its full sequence

## Deferred Work

- Persisting AI review history
- Live provider/tool integration exercises
- GPU performance benchmarks
- Fully automated scoring of free-form system-design reasoning
- Completing all approximately 50 exercises in the first pull request
