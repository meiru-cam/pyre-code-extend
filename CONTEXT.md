# Pyre Code Curriculum Glossary

This glossary defines the curriculum language used when designing and reviewing Pyre Code learning experiences.

## Terms

**Curriculum**:
The reusable collection of learning paths, exercises, hints, references, and assessments offered to every learner.
_Avoid_: Personal study plan, lesson archive

**Learning path**:
An ordered sequence of exercises that develops one coherent implementation capability from prerequisites through a capstone.
_Avoid_: Category, tag, problem list

**Entry baseline**:
The minimum knowledge assumed before a learner begins a learning path. For the advanced paths, this means basic PyTorch tensor operations, modules, gradients, and ordinary multi-head attention—not prior knowledge of frontier architectures.
_Avoid_: Beginner course, PyTorch benchmark

**Advisory prerequisite**:
A recommended earlier exercise or capability that helps a learner succeed but never prevents access to a learning path or exercise.
_Avoid_: Gate, required unlock

**Exercise**:
A code implementation task with a stated contract, independently revealable hints, deterministic checks, and a reference implementation.
_Avoid_: Lesson, demo

**Visible example**:
An evaluator case shown in the exercise description to clarify the contract and expected behavior.
_Avoid_: Complete test suite

**Unshown evaluator case**:
An evaluator case omitted from the normal exercise UI so that learners must implement the general rule. It remains inspectable in the open-source repository and is not treated as secret.
_Avoid_: Hidden test, secret test

**Behavior category**:
A stable, author-assigned label identifying the capability tested by an evaluator case, used to provide actionable failure feedback without exposing its inputs or implementation.
_Avoid_: Inferred error type, traceback category

**Hint level**:
One independently revealed layer of assistance for an exercise. Level 1 contains guiding questions; Level 2 contains analytical guidance without disclosing the complete implementation.
_Avoid_: Hint round, solution step

**Evaluation harness**:
The reusable deterministic environment that supplies fixtures, simulated dependencies, failure schedules, traces, and assertions for grading an exercise.
_Avoid_: Test dataset

**Data harness**:
The fixture-loading, validation, and versioning component within an evaluation harness.
_Avoid_: Evaluation harness

**Reference implementation**:
The repository's correct, minimal implementation of an exercise contract, used for explanation and independent validation of the grader.
_Avoid_: Oracle

**Author-quality gate**:
The evidence required before an exercise is merged: independent comparison, seeded cases, applicable gradient checks, mutation rejection, correct behavior diagnostics, deterministic execution, and pinned provenance. Passing the reference implementation alone does not satisfy this gate.
_Avoid_: Reference-solution test, smoke test

**Executable criterion**:
A system-design decision expressed through code, schemas, or state transitions and graded with deterministic evaluator cases.
_Avoid_: Architectural opinion

**Design note**:
A short structured explanation of architectural choices, failure behavior, and tradeoffs, reviewed against a visible rubric rather than treated as objectively pass/fail.
_Avoid_: Essay, automatically graded answer

**AI design review**:
Optional model-generated feedback on a design note. It supplements the visible rubric and never determines whether an offline exercise passes.
_Avoid_: AI judge, grading authority

**Implementation status**:
The deterministic result of executing all required evaluator cases. An exercise is `Solved` only through this status.
_Avoid_: AI score, design quality score

**Design-note status**:
Whether the learner has supplied the structured architectural reasoning requested by an exercise, independently of implementation correctness.
_Avoid_: Solved status

**AI-review status**:
Whether optional model feedback has been requested and returned for a design note. It records `Not requested` or `Reviewed` without altering implementation status.
_Avoid_: Pass, fail

**AI Help query**:
An explicit learner-authored question sent with exercise context to an external model. Sharing the learner's current code is a separate opt-in choice.
_Avoid_: Automatic code upload, AI grading request

**AI Help mode**:
One of two explicit interactions: `Ask a question`, which sends a learner query and exercise context, or `Review my code`, which additionally sends the learner's current implementation after consent.
_Avoid_: Automatic tutor invocation

**Recommended AI reviewer**:
The preconfigured model suggested for optional design-note feedback. DeepSeek V4 Pro is the recommended preset, while the integration remains compatible with other OpenAI-style providers.
_Avoid_: Required model, grading dependency

**Model connection**:
A documented mapping between an exercise concept and a pinned implementation or architecture used by a real model or agent system.
_Avoid_: Model copy, production implementation

**Source pin**:
An immutable repository commit and source location defining the upstream behavior discussed by a model connection.
_Avoid_: Latest branch, floating link

**Exercise revision**:
An explicit successor to an exercise whose contract or pinned architectural behavior changed meaningfully. Upstream changes never silently redefine an existing revision.
_Avoid_: Silent test update, corrected completion

**Submission attempt**:
An immutable snapshot of a learner's code or design note submitted against one exercise revision. New learner responses create attempts, not exercise revisions.
_Avoid_: Exercise version

**Draft**:
The learner's unsent code or design note stored in browser-local storage for the current exercise.
_Avoid_: Submission attempt

**Transient AI feedback**:
Optional AI Help or design-review output held only in current UI state, matching the existing AI Help lifecycle. It is not written to submission history in the first release.
_Avoid_: Submission, durable review record

**Version-aware progress**:
Progress identified by both stable exercise ID and contract version. Completion of an earlier revision remains valid and visible when a newer revision becomes available.
_Avoid_: Reset progress, overwrite completion

**Capstone**:
The final integrative exercise in a learning path, requiring the learner to combine several previously implemented capabilities under realistic constraints.
_Avoid_: Final exam

**Vertical slice**:
A staged curriculum release containing shared infrastructure plus one primitive, one subsystem exercise, and one integrative exercise from each new learning path. It validates the full learner and author workflow before the paths are expanded.
_Avoid_: Incomplete prototype, full path
