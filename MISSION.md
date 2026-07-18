# Mission: Production AI Systems Implementation Curriculum

## Why

Enable learners with basic PyTorch knowledge to move from recognizing modern AI architecture names to implementing, testing, and reasoning about the mechanisms used in frontier attention, MoE models, agent runtimes, and agent guardrails.

## Success looks like

- Implement attention and MoE components and explain their architectural tradeoffs.
- Connect minimal implementations to pinned code from real models such as Kimi, GLM, Qwen, Llama, Gemma, and Mistral.
- Build an offline agent runtime that handles communication, workload control, API failures, budgets, checkpointing, and recovery.
- Secure that runtime with testable guardrails and explain the associated safety and usability tradeoffs.
- Validate implementations through deterministic tests, independent comparisons, invariants, gradients, failure injection, and mutation testing.
- Require every new exercise to satisfy the author-quality gate before merge.

## Constraints

- Require only basic PyTorch knowledge and no GPU.
- Keep prerequisites advisory: recommend preparation and readiness checks without locking content.
- Combine visible examples with unshown evaluator cases and behavior-level failure diagnostics.
- Keep exercises fully usable offline without model-provider API keys.
- Use independently revealed Level 1 question hints and Level 2 analytical hints.
- Prefer short feedback loops and progressive implementation over passive explanation.
- Validate the shared curriculum infrastructure through a vertical-slice release before expanding all four paths to their full exercise counts.
- Link to pinned upstream implementations without copying substantial production code.
- Preserve reproducibility through immutable source pins and explicit exercise revisions when contracts or architectural behavior change.
- Preserve learner attempts and completion against the exercise revision they actually used.
- Follow existing persistence conventions in the first release: browser-local drafts, SQLite submissions and progress, and transient optional AI feedback.
- Grade executable system-design decisions deterministically; review written tradeoff analysis with a visible rubric and optional AI feedback.
- Recommend DeepSeek V4 Pro for optional design review without making any provider a curriculum dependency.
- Track deterministic implementation, design-note completion, and optional AI review as separate outcomes.
- Require explicit learner queries and consent before sending exercise context, code, or design notes to an external AI provider.

## Out of scope

- Reproducing or pretraining full frontier-scale models.
- Requiring live commercial model APIs for grading.
- Teaching introductory Python or all of PyTorch inside the advanced paths.
- Treating a passing reference implementation as sufficient evidence that a grader is correct.
