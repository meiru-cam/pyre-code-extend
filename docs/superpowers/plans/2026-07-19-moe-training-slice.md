# MoE Architecture and Training Vertical Slice (Plan 3 of 6)

**Goal:** Add a separate `moe-training` path with a correct top-k router, capacity-aware sparse dispatch/gather, and a complete differentiable TinyMoE training step.

**Base and safety:** Branch `feature/moe-training-slice` from the merged Plan 2 `origin/main` in a new ignored worktree. Offline deterministic grading only; never read/stage `web/.env`.

## Task 1: MoE oracle and evaluator helpers

**Files:** extend `torch_judge/harness/tensor/`; create `tests/test_moe_harness.py`, `tests/quality/test_moe_mutations.py`.

- RED then GREEN helpers for dense dispatch oracles, token/expert invocation accounting, selected/unselected gradient assertions, expert-permutation transforms, and seeded tiny-domain fixtures.
- Harness code must not import task solutions. Harness exceptions are distinct from learner assertion failures.

## Task 2: `moe_topk_router`

**Contract:** `moe_topk_router(router_logits, k) -> (expert_indices, expert_weights)`. Inputs are `[tokens, experts]`; softmax is over experts before top-k; returned tensors are `[tokens, k]`; selected weights are renormalized to sum to one. Validate `1 <= k <= experts`.

- Cases: hand calculation, ties with deterministic PyTorch `topk` semantics, shape/dtype/device, randomized oracle, normalization, gradients through selected weights, expert permutation.
- Reject: wrong softmax dimension, top-k before softmax without renormalization, smallest-k, detached weights, shared route for all tokens.
- Connect DeepSeek-V3/Kimi K2/GLM-4.5/Qwen3-MoE router code and precise report sections; distinguish selection probabilities from auxiliary load-balancing objectives.

## Task 3: `moe_capacity_dispatch`

**Contract:** `moe_capacity_dispatch(x, expert_indices, expert_weights, experts, capacity) -> (output, stats)`. Route in token-major then slot order; each expert accepts at most `capacity`; overflow assignments are dropped; accepted weighted outputs are accumulated back to exactly one token row; `stats` contains integer `accepted`, `dropped`, and per-expert `loads`.

- Cases: dense-oracle equivalence, token conservation, no duplication, capacity boundaries including zero, sparse invocation counts, dtype/device, gradients only through selected accepted paths, expert permutation.
- Reject: no capacity, global instead of per-expert capacity, duplicated gather, unweighted gather, execute-all-experts, count shared expert as routed.
- Explain dropless versus dropping/padding, capacity-factor utilization, communication regularity, and quality loss.

## Task 4: `tiny_moe_train_step`

**Contract:** `tiny_moe_train_step(model, optimizer, x, targets, balance_weight=0.01, z_weight=0.001) -> dict`. The supplied model returns `(logits, router_logits)`; compute cross entropy plus differentiable load-balance and router z-loss, zero gradients, backpropagate once, step once, and return detached scalar metrics `loss`, `task_loss`, `balance_loss`, `z_loss`.

- Cases: hand-computed losses, exact parameter delta versus an independent PyTorch step, router/expert gradient reachability, zeroing between steps, seeded synthetic-domain loss/routing signals, deterministic repeated run.
- Reject: task loss only, detached auxiliary losses, wrong z-loss reduction, missing zero-grad, double step, loss-only-without-step.
- Teach how routing is trained: task gradients through selected gates, auxiliary load balancing, z-loss stability, noisy/exploratory routing, expert specialization signals, and why decreasing loss alone is insufficient evidence.

## Task 5: Path, mutation gate, and merge

- Add `moe-training` path in progression with advisory existing `moe` and `moe_load_balance`; do not modify `llm-frontiers`.
- Add exact two-level hints, model connections, pros/cons, commit-pinned code provenance, precise paper sections, starters, and generated catalogs.
- Require reference pass plus rejection of every named mutation in three seeded runs.
- Full Python/export/TypeScript/Vitest/Next-build verification, review, fixes, PR, and merge before Plan 4.
