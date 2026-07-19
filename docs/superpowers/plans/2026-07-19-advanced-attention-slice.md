# Advanced Attention Vertical Slice Implementation Plan (Plan 2 of 6)

> Required skills: `superpowers:executing-plans`, `superpowers:test-driven-development`, `superpowers:verification-before-completion`, and `superpowers:finishing-a-development-branch`.

**Goal:** Add a reusable tensor-evaluation harness and three offline, mutation-tested Advanced Attention exercises: QK normalization, a local/global layer schedule, and a configurable frontier attention block.

**Base:** Start an isolated `feature/advanced-attention-slice` worktree from merged `origin/main`. Never read or stage `web/.env`.

**Learner contracts:**

- `qk_norm(q, k, eps=1e-6) -> tuple[Tensor, Tensor]`: RMS-normalize Q and K independently over the final/head dimension, preserving shape, dtype, device, and gradients.
- `hybrid_attention_schedule(Q, K, V, layer_index, window_size, global_every) -> Tensor`: causal scaled-dot-product attention; every `global_every`-th layer using one-based layer numbering is global, other layers can see only the current token and at most `window_size` preceding tokens.
- `FrontierAttentionBlock(d_model, num_heads, num_kv_heads, layer_index, window_size, global_every, use_qk_norm=True, eps=1e-6)`: an `nn.Module` with bias-free `q_proj`, `k_proj`, `v_proj`, and `o_proj`; GQA repeats KV heads, optional per-head QK RMS normalization precedes attention, and the same causal local/global rule applies.

## Task 1: Shared tensor and mutation harness

**Files:** create `torch_judge/harness/__init__.py`, `torch_judge/harness/tensor/{__init__,assertions,differential,gradients}.py`, `tests/quality/{__init__,mutation_runner}.py`; test `tests/test_tensor_harness.py` and `tests/quality/test_mutation_runner.py`.

1. RED: test `assert_tensor_close` rejects shape/dtype/value differences; `assert_finite` rejects NaN/Inf; `DifferentialCase` plus `assert_differential` compares independent callables without importing a task solution; `assert_gradients_reach` checks finite non-zero gradients for named tensors; `assert_mutations_rejected(task_id, mutations)` proves the reference solution passes all cases and every named mutation fails at least one required behavior.
2. Run `.venv/bin/python -m pytest tests/test_tensor_harness.py tests/quality/test_mutation_runner.py -v` and confirm import/contract failures.
3. GREEN: implement only the tested helpers. A grading crash is raised as `HarnessFailure`; it is never counted as a learner failure.
4. Refactor docstrings and rerun the focused tests.

## Task 2: QK normalization exercise

**Files:** create `torch_judge/tasks/qk_norm.py`; modify `web/src/lib/starters.json`; create/extend `tests/quality/test_attention_mutations.py`.

1. RED metadata/quality tests require version 1, two independent hints (level 1 questions; level 2 analysis), visible and unshown cases, model connections, pros/cons, and precise sources. Reference output uses `torch.nn.functional.rms_norm` when available; the repository's pinned Torch 2.2 baseline instead computes RMS through the independent `torch.linalg.vector_norm / sqrt(head_dim)` identity.
2. Required behaviors: shape, dtype/device, finite zero input, randomized differential agreement, independent Q/K normalization, scale metamorphism, and gradient flow.
3. Named mutations that must be rejected: `l2_norm`, `normalize_dim_zero`, `shared_q_statistics`, `missing_eps`, and `detach_inputs`.
4. GREEN task content must connect Qwen3 and Gemma 3 QK norm to exact pinned code and explain stability versus extra normalization cost. Hints may not contain a complete function.
5. Run `.venv/bin/python -m pytest tests/quality/test_attention_mutations.py -k qk_norm -v`.

## Task 3: Hybrid local/global attention exercise

**Files:** create `torch_judge/tasks/hybrid_attention_schedule.py`; modify starter and quality test files.

1. RED cases use hand-calculated uniform-logit examples and an independent PyTorch mask oracle.
2. Required behaviors: output shape/dtype/device, causal masking, local boundary behavior, one-based global-layer selection, `window_size=0`, `global_every=1`, gradients, and deterministic randomized differential cases.
3. Reject mutations `always_global`, `always_local`, `future_visible`, `symmetric_window`, and `off_by_one_schedule`.
4. GREEN description connects Gemma 2/3 interleaving, Qwen3 layer types, and Mistral sliding-window tradeoffs; references are optional reading and point to exact source symbols/sections.
5. Run `.venv/bin/python -m pytest tests/quality/test_attention_mutations.py -k hybrid -v`.

## Task 4: FrontierAttentionBlock capstone

**Files:** create `torch_judge/tasks/frontier_attention_block.py`; modify starter and quality test files.

1. RED checks constructor validation (`d_model % num_heads == 0`, `num_heads % num_kv_heads == 0`, positive schedule arguments), projection shapes, numerical agreement after copying weights into an independent oracle, GQA head sharing, QK-norm toggle, local/global masking, finite gradients to input and every projection, and seeded repeatability.
2. Reject `no_gqa_repeat`, `normalize_values`, `always_global`, `noncausal`, `wrong_scale`, `softmax_wrong_dim`, and `detached_kv`.
3. GREEN metadata describes profiles and limits: Llama GQA, Mistral local attention, Gemma hybrid schedule/QK norm, Qwen3 GQA/QK norm, and Kimi/GLM MLA as a materially different branch deferred to the existing MLA prerequisite rather than falsely represented by this block.
4. Include memory/compute/quality/stability/implementation pros and cons and exact simplifications versus production kernels, RoPE, caches, tensor parallelism, and dropout.
5. Run `.venv/bin/python -m pytest tests/quality/test_attention_mutations.py -k frontier -v`.

## Task 5: Path, exports, determinism, and branch integration

**Files:** modify `web/src/lib/paths.json`, generated `problems.json` and `solutions.json`; test `tests/test_advanced_attention_path.py` and existing exporter suites.

1. RED path test requires `advanced-attention` as a separate path containing advisory foundations `gqa`, `sliding_window`, and `mla`, plus the three new exercises in progression order; existing `llm-frontiers` remains unchanged.
2. GREEN add English-first path copy and non-locking prerequisites, then run `npm run build:problems`, `npm run build:solutions`, and update starters.
3. Run the quality suite three times with fixed seeds; all reference solutions must pass and all named mutations must be rejected each run.
4. Verify `.venv/bin/python -m pytest -q`, `npm run build:problems`, `npm run build:solutions`, `cd web && npx tsc --noEmit && npm test -- --run && npm run build`.
5. Confirm `git status --short` contains no env file or dependency symlink, inspect `git diff --check`, commit without co-author trailers, push, open a PR to `main`, review, fix findings with TDD, merge, and verify `origin/main` contains the merge.
