"""ACT temporal ensembling: average every chunk's prediction for the current step, online."""

from ._interview import interview

TASK = {
    "title": "ACT Online Temporal Ensembling",
    "difficulty": "Medium",
    "version": 1,
    "function_name": "TemporalEnsembler",
    "description_en": r"""A policy that predicts action chunks is queried at every control step. Each query returns the next `chunk_size` actions, so the chunks overlap and several predictions cover the same step. Implement the online temporal ensemble from ACT, which executes their exponentially weighted average.

**Signature:** `TemporalEnsembler(coeff, chunk_size)` with methods `reset()` and `update(actions) -> Tensor`.

**Parameters:**
- `coeff` — float `m` in the weights `w_i = exp(-m * i)`. It may be zero or negative.
- `chunk_size` — integer `K >= 1`.
- `actions` — floating tensor of shape `(B, K, D)`. `actions[b, j]` is the action for the step `j` steps after the current one.

**Returns:** `update` returns a `(B, D)` tensor with the same dtype and device as `actions`: the ensembled action for the current step.

**The weighted average.** Number the calls since construction or the last `reset()` as steps `t = 0, 1, 2, ...`. The chunk from call `s` predicts step `t` when `s <= t <= s + K - 1`. Sort the chunks that predict step `t` from oldest to newest and give the `i`-th one the weight `w_i`, so `w_0` belongs to the oldest. The output at step `t` is

    sum_i w_i * prediction_i  /  sum_i w_i

**Constraints:**
- Compute the average online: keep one running average and one count per future step, not a history of past chunks.
- Every prediction counts, including actions that are exactly zero.
- Each batch row is ensembled independently.
- `reset()` forgets all history, so the next `update` behaves like the first one.
- Do not modify `actions` in place.
- Build the weights in the dtype of `actions`; float64 inputs are checked to about 1e-10.
- Raise `ValueError` when `chunk_size` is not an integer `>= 1`, when `actions` is not a 3-D floating tensor with `actions.shape[1] == chunk_size`, or when `B` or `D` changes without a `reset()`.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why chunks overlap.** ACT predicts `K` future actions at once to shorten the effective horizon and reduce compounding errors. Executing a chunk open-loop for `K` steps makes motion jerky at chunk boundaries, so ACT queries the policy every step instead and averages the overlapping predictions.

**The running average.** If `S_n = w_0 + ... + w_(n-1)` and `avg_n` is the average of the first `n` predictions for a step, then adding prediction `n` gives `avg_(n+1) = (avg_n * S_n + w_n * x_n) / S_(n+1)`. Keeping a count per step is enough to pick `w_n`, `S_n` and `S_(n+1)`. After each call, the entry for the current step is returned and dropped, and the newest chunk's last action starts a new entry with count 1.

**Choosing coeff.** With `m > 0` older predictions weigh more and the output is smoother; with `m < 0` newer predictions weigh more and the policy reacts faster; `m = 0` is a plain mean. ACT uses `m = 0.01`.

**Contrast with chunk scheduling.** The `embodied_action_stream` exercise lets a newer chunk replace an older plan. Temporal ensembling keeps every plan and averages them.""",
    "advisory_prerequisites": ["embodied_action_stream"],
    "hints": [
        {"level": 1, "kind": "questions", "content": "At call t, which earlier chunks predict step t, and which index inside each chunk holds that prediction? If you know the running average of n predictions and the sum of their weights, what do you need to fold in prediction n? Why is a count per future step enough to recover both weight sums?"},
        {"level": 2, "kind": "analysis", "content": "Precompute the weights and their cumulative sums once. Keep a running average and a count for each of the next K - 1 steps. Each call folds the overlapping part of the new chunk into those averages with the running-average identity from the background, opens a new entry for the chunk's last step, then returns and drops the oldest entry. Watch the off-by-one between a count and the weight index it selects, and the first call, which has no history."},
    ],
    "model_connections": [
        "ACT (Zhao et al. 2023, ALOHA) queries its policy every step and executes the exponentially weighted ensemble of overlapping chunks.",
        "LeRobot's ACTPolicy implements the online form as ACTTemporalEnsembler; ACT's original code computes the same average offline from a buffer of every past chunk.",
        "Later chunked policies (pi0, SmolVLA) execute chunks with asynchronous inference or real-time chunking instead of a per-step ensemble.",
    ],
    "pro_con_analysis": {
        "pros": ["Smooth motion with no jumps at chunk boundaries, constant memory per step, and no extra training cost."],
        "cons": ["Needs a policy query every step, and averaging predictions from different modes can produce an action that belongs to neither."],
    },
    "sources": [
        {
            "kind": "code",
            "url": "https://github.com/huggingface/lerobot/blob/75cd2ba0ce9ff47686bb2bd7b7f65840606c02ee/src/lerobot/policies/act/modeling_act.py",
            "commit": "75cd2ba0ce9ff47686bb2bd7b7f65840606c02ee",
            "path": "src/lerobot/policies/act/modeling_act.py",
            "symbol": "ACTTemporalEnsembler",
            "license": "Apache-2.0",
            "adapted": "Same online update: running weighted average per future step, weights indexed by a per-step count, pop the current step.",
            "simplifications": "Counts are kept as a 1-D tensor and never clamped, since they cannot exceed chunk_size - 1 before a merge; "
                               "weights are built in the actions dtype instead of float32; shape and batch changes raise ValueError.",
        },
        {
            "kind": "code",
            "url": "https://github.com/tonyzhaozh/act/blob/742c753c0d4a5d87076c8f69e5628c79a8cc5488/imitate_episodes.py",
            "commit": "742c753c0d4a5d87076c8f69e5628c79a8cc5488",
            "path": "imitate_episodes.py",
            "symbol": "eval_bc (temporal_agg branch, lines 250-259)",
            "license": "MIT",
            "adapted": "The offline definition the online update must match: weight w_i = exp(-k * i) over the predictions for step t, oldest first.",
            "simplifications": "Divergence: the original finds populated rows with torch.all(actions != 0, axis=1), so a real prediction containing a zero "
                               "entry is dropped from the average. The exercise counts every prediction, as LeRobot does. The original also keeps a "
                               "(T, T + K, D) buffer of all chunks; the exercise is online with O(K) state.",
        },
        {
            "kind": "paper",
            "url": "https://arxiv.org/abs/2304.13705",
            "section": "Zhao et al. 2023, Section IV-A Action Chunking and Temporal Ensemble, Figure 4 and Algorithm 2",
            "note": "Defines w_i = exp(-m * i) with w_0 the weight of the oldest action.",
        },
    ],
    "tests": [
        {"name": "Uniform weights on overlapping chunks", "behavior": "metrics.averaging", "code": r"""
import torch
e = {fn}(0.0, 3)
c0 = torch.tensor([[[1.0], [2.0], [3.0]]])
c1 = torch.tensor([[[5.0], [6.0], [7.0]]])
c2 = torch.tensor([[[9.0], [10.0], [11.0]]])
out0 = e.update(c0)
assert out0.shape == (1, 1) and torch.allclose(out0, torch.tensor([[1.0]])), out0
out1 = e.update(c1)  # step 1 is predicted by c0[1] = 2 and c1[0] = 5
assert torch.allclose(out1, torch.tensor([[3.5]])), out1
out2 = e.update(c2)  # step 2: c0[2] = 3, c1[1] = 6, c2[0] = 9
assert torch.allclose(out2, torch.tensor([[6.0]])), out2
"""},
        {"name": "Exponential weights favour the oldest prediction", "behavior": "metrics.averaging", "code": r"""
import math, torch
m = math.log(2.0)  # w_0 = 1, w_1 = 0.5
e = {fn}(m, 2)
e.update(torch.tensor([[[0.0, 1.0], [4.0, 2.0]]]))
out = e.update(torch.tensor([[[10.0, -2.0], [0.0, 0.0]]]))
want = torch.tensor([[(1.0 * 4.0 + 0.5 * 10.0) / 1.5, (1.0 * 2.0 + 0.5 * -2.0) / 1.5]])
assert torch.allclose(out, want), (out, want)
"""},
        {"name": "Online output equals the offline weighted average", "visibility": "unshown", "behavior": "metrics.averaging", "failure_message": "At every step, weight the predictions for that step by exp(-coeff * i) with i = 0 for the oldest chunk, and divide by the sum of those weights.", "code": r"""
import math, torch
def offline(chunks, m, K):
    outs = []
    for t in range(len(chunks)):
        num, den = 0.0, 0.0
        for i, s in enumerate(range(max(0, t - K + 1), t + 1)):
            w = math.exp(-m * i)
            num = num + w * chunks[s][:, t - s]
            den += w
        outs.append(num / den)
    return outs
for seed, m, K, B, D, T in ((0, 0.01, 4, 2, 3, 13), (1, 0.7, 5, 3, 2, 17), (2, -0.4, 3, 1, 4, 11), (3, 1.3, 2, 2, 1, 9), (4, 0.25, 6, 2, 2, 20)):
    g = torch.Generator().manual_seed(seed)
    chunks = [torch.randn(B, K, D, generator=g, dtype=torch.float64) for _ in range(T)]
    e = {fn}(m, K)
    for t, (got, want) in enumerate(zip([e.update(c) for c in chunks], offline(chunks, m, K))):
        assert got.shape == (B, D), got.shape
        assert torch.allclose(got, want, atol=1e-10, rtol=0), (seed, t, got, want)
"""},
        {"name": "Weights saturate once chunk_size chunks overlap", "visibility": "unshown", "behavior": "state.invariant", "failure_message": "In steady state each step has exactly chunk_size predictions weighted w_0 (oldest) to w_(K-1) (newest); index weights by the per-step count, not by time.", "code": r"""
import math, torch
m = math.log(3.0)  # w = 1, 1/3, 1/9
e = {fn}(m, 3)
for t in range(12):
    out = e.update(torch.full((1, 3, 1), float(t), dtype=torch.float64) + torch.tensor([0.0, 100.0, 200.0], dtype=torch.float64).view(1, 3, 1))
# step 11 is predicted by chunk 9 (offset 2), chunk 10 (offset 1) and chunk 11 (offset 0)
preds = [9.0 + 200.0, 10.0 + 100.0, 11.0]
want = (preds[0] * 1 + preds[1] / 3 + preds[2] / 9) / (1 + 1 / 3 + 1 / 9)
assert abs(out.item() - want) < 1e-9, (out, want)
"""},
        {"name": "Negative coeff weights newer predictions more", "visibility": "unshown", "behavior": "metrics.averaging", "failure_message": "w_i = exp(-coeff * i) with i counting from the oldest prediction, so a negative coeff gives the newest prediction the largest weight.", "code": r"""
import math, torch
m = -math.log(4.0)  # w_0 = 1, w_1 = 4
e = {fn}(m, 2)
e.update(torch.tensor([[[0.0], [1.0]]], dtype=torch.float64))
out = e.update(torch.tensor([[[6.0], [0.0]]], dtype=torch.float64))
assert abs(out.item() - (1.0 * 1.0 + 4.0 * 6.0) / 5.0) < 1e-12, out
"""},
        {"name": "Exact-zero actions are real predictions", "visibility": "unshown", "behavior": "edge.empty_or_boundary", "failure_message": "Do not treat zeros as missing predictions; an action of exactly 0 is averaged like any other value.", "code": r"""
import torch
e = {fn}(0.0, 3)
e.update(torch.tensor([[[1.0, 1.0], [0.0, 2.0], [0.0, 0.0]]]))
out1 = e.update(torch.tensor([[[4.0, 4.0], [6.0, 0.0], [3.0, 3.0]]]))
assert torch.allclose(out1, torch.tensor([[2.0, 3.0]])), out1
out2 = e.update(torch.tensor([[[3.0, 3.0], [1.0, 1.0], [1.0, 1.0]]]))
assert torch.allclose(out2, torch.tensor([[3.0, 1.0]])), out2
"""},
        {"name": "Batch rows are ensembled independently", "visibility": "unshown", "behavior": "state.invariant", "failure_message": "Keep a separate running average per batch row; share only the per-step counts.", "code": r"""
import torch
g = torch.Generator().manual_seed(7)
chunks = [torch.randn(3, 4, 2, generator=g, dtype=torch.float64) for _ in range(9)]
joint = {fn}(0.3, 4)
singles = [{fn}(0.3, 4) for _ in range(3)]
for c in chunks:
    out = joint.update(c)
    for b in range(3):
        assert torch.allclose(out[b:b + 1], singles[b].update(c[b:b + 1]), atol=1e-12), b
"""},
        {"name": "reset forgets all history", "visibility": "unshown", "behavior": "state.invariant", "failure_message": "reset() must clear the running averages and counts, so the next update returns that chunk's first action.", "code": r"""
import torch
g = torch.Generator().manual_seed(11)
e = {fn}(0.1, 3)
for _ in range(5):
    e.update(torch.randn(2, 3, 2, generator=g, dtype=torch.float64))
e.reset()
fresh = {fn}(0.1, 3)
new = [torch.randn(2, 3, 2, generator=g, dtype=torch.float64) for _ in range(4)]
for c in new:
    assert torch.allclose(e.update(c), fresh.update(c), atol=1e-12)
e.reset()
c = torch.randn(4, 3, 1, generator=g, dtype=torch.float64)
assert torch.equal(e.update(c), c[:, 0])
"""},
        {"name": "chunk_size 1 executes each new action", "visibility": "unshown", "behavior": "edge.empty_or_boundary", "failure_message": "With chunk_size 1 no chunks overlap, so every update returns actions[:, 0].", "code": r"""
import torch
g = torch.Generator().manual_seed(3)
e = {fn}(0.5, 1)
for _ in range(4):
    c = torch.randn(2, 1, 3, generator=g)
    out = e.update(c)
    assert out.shape == (2, 3) and torch.equal(out, c[:, 0]), out
"""},
        {"name": "Keeps dtype and does not mutate the input", "visibility": "unshown", "behavior": "tensor.dtype_device", "failure_message": "Return the input dtype and never modify the caller's actions tensor in place (clone what you store).", "code": r"""
import torch
g = torch.Generator().manual_seed(5)
e = {fn}(0.2, 3)
chunks = [torch.randn(2, 3, 2, generator=g, dtype=torch.float64) for _ in range(5)]
copies = [c.clone() for c in chunks]
for c in chunks:
    out = e.update(c)
    assert out.dtype == torch.float64, out.dtype
for c, k in zip(chunks, copies):
    assert torch.equal(c, k), "update modified its input"
"""},
        {"name": "Rejects bad shapes and arguments", "visibility": "unshown", "behavior": "contract.signature", "failure_message": "Raise ValueError for a non-integer or non-positive chunk_size, a wrong actions shape or dtype, or a batch/action-dim change without reset().", "code": r"""
import torch
for k in (0, -1, 2.5):
    try:
        {fn}(0.1, k)
    except ValueError:
        pass
    else:
        raise AssertionError(f"accepted chunk_size={k}")
for bad in (torch.zeros(3, 2), torch.zeros(1, 4, 2), torch.zeros(1, 2, 2), torch.zeros(1, 3, 2, dtype=torch.long)):
    try:
        {fn}(0.1, 3).update(bad)
    except ValueError:
        pass
    else:
        raise AssertionError(f"accepted actions {tuple(bad.shape)} {bad.dtype}")
for second in (torch.zeros(2, 3, 2), torch.zeros(1, 3, 3)):
    e = {fn}(0.1, 3)
    e.update(torch.zeros(1, 3, 2))
    try:
        e.update(second)
    except ValueError:
        pass
    else:
        raise AssertionError(f"accepted shape change to {tuple(second.shape)}")
"""},
    ],
    "solution": '''import torch


class TemporalEnsembler:
    def __init__(self, coeff, chunk_size):
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
            raise ValueError("chunk_size must be an integer >= 1")
        self.coeff = float(coeff)
        self.chunk_size = chunk_size
        self.reset()

    def reset(self):
        self.avg = None    # (B, K - 1, D) running average for each future step
        self.count = None  # (K - 1,) predictions merged into each future step

    def update(self, actions):
        K = self.chunk_size
        if not torch.is_tensor(actions) or actions.ndim != 3 or actions.shape[1] != K:
            raise ValueError("actions must have shape (B, chunk_size, D)")
        if not actions.is_floating_point():
            raise ValueError("actions must be a floating tensor")
        w = torch.exp(-self.coeff * torch.arange(K, dtype=actions.dtype, device=actions.device))
        cum = torch.cumsum(w, dim=0)
        if self.avg is None:
            avg = actions.clone()
            count = torch.ones(K, dtype=torch.long, device=actions.device)
        else:
            if self.avg.shape[0] != actions.shape[0] or self.avg.shape[2] != actions.shape[2]:
                raise ValueError("batch size or action dim changed without reset()")
            n = self.count
            merged = (self.avg * cum[n - 1].view(1, -1, 1) + actions[:, :-1] * w[n].view(1, -1, 1)) / cum[n].view(1, -1, 1)
            avg = torch.cat([merged, actions[:, -1:]], dim=1)
            count = torch.cat([n + 1, n.new_ones(1)])
        self.avg, self.count = avg[:, 1:], count[1:]
        return avg[:, 0]
''',
    "interview_questions": interview(
        concept=[
            "What problem does temporal ensembling solve when a policy predicts action chunks?",
            "In ACT's weights w_i = exp(-m * i), which prediction gets w_0, and what does the sign of m control?",
        ],
        deep_dive=[
            "Derive the online update that adds one prediction to a running weighted average without storing past chunks.",
            "Why is a per-step count enough state to pick the right weights, and why does it never exceed chunk_size?",
            "The original ACT code finds populated predictions with actions != 0. What goes wrong, and how does the online form avoid it?",
        ],
        tradeoffs=[
            "Temporal ensembling versus executing each chunk open-loop for K steps: smoothness, reactivity and compute?",
            "Temporal ensembling versus real-time chunking or asynchronous inference for slow, large VLA policies?",
        ],
    ),
}
