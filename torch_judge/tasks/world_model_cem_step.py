"""One cross-entropy-method refit for planning with a world model, in the V-JEPA 2-AC and TD-MPC variants."""

from ._interview import interview

TASK = {
    "title": "World Model CEM Planning Step",
    "difficulty": "Medium",
    "version": 1,
    "function_name": "cem_refit",
    "description_en": r"""A planner that uses a world model samples `S` action trajectories from a Gaussian, scores each one by rolling it through the model, and refits the Gaussian to the best samples. Implement that refit step, the core of the cross-entropy method (CEM), in the two variants used by V-JEPA 2-AC and TD-MPC.

**Signature:** `cem_refit(samples, energies, mean, std, topk, momentum_mean, momentum_std, mode="vjepa2", temperature=0.5) -> (new_mean, new_std)`

**Parameters:**
- `samples` — floating tensor `(S, H, A)`: `S` trajectories of `H` steps with `A` action dims.
- `energies` — floating tensor `(S,)`, one score per sample. In `"vjepa2"` mode it is an energy (lower is better, e.g. the distance of the predicted final state to a goal). In `"tdmpc"` mode it is a value (higher is better).
- `mean`, `std` — floating tensors `(H, A)`: the Gaussian the samples were drawn from.
- `topk` — integer number of elite samples, `1 <= topk <= S`.
- `momentum_mean`, `momentum_std` — floats in `[0, 1]`.
- `mode` — `"vjepa2"` or `"tdmpc"`.
- `temperature` — float `>= 0`, used only in `"tdmpc"` mode.

**Returns:** a tuple `(new_mean, new_std)` of `(H, A)` tensors with the dtype of `samples`.

**Choosing the elites.**
- `"vjepa2"`: the `topk` samples with the **lowest** energies.
- `"tdmpc"`: the `topk` samples with the **highest** values.
- Ties are broken by sample index: among equal scores, the lower index ranks first.

**Elite statistics** (per step and action dim, over the elites):
- `"vjepa2"`: the plain mean `mu`, and the **unbiased** standard deviation, which divides by `topk - 1`.
- `"tdmpc"`: weights `w_i = exp(temperature * (v_i - max_v))`, where `max_v` is the largest elite value, normalised to sum to 1. Then

      mu    = sum_i w_i * x_i
      sigma = sqrt(sum_i w_i * (x_i - mu) ** 2)

**Momentum blend** (both modes):

      new_mean = mu    * (1 - momentum_mean) + mean * momentum_mean
      new_std  = sigma * (1 - momentum_std)  + std  * momentum_std

**Constraints:**
- `"vjepa2"` needs at least two elites: raise `ValueError` when `topk == 1` in that mode, since the unbiased std of one sample is undefined. `"tdmpc"` accepts `topk == 1` and gives that elite weight 1 and `sigma = 0`.
- Subtract `max_v` before exponentiating, so values in the thousands still give finite weights.
- Do not modify any input in place.
- float64 inputs are checked to about 1e-10, float32 inputs to about 1e-5.
- Raise `ValueError` for an unknown `mode`; for `samples` that is not a 3-D floating tensor; for `energies`, `mean` or `std` with the wrong shape or a dtype different from `samples`; for non-finite energies; for a `topk` that is not an integer in `1..S`; for a momentum outside `[0, 1]`; and for a negative `temperature`.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**The planning loop.** Start from a Gaussian over action trajectories, for example `mean = 0`, `std = 1`. Each iteration samples `S` trajectories, rolls each through the world model from the current state, scores the outcome, and calls this refit. After a few iterations the planner executes the first action of `mean` and replans at the next step (model predictive control).

**V-JEPA 2-AC.** The world model is an action-conditioned predictor in the latent space of a frozen video encoder. The energy of a trajectory is the L1 distance between the predicted final embedding and the embedding of a goal image: `mean(|z_H - z_goal|)` over the embedding dims. The refit keeps the closest `topk` and averages them uniformly. The released planner uses a high std momentum (0.95), so the search narrows slowly.

**TD-MPC.** The world model is a latent dynamics model with reward and Q heads; a trajectory's value is its discounted predicted reward plus a terminal Q estimate. The refit keeps the `topk` highest values and weights them with a softmax, so better elites pull the mean harder. Subtracting the maximum does not change the normalised weights but keeps `exp` from overflowing. LeRobot's TD-MPC blends the mean with momentum 0.1, replaces the std outright (`momentum_std = 0`) and clamps it to `[0.05, 2.0]`.

**Why momentum.** A refit that jumps straight to the elite statistics can collapse the std after one lucky batch. Blending with the previous Gaussian keeps exploration alive across iterations.""",
    "advisory_prerequisites": ["softmax", "topk_sampling", "rssm_kl_balance", "jepa_block_masks_and_loss"],
    "hints": [
        {"level": 1, "kind": "questions", "content": "Which direction is better in each mode, and how do you take the first topk samples in that order while keeping lower indices first among ties? Which divisor makes a standard deviation unbiased, and why does that rule out one elite? In the weighted variant, why does subtracting the largest elite value leave the normalised weights unchanged?"},
        {"level": 2, "kind": "analysis", "content": "Validate the arguments first. Rank the scores with a stable sort, ascending for energies and descending for values, and keep the first topk indices. Gather those elites. In vjepa2 mode take the plain mean and the std with correction 1; in tdmpc mode build the shifted exponential weights from the elite values only, normalise them, then take the weighted mean and the square root of the weighted squared deviations. Finally blend each statistic with its old value. Pitfalls: momentum weights the old value, not the new one; torch.topk does not promise an order among ties; std must not share a code path that divides by topk - 1 in tdmpc mode."},
    ],
    "model_connections": [
        "V-JEPA 2-AC (Assran et al. 2025) plans robot arm actions by CEM over a latent world model, scoring each trajectory by the L1 distance to a goal-image embedding.",
        "TD-MPC and TD-MPC2 (Hansen et al.) plan with MPPI-style CEM over a learned latent model, weighting elites by a softmax over predicted returns; LeRobot ships it as TDMPCPolicy.",
        "PlaNet and Dreamer's early planners, and many model-based RL baselines (PETS), use the same CEM refit with a uniform elite average.",
    ],
    "pro_con_analysis": {
        "pros": ["Gradient-free, so it works with any world model and any scoring function, and every sample in an iteration is scored in parallel."],
        "cons": ["Sample cost grows quickly with horizon times action dims, and a Gaussian proposal cannot represent several distinct good plans at once."],
    },
    "sources": [
        {
            "kind": "code",
            "url": "https://github.com/facebookresearch/vjepa2/blob/204698b45b3712590f06245fbfba32d3be539812/notebooks/utils/mpc_utils.py",
            "commit": "204698b45b3712590f06245fbfba32d3be539812",
            "path": "notebooks/utils/mpc_utils.py",
            "symbol": "cem (line 28), select_topk_action_traj and the refit lines 127-150",
            "license": "MIT",
            "adapted": "The vjepa2 mode: top-k lowest objective, unweighted mean, Tensor.std (unbiased), and the momentum blend new * (1 - m) + old * m for both mean and std.",
            "simplifications": "Divergences: upstream uses separate momenta for the gripper dimension and a 7-dim action whose rotation dims are fixed at zero; "
                               "the exercise uses one momentum pair for all dims. Upstream torch.topk leaves the order of tied samples unspecified; "
                               "the exercise breaks ties by lower index. Upstream topk = 1 produces a NaN std; the exercise raises ValueError. "
                               "Sampling, clipping and the world-model rollout stay outside the function, which receives the energies.",
        },
        {
            "kind": "code",
            "url": "https://github.com/huggingface/lerobot/blob/75cd2ba0ce9ff47686bb2bd7b7f65840606c02ee/src/lerobot/policies/tdmpc/modeling_tdmpc.py",
            "commit": "75cd2ba0ce9ff47686bb2bd7b7f65840606c02ee",
            "path": "src/lerobot/policies/tdmpc/modeling_tdmpc.py",
            "symbol": "TDMPCPolicy.plan (line 167), elite selection and gaussian refit lines 225-251",
            "license": "Apache-2.0",
            "adapted": "The tdmpc mode: top-k highest value, softmax weights exp(temperature * (v - max v)) over the elites, weighted mean and biased weighted std, and the mean EMA momentum * old + (1 - momentum) * new.",
            "simplifications": "Divergences: upstream replaces the std outright and clamps it to [min_std, max_std]; the exercise blends it with momentum_std "
                               "(0 reproduces the replacement) and does not clamp. Upstream maps NaN values to 0; the exercise rejects non-finite scores. "
                               "Upstream plans a batch of environments at once with layout (horizon, n, batch, action_dim); the exercise plans one "
                               "with layout (S, H, A). Tie order follows lower index instead of torch.topk.",
        },
        {
            "kind": "paper",
            "url": "https://arxiv.org/abs/2506.09985",
            "section": "Assran et al. 2025, V-JEPA 2, Section 4 (planning with V-JEPA 2-AC: energy minimisation with the cross-entropy method)",
            "note": "Defines the goal-conditioned energy as the L1 distance between predicted and goal embeddings and optimises it with CEM.",
        },
        {
            "kind": "paper",
            "url": "https://arxiv.org/abs/2203.04955",
            "section": "Hansen et al. 2022, TD-MPC, Section 3 (Planning) and Algorithm 1",
            "note": "Elite selection by estimated return and the return-weighted update of the sampling distribution.",
        },
    ],
    "tests": [
        {"name": "vjepa2: uniform mean and unbiased std of the lowest energies", "behavior": "routing.selection", "code": r"""
import math, torch
samples = torch.tensor([[[0.0]], [[2.0]], [[4.0]], [[10.0]]])
energies = torch.tensor([0.3, 0.1, 0.2, 5.0])
mean, std = torch.zeros(1, 1), torch.ones(1, 1)
new_mean, new_std = {fn}(samples, energies, mean, std, 2, 0.25, 0.5, "vjepa2")
# elites are samples 1 and 2 (values 2 and 4): mu = 3, unbiased sigma = sqrt(2)
assert new_mean.shape == (1, 1) and new_std.shape == (1, 1)
assert torch.allclose(new_mean, torch.tensor([[3.0 * 0.75]])), new_mean
assert torch.allclose(new_std, torch.tensor([[math.sqrt(2.0) * 0.5 + 0.5]])), new_std
"""},
        {"name": "tdmpc: softmax-weighted mean and std of the highest values", "behavior": "metrics.averaging", "code": r"""
import math, torch
samples = torch.tensor([[[0.0]], [[4.0]], [[8.0]], [[100.0]]], dtype=torch.float64)
values = torch.tensor([1.0, 3.0, 2.0, -5.0], dtype=torch.float64)
mean, std = torch.zeros(1, 1, dtype=torch.float64), torch.ones(1, 1, dtype=torch.float64)
new_mean, new_std = {fn}(samples, values, mean, std, 2, 0.0, 0.0, "tdmpc", temperature=math.log(3.0))
# elites: value 3 (x = 4) and value 2 (x = 8); weights 1 and 1/3, normalised to 0.75 and 0.25
assert abs(new_mean.item() - 5.0) < 1e-12, new_mean
assert abs(new_std.item() - math.sqrt(3.0)) < 1e-12, new_std  # 0.75 * 1 + 0.25 * 9
"""},
        {"name": "Matches the refit formulas on random inputs", "visibility": "unshown", "behavior": "metrics.averaging", "failure_message": "Rank the scores (ascending energies for vjepa2, descending values for tdmpc), take the first topk, compute the mode's elite mean and std, then blend each with the old value.", "code": r"""
import torch
def expect(x, e, mean, std, k, mm, ms, mode, temp):
    order = torch.argsort(e, stable=True, descending=(mode == "tdmpc"))[:k]
    el = x[order]
    if mode == "vjepa2":
        mu = el.mean(0)
        sd = ((el - mu) ** 2).sum(0).div(k - 1).sqrt()
    else:
        v = e[order]
        w = torch.exp(temp * (v - v.max()))
        w = (w / w.sum()).view(-1, 1, 1)
        mu = (w * el).sum(0)
        sd = (w * (el - mu) ** 2).sum(0).sqrt()
    return mu * (1 - mm) + mean * mm, sd * (1 - ms) + std * ms
for seed, S, H, A, k, mm, ms, mode, temp in (
    (0, 32, 4, 3, 6, 0.25, 0.95, "vjepa2", 0.5),
    (1, 20, 2, 5, 3, 0.1, 0.0, "tdmpc", 0.5),
    (2, 50, 3, 2, 10, 0.4, 0.2, "vjepa2", 1.0),
    (3, 16, 5, 1, 4, 0.7, 0.3, "tdmpc", 2.5),
    (4, 9, 1, 4, 2, 0.0, 1.0, "vjepa2", 0.5),
    (5, 40, 3, 3, 7, 0.33, 0.66, "tdmpc", 0.05),
):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(S, H, A, generator=g, dtype=torch.float64)
    e = torch.randn(S, generator=g, dtype=torch.float64) * 3
    mean = torch.randn(H, A, generator=g, dtype=torch.float64)
    std = torch.rand(H, A, generator=g, dtype=torch.float64) + 0.1
    got_m, got_s = {fn}(x, e, mean, std, k, mm, ms, mode, temperature=temp)
    want_m, want_s = expect(x, e, mean, std, k, mm, ms, mode, temp)
    assert got_m.shape == (H, A) and got_s.shape == (H, A), (got_m.shape, got_s.shape)
    assert torch.allclose(got_m, want_m, atol=1e-10, rtol=0), (seed, got_m, want_m)
    assert torch.allclose(got_s, want_s, atol=1e-10, rtol=0), (seed, got_s, want_s)
"""},
        {"name": "vjepa2 keeps low energies, tdmpc keeps high values", "visibility": "unshown", "behavior": "routing.selection", "failure_message": "In vjepa2 mode keep the topk LOWEST energies; in tdmpc mode keep the topk HIGHEST values.", "code": r"""
import torch
samples = torch.arange(6, dtype=torch.float64).view(6, 1, 1)
scores = torch.tensor([5.0, 0.0, 4.0, 1.0, 3.0, 2.0], dtype=torch.float64)
z, o = torch.zeros(1, 1, dtype=torch.float64), torch.ones(1, 1, dtype=torch.float64)
m, s = {fn}(samples, scores, z, o, 3, 0.0, 0.0, "vjepa2")
assert abs(m.item() - 3.0) < 1e-12, ("vjepa2 should keep samples 1, 3, 5", m)  # (1 + 3 + 5) / 3
m, s = {fn}(samples, scores, z, o, 3, 0.0, 0.0, "tdmpc", temperature=0.0)
assert abs(m.item() - 2.0) < 1e-12, ("tdmpc should keep samples 0, 2, 4", m)  # (0 + 2 + 4) / 3
"""},
        {"name": "Ties go to the lower sample index", "visibility": "unshown", "behavior": "metrics.ties", "failure_message": "Rank with a stable order so that, among equal scores, the sample with the lower index is chosen first.", "code": r"""
import torch
samples = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0, 60.0], dtype=torch.float64).view(6, 1, 1)
z, o = torch.zeros(1, 1, dtype=torch.float64), torch.ones(1, 1, dtype=torch.float64)
# energies: sample 4 is best, then samples 0, 2, 5 tie; topk = 3 must take 4, 0, 2
energies = torch.tensor([1.0, 7.0, 1.0, 9.0, 0.5, 1.0], dtype=torch.float64)
m, _ = {fn}(samples, energies, z, o, 3, 0.0, 0.0, "vjepa2")
assert abs(m.item() - (50.0 + 10.0 + 30.0) / 3) < 1e-12, m
# values: sample 1 is best, then samples 2, 3, 5 tie; topk = 2 must take 1 and 2
values = torch.tensor([-1.0, 4.0, 2.0, 2.0, -3.0, 2.0], dtype=torch.float64)
m, _ = {fn}(samples, values, z, o, 2, 0.0, 0.0, "tdmpc", temperature=0.0)
assert abs(m.item() - 25.0) < 1e-12, m
"""},
        {"name": "Momentum weights the old Gaussian", "visibility": "unshown", "behavior": "state.invariant", "failure_message": "Blend as new_stat * (1 - momentum) + old_stat * momentum, separately for the mean (momentum_mean) and the std (momentum_std).", "code": r"""
import torch
g = torch.Generator().manual_seed(9)
x = torch.randn(12, 2, 3, generator=g, dtype=torch.float64)
e = torch.randn(12, generator=g, dtype=torch.float64)
mean = torch.full((2, 3), 7.0, dtype=torch.float64)
std = torch.full((2, 3), 0.25, dtype=torch.float64)
for mode in ("vjepa2", "tdmpc"):
    m1, s1 = {fn}(x, e, mean, std, 4, 1.0, 1.0, mode)
    assert torch.equal(m1, mean) and torch.equal(s1, std), (mode, "momentum 1 must keep the old Gaussian")
    m0, s0 = {fn}(x, e, mean, std, 4, 0.0, 0.0, mode)
    m, s = {fn}(x, e, mean, std, 4, 0.2, 0.9, mode)
    assert torch.allclose(m, 0.8 * m0 + 0.2 * mean, atol=1e-12), mode
    assert torch.allclose(s, 0.1 * s0 + 0.9 * std, atol=1e-12), mode
"""},
        {"name": "Weights stay finite for large values", "visibility": "unshown", "behavior": "numerics.stability", "failure_message": "Compute the weights as exp(temperature * (v - max_v)) over the elite values, so large values do not overflow to inf or NaN.", "code": r"""
import torch
samples = torch.tensor([1.0, 3.0, 5.0, 7.0], dtype=torch.float64).view(4, 1, 1)
values = torch.tensor([2000.0, 2001.0, 1999.0, -50.0], dtype=torch.float64)
z, o = torch.zeros(1, 1, dtype=torch.float64), torch.ones(1, 1, dtype=torch.float64)
m, s = {fn}(samples, values, z, o, 3, 0.0, 0.0, "tdmpc", temperature=1.0)
w = torch.exp(torch.tensor([0.0, -1.0, -2.0], dtype=torch.float64))
w = w / w.sum()
x = torch.tensor([3.0, 1.0, 5.0], dtype=torch.float64)
mu = (w * x).sum()
assert torch.isfinite(m).all() and torch.isfinite(s).all(), (m, s)
assert abs(m.item() - mu.item()) < 1e-10, (m, mu)
assert abs(s.item() - (w * (x - mu) ** 2).sum().sqrt().item()) < 1e-10, s
x32 = samples.float()
m32, s32 = {fn}(x32, values.float() * 10, z.float(), o.float(), 3, 0.0, 0.0, "tdmpc", temperature=0.5)
assert torch.isfinite(m32).all() and torch.isfinite(s32).all(), (m32, s32)
"""},
        {"name": "tdmpc std is the biased weighted std", "visibility": "unshown", "behavior": "metrics.averaging", "failure_message": "In tdmpc mode sigma = sqrt(sum_i w_i * (x_i - mu)^2) with normalised weights; do not divide by topk - 1.", "code": r"""
import torch
samples = torch.tensor([0.0, 2.0, 4.0, 6.0], dtype=torch.float64).view(4, 1, 1)
values = torch.ones(4, dtype=torch.float64)
z, o = torch.zeros(1, 1, dtype=torch.float64), torch.ones(1, 1, dtype=torch.float64)
m, s = {fn}(samples, values, z, o, 4, 0.0, 0.0, "tdmpc", temperature=0.5)
assert abs(m.item() - 3.0) < 1e-12, m
assert abs(s.item() - 5.0 ** 0.5) < 1e-12, s  # population variance of 0, 2, 4, 6 is 5
"""},
        {"name": "vjepa2 std is unbiased and ignores the energies' magnitudes", "visibility": "unshown", "behavior": "metrics.averaging", "failure_message": "In vjepa2 mode average the elites uniformly and use the unbiased std (divide by topk - 1); the energies only choose the elites.", "code": r"""
import torch
samples = torch.tensor([0.0, 2.0, 4.0, 6.0, 100.0], dtype=torch.float64).view(5, 1, 1)
energies = torch.tensor([0.0, 1.0, 50.0, 2.0, 99.0], dtype=torch.float64)
z, o = torch.zeros(1, 1, dtype=torch.float64), torch.ones(1, 1, dtype=torch.float64)
m, s = {fn}(samples, energies, z, o, 4, 0.0, 0.0, "vjepa2")
assert abs(m.item() - 3.0) < 1e-12, m
assert abs(s.item() - (20.0 / 3.0) ** 0.5) < 1e-12, s
"""},
        {"name": "topk = 1 and topk = S", "visibility": "unshown", "behavior": "edge.empty_or_boundary", "failure_message": "tdmpc with topk 1 returns the best sample with std 0; vjepa2 with topk 1 raises ValueError; topk = S uses every sample.", "code": r"""
import torch
g = torch.Generator().manual_seed(4)
x = torch.randn(7, 3, 2, generator=g, dtype=torch.float64)
v = torch.randn(7, generator=g, dtype=torch.float64)
z, o = torch.zeros(3, 2, dtype=torch.float64), torch.ones(3, 2, dtype=torch.float64)
m, s = {fn}(x, v, z, o, 1, 0.0, 0.0, "tdmpc")
assert torch.equal(m, x[int(v.argmax())]) and torch.equal(s, torch.zeros(3, 2, dtype=torch.float64)), (m, s)
try:
    {fn}(x, v, z, o, 1, 0.0, 0.0, "vjepa2")
except ValueError:
    pass
else:
    raise AssertionError("vjepa2 with topk = 1 must raise ValueError")
m, s = {fn}(x, v, z, o, 7, 0.0, 0.0, "vjepa2")
assert torch.allclose(m, x.mean(0), atol=1e-12) and torch.allclose(s, x.std(0), atol=1e-12)
"""},
        {"name": "CEM with a toy linear world model reaches the goal", "visibility": "unshown", "behavior": "routing.selection", "failure_message": "Repeated refits must move the Gaussian toward low energy (vjepa2) or high value (tdmpc); check the selection direction and the momentum blend.", "code": r"""
import torch
g = torch.Generator().manual_seed(0)
D, H, A = 4, 3, 2
Wz = torch.eye(D, dtype=torch.float64) * 0.9 + 0.05 * torch.randn(D, D, generator=g, dtype=torch.float64)
Wa = torch.randn(D, A, generator=g, dtype=torch.float64)
z0 = torch.randn(D, generator=g, dtype=torch.float64)
def rollout(acts):  # (S, H, A) -> final latent (S, D)
    z = z0.expand(acts.shape[0], D)
    for h in range(H):
        z = z @ Wz.T + acts[:, h] @ Wa.T
    return z
goal = rollout(0.6 * torch.randn(1, H, A, generator=g, dtype=torch.float64))[0]
energy = lambda acts: (rollout(acts) - goal).abs().mean(-1)  # L1 distance to the goal embedding
for mode, k, mm, ms in (("vjepa2", 8, 0.25, 0.3), ("tdmpc", 8, 0.1, 0.0)):
    mean, std = torch.zeros(H, A, dtype=torch.float64), torch.ones(H, A, dtype=torch.float64)
    start = energy(mean[None]).item()
    for _ in range(30):
        samples = mean + std * torch.randn(64, H, A, generator=g, dtype=torch.float64)
        e = energy(samples)
        mean, std = {fn}(samples, e if mode == "vjepa2" else -e, mean, std, k, mm, ms, mode, temperature=5.0)
    final = energy(mean[None]).item()
    assert final < 0.1 * start, (mode, start, final)
"""},
        {"name": "Keeps dtype and does not modify inputs", "visibility": "unshown", "behavior": "tensor.dtype_device", "failure_message": "Return tensors in the dtype of samples and never write into samples, energies, mean or std.", "code": r"""
import torch
g = torch.Generator().manual_seed(2)
for dtype in (torch.float32, torch.float64):
    for mode in ("vjepa2", "tdmpc"):
        x = torch.randn(10, 2, 3, generator=g).to(dtype)
        e = torch.randn(10, generator=g).to(dtype)
        mean = torch.randn(2, 3, generator=g).to(dtype)
        std = torch.rand(2, 3, generator=g).to(dtype)
        copies = [t.clone() for t in (x, e, mean, std)]
        m, s = {fn}(x, e, mean, std, 3, 0.3, 0.6, mode)
        assert m.dtype == dtype and s.dtype == dtype, (mode, m.dtype, s.dtype)
        for t, c in zip((x, e, mean, std), copies):
            assert torch.equal(t, c), (mode, "an input was modified in place")
"""},
        {"name": "Rejects bad arguments", "visibility": "unshown", "behavior": "contract.signature", "failure_message": "Raise ValueError for an unknown mode, wrong shapes or dtypes, non-finite energies, topk outside 1..S (or 1 in vjepa2 mode), momentum outside [0, 1], or a negative temperature.", "code": r"""
import torch
x = torch.zeros(5, 2, 3)
e = torch.arange(5.0)
m, s = torch.zeros(2, 3), torch.ones(2, 3)
good = dict(samples=x, energies=e, mean=m, std=s, topk=2, momentum_mean=0.2, momentum_std=0.5, mode="vjepa2", temperature=0.5)
{fn}(**good)
bad = [
    dict(mode="mppi"),
    dict(samples=torch.zeros(5, 6)),
    dict(samples=torch.zeros(5, 2, 3, dtype=torch.long)),
    dict(energies=torch.arange(4.0)),
    dict(energies=torch.arange(5.0, dtype=torch.float64)),
    dict(energies=torch.tensor([0.0, 1.0, float("nan"), 2.0, 3.0])),
    dict(energies=torch.tensor([0.0, 1.0, float("inf"), 2.0, 3.0])),
    dict(mean=torch.zeros(2, 2)),
    dict(std=torch.ones(3, 3)),
    dict(std=torch.ones(2, 3, dtype=torch.float64)),
    dict(topk=0),
    dict(topk=6),
    dict(topk=2.0),
    dict(topk=1),
    dict(momentum_mean=1.5),
    dict(momentum_std=-0.1),
    dict(mode="tdmpc", temperature=-1.0),
]
for change in bad:
    try:
        {fn}(**{**good, **change})
    except ValueError:
        pass
    else:
        raise AssertionError(f"accepted {change}")
"""},
    ],
    "solution": '''import torch


def cem_refit(samples, energies, mean, std, topk, momentum_mean, momentum_std, mode="vjepa2", temperature=0.5):
    if mode not in ("vjepa2", "tdmpc"):
        raise ValueError(f"unknown mode {mode!r}")
    if not torch.is_tensor(samples) or samples.ndim != 3 or not samples.is_floating_point():
        raise ValueError("samples must be a floating (S, H, A) tensor")
    S, H, A = samples.shape
    for name, t, shape in (("energies", energies, (S,)), ("mean", mean, (H, A)), ("std", std, (H, A))):
        if not torch.is_tensor(t) or tuple(t.shape) != shape or t.dtype != samples.dtype:
            raise ValueError(f"{name} must have shape {shape} and the dtype of samples")
    if not torch.isfinite(energies).all():
        raise ValueError("energies must be finite")
    if isinstance(topk, bool) or not isinstance(topk, int) or not 1 <= topk <= S:
        raise ValueError("topk must be an integer in 1..S")
    if mode == "vjepa2" and topk < 2:
        raise ValueError("vjepa2 mode needs topk >= 2 for an unbiased std")
    for m in (momentum_mean, momentum_std):
        if not 0.0 <= m <= 1.0:
            raise ValueError("momentum must be in [0, 1]")
    if temperature < 0:
        raise ValueError("temperature must be >= 0")

    # Stable sort: among equal scores the lower sample index ranks first.
    order = torch.argsort(energies, descending=(mode == "tdmpc"), stable=True)
    elite_idx = order[:topk]
    elites = samples[elite_idx]  # (topk, H, A)

    if mode == "vjepa2":
        mu = elites.mean(dim=0)
        sigma = elites.std(dim=0, correction=1)
    else:
        values = energies[elite_idx]
        w = torch.exp(temperature * (values - values.max()))
        w = (w / w.sum()).view(-1, 1, 1)
        mu = (w * elites).sum(dim=0)
        sigma = torch.sqrt((w * (elites - mu) ** 2).sum(dim=0))

    new_mean = mu * (1.0 - momentum_mean) + mean * momentum_mean
    new_std = sigma * (1.0 - momentum_std) + std * momentum_std
    return new_mean, new_std
''',
    "interview_questions": interview(
        concept=[
            "How does the cross-entropy method turn a world model into a planner, and what does one refit step do?",
            "Why does V-JEPA 2-AC minimise an energy while TD-MPC maximises a value, and what does each score measure?",
        ],
        deep_dive=[
            "Why does TD-MPC subtract the maximum elite value before exponentiating, and why does that not change the weights?",
            "What goes wrong with an unbiased std when only one elite is kept, and how would you handle it in a planner?",
            "How do the momentum values on the mean and the std change how fast CEM narrows its search, and what happens at the extremes?",
        ],
        tradeoffs=[
            "Uniform elite averaging versus softmax weighting by value: robustness to noisy scores versus speed of convergence?",
            "CEM planning at test time versus a learned policy that outputs actions directly: compute, adaptability to new goals and model errors?",
        ],
    ),
}
