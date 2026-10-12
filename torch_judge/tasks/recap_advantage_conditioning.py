"""RECAP (pi*0.6): steps-to-success values, n-step advantages and the binarized advantage indicator."""

from ._interview import interview

# A plain-loop n-step advantage, written independently of any vectorized solution.
_LOOP_ADVANTAGE = r"""
import torch

def loop_advantage(rewards, values, n):
    r, v = rewards.tolist(), values.tolist()
    last = len(r) - 1
    out = []
    for t in range(len(r)):
        stop = last + 1 if n is None else min(t + n, last + 1)
        boot = v[t + n] if (n is not None and t + n <= last) else 0.0
        out.append(sum(r[t:stop]) + boot - v[t])
    return torch.tensor(out, dtype=torch.float64)
"""

TASK = {
    "title": "RECAP Advantage Conditioning",
    "difficulty": "Hard",
    "version": 1,
    "function_name": "RecapAdvantageConditioning",
    "description_en": r"""Build the data side of RECAP, the method that trains pi*0.6 from its own rollouts: the steps-to-success reward, the value-bin targets, the n-step advantage, the per-task threshold that turns advantages into the binarized "Advantage: positive / negative" input, and the classifier-free guidance combination used at inference.

**Signature:** `RecapAdvantageConditioning(num_bins=201)` with these methods:
- `episode_rewards(length, success, max_len, c_fail) -> Tensor`
- `episode_returns(length, success, max_len, c_fail) -> Tensor`
- `value_bins(returns) -> LongTensor`
- `expected_value(logits) -> Tensor`
- `nstep_advantage(rewards, values, lookahead=None) -> Tensor`
- `threshold(advantages, positive_fraction) -> float`
- `indicators(advantages, task_ids, positive_fraction, is_correction, drop_u, drop_prob=0.3) -> LongTensor`
- `indicator_text(code) -> str`
- `cfg_velocity(v_uncond, v_cond, beta) -> Tensor`

**Rewards and returns.** An episode of `length = T` has observations `o_0 .. o_T` and rewards `r_0 .. r_T`, so both methods return a float64 tensor of shape `(T + 1,)`.
- `r_t = -1` for `t < T`. The terminal reward `r_T` is `0` on success and `-c_fail` on failure.
- Divide every reward by `max_len`, the task's maximum episode length.
- `episode_returns` gives `R_t = r_t + r_(t+1) + ... + r_T` (no discount). A successful episode therefore has `R_t = -(T - t) / max_len`.
- Raise `ValueError` when `length` is not a non-negative integer, when `max_len <= 0` or `max_len < length`, or when `c_fail < 0`.

**Value bins.** The `num_bins` bin values are `torch.linspace(-1, 0, num_bins)`.
- `value_bins` maps each return to the index of the nearest bin value, clamped to `0 .. num_bins - 1`. It keeps the input shape. Returns never sit exactly halfway between two bin values in the tests.
- `expected_value(logits)` takes logits of shape `(..., num_bins)` and returns the softmax-weighted sum of the bin values, shape `(...)`, in the dtype of `logits`. Raise `ValueError` when the last dimension is not `num_bins`.
- Raise `ValueError` in the constructor when `num_bins` is not an integer of at least 2.

**n-step advantage.** `rewards` and `values` are matching non-empty 1-D tensors for one episode (`values[t]` is the predicted `V(o_t)`).
- With `lookahead = N`: `A_t = r_t + ... + r_(t+N-1) + values[t+N] - values[t]`.
- Rewards past the last index are not added. When `t + N` is past the last index, the bootstrap term is 0; when `t + N` is exactly the last index, use `values[t+N]`.
- With `lookahead=None` (pre-training, N = T): `A_t = R_t - values[t]`, the rewards from `t` to the end with no bootstrap.
- Keep the input dtype. Raise `ValueError` for shape mismatches, empty or non-1-D inputs, or a `lookahead` that is not a positive integer or `None`.

**Threshold.** `threshold` returns `eps`, the `(1 - positive_fraction)` quantile of the 1-D `advantages`, with linear interpolation between sorted values (what `torch.quantile` does by default). About `positive_fraction` of the advantages are then strictly above it. Raise `ValueError` unless `0 < positive_fraction < 1`, or when `advantages` is empty or not 1-D.

**Indicators.** All five tensor arguments have shape `(M,)`. Return a long tensor of codes: `1` positive, `0` negative, `-1` dropped.
- Compute one threshold per distinct value of `task_ids`, over all of that task's samples (corrections included), with `threshold`.
- A sample is positive when its advantage is strictly greater than its task's threshold, or when `is_correction` is true (a human correction is always positive).
- Then drop the indicator, code `-1`, wherever `drop_u < drop_prob`, whatever its sign.
- Raise `ValueError` for mismatched shapes or a `drop_prob` outside `[0, 1]`.

**Text and guidance.**
- `indicator_text` maps `1` to `"Advantage: positive"`, `0` to `"Advantage: negative"` and `-1` to `""` (no indicator in the prompt). Any other code raises `ValueError`.
- `cfg_velocity` returns `v_uncond + beta * (v_cond - v_uncond)`. Raise `ValueError` when the shapes differ.

**Example:**

    rc = RecapAdvantageConditioning()
    rc.episode_returns(4, True, 8, 10.0)    # [-0.5, -0.375, -0.25, -0.125, 0.0]
    r = rc.episode_rewards(4, True, 8, 10.0)
    v = torch.tensor([-0.4, -0.4, -0.2, -0.1, 0.0], dtype=torch.float64)
    rc.nstep_advantage(r, v, 2)             # [-0.05, 0.05, -0.05, -0.025, 0.0]
    a = torch.tensor([0.1, -0.2, 0.3, 0.0, 0.5, -0.1, 0.2, 0.4, -0.3, 0.05])
    rc.threshold(a, 0.3)                    # 0.23: exactly 0.3, 0.5 and 0.4 lie above it

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**What RECAP does.** RECAP (RL with Experience and Corrections via Advantage-conditioned Policies) alternates three steps: collect rollouts, with optional expert interventions; train a value function on all data so far; and train the VLA on all data with an extra input saying whether each action was better than the threshold. The value function is a Monte Carlo estimate for the behavior policy of the dataset, not an off-policy Q-function.

**The reward.** Eq. 5 of the paper gives -1 per step, 0 at a successful end and a large negative constant at a failed end, so the value is the negative number of steps to success. Values are normalized per task by the maximum episode length to lie between -1 and 0, and the value head predicts a distribution over 201 bins trained with cross-entropy (Eq. 1). The paper does not say how returns are assigned to bins; nearest bin with clamping is this exercise's choice, and failed returns below -1 land in bin 0.

**The advantage.** In post-training the paper uses an n-step estimate with N = 50 lookahead steps. In pre-training it uses N = T, which needs only one value-function call per example and so can run on the fly during VLA training. The appendix writes that sum starting at t' = 0; this exercise reads it as starting at t, which is what N = T gives in the n-step formula.

**The threshold, and an ambiguity.** Section V-D says eps is set "to the 30% percentile of values predicted by the value function". Appendix F says the threshold is chosen "such that approximately 30% of the demonstration data has positive advantage" in pre-training, about 40% of evaluation rollouts in fine-tuning, and about 10% for the laundry task where demonstrations are slow but reliable. Taken literally, the first gives about 70% positive and the second about 30%. This exercise follows Appendix F, the more detailed statement: eps is the (1 - fraction) quantile, so the fraction names the positive share. The exact quantile rule is not in the paper.

**Why condition instead of weight.** Weighted regression such as AWR discards or downweights much of the data. Advantage conditioning trains on every example with plain supervised learning, labelled as better or worse. By Bayes rule, the indicator-conditioned policy is the improved policy of Eq. 2 with beta = 1. The text "Advantage: positive" or "Advantage: negative" sits after the predicted subtask and before the action tokens, so it changes only the action likelihoods.

**Dropout and guidance.** Appendix F drops the indicator 30% of the time, which replaces the loss weight alpha of Eq. 3 and trains the unconditional policy too. At inference the model can be prompted with "Advantage: positive" (beta = 1), or the conditional and unconditional outputs can be mixed as in Eq. 13 for beta > 1. The paper reports that large beta pushes actions to the edge of their support and gives aggressive motions, relies mainly on the threshold, and uses moderate values such as beta in [1.5, 2.5] where useful.

**Corrections and SFT.** Actions from human corrections are forced positive, on the assumption that experts correct well. When first fine-tuning on a task's demonstrations, the indicator is fixed to positive, which the paper found slightly better; that stage is plain SFT.""",
    "advisory_prerequisites": ["reinforce_discounted_returns", "gae_advantage", "pi0_flow_matching_chunk"],
    "hints": [
        {"level": 1, "kind": "questions", "content": "Where does the terminal reward sit in an episode of length T, and how many observations does that episode have? When t + N runs past the end, what is left to bootstrap from? If 30% of the data should be positive, which quantile of the advantages is the threshold? Where in the order of operations do corrections and dropout apply, and should corrections change the threshold?"},
        {"level": 2, "kind": "analysis", "content": "Build T + 1 rewards of -1, overwrite the last with 0 or -c_fail, divide by max_len, and take a reverse cumulative sum for returns. For value bins, scale (R + 1) by num_bins - 1, round and clamp. For the advantage, use a reverse cumulative sum padded with a zero so a window sum is a difference of two entries, and pad the values with a zero for the bootstrap. Threshold with the (1 - fraction) quantile; loop over unique task ids, compare strictly, OR in corrections, then overwrite dropped entries with -1."},
    ],
    "model_connections": [
        "pi*0.6 (Physical Intelligence, 2025) is pi0.6 trained with RECAP; the paper reports that full RECAP more than doubles throughput and roughly halves the failure rate on the diverse laundry folding and espresso tasks.",
        "CFGRL (Frans et al.) is the closest prior formulation; it used a threshold of 0 and tuned beta at test time, where RECAP sets a per-task threshold during training instead.",
        "The forward Euler sampler of pi0_flow_matching_chunk is where cfg_velocity plugs in: call the action expert with and without the indicator and step along the mixed velocity.",
    ],
    "pro_con_analysis": {
        "pros": [
            "Every example is trained with supervised learning, so the method scales to flow-matching action heads that have no tractable log-likelihood.",
            "Demonstrations, autonomous rollouts from old and new policies, and expert corrections all fit in one dataset with one label per action.",
            "The threshold is a direct knob for how selective the positive label is, with no test-time guidance weight needed.",
        ],
        "cons": [
            "The value function is a Monte Carlo estimate for the data's behavior policy, which the paper notes is less optimal than an off-policy Q-function.",
            "The positive fraction is a per-task hyperparameter (30%, 40% or 10% in the paper), and the paper's main text and appendix describe it differently.",
            "Forcing corrections positive assumes the expert always corrects well.",
        ],
    },
    "sources": [
        {
            "kind": "paper",
            "url": "https://arxiv.org/abs/2511.14759",
            "section": "Physical Intelligence 2025 (arXiv 2511.14759v2), Section IV-A (Eq. 1, expected value over bins) and Section IV-B (Eq. 2-3, I_t = 1[A > eps_l], corrections forced to True)",
            "note": "The distributional value function and the advantage-conditioned objective. No official RECAP code exists; the oracle is torch_judge/harness/embodied_frontier/vendored/recap_paper.py, transcribed from these definitions.",
        },
        {
            "kind": "paper",
            "url": "https://arxiv.org/abs/2511.14759",
            "section": "Section V-B (indicator text after the subtask, before the actions), Section V-C (Eq. 5 reward, normalization to (-1, 0) by maximum episode length) and Section V-D (threshold and SFT with I_t = True)",
            "note": "Section V-D says eps_l is 'the 30% percentile of values predicted by the value function'. The exercise follows Appendix F instead (see below) and treats positive_fraction as the share above the threshold.",
        },
        {
            "kind": "paper",
            "url": "https://arxiv.org/abs/2511.14759",
            "section": "Appendix F (Advantage Estimation, Advantage conditioning dropout, Advantage threshold) and Appendix E (Eq. 13, CFG with beta > 1)",
            "note": "N = 50 in post-training and N = T in pre-training; 30% indicator dropout; threshold chosen so about 30% (pre-training), 40% (fine-tuning) or 10% (laundry) of the data is positive. The N = T sum is written from t' = 0 in the appendix; the exercise starts it at t. The paper names no quantile rule, bin assignment or boundary bootstrap; the exercise uses linear-interpolation quantiles, the nearest bin, and a zero bootstrap past the last observation.",
        },
    ],
    "tests": [
        {"name": "Steps-to-success rewards and returns", "behavior": "state.invariant", "code": r"""
import torch
rc = {fn}()
r = rc.episode_rewards(4, True, 8, 10.0)
assert r.dtype == torch.float64 and r.shape == (5,), (r.dtype, r.shape)
assert torch.allclose(r, torch.tensor([-0.125, -0.125, -0.125, -0.125, 0.0], dtype=torch.float64)), r
R = rc.episode_returns(4, True, 8, 10.0)
assert torch.allclose(R, torch.tensor([-0.5, -0.375, -0.25, -0.125, 0.0], dtype=torch.float64)), R
R = rc.episode_returns(3, False, 8, 4.0)
assert torch.allclose(R, torch.tensor([-0.875, -0.75, -0.625, -0.5], dtype=torch.float64)), R
"""},
        {"name": "n-step advantage with a short lookahead", "behavior": "state.invariant", "code": r"""
import torch
rc = {fn}()
r = rc.episode_rewards(4, True, 8, 10.0)
v = torch.tensor([-0.4, -0.4, -0.2, -0.1, 0.0], dtype=torch.float64)
A = rc.nstep_advantage(r, v, 2)
assert torch.allclose(A, torch.tensor([-0.05, 0.05, -0.05, -0.025, 0.0], dtype=torch.float64)), A
A = rc.nstep_advantage(r, v)
assert torch.allclose(A, torch.tensor([-0.1, 0.025, -0.05, -0.025, 0.0], dtype=torch.float64)), A
"""},
        {"name": "Threshold keeps the top 30% and corrections stay positive", "behavior": "state.invariant", "code": r"""
import torch
rc = {fn}()
a = torch.tensor([0.1, -0.2, 0.3, 0.0, 0.5, -0.1, 0.2, 0.4, -0.3, 0.05], dtype=torch.float64)
assert abs(rc.threshold(a, 0.3) - 0.23) < 1e-9, rc.threshold(a, 0.3)
tasks = torch.zeros(10, dtype=torch.long)
corr = torch.zeros(10, dtype=torch.bool); corr[1] = True
u = torch.ones(10); u[4] = 0.1
codes = rc.indicators(a, tasks, 0.3, corr, u, 0.3)
assert codes.tolist() == [0, 1, 1, 0, -1, 0, 0, 1, 0, 0], codes
assert rc.indicator_text(1) == "Advantage: positive" and rc.indicator_text(-1) == ""
"""},
        {"name": "Seeded rewards and returns follow Eq. 5", "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "Use T + 1 rewards: -1 before the end, 0 or -c_fail at index T, all divided by max_len; returns are undiscounted suffix sums.",
         "code": r"""
import torch
rc = {fn}()
g = torch.Generator().manual_seed(5)
for _ in range(12):
    T = int(torch.randint(0, 30, (), generator=g))
    max_len = T + int(torch.randint(1, 20, (), generator=g))
    c_fail = float(torch.rand((), generator=g) * 50)
    success = bool(torch.rand((), generator=g) < 0.5)
    r = rc.episode_rewards(T, success, max_len, c_fail)
    want = [-1.0] * T + [0.0 if success else -c_fail]
    want = torch.tensor(want, dtype=torch.float64) / max_len
    assert r.dtype == torch.float64 and r.shape == (T + 1,), (r.dtype, r.shape)
    assert torch.allclose(r, want, atol=1e-12), (T, success)
    R = rc.episode_returns(T, success, max_len, c_fail)
    suffix = torch.tensor([float(want[t:].sum()) for t in range(T + 1)], dtype=torch.float64)
    assert torch.allclose(R, suffix, atol=1e-12), (T, success)
    if success:
        assert torch.allclose(R, -(T - torch.arange(T + 1, dtype=torch.float64)) / max_len, atol=1e-12)
"""},
        {"name": "Length-zero episodes and the failure constant", "visibility": "unshown", "behavior": "edge.empty_or_boundary",
         "failure_message": "An episode of length 0 has one observation whose return is only the terminal reward; a failure adds -c_fail / max_len, not an extra -1.",
         "code": r"""
import torch
rc = {fn}()
assert torch.allclose(rc.episode_returns(0, True, 5, 3.0), torch.tensor([0.0], dtype=torch.float64))
assert torch.allclose(rc.episode_returns(0, False, 5, 3.0), torch.tensor([-0.6], dtype=torch.float64))
R = rc.episode_returns(2, False, 4, 0.0)
assert torch.allclose(R, torch.tensor([-0.5, -0.25, 0.0], dtype=torch.float64)), R
R = rc.episode_returns(5, False, 5, 10.0)
assert torch.allclose(R[0], torch.tensor(-3.0, dtype=torch.float64)), R
"""},
        {"name": "Returns map to the nearest bin value, clamped", "visibility": "unshown", "behavior": "numerics.stability",
         "failure_message": "Bin values are linspace(-1, 0, num_bins); pick the nearest one (round, do not floor) and clamp returns outside [-1, 0] to the end bins.",
         "code": r"""
import torch
for B in (201, 11, 2):
    rc = {fn}(B)
    centers = torch.linspace(-1, 0, B, dtype=torch.float64)
    g = torch.Generator().manual_seed(B)
    R = torch.rand(4, 25, generator=g, dtype=torch.float64) * 1.6 - 1.3
    width = 1.0 / (B - 1)
    frac = ((R + 1) / width) % 1.0
    R = torch.where((frac - 0.5).abs() < 1e-3, R + 0.1 * width, R)
    got = rc.value_bins(R)
    want = (R[..., None] - centers).abs().argmin(-1)
    assert got.shape == R.shape and got.dtype == torch.long, (got.shape, got.dtype)
    assert torch.equal(got, want), B
rc = {fn}()
assert rc.value_bins(torch.tensor([-1.7, 0.4, -0.5, -0.0049, -0.9951], dtype=torch.float64)).tolist() == [0, 200, 100, 199, 1]
"""},
        {"name": "Expected value is the softmax-weighted bin value", "visibility": "unshown", "behavior": "numerics.stability",
         "failure_message": "Softmax over the last dimension, then sum probabilities times linspace(-1, 0, num_bins); keep the logits' dtype and batch shape.",
         "code": r"""
import torch
for B in (201, 7):
    rc = {fn}(B)
    centers = torch.linspace(-1, 0, B, dtype=torch.float64)
    g = torch.Generator().manual_seed(B + 1)
    logits = torch.randn(2, 3, B, generator=g, dtype=torch.float64) * 3
    V = rc.expected_value(logits)
    assert V.shape == (2, 3) and V.dtype == torch.float64, (V.shape, V.dtype)
    assert torch.allclose(V, (torch.softmax(logits, -1) * centers).sum(-1), atol=1e-12), B
rc = {fn}()
peak = torch.full((201,), -1e4, dtype=torch.float32); peak[150] = 1e4
V = rc.expected_value(peak)
assert V.dtype == torch.float32 and torch.isfinite(V) and abs(float(V) + 0.25) < 1e-5, V
flat = rc.expected_value(torch.zeros(201))
assert abs(float(flat) + 0.5) < 1e-6, flat
"""},
        {"name": "n-step advantage matches a loop for every lookahead", "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "Sum rewards t .. t+N-1 (stopping at the last index), add values[t+N] only when t+N is a valid index, subtract values[t].",
         "code": _LOOP_ADVANTAGE + r"""
rc = {fn}()
for seed in range(6):
    g = torch.Generator().manual_seed(seed)
    T = int(torch.randint(1, 25, (), generator=g))
    rewards = rc.episode_rewards(T, bool(seed % 2), T + 3, 7.0)
    values = -torch.rand(T + 1, generator=g, dtype=torch.float64)
    for n in (1, 2, 5, T, T + 1, 50, None):
        got = rc.nstep_advantage(rewards, values, n)
        assert got.shape == (T + 1,), got.shape
        assert torch.allclose(got, loop_advantage(rewards, values, n), atol=1e-12), (seed, T, n)
"""},
        {"name": "Bootstrap boundary at the last observation", "visibility": "unshown", "behavior": "edge.empty_or_boundary",
         "failure_message": "When t + N is exactly the last index use values[t+N]; past it the bootstrap is 0 and the sum stops at the last reward.",
         "code": r"""
import torch
rc = {fn}()
r = torch.tensor([1.0, 2.0, 4.0], dtype=torch.float64)
v = torch.tensor([10.0, 20.0, 40.0], dtype=torch.float64)
assert torch.allclose(rc.nstep_advantage(r, v, 1), torch.tensor([11.0, 22.0, -36.0], dtype=torch.float64))
assert torch.allclose(rc.nstep_advantage(r, v, 2), torch.tensor([33.0, -14.0, -36.0], dtype=torch.float64))
assert torch.allclose(rc.nstep_advantage(r, v, 3), torch.tensor([-3.0, -14.0, -36.0], dtype=torch.float64))
assert torch.allclose(rc.nstep_advantage(r, v), torch.tensor([-3.0, -14.0, -36.0], dtype=torch.float64))
one = rc.nstep_advantage(torch.tensor([0.5]), torch.tensor([0.25]), 4)
assert one.dtype == torch.float32 and torch.allclose(one, torch.tensor([0.25])), one
"""},
        {"name": "Threshold is the (1 - fraction) quantile", "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "positive_fraction is the share above the threshold: use the (1 - positive_fraction) quantile with linear interpolation, not the positive_fraction quantile.",
         "code": r"""
import torch
rc = {fn}()
for seed, q in ((0, 0.3), (1, 0.4), (2, 0.1), (3, 0.75)):
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(int(torch.randint(5, 200, (), generator=g)), generator=g, dtype=torch.float64)
    eps = rc.threshold(a, q)
    assert isinstance(eps, float), type(eps)
    assert abs(eps - float(torch.quantile(a, 1 - q))) < 1e-12, (seed, q, eps)
    share = float((a > eps).double().mean())
    assert abs(share - q) <= 1.0 / a.numel() + 1e-9, (seed, q, share)
"""},
        {"name": "One threshold per task, compared strictly", "visibility": "unshown", "behavior": "routing.selection",
         "failure_message": "Group samples by task_id, compute each task's own threshold over that task's advantages, and mark positive only when A > eps (ties are negative).",
         "code": r"""
import torch
rc = {fn}()
g = torch.Generator().manual_seed(9)
a0 = torch.randn(40, generator=g, dtype=torch.float64) * 0.01 - 0.5
a1 = torch.randn(60, generator=g, dtype=torch.float64) * 3.0 + 2.0
adv = torch.cat([a0, a1]); tasks = torch.cat([torch.full((40,), 7), torch.full((60,), 2)])
perm = torch.randperm(100, generator=g)
adv, tasks = adv[perm], tasks[perm]
none = torch.zeros(100, dtype=torch.bool)
codes = rc.indicators(adv, tasks, 0.3, none, torch.ones(100))
for t in (7, 2):
    rows = tasks == t
    eps = float(torch.quantile(adv[rows], 0.7))
    assert torch.equal(codes[rows], (adv[rows] > eps).long()), t
ties = torch.tensor([0.0, 1, 1, 1, 1, 1, 2, 2, 2, 2], dtype=torch.float64)
codes = rc.indicators(ties, torch.zeros(10, dtype=torch.long), 0.3, torch.zeros(10, dtype=torch.bool), torch.ones(10))
assert codes.tolist() == [0] * 10, codes
"""},
        {"name": "Corrections are forced positive after thresholding", "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "Compute each task's threshold over all of its samples using their real advantages, then set every correction to positive.",
         "code": r"""
import torch
rc = {fn}()
g = torch.Generator().manual_seed(4)
adv = torch.randn(30, generator=g, dtype=torch.float64)
corr = torch.zeros(30, dtype=torch.bool); corr[:6] = True
adv[:6] = -5.0 - torch.rand(6, generator=g, dtype=torch.float64)
tasks = torch.zeros(30, dtype=torch.long)
codes = rc.indicators(adv, tasks, 0.4, corr, torch.ones(30))
eps = float(torch.quantile(adv, 0.6))
assert torch.all(codes[:6] == 1), codes
assert torch.equal(codes[6:], (adv[6:] > eps).long()), codes
"""},
        {"name": "Dropout overrides the indicator", "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "Set code -1 wherever drop_u < drop_prob (strictly), for positives, negatives and corrections alike.",
         "code": r"""
import torch
rc = {fn}()
g = torch.Generator().manual_seed(12)
adv = torch.randn(50, generator=g, dtype=torch.float64)
tasks = (torch.rand(50, generator=g) < 0.5).long()
corr = torch.rand(50, generator=g) < 0.2
u = torch.rand(50, generator=g)
u[0] = 0.3; corr[1] = True; u[1] = 0.05
base = rc.indicators(adv, tasks, 0.3, corr, torch.ones(50), 0.3)
codes = rc.indicators(adv, tasks, 0.3, corr, u, 0.3)
assert codes.dtype == torch.long, codes.dtype
assert torch.equal(codes, torch.where(u < 0.3, torch.full_like(base, -1), base)), codes
assert int(codes[0]) != -1 and int(codes[1]) == -1, codes[:2]
assert torch.equal(rc.indicators(adv, tasks, 0.3, corr, u, 0.0), base)
assert torch.all(rc.indicators(adv, tasks, 0.3, corr, u * 0.999, 1.0) == -1)
"""},
        {"name": "Indicator text for each code", "visibility": "unshown", "behavior": "contract.signature",
         "failure_message": "Return 'Advantage: positive' for 1, 'Advantage: negative' for 0, '' for -1, and raise ValueError for any other code.",
         "code": r"""
rc = {fn}()
assert rc.indicator_text(1) == "Advantage: positive"
assert rc.indicator_text(0) == "Advantage: negative"
assert rc.indicator_text(-1) == ""
for bad in (2, -2):
    try:
        rc.indicator_text(bad)
    except ValueError:
        pass
    else:
        raise AssertionError(f"code {bad} was accepted")
"""},
        {"name": "Guidance mixes from the unconditional output", "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "Return v_uncond + beta * (v_cond - v_uncond): beta = 0 gives v_uncond and beta = 1 gives v_cond.",
         "code": r"""
import torch
rc = {fn}()
g = torch.Generator().manual_seed(2)
vu = torch.randn(2, 5, 3, generator=g, dtype=torch.float64)
vc = torch.randn(2, 5, 3, generator=g, dtype=torch.float64)
assert torch.allclose(rc.cfg_velocity(vu, vc, 1.0), vc, atol=1e-12)
assert torch.allclose(rc.cfg_velocity(vu, vc, 0.0), vu, atol=1e-12)
assert torch.allclose(rc.cfg_velocity(vu, vc, 2.5), vu + 2.5 * (vc - vu), atol=1e-12)
"""},
        {"name": "Inputs are not modified", "visibility": "unshown", "behavior": "tensor.dtype_device",
         "failure_message": "Build new tensors; never write into the rewards, values, advantages or drop draws you were given.",
         "code": r"""
import torch
rc = {fn}()
g = torch.Generator().manual_seed(8)
r = torch.randn(9, generator=g, dtype=torch.float64); v = torch.randn(9, generator=g, dtype=torch.float64)
adv = torch.randn(9, generator=g, dtype=torch.float64); u = torch.rand(9, generator=g)
corr = torch.zeros(9, dtype=torch.bool); corr[2] = True
saved = [x.clone() for x in (r, v, adv, u, corr)]
rc.nstep_advantage(r, v, 3); rc.nstep_advantage(r, v)
rc.indicators(adv, torch.zeros(9, dtype=torch.long), 0.3, corr, u)
rc.threshold(adv, 0.5)
for before, after in zip(saved, (r, v, adv, u, corr)):
    assert torch.equal(before, after), "an input was modified"
"""},
        {"name": "Rejects bad arguments", "visibility": "unshown", "behavior": "contract.signature",
         "failure_message": "Raise ValueError for bad lengths, max_len, c_fail, num_bins, logits width, lookahead, positive_fraction, drop_prob and mismatched shapes.",
         "code": r"""
import torch
rc = {fn}()
x = torch.zeros(4, dtype=torch.float64)
calls = [
    lambda: {fn}(1),
    lambda: rc.episode_rewards(-1, True, 5, 1.0),
    lambda: rc.episode_rewards(6, True, 5, 1.0),
    lambda: rc.episode_rewards(2, True, 0, 1.0),
    lambda: rc.episode_returns(2, False, 5, -1.0),
    lambda: rc.expected_value(torch.zeros(3, 200)),
    lambda: rc.nstep_advantage(x, torch.zeros(5, dtype=torch.float64), 2),
    lambda: rc.nstep_advantage(x, x, 0),
    lambda: rc.nstep_advantage(torch.zeros(0), torch.zeros(0), 1),
    lambda: rc.threshold(x, 0.0),
    lambda: rc.threshold(x, 1.0),
    lambda: rc.threshold(torch.zeros(0), 0.5),
    lambda: rc.indicators(x, torch.zeros(3, dtype=torch.long), 0.3, torch.zeros(4, dtype=torch.bool), torch.ones(4)),
    lambda: rc.indicators(x, torch.zeros(4, dtype=torch.long), 0.3, torch.zeros(4, dtype=torch.bool), torch.ones(4), 1.5),
    lambda: rc.cfg_velocity(torch.zeros(2, 3), torch.zeros(3, 2), 1.5),
]
for i, call in enumerate(calls):
    try:
        call()
    except ValueError:
        pass
    else:
        raise AssertionError(f"case {i} was accepted")
"""},
    ],
    "solution": '''import torch

_TEXT = {1: "Advantage: positive", 0: "Advantage: negative", -1: ""}


class RecapAdvantageConditioning:
    def __init__(self, num_bins=201):
        if isinstance(num_bins, bool) or not isinstance(num_bins, int) or num_bins < 2:
            raise ValueError("num_bins must be an integer >= 2")
        self.num_bins = num_bins

    def episode_rewards(self, length, success, max_len, c_fail):
        if isinstance(length, bool) or not isinstance(length, int) or length < 0:
            raise ValueError("length must be a non-negative integer")
        if max_len <= 0 or max_len < length:
            raise ValueError("max_len must be positive and at least length")
        if c_fail < 0:
            raise ValueError("c_fail must be non-negative")
        rewards = torch.full((length + 1,), -1.0, dtype=torch.float64)
        rewards[length] = 0.0 if success else -float(c_fail)  # terminal reward, Eq. 5
        return rewards / max_len

    def episode_returns(self, length, success, max_len, c_fail):
        rewards = self.episode_rewards(length, success, max_len, c_fail)
        return rewards.flip(0).cumsum(0).flip(0)

    def bin_centers(self, dtype=torch.float64, device=None):
        return torch.linspace(-1.0, 0.0, self.num_bins, dtype=dtype, device=device)

    def value_bins(self, returns):
        width = 1.0 / (self.num_bins - 1)
        index = torch.round((returns.double() + 1.0) / width)
        return index.clamp(0, self.num_bins - 1).long()

    def expected_value(self, logits):
        if logits.ndim < 1 or logits.shape[-1] != self.num_bins:
            raise ValueError("logits must end in a num_bins dimension")
        probs = torch.softmax(logits, dim=-1)
        return probs @ self.bin_centers(logits.dtype, logits.device)

    def nstep_advantage(self, rewards, values, lookahead=None):
        if rewards.ndim != 1 or values.shape != rewards.shape or rewards.numel() == 0:
            raise ValueError("rewards and values must be matching non-empty 1-D tensors")
        if lookahead is not None and (isinstance(lookahead, bool) or not isinstance(lookahead, int) or lookahead < 1):
            raise ValueError("lookahead must be a positive integer or None")
        steps = rewards.numel()
        tail = rewards.flip(0).cumsum(0).flip(0)  # tail[t] = sum of rewards[t:]
        tail = torch.cat([tail, tail.new_zeros(1)])
        if lookahead is None:
            return tail[:steps] - values
        idx = torch.arange(steps, device=rewards.device)
        end = (idx + lookahead).clamp(max=steps)
        padded = torch.cat([values, values.new_zeros(1)])  # no value after the last observation
        return tail[idx] - tail[end] + padded[end] - values

    def threshold(self, advantages, positive_fraction):
        if advantages.ndim != 1 or advantages.numel() == 0:
            raise ValueError("advantages must be a non-empty 1-D tensor")
        if not 0.0 < positive_fraction < 1.0:
            raise ValueError("positive_fraction must be in (0, 1)")
        return float(torch.quantile(advantages.double(), 1.0 - positive_fraction))

    def indicators(self, advantages, task_ids, positive_fraction, is_correction, drop_u, drop_prob=0.3):
        if advantages.ndim != 1:
            raise ValueError("advantages must be 1-D")
        for other in (task_ids, is_correction, drop_u):
            if other.shape != advantages.shape:
                raise ValueError("task_ids, is_correction and drop_u must match advantages")
        if not 0.0 <= drop_prob <= 1.0:
            raise ValueError("drop_prob must be in [0, 1]")
        positive = torch.zeros(advantages.shape, dtype=torch.bool, device=advantages.device)
        for task in torch.unique(task_ids):
            rows = task_ids == task
            eps = self.threshold(advantages[rows], positive_fraction)  # one threshold per task
            positive[rows] = advantages[rows].double() > eps
        codes = (positive | is_correction.bool()).long()  # human corrections are always positive
        return torch.where(drop_u < drop_prob, torch.full_like(codes, -1), codes)

    def indicator_text(self, code):
        if isinstance(code, bool) or code not in _TEXT:
            raise ValueError("code must be 1, 0 or -1")
        return _TEXT[code]

    def cfg_velocity(self, v_uncond, v_cond, beta):
        if v_uncond.shape != v_cond.shape:
            raise ValueError("v_uncond and v_cond must have the same shape")
        return v_uncond + beta * (v_cond - v_uncond)
''',
    "interview_questions": interview(
        concept=[
            "What does the improvement indicator in RECAP mean, and why does conditioning the policy on it give an improved policy without weighting or discarding any examples?",
            "Why does RECAP define the reward as -1 per step with 0 at success and a large negative constant at failure, and what does the resulting value function measure?",
        ],
        deep_dive=[
            "Walk through the n-step advantage RECAP uses in post-training: which rewards are summed, which value bootstraps, and what changes when t + N runs past the end of the episode or when N = T in pre-training?",
            "The pi*0.6 paper describes the advantage threshold as a 30% percentile in the main text and as the level at which about 30% of the data is positive in the appendix. How do these readings differ, and how would you compute the threshold per task?",
            "Why does RECAP drop the advantage indicator 30% of the time during training, and how are the conditional and unconditional outputs combined at inference for beta > 1?",
        ],
        tradeoffs=[
            "Advantage conditioning versus advantage-weighted regression and versus PPO-style policy gradients for a flow-matching VLA: what does each need from the model, and what does each do with bad data?",
            "Raising the positive fraction versus raising the guidance weight beta: how does each trade imitation of the data against optimality, and what failure did the paper report for large beta?",
        ],
    ),
}
