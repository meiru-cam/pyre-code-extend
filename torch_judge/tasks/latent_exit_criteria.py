"""Huginn per-token adaptive exit: the first recurrent step at which each token may stop."""

from ._interview import interview

TASK = {
    "title": "Recurrent-Depth Adaptive Exit Criteria",
    "difficulty": "Medium",
    "version": 1,
    "function_name": "first_exit_step",
    "description_en": r"""A recurrent-depth language model (Huginn) iterates one core block up to `R` times for every token and may stop a token early once its prediction has settled. You are given the logits and latent states the model produced after every iteration. Return, for every token, the step at which an exit criterion first fires.

**Signature:** `first_exit_step(logits_per_step, latents_per_step, criterion, threshold="auto") -> LongTensor`

**Parameters:**
- `logits_per_step` — floating tensor `(R, N, V)`. Row `r` (0-based) holds the logits for `N` tokens after recurrent step `r + 1`.
- `latents_per_step` — floating tensor `(R, N, h)`. Row `r` holds the latent states after step `r + 1`. Only `latent-diff` reads it.
- `criterion` — one of `"kl"`, `"entropy-diff"`, `"latent-diff"`, `"argmax-stability"`.
- `threshold` — a number, or the string `"auto"` for the criterion's default.

**Returns:** a `torch.long` tensor of shape `(N,)` on the input device. The value is **1-based**: the number of recurrent steps run when the token exits, so step `t` means row `t - 1` fired first. A token whose criterion never fires returns `R`.

**The criteria.** For each token, compute one exit value per step `t = 1..R`, comparing step `t` with step `t - 1`:
- `kl`: `p_t = log_softmax(logits_t)`. The value is the KL divergence of the previous step's distribution to the current one, exactly `F.kl_div(p_t, p_(t-1), reduction="none", log_target=True).sum(-1)`, which equals `sum(exp(p_(t-1)) * (p_(t-1) - p_t))`. Before step 1, `p_0` is the uniform distribution: every entry is `log(1 / V)`.
- `entropy-diff`: `q_t = softmax(logits_t)` and `H_t = -sum(q_t * log(q_t + 1e-10))`. The value is `abs(H_t - H_(t-1))`, with the sentinel `H_0 = 100.0`.
- `latent-diff`: the relative change `norm(s_t - s_(t-1)) / norm(s_t)`, L2 norms over `h`. There is no state before step 1, so this criterion can never fire at step 1.
- `argmax-stability`: `a_t = logits_t.argmax(-1)`, taking the lowest index when logits tie. A counter `c_t` counts how many consecutive steps the argmax has stayed the same: `c_t = c_(t-1) + 1` if `a_t == a_(t-1)`, else `0`. Before step 1 there is no argmax, so `c_1 = 0`.

**Firing rule and ties:**
- `kl`, `entropy-diff` and `latent-diff` fire when the value is **strictly less than** the threshold. A value exactly equal to the threshold does not fire.
- `argmax-stability` fires when `c_t >= int(threshold)`, so a counter exactly equal to the threshold fires. With threshold `k` the earliest possible exit is step `k + 1`.

**Defaults for `"auto"`:** `kl` 1e-3, `entropy-diff` 1e-3, `latent-diff` 0.03, `argmax-stability` 5.

**Constraints:**
- Compute in the dtype of the inputs; float64 inputs are checked exactly against a step-by-step evaluation.
- Every token is evaluated independently.
- Raise `ValueError` for an unknown criterion, when either input is not 3-D, when `R < 1`, or when the two inputs disagree on `R` or `N`.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why exit early.** Huginn (Geiping et al. 2025) runs a prelude, then the same core block `r` times, then a coda that produces logits. More iterations buy more compute per token. Easy tokens converge after a few iterations, so the model card offers per-token adaptive compute: stop iterating a token once a convergence test passes, sample it, and move on.

**How upstream evaluates it.** `generate_with_adaptive_compute` runs one iteration at a time and calls an exit evaluator after each, keeping the previous step's entropy, log-probabilities, latents or argmax as state. A token that exits at loop index `k` records `k + 1` steps; one that never exits records the maximum. This exercise computes the same answer for all steps at once.

**Choosing a criterion.** The model card suggests `kl` for interesting exits and `argmax-stability` for conservative ones. The paper's experiments use KL between successive steps with threshold 5e-4; the code's `"auto"` default for `kl` is 1e-3. `latent-diff` needs no coda call, so it is the cheapest check; the logit-based criteria must run the coda after every step.

**The sentinels.** The entropy sentinel 100.0 is far above any real entropy for practical vocabularies, so step 1 fires only with a huge threshold. The uniform start for `kl` lets a token whose first prediction is already nearly uniform exit at step 1.""",
    "advisory_prerequisites": ["softmax_entropy", "masked_kd_kl", "recurrent_depth_unroll"],
    "hints": [
        {"level": 1, "kind": "questions", "content": "What does each criterion compare step t against, and what stands in for step 0? In F.kl_div(input, target, log_target=True), which argument is the reference distribution? Once you have a (R, N) boolean tensor of 'fires at this step', how do you find the first true row per column, and what should a column with no true rows return?"},
        {"level": 2, "kind": "analysis", "content": "Compute an exit value for all steps at once by shifting along the step axis: prepend the step-0 stand-in (uniform log-probabilities, the 100.0 entropy, or 'cannot fire' for latents) and compare each row with the one before it. Argmax stability is a run-length counter, so a short loop over R is fine. Then take the smallest 1-based step index where the condition holds, defaulting to R. Watch strict versus non-strict comparisons and the off-by-one between row index and step count."},
    ],
    "model_connections": [
        "Huginn-0125 (Geiping et al. 2025) exposes entropy-diff, latent-diff, kl and argmax-stability as `criterion` in generate_with_adaptive_compute.",
        "Early-exit transformers (CALM, Depth-Adaptive Transformer) need trained exit heads; a recurrent-depth model can exit zero-shot because every iteration feeds the same coda.",
        "The same convergence tests appear in Universal Transformers' ACT halting and in PonderNet, which learn the halting probability instead of thresholding a difference.",
    ],
    "pro_con_analysis": {
        "pros": ["Saves compute on easy tokens with no retraining, and each criterion is a cheap comparison of consecutive steps."],
        "cons": ["Thresholds are tuned per model, logit criteria need a coda call every step, and batched decoding wastes compute because the KV cache holds the deepest step of the batch."],
    },
    "sources": [
        {
            "kind": "code",
            "url": "https://github.com/seal-rg/recurrent-pretraining/blob/1ea7220ec7eb42d13e89db0663df254d0bcdc28e/recpre/raven_modeling_minimal.py",
            "commit": "1ea7220ec7eb42d13e89db0663df254d0bcdc28e",
            "path": "recpre/raven_modeling_minimal.py",
            "symbol": "EntropyDiffExitEvaluator, LatentDiffExitEvaluator, KLExitEvaluator, ArgmaxStabilityExitEvaluator (lines 2128-2240); generate_with_adaptive_compute (line 1552)",
            "license": "Apache-2.0",
            "adapted": "Same exit values, sentinels, 'auto' thresholds and comparisons (strict < for differences, >= for the argmax counter); "
                       "the 1-based step count and the never-exit value R come from generate_with_adaptive_compute.",
            "simplifications": "Vectorized over all steps instead of evaluated one iteration at a time. Each token is its own sequence of length 1, so "
                               "upstream's reduce over the sequence is a no-op. Divergence: upstream's latent-diff compares step 1 with the random "
                               "initial state from initialize_state; the exercise has no initial state, so latent-diff cannot fire at step 1. "
                               "The KL uniform start uses V from the logits instead of padded_vocab_size.",
        },
        {
            "kind": "paper",
            "url": "https://arxiv.org/abs/2502.05171",
            "section": "Geiping et al. 2025, Section 6.1 Zero-Shot Adaptive Compute at Test-Time, Figure 10",
            "note": "Exits when the KL divergence between two successive steps falls below 5e-4.",
        },
        {
            "kind": "paper",
            "url": "https://huggingface.co/tomg-group-umd/huginn-0125",
            "section": "Model card, Per-Token Adaptive Compute",
            "note": "Lists the four criteria and suggests kl for interesting exits and argmax-stability for conservative exits.",
        },
    ],
    "tests": [
        {"name": "KL exits start from a uniform distribution", "behavior": "rl.kl_estimator", "code": r"""
import torch
logits = torch.tensor([
    [[0.0, 0.0], [2.0, 0.0]],  # step 1
    [[0.0, 0.0], [2.0, 0.0]],  # step 2
    [[1.0, 0.0], [2.0, 0.0]],  # step 3
])
latents = torch.zeros(3, 2, 4)
out = {fn}(logits, latents, "kl", "auto")
# token 0 is uniform at step 1, so KL(uniform || uniform) = 0 fires at once;
# token 1 differs from uniform at step 1 and repeats itself at step 2.
assert out.dtype == torch.long and out.tolist() == [1, 2], out
"""},
        {"name": "Argmax stability counts consecutive repeats", "behavior": "state.invariant", "code": r"""
import torch
seqs = [[0, 0, 0, 0, 0], [0, 1, 1, 1, 2], [0, 1, 0, 1, 0]]
logits = torch.stack([torch.nn.functional.one_hot(torch.tensor([s[t] for s in seqs]), 3).float() * 5 for t in range(5)])
out = {fn}(logits, torch.zeros(5, 3, 2), "argmax-stability", 2)
# counters: [0,1,2,..] -> step 3; [0,0,1,2,..] -> step 4; never -> R = 5
assert out.tolist() == [3, 4, 5], out
"""},
        {"name": "Latent-diff uses the relative change and skips step 1", "behavior": "numerics.stability", "code": r"""
import torch
latents = torch.tensor([
    [[1.0, 0.0], [1.0, 0.0]],
    [[1.0, 0.0], [3.0, 0.0]],
    [[1.0, 0.0], [3.2, 0.0]],
])
out = {fn}(torch.zeros(3, 2, 5), latents, "latent-diff", 0.1)
# token 1: step 2 changes by 2 / 3, step 3 by 0.2 / 3.2 = 0.0625 < 0.1
assert out.tolist() == [2, 3], out
"""},
        {"name": "Matches a step-by-step evaluation on random trajectories", "visibility": "unshown", "behavior": "state.invariant", "failure_message": "Compare each step with the one before it using the exact formulas in the requirement, and return the first 1-based step that fires or R.", "code": r"""
import math, torch
def slow(logits, latents, crit, thr):
    R, N, V = logits.shape
    out = []
    for n in range(N):
        step, prev, count, prev_arg = R, None, 0, -1
        for r in range(R):
            x = logits[r, n]
            if crit == "kl":
                lp = torch.log_softmax(x, -1)
                q = torch.full_like(lp, math.log(1.0 / V)) if prev is None else prev
                fire = (q.exp() * (q - lp)).sum().item() < thr
                prev = lp
            elif crit == "entropy-diff":
                p = torch.softmax(x, -1)
                H = -(p * torch.log(p + 1e-10)).sum().item()
                fire = abs(H - (100.0 if prev is None else prev)) < thr
                prev = H
            elif crit == "latent-diff":
                s = latents[r, n]
                fire = prev is not None and ((s - prev).norm() / s.norm()).item() < thr
                prev = s
            else:
                a = int(x.argmax())
                count = count + 1 if a == prev_arg else 0
                prev_arg = a
                fire = count >= thr
            if fire:
                step = r + 1
                break
        out.append(step)
    return out
for seed, crit, thr in ((0, "kl", 1e-3), (1, "kl", 2e-2), (2, "entropy-diff", 1e-3), (3, "entropy-diff", 3e-2),
                        (4, "latent-diff", 0.03), (5, "latent-diff", 0.2), (6, "argmax-stability", 3), (7, "argmax-stability", 1)):
    g = torch.Generator().manual_seed(seed)
    R, N, V, h = 12, 24, 7, 6
    decay = 0.6 ** torch.arange(R, dtype=torch.float64).view(R, 1, 1)
    scale = torch.rand(1, N, 1, generator=g, dtype=torch.float64) * 4
    logits = torch.randn(1, N, V, generator=g, dtype=torch.float64) * 2 + scale * decay * torch.randn(R, N, V, generator=g, dtype=torch.float64)
    latents = torch.randn(1, N, h, generator=g, dtype=torch.float64) + scale * decay * torch.randn(R, N, h, generator=g, dtype=torch.float64)
    got = {fn}(logits, latents, crit, thr)
    want = slow(logits, latents, crit, thr)
    assert got.tolist() == want, (crit, thr, got.tolist(), want)
    assert len(set(want)) > 2, "test data should spread the exit steps"
"""},
        {"name": "KL runs from the previous step to the current one", "visibility": "unshown", "behavior": "rl.kl_estimator", "failure_message": "Use the previous step's distribution as the reference: sum(exp(p_prev) * (p_prev - p_cur)), i.e. F.kl_div(p_cur, p_prev, log_target=True).", "code": r"""
import torch
P = torch.tensor([0.99, 0.01]).log()
Q = torch.tensor([0.9, 0.1]).log()
# KL(P || Q) ~ 0.071 and KL(Q || P) ~ 0.144; the threshold 0.1 sits between them.
logits = torch.stack([torch.stack([P, Q]), torch.stack([Q, P]), torch.stack([Q, P])])
out = {fn}(logits, torch.zeros(3, 2, 3), "kl", 0.1)
assert out.tolist() == [2, 3], out
"""},
        {"name": "Entropy-diff starts from the 100.0 sentinel", "visibility": "unshown", "behavior": "edge.empty_or_boundary", "failure_message": "Compare step 1's entropy with the sentinel 100.0: a near-zero entropy must not fire at step 1, but a threshold above 100 - H_1 must.", "code": r"""
import torch
peaked = torch.tensor([30.0, 0.0, 0.0, 0.0])
logits = peaked.expand(3, 2, 4).clone()
out = {fn}(logits, torch.zeros(3, 2, 2), "entropy-diff", "auto")
assert out.tolist() == [2, 2], out  # H_1 ~ 0 is far from 100, step 2 repeats it
flat = torch.zeros(3, 1, 4)  # H = log 4 ~ 1.386, |1.386 - 100| ~ 98.6
assert {fn}(flat, torch.zeros(3, 1, 2), "entropy-diff", 150.0).tolist() == [1]
assert {fn}(flat, torch.zeros(3, 1, 2), "entropy-diff", 98.0).tolist() == [2]
"""},
        {"name": "KL can fire at step 1 against the uniform start", "visibility": "unshown", "behavior": "edge.empty_or_boundary", "failure_message": "Before step 1 the previous distribution is uniform (log(1/V) everywhere), so a uniform first prediction fires at step 1.", "code": r"""
import torch
logits = torch.zeros(4, 3, 6)
logits[:, 1, 0] = 9.0
out = {fn}(logits, torch.zeros(4, 3, 2), "kl", 1e-6)
assert out.tolist() == [1, 2, 1], out
"""},
        {"name": "Latent-diff is scale invariant", "visibility": "unshown", "behavior": "numerics.stability", "failure_message": "Divide the change norm(s_t - s_(t-1)) by norm(s_t) so the criterion does not depend on the scale of the latents.", "code": r"""
import torch
base = torch.tensor([
    [[1.0, 0.0], [1.0, 0.0]],
    [[1.02, 0.0], [2.0, 0.0]],
    [[1.02, 0.0], [2.0, 0.0]],
], dtype=torch.float64)
for scale in (1.0, 1000.0, 0.001):
    out = {fn}(torch.zeros(3, 2, 3, dtype=torch.float64), base * scale, "latent-diff", "auto")
    assert out.tolist() == [2, 3], (scale, out)
"""},
        {"name": "Latent-diff divides by the current state", "visibility": "unshown", "behavior": "numerics.stability", "failure_message": "The denominator is the norm of the current state s_t, not of the previous state.", "code": r"""
import torch
latents = torch.tensor([
    [[1.0, 0.0], [2.0, 0.0]],
    [[2.0, 0.0], [1.0, 0.0]],
    [[2.0, 0.0], [1.0, 0.0]],
])
# token 0: 1 / 2 = 0.5 < 0.75 at step 2; token 1: 1 / 1 = 1 at step 2, 0 at step 3
assert {fn}(torch.zeros(3, 2, 3), latents, "latent-diff", 0.75).tolist() == [2, 3]
"""},
        {"name": "Latent-diff never fires at step 1", "visibility": "unshown", "behavior": "edge.empty_or_boundary", "failure_message": "There is no latent state before step 1, so latent-diff cannot fire there even with a huge threshold.", "code": r"""
import torch
g = torch.Generator().manual_seed(3)
latents = torch.randn(3, 5, 4, generator=g)
assert {fn}(torch.zeros(3, 5, 2), latents, "latent-diff", 10.0).tolist() == [2] * 5
assert {fn}(torch.zeros(1, 5, 2), latents[:1], "latent-diff", 10.0).tolist() == [1] * 5  # never fires -> R = 1
"""},
        {"name": "Exact ties at the threshold", "visibility": "unshown", "behavior": "metrics.ties", "failure_message": "Difference criteria fire only when the value is strictly below the threshold; argmax-stability fires when the counter is greater than or equal to it.", "code": r"""
import torch
g = torch.Generator().manual_seed(0)
same = torch.randn(1, 2, 5, generator=g).expand(4, 2, 5).clone()
z = torch.zeros(4, 2, 3)
assert {fn}(same, z, "kl", 0.0).tolist() == [4, 4]
assert {fn}(same, z, "entropy-diff", 0.0).tolist() == [4, 4]
lat = torch.tensor([[[1.0, 0.0]], [[2.0, 0.0]], [[4.0, 0.0]]])  # relative change exactly 0.5 twice
assert {fn}(torch.zeros(3, 1, 2), lat, "latent-diff", 0.5).tolist() == [3]
assert {fn}(torch.zeros(3, 1, 2), lat, "latent-diff", 0.5000001).tolist() == [2]
arg = torch.zeros(6, 1, 3)
arg[..., 1] = 1.0  # argmax 1 at every step: counters 0, 1, 2, 3, ...
assert {fn}(arg, torch.zeros(6, 1, 2), "argmax-stability", 3).tolist() == [4]
assert {fn}(arg, torch.zeros(6, 1, 2), "argmax-stability", 0).tolist() == [1]
"""},
        {"name": "'auto' picks each criterion's default threshold", "visibility": "unshown", "behavior": "contract.signature", "failure_message": "Use the upstream defaults for 'auto': kl 1e-3, entropy-diff 1e-3, latent-diff 0.03, argmax-stability 5.", "code": r"""
import torch
g = torch.Generator().manual_seed(42)
R, N, V, h = 14, 64, 6, 5
decay = 0.55 ** torch.arange(R, dtype=torch.float64).view(R, 1, 1)
scale = torch.rand(1, N, 1, generator=g, dtype=torch.float64) * 5
logits = torch.randn(1, N, V, generator=g, dtype=torch.float64) + scale * decay * torch.randn(R, N, V, generator=g, dtype=torch.float64)
latents = torch.randn(1, N, h, generator=g, dtype=torch.float64) + scale * decay * torch.randn(R, N, h, generator=g, dtype=torch.float64)
for crit, default, others in (("kl", 1e-3, (5e-4, 2e-3)), ("entropy-diff", 1e-3, (5e-4, 3e-3)),
                              ("latent-diff", 0.03, (0.01, 0.05)), ("argmax-stability", 5, (4, 6))):
    auto = {fn}(logits, latents, crit, "auto")
    assert torch.equal(auto, {fn}(logits, latents, crit, default)), crit
    for other in others:
        assert not torch.equal(auto, {fn}(logits, latents, crit, other)), (crit, other)
"""},
        {"name": "Tokens that never exit return R as a long tensor", "visibility": "unshown", "behavior": "tensor.dtype_device", "failure_message": "Return a torch.long tensor of shape (N,) whose value is R for every token whose criterion never fires.", "code": r"""
import torch
g = torch.Generator().manual_seed(9)
logits = torch.randn(7, 4, 5, generator=g, dtype=torch.float64) * 10
latents = torch.randn(7, 4, 3, generator=g, dtype=torch.float64)
for crit in ("kl", "entropy-diff", "latent-diff"):
    out = {fn}(logits, latents, crit, 0.0)
    assert out.dtype == torch.long and out.shape == (4,) and out.tolist() == [7] * 4, (crit, out)
out = {fn}(logits, latents, "argmax-stability", 100)
assert out.dtype == torch.long and out.tolist() == [7] * 4, out
"""},
        {"name": "Tied logits take the lowest argmax index", "visibility": "unshown", "behavior": "metrics.ties", "failure_message": "Use argmax over the vocabulary, which returns the lowest index among tied maxima.", "code": r"""
import torch
logits = torch.tensor([[[1.0, 1.0, 0.0]], [[1.0, 0.0, 1.0]], [[0.0, 1.0, 1.0]]])
# argmaxes 0, 0, 1 -> counters 0, 1, 0
assert {fn}(logits, torch.zeros(3, 1, 2), "argmax-stability", 1).tolist() == [2]
"""},
        {"name": "Rejects bad arguments", "visibility": "unshown", "behavior": "contract.signature", "failure_message": "Raise ValueError for an unknown criterion, non-3-D inputs, R < 1, or inputs that disagree on R or N.", "code": r"""
import torch
good_l, good_s = torch.zeros(3, 2, 4), torch.zeros(3, 2, 5)
bad = [
    (good_l, good_s, "cosine"),
    (good_l, good_s, "KL"),
    (torch.zeros(3, 4), good_s, "kl"),
    (good_l, torch.zeros(3, 2), "kl"),
    (good_l, torch.zeros(4, 2, 5), "kl"),
    (good_l, torch.zeros(3, 3, 5), "latent-diff"),
    (torch.zeros(0, 2, 4), torch.zeros(0, 2, 5), "kl"),
]
for logits, latents, crit in bad:
    try:
        {fn}(logits, latents, crit, "auto")
    except ValueError:
        pass
    else:
        raise AssertionError(f"accepted {tuple(logits.shape)}, {tuple(latents.shape)}, {crit}")
"""},
    ],
    "solution": '''import math

import torch
import torch.nn.functional as F

AUTO_THRESHOLDS = {"kl": 1e-3, "entropy-diff": 1e-3, "latent-diff": 0.03, "argmax-stability": 5}


def first_exit_step(logits_per_step, latents_per_step, criterion, threshold="auto"):
    if criterion not in AUTO_THRESHOLDS:
        raise ValueError(f"unknown criterion {criterion!r}")
    if logits_per_step.ndim != 3 or latents_per_step.ndim != 3:
        raise ValueError("logits_per_step and latents_per_step must be 3-D")
    R, N, V = logits_per_step.shape
    if R < 1 or latents_per_step.shape[:2] != (R, N):
        raise ValueError("inputs must share R >= 1 and N")
    if isinstance(threshold, str) and threshold == "auto":
        threshold = AUTO_THRESHOLDS[criterion]

    if criterion == "argmax-stability":
        argmax = logits_per_step.argmax(dim=-1)  # (R, N)
        count = torch.zeros(N, dtype=torch.long, device=argmax.device)
        counts = [count]
        for r in range(1, R):
            count = torch.where(argmax[r] == argmax[r - 1], count + 1, torch.zeros_like(count))
            counts.append(count)
        fires = torch.stack(counts) >= int(threshold)
    else:
        if criterion == "kl":
            log_p = F.log_softmax(logits_per_step, dim=-1)
            uniform = torch.full_like(log_p[:1], math.log(1.0 / V))
            prev = torch.cat([uniform, log_p[:-1]])
            values = F.kl_div(log_p, prev, reduction="none", log_target=True).sum(dim=-1)
        elif criterion == "entropy-diff":
            p = F.softmax(logits_per_step, dim=-1)
            entropy = -torch.sum(p * torch.log(p + 1e-10), dim=-1)  # (R, N)
            prev = torch.cat([torch.full_like(entropy[:1], 100.0), entropy[:-1]])
            values = (entropy - prev).abs()
        else:
            s = latents_per_step
            change = (s[1:] - s[:-1]).norm(dim=-1) / s[1:].norm(dim=-1)
            values = torch.cat([torch.full_like(s[:1, :, 0], float("inf")), change])
        fires = values < float(threshold)

    steps = torch.arange(1, R + 1, device=fires.device).view(R, 1).expand(R, N)
    never = torch.full_like(steps, R)
    return torch.where(fires, steps, never).min(dim=0).values
''',
    "interview_questions": interview(
        concept=[
            "Why can a recurrent-depth model like Huginn exit early per token without training an exit head?",
            "What do the four exit criteria (kl, entropy-diff, latent-diff, argmax-stability) each measure, and which ones need the coda at every step?",
        ],
        deep_dive=[
            "In F.kl_div(log_p_cur, log_p_prev, log_target=True), which distribution is the reference, and why might the direction matter for an exit test?",
            "What stands in for step 0 in each criterion, and how does that choice decide whether a token can exit at step 1?",
            "Why does batched adaptive-compute decoding make early exits look better than they are, and what does upstream recommend for benchmarking?",
        ],
        tradeoffs=[
            "Thresholding a difference between steps versus learning a halting probability as in ACT or PonderNet?",
            "A conservative criterion such as argmax-stability versus an aggressive one such as kl: what do you trade in compute and output quality?",
        ],
    ),
}
