"""Flow-SDE log-probabilities for a flow-matching action head: one SDE step or the whole denoising chain."""

from ._interview import interview

# Shared test fixtures: a nonlinear velocity field and an independent step-by-step oracle.
# The oracle uses the closed-form drift (Euler step plus score correction) with plain-float
# schedules and a per-sample loop, a different route from the weighted-prediction form.
_ORACLE = r"""
import math
import torch

def make_field(H, seed, log=None):
    g = torch.Generator().manual_seed(seed)
    W = torch.randn(H, H, generator=g, dtype=torch.float64) / H
    bias = torch.randn(1, H, 1, generator=g, dtype=torch.float64) * 0.2
    def v(x, t):
        if log is not None:
            log.append((x.detach().clone(), t.detach().clone()))
        return torch.einsum("hk,bkd->bhd", W.to(x.dtype), torch.tanh(x)) * (1 + t[:, None, None]) - 0.4 * x + bias.to(x.dtype)
    return v

def step_oracle(x, v, k, K, a):
    t = 1 - k / K
    d = 1 / K
    den = d if k == 0 else 1 - t
    sig = a * math.sqrt(t / den)
    mean = x - v * d - (sig * sig * d / (2 * t)) * (x + (1 - t) * v)
    return mean, math.sqrt(d) * sig

def normal_lp(x, mean, std):
    if std == 0:
        return torch.zeros_like(x)
    return -math.log(std) - 0.5 * math.log(2 * math.pi) - 0.5 * ((x - mean) / std) ** 2

def oracle(chain, v, idx, a, joint):
    B, K = chain.shape[0], chain.shape[1] - 1
    out = []
    for b in range(B):
        terms = [normal_lp(chain[b, 0], 0.0, 1.0)] if joint else []
        for k in (range(K) if joint else [int(idx[b])]):
            x = chain[b:b + 1, k]
            vel = v(x, torch.tensor([1 - k / K], dtype=chain.dtype))
            mean, std = step_oracle(x, vel, k, K, a)
            terms.append(normal_lp(chain[b:b + 1, k + 1], mean, std)[0])
        out.append(sum(terms) / len(terms))
    return torch.stack(out)

def sampled_chain(B, K, H, D, v, a, seed, sde_index=None):
    # Roll out the hybrid sampler: SDE at sde_index[b] (or every step when None), ODE elsewhere.
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, H, D, generator=g, dtype=torch.float64)
    states, eps_used = [x], []
    for k in range(K):
        nxt = []
        for b in range(B):
            xb = x[b:b + 1]
            vel = v(xb, torch.tensor([1 - k / K], dtype=torch.float64))
            sde = sde_index is None or int(sde_index[b]) == k
            mean, std = step_oracle(xb, vel, k, K, a if sde else 0.0)
            eps = torch.randn(xb.shape, generator=g, dtype=torch.float64)
            if sde:
                eps_used.append((b, k, eps[0], std))
            nxt.append(mean + std * eps)
        x = torch.cat(nxt)
        states.append(x)
    return torch.stack(states, dim=1), eps_used
"""

TASK = {
    "title": "Flow-SDE Chain Log-Probability for a Flow Action Head",
    "difficulty": "Hard",
    "version": 1,
    "function_name": "flow_chain_logprob",
    "description_en": r"""PPO needs `log pi(action | obs)`, and a flow-matching action head such as pi0 has none: Euler denoising is deterministic. Implement the Flow-SDE fix used by pi_RL in RLinf. Turn a denoising step into a Gaussian transition, then score a stored denoising chain under it.

**Signatures:** define both functions at the top level of your code.
- `flow_sde_mean_std(x_t, v_t, idx, num_steps, method, noise_level) -> (mean, std)`
- `flow_chain_logprob(chain, velocity_fn, sde_index, noise_level, joint=False, level="chunk_level", action_chunk=None, action_dim=None) -> Tensor`

**Time grid.** The noise sits at `t = 1` and the action at `t = 0`. With `K = num_steps`, the grid is `t_k = 1 - k / K` for `k = 0..K`. Step `k` moves a sample from `t_k` to `t_{k+1}`, so every step has `delta = 1 / K`.

**flow_sde_mean_std:**
- `x_t` and `v_t` are float tensors of shape `(B, H, D)`: the sample at time `t_idx` and the network's velocity there. `idx` is an integer tensor of shape `(B,)`. Each sample uses its own step, `0 <= idx[b] <= K - 1`.
- Let `t = t_idx`, `t_next = t_{idx+1}`, `action_pred = x_t - t * v_t` and `noise_pred = x_t + (1 - t) * v_t`.
- `"flow_ode"`: `mean = action_pred * (1 - t_next) + noise_pred * t_next`, and `std` is all zeros.
- `"flow_sde"`: `sigma_k = noise_level * sqrt(t_k / (1 - s_k))`, where `s_k = t_k` except at `t_k = 1`. There `s_0 = t_1`, the next grid time, so the denominator stays finite. The numerator keeps `t_0 = 1`.
- `"flow_sde"` then uses `mean = action_pred * (1 - t_next) + noise_pred * (t_next - sigma^2 * delta / (2 * t))` and `std = sqrt(delta) * sigma`, with `sigma = sigma_idx`.
- `mean` and `std` both have the shape and dtype of `x_t`.
- Raise `ValueError` for an unknown `method`, a `num_steps` that is not a positive integer, an `x_t` that is not 3-D, a `v_t` of a different shape, or an `idx` that is not `(B,)` or is out of range.

**flow_chain_logprob:**
- `chain` has shape `(B, K + 1, H, D)` with `K >= 1`. `chain[:, 0]` is the initial noise at `t = 1`, `chain[:, k]` is the sample at `t_k`, and `chain[:, K]` is the action.
- `velocity_fn(x, t)` takes `x` of shape `(B, H, D)` and `t` of shape `(B,)` holding grid times in `chain`'s dtype. It returns a velocity shaped like `x`.
- The per-element Gaussian log-density is `-log(std) - 0.5 * log(2 * pi) - 0.5 * ((x - mean) / std)^2`. It is exactly 0 wherever `std == 0`.
- `joint=False` scores one SDE step per sample. `sde_index` is an integer `(B,)` tensor with `0 <= sde_index[b] <= K - 1`. Call `velocity_fn` exactly once, with `x[b] = chain[b, i_b]` and `t[b] = t_{i_b}`. Get the mean and std from `flow_sde_mean_std(..., "flow_sde", noise_level)` and score `chain[b, i_b + 1]`.
- `joint=True` scores the whole chain and ignores `sde_index`. The terms are the standard-normal log-density of `chain[:, 0]`, plus, for each `k = 0..K-1`, the `"flow_sde"` log-density of `chain[:, k + 1]` given `chain[:, k]`. Call `velocity_fn` `K` times in order `k = 0..K-1`, with `chain[:, k]` and every time equal to `t_k`.
- In joint mode, return the **mean** of these `K + 1` per-element terms, not their sum.
- Keep only the first `action_chunk` positions along `H` and the first `action_dim` entries along `D` before reducing. `None` keeps them all.
- `level="token_level"` returns `(B, h, d)`. `"action_level"` sums over `d` and returns `(B, h)`. `"chunk_level"` sums over `h` and `d` and returns `(B,)`. Never sum over the batch.
- The result has `chain`'s dtype. It stays differentiable through `velocity_fn`: no `detach` and no `no_grad`.
- Never modify `chain` or `sde_index` in place.
- Raise `ValueError` for an unknown `level`, a `chain` that is not 4-D or has `K < 1`, and, when `joint` is false, a missing `sde_index`, one that is not `(B,)`, or one that is out of range. Also raise it when `action_chunk` is not an integer in `[1, H]` or `action_dim` is not an integer in `[1, D]`.

**Example:**

    mean, std = flow_sde_mean_std(torch.ones(1, 1, 1), torch.full((1, 1, 1), 2.0), torch.tensor([0]), 2, "flow_sde", 0.5)
    # t = 1, t_next = 0.5, sigma^2 = 0.25 * 1 / (1 - 0.5) = 0.5
    # action_pred = -1, noise_pred = 1: mean = -0.5 + (0.5 - 0.125) = -0.125, std = sqrt(0.5 * 0.5) = 0.5
    # with "flow_ode": mean = 0.0, std = 0.0
    chain = torch.tensor([1.0, 0.375, 0.0]).reshape(1, 3, 1, 1)
    flow_chain_logprob(chain, lambda x, t: torch.full_like(x, 2.0), torch.tensor([0]), 0.5)
    # log-density of 0.375 under N(-0.125, 0.5^2) = log 2 - 0.5 * log(2 * pi) - 0.5 = -0.7258, shape (1,)

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why a chain.** A deterministic Euler step is a point mass, so the action a flow head produces has no tractable density for a policy-gradient ratio. Adding Gaussian noise to a step makes that step a Gaussian transition with a closed-form log-density. ReinFlow (Eq. 6 and 7) and pi_RL's Flow-Noise (Eq. 5) write the log-probability of the whole denoising chain as the log-density of the initial noise plus the sum of the per-step terms. Flow-Noise learns the per-step std with a noise network. Flow-SDE, this exercise, fixes it with a schedule.

**Flow-SDE.** pi_RL Section 4.2.1, following Flow-GRPO, converts the probability-flow ODE into an SDE meant to keep the same marginals. It uses the schedule `sigma = a * sqrt(tau / (1 - tau))` and a Gaussian transition with variance `sigma^2 * delta` (Eq. 8 and 9). The `- sigma^2 * delta / (2 t)` weight on the predicted noise is the score-based drift correction. Expanded, the mean is `x - v * delta - sigma^2 * delta / (2 t) * (x + (1 - t) * v)`: the plain Euler step plus that correction. pi_RL writes its flow path with noise at `tau = 0` (Section 3.2). RLinf's code puts the noise at `t = 1`, and this exercise follows the code.

**The t = 1 guard.** The schedule divides by `1 - t`, which is zero at the first step. RLinf's `sample_mean_var` substitutes the next grid time in the denominator only. Since `1 - t_1 = 1 / K`, the first step gets `sigma = a * sqrt(K)` and a standard deviation of exactly `a`.

**Hybrid ODE-SDE rollouts.** pi_RL Section 4.2.3 draws one denoising step per action at random to run as SDE and runs the rest as ODE steps. RLinf's rollout draws that index once per call with `random.randint`, shares it across the batch and stores it per sample with the whole chain. At training time it recomputes only that one step. Its `ignore_last` option keeps the last step out of the draw.

**Joint mode.** RLinf's `joint_logprob` runs every step as SDE and averages the `K + 1` terms with `log_probs.mean(dim=1)`. The result is the ReinFlow chain log-probability divided by `K + 1`. This scales the PPO log-ratio, and with it how quickly the clip range is reached.

**Granularity.** RLinf's `preprocess_loss_inputs` reduces per-element log-probabilities by `logprob_type`: `token_level` keeps each action dimension, `action_level` sums over dimensions and `chunk_level` sums over the chunk. pi_RL treats a chunk as one macro-step whose reward is the sum over the chunk (Section 4.3.1). Before reducing, RLinf slices to the executed chunk length and the environment's action width, because openpi's pi0 predicts a 50-step horizon over 32 padded action dimensions.

**Noise level.** pi_RL Section 5.4 reports that a larger noise level lowers success during stochastic rollouts. A very small one (`a = 0.2`) trained unstably, with a much higher clip fraction. With `K = 1`, training-time success dropped sharply, which the authors attribute to ODE-to-SDE discretization error.""",
    "advisory_prerequisites": ["pi0_flow_matching_chunk", "ppo_clipped_policy_loss"],
    "hints": [
        {"level": 1, "kind": "questions", "content": "Which grid time holds the noise, and what is delta for K steps? Where does the sigma schedule divide by zero, and which single quantity is replaced there? With a different step index per sample, how do you gather each sample's current and next chain state in one indexing operation? In joint mode, how many terms are averaged, and which one does not come from a denoising step?"},
        {"level": 2, "kind": "analysis", "content": "Build the grid 1 - k / K in the input dtype. In the step function, gather t and t_next by idx, form the predicted action and the predicted noise, and weight them by 1 - t_next and t_next. For the SDE, build a sigma table whose denominator uses t_1 where t equals 1, subtract sigma squared times delta over 2t from the noise weight, and set std to sqrt(delta) times sigma. For the chain, write a Gaussian log-density that swaps std == 0 for a safe value and zeroes those entries. In single-step mode, gather chain rows with arange(B) and the index, call the velocity once with per-sample times, and score the next state. In joint mode, stack the standard-normal term with the K step terms and average. Slice to action_chunk and action_dim, then sum the axes the level asks for."},
    ],
    "model_connections": [
        "RLinf's Pi0RL stores the full denoising chain and the chosen SDE index at rollout. Its default_forward recomputes one step's log-probability, or every step's when joint_logprob is set, through rl_sampler.sample_mean_var and gaussian_logprob.",
        "pi_RL fine-tunes pi0 and pi0.5 in RLinf with Flow-SDE and Flow-Noise. On few-shot LIBERO, pi0 rises from 57.6% (SFT) to 96.1% with Flow-SDE and 97.6% with Flow-Noise (Appendix C.1).",
        "ReinFlow replaces the fixed schedule with a learned noise network and scores the whole chain (Eq. 7). RLinf's flow_noise method keeps the ODE mean and takes the std from such a head.",
        "The step log-probability feeds the clipped PPO ratio exactly as a token log-probability does in LLM RL. Only the policy's density changes.",
    ],
    "pro_con_analysis": {
        "pros": [
            "Every Gaussian transition has an exact closed-form log-density, so PPO's ratio needs no divergence estimator, even with few denoising steps.",
            "Scoring a single random SDE step needs one network call at training time, against K for the joint chain.",
            "Evaluation can still use the deterministic ODE sampler, because the injected noise only shapes exploration during training.",
        ],
        "cons": [
            "The noise level is a sensitive hyperparameter: too much noise degrades rollouts, and too little gives large gradients and unstable training (pi_RL Section 5.4).",
            "The Euler-discretized SDE only approximates the ODE's marginals, and the error grows as K shrinks.",
            "The chain must be stored per rollout step (K + 1 copies of the chunk), which costs memory in large on-policy batches.",
        ],
    },
    "sources": [
        {
            "kind": "code",
            "url": "https://github.com/RLinf/RLinf",
            "commit": "0e9438dee0088be7673a1fb4789018aea168db1a",
            "path": "rlinf/models/embodiment/openpi/sampling/rl_sampler.py",
            "symbol": "get_timesteps, sample_mean_var, gaussian_logprob",
            "license": "Apache-2.0",
            "adapted": "The noise-at-one grid, the flow_ode and flow_sde step mean and std with the t = 1 denominator guard, and the Gaussian log-density that is zero where std is zero.",
            "simplifications": "Only flow_ode and flow_sde are kept (flow_cps and flow_noise are dropped); "
                               "the grid is computed as 1 - k / K in the input dtype rather than a float32 linspace cast to it, which differs only by float rounding; "
                               "unknown methods raise ValueError instead of NotImplementedError, and shapes and indices are validated.",
        },
        {
            "kind": "code",
            "url": "https://github.com/RLinf/RLinf",
            "commit": "0e9438dee0088be7673a1fb4789018aea168db1a",
            "path": "rlinf/models/embodiment/openpi/tasks/rl.py",
            "symbol": "Pi0RL.default_forward",
            "license": "Apache-2.0",
            "adapted": "The log-probability recompute from a stored chain: one SDE step at a per-sample index, or the standard-normal term plus every step averaged when joint_logprob is set, then the slice to action_chunk and action_env_dim.",
            "simplifications": "The pi0 network is replaced by a velocity_fn(x, t) callable; "
                               "the chain keeps its dtype instead of being cast to float32; "
                               "the joint chain is passed as joint=True rather than an index tensor of arange rows; "
                               "values, entropy and the flow_noise head are dropped.",
        },
        {
            "kind": "code",
            "url": "https://github.com/RLinf/RLinf",
            "commit": "0e9438dee0088be7673a1fb4789018aea168db1a",
            "path": "rlinf/algorithms/utils.py",
            "symbol": "preprocess_loss_inputs",
            "license": "Apache-2.0",
            "adapted": "The token_level, action_level and chunk_level reductions of the per-element log-probabilities.",
            "simplifications": "Only the logprobs reshape and sum are kept; old_logprobs, advantages, masks, values and the sequence_token_level branch are dropped; "
                               "the input is already (B, h, d), so no single_action_dim reshape is needed.",
        },
        {
            "kind": "paper",
            "url": "https://arxiv.org/abs/2510.25889",
            "section": "pi_RL, Section 4.1.2 Eq. 5 (chain log-likelihood), Section 4.2.1 Eq. 8-9 (Flow-SDE mean and variance), Section 4.2.3 (hybrid ODE-SDE sampling), Section 4.3.1 (chunk-level reward), Section 5.4 (noise level and denoise steps)",
            "note": "Flow-SDE and Flow-Noise for pi0 and pi0.5, trained with PPO in RLinf.",
        },
        {
            "kind": "paper",
            "url": "https://arxiv.org/abs/2505.22094",
            "section": "ReinFlow, Section 4.1, Eq. 6-7",
            "note": "Noise injection turns the flow into a discrete-time Markov process whose joint log-probability is the initial-noise term plus the sum of Gaussian step terms.",
        },
    ],
    "tests": [
        {"name": "One step's mean and std", "behavior": "rl.logprob", "code": r"""
import torch
step = {fn}.__globals__["flow_sde_mean_std"]
x = torch.ones(1, 1, 1); v = torch.full((1, 1, 1), 2.0)
mean, std = step(x, v, torch.tensor([0]), 2, "flow_sde", 0.5)
assert mean.shape == (1, 1, 1) and std.shape == (1, 1, 1), (mean.shape, std.shape)
assert torch.allclose(mean, torch.tensor(-0.125)), mean
assert torch.allclose(std, torch.tensor(0.5)), std
mean, std = step(x, v, torch.tensor([0]), 2, "flow_ode", 0.5)
assert torch.allclose(mean, torch.tensor(0.0)) and torch.all(std == 0), (mean, std)
"""},
        {"name": "One SDE step scored inside a chain", "behavior": "rl.logprob", "code": r"""
import math, torch
chain = torch.tensor([1.0, 0.375, 0.0]).reshape(1, 3, 1, 1)
out = {fn}(chain, lambda x, t: torch.full_like(x, 2.0), torch.tensor([0]), 0.5)
want = math.log(2.0) - 0.5 * math.log(2 * math.pi) - 0.5
assert out.shape == (1,), out.shape
assert abs(out.item() - want) < 1e-5, (out, want)
"""},
        {"name": "Joint mode averages K + 1 terms", "behavior": "rl.logprob", "code": r"""
import math, torch
chain = torch.zeros(1, 3, 1, 1)
out = {fn}(chain, lambda x, t: torch.zeros_like(x), None, 0.0, joint=True, level="token_level")
# noise_level 0 makes both step terms 0; only log N(0; 0, 1) remains, averaged over 3 terms.
want = -0.5 * math.log(2 * math.pi) / 3
assert out.shape == (1, 1, 1), out.shape
assert abs(out.item() - want) < 1e-6, (out, want)
"""},
        {"name": "Seeded SDE steps match the closed-form drift", "visibility": "unshown", "behavior": "rl.logprob",
         "failure_message": "The flow_sde mean is the Euler step x - v*delta minus sigma^2*delta/(2t) times the predicted noise x + (1-t)v; std is sqrt(delta)*sigma.",
         "code": _ORACLE + r"""
step = {fn}.__globals__["flow_sde_mean_std"]
for seed, K in ((0, 2), (1, 3), (2, 5), (3, 10)):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(4, 3, 2, generator=g, dtype=torch.float64)
    v = torch.randn(4, 3, 2, generator=g, dtype=torch.float64)
    idx = torch.randint(0, K, (4,), generator=g)
    idx[0], idx[1] = 0, K - 1
    a = 0.3 + 0.2 * seed
    mean, std = step(x, v, idx, K, "flow_sde", a)
    assert mean.shape == x.shape and std.shape == x.shape, (mean.shape, std.shape)
    for b in range(4):
        m, s = step_oracle(x[b], v[b], int(idx[b]), K, a)
        assert torch.allclose(mean[b], m, atol=1e-10), (seed, b, (mean[b] - m).abs().max())
        assert torch.allclose(std[b], torch.full_like(std[b], s), atol=1e-12), (seed, b, std[b], s)
"""},
        {"name": "The ODE step is the plain Euler step", "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "flow_ode returns mean = x_t - v_t / K for every step index and a zero std with the full shape of x_t.",
         "code": r"""
import torch
step = {fn}.__globals__["flow_sde_mean_std"]
for seed, K in ((4, 1), (5, 4), (6, 10)):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(3, 4, 2, generator=g, dtype=torch.float64)
    v = torch.randn(3, 4, 2, generator=g, dtype=torch.float64)
    idx = torch.randint(0, K, (3,), generator=g)
    mean, std = step(x, v, idx, K, "flow_ode", 0.9)
    assert torch.allclose(mean, x - v / K, atol=1e-12), (seed, (mean - (x - v / K)).abs().max())
    assert std.shape == x.shape and torch.all(std == 0), (seed, std.shape)
"""},
        {"name": "Sigma schedule and the t = 1 guard", "visibility": "unshown", "behavior": "edge.empty_or_boundary",
         "failure_message": "At t = 1 replace only the denominator's t by the next grid time t_1 (so the first std equals noise_level); keep t_k / (1 - t_k) elsewhere.",
         "code": r"""
import math, torch
step = {fn}.__globals__["flow_sde_mean_std"]
for K in (1, 2, 4, 10):
    for a in (0.5, 1.3):
        x = torch.zeros(K, 1, 1, dtype=torch.float64)
        _, std = step(x, x.clone(), torch.arange(K), K, "flow_sde", a)
        assert torch.isfinite(std).all(), (K, std.flatten())
        for k in range(K):
            t = 1 - k / K
            den = 1 / K if k == 0 else 1 - t
            want = math.sqrt(1 / K) * a * math.sqrt(t / den)
            assert abs(std[k, 0, 0].item() - want) < 1e-10, (K, a, k, std[k, 0, 0].item(), want)
        assert abs(std[0, 0, 0].item() - a) < 1e-10, (K, a, "first step std must equal noise_level")
"""},
        {"name": "Single-step log-probability with per-sample indices", "visibility": "unshown", "behavior": "rl.logprob",
         "failure_message": "Score chain[b, i_b + 1] given chain[b, i_b] at time t_{i_b}, using each sample's own index, never sde_index[0] for the whole batch.",
         "code": _ORACLE + r"""
for seed, K in ((7, 2), (8, 4), (9, 10)):
    g = torch.Generator().manual_seed(seed)
    chain = torch.randn(5, K + 1, 3, 2, generator=g, dtype=torch.float64) * 0.5
    idx = torch.randint(0, K, (5,), generator=g)
    idx[0], idx[1] = 0, K - 1
    v = make_field(3, seed)
    got = {fn}(chain, v, idx, 0.6, level="token_level")
    want = oracle(chain, v, idx, 0.6, joint=False)
    assert got.shape == (5, 3, 2), got.shape
    assert torch.allclose(got, want, atol=1e-8), (seed, (got - want).abs().max())
"""},
        {"name": "Likelihood of transitions drawn from the sampler", "visibility": "unshown", "behavior": "rl.logprob",
         "failure_message": "For x_next = mean + std * eps, the per-element log-density must be -log(std) - 0.5*log(2*pi) - 0.5*eps^2.",
         "code": _ORACLE + r"""
for seed, K in ((11, 3), (12, 5), (13, 10)):
    g = torch.Generator().manual_seed(seed + 100)
    idx = torch.randint(0, K, (4,), generator=g)
    v = make_field(4, seed)
    chain, eps_used = sampled_chain(4, K, 4, 3, v, 0.7, seed, idx)
    got = {fn}(chain, v, idx, 0.7, level="token_level")
    for b, k, eps, std in eps_used:
        want = -math.log(std) - 0.5 * math.log(2 * math.pi) - 0.5 * eps ** 2
        assert torch.allclose(got[b], want, atol=1e-8), (seed, b, k, (got[b] - want).abs().max())
"""},
        {"name": "Velocity calls: once per single step, K in order for the chain", "visibility": "unshown", "behavior": "events.ordering",
         "failure_message": "Single-step mode calls velocity_fn once with chain[b, i_b] and t_{i_b}; joint mode calls it K times with chain[:, k] and t_k, k = 0..K-1.",
         "code": _ORACLE + r"""
K = 4
g = torch.Generator().manual_seed(14)
chain = torch.randn(3, K + 1, 2, 2, generator=g, dtype=torch.float64)
idx = torch.tensor([2, 0, 3])
log = []
{fn}(chain, make_field(2, 0, log), idx, 0.5)
assert len(log) == 1, len(log)
x, t = log[0]
assert t.shape == (3,) and t.dtype == torch.float64, (t.shape, t.dtype)
assert torch.allclose(t, torch.tensor([0.5, 1.0, 0.25], dtype=torch.float64)), t
assert torch.equal(x, chain[torch.arange(3), idx]), "velocity must see the stored state at the chosen step"
log = []
{fn}(chain, make_field(2, 0, log), None, 0.5, joint=True)
assert len(log) == K, len(log)
for k, (x, t) in enumerate(log):
    assert torch.equal(x, chain[:, k]), k
    assert torch.allclose(t, torch.full((3,), 1 - k / K, dtype=torch.float64)), (k, t)
"""},
        {"name": "Joint mode is the mean of the noise term and every step", "visibility": "unshown", "behavior": "rl.logprob",
         "failure_message": "Joint mode averages K + 1 per-element terms: log N(chain[:, 0]; 0, 1) and the flow_sde density of each of the K steps.",
         "code": _ORACLE + r"""
for seed, K in ((15, 1), (16, 3), (17, 6)):
    v = make_field(3, seed)
    chain, _ = sampled_chain(3, K, 3, 2, v, 0.8, seed)
    got = {fn}(chain, v, None, 0.8, joint=True, level="token_level")
    want = oracle(chain, v, None, 0.8, joint=True)
    assert got.shape == (3, 3, 2), got.shape
    assert torch.allclose(got, want, atol=1e-8), (seed, K, (got - want).abs().max())
"""},
        {"name": "Token, action and chunk levels", "visibility": "unshown", "behavior": "tensor.shape",
         "failure_message": "token_level keeps (B, h, d); action_level sums the last axis to (B, h); chunk_level sums both to (B,). Never sum over the batch.",
         "code": _ORACLE + r"""
g = torch.Generator().manual_seed(18)
K = 3
chain = torch.randn(2, K + 1, 5, 3, generator=g, dtype=torch.float64)
idx = torch.tensor([1, 2])
v = make_field(5, 18)
tok = {fn}(chain, v, idx, 0.5, level="token_level")
act = {fn}(chain, v, idx, 0.5, level="action_level")
chk = {fn}(chain, v, idx, 0.5, level="chunk_level")
assert tok.shape == (2, 5, 3) and act.shape == (2, 5) and chk.shape == (2,), (tok.shape, act.shape, chk.shape)
assert torch.allclose(act, tok.sum(-1), atol=1e-10) and torch.allclose(chk, tok.sum((1, 2)), atol=1e-10)
for joint in (False, True):
    one = {fn}(chain[:1], v, idx[:1], 0.5, joint=joint)
    both = {fn}(chain, v, idx, 0.5, joint=joint)
    assert torch.allclose(both[0], one[0], atol=1e-10), "a sample's log-prob must not depend on the rest of the batch"
"""},
        {"name": "Truncation to the executed chunk and action width", "visibility": "unshown", "behavior": "rl.masking",
         "failure_message": "Keep only positions [:action_chunk] along H and entries [:action_dim] along D before summing; padded dims never count.",
         "code": _ORACLE + r"""
for joint in (False, True):
    g = torch.Generator().manual_seed(19)
    K = 4
    chain = torch.randn(3, K + 1, 6, 4, generator=g, dtype=torch.float64)
    idx = torch.tensor([0, 3, 1])
    v = make_field(6, 19)
    full = {fn}(chain, v, idx, 0.5, joint=joint, level="token_level")
    cut = {fn}(chain, v, idx, 0.5, joint=joint, level="token_level", action_chunk=4, action_dim=2)
    assert cut.shape == (3, 4, 2), cut.shape
    assert torch.allclose(cut, full[:, :4, :2], atol=1e-10), joint
    chk = {fn}(chain, v, idx, 0.5, joint=joint, action_chunk=2, action_dim=3)
    assert torch.allclose(chk, full[:, :2, :3].sum((1, 2)), atol=1e-10), joint
    act = {fn}(chain, v, idx, 0.5, joint=joint, level="action_level", action_dim=1)
    assert act.shape == (3, 6) and torch.allclose(act, full[..., 0], atol=1e-10), joint
"""},
        {"name": "Zero noise gives zero step log-density", "visibility": "unshown", "behavior": "edge.empty_or_boundary",
         "failure_message": "With noise_level 0 the SDE mean equals the ODE mean, std is 0, and a zero-std element contributes exactly 0 (not inf or a clamped value).",
         "code": _ORACLE + r"""
step = {fn}.__globals__["flow_sde_mean_std"]
g = torch.Generator().manual_seed(20)
x = torch.randn(2, 3, 2, generator=g, dtype=torch.float64)
vv = torch.randn(2, 3, 2, generator=g, dtype=torch.float64)
m_sde, s_sde = step(x, vv, torch.tensor([0, 2]), 4, "flow_sde", 0.0)
m_ode, _ = step(x, vv, torch.tensor([0, 2]), 4, "flow_ode", 0.0)
assert torch.allclose(m_sde, m_ode, atol=1e-12) and torch.all(s_sde == 0)
K = 4
chain = torch.randn(2, K + 1, 3, 2, generator=g, dtype=torch.float64)
v = make_field(3, 20)
out = {fn}(chain, v, torch.tensor([0, 3]), 0.0, level="token_level")
assert torch.equal(out, torch.zeros_like(out)), out
joint = {fn}(chain, v, None, 0.0, joint=True, level="token_level")
want = (-0.5 * math.log(2 * math.pi) - 0.5 * chain[:, 0] ** 2) / (K + 1)
assert torch.allclose(joint, want, atol=1e-12), (joint - want).abs().max()
"""},
        {"name": "Gradients reach the velocity network", "visibility": "unshown", "behavior": "gradient.flow",
         "failure_message": "Do not detach the velocity or compute under no_grad: PPO differentiates the step log-probability through velocity_fn.",
         "code": _ORACLE + r"""
g = torch.Generator().manual_seed(21)
K = 3
chain = torch.randn(3, K + 1, 2, 2, generator=g, dtype=torch.float64)
idx = torch.tensor([0, 1, 2])
for joint in (False, True):
    W = torch.randn(2, 2, generator=g, dtype=torch.float64).requires_grad_()
    field = lambda x, t: torch.einsum("hk,bkd->bhd", W, torch.tanh(x)) * (1 + t[:, None, None])
    {fn}(chain, field, idx, 0.6, joint=joint).sum().backward()
    got = W.grad.clone()
    W.grad = None
    oracle(chain, field, idx, 0.6, joint).sum().backward()
    assert torch.isfinite(got).all() and got.abs().sum() > 0, got
    assert torch.allclose(got, W.grad, atol=1e-8), (joint, got, W.grad)
"""},
        {"name": "Keeps dtype and leaves inputs untouched", "visibility": "unshown", "behavior": "tensor.dtype_device",
         "failure_message": "Return chain's dtype (float32 stays float32), pass grid times in chain's dtype, and never write into chain or sde_index.",
         "code": _ORACLE + r"""
step = {fn}.__globals__["flow_sde_mean_std"]
for dtype in (torch.float32, torch.float64):
    g = torch.Generator().manual_seed(22)
    chain = torch.randn(2, 4, 3, 2, generator=g, dtype=torch.float64).to(dtype)
    idx = torch.tensor([1, 2])
    saved = (chain.clone(), idx.clone())
    log = []
    v = make_field(3, 22, log)
    for joint in (False, True):
        out = {fn}(chain, v, idx, 0.5, joint=joint, level="token_level")
        assert out.dtype == dtype, (dtype, out.dtype)
    assert all(t.dtype == dtype for _, t in log), "grid times must use chain's dtype"
    mean, std = step(chain[:, 0], chain[:, 1], idx, 3, "flow_sde", 0.5)
    assert mean.dtype == dtype and std.dtype == dtype, (mean.dtype, std.dtype)
    assert torch.equal(chain, saved[0]) and torch.equal(idx, saved[1]), "an input was modified"
"""},
        {"name": "Rejects bad methods, levels, shapes and indices", "visibility": "unshown", "behavior": "contract.signature",
         "failure_message": "Raise ValueError for an unknown method or level, a bad chain or index shape, an out-of-range index, a missing sde_index, or a bad action_chunk / action_dim.",
         "code": r"""
import torch
step = {fn}.__globals__["flow_sde_mean_std"]
x = torch.zeros(2, 3, 2); chain = torch.zeros(2, 4, 3, 2)
v = lambda x, t: torch.zeros_like(x)
ok = torch.tensor([0, 2])
bad = [
    lambda: step(x, x, ok, 3, "flow_cps", 0.5),
    lambda: step(x, x, ok, 0, "flow_sde", 0.5),
    lambda: step(x, torch.zeros(2, 3, 3), ok, 3, "flow_sde", 0.5),
    lambda: step(x[0], x[0], ok, 3, "flow_sde", 0.5),
    lambda: step(x, x, torch.tensor([0, 3]), 3, "flow_sde", 0.5),
    lambda: step(x, x, torch.tensor([0]), 3, "flow_sde", 0.5),
    lambda: {fn}(chain, v, ok, 0.5, level="sequence_level"),
    lambda: {fn}(chain[:, :1], v, torch.tensor([0, 0]), 0.5),
    lambda: {fn}(chain[0], v, ok, 0.5),
    lambda: {fn}(chain, v, None, 0.5),
    lambda: {fn}(chain, v, torch.tensor([0, 3]), 0.5),
    lambda: {fn}(chain, v, torch.tensor([-1, 0]), 0.5),
    lambda: {fn}(chain, v, torch.tensor([0, 1, 2]), 0.5),
    lambda: {fn}(chain, v, ok, 0.5, action_chunk=4),
    lambda: {fn}(chain, v, ok, 0.5, action_chunk=0),
    lambda: {fn}(chain, v, ok, 0.5, action_dim=3),
]
for i, call in enumerate(bad):
    try:
        call()
    except ValueError:
        pass
    else:
        raise AssertionError(f"case {i} was accepted")
"""},
    ],
    "solution": '''import math

import torch

_METHODS = ("flow_ode", "flow_sde")
_LEVELS = ("token_level", "action_level", "chunk_level")


def _timesteps(num_steps, dtype, device):
    # Noise at t = 1, the action at t = 0: t_k = 1 - k / K for k = 0..K.
    k = torch.arange(num_steps + 1, dtype=dtype, device=device)
    return 1.0 - k / num_steps


def flow_sde_mean_std(x_t, v_t, idx, num_steps, method, noise_level):
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}, got {method!r}")
    if isinstance(num_steps, bool) or not isinstance(num_steps, int) or num_steps < 1:
        raise ValueError("num_steps must be a positive integer")
    if x_t.ndim != 3 or v_t.shape != x_t.shape:
        raise ValueError("x_t and v_t must both have shape (B, H, D)")
    if idx.shape != x_t.shape[:1]:
        raise ValueError("idx must have shape (B,)")
    if bool((idx < 0).any()) or bool((idx >= num_steps).any()):
        raise ValueError("idx must lie in [0, num_steps - 1]")
    ts = _timesteps(num_steps, x_t.dtype, x_t.device)
    t = ts[idx][:, None, None]
    t_next = ts[idx + 1][:, None, None]
    delta = t - t_next
    action_pred = x_t - t * v_t           # where the straight line reaches t = 0
    noise_pred = x_t + (1 - t) * v_t      # where it reaches t = 1
    if method == "flow_ode":
        mean = action_pred * (1 - t_next) + noise_pred * t_next
        return mean, torch.zeros_like(x_t)
    # sigma_k = noise_level * sqrt(t_k / (1 - t_k)); at t = 1 the denominator uses t_1 instead.
    denom_t = torch.where(ts == 1, ts[1], ts)
    sigmas = noise_level * torch.sqrt(ts / (1 - denom_t))
    sigma = sigmas[idx][:, None, None]
    noise_weight = t_next - sigma * sigma * delta / (2 * t)
    mean = action_pred * (1 - t_next) + noise_pred * noise_weight
    std = (torch.sqrt(delta) * sigma).expand_as(x_t)
    return mean, std


def _gaussian_logprob(x, mean, std):
    zero = std == 0
    safe = torch.where(zero, torch.ones_like(std), std)
    lp = -torch.log(safe) - 0.5 * math.log(2 * math.pi) - 0.5 * ((x - mean) / safe) ** 2
    return torch.where(zero, torch.zeros_like(lp), lp)


def flow_chain_logprob(chain, velocity_fn, sde_index, noise_level, joint=False,
                       level="chunk_level", action_chunk=None, action_dim=None):
    if level not in _LEVELS:
        raise ValueError(f"level must be one of {_LEVELS}, got {level!r}")
    if chain.ndim != 4 or chain.shape[1] < 2:
        raise ValueError("chain must have shape (B, K + 1, H, D) with K >= 1")
    batch, num_states, horizon, dim = chain.shape
    for name, value, limit in (("action_chunk", action_chunk, horizon), ("action_dim", action_dim, dim)):
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= limit):
            raise ValueError(f"{name} must be an integer in [1, {limit}]")
    num_steps = num_states - 1
    ts = _timesteps(num_steps, chain.dtype, chain.device)
    rows = torch.arange(batch, device=chain.device)

    if joint:
        terms = [_gaussian_logprob(chain[:, 0], torch.zeros_like(chain[:, 0]), torch.ones_like(chain[:, 0]))]
        for k in range(num_steps):
            idx = torch.full((batch,), k, dtype=torch.long, device=chain.device)
            x = chain[:, k]
            v = velocity_fn(x, ts[idx])
            mean, std = flow_sde_mean_std(x, v, idx, num_steps, "flow_sde", noise_level)
            terms.append(_gaussian_logprob(chain[:, k + 1], mean, std))
        logp = torch.stack(terms, dim=1).mean(dim=1)
    else:
        if sde_index is None or sde_index.shape != (batch,):
            raise ValueError("sde_index must be a (B,) integer tensor when joint is False")
        if bool((sde_index < 0).any()) or bool((sde_index >= num_steps).any()):
            raise ValueError("sde_index must lie in [0, K - 1]")
        idx = sde_index.long()
        x = chain[rows, idx]
        v = velocity_fn(x, ts[idx])
        mean, std = flow_sde_mean_std(x, v, idx, num_steps, "flow_sde", noise_level)
        logp = _gaussian_logprob(chain[rows, idx + 1], mean, std)

    logp = logp[:, :action_chunk, :action_dim]  # drop padded positions and dims
    if level == "action_level":
        return logp.sum(dim=-1)
    if level == "chunk_level":
        return logp.sum(dim=(1, 2))
    return logp
''',
    "interview_questions": interview(
        concept=[
            "Why can't PPO use the likelihood of an action chunk from a flow-matching head such as pi0 directly, and how does turning denoising into a Gaussian Markov chain fix that?",
            "What does Flow-SDE change in a denoising step compared with the deterministic Euler step, and what is the extra drift term for?",
        ],
        deep_dive=[
            "Write the Flow-SDE step mean in terms of the predicted clean action and the predicted noise. Where does the sigma squared times delta over 2t term go, and what happens if you drop it?",
            "The schedule sigma = a * sqrt(t / (1 - t)) is infinite at t = 1 when noise sits at t = 1. How does RLinf handle the first denoising step, and what standard deviation does that step end up with?",
            "pi_RL runs one randomly chosen denoising step as SDE and the rest as ODE. What does the PPO ratio measure then, and why is it enough to store the chain and recompute a single step at training time?",
        ],
        tradeoffs=[
            "Joint chain log-probability over all K steps versus a single random SDE step: compare training compute, the size of the log-ratio and how much of the chain gets explored.",
            "Token-level, action-level and chunk-level log-probabilities with a chunk-level reward: how does each choice change what one PPO ratio covers and how clipping behaves?",
        ],
    ),
}
