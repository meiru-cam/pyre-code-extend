One point worth confirming before coding: the task and network are fixed by the statement, so the only freedom is in `train`'s keyword defaults (batch size, group size, number of inner steps, learning rate) and the reward threshold; the values used below are one setting that reaches the stated threshold comfortably and quickly, not the only one that works.

### Part 1

Run the script before reading it closely: with all three bugs present, `python grpo_train.py` crashes on its very first call to `torch.multinomial`. Fixing bugs one at a time and rerunning shows each bug's own symptom in turn.

- Bugs fixed so far: none · Observed behavior: `RuntimeError: probability tensor contains either inf, nan or element < 0`, on step 0 · Points to: bug 1, sampling
- Bugs fixed so far: 1 · Observed behavior: loss is `nan` on step 0, then the same `RuntimeError` on step 1 · Points to: bug 2, `group_advantages`
- Bugs fixed so far: 1, 2 · Observed behavior: runs to completion; held-out accuracy stalls around $0.2$, well under the $0.5$ threshold · Points to: bug 3, `logp_old`
- Bugs fixed so far: all three · Observed behavior: held-out accuracy reaches about $0.7$ with the seed used throughout this page

**Bug 1: `actions = torch.multinomial(logits_old, 1, generator=rng)`.** `torch.multinomial` treats its input as a set of non-negative, unnormalized weights over the categories — it does not apply a softmax first. `logits_old` comes straight out of a linear layer and is centered near zero, so it takes both signs; the very first call has a negative entry with near certainty, and PyTorch raises rather than silently clamping it. The fix turns the logits into an actual distribution before sampling:

```py
probs_old = F.softmax(logits_old, dim=-1)
actions = torch.multinomial(probs_old, 1, generator=rng).squeeze(1)   # was torch.multinomial(logits_old, ...)
```

**Bug 2: `return (rewards - mean) / std` in `group_advantages`.** When every response in a group gets the same reward — every deviation from the group mean is exactly $0$ — `std` is not "close to" zero, it is exactly $0$, and $0/0$ is `nan`, not a large or a small number. With `batch_size=100` groups of `group_size=16` and a freshly initialized policy near the $10\%$ chance level, the probability that all $100$ groups contain at least one correct response is negligible, so this `nan` appears on literally the first optimizer step. `loss.backward()` then writes `nan` into every parameter's gradient, and `opt.step()` writes `nan` into every parameter itself, permanently — the crash on the next `torch.multinomial` call (once bug 1 is fixed) is a downstream symptom of *this* bug, not a new one. The fix adds the epsilon already defined as `ADV_EPS`:

```py
return (rewards - mean) / (std + ADV_EPS)     # was (rewards - mean) / std
```

**Bug 3: `logp_old = logp_new.detach()` inside the `for _ in range(inner_steps):` loop.** Before the inner loop, `logp_old` is computed correctly, under `torch.no_grad()`, from the rollout snapshot $\theta_{\mathrm{old}}$. The flagged line then overwrites that name, on every inner iteration, with the *current* policy's log-probability, `.detach()`-ed only to block gradient flow through it — not to fix its value to the rollout snapshot. `logp_new` and this `logp_old` therefore always come from the same forward pass, so `logp_new - logp_old` is exactly $0$ and $\rho_i \equiv 1$ at every inner step, not just the first — where a ratio of exactly $1$ is in fact correct (Part 2a). Because $\rho_i$ never leaves $[1-\varepsilon, 1+\varepsilon]$, the clip never triggers, and the trust region it is meant to enforce is silently absent: every inner step performs a fresh, uncorrected REINFORCE update at the *current* parameters, reusing the advantages computed from the original rollout, more of which have gone stale with each inner step. Printing `ratio` shows no `nan` and no crash — training even continues to improve the reward, just far less than with all three bugs fixed, since none of the extra inner steps are actually protected from moving too far past the data they were computed from. The fix deletes the line, so the correct, no-grad `logp_old` computed once before the loop is reused, unchanged, at every inner step:

```py
ratio = torch.exp(logp_new - logp_old)      # logp_old: the fixed value from before the loop; no reassignment here
```

### Part 2

**(a)** At the first inner step, no update has happened yet, so $\theta = \theta_{\mathrm{old}}$ and $\log\pi_\theta(o_i\mid a,b)$ (a live node in the current graph) numerically equals $\log\pi_{\theta_{\mathrm{old}}}(o_i\mid a,b)$ (a detached constant stored earlier), so $\rho_i = \exp(0) = 1$. Writing $x(\theta) = \log\pi_\theta(o_i\mid a,b)$ and $c$ for the detached constant, $\rho_i(\theta) = \exp(x(\theta) - c)$, so

```math
\frac{\partial \rho_i}{\partial\theta} = \exp(x(\theta) - c)\cdot\frac{\partial x}{\partial\theta}
= \rho_i \cdot \nabla_\theta\log\pi_\theta(o_i\mid a,b),
```

which at $\rho_i = 1$ gives $\partial(\rho_i A_i)/\partial\theta = A_i\nabla_\theta\log\pi_\theta(o_i\mid a,b)$, exactly the REINFORCE term. Since $\rho_i = 1 \in [1-\varepsilon, 1+\varepsilon]$, the unclipped and clipped branches of $\min(\cdot,\cdot)$ agree there too, so clipping changes nothing at this exact point. This equality holds only because $c$ happened to equal $x(\theta)$'s value at this one instant; from the second inner step on, $\theta$ has moved away from $\theta_{\mathrm{old}}$ (the first step's update changed it) while $c = \log\pi_{\theta_{\mathrm{old}}}(o_i\mid a,b)$ stays fixed at the value it had before any update, so $x(\theta) \ne c$ in general and $\rho_i \ne 1$: the surrogate's gradient genuinely departs from plain REINFORCE, reweighted (and possibly clipped) by how far the policy has already moved on this batch. This is also why the bug in Part 1c is invisible whenever `inner_steps == 1`: with a single inner step, the buggy and the correct `logp_old` are numerically identical, and the two implementations compute the same update.

**(b)** $\operatorname{clip}(\rho,1-\varepsilon,1+\varepsilon)A$ is the value selected by $\min(\cdot,\cdot)$ exactly when $\rho$ has moved outside $[1-\varepsilon,1+\varepsilon]$ *in the direction that the sign of $A$ already favors*: $\rho > 1+\varepsilon$ with $A>0$ (already increasing a good response's probability past the trust region), or $\rho < 1-\varepsilon$ with $A<0$ (already decreasing a bad response's probability past it). The four pairs from the statement cover exactly these two cases and their opposites:

```text
(rho, A) = (0.7, 1):  unclipped 0.7 < clipped 0.8 -> min picks 0.7 (unclipped; rho has NOT overshot the favored direction)
(rho, A) = (1.5, 1):  unclipped 1.5 > clipped 1.2 -> min picks 1.2 (clipped;   rho HAS overshot the favored direction)
(rho, A) = (0.7, -1): unclipped -0.7 > clipped -0.8 -> min picks -0.8 (clipped;   overshot, favored direction is decreasing rho)
(rho, A) = (1.5, -1): unclipped -1.5 < clipped -1.2 -> min picks -1.5 (unclipped; not overshot)
```

Where the clipped branch is selected, $\operatorname{clip}(\rho,\dots)$ is locally constant in $\rho$, so $\partial(\text{surrogate})/\partial\rho = 0$: no gradient signal pushes $\rho$ any further in that direction. Where the unclipped branch is selected — including whenever $\rho$ moved *against* what $A$ favors — the gradient is the plain $A$, unreduced: clipping never blocks a correction, only the continuation of an already-exploited move.

**(c)** $\sum_{i=1}^G (r_i - \bar r) = \sum_i r_i - G\bar r = G\bar r - G\bar r = 0$, and dividing by the same $\operatorname{std}(r) + \epsilon_{\mathrm{adv}}$ for every $i$ in the group preserves this: $\sum_i A_i = 0/(\operatorname{std}(r)+\epsilon_{\mathrm{adv}}) = 0$. In an actor-critic estimator $\frac1G\sum_i (r_i - b(s))\nabla_\theta\log\pi_\theta(o_i\mid s)$, any baseline $b(s)$ that does not depend on $o_i$ leaves the estimator unbiased, because $\sum_o \pi_\theta(o\mid s)\nabla_\theta\log\pi_\theta(o\mid s) = \nabla_\theta \sum_o \pi_\theta(o\mid s) = \nabla_\theta 1 = 0$ for any fixed $s$ — subtracting $b(s)$ contributes exactly $0$ in expectation, while typically reducing the estimator's variance when $b(s)$ tracks $\mathbb{E}_{o\sim\pi}[r(o,s)\mid s]$. A learned $V_\phi(s)$ is trained by regression to approximate exactly that conditional expectation, at the cost of a second network, its own optimizer, and its own approximation error. The group mean $\bar r$ is a direct, sample-based estimate of the same quantity, computed from $G$ fresh samples of this exact prompt, so it plays the same variance-reducing role without a second network at all; dividing further by $\operatorname{std}(r)$ additionally rescales the advantage so that groups with more or less spread-out rewards contribute comparably.

**(d)** For $G=1$, `rewards.std(dim=1)` divides the sum of squared deviations by $G-1=0$: the result is `nan` regardless of `ADV_EPS`, since `nan + ADV_EPS` is still `nan` — the estimator itself is undefined for a single sample, so `group_advantages` cannot be used with $G=1$ at all. For $G\ge2$ with all $G$ rewards equal, every deviation from the mean is exactly $0$, so `std` is a well-defined, finite $0$ (a sum of $G$ zeros divided by $G-1>0$); it is the *subsequent* division `(rewards - mean) / std` that hits $0/0$. `ADV_EPS` repairs exactly this second case, turning it into $0/(0+\epsilon_{\mathrm{adv}}) = 0$ — a well-defined "no signal from this group," not a `nan` — but it cannot repair the first case, where the input to the division is already `nan`.

**(e)** The clip bounds how far $\rho_i$ (and so, loosely, $\pi_\theta$) can move from $\theta_{\mathrm{old}}$ within one outer step's `inner_steps` updates, but $\theta_{\mathrm{old}}$ is reset to the current $\theta$ at the start of every new outer step, so this bound resets every step and places no bound at all on how far $\pi_\theta$ can drift over many outer steps. $\beta D_{\mathrm{KL}}(\pi_\theta\Vert\pi_{\mathrm{ref}})$ is evaluated against the same fixed $\pi_{\mathrm{ref}}$ for the entire run, so it is the only term that limits cumulative drift. As $\beta\to0$, that anchor disappears: each individual outer step is still trust-region-bounded by the clip, but nothing stops $\pi_\theta$ from moving arbitrarily far from $\pi_{\mathrm{ref}}$ over enough outer steps, which on a less strictly verifiable reward than this one is exactly the setting in which reward hacking becomes possible (Follow-ups). As $\beta\to\infty$, the KL term dominates $L(\theta)$ and $\pi_\theta$ is pinned arbitrarily close to $\pi_{\mathrm{ref}}$ regardless of the reward signal, and training stops improving the reward at all.

### Follow-ups

- **`ratio = logp_new - logp_old`, without the `exp`.** For small updates this tracks the true ratio reasonably well, since $\log(1+x)\approx x$ near $x=0$, so the model can often still learn something; the failure is structural, not immediate. $\operatorname{clip}$ now bounds a *log*-difference to $[-\varepsilon,\varepsilon]$ rather than a probability ratio to $[1-\varepsilon,1+\varepsilon]$ — a much narrower band in probability terms once $|x|$ is not small — and the true ratio is bounded below by $0$ (a token the policy now strongly disfavors can contribute at most $-A_i$ to the surrogate) while a raw log-difference is unbounded below, so a single very-disfavored token can dominate the batch's gradient.
- **Reward hacking.** With a strictly verifiable reward like this one there is little room to exploit, but in general the reward is itself a model or a heuristic, and an unconstrained policy can drift toward inputs that score well under it without being genuinely better. The KL term and the ratio clip are exactly the two mechanisms that keep $\pi_\theta$ close enough to $\pi_{\mathrm{ref}}$, and close enough to the data it was just evaluated on, to limit this.
- **Entropy collapse.** Once a group's $G$ samples all land on the same response, every deviation from the group mean is $0$, so $A_i = 0$ for every $i$ in that group (Part 2d) regardless of whether the shared response is right or wrong: a prompt the policy has already made near-deterministic, correctly or not, stops producing any gradient signal for itself. A KL term toward a higher-entropy $\pi_{\mathrm{ref}}$ is a common mitigation, since it directly penalizes the policy for concentrating its probability mass.
- **Per-token vs. per-sequence averaging.** With multi-token responses, $\log\pi_\theta(o\mid a,b)$ is a sum over the response's tokens, so $\rho_i$ compounds one factor per token; averaging the surrogate over every token in the batch (rather than over every response, or every response first and then every prompt) gives longer responses more total weight in the gradient purely because they contain more tokens, independent of how good they are.
- **GRPO vs. PPO.** GRPO drops PPO's learned value network entirely (Part 2c), which halves the models trained and removes GAE's own bias-variance trade-off, at the cost of needing $G$ full rollouts per prompt instead of $1$, and a baseline that only compares responses to the same prompt, with no signal about which prompts are harder than others.
- **Gradient-norm clipping.** A global cap on $\lVert\nabla_\theta L\rVert$, applied right before `opt.step()`, is a common further stabilizer, orthogonal to the ratio clip: the ratio clip bounds how far a single sample's probability is allowed to move, while gradient clipping bounds the size of the aggregated update itself, regardless of which samples produced it.

```python
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
_PRIOR_TORCH_THREADS = torch.get_num_threads()  # NOTE: restored at the end of the checks below so this
torch.set_num_threads(1)                        # setting cannot leak into a later page's own torch calls

V = 10
D_MODEL = 16
HIDDEN = 32
CLIP_EPS = 0.2
KL_COEF = 0.01
ADV_EPS = 1e-6


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_a = nn.Embedding(V, D_MODEL)
        self.embed_b = nn.Embedding(V, D_MODEL)
        self.net = nn.Sequential(nn.Linear(D_MODEL, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, V))

    def forward(self, a, b):                   # a, b: (N,) digit ids -> logits (N, V)
        return self.net(self.embed_a(a) + self.embed_b(b))


def sample_prompts(n, rng):                     # n random (a, b) prompts, each digit uniform in [0, V)
    a = torch.randint(0, V, (n,), generator=rng)
    b = torch.randint(0, V, (n,), generator=rng)
    return a, b


def reward(a, b, answer):                       # verifiable reward: 1.0 iff answer == (a + b) mod 10
    return (answer == (a + b) % V).float()


def group_advantages(rewards):                   # rewards: (P, G) -> advantages (P, G)
    mean = rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, keepdim=True)
    return (rewards - mean) / (std + ADV_EPS)    # fix for bug 2: + ADV_EPS


def train(policy, ref_policy, steps=200, batch_size=100, group_size=16, inner_steps=4, lr=0.02, seed=0):
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    rng = torch.Generator().manual_seed(seed)
    for step in range(steps):
        a, b = sample_prompts(batch_size, rng)
        a_rep, b_rep = a.repeat_interleave(group_size), b.repeat_interleave(group_size)
        with torch.no_grad():
            logits_old = policy(a_rep, b_rep)
            probs_old = F.softmax(logits_old, dim=-1)                     # fix for bug 1: softmax before sampling
            actions = torch.multinomial(probs_old, 1, generator=rng).squeeze(1)
            logp_old = F.log_softmax(logits_old, dim=-1).gather(1, actions[:, None]).squeeze(1)
            logp_ref_all = F.log_softmax(ref_policy(a_rep, b_rep), dim=-1)
        r = reward(a_rep, b_rep, actions).view(batch_size, group_size)
        adv = group_advantages(r).reshape(-1)
        for _ in range(inner_steps):
            logits_new = policy(a_rep, b_rep)
            logp_new_all = F.log_softmax(logits_new, dim=-1)
            logp_new = logp_new_all.gather(1, actions[:, None]).squeeze(1)
            # fix for bug 3: no reassignment of logp_old here -- it stays the value from before the loop
            ratio = torch.exp(logp_new - logp_old)
            clipped = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS)
            pg_loss = -torch.min(ratio * adv, clipped * adv).mean()
            probs_new_all = logp_new_all.exp()
            kl = (probs_new_all * (logp_new_all - logp_ref_all)).sum(dim=-1).mean()
            loss = pg_loss + KL_COEF * kl
            opt.zero_grad()
            loss.backward()
            opt.step()
    return policy


@torch.no_grad()
def evaluate(policy, n, seed):                   # greedy accuracy on n freshly sampled held-out prompts
    rng = torch.Generator().manual_seed(seed)
    a, b = sample_prompts(n, rng)
    pred = policy(a, b).argmax(dim=-1)
    return reward(a, b, pred).mean().item()


if __name__ == "__main__":
    torch.manual_seed(0)
    policy = Policy()
    ref_policy = copy.deepcopy(policy)
    for p in ref_policy.parameters():
        p.requires_grad_(False)
    train(policy, ref_policy)
    accuracy = evaluate(policy, 2000, seed=999)
    print(f"held-out accuracy: {accuracy:.3f}")
    assert accuracy > 0.5, "the policy did not learn to add single digits"
    print("cleared the reward threshold")
```

```python
import warnings

# Part 1, statement: the fixed script clears the reward threshold with the seed used throughout this page
torch.manual_seed(0)
policy = Policy()
ref_policy = copy.deepcopy(policy)
for p in ref_policy.parameters():
    p.requires_grad_(False)
train(policy, ref_policy)
accuracy = evaluate(policy, 2000, seed=999)
assert accuracy > 0.5, f"held-out accuracy {accuracy:.3f} did not clear the threshold"

# Bug 1, reinserted in isolation: raw logits (which take both signs) crash torch.multinomial;
# softmax'd logits sample cleanly, exactly as Part 1's fix says
raw_logits = torch.tensor([[1.0, -0.5, 0.3, 2.0], [0.1, 0.2, -0.1, -0.4]])
try:
    torch.multinomial(raw_logits, 1)
    crashed_on_raw_logits = False
except RuntimeError:
    crashed_on_raw_logits = True
assert crashed_on_raw_logits, "sampling from raw logits should raise -- bug 1's exact failure mode"
sampled = torch.multinomial(F.softmax(raw_logits, dim=-1), 1)
assert sampled.shape == (2, 1)

# The advantage formula against the hand-computed example of the statement (G = 4, rewards (1, 0, 1, 0))
rewards_ex = torch.tensor([[1., 0., 1., 0.]])
adv_ex = group_advantages(rewards_ex)
expected_adv = torch.tensor([[3 ** 0.5 / 2, -(3 ** 0.5) / 2, 3 ** 0.5 / 2, -(3 ** 0.5) / 2]])
assert torch.allclose(adv_ex, expected_adv, atol=1e-4)


def buggy_group_advantages(rewards):             # bug 2 reinserted: no + ADV_EPS
    mean = rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, keepdim=True)
    return (rewards - mean) / std


# Bug 2, reinserted in isolation: a group with a constant reward is nan without ADV_EPS, 0 with it;
# a group with genuine variance is unaffected either way
constant_group = torch.tensor([[1., 1., 1., 1.], [0., 1., 0., 1.]])
assert torch.isnan(buggy_group_advantages(constant_group)[0]).all()
assert not torch.isnan(group_advantages(constant_group)).any()
assert torch.allclose(buggy_group_advantages(constant_group)[1], group_advantages(constant_group)[1], atol=1e-4)

# Part 2d: G = 1 is nan regardless of ADV_EPS (the std estimator itself is 0/0, not merely the division)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")              # torch warns about the degenerate degrees of freedom; expected here
    assert torch.isnan(group_advantages(torch.tensor([[1.0]]))).all()

# The clipped objective against the hand-computed example of the statement, all four clip/sign combinations
ratio_ex = torch.tensor([0.7, 1.5, 0.7, 1.5])
adv_ex2 = torch.tensor([1.0, 1.0, -1.0, -1.0])
clipped_ex = torch.clamp(ratio_ex, 1 - CLIP_EPS, 1 + CLIP_EPS)
surrogate_ex = torch.min(ratio_ex * adv_ex2, clipped_ex * adv_ex2)
assert torch.allclose(surrogate_ex, torch.tensor([0.7, 1.2, -0.8, -1.5]), atol=1e-6)
assert round((-surrogate_ex).mean().item(), 4) == 0.1


def run_inner_steps(policy, bug3, inner_steps=3, seed=1):
    """Three inner steps on one rollout batch, returning the ratio tensor computed at each step."""
    rng = torch.Generator().manual_seed(seed)
    a, b = sample_prompts(20, rng)
    a_rep, b_rep = a.repeat_interleave(8), b.repeat_interleave(8)
    opt = torch.optim.Adam(policy.parameters(), lr=0.05)
    with torch.no_grad():
        logits_old = policy(a_rep, b_rep)
        actions = torch.multinomial(F.softmax(logits_old, dim=-1), 1, generator=rng).squeeze(1)
        logp_old = F.log_softmax(logits_old, dim=-1).gather(1, actions[:, None]).squeeze(1)
    r = reward(a_rep, b_rep, actions).view(20, 8)
    adv = group_advantages(r).reshape(-1)
    ratios = []
    for _ in range(inner_steps):
        logp_new_all = F.log_softmax(policy(a_rep, b_rep), dim=-1)
        logp_new = logp_new_all.gather(1, actions[:, None]).squeeze(1)
        if bug3:
            logp_old = logp_new.detach()          # bug 3 reinserted
        ratio = torch.exp(logp_new - logp_old)
        ratios.append(ratio.detach().clone())
        clipped = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS)
        loss = -torch.min(ratio * adv, clipped * adv).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return ratios


# Bug 3, reinserted in isolation: both variants start at ratio == 1 exactly (Part 2a); from the second
# inner step on, the correct policy's ratio must move away from 1, while the buggy one stays pinned to it
torch.manual_seed(2)
policy_correct = Policy()
policy_buggy = copy.deepcopy(policy_correct)     # identical starting weights, so only the bug differs
ratios_correct = run_inner_steps(policy_correct, bug3=False)
ratios_buggy = run_inner_steps(policy_buggy, bug3=True)
ones = torch.ones(160)
assert torch.allclose(ratios_correct[0], ones, atol=1e-6)
assert torch.allclose(ratios_buggy[0], ones, atol=1e-6)
for step in (1, 2):
    assert not torch.allclose(ratios_correct[step], ones, atol=1e-3), "correct: ratio should move away from 1"
    assert torch.allclose(ratios_buggy[step], ones, atol=1e-6), "buggy: ratio should stay pinned to 1"

torch.set_num_threads(_PRIOR_TORCH_THREADS)  # NOTE: undo the pin so it cannot leak into another page

print("all checks passed")
```
