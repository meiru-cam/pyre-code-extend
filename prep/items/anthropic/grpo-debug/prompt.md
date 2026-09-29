The file `grpo_train.py` below trains a tiny policy with *Group Relative Policy Optimization* (GRPO) on a synthetic task with a verifiable reward: given two digits $a, b \in \{0,\dots,9\}$ as the prompt, the policy must answer with the single token $(a+b) \bmod 10$.

A *policy* $\pi_\theta$ maps a prompt $(a,b)$ to a categorical distribution over $V = 10$ candidate answer tokens $\{0,\dots,9\}$: $\pi_\theta(\cdot \mid a,b) = \operatorname{softmax}(f_\theta(a,b))$, where $f_\theta(a,b) \in \mathbb{R}^V$ are the logits produced by a small network. A *response* $o$ is a single token sampled from $\pi_\theta(\cdot\mid a,b)$, and its *reward* is $r(o,a,b) = \mathbb{1}[\,o = (a+b) \bmod 10\,]$.

For each prompt, a *group* of $G$ responses $o_1,\dots,o_G$ is sampled independently from the same distribution $\pi_{\theta_{\mathrm{old}}}(\cdot\mid a,b)$, where $\theta_{\mathrm{old}}$ is a snapshot of the parameters taken once, before any of the updates described below. Writing $r_i = r(o_i,a,b)$, the *group-relative advantage* of response $i$ is

```math
A_i = \frac{r_i - \bar r}{\operatorname{std}(r) + \epsilon_{\mathrm{adv}}}, \qquad
\bar r = \frac{1}{G}\sum_{j=1}^G r_j, \qquad
\operatorname{std}(r) = \sqrt{\frac{1}{G-1}\sum_{j=1}^G (r_j - \bar r)^2},
```

with a small constant $\epsilon_{\mathrm{adv}} > 0$. For example, a group of $G=4$ responses with rewards $(1,0,1,0)$ has $\bar r = 0.5$, $\operatorname{std}(r) = 1/\sqrt3 \approx 0.5774$ (an $\epsilon_{\mathrm{adv}}$ of $10^{-6}$ changes this negligibly), and

```text
A = (0.8660, -0.8660, 0.8660, -0.8660)   # (r_i - 0.5) / 0.5774, elementwise
```

The *ratio* of response $i$ compares the policy currently being optimized, $\pi_\theta$, against the policy that generated the sample, $\pi_{\theta_{\mathrm{old}}}$:

```math
\rho_i = \exp\!\big(\log \pi_\theta(o_i \mid a,b) - \log \pi_{\theta_{\mathrm{old}}}(o_i \mid a,b)\big).
```

The training loss for one flattened batch of $P \cdot G$ samples (all $G$ responses of all $P$ prompts) is

```math
L(\theta) = \frac{1}{PG}\sum_{i=1}^{PG} -\min\!\big(\rho_i A_i,\ \operatorname{clip}(\rho_i,\, 1-\varepsilon,\, 1+\varepsilon)\, A_i\big)
\;+\; \beta \cdot \frac{1}{PG}\sum_{i=1}^{PG} D_{\mathrm{KL}}\!\big(\pi_\theta(\cdot\mid a_i,b_i) \,\Vert\, \pi_{\mathrm{ref}}(\cdot\mid a_i,b_i)\big),
```

where $\operatorname{clip}(x, l, u) = \min(\max(x,l),u)$, $\varepsilon$ is the clip range, $\beta$ is a fixed coefficient, and $\pi_{\mathrm{ref}}$ is a second, frozen copy of the policy fixed at its value before training started (never updated by the optimizer, and unrelated to $\theta_{\mathrm{old}}$). For instance, with $\varepsilon = 0.2$ the four pairs $(\rho_i, A_i) \in \{(0.7,1), (1.5,1), (0.7,-1), (1.5,-1)\}$ give clipped values $\operatorname{clip}(\rho_i,0.8,1.2) \in \{0.8, 1.2, 0.8, 1.2\}$ and

```text
rho_i * A_i          =  0.7,  1.5, -0.7, -1.5
clip(rho_i) * A_i    =  0.8,  1.2, -0.8, -1.2
min(...)             =  0.7,  1.2, -0.8, -1.5     # the surrogate term, before the leading minus sign
-> loss contribution = -0.7, -1.2,  0.8,  1.5      # mean over the four: 0.1
```

Since a response here is a single token, the ratio's usual per-token definition and its per-sequence definition coincide; with a multi-token response, $\log \pi_\theta(o \mid a,b)$ would instead be the sum of the per-token log-probabilities and $A_i$ would be the same value broadcast to every token of response $i$.

`grpo_train.py` alternates two kinds of steps. In each *outer* step, it samples a batch of $P$ prompts and, for each, a group of $G$ responses from the current policy; this snapshot of the parameters is $\theta_{\mathrm{old}}$ for everything that follows. It computes every reward $r_i$ and advantage $A_i$ once. Then, in a loop of `inner_steps` *inner* steps, it repeatedly computes $L(\theta)$ against this same batch of $PG$ sampled responses and takes a gradient step — so $\theta$ changes at every inner step, while $\theta_{\mathrm{old}}$ (and hence every $\log \pi_{\theta_{\mathrm{old}}}(o_i\mid\cdot)$ used in $\rho_i$) stays fixed at the value it had at the start of the outer step. Only after all `inner_steps` updates does the next outer step resample a fresh batch and a fresh $\theta_{\mathrm{old}}$.

### Part 1 — Find and fix three bugs

The file contains three bugs: one causes an immediate crash, one is silent until the loss turns to `nan`, and one never crashes or produces `nan` at all. For each, give its location, explain what it does to sampling or to training and why, and fix it. With all three fixed, `python grpo_train.py` trains for 200 outer steps (a few seconds on a CPU) and reaches a mean reward above $0.5$ on a batch of 2,000 freshly sampled held-out prompts, evaluated by greedily taking $\arg\max_o \pi_\theta(o\mid a,b)$ rather than sampling (chance is $0.1$). `sample_prompts`, `reward`, the network architecture, and every hyperparameter are correct and need no change.

```py
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

V = 10                          # digits 0-9: both the two prompt digits and the possible answer tokens
D_MODEL = 16
HIDDEN = 32
CLIP_EPS = 0.2                   # epsilon: the clip range is [1 - CLIP_EPS, 1 + CLIP_EPS]
KL_COEF = 0.01                   # beta: weight of the KL penalty against the frozen reference policy
ADV_EPS = 1e-6                   # epsilon_adv: the advantage's standard-deviation denominator


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
    return (rewards - mean) / std


def train(policy, ref_policy, steps=200, batch_size=100, group_size=16, inner_steps=4, lr=0.02, seed=0):
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    rng = torch.Generator().manual_seed(seed)
    for step in range(steps):
        a, b = sample_prompts(batch_size, rng)
        a_rep, b_rep = a.repeat_interleave(group_size), b.repeat_interleave(group_size)
        with torch.no_grad():
            logits_old = policy(a_rep, b_rep)
            actions = torch.multinomial(logits_old, 1, generator=rng).squeeze(1)
            logp_old = F.log_softmax(logits_old, dim=-1).gather(1, actions[:, None]).squeeze(1)
            logp_ref_all = F.log_softmax(ref_policy(a_rep, b_rep), dim=-1)
        r = reward(a_rep, b_rep, actions).view(batch_size, group_size)
        adv = group_advantages(r).reshape(-1)
        for _ in range(inner_steps):
            logits_new = policy(a_rep, b_rep)
            logp_new_all = F.log_softmax(logits_new, dim=-1)
            logp_new = logp_new_all.gather(1, actions[:, None]).squeeze(1)
            logp_old = logp_new.detach()
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

### Part 2 — The mathematics behind the objective

Answer the following about the fixed trainer, using the quantities defined above.

**(a)** At the first inner step of an outer step, $\theta$ equals $\theta_{\mathrm{old}}$ exactly. Derive $\partial(\rho_i A_i)/\partial\theta$ at that point and show it equals the plain policy-gradient (REINFORCE) term $A_i \nabla_\theta \log\pi_\theta(o_i\mid a,b)$, and that $\rho_i = 1$ there contributes no clipping. Explain why this equality between the surrogate's gradient and the REINFORCE gradient can only hold at the first inner step of each outer step, and why $\rho_i$ is, in general, no longer $1$ from the second inner step onward.

**(b)** For $\varepsilon = 0.2$, state the general rule (any $\rho$, any sign of $A$) for when $\operatorname{clip}(\rho,1-\varepsilon,1+\varepsilon)A$ rather than $\rho A$ is the value selected by $\min(\cdot,\cdot)$, and what $\partial(\text{surrogate})/\partial\rho$ equals in that region.

**(c)** Show that $\frac{1}{G}\sum_{i=1}^G A_i = 0$ for every group, by construction. Use it to explain what role the group mean plays relative to a learned baseline $V_\phi(s)$ in an actor-critic method, and why GRPO needs no such network.

**(d)** State exactly what `group_advantages` returns for a group of size $G=1$, and separately for a group of size $G \ge 2$ whose $G$ rewards happen to be all equal. Explain why `ADV_EPS` repairs one of these cases but not the other.

**(e)** $\varepsilon$ bounds how far $\rho_i$ can move from $\theta_{\mathrm{old}}$, which is refreshed at the start of every outer step; $\beta$ penalizes distance from $\pi_{\mathrm{ref}}$, which is fixed for the whole run. Explain what each of the two terms bounds over a run of many outer steps, and what changes qualitatively as $\beta \to 0$ and as $\beta \to \infty$.
