"""DreamerV3 RSSM KL loss: categorical latents with unimix, KL balancing and free bits."""

from ._interview import interview

TASK = {
    "title": "RSSM KL Balancing and Free Bits",
    "difficulty": "Medium",
    "version": 1,
    "function_name": "rssm_kl_loss",
    "description_en": r"""A DreamerV3 world model keeps a latent state made of `S` independent categorical variables with `C` classes each. The encoder sees the current observation and outputs posterior logits; the dynamics model predicts prior logits without it. Implement the KL part of the world-model loss: unimix both distributions, measure the KL from posterior to prior, apply free bits, and split the KL into two terms whose gradients reach different networks.

**Signature:** `rssm_kl_loss(post_logits, prior_logits, free_nats=1.0, beta_dyn=1.0, beta_rep=0.1, unimix=0.01) -> (loss, dyn, rep)`

**Parameters:**
- `post_logits`, `prior_logits` — floating tensors of the same shape `(B, T, S, C)`: `S` groups of `C` class logits per batch row and time step.
- `free_nats` — float `>= 0`, the free-bits floor.
- `beta_dyn`, `beta_rep` — floats that weight the two terms.
- `unimix` — float in `[0, 1]`, the share of the uniform distribution mixed into each group.

**Returns:**
- `dyn` and `rep` — tensors of shape `(B, T)`, the two KL terms after free bits.
- `loss` — a 0-d tensor, `mean(beta_dyn * dyn + beta_rep * rep)` over all `B * T` entries.

**The steps, for each logits tensor and then for the pair:**
- Unimix the probabilities, not the logits: `probs = (1 - unimix) * softmax(logits) + unimix / C`, then take `log(probs)`. With `unimix = 0`, use `log_softmax(logits)` directly.
- KL per group: `kl_g = sum_c p_c * (log p_c - log q_c)`, with `p` the posterior and `q` the prior. Sum `kl_g` over the `S` groups to get one value per `(B, T)`.
- Free bits on that sum: `max(free_nats, kl)` per `(B, T)`, never per group.
- `dyn` is the KL with the posterior detached, so its gradient trains only the prior. `rep` is the KL with the prior detached, so its gradient trains only the posterior.

**Constraints:**
- `dyn` and `rep` have equal values; only their gradients differ.
- Where the summed KL is below `free_nats`, the term equals `free_nats` and its gradient is exactly zero.
- The gradient of `dyn` with respect to `post_logits` is zero (or `None`), and so is the gradient of `rep` with respect to `prior_logits`.
- Logits of `-1e4` or lower must give finite results, with or without unimix.
- Compute in the input dtype; float64 inputs are checked to about 1e-10, values and gradients alike.
- Do not modify the inputs in place.
- Raise `ValueError` when either input is not a 4-D floating tensor, when their shapes differ, or when `unimix` is outside `[0, 1]`.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why two terms.** The KL between posterior and prior can shrink in two ways: the prior learns to predict the posterior, or the posterior gives up information until it is easy to predict. DreamerV3 wants mostly the first. Splitting the same KL into `dyn` (only the prior moves) and `rep` (only the posterior moves) and weighting them 1.0 and 0.1 does that. DreamerV2 called this KL balancing.

**Why free bits.** Once the KL is below about one nat per time step, the model gains little from shrinking it further, and pushing on it collapses the latent. Clipping at `free_nats` turns the gradient off in that regime so the reconstruction and reward losses take over. It is applied to the total over groups, so a few informative groups can still carry the whole nat.

**Why unimix.** Mixing 1% uniform into each categorical keeps every probability at least `unimix / C`. The log-probabilities stay bounded, so the KL cannot blow up when the posterior puts all its mass on a class the prior thinks is impossible. Mixing the logits instead would not bound anything.

**The latent sample.** The world model feeds the posterior forward as a one-hot sample with a straight-through gradient, `onehot + probs - stopgrad(probs)`, using the unimixed probabilities. That sampling step is not part of this exercise.

**Upstream differences.** The current DreamerV3 (paper v2 and the JAX repo) uses `beta_dyn = 1.0`; the first arXiv version of the paper and the PyTorch port (dreamerv3-torch) use 0.5, so both betas are parameters here.

**Contrast with k3_kl_penalty.** That exercise estimates a KL from sampled tokens' log-probabilities. Here both distributions are small categoricals, so the KL is computed exactly.""",
    "advisory_prerequisites": ["softmax", "softmax_entropy", "k3_kl_penalty", "symlog_twohot"],
    "hints": [
        {"level": 1, "kind": "questions", "content": "If you detach the posterior inside the KL, which network still receives a gradient? In what order do you sum over classes, sum over groups and apply the floor, and what changes if you swap the last two? What is the smallest probability any class can have after unimix, and what does that do to log q?"},
        {"level": 2, "kind": "analysis", "content": "Write one helper that turns logits into log-probabilities: mix in probability space when unimix is positive, and use a log-softmax when it is zero. Write a second helper for the KL that sums over the class axis and then the group axis. Call the KL twice with the detach on opposite sides, clamp each result from below at free_nats, then combine with the betas and take the mean. Pitfalls: mixing the logits instead of the probabilities, applying the floor per group, and taking log of a softmax that underflowed to zero."},
    ],
    "model_connections": [
        "DreamerV3 (Hafner et al. 2023) trains its RSSM with exactly this loss: 32 groups, unimix 0.01, free bits of 1 nat, and dyn and rep weighted 1.0 and 0.1 in the final paper version.",
        "DreamerV2 introduced discrete latents and KL balancing with a single mixing weight of 0.8; DreamerV3 splits it into two separately weighted terms and adds free bits and unimix.",
        "The straight-through one-hot sample of the same unimixed probabilities is what the dynamics model and the actor-critic consume.",
    ],
    "pro_con_analysis": {
        "pros": ["Stable world-model training across domains with fixed hyperparameters: the prior is pushed toward the posterior, latents do not collapse, and the KL stays bounded."],
        "cons": ["Three more hyperparameters, and below the free-bits floor the KL gives no learning signal at all, so representations can stay more informative than they need to be."],
    },
    "sources": [
        {
            "kind": "code",
            "url": "https://github.com/danijar/dreamerv3/blob/e01491fad6434b2245a3b8ca201dd7faedcc458c/dreamerv3/rssm.py",
            "commit": "e01491fad6434b2245a3b8ca201dd7faedcc458c",
            "path": "dreamerv3/rssm.py",
            "symbol": "RSSM.loss (lines 120-131) and RSSM._dist (line 173)",
            "license": "MIT",
            "adapted": "Same loss: dyn = KL(sg(post) || prior), rep = KL(post || sg(prior)), each summed over the stoch groups by Agg and floored with jnp.maximum(kl, free_nats).",
            "simplifications": "JAX transcribed to torch. Upstream skips the floor when free_nats is 0; the exercise always applies it, which differs only for KL values that round below zero. "
                               "Upstream uses jnp.maximum (half the gradient at an exact tie); the exercise does not test ties.",
        },
        {
            "kind": "code",
            "url": "https://github.com/danijar/dreamerv3/blob/e01491fad6434b2245a3b8ca201dd7faedcc458c/embodied/jax/outs.py",
            "commit": "e01491fad6434b2245a3b8ca201dd7faedcc458c",
            "path": "embodied/jax/outs.py",
            "symbol": "Categorical (line 208), OneHot, Agg.kl",
            "license": "MIT",
            "adapted": "Unimix in probability space followed by log, and the exact categorical KL sum_c p (log p - log q).",
            "simplifications": "Upstream casts logits to float32; the exercise keeps the input dtype so float64 can be checked tightly. "
                               "The OneHot/Agg wrappers become two helpers.",
        },
        {
            "kind": "code",
            "url": "https://github.com/danijar/dreamerv3/blob/e01491fad6434b2245a3b8ca201dd7faedcc458c/dreamerv3/configs.yaml",
            "commit": "e01491fad6434b2245a3b8ca201dd7faedcc458c",
            "path": "dreamerv3/configs.yaml",
            "symbol": "agent.loss_scales (line 86), agent.dyn.rssm free_nats and unimix (line 91)",
            "license": "MIT",
            "adapted": "Defaults dyn 1.0, rep 0.1, free_nats 1.0, unimix 0.01; the total is sum(v.mean() * scale), as in Agent.loss.",
            "simplifications": "Only the dyn and rep terms of the world-model loss are kept.",
        },
        {
            "kind": "code",
            "url": "https://github.com/NM512/dreamerv3-torch/blob/6ef8646d807cd10ce0c88e10a7e943211e7fc44c/networks.py",
            "commit": "6ef8646d807cd10ce0c88e10a7e943211e7fc44c",
            "path": "networks.py",
            "symbol": "RSSM.kl_loss (line 272)",
            "license": "MIT",
            "adapted": "Cross-check: the same two detached KLs, floored with torch.clip(min=free) and weighted by dyn_scale and rep_scale.",
            "simplifications": "Divergence: the port's default dyn_scale is 0.5 (configs.yaml line 57) against 1.0 upstream, so the exercise takes both betas as parameters. "
                               "It returns the per-(B, T) loss and leaves the mean to the caller.",
        },
        {
            "kind": "code",
            "url": "https://github.com/NM512/dreamerv3-torch/blob/6ef8646d807cd10ce0c88e10a7e943211e7fc44c/tools.py",
            "commit": "6ef8646d807cd10ce0c88e10a7e943211e7fc44c",
            "path": "tools.py",
            "symbol": "OneHotDist (line 425)",
            "license": "MIT",
            "adapted": "Cross-check of the unimix step: probs * (1 - u) + u / C, then log.",
            "simplifications": "The KL comes from torch.distributions instead of an explicit sum; sampling and the straight-through mode are not part of the exercise.",
        },
        {
            "kind": "paper",
            "url": "https://arxiv.org/abs/2301.04104",
            "section": "Hafner et al. 2023, Mastering Diverse Domains through World Models, Section 2 World model learning, Equations 3-5 (dynamics and representation losses with free bits) and the unimix paragraph",
            "note": "beta_dyn = 1 and beta_rep = 0.1 in the final version; earlier versions used beta_dyn = 0.5.",
        },
    ],
    "tests": [
        {"name": "One group, two classes, no unimix", "behavior": "rl.kl_estimator", "code": r"""
import math, torch
post = torch.tensor([0.0, math.log(3.0)]).view(1, 1, 1, 2)   # p = [0.25, 0.75]
prior = torch.zeros(1, 1, 1, 2)                               # q = [0.5, 0.5]
loss, dyn, rep = {fn}(post, prior, free_nats=0.0, beta_dyn=1.0, beta_rep=0.1, unimix=0.0)
kl = 0.25 * math.log(0.5) + 0.75 * math.log(1.5)
assert dyn.shape == (1, 1) and rep.shape == (1, 1) and loss.shape == ()
assert abs(dyn.item() - kl) < 1e-6 and abs(rep.item() - kl) < 1e-6, (dyn, rep, kl)
assert abs(loss.item() - 1.1 * kl) < 1e-6, loss
"""},
        {"name": "Free bits apply to the sum over groups", "behavior": "rl.clipping", "code": r"""
import math, torch
# Two groups with a KL of about 0.1308 nats each, 0.2616 in total.
group_post = torch.tensor([0.0, math.log(3.0)])
post = group_post.repeat(2, 1).view(1, 1, 2, 2)
prior = torch.zeros(1, 1, 2, 2)
loss, dyn, rep = {fn}(post, prior, free_nats=0.2, beta_dyn=1.0, beta_rep=0.0, unimix=0.0)
kl = 2 * (0.25 * math.log(0.5) + 0.75 * math.log(1.5))   # 0.2616 > 0.2: not clipped
assert abs(dyn.item() - kl) < 1e-6, dyn
loss, dyn, rep = {fn}(post, prior, free_nats=1.0, beta_dyn=1.0, beta_rep=0.0, unimix=0.0)
assert abs(dyn.item() - 1.0) < 1e-6, dyn                 # 0.2616 < 1.0: clipped to the floor
"""},
        {"name": "Identical distributions sit on the floor with zero gradient", "behavior": "gradient.flow", "code": r"""
import torch
g = torch.Generator().manual_seed(0)
logits = torch.randn(2, 3, 4, 5, generator=g)
post = logits.clone().requires_grad_(True)
prior = logits.clone().requires_grad_(True)
loss, dyn, rep = {fn}(post, prior)
assert torch.allclose(dyn, torch.ones(2, 3)) and torch.allclose(rep, torch.ones(2, 3))
assert abs(loss.item() - 1.1) < 1e-6, loss
loss.backward()
assert torch.all(post.grad == 0) and torch.all(prior.grad == 0)
"""},
        {"name": "Seeded inputs match the exact formula", "visibility": "unshown", "behavior": "rl.kl_estimator",
         "failure_message": "Mix unimix into the softmax probabilities, take the log, sum p * (log p - log q) over classes and then groups, floor each (B, T) sum at free_nats, and return mean(beta_dyn * dyn + beta_rep * rep).", "code": r"""
import torch
def logp(x, u):
    if u == 0:
        return torch.log_softmax(x, -1)
    return torch.log((1 - u) * torch.softmax(x, -1) + u / x.shape[-1])
def oracle(post, prior, free, bd, br, u):
    lp, lq = logp(post, u), logp(prior, u)
    kl = (lp.exp() * (lp - lq)).sum(-1).sum(-1)
    kl = torch.clamp(kl, min=free)
    return (bd * kl + br * kl).mean(), kl
for seed, shape, scale, free, bd, br, u in (
    (0, (2, 3, 4, 5), 1.0, 1.0, 1.0, 0.1, 0.01),
    (1, (3, 2, 8, 4), 3.0, 0.5, 0.5, 0.1, 0.01),
    (2, (1, 5, 32, 16), 2.0, 1.0, 1.0, 0.1, 0.01),
    (3, (2, 2, 3, 7), 6.0, 0.0, 0.8, 0.2, 0.1),
    (4, (4, 1, 6, 3), 4.0, 2.0, 1.0, 0.1, 0.0),
    (5, (2, 4, 5, 9), 8.0, 3.0, 0.5, 0.3, 0.3),
):
    g = torch.Generator().manual_seed(seed)
    post = scale * torch.randn(*shape, generator=g, dtype=torch.float64)
    prior = scale * torch.randn(*shape, generator=g, dtype=torch.float64)
    loss, dyn, rep = {fn}(post, prior, free_nats=free, beta_dyn=bd, beta_rep=br, unimix=u)
    want_loss, want_kl = oracle(post, prior, free, bd, br, u)
    assert dyn.shape == shape[:2] and rep.shape == shape[:2] and loss.shape == (), (dyn.shape, loss.shape)
    assert torch.allclose(dyn, want_kl, atol=1e-10, rtol=0), (seed, dyn, want_kl)
    assert torch.allclose(rep, want_kl, atol=1e-10, rtol=0), (seed, rep, want_kl)
    assert torch.allclose(loss, want_loss, atol=1e-10, rtol=0), (seed, loss, want_loss)
"""},
        {"name": "dyn trains only the prior and rep only the posterior", "visibility": "unshown", "behavior": "gradient.flow",
         "failure_message": "Detach the posterior inside dyn and the prior inside rep, so dyn's gradient reaches only prior_logits and rep's only post_logits.", "code": r"""
import torch
def logp(x, u):
    return torch.log((1 - u) * torch.softmax(x, -1) + u / x.shape[-1])
def kl(p, q, u, free):
    lp, lq = logp(p, u), logp(q, u)
    return torch.clamp((lp.exp() * (lp - lq)).sum(-1).sum(-1), min=free)
def grads(out, post, prior):
    gp, gq = torch.autograd.grad(out, [post, prior], allow_unused=True, retain_graph=True)
    gp = torch.zeros_like(post) if gp is None else gp
    gq = torch.zeros_like(prior) if gq is None else gq
    return gp, gq
for seed, free in ((0, 0.0), (1, 0.5), (2, 1.0)):
    g = torch.Generator().manual_seed(seed)
    base_post = 3 * torch.randn(2, 3, 4, 6, generator=g, dtype=torch.float64)
    base_prior = 3 * torch.randn(2, 3, 4, 6, generator=g, dtype=torch.float64)
    post = base_post.clone().requires_grad_(True)
    prior = base_prior.clone().requires_grad_(True)
    w = torch.rand(2, 3, generator=g, dtype=torch.float64)
    loss, dyn, rep = {fn}(post, prior, free_nats=free, beta_dyn=0.7, beta_rep=0.2, unimix=0.01)
    dyn_post, dyn_prior = grads((w * dyn).sum(), post, prior)
    rep_post, rep_prior = grads((w * rep).sum(), post, prior)
    assert torch.all(dyn_post == 0), "dyn sends gradient to the posterior"
    assert torch.all(rep_prior == 0), "rep sends gradient to the prior"
    p2 = base_post.clone().requires_grad_(True)
    q2 = base_prior.clone().requires_grad_(True)
    want_prior, = torch.autograd.grad((w * kl(p2.detach(), q2, 0.01, free)).sum(), [q2])
    want_post, = torch.autograd.grad((w * kl(p2, q2.detach(), 0.01, free)).sum(), [p2])
    assert torch.allclose(dyn_prior, want_prior, atol=1e-10, rtol=0), seed
    assert torch.allclose(rep_post, want_post, atol=1e-10, rtol=0), seed
    want_loss = (0.7 * kl(p2.detach(), q2, 0.01, free) + 0.2 * kl(p2, q2.detach(), 0.01, free)).mean()
    want_loss_post, want_loss_prior = torch.autograd.grad(want_loss, [p2, q2])
    loss_post, loss_prior = grads(loss, post, prior)
    assert torch.allclose(loss_prior, want_loss_prior, atol=1e-10, rtol=0), seed
    assert torch.allclose(loss_post, want_loss_post, atol=1e-10, rtol=0), seed
"""},
        {"name": "Clipped entries get exactly zero gradient", "visibility": "unshown", "behavior": "rl.clipping",
         "failure_message": "Floor with max(free_nats, kl) (or clamp) so a (B, T) entry below free_nats has value free_nats and exactly zero gradient; do not pass gradient through the floor.", "code": r"""
import torch
g = torch.Generator().manual_seed(3)
post = torch.randn(1, 4, 3, 5, generator=g, dtype=torch.float64)
prior = post.clone()
prior[0, 0] += 0.05 * torch.randn(3, 5, generator=g, dtype=torch.float64)   # tiny KL: clipped
prior[0, 1] += 4.0 * torch.randn(3, 5, generator=g, dtype=torch.float64)    # large KL: not clipped
prior[0, 2] += 0.10 * torch.randn(3, 5, generator=g, dtype=torch.float64)   # tiny KL: clipped
prior[0, 3] += 5.0 * torch.randn(3, 5, generator=g, dtype=torch.float64)    # large KL: not clipped
post.requires_grad_(True)
prior.requires_grad_(True)
loss, dyn, rep = {fn}(post, prior, free_nats=1.0, beta_dyn=1.0, beta_rep=0.1, unimix=0.01)
assert torch.all(dyn[0, [0, 2]] == 1.0) and torch.all(dyn[0, [1, 3]] > 1.0), dyn
loss.backward()
for t in (0, 2):
    assert torch.all(post.grad[0, t] == 0) and torch.all(prior.grad[0, t] == 0), t
for t in (1, 3):
    assert post.grad[0, t].abs().sum() > 0 and prior.grad[0, t].abs().sum() > 0, t
"""},
        {"name": "Unimix bounds the KL of a confident miss", "visibility": "unshown", "behavior": "numerics.stability",
         "failure_message": "Mix in probability space: probs = (1 - unimix) * softmax(logits) + unimix / C, then log; mixing the logits leaves the KL unbounded.", "code": r"""
import math, torch
C, u = 4, 0.01
post = torch.full((1, 1, 1, C), -50.0, dtype=torch.float64); post[..., 0] = 50.0
prior = torch.full((1, 1, 1, C), -50.0, dtype=torch.float64); prior[..., 1] = 50.0
loss, dyn, rep = {fn}(post, prior, free_nats=0.0, beta_dyn=1.0, beta_rep=0.0, unimix=u)
hi, lo = 1 - u + u / C, u / C
want = hi * math.log(hi / lo) + lo * math.log(lo / hi)
assert abs(dyn.item() - want) < 1e-10, (dyn.item(), want)
assert abs(loss.item() - want) < 1e-10, loss
"""},
        {"name": "Very negative logits stay finite", "visibility": "unshown", "behavior": "numerics.stability",
         "failure_message": "With unimix = 0 use log_softmax, not log(softmax): a probability that underflows to zero must not turn the KL into nan or inf.", "code": r"""
import torch
post = torch.tensor([0.0, -1e4, -2e4]).view(1, 1, 1, 3)
prior = torch.tensor([0.0, 0.0, -1e4]).view(1, 1, 1, 3)
for u in (0.0, 0.01):
    p = post.clone().requires_grad_(True)
    q = prior.clone().requires_grad_(True)
    loss, dyn, rep = {fn}(p, q, free_nats=0.0, beta_dyn=1.0, beta_rep=0.1, unimix=u)
    assert torch.isfinite(loss) and torch.isfinite(dyn).all() and torch.isfinite(rep).all(), (u, loss, dyn)
    loss.backward()
    assert torch.isfinite(p.grad).all() and torch.isfinite(q.grad).all(), u
lp = torch.log_softmax(post, -1)
lq = torch.log_softmax(prior, -1)
want = (lp.exp() * (lp - lq)).sum()
loss, dyn, rep = {fn}(post.double(), prior.double(), free_nats=0.0, unimix=0.0)
assert abs(dyn.item() - want.item()) < 1e-6, (dyn, want)
"""},
        {"name": "Betas weight the batch mean", "visibility": "unshown", "behavior": "metrics.averaging",
         "failure_message": "loss is the mean over all B * T entries of beta_dyn * dyn + beta_rep * rep; do not sum, and do not swap the two betas.", "code": r"""
import torch
g = torch.Generator().manual_seed(9)
post = 2 * torch.randn(3, 5, 4, 4, generator=g, dtype=torch.float64)
prior = 2 * torch.randn(3, 5, 4, 4, generator=g, dtype=torch.float64)
_, dyn, rep = {fn}(post, prior, free_nats=0.5, beta_dyn=1.0, beta_rep=0.0)
for bd, br in ((1.0, 0.1), (0.5, 0.1), (0.0, 1.0), (2.0, 0.0)):
    loss, d, r = {fn}(post, prior, free_nats=0.5, beta_dyn=bd, beta_rep=br)
    assert torch.allclose(d, dyn) and torch.allclose(r, rep)
    want = (bd * dyn + br * rep).mean()
    assert abs(loss.item() - want.item()) < 1e-10, (bd, br, loss, want)
"""},
        {"name": "Keeps dtype and leaves the inputs alone", "visibility": "unshown", "behavior": "tensor.dtype_device",
         "failure_message": "Compute in the input dtype (float32 stays float32, float64 stays float64) and never modify post_logits or prior_logits in place.", "code": r"""
import torch
g = torch.Generator().manual_seed(4)
for dtype in (torch.float32, torch.float64):
    post = torch.randn(2, 2, 3, 4, generator=g).to(dtype)
    prior = torch.randn(2, 2, 3, 4, generator=g).to(dtype)
    a, b = post.clone(), prior.clone()
    loss, dyn, rep = {fn}(post, prior)
    assert loss.dtype == dyn.dtype == rep.dtype == dtype, (loss.dtype, dyn.dtype, rep.dtype)
    assert torch.equal(post, a) and torch.equal(prior, b), "inputs were modified"
"""},
        {"name": "Rejects bad inputs", "visibility": "unshown", "behavior": "contract.signature",
         "failure_message": "Raise ValueError for inputs that are not 4-D floating tensors, for mismatched shapes, and for unimix outside [0, 1].", "code": r"""
import torch
ok = torch.zeros(1, 2, 3, 4)
cases = [
    (torch.zeros(2, 3, 4), torch.zeros(2, 3, 4), {}),
    (ok, torch.zeros(1, 2, 3, 5), {}),
    (ok, torch.zeros(1, 2, 4, 4), {}),
    (torch.zeros(1, 2, 3, 4, dtype=torch.long), torch.zeros(1, 2, 3, 4, dtype=torch.long), {}),
    (ok, ok, {"unimix": -0.1}),
    (ok, ok, {"unimix": 1.5}),
]
for post, prior, kw in cases:
    try:
        {fn}(post, prior, **kw)
    except ValueError:
        pass
    else:
        raise AssertionError(f"accepted {tuple(post.shape)} {post.dtype} vs {tuple(prior.shape)} {kw}")
"""},
    ],
    "solution": '''import torch


def _log_probs(logits, unimix):
    """Log-probabilities of each group after mixing in the uniform distribution."""
    if unimix == 0:
        return torch.log_softmax(logits, dim=-1)
    probs = torch.softmax(logits, dim=-1)
    probs = (1 - unimix) * probs + unimix / logits.shape[-1]
    return torch.log(probs)


def _kl(log_p, log_q):
    """KL(p || q) per group, summed over the classes and then the S groups -> (B, T)."""
    per_group = (log_p.exp() * (log_p - log_q)).sum(dim=-1)
    return per_group.sum(dim=-1)


def rssm_kl_loss(post_logits, prior_logits, free_nats=1.0, beta_dyn=1.0, beta_rep=0.1, unimix=0.01):
    for name, x in (("post_logits", post_logits), ("prior_logits", prior_logits)):
        if not torch.is_tensor(x) or x.ndim != 4 or not x.is_floating_point():
            raise ValueError(f"{name} must be a floating tensor of shape (B, T, S, C)")
    if post_logits.shape != prior_logits.shape:
        raise ValueError("post_logits and prior_logits must have the same shape")
    if not 0 <= unimix <= 1:
        raise ValueError("unimix must be in [0, 1]")
    log_post = _log_probs(post_logits, unimix)
    log_prior = _log_probs(prior_logits, unimix)
    dyn = _kl(log_post.detach(), log_prior)   # trains the prior toward the posterior
    rep = _kl(log_post, log_prior.detach())   # trains the posterior toward the prior
    dyn = torch.clamp(dyn, min=free_nats)
    rep = torch.clamp(rep, min=free_nats)
    loss = (beta_dyn * dyn + beta_rep * rep).mean()
    return loss, dyn, rep
''',
    "interview_questions": interview(
        concept=[
            "Why does DreamerV3 split the KL between posterior and prior into a dynamics term and a representation term with different weights?",
            "What are free bits, and why are they applied to the KL summed over all latent groups rather than to each group?",
        ],
        deep_dive=[
            "Where exactly do the stop-gradients go in the dyn and rep terms, and how would you verify the gradient routing with autograd?",
            "Why is unimix applied to the probabilities and not to the logits, and what bound does it give on the KL?",
            "What happens to the gradient of an entry whose KL is below the free-bits floor, and why can that be a problem late in training?",
        ],
        tradeoffs=[
            "Discrete categorical latents with straight-through gradients versus Gaussian latents with the reparameterization trick in a world model?",
            "The current DreamerV3 uses beta_dyn 1.0 while the first paper version and the PyTorch port use 0.5. What changes in training when beta_dyn is lowered relative to beta_rep?",
        ],
    ),
}
