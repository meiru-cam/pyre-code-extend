"""SimPO and IPO: two preference losses that change DPO's reference and link function."""

from ._interview import interview

TASK = {
    "title": "SimPO and IPO Losses",
    "difficulty": "Medium",
    "version": 1,
    "function_name": "preference_loss",
    "description_en": r"""Implement the SimPO and IPO preference losses.

**Signature:** `preference_loss(chosen_logps, rejected_logps, loss_type, beta, chosen_lengths=None, rejected_lengths=None, gamma=0.0, ref_chosen_logps=None, ref_rejected_logps=None) -> Tensor`

**Parameters:**
- `chosen_logps`, `rejected_logps` — float tensors `(B,)`: the policy's summed log-probability of each response.
- `loss_type` — `"simpo"` or `"ipo"`.
- `beta` — positive float.
- `chosen_lengths`, `rejected_lengths` — integer tensors `(B,)` of response token counts. Required for SimPO.
- `gamma` — SimPO target reward margin.
- `ref_chosen_logps`, `ref_rejected_logps` — float tensors `(B,)` from the reference model. Required for IPO.

**Returns:** scalar tensor, the mean over the batch of the per-pair loss:

    simpo:  x = beta * (chosen_logps / chosen_lengths - rejected_logps / rejected_lengths) - gamma
            loss = -log(sigmoid(x))
    ipo:    h = (chosen_logps - ref_chosen_logps) - (rejected_logps - ref_rejected_logps)
            loss = (h - 1 / (2 * beta)) ** 2

**Constraints:**
- SimPO uses no reference model. IPO uses no sigmoid.
- The SimPO loss must stay finite for margins as large as 1000 in either direction.
- Keep the result differentiable with respect to the policy log-probabilities.
- Raise `ValueError` for an unknown `loss_type`, `beta <= 0`, a missing required argument, or a length that is not positive.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**SimPO.** DPO's implicit reward is a log-ratio against a frozen reference, which costs a second model and does not match how generation ranks sequences. SimPO replaces it with the length-normalized average log-probability, the same score beam search uses, and asks the chosen response to win by at least `gamma / beta`.

**IPO.** With deterministic preferences DPO keeps pushing the log-ratio gap toward infinity, which overfits and ignores the KL regularizer. IPO regresses the gap toward a finite target `1 / (2 * beta)` with a squared loss, so a larger `beta` means a smaller target and stronger regularization. TRL applies this same formula to length-averaged log-probabilities, which keeps `beta` comparable across response lengths.""",
    "advisory_prerequisites": ["dpo_loss"],
    "hints": [
        {"level": 1, "kind": "questions", "content": "Which loss uses the reference model and which uses response lengths? What is -log(sigmoid(x)) as a stable function? What target does IPO regress the gap toward?"},
        {"level": 2, "kind": "analysis", "content": "Validate arguments per loss_type first. SimPO: divide summed logps by lengths, form x, and use F.softplus(-x) instead of -log(sigmoid(x)). IPO: form h from the two log-ratios and square h - 1 / (2 * beta). Return the batch mean."},
    ],
    "model_connections": [
        "TRL's DPOTrainer and CPOTrainer expose `loss_type=\"ipo\"` and `loss_type=\"simpo\"`; SimPO was used to train strong Gemma and Llama chat models without a reference model.",
    ],
    "pro_con_analysis": {
        "pros": ["SimPO drops the reference model and aligns the training score with generation; IPO bounds the gap and resists overfitting on deterministic preferences."],
        "cons": ["SimPO loses the KL anchor to the reference and needs careful tuning of beta and gamma; IPO's squared loss is sensitive to beta and to label noise."],
    },
    "sources": [{'kind': 'code',
      'url': 'https://github.com/huggingface/trl',
      'commit': 'd947c4f5098c8d6ca30ea7ff58b98e6570fd6a01',
      'path': 'trl/experimental/cpo/cpo_trainer.py',
      'symbol': "CPOTrainer.cpo_loss (loss_type='simpo')",
      'license': 'Apache-2.0',
      'adapted': 'SimPO: -logsigmoid(beta * (average chosen logp - average rejected logp) - gamma).',
      'simplifications': 'No label smoothing or AlphaPO reward transform; lengths are passed explicitly instead of '
                         'averaging inside the trainer.'},
     {'kind': 'code',
      'url': 'https://github.com/huggingface/trl',
      'commit': 'd947c4f5098c8d6ca30ea7ff58b98e6570fd6a01',
      'path': 'trl/trainer/dpo_trainer.py',
      'symbol': "DPOTrainer loss_type='ipo' branch",
      'license': 'Apache-2.0',
      'adapted': 'IPO: squared distance of the log-ratio gap from 1 / (2 * beta).',
      'simplifications': 'DIVERGES: TRL divides each log-ratio by its completion length before taking the gap; this '
                         'exercise uses the summed form of the IPO paper (Eq. 17). Passing length-averaged '
                         'log-probabilities reproduces TRL.'},
     {'kind': 'paper',
      'url': 'https://arxiv.org/abs/2405.14734',
      'equation': 'SimPO paper, Eq. 6',
      'note': 'The SimPO objective.'},
     {'kind': 'paper',
      'url': 'https://arxiv.org/abs/2310.12036',
      'equation': 'IPO paper (Azar et al.), Eq. 17',
      'note': 'The IPO objective.'}],
    "tests": [
        {"name": "SimPO and IPO on a small batch", "behavior": "rl.logprob", "code": r"""
import math, torch
chosen = torch.tensor([-4.0, -6.0]); rejected = torch.tensor([-9.0, -4.0])
out = {fn}(chosen, rejected, "simpo", beta=2.0, chosen_lengths=torch.tensor([2, 3]), rejected_lengths=torch.tensor([3, 2]), gamma=0.5)
xs = [2.0 * (-2.0 + 3.0) - 0.5, 2.0 * (-2.0 + 2.0) - 0.5]
want = sum(math.log1p(math.exp(-x)) for x in xs) / 2
assert out.ndim == 0 and abs(out.item() - want) < 1e-6, (out, want)
out = {fn}(chosen, rejected, "ipo", beta=0.5, ref_chosen_logps=torch.tensor([-5.0, -5.0]), ref_rejected_logps=torch.tensor([-8.0, -5.0]))
hs = [(-4.0 + 5.0) - (-9.0 + 8.0), (-6.0 + 5.0) - (-4.0 + 5.0)]
want = sum((h - 1.0) ** 2 for h in hs) / 2
assert abs(out.item() - want) < 1e-6, (out, want)
"""},
        {"name": "Matches a seeded oracle and gradient for both losses", "visibility": "unshown", "behavior": "rl.logprob", "failure_message": "SimPO length-normalizes and subtracts gamma; IPO squares the log-ratio gap minus 1 / (2 * beta).", "code": r"""
import math, random, torch
for seed in (5, 29, 64):
    rng = random.Random(seed)
    B = rng.randint(2, 5)
    beta = rng.choice([0.1, 0.5, 2.0]); gamma = rng.uniform(0, 1.5)
    c = [rng.uniform(-40, -1) for _ in range(B)]; r = [rng.uniform(-40, -1) for _ in range(B)]
    lc = [rng.randint(1, 20) for _ in range(B)]; lr = [rng.randint(1, 20) for _ in range(B)]
    rc = [rng.uniform(-40, -1) for _ in range(B)]; rr = [rng.uniform(-40, -1) for _ in range(B)]
    x = torch.tensor(c, dtype=torch.float64, requires_grad=True)
    out = {fn}(x, torch.tensor(r, dtype=torch.float64), "simpo", beta=beta,
               chosen_lengths=torch.tensor(lc), rejected_lengths=torch.tensor(lr), gamma=gamma)
    margins = [beta * (c[i] / lc[i] - r[i] / lr[i]) - gamma for i in range(B)]
    want = sum(math.log1p(math.exp(-m)) for m in margins) / B
    assert abs(out.item() - want) < 1e-9, (seed, "simpo", out, want)
    out.backward()
    grad = [-(1 / (1 + math.exp(m))) * beta / lc[i] / B for i, m in enumerate(margins)]
    assert torch.allclose(x.grad, torch.tensor(grad, dtype=torch.float64), atol=1e-12), (seed, x.grad, grad)
    x = torch.tensor(c, dtype=torch.float64, requires_grad=True)
    out = {fn}(x, torch.tensor(r, dtype=torch.float64), "ipo", beta=beta,
               ref_chosen_logps=torch.tensor(rc, dtype=torch.float64), ref_rejected_logps=torch.tensor(rr, dtype=torch.float64))
    hs = [(c[i] - rc[i]) - (r[i] - rr[i]) for i in range(B)]
    want = sum((h - 1 / (2 * beta)) ** 2 for h in hs) / B
    assert abs(out.item() - want) < 1e-8 * max(1.0, want), (seed, "ipo", out, want)
    out.backward()
    grad = [2 * (h - 1 / (2 * beta)) / B for h in hs]
    assert torch.allclose(x.grad, torch.tensor(grad, dtype=torch.float64), atol=1e-10), (seed, x.grad, grad)
"""},
        {"name": "SimPO stays finite for extreme margins", "visibility": "unshown", "behavior": "numerics.stability", "failure_message": "Use a stable form such as softplus(-x) instead of -log(sigmoid(x)).", "code": r"""
import torch
chosen = torch.tensor([-2000.0, 0.0], dtype=torch.float64, requires_grad=True)
rejected = torch.tensor([0.0, -2000.0], dtype=torch.float64)
ones = torch.tensor([1, 1])
out = {fn}(chosen, rejected, "simpo", beta=0.5, chosen_lengths=ones, rejected_lengths=ones)
assert torch.isfinite(out) and abs(out.item() - 500.0) < 1e-6, out
out.backward()
assert torch.isfinite(chosen.grad).all(), chosen.grad
"""},
        {"name": "Rejects invalid or missing arguments", "visibility": "unshown", "behavior": "contract.signature", "failure_message": "SimPO needs positive lengths, IPO needs reference log-probabilities, and beta must be positive.", "code": r"""
import torch
x = torch.zeros(2); ones = torch.tensor([1, 1])
cases = [
    dict(loss_type="dpo", beta=1.0),
    dict(loss_type="simpo", beta=1.0),
    dict(loss_type="simpo", beta=1.0, chosen_lengths=torch.tensor([0, 1]), rejected_lengths=ones),
    dict(loss_type="simpo", beta=0.0, chosen_lengths=ones, rejected_lengths=ones),
    dict(loss_type="ipo", beta=1.0),
    dict(loss_type="ipo", beta=-1.0, ref_chosen_logps=x, ref_rejected_logps=x),
]
for kwargs in cases:
    try:
        {fn}(x, x, **kwargs)
    except ValueError:
        pass
    else:
        raise AssertionError(f"accepted {kwargs}")
"""},
    ],
    "solution": '''import torch
import torch.nn.functional as F

def preference_loss(chosen_logps, rejected_logps, loss_type, beta, chosen_lengths=None, rejected_lengths=None,
                    gamma=0.0, ref_chosen_logps=None, ref_rejected_logps=None):
    if beta <= 0:
        raise ValueError("beta must be positive")
    if loss_type == "simpo":
        if chosen_lengths is None or rejected_lengths is None:
            raise ValueError("SimPO needs response lengths")
        if bool((chosen_lengths <= 0).any()) or bool((rejected_lengths <= 0).any()):
            raise ValueError("lengths must be positive")
        margin = beta * (chosen_logps / chosen_lengths - rejected_logps / rejected_lengths) - gamma
        return F.softplus(-margin).mean()
    if loss_type == "ipo":
        if ref_chosen_logps is None or ref_rejected_logps is None:
            raise ValueError("IPO needs reference log-probabilities")
        gap = (chosen_logps - ref_chosen_logps) - (rejected_logps - ref_rejected_logps)
        return ((gap - 1 / (2 * beta)) ** 2).mean()
    raise ValueError(f"unknown loss_type {loss_type!r}")
''',
    "interview_questions": interview(
        concept=[
            "What problem with DPO does SimPO address, and what does it use instead of the reference log-ratio?",
            "What problem with DPO does IPO address, and why does a finite target help?",
        ],
        deep_dive=[
            "Write both losses. Why does SimPO divide by the response length, and what does gamma do?",
            "Derive the IPO gradient with respect to the chosen log-probability. What happens once the gap reaches 1 / (2 * beta)?",
            "Why can -log(sigmoid(x)) overflow, and what stable form avoids it?",
        ],
        tradeoffs=[
            "Without a reference model, what stops SimPO from drifting far from the SFT model? How would you detect drift?",
            "DPO, IPO and SimPO on noisy human labels versus clean verifiable labels: which would you pick for each?",
        ],
    ),
}
