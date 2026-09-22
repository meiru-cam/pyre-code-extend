"""The k3 KL estimator — the low-variance, non-negative KL penalty used by GRPO."""

TASK = {
    "title": "k3 KL Penalty",
    "difficulty": "Medium",
    "version": 1,
    "function_name": "k3_kl_penalty",
    "description_en": r"""Implement the k3 estimator of the KL divergence between the current policy and a frozen reference model.

**Signature:** `k3_kl_penalty(logprobs, ref_logprobs) -> Tensor`

**Parameters:**
- `logprobs` — float tensor. Per-token log-probabilities under the policy being trained.
- `ref_logprobs` — float tensor of the same shape. The same tokens under the frozen reference model.

**Returns:** float tensor of the same shape. Elementwise, with `r = ref_logprobs - logprobs`:

    penalty = exp(r) - r - 1

**Constraints:**
- Return one value per element. Do not reduce and do not mask; the caller owns both.
- The log-ratio is reference minus policy, in that order. Reversing it gives a different, wrong quantity.
- Keep the result differentiable with respect to `logprobs`. Do not detach.
- Do not clamp `r`. An overflow is a real signal, not a bug to hide.
- Raise a `ValueError` when the two tensors do not have the same shape.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why this form.** The quantity being estimated is the KL divergence from the policy to the reference, and only samples drawn from the policy are available. Three estimators are common:

    k1 = -r                  unbiased, but negative on individual samples
    k2 = r * r / 2           always non-negative, but biased
    k3 = exp(r) - r - 1      unbiased and always non-negative

k3 is the one GRPO uses. It is non-negative for every real `r` because `exp(r) >= 1 + r` with equality only at zero, so the penalty is exactly zero when the policy and the reference agree on a token and grows in both directions as they diverge. Its expectation under the policy equals the true KL, so it is unbiased, and its per-sample variance is far lower than k1's — which matters because the penalty is estimated from a handful of rollouts rather than from the full distribution.

**Why not clamp.** `r` is a difference of log-probabilities and can be large early in training. A strongly positive `r` makes `exp(r)` overflow, which says the policy has collapsed away from the reference. Production stacks clip the ratio inside the loss instead of hiding it here.""",
    "advisory_prerequisites": ["per_token_logprobs", "softmax"],
    "hints": [
        {
            "level": 1,
            "kind": "questions",
            "content": "The formula is one line, so the work is in getting it exactly right. Which of the two inputs is the minuend in the log-ratio — the policy or the reference? What happens to the whole expression when the two log-probabilities are equal, and does your implementation return exactly that? Try a few values of r on paper, both positive and negative: does your expression ever go below zero, and what does it mean if it does? Last, which input should carry a gradient, and does any operation in your version cut it off?",
        },
        {
            "level": 2,
            "kind": "analysis",
            "content": "Compute `r = ref_logprobs - logprobs` first, as a named intermediate, then return `torch.exp(r) - r - 1`. Two sign errors are easy to make and neither raises: computing `logprobs - ref_logprobs` gives you a different estimator that is still non-negative, so shape and sign checks pass while the penalty pushes the wrong way; and writing `exp(r) - r + 1` shifts everything by two so the penalty never reaches zero for an unchanged policy. Check both against the r=0 case, where the correct answer is exactly 0. Gradients flow through both `exp(r)` and the linear `-r` term back to `logprobs`; the derivative with respect to `r` is `exp(r) - 1`, so at r=0 the gradient vanishes, which is why an on-policy step at initialization feels no KL pressure. Validate the shapes before computing so a broadcast does not silently produce a larger tensor.",
        },
    ],
    "model_connections": [
        "simple_GRPO's GRPO_step computes 'torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1' inline, then subtracts beta times this term from the per-token objective.",
        "nano-aha-moment's compute_pg_loss builds the same 'exp(ref_logratio) - 1 - ref_logratio' expression and weights it with KL_COEFFICIENT.",
        "John Schulman's note on approximating KL is the origin of the k1/k2/k3 naming and the argument that k3 is both unbiased and non-negative.",
        "OpenRLHF exposes the same estimator behind a flag so the KL term can be applied either inside the reward or as a separate loss component.",
    ],
    "pro_con_analysis": {
        "pros": [
            "Unbiased and non-negative at the same time, which k1 and k2 cannot both achieve.",
            "Much lower per-sample variance than k1, so a small number of rollouts still gives a usable penalty.",
            "Exactly zero when the policy matches the reference, so an unchanged policy pays nothing.",
        ],
        "cons": [
            "Overflows for strongly positive log-ratios, so a diverging policy produces inf rather than a bounded penalty.",
            "Asymmetric in its growth: it punishes a policy that under-weights a reference token much more than the reverse.",
            "A per-token KL penalty constrains local token choice, which is a blunt proxy for constraining the behaviour of the whole response.",
        ],
    },
    "sources": [
        {
            "kind": "code",
            "url": "https://github.com/McGill-NLP/nano-aha-moment",
            "commit": "5314e6f8fc60efaa0f4b8fdb62353e9bd451638a",
            "path": "nano_r1_script.py",
            "symbol": "compute_pg_loss",
            "license": "MIT",
            "adapted": "The per-token KL term exp(ref_logratio) - 1 - ref_logratio computed between the policy and the frozen reference model.",
            "simplifications": "Isolated as a pure elementwise function over two log-probability tensors: no reference model forward pass, no coefficient, no masking and no reduction.",
        },
        {
            "kind": "paper",
            "url": "https://arxiv.org/abs/2402.03300",
            "section": "4.1 From PPO to GRPO, unbiased KL estimator",
        },
    ],
    "tests": [
        {
            "name": "Zero when the policy matches the reference",
            "behavior": "rl.kl_estimator",
            "code": r"""
import torch
lp = torch.tensor([-0.5, -2.0, -7.25])
out = {fn}(lp, lp.clone())
assert out.shape == lp.shape, f'expected {tuple(lp.shape)}, got {tuple(out.shape)}'
assert torch.allclose(out, torch.zeros(3), atol=1e-7), out
""",
        },
        {
            "name": "Hand-calculated log-ratio of one",
            "behavior": "rl.kl_estimator",
            "code": r"""
import torch
import math
# r = ref - policy = -1.0 - (-2.0) = 1.0  ->  exp(1) - 1 - 1
lp = torch.tensor([-2.0])
ref = torch.tensor([-1.0])
expected = math.exp(1.0) - 1.0 - 1.0
assert torch.allclose({fn}(lp, ref), torch.tensor([expected]), atol=1e-6), {fn}(lp, ref)
""",
        },
        {
            "name": "Never negative, for either direction of divergence",
            "visibility": "unshown",
            "behavior": "rl.kl_estimator",
            "failure_message": "The penalty went negative. exp(r) - r - 1 is non-negative for every real r; a negative value means the k1 estimator or a sign error.",
            "code": r"""
import torch
torch.manual_seed(0)
for _ in range(5):
    lp = torch.randn(64) * 2.0 - 1.0
    ref = torch.randn(64) * 2.0 - 1.0
    out = {fn}(lp, ref)
    assert torch.isfinite(out).all(), 'produced inf or nan on moderate inputs'
    assert (out >= -1e-6).all(), f'minimum was {out.min().item()}'
""",
        },
        {
            "name": "Matches the closed-form oracle",
            "visibility": "unshown",
            "behavior": "rl.kl_estimator",
            "failure_message": "Values disagree with exp(ref - policy) - (ref - policy) - 1. Check the order of subtraction and the constant term.",
            "code": r"""
import torch
torch.manual_seed(1)
for shape in ((6,), (3, 5), (2, 3, 4)):
    lp = torch.randn(shape)
    ref = torch.randn(shape)
    r = ref - lp
    expected = torch.exp(r) - r - 1.0
    out = {fn}(lp, ref)
    assert out.shape == expected.shape, f'expected {tuple(expected.shape)}, got {tuple(out.shape)}'
    assert torch.allclose(out, expected, atol=1e-5), (out - expected).abs().max()
""",
        },
        {
            "name": "Distinguishes k3 from k1 and k2",
            "visibility": "unshown",
            "behavior": "rl.kl_estimator",
            "failure_message": "The result matches the k1 or k2 estimator rather than k3. Use exp(r) - r - 1, not -r and not r squared over two.",
            "code": r"""
import torch
lp = torch.tensor([-3.0, -0.25, -1.0])
ref = torch.tensor([-1.0, -2.5, -1.5])
r = ref - lp
out = {fn}(lp, ref)
k1 = -r
k2 = r * r / 2.0
assert not torch.allclose(out, k1, atol=1e-3), 'this is the k1 estimator'
assert not torch.allclose(out, k2, atol=1e-3), 'this is the k2 estimator'
assert torch.allclose(out, torch.exp(r) - r - 1.0, atol=1e-5), out
""",
        },
        {
            "name": "Asymmetric in the log-ratio",
            "visibility": "unshown",
            "behavior": "rl.kl_estimator",
            "failure_message": "Swapping the two arguments produced the same values. The log-ratio is reference minus policy, so the estimator is not symmetric.",
            "code": r"""
import torch
lp = torch.tensor([-4.0, -0.5])
ref = torch.tensor([-0.5, -4.0])
forward = {fn}(lp, ref)
reverse = {fn}(ref, lp)
assert not torch.allclose(forward, reverse, atol=1e-3), 'estimator is symmetric; check the subtraction order'
# A positive r grows much faster than the equivalent negative r.
assert forward[0] > forward[1], f'expected the positive log-ratio to dominate, got {forward}'
""",
        },
        {
            "name": "Gradients reach the policy log-probabilities",
            "visibility": "unshown",
            "behavior": "gradient.flow",
            "failure_message": "No gradient reached logprobs, or the gradient value is wrong. The derivative with respect to logprobs is 1 - exp(r).",
            "code": r"""
import torch
torch.manual_seed(2)
lp = torch.randn(8, requires_grad=True)
ref = torch.randn(8)
out = {fn}(lp, ref)
assert out.requires_grad, 'output is detached from the graph'
out.sum().backward()
assert lp.grad is not None, 'logprobs received no gradient'
r = (ref - lp).detach()
assert torch.allclose(lp.grad, 1.0 - torch.exp(r), atol=1e-4), lp.grad
""",
        },
        {
            "name": "Gradient vanishes at the reference",
            "visibility": "unshown",
            "behavior": "numerics.stability",
            "failure_message": "The gradient is not zero where the policy equals the reference. At r=0 the derivative 1 - exp(r) is exactly zero.",
            "code": r"""
import torch
lp = torch.tensor([-1.5, -0.25, -6.0], requires_grad=True)
ref = lp.detach().clone()
out = {fn}(lp, ref)
out.sum().backward()
assert torch.allclose(lp.grad, torch.zeros(3), atol=1e-6), lp.grad
""",
        },
        {
            "name": "Rejects mismatched shapes",
            "visibility": "unshown",
            "behavior": "contract.signature",
            "failure_message": "Mismatched shapes were broadcast instead of rejected. Raise ValueError when the two tensors differ in shape.",
            "code": r"""
import torch
pairs = ((torch.randn(4), torch.randn(5)),
         (torch.randn(2, 3), torch.randn(3)),
         (torch.randn(4), torch.randn(4, 1)))
for lp, ref in pairs:
    try:
        {fn}(lp, ref)
    except ValueError:
        continue
    raise AssertionError(f'shapes {tuple(lp.shape)} and {tuple(ref.shape)} should raise ValueError')
""",
        },
        {
            "name": "Preserves dtype and per-element shape",
            "visibility": "unshown",
            "behavior": "tensor.dtype_device",
            "failure_message": "The output dtype changed or the result was reduced. Return one value per element in the input dtype.",
            "code": r"""
import torch
lp = torch.randn(3, 4, dtype=torch.float64)
ref = torch.randn(3, 4, dtype=torch.float64)
out = {fn}(lp, ref)
assert out.dtype == torch.float64, f'expected float64, got {out.dtype}'
assert out.shape == (3, 4), f'expected (3, 4), got {tuple(out.shape)}'
assert out.device == lp.device, f'expected {lp.device}, got {out.device}'
""",
        },
    ],
    "solution": '''import torch


def k3_kl_penalty(logprobs, ref_logprobs):
    if logprobs.shape != ref_logprobs.shape:
        raise ValueError(
            f"shape mismatch: {tuple(logprobs.shape)} vs {tuple(ref_logprobs.shape)}"
        )
    log_ratio = ref_logprobs - logprobs
    return torch.exp(log_ratio) - log_ratio - 1.0
''',
}
