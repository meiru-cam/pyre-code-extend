"""Router z-loss for logit stability."""

TASK = {
    "title": "Router z-loss",
    "difficulty": "Medium",
    "version": 1,
    "function_name": "router_z_loss",
    "description_en": r"""Implement the router z-loss that stabilizes Mixture-of-Experts training.

Signature: 'router_z_loss(router_logits) -> scalar'

'router_logits' has shape '(tokens, num_experts)'. The z-loss penalizes large router logits by squaring the log-partition function of each token and averaging over tokens:

'z_loss = mean_over_tokens( logsumexp(router_logits, dim=-1) ** 2 )'

Return a scalar tensor that stays differentiable with respect to 'router_logits'.

The log-partition 'logsumexp' grows with the magnitude of the logits, so squaring and minimizing it keeps the router's pre-softmax scores small. Small logits keep the softmax well-conditioned, reduce round-off in low precision, and prevent a few experts from acquiring runaway confidence early in training. Because this term only shapes logit scale, it complements — it does not replace — the load-balancing loss that equalizes expert usage.

Use a numerically stable log-sum-exp: exponentiating raw logits overflows for large magnitudes, while 'torch.logsumexp' subtracts the row max first. Average, do not sum, over tokens so the term's scale is independent of batch size. Validate that the input is a two-dimensional floating-point tensor.""",
    "advisory_prerequisites": ["softmax", "moe_topk_router", "moe_load_balance"],
    "hints": [
        {"level": 1, "kind": "questions", "content": "What quantity summarizes how large a token's logits are? Why square it rather than use it directly? What happens to exp(logit) when a logit is very large, and how does logsumexp avoid it? Should the batch dimension be summed or averaged?"},
        {"level": 2, "kind": "analysis", "content": "Compute the per-token log-partition with torch.logsumexp over the expert dimension, square it elementwise, and take the mean over tokens. The result is a scalar that keeps gradients to router_logits. Avoid log(exp(logits).sum()) because it overflows; logsumexp is the stable form the penalty targets."},
    ],
    "model_connections": [
        "ST-MoE introduced the router z-loss to stabilize sparse training, adding a small weighted penalty on the squared log-partition of router logits.",
        "Switch Transformers' router_z_loss_func in Transformers computes logsumexp over experts, squares it, and averages, exactly this contract.",
        "DeepSeek and Qwen MoE stacks keep router logits well-scaled so the softmax gate and top-k selection stay well-conditioned in low precision.",
    ],
    "pro_con_analysis": {
        "pros": [
            "Bounds the router log-partition, keeping the gate softmax numerically stable in bf16/fp16.",
            "Cheap scalar penalty with a clean gradient to the router logits.",
            "Complements load balancing without changing which experts are selected.",
        ],
        "cons": [
            "An overly large weight can suppress useful logit separation between experts.",
            "It shapes scale only; it does not fix expert under-utilization on its own.",
            "Adds one more coefficient to tune alongside the balance-loss weight.",
        ],
    },
    "sources": [
        {"kind": "code", "url": "https://github.com/huggingface/transformers", "commit": "7ea2320c76117e6742364808a666ef6f2fb40a67", "path": "src/transformers/models/switch_transformers/modeling_switch_transformers.py", "symbol": "router_z_loss_func", "license": "Apache-2.0", "adapted": "Squared log-partition of router logits averaged over tokens.", "simplifications": "Operates on a flat (tokens, experts) logit matrix without the sequence/model wrapper or loss weighting."},
        {"kind": "paper", "url": "https://arxiv.org/abs/2202.08906", "section": "3.2 Router z-loss"},
    ],
    "tests": [
        {"name": "Hand-calculated squared log-partition", "behavior": "numerics.stability", "code": r"""
import torch
import math
logits=torch.zeros(1,2)
# logsumexp([0,0]) = log(2); z = log(2)**2
assert torch.allclose({fn}(logits), torch.tensor(math.log(2.0)**2))
assert {fn}(logits).ndim==0
"""},
        {"name": "Rejects non-2D input", "behavior": "edge.empty_or_boundary", "code": r"""
import torch
for bad in (torch.randn(4), torch.randn(2,3,4)):
    try:
        {fn}(bad)
    except (ValueError, RuntimeError):
        pass
    else:
        raise AssertionError('non-2D router logits must raise')
"""},
        {"name": "Independent seeded oracle over tokens", "behavior": "numerics.stability", "visibility": "unshown", "failure_message": "z-loss must be the mean over tokens of the squared logsumexp of the expert logits.", "code": r"""
import torch
for seed,shape in [(3,(5,7)),(9,(1,4)),(21,(6,3))]:
    generator=torch.Generator().manual_seed(seed)
    logits=torch.randn(*shape,generator=generator,dtype=torch.float64)
    expected=torch.logsumexp(logits,dim=-1).square().mean()
    actual={fn}(logits)
    assert actual.ndim==0 and actual.dtype==logits.dtype
    assert torch.allclose(actual,expected,atol=1e-12)
"""},
        {"name": "Gradient reaches the router logits", "behavior": "gradient.flow", "visibility": "unshown", "failure_message": "z-loss must stay differentiable with respect to router_logits.", "code": r"""
import torch
logits=torch.randn(4,6,dtype=torch.float64,requires_grad=True)
loss={fn}(logits)
loss.backward()
assert logits.grad is not None and bool(torch.count_nonzero(logits.grad))
assert bool(torch.isfinite(logits.grad).all())
"""},
        {"name": "Stable and larger for large-magnitude logits", "behavior": "numerics.stability", "visibility": "unshown", "failure_message": "Use a stable log-sum-exp: large logits must yield a finite penalty that exceeds small logits.", "code": r"""
import torch
small={fn}(torch.zeros(3,8))
big={fn}(torch.full((3,8),1000.0))
assert bool(torch.isfinite(big)) and float(big)>float(small)
"""},
        {"name": "Averaged over tokens, not summed", "behavior": "numerics.stability", "visibility": "unshown", "failure_message": "The penalty must average over tokens so its scale is independent of batch size.", "code": r"""
import torch
generator=torch.Generator().manual_seed(5)
logits=torch.randn(4,6,generator=generator,dtype=torch.float64)
duplicated=torch.cat([logits,logits],dim=0)
assert torch.allclose({fn}(logits),{fn}(duplicated),atol=1e-12)
"""},
        {"name": "Expert permutation invariance", "behavior": "state.invariant", "visibility": "unshown", "failure_message": "The log-partition is over experts, so permuting expert columns must not change the loss.", "code": r"""
import torch
generator=torch.Generator().manual_seed(7)
logits=torch.randn(5,6,generator=generator,dtype=torch.float64)
perm=torch.randperm(6,generator=generator)
assert torch.allclose({fn}(logits),{fn}(logits[:,perm]),atol=1e-12)
"""},
    ],
    "solution": r'''def router_z_loss(router_logits):
    if not isinstance(router_logits, torch.Tensor) or router_logits.ndim != 2:
        raise ValueError("router_logits must be a (tokens, num_experts) tensor")
    if not router_logits.is_floating_point():
        raise ValueError("router_logits must be floating point")
    log_partition = torch.logsumexp(router_logits, dim=-1)
    return log_partition.square().mean()''',
    "demo": r"""logits=torch.randn(8,16)
print(float(router_z_loss(logits)))""",
}
