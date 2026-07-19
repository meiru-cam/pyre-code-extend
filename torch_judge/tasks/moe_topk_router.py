"""Probability-normalized token-choice top-k MoE router."""

TASK = {
    "title": "Top-k MoE Router",
    "difficulty": "Medium",
    "version": 1,
    "function_name": "moe_topk_router",
    "description_en": r"""Implement token-choice top-k routing.

Signature: 'moe_topk_router(router_logits, k) -> (expert_indices, expert_weights)'

'router_logits' is a floating tensor of shape '(tokens, experts)'. Validate '1 <= k <= experts'. Compute softmax over the expert dimension, select the largest 'k' probabilities independently for each token, and renormalize only the selected probabilities to sum to one. Return long indices and floating weights, both shaped '(tokens, k)', on the input device. Preserve gradient flow from weights to router logits.

The explicit full softmax makes the selection-probability contract easy to inspect. Some frontier routers use sigmoid scores, group-limited selection, routing biases, or a route scale instead. This exercise is therefore a reusable baseline—not a claim of checkpoint equivalence.

Softmax is monotone, and normalizing after selection cancels the probability mass assigned to unselected experts. Therefore selecting logits and applying softmax only to those selected logits is numerically equivalent to this contract. The evaluator grades the observable routes and normalized weights; the full-softmax order remains an explanatory baseline for connecting routing probabilities to auxiliary losses.

Optional code reading: start at DeepSeek-V3 'Gate.forward' and Qwen3-MoE 'Qwen3MoeTopKRouter.forward'; compare their score functions and auxiliary objectives after your implementation if useful.""",
    "advisory_prerequisites": ["softmax", "moe", "moe_load_balance"],
    "hints": [
        {"level": 1, "kind": "questions", "content": "Which dimension represents experts? Is each token routed independently? After discarding experts, do the retained probabilities still sum to one?"},
        {"level": 2, "kind": "analysis", "content": "Apply softmax on dim=-1, use topk on that same dimension, then divide each retained weight row by its own sum. Return topk's indices first and do not detach the weights."},
    ],
    "model_connections": [
        "DeepSeek-V3 Gate.forward implements group-limited top-k routing with softmax or sigmoid scores, optional routing bias, and route scaling; unlike this exercise, its softmax branch does not renormalize the selected weights.",
        "Kimi K2 is an MoE with 384 experts, 8 selected experts per token, and a shared expert; SGLang's Kimi-compatible DeepseekV2MoE wires router logits into configurable grouped TopK and fused expert backends.",
        "GLM-4.5's Transformers implementation uses Glm4MoeTopkRouter for group-limited selection, optional normalization, routing scale, and Glm4MoeMoE expert execution.",
        "Qwen3-MoE computes router logits, softmax routing weights, and top-k experts, and can normalize selected weights when norm_topk_prob is enabled.",
    ],
    "pro_con_analysis": {
        "pros": [
            "Activates only k experts per token, making conditional capacity much larger than dense compute.",
            "Normalized selected gates give an interpretable convex mixture and differentiable gate values.",
        ],
        "cons": [
            "Hard top-k indices are discontinuous and receive no ordinary gradient through the selection decision.",
            "Token-choice routing can overload popular experts and requires balancing or capacity policy.",
            "A full softmax over all experts can be more expensive than specialized grouped or sigmoid routers.",
        ],
    },
    "sources": [
        {"kind": "code", "url": "https://github.com/deepseek-ai/DeepSeek-V3", "commit": "9b4e9788e4a3a731f7567338ed15d3ec549ce03b", "path": "inference/model.py", "symbol": "Gate.forward", "license": "MIT", "adapted": "Per-token scoring, top-k expert indices, selected routing weights, and optional renormalization.", "simplifications": "Softmax only; no grouped selection, sigmoid branch, routing bias, route scale, shared experts, or expert parallelism."},
        {"kind": "code", "url": "https://github.com/huggingface/transformers", "commit": "7ea2320c76117e6742364808a666ef6f2fb40a67", "path": "src/transformers/models/qwen3_moe/modeling_qwen3_moe.py", "symbol": "Qwen3MoeTopKRouter.forward and Qwen3MoeSparseMoeBlock.forward", "license": "Apache-2.0", "adapted": "Softmax, per-token top-k selection, and selected-weight normalization boundary.", "simplifications": "Standalone logits function with no router projection, expert module, shared expert, padding mask, or fused kernels."},
        {"kind": "code", "url": "https://github.com/MoonshotAI/Kimi-K2", "commit": "1b4022bbb7187cf4011a8bdf0b4cd10e2daa26c4", "path": "docs/deploy_guidance.md", "symbol": "expert-parallel deployment options", "license": "Modified MIT", "adapted": "Production context for expert-parallel routing and dispatch.", "simplifications": "No distributed runtime or Kimi checkpoint-compatible router."},
        {"kind": "code", "url": "https://github.com/sgl-project/sglang", "commit": "d4801be447738522d2c62b49857c83787293be6f", "path": "python/sglang/srt/models/deepseek_v2.py", "symbol": "MoEGate.forward and DeepseekV2MoE.__init__/forward", "license": "Apache-2.0", "adapted": "The Kimi-compatible runtime boundary from router GEMM through configurable grouped TopK into fused experts.", "simplifications": "No quantization, fused kernels, shared-expert fusion, expert parallelism, or device-specific routing backends."},
        {"kind": "code", "url": "https://github.com/huggingface/transformers", "commit": "7ea2320c76117e6742364808a666ef6f2fb40a67", "path": "src/transformers/models/glm4_moe/modeling_glm4_moe.py", "symbol": "Glm4MoeTopkRouter.forward and Glm4MoeMoE.forward", "license": "Apache-2.0", "adapted": "Group-limited top-k selection, optional selected-weight normalization, route scaling, and the router-to-expert API.", "simplifications": "No router projection, group restriction, correction bias, shared experts, or model wrapper."},
        {"kind": "paper", "url": "https://arxiv.org/abs/2412.19437", "section": "2.1 Basic Architecture"},
        {"kind": "paper", "url": "https://arxiv.org/abs/2505.09388", "section": "2.1 Model Architecture"},
        {"kind": "paper", "url": "https://arxiv.org/abs/2507.20534", "section": "2.3 Model Architecture"},
        {"kind": "paper", "url": "https://arxiv.org/abs/2508.06471", "section": "2.1 Architecture"},
    ],
    "tests": [
        {"name": "Hand-calculated selection and normalization", "behavior": "routing.selection", "code": r"""
import torch
logits = torch.tensor([[0.0, 1.0, 2.0], [3.0, 1.0, 2.0]], dtype=torch.float64)
indices, weights = {fn}(logits, 2)
probs = torch.softmax(logits, dim=-1)
expected_values, expected_indices = torch.topk(probs, 2, dim=-1)
expected_values = expected_values / expected_values.sum(-1, keepdim=True)
assert torch.equal(indices, expected_indices)
assert torch.allclose(weights, expected_values, atol=1e-12)
"""},
        {"name": "Rejects invalid k", "behavior": "edge.empty_or_boundary", "code": r"""
import torch
for k in (0, 4, -1):
    try:
        {fn}(torch.randn(2, 3), k)
    except ValueError:
        pass
    else:
        raise AssertionError('invalid k must raise ValueError')
"""},
        {"name": "Independent seeded oracle and structure", "behavior": "routing.normalization", "visibility": "unshown", "failure_message": "Routes must match per-token largest softmax probabilities and selected weights must be renormalized.", "code": r"""
import torch
for seed, shape, k in [(17,(5,7),3),(23,(2,4),1),(31,(6,5),5)]:
    generator=torch.Generator().manual_seed(seed)
    logits=torch.randn(*shape, generator=generator, dtype=torch.float64)
    indices, weights={fn}(logits, k)
    probabilities=torch.exp(logits-torch.logsumexp(logits,dim=-1,keepdim=True))
    expected_weights, expected_indices=torch.topk(probabilities,k,dim=-1)
    expected_weights=expected_weights/expected_weights.sum(-1,keepdim=True)
    assert indices.shape==weights.shape==(shape[0],k)
    assert indices.dtype==torch.long and weights.dtype==logits.dtype
    assert indices.device==weights.device==logits.device
    assert torch.equal(indices,expected_indices)
    assert torch.allclose(weights,expected_weights,atol=1e-10,rtol=1e-9)
    assert torch.allclose(weights.sum(-1),torch.ones(shape[0],dtype=weights.dtype),atol=1e-12)
"""},
        {"name": "Each token has its own route and ties follow torch topk", "behavior": "routing.selection", "visibility": "unshown", "failure_message": "Routing must be independent per token and use deterministic torch.topk tie semantics.", "code": r"""
import torch
logits=torch.tensor([[9.,1.,0.],[0.,1.,9.],[2.,2.,0.]])
indices,weights={fn}(logits,2)
expected=torch.topk(torch.softmax(logits,-1),2,-1).indices
assert torch.equal(indices,expected)
assert not torch.equal(indices[0],indices[1])
"""},
        {"name": "Selected weights retain router gradients", "behavior": "training.router_gradient", "visibility": "unshown", "failure_message": "Selected routing weights must remain differentiable with respect to router logits.", "code": r"""
import torch
logits=torch.tensor([[2.,0.,-1.],[0.,3.,1.]],requires_grad=True)
_,weights={fn}(logits,2)
(weights*torch.tensor([[1.,3.],[2.,5.]])).sum().backward()
assert logits.grad is not None and torch.isfinite(logits.grad).all() and torch.count_nonzero(logits.grad)
"""},
        {"name": "Expert permutation preserves selected meanings", "behavior": "state.invariant", "visibility": "unshown", "failure_message": "Permuting expert columns must permute route ids while preserving corresponding gate weights.", "code": r"""
import torch
logits=torch.tensor([[4.,1.,-2.,2.],[0.,3.,1.,-1.]],dtype=torch.float64)
order=torch.tensor([2,0,3,1])
base_i,base_w={fn}(logits,2)
perm_i,perm_w={fn}(logits[:,order],2)
inverse=torch.empty_like(order); inverse[order]=torch.arange(4)
assert torch.equal(perm_i,inverse[base_i])
assert torch.allclose(perm_w,base_w,atol=1e-12)
"""},
    ],
    "solution": r'''def moe_topk_router(router_logits, k):
    if router_logits.ndim != 2:
        raise ValueError("router_logits must have shape (tokens, experts)")
    num_experts = router_logits.shape[-1]
    if not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= num_experts:
        raise ValueError("k must select between one and all experts")
    probabilities = torch.softmax(router_logits, dim=-1)
    weights, indices = torch.topk(probabilities, k, dim=-1)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    return indices, weights''',
    "demo": r"""router_logits=torch.tensor([[3.,1.,0.],[0.,2.,4.]])
expert_indices,expert_weights=moe_topk_router(router_logits,2)
print(expert_indices)
print(expert_weights, expert_weights.sum(-1))""",
}
