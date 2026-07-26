"""Noisy top-k routing for exploration during MoE training."""

TASK = {
    "title": "Noisy Top-k Routing",
    "difficulty": "Medium",
    "version": 1,
    "function_name": "noisy_topk_route",
    "description_en": r"""Implement noisy top-k gating, the exploration mechanism from the original sparsely-gated Mixture-of-Experts layer.

Signature: 'noisy_topk_route(clean_logits, noise, noise_std, k) -> (indices, weights)'

'clean_logits' and 'noise' both have shape '(tokens, num_experts)'. 'noise' holds standard-normal samples supplied by the caller so the exercise stays deterministic; in production the layer samples this noise itself and scales it by a learned per-expert amount. Add the scaled noise to the logits, softmax over experts, take the top-k experts per token, and renormalize the selected probabilities so each token's gate weights sum to one:

'noisy = clean_logits + noise_std * noise'
'probs = softmax(noisy, dim=-1)'
'weights, indices = topk(probs, k); weights = weights / weights.sum(-1, keepdim=True)'

Return long 'indices' of shape '(tokens, k)' and floating 'weights' of the same shape. When 'noise_std' is zero the result must exactly match noiseless top-k routing. The gate weights must stay differentiable with respect to 'clean_logits'; the injected noise is treated as a constant.

Noise perturbs which experts win the top-k, so tokens occasionally explore experts they would not otherwise pick. Early in training this spreads load and prevents a few experts from monopolizing routing before the load-balancing loss takes effect; annealing 'noise_std' toward zero recovers deterministic routing for inference. Validate the shapes, that 'k' is within '[1, num_experts]', and that 'noise_std' is non-negative.""",
    "advisory_prerequisites": ["softmax", "moe_topk_router"],
    "hints": [
        {"level": 1, "kind": "questions", "content": "At what point does the noise enter — before or after selecting experts? What must the result equal when noise_std is zero? Which tensor should gradients reach, and which is a constant? Do the selected weights still need to sum to one?"},
        {"level": 2, "kind": "analysis", "content": "Form noisy logits as clean_logits + noise_std * noise, softmax over the expert dimension, take torch.topk, and divide the selected probabilities by their per-token sum. Selection happens on the noisy probabilities but the whole path stays differentiable in clean_logits because noise and noise_std are constants. noise_std == 0 collapses to plain top-k routing."},
    ],
    "model_connections": [
        "Shazeer et al.'s sparsely-gated MoE adds tunable Gaussian noise before top-k gating to encourage exploration and balance expert load.",
        "GShard and Switch Transformer variants use jitter/noise on routing during training and disable it at inference, mirroring the noise_std anneal here.",
        "Modern DeepSeek/Qwen routers rely more on auxiliary balance losses than injected noise, but the exploration-versus-determinism tradeoff is the same design axis.",
    ],
    "pro_con_analysis": {
        "pros": [
            "Injecting noise before selection lets tokens explore non-argmax experts and spreads early load.",
            "Deterministic when noise_std is zero, so the same code serves training and inference.",
            "Keeps gate weights differentiable in the clean logits while treating noise as a constant.",
        ],
        "cons": [
            "Too much noise destabilizes routing and slows convergence.",
            "Exploration overlaps with the load-balancing loss, adding another schedule to tune.",
            "Sampling and annealing noise adds training-time bookkeeping absent at inference.",
        ],
    },
    "sources": [
        {"kind": "paper", "url": "https://arxiv.org/abs/1701.06538", "section": "2.1 Noisy Top-K Gating"},
    ],
    "tests": [
        {"name": "Zero noise matches plain top-k routing", "behavior": "routing.selection", "code": r"""
import torch
logits=torch.tensor([[0.,1.,2.],[3.,1.,2.]])
noise=torch.randn(2,3)
indices,weights={fn}(logits,noise,0.0,2)
probs=torch.softmax(logits,dim=-1)
ev,ei=torch.topk(probs,2,dim=-1); ev=ev/ev.sum(-1,keepdim=True)
assert torch.equal(indices,ei) and torch.allclose(weights,ev)
"""},
        {"name": "Rejects invalid k and mismatched noise", "behavior": "edge.empty_or_boundary", "code": r"""
import torch
logits=torch.randn(2,3); noise=torch.randn(2,3)
for bad_k in (0,4,-1):
    try: {fn}(logits,noise,0.1,bad_k)
    except ValueError: pass
    else: raise AssertionError('invalid k must raise')
try: {fn}(logits,torch.randn(2,4),0.1,2)
except (ValueError,RuntimeError): pass
else: raise AssertionError('mismatched noise shape must raise')
try: {fn}(logits,noise,-1.0,2)
except ValueError: pass
else: raise AssertionError('negative noise_std must raise')
"""},
        {"name": "Independent seeded oracle with active noise", "behavior": "routing.selection", "visibility": "unshown", "failure_message": "Noise must be scaled by noise_std and added before softmax and top-k, then selected weights renormalized.", "code": r"""
import torch
for seed,std,k in [(3,0.5,2),(8,1.5,1),(14,0.25,3)]:
    generator=torch.Generator().manual_seed(seed)
    logits=torch.randn(5,4,generator=generator,dtype=torch.float64)
    noise=torch.randn(5,4,generator=generator,dtype=torch.float64)
    indices,weights={fn}(logits,noise,std,k)
    probs=torch.softmax(logits+std*noise,dim=-1)
    ev,ei=torch.topk(probs,k,dim=-1); ev=ev/ev.sum(-1,keepdim=True)
    assert indices.shape==weights.shape==(5,k)
    assert indices.dtype==torch.long and weights.dtype==logits.dtype
    assert torch.equal(indices,ei) and torch.allclose(weights,ev,atol=1e-12)
    assert torch.allclose(weights.sum(-1),torch.ones(5,dtype=weights.dtype),atol=1e-12)
"""},
        {"name": "Gradient reaches clean logits, noise is constant", "behavior": "gradient.flow", "visibility": "unshown", "failure_message": "Gate weights must stay differentiable in clean_logits while noise is treated as a constant.", "code": r"""
import torch
generator=torch.Generator().manual_seed(31)
logits=torch.randn(4,6,generator=generator,dtype=torch.float64).requires_grad_(True)
noise=torch.randn(4,6,generator=generator,dtype=torch.float64)
indices,weights={fn}(logits,noise,0.7,3)
# weights renormalize to sum to one per token, so use a distribution-sensitive loss.
(weights.square().sum()).backward()
assert logits.grad is not None and bool(torch.count_nonzero(logits.grad))
assert bool(torch.isfinite(logits.grad).all())
"""},
        {"name": "Deterministic and expert-permutation consistent", "behavior": "state.invariant", "visibility": "unshown", "failure_message": "Given fixed noise the routing is deterministic and permuting experts selects the same experts with the same weights.", "code": r"""
import torch
generator=torch.Generator().manual_seed(19)
logits=torch.randn(5,6,generator=generator,dtype=torch.float64)
noise=torch.randn(5,6,generator=generator,dtype=torch.float64)
first={fn}(logits,noise,0.4,2); second={fn}(logits,noise,0.4,2)
assert torch.equal(first[0],second[0]) and torch.equal(first[1],second[1])
perm=torch.randperm(6,generator=generator)
base_idx,base_w={fn}(logits,noise,0.4,2)
perm_idx,perm_w={fn}(logits[:,perm],noise[:,perm],0.4,2)
back=perm[perm_idx]  # map permuted positions to original expert ids
for t in range(logits.shape[0]):
    base_map=dict(zip(base_idx[t].tolist(),base_w[t].tolist()))
    perm_map=dict(zip(back[t].tolist(),perm_w[t].tolist()))
    assert set(base_map)==set(perm_map)
    for e in base_map:
        assert abs(base_map[e]-perm_map[e])<1e-9
"""},
    ],
    "solution": r'''def noisy_topk_route(clean_logits, noise, noise_std, k):
    if clean_logits.ndim != 2 or not isinstance(noise, torch.Tensor) or noise.shape != clean_logits.shape:
        raise ValueError("clean_logits and noise must be matching (tokens, num_experts) tensors")
    if not clean_logits.is_floating_point():
        raise ValueError("clean_logits must be floating point")
    num_experts = clean_logits.shape[-1]
    if not isinstance(k, int) or isinstance(k, bool) or k < 1 or k > num_experts:
        raise ValueError("k must be an integer in [1, num_experts]")
    if noise_std < 0:
        raise ValueError("noise_std must be non-negative")
    noisy = clean_logits + noise_std * noise
    probs = torch.softmax(noisy, dim=-1)
    weights, indices = torch.topk(probs, k, dim=-1)
    weights = weights / weights.sum(-1, keepdim=True)
    return indices, weights''',
    "demo": r"""logits=torch.randn(4,8); noise=torch.randn(4,8)
idx,w=noisy_topk_route(logits,noise,0.5,2)
print(idx.shape, w.sum(-1))""",
}
