"""Shared always-on experts alongside routed sparse experts."""

TASK = {
    "title": "Shared and Routed Experts",
    "difficulty": "Medium",
    "version": 1,
    "function_name": "shared_routed_moe",
    "description_en": r"""Implement a Mixture-of-Experts forward that combines always-on shared experts with sparsely routed experts, the design DeepSeek and Qwen use.

Signature: 'shared_routed_moe(x, expert_indices, expert_weights, routed_experts, shared_experts) -> output'

'x' has shape '(tokens, d_model)'. 'expert_indices' and 'expert_weights' both have shape '(tokens, k)' and route only into 'routed_experts'. 'shared_experts' is a separate list of always-on modules. Each module maps '(n, d_model)' to '(n, d_model)'.

For every token the output is the sum of two branches:

- Shared branch: every shared expert runs on the full token batch and its outputs are added directly (no gate weighting).
- Routed branch: for each token, its k routed experts run sparsely and their outputs are weighted by the gate and accumulated back to the token.

'output = sum_over_shared( shared(x) ) + sum_over_routed_slots( weight * routed_expert(x_token) )'

Return one output row per token. Shared experts always execute on all tokens and must never appear in routed selection or in routed accounting; routed experts execute only on the tokens that selected them, and unused routed experts must never run.

Isolating a shared expert lets it capture common knowledge every token needs, so the routed experts specialize instead of each re-learning shared features. The shared branch is dense and cheap relative to the routed pool. Validate the routed route tensors, reject out-of-range routed ids, and require at least one routed expert; the shared list may be empty.""",
    "advisory_prerequisites": ["moe", "moe_topk_router", "moe_capacity_dispatch"],
    "hints": [
        {"level": 1, "kind": "questions", "content": "Which experts run for every token, and which run for only a few? Are shared experts gate-weighted? Should a shared expert ever appear in the routed indices or counts? How do the two branches combine into one output row?"},
        {"level": 2, "kind": "analysis", "content": "Accumulate the routed branch exactly like sparse dispatch: group accepted (token, slot) pairs per routed expert, call each used expert once, weight by the gate, and index_add to a routed output. Separately run every shared expert on the whole batch and sum. Return routed_output + shared_output. Keep the two accountings disjoint."},
    ],
    "model_connections": [
        "DeepSeekMoE isolates one or more shared experts that process every token, letting routed experts specialize on residual knowledge.",
        "DeepSeek-V3 MoE.forward adds an always-on shared-expert branch to the routed sum, exactly the two-branch structure here.",
        "Qwen and GLM MoE blocks similarly compose a shared/dense path with sparse routed experts and a weighted gather.",
    ],
    "pro_con_analysis": {
        "pros": [
            "A shared expert captures common features so routed experts specialize rather than duplicate them.",
            "The shared branch is dense and predictable, improving stability without large sparse cost.",
            "Keeping shared experts out of routed accounting yields clean load and utilization statistics.",
        ],
        "cons": [
            "The always-on branch adds fixed compute for every token.",
            "Too much shared capacity can crowd out expert specialization.",
            "Two branches complicate load-balancing and capacity bookkeeping.",
        ],
    },
    "sources": [
        {"kind": "code", "url": "https://github.com/deepseek-ai/DeepSeek-V3", "commit": "9b4e9788e4a3a731f7567338ed15d3ec549ce03b", "path": "inference/model.py", "symbol": "MoE.forward", "license": "MIT", "adapted": "Always-on shared-expert branch added to the routed sparse sum.", "simplifications": "Single process, explicit shared list, no expert parallelism, and no capacity limit."},
        {"kind": "code", "url": "https://github.com/huggingface/transformers", "commit": "7ea2320c76117e6742364808a666ef6f2fb40a67", "path": "src/transformers/models/glm4_moe/modeling_glm4_moe.py", "symbol": "Glm4MoeMoE.forward", "license": "Apache-2.0", "adapted": "Composition of a shared/dense branch with routed sparse experts and weighted gather.", "simplifications": "No routing groups, capacity, or model wrapper."},
        {"kind": "paper", "url": "https://arxiv.org/abs/2401.06066", "section": "3.2 Shared Expert Isolation"},
    ],
    "tests": [
        {"name": "Hand-calculated shared plus routed sum", "behavior": "dispatch.token_conservation", "code": r"""
import torch
import torch.nn as nn
def expert(scale):
    layer=nn.Linear(2,2,bias=False)
    with torch.no_grad(): layer.weight.copy_(torch.eye(2)*scale)
    return layer
x=torch.tensor([[1.,2.],[3.,4.]])
indices=torch.tensor([[0],[1]])
weights=torch.tensor([[0.5],[0.5]])
routed=[expert(2.),expert(4.)]
shared=[expert(1.)]
out={fn}(x,indices,weights,routed,shared)
# token0: shared 1*[1,2] + 0.5*2*[1,2] = [2,4]; token1: 1*[3,4] + 0.5*4*[3,4] = [9,12]
assert torch.allclose(out,torch.tensor([[2.,4.],[9.,12.]]))
"""},
        {"name": "Rejects malformed routed routes", "behavior": "edge.empty_or_boundary", "code": r"""
import torch
import torch.nn as nn
x=torch.randn(3,4); routed=[nn.Identity()]
for indices in (torch.tensor([[0],[0]]), torch.tensor([[1],[0],[0]])):
    try: {fn}(x, indices, torch.ones(indices.shape), routed, [])
    except (ValueError, RuntimeError): pass
    else: raise AssertionError('malformed routed routes must raise')
try: {fn}(x, torch.zeros(3,1,dtype=torch.long), torch.ones(3,1), [], [nn.Identity()])
except ValueError: pass
else: raise AssertionError('empty routed_experts must raise')
"""},
        {"name": "Independent oracle: dense shared plus sparse routed", "behavior": "dispatch.token_conservation", "visibility": "unshown", "failure_message": "Output must equal every shared expert on all tokens plus the gate-weighted sparse routed gather.", "code": r"""
import torch
import torch.nn as nn
for seed in (13,29):
    generator=torch.Generator().manual_seed(seed)
    tokens,d,ne,k=6,3,4,2
    x=torch.randn(tokens,d,generator=generator,dtype=torch.float64)
    indices=torch.randint(0,ne,(tokens,k),generator=generator)
    weights=torch.rand(tokens,k,generator=generator,dtype=torch.float64)
    def mk():
        layer=nn.Linear(d,d,bias=False,dtype=torch.float64)
        with torch.no_grad(): layer.weight.copy_(torch.randn(d,d,generator=generator,dtype=torch.float64))
        return layer
    routed=[mk() for _ in range(ne)]; shared=[mk() for _ in range(2)]
    actual={fn}(x.clone(),indices.clone(),weights.clone(),routed,shared)
    dense=torch.stack([e(x) for e in routed],dim=0)
    tok=torch.arange(tokens).unsqueeze(1).expand_as(indices)
    routed_expected=(dense[indices,tok]*weights.unsqueeze(-1)).sum(dim=1)
    shared_expected=sum(s(x) for s in shared)
    expected=routed_expected+shared_expected
    assert actual.shape==x.shape and actual.dtype==x.dtype
    assert torch.allclose(actual,expected,atol=1e-10,rtol=1e-9)
"""},
        {"name": "Shared run on all tokens, routed run sparsely", "behavior": "experts.sparsity", "visibility": "unshown", "failure_message": "Shared experts must process every token; routed experts run only on selected tokens and unused routed experts never run.", "code": r"""
import torch
import torch.nn as nn
from torch_judge.harness.tensor import CountingExpert
routed=[CountingExpert(nn.Identity()) for _ in range(4)]
shared=[CountingExpert(nn.Identity()) for _ in range(2)]
x=torch.randn(5,3)
indices=torch.tensor([[0],[0],[2],[0],[2]])
weights=torch.ones(5,1)
{fn}(x,indices,weights,routed,shared)
assert [(e.calls,e.tokens) for e in routed]==[(1,3),(0,0),(1,2),(0,0)]
assert all((e.calls,e.tokens)==(1,5) for e in shared)
"""},
        {"name": "Gradients reach selected routed, all shared, and gates", "behavior": "gradient.flow", "visibility": "unshown", "failure_message": "Selected routed experts, every shared expert, and selected gate weights must receive gradients.", "code": r"""
import torch
import torch.nn as nn
from torch_judge.harness.tensor import assert_expert_gradient_partition
routed=[nn.Linear(2,2,bias=False) for _ in range(3)]
shared=[nn.Linear(2,2,bias=False) for _ in range(2)]
x=torch.randn(3,2)
weights=torch.tensor([[.7,.3],[.6,.4],[.8,.2]],requires_grad=True)
indices=torch.tensor([[0,1],[0,1],[0,1]])
out={fn}(x,indices,weights,routed,shared)
(out*torch.tensor([[1.,2.],[3.,4.],[5.,6.]])).sum().backward()
assert_expert_gradient_partition(routed,{0,1})
assert all(all(p.grad is not None and bool(torch.count_nonzero(p.grad)) for p in s.parameters()) for s in shared)
assert torch.count_nonzero(weights.grad)==weights.numel()
"""},
        {"name": "Routed permutation preserves output", "behavior": "state.invariant", "visibility": "unshown", "failure_message": "Reordering routed experts with consistently remapped ids must preserve the output; the shared branch is unaffected.", "code": r"""
import torch
import torch.nn as nn
from torch_judge.harness.tensor import permute_expert_routes
routed=[]
for scale in (1.,2.,4.):
    layer=nn.Linear(2,2,bias=False)
    with torch.no_grad(): layer.weight.copy_(torch.eye(2)*scale)
    routed.append(layer)
shared=[nn.Linear(2,2,bias=False)]
x=torch.tensor([[1.,2.],[3.,1.],[2.,5.]])
indices=torch.tensor([[0,2],[1,0],[2,1]])
weights=torch.tensor([[.8,.2],[.3,.7],[.4,.6]])
base={fn}(x,indices,weights,routed,shared)
remapped,reordered=permute_expert_routes(indices,routed,[2,0,1])
other={fn}(x,remapped,weights,reordered,shared)
assert torch.allclose(base,other)
"""},
    ],
    "solution": r'''def shared_routed_moe(x, expert_indices, expert_weights, routed_experts, shared_experts):
    if x.ndim != 2 or expert_indices.ndim != 2 or expert_weights.shape != expert_indices.shape:
        raise ValueError("x and routed route tensors have invalid shapes")
    if expert_indices.shape[0] != x.shape[0] or expert_indices.dtype != torch.long:
        raise ValueError("routed route tensors must align with tokens and indices must be long")
    if not routed_experts:
        raise ValueError("at least one routed expert is required")
    if expert_indices.numel() and (int(expert_indices.min()) < 0 or int(expert_indices.max()) >= len(routed_experts)):
        raise ValueError("routed expert index out of range")
    routed_out = torch.zeros_like(x)
    routes = [[] for _ in routed_experts]
    for token in range(expert_indices.shape[0]):
        for slot in range(expert_indices.shape[1]):
            routes[int(expert_indices[token, slot])].append((token, slot))
    for expert_id, accepted in enumerate(routes):
        if not accepted:
            continue
        token_ids = torch.tensor([token for token, _ in accepted], device=x.device)
        slot_ids = torch.tensor([slot for _, slot in accepted], device=x.device)
        expert_output = routed_experts[expert_id](x.index_select(0, token_ids))
        route_weights = expert_weights[token_ids, slot_ids].to(expert_output.dtype).unsqueeze(-1)
        routed_out.index_add_(0, token_ids, expert_output * route_weights)
    shared_out = torch.zeros_like(x)
    for shared in shared_experts:
        shared_out = shared_out + shared(x)
    return routed_out + shared_out''',
    "demo": r"""routed=[nn.Linear(4,4,bias=False) for _ in range(3)]
shared=[nn.Linear(4,4,bias=False)]
x=torch.randn(5,4)
indices=torch.tensor([[0,1],[0,2],[1,2],[0,1],[2,0]])
weights=torch.full((5,2),.5)
print(shared_routed_moe(x,indices,weights,routed,shared).shape)""",
}
