"""Dense FFN versus sparse expert execution — the sparsity primitive."""

TASK = {
    "title": "Dense FFN versus Sparse Experts",
    "difficulty": "Medium",
    "version": 1,
    "function_name": "sparse_ffn_forward",
    "description_en": r"""Implement the sparse expert forward that a Mixture-of-Experts layer runs in place of one dense feed-forward network.

Signature: 'sparse_ffn_forward(x, expert_indices, expert_weights, experts) -> output'

'x' has shape '(tokens, d_model)'. 'expert_indices' and 'expert_weights' both have shape '(tokens, k)': for each token they name its k routed experts and the gate weight for each route. 'experts' is an ordered sequence of modules mapping '(n, d_model)' to '(n, d_model)'. Every routed assignment is kept — there is no capacity limit in this exercise.

For each token, the output is the weighted sum over its k routes of 'expert_weight * expert(x_token)'. Return one output row per input token with the same shape, dtype, and device as 'x'.

The point of the exercise is that this must be **sparse**: batch the accepted rows for one expert into a single call, execute only the experts that at least one token selected, and never run an unused expert. A dense oracle that evaluates every expert on every token and then gathers the selected results produces the identical answer, but does 'num_experts / k' times the work. Sparse execution is what makes a large expert pool affordable; the exercise measures it with expert invocation counts rather than wall-clock timing.

Validate shapes and route dtype up front, reject out-of-range expert ids, and require at least one expert. Real MoE blocks add capacity limits, shared always-on experts, and expert-parallel communication; those are separate exercises.""",
    "advisory_prerequisites": ["mlp", "moe"],
    "hints": [
        {"level": 1, "kind": "questions", "content": "How many experts should actually run for a batch that routes to only a few of them? Can a token receive output from more than one route? What does a dense reference compute that the sparse version must match exactly? Where do the gate weights enter?"},
        {"level": 2, "kind": "analysis", "content": "Group accepted (token, slot) pairs into one list per expert. For each non-empty list, index_select that expert's token rows, call the expert once, multiply by the matching gate weights, and index_add the weighted rows back to their tokens. Experts with no routed tokens are never called, which is what separates sparse execution from a dense gather."},
    ],
    "model_connections": [
        "Switch Transformer replaces the dense FFN sublayer with a single routed expert per token, trading constant FLOPs per token for a much larger parameter pool.",
        "Qwen3-MoE builds per-expert token masks and executes each expert only on its selected rows, then index-adds the weighted outputs — the sparse pattern this exercise isolates without capacity or shared experts.",
        "DeepSeek-V3 MoE.forward runs only the routed (and locally hosted) experts and sums their weighted outputs, demonstrating that the dense FFN is never materialized for every expert.",
        "GLM-4.5's Glm4MoeExperts executes selected-token expert calls and a weighted index-add gather, the same sparse-execution contract measured here by invocation counts.",
    ],
    "pro_con_analysis": {
        "pros": [
            "Executing only selected experts keeps per-token compute constant while the parameter count scales with the number of experts.",
            "One batched call per used expert avoids per-token kernel launches and matches production sparse blocks.",
            "Invocation-count checks make the sparse/dense distinction observable and testable without timing.",
        ],
        "cons": [
            "Python route lists and index_add do not represent fused distributed expert kernels.",
            "Without capacity limits, real load imbalance and communication cost are hidden.",
            "Materializing per-expert token batches costs gather/scatter memory traffic that a dense matmul avoids at small scale.",
        ],
    },
    "sources": [
        {"kind": "code", "url": "https://github.com/huggingface/transformers", "commit": "7ea2320c76117e6742364808a666ef6f2fb40a67", "path": "src/transformers/models/qwen3_moe/modeling_qwen3_moe.py", "symbol": "Qwen3MoeSparseMoeBlock.forward", "license": "Apache-2.0", "adapted": "Per-expert token selection and index_add gather of weighted expert outputs.", "simplifications": "No capacity, shared expert, gate module, sequence flattening, or model wrapper."},
        {"kind": "code", "url": "https://github.com/deepseek-ai/DeepSeek-V3", "commit": "9b4e9788e4a3a731f7567338ed15d3ec549ce03b", "path": "inference/model.py", "symbol": "MoE.forward", "license": "MIT", "adapted": "Executing only routed experts and summing weighted outputs.", "simplifications": "Single process, no shared expert, no capacity, and no expert-parallel communication."},
        {"kind": "code", "url": "https://github.com/huggingface/transformers", "commit": "7ea2320c76117e6742364808a666ef6f2fb40a67", "path": "src/transformers/models/glm4_moe/modeling_glm4_moe.py", "symbol": "Glm4MoeExperts.forward", "license": "Apache-2.0", "adapted": "Selected-token expert execution and weighted index-add gather.", "simplifications": "Omits capacity, shared experts, routing groups, and the transformer block."},
        {"kind": "paper", "url": "https://arxiv.org/abs/2101.03961", "section": "2.1 Simplifying Sparse Routing"},
    ],
    "tests": [
        {"name": "Hand-calculated weighted sparse sum", "behavior": "dispatch.token_conservation", "code": r"""
import torch
import torch.nn as nn
def expert(scale):
    layer=nn.Linear(2,2,bias=False)
    with torch.no_grad(): layer.weight.copy_(torch.eye(2)*scale)
    return layer
x=torch.tensor([[1.,2.],[3.,4.]])
indices=torch.tensor([[0,1],[1,0]])
weights=torch.tensor([[.75,.25],[.4,.6]])
out={fn}(x,indices,weights,[expert(1.),expert(10.)])
# token0: .75*[1,2] + .25*10*[1,2] = [3.25,6.5]; token1: .4*10*[3,4] + .6*[3,4] = [13.8,18.4]
assert torch.allclose(out,torch.tensor([[3.25,6.5],[13.8,18.4]]))
"""},
        {"name": "Rejects malformed routes", "behavior": "edge.empty_or_boundary", "code": r"""
import torch
import torch.nn as nn
experts=[nn.Identity()]
x=torch.randn(3,4)
for indices in (torch.tensor([[0],[0]]), torch.zeros(3,1,dtype=torch.float32), torch.tensor([[1],[0],[0]])):
    try:
        {fn}(x, indices, torch.ones(indices.shape), experts)
    except (ValueError, RuntimeError):
        pass
    else:
        raise AssertionError('malformed route tensors must raise')
"""},
        {"name": "Independent dense oracle and token conservation", "behavior": "dispatch.token_conservation", "visibility": "unshown", "failure_message": "Sparse output must equal a dense gather: weight each route once and accumulate all routes back to their token.", "code": r"""
import torch
import torch.nn as nn
for seed in (11,17,23):
    generator=torch.Generator().manual_seed(seed)
    tokens,d,num_experts,k=7,4,5,3
    x=torch.randn(tokens,d,generator=generator,dtype=torch.float64)
    indices=torch.randint(0,num_experts,(tokens,k),generator=generator)
    weights=torch.rand(tokens,k,generator=generator,dtype=torch.float64)
    experts=[]
    for _ in range(num_experts):
        layer=nn.Linear(d,d,bias=False,dtype=torch.float64)
        with torch.no_grad(): layer.weight.copy_(torch.randn(d,d,generator=generator,dtype=torch.float64))
        experts.append(layer)
    actual={fn}(x.clone(),indices.clone(),weights.clone(),experts)
    dense=torch.stack([e(x) for e in experts],dim=0)  # (num_experts, tokens, d)
    token_ids=torch.arange(tokens).unsqueeze(1).expand_as(indices)
    routed=dense[indices,token_ids]  # (tokens, k, d)
    expected=(routed*weights.unsqueeze(-1)).sum(dim=1)
    assert actual.shape==x.shape and actual.dtype==x.dtype and actual.device==x.device
    assert torch.allclose(actual,expected,atol=1e-10,rtol=1e-9)
"""},
        {"name": "Only selected experts run, once each", "behavior": "experts.sparsity", "visibility": "unshown", "failure_message": "Only experts that a token selected may run, each in a single batched call; unused experts must never execute.", "code": r"""
import torch
import torch.nn as nn
from torch_judge.harness.tensor import CountingExpert
experts=[CountingExpert(nn.Identity()) for _ in range(4)]
x=torch.randn(5,3)
indices=torch.tensor([[0],[0],[2],[0],[2]])
weights=torch.ones(5,1)
out={fn}(x,indices,weights,experts)
assert [(e.calls,e.tokens) for e in experts]==[(1,3),(0,0),(1,2),(0,0)]
"""},
        {"name": "Gradients reach selected experts and gates only", "behavior": "gradient.flow", "visibility": "unshown", "failure_message": "Only selected expert paths and their gate weights should receive gradients.", "code": r"""
import torch
import torch.nn as nn
from torch_judge.harness.tensor import assert_expert_gradient_partition
experts=[nn.Linear(2,2,bias=False) for _ in range(3)]
x=torch.randn(3,2)
weights=torch.tensor([[.7,.3],[.6,.4],[.8,.2]],requires_grad=True)
indices=torch.tensor([[0,1],[0,1],[0,1]])
out={fn}(x,indices,weights,experts)
(out*torch.tensor([[1.,2.],[3.,4.],[5.,6.]])).sum().backward()
assert_expert_gradient_partition(experts,{0,1})
assert torch.count_nonzero(weights.grad)==weights.numel()
"""},
        {"name": "Expert permutation preserves output", "behavior": "state.invariant", "visibility": "unshown", "failure_message": "Reordering experts with consistently remapped route ids must preserve the output.", "code": r"""
import torch
import torch.nn as nn
from torch_judge.harness.tensor import permute_expert_routes
experts=[]
for scale in (1.,2.,4.):
    layer=nn.Linear(2,2,bias=False)
    with torch.no_grad(): layer.weight.copy_(torch.eye(2)*scale)
    experts.append(layer)
x=torch.tensor([[1.,2.],[3.,1.],[2.,5.]])
indices=torch.tensor([[0,2],[1,0],[2,1]])
weights=torch.tensor([[.8,.2],[.3,.7],[.4,.6]])
base={fn}(x,indices,weights,experts)
remapped,reordered=permute_expert_routes(indices,experts,[2,0,1])
other={fn}(x,remapped,weights,reordered)
assert torch.allclose(base,other)
"""},
    ],
    "solution": r'''def sparse_ffn_forward(x, expert_indices, expert_weights, experts):
    if x.ndim != 2 or expert_indices.ndim != 2 or expert_weights.shape != expert_indices.shape:
        raise ValueError("x and route tensors have invalid shapes")
    if expert_indices.shape[0] != x.shape[0] or expert_indices.dtype != torch.long:
        raise ValueError("route tensors must align with tokens and indices must be long")
    if not experts:
        raise ValueError("at least one expert is required")
    if expert_indices.numel() and (int(expert_indices.min()) < 0 or int(expert_indices.max()) >= len(experts)):
        raise ValueError("expert index out of range")
    routes = [[] for _ in experts]
    for token in range(expert_indices.shape[0]):
        for slot in range(expert_indices.shape[1]):
            routes[int(expert_indices[token, slot])].append((token, slot))
    output = torch.zeros_like(x)
    for expert_id, accepted in enumerate(routes):
        if not accepted:
            continue
        token_ids = torch.tensor([token for token, _ in accepted], device=x.device)
        slot_ids = torch.tensor([slot for _, slot in accepted], device=x.device)
        expert_output = experts[expert_id](x.index_select(0, token_ids))
        route_weights = expert_weights[token_ids, slot_ids].to(expert_output.dtype).unsqueeze(-1)
        output.index_add_(0, token_ids, expert_output * route_weights)
    return output''',
    "demo": r"""experts=[nn.Linear(4,4,bias=False) for _ in range(3)]
x=torch.randn(5,4)
indices=torch.tensor([[0,1],[0,2],[1,2],[0,1],[2,0]])
weights=torch.full((5,2),.5)
print(sparse_ffn_forward(x,indices,weights,experts).shape)""",
}
