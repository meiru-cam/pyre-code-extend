"""Capacity-aware sparse expert dispatch and gather."""

TASK = {
    "title": "Capacity-aware MoE Dispatch",
    "difficulty": "Hard",
    "version": 1,
    "function_name": "moe_capacity_dispatch",
    "description_en": r"""Implement sparse dispatch, expert execution, and weighted gather.

Signature: 'moe_capacity_dispatch(x, expert_indices, expert_weights, experts, capacity) -> (output, stats)'

'x' has shape '(tokens, d_model)'; both route tensors have shape '(tokens, k)'. 'experts' is an ordered sequence of modules mapping '(n, d_model)' to the same shape. Consider assignments in token-major, then route-slot order. Each expert accepts at most its own non-negative integer 'capacity'; later overflow assignments are dropped. Batch all accepted rows for one expert into one call, multiply its results by route weights, and add every accepted route back to its original token. Return one output row per input token plus:

'{"accepted": int, "dropped": int, "loads": list[int]}'

This is a dropping capacity policy. A common capacity-factor conversion is 'capacity = ceil(capacity_factor * tokens * k / num_experts)': factor 1.0 allocates the average balanced load, while larger values trade padding/memory for fewer drops. Dropless systems keep all assignments but face irregular load and communication; fixed-capacity padding makes collective shapes regular but wastes compute; dropping bounds work but can lose token signal. Utilization should be read as each load divided by capacity alongside the drop rate. Real expert-parallel runtimes also exchange tokens across devices and often fuse sorting, all-to-all, expert kernels, and gather.

Shared experts are a separate always-on branch in models such as DeepSeek and Kimi. They are intentionally absent from this function's routed 'experts' list and from its accepted/dropped/load statistics.

Optional code reading: DeepSeek-V3 'MoE.forward' shows routed and shared experts; Qwen3-MoE's sparse block shows expert masking and indexed accumulation. The exercise makes capacity and sparse-call behavior more explicit than either inference loop.""",
    "advisory_prerequisites": ["moe", "moe_topk_router"],
    "hints": [
        {"level": 1, "kind": "questions", "content": "Is capacity global or per expert? In what order do assignments consume it? Can one token receive output from multiple accepted routes? Which experts should never be called?"},
        {"level": 2, "kind": "analysis", "content": "First scan token then slot, recording accepted (token, slot) pairs in one list per expert and updating per-expert loads. For each non-empty list, index-select its token batch, call that expert once, multiply by matching gate weights, and index_add results to output."},
    ],
    "model_connections": [
        "DeepSeek-V3 MoE.forward selects routed experts, executes only locally hosted experts, sums routed results, and adds always-on shared-expert output.",
        "Qwen3-MoE builds per-expert token masks, executes experts on selected token rows, weights them, and index-adds them into final hidden states.",
        "Kimi K2 deployment exposes DeepEP and expert-parallel controls; SGLang's Kimi-compatible DeepseekV2MoE connects TopK output to fused/shared expert execution.",
        "GLM-4.5's Glm4MoeExperts and Glm4MoeMoE implement selected-token expert execution, weighted gather, and a shared-expert branch.",
    ],
    "pro_con_analysis": {
        "pros": [
            "Per-expert capacity bounds worst-case expert compute and communication.",
            "One batched call per used expert preserves sparse execution while avoiding per-token kernel launches.",
            "Explicit statistics make overflow and utilization observable during tests and training.",
        ],
        "cons": [
            "Dropping overflow routes loses model signal and can bias training toward early assignments.",
            "Token-major acceptance is deterministic but may be less fair than prioritized or randomized policies.",
            "Python indexing and materialized route lists do not represent fused distributed production performance.",
        ],
    },
    "sources": [
        {"kind": "code", "url": "https://github.com/deepseek-ai/DeepSeek-V3", "commit": "9b4e9788e4a3a731f7567338ed15d3ec549ce03b", "path": "inference/model.py", "symbol": "MoE.forward", "license": "MIT", "adapted": "Sparse expert selection/execution, weighted gather, and shared-expert contrast.", "simplifications": "Single process, no shared expert, no expert ownership, no all-reduce, and an explicit dropping capacity."},
        {"kind": "code", "url": "https://github.com/huggingface/transformers", "commit": "7ea2320c76117e6742364808a666ef6f2fb40a67", "path": "src/transformers/models/qwen3_moe/modeling_qwen3_moe.py", "symbol": "Qwen3MoeSparseMoeBlock.forward", "license": "Apache-2.0", "adapted": "Per-expert token selection, weighted expert outputs, and index_add gather.", "simplifications": "No shared expert/gate, sequence flattening, compiler accommodation, or model wrapper."},
        {"kind": "code", "url": "https://github.com/MoonshotAI/Kimi-K2", "commit": "1b4022bbb7187cf4011a8bdf0b4cd10e2daa26c4", "path": "docs/deploy_guidance.md", "symbol": "DeepEP and expert-parallel launch options", "license": "Modified MIT", "adapted": "System context for dynamic expert-parallel dispatch and load balancing.", "simplifications": "Local deterministic tensors with no collective communication or placement."},
        {"kind": "code", "url": "https://github.com/sgl-project/sglang", "commit": "d4801be447738522d2c62b49857c83787293be6f", "path": "python/sglang/srt/models/deepseek_v2.py", "symbol": "DeepseekV2MoE.__init__ and forward", "license": "Apache-2.0", "adapted": "Kimi-compatible TopK-to-fused-expert dispatch and explicit shared-expert fusion choices.", "simplifications": "No fused kernel, quantization, expert parallel collective, redundant placement, or shared expert."},
        {"kind": "code", "url": "https://github.com/huggingface/transformers", "commit": "7ea2320c76117e6742364808a666ef6f2fb40a67", "path": "src/transformers/models/glm4_moe/modeling_glm4_moe.py", "symbol": "Glm4MoeExperts.forward and Glm4MoeMoE.forward", "license": "Apache-2.0", "adapted": "Selected-token expert execution, weighted index-add gather, and routed/shared branch composition.", "simplifications": "Adds explicit dropping capacity and omits shared experts, routing groups, and the transformer block."},
        {"kind": "paper", "url": "https://arxiv.org/abs/2101.03961", "section": "2.2 Improved Training and Inference Efficiency"},
        {"kind": "paper", "url": "https://arxiv.org/abs/2405.04434", "section": "2.1.2 Device-Limited Routing"},
        {"kind": "paper", "url": "https://arxiv.org/abs/2507.20534", "section": "2.4.2 Parallelism for Model Scaling"},
        {"kind": "paper", "url": "https://arxiv.org/abs/2508.06471", "section": "2.1 Architecture"},
    ],
    "tests": [
        {"name": "Token-major capacity and weighted accumulation", "behavior": "capacity.overflow", "code": r"""
import torch
import torch.nn as nn
def expert(scale):
    layer=nn.Linear(2,2,bias=False)
    with torch.no_grad(): layer.weight.copy_(torch.eye(2)*scale)
    return layer
x=torch.tensor([[1.,2.],[3.,4.]])
indices=torch.tensor([[0,1],[0,1]])
weights=torch.tensor([[.75,.25],[.4,.6]])
out,stats={fn}(x,indices,weights,[expert(1.),expert(10.)],capacity=1)
assert torch.allclose(out,torch.tensor([[3.25,6.5],[0.,0.]]))
assert stats=={'accepted':2,'dropped':2,'loads':[1,1]}
"""},
        {"name": "Capacity zero drops all work", "behavior": "edge.empty_or_boundary", "code": r"""
import torch
import torch.nn as nn
class FailsIfCalled(nn.Module):
    def forward(self,x): raise AssertionError('unused expert was called')
x=torch.randn(3,4)
out,stats={fn}(x,torch.zeros(3,1,dtype=torch.long),torch.ones(3,1),[FailsIfCalled()],0)
assert torch.equal(out,torch.zeros_like(x))
assert stats=={'accepted':0,'dropped':3,'loads':[0]}
"""},
        {"name": "Independent dense oracle and token conservation", "behavior": "dispatch.token_conservation", "visibility": "unshown", "failure_message": "Accepted routes must be capacity-limited per expert, weighted once, and accumulated to their original token.", "code": r"""
import torch
import torch.nn as nn
from torch_judge.harness.tensor import dense_capacity_dispatch_oracle
for seed,capacity in [(41,1),(43,2),(47,5)]:
    generator=torch.Generator().manual_seed(seed)
    x=torch.randn(6,3,generator=generator,dtype=torch.float64)
    indices=torch.randint(0,3,(6,2),generator=generator)
    weights=torch.rand(6,2,generator=generator,dtype=torch.float64)
    experts=[]
    for e in range(3):
        layer=nn.Linear(3,3,bias=False,dtype=torch.float64)
        with torch.no_grad(): layer.weight.copy_(torch.randn(3,3,generator=generator,dtype=torch.float64))
        experts.append(layer)
    actual,actual_stats={fn}(x.clone(),indices.clone(),weights.clone(),experts,capacity)
    expected,expected_stats=dense_capacity_dispatch_oracle(x,indices,weights,experts,capacity)
    assert actual.shape==x.shape and actual.dtype==x.dtype and actual.device==x.device
    assert torch.allclose(actual,expected,atol=1e-10,rtol=1e-9)
    assert actual_stats==expected_stats
    assert actual_stats['accepted']+actual_stats['dropped']==indices.numel()
"""},
        {"name": "Only used experts run, once each", "behavior": "experts.sparsity", "visibility": "unshown", "failure_message": "Dispatch must batch accepted rows and call each used expert once without executing unused experts.", "code": r"""
import torch
import torch.nn as nn
from torch_judge.harness.tensor import CountingExpert
experts=[CountingExpert(nn.Identity()) for _ in range(4)]
x=torch.randn(5,3)
indices=torch.tensor([[0],[0],[2],[0],[2]])
weights=torch.ones(5,1)
out,stats={fn}(x,indices,weights,experts,capacity=2)
assert [(e.calls,e.tokens) for e in experts]==[(1,2),(0,0),(1,2),(0,0)]
assert stats=={'accepted':4,'dropped':1,'loads':[2,0,2,0]}
"""},
        {"name": "Gradients reach accepted experts and gates only", "behavior": "gradient.flow", "visibility": "unshown", "failure_message": "Only accepted expert paths and accepted gate weights should receive gradients.", "code": r"""
import torch
import torch.nn as nn
from torch_judge.harness.tensor import assert_expert_gradient_partition
experts=[nn.Linear(2,2,bias=False) for _ in range(3)]
x=torch.randn(3,2,requires_grad=True)
weights=torch.tensor([[.7,.3],[.6,.4],[.8,.2]],requires_grad=True)
indices=torch.tensor([[0,1],[0,1],[0,1]])
out,stats={fn}(x,indices,weights,experts,capacity=1)
(out*torch.tensor([[1.,2.],[3.,4.],[5.,6.]])).sum().backward()
assert_expert_gradient_partition(experts,{0,1})
assert x.grad is not None and torch.count_nonzero(x.grad[0]) and not torch.count_nonzero(x.grad[1:])
assert torch.count_nonzero(weights.grad[0])==2 and not torch.count_nonzero(weights.grad[1:])
"""},
        {"name": "Expert permutation preserves output and statistics meaning", "behavior": "state.invariant", "visibility": "unshown", "failure_message": "Reordering experts with consistently remapped route ids must preserve output and permute loads.", "code": r"""
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
base,stats={fn}(x,indices,weights,experts,2)
remapped,reordered=permute_expert_routes(indices,experts,[2,0,1])
other,other_stats={fn}(x,remapped,weights,reordered,2)
assert torch.allclose(base,other)
assert other_stats['loads']==[stats['loads'][2],stats['loads'][0],stats['loads'][1]]
assert other_stats['accepted']==stats['accepted'] and other_stats['dropped']==stats['dropped']
"""},
    ],
    "solution": r'''def moe_capacity_dispatch(x, expert_indices, expert_weights, experts, capacity):
    if x.ndim != 2 or expert_indices.ndim != 2 or expert_weights.shape != expert_indices.shape:
        raise ValueError("x and route tensors have invalid shapes")
    if expert_indices.shape[0] != x.shape[0] or expert_indices.dtype != torch.long:
        raise ValueError("route tensors must align with tokens and indices must be long")
    if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity < 0 or not experts:
        raise ValueError("capacity and experts are invalid")
    if expert_indices.numel() and (int(expert_indices.min()) < 0 or int(expert_indices.max()) >= len(experts)):
        raise ValueError("expert index out of range")
    loads = [0 for _ in experts]
    routes = [[] for _ in experts]
    for token in range(expert_indices.shape[0]):
        for slot in range(expert_indices.shape[1]):
            expert_id = int(expert_indices[token, slot])
            if loads[expert_id] < capacity:
                routes[expert_id].append((token, slot))
                loads[expert_id] += 1
    output = torch.zeros_like(x)
    for expert_id, accepted in enumerate(routes):
        if not accepted:
            continue
        token_ids = torch.tensor([token for token, _ in accepted], device=x.device)
        slot_ids = torch.tensor([slot for _, slot in accepted], device=x.device)
        expert_output = experts[expert_id](x.index_select(0, token_ids))
        route_weights = expert_weights[token_ids, slot_ids].to(expert_output.dtype).unsqueeze(-1)
        output.index_add_(0, token_ids, expert_output * route_weights)
    accepted_count = sum(loads)
    return output, {"accepted": accepted_count, "dropped": expert_indices.numel() - accepted_count, "loads": loads}''',
    "demo": r"""experts=[nn.Linear(4,4,bias=False) for _ in range(3)]
x=torch.randn(5,4)
indices=torch.tensor([[0,1],[0,2],[1,2],[0,1],[2,0]])
weights=torch.full((5,2),.5)
output,stats=moe_capacity_dispatch(x,indices,weights,experts,capacity=2)
print(output.shape,stats)""",
}
