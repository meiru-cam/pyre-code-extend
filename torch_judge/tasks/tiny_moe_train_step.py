"""One complete differentiable TinyMoE optimization step."""

TASK = {
    "title": "Train a TinyMoE Router and Experts",
    "difficulty": "Hard",
    "version": 1,
    "function_name": "tiny_moe_train_step",
    "description_en": r"""Implement one complete MoE training step.

Signature: 'tiny_moe_train_step(model, optimizer, x, targets, balance_weight=0.01, z_weight=0.001) -> dict'

'model(x)' returns '(logits, router_logits)', with router logits shaped '(tokens, experts)'. Compute:

- 'task_loss = cross_entropy(logits, targets)'
- 'importance = softmax(router_logits, -1).mean(0)'
- 'balance_loss = num_experts * sum(importance ** 2)'
- 'z_loss = mean(logsumexp(router_logits, -1) ** 2)'
- 'loss = task_loss + balance_weight * balance_loss + z_weight * z_loss'

Call 'optimizer.zero_grad()', forward once, backward on the total loss once, and 'optimizer.step()' once. Return exactly 'loss', 'task_loss', 'balance_loss', and 'z_loss' as detached scalar tensors.

How routing learns: task gradients reach selected gate values and expert computation in a sparse model; the soft importance penalty provides a differentiable pressure against probability collapse; router z-loss discourages excessively large logits. Hard top-k index changes themselves are not ordinarily differentiable. Production systems additionally use token-count objectives, routing bias updates, noisy routing, capacity policy, expert-parallel collectives, and extensive utilization telemetry. A falling total loss alone does not prove healthy routing: inspect per-expert load, drops, gate entropy, specialization, and task quality.

Optional code reading: Qwen3-MoE's causal-LM forward integrates router auxiliary loss with the task loss. DeepSeek-V3 documents auxiliary-loss-free bias-based balancing, which is deliberately different from this exercise.""",
    "advisory_prerequisites": ["cross_entropy", "moe_topk_router", "moe_capacity_dispatch", "moe_load_balance", "adam"],
    "hints": [
        {"level": 1, "kind": "questions", "content": "Which terms must remain attached when total loss is built? Over which dimension is expert importance averaged? When should old gradients be cleared, and how many optimizer updates belong to one training step?"},
        {"level": 2, "kind": "analysis", "content": "Clear gradients, run the model, compute cross entropy, mean softmax probability per expert, E times its squared sum, and mean squared logsumexp. Backpropagate their weighted sum, step once, then detach each scalar only while constructing the metrics dictionary."},
    ],
    "model_connections": [
        "Qwen3-MoE adds a router load-balancing auxiliary loss to language-model loss during training when router logits are requested.",
        "DeepSeek-V3 replaces a conventional global auxiliary objective with auxiliary-loss-free routing-bias updates plus a small sequence-wise balance term; this exercise intentionally teaches the simpler differentiable baseline.",
        "Kimi K2 reports large-scale MoE pretraining with MuonClip stability work, while SGLang's Kimi-compatible runtime makes its grouped routing and expert execution boundaries inspectable.",
        "GLM-4.5's implementation exposes router logits and sparse/shared MoE composition; its training report shows why specialization also depends on data mixture and distributed optimization.",
    ],
    "pro_con_analysis": {
        "pros": [
            "One explicit step exposes how task, balancing, and stability gradients combine.",
            "Soft importance balancing is fully differentiable and has a clear uniform minimum of one.",
            "Exact optimizer-delta tests verify training behavior rather than accepting plausible metric output.",
        ],
        "cons": [
            "Soft probability balance does not guarantee equal hard top-k loads or prevent capacity overflow.",
            "The z-loss target depends on router parameterization and its coefficient needs tuning.",
            "A single local step omits expert parallelism, mixed precision, accumulation, clipping, scheduling, and checkpoint state.",
        ],
    },
    "sources": [
        {"kind": "code", "url": "https://github.com/huggingface/transformers", "commit": "7ea2320c76117e6742364808a666ef6f2fb40a67", "path": "src/transformers/models/qwen3_moe/modeling_qwen3_moe.py", "symbol": "load_balancing_loss_func and Qwen3MoeForCausalLM.forward", "license": "Apache-2.0", "adapted": "Router probability balancing and integration of an auxiliary router term with task loss.", "simplifications": "A standalone single-step function and soft-importance-square objective instead of Qwen's top-k token-frequency product."},
        {"kind": "code", "url": "https://github.com/deepseek-ai/DeepSeek-V3", "commit": "9b4e9788e4a3a731f7567338ed15d3ec549ce03b", "path": "inference/model.py", "symbol": "Gate.forward and MoE.forward", "license": "MIT", "adapted": "Separation between router logits/scores, routed expert computation, and weighted combination.", "simplifications": "Training-only losses absent from the inference file are specified independently from cited papers."},
        {"kind": "code", "url": "https://github.com/MoonshotAI/Kimi-K2", "commit": "1b4022bbb7187cf4011a8bdf0b4cd10e2daa26c4", "path": "README.md", "symbol": "Large-Scale Training and MuonClip sections", "license": "Modified MIT", "adapted": "Context that optimizer stability and MoE training must be evaluated together at scale.", "simplifications": "Plain optimizer API with no Muon, clipping, distributed state, or mixed precision."},
        {"kind": "code", "url": "https://github.com/sgl-project/sglang", "commit": "d4801be447738522d2c62b49857c83787293be6f", "path": "python/sglang/srt/models/deepseek_v2.py", "symbol": "MoEGate and DeepseekV2MoE", "license": "Apache-2.0", "adapted": "Inspectable Kimi-compatible router, TopK, fused expert, and shared-expert runtime boundaries.", "simplifications": "The exercise trains a tiny local model and does not reproduce this inference runtime or its kernels."},
        {"kind": "code", "url": "https://github.com/huggingface/transformers", "commit": "7ea2320c76117e6742364808a666ef6f2fb40a67", "path": "src/transformers/models/glm4_moe/modeling_glm4_moe.py", "symbol": "Glm4MoeTopkRouter.forward and Glm4MoeMoE.forward", "license": "Apache-2.0", "adapted": "The router's explicit logits output and the separation between sparse routed experts and the shared branch.", "simplifications": "The exercise exposes router logits to its loss directly and omits the transformer, shared expert, grouped sigmoid router, and distributed optimizer."},
        {"kind": "paper", "url": "https://arxiv.org/abs/2202.08906", "section": "2.4 Router z-loss"},
        {"kind": "paper", "url": "https://arxiv.org/abs/2412.19437", "section": "2.2.2 Auxiliary-Loss-Free Load Balancing"},
        {"kind": "paper", "url": "https://arxiv.org/abs/2507.20534", "section": "2.1 MuonClip and 2.5 Training recipe"},
        {"kind": "paper", "url": "https://arxiv.org/abs/2508.06471", "section": "2 Pre-Training"},
    ],
    "tests": [
        {"name": "Loss formulas and detached metric contract", "behavior": "training.load_balance", "code": r"""
import torch
import torch.nn as nn
import torch.nn.functional as F
class FixedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.logits=nn.Parameter(torch.tensor([[2.,0.,-1.],[0.,1.,3.]]))
        self.router=nn.Parameter(torch.tensor([[1.,0.],[2.,-1.]]))
    def forward(self,x): return self.logits,self.router
model=FixedModel(); optimizer=torch.optim.SGD(model.parameters(),lr=0.0)
targets=torch.tensor([0,2])
before_logits=model.logits.detach().clone(); before_router=model.router.detach().clone()
task=F.cross_entropy(before_logits,targets)
importance=torch.softmax(before_router,-1).mean(0)
balance=2*importance.square().sum()
z=torch.logsumexp(before_router,-1).square().mean()
expected=task+.2*balance+.03*z
metrics={fn}(model,optimizer,torch.empty(2,0),targets,.2,.03)
assert set(metrics)=={'loss','task_loss','balance_loss','z_loss'}
for value in metrics.values():
    assert isinstance(value,torch.Tensor) and value.ndim==0 and not value.requires_grad
assert torch.allclose(metrics['task_loss'],task)
assert torch.allclose(metrics['balance_loss'],balance)
assert torch.allclose(metrics['z_loss'],z)
assert torch.allclose(metrics['loss'],expected)
"""},
        {"name": "Exactly one zero, backward update, and step", "behavior": "contract.signature", "code": r"""
import torch
import torch.nn as nn
class CountingSGD(torch.optim.SGD):
    def __init__(self,params,lr): super().__init__(params,lr=lr); self.zero_calls=0; self.step_calls=0
    def zero_grad(self,*args,**kwargs): self.zero_calls+=1; return super().zero_grad(*args,**kwargs)
    def step(self,*args,**kwargs): self.step_calls+=1; return super().step(*args,**kwargs)
class Model(nn.Module):
    def __init__(self):
        super().__init__(); self.trunk=nn.Linear(3,3,bias=False); self.head=nn.Linear(3,2,bias=False); self.router=nn.Linear(3,2,bias=False); self.forward_calls=0; self.backward_calls=0
        def count_backward(gradient):
            self.backward_calls+=1
            return gradient
        self.trunk.weight.register_hook(count_backward)
    def forward(self,x):
        self.forward_calls+=1; hidden=self.trunk(x); return self.head(hidden),self.router(hidden)
torch.manual_seed(59)
model=Model(); optimizer=CountingSGD(model.parameters(),.05)
x=torch.randn(4,3); targets=torch.tensor([0,1,1,0])
before=[p.detach().clone() for p in model.parameters()]
{fn}(model,optimizer,x,targets)
assert model.forward_calls==1 and model.backward_calls==1
assert optimizer.zero_calls==1 and optimizer.step_calls==1
assert all(p.grad is not None for p in model.parameters())
assert any(not torch.equal(old,p) for old,p in zip(before,model.parameters()))
"""},
        {"name": "Exact parameter delta matches independent PyTorch step", "behavior": "training.router_gradient", "visibility": "unshown", "failure_message": "The optimizer update must use the attached weighted total loss exactly once after clearing old gradients.", "code": r"""
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
class Model(nn.Module):
    def __init__(self):
        super().__init__(); self.head=nn.Linear(4,3,bias=False); self.router=nn.Linear(4,3,bias=False)
    def forward(self,x): return self.head(x),self.router(x)
torch.manual_seed(61)
candidate=Model(); expected=copy.deepcopy(candidate)
x=torch.randn(7,4); targets=torch.tensor([0,1,2,0,2,1,0])
candidate_optimizer=torch.optim.SGD(candidate.parameters(),lr=.07,momentum=.2)
expected_optimizer=torch.optim.SGD(expected.parameters(),lr=.07,momentum=.2)
for parameter in candidate.parameters(): parameter.grad=torch.ones_like(parameter)*99
{fn}(candidate,candidate_optimizer,x,targets,.13,.017)
expected_optimizer.zero_grad()
logits,router=expected(x)
task=F.cross_entropy(logits,targets)
importance=torch.softmax(router,-1).mean(0)
balance=router.shape[-1]*importance.square().sum()
z=torch.logsumexp(router,-1).square().mean()
(task+.13*balance+.017*z).backward()
expected_optimizer.step()
for actual,oracle in zip(candidate.parameters(),expected.parameters()):
    assert torch.allclose(actual,oracle,atol=1e-8,rtol=1e-7)
"""},
        {"name": "Router and expert gradients remain observable", "behavior": "gradient.flow", "visibility": "unshown", "failure_message": "The total loss must send finite nonzero gradients into the router and task-selected expert computation.", "code": r"""
import torch
import torch.nn as nn
class SparseTinyMoE(nn.Module):
    def __init__(self):
        super().__init__(); self.router=nn.Linear(3,3,bias=False); self.experts=nn.ModuleList([nn.Linear(3,2,bias=False) for _ in range(3)])
        with torch.no_grad(): self.router.weight.copy_(torch.tensor([[1.,0.,0.],[0.,1.,0.],[-1.,-1.,0.]]))
    def forward(self,x):
        router_logits=self.router(x); probabilities=torch.softmax(router_logits,-1)
        gates,indices=torch.topk(probabilities,2,dim=-1); gates=gates/gates.sum(-1,keepdim=True)
        logits=torch.zeros(x.shape[0],2,dtype=x.dtype,device=x.device)
        for expert_id,expert in enumerate(self.experts):
            slots,tokens=torch.where(indices.transpose(0,1)==expert_id)
            if tokens.numel():
                logits.index_add_(0,tokens,expert(x[tokens])*gates[tokens,slots,None])
        return logits,router_logits
torch.manual_seed(67)
model=SparseTinyMoE(); optimizer=torch.optim.SGD(model.parameters(),lr=.01)
x=torch.tensor([[4.,0.,0.],[3.,1.,0.],[0.,4.,0.],[1.,3.,0.]])
{fn}(model,optimizer,x,torch.tensor([0,0,1,1]))
assert model.router.weight.grad is not None and torch.isfinite(model.router.weight.grad).all() and torch.count_nonzero(model.router.weight.grad)
from torch_judge.harness.tensor import assert_expert_gradient_partition
assert_expert_gradient_partition(model.experts,{0,1})
"""},
        {"name": "Seeded tiny domains train repeatably", "behavior": "state.invariant", "visibility": "unshown", "failure_message": "Repeated seeded TinyMoE runs must produce identical metrics and a useful task-loss signal.", "code": r"""
import torch
import torch.nn as nn
from torch_judge.harness.tensor import seeded_domain_batch
class DomainMoE(nn.Module):
    def __init__(self):
        super().__init__(); self.router=nn.Linear(4,3,bias=False); self.experts=nn.ModuleList([nn.Linear(4,3) for _ in range(3)])
        with torch.no_grad(): self.router.weight.copy_(torch.eye(3,4))
    def forward(self,x):
        r=self.router(x); routes=r.argmax(-1); logits=torch.zeros(x.shape[0],3,dtype=x.dtype,device=x.device)
        for expert_id,expert in enumerate(self.experts):
            tokens=torch.where(routes==expert_id)[0]
            if tokens.numel(): logits.index_copy_(0,tokens,expert(x[tokens]))
        return logits,r
def trajectory():
    torch.manual_seed(71)
    model=DomainMoE(); optimizer=torch.optim.SGD(model.parameters(),lr=.08)
    x,targets,_=seeded_domain_batch(73,tokens_per_domain=5,input_dim=4,num_domains=3)
    initial_routes=model.router(x).argmax(-1); values=[]
    for _ in range(8):
        metrics={fn}(model,optimizer,x,targets,.02,.001)
        values.append(tuple(float(metrics[key]) for key in ('loss','task_loss','balance_loss','z_loss')))
    final_routes=model.router(x).argmax(-1)
    loads=torch.bincount(final_routes,minlength=3)
    return values,initial_routes,final_routes,loads,targets
first=trajectory(); second=trajectory()
assert first[0]==second[0]
assert all(torch.equal(a,b) for a,b in zip(first[1:],second[1:]))
values,initial_routes,final_routes,loads,domains=first
assert values[-1][1] < values[0][1]
assert torch.equal(initial_routes,domains) and torch.equal(final_routes,domains)
assert torch.equal(loads,torch.tensor([5,5,5]))
"""},
    ],
    "solution": r'''def tiny_moe_train_step(model, optimizer, x, targets, balance_weight=0.01, z_weight=0.001):
    optimizer.zero_grad()
    logits, router_logits = model(x)
    task_loss = F.cross_entropy(logits, targets)
    importance = torch.softmax(router_logits, dim=-1).mean(dim=0)
    balance_loss = router_logits.shape[-1] * importance.square().sum()
    z_loss = torch.logsumexp(router_logits, dim=-1).square().mean()
    loss = task_loss + balance_weight * balance_loss + z_weight * z_loss
    loss.backward()
    optimizer.step()
    return {
        "loss": loss.detach(),
        "task_loss": task_loss.detach(),
        "balance_loss": balance_loss.detach(),
        "z_loss": z_loss.detach(),
    }''',
    "demo": r"""model=nn.Sequential()
# Supply a model whose forward returns (task_logits, router_logits), then:
# metrics=tiny_moe_train_step(model,optimizer,x,targets)
# print({name: float(value) for name,value in metrics.items()})""",
}
