"""Mutation gates and metadata contracts for the MoE training path."""

from __future__ import annotations

import pytest

from torch_judge.tasks import get_task
from torch_judge.tasks._schema import validate_task
from tests.quality.mutation_runner import Mutation, assert_mutations_rejected


ROUTER_MUTATIONS = [
    Mutation("raw_topk_logits", """def moe_topk_router(router_logits, k):
    weights, indices = torch.topk(router_logits, k, dim=-1)
    return indices, weights
"""),
    Mutation("wrong_softmax_dim", """def moe_topk_router(router_logits, k):
    probs = torch.softmax(router_logits, dim=0)
    weights, indices = torch.topk(probs, k, dim=-1)
    return indices, weights / weights.sum(-1, keepdim=True)
"""),
    Mutation("unnormalized_selected", """def moe_topk_router(router_logits, k):
    probs = torch.softmax(router_logits, dim=-1)
    weights, indices = torch.topk(probs, k, dim=-1)
    return indices, weights
"""),
    Mutation("smallest_k", """def moe_topk_router(router_logits, k):
    probs = torch.softmax(router_logits, dim=-1)
    weights, indices = torch.topk(probs, k, dim=-1, largest=False)
    return indices, weights / weights.sum(-1, keepdim=True)
"""),
    Mutation("detached_weights", """def moe_topk_router(router_logits, k):
    probs = torch.softmax(router_logits, dim=-1)
    weights, indices = torch.topk(probs, k, dim=-1)
    return indices, (weights / weights.sum(-1, keepdim=True)).detach()
"""),
    Mutation("shared_first_route", """def moe_topk_router(router_logits, k):
    probs = torch.softmax(router_logits, dim=-1)
    weights, indices = torch.topk(probs, k, dim=-1)
    weights = weights / weights.sum(-1, keepdim=True)
    return indices[:1].expand_as(indices), weights[:1].expand_as(weights)
"""),
]


DISPATCH_MUTATIONS = [
    Mutation("duplicated_gather", """def moe_capacity_dispatch(x, expert_indices, expert_weights, experts, capacity):
    out = torch.zeros_like(x); loads = [0] * len(experts)
    for t in range(x.shape[0]):
        for s in range(expert_indices.shape[1]):
            e = int(expert_indices[t, s])
            if loads[e] < capacity:
                loads[e] += 1; value = expert_weights[t, s] * experts[e](x[t:t+1])[0]; out[t] += value + value
    a=sum(loads); return out, {'accepted': a, 'dropped': expert_indices.numel()-a, 'loads': loads}
"""),
    Mutation("ignores_capacity", """def moe_capacity_dispatch(x, expert_indices, expert_weights, experts, capacity):
    out = torch.zeros_like(x); loads = [0] * len(experts)
    for t in range(x.shape[0]):
        for s in range(expert_indices.shape[1]):
            e = int(expert_indices[t, s]); loads[e] += 1
            out[t] += expert_weights[t, s] * experts[e](x[t:t+1])[0]
    return out, {'accepted': expert_indices.numel(), 'dropped': 0, 'loads': loads}
"""),
    Mutation("global_capacity", """def moe_capacity_dispatch(x, expert_indices, expert_weights, experts, capacity):
    out = torch.zeros_like(x); loads = [0] * len(experts); accepted = 0
    for t in range(x.shape[0]):
        for s in range(expert_indices.shape[1]):
            if accepted < capacity:
                e = int(expert_indices[t, s]); loads[e] += 1; accepted += 1
                out[t] += expert_weights[t, s] * experts[e](x[t:t+1])[0]
    return out, {'accepted': accepted, 'dropped': expert_indices.numel()-accepted, 'loads': loads}
"""),
    Mutation("unweighted_gather", """def moe_capacity_dispatch(x, expert_indices, expert_weights, experts, capacity):
    out = torch.zeros_like(x); loads = [0] * len(experts)
    for t in range(x.shape[0]):
        for s in range(expert_indices.shape[1]):
            e = int(expert_indices[t, s])
            if loads[e] < capacity:
                loads[e] += 1; out[t] += experts[e](x[t:t+1])[0]
    a=sum(loads); return out, {'accepted': a, 'dropped': expert_indices.numel()-a, 'loads': loads}
"""),
    Mutation("execute_all_experts", """def moe_capacity_dispatch(x, expert_indices, expert_weights, experts, capacity):
    all_outputs = [expert(x) for expert in experts]; out = torch.zeros_like(x); loads = [0] * len(experts)
    for t in range(x.shape[0]):
        for s in range(expert_indices.shape[1]):
            e = int(expert_indices[t,s])
            if loads[e] < capacity:
                loads[e] += 1; out[t] += expert_weights[t,s] * all_outputs[e][t]
    a=sum(loads); return out, {'accepted': a, 'dropped': expert_indices.numel()-a, 'loads': loads}
"""),
    Mutation("overwrite_duplicate_token", """def moe_capacity_dispatch(x, expert_indices, expert_weights, experts, capacity):
    out = torch.zeros_like(x); loads = [0] * len(experts)
    for t in range(x.shape[0]):
        for s in range(expert_indices.shape[1]):
            e = int(expert_indices[t,s])
            if loads[e] < capacity:
                loads[e] += 1; out[t] = expert_weights[t,s] * experts[e](x[t:t+1])[0]
    a=sum(loads); return out, {'accepted': a, 'dropped': expert_indices.numel()-a, 'loads': loads}
"""),
]


TRAIN_MUTATIONS = [
    Mutation("split_backward", """def tiny_moe_train_step(model, optimizer, x, targets, balance_weight=0.01, z_weight=0.001):
    optimizer.zero_grad(); logits,r=model(x); task=F.cross_entropy(logits,targets); p=torch.softmax(r,-1).mean(0); b=r.shape[-1]*p.square().sum(); z=torch.logsumexp(r,-1).square().mean(); auxiliary=balance_weight*b+z_weight*z; loss=task+auxiliary; task.backward(retain_graph=True); auxiliary.backward(); optimizer.step(); return {k:v.detach() for k,v in {'loss':loss,'task_loss':task,'balance_loss':b,'z_loss':z}.items()}
"""),
    Mutation("double_backward", """def tiny_moe_train_step(model, optimizer, x, targets, balance_weight=0.01, z_weight=0.001):
    optimizer.zero_grad(); logits,r=model(x); task=F.cross_entropy(logits,targets); p=torch.softmax(r,-1).mean(0); b=r.shape[-1]*p.square().sum(); z=torch.logsumexp(r,-1).square().mean(); loss=task+balance_weight*b+z_weight*z; loss.backward(retain_graph=True)
    for parameter in model.parameters(): parameter.grad=None
    loss.backward(); optimizer.step(); return {k:v.detach() for k,v in {'loss':loss,'task_loss':task,'balance_loss':b,'z_loss':z}.items()}
"""),
    Mutation("double_forward", """def tiny_moe_train_step(model, optimizer, x, targets, balance_weight=0.01, z_weight=0.001):
    optimizer.zero_grad(); model(x); logits,r=model(x); task=F.cross_entropy(logits,targets); p=torch.softmax(r,-1).mean(0); b=r.shape[-1]*p.square().sum(); z=torch.logsumexp(r,-1).square().mean(); loss=task+balance_weight*b+z_weight*z; loss.backward(); optimizer.step(); return {k:v.detach() for k,v in {'loss':loss,'task_loss':task,'balance_loss':b,'z_loss':z}.items()}
"""),
    Mutation("task_loss_only", """def tiny_moe_train_step(model, optimizer, x, targets, balance_weight=0.01, z_weight=0.001):
    optimizer.zero_grad(); logits, router_logits = model(x); task = F.cross_entropy(logits, targets)
    task.backward(); optimizer.step()
    z=torch.logsumexp(router_logits,-1).square().mean(); p=torch.softmax(router_logits,-1).mean(0); b=router_logits.shape[-1]*p.square().sum()
    return {k:v.detach() for k,v in {'loss':task,'task_loss':task,'balance_loss':b,'z_loss':z}.items()}
"""),
    Mutation("detached_auxiliary", """def tiny_moe_train_step(model, optimizer, x, targets, balance_weight=0.01, z_weight=0.001):
    optimizer.zero_grad(); logits, r = model(x); task=F.cross_entropy(logits,targets); p=torch.softmax(r,-1).mean(0); b=r.shape[-1]*p.square().sum(); z=torch.logsumexp(r,-1).square().mean(); loss=task+balance_weight*b.detach()+z_weight*z.detach(); loss.backward(); optimizer.step(); return {k:v.detach() for k,v in {'loss':loss,'task_loss':task,'balance_loss':b,'z_loss':z}.items()}
"""),
    Mutation("wrong_z_reduction", """def tiny_moe_train_step(model, optimizer, x, targets, balance_weight=0.01, z_weight=0.001):
    optimizer.zero_grad(); logits,r=model(x); task=F.cross_entropy(logits,targets); p=torch.softmax(r,-1).mean(0); b=r.shape[-1]*p.square().sum(); z=torch.logsumexp(r,-1).mean().square(); loss=task+balance_weight*b+z_weight*z; loss.backward(); optimizer.step(); return {k:v.detach() for k,v in {'loss':loss,'task_loss':task,'balance_loss':b,'z_loss':z}.items()}
"""),
    Mutation("missing_zero_grad", """def tiny_moe_train_step(model, optimizer, x, targets, balance_weight=0.01, z_weight=0.001):
    logits,r=model(x); task=F.cross_entropy(logits,targets); p=torch.softmax(r,-1).mean(0); b=r.shape[-1]*p.square().sum(); z=torch.logsumexp(r,-1).square().mean(); loss=task+balance_weight*b+z_weight*z; loss.backward(); optimizer.step(); return {k:v.detach() for k,v in {'loss':loss,'task_loss':task,'balance_loss':b,'z_loss':z}.items()}
"""),
    Mutation("double_step", """def tiny_moe_train_step(model, optimizer, x, targets, balance_weight=0.01, z_weight=0.001):
    optimizer.zero_grad(); logits,r=model(x); task=F.cross_entropy(logits,targets); p=torch.softmax(r,-1).mean(0); b=r.shape[-1]*p.square().sum(); z=torch.logsumexp(r,-1).square().mean(); loss=task+balance_weight*b+z_weight*z; loss.backward(); optimizer.step(); optimizer.step(); return {k:v.detach() for k,v in {'loss':loss,'task_loss':task,'balance_loss':b,'z_loss':z}.items()}
"""),
    Mutation("no_step", """def tiny_moe_train_step(model, optimizer, x, targets, balance_weight=0.01, z_weight=0.001):
    optimizer.zero_grad(); logits,r=model(x); task=F.cross_entropy(logits,targets); p=torch.softmax(r,-1).mean(0); b=r.shape[-1]*p.square().sum(); z=torch.logsumexp(r,-1).square().mean(); loss=task+balance_weight*b+z_weight*z; loss.backward(); return {k:v.detach() for k,v in {'loss':loss,'task_loss':task,'balance_loss':b,'z_loss':z}.items()}
"""),
]


SPARSE_FFN_MUTATIONS = [
    Mutation("execute_all_experts", """def sparse_ffn_forward(x, expert_indices, expert_weights, experts):
    dense = [e(x) for e in experts]; out = torch.zeros_like(x)
    for t in range(expert_indices.shape[0]):
        for s in range(expert_indices.shape[1]):
            e = int(expert_indices[t, s]); out[t] += expert_weights[t, s] * dense[e][t]
    return out
"""),
    Mutation("unweighted_gather", """def sparse_ffn_forward(x, expert_indices, expert_weights, experts):
    out = torch.zeros_like(x); routes = [[] for _ in experts]
    for t in range(expert_indices.shape[0]):
        for s in range(expert_indices.shape[1]): routes[int(expert_indices[t, s])].append((t, s))
    for eid, acc in enumerate(routes):
        if not acc: continue
        tid = torch.tensor([t for t, _ in acc]); out.index_add_(0, tid, experts[eid](x.index_select(0, tid)))
    return out
"""),
    Mutation("only_first_route", """def sparse_ffn_forward(x, expert_indices, expert_weights, experts):
    out = torch.zeros_like(x)
    for t in range(expert_indices.shape[0]):
        e = int(expert_indices[t, 0]); out[t] = expert_weights[t, 0] * experts[e](x[t:t+1])[0]
    return out
"""),
    Mutation("overwrite_token", """def sparse_ffn_forward(x, expert_indices, expert_weights, experts):
    out = torch.zeros_like(x); routes = [[] for _ in experts]
    for t in range(expert_indices.shape[0]):
        for s in range(expert_indices.shape[1]): routes[int(expert_indices[t, s])].append((t, s))
    for eid, acc in enumerate(routes):
        if not acc: continue
        tid = torch.tensor([t for t, _ in acc]); sid = torch.tensor([s for _, s in acc])
        o = experts[eid](x.index_select(0, tid)); w = expert_weights[tid, sid].to(o.dtype).unsqueeze(-1)
        out[tid] = o * w
    return out
"""),
    Mutation("detached_weights", """def sparse_ffn_forward(x, expert_indices, expert_weights, experts):
    out = torch.zeros_like(x); routes = [[] for _ in experts]
    for t in range(expert_indices.shape[0]):
        for s in range(expert_indices.shape[1]): routes[int(expert_indices[t, s])].append((t, s))
    for eid, acc in enumerate(routes):
        if not acc: continue
        tid = torch.tensor([t for t, _ in acc]); sid = torch.tensor([s for _, s in acc])
        o = experts[eid](x.index_select(0, tid)); w = expert_weights[tid, sid].detach().to(o.dtype).unsqueeze(-1)
        out.index_add_(0, tid, o * w)
    return out
"""),
    Mutation("per_token_calls", """def sparse_ffn_forward(x, expert_indices, expert_weights, experts):
    out = torch.zeros_like(x)
    for t in range(expert_indices.shape[0]):
        for s in range(expert_indices.shape[1]):
            e = int(expert_indices[t, s]); out[t] += expert_weights[t, s] * experts[e](x[t:t+1])[0]
    return out
"""),
]


ZLOSS_MUTATIONS = [
    Mutation("no_square", "def router_z_loss(router_logits):\n    return torch.logsumexp(router_logits, dim=-1).mean()\n"),
    Mutation("mean_then_square", "def router_z_loss(router_logits):\n    return torch.logsumexp(router_logits, dim=-1).mean().square()\n"),
    Mutation("sum_not_mean", "def router_z_loss(router_logits):\n    return torch.logsumexp(router_logits, dim=-1).square().sum()\n"),
    Mutation("naive_logsumexp", "def router_z_loss(router_logits):\n    return torch.log(torch.exp(router_logits).sum(dim=-1)).square().mean()\n"),
    Mutation("wrong_dim", "def router_z_loss(router_logits):\n    return torch.logsumexp(router_logits, dim=0).square().mean()\n"),
    Mutation("detached", "def router_z_loss(router_logits):\n    return torch.logsumexp(router_logits, dim=-1).square().mean().detach()\n"),
]


def _assert_metadata(task_id: str):
    task = get_task(task_id)
    assert task is not None
    validate_task(task_id, task)
    assert task["version"] == 1
    assert [hint["kind"] for hint in task["hints"]] == ["questions", "analysis"]
    assert any(test.get("visibility") == "unshown" for test in task["tests"])
    assert task["model_connections"] and task["pro_con_analysis"]["pros"] and task["pro_con_analysis"]["cons"]
    assert any(source["kind"] == "code" for source in task["sources"])


def test_moe_metadata_contracts_and_model_coverage():
    for task_id in ("moe_topk_router", "moe_capacity_dispatch", "tiny_moe_train_step"):
        _assert_metadata(task_id)
    text = " ".join(get_task("moe_topk_router")["model_connections"])
    for model in ("DeepSeek", "Kimi", "GLM", "Qwen"):
        assert model in text
    assert "does not renormalize" in text
    assert "norm_topk_prob" in text
    source_paths = {
        source["path"]
        for source in get_task("moe_topk_router")["sources"]
        if source["kind"] == "code"
    }
    assert "python/sglang/srt/models/deepseek_v2.py" in source_paths
    assert "src/transformers/models/glm4_moe/modeling_glm4_moe.py" in source_paths
    dispatch = get_task("moe_capacity_dispatch")
    assert "capacity_factor" in dispatch["description_en"]
    assert "Shared experts are a separate always-on branch" in dispatch["description_en"]
    for task_id in ("moe_capacity_dispatch", "tiny_moe_train_step"):
        paths = {
            source["path"]
            for source in get_task(task_id)["sources"]
            if source["kind"] == "code"
        }
        assert "python/sglang/srt/models/deepseek_v2.py" in paths
        assert "src/transformers/models/glm4_moe/modeling_glm4_moe.py" in paths


@pytest.mark.parametrize("_repeat", range(3))
def test_sparse_ffn_reference_and_mutations(_repeat):
    assert set(assert_mutations_rejected("dense_vs_sparse_ffn", SPARSE_FFN_MUTATIONS)) == {m.name for m in SPARSE_FFN_MUTATIONS}


@pytest.mark.parametrize("_repeat", range(3))
def test_router_zloss_reference_and_mutations(_repeat):
    assert set(assert_mutations_rejected("moe_router_zloss", ZLOSS_MUTATIONS)) == {m.name for m in ZLOSS_MUTATIONS}


@pytest.mark.parametrize("_repeat", range(3))
def test_router_reference_and_mutations(_repeat):
    assert set(assert_mutations_rejected("moe_topk_router", ROUTER_MUTATIONS)) == {m.name for m in ROUTER_MUTATIONS}


@pytest.mark.parametrize("_repeat", range(3))
def test_dispatch_reference_and_mutations(_repeat):
    assert set(assert_mutations_rejected("moe_capacity_dispatch", DISPATCH_MUTATIONS)) == {m.name for m in DISPATCH_MUTATIONS}


@pytest.mark.parametrize("_repeat", range(3))
def test_train_step_reference_and_mutations(_repeat):
    assert set(assert_mutations_rejected("tiny_moe_train_step", TRAIN_MUTATIONS)) == {m.name for m in TRAIN_MUTATIONS}
