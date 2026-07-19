"""Quality gates for the Advanced Attention exercise evaluators."""

from __future__ import annotations

from torch_judge.tasks import get_task
from torch_judge.tasks._schema import validate_task
from tests.quality.mutation_runner import Mutation, assert_mutations_rejected


QK_NORM_MUTATIONS = [
    Mutation(
        "l2_norm",
        """def qk_norm(q, k, eps=1e-6):
    return F.normalize(q, p=2, dim=-1, eps=eps), F.normalize(k, p=2, dim=-1, eps=eps)
""",
    ),
    Mutation(
        "normalize_dim_zero",
        """def qk_norm(q, k, eps=1e-6):
    return q * torch.rsqrt(q.square().mean(dim=0, keepdim=True) + eps), k * torch.rsqrt(k.square().mean(dim=0, keepdim=True) + eps)
""",
    ),
    Mutation(
        "shared_q_statistics",
        """def qk_norm(q, k, eps=1e-6):
    scale = torch.rsqrt((q.square() + k.square()).mean(dim=-1, keepdim=True) / 2 + eps)
    return q * scale, k * scale
""",
    ),
    Mutation(
        "missing_eps",
        """def qk_norm(q, k, eps=1e-6):
    return q * torch.rsqrt(q.square().mean(dim=-1, keepdim=True)), k * torch.rsqrt(k.square().mean(dim=-1, keepdim=True))
""",
    ),
    Mutation(
        "detach_inputs",
        """def qk_norm(q, k, eps=1e-6):
    q_out = q * torch.rsqrt(q.square().mean(dim=-1, keepdim=True) + eps)
    k_out = k * torch.rsqrt(k.square().mean(dim=-1, keepdim=True) + eps)
    return q_out.detach(), k_out.detach()
""",
    ),
]


def _hybrid_mutation(name: str, schedule: str, causal: bool = True) -> Mutation:
    causal_clause = "allowed = key_pos <= query_pos" if causal else "allowed = torch.ones_like(key_pos <= query_pos)"
    code = f'''def hybrid_attention_schedule(Q, K, V, layer_index, window_size, global_every):
    scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(Q.shape[-1])
    sequence = Q.shape[1]
    query_pos = torch.arange(sequence, device=Q.device).unsqueeze(1)
    key_pos = torch.arange(sequence, device=Q.device).unsqueeze(0)
    {causal_clause}
    is_global = {schedule}
    if not is_global:
        allowed = allowed & (key_pos >= query_pos - window_size)
    scores = scores.masked_fill(~allowed.unsqueeze(0), float("-inf"))
    return torch.bmm(torch.softmax(scores, dim=-1), V)
'''
    return Mutation(name, code)


HYBRID_MUTATIONS = [
    _hybrid_mutation("always_global", "True"),
    _hybrid_mutation("always_local", "False"),
    _hybrid_mutation("future_visible", "False", causal=False),
    Mutation(
        "symmetric_window",
        '''def hybrid_attention_schedule(Q, K, V, layer_index, window_size, global_every):
    scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(Q.shape[-1])
    positions = torch.arange(Q.shape[1], device=Q.device)
    allowed = (positions[:, None] - positions[None, :]).abs() <= window_size
    scores = scores.masked_fill(~allowed.unsqueeze(0), float("-inf"))
    return torch.bmm(torch.softmax(scores, dim=-1), V)
''',
    ),
    _hybrid_mutation("off_by_one_schedule", "layer_index % global_every == 0"),
]


def _frontier_mutation(
    name: str,
    *,
    repeat_mode: str = "contiguous",
    normalize_values: bool = False,
    mask_mode: str = "scheduled",
    scale: str = "self.head_dim ** -0.5",
    softmax_dim: int = -1,
    detach_kv: bool = False,
) -> Mutation:
    if repeat_mode == "contiguous":
        repeat_lines = """key = key.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
        value = value.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)"""
    elif repeat_mode == "interleaved":
        repeat_lines = """key = key.repeat(1, self.num_heads // self.num_kv_heads, 1, 1)
        value = value.repeat(1, self.num_heads // self.num_kv_heads, 1, 1)"""
    else:
        repeat_lines = "pass"
    value_norm = "value = norm(value)" if normalize_values else ""
    detach_lines = "key = key.detach(); value = value.detach()" if detach_kv else ""
    if mask_mode == "global":
        mask_lines = "allowed = key_pos <= query_pos"
    elif mask_mode == "noncausal":
        mask_lines = "allowed = torch.ones((sequence, sequence), dtype=torch.bool, device=x.device)"
    else:
        mask_lines = """allowed = key_pos <= query_pos
        if (self.layer_index + 1) % self.global_every != 0:
            allowed = allowed & (key_pos >= query_pos - self.window_size)"""
    return Mutation(
        name,
        f'''class FrontierAttentionBlock(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads, layer_index, window_size, global_every, use_qk_norm=True, eps=1e-6):
        super().__init__()
        if d_model % num_heads or num_heads % num_kv_heads or layer_index < 0 or window_size < 0 or global_every < 1:
            raise ValueError("invalid configuration")
        self.num_heads, self.num_kv_heads = num_heads, num_kv_heads
        self.head_dim, self.layer_index = d_model // num_heads, layer_index
        self.window_size, self.global_every = window_size, global_every
        self.use_qk_norm, self.eps = use_qk_norm, eps
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        batch, sequence, _ = x.shape
        query = self.q_proj(x).view(batch, sequence, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.k_proj(x).view(batch, sequence, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(x).view(batch, sequence, self.num_kv_heads, self.head_dim).transpose(1, 2)
        if self.use_qk_norm:
            def norm(t): return t * torch.rsqrt(t.square().mean(dim=-1, keepdim=True) + self.eps)
            query, key = norm(query), norm(key)
            {value_norm}
        {repeat_lines}
        {detach_lines}
        scores = torch.matmul(query, key.transpose(-2, -1)) * ({scale})
        query_pos = torch.arange(sequence, device=x.device).unsqueeze(1)
        key_pos = torch.arange(sequence, device=x.device).unsqueeze(0)
        {mask_lines}
        scores = scores.masked_fill(~allowed.view(1, 1, sequence, sequence), float("-inf"))
        weights = torch.softmax(scores, dim={softmax_dim})
        context = torch.matmul(weights, value).transpose(1, 2).reshape(batch, sequence, -1)
        return self.o_proj(context)
''',
    )


FRONTIER_MUTATIONS = [
    _frontier_mutation("no_gqa_repeat", repeat_mode="none"),
    _frontier_mutation("interleaved_gqa_repeat", repeat_mode="interleaved"),
    _frontier_mutation("normalize_values", normalize_values=True),
    _frontier_mutation("always_global", mask_mode="global"),
    _frontier_mutation("noncausal", mask_mode="noncausal"),
    _frontier_mutation("wrong_scale", scale="self.q_proj.in_features ** -0.5"),
    _frontier_mutation("softmax_wrong_dim", softmax_dim=-2),
    _frontier_mutation("detached_kv", detach_kv=True),
]


def test_qk_norm_metadata_has_curriculum_contract():
    task = get_task("qk_norm")
    assert task is not None
    validate_task("qk_norm", task, known_ids={"qk_norm", "rmsnorm", "attention"})
    assert task["version"] == 1
    assert task["advisory_prerequisites"] == ["rmsnorm", "attention"]
    assert [hint["level"] for hint in task["hints"]] == [1, 2]
    assert [hint["kind"] for hint in task["hints"]] == ["questions", "analysis"]
    assert len(task["model_connections"]) >= 2
    assert task["pro_con_analysis"]["pros"] and task["pro_con_analysis"]["cons"]
    assert any(source["kind"] == "code" for source in task["sources"])
    assert any(test.get("visibility") == "unshown" for test in task["tests"])


def test_qk_norm_reference_passes_and_all_named_mutations_are_rejected():
    result = assert_mutations_rejected("qk_norm", QK_NORM_MUTATIONS)
    assert set(result) == {mutation.name for mutation in QK_NORM_MUTATIONS}
    assert "gradient.flow" in result["detach_inputs"]
    assert "numerics.stability" in result["missing_eps"]


def test_hybrid_attention_metadata_has_curriculum_contract():
    task = get_task("hybrid_attention_schedule")
    assert task is not None
    validate_task(
        "hybrid_attention_schedule",
        task,
        known_ids={"hybrid_attention_schedule", "causal_attention", "sliding_window"},
    )
    assert task["advisory_prerequisites"] == ["causal_attention", "sliding_window"]
    assert [hint["kind"] for hint in task["hints"]] == ["questions", "analysis"]
    assert {"Gemma 3", "Qwen3", "Mistral"} <= {
        model
        for connection in task["model_connections"]
        for model in ("Gemma 3", "Qwen3", "Mistral")
        if model in connection
    }
    assert any(test["name"] == "global_every=1 makes every layer global" for test in task["tests"])


def test_hybrid_attention_reference_passes_and_mutations_are_rejected():
    result = assert_mutations_rejected("hybrid_attention_schedule", HYBRID_MUTATIONS)
    assert set(result) == {mutation.name for mutation in HYBRID_MUTATIONS}
    assert "attention.masking" in result["future_visible"]
    assert "attention.masking" in result["off_by_one_schedule"]


def test_frontier_attention_block_metadata_has_all_model_connections():
    task = get_task("frontier_attention_block")
    assert task is not None
    validate_task(
        "frontier_attention_block",
        task,
        known_ids={"frontier_attention_block", "qk_norm", "gqa", "hybrid_attention_schedule", "mla"},
    )
    text = " ".join(task["model_connections"])
    for model in ("Kimi", "GLM", "Qwen", "Llama", "Gemma", "Mistral"):
        assert model in text
    assert [hint["kind"] for hint in task["hints"]] == ["questions", "analysis"]
    assert len([source for source in task["sources"] if source["kind"] == "code"]) >= 4


def test_frontier_attention_reference_passes_and_mutations_are_rejected():
    result = assert_mutations_rejected("frontier_attention_block", FRONTIER_MUTATIONS)
    assert set(result) == {mutation.name for mutation in FRONTIER_MUTATIONS}
    assert "gradient.flow" in result["detached_kv"]
    assert "attention.masking" in result["noncausal"]
