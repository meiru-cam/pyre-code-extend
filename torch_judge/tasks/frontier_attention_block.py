"""Configurable GQA/QK-norm/local-global attention capstone."""

TASK = {
    "title": "Build a Configurable Frontier Attention Block",
    "difficulty": "Hard",
    "version": 1,
    "function_name": "FrontierAttentionBlock",
    "description_en": r"""Compose grouped-query attention, optional QK normalization, and a causal local/global schedule in one PyTorch module.

**Constructor:** `FrontierAttentionBlock(d_model, num_heads, num_kv_heads, layer_index, window_size, global_every, use_qk_norm=True, eps=1e-6)`

**Forward:** `block(x) -> Tensor`, where `x` and the output have shape `(batch, sequence, d_model)`.

Create bias-free projections named `q_proj`, `k_proj`, `v_proj`, and `o_proj`. Q has `num_heads`; K/V have `num_kv_heads` and are repeated in contiguous groups. When enabled, RMS-normalize Q and K independently over `head_dim` before the dot product. Apply causal attention; every `global_every`-th one-based layer is global, while other layers see the current token and at most `window_size` predecessors.

Validate the divisibility and schedule arguments. Preserve autograd connectivity to the input and every projection.

This is a learning profile, not a universal reproduction. Llama demonstrates GQA; Mistral demonstrates bounded local attention; Gemma combines local/global schedules and QK norm; Qwen3 combines GQA, QK norm, and layer-selected masks. Kimi K2 and GLM-4.5 use MLA-style attention, a materially different KV-compression design connected to the advisory `mla` exercise rather than falsely squeezed into this class.

Omitted production concerns include RoPE, cache updates, flash/flex kernels, attention dropout, tensor parallelism, mixed-precision upcasts, and model-specific residual/normalization placement.""",
    "advisory_prerequisites": ["qk_norm", "gqa", "hybrid_attention_schedule", "mla"],
    "hints": [
        {
            "level": 1,
            "kind": "questions",
            "content": (
                "What are the Q and KV projection widths? How many times must each KV head be "
                "repeated? At which point should QK normalization happen? Which two inequalities "
                "define a local causal mask?"
            ),
        },
        {
            "level": 2,
            "kind": "analysis",
            "content": (
                "Project and reshape to (B, heads, S, head_dim); normalize Q/K per final "
                "dimension; repeat each KV head num_heads // num_kv_heads times; scale logits by "
                "head_dim**-0.5; apply the scheduled causal mask; combine V, restore (B,S,D), "
                "then use o_proj."
            ),
        },
    ],
    "model_connections": [
        "Llama attention uses fewer key/value heads than query heads and repeats KV groups for GQA.",
        "Mistral inference represents sliding-window state per layer to bound long-context work.",
        "Gemma 3 combines QK normalization with a repeating local-sliding/global schedule.",
        "Qwen3 attention combines GQA, head-dimensional QK normalization, and layer-selected masks.",
        "Kimi K2 reports MLA, so its compressed KV path is better studied through the separate MLA exercise.",
        "GLM-4.5 also belongs to the MLA/MoE branch; this capstone explicitly does not claim implementation equivalence.",
    ],
    "pro_con_analysis": {
        "pros": [
            "GQA reduces KV projection/cache size relative to full multi-head KV while retaining multiple KV groups.",
            "QK norm controls logit magnitude; local/global scheduling trades bounded work for periodic long-range communication.",
            "One explicit configuration makes architectural choices and their interactions testable.",
        ],
        "cons": [
            "KV repetition is pedagogical but can materialize memory that optimized grouped kernels avoid.",
            "Local layers delay distant information flow and introduce cache/kernel scheduling complexity.",
            "Normalization adds reductions and the block omits model-specific features needed for checkpoint compatibility.",
            "MLA models such as Kimi and GLM require compressed latent KV projections, not this GQA representation.",
        ],
    },
    "sources": [
        {
            "kind": "code",
            "url": "https://github.com/huggingface/transformers",
            "commit": "7ea2320c76117e6742364808a666ef6f2fb40a67",
            "path": "src/transformers/models/qwen3/modeling_qwen3.py",
            "symbol": "repeat_kv, eager_attention_forward, and Qwen3Attention",
            "license": "Apache-2.0",
            "adapted": "GQA projection/repetition, head-dimensional QK norm, scaling, and mask selection boundaries.",
            "simplifications": "No RoPE/cache/backend dispatch/dropout and no learned norm weights.",
        },
        {
            "kind": "code",
            "url": "https://github.com/huggingface/transformers",
            "commit": "7ea2320c76117e6742364808a666ef6f2fb40a67",
            "path": "src/transformers/models/llama/modeling_llama.py",
            "symbol": "repeat_kv, eager_attention_forward, and LlamaAttention",
            "license": "Apache-2.0",
            "adapted": "The separate query-head/KV-head contract and contiguous KV-group repetition.",
            "simplifications": "Single-process CPU-sized tensors without cache or rotary embeddings.",
        },
        {
            "kind": "code",
            "url": "https://github.com/google/gemma_pytorch",
            "commit": "014acb7ac4563a5f77c76d7ff98f31b568c16508",
            "path": "gemma/config.py",
            "symbol": "GemmaConfig, AttentionType, and get_config_for_27b_v3",
            "license": "Apache-2.0",
            "adapted": "QK-norm and repeating local/global attention profile configuration.",
            "simplifications": "Arithmetic schedule rather than complete per-layer config and model block.",
        },
        {
            "kind": "code",
            "url": "https://github.com/mistralai/mistral-inference",
            "commit": "9eaeb91c17450e09021b6065a1d5cc69876507c8",
            "path": "src/mistral_inference/transformer_layers.py",
            "symbol": "repeat_kv and Attention",
            "license": "Apache-2.0",
            "adapted": "Bias-free GQA projections and repeated KV heads used with bounded inference state.",
            "simplifications": "Materialized attention mask with no xFormers kernel or rolling cache.",
        },
        {
            "kind": "paper",
            "url": "https://arxiv.org/abs/2507.20534",
            "section": "2 Model Architecture (Kimi K2 technical report)",
        },
        {
            "kind": "paper",
            "url": "https://arxiv.org/abs/2508.06471",
            "section": "2 Architecture (GLM-4.5 technical report)",
        },
    ],
    "tests": [
        {
            "name": "Module and projection contracts",
            "behavior": "contract.signature",
            "code": r"""
import torch
import torch.nn as nn
block = {fn}(d_model=16, num_heads=4, num_kv_heads=2, layer_index=0, window_size=2, global_every=3)
assert isinstance(block, nn.Module)
assert isinstance(block.q_proj, nn.Linear) and block.q_proj.weight.shape == (16, 16) and block.q_proj.bias is None
assert isinstance(block.k_proj, nn.Linear) and block.k_proj.weight.shape == (8, 16) and block.k_proj.bias is None
assert isinstance(block.v_proj, nn.Linear) and block.v_proj.weight.shape == (8, 16) and block.v_proj.bias is None
assert isinstance(block.o_proj, nn.Linear) and block.o_proj.weight.shape == (16, 16) and block.o_proj.bias is None
assert block(torch.randn(2, 5, 16)).shape == (2, 5, 16)
""",
        },
        {
            "name": "Constructor rejects invalid profiles",
            "behavior": "edge.empty_or_boundary",
            "code": r"""
bad_configs = [
    dict(d_model=10, num_heads=4, num_kv_heads=2, layer_index=0, window_size=1, global_every=2),
    dict(d_model=12, num_heads=3, num_kv_heads=2, layer_index=0, window_size=1, global_every=2),
    dict(d_model=12, num_heads=3, num_kv_heads=1, layer_index=-1, window_size=1, global_every=2),
    dict(d_model=12, num_heads=3, num_kv_heads=1, layer_index=0, window_size=-1, global_every=2),
    dict(d_model=12, num_heads=3, num_kv_heads=1, layer_index=0, window_size=1, global_every=0),
]
for config in bad_configs:
    raised = False
    try:
        {fn}(**config)
    except ValueError:
        raised = True
    assert raised
""",
        },
        {
            "name": "Independent numerical oracle",
            "behavior": "state.invariant",
            "visibility": "unshown",
            "failure_message": "Projection, GQA repetition, QK norm, scaling, masking, or output projection differs from the oracle.",
            "code": r"""
import torch
torch.manual_seed(101)
block = {fn}(16, 4, 2, layer_index=0, window_size=2, global_every=4, use_qk_norm=True)
x = torch.randn(2, 5, 16, dtype=torch.float32)
oracle_x = x.clone()
q_weight = block.q_proj.weight.detach().clone()
k_weight = block.k_proj.weight.detach().clone()
v_weight = block.v_proj.weight.detach().clone()
o_weight = block.o_proj.weight.detach().clone()
actual = block(x.clone())
B, S, D = x.shape
H, KV, HD = 4, 2, 4
q = torch.nn.functional.linear(oracle_x, q_weight).view(B, S, H, HD).transpose(1, 2)
k = torch.nn.functional.linear(oracle_x, k_weight).view(B, S, KV, HD).transpose(1, 2)
v = torch.nn.functional.linear(oracle_x, v_weight).view(B, S, KV, HD).transpose(1, 2)
q = q / torch.sqrt(q.square().mean(-1, keepdim=True) + block.eps)
k = k / torch.sqrt(k.square().mean(-1, keepdim=True) + block.eps)
k = k.repeat_interleave(H // KV, dim=1)
v = v.repeat_interleave(H // KV, dim=1)
scores = q @ k.transpose(-2, -1) * (HD ** -0.5)
query = torch.arange(S).unsqueeze(1)
key = torch.arange(S).unsqueeze(0)
allowed = (key <= query) & (key >= query - 2)
scores = scores.masked_fill(~allowed.view(1, 1, S, S), float('-inf'))
context = (torch.softmax(scores, -1) @ v).transpose(1, 2).reshape(B, S, D)
expected = torch.nn.functional.linear(context, o_weight)
assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)
""",
        },
        {
            "name": "QK normalization toggle changes only the configured operation",
            "behavior": "state.invariant",
            "visibility": "unshown",
            "failure_message": "use_qk_norm=False must skip Q/K normalization without changing the rest of the block.",
            "code": r"""
import torch
torch.manual_seed(107)
normalized = {fn}(8, 2, 1, 1, 1, 2, use_qk_norm=True)
plain = {fn}(8, 2, 1, 1, 1, 2, use_qk_norm=False)
plain.load_state_dict(normalized.state_dict())
x = torch.randn(1, 4, 8) * 4
plain_weights = {name: parameter.detach().clone() for name, parameter in plain.named_parameters()}
out_norm = normalized(x.clone())
out_plain = plain(x.clone())
assert out_norm.shape == out_plain.shape
assert not torch.allclose(out_norm, out_plain, atol=1e-5)
# Exact independent oracle for the disabled branch (this layer is global).
B, S, D = x.shape
q = torch.nn.functional.linear(x, plain_weights['q_proj.weight']).view(B, S, 2, 4).transpose(1, 2)
k = torch.nn.functional.linear(x, plain_weights['k_proj.weight']).view(B, S, 1, 4).transpose(1, 2)
v = torch.nn.functional.linear(x, plain_weights['v_proj.weight']).view(B, S, 1, 4).transpose(1, 2)
k = k.repeat_interleave(2, dim=1)
v = v.repeat_interleave(2, dim=1)
scores = q @ k.transpose(-2, -1) * 0.5
positions = torch.arange(S)
causal = positions.unsqueeze(0) <= positions.unsqueeze(1)
scores = scores.masked_fill(~causal.view(1, 1, S, S), float('-inf'))
context = (torch.softmax(scores, dim=-1) @ v).transpose(1, 2).reshape(B, S, D)
expected_plain = torch.nn.functional.linear(context, plain_weights['o_proj.weight'])
assert torch.allclose(out_plain, expected_plain, atol=1e-6, rtol=1e-5)
""",
        },
        {
            "name": "Local and global layers enforce their receptive fields",
            "behavior": "attention.masking",
            "visibility": "unshown",
            "failure_message": "The block did not apply causal local/global masking at the configured one-based layer interval.",
            "code": r"""
import torch
torch.manual_seed(109)
local = {fn}(8, 2, 1, layer_index=0, window_size=1, global_every=2)
global_block = {fn}(8, 2, 1, layer_index=1, window_size=1, global_every=2)
global_block.load_state_dict(local.state_dict())
x = torch.randn(1, 5, 8)
changed = x.clone()
changed[:, 0] += 20
local_base, local_changed = local(x), local(changed)
global_base, global_changed = global_block(x), global_block(changed)
assert torch.allclose(local_base[:, 4], local_changed[:, 4], atol=1e-6)
assert not torch.allclose(global_base[:, 4], global_changed[:, 4], atol=1e-5)
# Causality: changing future positions cannot affect earlier outputs in either profile.
future = x.clone(); future[:, 3:] -= 30
assert torch.allclose(local(x)[:, :3], local(future)[:, :3], atol=1e-6)
assert torch.allclose(global_block(x)[:, :3], global_block(future)[:, :3], atol=1e-6)
""",
        },
        {
            "name": "Gradients reach input and every projection",
            "behavior": "gradient.flow",
            "visibility": "unshown",
            "failure_message": "The block must preserve finite nonzero gradients to x and all four projections.",
            "code": r"""
import torch
torch.manual_seed(113)
block = {fn}(12, 3, 1, 0, 2, 3)
x = torch.randn(2, 5, 12, requires_grad=True)
out = block(x)
weights = torch.arange(1, out.numel() + 1, dtype=out.dtype).reshape_as(out)
(out * weights).sum().backward()
assert x.grad is not None and torch.isfinite(x.grad).all() and torch.count_nonzero(x.grad)
for name in ('q_proj', 'k_proj', 'v_proj', 'o_proj'):
    grad = getattr(block, name).weight.grad
    assert grad is not None and torch.isfinite(grad).all() and torch.count_nonzero(grad), name
""",
        },
        {
            "name": "Seeded construction is deterministic",
            "behavior": "state.invariant",
            "visibility": "unshown",
            "failure_message": "The block introduced unconfigured randomness or state-dependent output.",
            "code": r"""
import torch
torch.manual_seed(127)
first = {fn}(8, 2, 1, 0, 1, 2)
torch.manual_seed(127)
second = {fn}(8, 2, 1, 0, 1, 2)
x = torch.randn(2, 4, 8)
assert torch.equal(first(x), second(x))
""",
        },
    ],
    "solution": r'''class FrontierAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        num_kv_heads,
        layer_index,
        window_size,
        global_every,
        use_qk_norm=True,
        eps=1e-6,
    ):
        super().__init__()
        if d_model <= 0 or num_heads <= 0 or num_kv_heads <= 0:
            raise ValueError("model and head sizes must be positive")
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        if layer_index < 0 or window_size < 0 or global_every < 1:
            raise ValueError("invalid layer schedule")

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_heads
        self.layer_index = layer_index
        self.window_size = window_size
        self.global_every = global_every
        self.use_qk_norm = use_qk_norm
        self.eps = eps
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        if x.ndim != 3 or x.shape[-1] != self.q_proj.in_features or x.shape[1] == 0:
            raise ValueError("x must have shape (batch, non-empty sequence, d_model)")
        batch, sequence, _ = x.shape
        query = self.q_proj(x).view(batch, sequence, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.k_proj(x).view(batch, sequence, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(x).view(batch, sequence, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if self.use_qk_norm:
            query = query * torch.rsqrt(query.square().mean(dim=-1, keepdim=True) + self.eps)
            key = key * torch.rsqrt(key.square().mean(dim=-1, keepdim=True) + self.eps)

        repeats = self.num_heads // self.num_kv_heads
        key = key.repeat_interleave(repeats, dim=1)
        value = value.repeat_interleave(repeats, dim=1)
        scores = torch.matmul(query, key.transpose(-2, -1)) * (self.head_dim ** -0.5)

        query_position = torch.arange(sequence, device=x.device).unsqueeze(1)
        key_position = torch.arange(sequence, device=x.device).unsqueeze(0)
        allowed = key_position <= query_position
        if (self.layer_index + 1) % self.global_every != 0:
            allowed = allowed & (key_position >= query_position - self.window_size)
        scores = scores.masked_fill(~allowed.view(1, 1, sequence, sequence), float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(weights, value).transpose(1, 2).contiguous().view(batch, sequence, -1)
        return self.o_proj(context)''',
    "demo": r"""torch.manual_seed(0)
block = FrontierAttentionBlock(32, 8, 2, layer_index=0, window_size=4, global_every=6)
x = torch.randn(2, 16, 32)
print(block(x).shape)
print('KV projection width:', block.k_proj.out_features, 'vs Q:', block.q_proj.out_features)""",
}
