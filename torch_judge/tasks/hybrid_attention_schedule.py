"""Causal local/global attention scheduling exercise."""

TASK = {
    "title": "Hybrid Local/Global Attention Schedule",
    "difficulty": "Medium",
    "version": 1,
    "function_name": "hybrid_attention_schedule",
    "description_en": r"""Implement causal scaled-dot-product attention whose receptive field changes by layer.

**Signature:** `hybrid_attention_schedule(Q, K, V, layer_index, window_size, global_every) -> Tensor`

`Q` and `K` have shape `(batch, sequence, head_dim)` and `V` has shape `(batch, sequence, value_dim)`; `value_dim` may differ from `head_dim`. Layer numbering in the API is zero-based, but the schedule is expressed in human one-based terms: a layer is global when `(layer_index + 1) % global_every == 0`. Global layers may attend to every current or earlier token. Other layers are local and may attend to the current token plus at most `window_size` earlier tokens. Future tokens are never visible.

Validate matching non-empty Q/K/V shapes, `layer_index >= 0`, `window_size >= 0`, and `global_every >= 1`. Preserve dtype/device and gradients.

This small function isolates a real architectural decision. Gemma 3 repeats local layers followed by a global layer; Qwen3 represents full/sliding choices as layer types; Mistral uses sliding windows to bound cache and attention work. Production schedules and kernels vary—the exercise teaches the mask and indexing semantics, not a claim that these models share one block.""",
    "advisory_prerequisites": ["causal_attention", "sliding_window"],
    "hints": [
        {
            "level": 1,
            "kind": "questions",
            "content": (
                "For query position i, which key positions are causal? On a local layer, what "
                "additional lower bound does window_size impose? How does zero-based layer_index "
                "map to every Nth layer?"
            ),
        },
        {
            "level": 2,
            "kind": "analysis",
            "content": (
                "Build query and key position grids. Start with key <= query. Unless "
                "(layer_index + 1) is divisible by global_every, also require "
                "key >= query - window_size. Mask disallowed logits before softmax."
            ),
        },
    ],
    "model_connections": [
        "Gemma 3 config uses five LOCAL_SLIDING layers followed by one GLOBAL layer for larger variants.",
        "Qwen3 attention selects a causal mask from configured full_attention or sliding_attention layer types.",
        "Mistral inference derives per-layer cache sizes from an integer or repeating sliding-window list.",
    ],
    "pro_con_analysis": {
        "pros": [
            "Local layers bound receptive-field work and KV-cache demand while global layers restore long-range paths.",
            "A declarative schedule lets model code reuse one attention interface with different masks.",
        ],
        "cons": [
            "Information may need several layers to travel between distant tokens.",
            "Mixed masks complicate cache management, kernel selection, and layer-index bookkeeping.",
            "The best local/global ratio and window depend on training distribution and context length.",
        ],
    },
    "sources": [
        {
            "kind": "code",
            "url": "https://github.com/google/gemma_pytorch",
            "commit": "014acb7ac4563a5f77c76d7ff98f31b568c16508",
            "path": "gemma/config.py",
            "symbol": "AttentionType and get_config_for_27b_v3",
            "license": "Apache-2.0",
            "adapted": "The repeating local-sliding/global layer schedule as an explicit configuration.",
            "simplifications": "A single-head tensor function with no RoPE, cache, soft-capping, or fused kernel.",
        },
        {
            "kind": "code",
            "url": "https://github.com/huggingface/transformers",
            "commit": "7ea2320c76117e6742364808a666ef6f2fb40a67",
            "path": "src/transformers/models/qwen3/modeling_qwen3.py",
            "symbol": "Qwen3Model.forward and Qwen3Attention.forward",
            "license": "Apache-2.0",
            "adapted": "Selecting a full or sliding causal mask from the current layer type.",
            "simplifications": "Uses arithmetic periodicity instead of a full layer_types configuration list.",
        },
        {
            "kind": "code",
            "url": "https://github.com/mistralai/mistral-inference",
            "commit": "9eaeb91c17450e09021b6065a1d5cc69876507c8",
            "path": "src/mistral_inference/cache.py",
            "symbol": "get_cache_sizes",
            "license": "Apache-2.0",
            "adapted": "The idea that a repeating per-layer sliding-window profile drives bounded state.",
            "simplifications": "No inference cache; masking is materialized for tiny CPU tensors.",
        },
        {
            "kind": "paper",
            "url": "https://arxiv.org/abs/2503.19786",
            "section": "2 Model Architecture",
        },
    ],
    "tests": [
        {
            "name": "Hand-calculated local causal averages",
            "behavior": "attention.masking",
            "code": r"""
import torch
Q = torch.zeros(1, 3, 1)
K = torch.zeros(1, 3, 1)
V = torch.tensor([[[1.0], [2.0], [4.0]]])
out = {fn}(Q, K, V, layer_index=0, window_size=1, global_every=2)
expected = torch.tensor([[[1.0], [1.5], [3.0]]])
assert torch.allclose(out, expected, atol=1e-6), (out, expected)
""",
        },
        {
            "name": "Every Nth layer is global using one-based numbering",
            "behavior": "attention.masking",
            "code": r"""
import torch
Q = torch.zeros(1, 4, 1)
K = torch.zeros(1, 4, 1)
V = torch.tensor([[[1.0], [2.0], [4.0], [8.0]]])
out = {fn}(Q, K, V, layer_index=1, window_size=0, global_every=2)
expected = torch.tensor([[[1.0], [1.5], [7.0 / 3.0], [3.75]]])
assert torch.allclose(out, expected, atol=1e-6), (out, expected)
""",
        },
        {
            "name": "global_every=1 makes every layer global",
            "behavior": "edge.empty_or_boundary",
            "visibility": "unshown",
            "failure_message": "global_every=1 must select global causal attention for every layer index.",
            "code": r"""
import torch
Q = torch.zeros(1, 3, 1)
K = torch.zeros(1, 3, 1)
V = torch.tensor([[[2.0], [4.0], [8.0]]])
expected = torch.tensor([[[2.0], [3.0], [14.0 / 3.0]]])
for layer_index in (0, 1, 7):
    actual = {fn}(Q, K, V, layer_index, window_size=0, global_every=1)
    assert torch.allclose(actual, expected, atol=1e-6)
""",
        },
        {
            "name": "Preserves output structure",
            "behavior": "tensor.dtype_device",
            "code": r"""
import torch
Q = torch.randn(2, 5, 4, dtype=torch.float64)
K = torch.randn(2, 5, 4, dtype=torch.float64)
V = torch.randn(2, 5, 6, dtype=torch.float64)
out = {fn}(Q, K, V, 0, 2, 3)
assert out.shape == V.shape and out.dtype == V.dtype and out.device == V.device
""",
        },
        {
            "name": "Seeded differential mask oracle",
            "behavior": "attention.masking",
            "visibility": "unshown",
            "failure_message": "The causal local/global mask or scaled-dot-product computation is incorrect.",
            "code": r"""
import torch, math
def oracle(Q, K, V, layer_index, window_size, global_every):
    scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(Q.shape[-1])
    sequence = Q.shape[1]
    query = torch.arange(sequence, device=Q.device).unsqueeze(1)
    key = torch.arange(sequence, device=Q.device).unsqueeze(0)
    allowed = key <= query
    if (layer_index + 1) % global_every != 0:
        allowed = allowed & (key >= query - window_size)
    return torch.bmm(torch.softmax(scores.masked_fill(~allowed.unsqueeze(0), float('-inf')), dim=-1), V)
for seed, shape, layer, window, every in [
    (3, (2, 5, 4), 0, 0, 3),
    (7, (1, 6, 3), 2, 2, 4),
    (11, (2, 4, 5), 3, 1, 2),
]:
    torch.manual_seed(seed)
    Q = torch.randn(*shape, dtype=torch.float64)
    K = torch.randn(*shape, dtype=torch.float64)
    V = torch.randn(shape[0], shape[1], 6, dtype=torch.float64)
    actual = {fn}(Q.clone(), K.clone(), V.clone(), layer, window, every)
    expected = oracle(Q, K, V, layer, window, every)
    assert torch.allclose(actual, expected, atol=1e-9, rtol=1e-7)
""",
        },
        {
            "name": "Future and distant tokens cannot leak into a local output",
            "behavior": "attention.masking",
            "visibility": "unshown",
            "failure_message": "A local causal layer allowed a future or out-of-window token to affect its output.",
            "code": r"""
import torch
torch.manual_seed(17)
Q = torch.randn(1, 6, 4)
K = torch.randn(1, 6, 4)
V = torch.randn(1, 6, 3)
base = {fn}(Q, K, V, 0, 1, 4)
K_changed, V_changed = K.clone(), V.clone()
K_changed[:, 3:] += 1000
V_changed[:, 3:] += 1000
changed = {fn}(Q, K_changed, V_changed, 0, 1, 4)
assert torch.allclose(base[:, :3], changed[:, :3], atol=1e-6)
""",
        },
        {
            "name": "Boundary arguments are validated",
            "behavior": "edge.empty_or_boundary",
            "visibility": "unshown",
            "failure_message": "Reject negative indices/windows, non-positive global_every, and malformed Q/K/V shapes.",
            "code": r"""
import torch
Q = torch.randn(1, 2, 3)
K = torch.randn(1, 2, 3)
V = torch.randn(1, 2, 4)
bad_calls = [
    (Q, K, V, -1, 1, 2),
    (Q, K, V, 0, -1, 2),
    (Q, K, V, 0, 1, 0),
    (Q, K[:, :1], V, 0, 1, 2),
    (Q[:0], K[:0], V[:0], 0, 1, 2),
    (Q[..., :0], K[..., :0], V, 0, 1, 2),
    (Q, K, V[..., :0], 0, 1, 2),
]
for args in bad_calls:
    raised = False
    try:
        {fn}(*args)
    except ValueError:
        raised = True
    assert raised
""",
        },
        {
            "name": "Gradients reach Q K and V",
            "behavior": "gradient.flow",
            "visibility": "unshown",
            "failure_message": "Attention output must retain finite gradients to Q, K, and V.",
            "code": r"""
import torch
torch.manual_seed(23)
Q = torch.randn(2, 4, 3, requires_grad=True)
K = torch.randn(2, 4, 3, requires_grad=True)
V = torch.randn(2, 4, 5, requires_grad=True)
out = {fn}(Q, K, V, 0, 2, 3)
weights = torch.arange(1, out.numel() + 1, dtype=out.dtype).reshape_as(out)
(out * weights).sum().backward()
for tensor in (Q, K, V):
    assert tensor.grad is not None and torch.isfinite(tensor.grad).all() and torch.count_nonzero(tensor.grad)
""",
        },
    ],
    "solution": r'''def hybrid_attention_schedule(Q, K, V, layer_index, window_size, global_every):
    if Q.ndim != 3 or K.ndim != 3 or V.ndim != 3:
        raise ValueError("Q, K, and V must be rank-3 tensors")
    if Q.shape[:2] != K.shape[:2] or Q.shape[:2] != V.shape[:2] or Q.shape[-1] != K.shape[-1]:
        raise ValueError("Q, K, and V batch/sequence shapes and Q/K head dimensions must match")
    if Q.shape[0] == 0 or Q.shape[1] == 0 or Q.shape[-1] == 0 or V.shape[-1] == 0:
        raise ValueError("batch, sequence, head_dim, and value_dim must be non-empty")
    if layer_index < 0 or window_size < 0 or global_every < 1:
        raise ValueError("layer_index/window_size must be non-negative and global_every positive")

    scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(Q.shape[-1])
    sequence = Q.shape[1]
    query_position = torch.arange(sequence, device=Q.device).unsqueeze(1)
    key_position = torch.arange(sequence, device=Q.device).unsqueeze(0)
    allowed = key_position <= query_position
    if (layer_index + 1) % global_every != 0:
        allowed = allowed & (key_position >= query_position - window_size)
    scores = scores.masked_fill(~allowed.unsqueeze(0), float("-inf"))
    return torch.bmm(torch.softmax(scores, dim=-1), V)''',
    "demo": r"""Q = torch.zeros(1, 4, 1)
K = torch.zeros(1, 4, 1)
V = torch.arange(1, 5, dtype=torch.float32).view(1, 4, 1)
print('local:', hybrid_attention_schedule(Q, K, V, 0, 1, 2).flatten())
print('global:', hybrid_attention_schedule(Q, K, V, 1, 1, 2).flatten())""",
}
