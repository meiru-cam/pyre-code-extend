"""Per-head query/key RMS normalization exercise."""

TASK = {
    "title": "QK Normalization for Stable Attention",
    "difficulty": "Medium",
    "version": 1,
    "function_name": "qk_norm",
    "description_en": r"""Implement per-head RMS normalization for queries and keys.

**Signature:** `qk_norm(q, k, eps=1e-6) -> (q_normalized, k_normalized)`

`q` and `k` may have any floating-point shape whose final dimension is the head dimension. Normalize each Q vector and each K vector **independently** over that final dimension:

```
rmsnorm(x) = x * (mean(x**2, dim=-1) + eps) ** -0.5
```

Preserve shape, dtype, device, and autograd connectivity. `eps` must keep an all-zero vector finite. This exercise uses no learned scale; production blocks often attach a learned RMSNorm weight.

Why it matters: query/key magnitudes directly scale attention logits. QK normalization bounds that source of logit growth and can improve training stability, but it adds normalization work and changes the model's parameterization.

**Optional source connections:** Qwen3 applies learned head-dimensional Q/K RMSNorm immediately after projection. Gemma 3 enables QK norm in its attention configuration. These references are additional context—you can implement the contract without reading them.""",
    "advisory_prerequisites": ["rmsnorm", "attention"],
    "hints": [
        {
            "level": 1,
            "kind": "questions",
            "content": (
                "Which axis is the head-feature axis? Should Q and K share statistics? "
                "What happens to a zero vector if epsilon is omitted?"
            ),
        },
        {
            "level": 2,
            "kind": "analysis",
            "content": (
                "For each tensor separately, keep the last dimension while computing the mean "
                "of squared values, add eps, multiply by the reciprocal square root, and avoid "
                "detach or constructing a new tensor from the result."
            ),
        },
    ],
    "model_connections": [
        (
            "Qwen3: Qwen3Attention owns q_norm and k_norm over head_dim and applies them to "
            "the projected Q/K states before RoPE."
        ),
        (
            "Gemma 3: the official PyTorch configuration enables use_qk_norm together with "
            "a hybrid local/global attention schedule."
        ),
    ],
    "pro_con_analysis": {
        "pros": [
            "Controls query/key magnitude before the dot product and improves logit stability.",
            "Acts per head and per token, so it does not mix sequence positions or attention heads.",
        ],
        "cons": [
            "Adds two reductions and rescalings per attention layer.",
            "Changes optimization dynamics and may be redundant with other logit controls.",
            "Production implementations need careful mixed-precision and fused-kernel handling.",
        ],
    },
    "sources": [
        {
            "kind": "code",
            "url": "https://github.com/huggingface/transformers",
            "commit": "7ea2320c76117e6742364808a666ef6f2fb40a67",
            "path": "src/transformers/models/qwen3/modeling_qwen3.py",
            "symbol": "Qwen3Attention.__init__ and Qwen3Attention.forward",
            "license": "Apache-2.0",
            "adapted": "The placement of independent head-dimensional Q and K RMS normalization.",
            "simplifications": "No learned scale, RoPE, cache, dropout, or optimized attention kernel.",
        },
        {
            "kind": "code",
            "url": "https://github.com/google/gemma_pytorch",
            "commit": "014acb7ac4563a5f77c76d7ff98f31b568c16508",
            "path": "gemma/config.py",
            "symbol": "GemmaConfig.use_qk_norm and get_config_for_27b_v3",
            "license": "Apache-2.0",
            "adapted": "The explicit QK-normalization architecture option and its Gemma 3 use.",
            "simplifications": "A standalone tensor operation rather than a full Gemma attention layer.",
        },
        {
            "kind": "paper",
            "url": "https://arxiv.org/abs/2505.09388",
            "section": "2.1 Model Architecture",
        },
        {
            "kind": "paper",
            "url": "https://arxiv.org/abs/2503.19786",
            "section": "2 Model Architecture",
        },
    ],
    "tests": [
        {
            "name": "Hand-calculated RMS normalization",
            "behavior": "state.invariant",
            "code": r"""
import torch
q = torch.tensor([[[3.0, 4.0]]])
k = torch.tensor([[[1.0, -1.0]]])
q_out, k_out = {fn}(q, k, eps=0.0)
q_expected = q / torch.sqrt(q.square().mean(dim=-1, keepdim=True))
k_expected = k / torch.sqrt(k.square().mean(dim=-1, keepdim=True))
assert torch.allclose(q_out, q_expected, atol=1e-6)
assert torch.allclose(k_out, k_expected, atol=1e-6)
""",
        },
        {
            "name": "Preserves shape dtype and device",
            "behavior": "tensor.dtype_device",
            "code": r"""
import torch
q = torch.randn(2, 3, 4, 5, dtype=torch.float64)
k = torch.randn(2, 1, 4, 5, dtype=torch.float64)
q_out, k_out = {fn}(q, k)
assert q_out.shape == q.shape and k_out.shape == k.shape
assert q_out.dtype == q.dtype and k_out.dtype == k.dtype
assert q_out.device == q.device and k_out.device == k.device
""",
        },
        {
            "name": "Independent PyTorch vector-norm differential",
            "behavior": "state.invariant",
            "visibility": "unshown",
            "failure_message": "Q and K must independently match RMSNorm over the final dimension.",
            "code": r"""
import torch
import math
for seed, shape in [(7, (2, 3, 5)), (19, (1, 4, 3, 7)), (31, (6, 2))]:
    torch.manual_seed(seed)
    q = torch.randn(*shape, dtype=torch.float64)
    k = torch.randn(*shape, dtype=torch.float64) * 3.0 + 0.5
    eps = 3e-6
    q_out, k_out = {fn}(q.clone(), k.clone(), eps)
    q_rms = torch.linalg.vector_norm(q, dim=-1, keepdim=True) / math.sqrt(shape[-1])
    k_rms = torch.linalg.vector_norm(k, dim=-1, keepdim=True) / math.sqrt(shape[-1])
    q_ref = q / torch.sqrt(q_rms.square() + eps)
    k_ref = k / torch.sqrt(k_rms.square() + eps)
    assert torch.allclose(q_out, q_ref, atol=1e-9, rtol=1e-7)
    assert torch.allclose(k_out, k_ref, atol=1e-9, rtol=1e-7)
""",
        },
        {
            "name": "Zero vectors stay finite",
            "behavior": "numerics.stability",
            "visibility": "unshown",
            "failure_message": "Epsilon must keep zero-valued Q and K vectors finite.",
            "code": r"""
import torch
q = torch.zeros(2, 3, 4)
k = torch.zeros(2, 1, 4)
q_out, k_out = {fn}(q, k)
assert torch.isfinite(q_out).all() and torch.isfinite(k_out).all()
assert torch.equal(q_out, q) and torch.equal(k_out, k)
""",
        },
        {
            "name": "Large fp16 vectors avoid reduction overflow",
            "behavior": "numerics.stability",
            "visibility": "unshown",
            "failure_message": "RMS statistics must be accumulated safely for valid large fp16 values.",
            "code": r"""
import torch
q = torch.tensor([[[400.0, 400.0, 400.0, 400.0]]], dtype=torch.float16)
k = torch.tensor([[[400.0, -400.0, 200.0, -200.0]]], dtype=torch.float16)
q_out, k_out = {fn}(q, k)
def safe_oracle(x):
    work = x.float()
    return (work * torch.rsqrt(work.square().mean(-1, keepdim=True) + 1e-6)).to(x.dtype)
assert torch.isfinite(q_out).all() and torch.isfinite(k_out).all()
assert torch.allclose(q_out, safe_oracle(q), atol=2e-3, rtol=2e-3)
assert torch.allclose(k_out, safe_oracle(k), atol=2e-3, rtol=2e-3)
""",
        },
        {
            "name": "Q and K use independent statistics",
            "behavior": "state.invariant",
            "visibility": "unshown",
            "failure_message": "Query and key vectors must not share normalization statistics.",
            "code": r"""
import torch
q = torch.tensor([[[1.0, 1.0, 1.0, 1.0]]])
k = torch.tensor([[[10.0, 0.0, 0.0, 0.0]]])
q_out, k_out = {fn}(q, k, 0.0)
assert torch.allclose(q_out, torch.ones_like(q))
assert torch.allclose(k_out, torch.tensor([[[2.0, 0.0, 0.0, 0.0]]]))
""",
        },
        {
            "name": "Positive scale metamorphism",
            "behavior": "state.invariant",
            "visibility": "unshown",
            "failure_message": "With eps=0, positive rescaling of a nonzero vector must not change its normalized value.",
            "code": r"""
import torch
torch.manual_seed(41)
q = torch.randn(2, 3, 5, dtype=torch.float64) + 0.25
k = torch.randn(2, 2, 5, dtype=torch.float64) - 0.5
q1, k1 = {fn}(q, k, 0.0)
q2, k2 = {fn}(q * 7.0, k * 0.25, 0.0)
assert torch.allclose(q1, q2, atol=1e-10, rtol=1e-8)
assert torch.allclose(k1, k2, atol=1e-10, rtol=1e-8)
""",
        },
        {
            "name": "Gradients reach Q and K",
            "behavior": "gradient.flow",
            "visibility": "unshown",
            "failure_message": "The normalized outputs must retain finite, nonzero gradients to both inputs.",
            "code": r"""
import torch
torch.manual_seed(53)
q = torch.randn(2, 3, 4, requires_grad=True)
k = torch.randn(2, 2, 4, requires_grad=True)
q_out, k_out = {fn}(q, k)
q_weights = torch.arange(1, q_out.numel() + 1, dtype=q.dtype).reshape_as(q_out)
k_weights = torch.arange(1, k_out.numel() + 1, dtype=k.dtype).reshape_as(k_out)
loss = (q_out * q_weights).sum() + (k_out * k_weights).sum()
loss.backward()
assert q.grad is not None and torch.isfinite(q.grad).all() and torch.count_nonzero(q.grad)
assert k.grad is not None and torch.isfinite(k.grad).all() and torch.count_nonzero(k.grad)
""",
        },
    ],
    "solution": r'''def qk_norm(q, k, eps=1e-6):
    def normalize(x):
        work = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        inverse_rms = torch.rsqrt(work.square().mean(dim=-1, keepdim=True) + eps)
        return (work * inverse_rms).to(x.dtype)

    return normalize(q), normalize(k)''',
    "demo": r"""torch.manual_seed(0)
q = torch.randn(1, 2, 3, 4)
k = torch.randn(1, 1, 3, 4)
q_normalized, k_normalized = qk_norm(q, k)
print(q_normalized.square().mean(dim=-1))
print(k_normalized.square().mean(dim=-1))""",
}
