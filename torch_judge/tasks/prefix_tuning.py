"""Prefix tuning: trainable per-layer key/value prefixes in front of frozen causal attention."""

from ._interview import interview

TASK = {
    "title": "Prefix Tuning (Trainable KV Prefix)",
    "difficulty": "Hard",
    "version": 1,
    "function_name": "PrefixAttention",
    "description_en": r"""Add trainable key and value prefixes to a frozen causal multi-head self-attention layer, as prefix tuning and P-tuning v2 do.

**Signature:** `PrefixAttention(d_model, num_heads, num_prefix)` (nn.Module)

**Attributes:**
- `q_proj`, `k_proj`, `v_proj`, `o_proj` — `nn.Linear(d_model, d_model)` each, weights and biases frozen.
- `prefix_k`, `prefix_v` — parameters `(num_heads, num_prefix, head_dim)`, where `head_dim = d_model // num_heads`. Small random init.

**Forward:** `forward(x, attention_mask=None) -> Tensor`
- `x` — `(B, T, d_model)`.
- `attention_mask` — optional `(B, T)` bool or 0/1 tensor, true for real tokens. `None` means every token is real.
- Returns `(B, T, d_model)`.

**Attention rule:**
- Per head, keys are the `num_prefix` prefix keys followed by the `T` projected keys. Values likewise.
- Every query attends to every prefix slot, regardless of its position or the padding mask.
- Query `i` attends to token key `j` only if `j <= i` and token `j` is real.
- Scores use scale `1 / sqrt(head_dim)`. Apply `o_proj` to the concatenated heads.

**Constraints:**
- Use `prefix_k` and `prefix_v` directly as keys and values. Do not pass them through `k_proj` or `v_proj`.
- One prefix is shared by every sequence in the batch.
- Only `prefix_k` and `prefix_v` receive gradients.
- Raise `ValueError` if `d_model` is not divisible by `num_heads` or `num_prefix < 1`.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**What prefix tuning trains.** Li and Liang (2021) freeze the whole model and learn a few virtual tokens per layer. Unlike prompt tuning, which only adds embeddings at the input, the prefix lives in every layer's keys and values, so it steers every attention layer directly. P-tuning v2 applies the same idea to understanding tasks.

**How PEFT runs it.** HuggingFace PEFT stores the prefix as a tensor shaped `(num_virtual_tokens, num_layers * 2 * hidden)`, reshapes it into one `(batch, heads, num_virtual_tokens, head_dim)` key and value pair per layer, and passes them to the model as `past_key_values`. The model then treats the prefix exactly like a KV cache from earlier tokens. PEFT also prepends `num_virtual_tokens` ones to the attention mask.

**The causal-mask trap.** Building a lower-triangular mask over the full `num_prefix + T` key axis is wrong: it hides later prefix slots from early queries. The prefix acts as a past that every query has already seen, so the causal mask applies to the real-token keys only.

**Reparameterization.** Optimizing the prefix directly can be unstable, so the paper trains it through a small MLP from a smaller embedding and keeps only the MLP output after training. That is PEFT's `prefix_projection=True`. This exercise trains the prefix directly.""",
    "advisory_prerequisites": ["causal_attention", "mha", "kv_cache"],
    "hints": [
        {"level": 1, "kind": "questions", "content": "After concatenation the key axis has num_prefix + T slots: which of them may query i see? How do you turn the (B, T) padding mask and a (T, T) causal mask into one (B, 1, T, num_prefix + T) mask? How do you give a (heads, P, head_dim) prefix a batch dimension without copying it per sequence?"},
        {"level": 2, "kind": "analysis", "content": "Project x to q, k, v and reshape to (B, H, T, hd). Expand prefix_k and prefix_v to (B, H, P, hd) and concatenate before k and v on dim 2. Build token_ok = tril(ones(T, T)) combined with the padding mask on the key axis, then prepend an all-True (B, 1, T, P) block. Softmax the masked scores, multiply by v, merge heads and apply o_proj."},
    ],
    "model_connections": [
        "HuggingFace PEFT's PrefixTuningConfig feeds the prefix to transformers models as past_key_values.",
        "P-tuning v2 applies deep prompts to every layer for NLU tasks such as sequence labeling.",
    ],
    "pro_con_analysis": {
        "pros": ["Trains 0.1% of the parameters or less and stores one small prefix per task.", "Mixed-task batches can use a different prefix per row with one frozen model."],
        "cons": ["Each prefix slot takes attention compute and KV cache memory at inference, and it cannot be merged away like LoRA.", "Training is sensitive to initialization and learning rate, which is why the MLP reparameterization exists."],
    },
    "sources": [
        {
            "kind": "code",
            "url": "https://github.com/huggingface/peft",
            "commit": "b8674c86183a5dee38d0c3ede392e189593025e5",
            "path": "src/peft/peft_model.py",
            "symbol": "PeftModel.get_prompt, PeftModelForCausalLM.forward",
            "license": "Apache-2.0",
            "adapted": "The prefix reshaped into per-layer (heads, num_virtual_tokens, head_dim) key and value tensors shared "
                       "across the batch, and an all-ones prefix block prepended to the attention mask.",
            "simplifications": "One attention layer instead of num_layers, the prefix trained directly rather than through the "
                               "PrefixEncoder MLP, and no positional encoding.",
        },
        {
            "kind": "paper",
            "url": "https://arxiv.org/abs/2101.00190",
            "section": "Section 4.1 (Prefix-tuning method) and Section 4.3 (Parametrization of P)",
        },
    ],
    "tests": [
        {"name": "Output shape and trainable parameters", "behavior": "contract.signature", "code": r"""
import torch
torch.manual_seed(0)
layer = {fn}(d_model=8, num_heads=2, num_prefix=3)
assert layer.prefix_k.shape == (2, 3, 4) and layer.prefix_v.shape == (2, 3, 4)
trainable = {name for name, p in layer.named_parameters() if p.requires_grad}
assert trainable == {'prefix_k', 'prefix_v'}, trainable
out = layer(torch.randn(2, 5, 8))
assert out.shape == (2, 5, 8), out.shape
"""},
        {"name": "Matches a reference attention on seeded inputs", "visibility": "unshown", "behavior": "attention.masking", "failure_message": "Every query must see all prefix slots plus real token keys at positions <= its own, scaled by 1 / sqrt(head_dim).", "code": r"""
import math, torch
for seed in (1, 8, 29):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    H = int(torch.randint(1, 4, (1,), generator=g)); hd = int(torch.randint(2, 5, (1,), generator=g)); D = H * hd
    P = int(torch.randint(1, 5, (1,), generator=g)); B = 3; T = int(torch.randint(3, 8, (1,), generator=g))
    layer = {fn}(D, H, P).double()
    with torch.no_grad():
        layer.prefix_k.copy_(torch.randn(H, P, hd, generator=g, dtype=torch.float64))
        layer.prefix_v.copy_(torch.randn(H, P, hd, generator=g, dtype=torch.float64))
    x = torch.randn(B, T, D, generator=g, dtype=torch.float64)
    mask = torch.rand(B, T, generator=g) > 0.3
    mask[:, 0] = True
    mask[1, :2] = False
    out = layer(x, attention_mask=mask)
    q = layer.q_proj(x).view(B, T, H, hd); k = layer.k_proj(x).view(B, T, H, hd); v = layer.v_proj(x).view(B, T, H, hd)
    want = torch.zeros(B, T, H, hd, dtype=torch.float64)
    for b in range(B):
        for h in range(H):
            for i in range(T):
                keys = [layer.prefix_k[h, p] for p in range(P)] + [k[b, j, h] for j in range(T) if j <= i and mask[b, j]]
                vals = [layer.prefix_v[h, p] for p in range(P)] + [v[b, j, h] for j in range(T) if j <= i and mask[b, j]]
                scores = torch.stack([q[b, i, h] @ key for key in keys]) / math.sqrt(hd)
                want[b, i, h] = torch.softmax(scores, dim=0) @ torch.stack(vals)
    want = layer.o_proj(want.reshape(B, T, D))
    assert torch.allclose(out, want, atol=1e-9), (seed, (out - want).abs().max())
    none_out = layer(x)
    full = layer(x, attention_mask=torch.ones(B, T, dtype=torch.long))
    assert torch.allclose(none_out, full, atol=1e-12), 'attention_mask=None must mean all tokens are real'
"""},
        {"name": "Early queries see the whole prefix", "visibility": "unshown", "behavior": "attention.masking", "failure_message": "The causal mask applies to token keys only; query 0 must still attend to every prefix slot.", "code": r"""
import torch
torch.manual_seed(4)
layer = {fn}(d_model=4, num_heads=1, num_prefix=3).double()
with torch.no_grad():
    for proj in (layer.q_proj, layer.k_proj, layer.v_proj, layer.o_proj):
        proj.weight.zero_(); proj.bias.zero_()
    layer.o_proj.weight.copy_(torch.eye(4))
    layer.prefix_k.zero_()
    layer.prefix_v.copy_(torch.tensor([[[3., 0, 0, 0], [0, 3., 0, 0], [0, 0, 3., 0]]], dtype=torch.float64))
x = torch.randn(1, 4, 4, dtype=torch.float64)
out = layer(x)
# all scores are 0: query i averages 3 prefix values and i + 1 zero token values
for i in range(4):
    want = torch.tensor([3., 3., 3., 0], dtype=torch.float64) / (3 + i + 1)
    assert torch.allclose(out[0, i], want, atol=1e-12), (i, out[0, i], want)
"""},
        {"name": "Gradients reach only the prefix", "visibility": "unshown", "behavior": "gradient.flow", "failure_message": "Use prefix_k and prefix_v directly as keys and values, shared across the batch; keep all projections frozen.", "code": r"""
import torch
for seed in (6, 15, 33):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    layer = {fn}(6, 2, 2).double()
    with torch.no_grad():
        layer.prefix_k.copy_(torch.randn(2, 2, 3, generator=g, dtype=torch.float64))
        layer.prefix_v.copy_(torch.randn(2, 2, 3, generator=g, dtype=torch.float64))
    x = torch.randn(2, 4, 6, generator=g, dtype=torch.float64)
    mask = torch.tensor([[1, 1, 1, 1], [0, 1, 1, 1]])
    upstream = torch.randn(2, 4, 6, generator=g, dtype=torch.float64)
    (layer(x, attention_mask=mask) * upstream).sum().backward()
    eps = 1e-6
    for name, param in (('prefix_k', layer.prefix_k), ('prefix_v', layer.prefix_v)):
        base = param.detach().clone()
        for idx in [(0, 0, 0), (1, 1, 2), (0, 1, 1)]:
            with torch.no_grad():
                param.copy_(base); param[idx] += eps
            plus = (layer(x, attention_mask=mask) * upstream).sum()
            with torch.no_grad():
                param.copy_(base); param[idx] -= eps
            minus = (layer(x, attention_mask=mask) * upstream).sum()
            with torch.no_grad():
                param.copy_(base)
            numeric = (plus - minus) / (2 * eps)
            assert abs(param.grad[idx].item() - numeric.item()) < 1e-6, (seed, name, idx, param.grad[idx].item(), numeric.item())
    for proj in (layer.q_proj, layer.k_proj, layer.v_proj, layer.o_proj):
        assert proj.weight.grad is None and proj.bias.grad is None
"""},
        {"name": "Rejects invalid configurations", "visibility": "unshown", "behavior": "contract.signature", "failure_message": "Raise ValueError when d_model % num_heads != 0 or num_prefix < 1.", "code": r"""
for args in ((10, 3, 2), (8, 2, 0), (8, 2, -1)):
    try:
        {fn}(*args)
    except ValueError:
        pass
    else:
        raise AssertionError(f"accepted {args}")
"""},
    ],
    "solution": '''import math
import torch
import torch.nn as nn

class PrefixAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_prefix):
        super().__init__()
        if d_model % num_heads != 0 or num_prefix < 1:
            raise ValueError("need d_model divisible by num_heads and num_prefix >= 1")
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        for proj in (self.q_proj, self.k_proj, self.v_proj, self.o_proj):
            proj.requires_grad_(False)
        self.prefix_k = nn.Parameter(torch.randn(num_heads, num_prefix, self.head_dim) * 0.02)
        self.prefix_v = nn.Parameter(torch.randn(num_heads, num_prefix, self.head_dim) * 0.02)

    def forward(self, x, attention_mask=None):
        B, T, D = x.shape
        H, hd = self.num_heads, self.head_dim
        P = self.prefix_k.shape[1]
        q = self.q_proj(x).view(B, T, H, hd).transpose(1, 2)
        k = self.k_proj(x).view(B, T, H, hd).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, hd).transpose(1, 2)
        k = torch.cat([self.prefix_k.expand(B, H, P, hd), k], dim=2)
        v = torch.cat([self.prefix_v.expand(B, H, P, hd), v], dim=2)

        if attention_mask is None:
            attention_mask = torch.ones(B, T, dtype=torch.bool, device=x.device)
        causal = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
        token_ok = causal[None, :, :] & attention_mask.bool()[:, None, :]
        prefix_ok = torch.ones(B, T, P, dtype=torch.bool, device=x.device)
        allowed = torch.cat([prefix_ok, token_ok], dim=2)[:, None]

        scores = q @ k.transpose(-2, -1) / math.sqrt(hd)
        scores = scores.masked_fill(~allowed, float("-inf"))
        out = torch.softmax(scores, dim=-1) @ v
        return self.o_proj(out.transpose(1, 2).reshape(B, T, D))
''',
    "interview_questions": interview(
        concept=[
            "What is prefix tuning, and how does it differ from prompt tuning?",
            "What exactly is trained, and what shape does the prefix have per layer?",
        ],
        deep_dive=[
            "Why are the prefix keys and values not passed through the frozen k and v projections?",
            "How do you build the attention mask once the prefix is prepended, including causality and padding?",
            "Why can a naive lower-triangular mask over the full key axis give wrong results?",
            "How does HuggingFace PEFT hand the prefix to a transformers model, and why does that reuse the KV-cache path?",
        ],
        tradeoffs=[
            "Prefix tuning versus LoRA: trainable parameters, inference cost and mergeability?",
            "Why does the original paper reparameterize the prefix through an MLP, and when can you drop it?",
        ],
    ),
}
