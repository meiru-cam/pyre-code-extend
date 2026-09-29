"""DoRA: weight-decomposed low-rank adaptation for a linear layer."""

from ._interview import interview

TASK = {
    "title": "DoRA (Weight-Decomposed LoRA)",
    "difficulty": "Medium",
    "version": 1,
    "function_name": "DoRALinear",
    "description_en": r"""Implement a DoRA linear layer: LoRA updates the direction of each output row of the weight, and a trainable magnitude vector sets its length.

**Signature:** `DoRALinear(in_features, out_features, rank, alpha=1.0)` (nn.Module)

**Attributes:**
- `linear` — `nn.Linear(in_features, out_features)`, weight and bias frozen.
- `lora_A` — parameter `(rank, in_features)`, small random init.
- `lora_B` — parameter `(out_features, rank)`, initialized to zeros.
- `magnitude` — parameter `(out_features,)`, initialized to the L2 norm of each row of `linear.weight`.
- `scaling` — float, `alpha / rank`.

**Forward:** `forward(x) -> Tensor`, `x` of shape `(*, in_features)`.
- `W_adapted = linear.weight + scaling * lora_B @ lora_A`, shape `(out_features, in_features)`.
- `norm` = L2 norm of each row of `W_adapted`, shape `(out_features,)`, detached from autograd.
- `W_dora = (magnitude / norm)[:, None] * W_adapted`.

**Returns:** `F.linear(x, W_dora, linear.bias)`, shape `(*, out_features)`. The bias is not rescaled.

**Constraints:**
- A freshly built layer returns exactly `linear(x)`.
- Only `lora_A`, `lora_B` and `magnitude` receive gradients.
- `norm` receives no gradient.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**The decomposition.** DoRA (Liu et al., 2024) writes any weight as `W = m * V / ||V||`: a magnitude per output row times a unit direction. Plain LoRA changes both at once through one low-rank update. DoRA trains the magnitude directly and lets LoRA move only the direction.

**Why this helps.** The paper finds that full fine-tuning tends to change magnitude and direction in opposite amounts, while LoRA changes them in lock-step. Decoupling them brings DoRA's learning pattern, and usually its accuracy, closer to full fine-tuning at almost no extra parameter cost: one vector of `out_features` per layer.

**Why detach the norm.** Section 4.3 of the paper treats `||V + delta V||` as a constant during backward. That drops a large term from the gradient graph and cuts training memory substantially, with little accuracy loss. HuggingFace PEFT follows it with `weight_norm.detach()`.

**Merging.** The adapted layer is still one linear map, `(magnitude / norm)[:, None] * W_adapted`, so a trained DoRA adapter merges into the base weight for inference just like LoRA.""",
    "advisory_prerequisites": ["lora", "lora_merge"],
    "hints": [
        {"level": 1, "kind": "questions", "content": "Along which dimension of the (out, in) weight do you take the norm, so there is one number per output unit? What must the magnitude be at init for the layer to start as the base? Where does torch need to be told that the norm is a constant?"},
        {"level": 2, "kind": "analysis", "content": "In __init__, set magnitude = nn.Parameter(linear.weight.detach().norm(dim=1)). In forward, build W = linear.weight + scaling * lora_B @ lora_A, norm = W.norm(dim=1).detach(), W_dora = (magnitude / norm)[:, None] * W, and return F.linear(x, W_dora, linear.bias)."},
    ],
    "model_connections": [
        "HuggingFace PEFT exposes DoRA as LoraConfig(use_dora=True); the math lives in DoraLinearLayer.",
        "NVIDIA NeMo and Unsloth ship DoRA for LLaMA-family fine-tuning.",
    ],
    "pro_con_analysis": {
        "pros": ["Closer to full fine-tuning accuracy than LoRA at the same rank, and still mergeable for inference.", "Adds only one vector of size out_features per adapted layer."],
        "cons": ["Each training step rebuilds the full adapted weight to take its norm, which costs memory and time over plain LoRA.", "Needs the dequantized base weight each step when combined with a quantized base."],
    },
    "sources": [
        {
            "kind": "code",
            "url": "https://github.com/huggingface/peft",
            "commit": "b8674c86183a5dee38d0c3ede392e189593025e5",
            "path": "src/peft/tuners/lora/dora.py",
            "symbol": "DoraLinearLayer.get_weight_norm, DoraLinearLayer.update_layer, DoraLinearLayer.forward",
            "license": "Apache-2.0",
            "adapted": "Magnitude initialized to the row norms of the base weight, the adapted-weight norm computed per output "
                       "row and detached, and the output rescaled by magnitude / norm with the bias left unscaled.",
            "simplifications": "Computes the rescaled weight directly instead of adding a correction to the base output, "
                               "no dropout, caching, fan_in_fan_out, quantized base or embedding and conv variants.",
        },
        {
            "kind": "paper",
            "url": "https://arxiv.org/abs/2402.09353",
            "section": "Section 4 (Eq. 5) and Section 4.3, gradient of the detached norm",
        },
    ],
    "tests": [
        {"name": "Fresh layer equals the base layer", "behavior": "state.invariant", "code": r"""
import torch
torch.manual_seed(0)
layer = {fn}(8, 5, rank=2, alpha=4.0)
assert layer.magnitude.shape == (5,)
assert torch.allclose(layer.magnitude, layer.linear.weight.norm(dim=1), atol=1e-6)
x = torch.randn(3, 8)
assert torch.allclose(layer(x), layer.linear(x), atol=1e-5)
"""},
        {"name": "Forward matches the decomposed weight on seeded layers", "visibility": "unshown", "behavior": "numerics.stability", "failure_message": "Output must be x @ ((magnitude / row_norm(W + scaling * B @ A))[:, None] * (W + scaling * B @ A)).T + bias.", "code": r"""
import torch, torch.nn.functional as F
for seed in (1, 9, 40):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    fin, fout, r = int(torch.randint(4, 12, (1,), generator=g)), int(torch.randint(3, 9, (1,), generator=g)), int(torch.randint(1, 4, (1,), generator=g))
    alpha = float(torch.randint(1, 9, (1,), generator=g)) * 3.0
    layer = {fn}(fin, fout, rank=r, alpha=alpha)
    with torch.no_grad():
        layer.lora_A.copy_(torch.randn(r, fin, generator=g))
        layer.lora_B.copy_(torch.randn(fout, r, generator=g))
        layer.magnitude.copy_(torch.rand(fout, generator=g) * 2 + 0.5)
        layer.linear.bias.copy_(torch.randn(fout, generator=g))
    W = layer.linear.weight.detach() + (alpha / r) * layer.lora_B.detach() @ layer.lora_A.detach()
    col = torch.sqrt((W * W).sum(dim=1))
    x = torch.randn(2, 3, fin, generator=g)
    want = torch.einsum("bti,oi->bto", x, W) * (layer.magnitude.detach() / col) + layer.linear.bias.detach()
    out = layer(x)
    assert out.shape == (2, 3, fout), out.shape
    assert torch.allclose(out, want, atol=1e-4), (seed, (out - want).abs().max())
"""},
        {"name": "Gradients treat the norm as a constant", "visibility": "unshown", "behavior": "gradient.flow", "failure_message": "Detach the row norm: gradients of lora_A, lora_B and magnitude must match the paper's constant-norm form.", "code": r"""
import torch
for seed in (3, 17, 44):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    layer = {fn}(7, 4, rank=3, alpha=6.0).double()
    with torch.no_grad():
        layer.lora_A.copy_(torch.randn(3, 7, generator=g, dtype=torch.float64))
        layer.lora_B.copy_(torch.randn(4, 3, generator=g, dtype=torch.float64))
        layer.magnitude.copy_(torch.rand(4, generator=g, dtype=torch.float64) + 0.5)
    x = torch.randn(5, 7, generator=g, dtype=torch.float64)
    upstream = torch.randn(5, 4, generator=g, dtype=torch.float64)
    (layer(x) * upstream).sum().backward()
    W0 = layer.linear.weight.detach()
    A = layer.lora_A.detach().clone().requires_grad_()
    B = layer.lora_B.detach().clone().requires_grad_()
    m = layer.magnitude.detach().clone().requires_grad_()
    W = W0 + 2.0 * B @ A
    s = m / torch.sqrt((W.detach() ** 2).sum(dim=1))
    ((x @ W.T) * s * upstream).sum().backward()
    assert torch.allclose(layer.magnitude.grad, m.grad, atol=1e-9), seed
    assert torch.allclose(layer.lora_A.grad, A.grad, atol=1e-9), seed
    assert torch.allclose(layer.lora_B.grad, B.grad, atol=1e-9), seed
    assert layer.linear.weight.grad is None and layer.linear.bias.grad is None
"""},
        {"name": "Only the adapter and magnitude are trainable", "visibility": "unshown", "behavior": "contract.signature", "failure_message": "Freeze linear; train lora_A (rank, in), lora_B (out, rank) initialized to zero, and magnitude (out,) initialized to the row norms of linear.weight.", "code": r"""
import torch
torch.manual_seed(12)
layer = {fn}(12, 6, rank=4, alpha=8.0)
assert torch.allclose(layer.magnitude.detach(), torch.sqrt((layer.linear.weight.detach() ** 2).sum(dim=1)), atol=1e-6)
x = torch.randn(4, 12)
assert torch.allclose(layer(x), layer.linear(x), atol=1e-5), 'a fresh layer must equal the base layer'
trainable = {name for name, p in layer.named_parameters() if p.requires_grad}
assert trainable == {'lora_A', 'lora_B', 'magnitude'}, trainable
assert layer.lora_A.shape == (4, 12) and layer.lora_B.shape == (6, 4)
assert torch.count_nonzero(layer.lora_B) == 0
assert abs(layer.scaling - 2.0) < 1e-12
"""},
    ],
    "solution": '''import torch
import torch.nn as nn
import torch.nn.functional as F

class DoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank, alpha=1.0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.linear.weight.requires_grad_(False)
        self.linear.bias.requires_grad_(False)
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.scaling = alpha / rank
        self.magnitude = nn.Parameter(self.linear.weight.detach().norm(dim=1))

    def forward(self, x):
        weight = self.linear.weight + self.scaling * self.lora_B @ self.lora_A
        norm = weight.norm(dim=1).detach()
        weight = (self.magnitude / norm)[:, None] * weight
        return F.linear(x, weight, self.linear.bias)
''',
    "interview_questions": interview(
        concept=[
            "What is DoRA, and how does it differ from LoRA?",
            "What do the magnitude vector and the direction each represent, and how many extra parameters does DoRA add?",
        ],
        deep_dive=[
            "Along which axis is the weight norm taken, and why one value per output unit?",
            "How must the magnitude be initialized so training starts from the base model?",
            "Why does DoRA detach the norm in backward, and what does that save?",
            "How do you merge a trained DoRA adapter into the base weight for inference?",
        ],
        tradeoffs=[
            "DoRA versus LoRA at the same rank: accuracy, training memory and speed?",
            "What does DoRA cost when the base model is quantized, as in QDoRA?",
        ],
    ),
}
