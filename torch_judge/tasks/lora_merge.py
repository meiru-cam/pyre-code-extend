"""Merge a LoRA adapter into its base weight for inference, and take it back out."""

from ._interview import interview

TASK = {
    "title": "LoRA Merge and Unmerge",
    "difficulty": "Easy",
    "version": 1,
    "function_name": "MergeableLoRALinear",
    "description_en": r"""Extend a LoRA linear layer with `merge()` and `unmerge()`, which fold the adapter into the base weight and take it back out.

**Signature:** `MergeableLoRALinear(in_features, out_features, rank, alpha=1.0)` (nn.Module)

**Attributes:**
- `linear` — `nn.Linear(in_features, out_features)`, weight and bias frozen.
- `lora_A` — parameter `(rank, in_features)`, small random init.
- `lora_B` — parameter `(out_features, rank)`, initialized to zeros.
- `scaling` — float, `alpha / rank`.
- `merged` — bool, `False` after construction.

**Methods:**
- `forward(x)` — unmerged: `linear(x) + x @ lora_A.T @ lora_B.T * scaling`. Merged: `linear(x)` only.
- `merge()` — add `delta = lora_B @ lora_A * scaling` to `linear.weight`, set `merged = True`.
- `unmerge()` — subtract the same delta, set `merged = False`.

**Constraints:**
- `merge()` on a merged layer and `unmerge()` on an unmerged layer do nothing.
- Update `linear.weight` in place. It must stay the same `Parameter` object.
- Neither method records autograd history, and neither changes `lora_A` or `lora_B`.
- `merge()` then `unmerge()` restores `linear.weight` to within `1e-5`.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why merge.** An unmerged LoRA layer runs two extra small matmuls per call. After merging, the layer is a plain `nn.Linear` again, so inference has zero extra latency and the adapter disappears from the serving graph.

**Why unmerge.** Serving several adapters on one base model, or going back to training, needs the original weight. HuggingFace PEFT keeps the adapter matrices after `merge()` for exactly this. `merge_and_unload()` is the one-way version that drops them.

**In place matters.** An optimizer, an FSDP shard or a tied embedding holds a reference to the weight `Parameter`. Replacing it with a new tensor silently detaches all of them.

**Round-trip error.** In fp32 the round trip is exact to rounding. In bf16, `W + delta - delta` can differ from `W` by one ulp per element, which is why PEFT casts the delta to the weight dtype before subtracting and why production stacks keep an fp32 master copy.""",
    "advisory_prerequisites": ["lora"],
    "hints": [
        {"level": 1, "kind": "questions", "content": "What (out, in) matrix does the adapter add to the base weight? Which state stops a second merge from adding it twice? How do you change a Parameter's values without replacing it and without autograd recording the change?"},
        {"level": 2, "kind": "analysis", "content": "Write delta = lora_B @ lora_A * scaling. In merge, return early if merged, otherwise add delta to linear.weight inside torch.no_grad() with an in-place add_ and set merged. unmerge mirrors it with sub_. forward returns linear(x) alone when merged."},
    ],
    "model_connections": [
        "HuggingFace PEFT's LoraLayer.merge / unmerge and merge_and_unload fold adapters into the base weight before export.",
        "vLLM and LoRAX keep adapters unmerged to serve many of them over one shared base model.",
    ],
    "pro_con_analysis": {
        "pros": ["A merged layer costs exactly what the base layer costs at inference.", "Unmerge makes switching adapters on one base cheap and exact in fp32."],
        "cons": ["A merged model serves one adapter at a time; multi-tenant serving keeps adapters separate.", "Merging into a low-precision or quantized weight loses information, so the round trip is not exact."],
    },
    "sources": [{
        "kind": "code",
        "url": "https://github.com/huggingface/peft",
        "commit": "b8674c86183a5dee38d0c3ede392e189593025e5",
        "path": "src/peft/tuners/lora/layer.py",
        "symbol": "Linear.merge, Linear.unmerge, Linear.get_delta_weight",
        "license": "Apache-2.0",
        "adapted": "Delta weight lora_B @ lora_A * scaling added to and subtracted from base_layer.weight.data in place, "
                   "with the merged state guarding repeat calls and a merged forward that runs the base layer only.",
        "simplifications": "One adapter instead of a named set, no safe_merge NaN check, no lora_bias, fan_in_fan_out, "
                           "dropout or DoRA variant.",
    }],
    "tests": [
        {"name": "Merged output matches unmerged output", "behavior": "state.invariant", "code": r"""
import torch
torch.manual_seed(0)
layer = {fn}(8, 4, rank=2, alpha=4.0)
with torch.no_grad():
    layer.lora_B.normal_()
x = torch.randn(3, 8)
before = layer(x)
layer.merge()
assert layer.merged
assert torch.allclose(layer(x), before, atol=1e-5), 'merged forward must equal unmerged forward'
layer.unmerge()
assert not layer.merged
assert torch.allclose(layer(x), before, atol=1e-5)
"""},
        {"name": "Merge adds the scaled delta on seeded layers", "visibility": "unshown", "behavior": "numerics.stability", "failure_message": "merge() must add lora_B @ lora_A * (alpha / rank) to linear.weight, and the merged forward must use the base layer only.", "code": r"""
import torch, torch.nn.functional as F
for seed in (1, 7, 23):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    fin, fout, r = int(torch.randint(4, 12, (1,), generator=g)), int(torch.randint(3, 9, (1,), generator=g)), int(torch.randint(1, 4, (1,), generator=g))
    alpha = float(torch.randint(1, 9, (1,), generator=g))
    layer = {fn}(fin, fout, rank=r, alpha=alpha)
    with torch.no_grad():
        layer.lora_A.copy_(torch.randn(r, fin, generator=g))
        layer.lora_B.copy_(torch.randn(fout, r, generator=g))
    W0 = layer.linear.weight.detach().clone()
    b0 = layer.linear.bias.detach().clone()
    A, B = layer.lora_A.detach().clone(), layer.lora_B.detach().clone()
    x = torch.randn(4, fin, generator=g)
    want = F.linear(x, W0 + (B @ A) * (alpha / r), b0)
    assert torch.allclose(layer(x), want, atol=1e-4), seed
    layer.merge()
    assert torch.allclose(layer.linear.weight, W0 + (B @ A) * (alpha / r), atol=1e-5), seed
    assert torch.allclose(layer(x), want, atol=1e-4), seed
    assert torch.equal(layer.lora_A, A) and torch.equal(layer.lora_B, B), 'merge must not change the adapter'
"""},
        {"name": "Repeat calls are no-ops and unmerge restores the weight", "visibility": "unshown", "behavior": "state.invariant", "failure_message": "A second merge() or an unmerge() without a merge must leave the weight unchanged; merge then unmerge must restore it.", "code": r"""
import torch
for seed in (2, 11, 31):
    torch.manual_seed(seed)
    layer = {fn}(6, 5, rank=3, alpha=2.0)
    with torch.no_grad():
        layer.lora_B.normal_()
    W0 = layer.linear.weight.detach().clone()
    layer.unmerge()
    assert torch.equal(layer.linear.weight, W0), 'unmerge before merge changed the weight'
    assert not layer.merged
    layer.merge()
    W1 = layer.linear.weight.detach().clone()
    layer.merge()
    assert torch.equal(layer.linear.weight, W1), 'second merge added the delta twice'
    layer.unmerge()
    assert torch.allclose(layer.linear.weight, W0, atol=1e-5), 'unmerge did not restore the base weight'
    layer.unmerge()
    assert torch.allclose(layer.linear.weight, W0, atol=1e-5), 'second unmerge subtracted the delta twice'
"""},
        {"name": "Weight updated in place without autograd history", "visibility": "unshown", "behavior": "gradient.flow", "failure_message": "Update linear.weight in place under torch.no_grad(); keep the same Parameter, frozen, with no grad_fn.", "code": r"""
import torch
torch.manual_seed(5)
layer = {fn}(6, 4, rank=2)
with torch.no_grad():
    layer.lora_B.normal_()
weight = layer.linear.weight
layer.merge()
assert layer.linear.weight is weight, 'merge replaced the weight Parameter'
assert layer.linear.weight.grad_fn is None and not layer.linear.weight.requires_grad
layer.unmerge()
assert layer.linear.weight is weight, 'unmerge replaced the weight Parameter'
assert layer.linear.weight.grad_fn is None and not layer.linear.weight.requires_grad
params = dict(layer.named_parameters())
assert set(name for name, p in params.items() if p.requires_grad) == {'lora_A', 'lora_B'}
layer(torch.randn(2, 6)).sum().backward()
assert layer.lora_A.grad is not None and layer.lora_B.grad is not None
"""},
        {"name": "Fresh layer starts unmerged and equal to base", "visibility": "unshown", "behavior": "contract.signature", "failure_message": "Construct lora_A (rank, in), lora_B (out, rank) zeros, scaling alpha / rank and merged False.", "code": r"""
import torch
layer = {fn}(10, 6, rank=4, alpha=8.0)
assert layer.lora_A.shape == (4, 10) and layer.lora_B.shape == (6, 4)
assert torch.count_nonzero(layer.lora_B) == 0
assert abs(layer.scaling - 2.0) < 1e-12
assert layer.merged is False
x = torch.randn(3, 10)
assert torch.allclose(layer(x), layer.linear(x), atol=1e-6)
"""},
    ],
    "solution": '''import torch
import torch.nn as nn

class MergeableLoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank, alpha=1.0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.linear.weight.requires_grad_(False)
        self.linear.bias.requires_grad_(False)
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.scaling = alpha / rank
        self.merged = False

    def delta_weight(self):
        return self.lora_B @ self.lora_A * self.scaling

    @torch.no_grad()
    def merge(self):
        if self.merged:
            return
        self.linear.weight.add_(self.delta_weight())
        self.merged = True

    @torch.no_grad()
    def unmerge(self):
        if not self.merged:
            return
        self.linear.weight.sub_(self.delta_weight())
        self.merged = False

    def forward(self, x):
        if self.merged:
            return self.linear(x)
        return self.linear(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
''',
    "interview_questions": interview(
        concept=[
            "Why merge a LoRA adapter before serving, and what does it cost at inference if you do not?",
            "Why keep the adapter matrices after merging instead of deleting them?",
        ],
        deep_dive=[
            "Write the merged weight in terms of W, A, B, alpha and rank, and check its shape.",
            "Why must the merge update the weight in place under no_grad, and what breaks if it assigns a new tensor?",
            "What happens if merge() is called twice, and how do you prevent it?",
            "Is merge then unmerge exact in bf16? Why or why not?",
        ],
        tradeoffs=[
            "Merged single-adapter serving versus unmerged multi-adapter serving (S-LoRA, Punica): latency, memory and flexibility?",
            "Can you merge a LoRA trained on a 4-bit QLoRA base back into the quantized weights? What goes wrong?",
        ],
    ),
}
