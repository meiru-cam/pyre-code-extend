"""Finite scalar quantization (FSQ): bound, straight-through round, and the mixed-radix code index."""

from ._interview import interview

# A plain-Python, per-element model of the contract, written independently of the reference.
_MODEL = r"""
import math

def model_bound(x, L, eps=1e-3):
    half_l = (L - 1) * (1 + eps) / 2
    offset = 0.5 if L % 2 == 0 else 0.0
    return math.tanh(x + math.atanh(offset / half_l)) * half_l - offset

def model_code(x, L):
    return round(model_bound(x, L)) / (L // 2)

def model_index(digits, levels):
    index, place = 0, 1
    for digit, L in zip(digits, levels):
        index += digit * place
        place *= L
    return index
"""

TASK = {
    "title": "FSQ: Finite Scalar Quantization",
    "difficulty": "Medium",
    "version": 1,
    "function_name": "FSQ",
    "description_en": r"""Implement finite scalar quantization (FSQ), the quantizer in NVIDIA Cosmos-Tokenizer's discrete image and video tokenizers. Each of the `d` channels is squashed into a bounded range and rounded to one of `L_i` integer levels, so the codebook is implicit: every combination of levels is a code, and there is nothing to learn.

**Signature:** `FSQ(levels)` with an attribute and five methods:
- `codebook_size` — the integer product of all levels.
- `bound(z) -> Tensor`
- `quantize(z) -> Tensor`
- `codes_to_indices(zhat) -> Tensor`
- `indices_to_codes(indices) -> Tensor`
- calling the object, `fsq(z) -> (zhat, indices)`

**levels:** a non-empty list of integers, each at least 2. Channel `i` has `L = levels[i]` values. Raise `ValueError` for an empty list or any level below 2.

**bound(z):** `z` is a float tensor of shape `(..., d)` with `d = len(levels)`. Per channel, with `eps = 1e-3`:

    half_l = (L - 1) * (1 + eps) / 2
    offset = 0.5 if L is even else 0.0
    shift  = atanh(offset / half_l)
    bound(z) = tanh(z + shift) * half_l - offset

**quantize(z):** round `bound(z)` to the nearest integer with a straight-through estimator, then divide by `L // 2` (integer division). The result has the shape and dtype of `z`. In the forward pass it equals `round(bound(z)) / (L // 2)`; in the backward pass the round acts as the identity, so the gradient with respect to `z` is the derivative of `bound` divided by `L // 2`.

**codes_to_indices(zhat):** maps codes of shape `(..., d)` to int64 indices of shape `(...)`. Turn each code into a digit with `digit = round(zhat * (L // 2) + L // 2)`, which lies in `0..L-1`. The index is a mixed-radix number whose **first channel is the least significant**:

    index = digit_0 + digit_1 * L_0 + digit_2 * L_0 * L_1 + ...

**indices_to_codes(indices):** the inverse. Takes an integer tensor of any shape `(...)` and returns float32 codes of shape `(..., d)`: `digit_i = (index // (L_0 * ... * L_{i-1})) % L_i`, then `code_i = (digit_i - L_i // 2) / (L_i // 2)`.

**fsq(z):** returns `(zhat, indices)` with `zhat = quantize(z)` (it keeps its gradient) and `indices = codes_to_indices(zhat)`.

**Constraints:**
- Use exactly the bound above: `(1 + eps)`, `atanh` for the shift, and the 0.5 offset only on even levels.
- Do not clamp; the bound alone keeps every channel inside its `L` values.
- `bound`, `quantize`, `codes_to_indices` and `fsq` raise `ValueError` when the last dimension is not `len(levels)`.
- `indices_to_codes` raises `ValueError` for a non-integer tensor or any index outside `0..codebook_size-1`.

**Example:**

    fsq = FSQ([5, 4])          # codebook_size == 20
    fsq(torch.tensor([[0.0, 0.0], [10.0, 10.0], [-10.0, -10.0]]))
    # zhat    = [[0, 0], [1, 0.5], [-1, -1]]
    # indices = [12, 19, 0]

Channel 0 (`L = 5`) takes the values -1, -0.5, 0, 0.5, 1. Channel 1 (`L = 4`) takes -1, -0.5, 0, 0.5: an even level is asymmetric.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why FSQ.** VQ-VAE learns a codebook and maps each latent to its nearest entry, which needs a commitment loss, codebook updates and tricks against unused codes. FSQ (Mentzer et al. 2023) replaces all of that with "bound, then round". The codebook is the grid of level combinations, of size `L_0 * L_1 * ... * L_{d-1}`. The paper reports near-100% codebook usage without auxiliary losses and only a small drop in metrics when FSQ replaces VQ in MaskGIT and UViM.

**Choosing levels.** The paper recommends `L_i >= 5` and lists level sets that approximate powers of two (Table 1), for example `[8, 5, 5, 5]` for about 2^10 codes and `[8, 8, 8, 5, 5, 5]` for about 2^16. Cosmos-Tokenizer's discrete image and video configs use `[8, 8, 8, 5, 5, 5]`, so an encoder outputs 6 channels per position and the codebook has 64 000 entries.

**The even-level offset.** Rounding produces values symmetric around 0, so an odd `L` fits naturally: `(L - 1) / 2` steps on each side. An even `L` cannot be symmetric. Subtracting 0.5 moves the range to about `(-L/2, L/2 - 1)`, which rounds to exactly `L` integers. The `atanh` shift moves the input so that `bound(0)` is exactly 0, so a zero latent lands on the centre code.

**Paper versus code.** The JAX listing in the paper's Appendix A.1 writes `(1 - eps)` and `shift = tan(offset / half_l)`. Cosmos-Tokenizer, adapted from lucidrains' vector-quantize-pytorch, uses `(1 + eps)` and `atanh`. For a small `eps` both give exactly `L` rounded values per channel; they differ slightly in the continuous bound, and `tanh(tan(x))` is not exactly `x`, so the paper's `bound(0)` is close to 0 but not equal. This exercise follows Cosmos.

**Straight-through rounding.** `x + sg(round(x) - x)` equals `round(x)` in the forward pass, and its gradient is 1. The encoder therefore receives the decoder's gradient through the quantizer, scaled only by the slope of the bound.""",
    "advisory_prerequisites": ["vla_action_bins", "rssm_kl_balance"],
    "hints": [
        {"level": 1, "kind": "questions", "content": "For an even number of levels, which integers should rounding produce, and what does the 0.5 offset do to the range of tanh? Which tensor must be detached so that the forward pass rounds but the backward pass sees the identity? In the index, which channel changes fastest as the index counts up?"},
        {"level": 2, "kind": "analysis", "content": "Build per-channel tensors for the levels and for the basis [1, L_0, L_0*L_1, ...]. In bound, compute half_l, the even-level offset and the atanh shift in z's dtype, then apply tanh. In quantize, add the detached difference between the rounded and unrounded bound, then divide by L // 2. For indices, shift codes to digits 0..L-1, round, and take the dot product with the basis. For the inverse, floor-divide by the basis, take the remainder modulo the levels, and recentre."},
    ],
    "model_connections": [
        "NVIDIA Cosmos-Tokenizer's FSQuantizer implements this bound, round and mixed-radix index; its discrete image and video tokenizer configs use levels [8, 8, 8, 5, 5, 5].",
        "Cosmos-Tokenizer also has a ResidualFSQuantizer that stacks FSQ layers, each quantizing the residual the previous ones left.",
        "The FSQ paper swaps FSQ in for VQ in MaskGIT (image generation) and UViM (depth estimation, colorization, panoptic segmentation).",
    ],
    "pro_con_analysis": {
        "pros": [
            "No codebook parameters, no commitment or codebook loss, and the paper reports near-full codebook usage.",
            "Indices and codes convert with integer arithmetic, so a token model can predict indices and the decoder can rebuild codes exactly.",
        ],
        "cons": [
            "The codebook is an axis-aligned grid fixed by the levels; capacity is set by choosing levels rather than learned.",
            "Even levels are asymmetric, and very small levels are reported to work worse (the paper recommends at least 5 per channel).",
        ],
    },
    "sources": [
        {
            "kind": "code",
            "url": "https://github.com/NVIDIA/Cosmos-Tokenizer",
            "commit": "3584ae752ce8ebdbe06a420bf60d7513c0e878cc",
            "path": "cosmos_tokenizer/modules/quantizers.py",
            "symbol": "FSQuantizer (bound, quantize, _scale_and_shift, _scale_and_shift_inverse, codes_to_indices, indices_to_codes, forward)",
            "license": "Apache-2.0",
            "adapted": "The bound with its even-level offset and atanh shift, the straight-through round, normalization by L // 2, and the mixed-radix basis cumprod([1] + levels[:-1]) with its inverse.",
            "simplifications": "Single codebook with no project_in or project_out layers and no image/video channel packing; input is channels-last (..., d). "
                               "forward returns (zhat, indices) rather than (indices, codes, dummy_loss). "
                               "Outputs keep z's dtype rather than being cast to bfloat16, indices are int64 rather than int32, "
                               "codes are rounded to digits before the dot product rather than truncated by the int cast (identical on every exact code), "
                               "and indices_to_codes returns float32 and validates its range.",
        },
        {
            "kind": "code",
            "url": "https://github.com/NVIDIA/Cosmos-Tokenizer",
            "commit": "3584ae752ce8ebdbe06a420bf60d7513c0e878cc",
            "path": "cosmos_tokenizer/networks/configs.py",
            "symbol": "discrete_image, discrete_video",
            "license": "Apache-2.0",
            "adapted": "The levels [8, 8, 8, 5, 5, 5] used in the codebook-size test.",
            "simplifications": "Only the levels are used; the encoder and decoder settings are outside the exercise.",
        },
        {
            "kind": "paper",
            "url": "https://arxiv.org/abs/2309.15505",
            "section": "Mentzer et al. 2023, Section 3.1 (Finite Scalar Quantization), Section 3.2 and Table 1 (recommended levels), Appendix A.1 (JAX code)",
            "note": "The bound-then-round quantizer, round_ste, the implicit codebook and its index, and the L_i >= 5 heuristic. The appendix listing uses (1 - eps) and tan for the shift, unlike Cosmos.",
        },
    ],
    "tests": [
        {"name": "Corners and centre of a [5, 4] codebook", "behavior": "state.invariant", "code": r"""
import torch
fsq = {fn}([5, 4])
assert fsq.codebook_size == 20, fsq.codebook_size
zhat, idx = fsq(torch.tensor([[0.0, 0.0], [10.0, 10.0], [-10.0, -10.0]]))
assert torch.allclose(zhat, torch.tensor([[0.0, 0.0], [1.0, 0.5], [-1.0, -1.0]])), zhat
assert idx.tolist() == [12, 19, 0], idx
assert torch.allclose(fsq.bound(torch.zeros(2)), torch.zeros(2), atol=1e-6), fsq.bound(torch.zeros(2))
"""},
        {"name": "The first channel is the least significant digit", "behavior": "state.invariant", "code": r"""
import torch
fsq = {fn}([8, 5, 3])
codes = fsq.indices_to_codes(torch.tensor([0, 1, 8, 40, 119]))
want = torch.tensor([
    [-1.0, -1.0, -1.0],
    [-0.75, -1.0, -1.0],
    [-1.0, -0.5, -1.0],
    [-1.0, -1.0, 0.0],
    [0.75, 1.0, 1.0],
])
assert codes.shape == (5, 3) and codes.dtype == torch.float32, (codes.shape, codes.dtype)
assert torch.allclose(codes, want), codes
"""},
        {"name": "bound matches the formula per channel", "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "bound is tanh(z + atanh(offset / half_l)) * half_l - offset with half_l = (L - 1) * (1 + eps) / 2 and offset 0.5 only for even L.",
         "code": _MODEL + r"""
import torch
levels = [8, 5, 4, 7, 3, 6]
fsq = {fn}(levels)
for seed in (0, 3, 11):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(9, len(levels), generator=g, dtype=torch.float64) * 1.5
    out = fsq.bound(z)
    assert out.shape == z.shape, out.shape
    for r in range(z.shape[0]):
        for c, L in enumerate(levels):
            want = model_bound(z[r, c].item(), L)
            assert abs(out[r, c].item() - want) < 1e-6, (seed, r, L, out[r, c].item(), want)
"""},
        {"name": "Every channel takes exactly its L values", "visibility": "unshown", "behavior": "edge.empty_or_boundary",
         "failure_message": "Over the whole input range channel i must produce exactly levels[i] distinct codes, from -1 up to (L - 1 - L // 2) / (L // 2).",
         "code": r"""
import torch
levels = [8, 5, 4, 7, 6, 3]
fsq = {fn}(levels)
z = torch.linspace(-8.0, 8.0, 4001, dtype=torch.float64)[:, None].repeat(1, len(levels))
zhat = fsq.quantize(z)
for c, L in enumerate(levels):
    values = sorted(set(round(v, 9) for v in zhat[:, c].tolist()))
    hw = L // 2
    want = [round((k - hw) / hw, 9) for k in range(L)]
    assert values == want, (L, values, want)
"""},
        {"name": "quantize matches round(bound) / (L // 2) on seeded inputs", "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "quantize must round the bounded value to the nearest integer and divide by L // 2 (integer division), not by (L - 1) / 2.",
         "code": _MODEL + r"""
import torch
levels = [8, 6, 5, 4]
fsq = {fn}(levels)
for seed in (1, 5, 9):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(40, len(levels), generator=g, dtype=torch.float64) * 2
    zhat = fsq.quantize(z)
    assert zhat.dtype == torch.float64 and zhat.shape == z.shape, (zhat.dtype, zhat.shape)
    for r in range(z.shape[0]):
        for c, L in enumerate(levels):
            want = model_code(z[r, c].item(), L)
            assert abs(zhat[r, c].item() - want) < 1e-12, (seed, r, L, zhat[r, c].item(), want)
"""},
        {"name": "Index basis puts the first channel lowest", "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "index = sum of digit_i times the product of the levels before channel i; the first channel is the least significant digit.",
         "code": _MODEL + r"""
import random, torch
levels = [3, 7, 4, 6]
fsq = {fn}(levels)
rng = random.Random(17)
rows = [[rng.randrange(L) for L in levels] for _ in range(30)]
codes = torch.tensor([[(d - L // 2) / (L // 2) for d, L in zip(row, levels)] for row in rows], dtype=torch.float32)
idx = fsq.codes_to_indices(codes)
assert idx.dtype == torch.int64 and idx.shape == (30,), (idx.dtype, idx.shape)
assert idx.tolist() == [model_index(row, levels) for row in rows], idx
back = fsq.indices_to_codes(idx)
assert torch.allclose(back, codes), back
"""},
        {"name": "Round trip over the full Cosmos codebook", "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "indices_to_codes must invert codes_to_indices on every index; decode with floor division by the basis then modulo the level.",
         "code": r"""
import torch
fsq = {fn}([8, 8, 8, 5, 5, 5])
assert fsq.codebook_size == 64000, fsq.codebook_size
idx = torch.arange(64000)
codes = fsq.indices_to_codes(idx)
assert codes.shape == (64000, 6), codes.shape
# The round trip is the identity, so all 64 000 codes are distinct.
assert torch.equal(fsq.codes_to_indices(codes), idx)
assert codes.min().item() == -1.0 and codes.max().item() == 1.0
"""},
        {"name": "Gradient passes straight through the round", "visibility": "unshown", "behavior": "gradient.flow",
         "failure_message": "Use x + (round(x) - x).detach() so the gradient of quantize is the slope of bound divided by L // 2.",
         "code": r"""
import math, torch
levels = [8, 5, 6]
fsq = {fn}(levels)
g = torch.Generator().manual_seed(23)
z = (torch.randn(7, 3, generator=g, dtype=torch.float64) * 0.8).requires_grad_()
w = torch.randn(7, 3, generator=g, dtype=torch.float64)
zhat, _ = fsq(z)
(zhat * w).sum().backward()
for r in range(7):
    for c, L in enumerate(levels):
        half_l = (L - 1) * 1.001 / 2
        offset = 0.5 if L % 2 == 0 else 0.0
        t = math.tanh(z[r, c].item() + math.atanh(offset / half_l))
        want = w[r, c].item() * half_l * (1 - t * t) / (L // 2)
        assert abs(z.grad[r, c].item() - want) < 1e-9, (r, L, z.grad[r, c].item(), want)
"""},
        {"name": "Leading dimensions and dtypes are preserved", "visibility": "unshown", "behavior": "tensor.shape",
         "failure_message": "zhat keeps the shape and dtype of z, indices drop the last dimension and are int64, and fsq(z) agrees with quantize and codes_to_indices.",
         "code": r"""
import torch
fsq = {fn}([7, 5, 5, 5, 5])
for dtype in (torch.float32, torch.float64):
    g = torch.Generator().manual_seed(2)
    z = torch.randn(2, 3, 4, 5, generator=g).to(dtype) * 2
    zhat, idx = fsq(z)
    assert zhat.shape == z.shape and zhat.dtype == dtype, (zhat.shape, zhat.dtype)
    assert idx.shape == (2, 3, 4) and idx.dtype == torch.int64, (idx.shape, idx.dtype)
    assert torch.equal(zhat, fsq.quantize(z)) and torch.equal(idx, fsq.codes_to_indices(zhat))
    assert int(idx.min()) >= 0 and int(idx.max()) < fsq.codebook_size
    back = fsq.indices_to_codes(idx)
    assert back.shape == (2, 3, 4, 5) and torch.allclose(back.double(), zhat.double()), back
"""},
        {"name": "Rejects bad levels, widths and indices", "visibility": "unshown", "behavior": "contract.signature",
         "failure_message": "Raise ValueError for empty levels, a level below 2, a last dimension other than len(levels), and indices outside 0..codebook_size-1.",
         "code": r"""
import torch
for levels in ([], [8, 1]):
    try:
        {fn}(levels)
    except ValueError:
        pass
    else:
        raise AssertionError(f"accepted levels={levels}")
fsq = {fn}([5, 4])
for call in (lambda: fsq.bound(torch.zeros(3, 3)), lambda: fsq.quantize(torch.zeros(3)),
             lambda: fsq(torch.zeros(2, 1)), lambda: fsq.codes_to_indices(torch.zeros(4, 3)),
             lambda: fsq.indices_to_codes(torch.tensor([20])), lambda: fsq.indices_to_codes(torch.tensor([-1]))):
    try:
        call()
    except ValueError:
        pass
    else:
        raise AssertionError("missing ValueError")
"""},
    ],
    "solution": '''import math
import torch


class FSQ:
    def __init__(self, levels, eps=1e-3):
        levels = list(levels)
        if not levels or any(not isinstance(L, int) or isinstance(L, bool) or L < 2 for L in levels):
            raise ValueError("levels must be a non-empty list of integers >= 2")
        self.levels = levels
        self.eps = eps
        self.codebook_size = math.prod(levels)
        basis, running = [], 1
        for L in levels:
            basis.append(running)
            running *= L
        self._levels = torch.tensor(levels, dtype=torch.int64)
        self._basis = torch.tensor(basis, dtype=torch.int64)

    def _check(self, z):
        if z.shape[-1:] != (len(self.levels),):
            raise ValueError(f"last dimension must be {len(self.levels)}")

    def bound(self, z):
        self._check(z)
        levels = self._levels.to(z.dtype)
        half_l = (levels - 1) * (1 + self.eps) / 2
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0).to(z.dtype)
        shift = torch.atanh(offset / half_l)
        return torch.tanh(z + shift) * half_l - offset

    def quantize(self, z):
        bounded = self.bound(z)
        rounded = bounded + (bounded.round() - bounded).detach()
        return rounded / (self._levels // 2).to(z.dtype)

    def codes_to_indices(self, zhat):
        self._check(zhat)
        half_width = (self._levels // 2).to(zhat.dtype)
        digits = (zhat.detach() * half_width + half_width).round().long()
        return (digits * self._basis).sum(dim=-1)

    def indices_to_codes(self, indices):
        if indices.is_floating_point() or indices.is_complex() or indices.dtype == torch.bool:
            raise ValueError("indices must be an integer tensor")
        indices = indices.long()
        if indices.numel() and (indices.min() < 0 or indices.max() >= self.codebook_size):
            raise ValueError("indices out of range")
        digits = (indices.unsqueeze(-1) // self._basis) % self._levels
        half_width = self._levels // 2
        return (digits - half_width).to(torch.float32) / half_width.to(torch.float32)

    def __call__(self, z):
        zhat = self.quantize(z)
        return zhat, self.codes_to_indices(zhat)
''',
    "interview_questions": interview(
        concept=[
            "What is finite scalar quantization, and how does its codebook differ from a VQ-VAE codebook?",
            "Why does FSQ need no commitment loss or codebook updates?",
        ],
        deep_dive=[
            "Why does an even number of levels need an offset of 0.5, and what does the atanh shift add?",
            "How is a code turned into a single integer index, and how do you invert it?",
            "What gradient does the encoder receive through the quantizer?",
        ],
        tradeoffs=[
            "FSQ versus VQ for a video tokenizer that feeds an autoregressive world model: codebook usage, parameters and flexibility?",
            "How would you choose the levels for a target codebook size, and what do you lose with very few levels per channel?",
        ],
    ),
}
