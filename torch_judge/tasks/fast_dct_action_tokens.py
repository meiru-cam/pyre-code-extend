"""FAST action tokens before BPE: orthonormal DCT over time, scale-and-round, low frequencies first."""

from ._interview import interview

TASK = {
    "title": "FAST DCT Action Tokens",
    "difficulty": "Medium",
    "version": 1,
    "function_name": "fast_encode",
    "description_en": r"""Turn a chunk of continuous robot actions into the integer tokens FAST feeds to BPE, and decode them back.

**Signatures:** define both functions at the top level of your code.
- `fast_encode(chunk, scale=10.0, min_token=0) -> LongTensor`
- `fast_decode(tokens, T, D, scale=10.0, min_token=0) -> Tensor`

**Parameters:**
- `chunk` — float tensor of shape `(T, D)` or `(B, T, D)`: `T` time steps of a `D`-dimensional action. Values are already normalized (FAST maps the 1st and 99th percentiles to -1 and 1 beforehand).
- `scale` — positive float. Larger keeps more precision and makes more distinct token values.
- `min_token` — integer subtracted from every rounded coefficient, so the smallest expected coefficient becomes token 0.
- `tokens` — integer tensor of shape `(T * D,)` or `(B, T * D)`, as returned by `fast_encode`.

**Encode,** for each chunk, in this order:
- Apply the orthonormal DCT-II along the time axis, separately for every action dimension. With `n` the time index and `k` the frequency index, both in `0 .. T-1`:

        C[k, n] = sqrt(1 / T) * cos(pi * (2n + 1) * k / (2T))   if k == 0
        C[k, n] = sqrt(2 / T) * cos(pi * (2n + 1) * k / (2T))   otherwise
        coeff   = C @ chunk          # (T, D): row k holds frequency k of every dimension

- Round `coeff * scale` to the nearest integer with `torch.round`. Exact ties depend on the last bit of the DCT, so no case relies on them.
- Flatten the `(T, D)` result row-major: all dimensions of frequency 0, then all dimensions of frequency 1, and so on.
- Subtract `min_token` and clamp at 0.

**Decode** inverts the steps it can: reshape to `(T, D)`, add `min_token`, divide by `scale`, and apply the inverse DCT `C.T @ coeff` along time.

**Returns:**
- `fast_encode` — `torch.long` tensor of shape `(T * D,)` for a 2-D chunk or `(B, T * D)` for a 3-D chunk.
- `fast_decode` — `torch.float64` tensor of shape `(T, D)` for 1-D tokens or `(B, T, D)` for 2-D tokens.

**Constraints:**
- Build the DCT as a matrix in torch. Do not import scipy.
- Compute the DCT in `float64`, whatever the input dtype.
- The DCT runs over time, never over the action dimensions.
- Clamping is lossy: decode does not try to undo it.
- `T = 1` is valid: the DCT of one step is the step itself.
- `fast_encode` raises `ValueError` unless `chunk` is 2-D or 3-D.
- `fast_decode` raises `ValueError` unless `tokens` is 1-D or 2-D with a last dimension of exactly `T * D`.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why not one token per action value.** Binning every dimension at every step (RT-2, OpenVLA) gives `T * D` tokens per chunk. At high control rates neighbouring steps are nearly identical, so the next token is easy to guess from the previous one and the training signal per token is tiny. Autoregressive VLAs trained that way struggle on fast, dexterous tasks.

**What the DCT buys.** Smooth action trajectories put almost all their energy into a few low-frequency coefficients. After scaling and rounding, most high-frequency coefficients become 0. Flattening low frequencies first puts the informative coefficients at the front and long runs of zeros at the back, which BPE then merges into a handful of tokens. The orthonormal DCT keeps the round trip exact apart from rounding: each coefficient is off by at most `0.5 / scale`, and because `C` is orthogonal the L2 error of the whole chunk is at most `0.5 / scale * sqrt(T * D)`. A single action entry can be off by more than `0.5 / scale`.

**Where BPE comes in.** Upstream turns each integer into a character with `chr` and trains a byte-level BPE tokenizer on those strings (the `bpe` exercise covers that part). `min_token` is the smallest rounded coefficient seen while fitting, so every character is non-negative. openpi maps the resulting BPE ids onto the tail of the PaliGemma vocabulary for pi0-FAST.""",
    "advisory_prerequisites": ["bpe", "vla_action_bins"],
    "hints": [
        {"level": 1, "kind": "questions", "content": "Which axis of a (B, T, D) tensor is time, and how do you multiply every batch item by a (T, T) matrix along it? After the DCT, which index should vary fastest when you flatten so that frequency 0 of every dimension comes first?"},
        {"level": 2, "kind": "analysis", "content": "Build the matrix from an outer product of a frequency index and a time index, in float64, and give row 0 its own scale. A matrix product with the time axis on the right broadcasts over a batch dimension, and a row-major reshape of the (T, D) coefficients already puts frequency 0 first. Decoding reuses the same matrix, because an orthonormal matrix is inverted by its transpose."},
    ],
    "model_connections": [
        "pi0-FAST tokenizes 1-second action chunks with FAST and trains an autoregressive VLA on the BPE ids, matching diffusion pi0 on dexterous tasks while training several times faster.",
        "openpi's FASTTokenizer wraps the same processor and maps its BPE ids onto the last entries of the PaliGemma vocabulary.",
        "LeRobot's pi0fast policy loads the physical-intelligence/fast processor from the Hugging Face Hub.",
    ],
    "pro_con_analysis": {
        "pros": [
            "Few tokens per chunk for smooth trajectories, so autoregressive decoding is faster and each token carries more information.",
            "One scale parameter trades reconstruction error against vocabulary size, and the inverse is closed form.",
        ],
        "cons": [
            "Rounding and clamping are lossy; a coefficient below min_token is silently clipped.",
            "Token count varies with the trajectory after BPE, and a malformed sequence may not reshape back to (T, D) at all.",
        ],
    },
    "sources": [
        {
            "kind": "code",
            "url": "https://huggingface.co/physical-intelligence/fast/blob/ec4d7aa71691cac0b8bed6942be45684db2110f4/processing_action_tokenizer.py",
            "commit": "ec4d7aa71691cac0b8bed6942be45684db2110f4",
            "path": "processing_action_tokenizer.py",
            "symbol": "UniversalActionProcessor.__call__, UniversalActionProcessor.decode",
            "license": "Apache-2.0",
            "adapted": "scipy.fft.dct / idct with norm='ortho' along time rewritten as an explicit orthonormal DCT-II matrix in torch; "
                       "np.around becomes torch.round.",
            "simplifications": "BPE (chr, bpe_tokenizer, ord) is left out, so encode returns the integer code points and decode takes them. "
                               "T and D are passed to decode instead of being cached on the processor. "
                               "Where upstream decode catches a wrong-length token row, prints the error and returns all-zero actions, "
                               "this exercise raises ValueError. "
                               "Upstream keeps a float32 chunk in float32 through scipy's DCT; the exercise always computes in float64, "
                               "so it matches upstream run on the chunk cast to float64. fit(the min_token search and BPE training) is not part of the exercise.",
        },
        {
            "kind": "paper",
            "url": "https://arxiv.org/abs/2501.09747",
            "section": "Pertsch et al. 2025, Section V-A (DCT) and Section V-B, Algorithm 1 FAST Tokenizer",
            "note": "Normalize, DCT per dimension, scale and round, flatten with low-frequency components of all dimensions first, then BPE.",
        },
    ],
    "tests": [
        {"name": "Hand-computed tokens for a two-step chunk", "behavior": "state.invariant", "code": r"""
import torch
chunk = torch.tensor([[1.0, 0.0], [3.0, 0.5]], dtype=torch.float64)
# frequency 0 = sum / sqrt(2), frequency 1 = (x0 - x1) / sqrt(2), then * 10 and round:
#   dim 0: 28.28 -> 28, -14.14 -> -14      dim 1: 3.54 -> 4, -3.54 -> -4
# row-major flatten [f0d0, f0d1, f1d0, f1d1] = [28, 4, -14, -4], minus min_token = -20:
out = {fn}(chunk, scale=10.0, min_token=-20)
assert out.dtype == torch.long, out.dtype
assert out.tolist() == [48, 24, 6, 16], out.tolist()
"""},
        {"name": "Decode recovers the chunk at a fine scale", "behavior": "state.invariant", "code": r"""
import torch
fast_decode = {fn}.__globals__["fast_decode"]
g = torch.Generator().manual_seed(0)
chunk = torch.rand(6, 3, generator=g, dtype=torch.float64) * 2 - 1
tokens = {fn}(chunk, scale=1000.0, min_token=-5000)
assert tokens.shape == (18,), tokens.shape
back = fast_decode(tokens, 6, 3, scale=1000.0, min_token=-5000)
assert back.shape == (6, 3) and back.dtype == torch.float64, (back.shape, back.dtype)
assert torch.allclose(back, chunk, atol=2e-3), (back - chunk).abs().max()
"""},
        {"name": "Matches the DCT-II formula on seeded chunks", "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "Use the orthonormal DCT-II along time: row 0 scaled by sqrt(1/T), other rows by sqrt(2/T), then round(coeff * scale) - min_token.", "code": r"""
import math, random, torch
def expected(x, scale, min_token):
    T, D = len(x), len(x[0])
    out = []
    for k in range(T):
        a = math.sqrt((1.0 if k == 0 else 2.0) / T)
        for d in range(D):
            c = a * sum(x[n][d] * math.cos(math.pi * (2 * n + 1) * k / (2 * T)) for n in range(T))
            out.append(max(round(c * scale) - min_token, 0))
    return out
for seed, (T, D), scale, min_token in ((1, (5, 3), 10.0, -200), (2, (8, 2), 7.5, -300), (3, (3, 7), 25.0, -400), (4, (16, 4), 10.0, -500)):
    rng = random.Random(seed)
    x = [[rng.uniform(-1, 1) for _ in range(D)] for _ in range(T)]
    out = {fn}(torch.tensor(x, dtype=torch.float64), scale=scale, min_token=min_token)
    assert out.tolist() == expected(x, scale, min_token), (seed, out.tolist(), expected(x, scale, min_token))
"""},
        {"name": "Flattens low frequencies first with dimensions interleaved", "visibility": "unshown", "behavior": "tensor.shape",
         "failure_message": "Flatten the (T, D) coefficients row-major: frequency 0 of every dimension, then frequency 1, and the DCT must run over time.", "code": r"""
import torch
# dim 0 is constant 1, dim 1 is constant 2, dim 2 is constant 0: only frequency 0 is non-zero.
chunk = torch.tensor([[1.0, 2.0, 0.0]] * 4, dtype=torch.float64)
out = {fn}(chunk, scale=10.0, min_token=0)
# frequency 0 = value * sqrt(T) = value * 2
assert out.tolist() == [20, 40, 0] + [0] * 9, out.tolist()
"""},
        {"name": "Batched chunks encode each item independently", "visibility": "unshown", "behavior": "tensor.shape",
         "failure_message": "A (B, T, D) chunk returns (B, T * D) tokens, each row equal to encoding that item alone.", "code": r"""
import torch
g = torch.Generator().manual_seed(7)
chunk = torch.rand(3, 5, 2, generator=g, dtype=torch.float64) * 2 - 1
out = {fn}(chunk, scale=10.0, min_token=-40)
assert out.shape == (3, 10) and out.dtype == torch.long, (out.shape, out.dtype)
for b in range(3):
    assert torch.equal(out[b], {fn}(chunk[b], scale=10.0, min_token=-40)), b
"""},
        {"name": "Coefficients below min_token clamp to zero", "visibility": "unshown", "behavior": "edge.empty_or_boundary",
         "failure_message": "Subtract min_token from the rounded coefficients, then clamp negative results at 0.", "code": r"""
import torch
chunk = torch.tensor([[-3.0], [-1.0], [0.6], [2.0]], dtype=torch.float64)
# rounded coefficients * 10: [-7, -37, -3, -3]
out = {fn}(chunk, scale=10.0, min_token=-5)
assert out.tolist() == [0, 0, 2, 2], out.tolist()
out = {fn}(chunk, scale=10.0, min_token=-36)
assert out.tolist() == [29, 0, 33, 33], out.tolist()
"""},
        {"name": "Computes the DCT in float64", "visibility": "unshown", "behavior": "tensor.dtype_device",
         "failure_message": "Cast the chunk to float64 before the DCT so float32 inputs round the same way as float64 ones.", "code": r"""
import torch
# float64 DCT: frequency 1 = (v - w) / sqrt(2) * 1000 = 196.50002 -> 197.
# The same arithmetic in float32 lands exactly on 196.5 and rounds to 196.
chunk = torch.tensor([[0.44110000133514404], [0.1632070094347]], dtype=torch.float32)
out = {fn}(chunk, scale=1000.0, min_token=0)
assert out.dtype == torch.long, out.dtype
assert out.tolist() == [427, 197], out.tolist()
"""},
        {"name": "Single time step and constant chunks", "visibility": "unshown", "behavior": "edge.empty_or_boundary",
         "failure_message": "For T = 1 the coefficient is the action itself; a constant chunk has only frequency 0 non-zero, equal to value * sqrt(T).", "code": r"""
import torch
fast_decode = {fn}.__globals__["fast_decode"]
one = torch.tensor([[0.31, -0.74]], dtype=torch.float64)
out = {fn}(one, scale=10.0, min_token=-10)
assert out.tolist() == [13, 3], out.tolist()
back = fast_decode(out, 1, 2, scale=10.0, min_token=-10)
assert torch.allclose(back, torch.tensor([[0.3, -0.7]], dtype=torch.float64)), back
const = torch.full((9, 2), -0.4, dtype=torch.float64)
out = {fn}(const, scale=10.0, min_token=-20)
assert out.tolist() == [8, 8] + [20] * 16, out.tolist()
"""},
        {"name": "Decode inverts with the transposed DCT matrix", "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "Decode: reshape to (T, D), add min_token, divide by scale, then apply the inverse DCT C.T @ coeff along time.", "code": r"""
import math, random, torch
fast_decode = {fn}.__globals__["fast_decode"]
rng = random.Random(11)
T, D, scale, min_token = 5, 3, 4.0, -30
tokens = [rng.randint(0, 60) for _ in range(T * D)]
out = fast_decode(torch.tensor(tokens), T, D, scale=scale, min_token=min_token)
assert out.shape == (T, D) and out.dtype == torch.float64, (out.shape, out.dtype)
for n in range(T):
    for d in range(D):
        want = 0.0
        for k in range(T):
            a = math.sqrt((1.0 if k == 0 else 2.0) / T)
            want += a * math.cos(math.pi * (2 * n + 1) * k / (2 * T)) * (tokens[k * D + d] + min_token) / scale
        assert abs(out[n, d].item() - want) < 1e-9, (n, d, out[n, d].item(), want)
"""},
        {"name": "Batched round trip stays within the rounding budget", "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "decode(encode(x)) must return x up to rounding: each coefficient is off by at most 0.5 / scale.", "code": r"""
import torch
fast_decode = {fn}.__globals__["fast_decode"]
for seed, (B, T, D), scale in ((3, (4, 10, 3), 10.0), (5, (2, 7, 5), 50.0)):
    g = torch.Generator().manual_seed(seed)
    chunk = torch.rand(B, T, D, generator=g, dtype=torch.float64) * 2 - 1
    tokens = {fn}(chunk, scale=scale, min_token=-1000)
    back = fast_decode(tokens, T, D, scale=scale, min_token=-1000)
    assert back.shape == (B, T, D), back.shape
    err = ((back - chunk) ** 2).sum(dim=(1, 2)).sqrt()
    assert (err <= 0.5 / scale * (T * D) ** 0.5 + 1e-9).all(), err
    assert not torch.allclose(back, chunk, atol=1e-6)
"""},
        {"name": "Rejects malformed inputs", "visibility": "unshown", "behavior": "contract.signature",
         "failure_message": "fast_encode needs a 2-D or 3-D chunk; fast_decode needs 1-D or 2-D tokens whose last dimension is T * D.", "code": r"""
import torch
fast_decode = {fn}.__globals__["fast_decode"]
for bad in (torch.zeros(4), torch.zeros(1, 2, 3, 4)):
    try:
        {fn}(bad)
    except ValueError:
        pass
    else:
        raise AssertionError(f"fast_encode accepted shape {tuple(bad.shape)}")
for bad, T, D in ((torch.zeros(5, dtype=torch.long), 2, 3), (torch.zeros(2, 7, dtype=torch.long), 2, 3), (torch.zeros(1, 1, 6, dtype=torch.long), 2, 3)):
    try:
        fast_decode(bad, T, D)
    except ValueError:
        pass
    else:
        raise AssertionError(f"fast_decode accepted shape {tuple(bad.shape)} for T={T}, D={D}")
"""},
    ],
    "solution": '''import math
import torch

def _dct_matrix(T):
    n = torch.arange(T, dtype=torch.float64)
    k = torch.arange(T, dtype=torch.float64).unsqueeze(1)
    C = torch.cos(math.pi * (2 * n + 1) * k / (2 * T))
    C[0] *= math.sqrt(1.0 / T)
    C[1:] *= math.sqrt(2.0 / T)
    return C

def fast_encode(chunk, scale=10.0, min_token=0):
    if chunk.ndim not in (2, 3):
        raise ValueError("chunk must have shape (T, D) or (B, T, D)")
    T, D = chunk.shape[-2], chunk.shape[-1]
    coeff = _dct_matrix(T) @ chunk.to(torch.float64)
    ints = torch.round(coeff * scale).reshape(*chunk.shape[:-2], T * D)
    return torch.clamp(ints - min_token, min=0).long()

def fast_decode(tokens, T, D, scale=10.0, min_token=0):
    if tokens.ndim not in (1, 2) or tokens.shape[-1] != T * D:
        raise ValueError("tokens must have shape (T * D,) or (B, T * D)")
    coeff = (tokens.to(torch.float64) + min_token).reshape(*tokens.shape[:-1], T, D) / scale
    return _dct_matrix(T).T @ coeff
''',
    "interview_questions": interview(
        concept=[
            "What problem with per-dimension, per-timestep action binning does FAST solve for autoregressive VLAs?",
            "Why does applying a DCT along time make an action chunk compressible?",
        ],
        deep_dive=[
            "How do you build the orthonormal DCT-II matrix, and why is its inverse simply the transpose?",
            "Why flatten the coefficient matrix with all dimensions of frequency 0 first instead of all frequencies of dimension 0 first?",
            "Where does information get lost in the FAST encode-decode round trip, and how large can the reconstruction error be?",
        ],
        tradeoffs=[
            "How does the choice of scale trade reconstruction accuracy against vocabulary size and token count?",
            "FAST tokens with an autoregressive VLA versus a flow-matching action expert like pi0: what do you gain and lose?",
        ],
    ),
}
