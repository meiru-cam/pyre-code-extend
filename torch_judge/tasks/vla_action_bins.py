"""Uniform per-dimension action binning for VLAs: q01/q99 normalization, OpenVLA tail tokens and the RT-1 convention."""

from ._interview import interview

# Plain-torch model of both conventions, written independently of the reference.
_MODEL = r"""
import torch

def m_norm(x, low, high):
    n = (2 * (x - low) / (high - low + 1e-8) - 1).clamp(-1, 1)
    return torch.where(low == high, torch.zeros_like(n), n)

def m_encode(x, low, high, n_bins, vocab, conv):
    n = m_norm(x, low, high)
    if conv == "openvla":
        edges = torch.linspace(-1, 1, n_bins, dtype=n.dtype)
        idx = (n.unsqueeze(-1) >= edges).sum(-1)
        return vocab - idx
    return ((n + 1) / 2 * (n_bins - 1)).floor().long()

def m_decode(ids, low, high, n_bins, vocab, conv):
    if conv == "openvla":
        edges = torch.linspace(-1, 1, n_bins, dtype=low.dtype)
        centers = (edges[:-1] + edges[1:]) / 2
        n = centers[(vocab - ids - 1).clamp(0, n_bins - 2)]
        return 0.5 * (n + 1) * (high - low) + low
    return ids.to(low.dtype) / (n_bins - 1) * (high - low) + low

def stats(g, D, dtype=torch.float64):
    low = torch.rand(D, generator=g, dtype=dtype) * 2 - 1.5
    high = low + torch.rand(D, generator=g, dtype=dtype) * 3 + 0.1
    return low, high
"""

TASK = {
    "title": "VLA Action Binning: OpenVLA and RT-1",
    "difficulty": "Medium",
    "version": 1,
    "function_name": "ActionBinner",
    "description_en": r"""Turn continuous robot actions into discrete tokens and back, the way OpenVLA and RT-1 do, with per-dimension uniform bins.

**Signature:** `ActionBinner(low, high, n_bins, vocab_size, convention="openvla")`

**Parameters:**
- `low`, `high` — float tensors of shape `(D,)`: the per-dimension 1st and 99th percentiles (q01, q99) of the training actions. `high >= low`.
- `n_bins` — integer, at least 2.
- `vocab_size` — integer size of the language model's vocabulary. Used only by `"openvla"`.
- `convention` — `"openvla"` or `"rt1"`.

**Methods** (actions have shape `(..., D)`; outputs keep the leading shape):
- `normalize(actions) -> Tensor` — same dtype as `actions`:

        n = clamp(2 * (actions - low) / (high - low + 1e-8) - 1, -1, 1)

  Any dimension with `low == high` is set to `0`.
- `encode(actions) -> LongTensor` — normalize, then:
  - `"openvla"`: `edges = torch.linspace(-1, 1, n_bins)`. `bin` is the number of edges `<= n` (this is `np.digitize(n, edges)`, a value in `1..n_bins`). The token id is `vocab_size - bin`.
  - `"rt1"`: the token id is `floor((n + 1) / 2 * (n_bins - 1))`, a value in `0..n_bins-1`.
- `decode(token_ids) -> Tensor` — actions in the original units, dtype of `low`:
  - `"openvla"`: `i = clamp(vocab_size - token_ids - 1, 0, n_bins - 2)`, `n = centers[i]` with `centers = (edges[:-1] + edges[1:]) / 2`, then `action = 0.5 * (n + 1) * (high - low) + low`.
  - `"rt1"`: `action = token_ids / (n_bins - 1) * (high - low) + low`. Do not clamp the ids.

**Constraints:**
- Use the given `low` and `high`; never compute statistics from the actions you are encoding.
- Values outside `[low, high]` clip to the end bins.
- OpenVLA decodes to bin centers, so there are `n_bins - 1` reachable actions per dimension.
- Raise `ValueError` for an unknown `convention`, `n_bins < 2`, `low` and `high` of different shapes, any `high < low`, or actions whose last dimension is not `D`.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why bins.** RT-1, RT-2 and OpenVLA predict actions with the same cross-entropy head they use for text, so each continuous action dimension becomes one token from a fixed set of uniform bins. A 7-DoF action is 7 tokens per control step.

**Why q01/q99.** Normalizing with the dataset min and max lets one outlier stretch the range and waste most bins. OpenVLA maps the 1st to 99th percentile onto `[-1, 1]` and clips the rest. Dimensions that never move (for example an unused gripper axis) are set to 0 instead of dividing by nearly zero.

**The epsilon.** Because of the `+ 1e-8`, an action exactly at `high` normalizes to just below `+1`; only values clipped from above reach `+1` exactly.

**The off-by-one.** `np.digitize` over `n_bins` edges returns `1..n_bins`, but there are only `n_bins - 1` intervals. OpenVLA subtracts 1 and clips, so a value exactly at `+1` lands in the last interval. Its ids take the last `n_bins` slots of the Llama vocabulary, the least-used BPE tokens, so `bin` counts down from `vocab_size - 1`.

**RT-1.** RT-1 has its own action vocabulary of `n_bins` tokens. It truncates to the left end of each bucket and decodes to that end, so its reconstruction is biased low by up to one bucket, and only the exact maximum reaches the top token.""",
    "advisory_prerequisites": ["int8_quantization"],
    "hints": [
        {"level": 1, "kind": "questions", "content": "How many bin edges and how many bin centers do n_bins edges give? Which bin does a value exactly at +1 fall in, and how does decode keep that index in range? What should a dimension with low == high become?"},
        {"level": 2, "kind": "analysis", "content": "Normalize with the given low/high, clamp to [-1, 1] and zero the constant dims with torch.where. For OpenVLA, torch.bucketize(n, edges, right=True) equals np.digitize; ids are vocab_size - bin. Decode with clamp(vocab_size - id - 1, 0, n_bins - 2) into the centers. For RT-1, floor (n + 1) / 2 * (n_bins - 1)."},
    ],
    "model_connections": [
        "OpenVLA (Llama 2 backbone) overwrites the last 256 vocabulary tokens with action bins and normalizes each dimension with q01/q99 statistics.",
        "RT-1 discretizes each action dimension into 256 buckets of its own action vocabulary; RT-2 reuses the same 256-bin scheme inside a VLM.",
        "pi0-FAST and the FAST tokenizer start from the same q01/q99 normalization but compress a whole chunk with a DCT and BPE instead of one token per dimension.",
    ],
    "pro_con_analysis": {
        "pros": [
            "Reuses the language model's cross-entropy head unchanged, so a VLM becomes a policy with no new output layer.",
            "Each token is easy to read back: one dimension, one bin.",
        ],
        "cons": [
            "Resolution is fixed at (high - low) / (n_bins - 1) per dimension, and high-frequency control needs one token per dimension per step.",
            "Cross-entropy ignores bin order, so predicting a neighboring bin costs as much as predicting a distant one.",
        ],
    },
    "sources": [
        {'kind': 'code',
         'url': 'https://github.com/openvla/openvla/blob/c8f03f48af692657d3060c19588038c7220e9af9/prismatic/vla/action_tokenizer.py',
         'commit': 'c8f03f48af692657d3060c19588038c7220e9af9',
         'path': 'prismatic/vla/action_tokenizer.py',
         'symbol': 'ActionTokenizer.__call__, ActionTokenizer.decode_token_ids_to_actions',
         'license': 'MIT',
         'adapted': 'np.digitize over linspace(-1, 1, n_bins), ids vocab_size - bin, decode to bin centers with the clipped index, in torch.',
         'simplifications': 'Returns token ids instead of decoded strings; vocab_size is passed in instead of read from a Hugging Face tokenizer.'},
        {'kind': 'code',
         'url': 'https://github.com/openvla/openvla/blob/c8f03f48af692657d3060c19588038c7220e9af9/prismatic/vla/datasets/rlds/utils/data_utils.py',
         'commit': 'c8f03f48af692657d3060c19588038c7220e9af9',
         'path': 'prismatic/vla/datasets/rlds/utils/data_utils.py',
         'symbol': 'normalize_action_and_proprio',
         'license': 'MIT',
         'adapted': 'The BOUNDS_Q99 branch: 2 * (x - q01) / (q99 - q01 + 1e-8) - 1, clipped to [-1, 1].',
         'simplifications': 'Upstream zeroes dimensions where the dataset min equals max; the exercise only has q01/q99 and zeroes '
                            'dimensions where q01 equals q99. The two agree whenever min equals max; a dimension with q01 == q99 '
                            'but min != max is set to 0 here, while upstream pins it to -1 or +1. The optional per-dimension '
                            'normalization mask is dropped.'},
        {'kind': 'code',
         'url': 'https://github.com/openvla/openvla/blob/c8f03f48af692657d3060c19588038c7220e9af9/prismatic/extern/hf/modeling_prismatic.py',
         'commit': 'c8f03f48af692657d3060c19588038c7220e9af9',
         'path': 'prismatic/extern/hf/modeling_prismatic.py',
         'symbol': 'OpenVLAForActionPrediction.predict_action',
         'license': 'MIT',
         'adapted': 'Unnormalize: 0.5 * (n + 1) * (q99 - q01) + q01.',
         'simplifications': 'Only the arithmetic after generate is kept; the normalization mask is dropped.'},
        {'kind': 'code',
         'url': 'https://github.com/google-research/robotics_transformer/blob/4569641b8111f3f402c32d8e24becd2a6e952ecc/tokenizers/action_tokenizer.py',
         'commit': '4569641b8111f3f402c32d8e24becd2a6e952ecc',
         'path': 'tokenizers/action_tokenizer.py',
         'symbol': 'RT1ActionTokenizer.tokenize, RT1ActionTokenizer.detokenize',
         'license': 'Apache-2.0',
         'adapted': 'Float branch: int((a - min) / (max - min) * (V - 1)) and a / (V - 1) * (max - min) + min, in torch.',
         'simplifications': 'Applied to the q01/q99-normalized value with min = -1 and max = 1 instead of to the raw action with '
                            'the action-spec bounds; one (..., D) tensor instead of a TensorSpecStruct; the int32 one-hot '
                            'branch is dropped.'},
        {'kind': 'paper',
         'url': 'https://arxiv.org/abs/2406.09246',
         'section': 'Kim et al. 2024 (OpenVLA), Section 3.2 Training Procedure',
         'note': '256 uniform bins per dimension between the 1st and 99th quantile, overwriting the 256 least-used Llama tokens.'},
        {'kind': 'paper',
         'url': 'https://arxiv.org/abs/2212.06817',
         'section': 'Brohan et al. 2022 (RT-1), Section 4 RT-1: Robotics Transformer, Action tokenization',
         'note': 'Each action dimension discretized into 256 uniform bins.'},
    ],
    "tests": [
        {"name": "Normalize with q01/q99, clip, and zero a constant dimension", "behavior": "numerics.stability", "code": r"""
import torch
low = torch.tensor([0.0, -2.0, 0.5], dtype=torch.float64)
high = torch.tensor([1.0, 2.0, 0.5], dtype=torch.float64)
b = {fn}(low, high, 256, 32000)
x = torch.tensor([[0.5, 4.0, 0.7], [0.0, -1.0, 9.0]], dtype=torch.float64)
n = b.normalize(x)
want = torch.tensor([[0.0, 1.0, 0.0], [-1.0, -0.5, 0.0]], dtype=torch.float64)
assert n.shape == (2, 3) and n.dtype == torch.float64, (n.shape, n.dtype)
assert torch.allclose(n, want, atol=1e-6), n
"""},
        {"name": "OpenVLA: tail token ids and decode to bin centers", "behavior": "edge.empty_or_boundary", "code": r"""
import torch
low, high = -torch.ones(4, dtype=torch.float64), torch.ones(4, dtype=torch.float64)
b = {fn}(low, high, 5, 100, convention="openvla")
# edges [-1, -0.5, 0, 0.5, 1], centers [-0.75, -0.25, 0.25, 0.75]
# 1.5 is beyond high, so it clips to exactly +1 and digitizes to n_bins = 5
x = torch.tensor([-1.0, 0.1, -0.6, 1.5], dtype=torch.float64)
ids = b.encode(x)
assert ids.dtype == torch.long and ids.tolist() == [99, 97, 99, 95], ids
rec = b.decode(ids)
assert torch.allclose(rec, torch.tensor([-0.75, 0.25, -0.75, 0.75], dtype=torch.float64)), rec
"""},
        {"name": "RT-1: floor to a bucket and decode to its left end", "behavior": "edge.empty_or_boundary", "code": r"""
import torch
low = torch.tensor([0.0, 0.0], dtype=torch.float64)
high = torch.tensor([10.0, 10.0], dtype=torch.float64)
b = {fn}(low, high, 11, 32000, convention="rt1")
ids = b.encode(torch.tensor([[3.7, 12.0]], dtype=torch.float64))
assert ids.dtype == torch.long and ids.tolist() == [[3, 10]], ids
rec = b.decode(ids)
assert torch.allclose(rec, torch.tensor([[3.0, 10.0]], dtype=torch.float64)), rec
"""},
        {"name": "Normalization matches a seeded model", "visibility": "unshown", "behavior": "numerics.stability",
         "failure_message": "Map [low, high] onto [-1, 1] with the given statistics, clip, and set dimensions with low == high to 0.", "code": _MODEL + r"""
for seed in (0, 1, 2, 3, 4):
    g = torch.Generator().manual_seed(seed)
    D = 3 + seed
    low, high = stats(g, D)
    high[1] = low[1]
    x = torch.randn(4, 6, D, generator=g, dtype=torch.float64) * 2
    b = {fn}(low, high, 256, 32000)
    out = b.normalize(x)
    assert out.shape == x.shape, (seed, out.shape)
    assert torch.allclose(out, m_norm(x, low, high), atol=1e-12), seed
    assert (out[..., 1] == 0).all(), seed
"""},
        {"name": "Statistics come from low/high, not from the batch", "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "Normalize with the q01/q99 passed to the constructor; the batch's own min and max must not matter.", "code": _MODEL + r"""
low = torch.tensor([-1.0, 0.0], dtype=torch.float64)
high = torch.tensor([1.0, 4.0], dtype=torch.float64)
b = {fn}(low, high, 256, 32000)
x = torch.tensor([[0.2, 1.0], [0.4, 2.0]], dtype=torch.float64)
assert torch.allclose(b.normalize(x), torch.tensor([[0.2, -0.5], [0.4, 0.0]], dtype=torch.float64), atol=1e-6), b.normalize(x)
one = b.normalize(x[:1])
assert torch.allclose(one, torch.tensor([[0.2, -0.5]], dtype=torch.float64), atol=1e-6), one
"""},
        {"name": "OpenVLA encode and decode match a seeded model", "visibility": "unshown", "behavior": "numerics.stability",
         "failure_message": "Token id is vocab_size - digitize(n, linspace(-1, 1, n_bins)); decode clamps vocab_size - id - 1 to [0, n_bins - 2] and reads the bin centers.", "code": _MODEL + r"""
for seed, n_bins, vocab in ((0, 256, 32000), (1, 7, 50), (2, 16, 1000), (3, 256, 32064), (4, 3, 10)):
    g = torch.Generator().manual_seed(seed)
    D = 2 + seed
    low, high = stats(g, D)
    x = low + (high - low) * (torch.rand(5, D, generator=g, dtype=torch.float64) * 1.4 - 0.2)
    b = {fn}(low, high, n_bins, vocab, convention="openvla")
    ids = b.encode(x)
    want = m_encode(x, low, high, n_bins, vocab, "openvla")
    assert ids.dtype == torch.long and torch.equal(ids, want), (seed, ids, want)
    assert ids.min() >= vocab - n_bins and ids.max() <= vocab - 1, seed
    rec = b.decode(ids)
    assert torch.allclose(rec, m_decode(ids, low, high, n_bins, vocab, "openvla"), atol=1e-12), seed
    width = (high - low) / (n_bins - 1)
    inside = ((x >= low) & (x <= high))
    assert ((rec - x).abs() <= width + 1e-9)[inside].all(), seed
"""},
        {"name": "OpenVLA boundaries: +1, -1, out-of-range values and ids", "visibility": "unshown", "behavior": "edge.empty_or_boundary",
         "failure_message": "A value clipped to +1 digitizes to n_bins and must decode to the last center; values beyond q01/q99 clip to the end bins; out-of-range ids clamp.", "code": r"""
import torch
low, high = torch.zeros(3, dtype=torch.float64), torch.full((3,), 2.0, dtype=torch.float64)
b = {fn}(low, high, 256, 32000, convention="openvla")
ids = b.encode(torch.tensor([[50.0, 0.0, 1.0], [2.0, -50.0, 2.0]], dtype=torch.float64))
assert ids.tolist() == [[31744, 31999, 31872], [31745, 31999, 31745]], ids
edges = torch.linspace(-1, 1, 256, dtype=torch.float64)
centers = (edges[:-1] + edges[1:]) / 2
rec = b.decode(ids)
want = torch.stack([centers[-1], centers[0], centers[127], centers[-1], centers[0], centers[-1]]).reshape(2, 3) + 1
assert torch.allclose(rec, want, atol=1e-12), rec
far = b.decode(torch.tensor([40000, 5, 31998], dtype=torch.long))
assert torch.allclose(far, torch.stack([centers[0], centers[-1], centers[1]]) + 1, atol=1e-12), far
"""},
        {"name": "OpenVLA constant dimension encodes as the middle and decodes to low", "visibility": "unshown", "behavior": "numerics.stability",
         "failure_message": "A dimension with low == high normalizes to 0 before binning and unnormalizes back to low.", "code": _MODEL + r"""
low = torch.tensor([3.0, -1.0], dtype=torch.float64)
high = torch.tensor([3.0, 1.0], dtype=torch.float64)
b = {fn}(low, high, 256, 32000, convention="openvla")
x = torch.tensor([[2.0, 0.3], [7.0, -0.3]], dtype=torch.float64)
ids = b.encode(x)
assert torch.equal(ids[:, 0], torch.tensor([31872, 31872])), ids
assert torch.equal(ids, m_encode(x, low, high, 256, 32000, "openvla")), ids
rec = b.decode(ids)
assert torch.allclose(rec[:, 0], torch.tensor([3.0, 3.0], dtype=torch.float64)), rec
"""},
        {"name": "RT-1 encode and decode match a seeded model", "visibility": "unshown", "behavior": "numerics.stability",
         "failure_message": "RT-1 ids are floor((n + 1) / 2 * (n_bins - 1)) with no vocabulary offset, and decode is id / (n_bins - 1) mapped back to [low, high].", "code": _MODEL + r"""
for seed, n_bins in ((0, 256), (1, 11), (2, 3), (3, 100), (4, 2)):
    g = torch.Generator().manual_seed(seed)
    D = 2 + seed
    low, high = stats(g, D)
    x = low + (high - low) * (torch.rand(6, D, generator=g, dtype=torch.float64) * 1.4 - 0.2)
    b = {fn}(low, high, n_bins, 32000, convention="rt1")
    ids = b.encode(x)
    want = m_encode(x, low, high, n_bins, 32000, "rt1")
    assert ids.dtype == torch.long and torch.equal(ids, want), (seed, ids, want)
    assert ids.min() >= 0 and ids.max() <= n_bins - 1, seed
    rec = b.decode(ids)
    assert torch.allclose(rec, m_decode(ids, low, high, n_bins, 32000, "rt1"), atol=1e-12), seed
    assert (rec <= torch.maximum(x, low) + 1e-9).all(), seed
"""},
        {"name": "RT-1 decode does not clamp ids", "visibility": "unshown", "behavior": "edge.empty_or_boundary",
         "failure_message": "RT-1 decodes id / (n_bins - 1) * (high - low) + low for any id, without clamping.", "code": r"""
import torch
low, high = torch.tensor([0.0], dtype=torch.float64), torch.tensor([4.0], dtype=torch.float64)
b = {fn}(low, high, 5, 32000, convention="rt1")
rec = b.decode(torch.tensor([[0], [2], [4], [6]], dtype=torch.long))
assert torch.allclose(rec.reshape(-1), torch.tensor([0.0, 2.0, 4.0, 6.0], dtype=torch.float64)), rec
"""},
        {"name": "Keeps leading batch and time dimensions", "visibility": "unshown", "behavior": "tensor.shape",
         "failure_message": "Bin each of the D dimensions elementwise and keep every leading dimension of the input.", "code": r"""
import torch
g = torch.Generator().manual_seed(7)
low, high = -torch.ones(7), torch.ones(7)
for conv in ("openvla", "rt1"):
    b = {fn}(low, high, 256, 32000, convention=conv)
    x = torch.rand(2, 3, 4, 7, generator=g) * 2 - 1
    ids = b.encode(x)
    assert ids.shape == (2, 3, 4, 7) and ids.dtype == torch.long, (conv, ids.shape, ids.dtype)
    rec = b.decode(ids)
    assert rec.shape == (2, 3, 4, 7) and rec.dtype == torch.float32, (conv, rec.shape, rec.dtype)
    assert b.normalize(x).dtype == torch.float32, conv
"""},
        {"name": "Rejects invalid arguments", "visibility": "unshown", "behavior": "contract.signature",
         "failure_message": "Raise ValueError for an unknown convention, n_bins < 2, mismatched or inverted low/high, and actions with the wrong last dimension.", "code": r"""
import torch
lo, hi = torch.zeros(3), torch.ones(3)
bad = [
    lambda: {fn}(lo, hi, 256, 32000, convention="fast"),
    lambda: {fn}(lo, hi, 1, 32000),
    lambda: {fn}(lo, torch.ones(4), 256, 32000),
    lambda: {fn}(hi, lo, 256, 32000),
    lambda: {fn}(lo, hi, 256, 32000).encode(torch.zeros(2, 4)),
    lambda: {fn}(lo, hi, 256, 32000, convention="rt1").normalize(torch.zeros(5, 2)),
]
for i, call in enumerate(bad):
    try:
        call()
    except ValueError:
        pass
    else:
        raise AssertionError(f"case {i} accepted")
"""},
    ],
    "solution": '''import torch

class ActionBinner:
    def __init__(self, low, high, n_bins, vocab_size, convention="openvla"):
        if convention not in ("openvla", "rt1"):
            raise ValueError(f"unknown convention {convention!r}")
        if n_bins < 2:
            raise ValueError("n_bins must be at least 2")
        if low.shape != high.shape or low.ndim != 1:
            raise ValueError("low and high must both have shape (D,)")
        if (high < low).any():
            raise ValueError("high must be >= low")
        self.low, self.high = low, high
        self.n_bins, self.vocab_size, self.convention = n_bins, vocab_size, convention
        self.edges = torch.linspace(-1, 1, n_bins, dtype=low.dtype)
        self.centers = (self.edges[:-1] + self.edges[1:]) / 2

    def normalize(self, actions):
        if actions.shape[-1] != self.low.shape[0]:
            raise ValueError("actions must end in dimension D")
        low, high = self.low.to(actions.dtype), self.high.to(actions.dtype)
        n = torch.clamp(2 * (actions - low) / (high - low + 1e-8) - 1, -1, 1)
        return torch.where(low == high, torch.zeros_like(n), n)

    def encode(self, actions):
        n = self.normalize(actions)
        if self.convention == "openvla":
            bins = torch.bucketize(n, self.edges.to(n.dtype), right=True)
            return self.vocab_size - bins
        return ((n + 1) / 2 * (self.n_bins - 1)).long()

    def decode(self, token_ids):
        if self.convention == "openvla":
            i = torch.clamp(self.vocab_size - token_ids - 1, 0, self.n_bins - 2)
            n = self.centers[i]
            return 0.5 * (n + 1) * (self.high - self.low) + self.low
        return token_ids.to(self.low.dtype) / (self.n_bins - 1) * (self.high - self.low) + self.low
''',
    "interview_questions": interview(
        concept=[
            "Why do RT-1, RT-2 and OpenVLA turn each continuous action dimension into a discrete token?",
            "Why normalize actions with the 1st and 99th percentiles instead of the min and max?",
        ],
        deep_dive=[
            "With 256 bin edges from linspace(-1, 1, 256), how many intervals are there, what does np.digitize return for a value of exactly +1, and how does OpenVLA decode it?",
            "Why does OpenVLA place action tokens at the end of the Llama vocabulary, and how do you map a bin to a token id and back?",
            "Compare the reconstruction error of OpenVLA's center decoding with RT-1's floor-and-left-edge decoding for a value spread uniformly inside a bin.",
        ],
        tradeoffs=[
            "Per-dimension uniform bins versus a learned action tokenizer such as FAST or VQ: what changes in sequence length, resolution and smoothness?",
            "Discrete action tokens with cross-entropy versus a continuous action head trained with regression or flow matching: what do you gain and lose?",
        ],
    ),
}
