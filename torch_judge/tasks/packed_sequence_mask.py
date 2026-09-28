"""Sequence packing: block-diagonal causal mask, reset positions and cumulative sequence offsets."""

from ._interview import interview

TASK = {
    "title": "Packed Sequence Attention Mask",
    "difficulty": "Medium",
    "version": 1,
    "function_name": "packed_attention_mask",
    "description_en": r"""Build the attention metadata for several documents packed into one row of length `total_len`.

**Signature:** `packed_attention_mask(seq_lens, total_len) -> tuple[Tensor, Tensor, Tensor]`

**Parameters:**
- `seq_lens` — list of positive integers, the document lengths in packing order.
- `total_len` — integer, the row length. Positions after `sum(seq_lens)` are padding.

**Returns:** `(mask, position_ids, cu_seqlens)`
- `mask` — bool tensor `(total_len, total_len)`. `mask[q, k]` is True when query `q` may attend to key `k`: both belong to the same document and `k <= q`. A padding position attends only to itself.
- `position_ids` — int64 tensor `(total_len,)`. Positions restart at 0 at the start of every document; padding positions are 0.
- `cu_seqlens` — int32 tensor `(len(seq_lens) + 1,)` of cumulative document boundaries, starting at 0. Padding is not included.

**Constraints:**
- No token may attend across a document boundary.
- Every row of `mask` has at least one True entry.
- Raise `ValueError` when `seq_lens` is empty, contains a non-positive length, or sums to more than `total_len`.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why pack.** Padding every sample to the longest one wastes compute on pad tokens. Packing concatenates short documents into full rows so every position does useful work, which often speeds SFT up by 2x or more.

**Why the mask matters.** A plain causal mask lets the second document attend to the first, which leaks unrelated context and changes the loss. The block-diagonal causal mask, together with position ids that restart per document, makes a packed row compute exactly what the separate documents would.

**cu_seqlens.** FlashAttention's variable-length kernels never build the dense mask. They take the cumulative boundaries instead and run attention per segment, which is why packing is nearly free there.""",
    "advisory_prerequisites": ["causal_attention", "rope"],
    "hints": [
        {"level": 1, "kind": "questions", "content": "How can you label each position with its document id? Given those ids, what two conditions make mask[q, k] True? What should a padding row contain so softmax does not produce NaN?"},
        {"level": 2, "kind": "analysis", "content": "Build a doc_id vector with torch.repeat_interleave, giving each padding position its own unique id. Then mask = (doc_id[:, None] == doc_id[None, :]) & causal. position_ids = arange minus each document's start offset. cu_seqlens = cumsum of [0] + seq_lens as int32."},
    ],
    "model_connections": [
        "Hugging Face's DataCollatorWithFlattening, TRL's packing=True and Megatron's packed datasets pass cu_seqlens and reset position ids to FlashAttention varlen kernels.",
    ],
    "pro_con_analysis": {
        "pros": ["Removes padding waste while keeping each document's computation identical to training it alone."],
        "cons": ["The dense mask is quadratic in row length; real systems rely on varlen kernels, and packing changes per-document loss weighting unless the loss is normalized carefully."],
    },
    "sources": [{'kind': 'code',
      'url': 'https://github.com/huggingface/transformers',
      'commit': '2e703d6c7e81ec584c25edc39e8bd2f5a45b04bd',
      'path': 'src/transformers/data/data_collator.py',
      'symbol': 'DataCollatorWithFlattening',
      'license': 'Apache-2.0',
      'adapted': 'Per-document position ids that restart at zero and cumulative sequence offsets for varlen attention.',
      'simplifications': 'Builds the dense block-diagonal causal mask as well, for learning; adds explicit padding rows '
                         'that attend only to themselves.'}],
    "tests": [
        {"name": "Two documents and padding", "behavior": "attention.masking", "code": r"""
import torch
mask, pos, cu = {fn}([2, 3], 6)
want = torch.tensor([
    [1, 0, 0, 0, 0, 0],
    [1, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0],
    [0, 0, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 1],
], dtype=torch.bool)
assert mask.dtype == torch.bool and torch.equal(mask, want), mask
assert pos.dtype == torch.int64 and pos.tolist() == [0, 1, 0, 1, 2, 0], pos
assert cu.dtype == torch.int32 and cu.tolist() == [0, 2, 5], cu
"""},
        {"name": "Matches a seeded element-wise oracle", "visibility": "unshown", "behavior": "attention.masking", "failure_message": "Queries may only see earlier keys of the same document; padding sees only itself; positions restart per document.", "code": r"""
import random
for seed in (2, 31, 77):
    rng = random.Random(seed)
    lens = [rng.randint(1, 6) for _ in range(rng.randint(1, 5))]
    total = sum(lens) + rng.randint(2, 4)
    mask, pos, cu = {fn}(lens, total)
    doc, start, starts = [], 0, []
    for d, n in enumerate(lens):
        doc += [d] * n; starts += [start] * n; start += n
    for q in range(total):
        for k in range(total):
            if q >= start:
                want = q == k
            else:
                want = k < start and doc[q] == doc[k] and k <= q
            assert bool(mask[q, k]) == want, (seed, lens, q, k)
        want_pos = 0 if q >= start else q - starts[q]
        assert pos[q].item() == want_pos, (seed, q, pos)
    running = [0]
    for n in lens:
        running.append(running[-1] + n)
    assert cu.tolist() == running, (seed, cu, running)
"""},
        {"name": "Packed attention equals attention on each document alone", "visibility": "unshown", "behavior": "attention.masking", "failure_message": "With the mask applied, each document's outputs must match running attention on that document separately.", "code": r"""
import math, torch
g = torch.Generator().manual_seed(8)
lens = [3, 1, 4]
total = 9
d = 5
q = torch.randn(total, d, generator=g); k = torch.randn(total, d, generator=g); v = torch.randn(total, d, generator=g)
mask, _, _ = {fn}(lens, total)
scores = (q @ k.T / math.sqrt(d)).masked_fill(~mask, float("-inf"))
packed = torch.softmax(scores, dim=-1) @ v
assert torch.isfinite(packed).all()
start = 0
for n in lens:
    sl = slice(start, start + n)
    s = q[sl] @ k[sl].T / math.sqrt(d)
    s = s.masked_fill(torch.triu(torch.ones(n, n, dtype=torch.bool), 1), float("-inf"))
    alone = torch.softmax(s, dim=-1) @ v[sl]
    assert torch.allclose(packed[sl], alone, atol=1e-6), (n, packed[sl], alone)
    start += n
"""},
        {"name": "Rejects invalid lengths", "visibility": "unshown", "behavior": "contract.signature", "failure_message": "Empty lists, non-positive lengths and overflowing rows must raise ValueError.", "code": r"""
for lens, total in (([], 4), ([2, 0], 4), ([3, -1], 4), ([3, 3], 5)):
    try:
        {fn}(lens, total)
    except ValueError:
        pass
    else:
        raise AssertionError(f"accepted {lens} {total}")
"""},
    ],
    "solution": '''import torch

def packed_attention_mask(seq_lens, total_len):
    if not seq_lens or any(n <= 0 for n in seq_lens) or sum(seq_lens) > total_len:
        raise ValueError("seq_lens must be positive and fit in total_len")
    used = sum(seq_lens)
    lengths = torch.tensor(seq_lens)
    doc_id = torch.repeat_interleave(torch.arange(len(seq_lens)), lengths)
    pad_ids = torch.arange(len(seq_lens), len(seq_lens) + total_len - used)
    doc_id = torch.cat([doc_id, pad_ids])
    causal = torch.ones(total_len, total_len, dtype=torch.bool).tril()
    mask = (doc_id[:, None] == doc_id[None, :]) & causal

    cu_seqlens = torch.cat([torch.zeros(1, dtype=torch.long), torch.cumsum(lengths, 0)])
    starts = torch.repeat_interleave(cu_seqlens[:-1], lengths)
    position_ids = torch.zeros(total_len, dtype=torch.long)
    position_ids[:used] = torch.arange(used) - starts
    return mask, position_ids, cu_seqlens.to(torch.int32)
''',
    "interview_questions": interview(
        concept=[
            "What is sequence packing, and why does it speed up SFT and pretraining?",
            "What goes wrong if you pack documents but keep a plain causal mask and continuous position ids?",
        ],
        deep_dive=[
            "Build the block-diagonal causal mask from the document lengths. What must a padding row contain, and why?",
            "What is cu_seqlens, and how do FlashAttention varlen kernels use it instead of a dense mask?",
            "With RoPE, why must position ids restart at each document boundary?",
        ],
        tradeoffs=[
            "Packing changes how many tokens each document contributes to a batch. How does that interact with per-sequence versus per-token loss averaging?",
            "Greedy bin packing versus sorting by length versus truncation: effects on throughput, padding and data distribution?",
        ],
    ),
}
