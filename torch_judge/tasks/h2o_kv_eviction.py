"""H2O: evict KV entries by accumulated attention, keeping heavy hitters plus a recent window per head."""

from ._interview import interview

TASK = {
    "title": "H2O Heavy-Hitter KV Eviction",
    "difficulty": "Hard",
    "version": 1,
    "function_name": "H2OCache",
    "description_en": r"""H2O (Heavy-Hitter Oracle) bounds the KV cache by keeping, for each head, the tokens that have received the most attention so far plus the most recent tokens. Implement the per-step bookkeeping and eviction.

**Signature:** `H2OCache(heavy_size, recent_size)` with `update(keys, values, attn) -> (keys, values)` and a `scores` attribute (None before the first update).

**Inputs to `update`.**
- `keys`, `values` — `[heads, L, d]`: the cache returned by the previous update with this step's `n` new tokens appended (the whole prompt on the first call).
- `attn` — `[heads, n, L]`: the softmax attention of the `n` new queries over all `L` keys.

**Steps:**
1. **Validate.** Raise `ValueError` if `attn.shape[2] != L`, or if there are previous scores and their length is not `L - n`.
2. **Accumulate.** `step = attn.sum(dim=1)`, shape `[heads, L]`. On the first call `scores = step`; otherwise `scores = step` with the previous scores added to its first `L - n` columns.
3. **Evict.** If `L <= heavy_size + recent_size`, return the inputs unchanged. Otherwise, for each head independently: among positions `[0, L - recent_size)`, choose the `heavy_size` with the highest scores (on equal scores, the lower index wins), sort them ascending, and append positions `[L - recent_size, L)`. Gather keys, values and scores at those positions.
4. Return the new keys and values, each `[heads, heavy_size + recent_size, d]`; `scores` becomes `[heads, heavy_size + recent_size]`.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Heavy hitters.** Accumulated attention follows a power law: a small set of tokens receives most of it across steps. Keeping them preserves most of what attention reads, even with a small cache.

**Why a recent window as well.** New tokens have had few steps to accumulate score, so a pure score ranking would evict them before they could prove important; local context also matters for fluency.

**Why per head.** Heads attend to different things. Choosing per head keeps each head's important tokens, at the cost of a ragged layout that paged attention kernels do not handle directly.

**Relation to attention sinks.** The first tokens usually become heavy hitters on their own, which is why StreamingLLM's fixed sinks work: they are a hand-coded special case of this policy.

**The catch.** H2O needs the attention probabilities, which fused kernels such as FlashAttention never write out, so real systems either recompute them or use a cheaper proxy score.""",
    "advisory_prerequisites": ["attention_sink_cache", "kv_cache"],
    "hints": [
        {"level": 1, "kind": "questions", "content": "Where do the new tokens' scores come from, and which columns of the new score row line up with the previous scores? Which positions are candidates for heavy hitters, and how do you break ties deterministically?"},
        {"level": 2, "kind": "analysis", "content": "step = attn.sum(1); add old scores to step[:, :L - n]. For eviction, take scores[:, :L - recent], argsort descending with stable=True, keep the first heavy_size, sort them, cat with arange(L - recent, L), then torch.gather keys, values and scores with the index expanded over d."},
    ],
    "model_connections": [
        "H2O (NeurIPS 2023) introduced heavy-hitter eviction; SnapKV, Scissorhands and NVIDIA's kvpress build on accumulated or windowed attention scores.",
        "StreamingLLM's attention sinks are the special case where the heavy hitters are fixed to the first tokens.",
    ],
    "pro_con_analysis": {
        "pros": ["Adapts to the content: tokens that matter keep their place, so quality holds with a much smaller cache than a fixed window."],
        "cons": ["Needs attention probabilities that fused kernels do not produce, evicts irreversibly, and yields a per-head ragged layout that complicates paged memory."],
    },
    "sources": [
        {"kind": "paper", "url": "https://arxiv.org/abs/2306.14048", "section": "Section 3 and Algorithm 1, Heavy-Hitter Oracle"},
    ],
    "tests": [
        {"name": "Keeps heavy hitters and the recent window", "behavior": "attention.cache", "code": r"""
import torch
c = {fn}(heavy_size=1, recent_size=2)
keys = torch.arange(4.0).view(1, 4, 1)
attn = torch.tensor([[[0.7, 0.1, 0.1, 0.1], [0.1, 0.1, 0.6, 0.2]]])
k, v = c.update(keys, keys.clone(), attn)
assert k.flatten().tolist() == [0.0, 2.0, 3.0], k
torch.testing.assert_close(c.scores, torch.tensor([[0.8, 0.7, 0.3]]))
k, v = c.update(torch.cat([k, torch.tensor([[[4.0]]])], 1), torch.cat([v, torch.tensor([[[4.0]]])], 1),
                torch.tensor([[[0.1, 0.1, 0.1, 0.7]]]))
assert k.flatten().tolist() == [0.0, 3.0, 4.0] and torch.equal(k, v), k
"""},
        {"name": "Seeded multi-head streams match an oracle", "visibility": "unshown", "behavior": "attention.cache", "failure_message": "Accumulate summed attention onto the previous scores, pick heavy hitters per head from outside the recent window with lower-index tie-breaking, and keep positions in order.", "code": r"""
import random, torch
for seed in (12, 57, 83):
    rng = random.Random(seed)
    torch.manual_seed(seed)
    H, hh, rc, d = 3, rng.randint(0, 3), rng.randint(1, 4), 2
    c = {fn}(heavy_size=hh, recent_size=rc)
    L0 = rng.randint(1, 10)
    ids = [list(range(L0)) for _ in range(H)]
    sc = None
    K = torch.arange(L0, dtype=torch.float64).view(1, L0, 1).expand(H, L0, d).clone()
    nxt = L0
    for step in range(20):
        n = L0 if step == 0 else rng.randint(1, 3)
        if step > 0:
            new = torch.arange(nxt, nxt + n, dtype=torch.float64).view(1, n, 1).expand(H, n, d)
            K = torch.cat([K, new], 1)
            for h in range(H):
                ids[h] += list(range(nxt, nxt + n))
            nxt += n
        L = K.shape[1]
        logits = torch.randn(H, n, L, dtype=torch.float64)
        if rng.random() < 0.3:
            logits = logits.round()
        attn = torch.softmax(logits, -1)
        s = attn.sum(1).clone()
        if sc is not None:
            s[:, :L - n] += sc
        k, v = c.update(K, K * 2, attn)
        if L > hh + rc:
            keep = []
            for h in range(H):
                cand = sorted(range(L - rc), key=lambda i: (-s[h, i].item(), i))[:hh]
                keep.append(sorted(cand) + list(range(L - rc, L)))
            idx = torch.tensor(keep)
            want = torch.gather(K, 1, idx[..., None].expand(-1, -1, d))
            s = torch.gather(s, 1, idx)
        else:
            want = K
        torch.testing.assert_close(k, want, msg=f"{seed} {step}")
        torch.testing.assert_close(v, want * 2)
        torch.testing.assert_close(c.scores, s)
        K, sc = k, s
"""},
        {"name": "Recent tokens are never heavy-hitter candidates", "visibility": "unshown", "behavior": "attention.cache", "failure_message": "Heavy hitters come only from positions before the recent window, even when a recent token has the highest score.", "code": r"""
import torch
c = {fn}(heavy_size=1, recent_size=2)
keys = torch.arange(5.0).view(1, 5, 1)
attn = torch.tensor([[[0.1, 0.2, 0.05, 0.6, 0.05]]])
k, v = c.update(keys, keys, attn)
assert k.flatten().tolist() == [1.0, 3.0, 4.0], k
"""},
        {"name": "Validates shapes", "visibility": "unshown", "behavior": "contract.signature", "failure_message": "Reject attention whose key length differs from the cache, and new tokens that do not line up with the previous scores.", "code": r"""
import torch
c = {fn}(heavy_size=1, recent_size=1)
try:
    c.update(torch.zeros(1, 3, 2), torch.zeros(1, 3, 2), torch.zeros(1, 1, 4))
except ValueError:
    pass
else:
    raise AssertionError("accepted mismatched attention")
c.update(torch.zeros(1, 2, 2), torch.zeros(1, 2, 2), torch.full((1, 2, 2), 0.5))
try:
    c.update(torch.zeros(1, 5, 2), torch.zeros(1, 5, 2), torch.full((1, 1, 5), 0.2))
except ValueError:
    pass
else:
    raise AssertionError("accepted a cache that does not extend the previous one")
"""},
    ],
    "solution": '''import torch


class H2OCache:
    def __init__(self, heavy_size, recent_size):
        self.heavy_size = heavy_size
        self.recent_size = recent_size
        self.scores = None

    def update(self, keys, values, attn):
        length, n = keys.shape[1], attn.shape[1]
        if attn.shape[2] != length:
            raise ValueError("attention must cover every cached key")
        if self.scores is not None and self.scores.shape[1] != length - n:
            raise ValueError("new tokens must extend the previous cache")
        scores = attn.sum(dim=1)
        if self.scores is not None:
            scores = scores.clone()
            scores[:, :length - n] += self.scores
        self.scores = scores
        if length <= self.heavy_size + self.recent_size:
            return keys, values

        candidates = scores[:, :length - self.recent_size]
        order = torch.argsort(-candidates, dim=-1, stable=True)
        heavy = order[:, :self.heavy_size].sort(dim=-1).values
        recent = torch.arange(length - self.recent_size, length, device=keys.device).expand(keys.shape[0], -1)
        keep = torch.cat([heavy, recent], dim=-1)
        index = keep[..., None].expand(-1, -1, keys.shape[-1])
        self.scores = torch.gather(scores, 1, keep)
        return torch.gather(keys, 1, index), torch.gather(values, 1, index)
''',
    "interview_questions": interview(
        concept=[
            "What is a heavy hitter in the KV cache, and why does accumulated attention identify tokens worth keeping?",
            "Why does H2O keep a recent window in addition to the heavy hitters?",
        ],
        deep_dive=[
            "Walk through one decoding step: how are the new token's attention weights folded into the scores, and which columns line up with the old scores?",
            "Why choose heavy hitters per head rather than one set for the whole layer? What does that do to memory layout?",
            "How do attention sinks relate to H2O's heavy hitters?",
        ],
        tradeoffs=[
            "H2O's accumulated scores versus SnapKV's observation-window scores versus fixed sinks plus a window: accuracy, overhead and compatibility with FlashAttention?",
            "Irreversible eviction versus offloading evicted KV to CPU and recalling it when needed: latency, memory and complexity?",
        ],
    ),
}
