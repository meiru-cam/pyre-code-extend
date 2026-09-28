"""StreamingLLM: keep attention-sink tokens plus a recent window, with positions assigned inside the cache."""

from ._interview import interview

TASK = {
    "title": "Attention Sink KV Cache",
    "difficulty": "Hard",
    "version": 1,
    "function_name": "StreamingLLMCache",
    "description_en": r"""StreamingLLM lets a model generate far past its context length with a fixed-size KV cache: it keeps the first few tokens (attention sinks) and the most recent ones, and gives positions by place in the cache rather than in the text. Implement one attention layer with this cache.

**Signature:** `StreamingLLMCache(start_size, recent_size, base=10000.0)` with `attend(q, k, v) -> Tensor` and attributes `keys`, `values`.

**Shapes.** `q`, `k`, `v` are `[heads, n, d]` for `n` new tokens; `d` is even. `keys` and `values` are the cached, un-rotated `[heads, L, d]` tensors (L = 0 before the first call). The output is `[heads, n, d]`.

**`attend(q, k, v)`:**
1. Raise `ValueError` unless `1 <= n <= recent_size`.
2. **Evict for space.** With `L` cached tokens, if `L + n > start_size + recent_size`, keep cached tokens `[0, start_size)` and `[L - recent_size + n, L)`, in order.
3. Append `k` and `v` to the cache. Let `T` be the new cache length.
4. **Rotate by cache position.** Apply RoPE to the cached keys at positions `0 .. T-1` and to `q` at positions `T-n .. T-1`. RoPE of `x` at position `p`: `inv_freq[i] = 1 / base ** (2i / d)` for `i < d/2`; `angle = p * cat(inv_freq, inv_freq)`; result `x * cos(angle) + rotate_half(x) * sin(angle)` where `rotate_half(x) = cat(-x[..., d/2:], x[..., :d/2])`.
5. **Attend.** Scores `q_rot @ k_rot^T / sqrt(d)`; query `i` (position `T-n+i`) may see keys at positions `<= T-n+i`. Softmax over keys, multiply by the cached values, and return.

The cache stores keys before rotation; compute in the input dtype.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Attention sinks.** Softmax must put its mass somewhere. Trained models learn to dump attention on the first tokens, which every later token sees. Evicting them, as a plain sliding window does, shifts that mass onto other tokens and perplexity explodes. Keeping about four of them fixes it.

**Why positions inside the cache.** After eviction the text positions of kept tokens have gaps and grow without bound, past what the model was trained on. Numbering by cache slot keeps every distance within the training range.

**Why store un-rotated keys.** Positions change whenever tokens are evicted, so keys are rotated again at every step. The next exercise stores rotated keys and corrects them only when the cache shifts.

**What it does not do.** Evicted tokens are gone; StreamingLLM keeps generation fluent but does not extend what the model can remember.""",
    "advisory_prerequisites": ["kv_cache", "rope", "sliding_window"],
    "hints": [
        {"level": 1, "kind": "questions", "content": "Which cached tokens survive when n new tokens arrive and the cache is full? After eviction, what position does each key get, and what position do the new queries get? Which keys can the second of three new queries see?"},
        {"level": 2, "kind": "analysis", "content": "Evict with two slices and a cat along the sequence dim, then cat the new k and v. Build a rope(x, positions) helper from arange positions. Rotate all cached keys and the queries, compute scores, mask entries where key position > query position with -inf, softmax, and matmul with values."},
    ],
    "model_connections": [
        "StreamingLLM (MIT HAN Lab) patches Llama, Falcon and GPT-NeoX attention this way and is used in TensorRT-LLM and SwiftInfer for endless chat.",
        "OpenAI's gpt-oss models add a learned sink logit per head, building the sink into the architecture instead of relying on the first tokens.",
    ],
    "pro_con_analysis": {
        "pros": ["Constant memory and latency per token for any stream length, with no retraining, and much lower perplexity than a plain window."],
        "cons": ["Evicted context is lost for good, and re-rotating every cached key at every step costs extra compute."],
    },
    "sources": [
        {"kind": "code", "url": "https://github.com/mit-han-lab/streaming-llm", "commit": "2e5042606d69933d88fbf909bd77907456b9b4dd", "path": "streaming_llm/kv_cache.py", "symbol": "StartRecentKVCache.evict_for_space", "license": "MIT", "adapted": "Keeping the first start_size tokens and the recent window that leaves room for the incoming tokens.", "simplifications": "One layer, no batch dimension, and eviction applied inside attend instead of by the generation loop."},
        {"kind": "code", "url": "https://github.com/mit-han-lab/streaming-llm", "commit": "2e5042606d69933d88fbf909bd77907456b9b4dd", "path": "streaming_llm/pos_shift/modify_llama.py", "symbol": "llama_pos_shift_attention_forward", "license": "MIT", "adapted": "Caching un-rotated keys, rotating keys at their cache positions and queries at the cache positions of the new tokens.", "simplifications": "No projections, grouped-query heads or attention-mask argument; the causal mask is built from positions."},
        {"kind": "paper", "url": "https://arxiv.org/abs/2309.17453", "section": "Section 3.2, Rolling KV Cache with Attention Sinks"},
    ],
    "tests": [
        {"name": "Evicts the middle and keeps sinks", "behavior": "attention.cache", "code": r"""
import torch
torch.manual_seed(0)
c = {fn}(start_size=2, recent_size=3)
ks = [torch.full((1, 1, 4), float(i)) for i in range(8)]
for i in range(8):
    out = c.attend(torch.randn(1, 1, 4), ks[i], ks[i])
    assert out.shape == (1, 1, 4)
assert c.keys[0, :, 0].tolist() == [0.0, 1.0, 5.0, 6.0, 7.0], c.keys[0, :, 0]
assert torch.equal(c.values, c.keys)
"""},
        {"name": "Positions come from cache slots", "visibility": "unshown", "behavior": "attention.masking", "failure_message": "Rotate keys at their cache positions 0..T-1 and new queries at T-n..T-1, and let each new query see only keys at or before its position.", "code": r"""
import math, torch
def rope(x, pos, base=10000.0):
    d = x.shape[-1]
    inv = 1.0 / base ** (torch.arange(0, d, 2, dtype=x.dtype) / d)
    ang = pos.to(x.dtype)[:, None] * torch.cat([inv, inv])[None, :]
    rh = torch.cat([-x[..., d // 2:], x[..., :d // 2]], dim=-1)
    return x * ang.cos() + rh * ang.sin()
def ref(q, K, V, base):
    T, n, d = K.shape[1], q.shape[1], q.shape[-1]
    kr = rope(K, torch.arange(T), base)
    qr = rope(q, torch.arange(T - n, T), base)
    s = qr @ kr.transpose(-1, -2) / math.sqrt(d)
    qpos = torch.arange(T - n, T)[:, None]
    s = s.masked_fill(torch.arange(T)[None, :] > qpos, float("-inf"))
    return torch.softmax(s, -1) @ V
torch.manual_seed(3)
c = {fn}(start_size=1, recent_size=4, base=100.0)
raw_k, raw_v = [], []
for n in [3, 2, 1, 4, 2]:
    q, k, v = (torch.randn(2, n, 6, dtype=torch.float64) for _ in range(3))
    L = len(raw_k)
    if L + n > 5:
        keep = list(range(1)) + list(range(L - 4 + n, L))
        raw_k = [raw_k[i] for i in keep]; raw_v = [raw_v[i] for i in keep]
    raw_k += list(k.unbind(1)); raw_v += list(v.unbind(1))
    K, V = torch.stack(raw_k, 1), torch.stack(raw_v, 1)
    out = c.attend(q, k, v)
    assert out.dtype == torch.float64
    torch.testing.assert_close(out, ref(q, K, V, 100.0))
    torch.testing.assert_close(c.keys, K)
"""},
        {"name": "Seeded long streams match the reference", "visibility": "unshown", "behavior": "attention.cache", "failure_message": "Evict [start_size, L - recent_size + n) before appending, keep un-rotated keys, and rotate by cache position every call.", "code": r"""
import math, random, torch
def rope(x, pos, base=10000.0):
    d = x.shape[-1]
    inv = 1.0 / base ** (torch.arange(0, d, 2, dtype=x.dtype) / d)
    ang = pos.to(x.dtype)[:, None] * torch.cat([inv, inv])[None, :]
    rh = torch.cat([-x[..., d // 2:], x[..., :d // 2]], dim=-1)
    return x * ang.cos() + rh * ang.sin()
for seed in (11, 45, 78):
    rng = random.Random(seed)
    torch.manual_seed(seed)
    s, r = rng.randint(0, 3), rng.randint(2, 6)
    c = {fn}(start_size=s, recent_size=r)
    ks, vs = [], []
    for step in range(25):
        n = rng.randint(1, r)
        q, k, v = (torch.randn(3, n, 8, dtype=torch.float64) for _ in range(3))
        if len(ks) + n > s + r:
            keep = list(range(s)) + list(range(len(ks) - r + n, len(ks)))
            ks = [ks[i] for i in keep]; vs = [vs[i] for i in keep]
        ks += list(k.unbind(1)); vs += list(v.unbind(1))
        K, V = torch.stack(ks, 1), torch.stack(vs, 1)
        T = K.shape[1]
        sc = rope(q, torch.arange(T - n, T)) @ rope(K, torch.arange(T)).transpose(-1, -2) / math.sqrt(8)
        sc = sc.masked_fill(torch.arange(T)[None, :] > torch.arange(T - n, T)[:, None], float("-inf"))
        torch.testing.assert_close(c.attend(q, k, v), torch.softmax(sc, -1) @ V, msg=f"{seed} {step}")
        assert c.keys.shape[1] <= s + r
"""},
        {"name": "Rejects chunks larger than the window", "visibility": "unshown", "behavior": "contract.signature", "failure_message": "n must satisfy 1 <= n <= recent_size.", "code": r"""
import torch
c = {fn}(start_size=1, recent_size=2)
for n in (0, 3):
    try:
        c.attend(torch.zeros(1, n, 2), torch.zeros(1, n, 2), torch.zeros(1, n, 2))
    except ValueError:
        pass
    else:
        raise AssertionError(f"accepted n={n}")
"""},
    ],
    "solution": '''import math

import torch


def _rope(x, positions, base):
    d = x.shape[-1]
    inv_freq = 1.0 / base ** (torch.arange(0, d, 2, dtype=x.dtype, device=x.device) / d)
    angle = positions.to(x.dtype)[:, None] * torch.cat([inv_freq, inv_freq])[None, :]
    rotated_half = torch.cat([-x[..., d // 2:], x[..., :d // 2]], dim=-1)
    return x * angle.cos() + rotated_half * angle.sin()


class StreamingLLMCache:
    def __init__(self, start_size, recent_size, base=10000.0):
        self.start_size = start_size
        self.recent_size = recent_size
        self.base = base
        self.keys = None
        self.values = None

    def attend(self, q, k, v):
        n = q.shape[1]
        if not 1 <= n <= self.recent_size:
            raise ValueError("need 1 <= n <= recent_size")
        if self.keys is None:
            self.keys, self.values = k[:, :0], v[:, :0]
        length = self.keys.shape[1]
        if length + n > self.start_size + self.recent_size:
            cut = length - self.recent_size + n
            self.keys = torch.cat([self.keys[:, :self.start_size], self.keys[:, cut:]], dim=1)
            self.values = torch.cat([self.values[:, :self.start_size], self.values[:, cut:]], dim=1)
        self.keys = torch.cat([self.keys, k], dim=1)
        self.values = torch.cat([self.values, v], dim=1)

        total = self.keys.shape[1]
        key_pos = torch.arange(total, device=q.device)
        query_pos = torch.arange(total - n, total, device=q.device)
        k_rot = _rope(self.keys, key_pos, self.base)
        q_rot = _rope(q, query_pos, self.base)
        scores = q_rot @ k_rot.transpose(-1, -2) / math.sqrt(q.shape[-1])
        scores = scores.masked_fill(key_pos[None, :] > query_pos[:, None], float("-inf"))
        return torch.softmax(scores, dim=-1) @ self.values
''',
    "interview_questions": interview(
        concept=[
            "What is an attention sink, and why does a plain sliding-window KV cache fall apart once the first tokens are evicted?",
            "What problem does StreamingLLM solve, and what problem does it explicitly not solve?",
        ],
        deep_dive=[
            "Walk through evict-for-space when the cache is full and n new tokens arrive. Which slice is dropped and why does it depend on n?",
            "Why does StreamingLLM give keys positions by cache slot instead of their original text positions? What happens with RoPE if you use text positions?",
            "Why must the cache store keys before RoPE in this design, and what does that cost per decoding step?",
        ],
        tradeoffs=[
            "Attention sinks plus a recent window versus H2O-style importance-based eviction versus a learned sink logit (gpt-oss): quality, cost and implementation complexity?",
            "Bounded-memory streaming versus retrieval over an external store of evicted context for long-running assistants?",
        ],
    ),
}
