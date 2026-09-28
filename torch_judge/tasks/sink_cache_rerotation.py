"""Hugging Face SinkCache: shift rotated keys to new positions by composing rotations."""

from ._interview import interview

TASK = {
    "title": "Sink Cache Key Re-rotation",
    "difficulty": "Hard",
    "version": 1,
    "function_name": "sink_cache_update",
    "description_en": r"""Hugging Face's `SinkCache` implements attention sinks for models that cache keys after RoPE. When tokens are evicted, the kept keys move to earlier slots, so their rotation must change. The un-rotated keys are gone; instead, rotate the cached keys backwards by the shift. Implement one cache update.

**Signature:** `sink_cache_update(key_cache, value_cache, k, v, window_length, num_sink_tokens, base=10000.0) -> (keys, values)`

**Shapes.** `key_cache` and `value_cache` are `[heads, L, d]` (L may be 0); `k` and `v` are `[heads, n, d]`; `d` is even. Every cached key is already rotated at its slot index. `k` is not rotated yet. Do not modify the inputs.

**RoPE** of `x` at position `p`: `inv_freq[i] = 1 / base ** (2i / d)` for `i < d/2`; `angle = p * cat(inv_freq, inv_freq)`; result `x * cos(angle) + rotate_half(x) * sin(angle)`, where `rotate_half(x) = cat(-x[..., d/2:], x[..., :d/2])`.

**Update:**
1. Raise `ValueError` unless `1 <= n <= window_length - num_sink_tokens`.
2. Let `shift = max(0, L + n - window_length)`. If `shift > 0`, drop cached slots `[num_sink_tokens, num_sink_tokens + shift)`, keep the sinks unchanged, and rotate every kept non-sink key back by `shift` positions: `x * cos(a) - rotate_half(x) * sin(a)` with `a = shift * cat(inv_freq, inv_freq)`.
3. Rotate `k` at the slots it will occupy, `T-n .. T-1`, where `T` is the new length, and append `k` and `v`.
4. Return the new keys and values. Afterwards every key is rotated at its slot index, and `T <= window_length`.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why rotating back works.** RoPE rotates each pair of dimensions by an angle proportional to position, and rotations compose by adding angles. A key rotated at slot `p` and then rotated by `-shift` equals the original key rotated at `p - shift`, with no need for the original.

**Why this beats rotating every step.** The previous exercise re-rotates every cached key at every step. Here keys are rotated once when cached and corrected only on eviction, which works with the standard attention kernel and the model's usual RoPE call.

**Numerics.** Each correction adds rounding error; Hugging Face computes the correction's cos and sin in float32, and in half precision the error accumulates over very long streams.

**Status.** Recent Transformers releases no longer ship `SinkCache` in the core library; the rotation identity is the part worth knowing.""",
    "advisory_prerequisites": ["attention_sink_cache", "rope"],
    "hints": [
        {"level": 1, "kind": "questions", "content": "If a key was rotated at slot 7 and now belongs at slot 4, what single rotation fixes it? Which slots are dropped when n new tokens overflow the window by shift? Which keys must not be rotated?"},
        {"level": 2, "kind": "analysis", "content": "Compute inv_freq and a helper that rotates by an angle tensor. If shift > 0, split the cache into sinks and cache[:, sinks + shift:], rotate the latter by -shift, and cat. Then rotate k at arange(T - n, T) and cat keys and values."},
    ],
    "model_connections": [
        "transformers' SinkCache re-rotated kept keys with _get_rerotation_cos_sin; the attention_sinks package did the same for many Hugging Face models.",
        "Any RoPE cache that moves tokens, such as context shifting in llama.cpp, uses the same composition to avoid recomputing keys.",
    ],
    "pro_con_analysis": {
        "pros": ["Keys are rotated once and corrected only on eviction, so the attention kernel and model code stay unchanged."],
        "cons": ["Each correction adds rounding error, and it only works for position encodings that compose, such as RoPE, not learned absolute embeddings."],
    },
    "sources": [
        {"kind": "code", "url": "https://github.com/huggingface/transformers", "commit": "2ef31dec1676249d26044a8aa8abe33dbecf0d10", "path": "src/transformers/cache_utils.py", "symbol": "SinkCache.update and SinkCache._get_rerotation_cos_sin", "license": "Apache-2.0", "adapted": "Evicting the oldest non-sink keys, re-rotating the kept ones backwards with the angle-difference identity, and keeping sinks fixed.", "simplifications": "A stateless function over one layer without batch; new keys arrive un-rotated and are rotated at their final slot; the shift is exactly the number of evicted tokens; no partial rotation."},
        {"kind": "paper", "url": "https://arxiv.org/abs/2309.17453", "section": "Section 3.2, Rolling KV Cache with Attention Sinks"},
    ],
    "tests": [
        {"name": "Kept keys match keys rotated at their new slots", "behavior": "attention.cache", "code": r"""
import torch
def rope(x, pos, base=10000.0):
    d = x.shape[-1]
    inv = 1.0 / base ** (torch.arange(0, d, 2, dtype=x.dtype) / d)
    ang = pos.to(x.dtype)[:, None] * torch.cat([inv, inv])[None, :]
    return x * ang.cos() + torch.cat([-x[..., d // 2:], x[..., :d // 2]], -1) * ang.sin()
torch.manual_seed(0)
raw = torch.randn(1, 6, 4, dtype=torch.float64)
K, V = torch.zeros(1, 0, 4, dtype=torch.float64), torch.zeros(1, 0, 4, dtype=torch.float64)
for i in range(6):
    K, V = {fn}(K, V, raw[:, i:i + 1], raw[:, i:i + 1], window_length=4, num_sink_tokens=1)
kept = raw[:, [0, 3, 4, 5]]
torch.testing.assert_close(K, rope(kept, torch.arange(4)))
torch.testing.assert_close(V, kept)
"""},
        {"name": "Seeded streams stay consistent", "visibility": "unshown", "behavior": "attention.cache", "failure_message": "Drop slots [sinks, sinks + shift), rotate kept non-sink keys back by shift, leave sinks alone, and rotate new keys at slots T-n..T-1.", "code": r"""
import random, torch
def rope(x, pos, base):
    d = x.shape[-1]
    inv = 1.0 / base ** (torch.arange(0, d, 2, dtype=x.dtype) / d)
    ang = pos.to(x.dtype)[:, None] * torch.cat([inv, inv])[None, :]
    return x * ang.cos() + torch.cat([-x[..., d // 2:], x[..., :d // 2]], -1) * ang.sin()
for seed in (9, 40, 73):
    rng = random.Random(seed)
    torch.manual_seed(seed)
    w, s = rng.randint(3, 8), rng.randint(0, 2)
    base = rng.choice([10000.0, 50.0])
    K = V = torch.zeros(2, 0, 6, dtype=torch.float64)
    ks, vs = [], []
    for step in range(30):
        n = rng.randint(1, w - s)
        k, v = torch.randn(2, n, 6, dtype=torch.float64), torch.randn(2, n, 6, dtype=torch.float64)
        before = (K.clone(), V.clone(), k.clone())
        K2, V2 = {fn}(K, V, k, v, window_length=w, num_sink_tokens=s, base=base)
        assert torch.equal(K, before[0]) and torch.equal(V, before[1]) and torch.equal(k, before[2]), "inputs modified"
        shift = max(0, len(ks) + n - w)
        ks = ks[:s] + ks[s + shift:] + list(k.unbind(1)); vs = vs[:s] + vs[s + shift:] + list(v.unbind(1))
        want_k = rope(torch.stack(ks, 1), torch.arange(len(ks)), base)
        torch.testing.assert_close(K2, want_k, msg=f"{seed} {step}")
        torch.testing.assert_close(V2, torch.stack(vs, 1))
        K, V = K2, V2
"""},
        {"name": "Rejects bad chunk sizes", "visibility": "unshown", "behavior": "contract.signature", "failure_message": "n must satisfy 1 <= n <= window_length - num_sink_tokens.", "code": r"""
import torch
z = torch.zeros(1, 0, 2)
for n in (0, 3):
    try:
        {fn}(z, z, torch.zeros(1, n, 2), torch.zeros(1, n, 2), window_length=4, num_sink_tokens=2)
    except ValueError:
        pass
    else:
        raise AssertionError(f"accepted n={n}")
"""},
    ],
    "solution": '''import torch


def _rotate(x, angle):
    d = x.shape[-1]
    rotated_half = torch.cat([-x[..., d // 2:], x[..., :d // 2]], dim=-1)
    return x * angle.cos() + rotated_half * angle.sin()


def sink_cache_update(key_cache, value_cache, k, v, window_length, num_sink_tokens, base=10000.0):
    n, d = k.shape[1], k.shape[-1]
    if not 1 <= n <= window_length - num_sink_tokens:
        raise ValueError("need 1 <= n <= window_length - num_sink_tokens")
    inv_freq = 1.0 / base ** (torch.arange(0, d, 2, dtype=k.dtype, device=k.device) / d)
    freqs = torch.cat([inv_freq, inv_freq])

    shift = max(0, key_cache.shape[1] + n - window_length)
    if shift > 0:
        sinks = key_cache[:, :num_sink_tokens]
        kept = _rotate(key_cache[:, num_sink_tokens + shift:], -shift * freqs)
        key_cache = torch.cat([sinks, kept], dim=1)
        value_cache = torch.cat([value_cache[:, :num_sink_tokens], value_cache[:, num_sink_tokens + shift:]], dim=1)

    start = key_cache.shape[1]
    positions = torch.arange(start, start + n, dtype=k.dtype, device=k.device)
    new_keys = _rotate(k, positions[:, None] * freqs[None, :])
    return torch.cat([key_cache, new_keys], dim=1), torch.cat([value_cache, v], dim=1)
''',
    "interview_questions": interview(
        concept=[
            "Why can a RoPE key rotated at one position be moved to another position without the original key?",
            "What does a sink cache keep and what does it evict when the window overflows?",
        ],
        deep_dive=[
            "Derive the correction: a key rotated at slot p must end up rotated at p - shift. What angle do you apply and why are sinks excluded?",
            "Compare storing keys before RoPE and rotating every step (StreamingLLM) with storing them after RoPE and correcting on eviction (SinkCache). What does each cost per token?",
            "Why does repeated re-rotation accumulate error, and how would you bound it in half precision?",
        ],
        tradeoffs=[
            "Correcting rotated keys on eviction versus caching pre-RoPE keys and fusing RoPE into the attention kernel: kernel complexity, memory traffic and numerics?",
            "Shipping a cache strategy inside the modeling library versus as optional Hub code: maintenance cost versus discoverability?",
        ],
    ),
}
