"""coconut_latent_forward's reference solution, checked against upstream Coconut.forward.

Upstream reuses a KV cache and computes one new position per pass; the exercise
recomputes the full sequence each pass. On the batches upstream supports
(contiguous latents that start at the same index in every item) both must give
the same logits and the same gradients. The batches upstream does not support,
and the latent-at-position-0 case, are pinned as divergences below.

`transformers` is not installed in this repo's environment, so the oracle drives a
small torch-only GPT-2 stub with the Hugging Face call interface
(`inputs_embeds`, `attention_mask`, `position_ids`, `past_key_values`,
`output_hidden_states`, and legacy tuple caches).
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_judge.harness.reasoning.vendored.coconut_model import Coconut
from torch_judge.tasks import get_task

TASK_ID = "coconut_latent_forward"
VOCAB, LATENT, START, END, EOS = 16, 13, 12, 14, 15
ATOL = 1e-9


def reference():
    task = get_task(TASK_ID)
    assert task is not None, f"unknown task {TASK_ID!r}"
    namespace: dict = {}
    exec(task["solution"], namespace)
    return namespace[task["function_name"]]


class StubGPT2(nn.Module):
    """Pre-norm GPT-2 in float64 with learned positions, tied head and a KV cache."""

    def __init__(self, seed: int, d: int = 16, n_head: int = 2, n_layer: int = 2, n_pos: int = 64):
        super().__init__()
        g = torch.Generator().manual_seed(seed)

        def p(*shape, scale=0.3, base=0.0):
            return nn.Parameter(base + scale * torch.randn(*shape, generator=g, dtype=torch.float64))

        self.d, self.n_head, self.n_layer = d, n_head, n_layer
        self.wte = nn.Embedding(VOCAB, d).double()
        with torch.no_grad():
            self.wte.weight.copy_(torch.randn(VOCAB, d, generator=g, dtype=torch.float64))
        self.wpe = p(n_pos, d, scale=0.1)
        self.ln1_w, self.ln1_b, self.ln2_w, self.ln2_b = (
            nn.ParameterList([p(d, scale=0.1, base=1.0 if i % 2 == 0 else 0.0) for _ in range(n_layer)]) for i in range(4)
        )
        self.c_attn_w = nn.ParameterList([p(d, 3 * d) for _ in range(n_layer)])
        self.c_attn_b = nn.ParameterList([p(3 * d, scale=0.05) for _ in range(n_layer)])
        self.c_proj_w = nn.ParameterList([p(d, d) for _ in range(n_layer)])
        self.fc_w = nn.ParameterList([p(d, 4 * d) for _ in range(n_layer)])
        self.fc_proj_w = nn.ParameterList([p(4 * d, d, scale=0.15) for _ in range(n_layer)])
        self.lnf_w, self.lnf_b = p(d, scale=0.1, base=1.0), p(d, scale=0.1)

    def forward(self, inputs_embeds, attention_mask=None, position_ids=None, past_key_values=None, output_hidden_states=False):
        B, n, d = inputs_embeds.shape
        past_len = past_key_values[0][0].shape[2] if past_key_values else 0
        if attention_mask is not None:
            assert attention_mask.shape == (B, past_len + n) and bool(attention_mask.all()), "stub supports no padding"
        if position_ids is None:
            position_ids = torch.arange(past_len, past_len + n).expand(B, n)
        h = inputs_embeds + self.wpe[position_ids]
        hidden_states, presents = [h], []
        H, dh = self.n_head, d // self.n_head
        query_pos = torch.arange(past_len, past_len + n)[:, None]
        key_pos = torch.arange(past_len + n)[None, :]
        future = key_pos > query_pos
        for layer in range(self.n_layer):
            x = F.layer_norm(h, (d,), self.ln1_w[layer], self.ln1_b[layer])
            q, k, v = (x @ self.c_attn_w[layer] + self.c_attn_b[layer]).split(d, dim=-1)
            q, k, v = (t.view(B, n, H, dh).transpose(1, 2) for t in (q, k, v))
            if past_key_values:
                k = torch.cat([past_key_values[layer][0], k], dim=2)
                v = torch.cat([past_key_values[layer][1], v], dim=2)
            presents.append((k, v))
            scores = (q @ k.transpose(-1, -2) / math.sqrt(dh)).masked_fill(future, float("-inf"))
            attn = (scores.softmax(-1) @ v).transpose(1, 2).reshape(B, n, d)
            h = h + attn @ self.c_proj_w[layer]
            x = F.layer_norm(h, (d,), self.ln2_w[layer], self.ln2_b[layer])
            h = h + F.gelu(x @ self.fc_w[layer]) @ self.fc_proj_w[layer]
            if layer < self.n_layer - 1:
                hidden_states.append(h)
        h = F.layer_norm(h, (d,), self.lnf_w, self.lnf_b)
        hidden_states.append(h)  # like HF GPT-2, the last entry is after ln_f
        return SimpleNamespace(
            logits=h @ self.wte.weight.T,
            hidden_states=tuple(hidden_states) if output_hidden_states else None,
            past_key_values=presents,
        )


def exercise_model(stub: StubGPT2):
    """The exercise's interface: model(inputs_embeds) -> (logits, last_hidden), full sequence."""

    def model(inputs_embeds):
        out = stub(inputs_embeds=inputs_embeds, output_hidden_states=True)
        return out.logits, out.hidden_states[-1]

    return model


def upstream_forward(stub: StubGPT2, input_ids: torch.Tensor):
    coconut = Coconut(stub, stub.wte, LATENT, START, END, EOS)
    B, T = input_ids.shape
    return coconut(
        input_ids,
        attention_mask=torch.ones(B, T, dtype=torch.long),
        labels=input_ids.clone(),
        position_ids=torch.arange(T).expand(B, T),
    )


def aligned_batch(seed: int, batch: int, start: int, tail: int, counts: list[int]) -> torch.Tensor:
    """Every item's latents start at `start` and are contiguous, as Coconut's collator guarantees."""
    g = torch.Generator().manual_seed(seed)
    T = start + max(counts) + tail
    ids = torch.randint(0, 12, (batch, T), generator=g)
    for b, k in enumerate(counts):
        ids[b, start - 1] = START
        ids[b, start:start + k] = LATENT
        ids[b, start + k] = END
    return ids


CASES = [
    (0, 1, 3, 4, [2]),
    (1, 3, 4, 5, [3, 1, 2]),
    (2, 2, 2, 3, [4, 4]),
    (3, 4, 5, 2, [0, 2, 3, 1]),
    (4, 2, 6, 6, [5, 0]),
    (5, 3, 1, 4, [1, 1, 1]),
    (6, 1, 7, 3, [6]),
]


@pytest.mark.parametrize("seed,batch,start,tail,counts", CASES)
def test_full_recompute_matches_upstream_kv_cached_logits(seed, batch, start, tail, counts):
    ids = aligned_batch(seed, batch, start, tail, counts)
    stub = StubGPT2(seed)
    got = reference()(exercise_model(stub), stub.wte, ids, LATENT)
    want = upstream_forward(stub, ids).logits
    assert got.shape == want.shape == (batch, ids.shape[1], VOCAB)
    assert torch.allclose(got, want, atol=ATOL, rtol=0), (got - want).abs().max()


@pytest.mark.parametrize("seed,batch,start,tail,counts", CASES[:5])
def test_full_recompute_matches_upstream_gradients(seed, batch, start, tail, counts):
    ids = aligned_batch(seed, batch, start, tail, counts)
    ours, theirs = StubGPT2(seed), StubGPT2(seed)

    logits = reference()(exercise_model(ours), ours.wte, ids, LATENT)
    F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB), ids[:, 1:].reshape(-1)).backward()
    upstream_forward(theirs, ids).loss.backward()

    for (name, a), (_, b) in zip(ours.named_parameters(), theirs.named_parameters()):
        assert a.grad is not None and b.grad is not None, name
        assert torch.allclose(a.grad, b.grad, atol=ATOL, rtol=0), (name, (a.grad - b.grad).abs().max())
    # Thoughts flow back into the prefix: the cross-pass path carries real gradient.
    assert ours.wte.weight.grad.abs().sum() > 0


def test_no_latents_is_one_plain_forward_like_upstream():
    ids = torch.randint(0, 12, (3, 9), generator=torch.Generator().manual_seed(7))
    stub = StubGPT2(7)
    got = reference()(exercise_model(stub), stub.wte, ids, LATENT)
    want = upstream_forward(stub, ids).logits
    assert torch.allclose(got, want, atol=ATOL, rtol=0)


def test_stub_kv_cache_matches_full_forward():
    """Sanity check of the stub itself: cached incremental decoding equals one full pass."""
    stub = StubGPT2(8)
    x = torch.randn(2, 7, 16, generator=torch.Generator().manual_seed(8), dtype=torch.float64)
    full = stub(x, output_hidden_states=True)
    head = stub(x[:, :4], output_hidden_states=True)
    tail = stub(x[:, 4:], position_ids=torch.arange(4, 7).expand(2, 3), past_key_values=head.past_key_values, output_hidden_states=True)
    assert torch.allclose(torch.cat([head.logits, tail.logits], 1), full.logits, atol=1e-12)


def test_pins_divergence_unaligned_latents():
    """Upstream indexes hidden states from the batch's earliest latent and fails; the exercise matches per-item upstream runs."""
    ids = torch.tensor([
        [1, 2, START, LATENT, LATENT, END, 3, 4, 5],
        [1, 2, 3, 4, START, LATENT, END, 6, 7],
    ])
    stub = StubGPT2(9)
    with pytest.raises(IndexError):
        upstream_forward(stub, ids)
    got = reference()(exercise_model(stub), stub.wte, ids, LATENT)
    for b in range(2):
        want = upstream_forward(stub, ids[b:b + 1]).logits
        assert torch.allclose(got[b:b + 1], want, atol=ATOL, rtol=0), b


def test_pins_divergence_non_contiguous_latents():
    """Upstream computes one new position per pass, so a gap between latents breaks it; the exercise handles gaps."""
    ids = torch.tensor([[1, 2, LATENT, 3, LATENT, 4, 5]])
    stub = StubGPT2(10)
    with pytest.raises(IndexError):
        upstream_forward(stub, ids)
    model = exercise_model(stub)
    got = reference()(model, stub.wte, ids, LATENT)
    # Hand-rolled two-pass check.
    e = stub.wte(ids)
    e = e.index_put((torch.tensor([0]), torch.tensor([2])), model(e)[1][0:1, 1])
    e = e.index_put((torch.tensor([0]), torch.tensor([4])), model(e)[1][0:1, 3])
    assert torch.allclose(got, model(e)[0], atol=ATOL, rtol=0)


def test_pins_divergence_latent_at_position_zero():
    """Upstream would read position -1 (or run an empty prefix); the exercise rejects the input."""
    ids = torch.tensor([[LATENT, 1, 2, 3]])
    stub = StubGPT2(11)
    with pytest.raises(ValueError):
        reference()(exercise_model(stub), stub.wte, ids, LATENT)
