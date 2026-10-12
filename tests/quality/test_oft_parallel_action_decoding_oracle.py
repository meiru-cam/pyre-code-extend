"""The oft_parallel_action_decoding reference, checked against OpenVLA-OFT and its patched transformers.

The mutation gate proves the evaluator rejects wrong implementations; this file proves
the contract itself matches upstream, using the transcription in
`torch_judge/harness/embodied_frontier/vendored/openvla_oft_parallel_decoding.py`.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from torch_judge.harness.embodied_frontier.vendored.openvla_oft_parallel_decoding import (
    ACTION_TOKEN_BEGIN_IDX,
    finetune_actions_hidden_states,
    get_current_action_mask,
    get_next_actions_mask,
    hf_causal_mask,
    oft_bidirectional_mask,
    predict_action,
    prepare_input_for_action_prediction,
    prepare_labels_for_action_prediction,
)
from torch_judge.tasks import get_task

TASK_ID = "oft_parallel_action_decoding"
SEEDS = (0, 1, 2, 3, 4, 5, 6)
VOCAB, N_BINS = 48, 8
BEGIN = VOCAB - N_BINS - 1


def reference_class():
    task = get_task(TASK_ID)
    assert task is not None, f"unknown task {TASK_ID!r}"
    namespace: dict = {}
    exec(task["solution"], namespace)
    return namespace[task["function_name"]]


def world(seed: int, hidden: int = 6, vocab: int = VOCAB, n_bins: int = N_BINS):
    """A one-layer attention decoder, an embedding table and a call log."""
    g = torch.Generator().manual_seed(seed)
    table = torch.randn(97, hidden, generator=g, dtype=torch.float64)
    pos = torch.randn(512, hidden, generator=g, dtype=torch.float64) * 0.7
    wq, wk = (torch.randn(hidden, hidden, generator=g, dtype=torch.float64) / math.sqrt(hidden) for _ in range(2))
    wv = torch.randn(hidden, hidden, generator=g, dtype=torch.float64)
    wout = torch.randn(hidden, vocab, generator=g, dtype=torch.float64) * 2
    boost = torch.zeros(vocab, dtype=torch.float64)
    boost[vocab - n_bins:] = 4.0
    log: list = []

    def embed(ids):
        return table[ids % 97]

    def decoder(x, bias):
        log.append(bias.clone())
        z = x + pos[: x.shape[1]]
        scores = (z @ wq) @ (z @ wk).transpose(-1, -2) / math.sqrt(hidden) + bias[:, 0].to(z.dtype)
        h = torch.tanh(z + torch.softmax(scores, dim=-1) @ (z @ wv))
        return h, h @ wout + boost

    return embed, decoder, log


def prompt(seed: int, length: int, ends_with_empty: bool):
    g = torch.Generator().manual_seed(1000 + seed)
    ids = torch.randint(3, 900, (1, length), generator=g)
    ids[:, 0] = 1
    if ends_with_empty:
        ids[:, -1] = 29871
    return ids, torch.ones(1, length, dtype=torch.long)


SHAPES = ((2, 3, 4, 3), (3, 2, 2, 5), (1, 4, 3, 2), (4, 1, 5, 4), (2, 2, 1, 3), (3, 3, 2, 6), (8, 7, 3, 4))


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("ends_with_empty", (False, True))
def test_discrete_predict_matches_upstream(seed, ends_with_empty):
    K, D, P, L = SHAPES[seed]
    ids, mask = prompt(seed, L, ends_with_empty)
    patches = torch.randn(1, P, 6, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)
    embed, decoder, got_log = world(seed)
    got = reference_class()(K, D, VOCAB, N_BINS).predict(decoder, embed, ids, mask, patches)
    embed, decoder, want_log = world(seed)
    want, _ = predict_action(ids, mask, patches, embed, lambda x, m: decoder(x, m), D, K,
                             vocab_size=VOCAB, n_action_bins=N_BINS, action_token_begin_idx=BEGIN)
    assert got.shape == (1, K, D)
    # torch.linspace and np.linspace differ by an ulp; bins are 2 / (n_bins - 1) apart.
    torch.testing.assert_close(got[0], want, rtol=0, atol=1e-12)
    torch.testing.assert_close(got_log[0], want_log[0], rtol=0, atol=0)
    assert len(got_log) == len(want_log) == 1


@pytest.mark.parametrize("seed", SEEDS)
def test_discrete_predict_reaches_many_bins(seed):
    """Guard against a degenerate decoder whose argmax always clamps to one bin."""
    K, D, P, L = SHAPES[seed]
    ids, mask = prompt(seed, L, False)
    patches = torch.randn(1, P, 6, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)
    embed, decoder, _ = world(seed)
    got = reference_class()(K, D, VOCAB, N_BINS).predict(decoder, embed, ids, mask, patches)
    if K * D >= 6:
        assert got.unique().numel() >= 2


@pytest.mark.parametrize("seed", SEEDS)
def test_l1_predict_matches_upstream(seed):
    K, D, P, L = SHAPES[seed]
    ids, mask = prompt(seed, L, seed % 2 == 0)
    patches = torch.randn(1, P, 6, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)
    w = torch.randn(D * 6, D, generator=torch.Generator().manual_seed(50 + seed), dtype=torch.float64)

    def head(h):
        return torch.tanh(h @ w)

    embed, decoder, _ = world(seed)
    got = reference_class()(K, D, VOCAB, N_BINS).predict(decoder, embed, ids, mask, patches, action_head=head)
    embed, decoder, _ = world(seed)
    want, _ = predict_action(ids, mask, patches, embed, lambda x, m: decoder(x, m), D, K,
                             vocab_size=VOCAB, n_action_bins=N_BINS, action_token_begin_idx=BEGIN,
                             action_head=head)
    torch.testing.assert_close(got[0], want, rtol=0, atol=0)


def test_default_constants_match_llama2():
    """vocab_size - n_bins - 1 with the defaults is upstream's hard-coded ACTION_TOKEN_BEGIN_IDX."""
    dec = reference_class()(8, 7)
    assert dec.action_begin == ACTION_TOKEN_BEGIN_IDX == 31743
    ids, mask = torch.tensor([[1, 450, 3802]]), torch.ones(1, 3, dtype=torch.long)
    got_ids, got_mask, got_labels, n_prompt = dec.prepare_inputs(ids, mask)
    up_ids = torch.cat([ids, torch.tensor([[29871]])], dim=1)
    labels = torch.full_like(up_ids, -100)
    up_ids, up_mask = prepare_input_for_action_prediction(up_ids, mask, 7, 8)
    up_labels = prepare_labels_for_action_prediction(labels, up_ids)
    assert torch.equal(got_ids, up_ids) and torch.equal(got_mask, up_mask) and torch.equal(got_labels, up_labels)
    assert n_prompt == 3


@pytest.mark.parametrize("seed", SEEDS)
def test_action_masks_match_upstream(seed):
    g = torch.Generator().manual_seed(seed)
    D = 1 + seed % 4
    labels = torch.randint(BEGIN - 6, VOCAB, (4, 15), generator=g)
    labels[torch.rand(4, 15, generator=g) < 0.4] = -100
    cur, nxt = reference_class()(2, D, VOCAB, N_BINS).action_masks(labels)
    assert torch.equal(cur, get_current_action_mask(labels, D, BEGIN))
    assert torch.equal(nxt, get_next_actions_mask(labels, D, BEGIN))


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64, torch.bfloat16))
def test_attention_bias_matches_patched_transformers(seed, dtype):
    g = torch.Generator().manual_seed(seed)
    B, S = 3, 5 + seed
    mask = torch.ones(B, S, dtype=torch.long)
    mask[0, S - 1 - seed % 3:] = 0  # right padding, as the OFT collator uses
    mask[1, : 1 + seed % 2] = 0  # left padding
    mask[2] = (torch.rand(S, generator=g) > 0.3).long()
    mask[2, -1] = 1
    got = reference_class()(2, 2).attention_bias(mask, dtype)
    want = oft_bidirectional_mask(hf_causal_mask(mask, torch.zeros(B, S, 1, dtype=dtype)))
    assert got.shape == want.shape and got.dtype == want.dtype
    assert torch.equal(got, want)


def test_attention_bias_without_padding_is_all_zero():
    """With no padding, the SDPA path passes no mask and is_causal=False: full attention."""
    mask = torch.ones(2, 7, dtype=torch.long)
    want = oft_bidirectional_mask(hf_causal_mask(mask, torch.zeros(2, 7, 1)))
    got = reference_class()(2, 2).attention_bias(mask)
    assert torch.equal(got, want) and (got == 0).all()


def test_last_row_trick_differs_from_plain_causal():
    mask = torch.tensor([[1, 1, 1, 1, 0, 0]])
    causal = hf_causal_mask(mask, torch.zeros(1, 6, 1))
    patched = oft_bidirectional_mask(causal)
    assert (causal[0, 0, 0, 1:] != 0).all()  # causal: the first query sees only itself
    assert (patched[0, 0, 0, :4] == 0).all() and (patched[0, 0, 0, 4:] != 0).all()


@pytest.mark.parametrize("seed", SEEDS)
def test_train_readout_matches_finetune(seed):
    g = torch.Generator().manual_seed(seed)
    K, D, P, H = 1 + seed % 3, 1 + (seed + 1) % 3, 2 + seed % 2, 4
    rows = []
    for b in range(3):
        pre = 2 + (b + seed) % 4
        rows.append([-100] * pre + torch.randint(BEGIN + 1, VOCAB, (K * D,), generator=g).tolist() + [2])
    L = max(len(r) for r in rows) + 1
    labels = torch.tensor([r + [-100] * (L - len(r)) for r in rows])
    last = torch.randn(3, P + L, H, generator=g, dtype=torch.float64)
    got = reference_class()(K, D, VOCAB, N_BINS).train_action_hidden(last, labels, P)
    want = finetune_actions_hidden_states(last, labels, P, D, K, BEGIN)
    assert torch.equal(got, want)


def test_inference_and_training_use_the_same_offset():
    """The slot at multimodal index i is read from index i - 1 in both code paths."""
    K, D, P = 2, 3, 4
    dec = reference_class()(K, D, VOCAB, N_BINS)
    ids, mask = torch.tensor([[1, 7, 8, 29871]]), torch.ones(1, 4, dtype=torch.long)
    full_ids, _, labels, n_prompt = dec.prepare_inputs(ids, mask)
    S = P + full_ids.shape[1]
    last = torch.arange(S, dtype=torch.float64).reshape(1, S, 1)
    train = dec.train_action_hidden(last, labels, P)

    def decoder(x, bias):
        return last.expand(x.shape[0], -1, -1), torch.zeros(x.shape[0], S, VOCAB, dtype=x.dtype)

    infer = dec.predict(decoder, lambda i: torch.zeros(*i.shape, 1, dtype=torch.float64), ids, mask,
                        torch.zeros(1, P, 1, dtype=torch.float64), action_head=lambda h: h)
    first_slot = 1 + P + n_prompt
    assert train.reshape(-1).tolist() == list(range(first_slot - 1, first_slot - 1 + K * D))
    assert infer.reshape(-1).tolist() == train.reshape(-1).tolist()


# --------------------------------------------------------------------------
# Deliberate divergences from upstream, pinned so a change is noticed.


def test_divergence_batch_append_of_empty_token():
    """Upstream concatenates a (1, 1) tensor and fails for B > 1; the exercise appends to every row."""
    ids = torch.tensor([[1, 5, 6], [1, 7, 8]])
    mask = torch.ones(2, 3, dtype=torch.long)
    with pytest.raises(RuntimeError):
        predict_action(ids, mask, torch.zeros(2, 1, 6, dtype=torch.float64), world(0)[0],
                       lambda x, m: world(0)[1](x, m), 2, 2, vocab_size=VOCAB, n_action_bins=N_BINS,
                       action_token_begin_idx=BEGIN)
    got_ids = reference_class()(2, 2, VOCAB, N_BINS).prepare_inputs(ids, mask)[0]
    assert got_ids[:, 3].tolist() == [29871, 29871]


def test_divergence_discrete_dtype_follows_logits():
    """Upstream returns numpy float64 bin centers; the exercise uses the logits dtype."""
    centers = np.linspace(-1, 1, N_BINS)
    centers = (centers[:-1] + centers[1:]) / 2.0
    dec = reference_class()(1, 2, VOCAB, N_BINS)

    def decoder(x, bias):
        logits = torch.zeros(x.shape[0], x.shape[1], VOCAB, dtype=torch.float32)
        logits[..., VOCAB - 3] = 1.0
        return x, logits

    out = dec.predict(decoder, lambda i: torch.zeros(*i.shape, 2, dtype=torch.float32),
                      torch.tensor([[1, 5]]), torch.ones(1, 2, dtype=torch.long), torch.zeros(1, 2, 2))
    assert out.dtype == torch.float32
    assert torch.allclose(out, torch.full((1, 1, 2), float(centers[2])))
