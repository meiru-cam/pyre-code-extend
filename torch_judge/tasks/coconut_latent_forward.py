"""Coconut continuous-thought feedback: each latent slot takes the last hidden state before it."""

from ._interview import interview

# A tiny seeded causal LM shared by every test. It records every call so a test can
# inspect what the learner fed in. Built in float64 so comparisons can be tight.
_MODEL = r"""
import torch, torch.nn as nn, torch.nn.functional as F

class TinyCausalLM(nn.Module):
    def __init__(self, seed, vocab=12, d=8, layers=2, max_len=40):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        p = lambda *s: nn.Parameter(torch.randn(*s, generator=g, dtype=torch.float64) * 0.5)
        self.embed = nn.Embedding(vocab, d).double()
        with torch.no_grad():
            self.embed.weight.copy_(torch.randn(vocab, d, generator=g, dtype=torch.float64))
        self.d, self.layers = d, layers
        self.pos = p(max_len, d)
        self.gain = nn.Parameter(1 + 0.1 * torch.randn(d, generator=g, dtype=torch.float64))
        self.wq, self.wk, self.wv, self.wo, self.w1, self.w2 = (
            nn.ParameterList([p(d, d) for _ in range(layers)]) for _ in range(6))
        self.head = p(d, vocab)
        self.calls, self.hiddens = [], []

    def forward(self, inputs_embeds):
        assert inputs_embeds.dim() == 3 and inputs_embeds.shape[-1] == self.d, inputs_embeds.shape
        self.calls.append(inputs_embeds.detach().clone())
        T = inputs_embeds.shape[1]
        h = inputs_embeds * self.gain + self.pos[:T]  # saves inputs_embeds for backward, like an input norm
        future = torch.ones(T, T, dtype=torch.bool).triu(1)
        for l in range(self.layers):
            x = F.layer_norm(h, (self.d,))
            s = (x @ self.wq[l]) @ (x @ self.wk[l]).transpose(-1, -2) / self.d ** 0.5
            h = h + s.masked_fill(future, float('-inf')).softmax(-1) @ (x @ self.wv[l]) @ self.wo[l]
            h = h + torch.tanh(F.layer_norm(h, (self.d,)) @ self.w1[l]) @ self.w2[l]
        last = F.layer_norm(h, (self.d,))
        self.hiddens.append(last.detach().clone())
        return last @ self.head, last

LAT = 11
"""

# Independent full-recompute oracle, written as a plain loop over items.
_ORACLE = r"""
def oracle(model, embed, ids, latent_id):
    E = embed(ids)
    B, T = ids.shape
    lists = [[t for t in range(T) if int(ids[b, t]) == latent_id] for b in range(B)]
    n = max([len(l) for l in lists] + [0])
    for i in range(n):
        _, H = model(E)
        rows = []
        for b in range(B):
            row = [E[b, t] for t in range(T)]
            if len(lists[b]) > i:
                row[lists[b][i]] = H[b, lists[b][i] - 1]
            rows.append(torch.stack(row))
        E = torch.stack(rows)
    return model(E)[0]
"""


def _case(body: str, *, oracle: bool = False) -> str:
    return _MODEL + (_ORACLE if oracle else "") + body


TASK = {
    "title": "Coconut Continuous-Thought Forward",
    "difficulty": "Hard",
    "version": 1,
    "function_name": "coconut_forward",
    "description_en": r"""Coconut lets a language model reason in latent space. The question is followed by placeholder `<|latent|>` tokens, and each placeholder's input embedding is replaced by the model's own last hidden state at the position just before it, a "continuous thought". Implement this forward pass with full recomputation.

**Signature:** `coconut_forward(model, embed, input_ids, latent_id) -> Tensor`

**Parameters:**
- `model` — a causal language model. `model(inputs_embeds)` takes a floating tensor of shape `(B, T, d)` and returns `(logits, last_hidden)` with shapes `(B, T, V)` and `(B, T, d)`. Position `t` of either output depends only on input positions `0..t`.
- `embed` — an `nn.Embedding` that maps a `(B, T)` id tensor to `(B, T, d)` input embeddings.
- `input_ids` — integer tensor of shape `(B, T)`.
- `latent_id` — the integer id of the `<|latent|>` token.

**Returns:** the `logits` from the last call to `model`, shape `(B, T, V)`.

**The passes.** For item `b`, let `p_b0 < p_b1 < ...` be the positions where `input_ids[b] == latent_id`, and let `N` be the largest latent count over the batch.
- Start from `E = embed(input_ids)`.
- For pass `i = 0 .. N - 1`: call `(logits, H) = model(E)`. Then, for every item with more than `i` latents, set `E[b, p_bi] = H[b, p_bi - 1]`. Every other entry of `E` keeps its value.
- Pass `N`: call `model(E)` once more and return its logits.

**Constraints:**
- Call `model` exactly `N + 1` times, each time on the full `(B, T, d)` sequence. Do not use a KV cache or slice the sequence.
- When no item has a latent, call `model` once and return its logits.
- Items are independent: each has its own latent count and positions, and latents need not be contiguous. An item with fewer than `N` latents keeps its embeddings unchanged in the later passes.
- Keep every fed-back hidden state in the autograd graph: no `detach`, no `torch.no_grad`. Backpropagating from the returned logits must reach the parameters of `model` and `embed` through each continuous thought.
- Do not modify in place a tensor that an earlier `model` call received; build a new tensor for each pass (for example with `index_put` or `torch.where`).
- Do not modify `input_ids`.
- Raise `ValueError` when `input_ids` is not a 2-D integer tensor, or when any item has a latent at position 0 (there is no previous position to read).
- Tests use a float64 model and compare logits and gradients to about `1e-9`.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why one pass per latent.** The thought at `p_bi` is the hidden state at `p_bi - 1`, which in turn depends on every earlier thought. So the thoughts must be produced in order, one forward pass each. Different items fill their `i`-th thought in the same pass even when their latents sit at different positions.

**Why full recomputation is enough.** With a causal model the hidden state at `p_bi - 1` only sees positions up to `p_bi - 1`, all of which are already final by pass `i`. Recomputing the whole sequence each pass costs `O(N * T^2)` attention but gives exactly the same logits as the cached version.

**What upstream does instead.** Coconut's `Coconut.forward` computes the prefix before the earliest latent once, then on each pass feeds only the next position and reuses `past_key_values`. The logits of each pass are concatenated along the sequence. This is cheaper but needs every item's latents to be contiguous and to start at the same index, which Coconut's sample builder (contiguous latents) and collator (left-padding that aligns the first latent) guarantee together.

**Why gradients through thoughts matter.** Coconut trains the thoughts end to end: the loss on the answer tokens backpropagates through every fed-back hidden state into the earlier passes. Detaching a thought silently turns the latent chain into a fixed prompt and the model never learns what to put there.""",
    "advisory_prerequisites": ["gpt2_block", "kv_cache", "coconut_stage_sample"],
    "hints": [
        {"level": 1, "kind": "questions", "content": "Which hidden state becomes the input at a latent position: the one at that position, or the one before it? Why can thought i only be computed after thought i - 1 is in place? How many model calls does an item with three latents need if another item in the batch has five? How do you replace a few rows of the embedding tensor without breaking autograd for the earlier calls?"},
        {"level": 2, "kind": "analysis", "content": "First collect each item's sorted latent positions and validate them, including the position-0 case. Embed once, then loop N times: run the model on the whole sequence, gather the hidden states one position before each item's i-th latent, and write them into a fresh copy of the embeddings. A final call produces the logits. Pitfalls: reading the latent's own position, filling every latent from the first pass, detaching the gathered states, and writing in place into a tensor the model already consumed."},
    ],
    "model_connections": [
        "Coconut (Hao et al. 2024) trains GPT-2 to replace chain-of-thought language steps with continuous thoughts and runs this loop in Coconut.forward.",
        "The same loop works with any decoder that exposes its last hidden state; upstream tests it with GPT-2 and Llama 3.",
        "Pause tokens add inert positions for extra compute; Coconut makes those positions carry information by feeding hidden states back, and recurrent-depth models (Huginn) recur in depth instead of along the sequence.",
    ],
    "pro_con_analysis": {
        "pros": ["Reasoning steps are dense vectors instead of sampled tokens, so a thought can encode several candidate next steps at once and needs far fewer positions than a language chain of thought."],
        "cons": ["Each thought needs its own sequential forward pass, so training cannot be parallelised over the latent span, and the thoughts are not human-readable."],
    },
    "sources": [
        {
            "kind": "code",
            "url": "https://github.com/facebookresearch/coconut/blob/27273cb8cca4bb763c041a63b036d0c3b7cbbb48/coconut.py",
            "commit": "27273cb8cca4bb763c041a63b036d0c3b7cbbb48",
            "path": "coconut.py",
            "symbol": "Coconut.forward (lines 39-193)",
            "license": "MIT",
            "adapted": "Same feedback rule: latent_lists from input_ids == latent_token_id, max_n_latents + 1 passes, and in pass i each item's i-th latent "
                       "takes the last-layer hidden state at token_idx - 1; no in-place writes into embeddings the model has seen.",
            "simplifications": "Divergence: upstream reuses past_key_values and feeds only the new positions, with hidden_states_offset taken from the "
                               "earliest latent in the batch, so it requires contiguous latents that start at the same index in every item and raises "
                               "IndexError otherwise; a latent at position 0 has no previous position to read. The exercise recomputes the full sequence on each pass, "
                               "accepts any latent positions, and raises ValueError for a latent at position 0. The model takes only inputs_embeds "
                               "(no attention_mask or position_ids), returns the last pass's logits instead of concatenating per-pass logits, and "
                               "computes no loss.",
        },
        {
            "kind": "paper",
            "url": "https://arxiv.org/abs/2412.06769",
            "section": "Hao et al. 2024, Training Large Language Models to Reason in a Continuous Latent Space, Section 3 (Method), 'Continuous thought' and 'Training details', Figure 1",
            "note": "Defines the continuous thought as the last hidden state fed back as the next input embedding, and the n + 1 forward passes for n thoughts.",
        },
    ],
    "tests": [
        {"name": "No latents: one call, plain logits", "behavior": "edge.empty_or_boundary", "code": _case(r"""
m = TinyCausalLM(0)
ids = torch.tensor([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]])
out = {fn}(m, m.embed, ids, LAT)
assert len(m.calls) == 1, f"expected 1 model call, got {len(m.calls)}"
m.calls.clear()
want = m(m.embed(ids))[0]
assert out.shape == (2, 5, 12), out.shape
assert torch.allclose(out, want, atol=1e-12), (out - want).abs().max()
""")},
        {"name": "One latent takes the hidden state just before it", "behavior": "state.invariant", "code": _case(r"""
m = TinyCausalLM(1)
ids = torch.tensor([[1, 2, LAT, 4, 5]])
out = {fn}(m, m.embed, ids, LAT)
assert len(m.calls) == 2, f"N = 1 latent needs 2 model calls, got {len(m.calls)}"
# The second call sees the first call's last hidden state at position 1 in slot 2.
assert torch.allclose(m.calls[1][0, 2], m.hiddens[0][0, 1], atol=1e-12)
assert torch.equal(m.calls[1][0, [0, 1, 3, 4]], m.calls[0][0, [0, 1, 3, 4]])
assert torch.allclose(out, m(m.calls[1])[0], atol=1e-12)
""")},
        {"name": "Each pass fills the next latent of every item", "behavior": "state.invariant", "code": _case(r"""
m = TinyCausalLM(2)
ids = torch.tensor([[1, LAT, LAT, LAT, 6, 7],
                    [3, 4, LAT, 5, 6, 7]])
{fn}(m, m.embed, ids, LAT)
assert len(m.calls) == 4, f"max latent count is 3, so 4 calls; got {len(m.calls)}"
# Pass 0 fills (0, 1) and (1, 2); pass 1 fills (0, 2); pass 2 fills (0, 3).
assert torch.allclose(m.calls[1][0, 1], m.hiddens[0][0, 0], atol=1e-12)
assert torch.allclose(m.calls[1][1, 2], m.hiddens[0][1, 1], atol=1e-12)
assert torch.allclose(m.calls[2][0, 2], m.hiddens[1][0, 1], atol=1e-12)
assert torch.allclose(m.calls[3][0, 3], m.hiddens[2][0, 2], atol=1e-12)
assert torch.equal(m.calls[3][1], m.calls[1][1]), "item 1 has one latent and must not change after pass 0"
""")},
        {"name": "Matches the full-recompute loop on ragged batches", "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "In pass i, write H[b, p_bi - 1] from that pass's model call into each item's i-th latent slot, then run one final pass.",
         "code": _case(r"""
for seed in range(6):
    g = torch.Generator().manual_seed(100 + seed)
    B, T = 3, 12
    ids = torch.randint(0, 10, (B, T), generator=g)
    for b in range(B):
        k = int(torch.randint(0, 5, (1,), generator=g))
        start = int(torch.randint(1, T - k, (1,), generator=g))
        ids[b, start:start + k] = LAT
    m = TinyCausalLM(seed)
    got = {fn}(m, m.embed, ids, LAT)
    want = oracle(m, m.embed, ids, LAT)
    assert got.shape == want.shape, (got.shape, want.shape)
    assert torch.allclose(got, want, atol=1e-9, rtol=0), (seed, (got - want).abs().max())
""", oracle=True)},
        {"name": "Non-contiguous latents and different offsets", "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "Latents may be separated by ordinary tokens; the i-th latent of each item reads the hidden state at its own position minus one, wherever it is.",
         "code": _case(r"""
m = TinyCausalLM(7)
ids = torch.tensor([[1, LAT, 2, LAT, 3, LAT, 4, 5],
                    [1, 2, 3, 4, LAT, 6, LAT, 7],
                    [LAT if t == 7 else t % 10 for t in range(8)]])
got = {fn}(m, m.embed, ids, LAT)
m.calls.clear()
want = oracle(m, m.embed, ids, LAT)
assert torch.allclose(got, want, atol=1e-9, rtol=0), (got - want).abs().max()
""", oracle=True)},
        {"name": "Exactly N + 1 full-length model calls", "visibility": "unshown", "behavior": "contract.signature",
         "failure_message": "Call the model exactly max_latents + 1 times, each time on the whole (B, T, d) embedding tensor.",
         "code": _case(r"""
for ids, n in ((torch.tensor([[1, LAT, LAT, 2], [1, 2, 3, 4]]), 3),
               (torch.tensor([[1, 2, LAT, 4, 5, 6, LAT, LAT, LAT, LAT, 3]]), 6),
               (torch.tensor([[4, 3, 2, 1]]), 1)):
    m = TinyCausalLM(3)
    {fn}(m, m.embed, ids, LAT)
    assert len(m.calls) == n, f"expected {n} calls, got {len(m.calls)}"
    for c in m.calls:
        assert c.shape == (ids.shape[0], ids.shape[1], 8), c.shape
""")},
        {"name": "Each item matches running it alone", "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "Use each item's own latent positions and count; never reuse another item's positions or the batch minimum.",
         "code": _case(r"""
m = TinyCausalLM(4)
ids = torch.tensor([[1, 2, 3, LAT, LAT, 4, 5, 6, 7],
                    [1, LAT, LAT, LAT, LAT, 2, 3, 4, 5],
                    [9, 8, 7, 6, 5, 4, 3, 2, 1],
                    [1, 2, 3, 4, 5, 6, LAT, 8, 9]])
joint = {fn}(m, m.embed, ids, LAT)
for b in range(4):
    single = {fn}(m, m.embed, ids[b:b + 1], LAT)
    assert torch.allclose(joint[b:b + 1], single, atol=1e-9, rtol=0), (b, (joint[b:b + 1] - single).abs().max())
""")},
        {"name": "Gradients flow through every continuous thought", "visibility": "unshown", "behavior": "gradient.flow",
         "failure_message": "Do not detach the gathered hidden states or run a pass under no_grad; every continuous thought must stay in the autograd graph.",
         "code": _case(r"""
ids = torch.tensor([[1, 2, LAT, LAT, LAT, 3, 4], [5, LAT, 6, 7, LAT, 8, 9]])
w = torch.randn(7, 12, generator=torch.Generator().manual_seed(9), dtype=torch.float64)
grads = []
for impl in ({fn}, oracle):
    m = TinyCausalLM(5)
    (impl(m, m.embed, ids, LAT) * w).sum().backward()
    grads.append({n: p.grad.clone() for n, p in m.named_parameters()})
for name in grads[1]:
    assert grads[0][name] is not None, name
    assert torch.allclose(grads[0][name], grads[1][name], atol=1e-9, rtol=0), (name, (grads[0][name] - grads[1][name]).abs().max())
""", oracle=True)},
        {"name": "Latent at position 0 and bad input_ids are rejected", "visibility": "unshown", "behavior": "edge.empty_or_boundary",
         "failure_message": "Raise ValueError when any item has a latent at position 0, or input_ids is not a 2-D integer tensor.",
         "code": _case(r"""
m = TinyCausalLM(6)
for bad in (torch.tensor([[1, 2, 3], [LAT, 2, 3]]), torch.tensor([1, LAT, 3]), torch.tensor([[1.0, 2.0, 3.0]])):
    try:
        {fn}(m, m.embed, bad, LAT)
    except ValueError:
        pass
    else:
        raise AssertionError(f"accepted input_ids {bad.tolist()} ({bad.dtype})")
""")},
        {"name": "input_ids is not modified", "visibility": "unshown", "behavior": "tensor.dtype_device",
         "failure_message": "Treat input_ids as read-only; build the latent positions without writing into it.",
         "code": _case(r"""
m = TinyCausalLM(8)
ids = torch.tensor([[1, 2, LAT, LAT, 5], [1, LAT, 3, 4, 5]])
before = ids.clone()
out = {fn}(m, m.embed, ids, LAT)
assert torch.equal(ids, before), ids
assert out.dtype == torch.float64 and out.shape == (2, 5, 12), (out.dtype, out.shape)
""")},
        {"name": "Embeddings a pass consumed are never written in place", "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "Do not write into a tensor an earlier model call received; build a new embedding tensor for each pass.",
         "code": _case(r"""
m = TinyCausalLM(10)
seen = []
inner = m.forward
def recording_forward(inputs_embeds):
    seen.append((inputs_embeds, inputs_embeds._version, inputs_embeds.detach().clone()))
    return inner(inputs_embeds)
m.forward = recording_forward
ids = torch.tensor([[1, 2, LAT, LAT, LAT, 3, 4], [5, LAT, 6, 7, LAT, 8, 9]])
{fn}(m, m.embed, ids, LAT)
assert len(seen) == 4, len(seen)
for i, (t, version, values) in enumerate(seen):
    assert t._version == version, ("pass", i, "version", version, "->", t._version)
    assert torch.equal(t.detach(), values), ("pass", i, "values changed")
""")},
    ],
    "solution": '''import torch


def coconut_forward(model, embed, input_ids, latent_id):
    if not torch.is_tensor(input_ids) or input_ids.dim() != 2 or input_ids.is_floating_point() or input_ids.is_complex():
        raise ValueError("input_ids must be a 2-D integer tensor")
    B, T = input_ids.shape
    latent_lists = [torch.nonzero(input_ids[b] == latent_id).flatten().tolist() for b in range(B)]
    if any(positions and positions[0] == 0 for positions in latent_lists):
        raise ValueError("a latent token cannot be at position 0")
    n_latents = max((len(positions) for positions in latent_lists), default=0)

    inputs_embeds = embed(input_ids)
    for pass_idx in range(n_latents):
        _, hidden = model(inputs_embeds)
        pairs = [(b, positions[pass_idx]) for b, positions in enumerate(latent_lists) if len(positions) > pass_idx]
        rows = torch.tensor([b for b, _ in pairs], device=input_ids.device)
        cols = torch.tensor([t for _, t in pairs], device=input_ids.device)
        # Out of place, so the tensor the model just consumed stays intact for backward.
        inputs_embeds = inputs_embeds.index_put((rows, cols), hidden[rows, cols - 1])
    logits, _ = model(inputs_embeds)
    return logits
''',
    "interview_questions": interview(
        concept=[
            "What is a continuous thought in Coconut, and where does its input embedding come from?",
            "Why does a sequence with n latent tokens need n + 1 forward passes instead of one?",
        ],
        deep_dive=[
            "Why is the thought at a latent position taken from the hidden state one position earlier rather than at the latent position itself?",
            "How does Coconut's KV-cached loop decide which positions to compute in each pass, and what does it assume about where latents sit in a batch?",
            "What goes wrong in training if the fed-back hidden states are detached, and how would you detect it with a test?",
        ],
        tradeoffs=[
            "Full recomputation on every pass versus reusing a KV cache: cost, simplicity and constraints on the batch layout?",
            "Continuous thoughts versus language chain of thought versus pause tokens: what does each buy and cost at training and inference time?",
        ],
    ),
}
