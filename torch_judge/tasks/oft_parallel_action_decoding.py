"""OpenVLA-OFT parallel decoding: empty action slots, a bidirectional mask and a one-pass readout of a whole action chunk."""

from ._interview import interview

# A tiny one-layer attention "decoder" plus a plain-loop model of the contract,
# written independently of the reference solution.
_WORLD = r"""
import math
import torch

def make_world(seed, H=6, V=48, n_bins=8, T=97):
    g = torch.Generator().manual_seed(seed)
    table = torch.randn(T, H, generator=g, dtype=torch.float64)
    pos = torch.randn(512, H, generator=g, dtype=torch.float64) * 0.7
    Wq = torch.randn(H, H, generator=g, dtype=torch.float64) / math.sqrt(H)
    Wk = torch.randn(H, H, generator=g, dtype=torch.float64) / math.sqrt(H)
    Wv = torch.randn(H, H, generator=g, dtype=torch.float64)
    Wout = torch.randn(H, V, generator=g, dtype=torch.float64) * 2
    boost = torch.zeros(V, dtype=torch.float64)
    boost[V - n_bins:] = 4.0
    log = []

    def embed(ids):
        return table[ids % T]

    def decoder(x, bias):
        log.append((x.clone(), bias.clone()))
        S = x.shape[1]
        z = x + pos[:S]
        scores = (z @ Wq) @ (z @ Wk).transpose(-1, -2) / math.sqrt(H) + bias[:, 0].to(z.dtype)
        h = torch.tanh(z + torch.softmax(scores, dim=-1) @ (z @ Wv))
        return h, h @ Wout + boost

    return embed, decoder, log

def m_prepare(ids, mask, K, D, vocab, n_bins):
    rows, mrows, lrows = [], [], []
    append = any(int(r[-1]) != 29871 for r in ids)
    for r, m in zip(ids.tolist(), mask.tolist()):
        if append:
            r = r + [29871]
        n_prompt = len(r) - 1
        lab = [-100] * len(r) + [vocab - n_bins] * (K * D) + [2]
        r = r + [1] * (K * D) + [2]
        m = m + [1] * (len(r) - len(m))
        rows.append(r); mrows.append(m); lrows.append(lab)
    return torch.tensor(rows), torch.tensor(mrows), torch.tensor(lrows), n_prompt

def m_masks(labels, D, vocab, n_bins):
    cur = torch.zeros(labels.shape, dtype=torch.bool)
    nxt = torch.zeros(labels.shape, dtype=torch.bool)
    for b in range(labels.shape[0]):
        count = 0
        for j in range(labels.shape[1]):
            v = int(labels[b, j])
            if v != -100:
                count += 1
            if v >= vocab - n_bins:
                if count <= D:
                    cur[b, j] = count >= 1
                else:
                    nxt[b, j] = True
    return cur, nxt

def m_bias(mask, dtype):
    B, S = mask.shape
    out = torch.zeros(B, 1, S, S, dtype=dtype)
    lo = torch.finfo(dtype).min
    for b in range(B):
        for k in range(S):
            if float(mask[b, k]) == 0.0:
                out[b, 0, :, k] = lo
    return out

def m_predict(embed, decoder, ids, mask, patches, K, D, vocab, n_bins, head=None):
    ids2, mask2, labels, n_prompt = m_prepare(ids, mask, K, D, vocab, n_bins)
    cur, nxt = m_masks(labels, D, vocab, n_bins)
    e = embed(ids2).clone()
    e[cur | nxt] = 0.0
    B, P = patches.shape[:2]
    x = torch.cat([e[:, :1], patches, e[:, 1:]], dim=1)
    mm = torch.cat([mask2[:, :1], torch.ones(B, P, dtype=mask2.dtype), mask2[:, 1:]], dim=1)
    h, logits = decoder(x, m_bias(mm, x.dtype))
    first_action = 1 + P + n_prompt
    rows = range(first_action - 1, first_action - 1 + K * D)
    if head is not None:
        grouped = torch.stack([torch.cat([h[:, r] for r in list(rows)[t * D:(t + 1) * D]], dim=-1) for t in range(K)], dim=1)
        return head(grouped)
    edges = torch.linspace(-1, 1, n_bins, dtype=torch.float64)
    centers = (edges[:-1] + edges[1:]) / 2
    out = torch.zeros(B, K * D, dtype=torch.float64)
    for b in range(B):
        for i, r in enumerate(rows):
            tok = int(logits[b, r].argmax())
            out[b, i] = centers[min(max(vocab - tok - 1, 0), n_bins - 2)]
    return out.reshape(B, K, D)
"""

TASK = {
    "title": "OpenVLA-OFT Parallel Decoding of an Action Chunk",
    "difficulty": "Hard",
    "version": 1,
    "function_name": "OFTParallelDecoder",
    "description_en": r"""Predict a whole chunk of `K` future actions, each with `D` dimensions, in one decoder forward pass, the way OpenVLA-OFT does: append `K * D` empty action slots and a stop token to the prompt, zero their embeddings, let every token attend to every non-padding token, and read the chunk off the outputs.

**Signature:** `OFTParallelDecoder(chunk_len, action_dim, vocab_size=32000, n_bins=256)`

`chunk_len` is `K`, `action_dim` is `D`, and `N = K * D` is the number of action slots. Action token ids are the last `n_bins` ids of the vocabulary, so a label is an **action label** when `label > vocab_size - n_bins - 1` (31743 for the defaults). Fixed ids: ignore label `-100`, stop `2`, empty token `29871`, placeholder `1`.

**Methods:**
- `prepare_inputs(input_ids, attention_mask) -> (input_ids, attention_mask, labels, num_prompt_tokens)`. Inputs are `(B, L)`.
  - If any row's last id is not `29871`, append `29871` to every row.
  - `num_prompt_tokens` is the length after that step minus 1 (the BOS token is not counted).
  - Append `N` placeholder ids `1`, then one stop id `2`.
  - Extend `attention_mask` with ones (same dtype) to the new length.
  - `labels`: `-100` over the prompt (including any appended `29871`), `vocab_size - n_bins` over the `N` slots, `2` at the stop position.
- `action_masks(labels) -> (current, next)`, two bool tensors shaped like `labels` (`(B, S)`).
  - Let `count` be the running number of labels `!= -100` along the sequence, up to and including each position.
  - `current` is an action label with `1 <= count <= D`. `next` is an action label with `count > D`.
- `attention_bias(attention_mask, dtype=torch.float32) -> (B, 1, S, S)`, an additive mask.
  - Entry `[b, 0, q, k]` is `0` when `attention_mask[b, k] != 0` and `torch.finfo(dtype).min` when it is `0`. Every query row is the same, including the rows of padding queries. There is no causal part.
- `train_action_hidden(last_hidden, labels, num_patches) -> (B, N, H)`, for training batches.
  - `last_hidden` is `(B, num_patches + L, H)` and `labels` is `(B, L)`.
  - Take `last_hidden[:, num_patches:-1]` and `action_masks(labels[:, 1:])`, and keep the positions where `current | next` is true, in order.
- `predict(decoder, embed, input_ids, attention_mask, patch_embeds, action_head=None) -> (B, K, D)`.
  - Run `prepare_inputs`, then `x = embed(ids)` of shape `(B, S, H)`.
  - Set `x` to zero wherever `current | next` of the labels is true. The stop token keeps its embedding.
  - Insert `patch_embeds` (`(B, P, H)`) right after the first token: `[x[:, :1], patch_embeds, x[:, 1:]]`. Insert `P` ones into the attention mask at the same place.
  - Call `hidden, logits = decoder(embeds, attention_bias(mask, embeds.dtype))` **exactly once**. `hidden` is `(B, S + P, H)` and `logits` is `(B, S + P, vocab_size)`.
  - Read positions `start .. start + N - 1` with `start = P + num_prompt_tokens`. This is one position before each action slot, as in next-token prediction.
  - With `action_head`: return `action_head(h)` where `h` is the read hidden states reshaped to `(B, K, D * H)`, so time step `t` concatenates the `D` slots `t * D .. t * D + D - 1`.
  - Without it: `tok = argmax` of the read logits, `i = clamp(vocab_size - tok - 1, 0, n_bins - 2)`, and the action is `centers[i]` with `edges = torch.linspace(-1, 1, n_bins)` and `centers = (edges[:-1] + edges[1:]) / 2`, in the dtype of `logits`, reshaped to `(B, K, D)`.

**Constraints:**
- Never mask a key that has `attention_mask == 1`: prompt and patch tokens attend to the action slots and the stop token too.
- Raise `ValueError` if `chunk_len < 1`, `action_dim < 1`, `n_bins < 2` or `vocab_size <= n_bins`; if `input_ids` is not 2D or `attention_mask` has another shape; if `patch_embeds` has another batch size; if `last_hidden` is not `num_patches + L` long or a row of `labels[:, 1:]` does not hold exactly `N` action positions; or if `action_head` does not return `(B, K, D)`.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why not autoregressive.** OpenVLA decodes one token per action dimension, 7 per time step, so a chunk of `K` steps costs `K * D` sequential forward passes. The OFT paper (Kim, Finn, Liang 2025) replaces the input action tokens with empty action embeddings and the causal mask with bidirectional attention, so the whole chunk comes out of one pass. The paper reports no performance drop from parallel decoding in its experiments, and, together with chunking, 26x to 43x throughput speedups.

**Empty slots.** The slots enter as zero vectors (the placeholder id `1` is embedded and then multiplied by zero), so the decoder can only tell them apart by position, which Llama 2 encodes with rotary embeddings inside attention.

**The mask.** The OFT code installs a patched `transformers` whose SDPA attention takes the last row of Hugging Face's causal-plus-padding mask and copies it to every row. The last query sees every earlier key, so that row masks exactly the padding keys. The result is full bidirectional attention over the whole sequence, not a prefix-LM mask.

**The stop token.** OFT appends `</s>` at inference because it is present at training time, and with bidirectional attention every token sees it.

**The shift.** OFT keeps OpenVLA's next-token alignment: in training it reads `last_hidden[:, num_patches:-1]` against `labels[:, 1:]`, and at inference it starts one position before the first action slot. Both sides use the same offset, so the action head always sees the output of the slot's left neighbor.

**The heads.** The L1 regression head (an MLP with ReLU) maps each time step's `D` concatenated hidden states to its `D` actions. The discrete path keeps OpenVLA's 256 bins and its bin-center decoding. LIBERO uses `K = 8`, `D = 7`; ALOHA uses `K = 25`, `D = 14`.""",
    "advisory_prerequisites": ["vla_action_bins"],
    "hints": [
        {"level": 1, "kind": "questions", "content": "Where in the multimodal sequence does the first action slot sit once the patches are inserted after BOS? Which output position predicts it under next-token alignment? Why does the attention bias depend only on the key index? How is the 'current' window counted when a non-action label appears before the actions?"},
        {"level": 2, "kind": "analysis", "content": "Build labels alongside the ids so the action masks come from labels, not ids. Compute a running count of non-ignored labels and AND it with label > vocab_size - n_bins - 1. For the bias, broadcast the key padding mask over every query row and fill with finfo(dtype).min. In predict, zero the slots, splice the patches after position 0, call the decoder once, slice from P + num_prompt_tokens, then reshape (B, N, H) to (B, K, D * H) for the head or decode argmax ids to bin centers."},
    ],
    "model_connections": [
        "OpenVLA-OFT fine-tunes OpenVLA (Llama 2 7B) with parallel decoding, action chunking and an L1 regression head; OpenVLA-OFT+ adds FiLM for the ALOHA tasks.",
        "OpenVLA, the starting point, decodes 7 discrete action tokens per time step autoregressively with a causal mask.",
        "pi0 and pi0.5 also emit a whole action chunk at once, but by integrating a flow-matching field over several denoising steps instead of reading the chunk off one pass.",
        "ACT decodes a chunk in parallel from learned position queries with an L1 loss; the OFT paper cites it for both the empty, position-only action inputs and the L1 objective.",
    ],
    "pro_con_analysis": {
        "pros": [
            "One forward pass per chunk instead of K * D, which the paper measures as a large throughput gain and lower latency.",
            "Every slot sees every other slot and the whole prompt, so the chunk is predicted jointly rather than left to right.",
            "Works with both the discrete bin head and a continuous L1 head on the same hidden states.",
        ],
        "cons": [
            "Each slot is predicted independently given the shared context, so a multimodal joint action distribution cannot be sampled consistently from per-slot argmax or L1 outputs; the paper notes it may be less expressive than autoregression.",
            "Chunk length and action dimension are baked into the sequence layout and the head input size; changing K or D needs new training.",
            "Because the prompt also attends to the slots, nothing can be cached across passes: OFT's diffusion variant reruns the full decoder at each of its 50 denoising steps, while a prefix-LM such as pi0 computes the prefix KV cache once.",
        ],
    },
    "sources": [
        {'kind': 'code',
         'url': 'https://github.com/moojink/openvla-oft/blob/e4287e94541f459edc4feabc4e181f537cd569a8/prismatic/extern/hf/modeling_prismatic.py',
         'commit': 'e4287e94541f459edc4feabc4e181f537cd569a8',
         'path': 'prismatic/extern/hf/modeling_prismatic.py',
         'symbol': 'OpenVLAForActionPrediction.predict_action, _prepare_input_for_action_prediction, '
                   '_prepare_labels_for_action_prediction, _regression_or_discrete_prediction; '
                   'PrismaticForConditionalGeneration._process_action_masks, _build_multimodal_attention',
         'license': 'MIT',
         'adapted': 'The 29871 append, placeholder and stop packing, label construction, zeroed action embeddings, '
                    'patch insertion after BOS, the P + num_prompt_tokens readout slice and the bin-center decode.',
         'simplifications': 'The Llama decoder, vision backbone and tokenizer are callables passed in; the decoder takes the '
                            '4D additive mask instead of building it from the 2D mask. Upstream concatenates a (1, 1) '
                            '29871 tensor and only supports batch size 1; the exercise appends it to every row. Upstream '
                            'returns numpy float64 bin centers; the exercise returns them in the logits dtype. Proprio, '
                            'FiLM, diffusion and unnormalization are dropped.'},
        {'kind': 'code',
         'url': 'https://github.com/moojink/openvla-oft/blob/e4287e94541f459edc4feabc4e181f537cd569a8/prismatic/training/train_utils.py',
         'commit': 'e4287e94541f459edc4feabc4e181f537cd569a8',
         'path': 'prismatic/training/train_utils.py',
         'symbol': 'get_current_action_mask, get_next_actions_mask',
         'license': 'MIT',
         'adapted': 'Running count of non-ignored labels, windowed at action_dim, ANDed with label > ACTION_TOKEN_BEGIN_IDX.',
         'simplifications': 'ACTION_TOKEN_BEGIN_IDX (31743, hard-coded for Llama 2) becomes vocab_size - n_bins - 1, '
                            'which is 31743 for the defaults.'},
        {'kind': 'code',
         'url': 'https://github.com/moojink/openvla-oft/blob/e4287e94541f459edc4feabc4e181f537cd569a8/vla-scripts/finetune.py',
         'commit': 'e4287e94541f459edc4feabc4e181f537cd569a8',
         'path': 'vla-scripts/finetune.py',
         'symbol': 'run_forward_pass',
         'license': 'MIT',
         'adapted': 'last_hidden_states[:, num_patches:-1] masked by the action masks of labels[:, 1:], reshaped to (B, K * D, H).',
         'simplifications': 'Only the continuous-action readout is kept; losses, metrics and bfloat16 casts are dropped.'},
        {'kind': 'code',
         'url': 'https://github.com/moojink/openvla-oft/blob/e4287e94541f459edc4feabc4e181f537cd569a8/prismatic/models/action_heads.py',
         'commit': 'e4287e94541f459edc4feabc4e181f537cd569a8',
         'path': 'prismatic/models/action_heads.py',
         'symbol': 'L1RegressionActionHead.predict_action',
         'license': 'MIT',
         'adapted': 'The reshape (B, K * D, H) -> (B, K, D * H) before the MLP.',
         'simplifications': 'The MLPResNet is replaced by any callable action_head.'},
        {'kind': 'code',
         'url': 'https://github.com/moojink/transformers-openvla-oft/blob/bc339d9ad707454c0c115970db43c260067c61ab/src/transformers/models/llama/modeling_llama.py',
         'commit': 'bc339d9ad707454c0c115970db43c260067c61ab',
         'path': 'src/transformers/models/llama/modeling_llama.py',
         'symbol': 'LlamaSdpaAttention.forward (OFT bidirectional patch), LlamaModel._update_causal_mask',
         'license': 'Apache-2.0',
         'adapted': 'The last row of the causal-plus-padding mask copied to every row, used with is_causal=False.',
         'simplifications': 'Built directly as a key-padding mask broadcast over queries, which equals the last row of '
                            'the causal mask for any 2D padding pattern; the SDPA path that passes no mask when there is '
                            'no padding is the all-zero bias here.'},
        {'kind': 'paper',
         'url': 'https://arxiv.org/abs/2502.19645',
         'section': 'Kim, Finn, Liang 2025 (OpenVLA-OFT), Section IV-B "Implementing Alternative Design Components", '
                    'paragraph "Parallel decoding and action chunking"',
         'note': 'Empty action embeddings as input and a bidirectional mask in place of the causal one; K * D actions in one pass.'},
        {'kind': 'paper',
         'url': 'https://arxiv.org/abs/2502.19645',
         'section': 'Appendix B.1 "Parallel Decoding Implementation" and Appendix A (architecture changes 3-5)',
         'note': 'Empty action embeddings that differ only in their positional encoding; a 4-layer ReLU MLP head for L1 regression.'},
    ],
    "tests": [
        {"name": "prepare_inputs packs slots and labels; action_masks split current from next", "behavior": "state.invariant", "code": r"""
import torch
dec = {fn}(chunk_len=2, action_dim=3)
ids = torch.tensor([[1, 450, 3802]])
mask = torch.ones(1, 3, dtype=torch.long)
out_ids, out_mask, labels, n_prompt = dec.prepare_inputs(ids, mask)
assert out_ids.tolist() == [[1, 450, 3802, 29871, 1, 1, 1, 1, 1, 1, 2]], out_ids
assert out_mask.tolist() == [[1] * 11] and out_mask.dtype == torch.long, out_mask
assert labels.tolist() == [[-100, -100, -100, -100] + [31744] * 6 + [2]], labels
assert n_prompt == 3, n_prompt
cur, nxt = dec.action_masks(labels)
assert cur.dtype == torch.bool and cur[0].tolist() == [False] * 4 + [True] * 3 + [False] * 4, cur
assert nxt[0].tolist() == [False] * 7 + [True] * 3 + [False], nxt
# A prompt that already ends in 29871 gets nothing appended.
again = dec.prepare_inputs(torch.tensor([[1, 450, 29871]]), torch.ones(1, 3, dtype=torch.long))
assert again[0].shape == (1, 10) and again[3] == 2, again
"""},
        {"name": "attention_bias masks padding keys only, in every row", "behavior": "attention.masking", "code": r"""
import torch
dec = {fn}(chunk_len=1, action_dim=2)
bias = dec.attention_bias(torch.tensor([[1, 1, 1, 0]]), torch.float32)
lo = torch.finfo(torch.float32).min
row = torch.tensor([0.0, 0.0, 0.0, lo])
assert bias.shape == (1, 1, 4, 4) and bias.dtype == torch.float32, (bias.shape, bias.dtype)
assert torch.equal(bias[0, 0], row.expand(4, 4)), bias
"""},
        {"name": "predict reads one position before each slot, in a single decoder call", "behavior": "attention.masking", "code": r"""
import torch
dec = {fn}(chunk_len=2, action_dim=2, vocab_size=40, n_bins=8)
calls = []
def decoder(x, bias):
    calls.append(x.shape)
    B, S, _ = x.shape
    pos = torch.arange(S, dtype=x.dtype).reshape(1, S, 1).expand(B, S, 1)
    return pos, torch.zeros(B, S, 40, dtype=x.dtype)
embed = lambda ids: torch.ones(*ids.shape, 1, dtype=torch.float64)
patches = torch.ones(1, 3, 1, dtype=torch.float64)
# prompt [BOS, 450, 29871] -> num_prompt_tokens 2; P = 3; first slot at 1 + 3 + 2 = 6
out = dec.predict(decoder, embed, torch.tensor([[1, 450, 29871]]), torch.ones(1, 3, dtype=torch.long), patches,
                  action_head=lambda h: h)
assert len(calls) == 1 and calls[0] == (1, 3 + 3 + 4 + 1, 1), calls
assert out.shape == (1, 2, 2) and out.reshape(-1).tolist() == [5.0, 6.0, 7.0, 8.0], out
"""},
        {"name": "prepare_inputs matches a model across batches and the 29871 rule", "visibility": "unshown", "behavior": "edge.empty_or_boundary",
         "failure_message": "Append 29871 to every row unless every row already ends in it, then N placeholder ids 1 and a stop id 2; labels are -100, vocab_size - n_bins and 2; num_prompt_tokens excludes BOS.", "code": _WORLD + r"""
for seed, (K, D, vocab, n_bins) in enumerate(((2, 3, 32000, 256), (3, 2, 48, 8), (1, 1, 20, 2), (4, 7, 32000, 256))):
    g = torch.Generator().manual_seed(seed)
    B, L = 1 + seed % 3, 2 + seed
    ids = torch.randint(3, 900, (B, L), generator=g)
    ids[:, 0] = 1
    if seed % 2 == 0:
        ids[:, -1] = 29871
    elif B > 1:
        ids[0, -1] = 29871
    mask = (torch.rand(B, L, generator=g) > 0.2).long()
    got = {fn}(K, D, vocab, n_bins).prepare_inputs(ids, mask)
    want = m_prepare(ids, mask, K, D, vocab, n_bins)
    for name, a, b in zip(("ids", "mask", "labels"), got[:3], want[:3]):
        assert a.shape == b.shape and torch.equal(a.long(), b), (seed, name, a, b)
    assert got[1].dtype == mask.dtype, (seed, got[1].dtype)
    assert int(got[3]) == want[3], (seed, got[3], want[3])
mask_bool = torch.tensor([[True, True]])
assert {fn}(1, 2).prepare_inputs(torch.tensor([[1, 5]]), mask_bool)[1].dtype == torch.bool
"""},
        {"name": "action_masks follow the running label count, including padding and early labels", "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "Count every label != -100 (not only action labels) cumulatively; current is an action label with count in 1..D, next is an action label with count > D.", "code": _WORLD + r"""
dec = {fn}(chunk_len=3, action_dim=2, vocab_size=48, n_bins=8)
A = 40
labels = torch.tensor([
    [-100, -100, A, A + 3, A + 7, A, A + 1, A + 2, 2, -100, -100],
    [-100, 17, A, A + 1, A + 2, A + 3, A + 4, A + 5, 2, -100, -100],
    [-100, -100, -100, -100, A, A, A, A, A, A, 2],
    [-100, -100, 39, A, A, A, A, A, A, 2, -100],
])
cur, nxt = dec.action_masks(labels)
wc, wn = m_masks(labels, 2, 48, 8)
assert cur.dtype == torch.bool and nxt.dtype == torch.bool, (cur.dtype, nxt.dtype)
assert torch.equal(cur, wc), (cur, wc)
assert torch.equal(nxt, wn), (nxt, wn)
assert cur[1].tolist().count(True) == 1 and cur[3].tolist().count(True) == 1, cur
for seed in range(5):
    g = torch.Generator().manual_seed(seed)
    lab = torch.randint(30, 48, (3, 12), generator=g)
    lab[torch.rand(3, 12, generator=g) < 0.35] = -100
    d = {fn}(chunk_len=2, action_dim=1 + seed % 3, vocab_size=48, n_bins=8)
    c, n = d.action_masks(lab)
    wc, wn = m_masks(lab, 1 + seed % 3, 48, 8)
    assert torch.equal(c, wc) and torch.equal(n, wn), seed
"""},
        {"name": "attention_bias matches a model for any padding pattern and dtype", "visibility": "unshown", "behavior": "attention.masking",
         "failure_message": "Use no causal part: mask only keys whose attention_mask is 0 with finfo(dtype).min, identically in every query row, including padding rows.", "code": _WORLD + r"""
dec = {fn}(chunk_len=2, action_dim=3)
cases = [
    torch.tensor([[1, 1, 1, 1, 1]]),
    torch.tensor([[1, 1, 1, 0, 0], [0, 0, 1, 1, 1], [1, 0, 1, 0, 1]]),
    torch.tensor([[True, False, True, True]]),
    torch.tensor([[1.0, 1.0, 0.0]]),
]
for i, mask in enumerate(cases):
    for dtype in (torch.float32, torch.float64, torch.bfloat16):
        bias = dec.attention_bias(mask, dtype)
        want = m_bias(mask, dtype)
        assert bias.shape == want.shape and bias.dtype == dtype, (i, dtype, bias.shape, bias.dtype)
        assert torch.equal(bias, want), (i, dtype, bias)
assert (dec.attention_bias(torch.ones(2, 6)) == 0).all()
"""},
        {"name": "train_action_hidden uses the shifted readout against labels[:, 1:]", "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "Slice last_hidden[:, num_patches:-1], build the masks from labels[:, 1:], keep current | next in order and reshape to (B, K * D, H).", "code": _WORLD + r"""
K, D, P = 2, 3, 4
dec = {fn}(chunk_len=K, action_dim=D, vocab_size=48, n_bins=8)
for seed in range(5):
    g = torch.Generator().manual_seed(seed)
    B = 1 + seed % 3
    rows = []
    for b in range(B):
        pre = 2 + (b + seed) % 3
        acts = torch.randint(40, 48, (K * D,), generator=g).tolist()
        row = [-100] * pre + acts + [2]
        rows.append(row)
    L = max(len(r) for r in rows) + 1
    labels = torch.tensor([r + [-100] * (L - len(r)) for r in rows])
    H = 5
    last = torch.randn(B, P + L, H, generator=g, dtype=torch.float64)
    got = dec.train_action_hidden(last, labels, P)
    cur, nxt = m_masks(labels[:, 1:], D, 48, 8)
    want = last[:, P:-1][cur | nxt].reshape(B, K * D, H)
    assert got.shape == (B, K * D, H), (seed, got.shape)
    assert torch.equal(got, want), seed
"""},
        {"name": "predict with the discrete head matches a model", "visibility": "unshown", "behavior": "attention.masking",
         "failure_message": "Zero the action slots, insert patches after BOS, use the full bidirectional bias, call the decoder once, slice from P + num_prompt_tokens, and decode clamp(vocab_size - argmax - 1, 0, n_bins - 2) to bin centers.", "code": _WORLD + r"""
for seed, (K, D, P, L) in enumerate(((2, 3, 4, 3), (3, 2, 2, 5), (1, 4, 3, 2), (4, 1, 5, 4), (2, 2, 1, 3), (3, 3, 2, 6))):
    embed, decoder, log = make_world(seed)
    g = torch.Generator().manual_seed(100 + seed)
    B = 1 + seed % 2
    ids = torch.randint(3, 500, (B, L), generator=g)
    ids[:, 0] = 1
    mask = torch.ones(B, L, dtype=torch.long)
    patches = torch.randn(B, P, 6, generator=g, dtype=torch.float64)
    out = {fn}(K, D, 48, 8).predict(decoder, embed, ids, mask, patches)
    assert len(log) == 1, (seed, len(log))
    _, ref_decoder, _ = make_world(seed)
    want = m_predict(embed, ref_decoder, ids, mask, patches, K, D, 48, 8)
    assert out.shape == (B, K, D) and out.dtype == torch.float64, (seed, out.shape, out.dtype)
    assert torch.allclose(out, want, atol=1e-12), (seed, out, want)
    x, bias = log[0]
    assert x.shape[1] == bias.shape[-1] == L + (0 if int(ids[0, -1]) == 29871 else 1) + K * D + 1 + P, seed
    assert (bias == 0).all(), seed
"""},
        {"name": "predict feeds zeroed slots, keeps the stop embedding and places the patches after BOS", "visibility": "unshown", "behavior": "tensor.shape",
         "failure_message": "Zero only the current | next action positions (not the stop token) and splice patch_embeds between the first token and the rest.", "code": _WORLD + r"""
embed, decoder, log = make_world(3)
K, D, P = 2, 2, 3
ids = torch.tensor([[1, 77, 88]])
patches = torch.full((1, P, 6), 9.0, dtype=torch.float64)
{fn}(K, D, 48, 8).predict(decoder, embed, ids, torch.ones(1, 3, dtype=torch.long), patches)
x, _ = log[0]
full = torch.tensor([[1, 77, 88, 29871] + [1] * (K * D) + [2]])
e = embed(full)
assert torch.equal(x[0, 0], e[0, 0]), "first token must stay first"
assert torch.equal(x[0, 1:1 + P], patches[0]), "patches go right after the first token"
assert torch.equal(x[0, 1 + P:1 + P + 3], e[0, 1:4]), "prompt follows the patches"
assert (x[0, 1 + P + 3:1 + P + 3 + K * D] == 0).all(), "action slots must be zero"
assert torch.equal(x[0, -1], e[0, -1]) and x[0, -1].abs().sum() > 0, "stop keeps its embedding"
"""},
        {"name": "predict with an L1 head groups D slots per time step", "visibility": "unshown", "behavior": "tensor.shape",
         "failure_message": "Reshape the N read hidden states to (B, K, D * H) so step t concatenates slots t*D .. t*D + D - 1, then call action_head once.", "code": _WORLD + r"""
for seed, (K, D) in enumerate(((2, 3), (3, 2), (4, 4), (1, 5), (5, 1))):
    embed, decoder, log = make_world(seed)
    g = torch.Generator().manual_seed(200 + seed)
    W = torch.randn(D * 6, D, generator=g, dtype=torch.float64)
    seen = []
    def head(h):
        seen.append(h.shape)
        return torch.tanh(h @ W)
    ids = torch.tensor([[1, 5, 6, 29871]])
    patches = torch.randn(1, 2, 6, generator=g, dtype=torch.float64)
    out = {fn}(K, D, 48, 8).predict(decoder, embed, ids, torch.ones(1, 4, dtype=torch.long), patches, action_head=head)
    _, ref_decoder, _ = make_world(seed)
    want = m_predict(embed, ref_decoder, ids, torch.ones(1, 4, dtype=torch.long), patches, K, D, 48, 8, head=lambda h: torch.tanh(h @ W))
    assert seen == [(1, K, D * 6)], (seed, seen)
    assert out.shape == (1, K, D) and torch.allclose(out, want, atol=1e-12), (seed, out, want)
"""},
        {"name": "predict passes prompt padding through to the bias", "visibility": "unshown", "behavior": "attention.masking",
         "failure_message": "Extend the given attention_mask (do not replace it with ones), insert ones for the patches after the first entry, and build the bias from that multimodal mask.", "code": _WORLD + r"""
embed, decoder, log = make_world(8)
ids = torch.tensor([[1, 5, 6, 7, 29871], [1, 9, 10, 11, 29871]])
mask = torch.tensor([[1, 1, 1, 1, 1], [1, 0, 0, 1, 1]])
patches = torch.randn(2, 3, 6, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
out = {fn}(2, 2, 48, 8).predict(decoder, embed, ids, mask, patches)
_, bias = log[0]
mm = torch.cat([mask[:, :1], torch.ones(2, 3, dtype=torch.long), mask[:, 1:], torch.ones(2, 5, dtype=torch.long)], dim=1)
assert torch.equal(bias, m_bias(mm, torch.float64)), bias[1, 0, 0]
_, ref_decoder, _ = make_world(8)
assert torch.allclose(out, m_predict(embed, ref_decoder, ids, mask, patches, 2, 2, 48, 8), atol=1e-12)
"""},
        {"name": "Rejects invalid arguments", "visibility": "unshown", "behavior": "contract.signature",
         "failure_message": "Raise ValueError for bad sizes, mismatched shapes, a wrong number of action labels or a head with the wrong output shape.", "code": _WORLD + r"""
embed, decoder, _ = make_world(0)
ok = {fn}(2, 2, 48, 8)
ids, mask = torch.tensor([[1, 5, 29871]]), torch.ones(1, 3, dtype=torch.long)
bad = [
    lambda: {fn}(0, 2),
    lambda: {fn}(2, 0),
    lambda: {fn}(2, 2, 48, 1),
    lambda: {fn}(2, 2, 8, 8),
    lambda: ok.prepare_inputs(torch.tensor([1, 5]), torch.ones(2)),
    lambda: ok.prepare_inputs(ids, torch.ones(1, 4, dtype=torch.long)),
    lambda: ok.predict(decoder, embed, ids, mask, torch.zeros(2, 3, 6, dtype=torch.float64)),
    lambda: ok.predict(decoder, embed, ids, mask, torch.zeros(1, 3, 6, dtype=torch.float64), action_head=lambda h: h),
    lambda: ok.train_action_hidden(torch.zeros(1, 9, 4), torch.tensor([[-100, 40, 40, 40, 40, 2]]), 2),
    lambda: ok.train_action_hidden(torch.zeros(1, 8, 4), torch.tensor([[-100, -100, 40, 40, 40, 2]]), 2),
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

class OFTParallelDecoder:
    IGNORE_INDEX = -100
    STOP_ID = 2
    EMPTY_ID = 29871
    PLACEHOLDER_ID = 1

    def __init__(self, chunk_len, action_dim, vocab_size=32000, n_bins=256):
        if chunk_len < 1 or action_dim < 1:
            raise ValueError("chunk_len and action_dim must be at least 1")
        if n_bins < 2 or vocab_size <= n_bins:
            raise ValueError("need n_bins >= 2 and vocab_size > n_bins")
        self.chunk_len, self.action_dim = chunk_len, action_dim
        self.vocab_size, self.n_bins = vocab_size, n_bins
        self.num_slots = chunk_len * action_dim
        self.action_begin = vocab_size - n_bins - 1  # labels above this are action tokens
        edges = torch.linspace(-1, 1, n_bins, dtype=torch.float64)
        self.centers = (edges[:-1] + edges[1:]) / 2

    def prepare_inputs(self, input_ids, attention_mask):
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError("input_ids and attention_mask must both be (B, L)")
        batch = input_ids.shape[0]
        if not bool((input_ids[:, -1] == self.EMPTY_ID).all()):
            input_ids = torch.cat([input_ids, input_ids.new_full((batch, 1), self.EMPTY_ID)], dim=1)
        num_prompt_tokens = input_ids.shape[1] - 1
        slots = input_ids.new_full((batch, self.num_slots), self.PLACEHOLDER_ID)
        stop = input_ids.new_full((batch, 1), self.STOP_ID)
        labels = torch.cat([
            torch.full_like(input_ids, self.IGNORE_INDEX),
            torch.full_like(slots, self.action_begin + 1),
            stop,
        ], dim=1)
        input_ids = torch.cat([input_ids, slots, stop], dim=1)
        extra = attention_mask.new_ones((batch, input_ids.shape[1] - attention_mask.shape[1]))
        attention_mask = torch.cat([attention_mask, extra], dim=1)
        return input_ids, attention_mask, labels, num_prompt_tokens

    def action_masks(self, labels):
        count = (labels != self.IGNORE_INDEX).long().cumsum(dim=1)
        is_action = labels > self.action_begin
        current = is_action & (count >= 1) & (count <= self.action_dim)
        following = is_action & (count > self.action_dim)
        return current, following

    def attention_bias(self, attention_mask, dtype=torch.float32):
        batch, seq = attention_mask.shape
        padded_key = (attention_mask == 0)[:, None, None, :].expand(batch, 1, seq, seq)
        bias = torch.zeros(batch, 1, seq, seq, dtype=dtype, device=attention_mask.device)
        return bias.masked_fill(padded_key, torch.finfo(dtype).min)

    def train_action_hidden(self, last_hidden, labels, num_patches):
        batch, length = labels.shape
        if last_hidden.shape[1] != num_patches + length:
            raise ValueError("last_hidden must be num_patches + L long")
        text_hidden = last_hidden[:, num_patches:-1]
        current, following = self.action_masks(labels[:, 1:])
        keep = current | following
        if (keep.sum(dim=1) != self.num_slots).any():
            raise ValueError("every row needs exactly chunk_len * action_dim action labels")
        return text_hidden[keep].reshape(batch, self.num_slots, -1)

    def predict(self, decoder, embed, input_ids, attention_mask, patch_embeds, action_head=None):
        input_ids, attention_mask, labels, num_prompt_tokens = self.prepare_inputs(input_ids, attention_mask)
        batch = input_ids.shape[0]
        if patch_embeds.shape[0] != batch:
            raise ValueError("patch_embeds must have the same batch size")
        current, following = self.action_masks(labels)
        x = embed(input_ids)
        x = x * ~(current | following).unsqueeze(-1)  # empty action slots
        num_patches = patch_embeds.shape[1]
        embeds = torch.cat([x[:, :1], patch_embeds, x[:, 1:]], dim=1)
        patch_mask = attention_mask.new_ones((batch, num_patches))
        mask = torch.cat([attention_mask[:, :1], patch_mask, attention_mask[:, 1:]], dim=1)
        hidden, logits = decoder(embeds, self.attention_bias(mask, embeds.dtype))
        start = num_patches + num_prompt_tokens  # one position before the first slot
        read = slice(start, start + self.num_slots)
        if action_head is not None:
            grouped = hidden[:, read].reshape(batch, self.chunk_len, -1)
            actions = action_head(grouped)
            if tuple(actions.shape) != (batch, self.chunk_len, self.action_dim):
                raise ValueError("action_head must return (B, chunk_len, action_dim)")
            return actions
        tokens = logits[:, read].argmax(dim=-1)
        index = torch.clamp(self.vocab_size - tokens - 1, 0, self.n_bins - 2)
        centers = self.centers.to(device=logits.device, dtype=logits.dtype)
        return centers[index].reshape(batch, self.chunk_len, self.action_dim)
''',
    "interview_questions": interview(
        concept=[
            "Why does autoregressive OpenVLA need K * D sequential forward passes for a chunk of K actions with D dimensions, and how does parallel decoding bring that to one?",
            "If every action slot enters the decoder as the same zero vector, what lets the model produce different actions for different slots?",
        ],
        deep_dive=[
            "Walk through the multimodal sequence OFT builds at inference: where do BOS, the image patches, the prompt, the empty token, the action slots and the stop token sit, and which output positions are read for the actions?",
            "OFT's patched attention copies the last row of Hugging Face's causal-plus-padding mask to every row. Why does that row mask exactly the padding keys, and how does the result differ from a prefix-LM mask?",
            "The L1 head reshapes (B, K * D, H) hidden states to (B, K, D * H). What does each row of the head input contain, and what would go wrong if the reshape grouped slots by dimension instead of by time step?",
        ],
        tradeoffs=[
            "Parallel decoding with an L1 head versus autoregressive discrete tokens: what do you gain in latency and lose in modeling a multimodal action distribution?",
            "Parallel L1 decoding in one pass versus a flow-matching or diffusion head that needs several passes: how would you choose for a 25-50 Hz bimanual robot?",
        ],
    ),
}
