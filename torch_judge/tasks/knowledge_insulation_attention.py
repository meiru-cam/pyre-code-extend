"""Knowledge insulation: the joint VLM / FAST / action-expert mask and the stop-gradient attention."""

from ._interview import interview

# Independent helpers for the tests: a loop-built rule mask and plain masked attention.
_HELPERS = r"""
import math
import torch

def rule_mask(prefix_pad, fast_pad, H):
    B, P = prefix_pad.shape
    F = fast_pad.shape[1]
    N = P + F + H
    m = torch.zeros(B, N, N, dtype=torch.bool)
    for b in range(B):
        valid = list(prefix_pad[b].tolist()) + list(fast_pad[b].tolist()) + [True] * H
        for i in range(N):
            for j in range(N):
                if not (valid[i] and valid[j]):
                    continue
                if j < P:
                    ok = True
                elif j < P + F:
                    ok = P <= i < P + F and j <= i
                else:
                    ok = i >= P + F
                m[b, i, j] = ok
    return m

def rule_positions(prefix_pad, fast_pad, H):
    B, P = prefix_pad.shape
    out = []
    for b in range(B):
        row, seen = [], 0
        for v in list(prefix_pad[b].tolist()) + list(fast_pad[b].tolist()):
            seen += int(v)
            row.append(seen - 1)
        base = int(prefix_pad[b].sum())
        row += [base + h for h in range(H)]
        out.append(row)
    return torch.tensor(out, dtype=torch.long)

def dense_attention(q, k, v, mask):
    scores = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
    scores = scores.masked_fill(~mask[:, None], -1e30)
    return torch.softmax(scores, dim=-1) @ v

def qkv(seed, B, heads, N, d, grad=True):
    g = torch.Generator().manual_seed(seed)
    ts = [torch.randn(B, heads, N, d, generator=g, dtype=torch.float64) for _ in range(3)]
    if grad:
        ts = [t.requires_grad_(True) for t in ts]
    return ts

def pads(seed, B, P, F):
    g = torch.Generator().manual_seed(seed + 100)
    prefix_pad = torch.rand(B, P, generator=g) < 0.75
    prefix_pad[:, 0] = True
    fast_pad = torch.rand(B, F, generator=g) < 0.75
    if F:
        fast_pad[:, 0] = True
    return prefix_pad, fast_pad

def grads(loss, tensors):
    return torch.autograd.grad(loss, tensors, allow_unused=True, retain_graph=True)
"""

TASK = {
    "title": "Knowledge Insulation: Mask and Stop-Gradient Attention",
    "difficulty": "Hard",
    "version": 1,
    "function_name": "KnowledgeInsulatedAttention",
    "description_en": r"""Build the two attention pieces of a knowledge-insulated VLA (pi0.5 + KI). One sequence holds three groups of tokens, in this order: the VLM prefix (images, prompt, state; `P` tokens), the discrete FAST action tokens the VLM learns to predict (`F` tokens), and the continuous noisy-action tokens of the action expert (`H` tokens). `N = P + F + H`. The prefix and FAST tokens use the backbone weights; the action tokens use the action-expert weights.

**Signature:** `KnowledgeInsulatedAttention()` with two methods:
- `build_mask(prefix_pad, fast_pad, num_actions) -> (mask, position_ids)`
- `attend(q, k, v, mask, num_backbone) -> Tensor`

**build_mask:**
- `prefix_pad` is a bool tensor `(B, P)` and `fast_pad` a bool tensor `(B, F)`; `True` marks a real token, `False` padding. `F` may be 0. `num_actions` is `H`. Action tokens are never padding.
- `mask` is a bool tensor `(B, N, N)`. `mask[b, i, j]` is `True` when query `i` may attend to key `j`.
- A query or key that is padding allows nothing: its whole row and column are `False`.
- Prefix queries attend to every prefix key, and to nothing else.
- FAST query `i` attends to every prefix key and to FAST keys `j <= i` (causal). It never attends to action keys.
- Action queries attend to every prefix key and to every action key (bidirectional). They never attend to FAST keys.
- `position_ids` is a long tensor `(B, N)`. For the first `P + F` tokens it is `cumsum(pad) - 1` over the concatenated prefix and FAST pad masks. Action token `h` (from 0) gets `prefix_pad[b].sum() + h`, so the actions sit at the positions they would have if the FAST tokens were absent.
- Raise `ValueError` when either pad mask is not a 2-D bool tensor, when their batch sizes differ, or when `num_actions` is not a positive integer.

**attend:**
- `q`, `k` and `v` have shape `(B, heads, N, d)`; `mask` is a bool tensor `(B, N, N)` shared by all heads. Return `(B, heads, N, d)` in the input dtype.
- Scores are `q @ k^T / sqrt(d)`. Masked scores must not take part in the softmax. A query row with no allowed key (padding) must still give finite outputs and gradients; its values are not graded.
- Query rows `i < num_backbone` attend to `k` and `v` as given.
- Query rows `i >= num_backbone` attend to keys and values whose first `num_backbone` positions are cut from the graph with a stop-gradient. Only those positions, and only on this path.
- The forward output equals ordinary masked attention. Only gradients change: a loss on action rows sends no gradient into `k[:, :, :num_backbone]` or `v[:, :, :num_backbone]`, while `q`, and the keys and values at positions `>= num_backbone`, still get their usual gradients.
- `0 <= num_backbone <= N`. Raise `ValueError` when `q`, `k`, `v` are not 4-D with equal shapes, when `mask` is not `(B, N, N)`, or when `num_backbone` is out of range.

**Example:** `P = 2`, `F = 2`, `H = 2`, no padding:

    mask, pos = KnowledgeInsulatedAttention().build_mask(
        torch.tensor([[True, True]]), torch.tensor([[True, True]]), 2)
    # mask[0].int():               pos:
    # [[1, 1, 0, 0, 0, 0],         [[0, 1, 2, 3, 2, 3]]
    #  [1, 1, 0, 0, 0, 0],
    #  [1, 1, 1, 0, 0, 0],
    #  [1, 1, 1, 1, 0, 0],
    #  [1, 1, 0, 0, 1, 1],
    #  [1, 1, 0, 0, 1, 1]]

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why two action heads.** A flow-matching action expert gives fast, continuous actions, but its randomly initialized weights send noisy gradients into the pre-trained VLM, which the KI paper (Driess et al. 2025) links to worse language following. KI trains the backbone on FAST action tokens with an ordinary next-token loss, so the backbone still learns action-relevant features, and trains the expert with flow matching on top of those features.

**The stop-gradient.** The backbone and the expert interact only through attention. Equations 5 and 6 of the paper apply a stop-gradient to the backbone keys and values on the expert's side: `P_ab = softmax(Q_a sg(K_b)^T ...)` and `E_a = P_ab sg(V_b) + P_aa V_a`. The forward pass is unchanged, so inference is identical. The paper notes that with the flow loss now training an independent set of weights, its weight can simply be set to 1. LeRobot's pi0.5 implements this in `compute_layer_complete`: when `knowledge_insulation` is on, the action queries attend to `key_states[:, :, :prefix_len].detach()` and the matching values, where `prefix_len` covers every backbone token, FAST tokens included.

**The mask.** The paper's Appendix B: a full prefix mask over images, language and state; FAST tokens attend to the prefix and autoregressively to earlier FAST tokens; action-expert tokens attend to the prefix and to one another but not to FAST tokens, to avoid information leakage between the two representations of the same actions; no VLM token attends to the action expert. The FAST tokens encode the very chunk the expert is denoising, so an expert that could read them would learn to copy instead of to act. LeRobot's FineART-VLA builds this with big_vision-style `att_masks` (FAST tokens `1` each, actions `[1, 0, ..., 0]`) and then sets the action-to-FAST block to `False`, because the cumulative rule alone would let the later action block see the FAST tokens.

**Positions.** At inference the expert runs without FAST tokens, so FineART gives the action tokens RoPE positions that continue right after the non-FAST prefix, the same positions they get at inference.""",
    "advisory_prerequisites": ["pi0_flow_matching_chunk", "fast_dct_action_tokens"],
    "hints": [
        {"level": 1, "kind": "questions", "content": "For each pair of token groups (prefix, FAST, action), which direction of attention is allowed, and which group must never see which? Which tensors on the action-query path should be cut from the graph so a flow loss cannot reach the backbone, and why does that leave the forward values unchanged?"},
        {"level": 2, "kind": "analysis", "content": "Build the mask from per-position group labels: start from 'key is in the prefix', add a lower-triangular FAST block and a full action block, then AND with the outer product of the pad vector (actions always valid). Positions: cumsum of the backbone pads minus one, then the count of valid prefix tokens plus arange(H). In attend, split the query rows at num_backbone. The first group uses k and v directly. The second uses copies of k and v whose first num_backbone positions are detached and concatenated with the rest. Fill masked scores with a large finite negative before the softmax, and concatenate the two outputs along the token axis."},
    ],
    "model_connections": [
        "pi0.5 + KI (Driess et al. 2025) trains the VLM on FAST tokens and the action expert with flow matching, with a stop-gradient between them; at inference only the expert produces actions.",
        "LeRobot's PI05 compute_layer_complete has a knowledge_insulation flag (off in stock pi0.5) that detaches the VLM keys and values on the action-query path; FineART-VLA turns it on by default and adds the action-to-FAST mask.",
        "The masks are big_vision prefix-LM masks (make_att_2d_masks): a cumulative att_masks vector gives blocks that see themselves and every earlier block.",
        "HybridVLA, a baseline in the KI paper, lets autoregressive tokens attend to the diffusion inputs; KI's mask forbids attention in both directions between the two action representations.",
    ],
    "pro_con_analysis": {
        "pros": [
            "The backbone learns from a stable next-token loss and keeps more of its VLM knowledge, while the expert still gives fast continuous actions at inference.",
            "The stop-gradient changes no forward value, so a KI-trained model runs exactly like pi0.5 at test time.",
        ],
        "cons": [
            "Training sequences carry both the FAST tokens and the noisy action chunk, so they are longer than for either objective alone.",
            "The insulation only makes sense when the backbone also learns to predict actions; the paper calls the restriction sensible if and only if the backbone is trained on discrete actions too.",
        ],
    },
    "sources": [
        {
            "kind": "code",
            "url": "https://github.com/huggingface/lerobot",
            "commit": "75cd2ba0ce9ff47686bb2bd7b7f65840606c02ee",
            "path": "src/lerobot/policies/pi05/modeling_pi05.py",
            "symbol": "compute_layer_complete (knowledge_insulation branch), PI05Pytorch.embed_suffix",
            "license": "Apache-2.0",
            "adapted": "The split of query rows at prefix_len, with the VLM keys and values detached only on the action-query path, and the suffix att_masks [1, 0, ..., 0].",
            "simplifications": "Only the attention core is kept: no projections, RoPE, norms, MLPs, GQA or attention_fn hooks. "
                               "q, k, v come in as (B, heads, N, d) and the output stays in that layout (upstream returns (B, N, heads, d)). "
                               "Masked scores are filled with torch.finfo(dtype).min rather than adding -2.3819763e38, and the softmax runs in the input dtype rather than float32. "
                               "The mask is boolean (B, N, N) rather than the additive 4-D layout.",
        },
        {
            "kind": "code",
            "url": "https://github.com/huggingface/lerobot",
            "commit": "75cd2ba0ce9ff47686bb2bd7b7f65840606c02ee",
            "path": "src/lerobot/policies/fineart_vla/modeling_fineart_vla.py",
            "symbol": "FineARTVLAPytorch._embed_supervised_prefix, FineARTVLAPytorch._combined_prefix_and_flow",
            "license": "Apache-2.0",
            "adapted": "FAST tokens appended to the prefix with att_masks 1, the action-to-FAST block set to False, and action positions parallel to FAST.",
            "simplifications": "The non-FAST prefix is fully bidirectional: FineART's causal language-target marks are dropped. "
                               "The mask is built directly from pad masks instead of from embedded tokens. "
                               "Amortized flow (several action blocks per prefix) and FlexAttention block masks are dropped.",
        },
        {
            "kind": "code",
            "url": "https://github.com/huggingface/lerobot",
            "commit": "75cd2ba0ce9ff47686bb2bd7b7f65840606c02ee",
            "path": "src/lerobot/policies/common/vla_utils.py",
            "symbol": "make_att_2d_masks",
            "license": "Apache-2.0",
            "adapted": "The cumulative att_masks rule and the pad outer product that the oracle uses to build the reference mask.",
            "simplifications": "Learners write the mask from the group rules instead of from a cumulative att_masks vector; the two agree for this token layout.",
        },
        {
            "kind": "paper",
            "url": "https://arxiv.org/abs/2505.23705",
            "section": "Driess et al. 2025, Knowledge Insulating Vision-Language-Action Models, Sections 5.1-5.2 and Appendix B",
            "equation": "Equations 5 and 6 (stop-gradient on backbone keys and values for action-expert queries)",
            "note": "Joint FAST and flow training, the FAST-action attention mask, and the stop-gradient in attention.",
        },
    ],
    "tests": [
        {"name": "Mask and positions for a small layout", "behavior": "attention.masking", "code": r"""
import torch
ki = {fn}()
mask, pos = ki.build_mask(torch.tensor([[True, True]]), torch.tensor([[True, True]]), 2)
want = torch.tensor([[[1, 1, 0, 0, 0, 0],
                      [1, 1, 0, 0, 0, 0],
                      [1, 1, 1, 0, 0, 0],
                      [1, 1, 1, 1, 0, 0],
                      [1, 1, 0, 0, 1, 1],
                      [1, 1, 0, 0, 1, 1]]], dtype=torch.bool)
assert mask.dtype == torch.bool and torch.equal(mask, want), mask.int()
assert pos.dtype == torch.long and pos.tolist() == [[0, 1, 2, 3, 2, 3]], pos
"""},
        {"name": "Same forward as masked attention; action loss skips backbone K and V", "behavior": "gradient.flow", "code": _HELPERS + r"""
ki = {fn}()
mask, _ = ki.build_mask(torch.tensor([[True, True]]), torch.tensor([[True, True]]), 2)
q, k, v = qkv(0, 1, 2, 6, 4)
out = ki.attend(q, k, v, mask, 4)
assert out.shape == (1, 2, 6, 4), out.shape
assert torch.allclose(out, dense_attention(q, k, v, mask), atol=1e-10), "forward changed"
gq, gk, gv = grads(out[:, :, 4:].sum(), [q, k, v])
assert gk is not None and torch.all(gk[:, :, :4] == 0) and torch.all(gv[:, :, :4] == 0), "action loss reached backbone K/V"
assert gk[:, :, 4:].abs().sum() > 0 and gq[:, :, 4:].abs().sum() > 0, "action-side gradients must survive"
"""},
        {"name": "Seeded masks with padding match the rule", "visibility": "unshown", "behavior": "attention.masking",
         "failure_message": "Prefix sees only the prefix; FAST sees the prefix and earlier FAST tokens; actions see the prefix and all actions; padding clears its row and column.",
         "code": _HELPERS + r"""
ki = {fn}()
for seed, (B, P, F, H) in enumerate(((2, 5, 4, 3), (3, 3, 6, 5), (1, 7, 1, 2), (4, 2, 3, 1))):
    prefix_pad, fast_pad = pads(seed, B, P, F)
    mask, _ = ki.build_mask(prefix_pad, fast_pad, H)
    assert mask.dtype == torch.bool and mask.shape == (B, P + F + H, P + F + H), (mask.dtype, mask.shape)
    assert torch.equal(mask, rule_mask(prefix_pad, fast_pad, H)), (seed, mask.int())
"""},
        {"name": "Action tokens never see FAST tokens", "visibility": "unshown", "behavior": "attention.masking",
         "failure_message": "Set the action-query-to-FAST-key block to False: the expert must not read the discrete tokens of the chunk it denoises.",
         "code": r"""
import torch
ki = {fn}()
B, P, F, H = 2, 3, 4, 5
mask, _ = ki.build_mask(torch.ones(B, P, dtype=torch.bool), torch.ones(B, F, dtype=torch.bool), H)
assert not mask[:, P + F:, P:P + F].any(), mask[0].int()
assert mask[:, P + F:, :P].all() and mask[:, P + F:, P + F:].all(), "actions see the prefix and every action"
"""},
        {"name": "FAST tokens are causal and never see actions", "visibility": "unshown", "behavior": "attention.masking",
         "failure_message": "FAST query i sees FAST keys j <= i only, and no action key; the backbone never attends to the action expert.",
         "code": r"""
import torch
ki = {fn}()
B, P, F, H = 1, 2, 5, 3
mask, _ = ki.build_mask(torch.ones(B, P, dtype=torch.bool), torch.ones(B, F, dtype=torch.bool), H)
fast = mask[0, P:P + F, P:P + F]
assert torch.equal(fast, torch.tril(torch.ones(F, F, dtype=torch.bool))), fast.int()
assert not mask[0, :P + F, P + F:].any(), "a backbone token attends to an action token"
assert mask[0, P:P + F, :P].all(), "FAST tokens see the whole prefix"
"""},
        {"name": "Prefix tokens see only the prefix, bidirectionally", "visibility": "unshown", "behavior": "attention.masking",
         "failure_message": "Prefix queries attend to every valid prefix key and to no FAST or action key.",
         "code": r"""
import torch
ki = {fn}()
mask, _ = ki.build_mask(torch.ones(2, 4, dtype=torch.bool), torch.ones(2, 3, dtype=torch.bool), 2)
assert mask[:, :4, :4].all(), mask[0].int()
assert not mask[:, :4, 4:].any(), mask[0].int()
"""},
        {"name": "Padding clears rows and columns, including FAST padding", "visibility": "unshown", "behavior": "edge.empty_or_boundary",
         "failure_message": "A padded prefix or FAST token neither attends nor is attended to; action tokens are always valid.",
         "code": r"""
import torch
ki = {fn}()
prefix_pad = torch.tensor([[True, False, True], [True, True, True]])
fast_pad = torch.tensor([[True, True, False, False], [True, False, True, True]])
mask, _ = ki.build_mask(prefix_pad, fast_pad, 3)
pad = torch.cat([prefix_pad, fast_pad, torch.ones(2, 3, dtype=torch.bool)], dim=1)
for b in range(2):
    for i in range(10):
        if not pad[b, i]:
            assert not mask[b, i].any() and not mask[b, :, i].any(), (b, i, mask[b].int())
assert mask[0, 7:, 7:].all(), "action tokens are never padding"
"""},
        {"name": "Positions: cumsum on the backbone, actions parallel to FAST", "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "Backbone positions are cumsum(pad) - 1; action h gets (number of valid prefix tokens) + h, ignoring the FAST tokens.",
         "code": _HELPERS + r"""
ki = {fn}()
for seed, (B, P, F, H) in enumerate(((3, 5, 4, 3), (2, 4, 0, 4), (2, 1, 6, 2))):
    prefix_pad, fast_pad = pads(seed + 7, B, P, F)
    _, pos = ki.build_mask(prefix_pad, fast_pad, H)
    assert pos.dtype == torch.long and pos.shape == (B, P + F + H), (pos.dtype, pos.shape)
    assert torch.equal(pos, rule_positions(prefix_pad, fast_pad, H)), (seed, pos)
"""},
        {"name": "No FAST tokens", "visibility": "unshown", "behavior": "edge.empty_or_boundary",
         "failure_message": "With F = 0 the mask is the pi0.5 prefix-LM mask: prefix bidirectional, actions see the prefix and each other.",
         "code": _HELPERS + r"""
ki = {fn}()
prefix_pad = torch.tensor([[True, True, False], [True, True, True]])
fast_pad = torch.zeros(2, 0, dtype=torch.bool)
mask, pos = ki.build_mask(prefix_pad, fast_pad, 2)
assert mask.shape == (2, 5, 5), mask.shape
assert torch.equal(mask, rule_mask(prefix_pad, fast_pad, 2)), mask.int()
assert pos.tolist() == [[0, 1, 1, 2, 3], [0, 1, 2, 3, 4]], pos
"""},
        {"name": "Forward equals masked attention for any split", "visibility": "unshown", "behavior": "numerics.stability",
         "failure_message": "The stop-gradient changes no value: scores q @ k^T / sqrt(d), masked, softmax over keys, times v.",
         "code": _HELPERS + r"""
ki = {fn}()
for seed, (B, heads, P, F, H, d) in enumerate(((2, 3, 4, 3, 5, 8), (1, 1, 2, 0, 3, 4), (3, 2, 5, 2, 2, 6))):
    prefix_pad, fast_pad = pads(seed, B, P, F)
    mask, _ = ki.build_mask(prefix_pad, fast_pad, H)
    N = P + F + H
    q, k, v = qkv(seed, B, heads, N, d, grad=False)
    valid = mask.any(dim=-1)
    for nb in (0, P, P + F, N):
        out = ki.attend(q, k, v, mask, nb)
        want = dense_attention(q, k, v, mask)
        assert out.shape == q.shape and out.dtype == q.dtype, (out.shape, out.dtype)
        assert torch.allclose(out.transpose(1, 2)[valid], want.transpose(1, 2)[valid], atol=1e-10), (seed, nb)
"""},
        {"name": "Action-row loss sends no gradient into backbone keys or values", "visibility": "unshown", "behavior": "gradient.flow",
         "failure_message": "On the action-query path detach both k and v at positions < num_backbone (paper eq. 5 and 6).",
         "code": _HELPERS + r"""
ki = {fn}()
for seed in (1, 2, 3):
    B, heads, P, F, H, d = 2, 2, 4, 3, 4, 5
    prefix_pad, fast_pad = pads(seed, B, P, F)
    mask, _ = ki.build_mask(prefix_pad, fast_pad, H)
    q, k, v = qkv(seed, B, heads, P + F + H, d)
    w = torch.randn(B, heads, H, d, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)
    out = ki.attend(q, k, v, mask, P + F)
    gq, gk, gv = grads((out[:, :, P + F:] * w).sum(), [q, k, v])
    for name, g in (("k", gk), ("v", gv)):
        assert g is not None and torch.all(g[:, :, :P + F] == 0), (seed, name, "backbone positions got a gradient")
"""},
        {"name": "Action-side gradients match ordinary attention", "visibility": "unshown", "behavior": "gradient.flow",
         "failure_message": "Keep gradients to the action queries and to keys and values at positions >= num_backbone; detach nothing else.",
         "code": _HELPERS + r"""
ki = {fn}()
for seed in (4, 5):
    B, heads, P, F, H, d = 2, 3, 3, 2, 4, 6
    prefix_pad, fast_pad = pads(seed, B, P, F)
    mask, _ = ki.build_mask(prefix_pad, fast_pad, H)
    nb = P + F
    q, k, v = qkv(seed, B, heads, nb + H, d)
    w = torch.randn(B, heads, H, d, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)
    gq, gk, gv = grads((ki.attend(q, k, v, mask, nb)[:, :, nb:] * w).sum(), [q, k, v])
    rq, rk, rv = grads((dense_attention(q, k, v, mask)[:, :, nb:] * w).sum(), [q, k, v])
    assert gq is not None and torch.allclose(gq, rq, atol=1e-9), (seed, "q")
    assert torch.allclose(gk[:, :, nb:], rk[:, :, nb:], atol=1e-9), (seed, "action k")
    assert torch.allclose(gv[:, :, nb:], rv[:, :, nb:], atol=1e-9), (seed, "action v")
"""},
        {"name": "Backbone-row gradients are untouched", "visibility": "unshown", "behavior": "gradient.flow",
         "failure_message": "Rows < num_backbone attend to the original k and v; the stop-gradient applies only on the action-query path.",
         "code": _HELPERS + r"""
ki = {fn}()
for seed in (6, 7):
    B, heads, P, F, H, d = 2, 2, 4, 3, 3, 4
    prefix_pad, fast_pad = pads(seed, B, P, F)
    mask, _ = ki.build_mask(prefix_pad, fast_pad, H)
    nb = P + F
    q, k, v = qkv(seed, B, heads, nb + H, d)
    valid = torch.cat([prefix_pad, fast_pad], dim=1)[:, None, :, None].to(torch.float64)
    w = torch.randn(B, heads, nb, d, generator=torch.Generator().manual_seed(seed), dtype=torch.float64) * valid
    got = grads((ki.attend(q, k, v, mask, nb)[:, :, :nb] * w).sum(), [q, k, v])
    want = grads((dense_attention(q, k, v, mask)[:, :, :nb] * w).sum(), [q, k, v])
    for name, a, b in zip("qkv", got, want):
        assert a is not None and torch.allclose(a, b, atol=1e-9), (seed, name)
"""},
        {"name": "Split sits exactly at num_backbone", "visibility": "unshown", "behavior": "gradient.flow",
         "failure_message": "Detach exactly the first num_backbone key/value positions for the query rows >= num_backbone, whatever the mask allows.",
         "code": _HELPERS + r"""
ki = {fn}()
B, heads, N, d = 1, 2, 7, 3
full = torch.ones(B, N, N, dtype=torch.bool)
for nb in (1, 3, 5):
    q, k, v = qkv(nb, B, heads, N, d)
    out = ki.attend(q, k, v, full, nb)
    gq, gk, gv = grads(out[:, :, nb:].pow(2).sum(), [q, k, v])
    assert torch.all(gk[:, :, :nb] == 0) and torch.all(gv[:, :, :nb] == 0), nb
    assert torch.all(gk[:, :, nb:].abs().sum(-1) > 0) and torch.all(gv[:, :, nb:].abs().sum(-1) > 0), nb
    gq, gk, gv = grads(out[:, :, :nb].pow(2).sum(), [q, k, v])
    assert torch.all(gk[:, :, :nb].abs().sum(-1) > 0) and torch.all(gv[:, :, :nb].abs().sum(-1) > 0), nb
"""},
        {"name": "num_backbone = 0 or N is plain attention", "visibility": "unshown", "behavior": "edge.empty_or_boundary",
         "failure_message": "With no backbone rows nothing is detached; with no action rows the backbone path is ordinary attention.",
         "code": _HELPERS + r"""
ki = {fn}()
B, heads, N, d = 2, 2, 5, 4
mask = torch.rand(B, N, N, generator=torch.Generator().manual_seed(3)) < 0.7
mask |= torch.eye(N, dtype=torch.bool)
for nb in (0, N):
    q, k, v = qkv(10 + nb, B, heads, N, d)
    got = grads(ki.attend(q, k, v, mask, nb).pow(2).sum(), [q, k, v])
    want = grads(dense_attention(q, k, v, mask).pow(2).sum(), [q, k, v])
    for name, a, b in zip("qkv", got, want):
        assert a is not None and torch.allclose(a, b, atol=1e-9), (nb, name)
"""},
        {"name": "Padding rows stay finite", "visibility": "unshown", "behavior": "numerics.stability",
         "failure_message": "Fill masked scores with a large finite negative, not -inf, so a fully masked padding row does not produce NaN outputs or gradients.",
         "code": _HELPERS + r"""
ki = {fn}()
prefix_pad = torch.tensor([[True, False, True, False]])
fast_pad = torch.tensor([[True, False]])
mask, _ = ki.build_mask(prefix_pad, fast_pad, 3)
q, k, v = qkv(9, 1, 2, 9, 4)
out = ki.attend(q, k, v, mask, 6)
assert torch.isfinite(out).all(), "padding rows produced NaN or inf"
valid = mask.any(dim=-1)
gq, gk, gv = grads(out.transpose(1, 2)[valid].sum(), [q, k, v])
assert all(torch.isfinite(g).all() for g in (gq, gk, gv)), "non-finite gradient"
"""},
        {"name": "Two-expert layer: flow loss trains only the expert", "visibility": "unshown", "behavior": "gradient.flow",
         "failure_message": "Built together, the mask and attend must keep a loss on action outputs out of the backbone weights and keep FAST outputs blind to the actions.",
         "code": _HELPERS + r"""
ki = {fn}()
g = torch.Generator().manual_seed(42)
B, P, F, H, D, heads = 2, 4, 3, 3, 8, 2
d = D // heads
x_b = torch.randn(B, P + F, D, generator=g, dtype=torch.float64)
x_a = torch.randn(B, H, D, generator=g, dtype=torch.float64)
Wb = [(torch.randn(D, D, generator=g, dtype=torch.float64) / D ** 0.5).requires_grad_(True) for _ in range(3)]
Wa = [(torch.randn(D, D, generator=g, dtype=torch.float64) / D ** 0.5).requires_grad_(True) for _ in range(3)]
prefix_pad, fast_pad = torch.ones(B, P, dtype=torch.bool), torch.ones(B, F, dtype=torch.bool)
mask, _ = ki.build_mask(prefix_pad, fast_pad, H)
def proj(xb, xa, i):
    t = torch.cat([xb @ Wb[i], xa @ Wa[i]], dim=1)
    return t.view(B, P + F + H, heads, d).transpose(1, 2)
def layer(xb, xa):
    return ki.attend(proj(xb, xa, 0), proj(xb, xa, 1), proj(xb, xa, 2), mask, P + F)
out = layer(x_b, x_a)
flow_loss = out[:, :, P + F:].pow(2).sum()
gb = grads(flow_loss, Wb)
ga = grads(flow_loss, Wa)
assert all(t is None or torch.all(t == 0) for t in gb), "flow loss reached backbone weights"
assert all(t is not None and t.abs().sum() > 0 for t in ga), "flow loss must train the expert"
fast_loss = out[:, :, P:P + F].pow(2).sum()
assert all(t is not None and t.abs().sum() > 0 for t in grads(fast_loss, Wb)), "FAST loss must train the backbone"
x_a2 = x_a + torch.randn(B, H, D, generator=g, dtype=torch.float64)
out2 = layer(x_b, x_a2)
assert torch.allclose(out[:, :, :P + F], out2[:, :, :P + F], atol=1e-12), "backbone tokens read the actions"
x_b2 = x_b.clone(); x_b2[:, P:] += 1.0
out3 = layer(x_b2, x_a)
assert torch.allclose(out[:, :, P + F:], out3[:, :, P + F:], atol=1e-12), "action tokens read the FAST tokens"
"""},
        {"name": "Inputs, dtype and bad arguments", "visibility": "unshown", "behavior": "contract.signature",
         "failure_message": "Return the input dtype; raise ValueError for non-bool or non-2-D pads, mismatched batches, num_actions < 1, non-4-D or mismatched q/k/v, a wrong mask shape, or num_backbone outside 0..N.",
         "code": r"""
import torch
ki = {fn}()
q = torch.randn(1, 2, 5, 4, dtype=torch.float32)
mask, _ = ki.build_mask(torch.ones(1, 2, dtype=torch.bool), torch.ones(1, 1, dtype=torch.bool), 2)
out = ki.attend(q, q, q, mask, 3)
assert out.dtype == torch.float32 and out.shape == q.shape, (out.dtype, out.shape)
bad = [
    lambda: ki.build_mask(torch.ones(1, 2), torch.ones(1, 1, dtype=torch.bool), 2),
    lambda: ki.build_mask(torch.ones(2, dtype=torch.bool), torch.ones(1, 1, dtype=torch.bool), 2),
    lambda: ki.build_mask(torch.ones(1, 2, dtype=torch.bool), torch.ones(2, 1, dtype=torch.bool), 2),
    lambda: ki.build_mask(torch.ones(1, 2, dtype=torch.bool), torch.ones(1, 1, dtype=torch.bool), 0),
    lambda: ki.build_mask(torch.ones(1, 2, dtype=torch.bool), torch.ones(1, 1, dtype=torch.bool), 2.0),
    lambda: ki.attend(q[0], q[0], q[0], mask, 3),
    lambda: ki.attend(q, q[:, :, :4], q, mask, 3),
    lambda: ki.attend(q, q, q, mask[:, :4, :4], 3),
    lambda: ki.attend(q, q, q, mask, -1),
    lambda: ki.attend(q, q, q, mask, 6),
]
for i, call in enumerate(bad):
    try:
        call()
    except ValueError:
        pass
    else:
        raise AssertionError(f"case {i} was accepted")
"""},
    ],
    "solution": '''import math

import torch


class KnowledgeInsulatedAttention:
    @staticmethod
    def _check_pad(name, pad):
        if not isinstance(pad, torch.Tensor) or pad.dtype != torch.bool or pad.ndim != 2:
            raise ValueError(f"{name} must be a 2-D bool tensor")

    def build_mask(self, prefix_pad, fast_pad, num_actions):
        self._check_pad("prefix_pad", prefix_pad)
        self._check_pad("fast_pad", fast_pad)
        if fast_pad.shape[0] != prefix_pad.shape[0]:
            raise ValueError("prefix_pad and fast_pad must have the same batch size")
        if isinstance(num_actions, bool) or not isinstance(num_actions, int) or num_actions < 1:
            raise ValueError("num_actions must be a positive integer")
        batch, num_prefix = prefix_pad.shape
        num_backbone = num_prefix + fast_pad.shape[1]
        device = prefix_pad.device
        total = num_backbone + num_actions

        idx = torch.arange(total, device=device)
        is_prefix = idx < num_prefix
        is_fast = (idx >= num_prefix) & (idx < num_backbone)
        is_action = idx >= num_backbone
        query, key = idx[:, None], idx[None, :]
        # Everyone sees the prefix; FAST is causal; actions see each other but never FAST.
        allowed = is_prefix[None, :].expand(total, total)
        allowed = allowed | (is_fast[:, None] & is_fast[None, :] & (key <= query))
        allowed = allowed | (is_action[:, None] & is_action[None, :])

        actions_valid = torch.ones(batch, num_actions, dtype=torch.bool, device=device)
        pad = torch.cat([prefix_pad, fast_pad, actions_valid], dim=1)
        mask = allowed[None] & pad[:, :, None] & pad[:, None, :]

        backbone_pos = torch.cumsum(pad[:, :num_backbone].long(), dim=1) - 1
        # Actions take the positions they have at inference, when no FAST tokens exist.
        first_action = prefix_pad.long().sum(dim=1, keepdim=True)
        action_pos = first_action + torch.arange(num_actions, device=device)
        return mask, torch.cat([backbone_pos, action_pos], dim=1)

    def attend(self, q, k, v, mask, num_backbone):
        if q.ndim != 4 or k.shape != q.shape or v.shape != q.shape:
            raise ValueError("q, k and v must share one (B, heads, N, d) shape")
        batch, _, total, dim = q.shape
        if mask.shape != (batch, total, total):
            raise ValueError("mask must have shape (B, N, N)")
        if isinstance(num_backbone, bool) or not isinstance(num_backbone, int) or not 0 <= num_backbone <= total:
            raise ValueError("num_backbone must be an integer in [0, N]")
        scale = 1.0 / math.sqrt(dim)

        def masked_attention(queries, keys, values, rows):
            scores = queries @ keys.transpose(-1, -2) * scale
            scores = scores.masked_fill(~rows[:, None], torch.finfo(scores.dtype).min)
            return torch.softmax(scores, dim=-1) @ values

        nb = num_backbone
        backbone_out = masked_attention(q[:, :, :nb], k, v, mask[:, :nb])
        # Knowledge insulation: the action queries read backbone K/V without a gradient path.
        k_insulated = torch.cat([k[:, :, :nb].detach(), k[:, :, nb:]], dim=2)
        v_insulated = torch.cat([v[:, :, :nb].detach(), v[:, :, nb:]], dim=2)
        action_out = masked_attention(q[:, :, nb:], k_insulated, v_insulated, mask[:, nb:])
        return torch.cat([backbone_out, action_out], dim=2)
''',
    "interview_questions": interview(
        concept=[
            "What problem does knowledge insulation solve when a randomly initialized action expert is attached to a pre-trained VLM, and why does the backbone still need an action objective of its own?",
            "In a KI-trained pi0.5, which token groups may attend to which: the VLM prefix, the FAST action tokens and the action-expert tokens?",
        ],
        deep_dive=[
            "Where exactly does the stop-gradient go in the joint attention, and why does it leave the forward pass and inference unchanged?",
            "Why must the action-expert tokens be blocked from the FAST tokens, and why is that block needed on top of the cumulative prefix-LM mask?",
            "With FAST tokens in the training sequence but not at inference, what RoPE positions should the action tokens get, and what goes wrong if you simply continue counting after the FAST tokens?",
        ],
        tradeoffs=[
            "Knowledge insulation versus freezing the VLM versus full joint training: what does each do to language following, action quality and training speed?",
            "KI's mask forbids attention in both directions between FAST and continuous actions, while HybridVLA lets autoregressive tokens attend to the diffusion inputs. What does each choice buy and cost?",
        ],
    ),
}
