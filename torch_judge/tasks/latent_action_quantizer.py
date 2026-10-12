"""Latent action quantizer: VQ of a frame-feature difference with straight-through or noise-substitution gradients."""

from ._interview import interview

TASK = {
    "title": "Latent Action Quantizer",
    "difficulty": "Medium",
    "version": 1,
    "function_name": "latent_action_quantize",
    "description_en": r"""Turn the change between two video frames into a discrete latent action, the pseudo-label LAPA and UniVLA use to pretrain a VLA on video that has no robot actions.

**Signatures:** define both functions at the top level of your code.
- `latent_action_quantize(f_t, f_tH, codebook, mode="ste", noise=None, beta=0.25, eps=1e-12) -> (z_q, idx, perplexity, vq_loss)`
- `replace_dead_codes(codebook, usage, num_batches, threshold, jitter=None) -> (new_codebook, dead)`

**Parameters of `latent_action_quantize`:**
- `f_t`, `f_tH` — float tensors of the same shape `(..., D)`: encoder features of frame `t` and frame `t + H`. Any number of leading dimensions.
- `codebook` — float tensor of shape `(K, D)`.
- `mode` — `"ste"` (straight-through) or `"nsvq"` (noise substitution).
- `noise` — float tensor shaped like `f_t`. Required in `"nsvq"` mode, ignored in `"ste"` mode.
- `beta` — weight of the commitment term in `"ste"` mode.
- `eps` — small constant used by the NSVQ scale and the perplexity log.

**Steps:**
- The input vector is the feature difference `x = f_tH - f_t`.
- `idx` is the index of the nearest code to each vector of `x` in squared Euclidean distance. On an exact tie, pick the lowest index.
- `q = codebook[idx]`.
- `"ste"` mode: `z_q = x + stop_grad(q - x)`, and

        vq_loss = mean((stop_grad(x) - q) ** 2) + beta * mean((x - stop_grad(q)) ** 2)

  where each mean runs over every element.
- `"nsvq"` mode: with `||.||` the L2 norm over the last dimension, computed per vector,

        z_q = x + (||x - q|| / ||noise|| + eps) * noise

  Gradients flow through every term, including `||x - q||`. `vq_loss` is a zero scalar tensor.
- `perplexity`: let `p_k` be the fraction of all vectors (over every leading dimension together) whose `idx` is `k`. Then `perplexity = exp(-sum_k p_k * log(p_k + eps))`.

**Returns:** `z_q` with the shape and dtype of `f_t`; `idx` as a `torch.long` tensor of shape `f_t.shape[:-1]`; `perplexity` and `vq_loss` as 0-dim tensors in the dtype of `f_t`.

**`replace_dead_codes`** restarts codes that are rarely chosen:
- `usage` — tensor of shape `(K,)` holding how many times each code was chosen over the last `num_batches` batches.
- A code is dead when `usage[k] / num_batches < threshold`. Every other code is live.
- Let `dead` and `live` be the dead and live indices in ascending order. The `j`-th dead code is overwritten with a copy of code `live[j % len(live)]`.
- `jitter`, when given, has shape `(K, D)`. Row `k` of it is added to code `k` only if code `k` was replaced.
- If no code is live, nothing is copied: every code is dead, and `jitter` (when given) is added to every row.
- Return a new tensor that does not require grad, plus `dead` as a `torch.long` tensor (empty if no code is dead).

**Constraints:**
- Never modify `codebook` or `usage` in place.
- In `"ste"` mode the codebook receives gradient only through `vq_loss`, never through `z_q`.
- Raise `ValueError` for an unknown `mode`, for `"nsvq"` without a `noise` shaped like `f_t`, and when `f_t` and `f_tH` differ in shape or their last dimension is not `D`.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why latent actions.** Most video of people and robots has no action labels. LAPA trains a VQ-VAE on frame pairs: the encoder sees frames `t` and `t + H` and acts as an inverse-dynamics model, the decoder reconstructs frame `t + H` from frame `t` and the quantized code and acts as a world model. The code index becomes a pseudo-action, a VLM is pretrained to predict it from the image and instruction, and a small labelled set later swaps in a real action head.

**The difference.** LAPA's quantizer subtracts the two encoded frames (`d1 = e2 - e1` in its Appendix A) before looking up the codebook, so the code describes the change rather than the scene. UniVLA instead appends learned latent-action queries to DINOv2 patch features and reads the action off a transformer, but quantizes the result with the same nearest-code lookup.

**Two ways to pass a gradient through argmin.** The straight-through estimator copies the gradient of `z_q` onto `x` unchanged, so the codebook needs its own loss: the codebook term pulls codes toward the encoder outputs and the beta-weighted commitment term keeps the encoder near its codes. UniVLA trains with exactly this loss, with a default beta of 0.25. NSVQ (Vali and Backstrom, 2022), which LAPA adopts to avoid collapse, instead replaces the quantization error with a random direction of the same length. The decoder then sees an input as far from `x` as `q` is (up to eps), and the codebook learns through `||x - q||` with no extra loss term.

**Dead codes.** Codes that no encoder output reaches stop learning. LAPA calls a replacement routine during early training that copies used codes, plus a tiny random perturbation, into codes whose usage fell below a threshold. Upstream shuffles the donor list when there are fewer live codes than dead ones; this exercise uses the unshuffled round-robin order so results are deterministic. Perplexity, the exponentiated entropy of code usage, is the usual health check: it equals the number of codes in use when usage is uniform.""",
    "advisory_prerequisites": ["fast_dct_action_tokens", "rssm_kl_balance", "softmax_entropy"],
    "hints": [
        {"level": 1, "kind": "questions", "content": "Which tensor does argmin give no gradient to, and how does each mode route a gradient around it? In STE mode, what would the codebook learn if vq_loss were missing? Which norm in the NSVQ formula carries the codebook gradient? Why does a code with zero usage need eps inside the log?"},
        {"level": 2, "kind": "analysis", "content": "Validate the mode and shapes, form x as the later features minus the earlier ones, and flatten the leading dimensions. Expand the squared distance as |x|^2 - 2 x.c + |c|^2 and take argmin, which returns the first minimum. Gather the codes, then build z_q and vq_loss for the chosen mode with detach in the right places. Count code usage with bincount over all flattened vectors for the perplexity. For replacement, split indices by the usage rate, pair the j-th dead index with live index j modulo the live count, copy, and add jitter rows for replaced codes only, all on a detached clone."},
    ],
    "model_connections": [
        "LAPA (Ye et al., 2024) quantizes the difference of two encoded frames with NSVQ, uses the code as a latent-action label to pretrain a VLM, then fine-tunes a fresh action head on a small set of robot trajectories.",
        "UniVLA's latent action model quantizes DINOv2-based latent-action tokens with a nearest-code VectorQuantizer and a straight-through estimator, trained with a codebook loss plus 0.25 times a commitment loss.",
        "GR00T N1 trains a VQ-VAE on consecutive video frames and uses its continuous pre-quantized embedding as the action label of a separate \"LAPA\" embodiment during pretraining.",
    ],
    "pro_con_analysis": {
        "pros": [
            "Turns unlabelled human and robot video into action-like supervision for VLA pretraining.",
            "A small codebook gives the VLM a short discrete action vocabulary that its token head can predict directly.",
            "NSVQ trains the codebook without a separate codebook or commitment loss.",
        ],
        "cons": [
            "Latent codes can capture camera motion or other changes unrelated to the agent; UniVLA adds a language-conditioned stage to separate task-relevant dynamics for this reason.",
            "Codes still need a mapping to real actions, so a labelled fine-tuning set and a new action head are required.",
            "VQ training needs care against unused codes: replacement, perplexity monitoring and tuning of the threshold and period.",
        ],
    },
    "sources": [
        {'kind': 'code',
         'url': 'https://github.com/LatentActionPretraining/LAPA/blob/46aca51d7faebcec02d7d323bbb3820c2df07bc6/laq/laq_model/nsvq.py',
         'commit': '46aca51d7faebcec02d7d323bbb3820c2df07bc6',
         'path': 'laq/laq_model/nsvq.py',
         'symbol': 'NSVQ.forward, NSVQ.replace_unused_codebooks',
         'license': 'MIT',
         'adapted': 'Feature difference, expanded squared-distance argmin, NSVQ noise substitution with eps added to the norm ratio, perplexity with eps inside the log, and threshold-based dead-code replacement.',
         'simplifications': 'The project_in/CNN encode and project_out decode are dropped and the features arrive already encoded. '
                            'The normal noise is passed in instead of sampled. Dead-code donors are the live codes in round-robin order '
                            '(upstream shuffles the tiled donor list when live codes are fewer than dead ones), and the eps * randn '
                            'perturbation is a caller-supplied jitter. Usage counting and its reset are left to the caller.'},
        {'kind': 'code',
         'url': 'https://github.com/OpenDriveLab/UniVLA/blob/0ab9e9dd3074ffa14fdd1a00aa2aa84e4f8e1352/latent_action_model/genie/modules/blocks.py',
         'commit': '0ab9e9dd3074ffa14fdd1a00aa2aa84e4f8e1352',
         'path': 'latent_action_model/genie/modules/blocks.py',
         'symbol': 'VectorQuantizer.forward',
         'license': 'Apache-2.0',
         'adapted': 'Nearest-code lookup over the last dimension with any leading dimensions, and the straight-through output x + (z - x).detach().',
         'simplifications': 'torch.cdist is replaced by the expanded squared distance, which has the same argmin; the usage buffer and random_restart are not used.'},
        {'kind': 'code',
         'url': 'https://github.com/OpenDriveLab/UniVLA/blob/0ab9e9dd3074ffa14fdd1a00aa2aa84e4f8e1352/latent_action_model/genie/model.py',
         'commit': '0ab9e9dd3074ffa14fdd1a00aa2aa84e4f8e1352',
         'path': 'latent_action_model/genie/model.py',
         'symbol': 'DINO_LAM.shared_step (q_loss, commit_loss, vq_beta = 0.25)',
         'license': 'Apache-2.0',
         'adapted': 'vq_loss = mean((sg(emb) - z)^2) + beta * mean((emb - sg(z))^2).',
         'simplifications': 'The reconstruction loss and the stage-2 task-irrelevant codebook terms are omitted.'},
        {'kind': 'paper',
         'url': 'https://arxiv.org/abs/2410.11758',
         'section': 'LAPA, Ye et al. 2024: Section 3.1 Latent Action Quantization and Appendix A (d1 = e2 - e1, NSVQ Eq. 2)',
         'note': 'Eq. 2 has no eps; the code adds eps to the norm ratio, which this exercise follows.'},
        {'kind': 'paper',
         'url': 'https://arxiv.org/abs/2505.06111',
         'section': 'UniVLA, Bu et al. 2025: Section III-A Latent action quantization and Latent action decoupling',
         'note': 'DINOv2 features, VQ-VAE objective, two-stage task-centric latent actions.'},
        {'kind': 'paper',
         'url': 'https://arxiv.org/abs/2503.14734',
         'section': 'GR00T N1, NVIDIA 2025: Section 2.2 Latent Actions',
         'note': 'VQ-VAE latent actions from consecutive frames used as a "LAPA" embodiment.'},
    ],
    "tests": [
        {"name": "Straight-through quantization of a frame difference", "behavior": "routing.selection", "code": r"""
import math, torch
codebook = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]])
f_t = torch.tensor([[1.0, 1.0], [0.0, 0.0], [2.0, 0.0]])
f_tH = torch.tensor([[1.9, 1.2], [0.1, 1.8], [2.1, -0.1]])
# x = f_tH - f_t = [[0.9, 0.2], [0.1, 1.8], [0.1, -0.1]] -> nearest codes 1, 2, 0
z_q, idx, perplexity, vq_loss = {fn}(f_t, f_tH, codebook, mode="ste")
assert idx.dtype == torch.long and idx.tolist() == [1, 2, 0], idx
assert torch.allclose(z_q, codebook[idx], atol=1e-6), z_q
assert abs(perplexity.item() - 3.0) < 1e-5, perplexity
x = f_tH - f_t
sq = ((x - codebook[idx]) ** 2).mean().item()
assert abs(vq_loss.item() - 1.25 * sq) < 1e-6, (vq_loss, 1.25 * sq)
"""},
        {"name": "Noise substitution keeps the quantization error's length", "behavior": "state.invariant", "code": r"""
import torch
codebook = torch.tensor([[0.0, 0.0], [3.0, 0.0]], dtype=torch.float64)
f_t = torch.zeros(1, 2, dtype=torch.float64)
f_tH = torch.tensor([[3.0, 4.0]], dtype=torch.float64)  # x = (3, 4), nearest code 1, ||x - q|| = 4
noise = torch.tensor([[0.0, -2.0]], dtype=torch.float64)  # ||noise|| = 2
z_q, idx, perplexity, vq_loss = {fn}(f_t, f_tH, codebook, mode="nsvq", noise=noise, eps=0.0)
assert idx.tolist() == [1], idx
assert torch.allclose(z_q, torch.tensor([[3.0, 0.0]], dtype=torch.float64)), z_q  # x + (4 / 2) * noise
assert vq_loss.ndim == 0 and vq_loss.item() == 0.0, vq_loss
"""},
        {"name": "Dead codes copy live codes in round-robin order", "behavior": "state.invariant", "code": r"""
import torch
replace_dead_codes = {fn}.__globals__["replace_dead_codes"]
codebook = torch.arange(10, dtype=torch.float32).reshape(5, 2)
usage = torch.tensor([0, 9, 1, 4, 0])
new, dead = replace_dead_codes(codebook, usage, num_batches=10, threshold=0.2)
# rates 0, .9, .1, .4, 0 -> dead [0, 2, 4], live [1, 3]; donors 1, 3, 1
assert dead.tolist() == [0, 2, 4], dead
assert new.tolist() == [[2, 3], [2, 3], [6, 7], [6, 7], [2, 3]], new
assert codebook[0].tolist() == [0, 1], "codebook was modified in place"
"""},
        {"name": "Nearest codes match a brute-force search over leading dims", "visibility": "unshown", "behavior": "routing.selection",
         "failure_message": "Quantize x = f_tH - f_t (later minus earlier) with squared Euclidean distance over the last dimension, keeping every leading dimension in idx.", "code": r"""
import torch
for seed in (0, 1, 2, 3, 4):
    g = torch.Generator().manual_seed(seed)
    B, T, D, K = 3, 4, 5, 7
    codebook = torch.randn(K, D, generator=g, dtype=torch.float64)
    f_t = torch.randn(B, T, D, generator=g, dtype=torch.float64)
    f_tH = f_t + torch.randn(B, T, D, generator=g, dtype=torch.float64) * 1.5 + 0.7
    z_q, idx, _, _ = {fn}(f_t, f_tH, codebook, mode="ste")
    assert idx.shape == (B, T) and idx.dtype == torch.long, (idx.shape, idx.dtype)
    assert z_q.shape == (B, T, D) and z_q.dtype == torch.float64, (z_q.shape, z_q.dtype)
    x = f_tH - f_t
    for b in range(B):
        for t in range(T):
            d = [((x[b, t] - codebook[k]) ** 2).sum().item() for k in range(K)]
            assert idx[b, t].item() == min(range(K), key=lambda k: d[k]), (seed, b, t, idx[b, t], d)
    assert torch.allclose(z_q, codebook[idx]), seed
"""},
        {"name": "Exact ties pick the lowest code index", "visibility": "unshown", "behavior": "metrics.ties",
         "failure_message": "When several codes are equally near, return the lowest index, as torch.argmin does.", "code": r"""
import torch
codebook = torch.tensor([[2.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
f_t = torch.zeros(3, 2)
f_tH = torch.tensor([[0.0, 0.0], [1.5, 0.0], [1.0, 0.0]])
# x=(0,0): codes 1, 2, 3 tie at 1 -> 1;  x=(1.5,0): codes 0, 1, 4 tie at .25 -> 0;  x=(1,0): codes 1 and 4 tie at 0 -> 1
_, idx, _, _ = {fn}(f_t, f_tH, codebook, mode="ste")
assert idx.tolist() == [1, 0, 1], idx
"""},
        {"name": "Straight-through gradient is the identity and skips the codebook", "visibility": "unshown", "behavior": "gradient.flow",
         "failure_message": "In ste mode z_q = x + (q - x).detach(): its gradient reaches f_tH as +g and f_t as -g, and never reaches the codebook.", "code": r"""
import torch
g = torch.Generator().manual_seed(7)
codebook = torch.randn(6, 4, generator=g, dtype=torch.float64, requires_grad=True)
f_t = torch.randn(2, 5, 4, generator=g, dtype=torch.float64, requires_grad=True)
f_tH = torch.randn(2, 5, 4, generator=g, dtype=torch.float64, requires_grad=True)
w = torch.randn(2, 5, 4, generator=g, dtype=torch.float64)
z_q, idx, _, _ = {fn}(f_t, f_tH, codebook, mode="ste")
assert torch.allclose(z_q, codebook.detach()[idx]), "ste forward value must equal the chosen code"
(z_q * w).sum().backward()
assert f_tH.grad is not None and torch.allclose(f_tH.grad, w), "f_tH must receive the upstream gradient unchanged"
assert torch.allclose(f_t.grad, -w), "f_t must receive minus the upstream gradient"
assert codebook.grad is None or torch.count_nonzero(codebook.grad) == 0, "z_q must not send gradient to the codebook"
"""},
        {"name": "Codebook and commitment losses route gradients separately", "visibility": "unshown", "behavior": "gradient.flow",
         "failure_message": "vq_loss = mean((x.detach() - q)^2) + beta * mean((x - q.detach())^2): the codebook term moves codes, the beta term moves the encoder.", "code": r"""
import torch
for seed, beta in ((1, 0.25), (2, 0.4), (3, 1.7)):
    g = torch.Generator().manual_seed(seed)
    codebook = torch.randn(4, 3, generator=g, dtype=torch.float64, requires_grad=True)
    f_t = torch.randn(9, 3, generator=g, dtype=torch.float64, requires_grad=True)
    f_tH = torch.randn(9, 3, generator=g, dtype=torch.float64, requires_grad=True)
    _, idx, _, vq_loss = {fn}(f_t, f_tH, codebook, mode="ste", beta=beta)
    vq_loss.backward()
    x = (f_tH - f_t).detach()
    q = codebook.detach()[idx]
    n = x.numel()
    want_loss = (1 + beta) * ((x - q) ** 2).mean()
    assert abs(vq_loss.item() - want_loss.item()) < 1e-10, (seed, vq_loss, want_loss)
    want_cb = torch.zeros_like(codebook).index_add_(0, idx, 2 * (q - x) / n)
    assert torch.allclose(codebook.grad, want_cb), (seed, "codebook gradient must come from the unweighted codebook term")
    assert torch.allclose(f_tH.grad, beta * 2 * (x - q) / n), (seed, "encoder gradient must come only from beta * commitment")
"""},
        {"name": "NSVQ output matches the formula with eps outside the ratio", "visibility": "unshown", "behavior": "numerics.stability",
         "failure_message": "In nsvq mode use x + (||x - q|| / ||noise|| + eps) * noise with per-vector L2 norms over the last dimension; eps is added to the ratio, not to a norm.", "code": r"""
import torch
for seed, eps in ((0, 1e-12), (1, 0.5), (2, 2.0)):
    g = torch.Generator().manual_seed(seed)
    codebook = torch.randn(5, 4, generator=g, dtype=torch.float64)
    f_t = torch.randn(2, 3, 4, generator=g, dtype=torch.float64)
    f_tH = torch.randn(2, 3, 4, generator=g, dtype=torch.float64)
    noise = torch.randn(2, 3, 4, generator=g, dtype=torch.float64) * 3.0
    z_q, idx, _, _ = {fn}(f_t, f_tH, codebook, mode="nsvq", noise=noise, eps=eps)
    x = f_tH - f_t
    want = torch.empty_like(x)
    for b in range(2):
        for t in range(3):
            r = torch.sqrt(((x[b, t] - codebook[idx[b, t]]) ** 2).sum())
            m = torch.sqrt((noise[b, t] ** 2).sum())
            want[b, t] = x[b, t] + (r / m + eps) * noise[b, t]
    assert torch.allclose(z_q, want, atol=1e-12), (seed, eps, (z_q - want).abs().max())
"""},
        {"name": "NSVQ trains the codebook through the residual norm", "visibility": "unshown", "behavior": "gradient.flow",
         "failure_message": "In nsvq mode keep ||x - q|| in the graph: the codebook learns through it and the encoder gradient includes its derivative.", "code": r"""
import torch
for seed in (3, 4, 5):
    g = torch.Generator().manual_seed(seed)
    cb0 = torch.randn(4, 3, generator=g, dtype=torch.float64)
    a0 = torch.randn(6, 3, generator=g, dtype=torch.float64)
    b0 = torch.randn(6, 3, generator=g, dtype=torch.float64)
    noise = torch.randn(6, 3, generator=g, dtype=torch.float64)
    w = torch.randn(6, 3, generator=g, dtype=torch.float64)
    cb, a, b = (t.clone().requires_grad_(True) for t in (cb0, a0, b0))
    z_q, idx, _, _ = {fn}(a, b, cb, mode="nsvq", noise=noise)
    (z_q * w).sum().backward()
    cb2, a2, b2 = (t.clone().requires_grad_(True) for t in (cb0, a0, b0))
    x = b2 - a2
    q = cb2[idx]
    r = (x - q).pow(2).sum(-1, keepdim=True).sqrt()
    m = noise.pow(2).sum(-1, keepdim=True).sqrt()
    ((x + (r / m + 1e-12) * noise) * w).sum().backward()
    assert cb.grad is not None and torch.count_nonzero(cb.grad) > 0, "the codebook must receive gradient in nsvq mode"
    assert torch.allclose(cb.grad, cb2.grad), seed
    assert torch.allclose(b.grad, b2.grad) and torch.allclose(a.grad, a2.grad), seed
"""},
        {"name": "Perplexity counts usage over every vector and stays finite", "visibility": "unshown", "behavior": "metrics.averaging",
         "failure_message": "perplexity = exp(-sum_k p_k * log(p_k + eps)) with p_k the share of all flattened vectors assigned to code k; unused codes contribute zero.", "code": r"""
import math, torch
codebook = torch.eye(8, dtype=torch.float64)
def run(assign, shape):
    f_tH = codebook[torch.tensor(assign)].reshape(*shape, 8)
    return {fn}(torch.zeros_like(f_tH), f_tH, codebook, mode="ste")[2]
p = run([0, 0, 0, 0], (2, 2))
assert p.ndim == 0 and abs(p.item() - 1.0) < 1e-9, p
p = run([0, 1, 2, 3, 0, 1, 2, 3], (2, 4))
assert abs(p.item() - 4.0) < 1e-9, p
assign = [0, 0, 0, 1, 1, 5]
p = run(assign, (3, 2))
probs = [3 / 6, 2 / 6, 1 / 6]
want = math.exp(-sum(q * math.log(q) for q in probs))
assert math.isfinite(p.item()) and abs(p.item() - want) < 1e-9, (p, want)
"""},
        {"name": "Validates mode, noise and shapes", "visibility": "unshown", "behavior": "contract.signature",
         "failure_message": "Raise ValueError for an unknown mode, for nsvq without noise shaped like f_t, and for mismatched feature shapes.", "code": r"""
import torch
cb = torch.randn(4, 3)
a, b = torch.randn(5, 3), torch.randn(5, 3)
bad = [
    lambda: {fn}(a, b, cb, mode="fsq"),
    lambda: {fn}(a, b, cb, mode="nsvq"),
    lambda: {fn}(a, b, cb, mode="nsvq", noise=torch.randn(5, 2)),
    lambda: {fn}(a, torch.randn(4, 3), cb),
    lambda: {fn}(torch.randn(5, 2), torch.randn(5, 2), cb),
]
for i, call in enumerate(bad):
    try:
        call()
    except ValueError:
        pass
    else:
        raise AssertionError(f"invalid call {i} was accepted")
z_q, idx, p, loss = {fn}(a, b, cb, mode="ste", noise=None)
assert z_q.shape == (5, 3)
"""},
        {"name": "Replacement threshold, jitter and copy semantics", "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "A code is dead only when usage / num_batches < threshold; add jitter only to replaced rows; return a detached new tensor and leave the inputs untouched.", "code": r"""
import torch
replace_dead_codes = {fn}.__globals__["replace_dead_codes"]
g = torch.Generator().manual_seed(11)
codebook = torch.randn(7, 3, generator=g, requires_grad=True)
before = codebook.detach().clone()
usage = torch.tensor([1, 0, 8, 0, 0, 0, 2])
usage_before = usage.clone()
jitter = torch.randn(7, 3, generator=g) * 1e-3
new, dead = replace_dead_codes(codebook, usage, num_batches=4, threshold=0.25, jitter=jitter)
# rates .25 .0 2 .0 .0 .0 .5 -> rate == threshold is live: live [0, 2, 6], dead [1, 3, 4, 5]
assert dead.dtype == torch.long and dead.tolist() == [1, 3, 4, 5], dead
want = before.clone()
for j, k in enumerate([1, 3, 4, 5]):
    want[k] = before[[0, 2, 6][j % 3]] + jitter[k]
assert torch.allclose(new, want), (new - want).abs().max()
assert not new.requires_grad, "the new codebook must not require grad"
assert torch.equal(codebook.detach(), before) and torch.equal(usage, usage_before), "inputs were modified"
"""},
        {"name": "No dead codes and no live codes", "visibility": "unshown", "behavior": "edge.empty_or_boundary",
         "failure_message": "With no dead code return an unchanged copy and an empty index; with no live code copy nothing and add jitter to every row.", "code": r"""
import torch
replace_dead_codes = {fn}.__globals__["replace_dead_codes"]
codebook = torch.arange(8, dtype=torch.float64).reshape(4, 2)
new, dead = replace_dead_codes(codebook, torch.tensor([5, 5, 5, 5]), num_batches=5, threshold=0.5)
assert dead.numel() == 0 and dead.dtype == torch.long, dead
assert torch.equal(new, codebook) and new.data_ptr() != codebook.data_ptr(), "return a copy"
jitter = torch.full((4, 2), 0.125, dtype=torch.float64)
new, dead = replace_dead_codes(codebook, torch.zeros(4, dtype=torch.long), num_batches=5, threshold=0.5, jitter=jitter)
assert dead.tolist() == [0, 1, 2, 3], dead
assert torch.equal(new, codebook + 0.125), new
new, _ = replace_dead_codes(codebook, torch.zeros(4, dtype=torch.long), num_batches=5, threshold=0.5)
assert torch.equal(new, codebook), new
"""},
    ],
    "solution": '''import torch

def latent_action_quantize(f_t, f_tH, codebook, mode="ste", noise=None, beta=0.25, eps=1e-12):
    if mode not in ("ste", "nsvq"):
        raise ValueError(f"unknown mode {mode!r}")
    if f_t.shape != f_tH.shape or codebook.ndim != 2 or f_t.shape[-1] != codebook.shape[-1]:
        raise ValueError("f_t and f_tH must share a shape whose last dim matches the codebook")
    x = f_tH - f_t
    K, D = codebook.shape
    flat = x.reshape(-1, D)
    dist = (flat.pow(2).sum(-1, keepdim=True)
            - 2 * flat @ codebook.t()
            + codebook.pow(2).sum(-1)[None, :])
    idx_flat = torch.argmin(dist, dim=-1)
    idx = idx_flat.reshape(x.shape[:-1])
    q = codebook[idx]

    if mode == "ste":
        z_q = x + (q - x).detach()
        codebook_loss = (x.detach() - q).pow(2).mean()
        commit_loss = (x - q.detach()).pow(2).mean()
        vq_loss = codebook_loss + beta * commit_loss
    else:
        if noise is None or noise.shape != x.shape:
            raise ValueError("nsvq mode needs noise with the same shape as f_t")
        residual_norm = (x - q).pow(2).sum(-1, keepdim=True).sqrt()
        noise_norm = noise.pow(2).sum(-1, keepdim=True).sqrt()
        z_q = x + (residual_norm / noise_norm + eps) * noise
        vq_loss = x.new_zeros(())

    counts = torch.bincount(idx_flat, minlength=K).to(x.dtype)
    probs = counts / idx_flat.numel()
    perplexity = torch.exp(-(probs * torch.log(probs + eps)).sum())
    return z_q, idx, perplexity, vq_loss


def replace_dead_codes(codebook, usage, num_batches, threshold, jitter=None):
    with torch.no_grad():
        source = codebook.detach()
        new = source.clone()
        rate = usage.to(torch.float64) / num_batches
        dead = torch.nonzero(rate < threshold).flatten()
        live = torch.nonzero(rate >= threshold).flatten()
        if live.numel() == 0:
            if jitter is not None:
                new = new + jitter
            return new, dead
        donors = live[torch.arange(dead.numel()) % live.numel()]
        new[dead] = source[donors]
        if jitter is not None:
            new[dead] = new[dead] + jitter[dead]
        return new, dead
''',
    "interview_questions": interview(
        concept=[
            "What is a latent action, and how do LAPA and UniVLA use it to pretrain a VLA on video without action labels?",
            "Why does a VQ-VAE need a special trick to pass gradients through the quantizer, and what does the straight-through estimator do?",
        ],
        deep_dive=[
            "In straight-through VQ, which parameters does each of the codebook loss and the commitment loss update, and what goes wrong if you drop either?",
            "Write the NSVQ substitution and explain why the codebook still learns without a codebook loss.",
            "How do you detect and repair dead codes, and why is perplexity a useful health metric?",
        ],
        tradeoffs=[
            "Quantizing the difference of frame features versus quantizing the pair jointly: what does each make the code capture, and what can leak in?",
            "Discrete latent action codes versus GR00T N1's continuous pre-quantized embedding as a pretraining label: what does each give the policy?",
        ],
    ),
}
