"""Chunked LM-head cross-entropy that never keeps the full logits tensor alive for backward."""

from ._interview import interview

TASK = {
    "title": "Chunked Cross-Entropy",
    "difficulty": "Hard",
    "version": 1,
    "function_name": "chunked_cross_entropy",
    "description_en": r"""Compute the language-model loss `cross_entropy(hidden @ weight.T, targets)` without keeping the full `(N, V)` logits in memory for the backward pass.

**Signature:** `chunked_cross_entropy(hidden, weight, targets, chunk_size, ignore_index=-100) -> Tensor`

**Parameters:**
- `hidden` — float tensor `(N, D)`, final hidden states for N tokens.
- `weight` — float tensor `(V, D)`, the LM head (unembedding) matrix.
- `targets` — int64 tensor `(N,)`. Entries equal to `ignore_index` are excluded.
- `chunk_size` — positive integer, the number of tokens per chunk.

**Returns:** scalar tensor, the mean cross-entropy over non-ignored tokens. Value and gradients with respect to `hidden` and `weight` must match the unchunked computation.

**Constraints:**
- The tensors that autograd saves for backward must total less than one quarter of the size of the full `(N, V)` logits. Materializing one chunk of logits at a time is not enough on its own, because autograd would keep every chunk alive; recompute each chunk during backward, for example with `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)`.
- Divide by the number of non-ignored tokens across all chunks, not by a per-chunk count.
- Raise `ValueError` for `chunk_size < 1`, mismatched shapes, or no non-ignored token.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why the LM head dominates memory.** With a 128k vocabulary and 16k tokens per micro-batch, fp32 logits take 8 GB, and autograd keeps them plus the softmax for backward. That single tensor can exceed all the activations of the transformer body.

**Two fixes.** Chunking plus recomputation, as here, trades one extra matmul per chunk for memory. Fused kernels such as Liger's fused linear cross-entropy go further and compute the gradient during the forward pass, so the logits never need to be recomputed.""",
    "advisory_prerequisites": ["cross_entropy", "activation_checkpointing"],
    "hints": [
        {"level": 1, "kind": "questions", "content": "If you loop over chunks and sum their losses, which tensors does autograd keep? How do you make it keep only the inputs of each chunk? What should you divide the summed loss by?"},
        {"level": 2, "kind": "analysis", "content": "Count valid = (targets != ignore_index).sum() first. Define a function chunk_loss(h, w, t) that returns F.cross_entropy(h @ w.T, t, ignore_index=ignore_index, reduction='sum'). Loop over chunks, wrap each call in checkpoint(chunk_loss, ..., use_reentrant=False), sum the results and divide by valid."},
    ],
    "model_connections": [
        "Liger Kernel's FusedLinearCrossEntropy, torchtune's chunked output loss and Apple's Cut Cross-Entropy all avoid materializing full LM-head logits.",
    ],
    "pro_con_analysis": {
        "pros": ["Cuts peak memory from the LM head by the number of chunks, which allows longer sequences or larger micro-batches."],
        "cons": ["Recomputation adds one extra LM-head matmul per chunk, and the Python loop launches more kernels than a fused implementation."],
    },
    "tests": [
        {"name": "Matches the full loss on a small batch", "behavior": "numerics.stability", "code": r"""
import torch, torch.nn.functional as F
g = torch.Generator().manual_seed(0)
hidden = torch.randn(10, 4, generator=g); weight = torch.randn(7, 4, generator=g)
targets = torch.tensor([1, 2, -100, 3, 4, 5, 6, -100, 0, 1])
out = {fn}(hidden, weight, targets, chunk_size=3)
want = F.cross_entropy(hidden @ weight.T, targets)
assert out.ndim == 0 and torch.allclose(out, want, atol=1e-6), (out, want)
"""},
        {"name": "Matches values and gradients on seeded inputs", "visibility": "unshown", "behavior": "gradient.flow", "failure_message": "Loss and gradients must equal the unchunked mean cross-entropy over all non-ignored tokens.", "code": r"""
import torch, torch.nn.functional as F
for seed in (1, 12, 47):
    g = torch.Generator().manual_seed(seed)
    N = int(torch.randint(5, 40, (1,), generator=g)); D = 6; V = 11
    chunk = int(torch.randint(1, 9, (1,), generator=g))
    h0 = torch.randn(N, D, generator=g, dtype=torch.float64); w0 = torch.randn(V, D, generator=g, dtype=torch.float64)
    targets = torch.randint(0, V, (N,), generator=g)
    targets[torch.rand(N, generator=g) < 0.3] = -1
    targets[0] = 2
    h1, w1 = h0.clone().requires_grad_(), w0.clone().requires_grad_()
    out = {fn}(h1, w1, targets, chunk_size=chunk, ignore_index=-1)
    out.backward()
    logits = torch.einsum("nd,vd->nv", h0, w0)
    valid = targets != -1
    logp = logits[valid] - torch.logsumexp(logits[valid], dim=-1, keepdim=True)
    want = -logp.gather(1, targets[valid][:, None]).mean()
    assert abs(out.item() - want.item()) < 1e-10, (seed, out, want)
    p = torch.softmax(logits, dim=-1)
    dlogits = p.clone(); dlogits[valid, targets[valid]] -= 1; dlogits[~valid] = 0
    dlogits /= valid.sum()
    assert torch.allclose(h1.grad, dlogits @ w0, atol=1e-10), seed
    assert torch.allclose(w1.grad, dlogits.T @ h0, atol=1e-10), seed
"""},
        {"name": "Saves far less than the full logits for backward", "visibility": "unshown", "behavior": "tensor.shape", "failure_message": "Autograd kept chunk logits alive; recompute each chunk in backward with torch.utils.checkpoint.", "code": r"""
import torch
N, D, V = 512, 32, 4000
g = torch.Generator().manual_seed(3)
hidden = torch.randn(N, D, generator=g, requires_grad=True)
weight = torch.randn(V, D, generator=g, requires_grad=True)
targets = torch.randint(0, V, (N,), generator=g)
storages = {}
def pack(t):
    storages[t.untyped_storage().data_ptr()] = t.untyped_storage().nbytes()
    return t
with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
    out = {fn}(hidden, weight, targets, chunk_size=64)
saved = sum(storages.values())
assert saved < N * V * 4 / 4, (saved, N * V * 4)
out.backward()
assert hidden.grad is not None and weight.grad is not None
"""},
        {"name": "Rejects invalid arguments", "visibility": "unshown", "behavior": "contract.signature", "failure_message": "Bad chunk sizes, mismatched shapes and batches with no valid token must raise ValueError.", "code": r"""
import torch
h = torch.randn(4, 3); w = torch.randn(5, 3)
cases = [
    (h, w, torch.tensor([0, 1, 2, 3]), 0),
    (h, torch.randn(5, 2), torch.tensor([0, 1, 2, 3]), 2),
    (h, w, torch.tensor([0, 1, 2]), 2),
    (h, w, torch.full((4,), -100), 2),
]
for hidden, weight, targets, chunk in cases:
    try:
        {fn}(hidden, weight, targets, chunk)
    except ValueError:
        pass
    else:
        raise AssertionError(f"accepted chunk={chunk} {tuple(weight.shape)} {targets.tolist()}")
"""},
    ],
    "solution": '''import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

def chunked_cross_entropy(hidden, weight, targets, chunk_size, ignore_index=-100):
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if hidden.ndim != 2 or weight.ndim != 2 or hidden.shape[1] != weight.shape[1] or targets.shape != hidden.shape[:1]:
        raise ValueError("expected hidden (N, D), weight (V, D) and targets (N,)")
    valid = int((targets != ignore_index).sum())
    if valid == 0:
        raise ValueError("no target to score")

    def chunk_loss(h, w, t):
        return F.cross_entropy(h @ w.T, t, ignore_index=ignore_index, reduction="sum")

    total = hidden.new_zeros(())
    for start in range(0, hidden.shape[0], chunk_size):
        end = start + chunk_size
        total = total + checkpoint(chunk_loss, hidden[start:end], weight, targets[start:end], use_reentrant=False)
    return total / valid
''',
    "interview_questions": interview(
        concept=[
            "Why can the LM-head logits be the largest single tensor in LLM training? Estimate their size for a 128k vocabulary and 16k tokens.",
            "What does chunking the cross-entropy save, and why does chunking alone not save memory under autograd?",
        ],
        deep_dive=[
            "Walk through which tensors autograd saves with and without checkpointing each chunk.",
            "How do you keep the mean correct when chunks contain different numbers of ignored tokens?",
            "Write the gradient of cross-entropy with respect to the logits. How does a fused kernel use it to avoid recomputing the logits?",
        ],
        tradeoffs=[
            "Chunk-and-recompute versus a fused linear cross-entropy kernel versus vocabulary parallelism: memory, speed and complexity?",
            "How do you choose the chunk size, and what happens at the extremes?",
        ],
    ),
}
