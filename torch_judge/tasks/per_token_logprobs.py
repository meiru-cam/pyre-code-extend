"""Per-token log-probabilities — the gradient carrier of every RL post-training loss."""

TASK = {
    "title": "Per-Token Log-Probabilities",
    "difficulty": "Easy",
    "version": 1,
    "function_name": "per_token_logprobs",
    "description_en": r"""Implement the per-token log-probability lookup that every policy-gradient loss is built on.

**Signature:** `per_token_logprobs(logits, input_ids) -> Tensor`

**Parameters:**
- `logits` — float tensor of shape `(B, T, V)`. Raw scores over the vocabulary at each position.
- `input_ids` — integer tensor of shape `(B, T)`. The token actually realized at each position.

**Returns:** float tensor of shape `(B, T)`. Entry `[b, t]` is the log-probability the model assigned to token `input_ids[b, t]` at position `t`:

    out[b, t] = log_softmax(logits[b, t, :], dim=-1)[input_ids[b, t]]

**Constraints:**
- Normalize over the vocabulary dimension, not over positions and not over the batch.
- Use a fused `log_softmax`. Do not take `softmax` first and then `log`.
- The inputs are already aligned. Do not shift `logits` or `input_ids` by one position.
- Return one value per position. Do not reduce to a scalar.
- Keep the result differentiable with respect to `logits`.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why the fused log-softmax.** For a token whose probability is below about 1e-45 in float32, `log(softmax(x))` returns `-inf` while `log_softmax(x)` returns a large finite number. RL rollouts routinely sample such tokens, and one `-inf` poisons the whole batch loss.

**Why the caller does the shift.** A real training loop applies the causal shift before calling this function, so shifting again here would double-shift the batch.

**Why this is the right primitive.** GRPO, PPO, DPO and the KL penalties all consume exactly this tensor. The policy ratio is a difference of two of these, and the KL estimator is a function of the difference between the policy's and the reference model's values. Getting the gather axis or the stability wrong here corrupts every loss downstream, and the symptom appears as a mysteriously flat or exploding reward curve rather than as a shape error.""",
    "advisory_prerequisites": ["softmax", "cross_entropy"],
    "hints": [
        {
            "level": 1,
            "kind": "questions",
            "content": "Which of the three dimensions of `logits` is the one that has to sum to one in probability space? After you turn logits into log-probabilities, you hold a `(B, T, V)` tensor but you want `(B, T)` — what operation picks exactly one entry from the last axis per position, and what shape does the index tensor need to have? Is there a PyTorch function that computes log and softmax together, and why would that be better than calling the two separately? Finally: the caller passes aligned tensors, so how many positions should the output have compared with the input?",
        },
        {
            "level": 2,
            "kind": "analysis",
            "content": "Two steps. First, `torch.log_softmax(logits, dim=-1)` gives a `(B, T, V)` tensor of log-probabilities normalized over the vocabulary. The fused version subtracts the row max internally, so it stays finite where a separate `log(softmax(x))` underflows to `-inf`. Second, select the realized token at each position. `torch.gather` needs an index tensor of the same rank as the source, so unsqueeze `input_ids` to `(B, T, 1)`, gather along the last dimension, then squeeze that dimension back off to reach `(B, T)`. An equivalent selection is advanced indexing with batch and position index grids, but gather is what the reference implementations use. Do no shifting and no reduction: the tensor you return has one entry per input position, and it must still carry a grad_fn back to `logits`.",
        },
    ],
    "model_connections": [
        "simple_GRPO's get_per_token_logps applies log_softmax over the vocabulary and then torch.gather with the realized ids unsqueezed to a trailing dimension — the same two steps this contract asks for.",
        "nano-aha-moment's compute_pg_loss calls this quantity for both the policy and the frozen reference model; their difference is the only input the k3 KL estimator needs.",
        "OpenRLHF's PolicyLoss takes log_probs and old_log_probs of exactly this shape and forms the importance ratio as exp of their difference, which is why the tensor is never reduced here.",
        "verl and slime both compute this inside their training worker and hand the per-token tensor to the loss function, keeping masking and reduction as a separate concern.",
    ],
    "pro_con_analysis": {
        "pros": [
            "One small contract that every post-training loss reuses, so a correct implementation is leveraged by GRPO, PPO, DPO and the KL terms alike.",
            "The fused log_softmax is both more stable and faster than a separate softmax followed by log.",
            "Returning a per-token tensor keeps masking and reduction under the caller's control, which is what makes response-only training possible.",
        ],
        "cons": [
            "Materializing a (B, T, V) log-probability tensor is memory-heavy for large vocabularies; production stacks fuse the gather into the kernel or chunk over positions.",
            "The contract shifts the alignment burden onto the caller, so a caller that forgets the causal shift gets silently wrong values instead of an error.",
            "Per-token output invites the mistake of averaging over padding, because the function itself cannot know which positions are real.",
        ],
    },
    "sources": [
        {
            "kind": "code",
            "url": "https://github.com/McGill-NLP/nano-aha-moment",
            "commit": "5314e6f8fc60efaa0f4b8fdb62353e9bd451638a",
            "path": "nano_r1_script.py",
            "symbol": "compute_pg_loss",
            "license": "MIT",
            "adapted": "The log-softmax then gather step that produces per-token log-probabilities for the policy and the reference model.",
            "simplifications": "Isolated as a pure tensor function: no model forward pass, no reference model, no temperature division, no masking and no reduction.",
        },
        {
            "kind": "paper",
            "url": "https://arxiv.org/abs/2402.03300",
            "section": "4.1 From PPO to GRPO",
        },
    ],
    "tests": [
        {
            "name": "Hand-calculated uniform case",
            "behavior": "rl.logprob",
            "code": r"""
import torch
import math
# Two positions, two-token vocabulary, all logits equal -> every token has probability 0.5.
logits = torch.zeros(1, 2, 2)
input_ids = torch.tensor([[0, 1]])
out = {fn}(logits, input_ids)
assert out.shape == (1, 2), f'expected (1, 2), got {tuple(out.shape)}'
assert torch.allclose(out, torch.full((1, 2), -math.log(2.0)), atol=1e-6), out
""",
        },
        {
            "name": "Picks the realized token, not a neighbour",
            "behavior": "rl.logprob",
            "code": r"""
import torch
# Distinct logits per position so a wrong pick is visible in the value.
logits = torch.tensor([[[0.0, 10.0, 0.0],
                        [10.0, 0.0, 0.0],
                        [0.0, 0.0, 10.0]]])
input_ids = torch.tensor([[1, 0, 2]])
out = {fn}(logits, input_ids)
# Each selected token is the argmax of its row, so each log-prob is near zero.
assert torch.all(out > -0.01), f'expected near-zero log-probs, got {out}'
input_ids_wrong = torch.tensor([[0, 1, 0]])
out_wrong = {fn}(logits, input_ids_wrong)
assert torch.all(out_wrong < -9.0), f'expected strongly negative log-probs, got {out_wrong}'
""",
        },
        {
            "name": "Matches an independent cross-entropy oracle",
            "visibility": "unshown",
            "behavior": "rl.logprob",
            "failure_message": "Values disagree with the negative per-token cross-entropy oracle. Check that you normalize over the vocabulary dimension and select the realized token id.",
            "code": r"""
import torch
import torch.nn.functional as F
torch.manual_seed(0)
for B, T, V in ((2, 5, 7), (3, 1, 4), (1, 8, 11)):
    logits = torch.randn(B, T, V)
    input_ids = torch.randint(0, V, (B, T))
    out = {fn}(logits, input_ids)
    oracle = -F.cross_entropy(
        logits.reshape(B * T, V), input_ids.reshape(B * T), reduction='none'
    ).reshape(B, T)
    assert out.shape == oracle.shape, f'expected {tuple(oracle.shape)}, got {tuple(out.shape)}'
    assert torch.allclose(out, oracle, atol=1e-5), (out - oracle).abs().max()
""",
        },
        {
            "name": "No causal shift is applied",
            "visibility": "unshown",
            "behavior": "contract.signature",
            "failure_message": "The output length or alignment is wrong. The inputs are already aligned; do not drop a position or shift logits against input_ids.",
            "code": r"""
import torch
# Position t strongly favours token t. A shift of one would select the wrong row.
V = 4
logits = torch.full((1, V, V), -20.0)
for t in range(V):
    logits[0, t, t] = 20.0
input_ids = torch.arange(V).unsqueeze(0)
out = {fn}(logits, input_ids)
assert out.shape == (1, V), f'expected (1, {V}), got {tuple(out.shape)}'
assert torch.all(out > -0.01), f'a shift would give strongly negative values, got {out}'
""",
        },
        {
            "name": "Stable for extreme logits",
            "visibility": "unshown",
            "behavior": "numerics.stability",
            "failure_message": "Produced inf or nan on large-magnitude logits. Use a fused log_softmax instead of taking log after softmax.",
            "code": r"""
import torch
# A token whose probability underflows float32: log(softmax(x)) gives -inf, log_softmax does not.
logits = torch.tensor([[[0.0, 200.0]]])
input_ids = torch.tensor([[0]])
out = {fn}(logits, input_ids)
assert torch.isfinite(out).all(), f'expected a finite value, got {out}'
assert torch.allclose(out, torch.tensor([[-200.0]]), atol=1e-2), out
big = torch.tensor([[[1e4, 1e4 + 1.0, -1e4]]])
assert torch.isfinite({fn}(big, torch.tensor([[2]]))).all(), 'large-magnitude logits overflowed'
""",
        },
        {
            "name": "Invariant to a constant logit shift",
            "visibility": "unshown",
            "behavior": "numerics.stability",
            "failure_message": "Adding a constant to every logit changed the result. Softmax is shift-invariant, so a correct normalization over the vocabulary is too.",
            "code": r"""
import torch
torch.manual_seed(1)
logits = torch.randn(2, 4, 6)
input_ids = torch.randint(0, 6, (2, 4))
base = {fn}(logits, input_ids)
shifted = {fn}(logits + 7.5, input_ids)
assert torch.allclose(base, shifted, atol=1e-5), (base - shifted).abs().max()
""",
        },
        {
            "name": "Gradients reach the logits",
            "visibility": "unshown",
            "behavior": "gradient.flow",
            "failure_message": "No gradient reached logits. Do not detach, and do not rebuild the result from Python floats.",
            "code": r"""
import torch
torch.manual_seed(2)
logits = torch.randn(2, 3, 5, requires_grad=True)
input_ids = torch.randint(0, 5, (2, 3))
out = {fn}(logits, input_ids)
assert out.requires_grad, 'output is detached from the graph'
out.sum().backward()
assert logits.grad is not None, 'logits received no gradient'
assert torch.isfinite(logits.grad).all(), 'gradient contains inf or nan'
# The selected entry gets 1 - p, the others get -p, so each row sums to zero.
assert torch.allclose(logits.grad.sum(dim=-1), torch.zeros(2, 3), atol=1e-5)
""",
        },
        {
            "name": "Preserves dtype and device",
            "visibility": "unshown",
            "behavior": "tensor.dtype_device",
            "failure_message": "The output dtype or device does not follow the input logits. Do not cast to float32 or move the tensor.",
            "code": r"""
import torch
logits = torch.randn(2, 3, 5, dtype=torch.float64)
input_ids = torch.randint(0, 5, (2, 3))
out = {fn}(logits, input_ids)
assert out.dtype == torch.float64, f'expected float64, got {out.dtype}'
assert out.device == logits.device, f'expected {logits.device}, got {out.device}'
""",
        },
        {
            "name": "Single position and single-token vocabulary",
            "visibility": "unshown",
            "behavior": "edge.empty_or_boundary",
            "failure_message": "A boundary shape failed. T=1 and V=1 must still return a (B, T) tensor, and a one-token vocabulary has log-probability zero.",
            "code": r"""
import torch
out = {fn}(torch.zeros(3, 1, 1), torch.zeros(3, 1, dtype=torch.long))
assert out.shape == (3, 1), f'expected (3, 1), got {tuple(out.shape)}'
assert torch.allclose(out, torch.zeros(3, 1), atol=1e-6), out
out = {fn}(torch.randn(1, 1, 9), torch.tensor([[4]]))
assert out.shape == (1, 1), f'expected (1, 1), got {tuple(out.shape)}'
""",
        },
    ],
    "solution": '''import torch


def per_token_logprobs(logits, input_ids):
    log_probs = torch.log_softmax(logits, dim=-1)
    return torch.gather(log_probs, dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)
''',
}
