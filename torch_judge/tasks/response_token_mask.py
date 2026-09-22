"""Response token mask — the mask that keeps prompt and padding out of the policy gradient."""

TASK = {
    "title": "Response Token Mask",
    "difficulty": "Easy",
    "version": 1,
    "function_name": "response_token_mask",
    "description_en": r"""Build the boolean mask that selects exactly the tokens a policy generated, excluding the prompt it was given and the padding added to square off the batch.

**Signature:** `response_token_mask(prompt_lens, total_lens, max_len) -> Tensor`

**Parameters:**
- `prompt_lens` — integer tensor of shape `(B,)`. The number of prompt tokens in each row.
- `total_lens` — integer tensor of shape `(B,)`. Prompt tokens plus generated tokens in each row, before padding.
- `max_len` — positive integer. The padded sequence length of the batch.

**Returns:** boolean tensor of shape `(B, max_len)`. Entry `[b, t]` is True when position `t` holds a token that row `b` generated:

    mask[b, t] = (prompt_lens[b] <= t) and (t < total_lens[b])

The layout is right-padded: prompt first, generated tokens next, padding last.

**Constraints:**
- Exclude every prompt position. The mask starts at index `prompt_lens`, not at 0.
- Exclude every padding position. The mask stops at index `total_lens`.
- Include the final generated token. The comparison against `total_lens` is exclusive, not inclusive.
- Return a boolean tensor, not a float one.
- Raise a `ValueError` when any row violates `0 <= prompt_lens <= total_lens <= max_len`, or when the two length tensors have different shapes.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why the prompt is excluded.** The policy did not produce those tokens, so a gradient through them trains the model to predict its own input.

**Why padding is excluded.** Padding tokens carry arbitrary ids; including them adds noise that scales with how ragged the batch is.

**Why the last token matters.** The end-of-sequence token is a real decision the policy made and a real place to assign credit. An off-by-one that drops it silently removes the strongest stopping signal.

**Why the caller cannot skip this.** Every loss in this path is computed per token and then reduced, and the reduction is where masking bites: dividing a masked sum by the total number of positions instead of by the number of unmasked positions makes the loss depend on how much padding a batch happens to contain, so the same data produces a different gradient depending on batching. Building the mask is the cheap half; using its count as the denominator is the half that gets forgotten.""",
    "advisory_prerequisites": ["causal_attention", "per_token_logprobs"],
    "hints": [
        {
            "level": 1,
            "kind": "questions",
            "content": "You need a tensor where every row asks a different question about the same column indices. What single tensor holds the positions 0 to max_len minus 1, and what happens when you compare it against a column vector of per-row lengths? Which of the two comparisons needs to be inclusive and which exclusive — write out a tiny example with prompt_lens=2 and total_lens=5 and list the positions that should be True. How do the two conditions combine? And before any of that: what should happen if a row claims more prompt tokens than total tokens?",
        },
        {
            "level": 2,
            "kind": "analysis",
            "content": "Create the position row with `torch.arange(max_len)`, shaped `(1, max_len)`. Broadcast it against `prompt_lens.unsqueeze(-1)` and `total_lens.unsqueeze(-1)`, both `(B, 1)`, and the result is `(B, max_len)` with no loop. The two comparisons are deliberately asymmetric: `positions >= prompt_lens` is inclusive because position `prompt_lens[b]` is the first generated token, while `positions < total_lens` is exclusive because `total_lens[b]` is one past the last one. Combine them with a logical and. Put the arange on the same device as the inputs so the mask does not land on the wrong device in a multi-device setting. Validate first: check the two length tensors have the same shape, then that the whole chain 0, prompt, total, max_len is non-decreasing, and raise ValueError naming which row failed.",
        },
    ],
    "model_connections": [
        "nano-aha-moment's compute_pg_loss carries a labels_mask exactly of this shape and applies it before summing the per-token loss, then divides by the total response length rather than by the padded size.",
        "simple_GRPO builds the same mask from per-row completion lengths and uses its sum as the loss denominator.",
        "OpenRLHF passes an action_mask into PolicyLoss and ValueLoss for the same purpose; every loss in that file takes the mask as an explicit argument rather than inferring it.",
        "The distinction also appears in supervised fine-tuning, where the prompt is masked out of the cross-entropy so only completion tokens are trained.",
    ],
    "pro_con_analysis": {
        "pros": [
            "One boolean tensor serves every per-token loss, so prompt and padding exclusion is defined once instead of re-derived in each loss.",
            "Broadcasting makes it allocation-light and free of Python loops over the batch.",
            "Keeping it boolean lets the caller choose between multiplying and indexing, and makes the unmasked count a plain sum.",
        ],
        "cons": [
            "It hard-codes the right-padded layout; a left-padded batch needs a different construction and will silently produce a wrong mask here.",
            "The mask alone does not enforce the matching denominator, which is the more common source of a wrong loss.",
            "For very ragged batches the mask is mostly False, which is a signal to sort by length rather than something the mask can fix.",
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
            "adapted": "The labels_mask that restricts the per-token policy-gradient loss to generated response tokens.",
            "simplifications": "Isolated as a pure function over per-row lengths: no tokenizer, no episode dictionary, no loss and no denominator.",
        },
        {
            "kind": "paper",
            "url": "https://arxiv.org/abs/2402.03300",
            "section": "4.1 From PPO to GRPO, objective over output tokens",
        },
    ],
    "tests": [
        {
            "name": "Hand-calculated single row",
            "behavior": "rl.masking",
            "code": r"""
import torch
# Prompt occupies positions 0 and 1; the response occupies 2, 3, 4; 5 and 6 are padding.
mask = {fn}(torch.tensor([2]), torch.tensor([5]), 7)
assert mask.shape == (1, 7), f'expected (1, 7), got {tuple(mask.shape)}'
assert mask.dtype == torch.bool, f'expected bool, got {mask.dtype}'
expected = torch.tensor([[False, False, True, True, True, False, False]])
assert torch.equal(mask, expected), mask
""",
        },
        {
            "name": "Rows are masked independently",
            "behavior": "rl.masking",
            "code": r"""
import torch
mask = {fn}(torch.tensor([0, 3]), torch.tensor([2, 4]), 5)
expected = torch.tensor([[True, True, False, False, False],
                         [False, False, False, True, False]])
assert torch.equal(mask, expected), mask
assert mask.sum(dim=-1).tolist() == [2, 1], mask.sum(dim=-1)
""",
        },
        {
            "name": "Excludes every prompt position",
            "visibility": "unshown",
            "behavior": "rl.masking",
            "failure_message": "Prompt positions were left unmasked. The mask must start at index prompt_lens, not at index 0.",
            "code": r"""
import torch
torch.manual_seed(0)
prompt = torch.tensor([4, 1, 7])
total = torch.tensor([9, 5, 8])
mask = {fn}(prompt, total, 12)
for b in range(3):
    assert not mask[b, :prompt[b]].any(), f'row {b} left a prompt position unmasked'
""",
        },
        {
            "name": "Excludes every padding position",
            "visibility": "unshown",
            "behavior": "rl.masking",
            "failure_message": "Padding positions were left unmasked. The mask must stop at index total_lens.",
            "code": r"""
import torch
prompt = torch.tensor([2, 0, 5])
total = torch.tensor([6, 3, 9])
max_len = 11
mask = {fn}(prompt, total, max_len)
for b in range(3):
    assert not mask[b, total[b]:].any(), f'row {b} left a padding position unmasked'
""",
        },
        {
            "name": "Includes the final generated token",
            "visibility": "unshown",
            "behavior": "edge.empty_or_boundary",
            "failure_message": "The last generated token is masked out. An off-by-one here removes the end-of-sequence decision from the gradient.",
            "code": r"""
import torch
prompt = torch.tensor([3, 0])
total = torch.tensor([7, 2])
mask = {fn}(prompt, total, 9)
for b in range(2):
    last = int(total[b]) - 1
    assert bool(mask[b, last]), f'row {b} masked out its final generated token at index {last}'
""",
        },
        {
            "name": "Unmasked count equals the generated length",
            "visibility": "unshown",
            "behavior": "state.invariant",
            "failure_message": "The number of True entries does not equal total_lens minus prompt_lens. That count is the loss denominator, so an error here rescales every loss.",
            "code": r"""
import torch
torch.manual_seed(1)
for _ in range(4):
    B = 6
    prompt = torch.randint(0, 5, (B,))
    gen = torch.randint(1, 6, (B,))
    total = prompt + gen
    max_len = int(total.max()) + 3
    mask = {fn}(prompt, total, max_len)
    assert torch.equal(mask.sum(dim=-1), gen), f'expected {gen.tolist()}, got {mask.sum(dim=-1).tolist()}'
""",
        },
        {
            "name": "Empty response and full-length response",
            "visibility": "unshown",
            "behavior": "edge.empty_or_boundary",
            "failure_message": "A boundary case failed. An empty response must produce an all-False row, and a row with no prompt and no padding must produce an all-True row.",
            "code": r"""
import torch
# Row 0 generated nothing; row 1 is prompt-free and fills the whole window.
mask = {fn}(torch.tensor([3, 0]), torch.tensor([3, 4]), 4)
assert not mask[0].any(), f'empty response should be all False, got {mask[0]}'
assert mask[1].all(), f'full row should be all True, got {mask[1]}'
""",
        },
        {
            "name": "Rejects inconsistent lengths",
            "visibility": "unshown",
            "behavior": "contract.signature",
            "failure_message": "An inconsistent length was accepted. Raise ValueError unless every row satisfies 0 <= prompt_lens <= total_lens <= max_len.",
            "code": r"""
import torch
cases = ((torch.tensor([5]), torch.tensor([3]), 8),
         (torch.tensor([1]), torch.tensor([9]), 8),
         (torch.tensor([-1]), torch.tensor([3]), 8),
         (torch.tensor([1, 2]), torch.tensor([3]), 8))
for prompt, total, max_len in cases:
    try:
        {fn}(prompt, total, max_len)
    except ValueError:
        continue
    raise AssertionError(f'prompt={prompt.tolist()} total={total.tolist()} max_len={max_len} should raise ValueError')
""",
        },
        {
            "name": "Returns bool on the input device",
            "visibility": "unshown",
            "behavior": "tensor.dtype_device",
            "failure_message": "The mask is not a boolean tensor on the same device as the inputs. Do not return floats or integers.",
            "code": r"""
import torch
prompt = torch.tensor([1, 2])
total = torch.tensor([4, 5])
mask = {fn}(prompt, total, 6)
assert mask.dtype == torch.bool, f'expected bool, got {mask.dtype}'
assert mask.device == prompt.device, f'expected {prompt.device}, got {mask.device}'
assert mask.shape == (2, 6), f'expected (2, 6), got {tuple(mask.shape)}'
""",
        },
    ],
    "solution": '''import torch


def response_token_mask(prompt_lens, total_lens, max_len):
    if prompt_lens.shape != total_lens.shape:
        raise ValueError(
            f"shape mismatch: {tuple(prompt_lens.shape)} vs {tuple(total_lens.shape)}"
        )
    if not isinstance(max_len, int) or max_len <= 0:
        raise ValueError(f"max_len must be a positive integer, got {max_len!r}")
    if (prompt_lens < 0).any() or (total_lens < prompt_lens).any() or (total_lens > max_len).any():
        raise ValueError("every row must satisfy 0 <= prompt_lens <= total_lens <= max_len")
    positions = torch.arange(max_len, device=prompt_lens.device).unsqueeze(0)
    return (positions >= prompt_lens.unsqueeze(-1)) & (positions < total_lens.unsqueeze(-1))
''',
}
