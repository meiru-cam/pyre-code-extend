"""Rollout batch assembly — turning ragged generations into the padded batch a loss can consume."""

TASK = {
    "title": "Rollout Batch Assembly",
    "difficulty": "Hard",
    "version": 1,
    "function_name": "rollout_batch_assembly",
    "description_en": r"""Assemble raw generations into the padded, masked, advantage-broadcast batch that every loss in this path expects.

**Signature:** `rollout_batch_assembly(prompt_ids, generation_ids, advantages, pad_id=0) -> dict`

**Parameters:**
- `prompt_ids` — a list of `B` lists of integers. The prompt tokens of each response.
- `generation_ids` — a list of `B` lists of integers. The tokens the policy generated. May be empty for a response that produced nothing.
- `advantages` — float tensor of shape `(B,)`. One scalar advantage per response.
- `pad_id` — the integer token id used to fill each row out to the batch width.

**Returns:** a dictionary with exactly four keys, where `T` is the longest prompt-plus-generation length in the batch:

- `input_ids` — integer tensor `(B, T)`. Each row holds its prompt, then its generation, then `pad_id` repeated to width `T`.
- `mask` — boolean tensor `(B, T)`. True exactly on generated tokens; False on prompt positions and False on padding.
- `token_advantages` — float tensor `(B, T)`. The response's advantage at every unmasked position, and exactly 0.0 everywhere else.
- `lengths` — integer tensor `(B,)`. The number of generated tokens in each row.

The layout is right-padded: prompt first, generation next, padding last.

**Constraints:**
- `T` is the maximum of prompt length plus generation length, not the longest prompt or the longest generation alone.
- Conserve tokens. Each row's leading slice must equal its prompt followed by its generation, in order, with nothing dropped or duplicated.
- `mask` is True precisely on the generation segment.
- `token_advantages` is exactly 0.0 wherever `mask` is False.
- `lengths` equals the per-row count of True entries in `mask`.
- Do not mutate the input lists.
- A response with an empty generation is valid: no unmasked tokens, length 0.
- Raise a `ValueError` when the three inputs do not have the same length `B`, when `B` is zero, or when `advantages` is not one-dimensional.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why these three invariants are checked separately.** Each is a real failure that shapes alone will not reveal.

**Token conservation.** Everything upstream produces ragged Python lists; everything downstream needs rectangular tensors and cannot recover the raggedness once it is lost. A dropped or reordered token is invisible in every shape assertion.

**Mask alignment.** A mask that also covers the prompt trains the model on its own input; a mask that covers padding trains it on `pad_id`.

**Advantage containment.** A broadcast that reaches prompt or padding positions leaks a non-zero gradient into tokens the policy never chose. Because the loss then multiplies by the mask anyway, the error is invisible in the final scalar while remaining a real bug in any code path that forgets the multiplication.

This is the function that makes the difference between a loss that is correct on paper and a training step that is correct in practice.""",
    "advisory_prerequisites": [
        "response_token_mask",
        "group_relative_advantage",
        "grpo_token_loss",
    ],
    "hints": [
        {
            "level": 1,
            "kind": "questions",
            "content": "First settle the width: what determines `T`, and is it the longest prompt, the longest generation, or something else? Now think about one row at a time — you know where its generation starts and where it ends, so which existing exercise already turns a pair of per-row boundaries into a boolean mask? For the advantages, you have a vector of length B and need a matrix: what shape must the advantage take before it can multiply a mask, and what does multiplying by a boolean-turned-float give you at the masked-out positions for free? Last, check the empty-generation row against every one of your four outputs.",
        },
        {
            "level": 2,
            "kind": "analysis",
            "content": "Compute the per-row boundaries first: `prompt_lens`, `gen_lens` and `total_lens` as integer tensors, then `T = int(total_lens.max())` — using only the generation lengths gives a width too small to hold the prompts. Allocate `input_ids` full of `pad_id` and write each row's concatenated tokens into its leading slice with a loop; the loop is over B, not over tokens, so it is cheap. Build the mask with the same arange-broadcast as `response_token_mask`: `positions >= prompt_lens.unsqueeze(-1)` and `positions < total_lens.unsqueeze(-1)`. For the advantages, `advantages.unsqueeze(-1) * mask.to(dtype)` does the broadcast and the zeroing in one step — the multiplication by a zero is what guarantees invariant 3, so prefer it to `torch.where` or to assigning into a zeros tensor, both of which leave room for an off-by-one. Guard `T` when every row is empty, since `max()` over an all-zero length tensor is 0 and the resulting width-zero tensors are still valid. Validate the three lengths agree before any of this, so the error names the real mismatch.",
        },
    ],
    "model_connections": [
        "nano-aha-moment's create_training_episodes performs exactly this job: it groups generations, computes the group advantages, and emits padded token, mask and advantage tensors for the training step.",
        "simple_GRPO's gen_samples returns prompt ids, output ids and stacked rewards, leaving this padding and masking step to its training loop.",
        "OpenRLHF keeps the assembled rollout in an experience buffer with the action mask stored alongside the token ids, because every loss in that file takes the mask as an argument.",
        "verl and slime both treat rollout assembly as a distinct stage between the generation engine and the training worker, which is why the two can run on different hardware.",
    ],
    "pro_con_analysis": {
        "pros": [
            "One function establishes the layout contract, so every loss downstream can assume right-padding and a correct mask.",
            "Zeroing the advantages outside the mask makes the tensors safe to use even in code paths that forget to multiply by the mask.",
            "Returning the lengths alongside the mask gives the loss its denominator without recomputing it.",
        ],
        "cons": [
            "Padding to the longest row wastes memory proportional to how ragged the batch is; production stacks sort by length or pack sequences instead.",
            "The right-padded convention is implicit in the tensors, so a caller that assumes left-padding gets silently wrong masks.",
            "Materializing a full token-advantage matrix is redundant when every row carries one repeated scalar, but it keeps the losses uniform.",
        ],
    },
    "sources": [
        {
            "kind": "code",
            "url": "https://github.com/McGill-NLP/nano-aha-moment",
            "commit": "5314e6f8fc60efaa0f4b8fdb62353e9bd451638a",
            "path": "nano_r1_script.py",
            "symbol": "create_training_episodes",
            "license": "MIT",
            "adapted": "The assembly of ragged prompt and generation token lists into padded input ids, a response-only mask, and per-token advantages broadcast from one scalar per response.",
            "simplifications": "Takes advantages as a given argument rather than computing them from grouped rewards, and omits the tokenizer, the generation engine and the finish-reason bookkeeping.",
        },
        {
            "kind": "paper",
            "url": "https://arxiv.org/abs/2402.03300",
            "section": "4.1 From PPO to GRPO, objective over output tokens",
        },
    ],
    "tests": [
        {
            "name": "Hand-calculated two-row batch",
            "behavior": "rl.rollout_assembly",
            "code": r"""
import torch
batch = {fn}([[1, 2], [3]], [[4, 5, 6], [7]], torch.tensor([0.5, -0.5]), pad_id=0)
assert set(batch) == {'input_ids', 'mask', 'token_advantages', 'lengths'}, sorted(batch)
assert torch.equal(batch['input_ids'], torch.tensor([[1, 2, 4, 5, 6],
                                                     [3, 7, 0, 0, 0]])), batch['input_ids']
assert torch.equal(batch['mask'], torch.tensor([[False, False, True, True, True],
                                                [False, True, False, False, False]])), batch['mask']
assert torch.equal(batch['lengths'], torch.tensor([3, 1])), batch['lengths']
""",
        },
        {
            "name": "Advantages reach generated tokens only",
            "behavior": "rl.rollout_assembly",
            "code": r"""
import torch
batch = {fn}([[1, 2], [3]], [[4, 5, 6], [7]], torch.tensor([0.5, -0.5]), pad_id=0)
expected = torch.tensor([[0.0, 0.0, 0.5, 0.5, 0.5],
                         [0.0, -0.5, 0.0, 0.0, 0.0]])
assert torch.allclose(batch['token_advantages'], expected, atol=1e-6), batch['token_advantages']
""",
        },
        {
            "name": "Tokens are conserved in order",
            "visibility": "unshown",
            "behavior": "dispatch.token_conservation",
            "failure_message": "A token was dropped, duplicated or reordered. Each row's leading slice must equal its prompt followed by its generation.",
            "code": r"""
import torch
import random
random.seed(0)
prompts = [[random.randrange(1, 99) for _ in range(random.randrange(1, 5))] for _ in range(4)]
gens = [[random.randrange(1, 99) for _ in range(random.randrange(1, 6))] for _ in range(4)]
batch = {fn}(prompts, gens, torch.randn(4), pad_id=0)
for b in range(4):
    total = len(prompts[b]) + len(gens[b])
    got = batch['input_ids'][b, :total].tolist()
    assert got == prompts[b] + gens[b], f'row {b}: {got} vs {prompts[b] + gens[b]}'
""",
        },
        {
            "name": "Width covers the longest prompt plus generation",
            "visibility": "unshown",
            "behavior": "tensor.shape",
            "failure_message": "The batch width is wrong. It is the maximum of prompt length plus generation length, not the longest prompt or the longest generation alone.",
            "code": r"""
import torch
# Row 0 has the longest prompt, row 1 the longest generation, row 2 the longest total.
prompts = [[1, 2, 3, 4], [5], [6, 6, 6]]
gens = [[7], [8, 9, 10, 11, 12], [13, 14, 15, 16]]
batch = {fn}(prompts, gens, torch.zeros(3), pad_id=0)
assert batch['input_ids'].shape == (3, 7), f"expected (3, 7), got {tuple(batch['input_ids'].shape)}"
assert batch['mask'].shape == (3, 7), tuple(batch['mask'].shape)
assert batch['token_advantages'].shape == (3, 7), tuple(batch['token_advantages'].shape)
""",
        },
        {
            "name": "Padding is filled with pad_id and left unmasked",
            "visibility": "unshown",
            "behavior": "rl.masking",
            "failure_message": "Padding positions carry the wrong id or are marked True in the mask.",
            "code": r"""
import torch
batch = {fn}([[1], [2]], [[3], [4, 5, 6]], torch.tensor([1.0, 1.0]), pad_id=99)
# Row 0 is prompt + one token, so positions 2 and 3 are padding.
assert torch.equal(batch['input_ids'][0], torch.tensor([1, 3, 99, 99])), batch['input_ids'][0]
assert not batch['mask'][0, 2:].any(), batch['mask'][0]
# Row 1 fills the width exactly, so it carries no padding at all.
assert torch.equal(batch['input_ids'][1], torch.tensor([2, 4, 5, 6])), batch['input_ids'][1]
assert torch.equal(batch['mask'][1], torch.tensor([False, True, True, True])), batch['mask'][1]
# pad_id must never be marked as a generated token, whatever its value.
assert not batch['mask'][batch['input_ids'] == 99].any(), 'a pad position was left unmasked'
""",
        },
        {
            "name": "The mask excludes every prompt position",
            "visibility": "unshown",
            "behavior": "rl.masking",
            "failure_message": "Prompt positions were marked True. The policy did not generate its own input.",
            "code": r"""
import torch
prompts = [[1, 2, 3], [4], [5, 6]]
gens = [[7, 8], [9, 10, 11], [12]]
batch = {fn}(prompts, gens, torch.randn(3), pad_id=0)
for b in range(3):
    assert not batch['mask'][b, :len(prompts[b])].any(), f'row {b} unmasked a prompt position'
    assert bool(batch['mask'][b, len(prompts[b]):len(prompts[b]) + len(gens[b])].all()), f'row {b} masked a generated token'
""",
        },
        {
            "name": "Advantages are exactly zero outside the mask",
            "visibility": "unshown",
            "behavior": "rl.rollout_assembly",
            "failure_message": "A non-zero advantage appeared at a prompt or padding position. Multiply the broadcast advantage by the mask.",
            "code": r"""
import torch
torch.manual_seed(0)
prompts = [[1, 2], [3, 4, 5], [6]]
gens = [[7, 8, 9], [10], [11, 12]]
advantages = torch.tensor([2.5, -3.5, 7.0])
batch = {fn}(prompts, gens, advantages, pad_id=0)
outside = batch['token_advantages'][~batch['mask']]
assert torch.equal(outside, torch.zeros_like(outside)), f'non-zero outside the mask: {outside}'
for b in range(3):
    inside = batch['token_advantages'][b][batch['mask'][b]]
    assert torch.allclose(inside, torch.full_like(inside, float(advantages[b])), atol=1e-6), inside
""",
        },
        {
            "name": "Lengths agree with the mask",
            "visibility": "unshown",
            "behavior": "state.invariant",
            "failure_message": "lengths does not equal the per-row count of True entries in mask. The loss uses one or the other as its denominator, so they must agree.",
            "code": r"""
import torch
import random
random.seed(1)
for _ in range(3):
    B = 5
    prompts = [[1] * random.randrange(1, 4) for _ in range(B)]
    gens = [[2] * random.randrange(0, 5) for _ in range(B)]
    batch = {fn}(prompts, gens, torch.randn(B), pad_id=0)
    assert torch.equal(batch['lengths'], batch['mask'].sum(dim=-1)), (batch['lengths'], batch['mask'].sum(dim=-1))
    assert batch['lengths'].tolist() == [len(g) for g in gens], batch['lengths']
""",
        },
        {
            "name": "An empty generation yields an empty row",
            "visibility": "unshown",
            "behavior": "edge.empty_or_boundary",
            "failure_message": "A response that generated nothing was not handled. Its row has no unmasked tokens and a length of 0.",
            "code": r"""
import torch
batch = {fn}([[1, 2], [3]], [[], [4, 5]], torch.tensor([9.0, 1.0]), pad_id=0)
assert not batch['mask'][0].any(), batch['mask'][0]
assert int(batch['lengths'][0]) == 0, batch['lengths']
assert torch.equal(batch['token_advantages'][0], torch.zeros(3)), batch['token_advantages'][0]
assert torch.equal(batch['input_ids'][0], torch.tensor([1, 2, 0])), batch['input_ids'][0]
# Every row empty is still a valid batch.
allempty = {fn}([[], []], [[], []], torch.tensor([1.0, 2.0]), pad_id=0)
assert int(allempty['lengths'].sum()) == 0, allempty['lengths']
assert not allempty['mask'].any(), allempty['mask']
""",
        },
        {
            "name": "Rejects mismatched batch lengths",
            "visibility": "unshown",
            "behavior": "contract.signature",
            "failure_message": "An inconsistent batch was accepted. prompt_ids, generation_ids and advantages must all describe the same B responses.",
            "code": r"""
import torch
bad = (([[1]], [[2], [3]], torch.randn(1)),
       ([[1], [2]], [[3]], torch.randn(2)),
       ([[1], [2]], [[3], [4]], torch.randn(3)),
       ([], [], torch.randn(0)),
       ([[1]], [[2]], torch.randn(1, 1)))
for prompts, gens, advantages in bad:
    try:
        {fn}(prompts, gens, advantages, pad_id=0)
    except ValueError:
        continue
    raise AssertionError(f'{len(prompts)}/{len(gens)}/{tuple(advantages.shape)} should raise ValueError')
""",
        },
    ],
    "solution": '''import torch


def rollout_batch_assembly(prompt_ids, generation_ids, advantages, pad_id=0):
    batch_size = len(prompt_ids)
    if batch_size == 0:
        raise ValueError("the batch must hold at least one response")
    if len(generation_ids) != batch_size:
        raise ValueError(
            f"prompt_ids has {batch_size} rows but generation_ids has {len(generation_ids)}"
        )
    if advantages.ndim != 1 or advantages.shape[0] != batch_size:
        raise ValueError(
            f"advantages must have shape ({batch_size},), got {tuple(advantages.shape)}"
        )

    device = advantages.device
    prompt_lens = torch.tensor([len(p) for p in prompt_ids], device=device)
    gen_lens = torch.tensor([len(g) for g in generation_ids], device=device)
    total_lens = prompt_lens + gen_lens
    width = int(total_lens.max())

    input_ids = torch.full((batch_size, width), pad_id, dtype=torch.long, device=device)
    for row, (prompt, generation) in enumerate(zip(prompt_ids, generation_ids)):
        tokens = list(prompt) + list(generation)
        if tokens:
            input_ids[row, : len(tokens)] = torch.tensor(
                tokens, dtype=torch.long, device=device
            )

    positions = torch.arange(width, device=device).unsqueeze(0)
    mask = (positions >= prompt_lens.unsqueeze(-1)) & (positions < total_lens.unsqueeze(-1))
    token_advantages = advantages.unsqueeze(-1) * mask.to(advantages.dtype)

    return {
        "input_ids": input_ids,
        "mask": mask,
        "token_advantages": token_advantages,
        "lengths": gen_lens,
    }
''',
}
