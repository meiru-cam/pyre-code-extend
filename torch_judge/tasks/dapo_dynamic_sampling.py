"""DAPO dynamic sampling — dropping the groups that carry no gradient signal."""

TASK = {
    "title": "DAPO Dynamic Sampling",
    "difficulty": "Medium",
    "version": 1,
    "function_name": "dapo_dynamic_sampling",
    "description_en": r"""Implement the filter that removes degenerate groups from a GRPO batch before the advantage is computed.

**Signature:** `dapo_dynamic_sampling(rewards, group_size, tol=0.0) -> Tensor`

**Parameters:**
- `rewards` — float tensor of shape `(N,)`. One reward per sampled response, laid out group-major: the first `group_size` entries belong to prompt 0, and so on.
- `group_size` — positive integer that divides `N` exactly.
- `tol` — non-negative float. A group counts as degenerate when the spread of its rewards is at most this value.

**Returns:** boolean tensor of shape `(N,)`. True for every response whose group should be kept.

    spread[g]     = max(group g) - min(group g)
    degenerate[g] = spread[g] <= tol
    keep[i]       = not degenerate[group of i]

**Constraints:**
- Keep or drop whole groups. Every member of a group shares one decision.
- Repeat each group's decision `group_size` times in place. Do not tile the decision vector.
- Both an all-correct group and an all-wrong group are degenerate. The criterion is spread, not magnitude.
- A group of size one is always degenerate.
- Return a boolean mask over responses, not over groups, and not a filtered tensor.
- Raise a `ValueError` when `group_size` is not positive, does not divide `N`, or when `tol` is negative.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why these groups are worthless.** GRPO's advantage standardizes each reward against its own group. When every member received the same reward, the numerator is zero for every member, so every advantage in the group is zero and the group contributes exactly nothing to the gradient. It still cost a full set of generations to produce, and it still dilutes the gradient from the groups that do carry signal.

**Why both ends count.** A prompt the policy always solves and a prompt it never solves both yield a constant group. An implementation that drops only the all-correct groups keeps every hopeless prompt in the batch and slowly biases training toward easy prompts.

**Why whole groups.** Dropping individual responses destroys the group structure the advantage depends on; the surviving members would be standardized against a group that no longer exists.

**Note the ordering.** This filter runs before `group_relative_advantage`, not after. Running it afterwards would mean standardizing the degenerate groups first, which is where the division by a near-zero standard deviation lives — the epsilon guard in that function exists precisely because this filter is optional.""",
    "advisory_prerequisites": ["group_relative_advantage", "grpo_token_loss"],
    "hints": [
        {
            "level": 1,
            "kind": "questions",
            "content": "The rewards arrive flat but the decision is per group — what reshape puts each group on its own row, and which axis do max and min then run along? You end up with one boolean per group but the contract wants one per response: what operation turns a length-G vector into a length-N one with each entry repeated group_size times, and does the result line up with the original group-major layout? Think about why the criterion is a spread rather than a check against a specific reward value. And what does your expression say about a group of size one?",
        },
        {
            "level": 2,
            "kind": "analysis",
            "content": "Reshape to `(num_groups, group_size)`, then `grouped.max(dim=-1).values - grouped.min(dim=-1).values` gives the spread per group. Compare it against `tol` to get a `(num_groups,)` boolean of degenerate groups and negate it to get the keep decision. To expand back, `keep.unsqueeze(-1).expand(-1, group_size).reshape(-1)` restores the group-major layout exactly; `repeat_interleave` on the group axis does the same thing and is easier to read. Do not use `repeat`, which tiles the whole vector rather than repeating each element, and produces a mask that is silently misaligned with the rewards. A group of size one falls out correctly with no special case, since its max equals its min. Validate `group_size` and `tol` before the reshape so the error names the real problem. Checking variance instead of spread also works, but the spread is exact for a constant group while a variance can carry float error, which matters when `tol` is zero.",
        },
    ],
    "model_connections": [
        "DAPO introduced dynamic sampling as one of its four modifications to GRPO, alongside clip-higher, token-level loss and overlong-reward shaping.",
        "OpenRLHF and verl both implement the filter as a pre-advantage step over the rollout buffer, optionally resampling to refill the batch to its target size.",
        "The exercise pairs with group_relative_advantage: that function's epsilon guard is what keeps a degenerate group finite when this filter is not applied.",
        "The same idea appears in curriculum and difficulty-sampling work, where prompts that are always or never solved are removed from the training pool rather than from the batch.",
    ],
    "pro_con_analysis": {
        "pros": [
            "Removes responses that provably contribute zero gradient, so the effective batch is all signal.",
            "Symmetric in difficulty: it drops hopeless prompts as readily as trivial ones, avoiding a drift toward easy data.",
            "Cheap — two reductions over a small tensor, computed before any model forward pass on the filtered batch.",
        ],
        "cons": [
            "The surviving batch size is data-dependent, so gradient noise varies between steps unless the batch is refilled by resampling.",
            "Resampling to a fixed batch size costs more generation, which is the main expense of the whole method.",
            "A prompt that is currently hopeless may become learnable later, and dropping it every step means the policy never gets the chance.",
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
            "adapted": "The group-major reward layout and per-group reduction that this filter operates on, taken from the episode construction that computes group advantages.",
            "simplifications": "Pure boolean filter over a flat reward vector: no resampling, no episode rebuilding and no batch refill.",
        },
        {
            "kind": "paper",
            "url": "https://arxiv.org/abs/2503.14476",
            "section": "3.2 Dynamic Sampling",
        },
    ],
    "tests": [
        {
            "name": "Drops an all-identical group and keeps a varied one",
            "behavior": "rl.rollout_assembly",
            "code": r"""
import torch
# Group 0 is constant; group 1 varies.
rewards = torch.tensor([1.0, 1.0, 0.0, 1.0])
out = {fn}(rewards, 2)
assert out.dtype == torch.bool, f'expected bool, got {out.dtype}'
assert out.shape == (4,), f'expected (4,), got {tuple(out.shape)}'
assert torch.equal(out, torch.tensor([False, False, True, True])), out
""",
        },
        {
            "name": "All-wrong groups are dropped too",
            "behavior": "rl.rollout_assembly",
            "code": r"""
import torch
# Group 0 never solved, group 1 always solved, group 2 mixed.
rewards = torch.tensor([0.0, 0.0, 1.0, 1.0, 0.0, 1.0])
out = {fn}(rewards, 2)
assert torch.equal(out, torch.tensor([False, False, False, False, True, True])), out
""",
        },
        {
            "name": "Whole groups are kept or dropped together",
            "visibility": "unshown",
            "behavior": "rl.rollout_assembly",
            "failure_message": "Individual responses were filtered instead of whole groups. Every member of a group must share one decision.",
            "code": r"""
import torch
torch.manual_seed(0)
group_size = 4
rewards = torch.randn(5 * group_size)
rewards[:group_size] = 0.5
out = {fn}(rewards, group_size)
grouped = out.reshape(5, group_size)
for g in range(5):
    row = grouped[g]
    assert bool(row.all()) or not bool(row.any()), f'group {g} was split: {row}'
assert not bool(grouped[0].any()), 'the constant group should be dropped'

# A varied group whose middle member sits exactly on the group mean. A per-response
# filter would drop that one member and keep the rest; the decision is per group.
on_the_mean = torch.tensor([0.0, 1.0, 2.0])
out = {fn}(on_the_mean, 3)
assert torch.equal(out, torch.ones(3, dtype=torch.bool)), f'the whole group must be kept, got {out}'

# The same trap with a degenerate group alongside it.
mixed = torch.tensor([0.0, 1.0, 2.0, 4.0, 4.0, 4.0])
out = {fn}(mixed, 3)
expected = torch.tensor([True, True, True, False, False, False])
assert torch.equal(out, expected), f'{out} vs {expected}'
""",
        },
        {
            "name": "The mask is aligned with the group-major layout",
            "visibility": "unshown",
            "behavior": "rl.rollout_assembly",
            "failure_message": "The per-group decision was tiled rather than repeated, so the mask is misaligned with the rewards. Repeat each decision group_size times in place.",
            "code": r"""
import torch
# Groups alternate degenerate and varied, which a tiled mask would get wrong.
rewards = torch.tensor([0.0, 0.0,   0.0, 1.0,   2.0, 2.0,   3.0, 9.0])
out = {fn}(rewards, 2)
expected = torch.tensor([False, False, True, True, False, False, True, True])
assert torch.equal(out, expected), f'{out} vs {expected}'
""",
        },
        {
            "name": "Matches a per-group oracle on random batches",
            "visibility": "unshown",
            "behavior": "rl.rollout_assembly",
            "failure_message": "The decision disagrees with an independent per-group spread computation.",
            "code": r"""
import torch
torch.manual_seed(1)
for num_groups, group_size in ((6, 4), (3, 8), (5, 2)):
    rewards = (torch.rand(num_groups * group_size) > 0.5).float()
    out = {fn}(rewards, group_size)
    expected = torch.zeros(num_groups * group_size, dtype=torch.bool)
    for g in range(num_groups):
        chunk = rewards[g * group_size:(g + 1) * group_size]
        keep = float(chunk.max() - chunk.min()) > 0.0
        expected[g * group_size:(g + 1) * group_size] = keep
    assert torch.equal(out, expected), f'{out} vs {expected}'
""",
        },
        {
            "name": "Dropped groups are exactly the ones with zero advantage",
            "visibility": "unshown",
            "behavior": "rl.advantage",
            "failure_message": "The filter does not agree with the advantage it exists to protect. A dropped group must be one whose group-relative advantages are all zero.",
            "code": r"""
import torch
torch.manual_seed(2)
group_size = 4
rewards = torch.tensor([1.0, 1.0, 1.0, 1.0,
                        0.0, 1.0, 0.0, 1.0,
                        0.0, 0.0, 0.0, 0.0])
keep = {fn}(rewards, group_size)
grouped = rewards.reshape(3, group_size)
advantages = (grouped - grouped.mean(dim=-1, keepdim=True)) / (
    grouped.std(dim=-1, unbiased=False, keepdim=True) + 1e-4
)
for g in range(3):
    all_zero = bool(torch.allclose(advantages[g], torch.zeros(group_size), atol=1e-6))
    dropped = not bool(keep.reshape(3, group_size)[g].any())
    assert all_zero == dropped, f'group {g}: zero-advantage={all_zero} dropped={dropped}'
""",
        },
        {
            "name": "Tolerance widens the definition of degenerate",
            "visibility": "unshown",
            "behavior": "edge.empty_or_boundary",
            "failure_message": "The tol parameter was ignored or compared with the wrong strictness. A spread equal to tol counts as degenerate.",
            "code": r"""
import torch
rewards = torch.tensor([1.0, 1.05, 0.0, 5.0])
assert torch.equal({fn}(rewards, 2, tol=0.0), torch.tensor([True, True, True, True]))
# A spread of 0.05 is at or below tol, so group 0 becomes degenerate.
assert torch.equal({fn}(rewards, 2, tol=0.05), torch.tensor([False, False, True, True]))
assert torch.equal({fn}(rewards, 2, tol=0.1), torch.tensor([False, False, True, True]))
""",
        },
        {
            "name": "A group of one is always degenerate",
            "visibility": "unshown",
            "behavior": "edge.empty_or_boundary",
            "failure_message": "group_size=1 did not drop every response. A single completion has zero spread, so it carries no group-relative signal.",
            "code": r"""
import torch
out = {fn}(torch.tensor([5.0, -3.0, 0.0]), 1)
assert torch.equal(out, torch.zeros(3, dtype=torch.bool)), out
""",
        },
        {
            "name": "Rejects an invalid group size or tolerance",
            "visibility": "unshown",
            "behavior": "contract.signature",
            "failure_message": "An invalid argument was accepted. Raise ValueError for a non-positive or non-dividing group_size, or a negative tol.",
            "code": r"""
import torch
rewards = torch.randn(6)
bad = ((0, 0.0), (-2, 0.0), (4, 0.0), (5, 0.0), (2, -0.1))
for group_size, tol in bad:
    try:
        {fn}(rewards, group_size, tol=tol)
    except ValueError:
        continue
    raise AssertionError(f'group_size={group_size} tol={tol} should raise ValueError')
""",
        },
    ],
    "solution": '''import torch


def dapo_dynamic_sampling(rewards, group_size, tol=0.0):
    if not isinstance(group_size, int) or group_size <= 0:
        raise ValueError(f"group_size must be a positive integer, got {group_size!r}")
    if rewards.numel() % group_size != 0:
        raise ValueError(
            f"group_size {group_size} does not divide {rewards.numel()} rewards"
        )
    if tol < 0:
        raise ValueError(f"tol must be non-negative, got {tol}")

    grouped = rewards.reshape(-1, group_size)
    spread = grouped.max(dim=-1).values - grouped.min(dim=-1).values
    keep = spread > tol
    return keep.repeat_interleave(group_size)
''',
}
