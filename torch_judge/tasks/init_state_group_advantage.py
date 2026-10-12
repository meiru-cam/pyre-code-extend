"""Init-state group advantage: GRPO and RLOO for robot rollouts on an action-chunk token grid."""

from ._interview import interview

# An independent per-rollout oracle, written with plain Python lists and dicts.
_LOOP_ORACLE = r"""
import math

def loop_oracle(success, task_ids, init_ids, finish, tpa, num_chunks, apc, mode, drop, eps):
    n = len(success)
    groups = {}
    for i in range(n):
        groups.setdefault((task_ids[i], init_ids[i]), []).append(i)
    adv = [0.0] * n
    keep = [False] * n
    for members in groups.values():
        rs = [1.0 if success[i] else 0.0 for i in members]
        if drop and (min(rs) == max(rs)):
            continue
        k = len(rs)
        for j, i in enumerate(members):
            keep[i] = True
            if k == 1:
                adv[i] = rs[j] / (1.0 + eps) if mode == "grpo" else rs[j]
            elif mode == "grpo":
                mean = sum(rs) / k
                std = math.sqrt(sum((x - mean) ** 2 for x in rs) / (k - 1))
                adv[i] = (rs[j] - mean) / (std + eps)
            else:
                adv[i] = rs[j] - (sum(rs) - rs[j]) / (k - 1)
    slots = num_chunks * apc * tpa
    mask = [[keep[i] and t < finish[i] * tpa for t in range(slots)] for i in range(n)]
    grid = [[adv[i] if mask[i][t] else 0.0 for t in range(slots)] for i in range(n)]
    return keep, adv, mask, grid
"""

TASK = {
    "title": "Init-State Group Advantage for Robot Rollouts",
    "difficulty": "Medium",
    "version": 1,
    "function_name": "robot_group_advantage",
    "description_en": r"""Turn binary success flags from robot rollouts into per-token advantages: group the rollouts by task and initial state, drop groups with no signal, compute a GRPO or RLOO advantage per rollout, and spread it over the action tokens the robot actually executed.

**Signature:** `robot_group_advantage(success, task_ids, init_state_ids, finish_steps, tokens_per_action, num_chunks, actions_per_chunk, mode="grpo", drop_uniform=True, eps=1e-6) -> dict`

**Parameters:**
- `success` — bool tensor `(N,)`. The reward of rollout `i` is `1.0` if `success[i]` else `0.0`.
- `task_ids`, `init_state_ids` — integer tensors `(N,)`. Rollout `i` started task `task_ids[i]` from that task's initial state number `init_state_ids[i]`.
- `finish_steps` — integer tensor `(N,)`. The number of environment steps (single actions, not chunks) rollout `i` executed before it ended.
- `tokens_per_action` — positive int. Tokens the policy emits per action.
- `num_chunks`, `actions_per_chunk` — positive ints. Each rollout's token grid holds `num_chunks` chunks of `actions_per_chunk` actions, so it has `S = num_chunks * actions_per_chunk * tokens_per_action` slots, laid out chunk by chunk and, inside a chunk, action by action.
- `mode` — `"grpo"` or `"rloo"`.
- `drop_uniform` — when True, drop every group whose rollouts all have the same outcome.
- `eps` — float added to the GRPO standard deviation.

**Returns:** a dict with
- `"keep"` — bool `(N,)`, True for rollouts of kept groups.
- `"rollout_advantages"` — float32 `(N,)`, one advantage per rollout, `0.0` for dropped rollouts.
- `"mask"` — bool `(N, S)`, `mask[i, t] = keep[i] and t < finish_steps[i] * tokens_per_action`.
- `"advantages"` — float32 `(N, S)`, `rollout_advantages[i]` where `mask[i, t]`, else `0.0`.

**Groups.** A group is all rollouts with the same `(task_ids[i], init_state_ids[i])` pair. Rollouts of one group can sit anywhere in the batch, and groups can have different sizes. For a group with rewards `r` of size `k`:

    grpo, k >= 2:  A_i = (r_i - mean(r)) / (std(r) + eps)     std with the k - 1 divisor
    rloo, k >= 2:  A_i = r_i - (sum(r) - r_i) / (k - 1)
    k == 1:        A_i = r_i / (1 + eps) for grpo,  A_i = r_i for rloo

**Constraints:**
- Key groups by the pair. The same initial-state number under two tasks is two groups, and one task from two initial states is two groups.
- GRPO uses the sample standard deviation (divide by `k - 1`), not the population one.
- With `drop_uniform=True`, an all-success group and an all-failure group are both dropped, and so is a group of one.
- A dropped rollout has `keep` False, an all-False mask row, and zero advantages.
- The mask counts environment steps times `tokens_per_action`. When a rollout ends mid-chunk, the tokens of the actions it never executed are masked out.
- Raise `ValueError` when the four tensors are not 1-D with one length, when `mode` is unknown, when `tokens_per_action`, `num_chunks` or `actions_per_chunk` is below 1, or when any `finish_steps` is outside `[0, num_chunks * actions_per_chunk]`.

**Example.** `tokens_per_action=2`, `num_chunks=2`, `actions_per_chunk=2` (so `S = 8`), `task_ids=[0,0,0,1,1]`, `init_state_ids=[3,3,3,3,3]`, `success=[T,F,F,T,T]`, `finish_steps=[2,4,3,1,4]`, GRPO. Group `(0,3)` has rewards `[1,0,0]`: mean `1/3`, sample std `0.5774`, advantages about `[1.1547, -0.5774, -0.5774]`. Group `(1,3)` is all-success and is dropped. Mask rows: 4, 8, 6, 0 and 0 leading True slots.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why the group key is the initial state.** In LLM GRPO a group is the samples for one prompt. For a robot, the same instruction from a different initial object layout is a different problem, so comparing a rollout from an easy layout against one from a hard layout would reward the layout rather than the policy. SimpleVLA-RL replicates each `(task_id, trial_id)` dataset row `n_samples` times under one `uid` and the env worker loads initial state `trial_id` of that task; RIPT-VLA tiles one initial state across its leave-one-out batch.

**Why drop uniform groups.** When every rollout in a group succeeds, or every one fails, each advantage is zero and the group adds no gradient. SimpleVLA-RL (Section 3.3) excludes such groups during dynamic sampling and keeps sampling until the batch holds only mixed-outcome groups.

**Why env steps, not chunks.** The env worker stops in the middle of a chunk as soon as the episode is done, so the remaining actions of that chunk were generated but never executed. SimpleVLA-RL's mask is `arange(S) < finish_step * action_token_len`, which keeps exactly the executed actions.

**Why the sample standard deviation and the singleton rule.** They are what SimpleVLA-RL's `compute_grpo_outcome_advantage` does: it calls `torch.std`, whose default is unbiased, and gives a group of one a mean of 0 and a standard deviation of 1. The `group_relative_advantage` exercise uses the population standard deviation; the two differ by a factor of `sqrt((k - 1) / k)`.""",
    "advisory_prerequisites": ["group_relative_advantage", "dapo_dynamic_sampling", "response_token_mask"],
    "hints": [
        {"level": 1, "kind": "questions", "content": "What single key identifies the group of a rollout, and what goes wrong if you key by task alone or by initial-state number alone? Which standard deviation does the contract ask for, and what is it for a group of one? How many token slots did a rollout that stopped after 5 actions really produce, and what happens to the rest of its last chunk? Where must a dropped group disappear: in the advantages, in the mask, or both?"},
        {"level": 2, "kind": "analysis", "content": "Validate the inputs first. Stack task and initial-state ids into pairs and map each pair to a group number, for example with unique over rows and return_inverse. For each group, gather its rewards, skip it when drop_uniform is set and all rewards are equal, otherwise mark it kept and fill in its advantages with the rule for its size and mode. Build the mask by comparing a position row of length S against finish_steps times tokens_per_action, and with keep. Multiply the per-rollout advantages, as a column, by the mask."},
    ],
    "model_connections": [
        "SimpleVLA-RL computes GRPO advantages per uid with compute_grpo_outcome_advantage and masks them with arange < finish_step * action_token_len before the PPO-style clipped loss.",
        "RIPT-VLA runs its leave-one-out batch from one tiled initial state and discards a sample whose rollouts are all successes or all failures when dynamic sampling is on.",
        "RLinf's ManiSkill environment draws one reset-state id per group and repeats it group_size times; with use_fixed_reset_state_ids set, each env resets from its group's id.",
    ],
    "pro_con_analysis": {
        "pros": [
            "No value network: the other rollouts from the same initial state give the baseline.",
            "Keying by initial state keeps easy and hard layouts from being compared against each other.",
            "Dropping uniform groups keeps the batch made of groups that carry gradient.",
        ],
        "cons": [
            "Each initial state needs several full simulator rollouts, which are far more expensive than LLM samples.",
            "A binary outcome spread uniformly over all executed tokens cannot tell which action in a long episode helped.",
            "Dynamic sampling has to roll out more groups than it keeps once most initial states are always solved or never solved.",
        ],
    },
    "sources": [
        {
            "kind": "code",
            "url": "https://github.com/PRIME-RL/SimpleVLA-RL",
            "commit": "7c51662df27b586f9e8a1ab35fcf849f2b8852f9",
            "path": "verl/trainer/ppo/core_algos.py",
            "symbol": "compute_grpo_outcome_advantage",
            "license": "MIT",
            "adapted": "Per-uid mean and unbiased torch.std, singleton groups with mean 0 and std 1, eps added to the std, advantage tiled over the masked token grid.",
            "simplifications": "Rewards arrive as success flags instead of token-level rewards summed per row; the group key is an explicit (task_id, init_state_id) pair instead of a uid string; advantages are float32 throughout.",
        },
        {
            "kind": "code",
            "url": "https://github.com/PRIME-RL/SimpleVLA-RL",
            "commit": "7c51662df27b586f9e8a1ab35fcf849f2b8852f9",
            "path": "verl/trainer/ppo/ray_trainer.py",
            "symbol": "compute_advantage (response mask); RayTrainer.filter (accuracy branch)",
            "license": "MIT",
            "adapted": "The prefix mask arange(traj_len * chunk_tokens) < finish_step * action_token_len, and dropping groups by their success rate.",
            "simplifications": "Upstream keeps groups whose mean accuracy lies in [accuracy_lower_bound, accuracy_upper_bound] over consecutive blocks of n_samples rollouts; here only all-equal groups are dropped, and groups are found by key so they need not be contiguous. Dropped rollouts are masked instead of removed from the batch. SimpleVLA-RL's own rloo estimator (compute_rloo_returns) also discounts and whitens across the batch; the rloo mode here is the plain leave-one-out baseline of verl's compute_rloo_outcome_advantage.",
        },
        {
            "kind": "code",
            "url": "https://github.com/volcengine/verl",
            "commit": "00094bd9cd3fef9cf8903daf60ea4a2bcf832efc",
            "path": "verl/trainer/ppo/core_algos.py",
            "symbol": "compute_rloo_outcome_advantage",
            "license": "Apache-2.0",
            "adapted": "Leave-one-out baseline per group index, with a group of one keeping its raw score.",
            "simplifications": "Applied to success flags grouped by (task_id, init_state_id).",
        },
        {
            "kind": "paper",
            "url": "https://arxiv.org/abs/2509.09674",
            "section": "SimpleVLA-RL, Section 2.3 (GRPO, Eq. 8 and 12), Section 3.2 (binary outcome reward), Section 3.3 (dynamic sampling)",
            "note": "Groups of trajectories from one initial state, 0/1 trajectory reward propagated to action tokens, and exclusion of all-success and all-failure groups.",
        },
        {
            "kind": "paper",
            "url": "https://arxiv.org/abs/2505.17016",
            "section": "RIPT-VLA abstract; code ript/algos/rl_optimizers/rollout_generator.py at commit 440990e8864e12e4578b490ff6359e4f2c49ae3e (no license, cited only)",
            "note": "Leave-one-out advantages over rollouts from one tiled initial state, with dynamic sampling that discards uniform outcomes.",
        },
    ],
    "tests": [
        {"name": "GRPO with one dropped all-success group", "behavior": "rl.advantage", "code": r"""
import torch
T, F = True, False
out = {fn}(torch.tensor([T, F, F, T, T]), torch.tensor([0, 0, 0, 1, 1]), torch.tensor([3, 3, 3, 3, 3]),
           torch.tensor([2, 4, 3, 1, 4]), 2, 2, 2)
assert out["keep"].tolist() == [True, True, True, False, False], out["keep"]
want = torch.tensor([1.1547, -0.5774, -0.5774, 0.0, 0.0])
assert torch.allclose(out["rollout_advantages"], want, atol=1e-3), out["rollout_advantages"]
assert out["mask"].sum(dim=1).tolist() == [4, 8, 6, 0, 0], out["mask"]
assert torch.allclose(out["advantages"][0], torch.tensor([1.1547] * 4 + [0.0] * 4), atol=1e-3), out["advantages"][0]
"""},
        {"name": "RLOO keeps uniform and single-rollout groups when drop_uniform is False", "behavior": "rl.advantage", "code": r"""
import torch
out = {fn}(torch.tensor([True, True, False, False, True]), torch.tensor([0, 0, 0, 0, 2]),
           torch.tensor([1, 1, 1, 1, 1]), torch.tensor([4, 4, 4, 4, 4]), 1, 1, 4,
           mode="rloo", drop_uniform=False)
want = torch.tensor([2 / 3, 2 / 3, -2 / 3, -2 / 3, 1.0])
assert out["keep"].all(), out["keep"]
assert torch.allclose(out["rollout_advantages"], want, atol=1e-5), out["rollout_advantages"]
assert torch.allclose(out["advantages"], want.unsqueeze(1).expand(5, 4), atol=1e-5), out["advantages"]
"""},
        {"name": "Matches a per-rollout oracle on shuffled batches", "visibility": "unshown", "behavior": "rl.advantage",
         "failure_message": "Advantages disagree with a per-group loop; group by the (task, initial state) pair wherever the rollouts sit and apply the rule for the group's size and mode.",
         "code": _LOOP_ORACLE + r"""
import random, torch
for seed in range(6):
    rng = random.Random(seed)
    rows = []
    for g in range(rng.randint(3, 6)):
        task, init = rng.randint(0, 2), g
        p = rng.random()
        for _ in range(rng.choice([1, 2, 3, 5, 8])):
            rows.append((task, init, rng.random() < p))
    rng.shuffle(rows)
    tpa, nc, apc = rng.randint(1, 3), rng.randint(1, 3), rng.randint(2, 4)
    finish = [rng.randint(0, nc * apc) for _ in rows]
    for mode in ("grpo", "rloo"):
        for drop in (True, False):
            tids = [r[0] for r in rows]; iids = [r[1] for r in rows]; succ = [r[2] for r in rows]
            out = {fn}(torch.tensor(succ), torch.tensor(tids), torch.tensor(iids), torch.tensor(finish),
                       tpa, nc, apc, mode=mode, drop_uniform=drop)
            keep, adv, mask, grid = loop_oracle(succ, tids, iids, finish, tpa, nc, apc, mode, drop, 1e-6)
            assert out["keep"].tolist() == keep, (seed, mode, drop, "keep")
            assert torch.allclose(out["rollout_advantages"].double(), torch.tensor(adv, dtype=torch.float64), atol=1e-5), (seed, mode, drop)
            assert out["mask"].tolist() == mask, (seed, mode, drop, "mask")
            assert torch.allclose(out["advantages"].double(), torch.tensor(grid, dtype=torch.float64), atol=1e-5), (seed, mode, drop)
"""},
        {"name": "Group key is the (task, initial state) pair", "visibility": "unshown", "behavior": "rl.advantage",
         "failure_message": "Rollouts were grouped by task alone or by initial-state number alone; a group is all rollouts sharing both ids.",
         "code": r"""
import torch
# Pairs: (0,0) -> [1,0]; (1,0) -> [1,1,0]; (0,1) -> [0,0,1,1]
tids = torch.tensor([0, 0, 1, 1, 1, 0, 0, 0, 0])
iids = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1])
succ = torch.tensor([True, False, True, True, False, False, False, True, True])
fin = torch.full((9,), 2)
out = {fn}(succ, tids, iids, fin, 1, 1, 2, mode="rloo")
want = torch.tensor([1.0, -1.0, 0.5, 0.5, -1.0, -2 / 3, -2 / 3, 2 / 3, 2 / 3])
assert torch.allclose(out["rollout_advantages"], want, atol=1e-5), out["rollout_advantages"]
"""},
        {"name": "GRPO uses the sample standard deviation", "visibility": "unshown", "behavior": "rl.advantage",
         "failure_message": "The GRPO scale is off by the Bessel factor; divide by the standard deviation with the k - 1 divisor, as torch.std does by default.",
         "code": r"""
import torch
out = {fn}(torch.tensor([True, False, False, False]), torch.zeros(4, dtype=torch.long), torch.zeros(4, dtype=torch.long),
           torch.ones(4, dtype=torch.long), 1, 1, 1)
r = torch.tensor([1.0, 0.0, 0.0, 0.0])
want = (r - r.mean()) / (r.std(unbiased=True) + 1e-6)
assert torch.allclose(out["rollout_advantages"], want, atol=1e-5), (out["rollout_advantages"], want)
"""},
        {"name": "eps is added to the standard deviation", "visibility": "unshown", "behavior": "numerics.stability",
         "failure_message": "The eps argument was ignored or placed elsewhere; the denominator is std + eps.",
         "code": r"""
import math, torch
out = {fn}(torch.tensor([True, False]), torch.zeros(2, dtype=torch.long), torch.zeros(2, dtype=torch.long),
           torch.ones(2, dtype=torch.long), 1, 1, 1, eps=0.5)
a = 0.5 / (math.sqrt(0.5) + 0.5)
assert torch.allclose(out["rollout_advantages"], torch.tensor([a, -a]), atol=1e-5), out["rollout_advantages"]
"""},
        {"name": "Mask counts executed actions, including a mid-chunk stop", "visibility": "unshown", "behavior": "rl.masking",
         "failure_message": "The mask must cover finish_steps * tokens_per_action leading slots: count env steps, not chunks, and drop the unexecuted rest of the last chunk.",
         "code": r"""
import torch
# 3 chunks x 4 actions x 3 tokens = 36 slots.
succ = torch.tensor([True, False, True, False])
out = {fn}(succ, torch.zeros(4, dtype=torch.long), torch.zeros(4, dtype=torch.long),
           torch.tensor([5, 12, 1, 4]), 3, 3, 4)
assert out["mask"].shape == (4, 36), out["mask"].shape
for row, steps in enumerate([5, 12, 1, 4]):
    want = torch.arange(36) < steps * 3
    assert torch.equal(out["mask"][row], want), (row, out["mask"][row].int().tolist())
assert (out["advantages"][0, 15:] == 0).all() and (out["advantages"][0, :15] != 0).all(), out["advantages"][0]
"""},
        {"name": "Dropped groups leave no trace", "visibility": "unshown", "behavior": "rl.masking",
         "failure_message": "With drop_uniform=True, all-success, all-failure and single-rollout groups must get keep False, an empty mask row and zero advantages.",
         "code": r"""
import torch
# (0,0) all success; (0,1) all failure; (0,2) singleton; (0,3) mixed.
iids = torch.tensor([0, 0, 0, 1, 1, 2, 3, 3])
succ = torch.tensor([True, True, True, False, False, True, True, False])
for mode in ("grpo", "rloo"):
    out = {fn}(succ, torch.zeros(8, dtype=torch.long), iids, torch.full((8,), 3), 2, 1, 3, mode=mode)
    assert out["keep"].tolist() == [False] * 6 + [True, True], (mode, out["keep"])
    assert not out["mask"][:6].any(), (mode, out["mask"])
    assert (out["advantages"][:6] == 0).all() and (out["rollout_advantages"][:6] == 0).all(), mode
    assert out["mask"][6:].sum().item() == 12, (mode, out["mask"])
"""},
        {"name": "Without dropping, uniform groups give zero and singletons follow upstream", "visibility": "unshown", "behavior": "edge.empty_or_boundary",
         "failure_message": "With drop_uniform=False every group is kept: a uniform group gets zero advantage with a live mask, and a group of one gets r / (1 + eps) for grpo and r for rloo.",
         "code": r"""
import torch
iids = torch.tensor([0, 0, 0, 1, 2])
succ = torch.tensor([True, True, True, True, False])
for mode, single in (("grpo", 1.0 / (1.0 + 0.25)), ("rloo", 1.0)):
    out = {fn}(succ, torch.zeros(5, dtype=torch.long), iids, torch.full((5,), 2), 1, 1, 2,
               mode=mode, drop_uniform=False, eps=0.25)
    assert out["keep"].all(), (mode, out["keep"])
    assert out["mask"].all(), (mode, out["mask"])
    want = torch.tensor([0.0, 0.0, 0.0, single, 0.0])
    assert torch.allclose(out["rollout_advantages"], want, atol=1e-6), (mode, out["rollout_advantages"])
"""},
        {"name": "Output shapes and dtypes", "visibility": "unshown", "behavior": "tensor.shape",
         "failure_message": "Return keep (N,) bool, rollout_advantages (N,) float32, mask (N, S) bool and advantages (N, S) float32 with S = num_chunks * actions_per_chunk * tokens_per_action.",
         "code": r"""
import torch
out = {fn}(torch.tensor([True, False, True]), torch.tensor([4, 4, 4]), torch.tensor([7, 7, 7]),
           torch.tensor([3, 6, 0]), 7, 2, 3)
assert set(out) >= {"keep", "rollout_advantages", "mask", "advantages"}, set(out)
assert out["keep"].shape == (3,) and out["keep"].dtype == torch.bool
assert out["rollout_advantages"].shape == (3,) and out["rollout_advantages"].dtype == torch.float32
assert out["mask"].shape == (3, 42) and out["mask"].dtype == torch.bool
assert out["advantages"].shape == (3, 42) and out["advantages"].dtype == torch.float32
"""},
        {"name": "Rejects invalid inputs", "visibility": "unshown", "behavior": "contract.signature",
         "failure_message": "Raise ValueError for mismatched or non-1-D inputs, an unknown mode, a size argument below 1, or finish_steps outside [0, num_chunks * actions_per_chunk].",
         "code": r"""
import torch
s = torch.tensor([True, False]); z = torch.zeros(2, dtype=torch.long); f = torch.tensor([1, 2])
cases = [
    lambda: {fn}(torch.tensor([True]), z, z, f, 1, 1, 2),
    lambda: {fn}(s.reshape(1, 2), z.reshape(1, 2), z.reshape(1, 2), f.reshape(1, 2), 1, 1, 2),
    lambda: {fn}(s, z, z, f, 1, 1, 2, mode="ppo"),
    lambda: {fn}(s, z, z, f, 0, 1, 2),
    lambda: {fn}(s, z, z, f, 1, 0, 2),
    lambda: {fn}(s, z, z, f, 1, 1, 0),
    lambda: {fn}(s, z, z, torch.tensor([1, 3]), 1, 1, 2),
    lambda: {fn}(s, z, z, torch.tensor([-1, 2]), 1, 1, 2),
]
for i, call in enumerate(cases):
    try:
        call()
    except ValueError:
        continue
    raise AssertionError(f"case {i} should raise ValueError")
"""},
    ],
    "solution": '''import torch


def robot_group_advantage(success, task_ids, init_state_ids, finish_steps, tokens_per_action,
                          num_chunks, actions_per_chunk, mode="grpo", drop_uniform=True, eps=1e-6):
    tensors = (success, task_ids, init_state_ids, finish_steps)
    if any(t.ndim != 1 for t in tensors) or len({t.shape[0] for t in tensors}) != 1:
        raise ValueError("success, task_ids, init_state_ids and finish_steps must be 1-D with one length")
    if mode not in ("grpo", "rloo"):
        raise ValueError(f"mode must be 'grpo' or 'rloo', got {mode!r}")
    if tokens_per_action < 1 or num_chunks < 1 or actions_per_chunk < 1:
        raise ValueError("tokens_per_action, num_chunks and actions_per_chunk must be positive")
    max_steps = num_chunks * actions_per_chunk
    if bool((finish_steps < 0).any()) or bool((finish_steps > max_steps).any()):
        raise ValueError(f"finish_steps must lie in [0, {max_steps}]")

    n = success.shape[0]
    rewards = success.to(torch.float32)
    rollout_adv = torch.zeros(n, dtype=torch.float32)
    keep = torch.zeros(n, dtype=torch.bool)

    keys = torch.stack([task_ids.long(), init_state_ids.long()], dim=1)
    _, group = torch.unique(keys, dim=0, return_inverse=True)
    for g in group.unique():
        members = (group == g).nonzero(as_tuple=True)[0]
        r = rewards[members]
        if drop_uniform and bool((r == r[0]).all()):
            continue
        keep[members] = True
        size = r.numel()
        if size == 1:
            rollout_adv[members] = r / (1.0 + eps) if mode == "grpo" else r
        elif mode == "grpo":
            rollout_adv[members] = (r - r.mean()) / (r.std(unbiased=True) + eps)
        else:
            rollout_adv[members] = r - (r.sum() - r) / (size - 1)

    num_slots = max_steps * tokens_per_action
    positions = torch.arange(num_slots).unsqueeze(0)
    mask = (positions < (finish_steps.long() * tokens_per_action).unsqueeze(1)) & keep.unsqueeze(1)
    advantages = rollout_adv.unsqueeze(1) * mask
    return {"keep": keep, "rollout_advantages": rollout_adv, "mask": mask, "advantages": advantages}
''',
    "interview_questions": interview(
        concept=[
            "Why should GRPO groups for a robot policy be keyed by task and initial state rather than by the instruction text?",
            "Why do all-success and all-failure groups contribute nothing to a GRPO or RLOO update?",
        ],
        deep_dive=[
            "An episode ends after the third action of an eight-action chunk. Which of that chunk's tokens should carry advantage, and why?",
            "How does the sample standard deviation change GRPO advantages compared with the population one, and what does SimpleVLA-RL do for a group of one?",
            "Why must dropped groups also be removed from the token mask, even though their GRPO advantages are already zero?",
        ],
        tradeoffs=[
            "Several rollouts from one initial state versus one rollout from each of many initial states: what do you gain and lose in variance and coverage?",
            "Spreading one binary episode outcome over every executed token versus a learned critic with per-step values: when is each the better choice for robot RL?",
        ],
    ),
}
