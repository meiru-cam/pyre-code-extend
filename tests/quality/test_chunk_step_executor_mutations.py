"""Mutation gate for the chunk_step_executor exercise."""

from __future__ import annotations

import pytest

from torch_judge.tasks import get_task
from torch_judge.tasks._schema import validate_task
from tests.quality.mutation_runner import Mutation, assert_mutations_rejected

TASK_ID = "chunk_step_executor"


def _mutation(name: str, *edits: tuple[str, str]) -> Mutation:
    task = get_task(TASK_ID)
    assert task is not None, TASK_ID
    solution = task["solution"]
    for old, new in edits:
        assert old in solution, f"mutation target drifted: {name}"
        solution = solution.replace(old, new, 1)
    return Mutation(name, solution)


# RLinf-style vectorized count, but without the torch.where for envs that never
# ended: argmax of an all-False row is 0, so such an env gets a count of 1.
ARGMAX_WITHOUT_WHERE = '''import torch


def chunk_step(env_step, chunk):
    if not torch.is_tensor(chunk) or chunk.ndim != 3 or not chunk.is_floating_point():
        raise ValueError("bad chunk")
    B, H, _ = chunk.shape
    if B == 0 or H == 0:
        raise ValueError("bad chunk")
    obs, rew, te, tr = [], [], [], []
    ended = torch.zeros(B, dtype=torch.bool)
    for h in range(H):
        o, r, a, b = env_step(chunk[:, h])
        obs.append(o.clone()); rew.append(r.clone()); te.append(a.bool().clone()); tr.append(b.bool().clone())
        ended = ended | a.bool() | b.bool()
        if ended.all():
            break
    steps = len(rew)
    R = torch.zeros(B, H, dtype=rew[0].dtype)
    R[:, :steps] = torch.stack(rew, 1)
    TE = torch.stack(te, 1)
    TR = torch.stack(tr, 1)
    counts = (TE | TR).long().argmax(dim=1) + 1
    mask = torch.arange(H).unsqueeze(0) < counts.unsqueeze(1)
    rewards = R * mask
    idx = torch.arange(B)
    O = torch.stack(obs, 1)
    return {
        "num_executed": counts,
        "valid_mask": mask,
        "rewards": rewards,
        "chunk_reward": rewards.sum(dim=1),
        "terminated": TE[idx, counts - 1],
        "truncated": TR[idx, counts - 1],
        "final_obs": O[idx, counts - 1],
    }
'''


MUTATIONS = [
    _mutation(
        "mask excludes the ending action",
        ("        num_executed[active] += 1\n        active = active & ~(term | trunc)\n",
         "        active = active & ~(term | trunc)\n        num_executed[active] += 1\n"),
    ),
    _mutation(
        "chunk reward summed over every returned reward",
        ("        rewards[active, h] = reward[active]\n",
         "        rewards[active, h] = reward[active]\n        raw_sum = reward.clone() if h == 0 else raw_sum + reward\n"),
        ('"chunk_reward": rewards.sum(dim=1)', '"chunk_reward": raw_sum'),
    ),
    _mutation(
        "per-slot rewards not zeroed after the end",
        ("        rewards[active, h] = reward[active]\n", "        rewards[:, h] = reward\n"),
        ('"chunk_reward": rewards.sum(dim=1)', '"chunk_reward": (rewards * valid_mask).sum(dim=1)'),
    ),
    _mutation(
        "final_obs taken from the last call",
        ("        final_obs[active] = obs[active]\n", "        final_obs = obs.clone()\n"),
    ),
    _mutation(
        "flags folded with any() over the chunk",
        ("        terminated[active] = term[active]\n        truncated[active] = trunc[active]\n",
         "        terminated |= term\n        truncated |= trunc\n"),
    ),
    _mutation(
        "always steps the full chunk",
        ("        if not active.any():\n            break\n", "        pass\n"),
    ),
    _mutation(
        "stops when any env ends",
        ("        if not active.any():\n", "        if not active.all():\n"),
    ),
    _mutation(
        "only termination ends an episode",
        ("        active = active & ~(term | trunc)\n", "        active = active & ~term\n"),
    ),
    _mutation(
        "keeps the simulator's observation buffer without copying",
        ("            final_obs = obs.clone()\n", "            final_obs = obs\n"),
    ),
    _mutation(
        "rewards cast to float32",
        ("dtype=reward.dtype", "dtype=torch.float32"),
    ),
    _mutation(
        "accepts an integer chunk",
        ("chunk.ndim != 3 or not chunk.is_floating_point()", "chunk.ndim != 3"),
    ),
    Mutation("never-ended env counted as one action", ARGMAX_WITHOUT_WHERE),
]


def test_chunk_step_executor_metadata_is_valid():
    task = get_task(TASK_ID)
    assert task is not None
    validate_task(TASK_ID, task)
    assert not any(key.endswith("_zh") for key in task), "new exercises are English only"
    assert any(t.get("visibility", "visible") == "visible" for t in task["tests"])


@pytest.mark.parametrize("_repeat", range(3))
def test_chunk_step_executor_reference_and_mutations(_repeat):
    assert set(assert_mutations_rejected(TASK_ID, MUTATIONS, require_unshown=True)) == {m.name for m in MUTATIONS}
