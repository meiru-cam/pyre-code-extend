"""chunk_step_executor's reference solution, checked against its upstream counterparts.

RLinf's LIBERO chunk_step is the oracle for the executed-action count and the
prefix valid mask. ManiSkill's ActionRepeatWrapper is the oracle for the
not-done update of observation, reward and flags and for the early break. The
deliberate divergences from RLinf (flags and observation from the ending step,
early stop) are pinned at the bottom.
"""

from __future__ import annotations

import random

import pytest
import torch

from torch_judge.harness.embodied_rl.vendored.rlinf_maniskill_chunk_step import (
    maniskill_action_repeat_step,
    rlinf_libero_chunk_step,
    valid_action_mask_from_counts,
)
from torch_judge.tasks import get_task

TASK_ID = "chunk_step_executor"


def reference():
    task = get_task(TASK_ID)
    assert task is not None, f"unknown task {TASK_ID!r}"
    namespace: dict = {}
    exec(task["solution"], namespace)
    return namespace[task["function_name"]]


class FakeEnv:
    """Env b ends at slot ends[b] (None: never) with flags kinds[b] = (term, trunc).

    ``junk=False``: after the end the env reports both flags False and keeps its
    normal reward and observation (RLinf's any() fold then agrees with the
    ending step). ``junk=True``: after the end it reports seeded random flags, a
    large reward and an observation of -999. Every call returns fresh tensors,
    because ManiSkill's wrapper writes into the first step's tensors.
    """

    def __init__(self, ends, kinds, H, seed, junk):
        self.ends, self.kinds, self.junk = list(ends), list(kinds), junk
        g = torch.Generator().manual_seed(seed)
        self.r = torch.randn(len(ends), H + 1, generator=g, dtype=torch.float64)
        self.junk_flags = torch.rand(len(ends), H + 1, 2, generator=g) < 0.5
        self.t = 0

    def __call__(self, actions):
        t = self.t
        self.t += 1
        B = len(self.ends)
        obs = torch.stack([torch.full((B,), float(t), dtype=torch.float64), actions.sum(-1).double()], dim=1)
        reward = self.r[:, t].clone()
        term = torch.zeros(B, dtype=torch.bool)
        trunc = torch.zeros(B, dtype=torch.bool)
        for b, e in enumerate(self.ends):
            if e is not None and t == e:
                term[b], trunc[b] = self.kinds[b]
            elif e is not None and t > e and self.junk:
                reward[b] = 1000.0 + t
                term[b] = bool(self.junk_flags[b, t, 0])
                trunc[b] = bool(self.junk_flags[b, t, 1])
                obs[b] = -999.0
        return obs, reward, term, trunc


KIND_CHOICES = [(True, False), (False, True), (True, True)]


def random_script(seed: int):
    rng = random.Random(seed)
    B, H = rng.randint(1, 6), rng.randint(1, 8)
    ends = [rng.choice([None] + list(range(H))) for _ in range(B)]
    kinds = [rng.choice(KIND_CHOICES) for _ in range(B)]
    return B, H, ends, kinds


EDGE_SCRIPTS = [
    (3, [None, None, None], [(True, False)] * 3),
    (4, [0, 0], [(True, False), (False, True)]),
    (1, [0, None], [(True, True), (True, False)]),
    (5, [4, 2, None, 0], [(False, True), (True, True), (True, False), (True, False)]),
    (6, [5], [(True, False)]),
]


def scripts():
    out = []
    for seed in range(8):
        B, H, ends, kinds = random_script(seed)
        out.append((seed, H, ends, kinds))
    for i, (H, ends, kinds) in enumerate(EDGE_SCRIPTS):
        out.append((100 + i, H, ends, kinds))
    return out


@pytest.mark.parametrize("seed,H,ends,kinds", scripts())
def test_matches_rlinf_mask_counts_and_rewards(seed, H, ends, kinds):
    B = len(ends)
    chunk = torch.randn(B, H, 3, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)
    out = reference()(FakeEnv(ends, kinds, H, seed, junk=False), chunk)
    obs_list, rewards, terms, truncs, mask = rlinf_libero_chunk_step(FakeEnv(ends, kinds, H, seed, junk=False), chunk)

    assert torch.equal(out["valid_mask"], mask)
    assert torch.equal(out["num_executed"], mask.sum(dim=1))
    assert torch.equal(out["valid_mask"], valid_action_mask_from_counts(out["num_executed"], H))
    assert torch.allclose(out["rewards"], rewards * mask)
    assert torch.allclose(out["chunk_reward"], (rewards * mask).sum(dim=1))
    # Without post-end flags, RLinf's any() fold equals the ending step's flags.
    assert torch.equal(out["terminated"], terms[:, -1])
    assert torch.equal(out["truncated"], truncs[:, -1])
    for b in range(B):
        k = int(out["num_executed"][b])
        assert torch.equal(out["final_obs"][b], obs_list[k - 1][b])


@pytest.mark.parametrize("seed,H,ends,kinds", scripts())
def test_matches_maniskill_action_repeat_on_a_repeated_action(seed, H, ends, kinds):
    B = len(ends)
    action = torch.randn(B, 2, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)
    chunk = action.unsqueeze(1).expand(B, H, 2).contiguous()

    ref_env = FakeEnv(ends, kinds, H, seed, junk=True)
    out = reference()(ref_env, chunk)
    ms_env = FakeEnv(ends, kinds, H, seed, junk=True)
    obs, rew, term, trunc = maniskill_action_repeat_step(ms_env, action, H)

    assert torch.allclose(out["chunk_reward"], rew)
    assert torch.equal(out["final_obs"], obs)
    assert torch.equal(out["terminated"], term)
    assert torch.equal(out["truncated"], trunc)
    assert ref_env.t == ms_env.t, "both stop once every env is done"


def test_divergence_rlinf_folds_post_end_flags_with_any():
    """Env 0 truncates at slot 1, then the simulator reports terminated at slot 2.

    RLinf's any() fold reports both flags; the exercise keeps the ending step's.
    """
    term = torch.tensor([[False, False, True], [False, False, False]])
    trunc = torch.tensor([[False, True, False], [False, False, False]])

    def make():
        state = {"t": 0}

        def step(actions):
            t = state["t"]
            state["t"] += 1
            return torch.full((2, 1), float(t)), torch.ones(2), term[:, t].clone(), trunc[:, t].clone()

        return step

    chunk = torch.zeros(2, 3, 1)
    out = reference()(make(), chunk)
    obs_list, _, terms, truncs, mask = rlinf_libero_chunk_step(make(), chunk)
    assert terms[:, -1].tolist() == [True, False] and truncs[:, -1].tolist() == [True, False]
    assert out["terminated"].tolist() == [False, False] and out["truncated"].tolist() == [True, False]
    assert torch.equal(out["valid_mask"], mask)
    # RLinf's last-slot observation is from after env 0's end; the exercise keeps slot 1.
    assert obs_list[-1][0].item() == 2.0 and out["final_obs"][0].item() == 1.0


def test_divergence_rlinf_always_runs_the_full_chunk():
    ends, kinds, H = [0, 1], [(True, False), (False, True)], 6
    chunk = torch.zeros(2, H, 1, dtype=torch.float64)
    ref_env = FakeEnv(ends, kinds, H, 0, junk=True)
    rl_env = FakeEnv(ends, kinds, H, 0, junk=True)
    reference()(ref_env, chunk)
    rlinf_libero_chunk_step(rl_env, chunk)
    assert ref_env.t == 2 and rl_env.t == H


def test_rlinf_mask_counts_the_ending_action():
    assert valid_action_mask_from_counts(torch.tensor([1, 3, 0]), 3).tolist() == [
        [True, False, False],
        [True, True, True],
        [False, False, False],
    ]
