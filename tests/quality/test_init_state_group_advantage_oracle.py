"""init_state_group_advantage, checked against SimpleVLA-RL and verl upstream code.

The GRPO advantage and the token mask are compared with SimpleVLA-RL's
compute_grpo_outcome_advantage and compute_advantage mask; the drop rule with its
accuracy filter; the RLOO mode with verl's compute_rloo_outcome_advantage.
"""

from __future__ import annotations

import random

import pytest
import torch

from torch_judge.harness.embodied_rl.vendored.simplevla_rl_core_algos import (
    svr_accuracy_filter,
    svr_grpo_outcome_advantage,
    svr_response_mask,
)
from torch_judge.harness.rl.vendored.verl_core_algos import verl_rloo_outcome_advantage
from torch_judge.tasks import get_task

TASK_ID = "init_state_group_advantage"


def reference():
    task = get_task(TASK_ID)
    namespace: dict = {}
    exec(task["solution"], namespace)
    return namespace[task["function_name"]]


def random_batch(seed: int, *, contiguous: bool = False, group_size: int | None = None):
    """Seeded rollouts with mixed, uniform and (unless group_size is fixed) single-rollout groups."""
    rng = random.Random(seed)
    rows = []
    for g in range(rng.randint(3, 7)):
        task, init = rng.randint(0, 2), g
        k = group_size or rng.choice([1, 2, 3, 4, 8])
        p = rng.choice([0.0, 1.0, rng.random()])
        rows += [(task, init, rng.random() < p) for _ in range(k)]
    if not contiguous:
        rng.shuffle(rows)
    tpa, nc, apc = rng.randint(1, 4), rng.randint(1, 4), rng.randint(1, 5)
    finish = [rng.randint(0, nc * apc) for _ in rows]
    return dict(
        success=torch.tensor([r[2] for r in rows]),
        task_ids=torch.tensor([r[0] for r in rows]),
        init_state_ids=torch.tensor([r[1] for r in rows]),
        finish_steps=torch.tensor(finish),
        tokens_per_action=tpa,
        num_chunks=nc,
        actions_per_chunk=apc,
    )


def upstream_inputs(batch):
    """Token-level rewards (the outcome in slot 0), SimpleVLA-RL's mask and a uid per (task, init) pair."""
    mask = svr_response_mask(batch["finish_steps"], batch["tokens_per_action"], batch["num_chunks"],
                             batch["actions_per_chunk"] * batch["tokens_per_action"])
    rewards = torch.zeros(mask.shape, dtype=torch.float32)
    rewards[:, 0] = batch["success"].float()
    index = [f"{t}-{i}" for t, i in zip(batch["task_ids"].tolist(), batch["init_state_ids"].tolist())]
    return rewards, mask, index


@pytest.mark.parametrize("seed", range(8))
def test_grpo_matches_simplevla_rl_without_dropping(seed):
    batch = random_batch(seed)
    rewards, mask, index = upstream_inputs(batch)
    out = reference()(**batch, mode="grpo", drop_uniform=False)
    want = svr_grpo_outcome_advantage(rewards, mask, index)
    assert torch.equal(out["mask"], mask)
    torch.testing.assert_close(out["advantages"], want, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("seed", range(8))
def test_rloo_matches_verl_without_dropping(seed):
    batch = random_batch(seed)
    rewards, mask, index = upstream_inputs(batch)
    out = reference()(**batch, mode="rloo", drop_uniform=False)
    want = verl_rloo_outcome_advantage(rewards, mask.float(), index)
    torch.testing.assert_close(out["advantages"], want, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("seed", range(6))
def test_drop_rule_matches_the_accuracy_filter_then_grpo_on_the_kept_rows(seed):
    # Upstream filters consecutive blocks of n_samples by mean accuracy, then computes
    # GRPO on the surviving rows. Bounds strictly between 0 and 1 drop exactly the
    # uniform groups. Deliberate divergence, pinned here: the exercise masks dropped
    # rows instead of removing them, so kept rows must equal upstream on the sliced batch.
    n = 4
    batch = random_batch(seed, contiguous=True, group_size=n)
    keep_up = svr_accuracy_filter(batch["success"].float().unsqueeze(1), n, lower=0.5 / n, upper=1 - 0.5 / n)
    out = reference()(**batch, mode="grpo", drop_uniform=True)
    assert torch.equal(out["keep"], keep_up)
    rewards, mask, index = upstream_inputs(batch)
    kept = keep_up.nonzero(as_tuple=True)[0]
    want = svr_grpo_outcome_advantage(rewards[kept], mask[kept], [index[i] for i in kept.tolist()])
    torch.testing.assert_close(out["advantages"][kept], want, atol=1e-5, rtol=1e-5)
    assert not out["mask"][~keep_up].any()
    assert (out["advantages"][~keep_up] == 0).all()


def test_edge_cases_against_upstream():
    fn = reference()
    cases = [
        # One singleton success: upstream mean 0, std 1 -> 1 / (1 + 1e-6).
        dict(success=[True], task_ids=[0], init_state_ids=[0], finish_steps=[2]),
        # A uniform pair: std 0 -> zero advantages, not NaN.
        dict(success=[True, True], task_ids=[0, 0], init_state_ids=[0, 0], finish_steps=[1, 2]),
        # A rollout that executed nothing has an empty mask row.
        dict(success=[True, False, False], task_ids=[1, 1, 1], init_state_ids=[2, 2, 2], finish_steps=[0, 2, 1]),
        # The full horizon is allowed.
        dict(success=[False, True], task_ids=[0, 0], init_state_ids=[5, 5], finish_steps=[4, 4]),
    ]
    for case in cases:
        batch = {k: torch.tensor(v) for k, v in case.items()}
        batch.update(tokens_per_action=3, num_chunks=2, actions_per_chunk=2)
        rewards, mask, index = upstream_inputs(batch)
        out = fn(**batch, mode="grpo", drop_uniform=False)
        torch.testing.assert_close(out["advantages"], svr_grpo_outcome_advantage(rewards, mask, index),
                                   atol=1e-6, rtol=1e-6)
        assert torch.isfinite(out["advantages"]).all()
