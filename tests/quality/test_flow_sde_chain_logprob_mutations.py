"""Mutation gate for the flow_sde_chain_logprob exercise."""

from __future__ import annotations

import pytest

from tests.quality.mutation_runner import Mutation, assert_mutations_rejected
from torch_judge.tasks import TASKS, get_task
from torch_judge.tasks._schema import validate_task

TASK_ID = "flow_sde_chain_logprob"
REFERENCE = get_task(TASK_ID)["solution"]


def edit(*replacements: tuple[str, str]) -> str:
    """Apply exact-substring edits to the reference; each old string must occur once."""
    code = REFERENCE
    for old, new in replacements:
        assert code.count(old) == 1, f"edit target not unique: {old!r}"
        code = code.replace(old, new)
    return code


MUTATIONS = [
    # Research section 5 R8: the SDE drift correction dropped, so the mean is the ODE step.
    Mutation("sde_correction_dropped", edit((
        "noise_weight = t_next - sigma * sigma * delta / (2 * t)",
        "noise_weight = t_next",
    ))),
    # Research section 5 R8: the log-probability of the final action only, not the chain.
    Mutation("joint_scores_final_step_only", edit((
        "        logp = torch.stack(terms, dim=1).mean(dim=1)\n",
        "        logp = terms[-1]\n",
    ))),
    # Research section 5 R8: chunk_level sums over the batch axis.
    Mutation("chunk_level_sums_batch", edit((
        "        return logp.sum(dim=(1, 2))\n",
        "        return logp.sum(dim=(0, 2))\n",
    ))),
    Mutation("t1_guard_shifts_numerator_too", edit((
        "    sigmas = noise_level * torch.sqrt(ts / (1 - denom_t))\n",
        "    sigmas = noise_level * torch.sqrt(denom_t / (1 - denom_t))\n",
    ))),
    Mutation("first_step_gets_no_noise", edit((
        "    denom_t = torch.where(ts == 1, ts[1], ts)\n    sigmas = noise_level * torch.sqrt(ts / (1 - denom_t))\n",
        "    denom_t = torch.where(ts == 1, ts[1], ts)\n"
        "    sigmas = torch.where(ts == 1, torch.zeros_like(ts), noise_level * torch.sqrt(ts / (1 - denom_t)))\n",
    ))),
    Mutation("joint_sums_instead_of_mean", edit((
        "        logp = torch.stack(terms, dim=1).mean(dim=1)\n",
        "        logp = torch.stack(terms, dim=1).sum(dim=1)\n",
    ))),
    Mutation("joint_drops_initial_noise_term", edit((
        "        logp = torch.stack(terms, dim=1).mean(dim=1)\n",
        "        logp = torch.stack(terms[1:], dim=1).mean(dim=1)\n",
    ))),
    Mutation("std_missing_sqrt_delta", edit((
        "    std = (torch.sqrt(delta) * sigma).expand_as(x_t)\n",
        "    std = sigma.expand_as(x_t)\n",
    ))),
    Mutation("first_index_for_whole_batch", edit((
        "        idx = sde_index.long()\n",
        "        idx = sde_index.long()[:1].expand(batch)\n",
    ))),
    Mutation("velocity_at_next_state", edit((
        "        x = chain[rows, idx]\n        v = velocity_fn(x, ts[idx])\n",
        "        x = chain[rows, idx]\n        v = velocity_fn(chain[rows, idx + 1], ts[idx])\n",
    ))),
    Mutation("truncation_ignored", edit((
        "    logp = logp[:, :action_chunk, :action_dim]  # drop padded positions and dims\n",
        "",
    ))),
    Mutation("zero_std_clamped_not_masked", edit((
        "    zero = std == 0\n    safe = torch.where(zero, torch.ones_like(std), std)\n",
        "    zero = torch.zeros_like(std, dtype=torch.bool)\n    safe = std.clamp_min(1e-6)\n",
    ))),
    Mutation("action_level_sums_horizon", edit((
        "        return logp.sum(dim=-1)\n",
        "        return logp.sum(dim=1)\n",
    ))),
    Mutation("noise_at_zero_grid", edit((
        "    return 1.0 - k / num_steps\n",
        "    return k / num_steps\n",
    ))),
    Mutation("velocity_detached", edit((
        "        v = velocity_fn(x, ts[idx])\n        mean, std = flow_sde_mean_std(x, v, idx, num_steps, \"flow_sde\", noise_level)\n        logp",
        "        v = velocity_fn(x, ts[idx]).detach()\n        mean, std = flow_sde_mean_std(x, v, idx, num_steps, \"flow_sde\", noise_level)\n        logp",
    ))),
    Mutation("variance_used_as_std", edit((
        "    std = (torch.sqrt(delta) * sigma).expand_as(x_t)\n",
        "    std = (delta * sigma * sigma).expand_as(x_t)\n",
    ))),
    Mutation("grid_times_float32", edit((
        "    ts = _timesteps(num_steps, chain.dtype, chain.device)\n",
        "    ts = _timesteps(num_steps, torch.float32, chain.device).to(chain.dtype) if chain.dtype != torch.float64 else _timesteps(num_steps, chain.dtype, chain.device).float()\n",
    ))),
]


@pytest.mark.parametrize("_repeat", range(3))
def test_mutations_rejected(_repeat):
    rejected = assert_mutations_rejected(TASK_ID, MUTATIONS, require_unshown=True)
    assert set(rejected) == {mutation.name for mutation in MUTATIONS}


def test_mutation_names_are_unique():
    names = [mutation.name for mutation in MUTATIONS]
    assert len(names) == len(set(names)) and len(names) >= 6


def test_task_metadata_is_valid():
    task = get_task(TASK_ID)
    validate_task(TASK_ID, task, known_ids=set(TASKS))
    assert not any(key.endswith("_zh") for key in task)
    assert set(task["advisory_prerequisites"]) <= set(TASKS)
    assert len(task["interview_questions"]) == 7
    commits = {source["commit"] for source in task["sources"] if source["kind"] == "code"}
    assert commits == {"0e9438dee0088be7673a1fb4789018aea168db1a"}
