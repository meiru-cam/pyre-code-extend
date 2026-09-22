"""Every RL reference solution, checked against its upstream counterpart.

The mutation gate in `test_rl_posttraining_mutations.py` proves each evaluator is
strong enough to reject wrong implementations. It cannot prove the *contract* is
right: an evaluator and a reference solution written by one author can encode the
same misunderstanding twice and agree with each other perfectly.

These tests close that gap by comparing each reference solution against code
transcribed from the framework it was derived from, in
`torch_judge/harness/rl/vendored/`.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from torch_judge.harness.rl.vendored import (
    NANO_ADVANTAGE_EPS,
    OPENRLHF_LOG_RATIO_CLAMP,
    nano_group_advantages,
    nano_k3_kl_penalty,
    nano_log_softmax_and_gather,
    nano_pg_loss,
    nano_vineppo_token_advantages,
    openrlhf_gspo_ratio,
    openrlhf_policy_loss,
    openrlhf_value_loss,
)
from torch_judge.tasks import get_task


def reference(task_id: str):
    """Load a task's reference solution as a callable."""
    task = get_task(task_id)
    assert task is not None, f"unknown task {task_id!r}"
    namespace: dict = {}
    exec(task["solution"], namespace)
    return namespace[task["function_name"]]


def masked_batch(seed: int, batch: int = 5, length: int = 7):
    """A seeded ragged batch: at least one unmasked token per row."""
    generator = torch.Generator().manual_seed(seed)
    mask = torch.rand(batch, length, generator=generator) > 0.3
    mask[:, 0] = True
    return mask


# --------------------------------------------------------------------------
# nano-aha-moment
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_per_token_logprobs_matches_nano_log_softmax_and_gather(seed):
    fn = reference("per_token_logprobs")
    generator = torch.Generator().manual_seed(seed)
    for shape in ((3, 5, 9), (2, 1, 4), (4, 6, 17)):
        logits = torch.randn(*shape, generator=generator)
        input_ids = torch.randint(0, shape[-1], shape[:-1], generator=generator)
        assert torch.allclose(
            fn(logits, input_ids),
            nano_log_softmax_and_gather(logits, input_ids),
            atol=1e-7,
        )


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_group_relative_advantage_matches_nano_group_advantages(seed):
    fn = reference("group_relative_advantage")
    generator = torch.Generator().manual_seed(seed)
    for num_groups, group_size in ((4, 4), (3, 8), (6, 2)):
        rewards = torch.randn(num_groups * group_size, generator=generator)
        mine = fn(rewards, group_size, eps=NANO_ADVANTAGE_EPS)
        # Upstream normalizes one prompt's group at a time inside a loop.
        theirs = torch.cat(
            [
                nano_group_advantages(rewards[g * group_size : (g + 1) * group_size])
                for g in range(num_groups)
            ]
        )
        assert torch.allclose(mine, theirs, atol=1e-6), (mine - theirs).abs().max()


def test_group_relative_advantage_default_eps_matches_upstream():
    """The default epsilon is upstream's 1e-4, not an arbitrary choice."""
    fn = reference("group_relative_advantage")
    rewards = torch.tensor([0.0, 1.0, 2.0, 3.0])
    assert torch.allclose(fn(rewards, 4), nano_group_advantages(rewards), atol=1e-7)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_k3_kl_penalty_matches_nano_kl(seed):
    fn = reference("k3_kl_penalty")
    generator = torch.Generator().manual_seed(seed)
    logps = torch.randn(4, 6, generator=generator)
    ref_logps = torch.randn(4, 6, generator=generator)
    assert torch.allclose(
        fn(logps, ref_logps), nano_k3_kl_penalty(logps, ref_logps), atol=1e-6
    )


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("beta", [0.0, 0.04, 0.5])
def test_grpo_token_loss_matches_nano_pg_loss(seed, beta):
    fn = reference("grpo_token_loss")
    generator = torch.Generator().manual_seed(seed)
    batch, length = 5, 7
    logps = torch.randn(batch, length, generator=generator) * 0.5
    ref_logps = torch.randn(batch, length, generator=generator) * 0.5
    per_response = torch.randn(batch, generator=generator)
    mask = masked_batch(seed, batch, length)

    mine = fn(logps, ref_logps, per_response, mask, beta=beta)
    # Upstream takes advantages already broadcast to every token of a response.
    theirs, _ = nano_pg_loss(
        logps, ref_logps, per_response.unsqueeze(-1).expand(-1, length), mask, beta
    )
    assert torch.allclose(mine, theirs, atol=1e-5), f"{mine.item()} vs {theirs.item()}"


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_vineppo_mc_value_matches_nano_token_advantages(seed):
    fn = reference("vineppo_mc_value")
    generator = torch.Generator().manual_seed(seed)
    states = 5
    rollout_rewards = torch.rand(states, 4, generator=generator)
    final_reward = torch.rand((), generator=generator)

    values, mine = fn(rollout_rewards, final_reward)
    # Upstream works from a state boundary list and repeats each advantage over
    # that state's token span; one token per state isolates the arithmetic.
    estimates = values.tolist() + [float(final_reward)]
    theirs = nano_vineppo_token_advantages(list(range(states + 1)), estimates)
    assert torch.allclose(mine, torch.tensor(theirs, dtype=mine.dtype), atol=1e-6)


# --------------------------------------------------------------------------
# OpenRLHF
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("dual_clip", [None, 2.0, 5.0])
def test_ppo_clipped_policy_loss_matches_openrlhf(seed, dual_clip):
    fn = reference("ppo_clipped_policy_loss")
    generator = torch.Generator().manual_seed(seed)
    batch, length = 5, 7
    logps = torch.randn(batch, length, generator=generator) * 0.5
    old_logps = torch.randn(batch, length, generator=generator) * 0.5
    advantages = torch.randn(batch, length, generator=generator)
    mask = masked_batch(seed, batch, length)

    mine = fn(logps, old_logps, advantages, mask, clip_eps=0.2, dual_clip=dual_clip)
    theirs, _, _ = openrlhf_policy_loss(
        logps, old_logps, advantages, mask, 0.2, 0.2, dual_clip
    )
    assert torch.allclose(mine, theirs, atol=1e-6), f"{mine.item()} vs {theirs.item()}"


def test_ppo_clipped_policy_loss_matches_openrlhf_where_dual_clip_binds():
    """A random batch rarely triggers the dual-clip branch; force it."""
    fn = reference("ppo_clipped_policy_loss")
    mask = torch.ones(1, 1, dtype=torch.bool)
    for log_ratio, advantage in ((3.0, -2.0), (8.0, -1.5), (5.0, -0.25)):
        logps = torch.tensor([[0.0]])
        old_logps = torch.tensor([[-log_ratio]])
        advantages = torch.tensor([[advantage]])
        for dual_clip in (2.0, 3.0):
            mine = fn(logps, old_logps, advantages, mask, clip_eps=0.2, dual_clip=dual_clip)
            theirs, _, _ = openrlhf_policy_loss(
                logps, old_logps, advantages, mask, 0.2, 0.2, dual_clip
            )
            assert torch.allclose(mine, theirs, atol=1e-5), (
                f"log_ratio={log_ratio} A={advantage} dual_clip={dual_clip}: "
                f"{mine.item()} vs {theirs.item()}"
            )
            # The bound must actually be active, or this proves nothing.
            unbounded, _, _ = openrlhf_policy_loss(
                logps, old_logps, advantages, mask, 0.2, 0.2, None
            )
            assert theirs < unbounded, "dual_clip did not bind in this fixture"


def test_ppo_clipped_policy_loss_documents_the_log_ratio_clamp_difference():
    """Upstream clamps the log-ratio to +/-20 before exp; the exercise does not.

    This is a deliberate, declared difference rather than an oversight. Inside the
    band the two agree exactly; outside it they must diverge, and the exercise's
    description says why.
    """
    fn = reference("ppo_clipped_policy_loss")
    mask = torch.ones(1, 1, dtype=torch.bool)
    advantages = torch.tensor([[-1.0]])

    inside = torch.tensor([[0.0]]), torch.tensor([[-(OPENRLHF_LOG_RATIO_CLAMP - 1.0)]])
    mine, _ = fn(*inside, advantages, mask, clip_eps=0.2), None
    theirs, _, _ = openrlhf_policy_loss(*inside, advantages, mask, 0.2, 0.2, None)
    assert torch.allclose(mine, theirs, atol=1e-3), "inside the clamp band they must agree"

    outside = torch.tensor([[0.0]]), torch.tensor([[-(OPENRLHF_LOG_RATIO_CLAMP + 15.0)]])
    mine = fn(*outside, advantages, mask, clip_eps=0.2)
    clamped, _, _ = openrlhf_policy_loss(*outside, advantages, mask, 0.2, 0.2, None)
    unclamped, _, _ = openrlhf_policy_loss(
        *outside, advantages, mask, 0.2, 0.2, None, apply_log_ratio_clamp=False
    )
    assert not torch.allclose(mine, clamped), "the declared difference disappeared"
    assert torch.allclose(mine, unclamped, atol=1e-3), (
        "outside the band the exercise must match the unclamped upstream formula"
    )
    description = get_task("ppo_clipped_policy_loss")["description_en"]
    assert "clamp" in description.lower(), "the difference must be documented in the task"


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_ppo_value_loss_matches_openrlhf(seed):
    fn = reference("ppo_value_loss")
    generator = torch.Generator().manual_seed(seed)
    batch, length = 5, 7
    values = torch.randn(batch, length, generator=generator) * 2.0
    old_values = torch.randn(batch, length, generator=generator) * 2.0
    returns = torch.randn(batch, length, generator=generator) * 2.0
    mask = masked_batch(seed, batch, length)

    for clip_eps in (0.05, 0.2, 1.0):
        mine = fn(values, old_values, returns, mask, clip_eps=clip_eps)
        theirs = openrlhf_value_loss(values, old_values, returns, mask, clip_eps)
        assert torch.allclose(mine, theirs, atol=1e-6), f"clip_eps={clip_eps}"


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_gspo_sequence_ratio_matches_openrlhf(seed):
    fn = reference("gspo_sequence_ratio")
    generator = torch.Generator().manual_seed(seed)
    batch, length = 5, 7
    logps = torch.randn(batch, length, generator=generator) * 0.4
    old_logps = torch.randn(batch, length, generator=generator) * 0.4
    mask = masked_batch(seed, batch, length)
    assert torch.allclose(
        fn(logps, old_logps, mask),
        openrlhf_gspo_ratio(logps, old_logps, mask),
        atol=1e-6,
    )


# --------------------------------------------------------------------------
# Provenance hygiene
# --------------------------------------------------------------------------


VENDORED_DIR = Path(__file__).resolve().parents[2] / "torch_judge" / "harness" / "rl" / "vendored"


def test_vendored_modules_never_import_a_task_solution():
    """Match import statements, not prose — the package docstring names the rule."""
    for path in VENDORED_DIR.glob("*.py"):
        for line in path.read_text().splitlines():
            stripped = line.strip()
            if not (stripped.startswith("import ") or stripped.startswith("from ")):
                continue
            assert "torch_judge.tasks" not in stripped, (
                f"{path.name} imports task solutions: {stripped}"
            )


@pytest.mark.parametrize("module", ["openrlhf_loss.py", "nano_aha_moment.py"])
def test_vendored_modules_carry_provenance_and_license(module):
    source = (VENDORED_DIR / module).read_text()
    for marker in ("Upstream:", "License:", "WHAT WAS CHANGED"):
        assert marker in source, f"{module} is missing {marker!r}"
    assert "https://github.com/" in source, f"{module} has no upstream URL"


def test_format_reward_is_declared_as_a_divergent_contract():
    """rlvr_format_reward is not a transcription; its sources must say so."""
    sources = get_task("rlvr_format_reward")["sources"]
    code_sources = [s for s in sources if s["kind"] == "code"]
    assert code_sources, "the exercise still needs pinned provenance"
    prose = " ".join(s["adapted"] + " " + s["simplifications"] for s in code_sources)
    assert "Countdown" in prose or "differ" in prose.lower(), (
        "the sources entry must state that the tiers differ from upstream"
    )
