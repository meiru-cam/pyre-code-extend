"""Public RL-harness API used by authored evaluator cases."""

from torch_judge.harness import HarnessFailure
from torch_judge.harness.rl.scripted_rollout import (
    TRAINABLE_ROLES,
    assistant_turn,
    final_turn,
    reference_agentic_rollout,
    scripted_agent,
    scripted_tool,
    trajectory_reward_from_marker,
)
from torch_judge.harness.rl.tiny_policy import (
    TinyPolicy,
    assert_parameters_changed,
    assert_parameters_unchanged,
    drifted_reference,
    frozen_copy,
    parameter_snapshot,
    reference_grpo_step,
    seeded_rollout_batch,
)

__all__ = [
    "HarnessFailure",
    "TRAINABLE_ROLES",
    "TinyPolicy",
    "assistant_turn",
    "final_turn",
    "reference_agentic_rollout",
    "scripted_agent",
    "scripted_tool",
    "trajectory_reward_from_marker",
    "assert_parameters_changed",
    "assert_parameters_unchanged",
    "drifted_reference",
    "frozen_copy",
    "parameter_snapshot",
    "reference_grpo_step",
    "seeded_rollout_batch",
]
