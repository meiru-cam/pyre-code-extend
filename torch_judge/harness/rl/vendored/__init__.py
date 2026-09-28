"""Ground-truth oracles transcribed from upstream RL training frameworks.

Each module here carries its upstream URL, pinned commit, symbol list and
license in its docstring, plus an explicit account of what was changed and why.
The arithmetic is transcribed line for line; only infrastructure that cannot run
in the offline grading sandbox is removed.

These exist so an exercise is not graded solely against an oracle written by the
same author as its reference solution. A transcription error is still possible,
so `tests/quality/test_rl_vendored_oracles.py` compares every reference solution
against its upstream counterpart directly.

No module in this package may import `torch_judge.tasks`.

Licenses of vendored material:
  * openrlhf_loss.py    — Apache License 2.0, the OpenRLHF authors
  * nano_aha_moment.py  — MIT License, Copyright (c) 2025 McGill NLP
  * verl_agent_loop.py  — Apache License 2.0, Bytedance Ltd. and affiliates
  * verl_core_algos.py  — Apache License 2.0, Bytedance Ltd. and affiliates
  * trl_preference.py   — Apache License 2.0, the HuggingFace Team
"""

from torch_judge.harness.rl.vendored.nano_aha_moment import (
    NANO_ADVANTAGE_EPS,
    nano_group_advantages,
    nano_k3_kl_penalty,
    nano_log_softmax_and_gather,
    nano_pg_loss,
    nano_vineppo_token_advantages,
)
from torch_judge.harness.rl.vendored.openrlhf_loss import (
    OPENRLHF_LOG_RATIO_CLAMP,
    aggregate_loss,
    masked_mean,
    openrlhf_gspo_ratio,
    openrlhf_policy_loss,
    openrlhf_value_loss,
)
from torch_judge.harness.rl.vendored.trl_preference import trl_ipo_loss, trl_simpo_loss
from torch_judge.harness.rl.vendored.verl_agent_loop import verl_finalize_agent_rollout
from torch_judge.harness.rl.vendored.verl_core_algos import (
    VERL_IS_SAFETY_BOUND,
    verl_agg_loss,
    verl_rloo_outcome_advantage,
    verl_rollout_is_weights,
)

__all__ = [
    "NANO_ADVANTAGE_EPS",
    "OPENRLHF_LOG_RATIO_CLAMP",
    "aggregate_loss",
    "masked_mean",
    "nano_group_advantages",
    "nano_k3_kl_penalty",
    "nano_log_softmax_and_gather",
    "nano_pg_loss",
    "nano_vineppo_token_advantages",
    "openrlhf_gspo_ratio",
    "openrlhf_policy_loss",
    "openrlhf_value_loss",
    "trl_ipo_loss",
    "trl_simpo_loss",
    "VERL_IS_SAFETY_BOUND",
    "verl_agg_loss",
    "verl_finalize_agent_rollout",
    "verl_rloo_outcome_advantage",
    "verl_rollout_is_weights",
]
