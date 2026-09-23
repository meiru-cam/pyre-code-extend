"""Vendored agent-loop finalization oracle transcribed from verl.

Upstream:   https://github.com/volcengine/verl
Commit:     12ebe0cb4d300c58449fb6c675379e8700015c51
Files:      verl/experimental/agent_loop/tool_agent_loop.py,
            verl/experimental/agent_loop/agent_loop.py
Symbols:    ToolAgentLoop.run, AgentLoopOutput.as_dict
Retrieved:  2026-09-23
License:    Apache License 2.0, Copyright 2024 Bytedance Ltd. and affiliates

WHAT WAS CHANGED, AND WHY
-------------------------
The asynchronous agent state machine, tokenizer, generation server, multimodal
payloads and Pydantic output model are removed. The retained pure operation is
the shared prefix slice for response ids, response mask and rollout
log-probabilities, followed by terminal trajectory-reward placement.

The exercise's overlong penalty is intentionally absent: verl performs length
shaping elsewhere. Oracle comparisons pass zero penalty so this function checks
only the behavior directly transcribed from the pinned agent-loop symbols.

NOTHING HERE IMPORTS A TASK SOLUTION.
"""

from __future__ import annotations

import torch


def verl_finalize_agent_rollout(
    response_ids: torch.Tensor,
    response_mask: torch.Tensor,
    rollout_log_probs: torch.Tensor,
    reward: float,
    max_response_length: int,
) -> dict[str, torch.Tensor]:
    kept_response_ids = response_ids[:max_response_length]
    kept_response_mask = response_mask[:max_response_length]
    kept_rollout_log_probs = rollout_log_probs[:max_response_length]

    token_rewards = torch.zeros_like(kept_rollout_log_probs)
    token_rewards[-1] = reward
    return {
        "response_ids": kept_response_ids,
        "response_mask": kept_response_mask,
        "rollout_log_probs": kept_rollout_log_probs,
        "token_rewards": token_rewards,
    }
