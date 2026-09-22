"""Deterministic fixtures for multi-turn agentic rollouts.

Builds on ``harness/agents/scripted_model.py`` and ``harness/agents/fake_tools.py``
rather than re-implementing them: the model and the tools are already scripted and
already refuse to run past their script. What this module adds is the RL side —
which messages carry gradient, how a trajectory is scored, and an independent
oracle for the whole loop.

Nothing here imports a task solution.
"""

from __future__ import annotations

from typing import Any, Callable

from torch_judge.harness import HarnessFailure
from torch_judge.harness.agents.fake_tools import FakeTool
from torch_judge.harness.agents.scripted_model import ScriptedModel

__all__ = [
    "TRAINABLE_ROLES",
    "assistant_turn",
    "final_turn",
    "reference_agentic_rollout",
    "scripted_agent",
    "scripted_tool",
    "trajectory_reward_from_marker",
]

# Only tokens the policy produced can carry a gradient. Tool output is text the
# environment wrote; training on it teaches the model to predict its own inputs.
TRAINABLE_ROLES = frozenset({"assistant"})


def assistant_turn(content: str, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """A scripted assistant message that calls one tool and expects to continue."""
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": [{"name": tool, "arguments": dict(arguments)}],
    }


def final_turn(content: str) -> dict[str, Any]:
    """A scripted assistant message with no tool call, which ends the rollout."""
    return {"role": "assistant", "content": content}


def scripted_agent(turns: list[dict[str, Any]]) -> ScriptedModel:
    """A ScriptedModel that emits the given turns in order."""
    return ScriptedModel(turns)


def scripted_tool(name: str, results: list[Any]) -> FakeTool:
    """A FakeTool returning the given results in order."""
    return FakeTool(
        name,
        f"scripted {name}",
        {"query": str},
        outcomes=results,
    )


def trajectory_reward_from_marker(marker: str) -> Callable[[list[dict[str, Any]]], float]:
    """Score 1.0 when the final assistant message contains ``marker``, else 0.0.

    A deliberately trivial verifiable reward: the exercise is about trajectory
    assembly, not about reward design.
    """

    def score(messages: list[dict[str, Any]]) -> float:
        for message in reversed(messages):
            if message.get("role") == "assistant":
                return 1.0 if marker in str(message.get("content", "")) else 0.0
        return 0.0

    return score


def reference_agentic_rollout(
    model: ScriptedModel,
    tools: dict[str, FakeTool],
    initial_messages: list[dict[str, Any]],
    max_turns: int,
    reward_fn: Callable[[list[dict[str, Any]]], float],
) -> dict[str, Any]:
    """An independent implementation of the rollout loop, used only as an oracle.

    Written from the contract rather than from the reference solution, so a
    learner's loop can be compared against a trajectory built outside their code.
    """
    if max_turns < 1:
        raise HarnessFailure(f"max_turns must be at least 1, got {max_turns}")

    messages = [dict(message) for message in initial_messages]
    trainable = [message.get("role") in TRAINABLE_ROLES for message in messages]
    turns = 0
    tool_calls = 0
    stop_reason = "max_turns"

    while turns < max_turns:
        response = model.call(messages)
        turns += 1

        assistant = {"role": "assistant", "content": response.get("content", "")}
        messages.append(assistant)
        trainable.append(True)

        requested = response.get("tool_calls") or []
        if not requested:
            stop_reason = "finished"
            break

        for call in requested:
            name = call["name"]
            if name not in tools:
                raise HarnessFailure(f"scripted turn called unknown tool {name!r}")
            result = tools[name].invoke(call.get("arguments", {}))
            tool_calls += 1
            messages.append({"role": "tool", "name": name, "content": result})
            trainable.append(False)

    return {
        "messages": messages,
        "trainable": trainable,
        "turns": turns,
        "tool_calls": tool_calls,
        "stop_reason": stop_reason,
        "reward": float(reward_fn(messages)),
    }
