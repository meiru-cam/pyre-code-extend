"""Mutation gate for the agent context-engineering exercises."""

from __future__ import annotations

from grading_service.main import _execute_tests
from torch_judge.tasks import get_task
from tests.quality.mutation_runner import Mutation, assert_mutations_rejected


def _mutation(task_id: str, name: str, old: str, new: str) -> Mutation:
    solution = get_task(task_id)["solution"]
    assert old in solution, f"mutation target drifted: {task_id}/{name}"
    return Mutation(name, solution.replace(old, new, 1))


def test_context_compaction_reference_contract():
    task = get_task("context_compaction")
    assert task is not None
    response = _execute_tests(task["solution"], task, capture_output=False)
    assert response.allPassed, [
        (result.name, result.error) for result in response.results if not result.passed
    ]


def test_context_compaction_rejects_every_documented_mistake():
    rejected = assert_mutations_rejected(
        "context_compaction",
        [
            _mutation(
                "context_compaction",
                "keeps the oldest blocks instead of the newest",
                "    for index in range(len(blocks) - 1, -1, -1):",
                "    for index in range(len(blocks)):",
            ),
            _mutation(
                "context_compaction",
                "does not reserve room for the summary",
                "    available = token_budget - pinned_tokens - summary_budget",
                "    available = token_budget - pinned_tokens",
            ),
            _mutation(
                "context_compaction",
                "fits message by message so a tool block can be split",
                '        if message["role"] == "tool":\n'
                '            if not blocks or not blocks[-1][0].get("tool_calls"):\n'
                '                raise ValueError("a tool message must follow the assistant turn that called it")\n'
                "            blocks[-1].append(message)\n"
                "        else:\n"
                "            blocks.append([message])",
                "        blocks.append([message])",
            ),
            _mutation(
                "context_compaction",
                "skips an oversized block instead of stopping at it",
                "            break",
                "            continue",
            ),
            _mutation(
                "context_compaction",
                "summarizes what it kept instead of what it dropped",
                "    dropped = [message for block in blocks[:start] for message in block]",
                "    dropped = [message for block in blocks[start:] for message in block]",
            ),
            _mutation(
                "context_compaction",
                "puts the summary ahead of the pinned system message",
                '"messages": [dict(message) for message in pinned] + [summary] + kept,',
                '"messages": [summary] + [dict(message) for message in pinned] + kept,',
            ),
            _mutation(
                "context_compaction",
                "hands back the caller's dictionaries by reference",
                '            "messages": [dict(message) for message in messages],',
                '            "messages": messages,',
            ),
            _mutation(
                "context_compaction",
                "accepts a summary that overruns its budget",
                "    if summary_tokens > summary_budget:\n"
                '        raise ValueError("the summary exceeded summary_budget")',
                "    if False:\n"
                '        raise ValueError("the summary exceeded summary_budget")',
            ),
            _mutation(
                "context_compaction",
                "compacts even when the conversation already fits",
                "    if total <= token_budget:",
                "    if False:",
            ),
            _mutation(
                "context_compaction",
                "accepts an orphaned tool message",
                '            if not blocks or not blocks[-1][0].get("tool_calls"):\n'
                '                raise ValueError("a tool message must follow the assistant turn that called it")\n',
                "            if not blocks:\n"
                "                blocks.append([message])\n"
                "                continue\n",
            ),
            _mutation(
                "context_compaction",
                "reports a token total it did not return",
                '        "tokens": pinned_tokens + summary_tokens + used,',
                '        "tokens": used,',
            ),
            _mutation(
                "context_compaction",
                "never pins the leading system message",
                '    pinned = messages[:1] if messages and messages[0]["role"] == "system" else []',
                "    pinned = []",
            ),
        ],
    )
    assert len(rejected) == 12
