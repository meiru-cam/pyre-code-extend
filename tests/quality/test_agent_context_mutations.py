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


_DAG_SAME_ROUND_LOOP = '''    while True:
        ready = []
        for call in calls:
            if len(ready) >= max_parallel:
                break
            call_id = call["id"]
            if status[call_id] == "pending" and all(status[dep] == "succeeded" for dep in call["deps"]):
                ready.append(call_id)
                try:
                    results[call_id] = execute(call, {dep: results[dep] for dep in call["deps"]})
                    status[call_id] = "succeeded"
                except Exception as error:
                    errors[call_id] = error
                    status[call_id] = "failed"
                    stack = list(dependents[call_id])
                    while stack:
                        child = stack.pop()
                        if status[child] == "pending":
                            status[child] = "skipped"
                            stack.extend(dependents[child])
        if not ready:
            break
        rounds.append(ready)
'''


def _dag_same_round_mutation() -> Mutation:
    solution = get_task("tool_call_dag")["solution"]
    start = solution.index("    while True:\n")
    end = solution.index("    skipped = [")
    return Mutation(
        "updates readiness mid-round, so a chain runs in one round",
        solution[:start] + _DAG_SAME_ROUND_LOOP + "\n" + solution[end:],
    )


def test_tool_call_dag_reference_contract():
    task = get_task("tool_call_dag")
    assert task is not None
    response = _execute_tests(task["solution"], task, capture_output=False)
    assert response.allPassed, [
        (result.name, result.error) for result in response.results if not result.passed
    ]


def test_tool_call_dag_rejects_every_documented_mistake():
    task_id = "tool_call_dag"
    rejected = assert_mutations_rejected(
        task_id,
        [
            _dag_same_round_mutation(),
            _mutation(
                task_id,
                "skips only direct dependents",
                "                        stack.extend(dependents[child])",
                "                        pass",
            ),
            _mutation(
                task_id,
                "aborts every pending call on the first failure",
                "                stack = list(dependents[call_id])",
                '                stack = [c for c in by_id if status[c] == "pending"]',
            ),
            _mutation(
                task_id,
                "ignores the concurrency cap",
                "        ][:max_parallel]",
                "        ]",
            ),
            _mutation(
                task_id,
                "does not detect cycles",
                "    if emitted != len(by_id):",
                "    if False:",
            ),
            _mutation(
                task_id,
                "silently ignores an unknown dependency",
                "            if dep not in by_id:\n"
                '                raise ValueError(f"{call_id!r} depends on unknown id {dep!r}")',
                "            if dep not in by_id:\n"
                "                continue",
            ),
            _mutation(
                task_id,
                "leaks every result into dep_results",
                '            dep_results = {dep: results[dep] for dep in call["deps"]}',
                "            dep_results = dict(results)",
            ),
            _mutation(
                task_id,
                "swallows BaseException as a call failure",
                "            except Exception as error:",
                "            except BaseException as error:",
            ),
            _mutation(
                task_id,
                "copies the call before executing it",
                "                results[call_id] = execute(call, dep_results)",
                "                results[call_id] = execute(dict(call), dep_results)",
            ),
            _mutation(
                task_id,
                "reports skipped ids alphabetically instead of in input order",
                '    skipped = [call["id"] for call in calls if status[call["id"]] == "skipped"]',
                '    skipped = sorted(i for i in by_id if status[i] == "skipped")',
            ),
            _mutation(
                task_id,
                "accepts a boolean cap",
                "    if not isinstance(max_parallel, int) or isinstance(max_parallel, bool) or max_parallel < 1:",
                "    if not isinstance(max_parallel, int) or max_parallel < 1:",
            ),
            _mutation(
                task_id,
                "accepts duplicate ids",
                "        if call_id in by_id:",
                "        if False:",
            ),
        ],
    )
    assert len(rejected) == 12


def test_tool_call_stream_parser_reference_contract():
    task = get_task("tool_call_stream_parser")
    assert task is not None
    response = _execute_tests(task["solution"], task, capture_output=False)
    assert response.allPassed, [
        (result.name, result.error) for result in response.results if not result.passed
    ]


def test_tool_call_stream_parser_rejects_every_documented_mistake():
    task_id = "tool_call_stream_parser"
    rejected = assert_mutations_rejected(
        task_id,
        [
            _mutation(
                task_id,
                "releases everything, so a split tag leaks into the text",
                "            held = self._held_back(self._buffer)",
                "            held = 0",
            ),
            _mutation(
                task_id,
                "always holds back a tag's length minus one",
                "            held = self._held_back(self._buffer)",
                "            held = min(len(self._buffer), len(self._open) - 1)",
            ),
            _mutation(
                task_id,
                "holds all text until finish",
                "            held = self._held_back(self._buffer)",
                "            held = len(self._buffer)",
            ),
            _mutation(
                task_id,
                "holds back only a trailing first character",
                "        for size in range(min(len(text), len(self._open) - 1), 0, -1):\n"
                "            if self._open.startswith(text[-size:]):\n"
                "                return size\n"
                "        return 0",
                "        return 1 if text.endswith(self._open[0]) else 0",
            ),
            _mutation(
                task_id,
                "stops after the first call in a chunk",
                "                events.append(self._parse(body))\n"
                "                continue",
                "                events.append(self._parse(body))\n"
                "                break",
            ),
            _mutation(
                task_id,
                "drops malformed calls silently, as verl's HermesToolParser does",
                "                events.append(self._parse(body))",
                "                parsed = self._parse(body)\n"
                '                if parsed["type"] == "tool_call":\n'
                "                    events.append(parsed)",
            ),
            _mutation(
                task_id,
                "requires an arguments key, as verl's HermesToolParser does",
                '        arguments = payload.get("arguments", {})',
                '        arguments = payload.get("arguments")',
            ),
            _mutation(
                task_id,
                "ends a call at the last close tag instead of the first",
                "                end = self._buffer.find(self._close)",
                "                end = self._buffer.rfind(self._close)",
            ),
            _mutation(
                task_id,
                "flushes an unterminated call as text",
                "        if self._inside:\n"
                '            events.append({"type": "error", "raw": self._buffer, "reason": "unterminated"})\n'
                "        elif self._buffer:",
                "        if self._buffer:",
            ),
            _mutation(
                task_id,
                "loses held-back text at finish",
                "        elif self._buffer:",
                "        elif False:",
            ),
            _mutation(
                task_id,
                "reports invalid JSON as not_a_call",
                '            return {"type": "error", "raw": body, "reason": "invalid_json"}',
                '            return {"type": "error", "raw": body, "reason": "not_a_call"}',
            ),
            _mutation(
                task_id,
                "emits empty text events",
                "            if release:",
                "            if True:",
            ),
            _mutation(
                task_id,
                "accepts input after finish",
                "        if self._finished:\n"
                '            raise ValueError("feed called after finish")',
                "        if False:\n"
                '            raise ValueError("feed called after finish")',
            ),
            _mutation(
                task_id,
                "accepts identical open and close tags",
                "        if open_tag == close_tag:",
                "        if False:",
            ),
        ],
    )
    assert len(rejected) == 14
