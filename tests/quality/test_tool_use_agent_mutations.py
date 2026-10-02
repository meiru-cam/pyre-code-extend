"""Mutation gate for the multi-part agent tool-use loop exercise."""

from __future__ import annotations

import pytest

from torch_judge.tasks import get_task
from torch_judge.tasks._schema import validate_task
from tests.quality.parts_gate import assert_first_fails_in_part, assert_part_mutations_rejected

TASK_ID = "tool_use_agent"

MUTATIONS = [
    ("assistant turn dropped", 1, [("            messages.append({\"role\": \"assistant\", \"content\": response.content})  # the model's own turn first\n", "")]),
    ("results paired by name", 1, [("\"tool_use_id\": b[\"id\"]", "\"tool_use_id\": b[\"name\"]")]),
    ("only the first call answered", 1, [("[b for b in response.content if b[\"type\"] == \"tool_use\"])", "[b for b in response.content if b[\"type\"] == \"tool_use\"][:1])")]),
    ("results reversed", 1, [("for b, (text, is_error) in zip(uses, outcomes)]", "for b, (text, is_error) in zip(uses[::-1], outcomes[::-1])]")]),
    ("result not stringified", 1, [("return str(self.functions[name](**tool_input)), False", "return self.functions[name](**tool_input), False")]),
    ("system not passed", 1, [("self.client.create(system=system, messages=messages, tools=tools)", "self.client.create(system=\"\", messages=messages, tools=tools)")]),
    ("errors raise", 2, [("        except Exception as e:\n            return str(e), True", "        except ZeroDivisionError as e:\n            return str(e), True")]),
    ("errors flagged as success", 2, [("            return str(e), True", "            return str(e), False")]),
    ("unknown tool raises", 2, [("        if name not in self.functions:\n            return f\"unknown tool: {name}\", True\n", "")]),
    ("no cap", 2, [("        for calls in range(1, max_calls + 1):", "        for calls in iter(lambda: 1, 0):")]),
    ("one call too many", 2, [("for calls in range(1, max_calls + 1):", "for calls in range(1, max_calls + 2):"), ("if calls == max_calls:", "if calls == max_calls + 1:")]),
    ("tools run at the cap", 2, [("            if calls == max_calls:\n                break  # no call left to read the results, so the tools are not run\n", ""),
                                 ("        return f\"gave up after {max_calls} calls\"", "        return f\"gave up after {max_calls} calls\" if response.stop_reason == \"tool_use\" else None")]),
    ("batch keeps partial results", 3, [("            return [function(**call) for call in calls]", "            out = []\n            for call in calls:\n                try:\n                    out.append(function(**call))\n                except Exception:\n                    break\n            return out")]),
    ("batch name", 3, [("\"name\": tool[\"name\"] + \"_batch\"", "\"name\": tool[\"name\"] + \"s\"")]),
    ("batch schema not required", 3, [("                \"required\": [\"calls\"],\n", "")]),
    ("batch changes the tool", 3, [("        spec = {", "        tool[\"batched\"] = True\n        spec = {")]),
    ("tools run one at a time", 4, [("ThreadPoolExecutor(max_workers=max(1, len(uses)))", "ThreadPoolExecutor(max_workers=1)")]),
]


def test_mutations_rejected():
    assert_part_mutations_rejected(TASK_ID, MUTATIONS)


@pytest.mark.parametrize(("name", "part", "edits"), MUTATIONS, ids=[m[0] for m in MUTATIONS])
def test_each_mutation_first_fails_in_its_part(name, part, edits):
    assert_first_fails_in_part(TASK_ID, part, edits)


def test_task_metadata_is_valid():
    task = get_task(TASK_ID)
    validate_task(TASK_ID, task)
    assert not {"title_zh", "description_zh", "hint_zh"} & set(task)
