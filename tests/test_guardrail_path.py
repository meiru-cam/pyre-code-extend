"""Integration contract for the separate executable guardrails path."""

import json
from pathlib import Path

from torch_judge.tasks import get_task


ROOT = Path(__file__).parent.parent


def _json(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def test_guardrail_path_is_separate_and_progressive():
    paths = {item["id"]: item for item in _json("web/src/lib/paths.json")["paths"]}
    path = paths["agent-guardrails-security"]
    assert path["problems"] == ["policy_engine", "approval_gate", "guarded_runtime"]
    assert path["prerequisites"] == ["agent-runtime-system-design"]
    assert "guardrail" in path["titleEn"].lower()


def test_guardrail_tasks_export_with_starters_hints_and_security_metadata():
    ids = {"policy_engine", "approval_gate", "guarded_runtime"}
    starters = _json("web/src/lib/starters.json")
    problems = {item["id"]: item for item in _json("web/src/lib/problems.json")["problems"]}
    solutions = _json("web/src/lib/solutions.json")
    assert ids <= starters.keys() and ids <= problems.keys() and ids <= solutions.keys()
    for task_id in ids:
        assert "pass" in starters[task_id]
        assert [hint["kind"] for hint in problems[task_id]["hints"]] == ["questions", "analysis"]
        assert len(problems[task_id]["designNoteRubric"]) == 8
        assert problems[task_id]["modelConnections"]
        assert problems[task_id]["proConAnalysis"]["pros"]
        assert problems[task_id]["proConAnalysis"]["cons"]
        assert any(test.get("visibility") == "unshown" for test in problems[task_id]["tests"])


def test_agent_runtime_path_remains_unchanged():
    paths = {item["id"]: item for item in _json("web/src/lib/paths.json")["paths"]}
    assert paths["agent-runtime-system-design"]["problems"] == [
        "tool_registry", "budgeted_agent_loop", "supervisor_orchestration"
    ]


def test_generated_guardrail_solutions_are_fresh():
    solutions = _json("web/src/lib/solutions.json")
    for task_id in ("policy_engine", "approval_gate", "guarded_runtime"):
        assert solutions[task_id]["cells"][0] == {
            "type": "code",
            "source": get_task(task_id)["solution"].strip(),
            "role": "solution",
        }
