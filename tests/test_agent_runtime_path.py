"""Integration contract for the separate Agent Runtime and System Design path."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).parent.parent


def _json(relative_path: str):
    return json.loads((ROOT / relative_path).read_text(encoding="utf-8"))


def test_agent_runtime_path_is_separate_and_progressive():
    paths = {path["id"]: path for path in _json("web/src/lib/paths.json")["paths"]}
    path = paths["agent-runtime-system-design"]
    assert path["problems"] == [
        "tool_registry",
        "budgeted_agent_loop",
        "supervisor_orchestration",
    ]
    assert path["prerequisites"] == []
    assert "system" in path["titleEn"].lower()


def test_legacy_alignment_path_is_not_repurposed():
    paths = {path["id"]: path for path in _json("web/src/lib/paths.json")["paths"]}
    assert paths["alignment-agents"]["problems"] == [
        "cross_entropy", "reward_model", "dpo_loss", "grpo_loss", "ppo_loss", "mcts_search"
    ]


def test_agent_exercises_export_with_two_independent_hints_and_design_tradeoffs():
    ids = {"tool_registry", "budgeted_agent_loop", "supervisor_orchestration"}
    starters = _json("web/src/lib/starters.json")
    problems = {problem["id"]: problem for problem in _json("web/src/lib/problems.json")["problems"]}
    solutions = _json("web/src/lib/solutions.json")
    assert ids <= starters.keys() and ids <= problems.keys() and ids <= solutions.keys()
    for task_id in ids:
        assert "pass" in starters[task_id]
        assert [hint["kind"] for hint in problems[task_id]["hints"]] == ["questions", "analysis"]
        assert problems[task_id]["modelConnections"]
        assert problems[task_id]["proConAnalysis"]["pros"]
        assert problems[task_id]["proConAnalysis"]["cons"]
        assert [item["field"] for item in problems[task_id]["designNoteRubric"]] == [
            "api_boundaries",
            "state_ownership",
            "failure_recovery",
            "backpressure_concurrency",
            "durability_idempotency",
            "observability",
            "security",
            "tradeoffs",
        ]
        assert all(item["label"] for item in problems[task_id]["designNoteRubric"])
        assert any(source["kind"] == "code" for source in problems[task_id]["sources"])
