"""Integration contract for the separate MoE Architecture and Training path."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).parent.parent


def _json(relative_path: str):
    return json.loads((ROOT / relative_path).read_text(encoding="utf-8"))


def test_moe_training_is_separate_and_progressive():
    paths = {path["id"]: path for path in _json("web/src/lib/paths.json")["paths"]}
    path = paths["moe-training"]
    assert path["problems"] == [
        "moe",
        "moe_load_balance",
        "moe_topk_router",
        "moe_capacity_dispatch",
        "tiny_moe_train_step",
    ]
    assert path["prerequisites"] == ["transformer-internals"]


def test_llm_frontiers_is_not_repurposed_for_the_new_path():
    paths = {path["id"]: path for path in _json("web/src/lib/paths.json")["paths"]}
    assert paths["llm-frontiers"]["problems"] == [
        "mha", "gqa", "diff_attention", "mla", "moe", "moe_load_balance", "multi_token_prediction"
    ]


def test_moe_exercises_are_exported_with_two_independent_hints():
    ids = {"moe_topk_router", "moe_capacity_dispatch", "tiny_moe_train_step"}
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
