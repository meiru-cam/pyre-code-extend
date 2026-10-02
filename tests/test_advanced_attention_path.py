"""Integration contract for the advanced attention exercises, merged into the Attention path."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).parent.parent


def _json(relative_path: str):
    return json.loads((ROOT / relative_path).read_text(encoding="utf-8"))


def test_advanced_attention_follows_the_basics_in_one_path():
    paths = {path["id"]: path for path in _json("web/src/lib/paths.json")["paths"]}
    assert "advanced-attention" not in paths
    problems = paths["attention-position"]["problems"]
    assert problems[-4:] == ["qk_norm", "hybrid_attention_schedule", "mla", "frontier_attention_block"]
    # GQA and sliding windows, which the frontier block builds on, come earlier in the same path.
    assert problems.index("gqa") < problems.index("qk_norm")
    assert problems.index("sliding_window") < problems.index("qk_norm")


def test_existing_frontier_overview_is_unchanged():
    paths = {path["id"]: path for path in _json("web/src/lib/paths.json")["paths"]}
    assert paths["llm-frontiers"]["problems"] == [
        "mha", "gqa", "diff_attention", "mla", "moe", "moe_load_balance", "multi_token_prediction"
    ]


def test_new_exercises_have_starters_and_generated_entries():
    ids = {"qk_norm", "hybrid_attention_schedule", "frontier_attention_block"}
    starters = _json("web/src/lib/starters.json")
    problems = {problem["id"]: problem for problem in _json("web/src/lib/problems.json")["problems"]}
    solutions = _json("web/src/lib/solutions.json")
    assert ids <= starters.keys()
    assert ids <= problems.keys()
    assert ids <= solutions.keys()
    for task_id in ids:
        assert "pass" in starters[task_id]
        assert len(problems[task_id]["hints"]) == 2
        assert problems[task_id]["modelConnections"]
        assert problems[task_id]["proConAnalysis"]["pros"]
