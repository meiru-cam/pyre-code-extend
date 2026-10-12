"""Integration contract for the Latent Reasoning path."""

from __future__ import annotations

import json
from pathlib import Path

from torch_judge.tasks import get_task

ROOT = Path(__file__).parent.parent
PATH_IDS = [
    "pause_token_training_mask",
    "coconut_stage_sample",
    "coconut_latent_forward",
    "recurrent_depth_unroll",
    "latent_exit_criteria",
]


def _json(relative_path: str):
    return json.loads((ROOT / relative_path).read_text(encoding="utf-8"))


def test_path_runs_from_pause_tokens_to_adaptive_exit():
    paths = {path["id"]: path for path in _json("web/src/lib/paths.json")["paths"]}
    path = paths["latent-reasoning"]
    assert path["problems"] == PATH_IDS
    assert path["group"] == "architectures"
    assert path["prerequisites"] == ["transformer-internals"]


def test_advisory_prerequisites_never_point_forward():
    for index, task_id in enumerate(PATH_IDS):
        later = set(PATH_IDS[index:])
        assert not later & set(get_task(task_id).get("advisory_prerequisites", [])), task_id


def test_exercises_are_english_only_and_declare_their_oracle():
    for task_id in PATH_IDS:
        task = get_task(task_id)
        assert not {"title_zh", "description_zh", "hint_zh"} & set(task), task_id
        assert task["sources"], f"{task_id} has no sources"
    # Pause tokens have no official code, so the paper is the only oracle and must be the only source.
    assert [source["kind"] for source in get_task("pause_token_training_mask")["sources"]] == ["paper"]


def test_exercises_are_exported():
    problems = {problem["id"] for problem in _json("web/src/lib/problems.json")["problems"]}
    solutions = _json("web/src/lib/solutions.json")
    starters = _json("web/src/lib/starters.json")
    for task_id in PATH_IDS:
        assert task_id in problems and task_id in solutions and task_id in starters, task_id
