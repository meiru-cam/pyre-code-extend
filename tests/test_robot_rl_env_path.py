"""Integration contract for the Robot RL Environments path."""

from __future__ import annotations

import json
from pathlib import Path

from torch_judge.tasks import get_task

ROOT = Path(__file__).parent.parent
PATH_IDS = [
    "point_reach_env",
    "vla_action_adapter",
    "chunk_step_executor",
    "vector_autoreset",
    "episode_metrics_tracker",
    "truncation_bootstrap_gae",
    "init_state_group_advantage",
    "flow_sde_chain_logprob",
]


def _json(relative_path: str):
    return json.loads((ROOT / relative_path).read_text(encoding="utf-8"))


def test_path_runs_from_the_env_contract_to_flow_policy_likelihoods():
    paths = {path["id"]: path for path in _json("web/src/lib/paths.json")["paths"]}
    path = paths["robot-rl-env"]
    assert path["problems"] == PATH_IDS
    assert path["group"] == "training"
    assert path["prerequisites"] == ["rl-posttraining", "embodied-ai"]


def test_advisory_prerequisites_never_point_forward():
    for index, task_id in enumerate(PATH_IDS):
        later = set(PATH_IDS[index:])
        assert not later & set(get_task(task_id).get("advisory_prerequisites", [])), task_id


def test_exercises_are_english_only_and_declare_their_oracle():
    for task_id in PATH_IDS:
        task = get_task(task_id)
        assert not {"title_zh", "description_zh", "hint_zh"} & set(task), task_id
        assert any(source["kind"] == "code" for source in task["sources"]), f"{task_id} has no code source"


def test_exercises_are_exported():
    problems = {problem["id"] for problem in _json("web/src/lib/problems.json")["problems"]}
    solutions = _json("web/src/lib/solutions.json")
    starters = _json("web/src/lib/starters.json")
    for task_id in PATH_IDS:
        assert task_id in problems and task_id in solutions and task_id in starters, task_id
