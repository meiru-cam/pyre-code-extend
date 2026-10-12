"""Integration contract for the Embodied AI II (frontier VLAs and generative world models) path."""

from __future__ import annotations

import json
from pathlib import Path

from torch_judge.tasks import get_task

ROOT = Path(__file__).parent.parent
PATH_IDS = [
    "oft_parallel_action_decoding",
    "latent_action_quantizer",
    "knowledge_insulation_attention",
    "recap_advantage_conditioning",
    "fsq_quantizer",
    "diffusion_forcing_schedule",
    "shortcut_forcing_loss",
    "self_forcing_rollout",
]


def _json(relative_path: str):
    return json.loads((ROOT / relative_path).read_text(encoding="utf-8"))


def test_path_follows_embodied_ai_with_vlas_then_world_models():
    paths = {path["id"]: path for path in _json("web/src/lib/paths.json")["paths"]}
    path = paths["embodied-frontier"]
    assert path["problems"] == PATH_IDS
    assert path["group"] == "architectures"
    assert path["prerequisites"] == ["embodied-ai"]


def test_advisory_prerequisites_never_point_forward():
    for index, task_id in enumerate(PATH_IDS):
        later = set(PATH_IDS[index:])
        assert not later & set(get_task(task_id).get("advisory_prerequisites", [])), task_id


def test_exercises_are_english_only_and_declare_their_oracle():
    for task_id in PATH_IDS:
        task = get_task(task_id)
        assert not {"title_zh", "description_zh", "hint_zh"} & set(task), task_id
        assert task["sources"], f"{task_id} has no sources"
    # pi*0.6 has no official code, so the paper is the only oracle and must be the only source.
    assert {source["kind"] for source in get_task("recap_advantage_conditioning")["sources"]} == {"paper"}


def test_exercises_are_exported():
    problems = {problem["id"] for problem in _json("web/src/lib/problems.json")["problems"]}
    solutions = _json("web/src/lib/solutions.json")
    starters = _json("web/src/lib/starters.json")
    for task_id in PATH_IDS:
        assert task_id in problems and task_id in solutions and task_id in starters, task_id
