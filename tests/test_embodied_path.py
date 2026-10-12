"""Integration contract for the Embodied AI path (VLA module)."""

from __future__ import annotations

import json
from pathlib import Path

from torch_judge.tasks import get_task

ROOT = Path(__file__).parent.parent
VLA_IDS = [
    "vla_action_bins",
    "fast_dct_action_tokens",
    "act_temporal_ensemble",
    "pi0_flow_matching_chunk",
    "rtc_prefix_guidance",
]


def _json(relative_path: str):
    return json.loads((ROOT / relative_path).read_text(encoding="utf-8"))


def test_vla_module_runs_from_tokens_to_real_time_chunking():
    paths = {path["id"]: path for path in _json("web/src/lib/paths.json")["paths"]}
    path = paths["embodied-ai"]
    assert path["problems"] == VLA_IDS
    assert path["group"] == "architectures"
    assert path["prerequisites"] == ["vision-transformer"]


def test_advisory_prerequisites_never_point_forward():
    for index, task_id in enumerate(VLA_IDS):
        later = set(VLA_IDS[index:])
        assert not later & set(get_task(task_id).get("advisory_prerequisites", [])), task_id


def test_vla_exercises_are_english_only_with_pinned_code_sources():
    for task_id in VLA_IDS:
        task = get_task(task_id)
        assert not {"title_zh", "description_zh", "hint_zh"} & set(task), task_id
        code = [source for source in task["sources"] if source["kind"] == "code"]
        assert code, f"{task_id} has no pinned code source"


def test_vla_exercises_are_exported():
    problems = {problem["id"] for problem in _json("web/src/lib/problems.json")["problems"]}
    solutions = _json("web/src/lib/solutions.json")
    starters = _json("web/src/lib/starters.json")
    for task_id in VLA_IDS:
        assert task_id in problems and task_id in solutions and task_id in starters, task_id
