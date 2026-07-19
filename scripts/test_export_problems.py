"""Tests for export_problems.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from export_problems import OUTPUT, build_problem_catalog
from torch_judge.tasks import TASKS


def test_build_problem_catalog_matches_committed_json():
    expected = build_problem_catalog()
    actual = json.loads(OUTPUT.read_text(encoding="utf-8"))
    assert actual == expected


def test_build_problem_catalog_includes_all_registered_tasks():
    data = build_problem_catalog()
    exported_ids = [problem["id"] for problem in data["problems"]]
    assert exported_ids == list(dict.fromkeys(exported_ids))
    assert set(exported_ids) == set(TASKS)
    assert len(exported_ids) == len(TASKS)


def test_exported_problem_shape():
    problem = build_problem_catalog()["problems"][0]
    required = {
        "id", "title", "titleZh", "difficulty", "functionName",
        "hint", "hintZh", "descriptionEn", "descriptionZh", "tests", "version",
    }
    assert required <= set(problem)
    assert isinstance(problem["version"], int) and problem["version"] >= 1


def test_build_problem_catalog_recovers_from_malformed_existing_json(tmp_path):
    output = tmp_path / "problems.json"
    output.write_text("<<<<<<< HEAD\nnot json\n", encoding="utf-8")

    data = build_problem_catalog(output)

    exported_ids = [problem["id"] for problem in data["problems"]]
    assert set(exported_ids) == set(TASKS)
    assert len(exported_ids) == len(TASKS)


def test_visible_test_entries_keep_code():
    for problem in build_problem_catalog()["problems"]:
        for test in problem["tests"]:
            if test.get("visibility") != "unshown":
                assert isinstance(test["code"], str) and test["code"].strip()


def test_unshown_test_entries_have_no_code():
    for problem in build_problem_catalog()["problems"]:
        for test in problem["tests"]:
            if test.get("visibility") == "unshown":
                assert "code" not in test
                assert "failure_message" not in test
                assert test["behavior"]
