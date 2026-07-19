"""Unit tests for behavior-aware grading results (no HTTP server needed)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from grading_service.main import _execute_tests
from torch_judge.harness import HarnessFailure

TASK = {
    "title": "Add",
    "difficulty": "Easy",
    "function_name": "add",
    "description_en": "x",
    "hint": "x",
    "tests": [
        {"name": "visible pass", "code": "assert {fn}(1, 2) == 3", "behavior": "contract.signature"},
        {
            "name": "unshown case",
            "code": "print('leak'); assert {fn}(2, 2) == 4, 'raw assertion detail'",
            "behavior": "routing.normalization",
            "visibility": "unshown",
            "failure_message": "Routing normalization failed: probabilities must sum to one.",
        },
    ],
}

GOOD = "def add(a, b):\n    return a + b"
BAD = "def add(a, b):\n    return a - b"


def test_behavior_and_visibility_on_results():
    response = _execute_tests(GOOD, TASK)
    assert response.allPassed
    assert response.results[0].behavior == "contract.signature"
    assert response.results[0].visibility == "visible"
    assert response.results[0].testIndex == 0
    assert response.results[1].visibility == "unshown"
    assert response.results[1].testIndex == 1


def test_selected_run_preserves_original_test_index():
    response = _execute_tests(GOOD, TASK, test_indices=[1])
    assert len(response.results) == 1
    assert response.results[0].testIndex == 1


def test_empty_selected_run_is_not_reported_as_passing():
    response = _execute_tests(GOOD, TASK, test_indices=[])

    assert not response.allPassed
    assert response.total == 0
    assert response.error == "No visible test cases are available to run."


def test_unshown_failure_masks_detail():
    response = _execute_tests(BAD, TASK)
    failing = response.results[1]
    assert not failing.passed
    assert failing.error == "Routing normalization failed: probabilities must sum to one."
    assert "raw assertion detail" not in (failing.error or "")
    assert failing.output is None


def test_unshown_pass_hides_output():
    response = _execute_tests(GOOD, TASK)
    assert response.results[1].output is None


def test_unshown_failure_without_message_falls_back_to_behavior():
    task = {**TASK, "tests": [dict(TASK["tests"][1])]}
    del task["tests"][0]["failure_message"]
    response = _execute_tests(BAD, task)
    assert response.results[0].error == "Behavior check failed: routing.normalization"


@pytest.mark.parametrize("capture_output", [True, False])
def test_harness_failures_are_not_converted_to_learner_failures(capture_output):
    task = {
        **TASK,
        "tests": [
            {
                "name": "broken evaluator",
                "code": "from torch_judge.harness import HarnessFailure\nraise HarnessFailure('oracle is invalid')",
                "behavior": "state.invariant",
            }
        ],
    }

    with pytest.raises(HarnessFailure, match="oracle is invalid"):
        _execute_tests(GOOD, task, capture_output=capture_output)
