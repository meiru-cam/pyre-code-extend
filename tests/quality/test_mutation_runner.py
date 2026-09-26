"""Mutation-runner contract tests."""

from __future__ import annotations

import pytest

from tests.quality.mutation_runner import Mutation, assert_mutations_rejected


def test_mutation_runner_requires_reference_to_pass(monkeypatch):
    monkeypatch.setattr(
        "tests.quality.mutation_runner.get_task",
        lambda _task_id: {
            "function_name": "identity",
            "solution": "def identity(x):\n    return x + 1",
            "tests": [{"name": "identity", "code": "assert {fn}(2) == 2"}],
        },
    )
    with pytest.raises(AssertionError, match="reference solution"):
        assert_mutations_rejected("fake", [])


def test_mutation_runner_reports_surviving_mutation(monkeypatch):
    monkeypatch.setattr(
        "tests.quality.mutation_runner.get_task",
        lambda _task_id: {
            "function_name": "identity",
            "solution": "def identity(x):\n    return x",
            "tests": [{"name": "identity", "behavior": "state.invariant", "code": "assert {fn}(2) == 2"}],
        },
    )
    with pytest.raises(AssertionError, match="survived.*same"):
        assert_mutations_rejected("fake", [Mutation("same", "def identity(x):\n    return x")])


def test_mutation_runner_returns_failed_behaviors(monkeypatch):
    monkeypatch.setattr(
        "tests.quality.mutation_runner.get_task",
        lambda _task_id: {
            "function_name": "identity",
            "solution": "def identity(x):\n    return x",
            "tests": [
                {"name": "identity", "behavior": "state.invariant", "code": "assert {fn}(2) == 2"},
                {"name": "boundary", "behavior": "edge.empty_or_boundary", "code": "assert {fn}(0) == 0"},
            ],
        },
    )
    result = assert_mutations_rejected(
        "fake",
        [Mutation("adds_one", "def identity(x):\n    return x + 1")],
    )
    assert result == {"adds_one": {"state.invariant", "edge.empty_or_boundary"}}


def test_mutation_runner_can_use_a_seeded_task_variant(monkeypatch):
    monkeypatch.setattr("tests.quality.mutation_runner.get_task", lambda _task_id: None)
    variant = {
        "function_name": "identity",
        "solution": "def identity(x):\n    return x",
        "tests": [
            {"name": "unshown nonzero", "visibility": "unshown", "behavior": "state.invariant",
             "failure_message": "wrong result", "code": "assert {fn}(7) == 7"},
        ],
    }
    result = assert_mutations_rejected(
        "fake", [Mutation("zeroes", "def identity(x):\n    return 0")],
        require_unshown=True, task_override=variant,
    )
    assert result == {"zeroes": {"state.invariant"}}
