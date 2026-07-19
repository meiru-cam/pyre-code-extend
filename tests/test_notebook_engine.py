"""Notebook engine visibility and harness-failure behavior."""

from __future__ import annotations

import pytest

from torch_judge import engine


def _task(test):
    return {
        "title": "Demo",
        "difficulty": "Medium",
        "function_name": "candidate",
        "tests": [test],
    }


def test_unshown_notebook_failure_masks_raw_exception(monkeypatch, capsys):
    monkeypatch.setattr(engine, "get_task", lambda _task_id: _task({
        "name": "adversarial values",
        "code": "raise RuntimeError('raw input: API_KEY=secret')",
        "behavior": "numerics.stability",
        "visibility": "unshown",
        "failure_message": "The implementation must remain finite for large values.",
    }))
    monkeypatch.setattr(engine, "_get_user_namespace", lambda: {"candidate": lambda: None})
    monkeypatch.setattr(engine, "mark_attempted", lambda _task_id: None)

    engine.check("demo")

    output = capsys.readouterr().out
    assert "The implementation must remain finite for large values." in output
    assert "numerics.stability" in output
    assert "API_KEY" not in output
    assert "RuntimeError" not in output


def test_notebook_harness_failure_does_not_mark_attempted(monkeypatch, capsys):
    attempted = []
    monkeypatch.setattr(engine, "get_task", lambda _task_id: _task({
        "name": "broken oracle",
        "code": "from torch_judge.harness import HarnessFailure\nraise HarnessFailure('bad fixture')",
        "behavior": "state.invariant",
    }))
    monkeypatch.setattr(engine, "_get_user_namespace", lambda: {"candidate": lambda: None})
    monkeypatch.setattr(engine, "mark_attempted", attempted.append)

    engine.check("demo")

    output = capsys.readouterr().out
    assert "Evaluator harness failure" in output
    assert "bad fixture" in output
    assert attempted == []
