"""Notebook hint() must support explicit levels while keeping legacy behavior."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from torch_judge import engine
from torch_judge.tasks import TASKS

NEW_TASK = {
    "title": "Top-k Router",
    "difficulty": "Medium",
    "function_name": "topk_route",
    "description_en": "x",
    "hints": [
        {"level": 1, "kind": "questions", "content": "Which dim is experts?"},
        {"level": 2, "kind": "analysis", "content": "Normalize logits per token."},
    ],
    "tests": [{"name": "t", "code": "assert True"}],
}


def test_legacy_hint_unchanged(capsys, monkeypatch):
    monkeypatch.setitem(TASKS, "_legacy", {"title": "L", "hint": "old style hint"})
    engine.hint("_legacy")
    assert "old style hint" in capsys.readouterr().out


def test_default_level_is_one(capsys, monkeypatch):
    monkeypatch.setitem(TASKS, "_new", NEW_TASK)
    engine.hint("_new")
    output = capsys.readouterr().out
    assert "Which dim is experts?" in output
    assert "Normalize logits" not in output


def test_explicit_level_two(capsys, monkeypatch):
    monkeypatch.setitem(TASKS, "_new", NEW_TASK)
    engine.hint("_new", level=2)
    output = capsys.readouterr().out
    assert "Normalize logits per token." in output
    assert "Which dim is experts?" not in output


def test_missing_level_reports_available(capsys, monkeypatch):
    monkeypatch.setitem(TASKS, "_new", NEW_TASK)
    engine.hint("_new", level=3)
    output = capsys.readouterr().out
    assert "3" in output and "1, 2" in output
