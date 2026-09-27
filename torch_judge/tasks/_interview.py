"""Authoring helper for per-task interview questions."""

from __future__ import annotations


def interview(*, concept: list[str], deep_dive: list[str], tradeoffs: list[str] = ()) -> list[dict]:
    """Build the ordered `interview_questions` list validated by `_schema`."""
    return [
        {"stage": stage, "question": question}
        for stage, questions in (("concept", concept), ("deep_dive", deep_dive), ("tradeoffs", tradeoffs))
        for question in questions
    ]
