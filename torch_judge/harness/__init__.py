"""Reusable deterministic evaluation harnesses."""


class HarnessFailure(RuntimeError):
    """Raised when an evaluator or fixture is invalid rather than learner code."""
