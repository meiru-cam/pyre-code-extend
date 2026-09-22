"""Validation for task metadata — legacy fields plus advanced-curriculum extensions."""

from __future__ import annotations

import re
from typing import Any

DIFFICULTIES = frozenset({"Easy", "Medium", "Hard"})
HINT_KINDS = frozenset({"questions", "analysis"})
VISIBILITIES = frozenset({"visible", "unshown"})

DESIGN_NOTE_DIMENSIONS = (
    ("api_boundaries", "API boundaries and class responsibilities"),
    ("state_ownership", "State and ownership"),
    ("failure_recovery", "Failure behavior and recovery"),
    ("backpressure_concurrency", "Backpressure and concurrency"),
    ("durability_idempotency", "Durability and idempotency"),
    ("observability", "Observability"),
    ("security", "Security"),
    ("tradeoffs", "Explicit tradeoffs"),
)

BEHAVIOR_CATEGORIES = frozenset({
    "contract.signature",
    "tensor.shape",
    "tensor.dtype_device",
    "numerics.stability",
    "edge.empty_or_boundary",
    "state.invariant",
    "gradient.flow",
    "attention.masking",
    "attention.cache",
    "routing.selection",
    "routing.normalization",
    "dispatch.token_conservation",
    "capacity.overflow",
    "experts.sparsity",
    "training.router_gradient",
    "training.load_balance",
    "protocol.validation",
    "events.ordering",
    "retry.classification",
    "retry.backoff",
    "effects.idempotency",
    "scheduler.concurrency",
    "budget.enforcement",
    "checkpoint.recovery",
    "security.permission",
    "security.injection",
    "security.redaction",
    "rl.logprob",
    "rl.advantage",
    "rl.kl_estimator",
    "rl.clipping",
    "rl.masking",
    "rl.reward_verifiable",
    "rl.rollout_assembly",
    "rl.trajectory",
})


# The description renderer (web MarkdownContent) has no KaTeX support and no
# task uses LaTeX; formulas are written in plain text or code blocks. Reject
# LaTeX/math markup so it cannot silently render as literal source.
_MATH_MARKUP = re.compile(
    r"\\\[|\\\]|\\\(|\\\)|\$\$"
    r"|\\(?:operatorname|frac|sqrt|sum|prod|int|epsilon|varepsilon|alpha|beta"
    r"|gamma|delta|sigma|lambda|theta|mu|rho|phi|psi|omega|cdot|times|div|odot"
    r"|otimes|mathrm|mathbb|mathcal|mathbf|boldsymbol|partial|nabla|langle"
    r"|rangle|leq|geq|approx|neq|equiv|infty|begin|end|left|right|hat|bar"
    r"|tilde|vec|text)\b"
)


class TaskValidationError(ValueError):
    def __init__(self, task_id: str, field: str, message: str):
        self.task_id = task_id
        self.field = field
        super().__init__(f"Task '{task_id}', field '{field}': {message}")


def _reject_math_markup(task_id: str, field: str, text: str) -> None:
    match = _MATH_MARKUP.search(text)
    if match is not None:
        raise TaskValidationError(
            task_id,
            field,
            f"unsupported LaTeX/math markup {match.group(0)!r}; the description "
            "renderer has no KaTeX, so write formulas in plain text or a code block",
        )


def build_design_note_rubric() -> list[dict[str, str]]:
    """Return isolated, frontend-ready metadata for the shared design rubric."""
    return [{"field": field, "label": label} for field, label in DESIGN_NOTE_DIMENSIONS]


def _require_str(task_id: str, task: dict, key: str) -> None:
    value = task.get(key)
    if not isinstance(value, str) or not value.strip():
        raise TaskValidationError(task_id, key, "required non-empty string")


def _validate_hints(task_id: str, hints: Any) -> None:
    if not isinstance(hints, list) or not hints:
        raise TaskValidationError(task_id, "hints", "must be a non-empty list")
    seen_levels: set[int] = set()
    for i, hint in enumerate(hints):
        if not isinstance(hint, dict):
            raise TaskValidationError(task_id, "hints", f"entry {i} must be a dict")
        level = hint.get("level")
        if level not in (1, 2):
            raise TaskValidationError(task_id, "hints", f"entry {i}: level must be 1 or 2")
        if level in seen_levels:
            raise TaskValidationError(task_id, "hints", f"duplicate level {level}")
        seen_levels.add(level)
        if hint.get("kind") not in HINT_KINDS:
            raise TaskValidationError(
                task_id, "hints", f"entry {i}: kind must be one of {sorted(HINT_KINDS)}"
            )
        if not isinstance(hint.get("content"), str) or not hint["content"].strip():
            raise TaskValidationError(task_id, "hints", f"entry {i}: content required")
        expected_kind = {1: "questions", 2: "analysis"}[level]
        if hint["kind"] != expected_kind:
            raise TaskValidationError(
                task_id, "hints", f"level {level} must use kind '{expected_kind}'"
            )
    if seen_levels != {1, 2}:
        raise TaskValidationError(task_id, "hints", "new-style hints require levels 1 and 2")


def _validate_test(task_id: str, index: int, test: Any) -> None:
    field = f"tests[{index}]"
    if not isinstance(test, dict):
        raise TaskValidationError(task_id, field, "must be a dict")
    if not isinstance(test.get("name"), str) or not test["name"].strip():
        raise TaskValidationError(task_id, field, "name required")
    if not isinstance(test.get("code"), str) or not test["code"].strip():
        raise TaskValidationError(task_id, field, "code required")
    visibility = test.get("visibility", "visible")
    if visibility not in VISIBILITIES:
        raise TaskValidationError(task_id, field, f"visibility must be one of {sorted(VISIBILITIES)}")
    behavior = test.get("behavior")
    if behavior is not None and behavior not in BEHAVIOR_CATEGORIES:
        raise TaskValidationError(task_id, field, f"unknown behavior category '{behavior}'")
    if visibility == "unshown":
        if behavior is None:
            raise TaskValidationError(task_id, field, "unshown case requires a behavior category")
        if not isinstance(test.get("failure_message"), str) or not test["failure_message"].strip():
            raise TaskValidationError(task_id, field, "unshown case requires failure_message")


def _validate_sources(task_id: str, sources: Any) -> None:
    if not isinstance(sources, list):
        raise TaskValidationError(task_id, "sources", "must be a list")
    for i, source in enumerate(sources):
        if not isinstance(source, dict):
            raise TaskValidationError(task_id, "sources", f"entry {i} must be a dict")
        kind = source.get("kind")
        if kind not in {"code", "paper"}:
            raise TaskValidationError(task_id, "sources", f"entry {i}: kind must be code or paper")
        if not isinstance(source.get("url"), str) or not source["url"].strip():
            raise TaskValidationError(task_id, "sources", f"entry {i}: url required")
        if kind == "code":
            commit = source.get("commit")
            if not isinstance(commit, str) or re.fullmatch(r"[0-9a-fA-F]{40}", commit) is None:
                raise TaskValidationError(
                    task_id, "sources", f"entry {i}: code source requires a 40-character commit hash"
                )
            for key in ("path", "symbol", "license", "adapted", "simplifications"):
                if not isinstance(source.get(key), str) or not source[key].strip():
                    raise TaskValidationError(task_id, "sources", f"entry {i}: {key} required")
        else:
            locators = ("section", "equation", "figure", "pages")
            if not any(isinstance(source.get(key), str) and source[key].strip() for key in locators):
                raise TaskValidationError(
                    task_id, "sources", f"entry {i}: paper source requires a precise locator"
                )


def validate_task(task_id: str, task: dict, known_ids: set[str] | None = None) -> None:
    """Validate one TASK dict. Raises TaskValidationError with a precise field."""
    if not isinstance(task, dict):
        raise TaskValidationError(task_id, "<root>", "TASK must be a dict")
    for key in ("title", "difficulty", "function_name", "description_en", "solution"):
        _require_str(task_id, task, key)
    if task["difficulty"] not in DIFFICULTIES:
        raise TaskValidationError(
            task_id, "difficulty", f"must be one of {sorted(DIFFICULTIES)}"
        )

    _reject_math_markup(task_id, "description_en", task["description_en"])
    if isinstance(task.get("description_zh"), str):
        _reject_math_markup(task_id, "description_zh", task["description_zh"])

    version = task.get("version", 1)
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        raise TaskValidationError(task_id, "version", "must be an integer >= 1")

    if "hint" not in task and "hints" not in task:
        raise TaskValidationError(task_id, "hints", "task needs legacy 'hint' or new 'hints'")
    if "hint" in task:
        _require_str(task_id, task, "hint")
        _reject_math_markup(task_id, "hint", task["hint"])
    if "hints" in task:
        _validate_hints(task_id, task["hints"])
        for i, hint in enumerate(task["hints"]):
            if isinstance(hint, dict) and isinstance(hint.get("content"), str):
                _reject_math_markup(task_id, f"hints[{i}]", hint["content"])

    tests = task.get("tests")
    if not isinstance(tests, list) or not tests:
        raise TaskValidationError(task_id, "tests", "required non-empty list")
    for index, test in enumerate(tests):
        _validate_test(task_id, index, test)
    if not any(test.get("visibility", "visible") == "visible" for test in tests):
        raise TaskValidationError(task_id, "tests", "requires at least one visible case")

    if "advisory_prerequisites" in task:
        prerequisites = task["advisory_prerequisites"]
        if not isinstance(prerequisites, list) or not all(isinstance(item, str) for item in prerequisites):
            raise TaskValidationError(task_id, "advisory_prerequisites", "must be a list of task ids")
        if known_ids is not None:
            unknown = sorted(set(prerequisites) - known_ids)
            if unknown:
                raise TaskValidationError(
                    task_id, "advisory_prerequisites", f"unknown task ids: {unknown}"
                )

    if "design_note_rubric" in task:
        rubric = task["design_note_rubric"]
        expected = [field for field, _label in DESIGN_NOTE_DIMENSIONS]
        valid = (
            isinstance(rubric, list)
            and [item.get("field") for item in rubric if isinstance(item, dict)] == expected
            and len(rubric) == len(expected)
            and all(
                isinstance(item, dict)
                and isinstance(item.get("label"), str)
                and bool(item["label"].strip())
                for item in rubric
            )
        )
        if not valid:
            raise TaskValidationError(
                task_id,
                "design_note_rubric",
                "must declare every structured design dimension once in canonical order",
            )

    if "pro_con_analysis" in task:
        analysis = task["pro_con_analysis"]
        valid = (
            isinstance(analysis, dict)
            and isinstance(analysis.get("pros"), list)
            and isinstance(analysis.get("cons"), list)
            and all(isinstance(item, str) for item in analysis["pros"] + analysis["cons"])
        )
        if not valid:
            raise TaskValidationError(
                task_id, "pro_con_analysis", "must be {'pros': [str], 'cons': [str]}"
            )

    if "model_connections" in task:
        connections = task["model_connections"]
        if not isinstance(connections, list) or not all(isinstance(item, str) for item in connections):
            raise TaskValidationError(task_id, "model_connections", "must be a list of strings")

    if "sources" in task:
        _validate_sources(task_id, task["sources"])
