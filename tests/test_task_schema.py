"""Tests for torch_judge.tasks._schema.validate_task."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from torch_judge.tasks._schema import (
    BEHAVIOR_CATEGORIES,
    TaskValidationError,
    validate_task,
)


def make_legacy_task() -> dict:
    return {
        "title": "ReLU",
        "title_zh": "ReLU",
        "difficulty": "Easy",
        "function_name": "relu",
        "hint": "clamp at zero",
        "hint_zh": "在零处截断",
        "description_en": "Implement ReLU.",
        "description_zh": "实现 ReLU。",
        "tests": [{"name": "basic", "code": "assert {fn} is not None"}],
        "solution": "def relu(x):\n    return x.clamp(min=0)",
    }


def make_new_task() -> dict:
    return {
        "title": "Top-k Expert Router",
        "difficulty": "Medium",
        "function_name": "topk_route",
        "version": 1,
        "description_en": "Implement a top-k router.",
        "advisory_prerequisites": ["softmax", "moe"],
        "hints": [
            {"level": 1, "kind": "questions", "content": "Which dim is experts?"},
            {"level": 2, "kind": "analysis", "content": "Normalize logits per token."},
        ],
        "model_connections": ["DeepSeek-V3 uses fine-grained experts."],
        "pro_con_analysis": {"pros": ["sparse compute"], "cons": ["routing instability"]},
        "sources": [
            {
                "kind": "code",
                "url": "https://github.com/deepseek-ai/DeepSeek-V3",
                "commit": "a" * 40,
                "path": "inference/model.py",
                "symbol": "DeepseekV3ForCausalLM.forward",
                "license": "MIT",
                "adapted": "Router input/output contract and top-k selection semantics.",
                "simplifications": "Single-process execution with tiny tensors.",
            }
        ],
        "tests": [
            {"name": "shape", "code": "assert True", "behavior": "tensor.shape"},
            {
                "name": "normalization",
                "code": "assert True",
                "behavior": "routing.normalization",
                "visibility": "unshown",
                "failure_message": "Expert probabilities must sum to one per token.",
            },
        ],
        "solution": "def topk_route(logits, k):\n    ...",
    }


def test_legacy_task_validates():
    validate_task("relu", make_legacy_task())


def test_new_task_validates():
    validate_task("moe_topk_router", make_new_task(), known_ids={"softmax", "moe", "moe_topk_router"})


@pytest.mark.parametrize("key", ["title", "difficulty", "function_name", "description_en", "solution"])
def test_missing_required_field(key):
    task = make_legacy_task()
    del task[key]
    with pytest.raises(TaskValidationError, match=key):
        validate_task("relu", task)


@pytest.mark.parametrize("markup", [
    r"\[ x = 1 \]",
    r"\operatorname{rmsnorm}(x)",
    r"$$ y = x^2 $$",
    r"the value \frac{a}{b} appears",
    r"\epsilon keeps it finite",
])
def test_latex_math_markup_rejected(markup):
    task = make_new_task()
    task["description_en"] = f"Implement it. {markup}"
    with pytest.raises(TaskValidationError, match="LaTeX/math markup"):
        validate_task("t", task, known_ids={"softmax", "moe"})


def test_latex_math_markup_rejected_in_hint():
    task = make_new_task()
    task["hints"][1]["content"] = r"compute \sqrt{mean(x**2)}"
    with pytest.raises(TaskValidationError, match="LaTeX/math markup"):
        validate_task("t", task, known_ids={"softmax", "moe"})


def test_plain_formula_still_valid():
    task = make_new_task()
    task["description_en"] = "Normalize: rmsnorm(x) = x * (mean(x**2, dim=-1) + eps) ** -0.5"
    validate_task("t", task, known_ids={"softmax", "moe"})


def test_bad_difficulty():
    task = make_legacy_task()
    task["difficulty"] = "medium"
    with pytest.raises(TaskValidationError, match="difficulty"):
        validate_task("relu", task)


def test_needs_hint_or_hints():
    task = make_legacy_task()
    del task["hint"]
    with pytest.raises(TaskValidationError, match="hints"):
        validate_task("relu", task)


def test_duplicate_hint_levels_rejected():
    task = make_new_task()
    task["hints"].append({"level": 1, "kind": "questions", "content": "dup"})
    with pytest.raises(TaskValidationError, match="hints"):
        validate_task("t", task)


def test_hint_level_out_of_range_rejected():
    task = make_new_task()
    task["hints"][0]["level"] = 3
    with pytest.raises(TaskValidationError, match="hints"):
        validate_task("t", task)


def test_bad_hint_kind_rejected():
    task = make_new_task()
    task["hints"][0]["kind"] = "spoiler"
    with pytest.raises(TaskValidationError, match="hints"):
        validate_task("t", task)


def test_both_hint_levels_are_required():
    task = make_new_task()
    task["hints"] = task["hints"][:1]
    with pytest.raises(TaskValidationError, match="levels 1 and 2"):
        validate_task("t", task)


def test_hint_kind_is_fixed_by_level():
    task = make_new_task()
    task["hints"][0]["kind"] = "analysis"
    with pytest.raises(TaskValidationError, match="level 1 must use kind 'questions'"):
        validate_task("t", task)


def test_version_must_be_positive_int():
    task = make_new_task()
    task["version"] = 0
    with pytest.raises(TaskValidationError, match="version"):
        validate_task("t", task)


def test_unknown_behavior_category_rejected():
    task = make_new_task()
    task["tests"][0]["behavior"] = "routing.made_up"
    with pytest.raises(TaskValidationError, match="tests"):
        validate_task("t", task)


def test_unshown_requires_behavior_and_failure_message():
    task = make_new_task()
    del task["tests"][1]["failure_message"]
    with pytest.raises(TaskValidationError, match="failure_message"):
        validate_task("t", task)


def test_task_requires_at_least_one_visible_case():
    task = make_new_task()
    for test in task["tests"]:
        test["visibility"] = "unshown"
        test["failure_message"] = "The evaluator behavior failed."

    with pytest.raises(TaskValidationError, match="at least one visible case"):
        validate_task("t", task)


def test_unknown_advisory_prerequisite_rejected():
    task = make_new_task()
    with pytest.raises(TaskValidationError, match="advisory_prerequisites"):
        validate_task("t", task, known_ids={"softmax"})


def test_source_with_path_requires_commit():
    task = make_new_task()
    del task["sources"][0]["commit"]
    with pytest.raises(TaskValidationError, match="sources"):
        validate_task("t", task)


def test_code_source_requires_full_provenance():
    task = make_new_task()
    task["sources"][0]["commit"] = "abc1234"
    with pytest.raises(TaskValidationError, match="40-character commit"):
        validate_task("t", task)


def test_paper_source_requires_precise_locator():
    task = make_new_task()
    task["sources"] = [{"kind": "paper", "url": "https://arxiv.org/abs/2412.19437"}]
    with pytest.raises(TaskValidationError, match="precise locator"):
        validate_task("t", task)


def test_design_note_rubric_requires_all_structured_dimensions():
    task = make_new_task()
    task["design_note_rubric"] = [{"field": "tradeoffs", "label": "Tradeoffs"}]
    with pytest.raises(TaskValidationError, match="design_note_rubric"):
        validate_task("t", task)


RL_BEHAVIOR_CATEGORIES = frozenset({
    "rl.logprob",
    "rl.advantage",
    "rl.kl_estimator",
    "rl.clipping",
    "rl.masking",
    "rl.reward_verifiable",
    "rl.rollout_assembly",
    "rl.trajectory",
})


def test_behavior_categories_match_spec_count():
    """The 27 categories of the advanced-curriculum spec, plus the RL path's 8."""
    assert RL_BEHAVIOR_CATEGORIES <= BEHAVIOR_CATEGORIES
    assert len(BEHAVIOR_CATEGORIES - RL_BEHAVIOR_CATEGORIES) == 27
    assert len(BEHAVIOR_CATEGORIES) == 35


def test_rl_behavior_categories_are_namespaced():
    """RL categories stay under the rl. prefix so the two sets never collide."""
    assert all(name.startswith("rl.") for name in RL_BEHAVIOR_CATEGORIES)
    assert not any(
        name.startswith("rl.") for name in BEHAVIOR_CATEGORIES - RL_BEHAVIOR_CATEGORIES
    )
