"""Contracts for versioned, paired offline guardrail fixtures."""

from pathlib import Path

import pytest

from torch_judge.harness import HarnessFailure
from torch_judge.harness.agents import evaluate_guardrail_case, load_guardrail_fixtures


FIXTURES = Path(__file__).parent / "fixtures" / "guardrails"


def test_guardrail_fixture_corpus_is_complete_paired_and_isolated():
    pairs = load_guardrail_fixtures(FIXTURES)
    assert len(pairs) == 8
    assert {pair.primary.fixture_class for pair in pairs} == {
        "legitimate_request",
        "direct_injection",
        "indirect_tool_output_injection",
        "secret_pii_egress",
        "privilege_escalation",
        "poisoned_memory",
        "recursive_delegation",
        "cancellation",
    }
    assert all(pair.version == 1 for pair in pairs)
    assert all(pair.legitimate.fixture_class == "legitimate_request" for pair in pairs)
    assert all(pair.legitimate.expected_action == "allow" for pair in pairs)
    ids = [case.case_id for pair in pairs for case in (pair.primary, pair.legitimate)]
    assert len(ids) == len(set(ids))
    pairs[0].primary.payload["mutated"] = True
    assert "mutated" not in load_guardrail_fixtures(FIXTURES)[0].primary.payload


def test_every_adversarial_pair_executes_with_expected_false_positive_behavior():
    pairs = load_guardrail_fixtures(FIXTURES)
    for pair in pairs:
        assert evaluate_guardrail_case(pair.primary) == pair.primary.expected_action
        assert evaluate_guardrail_case(pair.legitimate) == "allow"


@pytest.mark.parametrize(
    "document,match",
    [
        (
            {"version": 1, "fixture_class": "direct_injection", "primary": {
                "id": "bad", "expected_action": "deny", "provenance": {"source": "user"},
                "payload": {},
            }, "legitimate": {"id": "ok", "expected_action": "allow", "provenance": {
                "source": "user", "trust": "trusted"}, "payload": {}}},
            "provenance",
        ),
        (
            {"version": 1, "fixture_class": "direct_injection", "primary": {
                "id": "bad", "expected_action": ["allow", "deny"], "provenance": {
                    "source": "user", "trust": "untrusted"}, "payload": {}},
             "legitimate": {"id": "ok", "expected_action": "allow", "provenance": {
                 "source": "user", "trust": "trusted"}, "payload": {}}},
            "expected_action",
        ),
    ],
)
def test_invalid_guardrail_fixture_is_a_harness_failure(tmp_path, document, match):
    import json

    (tmp_path / "case.json").write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(HarnessFailure, match=match):
        load_guardrail_fixtures(tmp_path)
