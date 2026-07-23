"""Reference, metadata, and repeated mutation gates for guardrail exercises."""

import pytest

from grading_service.main import _execute_tests
from torch_judge.tasks import get_task
from torch_judge.tasks._schema import validate_task
from tests.quality.mutation_runner import Mutation, assert_mutations_rejected


IDS = ("policy_engine", "approval_gate", "guarded_runtime")


def _mutation(task_id, name, old, new):
    solution = get_task(task_id)["solution"]
    assert old in solution, f"mutation target drifted: {task_id}/{name}"
    return Mutation(name, solution.replace(old, new))


@pytest.mark.parametrize("task_id", IDS)
def test_guardrail_reference_contract(task_id):
    task = get_task(task_id)
    result = _execute_tests(task["solution"], task, capture_output=False)
    assert result.allPassed, [(item.name, item.error) for item in result.results if not item.passed]


def test_guardrail_metadata_and_sources():
    combined = []
    for task_id in IDS:
        task = get_task(task_id)
        validate_task(task_id, task)
        assert task["version"] == 1
        assert [hint["kind"] for hint in task["hints"]] == ["questions", "analysis"]
        assert len(task["design_note_rubric"]) == 8
        assert any(test.get("visibility") == "unshown" for test in task["tests"])
        assert task["model_connections"] and task["pro_con_analysis"]["pros"]
        assert task["pro_con_analysis"]["cons"]
        assert all(len(source["commit"]) == 40 for source in task["sources"] if source["kind"] == "code")
        combined.extend(source["url"] for source in task["sources"])
    text = " ".join(combined)
    for source in ("openai/openai-agents-python", "NVIDIA/NeMo-Guardrails", "meta-llama/PurpleLlama", "openclaw/openclaw", "NousResearch/hermes-agent", "owasp"):
        assert source.lower() in text.lower()


def _policy_mutations():
    return [
        _mutation("policy_engine", "reverses_policy_order", "self._policies = tuple(policies)", "self._policies = tuple(reversed(policies))"),
        _mutation("policy_engine", "first_policy_wins", "for policy in self._policies:", "for policy in self._policies[:1]:"),
        _mutation("policy_engine", "unordered_set", "self._policies = tuple(policies)", "self._policies = tuple(sorted(set(policies), key=lambda policy: policy.name))"),
        _mutation("policy_engine", "allow_overrides_deny", 'if action == "deny":', "if False:"),
        _mutation("policy_engine", "transform_after_deny", '''            if action == "deny":
                return PolicyDecision(
                    "deny", working, [f"{policy.name}: {reason or 'denied'}"]
                )
''', '''            if action == "deny":
                transformed = True
                reasons.append(f"{policy.name}: {reason or 'denied'}")
                continue
'''),
        _mutation("policy_engine", "fail_open_exception", "except Exception as error:", "except () as error:"),
        _mutation("policy_engine", "does_not_compose_transforms", "working.update(changes)", "pass"),
        _mutation("policy_engine", "shares_policy_context", "policy.evaluate(deepcopy(working))", "policy.evaluate(working)"),
        _mutation("policy_engine", "swallows_harness_failure", '''            except HarnessFailure:
                raise
            except Exception as error:
''', '''            except Exception as error:
'''),
    ]


def _approval_mutations():
    return [
        _mutation("approval_gate", "approve_by_tool_name_only", 'bound_action = {key: value for key, value in action.items() if key != "approval_token"}', 'bound_action = {"tool": action["tool"]}'),
        _mutation("approval_gate", "ignore_arguments", 'bound_action = {key: value for key, value in action.items() if key != "approval_token"}', 'bound_action = {key: value for key, value in action.items() if key not in {"approval_token", "arguments"}}'),
        _mutation("approval_gate", "reusable_token", 'grant["consumed"] = True', "pass"),
        _mutation("approval_gate", "no_expiry", 'if self._clock.now() >= grant["expires_at"]:', "if False:"),
        _mutation("approval_gate", "privilege_inheritance", 'if copied["capability"] not in capabilities:', 'if not any(copied["capability"].startswith(grant) for grant in capabilities):'),
        _mutation("approval_gate", "trusts_untrusted_provenance", 'if source["trust"] != "trusted":', "if False:"),
        _mutation("approval_gate", "ignores_identity_mismatch", 'if source["identity"] != copied["actor"]:', "if False:"),
        _mutation("approval_gate", "trusts_mutated_request", '"digest": stored["digest"],', '"digest": request.digest,'),
        _mutation("approval_gate", "multiple_grants_per_request", 'if stored["approved"]:', "if False:"),
    ]


def _runtime_mutations():
    return [
        _mutation("guarded_runtime", "sanitize_input_only", "checked, failure = apply_policy(context)", "checked, failure = context, None"),
        _mutation("guarded_runtime", "redact_after_audit", '"data": sanitized,', '"data": data,'),
        _mutation("guarded_runtime", "audit_secrets", 'sanitized = redactor.redact(copy_json(data, field="audit data JSON"))', 'sanitized = copy_json(data, field="audit data JSON")'),
        _mutation("guarded_runtime", "uncapped_delegation", "if depth > max_depth:", "if False:"),
        _mutation("guarded_runtime", "drops_provenance", 'context = {"kind": event["kind"], "provenance": event_provenance,', 'context = {"kind": event["kind"], "provenance": {"source":"lost","trust":"trusted"},'),
        _mutation("guarded_runtime", "approval_request_bypass", 'if outcome_name == "ApprovalRequest":', "if False:"),
        _mutation("guarded_runtime", "cancel_without_propagation", "runtime.cancel()", "pass"),
        _mutation("guarded_runtime", "no_step_budget", "if steps > max_steps:", "if False:"),
        _mutation("guarded_runtime", "audit_failure_as_success", 'return GuardedResult("failed", failure={"type": "audit_failure",', 'return GuardedResult("completed", failure={"type": "audit_failure",'),
        _mutation("guarded_runtime", "audit_only_after_tool_commit", 'audit("tool.approved", event)', "pass"),
        _mutation("guarded_runtime", "raw_action_egress", 'event["action"] = redactor.redact(deepcopy(event["action"]))', 'event["action"] = deepcopy(event["action"])'),
        _mutation("guarded_runtime", "partial_tool_failure_as_success", '                runtime.commit(deepcopy(event))\n', '                try:\n                    runtime.commit(deepcopy(event))\n                except Exception:\n                    pass\n'),
    ]


@pytest.mark.parametrize("_repeat", range(3))
def test_policy_mutations(_repeat):
    mutations = _policy_mutations()
    assert set(assert_mutations_rejected("policy_engine", mutations)) == {m.name for m in mutations}


@pytest.mark.parametrize("_repeat", range(3))
def test_approval_mutations(_repeat):
    mutations = _approval_mutations()
    assert set(assert_mutations_rejected("approval_gate", mutations)) == {m.name for m in mutations}


@pytest.mark.parametrize("_repeat", range(3))
def test_runtime_mutations(_repeat):
    mutations = _runtime_mutations()
    assert set(assert_mutations_rejected("guarded_runtime", mutations)) == {m.name for m in mutations}
