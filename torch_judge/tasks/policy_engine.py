"""Deterministic fail-closed policy composition for agent guardrails."""

from torch_judge.tasks._schema import build_design_note_rubric


TASK = {
    "title": "Composable Fail-Closed Policy Engine",
    "difficulty": "Medium",
    "version": 1,
    "function_name": "PolicyEngine",
    "description_en": r"""Implement 'PolicyDecision' and 'PolicyEngine(policies)'. Each policy has a non-empty 'name' and 'evaluate(context)' method returning a dictionary with 'action' equal to 'allow', 'deny', or 'transform', an optional string 'reason', and, for transform only, a 'changes' dictionary.

'PolicyEngine.evaluate(context) -> PolicyDecision' evaluates policies in constructor order on isolated JSON-like state. Transformations compose in that order by updating the working context. An allow never removes a previous transform. Deny wins immediately, so later policies must not run. A policy exception or malformed result fails closed as deny with a stable reason naming the failing policy. Never mutate the caller's context or leak a policy's mutable result.

'PolicyDecision' exposes 'action', 'context', and 'reasons'. The final action is deny, transform when any change was applied, otherwise allow. This deliberately uses deterministic local policy functions rather than live moderation. Production systems may combine classifiers, rules, and remote checks, but they still need explicit ordering, failure semantics, observability, and false-positive review.

Optional code reading: OpenAI Agents SDK separates input/output guardrail execution and tripwire results. NeMo Guardrails composes configured input, retrieval, and output rails and exposes blocking or modified results. Compare their lifecycle placement with this compact sequential composition core.""",
    "advisory_prerequisites": ["tool_registry"],
    "design_note_rubric": build_design_note_rubric(),
    "hints": [
        {"level": 1, "kind": "questions", "content": "What state does each policy see? If one policy transforms and the next allows, should the transform disappear? Which outcomes must stop evaluation? What should happen when policy code raises or returns an ambiguous result?"},
        {"level": 2, "kind": "analysis", "content": "Deep-copy the input, keep an ordered reason list and a transformed flag, then evaluate each policy against another copy. Validate every response before applying it. Update only from transform changes, return immediately on deny, and convert exceptions or malformed results into a deny that names the policy. Return newly copied state in the decision."},
    ],
    "model_connections": [
        "OpenAI Agents SDK InputGuardrail and OutputGuardrail wrap callable checks and expose tripwire outcomes; their run_in_parallel choice adds a latency-versus-prevention tradeoff beyond this sequential exercise.",
        "NeMo Guardrails LLMRails.check_async selects input/output rails from message roles and reports passed, modified, or blocked results; this exercise isolates deterministic composition and fail-closed semantics.",
        "A provider-neutral internal decision prevents model or moderation-provider response formats from controlling application policy directly.",
    ],
    "pro_con_analysis": {
        "pros": ["Deterministic order makes conflicts reproducible and auditable.", "Deny-wins and fail-closed behavior protect side-effecting boundaries.", "Composed transforms support least-data minimization without forcing every safe request to fail."],
        "cons": ["Fail-closed dependencies can reduce availability.", "Sequential checks add latency compared with parallel evaluation.", "Conservative policies can create false positives and require an appeal or override path."],
    },
    "sources": [
        {"kind": "code", "url": "https://github.com/openai/openai-agents-python", "commit": "2fa463571e76dae8ff267622f1018eaf06ffeb9f", "path": "src/agents/guardrail.py", "symbol": "GuardrailFunctionOutput, InputGuardrail.run, OutputGuardrail.run", "license": "MIT", "adapted": "Typed check outcome, named guardrail callable boundary, and explicit pre/post lifecycle placement.", "simplifications": "Synchronous local policies only; no asyncio, tracing spans, agents, generic context, or parallel execution."},
        {"kind": "code", "url": "https://github.com/NVIDIA/NeMo-Guardrails", "commit": "8cfaa12b79e68cf6ba3ac93ff675c16ff46e2716", "path": "nemoguardrails/rails/llm/llmrails.py", "symbol": "LLMRails.check_async and _determine_rails_from_messages", "license": "Apache-2.0", "adapted": "Input/output rail selection and passed/modified/blocked outcome vocabulary.", "simplifications": "No Colang runtime, LLM calls, retrieval rails, streaming buffers, action dispatcher, or async parallel rails."},
    ],
    "tests": [
        {"name": "Transforms compose in deterministic order", "behavior": "security.permission", "code": r"""
class Policy:
    def __init__(self,name,fn): self.name=name; self.fn=fn; self.seen=[]
    def evaluate(self,context): self.seen.append(dict(context)); return self.fn(context)
first=Policy('minimize',lambda c:{'action':'transform','reason':'drop private','changes':{'private':None,'step':1}})
second=Policy('label',lambda c:{'action':'transform','reason':'label','changes':{'step':c['step']+1}})
context={'public':'ok','private':'secret'}
decision={fn}([first,second]).evaluate(context)
assert decision.action=='transform'
assert decision.context=={'public':'ok','private':None,'step':2}
assert decision.reasons==['minimize: drop private','label: label']
assert context=={'public':'ok','private':'secret'}
assert second.seen==[{'public':'ok','private':None,'step':1}]
"""},
        {"name": "Deny wins and stops later policies", "behavior": "security.permission", "code": r"""
class Policy:
    def __init__(self,name,result): self.name=name; self.result=result; self.calls=0
    def evaluate(self,context): self.calls+=1; return self.result
before=Policy('before',{'action':'transform','changes':{'safe':True}})
deny=Policy('capability',{'action':'deny','reason':'not granted'})
after=Policy('after',{'action':'transform','changes':{'bypass':True}})
decision={fn}([before,deny,after]).evaluate({'safe':False})
assert decision.action=='deny' and decision.context=={'safe':True}
assert decision.reasons==['capability: not granted']
assert (before.calls,deny.calls,after.calls)==(1,1,0)
"""},
        {"name": "Exceptions fail closed without mutating caller state", "behavior": "state.invariant", "visibility": "unshown", "failure_message": "Policy exceptions must become typed deny decisions and must not mutate the caller's context.", "code": r"""
class Broken:
    name='remote-check'
    def evaluate(self,context): context['changed']=True; raise TimeoutError('late')
source={'value':1}
decision={fn}([Broken()]).evaluate(source)
assert decision.action=='deny' and decision.context=={'value':1}
assert decision.reasons==['remote-check: policy_error: TimeoutError: late']
assert source=={'value':1}
"""},
        {"name": "Malformed and ambiguous results fail closed", "behavior": "security.permission", "visibility": "unshown", "failure_message": "Reject unknown actions, transform without changes, invalid reasons, and duplicate policy names before unsafe continuation.", "code": r"""
class P:
    def __init__(self,name,result): self.name=name; self.result=result
    def evaluate(self,context): return self.result
bad=[{'action':'maybe'},{'action':'transform'},{'action':'allow','reason':3},None]
for result in bad:
    decision={fn}([P('bad',result)]).evaluate({'x':1})
    assert decision.action=='deny' and decision.reasons[0].startswith('bad: invalid_policy_result')
try: {fn}([P('same',{'action':'allow'}),P('same',{'action':'allow'})])
except ValueError: pass
else: raise AssertionError('duplicate policy names accepted')
"""},
        {"name": "Empty policy set allows an isolated context", "behavior": "state.invariant", "visibility": "unshown", "failure_message": "With no policies, return allow and a fresh copy of the original context.", "code": r"""
source={'nested':{'x':1}}
decision={fn}([]).evaluate(source)
assert decision.action=='allow' and decision.context==source and decision.context is not source
decision.context['nested']['x']=2
assert source=={'nested':{'x':1}}
"""},
        {"name": "Harness author failures remain evaluator errors", "behavior": "state.invariant", "visibility": "unshown", "failure_message": "A HarnessFailure raised by a deterministic fixture denotes an evaluator bug and must escape rather than become a learner-facing deny.", "code": r"""
from torch_judge.harness import HarnessFailure
class Broken:
    name='fixture'
    def evaluate(self,context): raise HarnessFailure('fixture exhausted')
try: {fn}([Broken()]).evaluate({'x':1})
except HarnessFailure: pass
else: raise AssertionError('HarnessFailure was swallowed')
"""},
    ],
    "solution": r'''class PolicyDecision:
    def __init__(self, action, context, reasons):
        from copy import deepcopy
        self.action = action
        self.context = deepcopy(context)
        self.reasons = list(reasons)


class PolicyEngine:
    def __init__(self, policies):
        if not isinstance(policies, (list, tuple)):
            raise TypeError("policies must be an ordered sequence")
        names = []
        for policy in policies:
            name = getattr(policy, "name", None)
            if not isinstance(name, str) or not name or not callable(getattr(policy, "evaluate", None)):
                raise TypeError("each policy needs a name and evaluate method")
            names.append(name)
        if len(names) != len(set(names)):
            raise ValueError("policy names must be unique")
        self._policies = tuple(policies)

    def evaluate(self, context):
        from copy import deepcopy
        from torch_judge.harness import HarnessFailure
        from torch_judge.harness.agents.protocol import copy_json

        working = copy_json(context, field="policy context JSON")
        if not isinstance(working, dict):
            raise TypeError("policy context must be a dictionary")
        transformed = False
        reasons = []
        for policy in self._policies:
            try:
                result = policy.evaluate(deepcopy(working))
            except HarnessFailure:
                raise
            except Exception as error:
                return PolicyDecision(
                    "deny",
                    working,
                    [f"{policy.name}: policy_error: {type(error).__name__}: {error}"],
                )
            valid = isinstance(result, dict)
            action = result.get("action") if valid else None
            reason = result.get("reason") if valid else None
            changes = result.get("changes") if valid else None
            if (
                action not in {"allow", "deny", "transform"}
                or (reason is not None and not isinstance(reason, str))
                or (action == "transform" and not isinstance(changes, dict))
                or (action != "transform" and changes is not None)
            ):
                return PolicyDecision(
                    "deny", working, [f"{policy.name}: invalid_policy_result"]
                )
            if action == "deny":
                return PolicyDecision(
                    "deny", working, [f"{policy.name}: {reason or 'denied'}"]
                )
            if action == "transform":
                try:
                    changes = copy_json(changes, field="policy changes JSON")
                except Exception:
                    return PolicyDecision(
                        "deny", working, [f"{policy.name}: invalid_policy_result"]
                    )
                working.update(changes)
                transformed = True
                if reason:
                    reasons.append(f"{policy.name}: {reason}")
        return PolicyDecision("transform" if transformed else "allow", working, reasons)''',
    "demo": "engine=PolicyEngine([])\nprint(engine.evaluate({'request':'hello'}).action)",
}
