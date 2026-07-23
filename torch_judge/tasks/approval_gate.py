"""Least-privilege action approval with digest-bound, one-use grants."""

from torch_judge.tasks._schema import build_design_note_rubric


TASK = {
    "title": "Capability-Bound Human Approval Gate",
    "difficulty": "Hard",
    "version": 1,
    "function_name": "ApprovalGate",
    "description_en": r"""Implement 'ApprovalGate(approval_required, clock, ttl=60)' with 'classify(action, capabilities, provenance)' and 'approve(request, approver)'. Also define typed 'ApprovalRequest', 'ApprovedAction', and 'DeniedAction' results.

'action' is a JSON-like dictionary containing non-empty 'tool', 'capability', 'actor', and 'arguments'. 'capabilities' is the exact set granted to that actor—do not infer parent/admin privileges or wildcard expansion. 'provenance' contains non-empty 'identity' and 'source' plus 'trust' ('trusted' or 'untrusted'); identity must equal action.actor. Deny malformed provenance, identity mismatch, or unauthorized capability before approval.

For authorized capabilities not in 'approval_required', return 'ApprovedAction'. For configured high-risk capabilities, untrusted provenance is denied; otherwise return an 'ApprovalRequest' bound to a canonical digest of tool, arguments, capability, actor, and provenance identity/source, with 'expires_at=clock.now()+ttl'. 'approve(request, approver)' accepts a trusted approver identity and creates a deterministic opaque token. A later action may include 'approval_token'. Accept it only when digest, actor identity, and expiry still match; consume it exactly once. Changed arguments, replay, expiry, or cross-identity reuse are denied.

This exercise models the authorization race explicitly: approval describes one immutable intended action, not a tool name. Production systems additionally persist grants transactionally, authenticate approvers, revoke sessions, and coordinate concurrent consumers.""",
    "advisory_prerequisites": ["policy_engine", "tool_registry"],
    "design_note_rubric": build_design_note_rubric(),
    "hints": [
        {"level": 1, "kind": "questions", "content": "Which exact fields define the approved side effect? Can a tool name alone distinguish two transfers? Where do identity and expiry enter the digest? What state makes a token one-use? Which checks must happen before asking a human?"},
        {"level": 2, "kind": "analysis", "content": "Validate/copy action and provenance, enforce exact capability membership and identity equality, then canonicalize the approval-bound fields with sorted JSON and SHA-256. Store issued token records containing digest, identity, expiry, and consumed state. Validate all fields and time before atomically marking a token consumed; never reuse an invalid token by silently creating a new approval request."},
    ],
    "model_connections": [
        "OpenAI Agents SDK function tools expose needs_approval so a run can interrupt before tool execution and later resume from approved or rejected tool-call state.",
        "OpenClaw tool policy layers construct a per-run allowed inventory; approval remains a second boundary for a particular high-risk invocation rather than a substitute for capability filtering.",
        "Hermes separates tool schemas/dispatch from user-facing confirmation flows; digest binding prevents a confirmed description from authorizing changed execution arguments.",
    ],
    "pro_con_analysis": {
        "pros": ["Exact capability checks enforce least privilege before involving a human.", "Digest-bound one-use grants close changed-argument and replay races.", "Typed outcomes make approval workflow state explicit and auditable."],
        "cons": ["Human approval adds latency and can become notification fatigue.", "In-memory grants do not survive restarts or coordinate multiple processes.", "Canonicalization and identity design must remain consistent across UI, gateway, and executor."],
    },
    "sources": [
        {"kind": "code", "url": "https://github.com/openai/openai-agents-python", "commit": "2fa463571e76dae8ff267622f1018eaf06ffeb9f", "path": "src/agents/tool.py", "symbol": "FunctionTool.needs_approval", "license": "MIT", "adapted": "Per-call approval boundary before function tool invocation.", "simplifications": "Synchronous deterministic token store; no RunState, async callbacks, hosted tools, MCP, persistence, or streaming interruptions."},
        {"kind": "code", "url": "https://github.com/openclaw/openclaw", "commit": "40d31f34813c2a01284b097c0d0d785fbb173400", "path": "src/agents/openclaw-tools.ts", "symbol": "createOpenClawTools and filterToolsByClientCaps", "license": "MIT", "adapted": "Run-scoped capability filtering before tools become invocable.", "simplifications": "Exact local capability strings; no plugins, channels, owner policy, sandbox, or dynamic runtime inventory."},
    ],
    "tests": [
        {"name": "Low-risk exact capability is approved without a grant", "behavior": "security.permission", "code": r"""
from torch_judge.harness.agents import VirtualClock
gate={fn}({'payments.write'},VirtualClock(),ttl=10)
action={'tool':'search','capability':'public.read','actor':'agent-1','arguments':{'q':'safe'}}
provenance={'identity':'agent-1','source':'user-session','trust':'trusted'}
result=gate.classify(action,{'public.read'},provenance)
assert type(result).__name__=='ApprovedAction' and result.action==action
result.action['arguments']['q']='changed'
assert action['arguments']['q']=='safe'
"""},
        {"name": "High-risk approval is bound, expiring, and one-use", "behavior": "security.permission", "code": r"""
from torch_judge.harness.agents import VirtualClock
clock=VirtualClock(); gate={fn}({'payments.write'},clock,ttl=5)
base={'tool':'transfer','capability':'payments.write','actor':'agent-1','arguments':{'amount':10,'to':'A'}}
prov={'identity':'agent-1','source':'user-session','trust':'trusted'}
request=gate.classify(base,{'payments.write'},prov)
assert type(request).__name__=='ApprovalRequest' and request.expires_at==5
token=gate.approve(request,{'identity':'human-1','trust':'trusted'})
approved=gate.classify({**base,'approval_token':token},{'payments.write'},prov)
assert type(approved).__name__=='ApprovedAction'
replay=gate.classify({**base,'approval_token':token},{'payments.write'},prov)
assert type(replay).__name__=='DeniedAction' and replay.reason=='approval_replayed'
"""},
        {"name": "Changed arguments and cross-identity reuse are denied", "behavior": "security.permission", "visibility": "unshown", "failure_message": "Bind approval to canonical arguments and actor/provenance identity; never authorize a changed or cross-identity action.", "code": r"""
from torch_judge.harness.agents import VirtualClock
gate={fn}({'danger'},VirtualClock(),ttl=10)
action={'tool':'delete','capability':'danger','actor':'a','arguments':{'path':'one'}}
prov={'identity':'a','source':'session','trust':'trusted'}
request=gate.classify(action,{'danger'},prov); token=gate.approve(request,{'identity':'h','trust':'trusted'})
changed={**action,'arguments':{'path':'two'},'approval_token':token}
assert gate.classify(changed,{'danger'},prov).reason=='approval_mismatch'
other={**action,'actor':'b','approval_token':token}; other_prov={**prov,'identity':'b'}
assert gate.classify(other,{'danger'},other_prov).reason=='approval_mismatch'
"""},
        {"name": "Expiry, provenance, and least privilege fail closed", "behavior": "security.permission", "visibility": "unshown", "failure_message": "Deny expired grants, untrusted high-risk provenance, identity mismatch, and capabilities that were not granted exactly.", "code": r"""
from torch_judge.harness.agents import VirtualClock
clock=VirtualClock(); gate={fn}({'admin.delete'},clock,ttl=1)
action={'tool':'delete','capability':'admin.delete','actor':'a','arguments':{}}
trusted={'identity':'a','source':'session','trust':'trusted'}
assert gate.classify(action,{'admin'},trusted).reason=='capability_not_granted'
assert gate.classify(action,{'admin.delete'},{**trusted,'trust':'untrusted'}).reason=='untrusted_provenance'
assert gate.classify(action,{'admin.delete'},{**trusted,'identity':'b'}).reason=='identity_mismatch'
request=gate.classify(action,{'admin.delete'},trusted); token=gate.approve(request,{'identity':'h','trust':'trusted'})
clock.sleep(1)
assert gate.classify({**action,'approval_token':token},{'admin.delete'},trusted).reason=='approval_expired'
"""},
        {"name": "Approval issuance validates approver and request", "behavior": "state.invariant", "visibility": "unshown", "failure_message": "Only a trusted named approver may issue a grant for an ApprovalRequest created by this gate.", "code": r"""
from torch_judge.harness.agents import VirtualClock
gate={fn}({'write'},VirtualClock(),ttl=2)
action={'tool':'x','capability':'write','actor':'a','arguments':{}}
request=gate.classify(action,{'write'},{'identity':'a','source':'s','trust':'trusted'})
for approver in ({'identity':'','trust':'trusted'},{'identity':'h','trust':'untrusted'}):
    try: gate.approve(request,approver)
    except (TypeError,ValueError): pass
    else: raise AssertionError('invalid approver accepted')
"""},
        {"name": "Approval request state is immutable to callers and approved once", "behavior": "security.permission", "visibility": "unshown", "failure_message": "Use gate-owned request state so caller mutation cannot change the grant, and do not issue multiple grants for one human decision.", "code": r"""
from torch_judge.harness.agents import VirtualClock
gate={fn}({'write'},VirtualClock(),ttl=5)
action={'tool':'write','capability':'write','actor':'a','arguments':{'value':1}}
prov={'identity':'a','source':'s','trust':'trusted'}
request=gate.classify(action,{'write'},prov)
request.digest='tampered'; request.identity='other'; request.expires_at=999
token=gate.approve(request,{'identity':'h','trust':'trusted'})
assert type(gate.classify({**action,'approval_token':token},{'write'},prov)).__name__=='ApprovedAction'
try: gate.approve(request,{'identity':'h','trust':'trusted'})
except ValueError: pass
else: raise AssertionError('one approval request issued multiple grants')
"""},
    ],
    "solution": r'''class ApprovalRequest:
    def __init__(self, digest, action, identity, expires_at):
        from copy import deepcopy
        self.digest = digest
        self.action = deepcopy(action)
        self.identity = identity
        self.expires_at = expires_at


class ApprovedAction:
    def __init__(self, action):
        from copy import deepcopy
        self.action = deepcopy(action)


class DeniedAction:
    def __init__(self, reason):
        self.reason = reason


class ApprovalGate:
    def __init__(self, approval_required, clock, ttl=60):
        from math import isfinite
        if not isinstance(approval_required, set) or not all(
            isinstance(item, str) and item for item in approval_required
        ):
            raise TypeError("approval_required must be a set of capability strings")
        if (isinstance(ttl, bool) or not isinstance(ttl, (int, float))
                or not isfinite(float(ttl)) or ttl <= 0):
            raise ValueError("ttl must be a positive finite number")
        self._approval_required = frozenset(approval_required)
        self._clock = clock
        self._ttl = float(ttl)
        self._requests = {}
        self._grants = {}
        self._sequence = 0

    def _validated(self, action, capabilities, provenance):
        from copy import deepcopy
        from torch_judge.harness.agents.protocol import copy_json
        try:
            copied = copy_json(action, field="approval action JSON")
            source = copy_json(provenance, field="approval provenance JSON")
        except Exception:
            return DeniedAction("malformed_request")
        if not isinstance(copied, dict) or not isinstance(source, dict):
            return DeniedAction("malformed_request")
        if not isinstance(capabilities, set) or not all(
            isinstance(item, str) and item for item in capabilities
        ):
            return DeniedAction("malformed_capabilities")
        required = ("tool", "capability", "actor", "arguments")
        if (any(not isinstance(copied.get(key), str) or not copied[key]
                for key in required[:3]) or not isinstance(copied.get("arguments"), dict)):
            return DeniedAction("malformed_request")
        if (not isinstance(source.get("identity"), str) or not source["identity"]
                or not isinstance(source.get("source"), str) or not source["source"]
                or source.get("trust") not in {"trusted", "untrusted"}):
            return DeniedAction("malformed_provenance")
        if source["identity"] != copied["actor"]:
            return DeniedAction("identity_mismatch")
        if copied["capability"] not in capabilities:
            return DeniedAction("capability_not_granted")
        return deepcopy(copied), deepcopy(source)

    @staticmethod
    def _digest(action, provenance):
        import hashlib
        import json
        bound_action = {key: value for key, value in action.items() if key != "approval_token"}
        bound = {
            "action": bound_action,
            "identity": provenance["identity"],
            "source": provenance["source"],
        }
        encoded = json.dumps(bound, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def classify(self, action, capabilities, provenance):
        checked = self._validated(action, capabilities, provenance)
        if isinstance(checked, DeniedAction):
            return checked
        copied, source = checked
        capability = copied["capability"]
        clean_action = {key: value for key, value in copied.items() if key != "approval_token"}
        if capability not in self._approval_required:
            return ApprovedAction(clean_action)
        if source["trust"] != "trusted":
            return DeniedAction("untrusted_provenance")
        digest = self._digest(copied, source)
        token = copied.get("approval_token")
        if token is None:
            request = ApprovalRequest(
                digest, clean_action, source["identity"], self._clock.now() + self._ttl
            )
            self._requests[id(request)] = {
                "digest": digest,
                "identity": source["identity"],
                "expires_at": request.expires_at,
                "approved": False,
            }
            return request
        grant = self._grants.get(token) if isinstance(token, str) else None
        if grant is None:
            return DeniedAction("approval_invalid")
        if grant["consumed"]:
            return DeniedAction("approval_replayed")
        if grant["digest"] != digest or grant["identity"] != source["identity"]:
            return DeniedAction("approval_mismatch")
        if self._clock.now() >= grant["expires_at"]:
            return DeniedAction("approval_expired")
        grant["consumed"] = True
        return ApprovedAction(clean_action)

    def approve(self, request, approver):
        import hashlib
        if not isinstance(request, ApprovalRequest) or id(request) not in self._requests:
            raise TypeError("request must originate from this gate")
        if (not isinstance(approver, dict)
                or not isinstance(approver.get("identity"), str) or not approver["identity"]
                or approver.get("trust") != "trusted"):
            raise ValueError("approver must be a trusted named identity")
        stored = self._requests[id(request)]
        if stored["approved"]:
            raise ValueError("approval request was already approved")
        stored["approved"] = True
        self._sequence += 1
        token = hashlib.sha256(
            f"{stored['digest']}:{approver['identity']}:{self._sequence}".encode("utf-8")
        ).hexdigest()
        self._grants[token] = {
            "digest": stored["digest"],
            "identity": stored["identity"],
            "expires_at": stored["expires_at"],
            "consumed": False,
        }
        return token''',
    "demo": "from torch_judge.harness.agents import VirtualClock\ngate=ApprovalGate(set(),VirtualClock())",
}
