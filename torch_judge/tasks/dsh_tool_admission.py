"""Decide whether a tool call may run: policy middleware, approval, then deny-only guards."""

from ._interview import interview

TASK = {
    "title": "Tool Call Admission",
    "difficulty": "Medium",
    "version": 1,
    "function_name": "admit",
    "description_en": r"""Before DeepSeek Harness runs a tool, it decides whether the call is allowed. Implement that decision.

**Signature:** `admit(call, pre_execute, guards, approve=None) -> dict`

**Parameters:**
- `call` — dict with `name` and `arguments`.
- `pre_execute` — list of middleware functions `f(call, next)`. Calling `next()` runs the next function in the list; after the last one, `next()` returns `{"kind": "allow"}`. A function may return a decision without calling `next()`.
- `guards` — list of functions `guard(call) -> str | None`.
- `approve` — optional function `approve(call, reason) -> "allowed-once" | "rejected" | "cancelled"`; `None` means there is no approval channel.

**Decisions** are `{"kind": "allow"}`, `{"kind": "deny", "reason": str}`, `{"kind": "cancel"}` and `{"kind": "ask", "reason"?: str}`. `admit` never returns `ask`.

**Steps:**
- **Policy.** Run the `pre_execute` chain and take its decision.
- **Ask.** If the decision is `ask`, resolve it. With no approval channel, deny with the ask's `reason`, or with `f'tool "{name}" requires approval (not yet supported)'` if it has none. Otherwise call `approve(call, reason)` exactly once, where `reason` is the ask's reason or None: `"allowed-once"` allows; `"rejected"` denies with `f'the user rejected tool "{name}"'`; `"cancelled"` denies with `f'approval for tool "{name}" was cancelled'`.
- **Guards.** Only if the decision is now `allow`: call each guard in order; the first one that returns a string turns the decision into a deny with that reason, and later guards are not called.
- Return the decision. `deny` and `cancel` decisions from the policy step are returned as they are. Exceptions from any function propagate to the caller.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why middleware for policy.** Hooks, permission modes and sandbox rules are plugins. Each can decide, delegate, or wrap the decision of the ones after it.

**Why guards are separate and deny-only.** Middleware can override each other, so a security rule written as middleware could be undone by a plugin registered in front of it. Guards run after every middleware and can only deny, so no ordering of plugins can turn a guard's denial back into an allow.

**Why deny without a channel.** A headless run has nobody to ask. Failing closed is the only safe default for a call that required approval.""",
    "advisory_prerequisites": ["dsh_event_bus", "approval_gate"],
    "hints": [
        {"level": 1, "kind": "questions", "content": "How do you build next() for a list of middleware? After an ask is approved, do the guards still run? What reason goes to approve() when the ask has none?"},
        {"level": 2, "kind": "analysis", "content": "Copy the middleware list into a queue and define next() that pops the head and calls it with (call, next), or returns {'kind': 'allow'} when empty. Resolve ask with a small if-chain on the approval outcome. Then, only for allow, loop over guards and return the first denial."},
    ],
    "model_connections": [
        "DeepSeek Harness's ToolRuntime.prepareExecution runs the tools/pre-execute waterfall, resolves ask through the approval service, then applies registered monotonic guards.",
        "Claude Code's permission modes and PreToolUse hooks make the same allow, deny or ask decision before a tool runs.",
    ],
    "pro_con_analysis": {
        "pros": ["Extensible policy with a hard floor: plugins compose freely, while guards give security rules that no plugin order can bypass."],
        "cons": ["Two policy mechanisms to learn, and asking the user per call adds friction that users often answer without reading."],
    },
    "sources": [
        {"kind": "code", "url": "https://github.com/deepseek-ai/deepseek-harness", "commit": "21638c56315ae6a2b552d6091945d3144c9af32e", "path": "packages/core/tools/src/index.ts", "symbol": "ToolRuntime.prepareExecution, serviceAsk and ToolLayer.guardReason", "license": "MIT", "adapted": "Pre-execute waterfall with an allow default, ask resolution wording, fail-closed without an approval channel, and deny-only guards after an allow.", "simplifications": "Synchronous; no scopes, caller cancellation, display reasons or the unavailable approval outcome."},
    ],
    "tests": [
        {"name": "Guards deny after policy allows", "behavior": "security.permission", "code": r"""
call = {"name": "rm", "arguments": {"path": "/"}}
assert {fn}(call, [], []) == {"kind": "allow"}
out = {fn}(call, [lambda c, next: next()], [lambda c: None, lambda c: "rm is forbidden"])
assert out == {"kind": "deny", "reason": "rm is forbidden"}, out
out = {fn}(call, [lambda c, next: {"kind": "deny", "reason": "policy"}], [lambda c: "guard"])
assert out == {"kind": "deny", "reason": "policy"}, out
"""},
        {"name": "Ask resolution covers every approval outcome", "visibility": "unshown", "behavior": "security.permission", "failure_message": "Resolve ask through approve exactly once with the stated wording, deny without a channel, and still apply guards after approval.", "code": r"""
call = {"name": "pay", "arguments": {}}
def run(outcome, reason=None, guards=()):
    asked = []
    ask = {"kind": "ask"} if reason is None else {"kind": "ask", "reason": reason}
    approve = None if outcome is None else (lambda c, r: asked.append(r) or outcome)
    return {fn}(call, [lambda c, next: ask], list(guards), approve), asked
out, asked = run("allowed-once", "costs money")
assert out == {"kind": "allow"} and asked == ["costs money"], (out, asked)
out, asked = run("allowed-once")
assert asked == [None], asked
assert run("rejected")[0] == {"kind": "deny", "reason": 'the user rejected tool "pay"'}
assert run("cancelled")[0] == {"kind": "deny", "reason": 'approval for tool "pay" was cancelled'}
assert run(None)[0] == {"kind": "deny", "reason": 'tool "pay" requires approval (not yet supported)'}
assert run(None, "needs a human")[0] == {"kind": "deny", "reason": "needs a human"}
assert run("allowed-once", guards=[lambda c: "over budget"])[0] == {"kind": "deny", "reason": "over budget"}
"""},
        {"name": "Seeded middleware chains match an oracle", "visibility": "unshown", "behavior": "events.ordering", "failure_message": "Run middleware in order with an allow default, let a middleware short-circuit, pass cancel and deny through untouched, and run guards only after an allow until the first denial.", "code": r"""
import random
for seed in (13, 47, 82):
    rng = random.Random(seed)
    for trial in range(200):
        trace = []
        pre, plan = [], []
        for i in range(rng.randint(0, 4)):
            kind = rng.choice(["pass", "pass", "allow", "deny", "cancel", "wrap"])
            plan.append(kind)
            if kind == "pass":
                pre.append(lambda c, n, i=i: trace.append(("pre", i)) or n())
            elif kind == "allow":
                pre.append(lambda c, n, i=i: trace.append(("pre", i)) or {"kind": "allow"})
            elif kind == "deny":
                pre.append(lambda c, n, i=i: trace.append(("pre", i)) or {"kind": "deny", "reason": f"d{i}"})
            elif kind == "cancel":
                pre.append(lambda c, n, i=i: trace.append(("pre", i)) or {"kind": "cancel"})
            else:
                pre.append(lambda c, n, i=i: trace.append(("pre", i)) or ({"kind": "deny", "reason": f"w{i}"} if n()["kind"] == "allow" else {"kind": "allow"}))
        results = [rng.choice([None, None, f"g{j}"]) for j in range(rng.randint(0, 3))]
        guards = [lambda c, j=j, r=r: trace.append(("guard", j)) or r for j, r in enumerate(results)]
        got = {fn}({"name": "t", "arguments": {}}, pre, guards)
        def decide(i):
            if i == len(plan):
                return {"kind": "allow"}, []
            kind = plan[i]
            if kind == "pass":
                d, t = decide(i + 1); return d, [("pre", i)] + t
            if kind == "allow":
                return {"kind": "allow"}, [("pre", i)]
            if kind == "deny":
                return {"kind": "deny", "reason": f"d{i}"}, [("pre", i)]
            if kind == "cancel":
                return {"kind": "cancel"}, [("pre", i)]
            d, t = decide(i + 1)
            return ({"kind": "deny", "reason": f"w{i}"} if d["kind"] == "allow" else {"kind": "allow"}), [("pre", i)] + t
        want, want_trace = decide(0)
        if want["kind"] == "allow":
            for j, r in enumerate(results):
                want_trace.append(("guard", j))
                if r is not None:
                    want = {"kind": "deny", "reason": r}; break
        assert got == want and trace == want_trace, (seed, trial, plan, results, got, want, trace)
"""},
        {"name": "Exceptions propagate", "visibility": "unshown", "behavior": "contract.signature", "failure_message": "admit must not swallow exceptions from middleware, approval or guards.", "code": r"""
def boom(*args):
    raise KeyError("x")
call = {"name": "t", "arguments": {}}
for args in (([boom], []), ([], [boom]), ([lambda c, n: {"kind": "ask"}], [], boom)):
    try:
        {fn}(call, *args)
    except KeyError:
        pass
    else:
        raise AssertionError("an exception was swallowed")
"""},
    ],
    "solution": '''def admit(call, pre_execute, guards, approve=None):
    queue = list(pre_execute)

    def next():
        if queue:
            return queue.pop(0)(call, next)
        return {"kind": "allow"}

    decision = next()
    name = call["name"]
    if decision["kind"] == "ask":
        reason = decision.get("reason")
        if approve is None:
            decision = {"kind": "deny", "reason": reason or f'tool "{name}" requires approval (not yet supported)'}
        else:
            outcome = approve(call, reason)
            if outcome == "allowed-once":
                decision = {"kind": "allow"}
            elif outcome == "rejected":
                decision = {"kind": "deny", "reason": f'the user rejected tool "{name}"'}
            else:
                decision = {"kind": "deny", "reason": f'approval for tool "{name}" was cancelled'}
    if decision["kind"] == "allow":
        for guard in guards:
            reason = guard(call)
            if reason is not None:
                return {"kind": "deny", "reason": reason}
    return decision
''',
    "interview_questions": interview(
        concept=[
            "How does an agent harness decide whether a tool call may run? What are allow, deny, ask and cancel?",
            "Why must a call that needs approval be denied when there is no one to ask?",
        ],
        deep_dive=[
            "Walk through how next() chains the policy middleware. What does a middleware that never calls next() do to the ones after it?",
            "Why do guards run after all middleware, only after an allow, and only with the power to deny? What attack or bug does that prevent?",
            "After the user approves a call, should guards still be able to deny it? Why?",
        ],
        tradeoffs=[
            "Approval per call versus per session versus allow-lists: user friction, safety and prompt-injection risk?",
            "Policy as code (middleware) versus policy as data (rules in config): flexibility, auditability and review cost?",
        ],
    ),
}
