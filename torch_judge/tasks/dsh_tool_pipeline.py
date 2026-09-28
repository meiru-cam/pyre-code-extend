"""Guarded tool execution pipeline: policy, approval, guards, around-dispatch and post-policy."""

from ._interview import interview

TASK = {
    "title": "Tool Pipeline Capstone",
    "difficulty": "Hard",
    "version": 1,
    "function_name": "ToolRuntime",
    "description_en": r"""Capstone: implement the whole pipeline every tool call in DeepSeek Harness goes through, from the model's request to the single result the model sees. It combines Tool Call Admission and Tool Dispatch and Result Policy, and adds failure containment and observers.

**Signature:** `ToolRuntime(tools, approve=None)` with attributes `pre_execute`, `guards`, `around`, `post_execute`, `observers` (lists you may append to) and method `execute(call) -> dict`.

**Parameters:**
- `tools` — dict from tool name to a function `body(arguments) -> list` of content blocks. It may raise.
- `approve` — optional function `approve(call, reason) -> "allowed-once" | "rejected" | "cancelled"`; `None` means no approval channel.
- `call` — dict with `name` and `arguments`.

**Results.** Every result is a dict. Success: `{"is_error": False, "content": blocks}`. An *error result for message `m` with code `c`* is `{"is_error": True, "content": [{"type": "text", "text": f"Error: {m}"}], "error": {"message": m, "code": c}}`, where `c` may be `None`. A raised exception `e` gives the error result for `str(e)` with code `None`. A result may also carry `additional_contexts`, a list, which is present only when non-empty.

**Stages, in order:**
- **Pre-execute.** Run `pre_execute` as middleware: each is `f(call, next)` and `next()` continues to the next one; after the last, `next()` returns `{"kind": "allow"}`. The decision is `allow`, `deny` with `reason`, `cancel`, or `ask` with optional `reason`.
- **Ask.** Resolve `ask` to `allow` or `deny`. With no approval channel: deny with the ask's `reason`, or `f'tool "{name}" requires approval (not yet supported)'` when it has none. Otherwise call `approve(call, reason)` once: `"allowed-once"` allows, `"rejected"` denies with `f'the user rejected tool "{name}"'`, `"cancelled"` denies with `f'approval for tool "{name}" was cancelled'`.
- **Cancel** becomes the error result for `"tool call aborted before dispatch"` with code `"ABORTED_BEFORE_DISPATCH"`.
- **Guards.** Only when the decision is `allow`: call each `guard(call)` in order; the first one that returns a string denies with that reason.
- **Deny** becomes the error result for the reason with code `None`.
- **Dispatch.** Only when allowed: run `around` as middleware, each `f(call, next)`; after the last, `next()` runs the tool body. The body step returns the success result, or the error result for `f'unknown tool "{name}"'` with code `"UNKNOWN_TOOL"` if the name is not in `tools`, or the exception's error result if the body raises.
- **Post-execute.** The cancel, deny and dispatch results all go through `post_execute` middleware, each `f(call, result, next)`; after the last, `next()` returns `{"kind": "accept"}`. `accept` keeps the result, replaces its `content` if the decision has `content`, and sets `additional_contexts` to the result's contexts followed by the decision's. `block` produces `{"is_error": True, "content": feedback, "error": {"message": text, "code": None}}` where `text` joins the `text` of the feedback's text blocks with newlines, and `additional_contexts` holds only the decision's contexts.
- **Pipeline failures.** If a `pre_execute`, guard, `approve`, `around` or `post_execute` function raises, stop right there: the result is that exception's error result, and later stages, including post-execute, are skipped.
- **Observers.** Call every `observers` function with `(call, result)` on the final result. An observer that raises is ignored and does not stop the others. Return the final result.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**One result per call.** Model APIs require exactly one result per tool call, so every failure, including a crash in a policy plugin, is converted into an error result instead of propagating.

**Why guards are separate.** Pre-execute listeners are extensible middleware, so one listener can override another. Guards run after all of them and can only deny, never allow, so a security rule cannot be undone by a later plugin.

**Why denials still reach post-execute.** Post-execute listeners see denied calls too; the repeat-call reminder counts them, because a model hammering a denied call is a loop.

**Why observers are contained.** A logging or telemetry observer must never change or break the result the model receives.""",
    "advisory_prerequisites": ["dsh_tool_admission", "dsh_tool_dispatch"],
    "hints": [
        {"level": 1, "kind": "questions", "content": "Which stages can produce a result that still goes through post-execute, and which failures skip it? Why do guards run only after an allow? How do you build next() for a list of middleware?"},
        {"level": 2, "kind": "analysis", "content": "Write one helper that chains middleware with an innermost default. In execute: wrap everything before observers in try/except that turns an exception into an error result and jumps to observers. Compute the pre-dispatch result (None when dispatching), otherwise run the around chain whose inner step catches body errors itself. Then run post-execute and apply accept or block."},
    ],
    "model_connections": [
        "DeepSeek Harness's ToolRuntime.execute runs tools/pre-execute, approval, monotonic guards, tools/execute, tools/post-execute and tools/result exactly in this order.",
        "Claude Code hooks (PreToolUse, PostToolUse) and permission prompts sit at the same seams.",
    ],
    "pro_con_analysis": {
        "pros": ["Policies, approvals, timeouts and reminders compose without touching tools or the loop, and every call yields exactly one well-formed result."],
        "cons": ["Converting every failure into a result can hide plugin bugs from developers, and the many stages make it hard to see why a call was denied."],
    },
    "sources": [
        {"kind": "code", "url": "https://github.com/deepseek-ai/deepseek-harness", "commit": "21638c56315ae6a2b552d6091945d3144c9af32e", "path": "packages/core/tools/src/index.ts", "symbol": "ToolRuntime.prepareExecution, serviceAsk, dispatchScheduledExecution, postExecute and notifyResult", "license": "MIT", "adapted": "Stage order, ask resolution wording, deny-only guards, cancel and deny results entering post-execute, pipeline failures skipping it, block and accept semantics, and contained observers.", "simplifications": "Synchronous, no scopes, signals, schemas, value snapshots, content projectors or finalizers; the unavailable approval outcome is folded into approve=None."},
    ],
    "tests": [
        {"name": "Allowed call runs and post-execute adds context", "behavior": "protocol.validation", "code": r"""
rt = {fn}({"read": lambda args: [{"type": "text", "text": "file:" + args["path"]}]})
rt.post_execute.append(lambda call, result, next: {**next(), "additional_contexts": ["note"]})
out = rt.execute({"name": "read", "arguments": {"path": "a.py"}})
assert out == {"is_error": False, "content": [{"type": "text", "text": "file:a.py"}], "additional_contexts": ["note"]}, out
"""},
        {"name": "Guards deny after policy allows and denials reach post-execute", "behavior": "security.permission", "code": r"""
rt = {fn}({"rm": lambda args: [{"type": "text", "text": "deleted"}]})
rt.pre_execute.append(lambda call, next: {"kind": "allow"})
rt.guards.append(lambda call: None)
rt.guards.append(lambda call: "rm is forbidden" if call["name"] == "rm" else None)
seen = []
rt.post_execute.append(lambda call, result, next: seen.append(result["is_error"]) or next())
out = rt.execute({"name": "rm", "arguments": {}})
assert out == {"is_error": True, "content": [{"type": "text", "text": "Error: rm is forbidden"}], "error": {"message": "rm is forbidden", "code": None}}, out
assert seen == [True]
"""},
        {"name": "Ask resolution covers every approval outcome", "visibility": "unshown", "behavior": "security.permission", "failure_message": "Resolve ask through the approval channel exactly once, with the stated denial wording, and deny when no channel exists.", "code": r"""
def run(approve, reason=None):
    calls = []
    body_runs = []
    def ap(call, r):
        calls.append(r); return approve
    rt = {fn}({"pay": lambda args: body_runs.append(1) or [{"type": "text", "text": "paid"}]}, approve=None if approve is None else ap)
    decision = {"kind": "ask"} if reason is None else {"kind": "ask", "reason": reason}
    rt.pre_execute.append(lambda call, next: decision)
    return rt.execute({"name": "pay", "arguments": {}}), calls, body_runs
out, calls, runs = run("allowed-once", "costs money")
assert out["is_error"] is False and calls == ["costs money"] and runs == [1], (out, calls)
out, calls, runs = run("rejected")
assert out["error"]["message"] == 'the user rejected tool "pay"' and runs == [], out
out, calls, runs = run("cancelled")
assert out["error"]["message"] == 'approval for tool "pay" was cancelled' and runs == [], out
out, _, runs = run(None)
assert out["error"]["message"] == 'tool "pay" requires approval (not yet supported)' and runs == [], out
out, _, _ = run(None, "needs a human")
assert out["error"]["message"] == "needs a human", out
rt = {fn}({"pay": lambda a: []}, approve=lambda c, r: "allowed-once")
rt.pre_execute.append(lambda call, next: {"kind": "ask"})
rt.guards.append(lambda call: "blocked by guard")
assert rt.execute({"name": "pay", "arguments": {}})["error"]["message"] == "blocked by guard"
"""},
        {"name": "Seeded pipelines match a stage oracle", "visibility": "unshown", "behavior": "protocol.validation", "failure_message": "Follow the stage order exactly: middleware chaining, guards only after allow, cancel and deny through post-execute, pipeline failures skipping it.", "code": r"""
import random
def err(m, c=None):
    return {"is_error": True, "content": [{"type": "text", "text": f"Error: {m}"}], "error": {"message": m, "code": c}}
class Boom(Exception):
    pass
for seed in (11, 37, 70):
    rng = random.Random(seed)
    for trial in range(80):
        name = rng.choice(["ok", "bad", "ghost"])
        tools = {"ok": lambda a: [{"type": "text", "text": "ok"}], "bad": lambda a: (_ for _ in ()).throw(ValueError("tool broke"))}
        pre_kind = rng.choice(["pass", "allow", "deny", "cancel", "raise"])
        guard_kind = rng.choice(["none", "none", "deny", "raise"])
        around_kind = rng.choice(["pass", "wrap", "raise"])
        post_kind = rng.choice(["pass", "accept_content", "block", "raise"])
        rt = {fn}(tools)
        if pre_kind == "allow":
            rt.pre_execute.append(lambda c, n: {"kind": "allow"})
        elif pre_kind == "deny":
            rt.pre_execute.append(lambda c, n: {"kind": "deny", "reason": "policy says no"})
        elif pre_kind == "cancel":
            rt.pre_execute.append(lambda c, n: {"kind": "cancel"})
        elif pre_kind == "raise":
            rt.pre_execute.append(lambda c, n: (_ for _ in ()).throw(Boom("pre crashed")))
        else:
            rt.pre_execute.append(lambda c, n: n())
        if guard_kind == "deny":
            rt.guards.append(lambda c: "guard says no")
        elif guard_kind == "raise":
            rt.guards.append(lambda c: (_ for _ in ()).throw(Boom("guard crashed")))
        if around_kind == "wrap":
            rt.around.append(lambda c, n: {**n(), "wrapped": True})
        elif around_kind == "raise":
            rt.around.append(lambda c, n: (_ for _ in ()).throw(Boom("around crashed")))
        else:
            rt.around.append(lambda c, n: n())
        if post_kind == "accept_content":
            rt.post_execute.append(lambda c, r, n: {"kind": "accept", "content": [{"type": "text", "text": "replaced"}], "additional_contexts": ["ctx"]})
        elif post_kind == "block":
            rt.post_execute.append(lambda c, r, n: {"kind": "block", "feedback": [{"type": "text", "text": "line1"}, {"type": "image"}, {"type": "text", "text": "line2"}]})
        elif post_kind == "raise":
            rt.post_execute.append(lambda c, r, n: (_ for _ in ()).throw(Boom("post crashed")))
        seen = []
        rt.observers.append(lambda c, r: (_ for _ in ()).throw(RuntimeError("observer broke")))
        rt.observers.append(lambda c, r: seen.append(r))
        got = rt.execute({"name": name, "arguments": {}})
        def expected():
            if pre_kind == "raise":
                return err("pre crashed")
            if pre_kind == "cancel":
                result = err("tool call aborted before dispatch", "ABORTED_BEFORE_DISPATCH")
            elif pre_kind == "deny":
                result = err("policy says no")
            else:
                if guard_kind == "raise":
                    return err("guard crashed")
                if guard_kind == "deny":
                    result = err("guard says no")
                else:
                    if around_kind == "raise":
                        return err("around crashed")
                    if name == "ok":
                        result = {"is_error": False, "content": [{"type": "text", "text": "ok"}]}
                    elif name == "bad":
                        result = err("tool broke")
                    else:
                        result = err('unknown tool "ghost"', "UNKNOWN_TOOL")
                    if around_kind == "wrap":
                        result = {**result, "wrapped": True}
            if post_kind == "raise":
                return err("post crashed")
            if post_kind == "block":
                return {"is_error": True, "content": [{"type": "text", "text": "line1"}, {"type": "image"}, {"type": "text", "text": "line2"}], "error": {"message": "line1\nline2", "code": None}}
            if post_kind == "accept_content":
                return {**result, "content": [{"type": "text", "text": "replaced"}], "additional_contexts": ["ctx"]}
            return result
        want = expected()
        assert got == want, (seed, trial, name, pre_kind, guard_kind, around_kind, post_kind, got, want)
        assert seen == [got], "observers must see the final result and one failing observer must not stop the others"
"""},
        {"name": "Middleware order and context merging", "visibility": "unshown", "behavior": "events.ordering", "failure_message": "Run middleware in registration order, keep the tool result's contexts before the decision's, and let block keep only its own contexts.", "code": r"""
trace = []
rt = {fn}({"t": lambda a: trace.append("body") or [{"type": "text", "text": "x"}]})
rt.pre_execute.append(lambda c, n: trace.append("pre1") or n())
rt.pre_execute.append(lambda c, n: trace.append("pre2") or n())
rt.around.append(lambda c, n: trace.append("around1") or n())
rt.around.append(lambda c, n: {**n(), "additional_contexts": ["from-tool"]})
rt.post_execute.append(lambda c, r, n: trace.append("post1") or n())
rt.post_execute.append(lambda c, r, n: {"kind": "accept", "additional_contexts": ["from-post"]})
out = rt.execute({"name": "t", "arguments": {}})
assert trace == ["pre1", "pre2", "around1", "body", "post1"], trace
assert out["additional_contexts"] == ["from-tool", "from-post"], out
rt2 = {fn}({"t": lambda a: []})
rt2.around.append(lambda c, n: {**n(), "additional_contexts": ["from-tool"]})
rt2.post_execute.append(lambda c, r, n: {"kind": "block", "feedback": [{"type": "text", "text": "no"}], "additional_contexts": ["why"]})
out = rt2.execute({"name": "t", "arguments": {}})
assert out["additional_contexts"] == ["why"] and out["is_error"] is True, out
rt3 = {fn}({"t": lambda a: []})
out = rt3.execute({"name": "t", "arguments": {}})
assert out == {"is_error": False, "content": []}, out
"""},
    ],
    "solution": '''def _chain(middleware, args, inner):
    queue = list(middleware)

    def next():
        if queue:
            return queue.pop(0)(*args, next)
        return inner()

    return next()


def _error(message, code=None):
    return {"is_error": True, "content": [{"type": "text", "text": f"Error: {message}"}],
            "error": {"message": message, "code": code}}


class ToolRuntime:
    def __init__(self, tools, approve=None):
        self.tools = tools
        self.approve = approve
        self.pre_execute = []
        self.guards = []
        self.around = []
        self.post_execute = []
        self.observers = []

    def _body(self, call):
        name = call["name"]
        if name not in self.tools:
            return _error(f'unknown tool "{name}"', "UNKNOWN_TOOL")
        try:
            return {"is_error": False, "content": self.tools[name](call["arguments"])}
        except Exception as error:
            return _error(str(error))

    def _resolve_ask(self, call, decision):
        name = call["name"]
        if self.approve is None:
            return {"kind": "deny", "reason": decision.get("reason") or f'tool "{name}" requires approval (not yet supported)'}
        outcome = self.approve(call, decision.get("reason"))
        if outcome == "allowed-once":
            return {"kind": "allow"}
        if outcome == "rejected":
            return {"kind": "deny", "reason": f'the user rejected tool "{name}"'}
        return {"kind": "deny", "reason": f'approval for tool "{name}" was cancelled'}

    def _post(self, call, result):
        decision = _chain(self.post_execute, (call, result), lambda: {"kind": "accept"})
        extra = list(decision.get("additional_contexts") or [])
        if decision["kind"] == "block":
            feedback = decision["feedback"]
            text = "\\n".join(b["text"] for b in feedback if b.get("type") == "text")
            blocked = {"is_error": True, "content": feedback, "error": {"message": text, "code": None}}
            if extra:
                blocked["additional_contexts"] = extra
            return blocked
        accepted = dict(result)
        if "content" in decision:
            accepted["content"] = decision["content"]
        contexts = list(result.get("additional_contexts") or []) + extra
        if contexts:
            accepted["additional_contexts"] = contexts
        return accepted

    def _run(self, call):
        decision = _chain(self.pre_execute, (call,), lambda: {"kind": "allow"})
        if decision["kind"] == "ask":
            decision = self._resolve_ask(call, decision)
        if decision["kind"] == "cancel":
            result = _error("tool call aborted before dispatch", "ABORTED_BEFORE_DISPATCH")
        else:
            reason = decision.get("reason") if decision["kind"] == "deny" else None
            if decision["kind"] == "allow":
                for guard in self.guards:
                    reason = guard(call)
                    if reason is not None:
                        break
            if reason is not None:
                result = _error(reason)
            else:
                result = _chain(self.around, (call,), lambda: self._body(call))
        return self._post(call, result)

    def execute(self, call):
        try:
            result = self._run(call)
        except Exception as error:
            result = _error(str(error))
        for observer in self.observers:
            try:
                observer(call, result)
            except Exception:
                pass
        return result
''',
    "interview_questions": interview(
        concept=[
            "Walk through what happens between a model emitting a tool call and the model seeing its result in a production agent harness.",
            "Why must every tool call produce exactly one result, even when a policy plugin crashes?",
        ],
        deep_dive=[
            "Why do guards run after all pre-execute middleware and only deny? What attack or bug does that prevent?",
            "Which failures go through post-execute and which skip it? Why does a denied call still go through post-execute?",
            "How is an ask decision resolved, and why deny by default when no approval channel exists?",
        ],
        tradeoffs=[
            "Converting every exception into an error result versus failing the turn: what helps the model, and what hides bugs from developers?",
            "Approval per call versus per session versus allow-lists: user friction, safety and prompt-injection risk?",
        ],
    ),
}
