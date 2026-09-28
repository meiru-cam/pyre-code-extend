"""Turn an admission decision into exactly one tool result: dispatch, then post-execute policy."""

from ._interview import interview

TASK = {
    "title": "Tool Dispatch and Result Policy",
    "difficulty": "Medium",
    "version": 1,
    "function_name": "settle_tool_call",
    "description_en": r"""Once a tool call has been admitted or refused, DeepSeek Harness produces exactly one result for it. Implement that second half of the pipeline.

**Signature:** `settle_tool_call(call, decision, tools, around, post_execute) -> dict`

**Parameters:**
- `call` — dict with `name` and `arguments`.
- `decision` — `{"kind": "allow"}`, `{"kind": "deny", "reason": str}` or `{"kind": "cancel"}`.
- `tools` — dict from tool name to `body(arguments) -> list` of content blocks. It may raise.
- `around` — list of middleware `f(call, next)` wrapped around the tool body.
- `post_execute` — list of middleware `f(call, result, next)`; after the last one, `next()` returns `{"kind": "accept"}`.

**Results.** Success is `{"is_error": False, "content": blocks}`. The *error result for message `m` with code `c`* is `{"is_error": True, "content": [{"type": "text", "text": f"Error: {m}"}], "error": {"message": m, "code": c}}`; `c` may be None.

**Step 1 — candidate result.**
- `cancel`: the error result for `"tool call aborted before dispatch"` with code `"ABORTED_BEFORE_DISPATCH"`.
- `deny`: the error result for its reason, code None. The tool does not run.
- `allow`: run the `around` chain; after the last middleware, `next()` runs the body step, which returns the success result, or the error result for `f'unknown tool "{name}"'` with code `"UNKNOWN_TOOL"` when the name is not in `tools`, or the error result for `str(e)` with code None when the body raises `e`.

**Step 2 — post-execute.** Run the `post_execute` chain on the candidate, whatever produced it.
- `{"kind": "accept", "content"?, "additional_contexts"?}` keeps the candidate, replaces its `content` if the decision has `content`, and sets `additional_contexts` to the candidate's contexts followed by the decision's.
- `{"kind": "block", "feedback": blocks, "additional_contexts"?}` returns `{"is_error": True, "content": feedback, "error": {"message": text, "code": None}}`, where `text` joins the `text` of the feedback's text blocks with newlines, plus only the decision's contexts.
- `additional_contexts` appears in a result only when it is non-empty.

Exceptions from `around` or `post_execute` functions propagate; only the tool body's exceptions become results.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why refused calls still get a result.** Model APIs require a result for every tool call, so a denial or cancellation is reported as an error result the model can read and react to.

**Why refusals go through post-execute.** Post-execute policies see every attempt. The repeat-call reminder counts denied calls there, since a model hammering a denied call is a loop.

**Why around-middleware.** Timeouts, retries and metrics wrap the body without the tool knowing; the next exercise builds the timeout wrapper on this seam.

**Why block keeps only its own contexts.** A blocked result replaces what the tool produced, so context the tool attached no longer applies.""",
    "advisory_prerequisites": ["dsh_tool_admission"],
    "hints": [
        {"level": 1, "kind": "questions", "content": "Which decisions skip the tool body but still reach post-execute? Where must the body's exceptions be caught so that around-middleware still sees a result? How do contexts merge on accept versus block?"},
        {"level": 2, "kind": "analysis", "content": "Write one chain helper that calls middleware with (*args, next) and falls back to an inner function. Build the candidate from the decision, running the around chain with a body step that catches its own exceptions. Then run post-execute and build a new dict for accept or block."},
    ],
    "model_connections": [
        "DeepSeek Harness's dispatchScheduledExecution runs the tools/execute waterfall around dispatchToolBody, and postExecute applies accept or block.",
        "Claude Code's PostToolUse hooks can likewise block a result and send feedback to the model.",
    ],
    "pro_con_analysis": {
        "pros": ["Every call yields one well-formed result, and wrappers and policies attach without changing tools."],
        "cons": ["Replacing or blocking results after the fact can hide what the tool really did unless the original is logged."],
    },
    "sources": [
        {"kind": "code", "url": "https://github.com/deepseek-ai/deepseek-harness", "commit": "21638c56315ae6a2b552d6091945d3144c9af32e", "path": "packages/core/tools/src/index.ts", "symbol": "ToolRuntime.dispatchScheduledExecution, dispatchToolBody, postExecute and toolAbortedBeforeDispatchResult", "license": "MIT", "adapted": "Cancel and deny results, the around chain over a body step that catches its own errors, UNKNOWN_TOOL, and accept and block semantics with context merging.", "simplifications": "Synchronous; no value snapshots, output schemas, content projectors, deferred contexts or cancellation signals."},
    ],
    "tests": [
        {"name": "Allowed call runs and post-execute adds context", "behavior": "protocol.validation", "code": r"""
tools = {"read": lambda args: [{"type": "text", "text": "file:" + args["path"]}]}
post = [lambda call, result, next: {**next(), "additional_contexts": ["note"]}]
out = {fn}({"name": "read", "arguments": {"path": "a.py"}}, {"kind": "allow"}, tools, [], post)
assert out == {"is_error": False, "content": [{"type": "text", "text": "file:a.py"}], "additional_contexts": ["note"]}, out
out = {fn}({"name": "read", "arguments": {"path": "a.py"}}, {"kind": "deny", "reason": "no"}, tools, [], [])
assert out == {"is_error": True, "content": [{"type": "text", "text": "Error: no"}], "error": {"message": "no", "code": None}}, out
"""},
        {"name": "Seeded settlements match an oracle", "visibility": "unshown", "behavior": "protocol.validation", "failure_message": "Build the candidate from the decision, catch only body errors, and apply accept or block exactly as stated.", "code": r"""
import random
def err(m, c=None):
    return {"is_error": True, "content": [{"type": "text", "text": f"Error: {m}"}], "error": {"message": m, "code": c}}
for seed in (8, 44, 91):
    rng = random.Random(seed)
    for trial in range(150):
        name = rng.choice(["ok", "bad", "ghost"])
        decision = rng.choice([{"kind": "allow"}, {"kind": "allow"}, {"kind": "deny", "reason": "nope"}, {"kind": "cancel"}])
        ran = []
        tools = {"ok": lambda a: ran.append(1) or [{"type": "text", "text": "ok"}],
                 "bad": lambda a: ran.append(1) or (_ for _ in ()).throw(ValueError("tool broke"))}
        around_kind = rng.choice(["none", "pass", "ctx"])
        around = [] if around_kind == "none" else [(lambda c, n: n()) if around_kind == "pass" else (lambda c, n: {**n(), "additional_contexts": ["t"]})]
        post_kind = rng.choice(["none", "accept", "content", "block"])
        post = {"none": [], "accept": [lambda c, r, n: n()],
                "content": [lambda c, r, n: {"kind": "accept", "content": [{"type": "text", "text": "new"}], "additional_contexts": ["p"]}],
                "block": [lambda c, r, n: {"kind": "block", "feedback": [{"type": "text", "text": "a"}, {"type": "image"}, {"type": "text", "text": "b"}], "additional_contexts": ["q"]}]}[post_kind]
        got = {fn}({"name": name, "arguments": {}}, decision, tools, around, post)
        if decision["kind"] == "cancel":
            cand = err("tool call aborted before dispatch", "ABORTED_BEFORE_DISPATCH")
        elif decision["kind"] == "deny":
            cand = err("nope")
        else:
            cand = {"ok": {"is_error": False, "content": [{"type": "text", "text": "ok"}]}, "bad": err("tool broke"),
                    "ghost": err('unknown tool "ghost"', "UNKNOWN_TOOL")}[name]
            if around_kind == "ctx":
                cand = {**cand, "additional_contexts": ["t"]}
        if post_kind == "block":
            want = {"is_error": True, "content": [{"type": "text", "text": "a"}, {"type": "image"}, {"type": "text", "text": "b"}], "error": {"message": "a\nb", "code": None}, "additional_contexts": ["q"]}
        elif post_kind == "content":
            want = {**cand, "content": [{"type": "text", "text": "new"}], "additional_contexts": cand.get("additional_contexts", []) + ["p"]}
        else:
            want = cand
        assert got == want, (seed, trial, name, decision, around_kind, post_kind, got, want)
        assert ran == ([1] if decision["kind"] == "allow" and name != "ghost" else []), (decision, name, ran)
"""},
        {"name": "Middleware order and propagation", "visibility": "unshown", "behavior": "events.ordering", "failure_message": "Run around and post-execute middleware in order around the body, and let their exceptions propagate.", "code": r"""
trace = []
tools = {"t": lambda a: trace.append("body") or []}
around = [lambda c, n: trace.append("a1") or n(), lambda c, n: trace.append("a2") or n()]
post = [lambda c, r, n: trace.append("p1") or n(), lambda c, r, n: trace.append(("p2", r["is_error"])) or n()]
out = {fn}({"name": "t", "arguments": {}}, {"kind": "allow"}, tools, around, post)
assert trace == ["a1", "a2", "body", "p1", ("p2", False)] and out == {"is_error": False, "content": []}, (trace, out)
def boom(*args):
    raise KeyError("x")
for around, post in (([boom], []), ([], [boom])):
    try:
        {fn}({"name": "t", "arguments": {}}, {"kind": "allow"}, tools, around, post)
    except KeyError:
        pass
    else:
        raise AssertionError("a middleware exception was swallowed")
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


def settle_tool_call(call, decision, tools, around, post_execute):
    name = call["name"]

    def body():
        if name not in tools:
            return _error(f'unknown tool "{name}"', "UNKNOWN_TOOL")
        try:
            return {"is_error": False, "content": tools[name](call["arguments"])}
        except Exception as error:
            return _error(str(error))

    if decision["kind"] == "cancel":
        result = _error("tool call aborted before dispatch", "ABORTED_BEFORE_DISPATCH")
    elif decision["kind"] == "deny":
        result = _error(decision["reason"])
    else:
        result = _chain(around, (call,), body)

    post = _chain(post_execute, (call, result), lambda: {"kind": "accept"})
    extra = list(post.get("additional_contexts") or [])
    if post["kind"] == "block":
        feedback = post["feedback"]
        text = "\\n".join(b["text"] for b in feedback if b.get("type") == "text")
        blocked = {"is_error": True, "content": feedback, "error": {"message": text, "code": None}}
        if extra:
            blocked["additional_contexts"] = extra
        return blocked
    accepted = dict(result)
    if "content" in post:
        accepted["content"] = post["content"]
    contexts = list(result.get("additional_contexts") or []) + extra
    if contexts:
        accepted["additional_contexts"] = contexts
    return accepted
''',
    "interview_questions": interview(
        concept=[
            "Why must every tool call, including a denied or cancelled one, produce exactly one result for the model?",
            "What is around-middleware on tool execution good for? Give three examples.",
        ],
        deep_dive=[
            "Why are the body's exceptions caught inside the innermost step rather than around the whole chain?",
            "Walk through accept versus block. Why does a blocked result drop the contexts the tool attached?",
            "Why do denied and cancelled calls still go through post-execute?",
        ],
        tradeoffs=[
            "Letting post-execute policies rewrite or block results versus only observing them: power versus auditability?",
            "Returning tool errors to the model versus retrying automatically inside the harness: which errors belong where?",
        ],
    ),
}
