"""Agent loop with steering and follow-up message queues."""

from ._interview import interview

TASK = {
    "title": "Agent Loop with Steering and Follow-ups",
    "difficulty": "Medium",
    "version": 1,
    "function_name": "agent_loop",
    "description_en": r"""Implement the Pi agent's main loop: call the model, run the tools it asks for, and fold in messages the user typed while the agent was working.

**Signature:** `agent_loop(prompts, model, execute_tool, get_steering, get_follow_up) -> list[dict]`

**Parameters:**
- `prompts` — list of initial messages.
- `model(context)` — given a copy of the context list, returns an assistant message `{"role": "assistant", "content": [...], "stop_reason": str}`. Content blocks with `"type": "toolCall"` have `id`, `name` and `arguments`.
- `execute_tool(call)` — runs one tool call and returns `{"content": [...], "is_error"?: bool, "terminate"?: bool}`; it may raise.
- `get_steering()` / `get_follow_up()` — each returns a list of queued user messages, possibly empty. Steering is delivered while the agent is working; follow-ups are delivered only when it would otherwise stop.

**Loop.** Keep a `context` list and a `new_messages` list; both start as copies of `prompts`. Set `pending = get_steering()`.
- Outer loop: set `more_tools = True`, then run the inner loop.
- Inner loop, while `more_tools` or `pending` is non-empty:
  - append each pending message to `context` and `new_messages`; clear `pending`;
  - call `model` with a copy of `context` and append the assistant message to both lists;
  - if its `stop_reason` is `"error"` or `"aborted"`, return `new_messages` immediately;
  - for each tool call in order, create a result message `{"role": "toolResult", "tool_call_id": id, "tool_name": name, "content": ..., "is_error": ...}`. If `stop_reason` is `"length"`, do not execute any call: each result is an error whose content is `[{"type": "text", "text": f'Tool call "{name}" was not executed: the response hit the output token limit, so its arguments may be truncated. Re-issue the tool call with complete arguments.'}]`. Otherwise call `execute_tool`; if it raises `e`, the result is an error with content `[{"type": "text", "text": str(e)}]`; if not, use its `content` and `bool(is_error)`. Append every result to both lists;
  - `more_tools` is True when there was at least one tool call and not every call's result had `terminate` true (exceptions and length-stop failures count as not terminating);
  - `pending = get_steering()`.
- After the inner loop: `follow = get_follow_up()`. If it is non-empty, set `pending = follow` and repeat the outer loop; otherwise return `new_messages`.

**Constraints:** Each call to `model` must receive a list that later loop steps do not modify.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Steering versus follow-up.** A user who notices the agent going the wrong way can type a correction that is injected before the next model call, without waiting for the task to finish. A follow-up is a new request queued for after the agent is done. Pi's TUI exposes the two as different keys.

**Why truncated tool calls are not executed.** When the output hits the token limit, the last tool call's JSON arguments may be cut off, and executing a half-written `bash` command or `write` payload is dangerous. Returning an error lets the model re-issue the call.

**Why a tool error is a message, not an exception.** The model can read the error and recover, for example by fixing a path, which is usually better than aborting the whole run.""",
    "advisory_prerequisites": ["budgeted_agent_loop"],
    "hints": [
        {"level": 1, "kind": "questions", "content": "What keeps the inner loop running: tool calls, steering, or both? When is the follow-up queue consulted? What should happen to tool calls in a response that was cut off by the token limit?"},
        {"level": 2, "kind": "analysis", "content": "Write two nested while loops exactly as described. Inside, a helper builds one toolResult per call, catching exceptions from execute_tool. Track whether every result terminated. Pass list(context) to the model so later appends do not change what it saw."},
    ],
    "model_connections": [
        "Pi's agent-loop runLoop implements steering and follow-up queues; Claude Code and Codex let users interrupt or queue messages in the same way.",
    ],
    "pro_con_analysis": {
        "pros": ["Users can correct a running agent without cancelling it, and malformed or failing tool calls become recoverable messages."],
        "cons": ["Steering can land between a tool result and the next model call in ways the model finds confusing, and queued follow-ups can pile up behind a long task."],
    },
    "sources": [
        {"kind": "code", "url": "https://github.com/badlogic/pi-mono", "commit": "6f7551516b84278eb9da1c340c8e7bc66be1a6ba", "path": "packages/agent/src/agent-loop.ts", "symbol": "runLoop, failToolCallsFromTruncatedMessage and shouldTerminateToolBatch", "license": "MIT", "adapted": "Inner loop over tool calls and steering, outer loop over follow-ups, error and aborted early exit, length-stop tool failure, and all-terminate batch rule.", "simplifications": "Synchronous, sequential tools only; no events, streaming, prepareNextTurn, finishTurn decisions or explicit continuation."},
    ],
    "tests": [
        {"name": "Runs tools until the model stops", "behavior": "rl.trajectory", "code": r"""
replies = [
    {"role": "assistant", "content": [{"type": "toolCall", "id": "1", "name": "ls", "arguments": {}}], "stop_reason": "toolUse"},
    {"role": "assistant", "content": [{"type": "text", "text": "done"}], "stop_reason": "stop"},
]
seen = []
def model(ctx):
    seen.append(ctx); return replies[len(seen) - 1]
out = {fn}([{"role": "user", "content": "list"}], model, lambda call: {"content": [{"type": "text", "text": "a.py"}]}, lambda: [], lambda: [])
assert [m["role"] for m in out] == ["user", "assistant", "toolResult", "assistant"], out
assert out[2] == {"role": "toolResult", "tool_call_id": "1", "tool_name": "ls", "content": [{"type": "text", "text": "a.py"}], "is_error": False}
assert len(seen[0]) == 1 and len(seen[1]) == 3, [len(s) for s in seen]
"""},
        {"name": "Matches a seeded script oracle", "visibility": "unshown", "behavior": "rl.trajectory", "failure_message": "Follow the inner and outer loop exactly: steering before each model call, follow-ups only when idle, and early return on error or aborted.", "code": r"""
import copy, random
LEN = 'Tool call "{}" was not executed: the response hit the output token limit, so its arguments may be truncated. Re-issue the tool call with complete arguments.'
def make_world(seed):
    rng = random.Random(seed)
    replies, steering, follow, outcomes = [], [], [], {}
    for k in range(80):
        calls = [{"type": "toolCall", "id": f"{k}.{j}", "name": rng.choice(["a", "b"]), "arguments": {"k": k}} for j in range(rng.choice([0, 0, 1, 2]))]
        stop = rng.choices(["stop", "toolUse", "length", "error", "aborted"], weights=[4, 6, 1, 0.15, 0.15])[0]
        replies.append({"role": "assistant", "content": [{"type": "text", "text": str(k)}] + calls, "stop_reason": stop})
        for c in calls:
            outcomes[c["id"]] = rng.choice(["ok", "err", "raise", "term"])
    for _ in range(200):
        steering.append([{"role": "user", "content": f"s{rng.random():.3f}"}] if rng.random() < 0.15 else [])
        follow.append([{"role": "user", "content": f"f{rng.random():.3f}"}] if rng.random() < 0.3 else [])
    return replies, steering, follow, outcomes
def run_tool(outcomes, call):
    o = outcomes[call["id"]]
    if o == "raise":
        raise RuntimeError("boom " + call["id"])
    return {"content": [{"type": "text", "text": call["id"]}], "is_error": o == "err", "terminate": o == "term"}
def oracle(prompts, replies, steering, follow, outcomes):
    s, f, r = list(steering), list(follow), list(replies)
    ctx, new = list(prompts), list(prompts)
    pending = s.pop(0) if s else []
    while True:
        more = True
        while more or pending:
            ctx += pending; new += pending; pending = []
            msg = r.pop(0); ctx.append(msg); new.append(msg)
            if msg["stop_reason"] in ("error", "aborted"):
                return new
            calls = [b for b in msg["content"] if b["type"] == "toolCall"]
            terms = []
            for c in calls:
                if msg["stop_reason"] == "length":
                    res = {"role": "toolResult", "tool_call_id": c["id"], "tool_name": c["name"], "content": [{"type": "text", "text": LEN.format(c["name"])}], "is_error": True}
                    terms.append(False)
                else:
                    try:
                        o = run_tool(outcomes, c)
                        res = {"role": "toolResult", "tool_call_id": c["id"], "tool_name": c["name"], "content": o["content"], "is_error": bool(o.get("is_error"))}
                        terms.append(bool(o.get("terminate")))
                    except Exception as e:
                        res = {"role": "toolResult", "tool_call_id": c["id"], "tool_name": c["name"], "content": [{"type": "text", "text": str(e)}], "is_error": True}
                        terms.append(False)
                ctx.append(res); new.append(res)
            more = bool(calls) and not all(terms)
            pending = s.pop(0) if s else []
        fu = f.pop(0) if f else []
        if not fu:
            return new
        pending = fu
for seed, world in ((s, w) for s in (7, 24, 63) for w in range(8)):
    replies, steering, follow, outcomes = make_world(seed * 100 + world)
    prompts = [{"role": "user", "content": "go"}]
    want = oracle(prompts, replies, steering, follow, outcomes)
    r, s, f = list(replies), list(steering), list(follow)
    snapshots = []
    def model(ctx):
        snapshots.append((ctx, copy.deepcopy(ctx))); return r.pop(0)
    got = {fn}(prompts, model, lambda c: run_tool(outcomes, c), lambda: s.pop(0) if s else [], lambda: f.pop(0) if f else [])
    assert got == want, (seed, got, want)
    assert all(a == b for a, b in snapshots), "a context passed to the model changed afterwards"
    assert prompts == [{"role": "user", "content": "go"}]
"""},
        {"name": "Length stops fail tools without running them", "visibility": "unshown", "behavior": "rl.trajectory", "failure_message": "Never execute tool calls from a response cut off by the token limit; return an error result for each and keep looping.", "code": r"""
ran = []
replies = iter([
    {"role": "assistant", "content": [{"type": "toolCall", "id": "w", "name": "write", "arguments": {"body": "trunc"}}], "stop_reason": "length"},
    {"role": "assistant", "content": [], "stop_reason": "stop"},
])
out = {fn}([], lambda ctx: next(replies), lambda c: ran.append(c) or {"content": []}, lambda: [], lambda: [])
assert ran == [], ran
assert out[1]["is_error"] is True and 'Tool call "write" was not executed' in out[1]["content"][0]["text"], out
assert out[-1]["stop_reason"] == "stop" and len(out) == 3, out
"""},
        {"name": "Terminate needs every result and steering is injected mid-run", "visibility": "unshown", "behavior": "rl.trajectory", "failure_message": "Stop tool looping only when every result terminates, and inject steering before the next model call.", "code": r"""
replies = iter([
    {"role": "assistant", "content": [{"type": "toolCall", "id": "1", "name": "t", "arguments": {}}, {"type": "toolCall", "id": "2", "name": "t", "arguments": {}}], "stop_reason": "toolUse"},
    {"role": "assistant", "content": [{"type": "toolCall", "id": "3", "name": "t", "arguments": {}}], "stop_reason": "toolUse"},
])
results = {"1": {"content": [], "terminate": True}, "2": {"content": []}, "3": {"content": [], "terminate": True}}
steer = iter([[], [{"role": "user", "content": "use v2"}], []])
out = {fn}([{"role": "user", "content": "go"}], lambda ctx: next(replies), lambda c: results[c["id"]], lambda: next(steer), lambda: [])
assert [m.get("content") if m["role"] == "user" else m["role"] for m in out] == ["go", "assistant", "toolResult", "toolResult", "use v2", "assistant", "toolResult"], out
"""},
    ],
    "solution": '''LENGTH_TEXT = ('Tool call "{}" was not executed: the response hit the output token limit, so its '
               'arguments may be truncated. Re-issue the tool call with complete arguments.')


def _run_tools(message, execute_tool):
    results, terminates = [], []
    for call in (b for b in message["content"] if b["type"] == "toolCall"):
        base = {"role": "toolResult", "tool_call_id": call["id"], "tool_name": call["name"]}
        if message["stop_reason"] == "length":
            results.append({**base, "content": [{"type": "text", "text": LENGTH_TEXT.format(call["name"])}], "is_error": True})
            terminates.append(False)
            continue
        try:
            outcome = execute_tool(call)
        except Exception as error:
            results.append({**base, "content": [{"type": "text", "text": str(error)}], "is_error": True})
            terminates.append(False)
            continue
        results.append({**base, "content": outcome["content"], "is_error": bool(outcome.get("is_error"))})
        terminates.append(bool(outcome.get("terminate")))
    return results, bool(results) and not all(terminates)


def agent_loop(prompts, model, execute_tool, get_steering, get_follow_up):
    context = list(prompts)
    new_messages = list(prompts)
    pending = get_steering()
    while True:
        more_tools = True
        while more_tools or pending:
            for message in pending:
                context.append(message)
                new_messages.append(message)
            pending = []
            reply = model(list(context))
            context.append(reply)
            new_messages.append(reply)
            if reply["stop_reason"] in ("error", "aborted"):
                return new_messages
            results, more_tools = _run_tools(reply, execute_tool)
            context.extend(results)
            new_messages.extend(results)
            pending = get_steering()
        follow = get_follow_up()
        if not follow:
            return new_messages
        pending = follow
''',
    "interview_questions": interview(
        concept=[
            "Describe the basic agent loop. What makes it stop?",
            "What is the difference between a steering message and a follow-up message, and why treat them differently?",
        ],
        deep_dive=[
            "Walk through when each queue is polled. What happens if the user steers while tools are running?",
            "Why are tool calls in a response that hit the output token limit not executed?",
            "Why turn tool exceptions into error results instead of raising, and why stop immediately on an errored or aborted model response?",
        ],
        tradeoffs=[
            "Injecting steering mid-run versus cancelling and restarting the turn: context coherence, latency and wasted work?",
            "Sequential versus parallel tool execution inside one step: speed, ordering of results, and conflicts between tools?",
        ],
    ),
}
