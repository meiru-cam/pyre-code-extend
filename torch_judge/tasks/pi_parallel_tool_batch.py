"""Execute one assistant message's tool calls in parallel while keeping results in call order."""

from ._interview import interview

TASK = {
    "title": "Parallel Tool Call Batch",
    "difficulty": "Medium",
    "version": 1,
    "function_name": "execute_tool_batch",
    "description_en": r"""When the model asks for several tools in one message, the Pi agent runs them concurrently but reports results in the order the model asked. Implement that batch executor.

**Signature:** `async execute_tool_batch(calls, tools, mode="parallel", before_tool_call=None, is_aborted=lambda: False, emit=lambda event: None) -> dict`

**Parameters:**
- `calls` — list of `{"id", "name", "arguments"}`.
- `tools` — dict from name to `{"execute": async fn(arguments) -> {"content": [...], "terminate"?: bool}, "sequential"?: bool}`. `execute` may raise.
- `before_tool_call` — optional async function `(call) -> None | {"block": True, "reason"?: str, "terminate"?: bool}`.
- `is_aborted()` — whether the user cancelled.
- `emit(event)` — receives `{"type": "start", "id"}` and `{"type": "end", "id"}` events.

**Preparing one call:** emit `start`. If the tool is unknown, the outcome is an error with text `f"Tool {name} not found"`. Otherwise await `before_tool_call` if given; if `is_aborted()` is then true, the outcome is an error `"Operation aborted"`; if it returned a block, the outcome is an error with its `reason` or `"Tool execution was blocked"`, carrying `terminate` when the block says so. Otherwise the call is *prepared* and still has to run. Outcomes decided here count as finished: emit `end` right away.

**Running a prepared call:** await `execute(arguments)`. If it raises `e`, the outcome is an error with text `str(e)`; otherwise use its `content` and `terminate`. Emit `end` when it finishes.

An error outcome has `content = [{"type": "text", "text": ...}]`. Each outcome becomes `{"role": "toolResult", "tool_call_id", "tool_name", "content", "is_error"}`.

**Modes:**
- **Sequential** — used when `mode == "sequential"` or any call names a known tool with `"sequential": True`. For each call in order: prepare it, run it if prepared, record the outcome; then stop if `is_aborted()`.
- **Parallel** — first prepare every call in order, stopping after the current call if `is_aborted()`. Only then start running all prepared calls concurrently. A prepared call that starts running while `is_aborted()` is true does not execute; its outcome is an error `"Operation aborted"`.

**Returns** `{"messages": [...], "terminate": bool}` with one message per recorded call, in call order. `terminate` is True when there is at least one message and every outcome has `terminate` true.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why parallel.** Models often request several independent reads or searches at once. Running them concurrently cuts latency roughly to that of the slowest call.

**Why results stay in call order.** Each result must follow its call, and a deterministic order keeps the transcript reproducible even though completion order varies.

**Why prepare first.** Approval prompts and policy hooks run in `before_tool_call`. Doing all of them before any tool starts means the user answers every prompt for the batch up front, and a block cannot race with an already running sibling.

**Why a sequential escape hatch.** Some tools, such as an interactive shell, must never overlap with others; declaring one forces the whole batch to run in order.""",
    "advisory_prerequisites": ["pi_agent_loop", "pi_file_mutation_queue"],
    "hints": [
        {"level": 1, "kind": "questions", "content": "Which work happens before any tool starts in parallel mode? How do you keep results in call order when they finish in a different order? When is the abort flag checked?"},
        {"level": 2, "kind": "analysis", "content": "Write prepare(call) returning either a finished outcome or a marker that the call is ready, and run(call) that executes and catches exceptions. Sequential mode loops over both. Parallel mode builds a list of finished outcomes or coroutines while preparing, then awaits asyncio.gather over them, which preserves order."},
    ],
    "model_connections": [
        "Pi's executeToolCallsParallel and executeToolCallsSequential; Claude Code and the OpenAI Agents SDK also run independent tool calls concurrently and return results in call order.",
    ],
    "pro_con_analysis": {
        "pros": ["Lower latency for independent calls, deterministic transcripts and up-front policy decisions."],
        "cons": ["Concurrent tools can conflict on shared state such as files, and one slow tool still delays the whole batch's results."],
    },
    "sources": [
        {"kind": "code", "url": "https://github.com/badlogic/pi-mono", "commit": "6f7551516b84278eb9da1c340c8e7bc66be1a6ba", "path": "packages/agent/src/agent-loop.ts", "symbol": "executeToolCalls, executeToolCallsParallel, executeToolCallsSequential, prepareToolCall and shouldTerminateToolBatch", "license": "MIT", "adapted": "Mode selection, prepare-then-run ordering, immediate outcomes for unknown or blocked calls, abort checks, ordered results and the all-terminate rule.", "simplifications": "No argument validation, afterToolCall hook, update streaming or result events beyond start and end."},
    ],
    "tests": [
        {"name": "Runs concurrently and returns results in call order", "behavior": "scheduler.concurrency", "code": r"""
import asyncio, time
async def main():
    async def slow(args):
        await asyncio.sleep(args["t"]); return {"content": [{"type": "text", "text": str(args["t"])}]}
    tools = {"sleep": {"execute": slow}}
    calls = [{"id": "a", "name": "sleep", "arguments": {"t": 0.2}}, {"id": "b", "name": "sleep", "arguments": {"t": 0.05}}]
    events = []
    start = time.perf_counter()
    out = await {fn}(calls, tools, emit=events.append)
    assert time.perf_counter() - start < 0.35
    assert [m["tool_call_id"] for m in out["messages"]] == ["a", "b"] and out["terminate"] is False, out
    assert out["messages"][1] == {"role": "toolResult", "tool_call_id": "b", "tool_name": "sleep", "content": [{"type": "text", "text": "0.05"}], "is_error": False}
    assert events == [{"type": "start", "id": "a"}, {"type": "start", "id": "b"}, {"type": "end", "id": "b"}, {"type": "end", "id": "a"}], events
asyncio.run(main())
"""},
        {"name": "Prepares every call before any runs, and unknown or blocked calls finish early", "visibility": "unshown", "behavior": "events.ordering", "failure_message": "In parallel mode run every before_tool_call first, emit end at once for calls decided during preparation, and only then start the prepared calls.", "code": r"""
import asyncio
async def main():
    log = []
    async def run(args):
        log.append(("run", args["i"])); await asyncio.sleep(0.01); return {"content": []}
    async def before(call):
        log.append(("before", call["id"]))
        if call["id"] == "c":
            return {"block": True}
        if call["id"] == "d":
            return {"block": True, "reason": "no network", "terminate": True}
        return None
    calls = [{"id": x, "name": "ghost" if x == "g" else "t", "arguments": {"i": x}} for x in ["a", "g", "c", "d", "e"]]
    events = []
    out = await {fn}(calls, {"t": {"execute": run}}, before_tool_call=before, emit=events.append)
    assert log[:4] == [("before", "a"), ("before", "c"), ("before", "d"), ("before", "e")], log
    assert sorted(log[4:]) == [("run", "a"), ("run", "e")], log
    texts = [m["content"][0]["text"] if m["is_error"] else None for m in out["messages"]]
    assert texts == [None, "Tool ghost not found", "Tool execution was blocked", "no network", None], out
    ends = [e["id"] for e in events if e["type"] == "end"]
    assert ends[:3] == ["g", "c", "d"] and sorted(ends[3:]) == ["a", "e"], events
asyncio.run(main())
"""},
        {"name": "Sequential tools force ordered execution and abort stops the batch", "visibility": "unshown", "behavior": "scheduler.concurrency", "failure_message": "One sequential tool makes the whole batch sequential; abort stops after the current call, and a prepared call that starts after abort is not executed.", "code": r"""
import asyncio
async def main():
    active, peak, ran = [0], [0], []
    async def work(args):
        active[0] += 1; peak[0] = max(peak[0], active[0]); ran.append(args["i"])
        await asyncio.sleep(0.02); active[0] -= 1
        return {"content": [], "terminate": True}
    tools = {"fast": {"execute": work}, "shell": {"execute": work, "sequential": True}}
    calls = [{"id": str(i), "name": "shell" if i == 1 else "fast", "arguments": {"i": i}} for i in range(3)]
    out = await {fn}(calls, tools)
    assert peak[0] == 1 and ran == [0, 1, 2] and out["terminate"] is True, (peak, ran, out)
    peak[0] = 0
    await {fn}([calls[0], calls[2]], tools)
    assert peak[0] == 2, "prepared calls must run concurrently in parallel mode"
    ran.clear()
    out = await {fn}(calls[:1] + calls[2:], tools, mode="sequential")
    assert ran == [0, 2]
    aborted = [False]
    async def aborting(args):
        ran.append(args["i"]); aborted[0] = True; return {"content": []}
    ran.clear()
    tools2 = {"t": {"execute": aborting, "sequential": True}}
    calls2 = [{"id": str(i), "name": "t", "arguments": {"i": i}} for i in range(3)]
    out = await {fn}(calls2, tools2, is_aborted=lambda: aborted[0])
    assert ran == [0] and len(out["messages"]) == 1, (ran, out)
    ran.clear(); aborted[0] = False
    async def before(call):
        if call["id"] == "1":
            aborted[0] = True
        return None
    out = await {fn}(calls2, {"t": {"execute": aborting}}, before_tool_call=before, is_aborted=lambda: aborted[0])
    assert ran == [] and [m["content"][0]["text"] for m in out["messages"]] == ["Operation aborted", "Operation aborted"], (ran, out)
asyncio.run(main())
"""},
        {"name": "Seeded batches match an ordering oracle", "visibility": "unshown", "behavior": "scheduler.concurrency", "failure_message": "Every recorded call yields one message in call order, errors carry the stated text, and terminate needs every outcome to terminate.", "code": r"""
import asyncio, random
async def main():
    for seed in (5, 26, 74):
        rng = random.Random(seed)
        for trial in range(40):
            async def ex(args):
                await asyncio.sleep(args["d"])
                if args["kind"] == "raise":
                    raise RuntimeError("fail " + args["id"])
                return {"content": [{"type": "text", "text": args["id"]}], "terminate": args["kind"] == "term"}
            async def before(call):
                return {"block": True, "terminate": call["arguments"]["bterm"]} if call["arguments"]["block"] else None
            calls, want = [], []
            for i in range(rng.randint(1, 5)):
                cid = f"c{i}"
                name = rng.choice(["t", "t", "t", "nope"])
                kind = rng.choice(["ok", "term", "term", "raise"])
                block = rng.random() < 0.2
                bterm = rng.random() < 0.5
                calls.append({"id": cid, "name": name, "arguments": {"id": cid, "d": rng.random() * 0.01, "kind": kind, "block": block, "bterm": bterm}})
                if name == "nope":
                    want.append((cid, True, f"Tool nope not found", False))
                elif block:
                    want.append((cid, True, "Tool execution was blocked", bterm))
                elif kind == "raise":
                    want.append((cid, True, "fail " + cid, False))
                else:
                    want.append((cid, False, cid, kind == "term"))
            out = await {fn}(calls, {"t": {"execute": ex}}, mode=rng.choice(["parallel", "sequential"]), before_tool_call=before)
            got = [(m["tool_call_id"], m["is_error"], m["content"][0]["text"]) for m in out["messages"]]
            assert got == [w[:3] for w in want], (seed, trial, got, want)
            assert out["terminate"] == all(w[3] for w in want), (seed, trial, out, want)
asyncio.run(main())
"""},
    ],
    "solution": '''import asyncio


def _error(text, terminate=False):
    return {"content": [{"type": "text", "text": text}], "is_error": True, "terminate": terminate}


async def execute_tool_batch(calls, tools, mode="parallel", before_tool_call=None,
                             is_aborted=lambda: False, emit=lambda event: None):
    async def prepare(call):
        emit({"type": "start", "id": call["id"]})
        tool = tools.get(call["name"])
        if tool is None:
            outcome = _error(f"Tool {call['name']} not found")
        else:
            outcome = None
            if before_tool_call is not None:
                decision = await before_tool_call(call)
                if is_aborted():
                    outcome = _error("Operation aborted")
                elif decision and decision.get("block"):
                    outcome = _error(decision.get("reason") or "Tool execution was blocked",
                                     decision.get("terminate") is True)
        if outcome is not None:
            emit({"type": "end", "id": call["id"]})
        return outcome

    async def run(call, check_abort):
        if check_abort and is_aborted():
            outcome = _error("Operation aborted")
        else:
            try:
                result = await tools[call["name"]]["execute"](call["arguments"])
                outcome = {"content": result["content"], "is_error": False,
                           "terminate": result.get("terminate") is True}
            except Exception as error:
                outcome = _error(str(error))
        emit({"type": "end", "id": call["id"]})
        return outcome

    sequential = mode == "sequential" or any(
        tools.get(c["name"], {}).get("sequential") for c in calls)
    recorded = []
    if sequential:
        for call in calls:
            outcome = await prepare(call)
            if outcome is None:
                outcome = await run(call, False)
            recorded.append((call, outcome))
            if is_aborted():
                break
    else:
        pending = []
        for call in calls:
            outcome = await prepare(call)
            pending.append((call, outcome))
            if is_aborted():
                break

        async def settle(call, outcome):
            return outcome if outcome is not None else await run(call, True)

        outcomes = await asyncio.gather(*(settle(c, o) for c, o in pending))
        recorded = list(zip([c for c, _ in pending], outcomes))

    messages = [{"role": "toolResult", "tool_call_id": call["id"], "tool_name": call["name"],
                 "content": outcome["content"], "is_error": outcome["is_error"]}
                for call, outcome in recorded]
    terminate = bool(recorded) and all(outcome["terminate"] for _, outcome in recorded)
    return {"messages": messages, "terminate": terminate}
''',
    "interview_questions": interview(
        concept=[
            "Why do agent harnesses run a model's tool calls in parallel, and what must stay deterministic when they do?",
            "What kinds of tools must never run concurrently with others?",
        ],
        deep_dive=[
            "Why run every before_tool_call hook before any tool starts? What race would appear otherwise?",
            "How do you return results in call order when calls finish in a different order?",
            "Where is the abort flag checked, and what result does each skipped or aborted call get?",
        ],
        tradeoffs=[
            "Unbounded parallelism versus a bounded pool versus fully sequential execution: latency, rate limits and resource contention?",
            "Declaring sequential per tool versus detecting conflicts from arguments (same file, same process): precision and complexity?",
        ],
    ),
}
