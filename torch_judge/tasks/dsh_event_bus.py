"""Cordis-style event bus: five dispatch modes and reversible listener registration."""

from ._interview import interview

TASK = {
    "title": "Plugin Event Bus with Dispatch Modes",
    "difficulty": "Medium",
    "version": 1,
    "function_name": "EventBus",
    "description_en": r"""Implement the event bus that every DeepSeek Harness plugin talks through.

**Signature:** `class EventBus` with the methods below. Listeners are plain callables; `serial` and `parallel` also accept `async def` listeners.

**Registration:**
- `on(name, listener, prepend=False) -> dispose` — append the listener to the event's list, or put it first when `prepend` is True. Return a zero-argument `dispose` function that removes this registration and returns True, or returns False if it was already removed.
- `once(name, listener, prepend=False) -> dispose` — like `on`, but the registration removes itself right before the listener runs for the first time.

**Dispatch.** Every mode first takes a snapshot of the event's listener list. Listeners added or removed during a dispatch do not change which listeners that dispatch calls. A value is a *bail value* when it is not `None` and not `False`.
- `emit(name, *args) -> None` — call every listener in order with `*args`; ignore return values.
- `bail(name, *args)` — call listeners in order and return the first bail value; return None if there is none.
- `async serial(name, *args)` — call listeners in order, awaiting each result if it is awaitable, and return the first bail value; return None if there is none.
- `async parallel(name, *args) -> None` — start every listener concurrently and wait for all of them. If any raised, raise an `ExceptionGroup` holding every raised exception, after all listeners have finished.
- `waterfall(name, *args, inner)` — around-middleware. Each listener is called as `listener(*args, next)`. Calling `next()` with no arguments runs the next listener in order, or `inner(*args)` after the last one, and returns its value. The first listener's return value is the result. A listener that returns without calling `next()` skips everything after it, including `inner`.

**Constraints:**
- An exception raised by a listener in `emit`, `bail`, `serial` or `waterfall` propagates to the caller.
- Events are independent: dispatching one name never calls listeners of another.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why a bus.** In DeepSeek Harness there is no privileged core. The model adapter, tool registry, session log and even the agent loop are plugins that register listeners and services on a shared context, so any of them can be swapped from configuration.

**Why five modes.** The dispatch mode is part of an event's contract. `emit` is fire-and-forget observation. `bail` and `serial` let the first listener that owns a decision answer it. `parallel` fans out independent work. `waterfall` is middleware: `tools/pre-execute`, `tools/execute` and `tools/post-execute` are waterfalls, which is how timeouts, approval prompts and repeat-call reminders wrap tool execution without touching the loop.

**Why disposers.** Every registration is an effect that returns its own undo. When a plugin unloads or hot-reloads, its listeners disappear with it.""",
    "advisory_prerequisites": ["tool_registry"],
    "hints": [
        {"level": 1, "kind": "questions", "content": "Why take a copy of the listener list before calling anyone? How can next() know which listener comes after the current one? How do you tell a disposer that already ran from one that has not?"},
        {"level": 2, "kind": "analysis", "content": "Store a list of registration records per name; dispose removes its own record by identity. For waterfall, copy the list into a queue and define next() that pops the head and calls it with (*args, next), falling back to inner(*args). For parallel use asyncio.gather(..., return_exceptions=True) and wrap every exception in one ExceptionGroup."},
    ],
    "model_connections": [
        "DeepSeek Harness builds on its vendored Cordis framework: tools/pre-execute, tools/execute and agent/pre-step are waterfalls, agent/turn-stopping is serial.",
        "The same around-middleware shape appears in Koa, Express and ASGI middleware stacks.",
    ],
    "pro_con_analysis": {
        "pros": ["Plugins extend behavior without importing each other or patching the loop, and registrations unwind cleanly on unload."],
        "cons": ["Control flow becomes implicit: the order of registration decides semantics, and a waterfall listener that forgets next() silently vetoes everything after it."],
    },
    "sources": [
        {"kind": "code", "url": "https://github.com/deepseek-ai/deepseek-harness", "commit": "21638c56315ae6a2b552d6091945d3144c9af32e", "path": "vendor/cordis/src/events.ts", "symbol": "EventsService.dispatch, emit, bail, serial, parallel, waterfall, on and once", "license": "MIT", "adapted": "Dispatch snapshot, bail-value rule, waterfall next() chaining, AggregateError for parallel, and disposer-returning registration.", "simplifications": "No fibers, context filters, this-binding, internal/* events or global hooks."},
    ],
    "tests": [
        {"name": "Waterfall wraps the inner call", "behavior": "events.ordering", "code": r"""
bus = {fn}()
calls = []
def outer(x, next):
    calls.append("outer-in"); r = next(); calls.append("outer-out"); return r + 1
def inner_listener(x, next):
    calls.append("inner"); return next() * 10
bus.on("tools/execute", outer)
bus.on("tools/execute", inner_listener)
out = bus.waterfall("tools/execute", 2, inner=lambda x: x + 3)
assert out == 51, out
assert calls == ["outer-in", "inner", "outer-out"], calls
"""},
        {"name": "Bail, prepend and dispose", "behavior": "events.ordering", "code": r"""
bus = {fn}()
bus.on("pick", lambda x: None)
dispose = bus.on("pick", lambda x: x * 2)
bus.on("pick", lambda x: "late")
bus.on("pick", lambda x: False, prepend=True)
assert bus.bail("pick", 4) == 8
assert dispose() is True and dispose() is False
assert bus.bail("pick", 4) == "late"
assert bus.bail("nobody", 1) is None
"""},
        {"name": "Waterfall short-circuit and snapshot semantics", "visibility": "unshown", "behavior": "events.ordering", "failure_message": "A listener that skips next() vetoes the rest; listeners added during a dispatch wait for the next dispatch.", "code": r"""
bus = {fn}()
seen = []
def veto(req, next):
    seen.append("veto")
    bus.on("pre", lambda req, next: seen.append("added") or next())
    return {"kind": "deny"}
bus.on("pre", veto)
bus.on("pre", lambda req, next: seen.append("second") or next())
out = bus.waterfall("pre", {}, inner=lambda req: seen.append("inner") or {"kind": "allow"})
assert out == {"kind": "deny"} and seen == ["veto"], (out, seen)
seen.clear()
bus2 = {fn}()
assert bus2.waterfall("empty", 3, inner=lambda x: x * 7) == 21
order = []
def remover(x, next):
    order.append("a"); d2(); return next()
bus2.on("chain", remover)
d2 = bus2.on("chain", lambda x, next: order.append("b") or next())
bus2.waterfall("chain", 0, inner=lambda x: order.append("inner"))
assert order == ["a", "b", "inner"], order
order.clear()
bus2.waterfall("chain", 0, inner=lambda x: order.append("inner"))
assert order == ["a", "inner"], order
"""},
        {"name": "Once removes itself before running and emit snapshots", "visibility": "unshown", "behavior": "state.invariant", "failure_message": "once must unregister before calling the listener, and emit must call exactly the listeners present when it started.", "code": r"""
bus = {fn}()
log = []
def reenter(v):
    log.append(("once", v))
    if v < 3:
        bus.emit("tick", v + 1)
d = bus.once("tick", reenter)
bus.on("tick", lambda v: log.append(("on", v)))
bus.emit("tick", 1)
assert log == [("once", 1), ("on", 2), ("on", 1)], log
assert d() is False
log.clear()
bus.emit("tick", 9)
assert log == [("on", 9)], log
b = {fn}()
got = []
b.on("a", lambda: got.append("a"))
b.on("b", lambda: got.append("b"))
b.emit("a")
assert got == ["a"], got
seq = []
def adder():
    seq.append(1); b.on("c", lambda: seq.append(2))
b.on("c", adder)
b.emit("c")
assert seq == [1], seq
p = {fn}()
order2 = []
p.on("x", lambda: order2.append(1))
p.on("x", lambda: order2.append(0), prepend=True)
p.on("x", lambda: order2.append(2))
p.emit("x")
assert order2 == [0, 1, 2], order2
"""},
        {"name": "Serial awaits in order and stops at the first bail value", "visibility": "unshown", "behavior": "events.ordering", "failure_message": "serial must await each listener before starting the next and return the first value that is not None or False.", "code": r"""
import asyncio
async def main():
    bus = {fn}()
    trace = []
    async def slow(x):
        trace.append("slow-start"); await asyncio.sleep(0.01); trace.append("slow-end"); return None
    def sync_false(x):
        trace.append("false"); return False
    async def decide(x):
        trace.append("decide"); return 0
    async def never(x):
        trace.append("never"); return "x"
    for f in (slow, sync_false, decide, never):
        bus.on("stop", f)
    out = await bus.serial("stop", 1)
    assert out == 0 and trace == ["slow-start", "slow-end", "false", "decide"], (out, trace)
    assert await bus.serial("none") is None
asyncio.run(main())
"""},
        {"name": "Parallel runs concurrently and groups every failure", "visibility": "unshown", "behavior": "scheduler.concurrency", "failure_message": "parallel must start all listeners together, wait for all of them, and raise one ExceptionGroup containing every failure.", "code": r"""
import asyncio, time
async def main():
    bus = {fn}()
    finished = []
    async def sleeper(tag):
        await asyncio.sleep(0.2); finished.append(tag)
    for tag in "abc":
        bus.on("fan", lambda _, tag=tag: sleeper(tag))
    start = time.perf_counter()
    assert await bus.parallel("fan", None) is None
    assert time.perf_counter() - start < 0.5 and sorted(finished) == ["a", "b", "c"], finished
    done = []
    async def boom(msg):
        await asyncio.sleep(0.01); raise ValueError(msg)
    async def ok(msg):
        await asyncio.sleep(0.05); done.append(msg)
    bus.on("fail", lambda m: boom("one"))
    bus.on("fail", ok)
    bus.on("fail", lambda m: boom("two"))
    try:
        await bus.parallel("fail", "m")
    except ExceptionGroup as group:
        assert sorted(str(e) for e in group.exceptions) == ["one", "two"], group.exceptions
    else:
        raise AssertionError("expected an ExceptionGroup")
    assert done == ["m"], "parallel returned before every listener finished"
asyncio.run(main())
"""},
    ],
    "solution": '''import asyncio
import inspect


class EventBus:
    def __init__(self):
        self._hooks = {}

    def on(self, name, listener, prepend=False):
        hooks = self._hooks.setdefault(name, [])
        record = [listener]
        if prepend:
            hooks.insert(0, record)
        else:
            hooks.append(record)

        def dispose():
            for i, item in enumerate(hooks):
                if item is record:
                    del hooks[i]
                    return True
            return False

        return dispose

    def once(self, name, listener, prepend=False):
        def wrapper(*args):
            dispose()
            return listener(*args)

        dispose = self.on(name, wrapper, prepend)
        return dispose

    def _snapshot(self, name):
        return [record[0] for record in self._hooks.get(name, [])]

    @staticmethod
    def _bailed(value):
        return value is not None and value is not False

    def emit(self, name, *args):
        for listener in self._snapshot(name):
            listener(*args)

    def bail(self, name, *args):
        for listener in self._snapshot(name):
            result = listener(*args)
            if self._bailed(result):
                return result
        return None

    async def serial(self, name, *args):
        for listener in self._snapshot(name):
            result = listener(*args)
            if inspect.isawaitable(result):
                result = await result
            if self._bailed(result):
                return result
        return None

    async def parallel(self, name, *args):
        async def run(listener):
            result = listener(*args)
            if inspect.isawaitable(result):
                await result

        results = await asyncio.gather(*(run(l) for l in self._snapshot(name)), return_exceptions=True)
        errors = [r for r in results if isinstance(r, BaseException)]
        if errors:
            raise ExceptionGroup(f"{len(errors)} listener(s) of {name!r} failed", errors)

    def waterfall(self, name, *args, inner):
        queue = self._snapshot(name)

        def next():
            if queue:
                return queue.pop(0)(*args, next)
            return inner(*args)

        return next()
''',
    "interview_questions": interview(
        concept=[
            "What problem does an event bus solve in an agent harness where everything is a plugin?",
            "Explain the difference between emit, bail, serial, parallel and waterfall. Which would you use for an approval check, and which for a timeout wrapper?",
        ],
        deep_dive=[
            "Why snapshot the listener list at the start of a dispatch? What bugs appear without it?",
            "Walk through how next() in a waterfall finds the next listener. What happens if a listener calls next() twice?",
            "Why should registration return a disposer instead of offering a separate off(name, listener) method?",
        ],
        tradeoffs=[
            "Should a failing emit listener stop the others, or should the dispatcher contain the exception? What does each choice cost?",
            "Waterfall middleware versus explicit hook points in the agent loop: debuggability, ordering bugs and extensibility?",
        ],
    ),
}
