"""Cooperative cancellation: abort signals, deadlines and a scoped tool timeout wrapper."""

from ._interview import interview

TASK = {
    "title": "Cooperative Tool Timeouts",
    "difficulty": "Hard",
    "version": 1,
    "function_name": "tool_timeout",
    "description_en": r"""Build the cancellation primitives DeepSeek Harness uses for tool timeouts, then the `tools/execute` wrapper that enforces a tool's time budget without abandoning the tool.

**Signature:** `tool_timeout` is a module-level namespace; define these names at module level and set `tool_timeout = SimpleNamespace(AbortController=..., any_signal=..., deadline=..., timeout_of=..., TimeoutReason=..., with_timeout=...)` (import `SimpleNamespace` from `types`).

**Signals.**
- `AbortController()` has `.signal` and `abort(reason)`. The signal has `.aborted` (bool), `.reason` (None until aborted) and `add_listener(fn)`. The first `abort` sets the reason and calls every listener once with the reason, in registration order; later `abort` calls do nothing. A listener added to an already aborted signal is called immediately.
- `any_signal(signals)` returns a signal that aborts with the reason of whichever input aborts first. If an input is already aborted, the result starts aborted with the first such input's reason.

**Deadlines.**
- `TimeoutReason(code, timeout)` is an `Exception` subclass with attributes `code` and `timeout`.
- `deadline(upstream, timeout, code)` returns an object with `.signal` and `dispose()`. If `timeout <= 0`, the signal is `upstream`, or a signal that never aborts when `upstream` is None. Otherwise schedule, on the running asyncio loop, an abort with `TimeoutReason(code, timeout)` after `timeout` seconds; the signal is that timer's signal, combined with `upstream` through `any_signal` when `upstream` is not None. `dispose()` cancels the timer.
- `timeout_of(signal, code=None)` returns the signal's reason if it is a `TimeoutReason` whose code matches `code` (any code when `code` is None), otherwise None.

**The wrapper** `async with_timeout(execution, next, timeout)`:
- `execution` is a dict holding the current `"signal"`; `next()` is an async function that runs the tool and returns its result dict.
- If `timeout` is None, return `await next()` unchanged.
- Otherwise create `deadline(execution["signal"], timeout, "TOOL_TIMEOUT")`, put its signal into `execution["signal"]` while `next()` runs, and restore the original signal afterwards even if `next()` raises. Always dispose the deadline.
- If this wrapper's own deadline fired, as judged by `timeout_of(signal, "TOOL_TIMEOUT")`, return `{"is_error": True, "content": [{"type": "text", "text": f"Error: tool call timed out after {timeout}s"}], "error": {"message": f"tool call timed out after {timeout}s", "code": "TOOL_TIMEOUT"}}`. Otherwise return the tool's result.
- Never stop waiting for `next()`; the tool is expected to notice the aborted signal and return.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Cooperative, not preemptive.** Killing a tool mid-write can corrupt files or leave child processes behind. The harness only asks the tool to stop, then waits for it to reach a quiet state before reporting the timeout.

**Why swap and restore the signal.** Middleware after this wrapper, such as post-execute policies, must see the caller's signal, not this wrapper's possibly aborted one.

**Why scope by code.** Deadlines nest: a turn deadline can fire inside a tool deadline. Only a timeout carrying this wrapper's own code is reported as a tool timeout; any other abort is an ordinary cancellation from upstream.

**First reason wins.** Combining signals adopts the reason of whichever source aborts first, so a race between an upstream cancel and the timer resolves to exactly one cause.""",
    "advisory_prerequisites": ["dsh_tool_pipeline"],
    "hints": [
        {"level": 1, "kind": "questions", "content": "What must happen to listeners when a signal aborts twice? How does a combined signal know which source fired first? How does the wrapper tell its own timeout from an outer one?"},
        {"level": 2, "kind": "analysis", "content": "A signal keeps aborted, reason and a listener list; abort is a no-op after the first call. any_signal creates a controller and adds listener lambda r: ctrl.abort(r) to each input. deadline uses asyncio.get_running_loop().call_later(timeout, ctrl.abort, TimeoutReason(...)) and returns an object whose dispose cancels the handle. The wrapper swaps execution['signal'] in try/finally and checks timeout_of on the deadline signal after next() returns."},
    ],
    "model_connections": [
        "DeepSeek Harness's timeout-policy plugin wraps tools/execute with dsh-timeout's deadline and timeoutOf; JavaScript's AbortSignal.any provides the first-reason-wins combination.",
        "Python's asyncio.timeout cancels the task instead, which is the preemptive alternative this design avoids.",
    ],
    "pro_con_analysis": {
        "pros": ["Tools get a chance to clean up, nested deadlines report the correct cause, and downstream middleware never sees a foreign signal."],
        "cons": ["A tool that ignores its signal can block the caller forever; cooperative cancellation only works when every tool honors it."],
    },
    "sources": [
        {"kind": "code", "url": "https://github.com/deepseek-ai/deepseek-harness", "commit": "21638c56315ae6a2b552d6091945d3144c9af32e", "path": "packages/guard/timeout-policy/src/index.ts", "symbol": "apply and toolTimeoutResult", "license": "MIT", "adapted": "Signal swap and restore around next(), scoped TOOL_TIMEOUT classification and the structured timeout result.", "simplifications": "Timeout comes as an argument instead of the tool definition's timeoutMs; seconds instead of milliseconds."},
        {"kind": "code", "url": "https://github.com/deepseek-ai/deepseek-harness", "commit": "21638c56315ae6a2b552d6091945d3144c9af32e", "path": "packages/util/timeout/src/index.ts", "symbol": "deadline, timeoutOf and TimeoutReason", "license": "MIT", "adapted": "Deadline combining an upstream signal with a timer, the no-timeout passthrough and code-scoped timeout classification.", "simplifications": "AbortController and AbortSignal.any are implemented by the learner; no idle watchdog or timer-delay validation."},
    ],
    "tests": [
        {"name": "Slow cooperative tool reports a timeout", "behavior": "retry.classification", "code": r"""
import asyncio
T = {fn}
async def main():
    root = T.AbortController()
    execution = {"signal": root.signal}
    async def tool():
        seen = execution["signal"]
        stopped = asyncio.Event()
        seen.add_listener(lambda reason: stopped.set())
        await stopped.wait()
        return {"is_error": True, "content": [{"type": "text", "text": "aborted"}]}
    out = await T.with_timeout(execution, tool, 0.05)
    assert out["error"] == {"message": "tool call timed out after 0.05s", "code": "TOOL_TIMEOUT"}, out
    assert execution["signal"] is root.signal and not root.signal.aborted
asyncio.run(main())
"""},
        {"name": "Signals abort once and combine first-reason-wins", "visibility": "unshown", "behavior": "state.invariant", "failure_message": "abort is idempotent, listeners fire once in order, late listeners fire immediately, and any_signal adopts the first reason.", "code": r"""
T = {fn}
c = T.AbortController()
log = []
c.signal.add_listener(lambda r: log.append(("a", r)))
c.signal.add_listener(lambda r: log.append(("b", r)))
assert c.signal.aborted is False and c.signal.reason is None
c.abort("first"); c.abort("second")
assert log == [("a", "first"), ("b", "first")] and c.signal.reason == "first", log
c.signal.add_listener(lambda r: log.append(("late", r)))
assert log[-1] == ("late", "first")
x, y, z = T.AbortController(), T.AbortController(), T.AbortController()
combo = T.any_signal([x.signal, y.signal])
y.abort("y wins"); x.abort("x late")
assert combo.aborted and combo.reason == "y wins", combo.reason
z.abort("pre")
w = T.AbortController()
combo2 = T.any_signal([w.signal, z.signal])
assert combo2.aborted and combo2.reason == "pre"
w.abort("after")
assert combo2.reason == "pre"
"""},
        {"name": "Deadlines fire, dispose, pass through and classify by code", "visibility": "unshown", "behavior": "retry.classification", "failure_message": "A positive timeout aborts with TimeoutReason after the delay, dispose cancels it, timeout <= 0 passes the upstream signal through, and timeout_of filters by code.", "code": r"""
import asyncio
T = {fn}
async def main():
    d = T.deadline(None, 0.02, "X")
    assert not d.signal.aborted
    await asyncio.sleep(0.05)
    r = T.timeout_of(d.signal, "X")
    assert d.signal.aborted and isinstance(r, T.TimeoutReason) and isinstance(r, Exception) and r.code == "X" and r.timeout == 0.02
    assert T.timeout_of(d.signal) is r and T.timeout_of(d.signal, "Y") is None
    d2 = T.deadline(None, 0.02, "X"); d2.dispose()
    await asyncio.sleep(0.05)
    assert not d2.signal.aborted
    up = T.AbortController()
    assert T.deadline(up.signal, 0, "X").signal is up.signal
    never = T.deadline(None, -1, "X").signal
    assert never.aborted is False
    up2 = T.AbortController()
    d3 = T.deadline(up2.signal, 5.0, "X")
    up2.abort("user cancel")
    assert d3.signal.aborted and d3.signal.reason == "user cancel" and T.timeout_of(d3.signal, "X") is None
    d3.dispose()
    plain = T.AbortController(); plain.abort(ValueError("no"))
    assert T.timeout_of(plain.signal) is None
asyncio.run(main())
"""},
        {"name": "Wrapper is scoped, restores the signal and waits for the tool", "visibility": "unshown", "behavior": "retry.classification", "failure_message": "Report TOOL_TIMEOUT only for this wrapper's own deadline, restore the caller's signal even on errors, wait for the tool to finish, and pass through when there is no budget.", "code": r"""
import asyncio
T = {fn}
async def main():
    turn = T.deadline(None, 0.03, "TURN_TIMEOUT")
    execution = {"signal": turn.signal}
    finished = []
    async def tool():
        stopped = asyncio.Event()
        execution["signal"].add_listener(lambda r: stopped.set())
        await stopped.wait()
        await asyncio.sleep(0.05)
        finished.append(True)
        return {"is_error": True, "content": [{"type": "text", "text": "cancelled by turn"}]}
    out = await T.with_timeout(execution, tool, 1.0)
    assert out["content"][0]["text"] == "cancelled by turn", out
    assert finished == [True] and execution["signal"] is turn.signal
    root = T.AbortController()
    execution = {"signal": root.signal}
    async def slow_but_done():
        stopped = asyncio.Event()
        execution["signal"].add_listener(lambda r: stopped.set())
        await stopped.wait()
        await asyncio.sleep(0.05)
        finished.append("drained")
        return {"is_error": False, "content": []}
    out = await T.with_timeout(execution, slow_but_done, 0.02)
    assert out["error"]["code"] == "TOOL_TIMEOUT" and finished[-1] == "drained", out
    held = []
    async def fast():
        held.append(execution["signal"])
        return {"is_error": False, "content": [{"type": "text", "text": "fast"}]}
    out = await T.with_timeout(execution, fast, 0.1)
    assert out == {"is_error": False, "content": [{"type": "text", "text": "fast"}]}
    await asyncio.sleep(0.2)
    assert not root.signal.aborted and not held[0].aborted, "the deadline must be disposed once the tool returns"
    async def boom():
        assert execution["signal"] is not root.signal
        raise KeyError("x")
    try:
        await T.with_timeout(execution, boom, 0.5)
    except KeyError:
        pass
    assert execution["signal"] is root.signal
    seen = []
    async def peek():
        seen.append(execution["signal"]); return {"is_error": False, "content": []}
    await T.with_timeout(execution, peek, None)
    assert seen == [root.signal]
asyncio.run(main())
"""},
    ],
    "solution": '''import asyncio
from types import SimpleNamespace


class _Signal:
    def __init__(self):
        self.aborted = False
        self.reason = None
        self._listeners = []

    def add_listener(self, fn):
        if self.aborted:
            fn(self.reason)
        else:
            self._listeners.append(fn)


class AbortController:
    def __init__(self):
        self.signal = _Signal()

    def abort(self, reason):
        signal = self.signal
        if signal.aborted:
            return
        signal.aborted = True
        signal.reason = reason
        listeners, signal._listeners = signal._listeners, []
        for fn in listeners:
            fn(reason)


def any_signal(signals):
    controller = AbortController()
    for signal in signals:
        if signal.aborted:
            controller.abort(signal.reason)
            return controller.signal
    for signal in signals:
        signal.add_listener(controller.abort)
    return controller.signal


class TimeoutReason(Exception):
    def __init__(self, code, timeout):
        super().__init__(f"{code} after {timeout}s")
        self.code = code
        self.timeout = timeout


class _Deadline:
    def __init__(self, signal, handle=None):
        self.signal = signal
        self._handle = handle

    def dispose(self):
        if self._handle is not None:
            self._handle.cancel()


def deadline(upstream, timeout, code):
    if timeout <= 0:
        return _Deadline(upstream if upstream is not None else AbortController().signal)
    timer = AbortController()
    handle = asyncio.get_running_loop().call_later(timeout, timer.abort, TimeoutReason(code, timeout))
    signal = timer.signal if upstream is None else any_signal([upstream, timer.signal])
    return _Deadline(signal, handle)


def timeout_of(signal, code=None):
    reason = signal.reason
    if not isinstance(reason, TimeoutReason):
        return None
    return reason if code is None or reason.code == code else None


async def with_timeout(execution, next, timeout):
    if timeout is None:
        return await next()
    d = deadline(execution["signal"], timeout, "TOOL_TIMEOUT")
    upstream = execution["signal"]
    execution["signal"] = d.signal
    try:
        result = await next()
        if timeout_of(d.signal, "TOOL_TIMEOUT") is not None:
            message = f"tool call timed out after {timeout}s"
            return {"is_error": True, "content": [{"type": "text", "text": f"Error: {message}"}],
                    "error": {"message": message, "code": "TOOL_TIMEOUT"}}
        return result
    finally:
        execution["signal"] = upstream
        d.dispose()


tool_timeout = SimpleNamespace(AbortController=AbortController, any_signal=any_signal, deadline=deadline,
                               timeout_of=timeout_of, TimeoutReason=TimeoutReason, with_timeout=with_timeout)
''',
    "interview_questions": interview(
        concept=[
            "What is the difference between cooperative and preemptive cancellation? Why does an agent harness prefer cooperative cancellation for tools?",
            "What does combining an upstream cancel signal with a timer signal give you, and why should the first reason win?",
        ],
        deep_dive=[
            "Walk through the timeout wrapper: why swap the signal onto the execution, and why restore it in a finally block?",
            "Deadlines nest: a turn deadline fires while a tool deadline is armed. How does scoping by code keep the tool from misreporting a timeout?",
            "Why must the wrapper keep waiting for the tool after the deadline fires instead of returning immediately?",
        ],
        tradeoffs=[
            "What do you do about a tool that ignores its signal? Compare a hard kill after a grace period, process isolation and just waiting.",
            "Per-tool budgets declared by the tool versus one global timeout configured by the operator: flexibility, safety and surprise?",
        ],
    ),
}
