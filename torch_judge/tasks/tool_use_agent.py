"""Fix and extend a tool-use agent loop against a scripted model: every tool call answered by id, errors handed back, a call cap, a batch tool, and parallel tool calls."""

from ._interview import interview

# A scripted client that replays fixed responses and checks every transcript it is sent.
_MODEL = r"""
import copy, random, threading, time

CLOSES = {
    ("ACME", "2026-03-02"): 71.5, ("ACME", "2026-03-03"): 72.0, ("ACME", "2026-03-04"): 70.25,
    ("BOLT", "2026-03-02"): 18.4, ("BOLT", "2026-03-03"): 19.0, ("BOLT", "2026-03-04"): 73.25,
    ("CRUX", "2026-03-02"): 305.0, ("CRUX", "2026-03-03"): 299.5, ("CRUX", "2026-03-04"): 310.75,
}
TICKERS = {t for t, _ in CLOSES}

def get_close(ticker, date):
    if ticker not in TICKERS:
        raise LookupError(f"no such ticker {ticker}")
    if (ticker, date) not in CLOSES:
        raise ValueError(f"no close for {ticker} on {date}")
    return CLOSES[ticker, date]

CLOSE_TOOL = {
    "name": "get_close",
    "description": "Closing price of one ticker on one date.",
    "input_schema": {"type": "object", "properties": {"ticker": {"type": "string"}, "date": {"type": "string"}},
                     "required": ["ticker", "date"]},
}
SYSTEM = "Answer questions about closing prices."

class Response:
    def __init__(self, content, stop_reason):
        self.content, self.stop_reason = content, stop_reason

def use(id_, name, **inp):
    return {"type": "tool_use", "id": id_, "name": name, "input": inp}

def ask(*blocks):
    return Response(list(blocks), "tool_use")

def answer(text):
    return Response([{"type": "text", "text": text}], "end_turn")

def expected_result(functions, block):
    # what a correct loop hands back for one tool_use block
    fn = functions.get(block["name"])
    if fn is None:
        return f"unknown tool: {block['name']}", True
    try:
        return str(fn(**block["input"])), False
    except Exception as e:
        return str(e), True

class Client:
    # replays script[k] on the k-th call after checking the transcript it was sent
    def __init__(self, script, functions, tools, question, endless=False):
        self.script, self.functions, self.tools, self.question = script, functions, tools, question
        self.endless, self.calls = endless, 0
    def response(self, k):
        if self.endless:
            return ask(use(f"loop_{k}", "get_close", ticker="ACME", date="2026-03-02"))
        assert k < len(self.script), f"create was called {k + 1} times; the model already ended its turn"
        return self.script[k]
    def create(self, *, system, messages, tools):
        k = self.calls
        self.calls += 1
        assert k < 40, "create was called 40 times"
        assert system == SYSTEM and tools == self.tools, ("call", k + 1, "system or tools differ from what run was given")
        want = [{"role": "user", "content": [{"type": "text", "text": self.question}]}]
        for i in range(k):
            r = self.response(i)
            uses = [b for b in r.content if b["type"] == "tool_use"]
            want.append({"role": "assistant", "content": r.content})
            want.append({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": b["id"], "content": t, "is_error": e}
                for b, (t, e) in ((b, expected_result(self.functions, b)) for b in uses)]})
        got = copy.deepcopy(messages)
        assert len(got) == len(want), ("call", k + 1, f"{len(got)} messages sent, expected {len(want)}", [m["role"] for m in got])
        for i, (g, w) in enumerate(zip(got, want)):
            assert g == w, ("call", k + 1, "message", i, "sent", g, "expected", w)
        return copy.deepcopy(self.response(k))

def run_bounded(target, seconds=5):
    box = {}
    def body():
        try:
            box["value"] = target()
        except BaseException as e:
            box["error"] = e
    t = threading.Thread(target=body, daemon=True)
    t.start()
    t.join(seconds)
    assert not t.is_alive(), f"run did not return within {seconds} seconds"
    if "error" in box:
        raise box["error"]
    return box["value"]

def random_script(rng, rounds, max_blocks, allow_errors):
    script, n = [], 0
    for _ in range(rounds):
        blocks = []
        for _ in range(rng.randint(1, max_blocks)):
            n += 1
            ticker = rng.choice(sorted(TICKERS) + (["ZZZ"] if allow_errors else []))
            date = rng.choice(["2026-03-02", "2026-03-03", "2026-03-04"] + (["2026-03-07"] if allow_errors else []))
            name = "get_close" if not allow_errors or rng.random() < 0.85 else "get_volume"
            blocks.append(use(f"toolu_{rng.randrange(10**6)}_{n}", name, ticker=ticker, date=date))
        script.append(ask(*blocks))
    script.append(answer(f"done after {rounds} rounds"))
    return script
class Counting:
    # wraps a function, counting calls and how many run at once
    def __init__(self, fn, delay=0.0):
        self.fn, self.delay, self.calls, self.active, self.peak = fn, delay, 0, 0, 0
        self.lock = threading.Lock()
    def __call__(self, **kw):
        with self.lock:
            self.calls += 1
            self.active += 1
            self.peak = max(self.peak, self.active)
        try:
            time.sleep(kw.pop("_delay", self.delay))
            return self.fn(**kw)
        finally:
            with self.lock:
                self.active -= 1

def plain_batch(calls):
    return [get_close(**c) for c in calls]
"""



TESTS = [
    {"name": "Part 1: the worked example", "part": 1, "behavior": "protocol.validation", "code": _MODEL + r"""
q = "What did BOLT close at on 2026-03-04?"
script = [ask(use("toolu_7", "get_close", ticker="BOLT", date="2026-03-04")), answer("BOLT closed at 73.25 on 2026-03-04.")]
client = Client(script, {"get_close": get_close}, [CLOSE_TOOL], q)
got = run_bounded(lambda: {fn}(client, {"get_close": get_close}).run(q, [CLOSE_TOOL], SYSTEM))
assert got == "BOLT closed at 73.25 on 2026-03-04.", got
assert client.calls == 2, client.calls
"""},
    {"name": "Part 1: several rounds and several calls per turn", "part": 1, "visibility": "unshown", "behavior": "protocol.validation",
     "failure_message": "With scripted turns of 1 to 3 successful tool calls over 0 to 4 rounds, a transcript sent to create was wrong (the model's turn missing, a result paired by name or out of order, a call left unanswered), the answer was wrong, or create was called the wrong number of times.",
     "code": _MODEL + r"""
for seed in range(60):
    rng = random.Random(seed)
    script = random_script(rng, seed % 5, 3, allow_errors=False)
    q = f"question {seed}"
    client = Client(script, {"get_close": get_close}, [CLOSE_TOOL], q)
    got = run_bounded(lambda: {fn}(client, {"get_close": get_close}).run(q, [CLOSE_TOOL], SYSTEM))
    assert got == script[-1].content[0]["text"], (seed, got)
    assert client.calls == len(script), (seed, client.calls, len(script))
"""},
    {"name": "Part 2: the worked example", "part": 2, "behavior": "retry.classification", "code": _MODEL + r"""
q = "What did CRUX close at on 2026-03-07?"
script = [ask(use("toolu_1", "get_close", ticker="CRUX", date="2026-03-07")), answer("There is no close for CRUX on 2026-03-07.")]
client = Client(script, {"get_close": get_close}, [CLOSE_TOOL], q)
got = run_bounded(lambda: {fn}(client, {"get_close": get_close}).run(q, [CLOSE_TOOL], SYSTEM))
assert got == "There is no close for CRUX on 2026-03-07.", got
client = Client([], {"get_close": get_close}, [CLOSE_TOOL], "loop", endless=True)
got = run_bounded(lambda: {fn}(client, {"get_close": get_close}).run("loop", [CLOSE_TOOL], SYSTEM, max_calls=3))
assert got == "gave up after 3 calls", got
assert client.calls == 3, client.calls
"""},
    {"name": "Part 2: failing tools and the call cap", "part": 2, "visibility": "unshown", "behavior": "budget.enforcement",
     "failure_message": "With unknown tickers, missing dates and an unknown tool name mixed in, a tool_result was not the error text with is_error True; or with max_calls from 1 to 6 the loop called create more than max_calls times, gave up when the last allowed call ended the turn, returned the wrong text, or ran the tools of the response that hit the cap.",
     "code": _MODEL + r"""
for seed in range(80):
    rng = random.Random(100 + seed)
    script = random_script(rng, rng.randint(0, 5), 3, allow_errors=True)
    cap = rng.randint(1, 6)
    q = f"question {seed}"
    counted = Counting(get_close)
    client = Client(script, {"get_close": get_close}, [CLOSE_TOOL], q)
    got = run_bounded(lambda: {fn}(client, {"get_close": counted}).run(q, [CLOSE_TOOL], SYSTEM, max_calls=cap))
    if len(script) <= cap:
        assert got == script[-1].content[0]["text"], (seed, cap, got)
        assert client.calls == len(script), (seed, cap, client.calls)
    else:
        assert got == f"gave up after {cap} calls", (seed, cap, got)
        assert client.calls == cap, (seed, cap, client.calls)
        ran = sum(1 for r in script[:cap - 1] for b in r.content if b["name"] == "get_close")
        assert counted.calls == ran, (seed, cap, "tools run", counted.calls, "expected", ran)
"""},
    {"name": "Part 3: the worked example", "part": 3, "behavior": "contract.signature", "code": _MODEL + r"""
spec, batch = {fn}.batch_tool(CLOSE_TOOL, get_close)
assert spec["name"] == "get_close_batch", spec["name"]
assert spec["input_schema"] == {"type": "object", "properties": {"calls": {"type": "array", "items": CLOSE_TOOL["input_schema"]}}, "required": ["calls"]}, spec["input_schema"]
q = "Which of ACME, BOLT and CRUX closed highest on 2026-03-03?"
calls = [{"ticker": t, "date": "2026-03-03"} for t in ("ACME", "BOLT", "CRUX")]
script = [ask(use("toolu_2", "get_close_batch", calls=calls)), answer("CRUX, at 299.5.")]
client = Client(script, {"get_close_batch": plain_batch}, [spec], q)
got = run_bounded(lambda: {fn}(client, {"get_close_batch": batch}).run(q, [spec], SYSTEM))
assert got == "CRUX, at 299.5.", got
assert client.calls == 2, client.calls
"""},
    {"name": "Part 3: batch failures, empty batches, other tools", "part": 3, "visibility": "unshown", "behavior": "effects.idempotency",
     "failure_message": "The batch tool returned results for a batch with a failing call, made calls after the first failure, did not return '[]' for no calls, changed the original tool spec, had no description, or named a batch of another tool wrongly.",
     "code": _MODEL + r"""
before = copy.deepcopy(CLOSE_TOOL)
counted = Counting(get_close)
spec, batch = {fn}.batch_tool(CLOSE_TOOL, counted)
assert CLOSE_TOOL == before, "batch_tool changed the tool spec it was given"
assert isinstance(spec.get("description"), str) and spec["description"].strip(), spec.get("description")
calls = [{"ticker": "ACME", "date": "2026-03-02"}, {"ticker": "CRUX", "date": "2026-03-09"}, {"ticker": "BOLT", "date": "2026-03-02"}]
script = [ask(use("t1", "get_close_batch", calls=calls), use("t2", "get_close_batch", calls=[])), answer("ok")]
client = Client(script, {"get_close_batch": plain_batch}, [spec], "q")
assert run_bounded(lambda: {fn}(client, {"get_close_batch": batch}).run("q", [spec], SYSTEM)) == "ok"
assert counted.calls == 2, f"the batch made {counted.calls} calls; it must stop at the first failure"
other = {"name": "lookup", "description": "Look one word up.", "input_schema": {"type": "object", "properties": {"word": {"type": "string"}}, "required": ["word"]}}
spec2, batch2 = {fn}.batch_tool(other, lambda word: word.upper())
assert spec2["name"] == "lookup_batch", spec2["name"]
assert spec2["input_schema"]["properties"]["calls"]["items"] == other["input_schema"]
assert batch2(calls=[{"word": "a"}, {"word": "bc"}]) == ["A", "BC"]
"""},
    {"name": "Part 4: the worked example", "part": 4, "behavior": "scheduler.concurrency", "code": _MODEL + r"""
slow = Counting(get_close)
blocks = [use(f"toolu_{i}", "get_close", ticker=t, date="2026-03-02", _delay=d) for i, (t, d) in enumerate([("ACME", 0.3), ("BOLT", 0.2), ("CRUX", 0.1)])]
script = [ask(*blocks), answer("ACME 71.5, BOLT 18.4, CRUX 305.0.")]
quiet = {"get_close": lambda _delay=0, **kw: get_close(**kw)}
client = Client(script, quiet, [CLOSE_TOOL], "q")
got = run_bounded(lambda: {fn}(client, {"get_close": slow}).run("q", [CLOSE_TOOL], SYSTEM))
assert got == "ACME 71.5, BOLT 18.4, CRUX 305.0.", got
assert slow.peak == 3, f"expected the 3 calls to run at once, saw at most {slow.peak}"
"""},
    {"name": "Part 4: parallel turns with errors", "part": 4, "visibility": "unshown", "behavior": "scheduler.concurrency",
     "failure_message": "Over several rounds with failing calls mixed in, the tool calls of one turn did not all run at once, results were not in block order, or an error in one call changed another call's result.",
     "code": _MODEL + r"""
for seed in range(6):
    rng = random.Random(300 + seed)
    script = random_script(rng, 2, 4, allow_errors=True)
    for r in script[:-1]:
        for b in r.content:
            b["input"]["_delay"] = rng.choice([0.05, 0.1, 0.15])
    widest = max(sum(1 for b in r.content if b["name"] == "get_close") for r in script[:-1])
    slow = Counting(get_close)
    quiet = {"get_close": lambda _delay=0, **kw: get_close(**kw)}
    client = Client(script, quiet, [CLOSE_TOOL], "q")
    got = run_bounded(lambda: {fn}(client, {"get_close": slow}).run("q", [CLOSE_TOOL], SYSTEM))
    assert got == script[-1].content[0]["text"], (seed, got)
    assert slow.peak == widest, (seed, "peak", slow.peak, "widest turn", widest)
"""},
]

TASK = {
    "title": "Agent Tool-Use Loop",
    "difficulty": "Medium",
    "version": 1,
    "function_name": "ToolAgent",
    "description_en": r"""Fix `ToolAgent`, a tool-use loop that the starter code gives you with several bugs, then extend it. The loop sends a conversation to a model, runs the tools the model asks for, hands the results back, and repeats until the model answers. The tests replace the model with a scripted client that checks every conversation it is sent.

The requirement arrives in parts. Each part keeps every earlier behavior, so one `ToolAgent` passes all parts at the end. Pass every test of the current part to reveal the next one.

**Rules for every part:**
- `ToolAgent(client, functions)`: `functions` maps a tool name to a Python function that takes the tool's input as keyword arguments.
- `client.create(system=..., messages=..., tools=...)` returns a response with `.content`, a list of blocks, and `.stop_reason`, which is `"end_turn"` or `"tool_use"`. Pass `system` and `tools` exactly as `run` received them.
- `messages` starts as one user message, `{"role": "user", "content": [{"type": "text", "text": question}]}`.
- On `"end_turn"`, the content is one text block, and its `"text"` is the answer that `run` returns.
- On `"tool_use"`, every block is `{"type": "tool_use", "id", "name", "input"}`. Append `{"role": "assistant", "content": response.content}`. Then append one user message whose content holds one `{"type": "tool_result", "tool_use_id", "content", "is_error"}` per tool_use block, in block order. Then call `create` again.
- `tool_use_id` is the block's `id`. `content` is `str()` of the function's return value, and `is_error` is `False` for a call that succeeds.
- Keep the constructor, `run_tool(name, tool_input) -> (text, is_error)` and `run(question, tools, system, max_calls=8) -> str`.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why this shows up in interviews:** the round checks that you know the tool-use protocol well enough to spot what a plausible first draft gets wrong, then asks how to make the agent cheaper and sturdier. Each later part adds one requirement: failures and a cap, a batch tool that saves model calls, and tool calls run in parallel.

**Where it is used:** every agent built on a chat model with tools runs this loop, from coding agents to research assistants. Pairing results by id, handing errors back, capping turns and batching lookups decide its cost and reliability.

Adapted from the agent tool-use loop in Schuture/Anthropic-Interview-Notes (CC BY-NC 4.0), reworded. The notebook globals become a `ToolAgent` class with a `run_tool` method, the live model becomes a scripted client, the stock table is new (`get_quote` becomes `get_close`, with new error texts), and `create` takes only `system`, `messages` and `tools`. The `unknown tool` error result and the rule that the capped response's tools are not run are new. The source's third part, a `get_quotes(lookups)` tool plus a prompt that asks the model to batch, becomes the graded `batch_tool`, which builds `<name>_batch(calls)` for any tool, and the parallel calls become a separate fourth part.""",
    "parts": [
        {
            "title": "Fix the loop",
            "description_en": r"""**Signature:** `ToolAgent(client, functions).run(question, tools, system, max_calls=8) -> str`

- Fix the starter so it follows the rules: the model's own turn goes back before the results, each result carries its block's `id`, every tool_use block gets a result, and the loop continues until `"end_turn"`.
- Tools in this part always succeed, and no question needs more than `max_calls` calls.

**Example:** with `functions = {"get_close": get_close}` and the question `"What did BOLT close at on 2026-03-04?"`:
- call 1 returns `tool_use` with one block, `id` `"toolu_7"`, `name` `"get_close"`, `input` `{"ticker": "BOLT", "date": "2026-03-04"}`
- call 2 is sent the question, that assistant turn, and a user turn with one `tool_result` whose `tool_use_id` is `"toolu_7"`, with `content` `"73.25"` and `is_error` `False`
- call 2 returns the text `"BOLT closed at 73.25 on 2026-03-04."`, which `run` returns; `create` was called exactly twice""",
        },
        {
            "title": "Failures and a cap",
            "description_en": r"""Keep Part 1. Tools can now fail, and the model may never stop.

- If the function raises, the result's `content` is `str(exception)` and `is_error` is `True`. A name not in `functions` gives `content` `f"unknown tool: {name}"` with `is_error` `True`. Neither stops the loop.
- `run` calls `create` at most `max_calls` times. If response number `max_calls` still asks for tools, return `f"gave up after {max_calls} calls"` without running them.

**Example:**
- for `"What did CRUX close at on 2026-03-07?"`, `get_close` raises `ValueError("no close for CRUX on 2026-03-07")`; the result sent back has that text as `content` with `is_error` `True`, and `run` returns the model's next answer
- a model that asks for a tool on every call, with `max_calls=3`, gets exactly 3 calls, and `run` returns `"gave up after 3 calls"`""",
        },
        {
            "title": "A batch tool",
            "description_en": r"""Keep Parts 1–2. Asking for one price per turn costs one model call per price. Add a static method `batch_tool(tool, function) -> (spec, batch_function)` that turns a one-call tool into a batch tool.

- `spec["name"]` is `tool["name"] + "_batch"`, and `spec["description"]` is a non-empty string.
- `spec["input_schema"]` is `{"type": "object", "properties": {"calls": {"type": "array", "items": tool["input_schema"]}}, "required": ["calls"]}`.
- `batch_function(calls)` calls `function(**call)` for each call in order and returns the list of results. The first exception propagates, and no later call is made. `batch_tool` never changes `tool`.

**Example:** with the batch of `get_close` offered, the question `"Which of ACME, BOLT and CRUX closed highest on 2026-03-03?"` takes two calls to `create`:
- the spec is named `"get_close_batch"`, and its `input_schema` wraps `get_close`'s schema as the items of `calls`
- call 1 returns one block for `"get_close_batch"` with three calls in `calls`
- its result's `content` is `"[72.0, 19.0, 299.5]"`, and call 2 answers""",
        },
        {
            "title": "Parallel tool calls",
            "description_en": r"""Keep Parts 1–3. Tool functions may be slow.

- All tool calls of one response run at the same time, each on its own thread. The results still go back in block order.
- The next `create` call waits until every call of the turn has finished.

**Example:** one response asks for three closes whose functions take `0.3`, `0.2` and `0.1` seconds:
- all three run at once
- the results go back in block order, though the first finishes last""",
        },
    ],
    "hints": [
        {"level": 1, "kind": "questions", "content": "Write down the messages a correct loop sends on its second create call for one tool request. Which message is missing from what the starter sends? Which field of the tool_use block should tool_use_id copy, and how many blocks can one response hold?"},
        {"level": 2, "kind": "analysis", "content": "The starter has four bugs: it never appends the assistant turn, it sets tool_use_id to the tool's name, it answers only the first tool_use block, and it makes one round at most. Loop until end_turn; each round append the response content as the assistant turn, then one user turn with a tool_result per tool_use block, built from that block's id."},
    ],
    "model_connections": [
        "Chat models with tool use return tool_use blocks that the caller must answer with tool_result blocks paired by id, in the same order, before the next call.",
        "Coding and research agents cap the number of turns, hand tool errors back to the model so it can recover, and batch or parallelize lookups to cut latency and cost.",
    ],
    "pro_con_analysis": {
        "pros": [
            "Sending the model's own turn back before the results keeps the transcript valid, so the model sees which request each result answers.",
            "Handing failures back as error results lets the model retry or explain instead of crashing the loop.",
            "A batch tool and parallel calls cut model calls and wall-clock time when the needed lookups are known up front.",
        ],
        "cons": [
            "A cap on calls can give up on questions that needed one more round.",
            "A batch with no partial success wastes the good lookups when one fails.",
            "Parallel tool calls need thread-safe tools and give no ordering between their side effects.",
        ],
    },
    "tests": TESTS,
    "solution": r'''# Adapted from Schuture/Anthropic-Interview-Notes (code under the MIT License).
from concurrent.futures import ThreadPoolExecutor


class ToolAgent:
    def __init__(self, client, functions):
        self.client = client  # client.create(system=..., messages=..., tools=...) -> response
        self.functions = functions  # tool name -> Python function taking the tool input as keywords

    def run_tool(self, name, tool_input):
        """(text, is_error); a failing tool becomes an error result instead of an exception."""
        if name not in self.functions:
            return f"unknown tool: {name}", True
        try:
            return str(self.functions[name](**tool_input)), False
        except Exception as e:
            return str(e), True

    def run(self, question, tools, system, max_calls=8):
        messages = [{"role": "user", "content": [{"type": "text", "text": question}]}]
        for calls in range(1, max_calls + 1):
            response = self.client.create(system=system, messages=messages, tools=tools)
            if response.stop_reason == "end_turn":
                return response.content[0]["text"]
            if calls == max_calls:
                break  # no call left to read the results, so the tools are not run
            messages.append({"role": "assistant", "content": response.content})  # the model's own turn first
            messages.append({"role": "user", "content": self._results(
                [b for b in response.content if b["type"] == "tool_use"])})
        return f"gave up after {max_calls} calls"

    def _results(self, uses):
        """One tool_result per tool_use, in block order; the calls of one turn run at once."""
        with ThreadPoolExecutor(max_workers=max(1, len(uses))) as pool:
            outcomes = list(pool.map(lambda b: self.run_tool(b["name"], b["input"]), uses))
        return [{"type": "tool_result", "tool_use_id": b["id"], "content": text, "is_error": is_error}
                for b, (text, is_error) in zip(uses, outcomes)]

    @staticmethod
    def batch_tool(tool, function):
        """A tool spec and function that make many calls of `tool` in one request."""
        spec = {
            "name": tool["name"] + "_batch",
            "description": f"Make several {tool['name']} calls at once. " + tool["description"],
            "input_schema": {
                "type": "object",
                "properties": {"calls": {"type": "array", "items": tool["input_schema"]}},
                "required": ["calls"],
            },
        }

        def run_batch(calls):
            # in order; the first failure propagates and nothing is returned for the rest
            return [function(**call) for call in calls]

        return spec, run_batch
''',
    "interview_questions": interview(
        concept=[
            "Why must the assistant turn holding the tool_use blocks go back before the tool_result turn, and what does the model see if it is missing?",
            "Why must tool_use_id copy the block's id rather than the tool's name?",
        ],
        deep_dive=[
            "How do you find all four bugs in the starter, and which test exposes each one?",
        ],
        tradeoffs=[
            "Why hand a tool failure back to the model as an error result instead of raising, and when should the loop stop instead?",
            "How do you choose max_calls, and what should the user see when the agent gives up?",
            "What does a batch tool save, and what is lost when one lookup in the batch fails?",
            "When is it safe to run one turn's tool calls in parallel, and why must the results still go back in block order?",
        ],
    ),
}
