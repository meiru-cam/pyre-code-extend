Confirm with the interviewer whether a failing tool call should ever stop the loop outright (assumed not here: `is_error` is handed back to the model like any other result) and whether the model may answer from memory without calling a tool at all (assumed not, stated directly in the system prompt below).

### Part 1

Reading the loop against the code line by line: the second call to `client.create` never receives the model's first turn, only the lone `tool_result`, because nothing appends `{"role": "assistant", "content": response.content}` — that is the first bug. The second pairs the `tool_result` with `tool_use["name"]`, the tool's name, a fixed string shared by every call to that tool, rather than `tool_use["id"]`, unique to this one request. The third reads only one block of a response that could hold several: `next(b for b in response.content if b["type"] == "tool_use")` throws away every `tool_use` block after the first. The fourth exits without checking the second response at all: the function calls `create` at most twice and returns `response.content[0]["text"]` unconditionally, rather than looping until `stop_reason` is `"end_turn"`.

```python
PRICES = {
    "AAPL": {"2026-01-05": 180.00, "2026-01-06": 181.50, "2026-01-07": 179.00,
             "2026-01-08": 184.00, "2026-01-09": 190.00, "2026-01-12": 188.00},
    "MSFT": {"2026-01-05": 410.00, "2026-01-06": 406.00, "2026-01-07": 412.00,
             "2026-01-08": 415.00, "2026-01-09": 420.00, "2026-01-12": 418.00},
    "GOOG": {"2026-01-05": 140.00, "2026-01-06": 141.40, "2026-01-07": 139.00,
             "2026-01-08": 142.50, "2026-01-09": 145.00, "2026-01-12": 146.50},
}


def get_quote(ticker: str, date: str) -> float:
    if ticker not in PRICES:
        raise ValueError(f"unknown ticker: {ticker}")
    if date not in PRICES[ticker]:
        raise ValueError(f"no trading data for {date}")
    return PRICES[ticker][date]


GET_QUOTE_TOOL = {
    "name": "get_quote",
    "description": "Look up a single stock's closing price on one trading day.",
    "input_schema": {
        "type": "object",
        "properties": {
            "ticker": {"type": "string", "description": "Stock ticker, e.g. 'AAPL'."},
            "date": {"type": "string", "description": "Trading day, formatted YYYY-MM-DD."},
        },
        "required": ["ticker", "date"],
    },
}

MODEL_NAME = "agent-loop-demo"
MAX_TOKENS = 1024
SYSTEM_PROMPT = (
    "You are a financial research assistant. Answer only using the get_quote tool; "
    "never state a price you did not just look up."
)

TOOL_FUNCTIONS = {"get_quote": get_quote}


def run_tool(name: str, tool_input: dict) -> tuple[str, bool]:
    try:
        return str(TOOL_FUNCTIONS[name](**tool_input)), False
    except Exception as e:
        return str(e), True


def _answered(block: dict) -> dict:
    text, is_error = run_tool(block["name"], block["input"])
    return {"type": "tool_result", "tool_use_id": block["id"], "content": text, "is_error": is_error}


def answer_question(question: str) -> str:
    messages = [{"role": "user", "content": [{"type": "text", "text": question}]}]
    while True:
        response = client.create(model=MODEL_NAME, system=SYSTEM_PROMPT, messages=messages,
                                  tools=[GET_QUOTE_TOOL], max_tokens=MAX_TOKENS)
        if response.stop_reason == "end_turn":                                # NOTE: loop until end_turn
            return response.content[0]["text"]
        messages.append({"role": "assistant", "content": response.content})   # NOTE: keep the model's own turn
        tool_uses = [b for b in response.content if b["type"] == "tool_use"]  # NOTE: every block, not just one
        messages.append({"role": "user", "content": [_answered(b) for b in tool_uses]})
```

`_answered` reads `block["id"]`, never `block["name"]`, so every `tool_result` is paired with the request it actually answers.

### Part 2

`run_agent` is Part 1's fixed loop with the tool catalog, the system prompt and a round cap as parameters instead of module-level constants; it reuses `_answered` unchanged, since pairing a `tool_result` with its `tool_use` block does not depend on which tools are offered.

```python
def run_agent(question: str, tools: list[dict], system: str, max_calls: int = 8) -> str:
    messages = [{"role": "user", "content": [{"type": "text", "text": question}]}]
    for _ in range(max_calls):                            # NOTE: a hard cap -- see Follow-ups
        response = client.create(model=MODEL_NAME, system=system, messages=messages,
                                  tools=tools, max_tokens=MAX_TOKENS)
        if response.stop_reason == "end_turn":
            return response.content[0]["text"]
        messages.append({"role": "assistant", "content": response.content})
        tool_uses = [b for b in response.content if b["type"] == "tool_use"]
        messages.append({"role": "user", "content": [_answered(b) for b in tool_uses]})
    return f"gave up after {max_calls} calls"
```

Nothing about error handling changes for the harder questions: `_answered` already carries whatever `run_tool` returns, success or failure, into `is_error`, so a failing lookup is just one more `tool_result` for the model to read on the next call, and the loop does not need to know that anything went wrong. What Part 1's question set never exercised is more than one `tool_use` block in a single response; `run_agent`'s comprehension over `response.content` answers every one of them, in one appended message, regardless of how many there are.

### Part 3

`run_agent` does not change; only what is passed to it does.

```python
def get_quotes(lookups: list[dict]) -> list[float]:
    return [get_quote(pair["ticker"], pair["date"]) for pair in lookups]   # NOTE: fails on the first bad pair


TOOL_FUNCTIONS["get_quotes"] = get_quotes

GET_QUOTES_TOOL = {
    "name": "get_quotes",
    "description": (
        "Look up closing prices for several (ticker, date) pairs in a single call. "
        "Prefer this over repeated get_quote calls whenever more than one price is needed."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "lookups": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"ticker": {"type": "string"}, "date": {"type": "string"}},
                    "required": ["ticker", "date"],
                },
            },
        },
        "required": ["lookups"],
    },
}

SYSTEM_PROMPT_BATCHED = SYSTEM_PROMPT + (
    " When a question needs more than one price, request every price you already know you will "
    "need in a single turn: one call to get_quotes if it is offered, or one get_quote call per "
    "price, in parallel, in the same turn, if it is not."
)
TOOLS_BATCHED = [GET_QUOTE_TOOL, GET_QUOTES_TOOL]
```

`get_quotes` reuses `get_quote` pair by pair inside a list comprehension, so it raises on the first invalid pair exactly as the statement requires, with no separate validation pass. Measured on all six questions (checks below): 17 calls to `create` with `[GET_QUOTE_TOOL]` and `SYSTEM_PROMPT`, 12 with `TOOLS_BATCHED` and `SYSTEM_PROMPT_BATCHED` — every question that needed more than one price now takes exactly two calls, the same floor a question that only ever needed one was already at.

### Follow-ups

- Context growth: every call resends the full `messages` so far, so a long-running question pays for its own earlier rounds again on each later call; once a tool result's content has been folded into a later turn, a loop meant to run for many rounds typically drops or summarizes the raw result rather than keeping it verbatim forever.
- Cost and latency: fewer, larger calls save more than the token count alone suggests, because every tool result already sitting in `messages` is resent as prompt tokens on each later call, and because time-to-first-token is paid once per call to `create`, not once per tool used.
- A loop that never ends: `max_calls` is one instance of a general requirement — an agent needs a stopping condition the model cannot override, since nothing stops a model from requesting tools indefinitely; a call count alone does not bound wall-clock time, since a slow or hanging tool call can make even a low cap take a long time to reach, so production loops pair the call count with an explicit wall-clock deadline.
- Prompt injection through tool results: `tool_result.content` is text the loop did not write itself — here a fixed table, in general a file or a web page — and a model that treats every input the same way may follow instructions that appear inside it; stating in the system prompt that tool output is data, not instructions, and gating any consequential action behind a step the loop controls rather than the model alone, both narrow this.
- When an agent is the wrong tool: a question whose steps are fixed in advance, such as "look up X, then Y, then compare", is answered more cheaply and more reliably by code that calls `get_quote` directly; a loop earns its cost only when which lookups are needed cannot be known until the question itself is read.

```python
# The checks below replace the real client with FakeModel, a deterministic stand-in used only here:
# it never calls a language model, it decides its next tool_use block (or its final answer) from
# fixed, per-question plans and from what the running `messages` already show was tried. It stands
# in for `client.create` only inside these checks, never for the interface described in the Problem.
import ast


class _Response:
    def __init__(self, content, stop_reason):
        self.content = content
        self.stop_reason = stop_reason


def _price(done, ticker, date):
    return float(done[(ticker, date)][0])


def _single(ticker, date):
    return [(ticker, date)], lambda done: f"price {_price(done, ticker, date):.2f}"


def _compare_change(t1, t2, d0, d1):
    pairs = [(t1, d0), (t1, d1), (t2, d0), (t2, d1)]

    def answer(done):
        pct = lambda t: (_price(done, t, d1) - _price(done, t, d0)) / _price(done, t, d0)
        return f"{t1 if pct(t1) > pct(t2) else t2} gained more"

    return pairs, answer


def _failing_lookup(ticker, date):
    return [(ticker, date)], lambda done: f"failed: {done[(ticker, date)][0]}"


def _highest(tickers, date):
    pairs = [(t, date) for t in tickers]
    return pairs, lambda done: f"{max(tickers, key=lambda t: _price(done, t, date))} is highest"


PLANS = {
    "What was AAPL's closing price on 2026-01-07?": _single("AAPL", "2026-01-07"),
    "What was MSFT's closing price on 2026-01-09?": _single("MSFT", "2026-01-09"),
    "What was GOOG's closing price on 2026-01-05?": _single("GOOG", "2026-01-05"),
    "Between 2026-01-05 and 2026-01-09, did AAPL or MSFT gain more, in percentage terms?":
        _compare_change("AAPL", "MSFT", "2026-01-05", "2026-01-09"),
    "What was GOOG's closing price on 2026-01-10?": _failing_lookup("GOOG", "2026-01-10"),
    "Which of AAPL, MSFT and GOOG had the highest closing price on 2026-01-08?":
        _highest(["AAPL", "MSFT", "GOOG"], "2026-01-08"),
}
SINGLE_CALL_QUESTIONS = list(PLANS)[:3]
MULTI_CALL_QUESTIONS = list(PLANS)[3:]
ALL_QUESTIONS = SINGLE_CALL_QUESTIONS + MULTI_CALL_QUESTIONS
CHECK_TEXT = {ALL_QUESTIONS[0]: "179.00", ALL_QUESTIONS[1]: "420.00", ALL_QUESTIONS[2]: "140.00",
             ALL_QUESTIONS[3]: "AAPL", ALL_QUESTIONS[4]: "no trading data", ALL_QUESTIONS[5]: "MSFT"}


class FakeModel:
    """Deterministic stand-in for a real model: correct, but not clever unless told to be."""

    def __init__(self, plans):
        self.plans = plans
        self.calls = 0
        self._next_id = 0

    def _new_id(self):
        self._next_id += 1
        return f"call_{self._next_id}"

    def create(self, model, system, messages, tools, max_tokens):
        self.calls += 1
        question = messages[0]["content"][0]["text"]
        pairs, answer_fn = self.plans[question]
        done = {}                                          # (ticker, date) -> (content, is_error), from messages
        for i, msg in enumerate(messages):
            if msg["role"] != "assistant":
                continue
            uses = {b["id"]: b for b in msg["content"] if b["type"] == "tool_use"}
            if not uses:
                continue
            for r in messages[i + 1]["content"]:
                assert r["type"] == "tool_result" and r["tool_use_id"] in uses, \
                    f"tool_result does not match any pending tool_use: {r}"
                block = uses[r["tool_use_id"]]
                if block["name"] == "get_quote":
                    done[(block["input"]["ticker"], block["input"]["date"])] = (r["content"], r["is_error"])
                elif r["is_error"]:
                    for pair in block["input"]["lookups"]:
                        done[(pair["ticker"], pair["date"])] = (r["content"], True)
                else:
                    values = ast.literal_eval(r["content"])
                    for pair, v in zip(block["input"]["lookups"], values):
                        done[(pair["ticker"], pair["date"])] = (str(v), False)
        remaining = [p for p in pairs if p not in done]
        if not remaining:
            return _Response([{"type": "text", "text": answer_fn(done)}], "end_turn")
        names = {t["name"] for t in tools}
        if "get_quotes" in names and len(remaining) > 1:               # a batch tool is offered: use it
            lookups = [{"ticker": t, "date": d} for t, d in remaining]
            return _Response([{"type": "tool_use", "id": self._new_id(), "name": "get_quotes",
                               "input": {"lookups": lookups}}], "tool_use")
        if "parallel" in system.lower() and len(remaining) > 1:        # told to parallelize: ask for all at once
            blocks = [{"type": "tool_use", "id": self._new_id(), "name": "get_quote",
                       "input": {"ticker": t, "date": d}} for t, d in remaining]
            return _Response(blocks, "tool_use")
        t, d = remaining[0]                                            # default: one lookup per turn
        return _Response([{"type": "tool_use", "id": self._new_id(), "name": "get_quote",
                           "input": {"ticker": t, "date": d}}], "tool_use")


def run_and_count(fn, *args, **kwargs):
    global client
    client = FakeModel(PLANS)
    return fn(*args, **kwargs), client.calls


def _expect_error(fn, *args, needle):
    try:
        fn(*args)
        assert False, "should have raised"
    except ValueError as e:
        assert needle in str(e), e


# --- get_quote, get_quotes and run_tool directly, including failures the example questions don't reach ---
assert get_quote("AAPL", "2026-01-07") == 179.00
_expect_error(get_quote, "TSLA", "2026-01-07", needle="unknown ticker")
_expect_error(get_quote, "AAPL", "2026-01-10", needle="no trading data")
assert get_quotes([{"ticker": "AAPL", "date": "2026-01-05"},
                   {"ticker": "MSFT", "date": "2026-01-05"}]) == [180.00, 410.00]
_expect_error(get_quotes, [{"ticker": "AAPL", "date": "2026-01-05"}, {"ticker": "AAPL", "date": "2026-01-10"}],
             needle="no trading data")                     # NOTE: all-or-nothing -- the first bad pair aborts
assert run_tool("get_quote", {"ticker": "AAPL", "date": "2026-01-07"}) == ("179.0", False)
text, is_error = run_tool("get_quote", {"ticker": "AAPL", "date": "2026-01-10"})
assert is_error and "no trading data" in text

# --- Part 1: answer_question on the three single-call questions, exactly two calls each ---
for q in SINGLE_CALL_QUESTIONS:
    answer, calls = run_and_count(answer_question, q)
    assert CHECK_TEXT[q] in answer, (q, answer)
    assert calls == 2, (q, calls)


# --- general contract: two tool_use blocks in one response, each answered with its own id ---
class _TwoToolStub:
    def __init__(self):
        self.calls = 0

    def create(self, model, system, messages, tools, max_tokens):
        self.calls += 1
        if self.calls == 1:
            return _Response([
                {"type": "tool_use", "id": "a1", "name": "get_quote",
                 "input": {"ticker": "AAPL", "date": "2026-01-07"}},
                {"type": "tool_use", "id": "a2", "name": "get_quote",
                 "input": {"ticker": "MSFT", "date": "2026-01-09"}},
            ], "tool_use")
        assistant_msg, result_msg = messages[-2], messages[-1]
        ids_asked = {b["id"] for b in assistant_msg["content"]}
        ids_answered = {r["tool_use_id"] for r in result_msg["content"]}
        assert ids_asked == ids_answered == {"a1", "a2"}, (ids_asked, ids_answered)
        contents = {r["tool_use_id"]: r["content"] for r in result_msg["content"]}
        assert contents == {"a1": "179.0", "a2": "420.0"}, contents
        return _Response([{"type": "text", "text": "done"}], "end_turn")


client = _TwoToolStub()
assert answer_question("(two independent lookups)") == "done"
assert client.calls == 2

# --- Part 2: run_agent with only get_quote offered, on all six questions ---
EXPECTED_BASELINE_CALLS = [2, 2, 2, 5, 2, 4]
baseline_total = 0
for q, expected_calls in zip(ALL_QUESTIONS, EXPECTED_BASELINE_CALLS):
    answer, calls = run_and_count(run_agent, q, [GET_QUOTE_TOOL], SYSTEM_PROMPT)
    assert CHECK_TEXT[q] in answer, (q, answer)
    assert calls == expected_calls, (q, calls, expected_calls)
    baseline_total += calls
assert baseline_total == 17, baseline_total


# --- Part 2: the cap on rounds, against a model that never stops asking for tools ---
class _NeverStopsStub:
    def __init__(self):
        self.calls = 0

    def create(self, model, system, messages, tools, max_tokens):
        self.calls += 1
        return _Response([{"type": "tool_use", "id": f"x{self.calls}", "name": "get_quote",
                           "input": {"ticker": "AAPL", "date": "2026-01-07"}}], "tool_use")


client = _NeverStopsStub()
assert run_agent("(never resolves)", [GET_QUOTE_TOOL], SYSTEM_PROMPT, max_calls=5) == "gave up after 5 calls"
assert client.calls == 5

# --- Part 3: run_agent with the batched tools and prompt, on the same six questions ---
batched_total = 0
for q in ALL_QUESTIONS:
    answer, calls = run_and_count(run_agent, q, TOOLS_BATCHED, SYSTEM_PROMPT_BATCHED)
    assert CHECK_TEXT[q] in answer, (q, answer)
    assert calls <= 2, (q, calls)
    batched_total += calls
assert batched_total == 12 < baseline_total
print(f"calls across all six questions: {baseline_total} with get_quote alone, {batched_total} batched")

print("all checks passed")
```
