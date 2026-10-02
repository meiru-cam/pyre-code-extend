A notebook defines `client`, an object with one method, `client.create(model, system, messages, tools, max_tokens) -> Response`, which sends the running conversation to a language model and returns its next turn.

`messages` is a list of `{"role": "user" | "assistant", "content": [...]}` dictionaries, oldest first. `content` is a list of *content blocks*, each one of:

- `{"type": "text", "text": "..."}` — plain text.
- `{"type": "tool_use", "id": "...", "name": "...", "input": {...}}` — a request from the model to call one tool. `id` identifies this particular request; `name` names one of the tools offered in this call's `tools`; `input` is that tool's arguments, already validated against the tool's schema (below). A `tool_use` block appears only inside an assistant message, because only the model produces one.
- `{"type": "tool_result", "tool_use_id": "...", "content": "...", "is_error": False}` — the outcome of running a tool, handed back to the model. `tool_use_id` must equal the `id` of the `tool_use` block it answers; `content` is the tool's output, or a description of the failure, as a string; `is_error` is `True` exactly when the tool call failed. A `tool_result` block appears only inside a user message, because only the calling code produces one — never the model.

`tools` is a list of `{"name": "...", "description": "...", "input_schema": {...}}` dictionaries: each tool's name, a natural-language description the model uses to decide when and how to call it, and a JSON Schema object (`{"type": "object", "properties": {...}, "required": [...]}`) restricting `input` to those properties. `max_tokens` caps the length of the model's reply and does not otherwise affect correctness here; the notebook passes a fixed constant.

The return value, a `Response`, has two attributes: `.content`, a list of content blocks in the shapes above (a response only ever contains `text` and `tool_use` blocks, never `tool_result`), and `.stop_reason`, a string that in this notebook is always either `"end_turn"` (the model is done and is not asking for a tool) or `"tool_use"` (`.content` contains at least one `tool_use` block that must be answered before the conversation can continue). A single response's `.content` may hold more than one `tool_use` block at once, when the model asks for several tool calls together.

Answering one question is this loop:

- Start `messages` with one message, `{"role": "user", "content": [{"type": "text", "text": question}]}`.
- Call `client.create(...)` with the current `messages`.
- If the response's `stop_reason` is `"end_turn"`, its `content` holds exactly one `text` block; that block's `text` is the answer, and the loop ends.
- If `stop_reason` is `"tool_use"`, append `{"role": "assistant", "content": response.content}` to `messages` — the model's own turn, exactly as produced, including every block. Then, for every `tool_use` block in `response.content`, run the tool it names on its `input`, and build one `tool_result` block whose `tool_use_id` equals that block's own `id`; collect all of these `tool_result` blocks into one new message, `{"role": "user", "content": [...]}`, and append it to `messages`. Go back to step 2.

In this notebook, whenever `stop_reason` is `"tool_use"`, every block of `.content` is a `tool_use` block; whenever `stop_reason` is `"end_turn"`, `.content` holds exactly the one `text` block described above.

The notebook offers one tool at first, `get_quote`, backed by the closing-price table below:

```text
              2026-01-05   2026-01-06   2026-01-07   2026-01-08   2026-01-09   2026-01-12
AAPL             180.00       181.50       179.00       184.00       190.00       188.00
MSFT             410.00       406.00       412.00       415.00       420.00       418.00
GOOG             140.00       141.40       139.00       142.50       145.00       146.50
```

No other ticker, and no other date — including 2026-01-10 and 2026-01-11, a Saturday and a Sunday — has any data. `get_quote(ticker: str, date: str) -> float` returns the closing price at the row and column of this table; called with a ticker not listed, it fails with the message `f"unknown ticker: {ticker}"`; called with a date not listed for a ticker that is, it fails with `f"no trading data for {date}"`.

The notebook also provides `run_tool(name: str, input: dict) -> tuple[str, bool]`: it calls the Python function registered for the tool named `name` with the keyword arguments in `input`, and returns `(str(result), False)`. If that call raises, `run_tool` catches the exception instead of propagating it and returns `(str(exception), True)` instead — a failing tool never crashes the loop; carrying `(text, is_error)` into the `tool_result` block above is the loop's job.

### Part 1 — Fix the starter agent

`SINGLE_CALL_QUESTIONS` lists three questions that each need exactly one call to `get_quote`:

```text
"What was AAPL's closing price on 2026-01-07?"   -> AAPL, 2026-01-07 -> 179.00
"What was MSFT's closing price on 2026-01-09?"   -> MSFT, 2026-01-09 -> 420.00
"What was GOOG's closing price on 2026-01-05?"   -> GOOG, 2026-01-05 -> 140.00
```

The function below is a first attempt at the loop above; it has exactly four bugs against it. Fixed, it returns a string stating the right price for each of the three questions above, and — for these or any other single-tool-call question — never departs from the loop above: it always sends the model's own turn back before the tool results that answer it, always pairs a `tool_result` with the `id` of the `tool_use` block it answers (not the tool's `name`, which is the same string on every call to that tool), answers every `tool_use` block a response contains rather than only one of them, and keeps exchanging turns with the model until `stop_reason` is `"end_turn"` rather than assuming one round of tool calls is always enough. Fixed, it also takes exactly two calls to `client.create` per question: one that receives the `tool_use` request, one that receives the final text after seeing the `tool_result`.

```py
def answer_question(question: str) -> str:
    messages = [{"role": "user", "content": [{"type": "text", "text": question}]}]
    response = client.create(model=MODEL_NAME, system=SYSTEM_PROMPT, messages=messages,
                              tools=[GET_QUOTE_TOOL], max_tokens=MAX_TOKENS)

    if response.stop_reason == "tool_use":
        tool_use = next(b for b in response.content if b["type"] == "tool_use")
        text, is_error = run_tool(tool_use["name"], tool_use["input"])
        messages.append({
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": tool_use["name"],
                         "content": text, "is_error": is_error}],
        })
        response = client.create(model=MODEL_NAME, system=SYSTEM_PROMPT, messages=messages,
                                  tools=[GET_QUOTE_TOOL], max_tokens=MAX_TOKENS)

    return response.content[0]["text"]
```

Trace of the first question above:

```text
messages = [{"role": "user", "content": [{"type": "text", "text":
             "What was AAPL's closing price on 2026-01-07?"}]}]

# call 1: no price is known yet, so the model asks for one
response.stop_reason = "tool_use"
response.content = [{"type": "tool_use", "id": "call_1", "name": "get_quote",
                      "input": {"ticker": "AAPL", "date": "2026-01-07"}}]

# the loop keeps the model's own turn, runs the tool, and hands back the result
messages += [{"role": "assistant", "content": response.content},
             {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_1",
                                            "content": "179.0", "is_error": False}]}]

# call 2: the model has the price and answers
response.stop_reason = "end_turn"
response.content = [{"type": "text", "text": "AAPL closed at $179.00 on 2026-01-07."}]
# the loop returns "AAPL closed at $179.00 on 2026-01-07."
```

### Part 2 — Questions that need several results

Generalize `answer_question` into

```py
def run_agent(question: str, tools: list[dict], system: str, max_calls: int = 8) -> str: ...
```

taking the tool catalog, the system prompt and a cap on rounds as arguments instead of the fixed globals above (Part 3 reuses it unchanged, with a different catalog and prompt). If the model has not reached `stop_reason == "end_turn"` after `max_calls` calls to `client.create`, stop and return the string `f"gave up after {max_calls} calls"` instead of calling `create` again.

With `tools=[GET_QUOTE_TOOL]` and `system=SYSTEM_PROMPT`, `run_agent` must answer every question in `MULTI_CALL_QUESTIONS`:

```text
"Between 2026-01-05 and 2026-01-09, did AAPL or MSFT gain more, in percentage terms?"
    -> names AAPL: (190.00 - 180.00) / 180.00 ~= 5.6%, against MSFT's (420.00 - 410.00) / 410.00 ~= 2.4%
"What was GOOG's closing price on 2026-01-10?"
    -> states that there is no trading data for that day
"Which of AAPL, MSFT and GOOG had the highest closing price on 2026-01-08?"
    -> names MSFT: 415.00 against 184.00 and 142.50
```

The first question needs four calls to `get_quote` (both tickers, at both dates); the third needs three (one call per ticker). With only `get_quote` offered, and nothing telling it to do otherwise, the model asks for one price at a time, so answering the three questions above costs `4 + 1`, `1 + 1` and `3 + 1` calls to `create` — 5, 2 and 4 — 11 in total. The second question's single call to `get_quote` fails, since there is no row for 2026-01-10; the loop must still hand the failure back as a `tool_result` with `is_error=True` and let the model produce an answer from it, rather than raising out of the loop.

Trace of the second question:

```text
messages[0] = {"role": "user", "content": [{"type": "text", "text":
               "What was GOOG's closing price on 2026-01-10?"}]}

# call 1
response.content = [{"type": "tool_use", "id": "call_1", "name": "get_quote",
                      "input": {"ticker": "GOOG", "date": "2026-01-10"}}]
response.stop_reason = "tool_use"

# run_tool("get_quote", {"ticker": "GOOG", "date": "2026-01-10"}) raises ValueError
# ("no trading data for 2026-01-10"); run_tool catches it and returns
# ("no trading data for 2026-01-10", True)
messages += [{"role": "assistant", "content": response.content},
             {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_1",
                                            "content": "no trading data for 2026-01-10", "is_error": True}]}]

# call 2: is_error does not stop the loop -- it is just another tool_result for the model to read
response.content = [{"type": "text", "text":
                      "I could not get GOOG's price on 2026-01-10: no trading data for 2026-01-10."}]
response.stop_reason = "end_turn"
```

### Part 3 — Fewer calls to `create`

Combined with Part 1's three questions (already at two calls each, the minimum for a question that needs a tool at all), the six questions of `SINGLE_CALL_QUESTIONS` and `MULTI_CALL_QUESTIONS` cost `6 + 11 = 17` calls to `client.create` under `tools=[GET_QUOTE_TOOL]` and `system=SYSTEM_PROMPT`. Without changing `run_agent`, bring this down to at most two calls per question — at most 12 in total — by changing only what is offered to the model:

- a second tool, `get_quotes`, that looks up several `(ticker, date)` pairs in one call:

  ```py
  def get_quotes(lookups: list[dict]) -> list[float]: ...
  ```

  `lookups` is a list of `{"ticker": str, "date": str}` dictionaries; the result lists one price per pair, in the same order. If any pair is invalid, `get_quotes` fails the same way `get_quote` would for that pair, scanning `lookups` in order, and returns nothing for the rest of the list — there is no partial success.
- a system prompt that tells the model to gather every price a question needs before answering: with `get_quotes` offered, in one call to it; without it, as several `get_quote` calls issued together, in parallel, in the same turn, rather than one call per turn.

Trace of the first `MULTI_CALL_QUESTIONS` question against this pair:

```text
messages[0] = {"role": "user", "content": [{"type": "text", "text":
               "Between 2026-01-05 and 2026-01-09, did AAPL or MSFT gain more, in percentage terms?"}]}

# call 1: every price the question needs is known up front, so the model asks for all four at once
response.content = [{"type": "tool_use", "id": "call_1", "name": "get_quotes",
                      "input": {"lookups": [{"ticker": "AAPL", "date": "2026-01-05"},
                                             {"ticker": "AAPL", "date": "2026-01-09"},
                                             {"ticker": "MSFT", "date": "2026-01-05"},
                                             {"ticker": "MSFT", "date": "2026-01-09"}]}}]
response.stop_reason = "tool_use"

# run_tool("get_quotes", ...) returns ("[180.0, 190.0, 410.0, 420.0]", False)
messages += [{"role": "assistant", "content": response.content},
             {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_1",
                                            "content": "[180.0, 190.0, 410.0, 420.0]", "is_error": False}]}]

# call 2: two calls total, against five for the same question in Part 2
response.content = [{"type": "text", "text":
                      "AAPL gained more in percentage terms between 2026-01-05 and 2026-01-09."}]
response.stop_reason = "end_turn"
```
