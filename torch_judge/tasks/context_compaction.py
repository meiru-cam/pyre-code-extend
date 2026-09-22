"""Token-budgeted history compaction that never splits a tool-call block."""

from torch_judge.tasks._schema import build_design_note_rubric

TASK = {
    "title": "Context Compaction",
    "difficulty": "Hard",
    "version": 1,
    "function_name": "compact_context",
    "description_en": r"""Compact a conversation so it fits a token budget, summarizing what you drop.

**Signature:** `compact_context(messages, token_budget, summary_budget, count_tokens, summarize) -> dict`

**Parameters:**
- `messages` — a list of message dictionaries. Every message has a non-empty string `role`. An assistant message may carry a non-empty `tool_calls` list; a message with role `tool` is the result of one such call.
- `token_budget` — non-negative integer. The returned conversation must cost no more than this.
- `summary_budget` — non-negative integer, no larger than `token_budget`. The room set aside for the summary when anything is dropped.
- `count_tokens` — `count_tokens(message) -> int`. The cost of one message. Call it as often as you like; it is pure.
- `summarize` — `summarize(dropped_messages) -> str`. Call it **at most once**, and only when something is actually dropped.

**Returns:** a dictionary with exactly four keys:

- `messages` — the compacted conversation, in order.
- `dropped` — the number of original messages that did not survive.
- `compacted` — True when a summary was inserted, False otherwise.
- `tokens` — the total `count_tokens` cost of `messages`.

**The pinned message.** If `messages[0]` has role `system`, that one message is pinned: it is never dropped, never summarized, and always stays first. Only index 0, and only one message.

**Blocks.** Group everything after the pinned message into atomic blocks:

- An assistant message with a non-empty `tool_calls` list opens a block, which extends through every `tool` message that immediately follows it.
- Every other message is a block of its own.

A block is kept whole or dropped whole. A `tool` message that does not immediately follow such an assistant message, or follows one only across an intervening message of another role, is malformed input.

**The rule:**

1. If the whole conversation already costs `token_budget` or less, return it unchanged. Do not call `summarize`.
2. Otherwise keep the **longest suffix of blocks** that costs no more than `token_budget - pinned - summary_budget`, scanning from the newest block backwards and stopping at the first block that does not fit.
3. Call `summarize` once with the dropped messages, in their original order. Insert `{"role": "system", "content": <the summary>, "compacted": True}` immediately after the pinned message, or at the front when there is no pinned message.

**Constraints:**
- Recency wins. Keep the newest blocks, not the oldest.
- The summary costs tokens too. Reserving `summary_budget` before choosing the suffix is the whole point of step 2.
- Never return a `tool` message whose opening assistant message was dropped.
- Do not mutate `messages` or any message in it. Return fresh dictionaries.
- Raise a `ValueError` when: `token_budget` or `summary_budget` is not a non-negative integer; `summary_budget` exceeds `token_budget`; a message has no non-empty string `role`; a `tool` message is orphaned; the pinned message alone exceeds `token_budget`; there is no room for the summary inside `token_budget`; or `summarize` returns a non-string, an empty string, or a summary costing more than `summary_budget`.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why blocks.** The tempting implementation walks backwards message by message until the budget runs out. It produces a conversation that starts with a `tool` message answering a call nobody made. Most providers reject that outright, and the ones that accept it give the model a result with no question attached. Tool-call atomicity is the single constraint that separates a working compactor from one that fails in production the first time a long tool result lands near the boundary.

**Why the summary budget is a parameter.** The cost of the summary depends on what you drop, and what you drop depends on how much room the summary leaves. That circularity has no clean fixed point, so production systems break it by reserving a fixed slice up front and treating an overrun as a failure of the summarizer rather than of the compactor. This exercise does the same, which is why an overrunning `summarize` raises instead of triggering a second pass.

**Why `compacted` is a separate flag.** `dropped == 0` and `compacted == False` are the same thing here, but the caller usually wants to log or meter compaction events, and a boolean that means exactly one thing survives refactoring better than a count that has to be compared against zero.

**What this leaves out.** Real compactors also score messages for importance rather than trusting recency, keep a pinned set that is not just index 0, summarize incrementally so the summary itself compounds, and re-order messages to keep a provider's prompt-cache prefix stable. Each of those is a separate decision on top of this one.""",
    "advisory_prerequisites": ["tool_registry"],
    "design_note_rubric": build_design_note_rubric(),
    "hints": [
        {
            "level": 1,
            "kind": "questions",
            "content": "Start with the grouping, not the budget. What exactly makes two adjacent messages inseparable, and what does your code do when a tool message appears with no assistant call in front of it? Once you have blocks, ask which end you scan from and why — if you scan from the oldest, what does the result look like? Then the budget: you subtract the pinned cost, but what is the second thing you must subtract before you start fitting blocks, and what goes wrong if you subtract it after? Finally, trace the case where the conversation already fits: how many times does summarize get called, and does your code agree?",
        },
        {
            "level": 2,
            "kind": "analysis",
            "content": "Build the block list in one forward pass: an assistant message with a non-empty tool_calls list appends a new block, a tool message appends to the current block after checking that the current block was opened by such an assistant message, and anything else appends a block of its own. That check is what turns an orphan tool message into a ValueError instead of a silently broken conversation. Then compute the total; the early return happens before any summary reasoning, so summarize stays uncalled. For the compaction path, walk the block list from the last index down, accumulating cost and breaking on the first block that would exceed token_budget minus pinned minus summary_budget — break, do not continue, or you will skip a large block and keep an older small one, which reorders nothing but silently drops a turn out of the middle of the window. Two errors here never raise on their own: reserving summary_budget after choosing the suffix leaves a result that is over budget by exactly the summary cost, and accumulating per message instead of per block cuts a tool block in half. Both pass a naive length check. The returned tokens value is pinned plus the summary cost plus the accumulated block cost, which should equal recounting the returned list — if it does not, you counted something you did not return.",
        },
    ],
    "model_connections": [
        "LangChain's trim_messages solves the same budget problem with strategy='last' and include_system, and its start_on parameter exists for the same reason blocks do here: a trimmed window that opens on the wrong role is invalid to the provider.",
        "Agent frameworks that summarize rather than trim insert the summary as a system message near the front, so the model reads the compressed past before the live window, which is the placement this exercise requires.",
        "In agentic RL the same boundary shows up during rollout: a trajectory that outgrows the context window has to be cut somewhere, and cutting inside a tool-call block corrupts the trajectory for both the rollout and the loss mask computed from it.",
    ],
    "pro_con_analysis": {
        "pros": [
            "Recency plus a summary keeps the live window exact and the distant past approximate, which matches how most agent errors are actually caused by the last few turns.",
            "Block atomicity makes every returned conversation valid to send, with no provider-specific repair pass afterwards.",
            "Reserving a fixed summary budget makes the result's cost predictable before the summarizer runs.",
        ],
        "cons": [
            "Recency is a crude importance proxy: a critical fact stated in turn two is compressed into prose while three turns of retries survive intact.",
            "A single large block near the boundary can force a much smaller window than the budget would otherwise allow, because the block cannot be split.",
            "Summarizing once per compaction loses information cumulatively; a long-running agent ends up summarizing summaries with no record of what was already lost.",
        ],
    },
    "sources": [
        {
            "kind": "code",
            "url": "https://github.com/langchain-ai/langchain",
            "commit": "877ad8c7bfa3f74377caa6fd33cbc20687e0052f",
            "path": "libs/core/langchain_core/messages/utils.py",
            "symbol": "trim_messages",
            "license": "MIT",
            "adapted": "The token-budgeted last-strategy trim: keep the newest messages within a budget, pin a leading system message, and refuse to start the kept window on a role that would be invalid to send.",
            "simplifications": "Replaced the message classes with plain dictionaries and the tokenizer with an injected count_tokens callable. Dropped partial-message splitting, the 'first' strategy, and the end_on parameter. Added the summary step and its reserved budget, which trim_messages does not have: it drops the old messages outright.",
        },
    ],
    "tests": [
        {
            "name": "A conversation that already fits is returned unchanged",
            "code": r"""
calls = []
def count(m):
    return len(m['content'])
def summarize(dropped):
    calls.append(list(dropped))
    return 'summary'

messages = [
    {'role': 'system', 'content': 'sys'},
    {'role': 'user', 'content': 'hello'},
    {'role': 'assistant', 'content': 'hi'},
]
out = {fn}(messages, 100, 20, count, summarize)
assert out['compacted'] is False, out['compacted']
assert out['dropped'] == 0, out['dropped']
assert out['tokens'] == 3 + 5 + 2, out['tokens']
assert [m['role'] for m in out['messages']] == ['system', 'user', 'assistant']
assert calls == [], 'summarize must not be called when nothing is dropped'
""",
        },
        {
            "name": "Dropping keeps the newest blocks and inserts the summary after the system message",
            "code": r"""
def count(m):
    return len(m['content'])
def summarize(dropped):
    return 'S' * 4

messages = [
    {'role': 'system', 'content': 'sys'},
    {'role': 'user', 'content': 'aaaaaaaaaa'},
    {'role': 'assistant', 'content': 'bbbbbbbbbb'},
    {'role': 'user', 'content': 'cc'},
    {'role': 'assistant', 'content': 'dd'},
]
out = {fn}(messages, 20, 6, count, summarize)
assert out['compacted'] is True, out['compacted']
assert out['dropped'] == 2, out['dropped']
roles = [m['role'] for m in out['messages']]
assert roles == ['system', 'system', 'user', 'assistant'], roles
assert out['messages'][1]['content'] == 'SSSS', out['messages'][1]
assert out['messages'][1].get('compacted') is True, out['messages'][1]
assert [m['content'] for m in out['messages'][2:]] == ['cc', 'dd']
assert out['tokens'] == 3 + 4 + 2 + 2, out['tokens']
assert out['tokens'] <= 20
""",
        },
        {
            "name": "A tool block is kept or dropped whole",
            "visibility": "unshown",
            "behavior": "protocol.validation",
            "failure_message": "A tool message survived without the assistant message that called it. An assistant turn with tool_calls and the tool results that follow it form one block that is kept or dropped together.",
            "code": r"""
def count(m):
    return m.get('cost', len(m['content']))
def summarize(dropped):
    return 'sum'

messages = [
    {'role': 'user', 'content': 'q', 'cost': 4},
    {'role': 'assistant', 'content': 'call', 'cost': 2, 'tool_calls': [{'id': 'c1', 'name': 't'}]},
    {'role': 'tool', 'content': 'r1', 'cost': 9, 'tool_call_id': 'c1'},
    {'role': 'tool', 'content': 'r2', 'cost': 9, 'tool_call_id': 'c1'},
]
# Budget leaves room for the two cheap messages but not the 20-token tool block.
out = {fn}(messages, 12, 3, count, summarize)
roles = [m['role'] for m in out['messages']]
assert 'tool' not in roles, roles
assert not any(m.get('tool_calls') for m in out['messages']), roles
assert out['dropped'] == 4, out['dropped']

# With room for the block, all four survive together.
wide = {fn}(messages, 100, 10, count, summarize)
assert [m['role'] for m in wide['messages']] == ['user', 'assistant', 'tool', 'tool']
""",
        },
        {
            "name": "The summary budget is reserved before the suffix is chosen",
            "visibility": "unshown",
            "behavior": "budget.enforcement",
            "failure_message": "The compacted conversation cost more than token_budget. The summary is counted too, so its budget has to be subtracted before blocks are fitted, not after.",
            "code": r"""
def count(m):
    return m.get('cost', len(m['content']))
def summarize(dropped):
    return 'x' * 7

for budget, reserve in ((30, 7), (24, 7), (18, 7), (40, 12)):
    messages = [{'role': 'system', 'content': 's', 'cost': 5}] + [
        {'role': 'user', 'content': str(i), 'cost': 6} for i in range(8)
    ]
    out = {fn}(messages, budget, reserve, count, summarize)
    total = sum(count(m) for m in out['messages'])
    assert total <= budget, f'budget {budget}: cost {total}'
    assert out['tokens'] == total, (out['tokens'], total)
""",
        },
        {
            "name": "Summarize is called once with exactly the dropped messages in order",
            "visibility": "unshown",
            "behavior": "state.invariant",
            "failure_message": "summarize was called the wrong number of times or with the wrong messages. It takes the dropped messages, in their original order, exactly once.",
            "code": r"""
calls = []
def count(m):
    return m.get('cost', len(m['content']))
def summarize(dropped):
    calls.append([m['content'] for m in dropped])
    return 'sum'

messages = [{'role': 'system', 'content': 'sys', 'cost': 2}] + [
    {'role': 'user', 'content': f'm{i}', 'cost': 5} for i in range(6)
]
out = {fn}(messages, 24, 4, count, summarize)
assert len(calls) == 1, f'summarize called {len(calls)} times'
kept = [m['content'] for m in out['messages'][2:]]
assert calls[0] + kept == [f'm{i}' for i in range(6)], (calls[0], kept)
assert out['dropped'] == len(calls[0]), (out['dropped'], calls[0])
""",
        },
        {
            "name": "The leading system message is pinned, never summarized",
            "visibility": "unshown",
            "behavior": "state.invariant",
            "failure_message": "The leading system message was dropped, summarized, or moved. It stays first and is never handed to summarize.",
            "code": r"""
seen = []
def count(m):
    return m.get('cost', len(m['content']))
def summarize(dropped):
    seen.extend(m['role'] for m in dropped)
    return 'sum'

messages = [{'role': 'system', 'content': 'sys', 'cost': 4}] + [
    {'role': 'user', 'content': str(i), 'cost': 9} for i in range(5)
]
out = {fn}(messages, 20, 3, count, summarize)
assert out['messages'][0]['content'] == 'sys', out['messages'][0]
assert 'system' not in seen, seen

# No pinned message: the summary goes to the very front.
tail = [{'role': 'user', 'content': str(i), 'cost': 9} for i in range(5)]
out2 = {fn}(tail, 20, 3, count, summarize)
assert out2['messages'][0].get('compacted') is True, out2['messages'][0]
""",
        },
        {
            "name": "The input is never mutated and the output holds fresh dictionaries",
            "visibility": "unshown",
            "behavior": "state.invariant",
            "failure_message": "The caller's messages were mutated or handed back by reference. Copy before returning so a later edit cannot reach back into the conversation.",
            "code": r"""
import copy
def count(m):
    return m.get('cost', len(m['content']))
def summarize(dropped):
    return 'sum'

messages = [{'role': 'system', 'content': 'sys', 'cost': 3}] + [
    {'role': 'user', 'content': str(i), 'cost': 7} for i in range(6)
]
before = copy.deepcopy(messages)
out = {fn}(messages, 26, 4, count, summarize)
assert messages == before, 'input was mutated'
originals = {id(m) for m in messages}
assert not any(id(m) in originals for m in out['messages']), 'returned a caller dictionary by reference'

# The same holds on the path where nothing is dropped.
whole = {fn}(messages, 1000, 4, count, summarize)
assert whole['compacted'] is False, whole['compacted']
assert messages == before, 'input was mutated'
assert not any(id(m) in originals for m in whole['messages']), 'returned a caller dictionary by reference'
""",
        },
        {
            "name": "Malformed input and impossible budgets raise ValueError",
            "visibility": "unshown",
            "behavior": "contract.signature",
            "failure_message": "An invalid budget, an orphaned tool message, or an oversized summary was accepted instead of raising ValueError.",
            "code": r"""
def count(m):
    return m.get('cost', len(m['content']))
def summarize(dropped):
    return 'sum'
def huge(dropped):
    return 'x' * 500

ok = [{'role': 'user', 'content': 'a', 'cost': 9} for _ in range(5)]
orphan = [{'role': 'user', 'content': 'a', 'cost': 1}, {'role': 'tool', 'content': 'r', 'cost': 1}]
split = [
    {'role': 'assistant', 'content': 'c', 'cost': 1, 'tool_calls': [{'id': 'c1'}]},
    {'role': 'user', 'content': 'u', 'cost': 1},
    {'role': 'tool', 'content': 'r', 'cost': 1},
]
cases = (
    (ok, -1, 2, summarize),
    (ok, 10, -1, summarize),
    (ok, 10, 20, summarize),
    (ok, 10, True, summarize),
    ([{'role': '', 'content': 'x', 'cost': 1}], 10, 2, summarize),
    (orphan, 10, 2, summarize),
    (split, 10, 2, summarize),
    ([{'role': 'system', 'content': 's', 'cost': 50}] + ok, 20, 2, summarize),
    (ok, 20, 2, huge),
    (ok, 20, 2, lambda dropped: ''),
    (ok, 20, 2, lambda dropped: None),
)
for messages, budget, reserve, fn_sum in cases:
    try:
        {fn}(messages, budget, reserve, count, fn_sum)
    except ValueError:
        continue
    raise AssertionError(f'budget={budget} reserve={reserve} should raise ValueError')
""",
        },
        {
            "name": "Empty and boundary conversations behave",
            "visibility": "unshown",
            "behavior": "edge.empty_or_boundary",
            "failure_message": "An empty conversation, an exact-fit budget, or a window with no room for any block was handled wrong.",
            "code": r"""
def count(m):
    return m.get('cost', len(m['content']))
def summarize(dropped):
    return 'ss'

empty = {fn}([], 10, 2, count, summarize)
assert empty['messages'] == [] and empty['dropped'] == 0 and empty['compacted'] is False, empty
assert empty['tokens'] == 0, empty

exact = [{'role': 'user', 'content': 'a', 'cost': 5}, {'role': 'user', 'content': 'b', 'cost': 5}]
out = {fn}(exact, 10, 3, count, summarize)
assert out['compacted'] is False and out['dropped'] == 0, out

# summarize returns 2 tokens; reserve is 2, so nothing else fits.
tight = [{'role': 'user', 'content': str(i), 'cost': 4} for i in range(3)]
out2 = {fn}(tight, 2, 2, count, summarize)
assert out2['dropped'] == 3, out2
assert [m.get('compacted') for m in out2['messages']] == [True], out2['messages']
assert out2['tokens'] == 2, out2
""",
        },
    ],
    "solution": '''def compact_context(messages, token_budget, summary_budget, count_tokens, summarize):
    if not isinstance(messages, list):
        raise ValueError("messages must be a list")
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("every message must be a dictionary")
        role = message.get("role")
        if not isinstance(role, str) or not role:
            raise ValueError("every message needs a non-empty string role")
    for name, value in (("token_budget", token_budget), ("summary_budget", summary_budget)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if summary_budget > token_budget:
        raise ValueError("summary_budget cannot exceed token_budget")

    pinned = messages[:1] if messages and messages[0]["role"] == "system" else []

    blocks = []
    for message in messages[len(pinned):]:
        if message["role"] == "tool":
            if not blocks or not blocks[-1][0].get("tool_calls"):
                raise ValueError("a tool message must follow the assistant turn that called it")
            blocks[-1].append(message)
        else:
            blocks.append([message])

    pinned_tokens = sum(count_tokens(message) for message in pinned)
    if pinned_tokens > token_budget:
        raise ValueError("the pinned system message alone exceeds token_budget")

    block_tokens = [sum(count_tokens(message) for message in block) for block in blocks]
    total = pinned_tokens + sum(block_tokens)
    if total <= token_budget:
        return {
            "messages": [dict(message) for message in messages],
            "dropped": 0,
            "compacted": False,
            "tokens": total,
        }

    if pinned_tokens + summary_budget > token_budget:
        raise ValueError("no room for the summary inside token_budget")
    available = token_budget - pinned_tokens - summary_budget

    start = len(blocks)
    used = 0
    for index in range(len(blocks) - 1, -1, -1):
        if used + block_tokens[index] > available:
            break
        used += block_tokens[index]
        start = index

    dropped = [message for block in blocks[:start] for message in block]
    text = summarize(dropped)
    if not isinstance(text, str) or not text.strip():
        raise ValueError("summarize must return a non-empty string")
    summary = {"role": "system", "content": text, "compacted": True}
    summary_tokens = count_tokens(summary)
    if summary_tokens > summary_budget:
        raise ValueError("the summary exceeded summary_budget")

    kept = [dict(message) for block in blocks[start:] for message in block]
    return {
        "messages": [dict(message) for message in pinned] + [summary] + kept,
        "dropped": len(dropped),
        "compacted": True,
        "tokens": pinned_tokens + summary_tokens + used,
    }
''',
}
