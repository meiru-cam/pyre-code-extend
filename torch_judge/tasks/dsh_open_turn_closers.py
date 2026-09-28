"""Crash recovery for an append-only agent session log: close the open turn."""

from ._interview import interview

TASK = {
    "title": "Close an Interrupted Session Turn",
    "difficulty": "Medium",
    "version": 1,
    "function_name": "open_turn_closers",
    "description_en": r"""After a crash, an agent's append-only session log can end in the middle of a turn. Compute the synthetic events that close it so the log can be resumed.

**Signature:** `open_turn_closers(events, cause="interrupted") -> list[dict]`

**Input events.** `events` is a list of dicts in log order, each with `type`, an integer `seq` (strictly increasing), a `time` and a `data` dict:
- `turn/start` `{turn}`, `turn/end` `{turn, reason}`
- `step/start` `{turn, step}`, `step/end` `{turn, step}`
- `assistant/message` `{turn, step, tool_call_ids}` — the ids of the tool calls the model requested, in order
- `tool/call` `{call_id}` — the harness recorded that the call started
- `tool/result` `{turn, step, call_id, surface_op}` — `surface_op` is `"append"` for a new result or `"replace"` for a rewrite of an older one
- any other type — ignore it

**Pending calls.** Scan the events in order:
- `turn/start`, `turn/end` and `step/end` forget every pending call.
- `assistant/message` makes each of its ids pending with that message's turn and step, not started. If an id is already pending, reset its entry to the new turn and step and to not started, but keep its original position in the pending order.
- `tool/call` marks a pending id as started and remembers the `seq` of that event.
- `tool/result` with `surface_op == "append"` and the same turn and step as the pending entry answers it and removes it.

**Returns.** If the log is empty or has no open turn (no `turn/start` after the last `turn/end`), return `[]`. Otherwise, with `last` the final event, return in this order:
- one `tool/result` event per pending call, in pending order, with `data = {"turn", "step", "call_id", "surface_op": "append", "is_error": True, "started": bool, "error_code": "TOOL_OUTCOME_UNKNOWN" if started else "TOOL_NOT_STARTED"}`, plus `"source_event_seqs": [seq of its tool/call]` in `data` when started;
- a `step/end` event `{turn, step}` if a step is still open;
- a `turn/end` event `{turn, reason: {"kind": cause}}`.

Synthetic events get consecutive `seq` values starting at `last["seq"] + 1` and reuse `last["time"]`.

**Constraints:**
- Do not modify the input.
- Raise `ValueError` when `cause` is not `"interrupted"` or `"forked"`.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why results must be synthesized.** Model APIs reject a history in which an assistant tool call has no matching result, so an unanswered call makes the session impossible to resume.

**Started versus not started.** A call with no start record never ran, so it is safe to retry. A call that started but has no durable result may have had side effects: the harness tells the model to retry only read-only or idempotent operations and otherwise to check external state or ask the user.

**Why append, never rewrite.** The log is the source of the model's context. Appending closers keeps every earlier event, and therefore the provider's KV cache prefix, intact. The same closers seal a fork that is cut in the middle of the parent's turn, with `cause="forked"`.""",
    "advisory_prerequisites": ["dsh_event_bus"],
    "hints": [
        {"level": 1, "kind": "questions", "content": "Which events end the lifetime of a pending call? What makes a tool result count as the answer to a pending call? Where do the step and turn closers get their numbers from?"},
        {"level": 2, "kind": "analysis", "content": "Keep open_turn and open_step cursors and an insertion-ordered dict of pending call_id -> {turn, step, call_seq}. Update the cursors and the dict per event type. At the end, return [] when open_turn is None; otherwise emit one result per pending entry, then step/end if open_step is set, then turn/end, numbering seq from last seq + 1."},
    ],
    "model_connections": [
        "DeepSeek Harness runs this recovery on resume and on fork; OpenAI and Anthropic chat APIs both reject assistant tool calls without matching results.",
    ],
    "pro_con_analysis": {
        "pros": ["The log stays append-only and replayable, and the model gets an honest, retry-safe explanation of what is unknown."],
        "cons": ["The model must reason about possibly duplicated side effects; the harness cannot know whether a started call finished."],
    },
    "sources": [
        {"kind": "code", "url": "https://github.com/deepseek-ai/deepseek-harness", "commit": "21638c56315ae6a2b552d6091945d3144c9af32e", "path": "packages/core/session/src/repair.ts", "symbol": "openTurnClosers and ToolCallRecovery.observe and results", "license": "MIT", "adapted": "Turn and step cursors, pending-call tracking with started records, append-only answer matching, synthetic result ordering and seq numbering.", "simplifications": "Model-visible wording, message ids and frozen message objects are replaced by started and error_code fields."},
    ],
    "tests": [
        {"name": "Closes an open step with one unanswered call", "behavior": "checkpoint.recovery", "code": r"""
events = [
    {"type": "turn/start", "seq": 1, "time": 10, "data": {"turn": 1}},
    {"type": "step/start", "seq": 2, "time": 11, "data": {"turn": 1, "step": 1}},
    {"type": "assistant/message", "seq": 3, "time": 12, "data": {"turn": 1, "step": 1, "tool_call_ids": ["a", "b"]}},
    {"type": "tool/call", "seq": 4, "time": 13, "data": {"call_id": "a"}},
    {"type": "tool/result", "seq": 5, "time": 14, "data": {"turn": 1, "step": 1, "call_id": "a", "surface_op": "append"}},
    {"type": "tool/call", "seq": 6, "time": 15, "data": {"call_id": "b"}},
]
out = {fn}(events)
assert out == [
    {"type": "tool/result", "seq": 7, "time": 15, "data": {"turn": 1, "step": 1, "call_id": "b", "surface_op": "append", "is_error": True, "started": True, "error_code": "TOOL_OUTCOME_UNKNOWN", "source_event_seqs": [6]}},
    {"type": "step/end", "seq": 8, "time": 15, "data": {"turn": 1, "step": 1}},
    {"type": "turn/end", "seq": 9, "time": 15, "data": {"turn": 1, "reason": {"kind": "interrupted"}}},
], out
"""},
        {"name": "Matches a seeded replay oracle", "visibility": "unshown", "behavior": "checkpoint.recovery", "failure_message": "Track pending calls per the rules, answer them only with appended results from the same turn and step, and number closers after the last event.", "code": r"""
import copy, random
def oracle(events, cause):
    open_turn = open_step = None
    pending = {}
    for e in events:
        t, d = e["type"], e["data"]
        if t in ("turn/start", "turn/end", "step/end"):
            pending = {}
        if t == "turn/start":
            open_turn, open_step = d["turn"], None
        elif t == "turn/end":
            open_turn = open_step = None
        elif t == "step/start":
            open_step = d["step"]
        elif t == "step/end":
            open_step = None
        elif t == "assistant/message":
            for cid in d["tool_call_ids"]:
                pending[cid] = {"turn": d["turn"], "step": d["step"], "call": None}
        elif t == "tool/call" and d["call_id"] in pending:
            pending[d["call_id"]]["call"] = e["seq"]
        elif t == "tool/result" and d["call_id"] in pending:
            p = pending[d["call_id"]]
            if d["surface_op"] == "append" and p["turn"] == d["turn"] and p["step"] == d["step"]:
                del pending[d["call_id"]]
    if not events or open_turn is None:
        return []
    seq, time, out = events[-1]["seq"], events[-1]["time"], []
    for cid, p in pending.items():
        seq += 1
        data = {"turn": p["turn"], "step": p["step"], "call_id": cid, "surface_op": "append", "is_error": True,
                "started": p["call"] is not None, "error_code": "TOOL_OUTCOME_UNKNOWN" if p["call"] is not None else "TOOL_NOT_STARTED"}
        if p["call"] is not None:
            data["source_event_seqs"] = [p["call"]]
        out.append({"type": "tool/result", "seq": seq, "time": time, "data": data})
    if open_step is not None:
        seq += 1
        out.append({"type": "step/end", "seq": seq, "time": time, "data": {"turn": open_turn, "step": open_step}})
    out.append({"type": "turn/end", "seq": seq + 1, "time": time, "data": {"turn": open_turn, "reason": {"kind": cause}}})
    return out
for seed in (8, 33, 71):
    rng = random.Random(seed)
    for trial in range(60):
        events, counter, turn, step, ids = [], [0], 0, 0, 0
        def add(t, **data):
            counter[0] += rng.randint(1, 3)
            events.append({"type": t, "seq": counter[0], "time": 100 + counter[0], "data": data})
        for _ in range(rng.randint(1, 3)):
            turn += 1; add("turn/start", turn=turn)
            for _ in range(rng.randint(0, 3)):
                step += 1; add("step/start", turn=turn, step=step)
                calls = [f"c{ids + i}" for i in range(rng.randint(0, 3))]
                if calls and rng.random() < 0.2:
                    calls.append(calls[0])
                ids += len(calls)
                add("assistant/message", turn=turn, step=step, tool_call_ids=calls)
                if rng.random() < 0.2:
                    add("progress", note="noise")
                for cid in calls:
                    if rng.random() < 0.7:
                        add("tool/call", call_id=cid)
                    if rng.random() < 0.5:
                        add("tool/result", turn=turn, step=rng.choice([step, step, step - 1]), call_id=cid, surface_op=rng.choice(["append", "append", "replace"]))
                if rng.random() < 0.6:
                    add("step/end", turn=turn, step=step)
            if rng.random() < 0.5:
                add("turn/end", turn=turn, reason={"kind": "done"})
        cut = rng.randint(0, len(events))
        prefix = events[:cut]
        cause = rng.choice(["interrupted", "forked"])
        frozen = copy.deepcopy(prefix)
        got = {fn}(prefix, cause)
        assert prefix == frozen, "input was modified"
        assert got == oracle(prefix, cause), (seed, trial, prefix, got)
"""},
        {"name": "Balanced logs and closed steps need no closers", "visibility": "unshown", "behavior": "edge.empty_or_boundary", "failure_message": "Return nothing for an empty or balanced log; step/end forgets unanswered calls and only an open step gets a synthetic step/end.", "code": r"""
assert {fn}([]) == []
done = [{"type": "turn/start", "seq": 1, "time": 0, "data": {"turn": 1}},
        {"type": "assistant/message", "seq": 2, "time": 0, "data": {"turn": 1, "step": 1, "tool_call_ids": ["x"]}},
        {"type": "turn/end", "seq": 3, "time": 0, "data": {"turn": 1, "reason": {"kind": "done"}}}]
assert {fn}(done) == []
closed = done[:2] + [{"type": "step/end", "seq": 3, "time": 5, "data": {"turn": 1, "step": 1}}]
assert {fn}(closed, "forked") == [{"type": "turn/end", "seq": 4, "time": 5, "data": {"turn": 1, "reason": {"kind": "forked"}}}]
again = [{"type": "turn/start", "seq": 1, "time": 0, "data": {"turn": 1}},
         {"type": "step/start", "seq": 2, "time": 0, "data": {"turn": 1, "step": 1}},
         {"type": "assistant/message", "seq": 3, "time": 0, "data": {"turn": 1, "step": 1, "tool_call_ids": ["k", "m"]}},
         {"type": "tool/call", "seq": 4, "time": 0, "data": {"call_id": "k"}},
         {"type": "assistant/message", "seq": 5, "time": 0, "data": {"turn": 1, "step": 1, "tool_call_ids": ["m", "k"]}}]
out = {fn}(again)
assert [(e["data"]["call_id"], e["data"]["started"]) for e in out[:2]] == [("k", False), ("m", False)], out
"""},
        {"name": "Rejects an unknown cause", "visibility": "unshown", "behavior": "contract.signature", "failure_message": "cause must be interrupted or forked.", "code": r"""
try:
    {fn}([], "crashed")
except ValueError:
    pass
else:
    raise AssertionError("accepted an unknown cause")
"""},
    ],
    "solution": '''CAUSES = ("interrupted", "forked")


def open_turn_closers(events, cause="interrupted"):
    if cause not in CAUSES:
        raise ValueError(f"unknown cause {cause!r}")
    open_turn = None
    open_step = None
    pending = {}
    for event in events:
        kind, data = event["type"], event["data"]
        if kind in ("turn/start", "turn/end", "step/end"):
            pending.clear()
        if kind == "turn/start":
            open_turn, open_step = data["turn"], None
        elif kind == "turn/end":
            open_turn, open_step = None, None
        elif kind == "step/start":
            open_step = data["step"]
        elif kind == "step/end":
            open_step = None
        elif kind == "assistant/message":
            for call_id in data["tool_call_ids"]:
                pending[call_id] = {"turn": data["turn"], "step": data["step"], "call_seq": None}
        elif kind == "tool/call":
            entry = pending.get(data["call_id"])
            if entry is not None:
                entry["call_seq"] = event["seq"]
        elif kind == "tool/result":
            entry = pending.get(data["call_id"])
            if (entry is not None and data["surface_op"] == "append"
                    and entry["turn"] == data["turn"] and entry["step"] == data["step"]):
                del pending[data["call_id"]]

    if not events or open_turn is None:
        return []
    last = events[-1]
    seq, time = last["seq"], last["time"]
    closers = []
    for call_id, entry in pending.items():
        seq += 1
        started = entry["call_seq"] is not None
        result = {
            "turn": entry["turn"], "step": entry["step"], "call_id": call_id, "surface_op": "append",
            "is_error": True, "started": started,
            "error_code": "TOOL_OUTCOME_UNKNOWN" if started else "TOOL_NOT_STARTED",
        }
        if started:
            result["source_event_seqs"] = [entry["call_seq"]]
        closers.append({"type": "tool/result", "seq": seq, "time": time, "data": result})
    if open_step is not None:
        seq += 1
        closers.append({"type": "step/end", "seq": seq, "time": time, "data": {"turn": open_turn, "step": open_step}})
    seq += 1
    closers.append({"type": "turn/end", "seq": seq, "time": time, "data": {"turn": open_turn, "reason": {"kind": cause}}})
    return closers
''',
    "interview_questions": interview(
        concept=[
            "Why does an agent session log need crash recovery at all? What breaks if you resume a log that ends in the middle of a tool call?",
            "Why does the harness distinguish a tool call that never started from one that started but has no result?",
        ],
        deep_dive=[
            "Walk through which events forget pending calls and which answer them. Why must an answering result be an append from the same turn and step?",
            "Why append closers instead of truncating the log back to the last complete step?",
            "How should the model be told to handle a started call whose outcome is unknown, for example a payment or a file deletion?",
        ],
        tradeoffs=[
            "Append-only event log with derived context versus a mutable message list: replay, forking, KV cache reuse and storage cost?",
            "Should recovery happen eagerly on load or lazily on the next request? What are the risks of each?",
        ],
    ),
}
