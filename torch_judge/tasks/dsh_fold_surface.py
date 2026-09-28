"""Derive the model-visible history from an append-only session log with replace operations."""

from ._interview import interview

TASK = {
    "title": "Fold the Session Surface",
    "difficulty": "Medium",
    "version": 1,
    "function_name": "fold_surface",
    "description_en": r"""In DeepSeek Harness the session log never changes; compaction and pruning append *replacement* events instead. Replay the log to find which messages the model currently sees.

**Signature:** `fold_surface(events) -> dict`

**Events.** Each event is a dict with `type`, `seq`, `data` and, for message events, `surface_op`. The *message types* are `system/message`, `user/message`, `assistant/message` and `tool/result`; their `data` holds a `message` dict. Every other type (for example `turn/start` or `assistant/attempt`) is invisible to the model.

**Surface operations:**
- `"append"` — add this event's `seq` at the end of the surface.
- `{"op": "replace", "start_seq": a, "end_seq": b}` — the surface nodes from `a` through `b`, inclusive, are removed and this event's `seq` takes their place.

**Validation.** Raise `ValueError` when:
- the `seq` values are not exactly `0, 1, 2, ...` in order;
- a message event has no `surface_op`, or a non-message event has one, or `surface_op` is neither `"append"` nor a valid replace dict;
- a replace refers to a `seq` that is not earlier than the event itself, or that is not currently on the surface, or whose start comes after its end on the surface;
- a `tool/result` replace does not cover exactly one node that is a `tool/result`, or changes anything other than `message["content"]` (compare `data` with `message["content"]` removed on both sides);
- a replace starts at surface position 0 while that node is a `system/message`, unless the replacing event is a `system/message` that covers exactly that one node.

**Returns:** `{"nodes": [...], "messages": [...], "replacements": [...]}`: the final surface as a list of seqs, the `data["message"]` of each node in order, and one `{"seq", "start", "end", "shadowed"}` dict per replace event in log order, where `shadowed` lists the seqs it removed.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Model-visible means logged.** Everything the model sees must be reconstructable from the log. Compaction does not delete old messages; it appends a summary that replaces a range, so replay, forking and audit still have the originals.

**Why tool-result rewrites are narrow.** A pruned tool result may shorten its content but must keep its call id, turn and step; otherwise it would stop answering its tool call.

**Why protect node 0.** The system prompt sits at the head of the surface. Letting a compaction summary swallow it would silently drop the agent's instructions.""",
    "advisory_prerequisites": ["dsh_open_turn_closers", "dsh_tool_result_pruner"],
    "hints": [
        {"level": 1, "kind": "questions", "content": "What is the surface before the first event, and how does each operation change it? Which checks need the surface state as it is right before the event, and which need the original events?"},
        {"level": 2, "kind": "analysis", "content": "Keep nodes as a list of seqs and events indexed by seq. For a replace, find start and end with nodes.index, check start_idx <= end_idx, run the tool/result and system-head checks against the shadowed slice, then do nodes[start_idx:end_idx + 1] = [seq]. Build messages from the final nodes."},
    ],
    "model_connections": [
        "DeepSeek Harness's foldSurface replays every session through these rules; compaction-basic and the tool-result pruner append replace events rather than rewriting history.",
    ],
    "pro_con_analysis": {
        "pros": ["History is never lost: replay, fork and audit work from the same log, and every model request is reproducible."],
        "cons": ["The log grows without bound and deriving the current view costs a replay or an incrementally maintained index."],
    },
    "sources": [
        {"kind": "code", "url": "https://github.com/deepseek-ai/deepseek-harness", "commit": "21638c56315ae6a2b552d6091945d3144c9af32e", "path": "packages/core/session/src/surface.ts", "symbol": "foldSurface, planSurfaceEvent, replacementRange, assertToolResultRewrite and assertSystemHeadRewrite", "license": "MIT", "adapted": "Contiguous seq check, append and replace folding, range validation, content-only tool-result rewrites and system-head protection.", "simplifications": "No developer messages, plugin message projections, source-event citations or content generations; message types are fixed."},
    ],
    "tests": [
        {"name": "Compaction replaces a range with a summary", "behavior": "checkpoint.recovery", "code": r"""
def msg(seq, t, text, op="append"):
    return {"type": t, "seq": seq, "data": {"message": {"text": text}}, "surface_op": op}
events = [
    msg(0, "system/message", "sys"),
    {"type": "turn/start", "seq": 1, "data": {"turn": 1}},
    msg(2, "user/message", "u1"),
    msg(3, "assistant/message", "a1"),
    {"type": "assistant/attempt", "seq": 4, "data": {"failed": True}},
    msg(5, "user/message", "u2"),
    msg(6, "user/message", "summary of u1 and a1", {"op": "replace", "start_seq": 2, "end_seq": 3}),
]
out = {fn}(events)
assert out["nodes"] == [0, 6, 5], out
assert [m["text"] for m in out["messages"]] == ["sys", "summary of u1 and a1", "u2"]
assert out["replacements"] == [{"seq": 6, "start": 2, "end": 3, "shadowed": [2, 3]}]
"""},
        {"name": "Matches a seeded fold oracle", "visibility": "unshown", "behavior": "checkpoint.recovery", "failure_message": "Append adds the seq at the end; replace swaps the inclusive range on the current surface for the new seq.", "code": r"""
import random
for seed in (9, 48, 83):
    rng = random.Random(seed)
    for trial in range(60):
        events, surface, reps = [], [], []
        events.append({"type": "system/message", "seq": 0, "data": {"message": {"k": 0}}, "surface_op": "append"}); surface.append(0)
        for seq in range(1, rng.randint(2, 25)):
            kind = rng.random()
            if kind < 0.15:
                events.append({"type": rng.choice(["turn/start", "step/end", "assistant/attempt"]), "seq": seq, "data": {}})
                continue
            t = rng.choice(["user/message", "assistant/message"])
            if kind < 0.35 and len(surface) > 2:
                i = rng.randint(1, len(surface) - 1)
                j = rng.randint(i, len(surface) - 1)
                op = {"op": "replace", "start_seq": surface[i], "end_seq": surface[j]}
                reps.append({"seq": seq, "start": surface[i], "end": surface[j], "shadowed": surface[i:j + 1]})
                surface[i:j + 1] = [seq]
            else:
                op = "append"; surface.append(seq)
            events.append({"type": t, "seq": seq, "data": {"message": {"k": seq}}, "surface_op": op})
        out = {fn}(events)
        assert out["nodes"] == surface and out["replacements"] == reps, (seed, trial, out, surface)
        assert out["messages"] == [{"k": s} for s in surface]
"""},
        {"name": "Tool results may only change content, and the system head is protected", "visibility": "unshown", "behavior": "protocol.validation", "failure_message": "A tool/result replace must cover exactly one tool/result and change only its content; node 0 system prompts may only be replaced by one system/message.", "code": r"""
def base():
    return [
        {"type": "system/message", "seq": 0, "data": {"message": {"text": "sys"}}, "surface_op": "append"},
        {"type": "user/message", "seq": 1, "data": {"message": {"text": "u"}}, "surface_op": "append"},
        {"type": "tool/result", "seq": 2, "data": {"turn": 1, "step": 1, "message": {"call_id": "c1", "content": "x" * 50}}, "surface_op": "append"},
    ]
ok = base() + [{"type": "tool/result", "seq": 3, "data": {"turn": 1, "step": 1, "message": {"call_id": "c1", "content": "x..."}}, "surface_op": {"op": "replace", "start_seq": 2, "end_seq": 2}}]
assert {fn}(ok)["nodes"] == [0, 1, 3]
bad_cases = [
    base() + [{"type": "tool/result", "seq": 3, "data": {"turn": 1, "step": 1, "message": {"call_id": "c2", "content": "x"}}, "surface_op": {"op": "replace", "start_seq": 2, "end_seq": 2}}],
    base() + [{"type": "tool/result", "seq": 3, "data": {"turn": 1, "step": 2, "message": {"call_id": "c1", "content": "x"}}, "surface_op": {"op": "replace", "start_seq": 2, "end_seq": 2}}],
    base() + [{"type": "tool/result", "seq": 3, "data": {"turn": 1, "step": 1, "message": {"call_id": "c1", "content": "x"}}, "surface_op": {"op": "replace", "start_seq": 1, "end_seq": 2}}],
    base() + [{"type": "tool/result", "seq": 3, "data": {"turn": 1, "step": 1, "message": {"call_id": "c1", "content": "x"}}, "surface_op": {"op": "replace", "start_seq": 1, "end_seq": 1}}],
    base() + [{"type": "user/message", "seq": 3, "data": {"message": {"text": "summary"}}, "surface_op": {"op": "replace", "start_seq": 0, "end_seq": 1}}],
    base() + [{"type": "system/message", "seq": 3, "data": {"message": {"text": "new sys"}}, "surface_op": {"op": "replace", "start_seq": 0, "end_seq": 1}}],
]
for events in bad_cases:
    try:
        {fn}(events)
    except ValueError:
        pass
    else:
        raise AssertionError(f"accepted {events[-1]}")
sys_ok = base() + [{"type": "system/message", "seq": 3, "data": {"message": {"text": "new sys"}}, "surface_op": {"op": "replace", "start_seq": 0, "end_seq": 0}}]
assert {fn}(sys_ok)["nodes"] == [3, 1, 2]
no_sys_head = base()[1:]
for e in no_sys_head:
    e["seq"] -= 1
no_sys_head.append({"type": "user/message", "seq": 2, "data": {"message": {"text": "s"}}, "surface_op": {"op": "replace", "start_seq": 0, "end_seq": 0}})
assert {fn}(no_sys_head)["nodes"] == [2, 1]
"""},
        {"name": "Rejects malformed logs", "visibility": "unshown", "behavior": "contract.signature", "failure_message": "Seqs must be contiguous from 0, message events need a surface_op and others must not have one, and replaces must reference earlier seqs on the current surface in order.", "code": r"""
u = lambda seq, op="append": {"type": "user/message", "seq": seq, "data": {"message": {}}, "surface_op": op}
bad = [
    [u(1)],
    [u(0), u(2)],
    [{"type": "user/message", "seq": 0, "data": {"message": {}}}],
    [{"type": "turn/start", "seq": 0, "data": {}, "surface_op": "append"}],
    [u(0, "prepend")],
    [u(0, {"op": "replace", "start_seq": 0, "end_seq": 0})],
    [u(0), u(1), u(2, {"op": "replace", "start_seq": 1, "end_seq": 0})],
    [u(0), u(1), u(2, {"op": "replace", "start_seq": 0, "end_seq": 0}), u(3, {"op": "replace", "start_seq": 0, "end_seq": 0})],
    [u(0), u(1, {"op": "replace", "start_seq": 0})],
]
for events in bad:
    try:
        {fn}(events)
    except ValueError:
        pass
    else:
        raise AssertionError(f"accepted {events}")
assert {fn}([]) == {"nodes": [], "messages": [], "replacements": []}
"""},
    ],
    "solution": '''MESSAGE_TYPES = {"system/message", "user/message", "assistant/message", "tool/result"}


def _without_content(data):
    return {**data, "message": {**data["message"], "content": None}}


def fold_surface(events):
    nodes = []
    replacements = []
    for index, event in enumerate(events):
        seq = event["seq"]
        if seq != index:
            raise ValueError(f"seq {seq} is not contiguous; expected {index}")
        op = event.get("surface_op")
        if event["type"] not in MESSAGE_TYPES:
            if op is not None:
                raise ValueError(f"{event['type']} cannot carry a surface_op")
            continue
        if op is None:
            raise ValueError(f"{event['type']} requires a surface_op")
        if op == "append":
            nodes.append(seq)
            continue
        if not (isinstance(op, dict) and op.get("op") == "replace"
                and isinstance(op.get("start_seq"), int) and isinstance(op.get("end_seq"), int)):
            raise ValueError("invalid surface_op")
        start, end = op["start_seq"], op["end_seq"]
        if start >= seq or end >= seq:
            raise ValueError("replace must reference earlier events")
        if start not in nodes or end not in nodes:
            raise ValueError("replace range is not on the current surface")
        start_idx, end_idx = nodes.index(start), nodes.index(end)
        if start_idx > end_idx:
            raise ValueError("replace start comes after its end")
        shadowed = nodes[start_idx:end_idx + 1]
        if event["type"] == "tool/result":
            if len(shadowed) != 1 or events[shadowed[0]]["type"] != "tool/result":
                raise ValueError("a tool/result replace must rewrite exactly one tool/result")
            if _without_content(events[shadowed[0]]["data"]) != _without_content(event["data"]):
                raise ValueError("a tool/result replace may change only content")
        if start_idx == 0 and events[nodes[0]]["type"] == "system/message":
            if event["type"] != "system/message" or len(shadowed) != 1:
                raise ValueError("the system prompt at node 0 may only be replaced by one system/message")
        nodes[start_idx:end_idx + 1] = [seq]
        replacements.append({"seq": seq, "start": start, "end": end, "shadowed": shadowed})
    return {
        "nodes": list(nodes),
        "messages": [events[s]["data"]["message"] for s in nodes],
        "replacements": replacements,
    }
''',
    "interview_questions": interview(
        concept=[
            "Why would an agent keep an append-only event log and derive the model's context from it, instead of storing the message list directly?",
            "What does 'model-visible means logged' buy you for debugging and reproducibility?",
        ],
        deep_dive=[
            "Walk through folding append and replace operations. Why must a replace reference nodes that are still on the surface?",
            "Why is a tool-result rewrite allowed to change only content? What breaks if it changes the call id?",
            "Why protect the system prompt at node 0 from range replacements?",
        ],
        tradeoffs=[
            "Replaying the full log versus maintaining the surface incrementally versus snapshotting it: cost, correctness and crash safety?",
            "Replacing ranges in place versus appending a summary at the end: which keeps the provider's KV cache prefix, and which keeps the conversation coherent?",
        ],
    ),
}
