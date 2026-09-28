"""Reconcile a changed system prompt with the session history while keeping the KV cache prefix."""

from ._interview import interview

TASK = {
    "title": "System Prompt Reconciliation",
    "difficulty": "Medium",
    "version": 1,
    "function_name": "reconcile_system_prompt",
    "description_en": r"""Before every model request, DeepSeek Harness re-renders the system prompt (skills, tools and settings may have changed) and decides how to record the change in the session history. Compute that decision.

**Signature:** `reconcile_system_prompt(system_nodes, rendered, in_history_capable, new_series) -> list`

**Parameters:**
- `system_nodes` — the `system/message` nodes currently on the surface, in order, as `(seq, text)` tuples. The first one is the *head*; the rest are *later* nodes.
- `rendered` — the newly rendered prompt string, possibly empty.
- `in_history_capable` — True when the model route accepts a system-prompt update appended later in the history.
- `new_series` — True when this request starts a new request series (for example right after compaction replaced part of the history).

**Returns** a list of operations, applied in order: `("append", text)` adds a new system node at the end of the history; `("replace", seq, text)` rewrites one existing node. The *effective prompt* is the text of the last node with non-empty text, or none.
- **No system node.** Return `[("append", rendered)]`, even when `rendered` is empty.
- **Empty rendering.** Replace every later node with non-empty text by `""`, in order, then replace the head by `""` if its text is non-empty. The route and series do not matter.
- **Continuing capable route** (`in_history_capable` and not `new_series`) with a non-empty rendering: return `[]` if the effective prompt equals `rendered`, otherwise `[("append", rendered)]`.
- **Otherwise** (incapable route or new series) with a non-empty rendering, *consolidate at the head*: replace every later node with non-empty text by `""`, in order, then replace the head by `rendered` if its text differs. Do this even when the effective prompt already equals `rendered`.

**Constraints:** Raise `ValueError` if `system_nodes` has duplicate seqs.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why this matters for cost.** Providers cache the key-value states of a request prefix. Rewriting the head system prompt changes the very first tokens, so the whole cached prefix is lost. Appending the new prompt after the cached history keeps everything before it reusable.

**Why not always append.** Some routes only honor a system prompt at the start, and a new request series (after compaction rewrote history) has no cache to protect, so the prompt is folded back into the head and older versions are blanked.

**Why blank instead of delete.** The log is append-only; a per-node replacement with empty text removes old instructions from what the model sees while keeping the history reconstructable. No older instructions may remain model-visible.""",
    "advisory_prerequisites": ["dsh_fold_surface"],
    "hints": [
        {"level": 1, "kind": "questions", "content": "What is the effective prompt when several system nodes exist? When is appending safe for the cache, and when must the prompt be folded back into the head? Why blank later nodes before touching the head?"},
        {"level": 2, "kind": "analysis", "content": "Handle the no-node case first. Compute the list of later nodes with non-empty text. For an empty rendering or a consolidation, emit a blanking replace for each of them, then a head replace if the head text differs from the target ('' or rendered). For a continuing capable route compare against the effective prompt."},
    ],
    "model_connections": [
        "DeepSeek Harness's agent loop applies this rule on every request; Anthropic prompt caching and DeepSeek's context caching both reward stable prefixes.",
    ],
    "pro_con_analysis": {
        "pros": ["Prompt changes cost only the new tokens on capable routes, and the model never sees two conflicting versions of its instructions."],
        "cons": ["A prompt that drifts through many appended updates is harder to read than one consolidated prompt, and the rule depends on accurately knowing each route's capabilities."],
    },
    "sources": [
        {"kind": "code", "url": "https://github.com/deepseek-ai/deepseek-harness", "commit": "21638c56315ae6a2b552d6091945d3144c9af32e", "path": "packages/core/agent-loop/README.md", "symbol": "Understand the implementation: prompt admission and prefix reuse", "license": "MIT", "adapted": "No-node append, empty-rendering clearing, append on a continuing in-history route, and head consolidation with per-node blanking otherwise.", "simplifications": "Series detection, tool-schema updates and developer messages are left to the caller; the decision returns operations instead of appending session events."},
    ],
    "tests": [
        {"name": "Capable route appends, new series consolidates", "behavior": "state.invariant", "code": r"""
nodes = [(0, "v1")]
assert {fn}(nodes, "v1", True, False) == []
assert {fn}(nodes, "v2", True, False) == [("append", "v2")]
nodes = [(0, "v1"), (9, "v2")]
assert {fn}(nodes, "v2", True, False) == []
assert {fn}(nodes, "v3", False, False) == [("replace", 9, ""), ("replace", 0, "v3")]
assert {fn}(nodes, "v2", True, True) == [("replace", 9, ""), ("replace", 0, "v2")]
"""},
        {"name": "Applying the operations always leaves exactly the rendered prompt visible", "visibility": "unshown", "behavior": "state.invariant", "failure_message": "After the operations the effective prompt must be the rendered text (or none when empty), with no older non-empty instruction left anywhere.", "code": r"""
import random
for seed in (4, 29, 66):
    rng = random.Random(seed)
    for trial in range(300):
        texts = [rng.choice(["", "a", "b", "c"]) for _ in range(rng.randint(0, 5))]
        nodes = [(i * 10, t) for i, t in enumerate(texts)]
        rendered = rng.choice(["", "a", "b", "c", "d"])
        capable, new = rng.random() < 0.5, rng.random() < 0.3
        ops = {fn}(nodes, rendered, capable, new)
        state = dict(nodes); order = [s for s, _ in nodes]
        for op in ops:
            if op[0] == "append":
                seq = 1000 + len(order); state[seq] = op[1]; order.append(seq)
            else:
                assert op[1] in state, (op, nodes)
                state[op[1]] = op[2]
        visible = [state[s] for s in order if state[s]]
        if not nodes:
            assert ops == [("append", rendered)], (nodes, ops)
        elif rendered == "":
            assert visible == [], (nodes, ops, visible)
            assert all(op[0] == "replace" for op in ops)
        else:
            assert visible[-1] == rendered, (nodes, rendered, ops, visible)
            if capable and not new:
                effective = next((t for t in reversed(texts) if t), None)
                assert ops == ([] if effective == rendered else [("append", rendered)]), (nodes, rendered, ops)
            else:
                assert visible == [rendered], (nodes, rendered, ops, visible)
                assert all(op[0] == "replace" for op in ops)
                assert [op for op in ops if op[1] != 0] == [("replace", s, "") for s, t in nodes[1:] if t], ops
                head_ops = [op for op in ops if op[1] == 0]
                assert head_ops == ([("replace", 0, rendered)] if texts[0] != rendered else []), ops
                assert not head_ops or ops[-1] == head_ops[0], ops
"""},
        {"name": "Empty renderings clear everything and consolidation ignores unchanged text", "visibility": "unshown", "behavior": "edge.empty_or_boundary", "failure_message": "An empty rendering blanks every non-empty node regardless of route; consolidation blanks later nodes even when the effective text is unchanged; no nodes means append even when empty.", "code": r"""
assert {fn}([], "", True, False) == [("append", "")]
assert {fn}([], "p", False, True) == [("append", "p")]
nodes = [(0, "old"), (3, ""), (5, "mid"), (8, "new")]
assert {fn}(nodes, "", True, False) == [("replace", 5, ""), ("replace", 8, ""), ("replace", 0, "")]
assert {fn}([(0, ""), (4, "")], "", False, True) == []
assert {fn}([(0, "new"), (7, "new")], "new", False, False) == [("replace", 7, "")]
assert {fn}([(0, ""), (2, "x")], "x", True, False) == []
assert {fn}([(0, "")], "x", True, False) == [("append", "x")]
"""},
        {"name": "Rejects duplicate node seqs", "visibility": "unshown", "behavior": "contract.signature", "failure_message": "Duplicate seqs in system_nodes must raise ValueError.", "code": r"""
try:
    {fn}([(1, "a"), (1, "b")], "c", True, False)
except ValueError:
    pass
else:
    raise AssertionError("accepted duplicate seqs")
"""},
    ],
    "solution": '''def reconcile_system_prompt(system_nodes, rendered, in_history_capable, new_series):
    seqs = [seq for seq, _ in system_nodes]
    if len(set(seqs)) != len(seqs):
        raise ValueError("duplicate system node seqs")
    if not system_nodes:
        return [("append", rendered)]
    head_seq, head_text = system_nodes[0]
    later = [(seq, text) for seq, text in system_nodes[1:] if text]

    def consolidate(target):
        ops = [("replace", seq, "") for seq, _ in later]
        if head_text != target:
            ops.append(("replace", head_seq, target))
        return ops

    if rendered == "":
        return consolidate("")
    if in_history_capable and not new_series:
        effective = next((text for _, text in reversed(system_nodes) if text), None)
        return [] if effective == rendered else [("append", rendered)]
    return consolidate(rendered)
''',
    "interview_questions": interview(
        concept=[
            "What is prompt caching (KV cache prefix reuse) at the provider, and why does editing the system prompt at the start of the conversation invalidate it?",
            "Why do agent harnesses need to change the system prompt mid-session at all?",
        ],
        deep_dive=[
            "Walk through the four cases: no system node, empty rendering, continuing capable route, and consolidation. What does each cost in cache misses?",
            "Why blank older system nodes with empty replacements instead of deleting them?",
            "Why does a new request series, for example right after compaction, consolidate at the head even if the prompt did not change?",
        ],
        tradeoffs=[
            "Appending prompt updates versus consolidating them: cache cost versus clarity of the instructions the model reads?",
            "Putting volatile context (time, open files, git status) in the system prompt versus in user messages: cache behavior and model attention?",
        ],
    ),
}
