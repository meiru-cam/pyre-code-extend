"""Advisory guard that nudges an agent out of identical tool-call loops."""

from ._interview import interview

TASK = {
    "title": "Repeat Tool-Call Reminder",
    "difficulty": "Medium",
    "version": 1,
    "function_name": "RepeatToolReminder",
    "description_en": r"""Implement the DeepSeek Harness guard that notices an agent calling the same tool with the same arguments again and again, and adds a reminder without ever blocking the call.

**Signature:** `RepeatToolReminder(thresholds=(3, 5, 8), include=(), exclude=(), arguments_preview_chars=500)` with methods `post_execute(agent, tool_name, arguments, next) -> dict` and `on_user_message(agent) -> None`.

**Construction.** Raise `ValueError` when `thresholds` is empty, contains a value that is not an `int` (a `bool` is not accepted), a value below 2, or a duplicate, or when `arguments_preview_chars` is not an `int` of at least 1. Sort thresholds ascending; the smallest is the *gentle* threshold.

**Tracking.**
- `include` and `exclude` are tool-name patterns where `*` matches any run of characters and every other character is literal; a pattern must match the whole name. A tool is tracked when (`include` is empty or some include pattern matches) and no exclude pattern matches.
- The canonical form of `arguments` is `json.dumps(arguments, sort_keys=True, separators=(",", ":"), ensure_ascii=False)`.
- Each agent has its own chain: the key `(tool_name, canonical)` of its last tracked call and a run length. A tracked call with the same key adds 1; a different key restarts at 1. Untracked calls leave the chain unchanged. Calls with `agent=None` are ignored.
- `on_user_message(agent)` clears that agent's chain.

**`post_execute`.** Advance the chain first, then call `next()` exactly once to get the downstream decision, a dict with `kind` equal to `"accept"` or `"block"` and an optional `additional_contexts` list. Return a new dict equal to the downstream one; if the run length equals a threshold, its `additional_contexts` is `[reminder] + downstream contexts`. Do not mutate the downstream dict. The reminder is a dict with:
- `summary` — `f"{tool_name} × {count}"`.
- `text` — at the gentle threshold exactly the `GENTLE` text below; at a higher threshold the `DETAILED` text with the tool name, the run length and the argument preview filled in.

The argument preview is the canonical string itself if it has at most `arguments_preview_chars` characters, otherwise its first `arguments_preview_chars` characters followed by `f"… (+{remaining} more chars)"`.

    GENTLE = ("You are repeating the exact same tool call with identical arguments. "
              "Carefully analyze the previous result before calling again: if the task is "
              "not complete, try a different approach or different arguments instead of "
              "repeating the call.")
    DETAILED = ("Repeated tool call detected:\n- tool: {tool}\n- consecutive_calls: {count}\n"
                "- arguments: {preview}\nThe repeated calls are not making progress. Do not call "
                "this tool with these exact arguments again. Inspect the latest result and choose a "
                "different action, different arguments, or finish the task if enough evidence has "
                "been gathered.")

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Advisory, not veto.** An agent stuck re-running `grep X` wastes tokens, but blocking the call can break a legitimate retry. The guard only adds a message after the tool result, and a later listener can still block.

**Why count in post-execute.** Post-execute also runs for calls that a policy denied, and a model hammering a denied call is exactly the loop worth breaking.

**Why bookkeeping tools are transparent.** If `todo_write` reset the chain, the model could launder a loop by interleaving it: `grep X, todo_write, grep X`.

**Why canonicalize.** The model can emit the same arguments with keys in a different order; sorting keys makes the identity exact and deterministic.""",
    "advisory_prerequisites": ["dsh_event_bus"],
    "hints": [
        {"level": 1, "kind": "questions", "content": "What state must be kept per agent, and when does it reset? Why must an excluded tool neither increment nor reset the chain? Why count before calling next()?"},
        {"level": 2, "kind": "analysis", "content": "Compile patterns with re.escape then replace the escaped star by '.*' and use fullmatch. Keep a dict agent -> (key, count). In post_execute: compute the reminder (or None) from the updated chain, call next(), and return {**downstream, 'additional_contexts': [reminder, *downstream.get('additional_contexts', [])]} only when there is a reminder."},
    ],
    "model_connections": [
        "DeepSeek Harness ships this guard in its base bundle with thresholds 3, 5 and 8; the reminder rides the post-execute decision's additionalContexts so the tool result itself stays unchanged for audit.",
    ],
    "pro_con_analysis": {
        "pros": ["Cheap, deterministic loop detection that never breaks a legitimate repeated call and leaves the tool output untouched."],
        "cons": ["Only exact repeats are caught; near-identical calls with a changed timestamp or offset slip through, and the in-memory chain resets on resume."],
    },
    "sources": [
        {"kind": "code", "url": "https://github.com/deepseek-ai/deepseek-harness", "commit": "21638c56315ae6a2b552d6091945d3144c9af32e", "path": "packages/guard/repeat-tool-reminder/src/index.ts", "symbol": "apply, observe, canonicalize, wildcardToRegExp, previewArguments and validateThresholds", "license": "MIT", "adapted": "Per-agent repeat chain, transparent untracked tools, threshold escalation, argument preview and folding the reminder into the downstream decision.", "simplifications": "No Cordis listeners, WeakMap keying, message sources or schemastery config; agents are plain hashable keys and user prompts arrive through on_user_message."},
    ],
    "tests": [
        {"name": "Gentle then detailed reminders", "behavior": "state.invariant", "code": r"""
guard = {fn}()
accept = lambda: {"kind": "accept"}
outs = [guard.post_execute("a1", "grep", {"q": "x", "path": "."}, accept) for _ in range(5)]
assert [bool(o.get("additional_contexts")) for o in outs] == [False, False, True, False, True], outs
gentle = outs[2]["additional_contexts"][0]
assert gentle["summary"] == "grep × 3" and gentle["text"].startswith("You are repeating the exact same tool call"), gentle
detailed = outs[4]["additional_contexts"][0]["text"]
assert "- tool: grep\n- consecutive_calls: 5\n- arguments: {\"path\":\".\",\"q\":\"x\"}\n" in detailed, detailed
assert outs[0] == {"kind": "accept"}
"""},
        {"name": "Seeded call streams match an independent chain oracle", "visibility": "unshown", "behavior": "state.invariant", "failure_message": "Track (tool, canonical arguments) per agent; untracked tools are transparent and a user message resets only that agent.", "code": r"""
import json, random, re
def canon(a):
    return json.dumps(a, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
for seed in (3, 14, 27):
    rng = random.Random(seed)
    guard = {fn}(thresholds=[4, 2, 6], exclude=["todo_*"])
    state = {}
    for step in range(120):
        agent = rng.choice(["a", "b", None])
        if rng.random() < 0.05:
            if agent is not None:
                guard.on_user_message(agent); state.pop(agent, None)
            continue
        tool = rng.choice(["read", "read", "todo_write", "bash"])
        args = rng.choice([{"p": 1, "q": [1, 2]}, {"q": [1, 2], "p": 1}, {"p": 2}])
        out = guard.post_execute(agent, tool, args, lambda: {"kind": "accept", "additional_contexts": [{"summary": "other"}]})
        want = None
        if agent is not None and not tool.startswith("todo_"):
            key = (tool, canon(args))
            prev = state.get(agent)
            count = prev[1] + 1 if prev and prev[0] == key else 1
            state[agent] = (key, count)
            if count in (2, 4, 6):
                want = f"{tool} × {count}"
        contexts = out["additional_contexts"]
        if want is None:
            assert contexts == [{"summary": "other"}], (seed, step, out)
        else:
            assert len(contexts) == 2 and contexts[0]["summary"] == want and contexts[1] == {"summary": "other"}, (seed, step, out)
            gentle = contexts[0]["text"].startswith("You are repeating")
            assert gentle == (want.endswith("× 2")), (seed, step, contexts[0])
"""},
        {"name": "Include and exclude use anchored wildcards", "visibility": "unshown", "behavior": "protocol.validation", "failure_message": "Only '*' is a wildcard, other characters are literal, and a pattern must match the whole tool name.", "code": r"""
guard = {fn}(thresholds=[2], include=["mcp_*", "a.b"], exclude=["mcp_slow"])
acc = lambda: {"kind": "accept"}
def fires(tool):
    guard.on_user_message("x")
    guard.post_execute("x", tool, {}, acc)
    return bool(guard.post_execute("x", tool, {}, acc).get("additional_contexts"))
assert fires("mcp_search") and fires("a.b")
assert not fires("mcp_slow") and not fires("axb") and not fires("xmcp_search") and not fires("bash")
"""},
        {"name": "Blocked decisions keep their feedback and gain the reminder", "visibility": "unshown", "behavior": "effects.idempotency", "failure_message": "Count before delegating, call next() exactly once, and fold the reminder into a new copy of whatever decision came back.", "code": r"""
guard = {fn}(thresholds=[2])
calls = []
downstream = {"kind": "block", "feedback": "denied", "additional_contexts": [{"summary": "policy"}]}
def nxt():
    calls.append(1); return downstream
guard.post_execute("a", "rm", {"path": "/"}, nxt)
out = guard.post_execute("a", "rm", {"path": "/"}, nxt)
assert calls == [1, 1], calls
assert out["kind"] == "block" and out["feedback"] == "denied", out
assert [c["summary"] for c in out["additional_contexts"]] == ["rm × 2", "policy"], out
assert downstream["additional_contexts"] == [{"summary": "policy"}], "downstream decision was mutated"
"""},
        {"name": "Long arguments are previewed but compared in full", "visibility": "unshown", "behavior": "budget.enforcement", "failure_message": "Truncate only the quoted preview; the chain key must use the full canonical arguments.", "code": r"""
guard = {fn}(thresholds=[2, 3], arguments_preview_chars=10)
acc = lambda: {"kind": "accept"}
body = "x" * 40
for _ in range(2):
    guard.post_execute("a", "write", {"body": body}, acc)
out = guard.post_execute("a", "write", {"body": body}, acc)
text = out["additional_contexts"][0]["text"]
canonical = '{"body":"' + body + '"}'
assert "- arguments: " + canonical[:10] + "… (+" + str(len(canonical) - 10) + " more chars)\n" in text, text
guard2 = {fn}(thresholds=[2], arguments_preview_chars=5)
guard2.post_execute("a", "write", {"body": body + "1"}, acc)
out = guard2.post_execute("a", "write", {"body": body + "2"}, acc)
assert not out.get("additional_contexts"), "different arguments that share a prefix were treated as repeats"
"""},
        {"name": "Rejects invalid configuration", "visibility": "unshown", "behavior": "contract.signature", "failure_message": "Empty, non-integer, too small or duplicate thresholds and a bad preview size must raise ValueError at construction.", "code": r"""
for kwargs in ({"thresholds": []}, {"thresholds": [1, 3]}, {"thresholds": [3, 3]}, {"thresholds": [2.5]},
               {"thresholds": [True, 3]}, {"arguments_preview_chars": 0}):
    try:
        {fn}(**kwargs)
    except ValueError:
        pass
    else:
        raise AssertionError(f"accepted {kwargs}")
"""},
    ],
    "solution": '''import json
import re

GENTLE = ("You are repeating the exact same tool call with identical arguments. "
          "Carefully analyze the previous result before calling again: if the task is "
          "not complete, try a different approach or different arguments instead of "
          "repeating the call.")
DETAILED = ("Repeated tool call detected:\\n- tool: {tool}\\n- consecutive_calls: {count}\\n"
            "- arguments: {preview}\\nThe repeated calls are not making progress. Do not call "
            "this tool with these exact arguments again. Inspect the latest result and choose a "
            "different action, different arguments, or finish the task if enough evidence has "
            "been gathered.")


def _pattern(text):
    return re.compile(re.escape(text).replace(r"\\*", ".*"))


class RepeatToolReminder:
    def __init__(self, thresholds=(3, 5, 8), include=(), exclude=(), arguments_preview_chars=500):
        values = list(thresholds)
        if not values:
            raise ValueError("thresholds must not be empty")
        for value in values:
            if type(value) is not int or value < 2:
                raise ValueError(f"invalid threshold {value!r}")
        if len(set(values)) != len(values):
            raise ValueError("thresholds must not contain duplicates")
        if type(arguments_preview_chars) is not int or arguments_preview_chars < 1:
            raise ValueError("arguments_preview_chars must be an integer >= 1")
        self.thresholds = sorted(values)
        self.include = [_pattern(p) for p in include]
        self.exclude = [_pattern(p) for p in exclude]
        self.preview_chars = arguments_preview_chars
        self.chains = {}

    def _tracked(self, name):
        if self.include and not any(p.fullmatch(name) for p in self.include):
            return False
        return not any(p.fullmatch(name) for p in self.exclude)

    def _observe(self, agent, name, arguments):
        if agent is None or not self._tracked(name):
            return None
        canonical = json.dumps(arguments, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        key = (name, canonical)
        chain = self.chains.get(agent)
        count = chain[1] + 1 if chain is not None and chain[0] == key else 1
        self.chains[agent] = (key, count)
        if count not in self.thresholds:
            return None
        if count == self.thresholds[0]:
            text = GENTLE
        else:
            preview = canonical
            if len(canonical) > self.preview_chars:
                preview = f"{canonical[:self.preview_chars]}… (+{len(canonical) - self.preview_chars} more chars)"
            text = DETAILED.format(tool=name, count=count, preview=preview)
        return {"summary": f"{name} × {count}", "text": text}

    def post_execute(self, agent, tool_name, arguments, next):
        reminder = self._observe(agent, tool_name, arguments)
        downstream = next()
        if reminder is None:
            return downstream
        return {**downstream, "additional_contexts": [reminder, *downstream.get("additional_contexts", [])]}

    def on_user_message(self, agent):
        self.chains.pop(agent, None)
''',
    "interview_questions": interview(
        concept=[
            "Why do long-running agents get stuck calling the same tool repeatedly, and why is a reminder better than blocking the call?",
            "What counts as the same call here, and why canonicalize the arguments?",
        ],
        deep_dive=[
            "Why must excluded bookkeeping tools neither increment nor reset the chain? Give a sequence that would launder a loop otherwise.",
            "Why count in post-execute rather than pre-execute, and why count before calling next()?",
            "The preview is truncated but the key uses the full arguments. What bug would appear if the key used the preview?",
        ],
        tradeoffs=[
            "Exact-match detection versus similarity-based detection of near-duplicate calls: false positives, false negatives and cost?",
            "The chain lives only in memory and resets on resume. When is that acceptable, and what would durable tracking require?",
        ],
    ),
}
