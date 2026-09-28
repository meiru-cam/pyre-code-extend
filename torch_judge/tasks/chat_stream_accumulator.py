"""Merge chat completion stream deltas into a message snapshot, including parallel tool calls."""

from ._interview import interview

TASK = {
    "title": "Chat Stream Delta Accumulator",
    "difficulty": "Medium",
    "version": 1,
    "function_name": "accumulate_delta",
    "description_en": r"""A streamed chat completion arrives as many small deltas: a few characters of content, or a fragment of one tool call's arguments. The client merges each delta into a snapshot of the message. Implement the merge used by the OpenAI Python client.

**Signature:** `accumulate_delta(acc: dict, delta: dict) -> dict` — updates `acc` in place and returns it.

For each `key, value` in `delta`, in order:
- **New key.** If `key` is missing from `acc` or `acc[key]` is None: when `value` is an *indexed list* (a list containing at least one dict with an `"index"` key), set `acc[key] = []` and merge into it as below; otherwise set `acc[key] = value` and move on.
- **Replace.** Keys `"index"` and `"type"` are replaced with `value`.
- **Strings** concatenate. **Numbers** (int or float, not bool) add. **Dicts** merge recursively with these same rules.
- **Lists.** If every entry already in `acc[key]` is a str, int or float, and either `acc[key]` is non-empty or `value` is not an indexed list, extend `acc[key]` with `value`. Otherwise, for each entry of `value`: it must be a dict with an int `"index"`, else raise `ValueError`. If `index < len(acc[key])`, merge the entry into `acc[key][index]` (raise `ValueError` if that element is not a dict); otherwise `acc[key].insert(index, entry)`.
- **Anything else** (for example a None value, or a string delta for a dict) leaves `acc[key]` unchanged.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**How tool calls stream.** The first fragment of a call carries its `index`, `id`, `type` and `function.name`; later fragments carry only the `index` and more `function.arguments` text. With parallel tool calls, fragments of different calls interleave, and `index` says which call each belongs to.

**Why `index` and `type` are replaced.** They identify an element or a union variant; concatenating them would turn `"function"` into `"functionfunction"`.

**Why an empty accumulator checks the delta.** One chunk can hold two fragments of the same call. Starting from `[]` and merging by index keeps them together; copying the list as-is would leave two entries for one call.

**Why None never overwrites.** Servers send `"content": null` alongside tool-call deltas; treating that as an update would erase text already received.""",
    "advisory_prerequisites": ["sse_decoder"],
    "hints": [
        {"level": 1, "kind": "questions", "content": "Which keys identify something rather than carry content? When a list first appears, how do you know whether to copy it or to merge it by index? What should a None value in a delta do?"},
        {"level": 2, "kind": "analysis", "content": "Write a helper that tests for an indexed list. In the loop, handle the missing-or-None case first (possibly creating []), then index/type, then dispatch on the pair of types: str, number, dict (recurse), list (extend or merge by index). Fall through without changing acc."},
    ],
    "model_connections": [
        "openai-python's ChatCompletionStreamState merges every chunk with accumulate_delta, then partially parses tool arguments for strict tools.",
        "vLLM and SGLang emit these deltas from their tool parsers, so any OpenAI-compatible client depends on the same merge rules.",
    ],
    "pro_con_analysis": {
        "pros": ["One generic merge handles content, refusals, reasoning fields and parallel tool calls without per-field code."],
        "cons": ["Relies on servers following conventions it cannot check: a server that repeats a string field in every chunk gets it concatenated."],
    },
    "sources": [
        {"kind": "code", "url": "https://github.com/openai/openai-python", "commit": "a380cf256abc2aa51cbf5d49437ae039b9aa9b12", "path": "src/openai/lib/streaming/_deltas.py", "symbol": "accumulate_delta and _has_indexed_entries", "license": "Apache-2.0", "adapted": "The full merge: new-key and indexed-list handling, index and type replacement, string and number accumulation, recursive dicts, primitive-list extension and merge by index.", "simplifications": "Raises ValueError for every malformed list entry instead of TypeError and RuntimeError, and excludes bools from number addition."},
    ],
    "tests": [
        {"name": "Merges content and a streamed tool call", "behavior": "protocol.validation", "code": r"""
acc = {}
for d in [{"role": "assistant", "content": ""}, {"content": "Hel"}, {"content": "lo"},
          {"content": None, "tool_calls": [{"index": 0, "id": "c1", "type": "function", "function": {"name": "get", "arguments": ""}}]},
          {"tool_calls": [{"index": 0, "function": {"arguments": '{"q":'}}]},
          {"tool_calls": [{"index": 0, "type": "function", "function": {"arguments": ' 1}'}}]}]:
    out = {fn}(acc, d)
    assert out is acc
assert acc == {"role": "assistant", "content": "Hello", "tool_calls": [
    {"index": 0, "id": "c1", "type": "function", "function": {"name": "get", "arguments": '{"q": 1}'}}]}, acc
"""},
        {"name": "Merge rules", "visibility": "unshown", "behavior": "protocol.validation", "failure_message": "Keep values on None or mismatched types, replace index and type, add numbers, extend primitive lists, merge a first indexed list by index, and raise ValueError on malformed entries.", "code": r"""
acc = {"a": "x", "n": 1, "f": 0.5, "type": "function", "index": 2, "p": [1], "d": {"k": "v"}, "z": None}
{fn}(acc, {"a": None, "n": 2, "f": 0.25, "type": "function", "index": 3, "p": [2, "s"], "d": "oops", "z": [3]})
assert acc == {"a": "x", "n": 3, "f": 0.75, "type": "function", "index": 3, "p": [1, 2, "s"], "d": {"k": "v"}, "z": [3]}, acc
acc = {}
{fn}(acc, {"calls": [{"index": 0, "arguments": "ab"}, {"index": 0, "arguments": "cd"}, {"index": 1, "arguments": "x"}]})
assert acc == {"calls": [{"index": 0, "arguments": "abcd"}, {"index": 1, "arguments": "x"}]}, acc
acc = {"calls": []}
{fn}(acc, {"calls": [{"index": 0, "a": "1"}, {"index": 0, "a": "2"}]})
assert acc == {"calls": [{"index": 0, "a": "12"}]}, acc
acc = {"calls": [{"index": 0}]}
{fn}(acc, {"calls": [{"index": 5, "a": "late"}]})
assert acc == {"calls": [{"index": 0}, {"index": 5, "a": "late"}]}, acc
for bad in ({"calls": ["str"]}, {"calls": [{"a": 1}]}, {"calls": [{"index": "0"}]}):
    try:
        {fn}({"calls": [{"index": 0}]}, bad)
    except ValueError:
        pass
    else:
        raise AssertionError(f"accepted {bad}")
try:
    {fn}({"calls": [7, {"index": 1}]}, {"calls": [{"index": 0}]})
except ValueError:
    pass
else:
    raise AssertionError("merged into a non-dict element")
"""},
        {"name": "Seeded parallel tool-call streams rebuild the message", "visibility": "unshown", "behavior": "protocol.validation", "failure_message": "Interleaved fragments of parallel tool calls, several fragments per chunk and null content deltas must rebuild exactly the final message.", "code": r"""
import random
PIECES = ['{"', 'path', '": "', 'a.py', '"}', ' ', '中', '1', ',', '[', ']']
for seed in (6, 38, 90):
    rng = random.Random(seed)
    for trial in range(150):
        content = "".join(rng.choice(PIECES) for _ in range(rng.randint(0, 6)))
        calls = [{"index": i, "id": f"call_{i}", "type": "function",
                  "function": {"name": f"tool{i}", "arguments": "".join(rng.choice(PIECES) for _ in range(rng.randint(0, 6)))}}
                 for i in range(rng.randint(0, 3))]
        frags = []
        cut = sorted(rng.sample(range(len(content) + 1), min(len(content) + 1, rng.randint(1, 3))))
        text_parts = [content[a:b] for a, b in zip([0] + cut, cut + [len(content)])]
        streams = []
        for c in calls:
            args = c["function"]["arguments"]
            k = rng.randint(1, 4)
            pts = sorted(rng.randint(0, len(args)) for _ in range(k - 1))
            parts = [args[a:b] for a, b in zip([0] + pts, pts + [len(args)])]
            first = {"index": c["index"], "id": c["id"], "type": "function", "function": {"name": c["function"]["name"], "arguments": parts[0]}}
            rest = [{"index": c["index"], "function": {"arguments": p}} for p in parts[1:]]
            streams.append([first] + rest)
        pending = [s for s in streams]
        deltas = [{"role": "assistant", "content": text_parts[0]}] + [{"content": t} for t in text_parts[1:]]
        started = [False] * len(pending)
        while any(pending):
            live = [i for i, s in enumerate(pending) if s and (i == 0 or started[i - 1])]
            i = rng.choice(live)
            frag = [pending[i].pop(0)]
            started[i] = True
            if pending[i] and rng.random() < 0.3:
                frag.append(pending[i].pop(0))
            delta = {"tool_calls": frag}
            if rng.random() < 0.3:
                delta["content"] = None
            deltas.append(delta)
        acc = {}
        for d in deltas:
            {fn}(acc, d)
        want = {"role": "assistant", "content": content}
        if calls:
            want["tool_calls"] = calls
        assert acc == want, (seed, trial, deltas, acc, want)
"""},
    ],
    "solution": '''def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_indexed(value):
    return isinstance(value, list) and any(isinstance(e, dict) and "index" in e for e in value)


def accumulate_delta(acc, delta):
    for key, value in delta.items():
        if key not in acc or acc[key] is None:
            if not _is_indexed(value):
                acc[key] = value
                continue
            acc[key] = []

        current = acc[key]
        if key == "index" or key == "type":
            acc[key] = value
        elif isinstance(current, str) and isinstance(value, str):
            acc[key] = current + value
        elif _is_number(current) and _is_number(value):
            acc[key] = current + value
        elif isinstance(current, dict) and isinstance(value, dict):
            accumulate_delta(current, value)
        elif isinstance(current, list) and isinstance(value, list):
            primitive = all(isinstance(e, (str, int, float)) for e in current)
            if primitive and (current or not _is_indexed(value)):
                current.extend(value)
                continue
            for entry in value:
                if not isinstance(entry, dict) or type(entry.get("index")) is not int:
                    raise ValueError(f"list delta entry needs an int index: {entry!r}")
                index = entry["index"]
                if index < len(current):
                    if not isinstance(current[index], dict):
                        raise ValueError("cannot merge into a non-dict element")
                    accumulate_delta(current[index], entry)
                else:
                    current.insert(index, entry)
    return acc
''',
    "interview_questions": interview(
        concept=[
            "What does a streamed chat completion chunk contain, and how does a client rebuild the full message from deltas?",
            "How are parallel tool calls streamed, and what tells the client which call a fragment belongs to?",
        ],
        deep_dive=[
            "Why must index and type be replaced rather than concatenated? What breaks if they are not?",
            "One chunk contains two fragments for the same tool call and the accumulator has no tool_calls yet. Walk through what a naive copy does versus merging by index.",
            "Why should a null content delta never overwrite text already received?",
        ],
        tradeoffs=[
            "Generic recursive delta merging versus a typed per-field accumulator: robustness to new fields versus catching malformed server output?",
            "Parsing tool arguments incrementally while they stream versus waiting for the call to finish: latency, error handling and UI?",
        ],
    ),
}
