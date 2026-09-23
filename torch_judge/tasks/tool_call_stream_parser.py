"""Incremental extraction of tagged tool calls from a streamed model response."""

from torch_judge.tasks._schema import build_design_note_rubric

TASK = {
    "title": "Streaming Tool-Call Parser",
    "difficulty": "Hard",
    "version": 1,
    "function_name": "ToolCallStreamParser",
    "description_en": r"""Parse tool calls out of a model's response as it streams in, chunk by chunk.

The model writes plain text and, between an open tag and a close tag, a JSON tool call:

    Let me look that up.<tool_call>{"name": "search", "arguments": {"q": "rope"}}</tool_call>

Chunks split the text anywhere — including in the middle of a tag.

**Define a class** `ToolCallStreamParser(open_tag="<tool_call>", close_tag="</tool_call>")` with:

- `feed(chunk) -> list` — consume the next piece of the stream and return the events it completes, in stream order.
- `finish() -> list` — the stream has ended. Return the events still pending, in order.

**Events** are dictionaries of exactly these shapes:

- `{"type": "text", "content": str}` — text outside any tool call. Never empty.
- `{"type": "tool_call", "name": str, "arguments": dict}` — one complete, well-formed call.
- `{"type": "error", "raw": str, "reason": str}` — a call that could not be used. `raw` is everything between the tags.

**Parsing a call.** The body between the tags must be a JSON object with a non-empty string `name`. `arguments` is optional and defaults to `{}`; when present it must be a JSON object. The **first** close tag after an open tag ends the body. The `reason` of an error is one of:

- `invalid_json` — the body is not valid JSON.
- `not_a_call` — valid JSON, but not an object, or `name` is missing, empty or not a string, or `arguments` is not an object.
- `unterminated` — the stream ended inside a call.

**Release text as early as possible.** Everything that cannot be the start of an open tag must be returned by the `feed` call that delivered it. Hold back only the **longest suffix** of the pending text that is a proper prefix of the open tag. Text may be split across several text events; how you split it is up to you.

**Constraints:**
- A malformed call produces an error event and parsing continues. One bad call never hides the calls or text after it.
- At `finish`, held-back text is released as text — a partial tag at the very end is just text. A call still open is reported as an `unterminated` error.
- Raise a `ValueError` when: a tag is not a non-empty string; the two tags are equal; a chunk is not a string; or `feed` or `finish` is called after `finish`.
- The events must not depend on how the stream was chunked, apart from how text is split.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why stream at all.** The point of streaming is that the user sees text while the model is still writing and the agent can start routing a call the moment it closes, instead of waiting for the whole response. A parser that buffers the entire response and runs a regex at the end is correct, and throws both of those away.

**The held-back suffix is the whole problem.** If a chunk ends in `<tool`, you cannot know yet whether it is the start of a tag or a less-than sign in prose. Release it and a tag split across chunks leaks into the text as literal characters, and the call inside is lost. Hold back a fixed number of characters instead and the stream stutters for no reason on text that could not possibly be a tag. The longest suffix that is also a prefix of the tag is the least you can hold and still be right. A tag that repeats its own first characters, such as `<<call>>`, is where a shortcut like "hold back from the last `<`" breaks.

**Why errors are events.** In a rollout, a malformed call is information: the policy produced it, and the reward and the loss mask both need to know. A parser that logs and drops it makes the trajectory look like the model chose not to call a tool, which trains the wrong lesson.

**What this leaves out.** Production parsers also stream the arguments of a call while it is still open — emitting the function name early so a router can warm up the tool — and handle formats with no closing tag at all, where the end of the response is the end of the call.""",
    "advisory_prerequisites": ["agentic_rollout_loop"],
    "design_note_rubric": build_design_note_rubric(),
    "hints": [
        {
            "level": 1,
            "kind": "questions",
            "content": "You are always in one of two states: outside a call, or inside one. What are you looking for in each? Outside a call, suppose the pending text ends in '<to' — can you release it? What about 'hello'? What exactly is the rule that decides how many characters at the end you must keep? Try it on a tag like '<<call>>' with pending text 'x<<' — does your rule give the right answer? Inside a call, the close tag can be split across chunks too; does your code notice a close tag that arrives in two halves? Last, what do you owe the caller when the stream ends in each state?",
        },
        {
            "level": 2,
            "kind": "analysis",
            "content": "Keep one string buffer and a flag for inside-or-outside. On each feed, append the chunk and loop. Outside: find the open tag in the buffer. If found, emit the text before it (if non-empty), drop the tag, flip to inside, and loop again — one chunk can hold several calls. If not found, compute how much to hold: for sizes from min(len(buffer), len(open_tag) - 1) down to 1, the first size whose buffer suffix is a prefix of the open tag is the answer; emit everything before that suffix and keep the suffix. Checking only whether the buffer ends in the tag's first character fails on self-overlapping tags. Inside: find the close tag in the buffer; if it is absent, keep buffering and stop — the close tag may complete on the next chunk. If found, the body is everything before it; parse it with json.loads, classify failures into invalid_json or not_a_call, and emit either a tool_call or an error. Then flip back to outside and loop. On finish, the buffer is either held-back text or an unfinished body, and each has exactly one right event. A good self-test: feed the same stream whole and one character at a time, merge adjacent text events, and compare.",
        },
    ],
    "model_connections": [
        "verl's HermesToolParser decodes the whole response and extracts calls with a regex over the completed text, so tool parsing in its agent loop happens once per turn rather than while tokens arrive.",
        "The same parser logs and discards a call whose JSON fails to load. The trajectory then carries no record that the model attempted a call, which is the case this exercise surfaces as an error event instead.",
        "verl notes its Hermes parser is adapted from vLLM's, whose serving path has a streaming variant for exactly this reason: a client streaming text cannot wait for the end of the response to find out whether it contained a call.",
        "Qwen and Hermes-format models are trained to wrap calls in these tags, which is why the same format shows up in both inference servers and RL rollout code.",
    ],
    "pro_con_analysis": {
        "pros": [
            "Text reaches the user with at most a tag's length of delay, and a call is available the moment its close tag arrives.",
            "Chunking-independence makes the parser testable: the same stream fed whole and one character at a time must give the same calls.",
            "Malformed calls stay visible as events, so both the agent loop and a training pipeline can react to them rather than silently losing them.",
        ],
        "cons": [
            "The first close tag ends a call, so a JSON string argument that contains the close tag literally is cut short. Fixing that needs a JSON-aware scanner, which costs more than it usually buys.",
            "Re-searching a growing call body on every chunk is quadratic in the body's length when chunks are tiny; a long call fed one token at a time pays for it.",
            "Arguments are only available once the call closes, so a slow call with large arguments cannot be routed early.",
        ],
    },
    "sources": [
        {
            "kind": "code",
            "url": "https://github.com/volcengine/verl",
            "commit": "12ebe0cb4d300c58449fb6c675379e8700015c51",
            "path": "verl/experimental/agent_loop/tool_parser.py",
            "symbol": "HermesToolParser.extract_tool_calls",
            "license": "Apache-2.0",
            "adapted": "Hermes-format tool-call extraction: calls are JSON objects with name and arguments between <tool_call> and </tool_call>, and the text outside the tags is returned as content.",
            "simplifications": "Upstream decodes token ids with a tokenizer and runs a non-greedy regex over the finished text; this exercise takes text chunks and parses incrementally, which upstream does not do. Upstream also requires an arguments key and drops a call that fails to load, logging the error; here arguments default to an empty object and a failed call becomes an error event. Upstream leaves an unterminated call in the content as literal text; here it is an unterminated error.",
        },
    ],
    "tests": [
        {
            "name": "A whole response yields text, the call, and the trailing text",
            "code": r"""
p = {fn}()
events = p.feed('Let me check.<tool_call>{"name": "search", "arguments": {"q": "rope"}}</tool_call>Done.')
events += p.finish()
assert events == [
    {'type': 'text', 'content': 'Let me check.'},
    {'type': 'tool_call', 'name': 'search', 'arguments': {'q': 'rope'}},
    {'type': 'text', 'content': 'Done.'},
], events
""",
        },
        {
            "name": "Tags split across chunks are held back, then recognized",
            "code": r"""
p = {fn}()
first = p.feed('Hi <tool_')
assert first == [{'type': 'text', 'content': 'Hi '}], first
second = p.feed('call>{"name": "clock"}</tool')
assert second == [], second
third = p.feed('_call>')
assert third == [{'type': 'tool_call', 'name': 'clock', 'arguments': {}}], third
assert p.finish() == []
""",
        },
        {
            "name": "Text that cannot start a tag is released immediately",
            "visibility": "unshown",
            "behavior": "events.ordering",
            "failure_message": "Text was held back longer than necessary. Only the longest suffix that is a prefix of the open tag may be withheld; everything before it belongs to the feed that delivered it.",
            "code": r"""
def text_of(events):
    return ''.join(e['content'] for e in events if e['type'] == 'text')

p = {fn}()
assert text_of(p.feed('hello world')) == 'hello world'
assert text_of(p.feed('a < b')) == 'a < b'
assert text_of(p.feed(' x<')) == ' x'
assert text_of(p.feed('b')) == '<b'
assert text_of(p.feed('<tool_ca')) == ''
assert text_of(p.feed('ke')) == '<tool_cake'
""",
        },
        {
            "name": "Self-overlapping tags hold back the longest matching suffix",
            "visibility": "unshown",
            "behavior": "protocol.validation",
            "failure_message": "A tag that repeats its own opening characters was missed or leaked into the text. Hold back the longest suffix that is a prefix of the tag, not just a trailing first character.",
            "code": r"""
p = {fn}(open_tag='<<call>>', close_tag='<</call>>')
events = p.feed('note<')
events += p.feed('<')
events += p.feed('call>>{"name": "ping"}<</call>>')
events += p.finish()
merged = []
for e in events:
    if e['type'] == 'text' and merged and merged[-1]['type'] == 'text':
        merged[-1] = {'type': 'text', 'content': merged[-1]['content'] + e['content']}
    else:
        merged.append(dict(e))
assert merged == [
    {'type': 'text', 'content': 'note'},
    {'type': 'tool_call', 'name': 'ping', 'arguments': {}},
], merged
""",
        },
        {
            "name": "Any chunking gives the same events",
            "visibility": "unshown",
            "behavior": "state.invariant",
            "failure_message": "The events changed with how the stream was chunked. Feeding the stream whole, one character at a time, or split at random must give the same calls, errors and text.",
            "code": r"""
import random

def merged(events):
    out = []
    for e in events:
        assert not (e['type'] == 'text' and e['content'] == ''), 'empty text event'
        if e['type'] == 'text' and out and out[-1]['type'] == 'text':
            out[-1] = {'type': 'text', 'content': out[-1]['content'] + e['content']}
        else:
            out.append(dict(e))
    return out

stream = (
    'Plan: a<b and <tool_call>{"name": "add", "arguments": {"x": 1, "y": 2}}</tool_call>'
    ' then <tool_call>{broken</tool_call> and <tool_call>{"name": "mul", "arguments": {"s": "</tool"}}</tool_call>'
    '<tool_call>[1, 2]</tool_call> end <tool_'
)
expected = [
    {'type': 'text', 'content': 'Plan: a<b and '},
    {'type': 'tool_call', 'name': 'add', 'arguments': {'x': 1, 'y': 2}},
    {'type': 'text', 'content': ' then '},
    {'type': 'error', 'raw': '{broken', 'reason': 'invalid_json'},
    {'type': 'text', 'content': ' and '},
    {'type': 'tool_call', 'name': 'mul', 'arguments': {'s': '</tool'}},
    {'type': 'error', 'raw': '[1, 2]', 'reason': 'not_a_call'},
    {'type': 'text', 'content': ' end <tool_'},
]

def run(chunks):
    p = {fn}()
    events = []
    for c in chunks:
        events += p.feed(c)
    return merged(events + p.finish())

assert run([stream]) == expected, run([stream])
assert run(list(stream)) == expected, 'one character at a time differs'
for seed in range(25):
    rng = random.Random(seed)
    cuts = sorted(rng.sample(range(1, len(stream)), rng.randint(1, 40)))
    pieces = [stream[i:j] for i, j in zip([0] + cuts, cuts + [len(stream)])]
    assert run(pieces) == expected, f'seed {seed} differs'
""",
        },
        {
            "name": "Malformed calls become errors and parsing continues",
            "visibility": "unshown",
            "behavior": "protocol.validation",
            "failure_message": "A malformed call was dropped, raised, or misclassified, or it hid the call after it. Report invalid_json or not_a_call as an error event and keep parsing.",
            "code": r"""
bodies = [
    ('{"name": "ok", "arguments": {"a": 1}}', {'type': 'tool_call', 'name': 'ok', 'arguments': {'a': 1}}),
    ('not json', {'type': 'error', 'raw': 'not json', 'reason': 'invalid_json'}),
    ('', {'type': 'error', 'raw': '', 'reason': 'invalid_json'}),
    ('"just a string"', {'type': 'error', 'raw': '"just a string"', 'reason': 'not_a_call'}),
    ('{"arguments": {}}', {'type': 'error', 'raw': '{"arguments": {}}', 'reason': 'not_a_call'}),
    ('{"name": ""}', {'type': 'error', 'raw': '{"name": ""}', 'reason': 'not_a_call'}),
    ('{"name": 7}', {'type': 'error', 'raw': '{"name": 7}', 'reason': 'not_a_call'}),
    ('{"name": "f", "arguments": [1]}', {'type': 'error', 'raw': '{"name": "f", "arguments": [1]}', 'reason': 'not_a_call'}),
    ('{"name": "f", "arguments": "{}"}', {'type': 'error', 'raw': '{"name": "f", "arguments": "{}"}', 'reason': 'not_a_call'}),
]
p = {fn}()
events = []
for body, _ in bodies:
    events += p.feed('<tool_call>' + body + '</tool_call>')
events += p.finish()
assert events == [want for _, want in bodies], events
""",
        },
        {
            "name": "Finish flushes held text, reports an open call, and closes the parser",
            "visibility": "unshown",
            "behavior": "edge.empty_or_boundary",
            "failure_message": "The end of the stream was handled wrong. Held-back text is released as text, an open call is an unterminated error with its body as raw, and the parser refuses input after finish.",
            "code": r"""
p = {fn}()
assert p.feed('costs <') == [{'type': 'text', 'content': 'costs '}]
assert p.finish() == [{'type': 'text', 'content': '<'}]

p = {fn}()
p.feed('go <tool_call>{"name": "slow", "argu')
assert p.finish() == [{'type': 'error', 'raw': '{"name": "slow", "argu', 'reason': 'unterminated'}]

p = {fn}()
assert p.feed('') == []
assert p.finish() == []
for action in (lambda: p.feed('x'), lambda: p.finish()):
    try:
        action()
    except ValueError:
        continue
    raise AssertionError('the parser must refuse feed and finish after finish')
""",
        },
        {
            "name": "Bad tags and non-string chunks are rejected",
            "visibility": "unshown",
            "behavior": "contract.signature",
            "failure_message": "An empty, non-string or identical pair of tags, or a non-string chunk, was accepted instead of raising ValueError.",
            "code": r"""
for kwargs in ({'open_tag': ''}, {'close_tag': ''}, {'open_tag': None},
               {'open_tag': '[x]', 'close_tag': '[x]'}):
    try:
        {fn}(**kwargs)
    except ValueError:
        continue
    raise AssertionError(f'{kwargs} should raise ValueError')

p = {fn}()
for bad in (None, b'<tool_call>', 3):
    try:
        p.feed(bad)
    except ValueError:
        continue
    raise AssertionError(f'feed({bad!r}) should raise ValueError')
""",
        },
    ],
    "solution": '''import json


class ToolCallStreamParser:
    def __init__(self, open_tag="<tool_call>", close_tag="</tool_call>"):
        for tag in (open_tag, close_tag):
            if not isinstance(tag, str) or not tag:
                raise ValueError("tags must be non-empty strings")
        if open_tag == close_tag:
            raise ValueError("the open and close tags must differ")
        self._open = open_tag
        self._close = close_tag
        self._buffer = ""
        self._inside = False
        self._finished = False

    def feed(self, chunk):
        if self._finished:
            raise ValueError("feed called after finish")
        if not isinstance(chunk, str):
            raise ValueError("chunk must be a string")
        self._buffer += chunk
        events = []
        while True:
            if self._inside:
                end = self._buffer.find(self._close)
                if end < 0:
                    break
                body = self._buffer[:end]
                self._buffer = self._buffer[end + len(self._close):]
                self._inside = False
                events.append(self._parse(body))
                continue
            start = self._buffer.find(self._open)
            if start >= 0:
                if start:
                    events.append({"type": "text", "content": self._buffer[:start]})
                self._buffer = self._buffer[start + len(self._open):]
                self._inside = True
                continue
            held = self._held_back(self._buffer)
            release = self._buffer[: len(self._buffer) - held]
            if release:
                events.append({"type": "text", "content": release})
            self._buffer = self._buffer[len(release):]
            break
        return events

    def finish(self):
        if self._finished:
            raise ValueError("finish called twice")
        self._finished = True
        events = []
        if self._inside:
            events.append({"type": "error", "raw": self._buffer, "reason": "unterminated"})
        elif self._buffer:
            events.append({"type": "text", "content": self._buffer})
        self._buffer = ""
        return events

    def _held_back(self, text):
        for size in range(min(len(text), len(self._open) - 1), 0, -1):
            if self._open.startswith(text[-size:]):
                return size
        return 0

    @staticmethod
    def _parse(body):
        try:
            payload = json.loads(body)
        except ValueError:
            return {"type": "error", "raw": body, "reason": "invalid_json"}
        if not isinstance(payload, dict):
            return {"type": "error", "raw": body, "reason": "not_a_call"}
        name = payload.get("name")
        arguments = payload.get("arguments", {})
        if not isinstance(name, str) or not name or not isinstance(arguments, dict):
            return {"type": "error", "raw": body, "reason": "not_a_call"}
        return {"type": "tool_call", "name": name, "arguments": arguments}
''',
}
