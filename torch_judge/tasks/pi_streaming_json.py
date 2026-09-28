"""Parse tool-call arguments while the model is still streaming them."""

from ._interview import interview

TASK = {
    "title": "Streaming Tool Arguments Parser",
    "difficulty": "Hard",
    "version": 1,
    "function_name": "parse_streaming_json",
    "description_en": r"""While a model streams a tool call, the arguments arrive as an incomplete JSON string. Pi parses every prefix so the UI can show the file path or command before the call finishes. Implement that parser.

**Signature:** `parse_streaming_json(text) -> value`

**Steps, in order:**
- If `text` is empty or only whitespace, return `{}`.
- Return `json.loads(text)` if it succeeds.
- Otherwise let `fixed = repair(text)`. If `fixed != text` and `json.loads(fixed)` succeeds, return it.
- Otherwise return `partial(text)`; if that fails, `partial(fixed)`; if that fails too, `{}`.

**`repair(text)`** walks the text tracking whether it is inside a string (a `"` toggles it; inside a string a backslash escape is consumed as described next, so `\"` does not end the string). Outside strings every character is copied. Inside a string:
- A raw control character (code point below 0x20) becomes its escape: `\b \f \n \r \t`, or `\u00XX` with lowercase hex for the others.
- A backslash followed by `u` and four hex digits is copied. A backslash followed by one of `" \ / b f n r t u` is copied with that character. Any other backslash, including one at the very end, becomes `\\` and the next character is processed normally.

**`partial(text)`** parses one JSON value that may stop anywhere. It *fails* on anything that is not valid JSON up to the point where the text ends, including trailing characters after a complete value. When the text ends inside a value:
- **String:** keep the characters decoded so far; drop an incomplete escape at the end, and drop a final high surrogate (U+D800 to U+DBFF) whose low half has not arrived yet.
- **Number:** remove trailing `+ - . e E`; keep the number if what remains is valid JSON, otherwise the value is *missing*.
- **`true`, `false`, `null`:** only a complete literal counts; a prefix of one is *missing*.
- **Array:** keep the complete elements, plus the last element if it is not missing.
- **Object:** keep the complete members, plus the last member if its key string is complete, its colon is present and its value is not missing.
- A missing top-level value makes the result `{}`.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why parse partial JSON.** A write tool call can stream for many seconds. Showing the target path and the first lines of content as they arrive lets the user interrupt a wrong call early, and it lets the harness start preparing, for example by reading the file to diff against.

**Why drop partial literals and keys.** A half key or `tru` has no safe meaning; a half string does, since it is a prefix of the final value.

**Why repair.** Models often emit raw newlines inside JSON strings or escapes such as `\d` from regular expressions. Strict parsers reject both, so a small repair pass rescues many calls that would otherwise fail.""",
    "advisory_prerequisites": ["tool_call_stream_parser"],
    "hints": [
        {"level": 1, "kind": "questions", "content": "What should '{\"a\": \"hel' return, and what about '{\"a\": tr'? How can the parser report both a value and whether the input ran out inside it? How do you decode escapes without writing your own unicode logic?"},
        {"level": 2, "kind": "analysis", "content": "Write a recursive descent parser where each parse function returns (value, next_index, complete) with a MISSING sentinel, and raises on invalid input. For strings collect the raw escaped text and decode it with json.loads('\"' + raw + '\"'), trimming an incomplete escape first. For numbers take the maximal run of number characters and validate it with a regex."},
    ],
    "model_connections": [
        "Pi's parseStreamingJson combines JSON.parse, a repair pass and the partial-json library; the Vercel AI SDK and OpenAI's streaming helpers use the same idea to render tool arguments live.",
    ],
    "pro_con_analysis": {
        "pros": ["Live previews of tool calls and early interruption, plus recovery from common model formatting mistakes."],
        "cons": ["Partial values can be misleading (a path prefix may name a different file), and repair can silently change what the model meant."],
    },
    "sources": [
        {"kind": "code", "url": "https://github.com/badlogic/pi-mono", "commit": "6f7551516b84278eb9da1c340c8e7bc66be1a6ba", "path": "packages/ai/src/utils/json-parse.ts", "symbol": "parseStreamingJson, parseJsonWithRepair and repairJson", "license": "MIT", "adapted": "The parse, repair, partial, partial-of-repaired fallback order and the repair rules for control characters and invalid escapes.", "simplifications": "The partial-json dependency is replaced by an explicitly specified partial parser."},
    ],
    "tests": [
        {"name": "Prefixes of a tool call", "behavior": "protocol.validation", "code": r"""
p = {fn}
assert p("") == {} and p("   ") == {}
assert p('{"path": "src/ma') == {"path": "src/ma"}
assert p('{"path": "a.py", "lines": [1, 2') == {"path": "a.py", "lines": [1, 2]}
assert p('{"path": "a.py", "overwrite": tr') == {"path": "a.py"}
assert p('{"path": "a.py", "mo') == {"path": "a.py"}
assert p('{"n": 12.') == {"n": 12}
assert p('{"cmd": "ls"}') == {"cmd": "ls"}
"""},
        {"name": "Every prefix of random JSON is consistent with the final value", "visibility": "unshown", "behavior": "protocol.validation", "failure_message": "Each prefix must yield a value consistent with the final one: complete members kept exactly, the last one only as a valid prefix, and nothing invented.", "code": r"""
import json, random
def rand_value(rng, depth):
    kind = rng.choice(["str", "int", "float", "bool", "null", "list", "dict"] if depth < 3 else ["str", "int", "bool", "null"])
    if kind == "str":
        return "".join(rng.choice('ab é"\\\n/😀\t') for _ in range(rng.randint(0, 6)))
    if kind == "int":
        return rng.randint(-500, 500)
    if kind == "float":
        return round(rng.uniform(-50, 50), 2) or 0.5
    if kind == "bool":
        return rng.random() < 0.5
    if kind == "null":
        return None
    if kind == "list":
        return [rand_value(rng, depth + 1) for _ in range(rng.randint(0, 4))]
    return {f"k{i}": rand_value(rng, depth + 1) for i in range(rng.randint(0, 4))}
def consistent(p, v):
    if isinstance(v, dict):
        if not isinstance(p, dict) or list(p) != list(v)[:len(p)]:
            return False
        keys = list(p)
        return all(p[k] == v[k] for k in keys[:-1]) and (not keys or consistent(p[keys[-1]], v[keys[-1]]))
    if isinstance(v, list):
        if not isinstance(p, list) or len(p) > len(v):
            return False
        return all(a == b for a, b in zip(p[:-1], v)) and (not p or consistent(p[-1], v[len(p) - 1]))
    if isinstance(v, str):
        return isinstance(p, str) and v.startswith(p)
    if isinstance(v, bool) or v is None:
        return p is v
    return isinstance(p, (int, float)) and not isinstance(p, bool) and abs(p) <= abs(v) + 1e-9 and (p == 0 or (p > 0) == (v > 0))
for seed in (3, 42, 95):
    rng = random.Random(seed)
    for trial in range(60):
        v = {f"top{i}": rand_value(rng, 1) for i in range(rng.randint(1, 4))}
        text = json.dumps(v, ensure_ascii=rng.random() < 0.5, separators=rng.choice([(",", ":"), (", ", ": ")]))
        for cut in range(len(text) + 1):
            got = {fn}(text[:cut])
            assert consistent(got, v), (seed, trial, text[:cut], got)
        assert {fn}(text) == v
        keys = list(v)
        for i in range(len(keys) - 1):
            start = text.index('"' + keys[i + 1] + '"')
            got = {fn}(text[:start])
            assert keys[i] in got and got[keys[i]] == v[keys[i]], (text[:start], got)
"""},
        {"name": "Repairs raw control characters and invalid escapes", "visibility": "unshown", "behavior": "protocol.validation", "failure_message": "Escape raw control characters inside strings and double backslashes before invalid escape characters, then parse.", "code": r"""
p = {fn}
assert p('{"text": "line1\nline2\ttab"}') == {"text": "line1\nline2\ttab"}
assert p('{"re": "\\d+\\.py"}') == {"re": "\\d+\\.py"}
assert p('{"re": "\\d+"}') == {"re": "\\d+"}
assert p('{"bell": "a\x07b"}') == {"bell": "a\x07b"}
assert p('{"u": "\\u00e9 \\x41"}') == {"u": "é \\x41"}
assert p('{"code": "x\ny') == {"code": "x\ny"}
assert p('{"q": "say \\"hi') == {"q": 'say "hi'}
"""},
        {"name": "Partial edge cases and invalid input", "visibility": "unshown", "behavior": "edge.empty_or_boundary", "failure_message": "Drop incomplete escapes, partial literals and dangling keys; strip trailing number characters; return {} for anything invalid.", "code": r"""
p = {fn}
cases = [
    ('{"a": "x\\u00', {"a": "x"}),
    ('{"a": "x\\', {"a": "x"}),
    ('{"a": -', {}),
    ('{"a": 1e', {"a": 1}),
    ('{"a": -2.5e+', {"a": -2.5}),
    ('{"a": [true, fal', {"a": [True]}),
    ('{"a": [nul', {"a": []}),
    ('{"a": {"b": {"c": 1', {"a": {"b": {"c": 1}}}),
    ('{"a": 1,', {"a": 1}),
    ('{"a"', {}),
    ('{"a":', {}),
    ('[1, 2, "th', [1, 2, "th"]),
    ('"hel', "hel"),
    ('tru', {}),
    ('{"a": 1} x', {}),
    ('{"a": 1,}', {}),
    ('{a: 1', {}),
    ('{"a": 01', {}),
    ('{"a": [1 2', {}),
]
for text, want in cases:
    got = p(text)
    assert got == want and type(got) is type(want), (text, got, want)
"""},
    ],
    "solution": '''import json
import re

_VALID_ESCAPES = set('"\\\\/bfnrtu')
_NUMBER = re.compile(r"-?(0|[1-9][0-9]*)(\\.[0-9]+)?([eE][+-]?[0-9]+)?")
_CONTROL = {"\\b": "\\\\b", "\\f": "\\\\f", "\\n": "\\\\n", "\\r": "\\\\r", "\\t": "\\\\t"}
_MISSING = object()


class _Invalid(Exception):
    pass


def repair(text):
    out = []
    in_string = False
    i = 0
    while i < len(text):
        ch = text[i]
        if not in_string:
            out.append(ch)
            in_string = ch == '"'
            i += 1
            continue
        if ch == '"':
            out.append(ch)
            in_string = False
        elif ch == "\\\\":
            nxt = text[i + 1] if i + 1 < len(text) else None
            if nxt == "u" and re.fullmatch(r"[0-9a-fA-F]{4}", text[i + 2:i + 6]):
                out.append(text[i:i + 6])
                i += 6
                continue
            if nxt is not None and nxt in _VALID_ESCAPES:
                out.append("\\\\" + nxt)
                i += 2
                continue
            out.append("\\\\\\\\")
        elif ord(ch) < 0x20:
            out.append(_CONTROL.get(ch, f"\\\\u{ord(ch):04x}"))
        else:
            out.append(ch)
        i += 1
    return "".join(out)


def _partial(s):
    n = len(s)

    def ws(i):
        while i < n and s[i] in " \\t\\n\\r":
            i += 1
        return i

    def string(i):
        j = i + 1
        while j < n:
            ch = s[j]
            if ch == '"':
                return json.loads(s[i:j + 1]), j + 1, True
            if ch == "\\\\":
                if j + 1 == n:
                    break
                esc = s[j + 1]
                if esc == "u":
                    digits = s[j + 2:j + 6]
                    if not re.fullmatch(r"[0-9a-fA-F]*", digits):
                        raise _Invalid()
                    if len(digits) < 4:
                        break
                    j += 6
                    continue
                if esc not in _VALID_ESCAPES:
                    raise _Invalid()
                j += 2
                continue
            if ord(ch) < 0x20:
                raise _Invalid()
            j += 1
        text = json.loads(s[i:j] + '"')
        if text and 0xD800 <= ord(text[-1]) <= 0xDBFF:
            text = text[:-1]
        return text, n, False

    def number(i):
        j = i
        while j < n and s[j] in "0123456789+-.eE":
            j += 1
        token = s[i:j]
        if j == n:
            token = token.rstrip("+-.eE")
            if _NUMBER.fullmatch(token):
                return json.loads(token), n, False
            return _MISSING, n, False
        if not _NUMBER.fullmatch(token):
            raise _Invalid()
        return json.loads(token), j, True

    def container(i, close, is_object):
        result = {} if is_object else []
        j = ws(i + 1)
        if j == n:
            return result, n, False
        if s[j] == close:
            return result, j + 1, True
        while True:
            j = ws(j)
            if j == n:
                return result, n, False
            key = None
            if is_object:
                if s[j] != '"':
                    raise _Invalid()
                key, j, done = string(j)
                if not done:
                    return result, n, False
                j = ws(j)
                if j == n:
                    return result, n, False
                if s[j] != ":":
                    raise _Invalid()
                j += 1
            item, j, done = value(j)
            if item is not _MISSING:
                if is_object:
                    result[key] = item
                else:
                    result.append(item)
            if not done:
                return result, n, False
            j = ws(j)
            if j == n:
                return result, n, False
            if s[j] == close:
                return result, j + 1, True
            if s[j] != ",":
                raise _Invalid()
            j += 1

    def value(i):
        i = ws(i)
        if i == n:
            return _MISSING, n, False
        ch = s[i]
        if ch == "{":
            return container(i, "}", True)
        if ch == "[":
            return container(i, "]", False)
        if ch == '"':
            return string(i)
        if ch == "-" or ch.isdigit():
            return number(i)
        for literal, parsed in (("true", True), ("false", False), ("null", None)):
            if s.startswith(literal, i):
                return parsed, i + len(literal), True
            if literal.startswith(s[i:]):
                return _MISSING, n, False
        raise _Invalid()

    result, end, done = value(0)
    if done and ws(end) != n:
        raise _Invalid()
    return {} if result is _MISSING else result


def parse_streaming_json(text):
    if not text or not text.strip():
        return {}
    try:
        return json.loads(text)
    except ValueError:
        pass
    fixed = repair(text)
    if fixed != text:
        try:
            return json.loads(fixed)
        except ValueError:
            pass
    for candidate in (text, fixed):
        try:
            return _partial(candidate)
        except (_Invalid, ValueError):
            continue
    return {}
''',
    "interview_questions": interview(
        concept=[
            "Why would an agent UI parse tool-call arguments before the model finishes streaming them?",
            "Which partial values are safe to show, and which must be dropped? Give an example of each.",
        ],
        deep_dive=[
            "Walk through parsing '{\"path\": \"src/ma' and '{\"force\": tr'. How does your parser report that it ran out of input inside a value?",
            "What does the repair pass fix, and why are raw newlines and escapes like \\d so common in model output?",
            "Why try the unrepaired text before the repaired text in the partial step?",
        ],
        tradeoffs=[
            "Repairing malformed tool arguments versus rejecting them and asking the model to retry: robustness versus executing something the model did not mean?",
            "Constrained decoding (grammar-guided JSON) versus post-hoc repair: latency, provider support and failure modes?",
        ],
    ),
}
