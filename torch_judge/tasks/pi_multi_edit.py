"""The coding-agent edit tool: exact-then-fuzzy matching of several replacements."""

from ._interview import interview

TASK = {
    "title": "Agent Edit Tool with Fuzzy Matching",
    "difficulty": "Hard",
    "version": 1,
    "function_name": "apply_edits",
    "description_en": r"""Implement the core of the Pi coding agent's `edit` tool: apply several `old_text -> new_text` replacements that the model wrote from memory, tolerating the small differences models typically introduce.

**Signature:** `apply_edits(content, edits) -> str`

**Parameters:**
- `content` — the file text. It may use `\n` or `\r\n` line endings.
- `edits` — list of dicts `{"old_text": str, "new_text": str}`.

**Line endings.** The file uses `\r\n` when its first `\n` is directly preceded by `\r`, otherwise `\n`. Convert `content` and every `old_text` and `new_text` to `\n` (replace `\r\n`, then any lone `\r`, with `\n`), do all work below in that form, and convert the result's `\n` back to `\r\n` if the file used `\r\n`.

**Fuzzy normalization** `norm(text)`, applied in this order: Unicode NFKC; strip trailing whitespace (`str.rstrip()`) from every line; map `‘ ’ ‚ ‛` to `'`; map `“ ” „ ‟` to `"`; map the dashes U+2010 to U+2015 and U+2212 to `-`; map U+00A0, U+2002 to U+200A, U+202F, U+205F and U+3000 to a space.

**Matching.**
- If any `old_text` is empty, raise `ValueError` with a message containing `must not be empty`.
- If some `old_text` does not occur exactly in the content but `norm(old_text)` occurs in `norm(content)`, fuzzy mode is on for the whole call. The *base* is `norm(content)` in fuzzy mode and the content itself otherwise.
- For every edit, find `old_text` in the base, first exactly and otherwise as `norm(old_text)`. If neither is found, raise `ValueError` containing `could not find`. If `norm(old_text)` occurs more than once in `norm(base)` (non-overlapping count), raise `ValueError` containing `must be unique`.
- All edits match against the same base, not against each other's output. If two matched ranges overlap, raise `ValueError` containing `overlap`.

**Replacement.**
- Normal mode: replace every matched range of the base with its `new_text`.
- Fuzzy mode: split the original content and the base into lines, keeping line endings. Group the edits by the base lines their ranges touch, merging groups whose line ranges overlap. Each group's lines are rewritten from the base with the edits applied; every line outside a group is copied unchanged from the original content.
- If the result equals the content, raise `ValueError` containing `no changes`.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why exact first.** A unique exact match is unambiguous; fuzzy matching is a fallback for the typographic noise models add, such as curly quotes, en dashes, non-breaking spaces and trailing spaces.

**Why uniqueness.** If `old_text` matches twice, the tool cannot know which one the model meant. Asking for more context is safer than guessing.

**Why one base for all edits.** Matching every edit against the original lets the model describe several changes from one reading of the file; applying replacements from the end backwards keeps earlier offsets valid.

**Why preserve untouched lines in fuzzy mode.** Writing back the normalized text would silently change quotes and trailing whitespace across the whole file, producing a huge diff. Only the lines an edit touches are taken from the normalized view.""",
    "advisory_prerequisites": ["tool_registry"],
    "hints": [
        {"level": 1, "kind": "questions", "content": "How do you decide once whether the whole call runs in fuzzy mode? Why does uniqueness need to be checked in normalized space even for exact matches? In fuzzy mode, how do you know which original lines an edit touched?"},
        {"level": 2, "kind": "analysis", "content": "Normalize endings, reject empty old texts, decide fuzzy mode, build the base, and collect (start, length, new_text) per edit with an index search. Sort by start, check overlaps, and apply from the end. In fuzzy mode compute line spans of the base with re.findall(r'[^\\n]*\\n|[^\\n]+'), map each range to its first and last touched line, merge overlapping line groups, and rebuild the file from original lines plus rewritten group slices."},
    ],
    "model_connections": [
        "Pi's edit tool, Claude Code's Edit tool and Aider's search/replace blocks all require a unique old string; Pi adds typographic normalization with line-preserving write-back.",
    ],
    "pro_con_analysis": {
        "pros": ["String replacement is robust to line-number drift and cheap for the model to produce; fuzzy fallback rescues most typographic mismatches."],
        "cons": ["The model must quote enough context to be unique, and normalization can still match a different region than intended when the file has near-duplicate lines."],
    },
    "sources": [
        {"kind": "code", "url": "https://github.com/badlogic/pi-mono", "commit": "6f7551516b84278eb9da1c340c8e7bc66be1a6ba", "path": "packages/coding-agent/src/core/tools/edit-diff.ts", "symbol": "applyEditsToNormalizedContent, fuzzyFindText, normalizeForFuzzyMatch, applyReplacementsPreservingUnchangedLines, detectLineEnding", "license": "MIT", "adapted": "Exact-then-fuzzy matching, whole-call fuzzy mode, uniqueness and overlap checks, reverse application, and line-preserving write-back.", "simplifications": "No file IO, BOM handling, diff rendering or per-edit error wording; errors are ValueError with required keywords."},
    ],
    "tests": [
        {"name": "Two exact edits against one reading of the file", "behavior": "protocol.validation", "code": r"""
src = "def f():\n    return 1\n\ndef g():\n    return 2\n"
out = {fn}(src, [{"old_text": "return 2", "new_text": "return 20"}, {"old_text": "return 1", "new_text": "return 10"}])
assert out == "def f():\n    return 10\n\ndef g():\n    return 20\n", out
"""},
        {"name": "Fuzzy match rewrites only the touched lines", "behavior": "protocol.validation", "code": r"""
src = "a = “x”   \nb = 1  \nc = ‘y’\n"
out = {fn}(src, [{"old_text": "c = 'y'", "new_text": "c = 'z'"}])
assert out == "a = “x”   \nb = 1  \nc = 'z'\n", repr(out)
"""},
        {"name": "Matches a seeded replacement oracle in exact mode", "visibility": "unshown", "behavior": "protocol.validation", "failure_message": "Match every edit against the original, apply all replacements, and keep CRLF files CRLF.", "code": r"""
import random
for seed in (4, 19, 88):
    rng = random.Random(seed)
    for trial in range(40):
        words = [f"w{i}_{rng.randint(0, 999)}" for i in range(rng.randint(4, 9))]
        lines = [f"line {w} = {rng.randint(0, 9)}" for w in words]
        crlf = rng.random() < 0.4
        content = ("\r\n" if crlf else "\n").join(lines) + ("\r\n" if crlf else "\n")
        picks = rng.sample(range(len(lines)), rng.randint(1, 3))
        edits = [{"old_text": lines[i], "new_text": lines[i].upper() + "!"} for i in picks]
        rng.shuffle(edits)
        want_lines = [l.upper() + "!" if i in picks else l for i, l in enumerate(lines)]
        want = ("\r\n" if crlf else "\n").join(want_lines) + ("\r\n" if crlf else "\n")
        assert {fn}(content, edits) == want, (seed, trial)
"""},
        {"name": "Errors for ambiguous, missing, overlapping, empty and no-op edits", "visibility": "unshown", "behavior": "contract.signature", "failure_message": "Reject empty old text, missing or non-unique matches, overlapping ranges and edits that change nothing, with the required wording.", "code": r"""
src = "x = 1\ny = 1\nz = 2\n"
cases = [
    ([{"old_text": "", "new_text": "a"}], "must not be empty"),
    ([{"old_text": "= 1", "new_text": "= 3"}], "must be unique"),
    ([{"old_text": "q = 9", "new_text": "q"}], "could not find"),
    ([{"old_text": "x = 1\ny", "new_text": "a"}, {"old_text": "y = 1", "new_text": "b"}], "overlap"),
    ([{"old_text": "z = 2", "new_text": "z = 2"}], "no changes"),
    ([{"old_text": "z = 2 ", "new_text": "z = 3"}, {"old_text": "x = 1", "new_text": "x = 1"}], None),
]
for edits, word in cases:
    try:
        out = {fn}(src, edits)
    except ValueError as e:
        assert word is not None and word in str(e).lower(), (edits, e)
    else:
        assert word is None and out == "x = 1\ny = 1\nz = 3\n", (edits, out)
dup = "a  \na\n"
try:
    {fn}(dup, [{"old_text": "a", "new_text": "b"}])
except ValueError as e:
    assert "must be unique" in str(e).lower()
else:
    raise AssertionError("exact single occurrence but two normalized occurrences must be ambiguous")
try:
    {fn}("x = \u20181\u2019\nx = '1'\n", [{"old_text": "x = '1'", "new_text": "x = 2"}])
except ValueError as e:
    assert "must be unique" in str(e).lower()
else:
    raise AssertionError("uniqueness must be checked in normalized space")
"""},
        {"name": "Fuzzy mode keeps untouched lines byte for byte", "visibility": "unshown", "behavior": "state.invariant", "failure_message": "In fuzzy mode rewrite only lines an edit touches; copy every other line from the original, including its trailing spaces and curly quotes.", "code": r"""
import random
for seed in (6, 50, 93):
    rng = random.Random(seed)
    for trial in range(30):
        n = rng.randint(4, 10)
        lines = []
        for i in range(n):
            body = f"v{i} = “{rng.randint(0, 99)}” – k{i}"
            lines.append(body + " " * rng.randint(0, 3))
        crlf = rng.random() < 0.3
        eol = "\r\n" if crlf else "\n"
        content = eol.join(lines) + eol
        i, j = sorted(rng.sample(range(n), 2))
        span = rng.random() < 0.5 and j == i + 1
        def plain(k):
            return f"v{k} = \"" + lines[k].split("“")[1].split("”")[0] + f"\" - k{k}"
        if span:
            edits = [{"old_text": plain(i) + "\n" + plain(j), "new_text": "MERGED"}]
            want = [l for k, l in enumerate(lines) if k not in (i, j)]
            want.insert(i, "MERGED")
        else:
            edits = [{"old_text": plain(i), "new_text": f"NEW{i}"}, {"old_text": plain(j), "new_text": f"NEW{j}"}]
            want = [f"NEW{k}" if k in (i, j) else l for k, l in enumerate(lines)]
        got = {fn}(content, edits)
        assert got == eol.join(want) + eol, (seed, trial, repr(got), repr(eol.join(want) + eol))
"""},
    ],
    "solution": '''import re
import unicodedata

_SINGLE = dict.fromkeys(map(ord, "\\u2018\\u2019\\u201a\\u201b"), "'")
_DOUBLE = dict.fromkeys(map(ord, "\\u201c\\u201d\\u201e\\u201f"), '"')
_DASH = dict.fromkeys(map(ord, "\\u2010\\u2011\\u2012\\u2013\\u2014\\u2015\\u2212"), "-")
_SPACE = dict.fromkeys([0x00A0, *range(0x2002, 0x200B), 0x202F, 0x205F, 0x3000], " ")
_TABLE = {**_SINGLE, **_DOUBLE, **_DASH, **_SPACE}


def _norm(text):
    text = unicodedata.normalize("NFKC", text)
    text = "\\n".join(line.rstrip() for line in text.split("\\n"))
    return text.translate(_TABLE)


def _to_lf(text):
    return text.replace("\\r\\n", "\\n").replace("\\r", "\\n")


def _line_spans(text):
    spans, offset = [], 0
    for line in re.findall(r"[^\\n]*\\n|[^\\n]+", text):
        spans.append((offset, offset + len(line)))
        offset += len(line)
    return spans


def _apply(text, replacements, offset=0):
    for start, length, new in sorted(replacements, reverse=True):
        start -= offset
        text = text[:start] + new + text[start + length:]
    return text


def apply_edits(content, edits):
    crlf_at, lf_at = content.find("\\r\\n"), content.find("\\n")
    crlf = lf_at != -1 and crlf_at != -1 and crlf_at <= lf_at
    original = _to_lf(content)
    edits = [(_to_lf(e["old_text"]), _to_lf(e["new_text"])) for e in edits]
    if any(not old for old, _ in edits):
        raise ValueError("old_text must not be empty")

    fuzzy = any(old not in original and _norm(old) in _norm(original) for old, _ in edits)
    base = _norm(original) if fuzzy else original

    matches = []
    for old, new in edits:
        index, length = base.find(old), len(old)
        if index == -1:
            index, length = base.find(_norm(old)), len(_norm(old))
        if index == -1:
            raise ValueError("could not find old_text in the file")
        if _norm(base).count(_norm(old)) > 1:
            raise ValueError("old_text must be unique; add more context")
        matches.append((index, length, new))
    matches.sort()
    for (s1, l1, _), (s2, _, _) in zip(matches, matches[1:]):
        if s1 + l1 > s2:
            raise ValueError("edits overlap; merge them or target disjoint regions")

    if not fuzzy:
        result = _apply(base, matches)
    else:
        original_lines = re.findall(r"[^\\n]*\\n|[^\\n]+", original)
        spans = _line_spans(base)
        groups = []
        for start, length, new in matches:
            first = next(i for i, (a, b) in enumerate(spans) if a <= start < b)
            last = first
            while spans[last][1] < start + length:
                last += 1
            if groups and first < groups[-1][1]:
                groups[-1][1] = max(groups[-1][1], last + 1)
                groups[-1][2].append((start, length, new))
            else:
                groups.append([first, last + 1, [(start, length, new)]])
        result, cursor = "", 0
        for first, end, reps in groups:
            result += "".join(original_lines[cursor:first])
            lo, hi = spans[first][0], spans[end - 1][1]
            result += _apply(base[lo:hi], reps, lo)
            cursor = end
        result += "".join(original_lines[cursor:])

    if result == original:
        raise ValueError("no changes: the replacements produced identical content")
    return result.replace("\\n", "\\r\\n") if crlf else result
''',
    "interview_questions": interview(
        concept=[
            "Why do coding agents edit files by string replacement instead of line numbers or full rewrites?",
            "Why must old_text be unique, and what should the tool tell the model when it is not?",
        ],
        deep_dive=[
            "Walk through exact-then-fuzzy matching. Which differences does the normalization forgive, and why is fuzzy mode decided once for the whole call?",
            "Why match all edits against the original content, and why apply replacements from the end of the file backwards?",
            "In fuzzy mode, how do you write back only the touched lines? What would go wrong if you wrote back the normalized file?",
        ],
        tradeoffs=[
            "String replacement versus unified diffs versus whole-file rewrites as the edit format for an LLM: token cost, failure modes and model accuracy?",
            "How far would you take fuzzy matching (indentation, whitespace inside lines, edit distance) before the risk of editing the wrong place outweighs the benefit?",
        ],
    ),
}
