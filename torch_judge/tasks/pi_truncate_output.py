"""Line- and byte-bounded truncation of tool output, from the head or the tail."""

from ._interview import interview

TASK = {
    "title": "Tool Output Truncation",
    "difficulty": "Medium",
    "version": 1,
    "function_name": "truncate_output",
    "description_en": r"""Implement the truncation the Pi coding agent applies to every `read` and `bash` result before the model sees it. Two limits apply at once; whichever is hit first wins.

**Signature:** `truncate_output(content, mode, max_lines=2000, max_bytes=51200) -> dict`

**Definitions.** Bytes are UTF-8 bytes. The *lines* of `content` are `content.split("\n")`, except that an empty string has no lines and a single trailing `"\n"` does not start an extra empty line. The output joins kept lines with `"\n"`.

**Returns** a dict with `content`, `truncated`, `truncated_by` (`"lines"`, `"bytes"` or None), `total_lines`, `total_bytes`, `output_lines`, `output_bytes` (UTF-8 size of the output `content`), `last_line_partial` and `first_line_exceeds_limit`.
- If `total_lines <= max_lines` and `total_bytes <= max_bytes`, return `content` unchanged with `truncated` False, `truncated_by` None, output counts equal to the totals and both flags False.
- Otherwise `truncated` is True, and:

**`mode="head"`** keeps whole lines from the start.
- If the first line alone is larger than `max_bytes`, return empty content, `truncated_by="bytes"`, zero output counts and `first_line_exceeds_limit=True`.
- Otherwise take lines in order while fewer than `max_lines` are kept. A line costs its bytes plus 1 for the joining newline if it is not the first kept line. Stop with `truncated_by="bytes"` before the first line that would push the total over `max_bytes`. If the loop ended for any other reason, `truncated_by="lines"`.

**`mode="tail"`** keeps whole lines from the end.
- Take lines from the last one backwards while fewer than `max_lines` are kept, with the same cost rule (the newline is counted for every line except the first one taken). If a line does not fit, set `truncated_by="bytes"`; if no line has been kept yet, keep the longest suffix of that line whose UTF-8 encoding fits in `max_bytes` without splitting a character, and set `last_line_partial=True`. Then stop. If the loop ended for any other reason, `truncated_by="lines"`.

**Constraints:** Raise `ValueError` for an unknown `mode` or for `max_lines` or `max_bytes` below 1.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why head for files and tail for commands.** A file's beginning shows its structure; a command's end shows the error or the final result. Pi truncates `read` from the head and `bash` from the tail.

**Why whole lines.** A half line is easy for the model to misread as real content. The only exception is a single enormous last line of command output, where a partial line beats nothing.

**Why two limits.** Line limits bound what the model must scan; byte limits bound tokens when lines are very long, such as minified files. The reported counts let the tool tell the model how to page through the rest.""",
    "advisory_prerequisites": ["dsh_tool_result_pruner"],
    "hints": [
        {"level": 1, "kind": "questions", "content": "How many lines does 'a\\nb\\n' have? What does one kept line cost in bytes, and when is the newline counted? How do you cut a UTF-8 string from the end without splitting a character?"},
        {"level": 2, "kind": "analysis", "content": "Split with the trailing-newline rule and measure with len(s.encode()). Loop with a running byte count and a kept list; break on the byte limit. For the partial tail line encode it, start at len(b) - max_bytes, advance while the byte is a continuation byte (b & 0xC0 == 0x80), and decode the rest."},
    ],
    "model_connections": [
        "Pi's read and bash tools use truncateHead and truncateTail with 2000 lines and 50 KB; Claude Code and Codex cap tool output the same way and tell the model how to read more.",
    ],
    "pro_con_analysis": {
        "pros": ["Predictable, cheap bounds on context usage, with exact counts the model can use to request the next page."],
        "cons": ["Byte limits approximate tokens only roughly, and the middle of long output is invisible unless the model asks for it."],
    },
    "sources": [
        {"kind": "code", "url": "https://github.com/badlogic/pi-mono", "commit": "6f7551516b84278eb9da1c340c8e7bc66be1a6ba", "path": "packages/coding-agent/src/core/tools/truncate.ts", "symbol": "truncateHead, truncateTail, splitLinesForCounting and truncateStringToBytesFromEnd", "license": "MIT", "adapted": "Line counting with the trailing-newline rule, head and tail loops with the newline cost, first-line overflow, partial last line on a UTF-8 boundary and the truncated_by classification.", "simplifications": "One function with a mode argument; max_lines and max_bytes are not echoed back; a partial tail line always reports truncated_by bytes, whereas the original reports lines when max_lines is 1."},
    ],
    "tests": [
        {"name": "Head and tail by lines", "behavior": "budget.enforcement", "code": r"""
text = "l1\nl2\nl3\nl4\n"
head = {fn}(text, "head", max_lines=2)
assert head["content"] == "l1\nl2" and head["truncated_by"] == "lines" and head["total_lines"] == 4 and head["output_lines"] == 2, head
tail = {fn}(text, "tail", max_lines=3)
assert tail["content"] == "l2\nl3\nl4" and tail["output_bytes"] == 8 and tail["total_bytes"] == 12, tail
same = {fn}(text, "head")
assert same["content"] == text and same["truncated"] is False and same["truncated_by"] is None and same["output_lines"] == 4, same
"""},
        {"name": "Matches a seeded reference on random text", "visibility": "unshown", "behavior": "budget.enforcement", "failure_message": "Count lines with the trailing-newline rule, charge a newline for every kept line but the first, and classify truncated_by as stated.", "code": r"""
import random
def lines_of(s):
    if s == "":
        return []
    parts = s.split("\n")
    if s.endswith("\n"):
        parts.pop()
    return parts
def nb(s):
    return len(s.encode("utf-8"))
def ref(content, mode, ml, mb):
    lines = lines_of(content); tb = nb(content)
    base = {"total_lines": len(lines), "total_bytes": tb, "last_line_partial": False, "first_line_exceeds_limit": False}
    if len(lines) <= ml and tb <= mb:
        return {**base, "content": content, "truncated": False, "truncated_by": None, "output_lines": len(lines), "output_bytes": tb}
    kept, used, by, partial = [], 0, "lines", False
    if mode == "head":
        if nb(lines[0]) > mb:
            return {**base, "content": "", "truncated": True, "truncated_by": "bytes", "output_lines": 0, "output_bytes": 0, "first_line_exceeds_limit": True}
        for i, line in enumerate(lines[:ml]):
            cost = nb(line) + (1 if i else 0)
            if used + cost > mb:
                by = "bytes"; break
            kept.append(line); used += cost
    else:
        for line in reversed(lines):
            if len(kept) >= ml:
                break
            cost = nb(line) + (1 if kept else 0)
            if used + cost > mb:
                by = "bytes"
                if not kept:
                    raw = line.encode("utf-8"); start = len(raw) - mb
                    while start < len(raw) and raw[start] & 0xC0 == 0x80:
                        start += 1
                    kept.insert(0, raw[start:].decode("utf-8")); partial = True
                break
            kept.insert(0, line); used += cost
    out = "\n".join(kept)
    return {**base, "content": out, "truncated": True, "truncated_by": by, "output_lines": len(kept), "output_bytes": nb(out), "last_line_partial": partial}
for seed in (2, 31, 80):
    rng = random.Random(seed)
    for trial in range(400):
        n = rng.randint(0, 12)
        words = ["", "a", "héllo", "日本語", "x" * 30, "😀😀", "tab\there"]
        content = "\n".join(rng.choice(words) + rng.choice(words) for _ in range(n))
        if rng.random() < 0.5 and content:
            content += "\n"
        mode = rng.choice(["head", "tail"])
        ml, mb = rng.randint(1, 8), rng.randint(1, 60)
        got = {fn}(content, mode, max_lines=ml, max_bytes=mb)
        assert got == ref(content, mode, ml, mb), (seed, trial, repr(content), mode, ml, mb, got)
"""},
        {"name": "Oversized lines at either end", "visibility": "unshown", "behavior": "edge.empty_or_boundary", "failure_message": "Head returns nothing when the first line is too big; tail keeps a partial last line cut on a character boundary.", "code": r"""
big = "x" * 100 + "\nshort\n"
head = {fn}(big, "head", max_bytes=50)
assert head["content"] == "" and head["first_line_exceeds_limit"] is True and head["truncated_by"] == "bytes" and head["output_lines"] == 0, head
tail = {fn}("ok\n" + "é" * 30, "tail", max_bytes=7)
assert tail["content"] == "é" * 3 and tail["last_line_partial"] is True and tail["output_bytes"] == 6 and tail["output_lines"] == 1, tail
assert tail["truncated_by"] == "bytes"
empty = {fn}("", "tail", max_lines=1, max_bytes=1)
assert empty["content"] == "" and empty["truncated"] is False and empty["total_lines"] == 0, empty
"""},
        {"name": "Rejects invalid arguments", "visibility": "unshown", "behavior": "contract.signature", "failure_message": "Unknown modes and limits below 1 must raise ValueError.", "code": r"""
for args in (("x", "middle"), ("x", "head", 0, 10), ("x", "tail", 10, 0)):
    try:
        {fn}(*args)
    except ValueError:
        pass
    else:
        raise AssertionError(f"accepted {args}")
"""},
    ],
    "solution": '''def _lines(content):
    if content == "":
        return []
    parts = content.split("\\n")
    if content.endswith("\\n"):
        parts.pop()
    return parts


def _size(text):
    return len(text.encode("utf-8"))


def _suffix_within(line, max_bytes):
    raw = line.encode("utf-8")
    if len(raw) <= max_bytes:
        return line
    start = len(raw) - max_bytes
    while start < len(raw) and raw[start] & 0xC0 == 0x80:
        start += 1
    return raw[start:].decode("utf-8")


def truncate_output(content, mode, max_lines=2000, max_bytes=51200):
    if mode not in ("head", "tail"):
        raise ValueError(f"unknown mode {mode!r}")
    if max_lines < 1 or max_bytes < 1:
        raise ValueError("limits must be at least 1")
    lines = _lines(content)
    total_bytes = _size(content)
    result = {"total_lines": len(lines), "total_bytes": total_bytes,
              "last_line_partial": False, "first_line_exceeds_limit": False}
    if len(lines) <= max_lines and total_bytes <= max_bytes:
        return {**result, "content": content, "truncated": False, "truncated_by": None,
                "output_lines": len(lines), "output_bytes": total_bytes}

    kept, used, truncated_by = [], 0, "lines"
    if mode == "head":
        if _size(lines[0]) > max_bytes:
            return {**result, "content": "", "truncated": True, "truncated_by": "bytes",
                    "output_lines": 0, "output_bytes": 0, "first_line_exceeds_limit": True}
        for line in lines[:max_lines]:
            cost = _size(line) + (1 if kept else 0)
            if used + cost > max_bytes:
                truncated_by = "bytes"
                break
            kept.append(line)
            used += cost
    else:
        for line in reversed(lines):
            if len(kept) >= max_lines:
                break
            cost = _size(line) + (1 if kept else 0)
            if used + cost > max_bytes:
                truncated_by = "bytes"
                if not kept:
                    kept.append(_suffix_within(line, max_bytes))
                    result["last_line_partial"] = True
                break
            kept.insert(0, line)
            used += cost
    output = "\\n".join(kept)
    return {**result, "content": output, "truncated": True, "truncated_by": truncated_by,
            "output_lines": len(kept), "output_bytes": _size(output)}
''',
    "interview_questions": interview(
        concept=[
            "Why must a coding agent truncate tool output before it reaches the model, and why cap both lines and bytes?",
            "Why truncate file reads from the head but command output from the tail?",
        ],
        deep_dive=[
            "Walk through the head loop: what does each kept line cost, and how do you decide whether lines or bytes caused the truncation?",
            "How do you cut a UTF-8 string to at most N bytes from the end without producing an invalid character?",
            "What should the tool tell the model after truncating so it can read the rest?",
        ],
        tradeoffs=[
            "Byte limits versus token limits for tool output: accuracy, speed and tokenizer dependence?",
            "Truncating with a note versus spilling the full output to a file the model can grep: context cost and model behavior?",
        ],
    ),
}
