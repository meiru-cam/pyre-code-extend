"""Deterministic head-marker-tail pruning of oversized tool results."""

from ._interview import interview

TASK = {
    "title": "Tool Result Pruner",
    "difficulty": "Medium",
    "version": 1,
    "function_name": "prune_content",
    "description_en": r"""Shrink an oversized tool result to a bounded head, a fixed marker and a bounded tail, keeping every non-text block in place.

**Signature:** `prune_content(blocks, threshold_chars=8192, head_chars=4096, tail_chars=1024) -> list | None`

**Parameters:**
- `blocks` — list of content-block dicts. A block with `"type": "text"` has a `"text"` string; any other type (an image, for example) is opaque.
- `threshold_chars`, `head_chars`, `tail_chars` — character budgets.

**Marker:** `MARKER = "\n\n[... tool result middle pruned ...]\n\n"`

**Returns.** Let `total` be the combined length of all text blocks, read in order as one string. If `total <= threshold_chars`, return `None`. Otherwise return a new list where:
- A character at combined position `i` is kept when `i < head_chars` or `i >= total - tail_chars`; every other text character is removed.
- `MARKER` appears exactly once, inside the first text block that loses at least one character, right after that block's kept head part.
- Each text block keeps all its other keys; a text block left with an empty string is dropped.
- Non-text blocks are kept unchanged, in their original positions relative to the text.

**Constraints:**
- Do not modify `blocks` or the dicts inside it.
- Raise `ValueError` if `threshold_chars` is not a positive `int`, if `head_chars` or `tail_chars` is not a non-negative `int`, or if `head_chars + len(MARKER) + tail_chars > threshold_chars`.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why prune before summarizing.** A single `cat` of a log file can be larger than the rest of the conversation. Trimming the middle of oversized tool output needs no model call and often relieves enough context pressure that summarization is not needed at all.

**Why the budget check.** Requiring head plus marker plus tail to fit inside the threshold makes pruning converge in one pass: every pruned result is at most the threshold and strictly smaller than its input, so it will never be pruned again.

**Replay safety.** In DeepSeek Harness the original tool result stays in the append-only session log; the pruned copy is appended as a replacement that cites the original, so replay can always recover the full output.

**Characters.** The harness counts Unicode code points so a slice never splits an emoji surrogate pair. Python strings are already sequences of code points.""",
    "advisory_prerequisites": ["context_compaction"],
    "hints": [
        {"level": 1, "kind": "questions", "content": "Where does the removed range start and end in combined coordinates? For one block, which part of it falls in the head and which in the tail? How do you remember that the marker was already placed?"},
        {"level": 2, "kind": "analysis", "content": "Validate, measure the total, and return None when within budget. Set removed_start = head_chars and removed_end = total - tail_chars. Walk the blocks with a running offset; for a text block at [start, end) keep text[:clamp(removed_start - start)] and text[clamp(removed_end - start):], insert the marker if the block intersects the removed range and no marker was placed yet, and skip empty results."},
    ],
    "model_connections": [
        "DeepSeek Harness's compaction-tool-result-pruner runs before compaction-basic summarizes history, with defaults of 8192, 4096 and 1024 code points.",
        "Coding agents such as Claude Code and Codex truncate long command output with a similar head-and-tail view before it reaches the model.",
    ],
    "pro_con_analysis": {
        "pros": ["Deterministic, free and replay-safe; errors usually live at the end of output and headers at the start, which head plus tail keeps."],
        "cons": ["The removed middle can hold the one line that mattered, and character counts only approximate tokens."],
    },
    "sources": [
        {"kind": "code", "url": "https://github.com/deepseek-ai/deepseek-harness", "commit": "21638c56315ae6a2b552d6091945d3144c9af32e", "path": "packages/compaction/compaction-tool-result-pruner/src/index.ts", "symbol": "ToolResultPruner.pruneContent and measureContent; resolveConfig in src/config.ts", "license": "MIT", "adapted": "Combined-coordinate head and tail retention, single marker in the first intersecting block, preserved non-text blocks, and the one-pass budget validation.", "simplifications": "No session rewrite, shadow-price event, token meter or replacement citation; JavaScript code-point arrays become Python strings."},
    ],
    "tests": [
        {"name": "Prunes one long text block", "behavior": "budget.enforcement", "code": r"""
MARKER = "\n\n[... tool result middle pruned ...]\n\n"
text = "".join(chr(ord("a") + i % 26) for i in range(100))
out = {fn}([{"type": "text", "text": text}], threshold_chars=60, head_chars=10, tail_chars=5)
assert out == [{"type": "text", "text": text[:10] + MARKER + text[-5:]}], out
assert {fn}([{"type": "text", "text": "short"}], threshold_chars=60, head_chars=10, tail_chars=5) is None
"""},
        {"name": "Matches a character-level oracle on seeded block layouts", "visibility": "unshown", "behavior": "budget.enforcement", "failure_message": "Keep combined positions below head_chars or at least total - tail_chars, insert one marker in the first block that loses text, and keep non-text blocks in place.", "code": r"""
import random
MARKER = "\n\n[... tool result middle pruned ...]\n\n"
def oracle(blocks, threshold, head, tail):
    total = sum(len(b["text"]) for b in blocks if b["type"] == "text")
    if total <= threshold:
        return None
    out, pos, placed = [], 0, False
    for b in blocks:
        if b["type"] != "text":
            out.append(b); continue
        kept_head, kept_tail, lost = [], [], False
        for ch in b["text"]:
            if pos < head:
                kept_head.append(ch)
            elif pos >= total - tail:
                kept_tail.append(ch)
            else:
                lost = True
            pos += 1
        marker = ""
        if lost and not placed:
            marker, placed = MARKER, True
        text = "".join(kept_head) + marker + "".join(kept_tail)
        if text:
            out.append({**b, "text": text})
    return out
for seed in (5, 22, 61):
    rng = random.Random(seed)
    for _ in range(40):
        blocks = []
        for i in range(rng.randint(1, 6)):
            if rng.random() < 0.3:
                blocks.append({"type": "image", "id": i})
            else:
                n = rng.choice([0, 1, 3, 8, 20, 45])
                blocks.append({"type": "text", "text": "".join(rng.choice("ab😀\n ") for _ in range(n)), "meta": i})
        head, tail = rng.randint(0, 12), rng.randint(0, 12)
        threshold = head + len(MARKER) + tail + rng.randint(0, 10)
        snapshot = [dict(b) for b in blocks]
        got = {fn}(blocks, threshold_chars=threshold, head_chars=head, tail_chars=tail)
        assert blocks == snapshot, "input was modified"
        assert got == oracle(blocks, threshold, head, tail), (seed, blocks, head, tail, threshold, got)
"""},
        {"name": "Marker lands once when the removed range spans blocks", "visibility": "unshown", "behavior": "edge.empty_or_boundary", "failure_message": "Only the first text block that loses characters gets the marker; fully removed blocks are dropped.", "code": r"""
MARKER = "\n\n[... tool result middle pruned ...]\n\n"
blocks = [{"type": "text", "text": "A" * 30}, {"type": "image", "src": "x"}, {"type": "text", "text": "B" * 50}, {"type": "text", "text": "C" * 30}]
out = {fn}(blocks, threshold_chars=60, head_chars=10, tail_chars=10)
assert out == [{"type": "text", "text": "A" * 10 + MARKER}, {"type": "image", "src": "x"}, {"type": "text", "text": "C" * 10}], out
out = {fn}([{"type": "text", "text": "x" * 70}], threshold_chars=len(MARKER), head_chars=0, tail_chars=0)
assert out == [{"type": "text", "text": MARKER}], out
assert {fn}([{"type": "text", "text": "x" * 30}, {"type": "text", "text": "y" * 30}], threshold_chars=60, head_chars=10, tail_chars=10) is None
"""},
        {"name": "Rejects budgets that cannot converge in one pass", "visibility": "unshown", "behavior": "contract.signature", "failure_message": "Budgets must be integers and head_chars + len(MARKER) + tail_chars must fit within threshold_chars.", "code": r"""
MARKER = "\n\n[... tool result middle pruned ...]\n\n"
bad = [dict(threshold_chars=0, head_chars=0, tail_chars=0), dict(threshold_chars=100, head_chars=-1, tail_chars=0),
       dict(threshold_chars=100, head_chars=40, tail_chars=40), dict(threshold_chars=100.0, head_chars=1, tail_chars=1),
       dict(threshold_chars=100, head_chars=1.5, tail_chars=1)]
for kwargs in bad:
    try:
        {fn}([{"type": "text", "text": "x"}], **kwargs)
    except ValueError:
        pass
    else:
        raise AssertionError(f"accepted {kwargs}")
ok = {fn}([{"type": "text", "text": "y" * 200}], threshold_chars=len(MARKER) + 80, head_chars=40, tail_chars=40)
assert ok is not None
"""},
    ],
    "solution": '''MARKER = "\\n\\n[... tool result middle pruned ...]\\n\\n"


def prune_content(blocks, threshold_chars=8192, head_chars=4096, tail_chars=1024):
    if type(threshold_chars) is not int or threshold_chars <= 0:
        raise ValueError("threshold_chars must be a positive integer")
    for name, value in (("head_chars", head_chars), ("tail_chars", tail_chars)):
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if head_chars + len(MARKER) + tail_chars > threshold_chars:
        raise ValueError("head_chars + marker + tail_chars must fit within threshold_chars")

    total = sum(len(block["text"]) for block in blocks if block["type"] == "text")
    if total <= threshold_chars:
        return None

    removed_start = head_chars
    removed_end = total - tail_chars
    pruned = []
    consumed = 0
    marker_placed = False
    for block in blocks:
        if block["type"] != "text":
            pruned.append(block)
            continue
        text = block["text"]
        start, end = consumed, consumed + len(text)
        head_end = min(len(text), max(0, removed_start - start))
        tail_start = min(len(text), max(0, removed_end - start))
        marker = ""
        if start < removed_end and end > removed_start and not marker_placed:
            marker, marker_placed = MARKER, True
        new_text = text[:head_end] + marker + text[tail_start:]
        if new_text:
            pruned.append({**block, "text": new_text})
        consumed = end
    return pruned
''',
    "interview_questions": interview(
        concept=[
            "Why do tool results dominate the context window in coding agents, and why trim them before summarizing history?",
            "Why keep both a head and a tail of the output rather than only the beginning?",
        ],
        deep_dive=[
            "Walk through how one block's kept head and kept tail are computed from combined coordinates. Where does the marker go when the removed range spans several blocks?",
            "Why must head plus marker plus tail fit inside the threshold? What goes wrong if it does not?",
            "Why keep the original result in the log and append a replacement instead of editing it in place?",
        ],
        tradeoffs=[
            "Character budgets versus token budgets for deciding what to prune: accuracy, speed and determinism?",
            "Head-and-tail pruning versus letting the model page through output with an offset and limit, or spilling the full output to a file: when would you pick each?",
        ],
    ),
}
