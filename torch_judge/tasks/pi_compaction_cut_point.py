"""Choose where to cut a session for compaction without orphaning tool results."""

from ._interview import interview

TASK = {
    "title": "Compaction Cut Point",
    "difficulty": "Medium",
    "version": 1,
    "function_name": "find_cut_point",
    "description_en": r"""When the Pi coding agent compacts a long session, it summarizes older entries and keeps the most recent ones verbatim. Choose the first entry to keep.

**Signature:** `find_cut_point(entries, keep_recent_tokens) -> dict`

**Parameters:**
- `entries` — list of dicts with `role` and a non-negative integer `tokens`. Roles: `user`, `assistant`, `toolResult`, `bashExecution`, `custom`, `branchSummary`, `compactionSummary`, and `meta` for bookkeeping entries the model never sees (a model change, a label).
- `keep_recent_tokens` — positive integer.

**Definitions:**
- A *cut point* is an entry whose role is neither `toolResult` nor `meta`.
- A *turn start* is an entry whose role is `user`, `bashExecution`, `custom`, `branchSummary` or `compactionSummary`.

**Algorithm:**
- If there is no cut point, return `{"first_kept_index": 0, "turn_start_index": -1, "is_split_turn": False}`.
- Walk from the last entry backwards, skipping entries with 0 tokens and adding the others' tokens to a running sum. At the first entry `i` where the sum reaches `keep_recent_tokens`, the cut is the first cut point with index `>= i`, or the last cut point if there is none. If the sum never reaches the budget, the cut is the first cut point.
- Then, while the entry just before the cut is `meta`, move the cut back by one.
- If the entry at the cut is a turn start, `turn_start_index` is -1. Otherwise it is the index of the closest turn start at or before the cut, or -1 if there is none.
- `is_split_turn` is True when the cut entry is not a turn start and a turn start was found.

**Returns:** `{"first_kept_index": int, "turn_start_index": int, "is_split_turn": bool}`.

**Constraints:** Raise `ValueError` for an unknown role, a negative or non-integer token count, or `keep_recent_tokens < 1`.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why never cut at a tool result.** A tool result must follow the assistant message that requested it; keeping a result whose call was summarized away produces an invalid transcript. Cutting at the assistant keeps the call and all its results together.

**Why prefer the cut point at or after the budget crossing.** It keeps at least the requested amount of recent context without splitting a tool exchange, and when trailing tool output alone exceeds the budget, the assistant call that produced it is kept rather than falling back to the start of the session.

**Split turns.** If the cut falls inside a turn, the start of that turn is summarized separately, so the summary can tell the model what request the kept messages belong to.

**Why pull metadata along.** Bookkeeping entries right before the cut, such as a model switch, describe the kept messages, so they stay with them.""",
    "advisory_prerequisites": ["context_compaction"],
    "hints": [
        {"level": 1, "kind": "questions", "content": "Which entries can the kept range start at, and why not a tool result? Once the running sum crosses the budget, which direction do you move to find a cut point? When is a cut a split turn?"},
        {"level": 2, "kind": "analysis", "content": "Validate, then list cut-point indices. Walk i from the end down to 0 accumulating nonzero tokens; on crossing, pick next((c for c in cuts if c >= i), cuts[-1]). Default to cuts[0]. Step back over meta entries. Then scan backwards from the cut for a turn start unless the cut itself is one."},
    ],
    "model_connections": [
        "Pi's findCutPoint drives both manual and automatic compaction; Claude Code and Codex also summarize old turns while keeping recent ones verbatim.",
    ],
    "pro_con_analysis": {
        "pros": ["Cheap, deterministic and always yields a valid transcript that keeps tool calls with their results."],
        "cons": ["Token counts are estimates, so the kept range only approximates the budget, and one huge tool result can force keeping more or less than intended."],
    },
    "sources": [
        {"kind": "code", "url": "https://github.com/badlogic/pi-mono", "commit": "6f7551516b84278eb9da1c340c8e7bc66be1a6ba", "path": "packages/coding-agent/src/core/compaction/compaction.ts", "symbol": "findCutPoint, findValidCutPoints, findTurnStartIndex, isCutPointMessage and isTurnStartMessage", "license": "MIT", "adapted": "Backward token accumulation, cut-point selection at or after the crossing, metadata pull-back and split-turn detection.", "simplifications": "Flat entries with precomputed token counts instead of session entries that project to messages; no compaction-entry boundaries or start and end index window."},
    ],
    "tests": [
        {"name": "Cuts at the assistant that owns trailing tool results", "behavior": "budget.enforcement", "code": r"""
entries = [
    {"role": "user", "tokens": 50},
    {"role": "assistant", "tokens": 20},
    {"role": "toolResult", "tokens": 30},
    {"role": "user", "tokens": 10},
    {"role": "assistant", "tokens": 10},
    {"role": "toolResult", "tokens": 500},
]
assert {fn}(entries, 100) == {"first_kept_index": 4, "turn_start_index": 3, "is_split_turn": True}
assert {fn}(entries, 540) == {"first_kept_index": 3, "turn_start_index": -1, "is_split_turn": False}
assert {fn}(entries, 10000) == {"first_kept_index": 0, "turn_start_index": -1, "is_split_turn": False}
"""},
        {"name": "Matches a seeded oracle", "visibility": "unshown", "behavior": "budget.enforcement", "failure_message": "Accumulate from the end, cut at the first valid point at or after the crossing, pull back over meta entries, and detect split turns.", "code": r"""
import random
ROLES = ["user", "assistant", "toolResult", "bashExecution", "custom", "branchSummary", "compactionSummary", "meta"]
STARTS = {"user", "bashExecution", "custom", "branchSummary", "compactionSummary"}
def oracle(entries, keep):
    cuts = [i for i, e in enumerate(entries) if e["role"] not in ("toolResult", "meta")]
    if not cuts:
        return {"first_kept_index": 0, "turn_start_index": -1, "is_split_turn": False}
    cut, total = cuts[0], 0
    for i in range(len(entries) - 1, -1, -1):
        if entries[i]["tokens"] == 0:
            continue
        total += entries[i]["tokens"]
        if total >= keep:
            later = [c for c in cuts if c >= i]
            cut = later[0] if later else cuts[-1]
            break
    while cut > 0 and entries[cut - 1]["role"] == "meta":
        cut -= 1
    if entries[cut]["role"] in STARTS:
        return {"first_kept_index": cut, "turn_start_index": -1, "is_split_turn": False}
    ts = next((j for j in range(cut, -1, -1) if entries[j]["role"] in STARTS), -1)
    return {"first_kept_index": cut, "turn_start_index": ts, "is_split_turn": ts != -1}
for seed in (3, 58, 91):
    rng = random.Random(seed)
    for trial in range(200):
        entries = []
        for _ in range(rng.randint(0, 14)):
            role = rng.choices(ROLES, weights=[3, 4, 4, 1, 1, 1, 1, 2])[0]
            tokens = 0 if role == "meta" or rng.random() < 0.1 else rng.randint(1, 60)
            entries.append({"role": role, "tokens": tokens})
        keep = rng.randint(1, 250)
        assert {fn}(entries, keep) == oracle(entries, keep), (seed, trial, entries, keep)
"""},
        {"name": "Pulls meta entries along and handles no cut points", "visibility": "unshown", "behavior": "edge.empty_or_boundary", "failure_message": "Move the cut back over adjacent meta entries; with no cut point keep everything from index 0.", "code": r"""
entries = [{"role": "user", "tokens": 10}, {"role": "assistant", "tokens": 10}, {"role": "meta", "tokens": 0},
           {"role": "meta", "tokens": 0}, {"role": "user", "tokens": 30}]
assert {fn}(entries, 25) == {"first_kept_index": 2, "turn_start_index": 0, "is_split_turn": True}
only_results = [{"role": "toolResult", "tokens": 100}, {"role": "meta", "tokens": 0}]
assert {fn}(only_results, 5) == {"first_kept_index": 0, "turn_start_index": -1, "is_split_turn": False}
no_start = [{"role": "assistant", "tokens": 5}, {"role": "toolResult", "tokens": 5}, {"role": "assistant", "tokens": 5}]
assert {fn}(no_start, 5) == {"first_kept_index": 2, "turn_start_index": -1, "is_split_turn": False}
"""},
        {"name": "Rejects invalid input", "visibility": "unshown", "behavior": "contract.signature", "failure_message": "Unknown roles, negative or non-integer tokens and a non-positive budget must raise ValueError.", "code": r"""
bad = [([{"role": "robot", "tokens": 1}], 5), ([{"role": "user", "tokens": -1}], 5),
       ([{"role": "user", "tokens": 1.5}], 5), ([{"role": "user", "tokens": 1}], 0)]
for entries, keep in bad:
    try:
        {fn}(entries, keep)
    except ValueError:
        pass
    else:
        raise AssertionError(f"accepted {entries} {keep}")
"""},
    ],
    "solution": '''ROLES = {"user", "assistant", "toolResult", "bashExecution", "custom", "branchSummary", "compactionSummary", "meta"}
TURN_STARTS = {"user", "bashExecution", "custom", "branchSummary", "compactionSummary"}


def find_cut_point(entries, keep_recent_tokens):
    if type(keep_recent_tokens) is not int or keep_recent_tokens < 1:
        raise ValueError("keep_recent_tokens must be a positive integer")
    for entry in entries:
        if entry["role"] not in ROLES:
            raise ValueError(f"unknown role {entry['role']!r}")
        if type(entry["tokens"]) is not int or entry["tokens"] < 0:
            raise ValueError("tokens must be a non-negative integer")

    cut_points = [i for i, e in enumerate(entries) if e["role"] not in ("toolResult", "meta")]
    if not cut_points:
        return {"first_kept_index": 0, "turn_start_index": -1, "is_split_turn": False}

    cut = cut_points[0]
    accumulated = 0
    for i in range(len(entries) - 1, -1, -1):
        tokens = entries[i]["tokens"]
        if tokens == 0:
            continue
        accumulated += tokens
        if accumulated >= keep_recent_tokens:
            cut = next((c for c in cut_points if c >= i), cut_points[-1])
            break

    while cut > 0 and entries[cut - 1]["role"] == "meta":
        cut -= 1

    if entries[cut]["role"] in TURN_STARTS:
        return {"first_kept_index": cut, "turn_start_index": -1, "is_split_turn": False}
    turn_start = next((j for j in range(cut, -1, -1) if entries[j]["role"] in TURN_STARTS), -1)
    return {"first_kept_index": cut, "turn_start_index": turn_start, "is_split_turn": turn_start != -1}
''',
    "interview_questions": interview(
        concept=[
            "What is context compaction in a coding agent, and why keep the most recent messages verbatim instead of summarizing everything?",
            "Why can the kept range never start at a tool result?",
        ],
        deep_dive=[
            "Walk through choosing the cut: accumulating from the end, the budget crossing, and picking the cut point at or after it. What happens when one trailing tool result exceeds the budget?",
            "What is a split turn, and what extra summary does it need so the kept messages still make sense?",
            "Why pull adjacent metadata entries into the kept range?",
        ],
        tradeoffs=[
            "Estimated tokens (chars / 4) versus the provider's reported usage for deciding when and where to compact: accuracy, cost and staleness?",
            "Cut-and-summarize compaction versus pruning tool outputs versus retrieval over the full log: what does each lose?",
        ],
    ),
}
