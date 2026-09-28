"""SIMA 2-style agents: parse structured model output into keyboard and mouse events on a real-time clock."""

from ._interview import interview

TASK = {
    "title": "Embodied Action Chunk Scheduler",
    "difficulty": "Hard",
    "version": 1,
    "function_name": "schedule_actions",
    "description_en": r"""SIMA 2 is a Gemini-based agent that watches a game's video frames and writes structured text that is parsed deterministically into keyboard and mouse commands, in chunks, while the game keeps running. Implement the parser and the scheduler that turns chunks arriving over time into low-level events. The paper does not publish its text format; this exercise uses the format below.

**Signature:** `schedule_actions(chunks, keys) -> dict`
- `chunks` — list of `(arrival_tick, text)` with non-decreasing integer ticks; raise `ValueError` otherwise.
- `keys` — set of valid key names.

**Format.** A chunk's text is a sequence of segments `<think>...</think>`, `<say>...</say>` and `<act>...</act>`, separated only by whitespace; segment bodies contain no `<`. Inside `<act>`, actions are separated by `;`, stripped, and empty ones are skipped. Actions, with single spaces between parts:
- `press K`, `hold K`, `release K` with `K` in `keys`.
- `move DX DY` with integers (optional leading `-`), each clamped to `[-100, 100]`.
- `click left` or `click right`.
- `wait N` with an integer `N >= 1`.

Anything else makes the chunk **malformed**: record its index in `rejected` and ignore it entirely; earlier chunks keep running.

**Scheduling a valid chunk arriving at tick `t`:**
1. **Preempt.** Drop every event scheduled at a tick `>= t`. For each key that is down after the remaining events, schedule `("up", K)` at `t`, in sorted key order.
2. **Dialogue.** Record `(t, body.strip())` for each `<say>` segment. `<think>` produces nothing.
3. **Actions**, with a cursor starting at `t`: `press` schedules `("down", K)` at the cursor and `("up", K)` at cursor + 1; `hold` schedules `("down", K)`; `release` schedules `("up", K)`; `move` schedules `("move", (dx, dy))`; `click` schedules `("click", B)`; `wait N` adds `N` to the cursor.

A key is down after an event sequence if its last event was `down`. Events at the same tick run in the order they were scheduled.

**Returns** `{"events": [(tick, kind, arg), ...], "dialogue": [(tick, text), ...], "rejected": [index, ...], "held": sorted keys down after all events}`, with events sorted by tick.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why text actions.** Writing actions as parseable text lets one Gemini model interleave reasoning, dialogue and control, and reuse everything it knows about language. The parser must be deterministic and strict: a half-understood command is worse than none.

**Why chunks and preemption.** Inference takes longer than a game frame, so the agent emits a short plan of actions. When a newer chunk arrives, it was computed from a newer frame, so the rest of the old plan is stale and is dropped.

**Why release held keys.** Dropping the rest of a plan can drop the release of a key it held. Without an explicit release the character keeps walking forever: the embodied version of a leaked lock.

**Why reject malformed chunks whole.** Applying half a chunk executes a plan the model did not intend; keeping the previous plan running is the safer fallback.

**Hierarchy.** SIMA 2 can also be steered by Gemini Pro, which watches the video at a slower cadence and issues new language instructions every k steps: a slow planner over a fast controller.""",
    "advisory_prerequisites": ["chat_stream_accumulator"],
    "hints": [
        {"level": 1, "kind": "questions", "content": "What must happen to the old plan's future events when a new chunk arrives, and which keys could be left pressed? Why should a malformed chunk not preempt anything? At which ticks do a press's down and up land?"},
        {"level": 2, "kind": "analysis", "content": "Parse the whole chunk first with a regex over segments and a small action parser; return None if anything fails. Keep a list of (tick, seq, kind, arg). For a valid chunk, filter out ticks >= t, replay the rest in (tick, seq) order to find held keys, append releases at t, then walk the actions with a cursor."},
    ],
    "model_connections": [
        "Google DeepMind's SIMA 2 emits structured text parsed into 96 keyboard keys, mouse clicks and discretized mouse moves, applied in chunks.",
        "Computer-use agents (Claude computer use, OpenAI Operator) and robot policies with action chunking (ACT, pi0) face the same stale-plan and preemption problem.",
    ],
    "pro_con_analysis": {
        "pros": ["One model reasons, talks and acts through text, and the strict parser plus preemption keep the game in a safe state when plans change."],
        "cons": ["Control is only as fast as inference plus parsing, and every chunk is computed from a frame that is already old when it executes."],
    },
    "sources": [
        {"kind": "paper", "url": "https://arxiv.org/abs/2512.04797", "section": "Section 3.2, Agent-Environment Interface (structured text actions, action chunks, 96 keys and discretized mouse movement)"},
        {"kind": "paper", "url": "https://arxiv.org/abs/2512.04797", "section": "Section 4.4, Gemini Instructing SIMA 2"},
    ],
    "tests": [
        {"name": "A new chunk preempts and releases held keys", "behavior": "events.ordering", "code": r"""
keys = {"W", "A", "space"}
out = {fn}([(0, "<think>go forward</think><act>hold W; wait 5; release W</act>"),
            (2, "<say> jumping </say><act>press space; move 3 -200</act>")], keys)
assert out == {
    "events": [(0, "down", "W"), (2, "up", "W"), (2, "down", "space"), (2, "move", (3, -100)), (3, "up", "space")],
    "dialogue": [(2, "jumping")], "rejected": [], "held": []}, out
"""},
        {"name": "Malformed chunks are rejected whole", "visibility": "unshown", "behavior": "protocol.validation", "failure_message": "Any invalid segment or action rejects the whole chunk: no dialogue, no preemption, and earlier plans keep running.", "code": r"""
keys = {"W", "S"}
bad = ["<act>press Q</act>", "<act>hold W; jump</act>", "<act>wait 0</act>", "<act>move 1</act>", "hello <act>hold W</act>",
       "<act>hold W</act", "<say>hi</say><dance>x</dance>", "<act>click middle</act>", "<act>move 1.5 2</act>",
       "<act>hold  W</act>", "<act>wait -1</act>", "<act>move +1 2</act>"]
for text in bad:
    out = {fn}([(0, "<act>hold S</act>"), (1, "<say>no</say>" + text)], keys)
    assert out == {"events": [(0, "down", "S")], "dialogue": [], "rejected": [1], "held": ["S"]}, (text, out)
ok = {fn}([(0, "  <act> hold W ;; ; </act>\n<think></think>  ")], keys)
assert ok["events"] == [(0, "down", "W")] and ok["held"] == ["W"], ok
try:
    {fn}([(3, "<act></act>"), (2, "<act></act>")], keys)
except ValueError:
    pass
else:
    raise AssertionError("accepted decreasing arrival ticks")
"""},
        {"name": "Preemption keeps past events and releases only held keys", "visibility": "unshown", "behavior": "events.ordering", "failure_message": "Keep events before the arrival tick, release keys still down (sorted) at the arrival tick before new events, and drop a pending press release.", "code": r"""
keys = {"A", "B", "C"}
out = {fn}([(0, "<act>hold C; press A; wait 1; hold B; release C; wait 3; release B</act>"),
            (1, "<act>click left</act>")], keys)
assert out["events"] == [(0, "down", "C"), (0, "down", "A"), (1, "up", "A"), (1, "up", "C"), (1, "click", "left")], out
out = {fn}([(0, "<act>press A</act>"), (1, "<act>move 0 0</act>")], keys)
assert out["events"] == [(0, "down", "A"), (1, "up", "A"), (1, "move", (0, 0))], out
out = {fn}([(0, "<act>hold B; hold A</act>"), (0, "<act>wait 2; press C</act>")], keys)
assert out["events"] == [(2, "down", "C"), (3, "up", "C")], out
assert out["held"] == [], out
"""},
        {"name": "Seeded chunk streams match an oracle", "visibility": "unshown", "behavior": "events.ordering", "failure_message": "Parse strictly, preempt at each valid arrival, release held keys in sorted order, and schedule actions with the cursor rules.", "code": r"""
import random, re
KEYS = {"W", "A", "S", "D", "E"}
def oracle(chunks):
    ev, dia, rej, seq = [], [], [], 0
    for idx, (t, text) in enumerate(chunks):
        segs, pos, ok = [], 0, True
        for m in re.finditer(r"\s*<(think|say|act)>([^<]*)</\1>\s*", text):
            if m.start() != pos:
                ok = False
            segs.append((m.group(1), m.group(2))); pos = m.end()
        if pos != len(text):
            ok = False
        acts = []
        for kind, body in segs:
            if kind != "act":
                continue
            for a in body.split(";"):
                a = a.strip()
                if not a:
                    continue
                p = a.split(" ")
                if len(p) == 2 and p[0] in ("press", "hold", "release") and p[1] in KEYS:
                    acts.append((p[0], p[1]))
                elif len(p) == 3 and p[0] == "move" and all(re.fullmatch(r"-?\d+", x) for x in p[1:]):
                    acts.append(("move", tuple(max(-100, min(100, int(x))) for x in p[1:])))
                elif len(p) == 2 and p[0] == "click" and p[1] in ("left", "right"):
                    acts.append(("click", p[1]))
                elif len(p) == 2 and p[0] == "wait" and re.fullmatch(r"\d+", p[1]) and int(p[1]) >= 1:
                    acts.append(("wait", int(p[1])))
                else:
                    ok = False
        if not ok:
            rej.append(idx); continue
        ev = [e for e in ev if e[0] < t]
        down = set()
        for e in sorted(ev, key=lambda e: (e[0], e[1])):
            if e[2] == "down": down.add(e[3])
            elif e[2] == "up": down.discard(e[3])
        for k in sorted(down):
            ev.append((t, seq, "up", k)); seq += 1
        dia += [(t, b.strip()) for k, b in segs if k == "say"]
        cur = t
        for kind, arg in acts:
            if kind == "wait":
                cur += arg; continue
            if kind == "press":
                ev.append((cur, seq, "down", arg)); seq += 1
                ev.append((cur + 1, seq, "up", arg)); seq += 1
                continue
            name = {"hold": "down", "release": "up"}.get(kind, kind)
            ev.append((cur, seq, name, arg)); seq += 1
    ev.sort(key=lambda e: (e[0], e[1]))
    down = set()
    for e in ev:
        if e[2] == "down": down.add(e[3])
        elif e[2] == "up": down.discard(e[3])
    return {"events": [(e[0], e[2], e[3]) for e in ev], "dialogue": dia, "rejected": rej, "held": sorted(down)}
def action(rng):
    r = rng.random()
    if r < 0.45:
        return rng.choice(["press", "hold", "hold", "release"]) + " " + rng.choice(sorted(KEYS) + (["Q"] if rng.random() < 0.05 else []))
    if r < 0.6:
        return f"move {rng.randint(-150, 150)} {rng.randint(-150, 150)}"
    if r < 0.7:
        return "click " + rng.choice(["left", "right", "left"])
    if r < 0.97:
        return f"wait {rng.randint(1, 3)}"
    return rng.choice(["wait 0", "jump", "move 1"])
for seed in (15, 52, 97):
    rng = random.Random(seed)
    for trial in range(120):
        chunks, t = [], 0
        for _ in range(rng.randint(1, 5)):
            t += rng.randint(0, 4)
            segs = []
            for _ in range(rng.randint(0, 3)):
                kind = rng.choice(["act", "act", "say", "think"])
                body = "; ".join(action(rng) for _ in range(rng.randint(0, 5))) if kind == "act" else rng.choice(["ok", " look left ", ""])
                segs.append(f"<{kind}>{body}</{kind}>")
            text = rng.choice(["", " ", "\n"]).join(segs)
            if rng.random() < 0.05:
                text = "noise" + text
            chunks.append((t, text))
        got = {fn}(chunks, set(KEYS))
        assert got == oracle(chunks), (seed, trial, chunks, got)
"""},
    ],
    "solution": '''import re

_SEGMENT = re.compile(r"\\s*<(think|say|act)>([^<]*)</\\1>\\s*")
_INT = re.compile(r"-?\\d+")


def _parse_action(text, keys):
    parts = text.split(" ")
    if len(parts) == 2 and parts[0] in ("press", "hold", "release") and parts[1] in keys:
        return parts[0], parts[1]
    if len(parts) == 3 and parts[0] == "move" and all(_INT.fullmatch(p) for p in parts[1:]):
        return "move", tuple(max(-100, min(100, int(p))) for p in parts[1:])
    if len(parts) == 2 and parts[0] == "click" and parts[1] in ("left", "right"):
        return "click", parts[1]
    if len(parts) == 2 and parts[0] == "wait" and parts[1].isdigit() and parts[1].isascii() and int(parts[1]) >= 1:
        return "wait", int(parts[1])
    return None


def _parse_chunk(text, keys):
    segments, pos = [], 0
    for match in _SEGMENT.finditer(text):
        if match.start() != pos:
            return None
        segments.append((match.group(1), match.group(2)))
        pos = match.end()
    if pos != len(text):
        return None
    dialogue, actions = [], []
    for kind, body in segments:
        if kind == "say":
            dialogue.append(body.strip())
        elif kind == "act":
            for raw in body.split(";"):
                raw = raw.strip()
                if not raw:
                    continue
                action = _parse_action(raw, keys)
                if action is None:
                    return None
                actions.append(action)
    return dialogue, actions


def _held(events):
    down = set()
    for _, _, kind, arg in sorted(events, key=lambda e: (e[0], e[1])):
        if kind == "down":
            down.add(arg)
        elif kind == "up":
            down.discard(arg)
    return down


def schedule_actions(chunks, keys):
    ticks = [t for t, _ in chunks]
    if any(b < a for a, b in zip(ticks, ticks[1:])):
        raise ValueError("arrival ticks must be non-decreasing")
    events, dialogue, rejected, seq = [], [], [], 0
    for index, (tick, text) in enumerate(chunks):
        parsed = _parse_chunk(text, keys)
        if parsed is None:
            rejected.append(index)
            continue
        says, actions = parsed
        events = [e for e in events if e[0] < tick]
        for key in sorted(_held(events)):
            events.append((tick, seq, "up", key))
            seq += 1
        dialogue.extend((tick, s) for s in says)
        cursor = tick
        for kind, arg in actions:
            if kind == "wait":
                cursor += arg
                continue
            if kind == "press":
                events.append((cursor, seq, "down", arg))
                events.append((cursor + 1, seq + 1, "up", arg))
                seq += 2
                continue
            name = {"hold": "down", "release": "up"}.get(kind, kind)
            events.append((cursor, seq, name, arg))
            seq += 1
    events.sort(key=lambda e: (e[0], e[1]))
    return {
        "events": [(t, kind, arg) for t, _, kind, arg in events],
        "dialogue": dialogue,
        "rejected": rejected,
        "held": sorted(_held(events)),
    }
''',
    "interview_questions": interview(
        concept=[
            "How does an agent like SIMA 2 turn a language model's output into keyboard and mouse control of a running game?",
            "Why do embodied agents emit actions in chunks, and why is every chunk already stale when it starts executing?",
        ],
        deep_dive=[
            "A new chunk arrives while the old plan still holds a key. Walk through what the scheduler drops, what it releases and in which order.",
            "Why should a malformed chunk be rejected entirely instead of executing the valid prefix, and why must it not preempt the running plan?",
            "How does a slow planner issuing instructions every k steps (Gemini Pro) combine with a fast controller (SIMA 2)? What goes wrong if the controller ignores stale instructions?",
        ],
        tradeoffs=[
            "Text actions from a general LLM versus a dedicated action head with discrete tokens or continuous outputs: generality, latency and precision?",
            "Long action chunks versus short chunks with frequent re-planning: smoothness, reactivity and inference cost?",
        ],
    ),
}
