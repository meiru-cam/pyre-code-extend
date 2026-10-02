"""Refactor a legacy chat bot function into bots behind one interface, then isolate them and connect them with events."""

from ._interview import interview

# Learner classes are reached through the room's method globals; the legacy function is the oracle.
_HELPERS = r"""
import random

def learner(name):
    funcs = [f for f in vars({fn}).values() if hasattr(f, "__globals__")]
    assert funcs, "{fn} must define its methods in Python"
    names = funcs[0].__globals__
    assert name in names, f"define a class named {name}"
    return names[name]

LEGACY = '''# Legacy chat bots: one function and module-level state. Its log is the behavior to keep.
cheers = {}
focus_end = {}
poll = {}
log = []


def clock_text(t):
    return f"{(t // 60) % 24:02d}:{t % 60:02d}"


def handle_message(sender, text, now):
    log.append(f"{sender}: {text}")

    # FocusBot reminders, in the order users first started a session
    for user, end in focus_end.items():
        if user in text and now < end:
            log.append(f"FocusBot: {user} is focusing until {clock_text(end)}, please wait.")

    # CheerBot
    if text.startswith("/cheer "):
        target = text[len("/cheer "):].strip()
        if len(target) > 1 and target[0] == "@":
            cheers[target[1:]] = cheers.get(target[1:], 0) + 1
            log.append(f"CheerBot: {sender} cheered {target}, now at {cheers[target[1:]]}.")

    # PollBot: a new poll
    if text.startswith("/poll "):
        pieces = [p.strip() for p in text[len("/poll "):].split("|")]
        if len(pieces) >= 3:
            labels = [chr(ord("A") + i) for i in range(len(pieces) - 1)]
            poll.clear()
            poll.update(author=sender, options=dict(zip(labels, pieces[1:])), votes=dict.fromkeys(labels, 0))
            shown = ", ".join(f"{label}) {option}" for label, option in poll["options"].items())
            log.append(f"PollBot: {sender} asks {pieces[0]} [{shown}]")

    # PollBot: a vote on the poll in progress
    if text.startswith("/vote "):
        letter = text[len("/vote "):].strip().upper()
        if poll and letter in poll["votes"]:
            poll["votes"][letter] += 1
            tally = ", ".join(f"{label}: {count}" for label, count in poll["votes"].items())
            log.append(f"PollBot: {sender} chose {letter} ({poll['options'][letter]}); {tally}")

    # FocusBot: a new session, replacing the sender's old one
    if text.startswith("/focus "):
        arg = text[len("/focus "):].strip()
        if arg.isascii() and arg.isdigit() and int(arg) >= 1:
            focus_end[sender] = now + int(arg)
            log.append(f"FocusBot: {sender} is focusing until {clock_text(now + int(arg))}.")
'''

def legacy_log(script):
    ns = {}
    exec(LEGACY, ns)
    for message in script:
        ns["handle_message"](*message)
    return ns["log"]

def default_room():
    Clock, FocusBot, CheerBot, PollBot = (learner(n) for n in ("Clock", "FocusBot", "CheerBot", "PollBot"))
    clock = Clock()
    room = {fn}(clock)
    room.register(FocusBot(clock, {}))
    room.register(CheerBot({}))
    room.register(PollBot({}))
    return room

def run(room, script):
    log = []
    for message in script:
        log = room.send(*message)
    return log

NAMES = ["Ines", "Omar", "Kai", "Om"]

def random_script(rng, n):
    now, out = rng.choice([0, 470, 1400]), []
    for _ in range(n):
        now += rng.choice([0, 1, 3, 10, 40])
        who, other = rng.choice(NAMES), rng.choice(NAMES)
        text = rng.choice([
            f"/cheer @{other}", f"/cheer {other}", "/cheer @", f"/cheer   @{other}  ", f"/Cheer @{other}", f"/cheer@{other}",
            f"/focus {rng.choice(['1', '5', '30', '0', '-3', 'abc', ' 12 ', '2.5', ''])}",
            f"/poll Q{rng.randint(0, 9)} | x | y", "/poll Lunch? | a | b | c", "/poll Only | one", "/poll  |  p | q ",
            f"/vote {rng.choice(['a', 'B', 'c', 'z', '', ' b '])}", "/votes A",
            f"hi {other}", f"{other} and {rng.choice(NAMES)}", "plain text",
        ])
        out.append((who, text, now))
    return out
"""

TESTS = [
    {"name": "Part 1: the worked example", "part": 1, "behavior": "state.invariant", "code": _HELPERS + r"""
room = default_room()
log = run(room, [("Ines", "Morning all", 480), ("Omar", "/focus 45", 482), ("Ines", "Omar, got a sec?", 490),
                 ("Kai", "/cheer @Ines", 495), ("Ines", "/poll Retro day? | Thu | Fri", 500), ("Omar", "/vote a", 505),
                 ("Kai", "/vote B", 506), ("Kai", "/cheer Omar", 527)])
assert log == [
    "Ines: Morning all",
    "Omar: /focus 45",
    "FocusBot: Omar is focusing until 08:47.",
    "Ines: Omar, got a sec?",
    "FocusBot: Omar is focusing until 08:47, please wait.",
    "Kai: /cheer @Ines",
    "CheerBot: Kai cheered @Ines, now at 1.",
    "Ines: /poll Retro day? | Thu | Fri",
    "PollBot: Ines asks Retro day? [A) Thu, B) Fri]",
    "Omar: /vote a",
    "PollBot: Omar chose A (Thu); A: 1, B: 0",
    "Kai: /vote B",
    "PollBot: Kai chose B (Fri); A: 1, B: 1",
    "Kai: /cheer Omar",
], log
"""},
    {"name": "Part 1: random scripts match the legacy log", "part": 1, "visibility": "unshown", "behavior": "state.invariant",
     "failure_message": "On a random message sequence, the room's log differed from the legacy handle_message log: check reminder order and timing, command parsing, poll replacement and vote matching.",
     "code": _HELPERS + r"""
for seed in range(150):
    script = random_script(random.Random(seed), 40)
    got, want = run(default_room(), script), legacy_log(script)
    for i, (g, w) in enumerate(zip(got, want)):
        assert g == w, (seed, i, g, w)
    assert len(got) == len(want), (seed, len(got), len(want))
"""},
    {"name": "Part 1: injected state, clock and fresh rooms", "part": 1, "visibility": "unshown", "behavior": "state.invariant",
     "failure_message": "Bots must keep state only in the dict they were given (changed in place, shared when the dict is shared), read time only from the injected clock, answer can_handle exactly when handle has lines, and two rooms must not share anything; send returns a copy of the log.",
     "code": _HELPERS + r"""
Clock, FocusBot, CheerBot, PollBot = (learner(n) for n in ("Clock", "FocusBot", "CheerBot", "PollBot"))
counts = {"Ines": 4}
cheer = CheerBot(counts)
assert cheer.can_handle("Kai", "/cheer @Ines") and not cheer.can_handle("Kai", "/cheer Ines")
assert cheer.handle("Kai", "/cheer @Ines") == ["CheerBot: Kai cheered @Ines, now at 5."]
assert counts == {"Ines": 5}
clock = Clock(100)
until = {"Omar": 130}
focus = FocusBot(clock, until)
assert focus.handle("Ann", "ask Omar") == ["FocusBot: Omar is focusing until 02:10, please wait."]
assert focus.handle("Ann", "/focus 5") == ["FocusBot: Ann is focusing until 01:45."] and until["Ann"] == 105
clock.now = 130
assert not focus.can_handle("Ann", "ask Omar"), "a session ending at 130 is over at 130"
assert focus.can_handle("Ann", "/focus 1440") and focus.handle("Ann", "/focus 1440") == ["FocusBot: Ann is focusing until 02:10."]
order = FocusBot(Clock(0), {})
order.handle("Kai", "/focus 50")
order.handle("Ines", "/focus 60")
order.handle("Kai", "/focus 90")
assert order.handle("Om", "Ines and Kai?") == ["FocusBot: Kai is focusing until 01:30, please wait.",
                                               "FocusBot: Ines is focusing until 01:00, please wait."], "first-start order, kept when a session is replaced"
shared = {}
first, second = PollBot(shared), PollBot(shared)
assert not first.can_handle("Kai", "/vote A"), "no poll yet"
first.handle("Ines", "/poll Q? | x | y")
assert shared, "the poll must live in the injected dict"
assert second.can_handle("Kai", "/vote b") and second.handle("Kai", "/vote b") == ["PollBot: Kai chose B (y); A: 0, B: 1"]
script = [("Omar", "/focus 9", 0), ("Kai", "/cheer @Ines", 1), ("Ines", "/poll Q | a | b", 2)]
room_a = default_room()
run(room_a, script)
room_b = default_room()
assert room_b.send("Ines", "Omar /vote a", 3) == ["Ines: Omar /vote a"], "a new room starts empty"
log = room_a.send("Kai", "/cheer @Ines", 3)
assert log[-1] == "CheerBot: Kai cheered @Ines, now at 2."
log.append("junk")
assert room_a.send("Kai", "x", 4)[-2:] == ["CheerBot: Kai cheered @Ines, now at 2.", "Kai: x"]
"""},
    {"name": "Part 2: the worked example", "part": 2, "behavior": "contract.signature", "code": _HELPERS + r"""
class PingBot:
    commands = ("/ping",)
    def can_handle(self, sender, text):
        return text == "/ping"
    def handle(self, sender, text):
        return ["PingBot: pong"]
class VoteCounter:
    commands = ("/vote", "/tally")
    def can_handle(self, sender, text):
        return True
    def handle(self, sender, text):
        return ["VoteCounter: here"]
class BrokenBot:
    def can_handle(self, sender, text):
        return True
    def handle(self, sender, text):
        raise RuntimeError("boom")
room = default_room()
room.register(PingBot())
assert room.send("Kai", "/ping", 10)[-2:] == ["Kai: /ping", "PingBot: pong"]
try:
    room.register(VoteCounter())
    raise AssertionError("a clashing command must raise ValueError")
except ValueError:
    pass
room.register(BrokenBot())
class LateBot:
    def can_handle(self, sender, text):
        return True
    def handle(self, sender, text):
        return ["LateBot: ok"]
room.register(LateBot())
assert room.send("Kai", "/ping", 11)[-4:] == ["Kai: /ping", "PingBot: pong", "BrokenBot: failed", "LateBot: ok"]
"""},
    {"name": "Part 2: dispatch rules and failures", "part": 2, "visibility": "unshown", "behavior": "protocol.validation",
     "failure_message": "handle must be called only after can_handle is True; a bot without commands claims nothing; a rejected bot must not be registered at all; built-in bots claim /cheer, /focus, /poll and /vote; an exception in can_handle or handle becomes one '<class name>: failed' line and later bots still run.",
     "code": _HELPERS + r"""
Clock = learner("Clock")
calls = []
class Picky:
    def can_handle(self, sender, text):
        calls.append("can")
        return text.startswith("!")
    def handle(self, sender, text):
        calls.append("handle")
        return ["Picky: " + text[1:]]
class Quiet:
    pass
class Silent:
    def can_handle(self, sender, text):
        return True
    def handle(self, sender, text):
        return []
class BadCheck:
    commands = ()
    def can_handle(self, sender, text):
        raise KeyError(text)
    def handle(self, sender, text):
        return ["never"]
room = {fn}(Clock())
room.register(Picky())
room.register(Silent())
room.register(Picky())
assert room.send("a", "hello", 1) == ["a: hello"] and calls == ["can", "can"]
assert room.send("a", "!hi", 2)[-2:] == ["Picky: hi", "Picky: hi"]
room.register(BadCheck())
assert room.send("a", "x", 3)[-2:] == ["a: x", "BadCheck: failed"]
for word in ["/cheer", "/focus", "/poll", "/vote"]:
    fresh = default_room()
    Clash = type("Clash", (Silent,), {"commands": ("/new", word)})
    try:
        fresh.register(Clash())
        raise AssertionError(f"{word} is owned by a built-in bot")
    except ValueError:
        pass
    New = type("New", (), {"commands": ("/new",), "can_handle": lambda s, a, t: t == "/new", "handle": lambda s, a, t: ["New: ok"]})
    fresh.register(New())
    assert fresh.send("a", "/new", 1)[-1] == "New: ok", "a rejected bot must not keep its claim on /new"
empty = {fn}(Clock())
assert empty.send("a", "/cheer @b", 5) == ["a: /cheer @b"]
"""},
    {"name": "Part 2: legacy log with extra bots", "part": 2, "visibility": "unshown", "behavior": "state.invariant",
     "failure_message": "With a failing bot and an always-silent bot registered after the three built-in bots, the log must equal the legacy log plus one '<class name>: failed' line per message.",
     "code": _HELPERS + r"""
class Flaky:
    def can_handle(self, sender, text):
        return True
    def handle(self, sender, text):
        raise ValueError("nope")
class Mute:
    commands = ("/mute",)
    def can_handle(self, sender, text):
        return False
    def handle(self, sender, text):
        raise AssertionError("handle called although can_handle was False")
for seed in range(60):
    script = random_script(random.Random(100 + seed), 30)
    room = default_room()
    room.register(Mute())
    room.register(Flaky())
    got = run(room, script)
    expected = []
    ns = {}
    exec(LEGACY, ns)
    for message in script:
        start = len(ns["log"])
        ns["handle_message"](*message)
        expected += ns["log"][start:] + ["Flaky: failed"]
    assert got == expected, seed
"""},
    {"name": "Part 3: the worked example", "part": 3, "behavior": "events.ordering", "code": _HELPERS + r"""
Clock, CheerBot, PollBot, EventBus = (learner(n) for n in ("Clock", "CheerBot", "PollBot", "EventBus"))
bus = EventBus()
room = {fn}(Clock())
room.register(PollBot({}, bus, close_at=2))
room.register(CheerBot({"Ines": 1}, bus))
room.send("Ines", "/poll Retro day? | Thu | Fri", 500)
room.send("Omar", "/vote b", 505)
assert room.send("Kai", "/vote B", 506)[-3:] == ["Kai: /vote B", "PollBot: Kai chose B (Fri); A: 0, B: 2", "CheerBot: Ines's poll is decided, now at 2."]
assert room.send("Omar", "/vote a", 507)[-1] == "Omar: /vote a"
"""},
    {"name": "Part 3: bus contract and closing rules", "part": 3, "visibility": "unshown", "behavior": "events.ordering",
     "failure_message": "publish must call handlers for the event's type in subscription order and return their lines concatenated; a poll closes when one option reaches close_at, publishes poll_closed with the author once, accepts no later votes, and a new poll can start; without a bus or close_at, bots behave as in Part 1; neither bot may hold the other.",
     "code": _HELPERS + r"""
Clock, CheerBot, PollBot, EventBus, Event = (learner(n) for n in ("Clock", "CheerBot", "PollBot", "EventBus", "Event"))
bus = EventBus()
seen = []
bus.subscribe("ping", lambda e: seen.append(("one", e.data)) or ["one"])
bus.subscribe("ping", lambda e: seen.append(("two", e.data)) or ["two", "three"])
bus.subscribe("other", lambda e: ["never"])
event = Event("ping", {"k": 1})
assert event.type == "ping" and event.data == {"k": 1}
assert bus.publish(event) == ["one", "two", "three"] and seen == [("one", {"k": 1}), ("two", {"k": 1})]
assert bus.publish(Event("nobody", {})) == []
closed = []
bus2 = EventBus()
bus2.subscribe("poll_closed", lambda e: closed.append(dict(e.data)) or [])
poll = PollBot({}, bus2, close_at=3)
for text in ["/poll Q | x | y | z", "/vote c", "/vote C", "/vote a"]:
    poll.handle("Ann", text)
assert closed == []
assert poll.handle("Bo", "/vote c") == ["PollBot: Bo chose C (z); A: 1, B: 0, C: 3"]
assert len(closed) == 1 and closed[0]["author"] == "Ann", closed
assert not poll.can_handle("Bo", "/vote a"), "a closed poll takes no votes"
assert poll.can_handle("Bo", "/poll Next | p | q")
poll.handle("Bo", "/poll Next | p | q")
assert poll.handle("Ann", "/vote a") == ["PollBot: Ann chose A (p); A: 1, B: 0"]
counts = {}
cheer = CheerBot(counts, bus2)
quiet = PollBot({}, None, close_at=1)
quiet.handle("Cy", "/poll R | u | v")
assert quiet.handle("Dee", "/vote a") == ["PollBot: Dee chose A (u); A: 1, B: 0"]
assert not quiet.can_handle("Dee", "/vote a"), "close_at applies without a bus too"
assert counts == {}
endless = PollBot({})
endless.handle("Cy", "/poll S | m | n")
for _ in range(5):
    endless.handle("Dee", "/vote a")
assert endless.can_handle("Dee", "/vote b"), "without close_at a poll never closes"
both = [CheerBot({}, EventBus()), PollBot({}, EventBus(), close_at=2)]
for bot in both:
    for value in vars(bot).values():
        assert not isinstance(value, (CheerBot, PollBot)), "PollBot and CheerBot must not hold each other"
"""},
    {"name": "Part 3: random scripts with closing polls", "part": 3, "visibility": "unshown", "behavior": "events.ordering",
     "failure_message": "With a bus and close_at = 3, the log must equal the legacy log except that a poll closes when an option reaches 3 votes and the author's cheer line follows the closing vote line.",
     "code": _HELPERS + r"""
Clock, FocusBot, CheerBot, PollBot, EventBus = (learner(n) for n in ("Clock", "FocusBot", "CheerBot", "PollBot", "EventBus"))
CLOSING = LEGACY.replace(
    '''            log.append(f"PollBot: {sender} chose {letter} ({poll['options'][letter]}); {tally}")''',
    '''            log.append(f"PollBot: {sender} chose {letter} ({poll['options'][letter]}); {tally}")
            if poll["votes"][letter] >= 3:
                author = poll["author"]
                poll.clear()
                cheers[author] = cheers.get(author, 0) + 1
                log.append(f"CheerBot: {author}'s poll is decided, now at {cheers[author]}.")''')
assert CLOSING != LEGACY
for seed in range(120):
    script = random_script(random.Random(500 + seed), 50)
    ns = {}
    exec(CLOSING, ns)
    for message in script:
        ns["handle_message"](*message)
    clock, bus = Clock(), EventBus()
    room = {fn}(clock)
    room.register(FocusBot(clock, {}))
    room.register(CheerBot({}, bus))
    room.register(PollBot({}, bus, close_at=3))
    assert run(room, script) == ns["log"], seed
"""},
]

TASK = {
    "title": "Chat Bot Refactoring",
    "difficulty": "Medium",
    "version": 1,
    "function_name": "ChatRoom",
    "description_en": r"""Refactor the legacy `handle_message` in the starter code into bot classes behind one interface, then make them safe to extend and let them talk through events.

The requirement arrives in parts. Each part keeps every earlier behavior, so one set of classes passes all parts at the end. Pass every test of the current part to reveal the next one.

**Rules for every part:**
- The starter's `handle_message(sender, text, now)` is the behavior to keep. `now` is minutes since midnight of the first day and never decreases. Times print as `HH:MM` of the day, so minute `1450` prints `00:10`.
- A command counts only at the very start of the text: the word, one space, then an argument whose surrounding whitespace is ignored.
- `/cheer @name` with at least one character after `@` adds 1 to `name`'s cheers. Line: `CheerBot: {sender} cheered {target}, now at {count}.` with the target as written, `@` included.
- `/focus n`, where `n` is ASCII digits worth at least `1`, sets the sender's session to end at `now + n`, replacing any old one. Line: `FocusBot: {sender} is focusing until {HH:MM}.`
- On every message, for each user with a session whose name is a substring of the text and whose end is after `now`, FocusBot first adds `FocusBot: {user} is focusing until {HH:MM}, please wait.` These lines come before every other bot line for the message, in the order users first started a session, and are decided before a `/focus` in the same message.
- `/poll question | option | option ...` splits on `|` and strips each piece. With at least two options it replaces any poll in progress; options get labels `A`, `B`, … and zero votes. Line: `PollBot: {sender} asks {question} [A) {option}, B) {option}]`.
- `/vote x` matches `x` case-insensitively against the current poll's labels and adds a vote. Line: `PollBot: {sender} chose {LABEL} ({option}); A: {votes}, B: {votes}`, every label in order.
- Anything else, including a malformed command, gets no bot line.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why this shows up in interviews:** it is a refactoring round: the code works, and the task is to change its shape without changing its output, then show the new shape is easy to extend and test. Each later part adds one requirement.

**Where it is used:** Slack and Discord bot frameworks, plugin systems with a registry, and services that replace module-level globals with injected state so each test gets a fresh instance.

Adapted from the chat bot refactoring question in Schuture/OpenAI-Interview-Notes (CC BY-NC 4.0), reworded, with new line formats and a new legacy function. The source's reading questions become graded rules: failure isolation and command ownership in Part 2. Its unit-testing part is not graded.""",
    "parts": [
        {
            "title": "Bots behind one interface",
            "description_en": r"""**Classes:**
- `Clock(now=0)` holds an attribute `now`.
- `CheerBot(cheer_counts)`, `FocusBot(clock, focus_until)` and `PollBot(poll_state)` each have `can_handle(sender, text) -> bool` and `handle(sender, text) -> list[str]`.
- `ChatRoom(clock)` has `register(bot)` and `send(sender, text, now) -> list[str]`.

**Rules:**
- Each bot keeps its state only in the dict it was given and changes that dict in place: `cheer_counts` maps names to counts, `focus_until` maps users to end minutes, and the layout of `poll_state` is yours, empty meaning no poll. Two bots given the same dict share that state. Nothing lives at module level.
- `FocusBot` reads the time only from `clock.now`. Never read the system clock.
- `can_handle` is `True` exactly when `handle` would return at least one line.
- `send` sets `clock.now = now`, appends `"{sender}: {text}"`, then, for each bot in registration order whose `can_handle` is `True`, appends the lines from `handle`. It returns a copy of the whole log.
- With `FocusBot`, `CheerBot` and `PollBot` registered in that order on one clock, the log equals `handle_message`'s for every sequence of messages.

**Example**, those three bots:
- `send("Omar", "/focus 45", 482)` adds `FocusBot: Omar is focusing until 08:47.`
- `send("Ines", "Omar, got a sec?", 490)` adds `FocusBot: Omar is focusing until 08:47, please wait.`
- `send("Ines", "/poll Retro day? | Thu | Fri", 500)` adds `PollBot: Ines asks Retro day? [A) Thu, B) Fri]`
- `send("Omar", "/vote a", 505)` adds `PollBot: Omar chose A (Thu); A: 1, B: 0`
- `send("Kai", "/cheer Omar", 527)` adds nothing: no `@`, and Omar's session ended at 527""",
        },
        {
            "title": "Isolation and command ownership",
            "description_en": r"""Keep Part 1. The room now accepts any bot and protects itself from bad ones.

- Any object with `can_handle` and `handle` can be registered. `handle` is called only when `can_handle` returned `True` for that message.
- Each bot may have an attribute `commands`, a tuple of command words such as `("/ping",)`. A bot without it claims nothing. The built-in bots claim `("/cheer",)`, `("/focus",)` and `("/poll", "/vote")`.
- `register` raises `ValueError` if the bot claims a command word that a registered bot already claims, and then registers nothing.
- If a bot's `can_handle` or `handle` raises, `send` appends `"{class name}: failed"` instead of its lines and goes on with the next bot.

**Example:** register the three built-in bots, then a `PingBot` with `commands = ("/ping",)` that answers `/ping` with `["PingBot: pong"]`:
- `send("Kai", "/ping", 10)` adds `PingBot: pong`
- registering another bot with `commands = ("/vote", "/tally")` raises `ValueError`, and later messages behave as if it was never offered
- a bot whose `handle` raises adds a line such as `BrokenBot: failed`, and bots after it still run""",
        },
        {
            "title": "Events between bots",
            "description_en": r"""Keep Parts 1–2. `PollBot` and `CheerBot` now cooperate without referring to each other.

**Classes:**
- `Event(type, data)` holds attributes `type` (a `str`) and `data` (a `dict`).
- `EventBus()` has `subscribe(event_type, handler)` and `publish(event) -> list[str]`. `publish` calls every handler subscribed to `event.type` in subscription order and returns all the lines they return, concatenated.
- `PollBot(poll_state, bus=None, close_at=None)` and `CheerBot(cheer_counts, bus=None)`. With the defaults, both behave exactly as in Part 1.

**Rules:**
- `CheerBot` given a bus subscribes to `"poll_closed"` when it is created. For each such event it adds 1 to the cheers of `event.data["author"]` and returns `CheerBot: {author}'s poll is decided, now at {count}.`
- When a vote brings an option to `close_at` votes, `PollBot` ends the poll: a later `/vote` behaves as if no poll is in progress. If it has a bus, it publishes `Event("poll_closed", {"author": ...})` and puts the returned lines right after its own vote line.
- `PollBot` and `CheerBot` hold no reference to each other.

**Example**, one bus, `PollBot({}, bus, close_at=2)` and `CheerBot({"Ines": 1}, bus)` registered:
- `send("Ines", "/poll Retro day? | Thu | Fri", 500)`, then `send("Omar", "/vote b", 505)`
- `send("Kai", "/vote B", 506)` adds `PollBot: Kai chose B (Fri); A: 0, B: 2` and then `CheerBot: Ines's poll is decided, now at 2.`
- `send("Omar", "/vote a", 507)` adds nothing""",
        },
    ],
    "hints": [
        {"level": 1, "kind": "questions", "content": "Which three module-level names hold state in the legacy code, and which bot owns each one? If a test creates two rooms, what must be different about where that state lives? Why does FocusBot need to be registered first?"},
        {"level": 2, "kind": "analysis", "content": "Move each legacy block into its bot's handle, reading and writing self._state, the dict passed to __init__. can_handle can call the same parsing helper and check for a result. FocusBot.handle builds the reminders first, then applies /focus. ChatRoom keeps a list of bots and a log; send sets clock.now, appends the echo line, loops over the bots and returns list(self._log)."},
    ],
    "model_connections": [
        "Agent frameworks route each tool call to one registered tool by name, and refuse a second tool claiming the same name for the same reason register refuses a clashing command.",
        "Training loops use callbacks and event hooks, such as on_step_end, so logging, checkpointing and early stopping react to the trainer without the trainer knowing about them.",
    ],
    "pro_con_analysis": {
        "pros": [
            "Injected state makes every bot testable alone with a fresh dict and a fake clock.",
            "A registry of command words turns a silent clash between bots into an error at registration.",
            "An event bus lets one bot react to another without either importing the other.",
        ],
        "cons": [
            "Behavior that spans bots, such as reminders coming first, now depends on registration order instead of being visible in one function.",
            "Catching every exception per bot can hide real bugs unless the failure is logged somewhere useful.",
            "Events make the flow harder to follow: the line after a vote comes from a handler you cannot see from PollBot.",
        ],
    },
    "tests": TESTS,
    "solution": r'''# Adapted from Schuture/OpenAI-Interview-Notes (code under the MIT License).
from abc import ABC, abstractmethod


def _clock_text(t):
    return f"{(t // 60) % 24:02d}:{t % 60:02d}"


def _argument(text, command):
    """The stripped text after 'command ', or None when the text does not start with it."""
    prefix = command + " "
    return text[len(prefix):].strip() if text.startswith(prefix) else None


class Clock:
    def __init__(self, now=0):
        self.now = now  # the room sets it; bots only read it


class Event:
    def __init__(self, type, data):
        self.type = type
        self.data = data


class EventBus:
    def __init__(self):
        self._handlers = {}

    def subscribe(self, event_type, handler):
        self._handlers.setdefault(event_type, []).append(handler)

    def publish(self, event):
        lines = []
        for handler in self._handlers.get(event.type, []):
            lines.extend(handler(event))
        return lines


class ChatBot(ABC):
    commands = ()

    @abstractmethod
    def can_handle(self, sender, text):
        """True exactly when handle would return at least one line."""

    @abstractmethod
    def handle(self, sender, text):
        """The lines this bot adds for this message."""


class CheerBot(ChatBot):
    commands = ("/cheer",)

    def __init__(self, cheer_counts, bus=None):
        self._counts = cheer_counts
        if bus is not None:
            bus.subscribe("poll_closed", self._on_poll_closed)

    @staticmethod
    def _target(text):
        target = _argument(text, "/cheer")
        return target if target is not None and len(target) > 1 and target[0] == "@" else None

    def _add(self, name):
        self._counts[name] = self._counts.get(name, 0) + 1
        return self._counts[name]

    def can_handle(self, sender, text):
        return self._target(text) is not None

    def handle(self, sender, text):
        target = self._target(text)
        return [f"CheerBot: {sender} cheered {target}, now at {self._add(target[1:])}."]

    def _on_poll_closed(self, event):
        author = event.data["author"]
        return [f"CheerBot: {author}'s poll is decided, now at {self._add(author)}."]


class FocusBot(ChatBot):
    commands = ("/focus",)

    def __init__(self, clock, focus_until):
        self._clock = clock
        self._until = focus_until  # insertion order is the order users first started a session

    @staticmethod
    def _minutes(text):
        arg = _argument(text, "/focus")
        if arg is None or not (arg.isascii() and arg.isdigit()) or int(arg) < 1:
            return None
        return int(arg)

    def _reminders(self, text):
        now = self._clock.now
        return [f"FocusBot: {user} is focusing until {_clock_text(end)}, please wait."
                for user, end in self._until.items() if user in text and now < end]

    def can_handle(self, sender, text):
        return bool(self._reminders(text)) or self._minutes(text) is not None

    def handle(self, sender, text):
        lines = self._reminders(text)  # decided before this message's own /focus takes effect
        minutes = self._minutes(text)
        if minutes is not None:
            end = self._clock.now + minutes
            self._until[sender] = end  # reassigning a key keeps its place in the dict
            lines.append(f"FocusBot: {sender} is focusing until {_clock_text(end)}.")
        return lines


class PollBot(ChatBot):
    commands = ("/poll", "/vote")

    def __init__(self, poll_state, bus=None, close_at=None):
        self._poll = poll_state  # empty means no poll in progress
        self._bus = bus
        self._close_at = close_at

    @staticmethod
    def _new_poll(text):
        arg = _argument(text, "/poll")
        if arg is None:
            return None
        pieces = [piece.strip() for piece in arg.split("|")]
        return pieces if len(pieces) >= 3 else None  # a question and at least two options

    def _vote(self, text):
        arg = _argument(text, "/vote")
        if arg is None or not self._poll:
            return None
        letter = arg.upper()
        return letter if letter in self._poll["votes"] else None

    def can_handle(self, sender, text):
        return self._new_poll(text) is not None or self._vote(text) is not None

    def handle(self, sender, text):
        pieces = self._new_poll(text)
        if pieces is not None:
            labels = [chr(ord("A") + i) for i in range(len(pieces) - 1)]
            self._poll.clear()
            self._poll.update(author=sender, options=dict(zip(labels, pieces[1:])), votes=dict.fromkeys(labels, 0))
            shown = ", ".join(f"{label}) {option}" for label, option in self._poll["options"].items())
            return [f"PollBot: {sender} asks {pieces[0]} [{shown}]"]
        letter = self._vote(text)
        votes = self._poll["votes"]
        votes[letter] += 1
        tally = ", ".join(f"{label}: {count}" for label, count in votes.items())
        lines = [f"PollBot: {sender} chose {letter} ({self._poll['options'][letter]}); {tally}"]
        if self._close_at is not None and votes[letter] >= self._close_at:
            author = self._poll["author"]
            self._poll.clear()  # later votes see no poll in progress
            if self._bus is not None:
                lines += self._bus.publish(Event("poll_closed", {"author": author}))
        return lines


class ChatRoom:
    def __init__(self, clock):
        self._clock = clock
        self._bots = []
        self._claimed = set()
        self._log = []

    def register(self, bot):
        commands = tuple(getattr(bot, "commands", ()))
        taken = [command for command in commands if command in self._claimed]
        if taken:
            raise ValueError(f"{type(bot).__name__} claims {taken}, which another bot already owns")
        self._claimed.update(commands)
        self._bots.append(bot)

    def send(self, sender, text, now):
        self._clock.now = now
        self._log.append(f"{sender}: {text}")
        for bot in self._bots:
            try:
                if bot.can_handle(sender, text):
                    self._log.extend(bot.handle(sender, text))
            except Exception:
                self._log.append(f"{type(bot).__name__}: failed")  # one bad bot never stops the others
        return list(self._log)
''',
    "interview_questions": interview(
        concept=[
            "Why does module-level state make the legacy function hard to test, and what does injecting a dict into each bot change?",
            "Why must FocusBot read time from an injected clock rather than the system clock?",
        ],
        deep_dive=[
            "How do you show that the refactored room produces exactly the legacy log, rather than trusting that the two code paths look alike?",
        ],
        tradeoffs=[
            "Should a room catch exceptions from bots, and what do you lose when it does?",
            "Why check command ownership at registration rather than when a message arrives?",
            "What does an event bus buy over PollBot calling CheerBot directly, and what does it cost in traceability?",
            "Should publish return lines to the publisher, or should handlers write to the log themselves?",
        ],
    ),
}
