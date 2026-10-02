A chat service delivers every incoming message to a set of bots that react to slash commands. A message has a sender name, a text, and an integer timestamp `now`: the number of minutes since midnight of the service's first day, never smaller than the previous message's, so 9:00 on the first day is `540` and 9:00 on the second day is `1980`. Nothing reads the system clock — every call passes `now` explicitly. Sending a message first appends `"{sender}: {text}"` to a shared log, then lets each bot look at the raw text and possibly append more lines to that same log. A command is recognized only at the very start of the text, as the command word followed by one space (`/cheer `, `/focus `, `/poll `, `/vote `); whitespace around the rest of the text is ignored. Three bots are registered.

**CheerBot**

- Recognizes `/cheer @<name>`, where `<name>` is one or more characters right after the `@`. Any other shape (no `@`, or nothing after it) is not a valid command.
- Keeps a running cheer count per username, starting at 0.
- On a valid command, adds 1 to the named user's count and announces the sender, the target exactly as written (with its leading `@`), and the new count.

**FocusBot**

- Recognizes `/focus <n>`, where `<n>` is one or more decimal digits spelling a number `>= 1`. Any other shape (missing, negative, non-numeric) is not a valid command.
- On a valid command, starts a focus session for the sender that ends at minute `now + n`, replacing any session the sender already has, and announces the sender and the time of day the session ends at, formatted `HH:MM` (a session can run past midnight: one started at 23:50 for 20 minutes ends at 00:10).
- Independently of that, on *every* incoming message — including this bot's own command and every other bot's — for every user whose name appears as a substring of the raw text and whose session is still running (this message's `now` strictly before the session's end), the bot appends one reminder line, in the order those users first ever started a session. Reminders come before every other bot's line for the message, and are decided before a `/focus` in the same message takes effect.

**PollBot**

- Recognizes `/poll <question> | <option 1> | <option 2> | ...`: split the text after `/poll ` on `|` and strip the surrounding whitespace from every piece; the first piece is the question, the rest are the options. Valid only with at least 2 options; a valid command replaces whatever poll is currently in progress (voted on or not) with a fresh one, options labelled `A`, `B`, `C`, ... in the given order, every count starting at 0. Announces the poll's author (the sender), its question, and its labelled options.
- Recognizes `/vote <letter>`, matched case-insensitively against the labels of the poll currently in progress. Valid only when a poll is in progress and the letter matches one of its labels; adds one vote to that option and announces the voter, the chosen option's text, and the updated tally for every label, in label order.
- A command that isn't a valid `/poll` or `/vote` by the rules above is not recognized.

A command that fails its bot's validity rule above gets no response from any bot as a command; its text still counts as an ordinary message for FocusBot's reminders.

Here is the current implementation:

```python
cheer_counts = {}
focus_until = {}
poll_state = {}
messages = []


def format_time(t):
    return f"{(t // 60) % 24:02d}:{t % 60:02d}"


def handle_message(name, msg, now):
    messages.append(name + ": " + msg)

    # FocusBot logic: warn if this message mentions someone who is still heads-down
    for user in focus_until:
        end = focus_until[user]
        if user in msg and now < end:
            messages.append("FocusBot: " + user + " is heads-down until " + format_time(end) + ", try again later.")

    # CheerBot logic
    if msg[:7] == "/cheer ":
        target = msg.split(" ", 1)[1].strip()
        if target[:1] == "@" and len(target) > 1:
            username = target[1:]
            if username not in cheer_counts:
                cheer_counts[username] = 0
            cheer_counts[username] += 1
            messages.append(
                "CheerBot: "
                + name
                + " cheers for "
                + target
                + "! "
                + target
                + "'s cheer count is now "
                + str(cheer_counts[username])
                + "."
            )

    # PollBot logic: start a new poll
    if msg[:6] == "/poll ":
        parts = msg[6:].split("|")
        parts = [p.strip() for p in parts]
        question = parts[0]
        options = parts[1:]
        if len(options) >= 2:
            poll_state.clear()
            poll_state["question"] = question
            poll_state["options"] = options
            votes = {}
            for i in range(len(options)):
                votes[chr(65 + i)] = 0
            poll_state["votes"] = votes
            poll_state["author"] = name
            labeled = []
            for i in range(len(options)):
                labeled.append(chr(65 + i) + ") " + options[i])
            messages.append(f"PollBot: {name} started a poll -- {question} [{', '.join(labeled)}]")

    # PollBot logic: vote on the poll in progress
    if msg[:6] == "/vote ":
        letter = msg.split(" ", 1)[1].strip().upper()
        if poll_state and letter in poll_state["votes"]:
            poll_state["votes"][letter] += 1
            idx = ord(letter) - 65
            option_text = poll_state["options"][idx]
            tally_bits = []
            for k in poll_state["votes"]:
                tally_bits.append(k + ": " + str(poll_state["votes"][k]))
            tally = ", ".join(tally_bits)
            messages.append(f"PollBot: {name} voted {letter} ({option_text}). Tally -- {tally}")

    # FocusBot logic: start a focus session
    if msg[:7] == "/focus ":
        arg = msg.split(" ", 1)[1].strip()
        if arg.isdecimal() and int(arg) >= 1:
            minutes = int(arg)
            end = now + minutes
            focus_until[name] = end
            messages.append(
                "FocusBot: " + name + " is heads-down for " + str(minutes) + " min, back at " + format_time(end) + "."
            )
```

Its output on the script below is the reference for "unchanged behavior" in every part:

```python
SCRIPT = [
    ("Maya", "Hello everyone", 540),
    ("Theo", "/focus 20", 545),
    ("Priya", "Hey Theo, quick question", 550),
    ("Noah", "/cheer @Priya", 552),
    ("Sana", "/cheer @Priya", 560),
    ("Leo", "/cheer @Theo", 561),
    ("Maya", "/poll Lunch spot? | Tacos | Pasta | Salad", 600),
    ("Theo", "/vote B", 605),
    ("Priya", "/vote B", 606),
    ("Noah", "/vote c", 607),
    ("Sana", "/vote Z", 608),
    ("Leo", "/focus abc", 609),
    ("Maya", "/cheer bob", 610),
    ("Theo", "/poll Coffee? | OnlyOneOption", 611),
    ("Priya", "/vote B", 612),
]

for sender, text, now in SCRIPT:
    handle_message(sender, text, now)

EXPECTED_LOG = [
    "Maya: Hello everyone",
    "Theo: /focus 20",
    "FocusBot: Theo is heads-down for 20 min, back at 09:25.",
    "Priya: Hey Theo, quick question",
    "FocusBot: Theo is heads-down until 09:25, try again later.",
    "Noah: /cheer @Priya",
    "CheerBot: Noah cheers for @Priya! @Priya's cheer count is now 1.",
    "Sana: /cheer @Priya",
    "CheerBot: Sana cheers for @Priya! @Priya's cheer count is now 2.",
    "Leo: /cheer @Theo",
    "FocusBot: Theo is heads-down until 09:25, try again later.",
    "CheerBot: Leo cheers for @Theo! @Theo's cheer count is now 1.",
    "Maya: /poll Lunch spot? | Tacos | Pasta | Salad",
    "PollBot: Maya started a poll -- Lunch spot? [A) Tacos, B) Pasta, C) Salad]",
    "Theo: /vote B",
    "PollBot: Theo voted B (Pasta). Tally -- A: 0, B: 1, C: 0",
    "Priya: /vote B",
    "PollBot: Priya voted B (Pasta). Tally -- A: 0, B: 2, C: 0",
    "Noah: /vote c",
    "PollBot: Noah voted C (Salad). Tally -- A: 0, B: 2, C: 1",
    "Sana: /vote Z",
    "Leo: /focus abc",
    "Maya: /cheer bob",
    "Theo: /poll Coffee? | OnlyOneOption",
    "Priya: /vote B",
    "PollBot: Priya voted B (Pasta). Tally -- A: 0, B: 3, C: 1",
]
assert messages == EXPECTED_LOG
```

### Part 1 — Reading the legacy code

Answer the following, each grounded in specific lines of the code above.

- Trace the call for `("Leo", "/cheer @Theo", 561)` in the script above. Which blocks of `handle_message` run, in what order, and why does the FocusBot line appear before the CheerBot line even though the message is a cheer command, not a focus one?
- Give a concrete scenario where calling `handle_message` twice — from two separate test functions, in the same process — makes the second test's result depend on what the first one did, and name the line(s) responsible.
- Suppose you add a fourth bot, `PingBot`, triggered by `/ping`. List every place in the file above you would have to touch, and explain what (if anything) stops two bots from silently claiming the same command prefix.
- Name two pieces of behavior that cannot be unit-tested without exercising all of `handle_message`, and explain why. What happens to the blocks after `# CheerBot logic` if the loop above it (lines 15-18) were to raise an exception partway through?

### Part 2 — A bot interface

Replace the single function with one class per bot behind a shared interface, and a room that dispatches an incoming message to whichever registered bots want to see it. State that used to live in a module-level dict — the cheer counts, the focus end times, the poll in progress — must instead be handed to each bot's constructor, and no bot may read the wall clock: one that needs `now` is given a clock object that the room advances.

```py
class ChatBot(ABC):
    def can_handle(self, sender: str, text: str) -> bool:
        """Whether this bot has anything to say about this message."""

    def handle(self, sender: str, text: str) -> list[str]:
        """The lines this bot appends to the log for this message."""


class Clock:
    """now: minutes since midnight of the first day, never decreasing. ChatRoom.send sets it; bots only read it."""

    def __init__(self, now: int = 0): ...


class ChatRoom:
    def __init__(self, clock: Clock): ...
    def register(self, bot: "ChatBot") -> None: ...
    def send(self, sender: str, text: str, now: int) -> list[str]:
        """Advances clock.now, appends "sender: text", dispatches to every registered bot, returns the log."""


class CheerBot(ChatBot):
    def __init__(self, cheer_counts: dict[str, int]): ...


class FocusBot(ChatBot):
    def __init__(self, clock: Clock, focus_until: dict[str, int]): ...


class PollBot(ChatBot):
    def __init__(self, poll_state: dict): ...
```

Implement these classes so that, for any sequence of `(sender, text, now)` triples, a `ChatRoom` with the three bots registered reproduces `handle_message`'s log line for line. There is no test suite to catch a mismatch: check this yourself by running `SCRIPT` through both `handle_message` and your `ChatRoom`, and diffing the two logs line by line, rather than reading the two code paths side by side and trusting they agree.

### Part 3 — Cross-bot events

`PollBot` and `CheerBot` must react to each other only by publishing and subscribing to events on a shared bus — neither may call the other's methods or hold a reference to it. New rule: once any option's vote count reaches 3, that poll ends immediately. `PollBot` must not accept any further `/vote` for it (a later `/vote` behaves exactly as if no poll were in progress), and `CheerBot` must add 1 to the cheer count of the poll's author and announce it, appended right after `PollBot`'s own line for that vote:

```text
CheerBot: {author}'s poll reached 3 votes! {author}'s cheer count is now {count}.
```

```py
class Event:
    def __init__(self, type: str, data: dict): ...


class EventBus:
    def subscribe(self, event_type: str, handler) -> None:
        """handler(event) is called for every published event of this type, in subscription order."""

    def publish(self, event: "Event") -> None: ...
```

In the script above, the last message, `Priya: /vote B` at minute 612, brings B's tally to 3. With the new rule, the log is `EXPECTED_LOG` followed by one more line:

```text
CheerBot: Maya's poll reached 3 votes! Maya's cheer count is now 1.
```

### Part 4 — Testing

Using the standard library's `unittest` (plain functions with `assert` work too):

- Write unit tests for at least two of the three bots, each constructing only that bot's own dependencies (an explicit `Clock`, an explicit `dict`) without going through `ChatRoom`, covering at least one core rule and at least one input the bot is specified to ignore.
- Write one end-to-end test that builds a fresh `ChatRoom` with the three bots, replays `SCRIPT`, and asserts the resulting log equals `EXPECTED_LOG`.
