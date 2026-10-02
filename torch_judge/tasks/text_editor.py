"""A text buffer that grows undo and redo, autocomplete, and replicas that converge."""

from ._interview import interview

# A slow model written straight from the statement: string slicing, whole-buffer snapshots, a scan.
_NAIVE = r"""
import copy, random, re
from collections import Counter

class Naive:
    def __init__(self, text=""):
        self.text, self.undos, self.redos = text, [], []
    def insert(self, pos, text):
        if not 0 <= pos <= len(self.text):
            raise IndexError(pos)
        if text:
            self.undos.append(self.text)
            self.redos.clear()
        self.text = self.text[:pos] + text + self.text[pos:]
    def delete(self, start, end):
        if not 0 <= start <= end <= len(self.text):
            raise IndexError((start, end))
        removed = self.text[start:end]
        if removed:
            self.undos.append(self.text)
            self.redos.clear()
        self.text = self.text[:start] + self.text[end:]
        return removed
    def read(self, start, end):
        if not 0 <= start <= end <= len(self.text):
            raise IndexError((start, end))
        return self.text[start:end]
    def undo(self):
        if not self.undos:
            return False
        self.redos.append(self.text)
        self.text = self.undos.pop()
        return True
    def redo(self):
        if not self.redos:
            return False
        self.undos.append(self.text)
        self.text = self.redos.pop()
        return True
    def suggest(self, prefix, k):
        counts = Counter(re.findall(r"[A-Za-z]+", self.text))
        return sorted((w for w in counts if w.startswith(prefix)), key=lambda w: (-counts[w], w))[:k]

def raises(error, make):
    try:
        make()
    except error:
        return True
    return False

def random_session(seed, ops, alphabet="abAB ", with_undo=False, with_suggest=False, initial=""):
    rng = random.Random(seed)
    buf, model = {fn}(initial), Naive(initial)
    for step in range(ops):
        n = len(model.text)
        roll = rng.random()
        if roll < 0.35 or n == 0:
            pos = rng.randint(0, n)
            text = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 4))) if n < 14 else ""
            buf.insert(pos, text)
            model.insert(pos, text)
        elif roll < 0.55:
            start = rng.randint(0, n)
            end = rng.randint(start, n)
            assert buf.delete(start, end) == model.delete(start, end), (seed, step)
        elif roll < 0.65:
            start = rng.randint(0, n)
            end = rng.randint(start, n)
            assert buf.read(start, end) == model.read(start, end), (seed, step)
        elif with_undo and roll < 0.8:
            assert buf.undo() == model.undo(), (seed, step)
        elif with_undo and roll < 0.9:
            assert buf.redo() == model.redo(), (seed, step)
        elif with_suggest:
            prefix, k = rng.choice(["", "a", "A", "b", "ab", "Ba", "x"]), rng.randint(0, 4)
            got = buf.suggest(prefix, k)
            assert got == model.suggest(prefix, k), (seed, step, model.text, prefix, k, got)
        assert buf.text() == model.text, (seed, step)
        assert len(buf) == len(model.text), (seed, step)
    return buf, model
"""

# Replicas exchanging operations over per-pair FIFO channels that lag and repeat.
_REPLICAS = r"""
import copy, random

def replicas(n):
    return [{fn}(site_id=chr(ord("A") + i)) for i in range(n)]

def deliver(target, op):
    target.apply(copy.deepcopy(op))  # as if it crossed the network
"""

TASK = {
    "title": "Text Editor Buffer",
    "difficulty": "Medium",
    "version": 1,
    "function_name": "TextBuffer",
    "description_en": r"""Build `TextBuffer`, the back end of a text editor.

The requirement arrives in parts. Each part keeps every earlier behavior, so one `TextBuffer` class passes all parts at the end. Pass every test of the current part to reveal the next one.

**Rules for every part:**
- Positions are character offsets from `0`.
- A range `(start, end)` is half-open, as in slicing: `start` included, `end` excluded.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why this shows up in interviews:** it starts as a warm-up on slicing and bounds, and each later part adds one requirement that tests whether the first design was clean.

**Where it is used:** every editor, from a browser text field to a code editor, keeps a buffer with an edit history.

Adapted from the text editor question in Schuture/OpenAI-Interview-Notes (CC BY-NC 4.0), reworded, on one class.""",
    "parts": [
        {
            "title": "Insert, delete, read",
            "description_en": r"""**Signature:** `TextBuffer(text="")`, `len(buf) -> int`, `text() -> str`, `insert(pos, text) -> None`, `delete(start, end) -> str`, `read(start, end) -> str`

- `TextBuffer(text)` starts with `text` as its contents. `len(buf)` is the number of characters, and `text()` returns the whole contents.
- `insert(pos, text)` inserts `text` so its first character lands at offset `pos`. It needs `0 <= pos <= len(buf)`.
- `delete(start, end)` removes the range and returns the removed text, `""` if `start == end`. It needs `0 <= start <= end <= len(buf)`.
- `read(start, end)` returns the range without changing anything. Same bounds as `delete`.
- A position or range outside those bounds raises `IndexError` and changes nothing.

**Example**, `buf = TextBuffer("pull request")`:
- `len(buf)` is `12`
- `buf.insert(4, " the")`; `buf.text()` is `"pull the request"`
- `buf.read(0, 4)` is `"pull"` and `buf.read(4, 8)` is `" the"`
- `buf.delete(4, 8)` returns `" the"`; `buf.text()` is `"pull request"` again
- `buf.insert(13, "!")` (past the end) and `buf.delete(7, 2)` (start after end) raise `IndexError`""",
        },
        {
            "title": "Undo and redo",
            "description_en": r"""Keep Part 1 and add an edit history.

**Signature:** `undo() -> bool`, `redo() -> bool`

- Each `insert` or `delete` that changes the buffer is one recorded edit. An edit that changes nothing (empty `text`, or `start == end`) and one that raises are not recorded and leave the history alone.
- `undo()` reverses the most recent recorded edit not yet undone and returns `True`, or returns `False` and changes nothing if there is none.
- `redo()` re-applies the most recently undone edit and returns `True`, or returns `False` if there is none.
- A new recorded edit discards everything that could be redone. `undo` and `redo` themselves are never recorded as edits.

**Example**, `buf = TextBuffer("note")`:
- `insert(4, "s")` makes `"notes"`, then `delete(0, 1)` makes `"otes"`
- `undo()` restores `"notes"`; a second `undo()` restores `"note"`; `redo()` brings back `"notes"`. All three return `True`.
- `insert(0, "my ")` makes `"my notes"`. The undone `delete` can no longer be redone: `redo()` returns `False`.
- `undo()` makes `"notes"`, `undo()` makes `"note"`, and one more `undo()` returns `False`""",
        },
        {
            "title": "Autocomplete",
            "description_en": r"""Keep Parts 1–2 and add word suggestions.

**Signature:** `suggest(prefix, k) -> list[str]`

- A word is a maximal run of the letters `A`–`Z` and `a`–`z`. Words are case-sensitive: `"Go"` and `"go"` are two words.
- The vocabulary is the words in the buffer's current contents, and a word's frequency is how many times it occurs there now.
- `suggest(prefix, k)` returns up to `k` words that start with `prefix`, most frequent first. Words with equal frequency come in string order, so every uppercase letter sorts before every lowercase one: `"Kiwi"` before `"banana"`.
- `prefix = ""` matches every word. `k = 0` returns `[]`, and `k < 0` raises `ValueError`.

**Example**, `buf = TextBuffer("to be or not to be")`:
- `buf.suggest("b", 5)` is `["be"]`
- `buf.suggest("", 3)` is `["be", "to", "not"]`: `be` and `to` occur twice, then `not` and `or` once each
- after `buf.delete(6, 9)` removes `"or "`, `buf.suggest("o", 5)` is `[]`""",
        },
        {
            "title": "Replicas that converge",
            "description_en": r"""Keep Parts 1–3. Several replicas now edit one document at the same time.

**Signature:** `TextBuffer(site_id=...)`, `local_insert(index, ch) -> op`, `local_delete(index) -> op`, `apply(op) -> None`

- A replica is `TextBuffer(site_id=s)` with a unique string `s`. It starts empty, `text()` returns its current document, and `len(buf)` is the length of `text()`. Tests do not call `insert`, `delete`, `undo` or `redo` on a replica.
- `local_insert(index, ch)` inserts one character at `index`, `0 <= index <= len(text())`, and returns an operation for the other replicas. Afterwards `text()[index] == ch`. A `ch` that is not exactly one character raises `ValueError`.
- `local_delete(index)` deletes the character at `index`, `0 <= index < len(text())`, and returns an operation. Both raise `IndexError` for an index outside those bounds.
- `apply(op)` applies an operation returned by any replica. Applying the same operation again changes nothing.
- Each operation is delivered to every other replica. Operations from one replica to another arrive in the order they were made, but nothing else is ordered: a delete can arrive before the insert of its character, and an operation can arrive twice. Operations may be copied on the way.
- Once two replicas have applied the same operations, their `text()` is equal.
- A character typed between two characters stays between them on every replica, as long as both are still there.

**Example:**
- `north = TextBuffer(site_id="north")` and `south = TextBuffer(site_id="south")` both start empty
- `north` types `"x"` at index 0 while `south`, not yet aware of it, types `"y"` at index 0
- each replica then applies the other's operation; both now show the same two-character text, `"xy"` or `"yx"`""",
        },
    ],
    "hints": [
        {"level": 1, "kind": "questions", "content": "What are the valid positions for an insert into a buffer of length n, and how do they differ from the valid ranges for a delete? Is the check the same when start == end? Which Python sequence lets you insert or remove a run in place with a slice?"},
        {"level": 2, "kind": "analysis", "content": "Keep the contents as a list of characters. insert checks 0 <= pos <= len, then does chars[pos:pos] = text. delete and read check 0 <= start <= end <= len; delete saves chars[start:end] as a string, then del chars[start:end]. text() joins the list. Check bounds before changing anything."},
    ],
    "model_connections": [
        "Collaborative notebooks and documents, including those used to edit prompts and evaluations together, sync edits between users with CRDTs or operational transformation.",
        "Code completion ranks candidate tokens by frequency in context, from tries over the open files up to language-model scores.",
    ],
    "pro_con_analysis": {
        "pros": [
            "Recording each edit with its position and text makes undo and redo cheap and exact.",
            "A trie answers a prefix query by walking only the prefix and the matching subtree.",
            "Giving every character a fixed, unique position makes replicas converge whatever order operations arrive in.",
        ],
        "cons": [
            "A list of characters makes each insert and delete O(n); a gap buffer, rope or piece table avoids that.",
            "Rebuilding the vocabulary on every suggest costs O(n); keeping it updated on each edit is faster but harder.",
            "A CRDT keeps deleted characters and ids forever unless replicas agree on what can be collected.",
        ],
    },
    "tests": [
        {"name": "Part 1: the worked example", "part": 1, "behavior": "state.invariant", "code": r"""
buf = {fn}("pull request")
assert len(buf) == 12
buf.insert(4, " the")
assert buf.text() == "pull the request"
assert buf.read(0, 4) == "pull" and buf.read(4, 8) == " the"
assert buf.delete(4, 8) == " the"
assert buf.text() == "pull request"
for bad in [lambda: buf.insert(13, "!"), lambda: buf.delete(7, 2)]:
    try:
        bad()
        raise AssertionError("expected IndexError")
    except IndexError:
        pass
assert buf.text() == "pull request"
"""},
        {"name": "Part 1: random edits match string slicing", "part": 1, "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "After a random mix of insert, delete and read, the buffer differed from doing the same edits with string slicing.",
         "code": _NAIVE + r"""
for seed in range(300):
    random_session(seed, 50, initial=["", "seed", "x y"][seed % 3])
"""},
        {"name": "Part 1: bounds", "part": 1, "visibility": "unshown", "behavior": "edge.empty_or_boundary",
         "failure_message": "insert needs 0 <= pos <= len and delete and read need 0 <= start <= end <= len; anything else raises IndexError without changing the buffer, and empty edits are allowed.",
         "code": _NAIVE + r"""
buf = {fn}()
assert len(buf) == 0 and buf.text() == ""
assert buf.read(0, 0) == "" and buf.delete(0, 0) == ""
buf.insert(0, "")
buf.insert(0, "abc")
buf.insert(3, "d")
assert buf.text() == "abcd"
for bad in [lambda: buf.insert(-1, "x"), lambda: buf.insert(5, "x"), lambda: buf.delete(-1, 2),
            lambda: buf.delete(2, 5), lambda: buf.delete(3, 2), lambda: buf.read(0, 5),
            lambda: buf.read(-2, -1), lambda: buf.read(3, 2)]:
    assert raises(IndexError, bad)
    assert buf.text() == "abcd"
assert buf.read(4, 4) == "" and buf.read(0, 4) == "abcd"
assert buf.delete(0, 4) == "abcd" and buf.text() == "" and len(buf) == 0
"""},
        {"name": "Part 2: the worked example", "part": 2, "behavior": "state.invariant", "code": r"""
buf = {fn}("note")
buf.insert(4, "s")
assert buf.delete(0, 1) == "n" and buf.text() == "otes"
assert buf.undo() is True and buf.text() == "notes"
assert buf.undo() is True and buf.text() == "note"
assert buf.redo() is True and buf.text() == "notes"
buf.insert(0, "my ")
assert buf.text() == "my notes"
assert buf.redo() is False and buf.text() == "my notes"
assert buf.undo() and buf.text() == "notes"
assert buf.undo() and buf.text() == "note"
assert buf.undo() is False and buf.text() == "note"
"""},
        {"name": "Part 2: random histories match snapshots", "part": 2, "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "After a random mix of edits, undos and redos, the buffer or a return value differed from keeping a snapshot of the whole text per recorded edit.",
         "code": _NAIVE + r"""
for seed in range(400):
    buf, model = random_session(seed, 60, with_undo=True)
    while True:
        done = buf.undo()
        assert done == model.undo(), seed
        assert buf.text() == model.text, seed
        if not done:
            break
"""},
        {"name": "Part 2: what is not recorded", "part": 2, "visibility": "unshown", "behavior": "edge.empty_or_boundary",
         "failure_message": "Edits that change nothing or raise must not be recorded or discard the redo history, and undoing a delete must put the text back where it was.",
         "code": _NAIVE + r"""
buf = {fn}("start")
assert buf.undo() is False and buf.redo() is False, "the initial text is not an edit"
buf.delete(1, 3)
assert buf.text() == "srt"
buf.insert(2, "")
buf.delete(1, 1)
assert raises(IndexError, lambda: buf.insert(9, "x"))
assert buf.undo() is True and buf.text() == "start", "the empty edits and the failed one were recorded"
buf.insert(0, "")
assert raises(IndexError, lambda: buf.delete(4, 99))
assert buf.redo() is True and buf.text() == "srt", "an empty or failed edit discarded the redo"
assert buf.undo() is True and buf.undo() is False and buf.text() == "start"
assert buf.redo() and buf.redo() is False
"""},
        {"name": "Part 3: the worked example", "part": 3, "behavior": "state.invariant", "code": r"""
buf = {fn}("to be or not to be")
assert buf.suggest("b", 5) == ["be"]
assert buf.suggest("", 3) == ["be", "to", "not"]
assert buf.delete(6, 9) == "or "
assert buf.suggest("o", 5) == []
"""},
        {"name": "Part 3: random edits match a word count", "part": 3, "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "After random edits, undos and redos, a suggestion differed from counting the words in the current text and sorting by frequency, then string order.",
         "code": _NAIVE + r"""
for seed in range(300):
    random_session(seed, 60, with_undo=True, with_suggest=True)
"""},
        {"name": "Part 3: what counts as a word", "part": 3, "visibility": "unshown", "behavior": "edge.empty_or_boundary",
         "failure_message": "Words are runs of A-Z and a-z only, case-sensitive; ties sort uppercase first; k = 0 gives [] and k < 0 raises ValueError.",
         "code": _NAIVE + r"""
buf = {fn}("pear Kiwi pear Kiwi fig")
assert buf.suggest("", 3) == ["Kiwi", "pear", "fig"]
assert buf.suggest("k", 5) == [] and buf.suggest("K", 5) == ["Kiwi"]
assert buf.suggest("", 0) == []
assert raises(ValueError, lambda: buf.suggest("", -1))
buf = {fn}("abc1def snake_case café x-ray A\tb")
assert buf.suggest("", 20) == ["A", "abc", "b", "caf", "case", "def", "ray", "snake", "x"]
assert buf.suggest("abc1", 5) == [] and buf.suggest("nope", 5) == []
buf = {fn}("go go go Go")
buf.insert(len(buf), " Go Go Go")
assert buf.suggest("", 2) == ["Go", "go"]
buf.undo()
assert buf.suggest("G", 2) == ["Go"] and buf.suggest("", 1) == ["go"]
"""},
        {"name": "Part 4: the worked example", "part": 4, "behavior": "events.ordering", "code": r"""
north, south = {fn}(site_id="north"), {fn}(site_id="south")
assert north.text() == "" and south.text() == ""
from_north = north.local_insert(0, "x")
from_south = south.local_insert(0, "y")
assert north.text() == "x" and south.text() == "y"
south.apply(from_north)
north.apply(from_south)
assert north.text() == south.text() and north.text() in ("xy", "yx")
"""},
        {"name": "Part 4: late deletes, repeats and neighbours", "part": 4, "visibility": "unshown", "behavior": "effects.idempotency",
         "failure_message": "A delete arriving before its insert, an operation applied twice, or an insert next to a concurrently deleted character left the replicas different or misplaced a character.",
         "code": _REPLICAS + r"""
a, b = replicas(2)
for ch in "abc":
    deliver(b, a.local_insert(len(a.text()), ch))
gone = a.local_delete(1)
typed = b.local_insert(2, "Y")
assert b.text() == "abYc"
deliver(b, gone)
deliver(a, typed)
assert a.text() == b.text() == "aYc"

a, b, c = replicas(3)
ins = a.local_insert(0, "z")
deliver(b, ins)
dele = b.local_delete(0)
deliver(c, dele)
deliver(c, ins)
deliver(a, dele)
assert a.text() == b.text() == c.text() == ""

(d,) = replicas(1)
op = d.local_insert(0, "w")
d.apply(op)
deliver(d, op)
assert d.text() == "w"
dele = d.local_delete(0)
deliver(d, dele)
deliver(d, dele)
assert d.text() == ""
for bad, error in [(lambda: d.local_insert(0, "ab"), ValueError), (lambda: d.local_insert(0, ""), ValueError),
                   (lambda: d.local_delete(0), IndexError), (lambda: d.local_insert(1, "x"), IndexError),
                   (lambda: d.local_insert(-1, "x"), IndexError)]:
    try:
        bad()
        raise AssertionError(f"expected {error.__name__}")
    except error:
        pass
assert d.text() == ""
"""},
        {"name": "Part 4: random interleavings converge", "part": 4, "visibility": "unshown", "behavior": "events.ordering",
         "failure_message": "With operations delivered late, repeated and interleaved, replicas that applied the same operations ended with different text, a local edit did not land where it was asked, or a character left the place between its neighbours.",
         "code": _REPLICAS + r"""
glyph = iter(range(0x4E00, 0x9FFF))
late_deletes = 0
for seed in range(150):
    rng = random.Random(seed)
    reps = replicas(3)
    queues = {(i, j): [] for i in range(3) for j in range(3) if i != j}
    placed = []  # (character, left neighbour or None, right neighbour or None) when typed
    seen = [set() for _ in reps]
    for round_ in range(40):
        for i, rep in enumerate(reps):
            if rng.random() < 0.5:
                continue
            before = rep.text()
            if not before or rng.random() < 0.6:
                index, ch = rng.randint(0, len(before)), chr(next(glyph))
                op = rep.local_insert(index, ch)
                assert rep.text() == before[:index] + ch + before[index:], "a local insert landed elsewhere"
                placed.append((ch, before[index - 1] if index else None, before[index] if index < len(before) else None))
                seen[i].add(ch)
                kind = ("insert", ch)
            else:
                index = rng.randrange(len(before))
                op = rep.local_delete(index)
                assert rep.text() == before[:index] + before[index + 1:], "a local delete removed another character"
                kind = ("delete", before[index])
            for j in range(3):
                if j != i:
                    queues[(i, j)].append((kind, op))
        for (i, j), queue in queues.items():
            if not queue or rng.random() < 0.5:
                continue
            k = rng.randint(1, len(queue))
            for (what, ch), op in queue[:k]:
                if what == "insert":
                    seen[j].add(ch)
                elif ch not in seen[j]:
                    late_deletes += 1
                deliver(reps[j], op)
                if rng.random() < 0.3:
                    deliver(reps[j], op)
            del queue[:k]
    for (i, j), queue in queues.items():
        for _, op in queue:
            deliver(reps[j], op)
    texts = [rep.text() for rep in reps]
    assert all(len(rep) == len(text) for rep, text in zip(reps, texts)), "len() must count a replica's visible characters"
    assert texts[0] == texts[1] == texts[2], (seed, texts)
    where = {ch: n for n, ch in enumerate(texts[0])}
    for ch, left, right in placed:
        if ch in where and left in where:
            assert where[left] < where[ch], (seed, "a character moved before its left neighbour")
        if ch in where and right in where:
            assert where[ch] < where[right], (seed, "a character moved after its right neighbour")
assert late_deletes > 50
"""},
    ],
    "solution": r'''# Adapted from Schuture/OpenAI-Interview-Notes (code under the MIT License).
import bisect
import random
import re
from collections import Counter
from fractions import Fraction

_WORD = re.compile(r"[A-Za-z]+")


class _Node:
    __slots__ = ("children", "count")

    def __init__(self):
        self.children = {}
        self.count = 0  # occurrences of the word ending here; 0 if none does


class TextBuffer:
    def __init__(self, text="", site_id=None):
        self._chars = list(text)
        self._undo, self._redo = [], []  # recorded edits: (kind, pos, text)
        # Replica state: every character gets a fixed position strictly between its neighbours.
        self.site_id = site_id
        self._counter = 0
        self._entries = []     # sorted (position, site, seq, ch)
        self._inserted = set()  # (site, seq) of inserts applied
        self._deleted = set()   # (site, seq) deleted anywhere, possibly before their insert arrives

    def __len__(self):
        return len(self._live()) if self.site_id is not None else len(self._chars)

    def text(self):
        if self.site_id is not None:
            return "".join(entry[3] for entry in self._live())
        return "".join(self._chars)

    def _check(self, start, end):
        if not 0 <= start <= end <= len(self._chars):
            raise IndexError(f"range ({start}, {end}) out of bounds for length {len(self._chars)}")

    def _insert(self, pos, text):
        self._chars[pos:pos] = text

    def _delete(self, start, end):
        removed = "".join(self._chars[start:end])
        del self._chars[start:end]
        return removed

    def insert(self, pos, text):
        self._check(pos, pos)
        self._insert(pos, text)
        if text:
            self._undo.append(("insert", pos, text))
            self._redo.clear()

    def delete(self, start, end):
        self._check(start, end)
        removed = self._delete(start, end)
        if removed:
            self._undo.append(("delete", start, removed))
            self._redo.clear()
        return removed

    def read(self, start, end):
        self._check(start, end)
        return "".join(self._chars[start:end])

    def _replay(self, edit, forward):
        kind, pos, text = edit
        if (kind == "insert") == forward:
            self._insert(pos, text)
        else:
            self._delete(pos, pos + len(text))

    def undo(self):
        if not self._undo:
            return False
        edit = self._undo.pop()
        self._replay(edit, forward=False)  # through the unrecorded helpers
        self._redo.append(edit)
        return True

    def redo(self):
        if not self._redo:
            return False
        edit = self._redo.pop()
        self._replay(edit, forward=True)
        self._undo.append(edit)
        return True

    def suggest(self, prefix, k):
        if k < 0:
            raise ValueError(f"k must be >= 0, got {k}")
        root = _Node()
        for word, count in Counter(_WORD.findall(self.text())).items():
            node = root
            for ch in word:
                node = node.children.setdefault(ch, _Node())
            node.count = count
        node = root
        for ch in prefix:
            node = node.children.get(ch)
            if node is None:
                return []
        found, stack = [], [(node, prefix)]
        while stack:
            node, word = stack.pop()
            if node.count:
                found.append((-node.count, word))
            stack.extend((child, word + ch) for ch, child in node.children.items())
        found.sort()  # most frequent first, then string order
        return [word for _, word in found[:k]]

    def _live(self):
        return [entry for entry in self._entries if entry[1:3] not in self._deleted]

    def local_insert(self, index, ch):
        if not isinstance(ch, str) or len(ch) != 1:
            raise ValueError(f"exactly one character, got {ch!r}")
        live = self._live()
        if not 0 <= index <= len(live):
            raise IndexError(f"insert index {index} out of range for length {len(live)}")
        left = live[index - 1][0] if index > 0 else Fraction(0)
        right = live[index][0] if index < len(live) else Fraction(1)
        # A random point, not the midpoint, so two replicas filling one gap rarely collide; an
        # exact collision (about 2**-32) still converges, since entries sort by (position, site, seq).
        position = left + (right - left) * Fraction(random.randint(1, 2 ** 32 - 1), 2 ** 32)
        self._counter += 1
        op = ("insert", position, self.site_id, self._counter, ch)
        self.apply(op)
        return op

    def local_delete(self, index):
        live = self._live()
        if not 0 <= index < len(live):
            raise IndexError(f"delete index {index} out of range for length {len(live)}")
        op = ("delete", live[index][1], live[index][2])
        self.apply(op)
        return op

    def apply(self, op):
        if op[0] == "insert":
            _, position, site, seq, ch = op
            if (site, seq) not in self._inserted:  # a repeated insert changes nothing
                self._inserted.add((site, seq))
                bisect.insort(self._entries, (position, site, seq, ch))
        else:
            self._deleted.add((op[1], op[2]))
''',
    "interview_questions": interview(
        concept=[
            "Why are the valid positions for insert different from the valid ranges for delete?",
            "Why is a list of characters a better buffer than a str, and what does each edit still cost?",
        ],
        deep_dive=[
            "Which structure would make inserts and deletes in the middle of a large file cheap, and what does it cost to read?",
        ],
        tradeoffs=[
            "What do you store per edit to undo it, and why does a new edit clear the redo stack?",
            "How does a trie answer a prefix query, and when would you keep it updated on each edit instead of rebuilding it?",
            "Why does sending an index such as insert at 5 not work between replicas, and what do you send instead?",
            "How can a delete arrive before the insert of its character, and how should a replica handle it?",
        ],
    ),
}
