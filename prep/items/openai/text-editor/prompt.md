Build up a text editor's backend in four stages, each adding methods to the same `TextBuffer` class. Positions are always character offsets into the buffer, counted from 0; a range `[start, end)` is half-open (`start` included, `end` excluded), following the usual slicing convention.

### Part 1 — Buffer: insert, delete, read

```py
class TextBuffer:
    def __init__(self, text: str = "") -> None:
        """Starts with these initial contents."""

    def __len__(self) -> int:
        """The current length, in characters."""

    def text(self) -> str:
        """The entire contents, equivalent to read(0, len(self))."""

    def insert(self, pos: int, text: str) -> None:
        """Inserts text so its first character lands at offset pos, shifting every character
        currently at or after pos to the right by len(text). Requires 0 <= pos <= len(self);
        raises IndexError otherwise."""

    def delete(self, start: int, end: int) -> str:
        """Removes the half-open range [start, end) and returns the removed text ("" if start == end).
        Requires 0 <= start <= end <= len(self); raises IndexError otherwise."""

    def read(self, start: int, end: int) -> str:
        """Returns [start, end) without modifying the buffer. Same bounds as delete."""
```

For example:

```text
buf = TextBuffer("draft notes")
len(buf)                      # 11
buf.insert(5, " short")
buf.text()                    # "draft short notes"
buf.read(0, 5)                # "draft"
buf.read(5, 11)               # " short"
buf.delete(5, 11)             # " short"
buf.text()                    # "draft notes"

buf.insert(100, "x")          # IndexError
buf.delete(3, 1)              # IndexError -- start > end
```

### Part 2 — Undo and redo

`TextBuffer` grows `undo` and `redo`, backed by two stacks. Each call to `insert` or `delete` that actually changes the buffer (a non-empty `text`, or a non-empty `[start, end)`) is one undoable operation; a call that changes nothing is not recorded and does not disturb either stack. `undo` reverses the most recently recorded operation and returns `True`, or leaves the buffer alone and returns `False` if there is nothing left to undo. `redo` re-applies the most recently undone operation and returns `True`, or returns `False` if there is nothing to redo. Recording a new operation (a real `insert` or `delete`, never `undo` or `redo` itself) clears the redo stack.

```py
def undo(self) -> bool:
    """Reverses the most recent recorded insert/delete. Returns whether there was one to reverse."""

def redo(self) -> bool:
    """Re-applies the most recently undone operation. Returns whether there was one to redo."""
```

For example:

```text
buf = TextBuffer("cat")
buf.insert(3, "s")            # "cats"
buf.insert(0, "wild")         # "wildcats"
buf.undo()                    # True  -> "cats"
buf.undo()                    # True  -> "cat"
buf.redo()                    # True  -> "cats"
buf.insert(4, "!")            # "cats!" -- a new edit, so the pending "wild" redo is discarded
buf.redo()                    # False -- nothing left to redo
buf.undo(); buf.undo()        # -> "cats" -> "cat"
buf.undo()                    # False -- nothing left to undo
```

### Part 3 — Autocomplete

`TextBuffer` grows `suggest`, backed by a prefix tree (trie) with a frequency count at each word. A *word* is a maximal run of the letters `A`–`Z` and `a`–`z` (case-sensitive, so `"The"` and `"the"` are different words). The vocabulary is exactly the words that currently appear in the buffer's contents: a word's frequency is how many times it currently occurs, and deleting its only occurrence removes it from the vocabulary. `suggest(prefix, k)` returns up to `k` words that start with `prefix`, ordered by frequency (most frequent first); words tied on frequency are ordered by comparing them as strings, uppercase before lowercase (`"Zoo"` before `"apple"`). `prefix = ""` matches every word; if fewer than `k` words match, all of them are returned. `k = 0` returns `[]`, and `k < 0` raises `ValueError`.

```py
def suggest(self, prefix: str, k: int) -> list[str]:
    """Up to k words of the buffer's current vocabulary that start with prefix, highest frequency
    first, ties broken by string order (uppercase before lowercase)."""
```

For example:

```text
buf = TextBuffer("")
buf.insert(0, "the cat sat on the mat")
buf.suggest("ca", 5)          # ["cat"]
buf.suggest("t", 5)           # ["the"]
buf.suggest("", 3)            # ["the", "cat", "mat"]  -- "the" occurs twice; the four words that
                              #   occur once ("cat", "mat", "on", "sat") follow in string order
buf.delete(4, 8)              # removes "cat ", the only occurrence of "cat"
buf.suggest("ca", 5)          # []
```

### Part 4 — Concurrent editing

Several replicas edit the same document at once, each identified by a unique `site_id`. A replica exposes:

```py
class CRDTDoc:
    def __init__(self, site_id: str) -> None:
        """A single replica, starting from an empty document."""

    def text(self) -> str:
        """This replica's current visible document."""

    def local_insert(self, index: int, ch: str) -> object:
        """Inserts one character (len(ch) == 1) at visible index (0 <= index <= len(text())) as a
        local edit, and returns an operation to broadcast to every other replica."""

    def local_delete(self, index: int) -> object:
        """Deletes the visible character at index (0 <= index < len(text())) as a local edit, and
        returns an operation to broadcast."""

    def apply(self, op: object) -> None:
        """Applies an operation returned by local_insert/local_delete on some replica (including
        this one). apply is idempotent: applying the same operation twice has no further effect."""
```

A multi-character edit is just several `local_insert` calls. An operation returned by one replica is eventually delivered, via `apply`, to every other replica; delivery from one specific replica to another preserves that replica's own generation order (as a reliable point-to-point connection would). Nothing else is ordered: operations issued on different replicas, and the receiving replica's own local edits, may interleave arbitrarily, so a deletion can reach a replica that has not yet seen the character it deletes, and the same operation may be delivered more than once. Design the representation of an operation, a character and its position however you like. The one requirement: once two replicas have applied the same set of operations, `text()` must agree on both, however the deliveries were interleaved along the way.

```text
a, b = CRDTDoc("A"), CRDTDoc("B")     # both start empty
op1 = a.local_insert(0, "H")          # replica A, locally: "H" -- b has not seen this yet
op2 = b.local_insert(0, "i")          # replica B, locally: "i" -- concurrently, a has not seen this
b.apply(op1)                          # the two operations arrive in whichever order
a.apply(op2)
assert a.text() == b.text()           # both must end up with the same document
```
