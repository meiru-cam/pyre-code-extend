A *vocabulary* `vocab` is a `dict[str, int]` whose keys are distinct, non-empty strings (*tokens*); its values are arbitrary integer ids and need not be distinct from one another. `unk_id` is an integer supplied separately from `vocab`, never equal to one of its values, so an id returned by the functions below is unambiguous: it is either a value of `vocab` (a real match) or `unk_id` (no match), never both. `text` is an ordinary Python string, possibly containing characters that appear in no token of `vocab` at all.

### Part 1 — Greedy longest match

Tokenize `text` from left to right. At the current position, a key of `vocab` *matches* there if `text` starting at that position begins with exactly that key. If at least one key matches, take the longest one — two different keys that both matched and had the same length would have to be the same string, so the longest match, when one exists, is always unique — append its id to the output, and move the position forward by that key's length. If no key matches, not even the single character at the position, append `unk_id` to the output and move the position forward by exactly one character. Repeat until the position reaches the end of `text`; `tokenize` returns `[]` on an empty `text`.

```py
def tokenize(text: str, vocab: dict[str, int], unk_id: int) -> list[int]:
    """Greedy longest match from the left. One id per matched token, unk_id for each
    character that matches no key of vocab."""
```

Example:

```text
vocab = {"a": 1, "ab": 2, "abc": 3, "b": 4, "bcd": 5}, unk_id = 0

tokenize("abcxbcdy", vocab, 0)
# position 0: "a", "ab" and "abc" all match here; the longest is "abc" -> emit 3, move to 3
# position 3: "x" matches no key                                     -> emit 0, move to 4
# position 4: "b" and "bcd" both match here; the longest is "bcd"    -> emit 5, move to 7
# position 7: "y" matches no key                                     -> emit 0, move to 8 (= len(text))
# -> [3, 0, 5, 0]
```

### Part 2 — Bounded and merged

Let $L$ be the length of the longest key in `vocab`, or $0$ when `vocab` is empty, and write $n$ for `len(text)`. `tokenize_merged` returns the same token sequence as `tokenize`, except that a maximal run of two or more consecutive `unk_id` entries collapses to a single `unk_id`; a run of any other, repeated id — two matched tokens that happen to carry the same id back to back — is left as it is. At each position, `tokenize_merged` must try at most $L$ candidate lengths, never the full scan down from the number of characters remaining to $1$ that `tokenize` does: that full scan is too slow once `text` is long and every key of `vocab` is short.

```py
def tokenize_merged(text: str, vocab: dict[str, int], unk_id: int) -> list[int]:
    """Same token sequence as tokenize(text, vocab, unk_id), except that a maximal run of
    unk_id collapses to one unk_id. Tries at most L candidate lengths per position, L = the
    longest key of vocab, instead of tokenize's full n - i."""
```

Example:

```text
vocab = {"a": 1, "ab": 2, "abc": 3, "b": 4, "bcd": 5}, unk_id = 0

tokenize("abczzbcd", vocab, 0)         -> [3, 0, 0, 5]   # "abc", "z", "z", "bcd"
tokenize_merged("abczzbcd", vocab, 0)  -> [3, 0, 5]      # the two consecutive UNKs merge into one
```

### Part 3 — Reusable trie

`vocab` is now large, and `tokenize_merged` is called on many separate texts that all share this one `vocab` — one call per document in a batch, for example — so the cost of preparing `vocab` for matching should be paid once, not repeated on every call. Implement `TrieTokenizer`: its constructor builds a trie from `vocab` once, and `tokenize` returns exactly what `tokenize_merged(text, vocab, unk_id)` would for whichever `text` it is called with, computed by walking that trie instead of rescanning `vocab`.

```py
class TrieTokenizer:
    def __init__(self, vocab: dict[str, int], unk_id: int) -> None:
        """Builds a trie from vocab once, for reuse across many calls to tokenize."""

    def tokenize(self, text: str) -> list[int]:
        """Returns exactly what tokenize_merged(text, vocab, unk_id) would, computed by
        walking the trie built in __init__."""
```

Example:

```text
vocab = {"a": 1, "ab": 2, "abc": 3, "b": 4, "bcd": 5}, unk_id = 0
tk = TrieTokenizer(vocab, 0)

tk.tokenize("abcxbcdy")  -> [3, 0, 5, 0]   # same result as tokenize's example above
tk.tokenize("abczzbcd")  -> [3, 0, 5]      # same result as tokenize_merged's example above
```
