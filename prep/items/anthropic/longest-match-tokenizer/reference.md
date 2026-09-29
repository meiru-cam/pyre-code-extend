Two points are worth confirming before coding: whether ties for the longest match are possible (they are not — two keys of equal length matching at the same position would have to be identical strings, hence the same dict key), and whether `unk_id` may coincide with a value already in `vocab` (assumed here not to, exactly so a returned id is never ambiguous about its source).

### Part 1

At each position, try every candidate length from the number of characters remaining down to `1`, and take the first one whose slice of `text` is a key of `vocab` — trying longest first means the first hit is automatically the longest. A `for`/`else` expresses this directly: `else` runs only when the loop finishes without `break`, i.e. when nothing matched.

```python
def tokenize(text: str, vocab: dict[str, int], unk_id: int) -> list[int]:
    ids = []
    i, n = 0, len(text)
    while i < n:
        for length in range(n - i, 0, -1):        # longest possible down to a single character
            piece = text[i:i + length]
            if piece in vocab:
                ids.append(vocab[piece])
                i += length
                break
        else:
            ids.append(unk_id)                     # NOTE: no key matched, not even text[i] alone
            i += 1
    return ids
```

The position advances by at least one on every iteration of the `while` loop, so it runs at most $n$ times; each one tries up to $n - i$ candidate lengths, for $O(n^2)$ time in the worst case — a long `text` that matches nothing in `vocab` at all.

### Part 2

Computing `max_len` once per call costs $O(k)$ for $k$ keys in `vocab`; after that, capping the candidate lengths at `max_len` instead of $n - i$ bounds the inner loop by $L$. Merging is done in the same pass: an `unk_id` is appended only if the previous id appended was not already `unk_id`, so a run never grows past length `1` in the output.

```python
def tokenize_merged(text: str, vocab: dict[str, int], unk_id: int) -> list[int]:
    max_len = max((len(token) for token in vocab), default=0)   # L: longest key, 0 if vocab is empty
    ids = []
    i, n = 0, len(text)
    while i < n:
        for length in range(min(max_len, n - i), 0, -1):        # NOTE: bounded by L, not by n - i
            piece = text[i:i + length]
            if piece in vocab:
                ids.append(vocab[piece])
                i += length
                break
        else:
            if not (ids and ids[-1] == unk_id):                 # NOTE: merge into the previous UNK
                ids.append(unk_id)
            i += 1
    return ids
```

The inner loop now tries at most $L$ candidate lengths per position, $O(n \cdot L)$ dictionary probes over the whole run. Each probe costs more than $O(1)$, though: `text[i:i + length]` copies `length` characters before `vocab` can even hash the result, so a probe of length $\ell$ costs $O(\ell)$, and the probes at one position sum to $O(L^2)$ — $O(n \cdot L^2)$ time overall in the worst case, not $O(n \cdot L)$. Part 3 removes that per-probe cost.

Greedy longest match does not minimize the number of tokens produced. With `vocab = {"aaaa": 1, "aaa": 2, "a": 3}` and `text = "aaaaaa"` (six copies of `"a"`), the greedy rule matches `"aaaa"` first — the longest key that matches at position `0` — leaving two characters that only `"a"` can match, one at a time: `tokenize("aaaaaa", vocab, unk_id)` is `[1, 3, 3]`, three tokens, and `tokenize_merged` agrees, since none of them is `unk_id`. Matching `"aaa"` twice instead covers the same six characters in two tokens, `[2, 2]` — fewer than greedy's three. (The same shape of counterexample defeats greedy coin change: with denominations `1`, `3` and `4`, greedy change for `6` also takes `4 + 1 + 1` instead of the optimal `3 + 3`.)

### Part 3

Each trie node needs only its children, keyed by the one character that reaches them, and an optional id marking a node where some key of `vocab` ends. Building the trie walks every key once, character by character, creating a child node only where one does not already exist: $O(V)$ time, where $V$ is the total length of every key in `vocab`. Matching at a position walks down from the root while the next character has a child, and separately remembers the id of the last node visited that was itself the end of a key — the longest match found so far along that walk, updated every time the walk passes through such a node, exactly as `tokenize_merged` updates its own longest match while trying shorter and shorter candidate lengths.

```python
class _TrieNode:
    __slots__ = ("children", "token_id")

    def __init__(self):
        self.children: dict[str, "_TrieNode"] = {}
        self.token_id: int | None = None    # None until some key of vocab ends exactly here


class TrieTokenizer:
    def __init__(self, vocab: dict[str, int], unk_id: int):
        self._unk_id = unk_id
        self._root = _TrieNode()
        for token, token_id in vocab.items():
            node = self._root
            for ch in token:
                node = node.children.setdefault(ch, _TrieNode())
            node.token_id = token_id

    def tokenize(self, text: str) -> list[int]:
        ids = []
        i, n = 0, len(text)
        while i < n:
            node, j, best = self._root, i, None    # best: (end index, id) of the last terminal node
            while j < n and text[j] in node.children:
                node = node.children[text[j]]
                j += 1
                if node.token_id is not None:
                    best = (j, node.token_id)
            if best is None:
                if not (ids and ids[-1] == self._unk_id):
                    ids.append(self._unk_id)
                i += 1
            else:
                end, token_id = best
                ids.append(token_id)
                i = end
        return ids
```

`tokenize` walks at most $L$ trie edges from each of the $n$ starting positions before it must stop — either `text` runs out or no child continues the walk. Every step is one dictionary lookup keyed by a single character, an $O(1)$ operation, rather than the $O(\ell)$ it costs to slice and hash a fresh length-$\ell$ candidate the way `tokenize_merged` does for each length it tries: that turns `tokenize_merged`'s $O(n \cdot L^2)$ into a genuine $O(n \cdot L)$ per call, not merely a smaller constant factor. What the trie buys beyond that is amortizing the $O(V)$ build cost of `__init__` across however many texts are tokenized with the same `vocab`, which the checks below measure directly.

### Follow-ups

- The bounded loop's $O(n \cdot L^2)$ against the trie's $O(n \cdot L)$ already favors the trie once tokens run more than a character or two long; on top of that, the trie's one-time $O(V)$ build cost is earned back once `vocab` is reused across many calls. Only a short-lived call against a small, short-token vocabulary does about as well with no preprocessing at all.
- `detokenize(tokens, vocab)` needs `vocab`'s tokens reversed by id, built once as `{v: k for k, v in vocab.items()}`; an id equal to `unk_id` has nothing to reverse to, since the character it replaced is gone. A lossless round trip has to keep the character span `(start, end)` that produced each `unk_id`, alongside the id, not the id alone.
- Fed bytes one at a time instead of the whole `text` up front, the tokenizer cannot commit to a match the instant one is found: with `vocab = {"ab": 1, "abc": 2}`, seeing `"ab"` so far must wait rather than emit `1`, since the next byte could still extend it to `"abc"`. The trie walk from Part 3 carries over directly — keep the current node alive between calls instead of restarting it at the root, and commit only once a byte arrives with no matching child, or input ends.
- A streamed completion is priced separately for input and output tokens, `price_in * n_input + price_out * n_output`. `n_output` is the count of tokens actually streamed back, summed chunk by chunk as they arrive (a provider's streaming API reports usage incrementally or in one final event); re-tokenizing the whole response so far on every chunk to recompute `n_output` costs quadratic time over the length of the stream, instead of linear.
- A bug-hunt variant pairs `tokenize`/`detokenize` functions whose `tokenize` grows a candidate one character at a time and commits to it — appending its id and resetting — the moment it first matches `vocab`, rather than extending toward the longest match. On `vocab = {"a": 1, "ab": 2}`, `text = "ab"` it commits to `"a"` at the very first character, never tries extending to `"ab"`, and, having no fallback for a candidate that never matches before the text runs out, fails on the trailing `"b"`. The fix is Part 1's rule: keep extending while a longer match is still possible, and commit only once no longer one is.

```python
import random
import time


def reference_tokenize(text, vocab, unk_id):
    """Independent restatement of Part 1's rule: at each position, try every possible
    match length from longest to shortest and take the first (hence longest) key of
    vocab that matches."""
    ids = []
    i, n = 0, len(text)
    while i < n:
        match = None
        for length in range(n - i, 0, -1):
            piece = text[i:i + length]
            if piece in vocab:
                match = (length, vocab[piece])
                break
        if match is None:
            ids.append(unk_id)
            i += 1
        else:
            length, token_id = match
            ids.append(token_id)
            i += length
    return ids


def merge_unk_runs(ids, unk_id):
    merged = []
    for token_id in ids:
        if token_id == unk_id and merged and merged[-1] == unk_id:
            continue
        merged.append(token_id)
    return merged


# --- the examples of the statement ---
vocab_ex = {"a": 1, "ab": 2, "abc": 3, "b": 4, "bcd": 5}
assert tokenize("abcxbcdy", vocab_ex, 0) == [3, 0, 5, 0]
assert tokenize("abczzbcd", vocab_ex, 0) == [3, 0, 0, 5]
assert tokenize_merged("abczzbcd", vocab_ex, 0) == [3, 0, 5]
tk_ex = TrieTokenizer(vocab_ex, 0)
assert tk_ex.tokenize("abcxbcdy") == [3, 0, 5, 0]
assert tk_ex.tokenize("abczzbcd") == [3, 0, 5]

# --- the token-count counterexample ---
vocab_cx = {"aaaa": 1, "aaa": 2, "a": 3}
assert tokenize("aaaaaa", vocab_cx, -1) == [1, 3, 3]
assert tokenize_merged("aaaaaa", vocab_cx, -1) == [1, 3, 3]

# --- edge cases the examples do not reach: empty text, and a vocab with no usable entries ---
assert tokenize("", vocab_ex, 0) == []
assert tokenize_merged("", vocab_ex, 0) == []
assert TrieTokenizer(vocab_ex, 0).tokenize("") == []
assert tokenize("xyz", {}, -1) == [-1, -1, -1]
assert tokenize_merged("xyz", {}, -1) == [-1]
assert TrieTokenizer({}, -1).tokenize("xyz") == [-1]


def random_vocab(rng, n_words, alphabet, max_len):
    words = set()
    while len(words) < n_words:
        words.add("".join(rng.choice(alphabet) for _ in range(rng.randint(1, max_len))))
    long_words = sorted(w for w in words if len(w) >= 2)     # NOTE: sorted -- order must not depend on hashing
    if long_words:
        base = rng.choice(long_words)                        # force a prefix-of-each-other case
        words.add(base[:rng.randint(1, len(base) - 1)])
    return {word: idx for idx, word in enumerate(sorted(words))}    # NOTE: sorted -- deterministic ids


def random_text(rng, vocab, alphabet, target_len):
    words = list(vocab)                # dict order is insertion order (from sorted words), not hash order
    out, total = [], 0
    while total < target_len:
        piece = rng.choice(words) if words and rng.random() < 0.6 else rng.choice(alphabet)
        out.append(piece)
        total += len(piece)
    return "".join(out)


saw_multi_unk_merge = False
saw_repeated_real_id = False   # a repeated id that is NOT unk_id -- must NOT be merged
for seed in range(300):
    rng = random.Random(seed)
    alphabet = "abc" if seed % 2 else "abcd"    # small alphabets make UNK runs and prefix collisions frequent
    vocab = random_vocab(rng, rng.randint(1, 8), alphabet, 4)
    unk_id = -1
    for _ in range(3):
        text = random_text(rng, vocab, alphabet, rng.randint(0, 40))
        expected_raw = reference_tokenize(text, vocab, unk_id)
        expected_merged = merge_unk_runs(expected_raw, unk_id)
        assert tokenize(text, vocab, unk_id) == expected_raw, (seed, text, vocab)
        assert tokenize_merged(text, vocab, unk_id) == expected_merged, (seed, text, vocab)
        assert TrieTokenizer(vocab, unk_id).tokenize(text) == expected_merged, (seed, text, vocab)
        if len(expected_raw) != len(expected_merged):
            saw_multi_unk_merge = True
        if any(a == b != unk_id for a, b in zip(expected_merged, expected_merged[1:])):
            saw_repeated_real_id = True
assert saw_multi_unk_merge and saw_repeated_real_id   # both rules were actually exercised, not just stated

# --- Part 3: one TrieTokenizer reused across many texts beats rebuilding it for every text ---
rng = random.Random(12345)
big_vocab = random_vocab(rng, 5000, "abcdefgh", 7)
texts = [random_text(rng, big_vocab, "abcdefgh", 300) for _ in range(80)]

start = time.perf_counter()
reused = TrieTokenizer(big_vocab, -1)
for text in texts:
    reused.tokenize(text)
reused_seconds = time.perf_counter() - start

start = time.perf_counter()
for text in texts:
    TrieTokenizer(big_vocab, -1).tokenize(text)      # rebuilds the trie before every single text
rebuilt_seconds = time.perf_counter() - start

assert rebuilt_seconds > 8 * reused_seconds, (rebuilt_seconds, reused_seconds)   # measured: 40x or more

print("all checks passed")
```
