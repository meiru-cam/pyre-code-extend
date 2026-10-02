An *example* is an opaque item drawn from one of several named datasets; nothing about its internal structure matters here. A `DataRegistry` gives access to $K$ datasets, each identified by a `str` name, through one method:

```py
class DataRegistry:
    def get_iterator(self, name: str, offset: int = 0):
        """Returns a fresh iterator over dataset `name`, opened directly at position `offset`;
        opening is fast at any offset, it never walks through the examples it skips. Writing L for
        the number of examples in `name`, the iterator yields the example at index offset % L, then
        (offset + 1) % L, then (offset + 2) % L, and so on forever: once it passes the last example
        it wraps back to the first and keeps going. It never raises StopIteration."""
```

Implement `DataBatcher(registry, weights, batch_size)`, where `weights` is a `dict[str, int]` mapping every dataset name to a positive integer weight (`weights` has at least one entry) and `batch_size` is a positive integer. Order the names alphabetically as $d_1 < d_2 < \cdots < d_K$ and write $w_k$ for $\mathrm{weights}[d_k]$ and $W = \sum_{k=1}^{K} w_k$ for the *total weight*. `next_batch()` returns a `list` of `batch_size` examples drawn from the $K$ datasets in proportion to `weights`. Every part below keeps the same rule for the order *between* datasets in a batch — ascending alphabetical order of name — and states its own rule for the order *within* one dataset's contribution.

### Part 1 — Exact weighted batches

Assume for this part that `batch_size` is divisible by $W$. For every $k$, `next_batch()` places exactly $\mathrm{batch\_size} \cdot w_k / W$ of dataset $d_k$'s examples in the batch, consecutively, in the order its own iterator produced them; the $K$ blocks are concatenated in order $d_1, d_2, \ldots, d_K$.

```py
class DataBatcher:
    def __init__(self, registry: DataRegistry, weights: dict[str, int], batch_size: int) -> None:
        """batch_size is divisible by sum(weights.values()) in this part."""

    def next_batch(self) -> list:
        """Returns batch_size examples: for each name in ascending alphabetical order,
        batch_size * weights[name] // sum(weights.values()) consecutive examples of that dataset."""
```

```text
weights = {"code": 2, "math": 1, "web": 1}      # W = 4
batch_size = 8                                    # 8 is divisible by 4

# code: 8 * 2 // 4 = 4 examples, math: 8 * 1 // 4 = 2, web: 8 * 1 // 4 = 2
next_batch() -> [code-0, code-1, code-2, code-3, math-0, math-1, web-0, web-1]
```

### Part 2 — Checkpoint and resume

Implement `state_dict()` and `load_state_dict(state)`. `state_dict()` returns a `dict` of plain `int`s and strings — small and JSON-serializable, its size fixed by $K$ and never growing with how many batches have been produced — capturing exactly enough to resume. Calling `load_state_dict` on a `DataBatcher` constructed with the same `registry`, `weights` and `batch_size` moves it to the position the state describes, so that its next `next_batch()` call returns exactly what the batcher the state came from would have returned next, whether or not the two are the same Python object. Restoring must not re-iterate over examples already produced: it uses `offset` to reopen each dataset directly where it left off.

```py
    def state_dict(self) -> dict:
        """A small, JSON-serializable dict sufficient to resume from here."""

    def load_state_dict(self, state: dict) -> None:
        """Moves this batcher to the position state describes."""
```

```text
b = DataBatcher(registry, {"code": 2, "math": 1, "web": 1}, batch_size=8)
b.next_batch()             # the batch shown in the Part 1 example
b.state_dict()              # {"offsets": {"code": 4, "math": 2, "web": 2}}
b.next_batch()              # [code-4, code-5, code-6, code-7, math-2, math-3, web-2, web-3]

# a second, independent batcher restored from the saved state continues identically:
b2 = DataBatcher(registry, {"code": 2, "math": 1, "web": 1}, batch_size=8)
b2.load_state_dict({"offsets": {"code": 4, "math": 2, "web": 2}})
b2.next_batch()              # [code-4, code-5, code-6, code-7, math-2, math-3, web-2, web-3]
```

### Part 3 — batch_size not divisible by W

Drop Part 1's divisibility assumption. `next_batch()` still returns `batch_size` examples in proportion to `weights`, but the per-dataset quota $\mathrm{batch\_size} \cdot w_k / W$ is now, in general, not an integer. `DataBatcher` gains two more constructor arguments, `allocation` and `seed`; whichever value `allocation` takes, `next_batch()` still falls back to Part 1's exact rule whenever `batch_size` *is* divisible by $W$.

Either way, when `batch_size` is not divisible by $W$, the batch is built one example at a time, `batch_size` times, each time choosing which dataset the next example comes from; the batch lists the `batch_size` examples in the order these choices were made, no longer grouped into per-dataset blocks. Write $n$ for how many such choices a batcher has made in total since it was constructed (accumulating across every `next_batch()` call so far, including calls before the current one), and $\mathrm{count}_k(n)$ for how many of the first $n$ choices picked dataset $d_k$; dataset $d_k$'s *ideal share* of the first $n$ choices is $n w_k / W$, generally not an integer.

- `allocation="deterministic"`: choose with no randomness, from the running per-dataset counts alone, such that for every $k$ and every $n$ (not only when $n$ is a multiple of `batch_size`), $\lvert \mathrm{count}_k(n) - n w_k / W \rvert \le C$ for a constant $C$ that does not depend on $n$. State your rule and prove such a $C$ for it.
- `allocation="stochastic"`: build a prefix-sum array over $w_1, \ldots, w_K$ in the same alphabetical order. Keep a `random.Random(seed)` instance on the batcher, and make each choice by drawing a uniform integer in $[0, W)$ from it and binary-searching the prefix-sum array for the dataset it lands in. `state_dict()` must then also carry enough of the generator's own state to make the draws resume exactly, not merely to have the correct distribution again.

```py
class DataBatcher:
    def __init__(self, registry: DataRegistry, weights: dict[str, int], batch_size: int,
                 allocation: str = "deterministic", seed: int = 0) -> None:
        """allocation is "deterministic" or "stochastic"; batch_size need not divide
        sum(weights.values()) any more. seed is used only when allocation == "stochastic"."""
```

```text
weights = {"code": 2, "math": 1, "web": 1}, batch_size = 6, allocation = "deterministic"
# W = 4, and 4 does not divide 6

next_batch() -> [code-0, math-0, web-0, code-1, code-2, math-1]
# final counts: code = 3 (ideal 6*2/4 = 3.0), math = 2 (ideal 1.5), web = 1 (ideal 1.5)
```
