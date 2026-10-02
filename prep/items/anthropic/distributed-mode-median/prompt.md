A cluster of $P = 10$ ranks, numbered $0$ through $9$, together hold a dataset of integers. Rank $r$'s *shard* is a list `shard` of zero or more integers, possibly with duplicates; shards may differ in length, and the full dataset is the concatenation of shard 0, shard 1, ..., shard 9 in that order. At least one shard is non-empty. Every rank knows $P = 10$ and its own `rank` in advance, and every rank runs the same function once, called with its own `rank`, its own `shard`, and three primitives, each already specialized to that rank so that no call needs to say who is making it:

- `send(dst: int, obj) -> None` ships `obj` to rank `dst` and returns immediately; it does not wait for `dst` to receive it.
- `recv(src: int) -> object` blocks until a value sent by rank `src` is available, then returns it. Two objects sent by the same `src` to the same rank arrive in the order `send` was called.
- `barrier() -> None` blocks the calling rank until all 10 ranks have called `barrier()`, then releases all of them together.

The only cost this exercise counts is network traffic: a call `send(dst, obj)` costs `len(pickle.dumps(obj))` bytes, and the total across every call any rank makes is what a solution is judged on — either as a whole, or, where stated below, as the largest amount any single rank sends or receives. Reading or scanning the local `shard`, sorting it, building a dictionary over it, or any other amount of local computation, costs nothing under this model, however large the shard is; only what crosses the network between ranks is counted.

Only rank 0's return value is examined; every other rank's return value is ignored, but every rank must still return rather than block forever. The *mode* of the dataset is the value with the highest count, summed across all ten shards; if several values tie for the highest count, the mode is the smallest of them. Writing $N$ for the total number of elements across all ten shards and $k = \lfloor (N-1)/2 \rfloor$, the *median* is the element at (zero-based) index $k$ of the dataset sorted in non-decreasing order — the usual middle element when $N$ is odd, and the smaller of the two middle elements (the *lower median*) when $N$ is even.

### Part 1 — Global mode

Implement `global_mode`. Let $D$ be the total number of (rank, distinct value) pairs across all ten shards — formally $D = \sum_{r=0}^{9} d_r$, where $d_r$ is the number of distinct values in shard $r$, counting a value once for every shard it appears in. A correct solution must additionally ensure that no single rank receives more than $O(D/P + P)$ bytes over the whole call.

```py
def global_mode(rank: int, shard: list[int], send, recv, barrier) -> int | None:
    """Every rank calls this once. After every rank has returned, rank 0's return value is the global
    mode of the ten shards combined; every other rank's return value is ignored."""
```

```text
rank 0: [4, 1, 4]
rank 1: [2, 4, 2]
rank 2: [1, 1]
ranks 3-9: []
# merged: [4, 1, 4, 2, 4, 2, 1, 1] -> counts {4: 3, 1: 3, 2: 2}
# 4 and 1 tie at count 3; the tie-break keeps the smaller value
global_mode(...) -> 1   # on rank 0
```

### Part 2 — Global median

Implement `global_median`. Let $R$ be one plus the difference between the largest and the smallest value across all ten shards. A correct solution must additionally ensure that the total number of bytes sent, summed over every rank, is $O\big(P \cdot (\log_2 R + 1)\big)$ — in particular, it must not grow with $N$.

```py
def global_median(rank: int, shard: list[int], send, recv, barrier) -> int | None:
    """Same calling convention as global_mode. After every rank has returned, rank 0's return value is
    the median as defined above; every other rank's return value is ignored."""
```

```text
rank 0: [10, 1]
rank 1: [7, 3]
rank 2: [15, 6]
rank 3: [8, 2]
ranks 4-9: []
# merged, sorted: [1, 2, 3, 6, 7, 8, 10, 15]   (N = 8, even)
# k = (8 - 1) // 2 = 3 (0-indexed) -> the element at index 3
global_median(...) -> 6   # the lower of the two middle values, 6 and 7
```

### Part 3 — Distributed sort

Implement `sample_sort`. After every rank has returned, write `result[r]` for rank $r$'s return value. Concatenating `result[0], result[1], ..., result[9]` in rank order must reproduce every element of every shard exactly once, in non-decreasing order — equivalently, `result[r]` is itself sorted, and every element of `result[r]` is at most every element of `result[r+1]`. Beyond this correctness requirement, on a dataset whose shard sizes and values are not overwhelmingly skewed, no single rank's send/recv traffic should exceed a small constant multiple of $N/P$ bytes, where $N$ is the total element count across all ten shards; a dataset dominated by one oversized shard or one repeated value may leave the traffic, or some `result[r]`, uneven, and is exempt from this bandwidth requirement, though never from the correctness requirement.

```py
def sample_sort(rank: int, shard: list[int], send, recv, barrier) -> list[int]:
    """Every rank calls this once. After every rank has returned, rank r's return value is result[r]
    as defined above."""
```

```text
rank 0: [5, 1]
rank 1: [9, 2]
rank 2: [4]
ranks 3-9: []
# merged, sorted: [1, 2, 4, 5, 9]
# one legal result: result[0] = [1, 2], result[1] = [4, 5], result[2] = [9], result[3..9] = []
# a different split of the same 5 values into 10 sorted, contiguous groups is equally legal
```
