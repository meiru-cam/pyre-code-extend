Two things are worth confirming before coding. Whether every rank already knows $P = 10$ and its own `rank` as constants, rather than discovering them at runtime — assumed yes, exactly as the setup states. And whether the values are ordinary bounded-magnitude integers, so that a search over the value range in Part 2 is guaranteed to land exactly on a real element — assumed yes below; arbitrary floats are a follow-up. The cost model counts network bytes only, so an $O(n \log n)$ local sort of one rank's own shard is free; only what crosses between ranks is charged.

### Part 1

Sending every element to rank 0 costs $O(N)$ bytes, no better than $O(D)$ whenever there are fewer distinct values than elements — which is exactly when a shard's histogram is worth building in the first place. Sending every rank's full local histogram to rank 0 instead — one message per rank, entries `(value, count)` — costs each rank only $O(\min(n_r, V))$ bytes to send, where $V$ is the number of distinct values in the merged dataset; for a small alphabet ($V \ll N$) this is already cheap and simple, and rank 0 merely sums matching keys across the ten histograms it receives. But every one of those entries still lands on rank 0: as $V$ grows, so does rank 0's share of the traffic and the work of merging it, while every other rank sits idle.

Partitioning by value instead of by rank fixes this. Give every value a fixed *owner* — some function of the value alone, the same on every rank — and have each rank forward its local counts for a value only to that value's owner, never to a fixed coordinator. Once every rank has done this, each owner has received, from every other rank, exactly that rank's contribution to every value the owner is responsible for; summed with its own local counts, the owner now holds the *true* global count for every value it owns, without needing to hear anything about values it does not own. The owner can find its own best value immediately — a *local winner* — and only one `(value, count)` pair per owner, ten numbers in total, needs to reach rank 0 for the final comparison.

```python
NUM_RANKS = 10


def _local_histogram(shard: list[int]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for v in shard:
        counts[v] = counts.get(v, 0) + 1
    return counts


def _owner(value: int) -> int:
    return value % NUM_RANKS   # NOTE: Python's % is already non-negative here, negative values included


def global_mode(rank, shard, send, recv, barrier):
    local = _local_histogram(shard)
    outgoing = [[] for _ in range(NUM_RANKS)]
    for value, count in local.items():
        outgoing[_owner(value)].append((value, count))
    for dst in range(NUM_RANKS):
        if dst != rank:
            send(dst, outgoing[dst])    # NOTE: send even [] -- recv(src) below waits specifically for dst

    owned = dict(outgoing[rank])
    for src in range(NUM_RANKS):
        if src != rank:
            for value, count in recv(src):
                owned[value] = owned.get(value, 0) + count

    if owned:
        best_count = max(owned.values())
        winner = min(v for v, c in owned.items() if c == best_count)
        candidate = (winner, owned[winner])
    else:
        candidate = None                # NOTE: this owner's slice of the key space is empty

    if rank != 0:
        send(0, candidate)
        return None
    candidates = [candidate] + [recv(src) for src in range(1, NUM_RANKS)]
    best_value, _ = min((c for c in candidates if c is not None), key=lambda vc: (-vc[1], vc[0]))
    return best_value
```

Every `(value, count)` pair from every rank's local histogram is sent exactly once, to its one owner, so the shuffle moves $D$ pairs in total; as long as the values are not concentrated on one residue mod $P$, each owner receives close to $D/P$ of them, and the ten-candidate final round adds $O(P)$ more to every rank — the $O(D/P + P)$ bound the statement asks for. Measured at $N = 4{,}000$ elements drawn from 2,000 distinct values, the busiest rank receives 2,551 bytes under this scheme against 22,576 bytes when every rank ships its histogram straight to rank 0 instead — nearly nine times as many. At 5 distinct values, gathering raw elements instead of histograms costs 7,344 bytes against 414, nearly eighteen times as many, confirming that compressing to counts is what matters at low cardinality, not which rank the counts land on.

### Part 2

Gathering the raw elements costs $O(N)$ bytes, the thing to avoid. Sorting each shard locally first and merging the ten sorted lists at rank 0 does not help: the merge step is cheaper in CPU time, but every element still has to physically cross the network to reach rank 0, so the byte cost is unchanged. What actually avoids moving the data is to never ask "what is the data" and only ever ask "how much of the data is less than $g$" for a guess $g$ — a question every rank can answer from its own shard alone, in two small numbers.

This turns finding the median into a binary search over the *value* range rather than over array indices. Every rank first reports its local minimum, maximum and count to rank 0, which combines them into the global range $[\mathit{lo}, \mathit{hi}]$ and the global count $N$, and computes $k = \lfloor(N-1)/2\rfloor$ once, up front — the *target rank* of the answer in sorted order, which never changes for the rest of the search (contrast this with quickselect over an array, which recurses into a shrinking sub-array and has to adjust $k$ at every step; here $k$ is always measured against the same, whole dataset, so it stays fixed). Each round, rank 0 sends every other rank a guess $g$; every rank reports back how many of its own elements are less than $g$ and how many equal $g$; rank 0 sums both counts over all ten ranks. If more than $k$ elements are below $g$, the answer is smaller, so the search continues in $[\mathit{lo}, g - 1]$. Otherwise, if the elements below-or-equal to $g$ already cover $k$, then $g$ itself is the answer — its occurrences span sorted position $k$. Otherwise the answer is larger, and the search continues in $[g + 1, \mathit{hi}]$. Each round strictly shrinks $[\mathit{lo}, \mathit{hi}]$, so this converges in $O(\log_2 R)$ rounds, where $R = \mathit{hi} - \mathit{lo} + 1$ at the start.

```python
def global_median(rank, shard, send, recv, barrier):
    local_sorted = sorted(shard)
    local_min = local_sorted[0] if local_sorted else None
    local_max = local_sorted[-1] if local_sorted else None

    if rank != 0:
        send(0, (local_min, local_max, len(shard)))
    else:
        lo, hi, total_n = local_min, local_max, len(shard)
        for src in range(1, NUM_RANKS):
            r_min, r_max, r_len = recv(src)
            total_n += r_len
            if r_len:                      # NOTE: an empty shard contributes no bound on lo/hi
                lo = r_min if lo is None else min(lo, r_min)
                hi = r_max if hi is None else max(hi, r_max)
    barrier()                              # every rank now has (or, on rank 0, knows) the global range

    if rank != 0:
        while True:
            guess = recv(0)
            if guess is None:              # NOTE: rank 0's signal that the answer is already found
                return None
            less = sum(1 for v in local_sorted if v < guess)
            equal = sum(1 for v in local_sorted if v == guess)
            send(0, (less, equal))

    k = (total_n - 1) // 2
    answer = None
    while answer is None:
        guess = (lo + hi) // 2
        for dst in range(1, NUM_RANKS):
            send(dst, guess)
        total_less = sum(1 for v in local_sorted if v < guess)
        total_equal = sum(1 for v in local_sorted if v == guess)
        for src in range(1, NUM_RANKS):
            r_less, r_equal = recv(src)
            total_less += r_less
            total_equal += r_equal
        if total_less > k:
            hi = guess - 1
        elif total_less + total_equal > k:
            answer = guess
        else:
            lo = guess + 1
    for dst in range(1, NUM_RANKS):
        send(dst, None)                    # NOTE: every worker is blocked in recv(0) -- release them all
    return answer
```

Each round costs $O(P)$ bytes — a guess out and a `(less, equal)` pair back, from and to nine other ranks — for $O(\log_2 R)$ rounds, plus the $O(P)$ initial exchange of local ranges: $O\big(P \cdot (\log_2 R + 1)\big)$ overall, independent of $N$. Measured with the value range held at 1,000 while $N$ grows from 200 to 100,000, the bisection's byte total moves from 3,105 to 2,988 — essentially flat, since it is governed by $R$, not $N$ — while gathering everything grows from 636 to 247,318, nearly 390 times more, tracking $N$'s own 500-fold growth; at the largest size, gathering costs over eighty times what the bisection does. The fixed overhead is not free, though: at only 200 elements, gathering (636 bytes) is still cheaper than the ten rounds of bisection (3,105 bytes), so this protocol is worth running once $N$ is large enough that $O(P \log_2 R)$ genuinely undercuts $O(N)$, not unconditionally.

### Part 3

Gathering everything at rank 0, sorting it there, and handing each rank back a contiguous slice is correct, and moves the theoretical minimum in total — every element crosses the network once on the way in and, to reach its final owner, once on the way out. The problem is where that traffic goes: all $2N$ bytes of it pass through rank 0, which becomes both the busiest sender and the busiest receiver while the other nine ranks wait.

*Sample sort* keeps the same two crossings per element but spreads them over all ten ranks instead of funnelling them through one. Every rank draws a small local sample of its own shard and sends it — not the shard itself — to rank 0, which combines the samples, sorts them, and picks nine *splitters* at evenly spaced positions, partitioning the value range into ten *buckets*: values up to the first splitter form bucket 0, values between the first and second splitters form bucket 1, and so on. Rank 0 broadcasts the splitters to everyone. Every rank then partitions its *own* shard by bucket and, in one all-to-all exchange, sends each bucket's slice to the rank owning that bucket — an element travels directly from its origin to its final owner, never through rank 0 unless rank 0 happens to be either. Each rank finally sorts what it received for its own bucket.

```python
import bisect
import random

SAMPLE_SIZE = 8


def _bucket_of(value: int, splitters: list[int]) -> int:
    return bisect.bisect_right(splitters, value)   # NOTE: ties go to the bucket starting at that splitter


def sample_sort(rank, shard, send, recv, barrier):
    rng = random.Random(rank)     # NOTE: seeded by rank alone, so the sample is reproducible regardless
    sample = rng.sample(shard, min(len(shard), SAMPLE_SIZE))   #      of thread-scheduling order

    if rank != 0:
        send(0, sample)
    else:
        combined = sorted(sample + [v for src in range(1, NUM_RANKS) for v in recv(src)])
        splitters = [combined[i * len(combined) // NUM_RANKS] for i in range(1, NUM_RANKS)] if combined else []
        for dst in range(1, NUM_RANKS):
            send(dst, splitters)

    if rank != 0:
        splitters = recv(0)
    barrier()                     # everyone now has the same splitters before the exchange begins

    outgoing = [[] for _ in range(NUM_RANKS)]
    for v in shard:
        outgoing[_bucket_of(v, splitters)].append(v)
    for dst in range(NUM_RANKS):
        if dst != rank:
            send(dst, outgoing[dst])   # NOTE: send even [] -- every rank recv's from every other rank once
    mine = outgoing[rank]
    for src in range(NUM_RANKS):
        if src != rank:
            mine.extend(recv(src))
    mine.sort()
    return mine
```

Sampling and broadcasting the splitters costs $O(P \cdot \min(S, n_r))$ per rank for a sample size $S$, a small constant; the exchange moves every element exactly once to its final owner, $O(N)$ in total, and — as long as the sampled splitters divide the value range roughly evenly — close to $N/P$ of it through any one rank, instead of $N$ through rank 0 alone. A rank's own shard size is also a floor on what it must send: no partitioning scheme moves less than $n_r$ bytes out of a rank that started with $n_r$ elements, so this guarantee additionally assumes shard sizes are themselves within a constant factor of $N/P$ — unlike Parts 1 and 2, which only ever move a summary of a shard, never the shard itself, and so do not depend on shard sizes at all. Measured at $N = 50{,}000$ uniformly distributed values spread over evenly sized shards, the busiest rank's traffic is 31,365 bytes under sample sort against 219,390 bytes under gather-sort-scatter — nearly seven times as many. The guarantee is only as good as the sample and the input: a dataset overwhelmingly dominated by one repeated value can put every occurrence of it in a single bucket regardless of how the splitters fall, and a dataset overwhelmingly dominated by one oversized shard forces that one rank to send most of it no matter how the buckets are drawn. Either way, the result is still correct — every `result[r]` is still sorted, and the ranks are still in non-decreasing order — only the balance is lost, exactly the case the statement exempts from the bandwidth requirement.

### Follow-ups

- One rank's shard is 90% of the data. Network cost in Parts 1 and 2 does not depend on shard size, only on distinct-value and range counts, so it is unaffected; local compute at the overloaded rank grows, and Part 3's fixed sample size per rank under-samples it, skewing the splitters. Sampling in proportion to `len(shard)`, or re-sharding the oversized shard across idle ranks before running any of the three algorithms, fixes both.
- The values are floats and the median must be exact. Part 2's search relies on the answer being a value the bisection can land on exactly; floats offer no such guarantee. Fall back to Part 3: run `sample_sort`, then have rank 0 collect the ten `len(result[r])` values (ten integers) and index directly into whichever rank's slice holds sorted position $k$ — one pass of $O(N)$ bytes instead of a raw gather, and exact.
- The value range is too large for $\log_2 R$ rounds to be cheap, or the values are continuous. Switch to a mergeable summary such as a t-digest or a KLL sketch: each rank builds one locally in $O(n_r)$ time, sends only the sketch (its size does not depend on $n_r$), and rank 0 merges the ten sketches and queries the combined one — an approximate median within a stated error bound, in a single round instead of $O(\log_2 R)$.
- The owner function in Part 1, or the splitters in Part 3, collide badly on adversarial input — many distinct values sharing one residue mod $P$, say. A multiplicative hash (`(value * 2654435761) % NUM_RANKS`) spreads correlated values apart better than `value % NUM_RANKS` while staying just as cheap to compute; in Part 3, a bucket that ends up far larger than $N/P$ can be split further with its own, narrower round of sampling.
- $P$ is large enough that even $O(P)$ messages per round through rank 0 matters. Route the broadcast and the reduction through a tree of ranks instead of a star, as in a collective all-reduce: a rank forwards to a constant number of children and waits for their replies before replying itself, trading an $O(P)$-per-round fan-in at one rank for $O(\log P)$ depth spread over every rank.

```python
import pickle
import queue
import threading


class Cluster:
    """A minimal synchronous message-passing simulator for NUM_RANKS ranks, run over real threads so a
    blocking recv() genuinely blocks until the matching send() has been made. send(dst, obj) never blocks:
    it serializes obj and enqueues it for dst. recv(src) blocks until a message from that specific src is
    available, then returns it. Every rank's function receives its own send/recv/barrier already bound to
    its rank, so it never has to say who it is."""

    def __init__(self, num_ranks: int = NUM_RANKS):
        self.num_ranks = num_ranks
        self._inboxes = [[queue.Queue() for _ in range(num_ranks)] for _ in range(num_ranks)]  # [dst][src]
        self._barrier = threading.Barrier(num_ranks)
        self._lock = threading.Lock()
        self.bytes_sent = 0
        self.bytes_received = [0] * num_ranks   # bytes landing at each rank -- the per-rank bottleneck

    def run(self, fn, shards: list) -> list:
        results = [None] * self.num_ranks
        errors = [None] * self.num_ranks

        def make_send(rank):
            def send(dst, obj):
                nbytes = len(pickle.dumps(obj))
                with self._lock:
                    self.bytes_sent += nbytes
                    self.bytes_received[dst] += nbytes
                self._inboxes[dst][rank].put(obj)
            return send

        def make_recv(rank):
            def recv(src):
                return self._inboxes[rank][src].get()
            return recv

        def worker(rank):
            try:
                results[rank] = fn(rank, shards[rank], make_send(rank), make_recv(rank), self._barrier.wait)
            except Exception as e:   # surface the real traceback instead of a silent hang on join()
                errors[rank] = e

        threads = [threading.Thread(target=worker, args=(r,)) for r in range(self.num_ranks)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        for rank, e in enumerate(errors):
            if e is not None:
                raise RuntimeError(f"rank {rank} raised") from e
        return results


def reference_mode(values: list[int]) -> int:
    counts: dict[int, int] = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    best = max(counts.values())
    return min(v for v, c in counts.items() if c == best)


def reference_median(values: list[int]) -> int:
    s = sorted(values)
    return s[(len(s) - 1) // 2]


def _gather_to_rank0(rank, shard, send, recv):
    """Baseline building block: every rank ships its raw shard to rank 0, which concatenates them.
    Used only to measure the cost of centralizing the data, never as part of an accepted solution."""
    if rank == 0:
        merged = list(shard)
        for src in range(1, NUM_RANKS):
            merged.extend(recv(src))
        return merged
    send(0, shard)
    return None


def gather_all_to_rank0(rank, shard, send, recv, barrier):
    return _gather_to_rank0(rank, shard, send, recv)


def naive_mode_gather_histograms(rank, shard, send, recv, barrier):
    """Baseline: every rank ships its full local histogram to rank 0, which merges them -- cheaper than
    gather_all_to_rank0 when there are few distinct values, but still funnels everything through rank 0."""
    local = _local_histogram(shard)
    if rank == 0:
        merged = dict(local)
        for src in range(1, NUM_RANKS):
            for v, c in recv(src):
                merged[v] = merged.get(v, 0) + c
        best = max(merged.values())
        return min(v for v, c in merged.items() if c == best)
    send(0, list(local.items()))
    return None


def _split_into_contiguous_chunks(items: list, k: int) -> list:
    n = len(items)
    base, extra = divmod(n, k)
    chunks, start = [], 0
    for i in range(k):
        size = base + (1 if i < extra else 0)
        chunks.append(items[start:start + size])
        start += size
    return chunks


def gather_sort_scatter(rank, shard, send, recv, barrier):
    """Baseline: gather everything to rank 0, sort it there, hand each rank a contiguous slice back --
    correct, but every one of the 2N bytes moved passes through rank 0."""
    if rank == 0:
        merged = list(shard)
        for src in range(1, NUM_RANKS):
            merged.extend(recv(src))
        merged.sort()
        chunks = _split_into_contiguous_chunks(merged, NUM_RANKS)
        for dst in range(1, NUM_RANKS):
            send(dst, chunks[dst])
        return chunks[0]
    send(0, shard)
    return recv(0)


def make_shards(rng: random.Random, total_n: int, kind: str = "uniform", value_range: int = 1000) -> list:
    sizes = [total_n // NUM_RANKS] * NUM_RANKS
    for i in range(total_n - sum(sizes)):
        sizes[i] += 1
    if kind == "skewed_sizes" and total_n > 0:
        big = int(total_n * 0.7)
        rest = total_n - big
        sizes = [big] + [rest // (NUM_RANKS - 1)] * (NUM_RANKS - 1)
        for i in range(total_n - sum(sizes)):
            sizes[1 + i] += 1
    shards = []
    for size in sizes:
        if kind == "duplicate_heavy":
            shards.append([7 if rng.random() < 0.85 else rng.randint(0, 5) for _ in range(size)])
        elif kind == "all_same":
            shards.append([42] * size)
        else:
            shards.append([rng.randint(0, value_range - 1) for _ in range(size)])
    return shards


# --- the three worked examples of the statement ---
example1 = [[4, 1, 4], [2, 4, 2], [1, 1], [], [], [], [], [], [], []]
assert Cluster().run(global_mode, example1)[0] == 1

example2 = [[10, 1], [7, 3], [15, 6], [8, 2], [], [], [], [], [], []]
assert Cluster().run(global_median, example2)[0] == 6

example3 = [[5, 1], [9, 2], [4], [], [], [], [], [], [], []]
res3 = Cluster().run(sample_sort, example3)
assert [v for r in res3 for v in r] == [1, 2, 4, 5, 9]
for i in range(NUM_RANKS - 1):
    if res3[i] and res3[i + 1]:
        assert res3[i][-1] <= res3[i + 1][0]

# --- correctness sweeps against independent references, several data shapes, fixed seeds ---
rng = random.Random(0)
for _ in range(30):
    kind = rng.choice(["uniform", "skewed_sizes", "duplicate_heavy", "all_same"])
    n = rng.randint(0, 300)
    shards = make_shards(rng, n, kind=kind, value_range=rng.choice([3, 10, 500]))
    flat = [v for s in shards for v in s]
    if not flat:
        continue
    expected = reference_mode(flat)
    for fn in (global_mode, naive_mode_gather_histograms):
        assert Cluster().run(fn, shards)[0] == expected, (kind, n, fn.__name__)

rng = random.Random(1)
for _ in range(30):
    kind = rng.choice(["uniform", "skewed_sizes", "duplicate_heavy", "all_same"])
    n = rng.randint(1, 300)
    shards = make_shards(rng, n, kind=kind, value_range=rng.choice([2, 10, 500]))
    flat = [v for s in shards for v in s]
    if not flat:
        continue
    assert Cluster().run(global_median, shards)[0] == reference_median(flat), (kind, n)

rng = random.Random(2)  # negative values
for _ in range(15):
    shards = [[rng.randint(-1000, 1000) for _ in range(rng.randint(0, 30))] for _ in range(NUM_RANKS)]
    flat = [v for s in shards for v in s]
    if not flat:
        continue
    assert Cluster().run(global_median, shards)[0] == reference_median(flat)
    assert Cluster().run(global_mode, shards)[0] == reference_mode(flat)

rng = random.Random(3)
for _ in range(20):
    kind = rng.choice(["uniform", "skewed_sizes", "duplicate_heavy", "all_same"])
    n = rng.randint(0, 400)
    shards = make_shards(rng, n, kind=kind, value_range=rng.choice([2, 5, 500]))
    flat_sorted = sorted(v for s in shards for v in s)
    for fn in (sample_sort, gather_sort_scatter):
        results = Cluster().run(fn, shards)
        assert [v for r in results for v in r] == flat_sorted, (kind, n, fn.__name__)
        for i in range(NUM_RANKS - 1):
            if results[i] and results[i + 1]:
                assert results[i][-1] <= results[i + 1][0]

rng = random.Random(4)  # tiny totals, including all-empty and single-element
for n in range(16):
    shards = make_shards(rng, n, kind="uniform", value_range=3)
    flat_sorted = sorted(v for s in shards for v in s)
    assert [v for r in Cluster().run(sample_sort, shards) for v in r] == flat_sorted
    if flat_sorted:
        assert Cluster().run(global_mode, shards)[0] == reference_mode(flat_sorted)
        assert Cluster().run(global_median, shards)[0] == reference_median(flat_sorted)

# --- communication claims, measured ---

# Part 2: bisection stays flat as N grows (range fixed); gathering grows with N
median_records = []
for n in (200, 2_000, 20_000, 100_000):
    rng = random.Random(42)
    shards = make_shards(rng, n, kind="uniform", value_range=1000)
    c1, c2 = Cluster(), Cluster()
    c1.run(global_median, shards)
    c2.run(gather_all_to_rank0, shards)
    median_records.append((n, c1.bytes_sent, c2.bytes_sent))
assert median_records[-1][1] < 5 * median_records[0][1], median_records
assert median_records[-1][2] > 50 * median_records[0][2], median_records
assert median_records[-1][2] > 10 * median_records[-1][1], median_records

# Part 1: the busiest rank's bytes under the shuffle grow much more slowly than under histogram-gather
mode_records = []
for v_count in (5, 50, 500, 2_000):
    rng = random.Random(7)
    shards = make_shards(rng, 4_000, kind="uniform", value_range=v_count)
    c1, c2 = Cluster(), Cluster()
    c1.run(global_mode, shards)
    c2.run(naive_mode_gather_histograms, shards)
    mode_records.append((v_count, max(c1.bytes_received), max(c2.bytes_received)))
assert mode_records[-1][2] > 3 * mode_records[-1][1], mode_records

rng = random.Random(11)  # low cardinality: raw gather vs. histogram gather
shards = make_shards(rng, 4_000, kind="uniform", value_range=5)
c1, c2 = Cluster(), Cluster()
c1.run(gather_all_to_rank0, shards)
c2.run(naive_mode_gather_histograms, shards)
assert c1.bytes_sent > 10 * c2.bytes_sent, (c1.bytes_sent, c2.bytes_sent)

# Part 3: the busiest rank's bytes under sample sort stay near N / P; gather-sort-scatter concentrates at rank 0
sort_records = []
for n in (2_000, 10_000, 50_000):
    rng = random.Random(9)
    shards = make_shards(rng, n, kind="uniform", value_range=1_000_000)
    c1, c2 = Cluster(), Cluster()
    c1.run(sample_sort, shards)
    c2.run(gather_sort_scatter, shards)
    sort_records.append((n, max(c1.bytes_received), max(c2.bytes_received)))
assert sort_records[-1][2] > 5 * sort_records[-1][1], sort_records

print("all checks passed")
```
