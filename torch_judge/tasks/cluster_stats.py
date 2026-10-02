"""Mode, median and sort over ten ranks that talk only by message passing, with bounded traffic."""

from ._interview import interview

# A ten-rank simulator: one thread per rank, pickled messages, and byte counters per rank.
_HELPERS = r"""
import pickle, queue, random, threading, time
from collections import Counter

P = 10

class Cluster:
    def __init__(self, shards):
        self.shards = [list(s) for s in shards]
        self.boxes = {(s, d): queue.Queue() for s in range(P) for d in range(P)}
        self.bar = threading.Barrier(P)
        self.sent, self.got, self.lock = [0] * P, [0] * P, threading.Lock()

    def tools(self, rank):
        def send(dst, obj):
            data = pickle.dumps(obj)
            with self.lock:
                self.sent[rank] += len(data)
                self.got[dst] += len(data)
            self.boxes[(rank, dst)].put(pickle.loads(data))
        def recv(src):
            try:
                return self.boxes[(src, rank)].get(timeout=4)
            except queue.Empty:
                raise AssertionError(f"rank {rank} waited for rank {src} forever") from None
        def barrier():
            self.bar.wait(timeout=4)
        return send, recv, barrier

    def run(self, method):
        results, errors = [None] * P, []
        def body(rank):
            try:
                send, recv, barrier = self.tools(rank)
                results[rank] = getattr({fn}(), method)(rank, list(self.shards[rank]), send, recv, barrier)
            except BaseException as e:
                errors.append(f"rank {rank}: {type(e).__name__}: {e}")
        threads = [threading.Thread(target=body, args=(r,), daemon=True) for r in range(P)]
        for t in threads:
            t.start()
        deadline = time.monotonic() + 7
        for t in threads:
            t.join(timeout=max(0.0, deadline - time.monotonic()))
        assert not errors, errors[0]
        assert not any(t.is_alive() for t in threads), "some rank never returned"
        return results

def spread(values, rng, empty=0.3):
    shards = [[] for _ in range(P)]
    live = [r for r in range(P) if rng.random() > empty] or [rng.randrange(P)]
    for v in values:
        shards[rng.choice(live)].append(v)
    return shards

def model_mode(shards):
    c = Counter(v for s in shards for v in s)
    return min(c.items(), key=lambda kv: (-kv[1], kv[0]))[0]

def model_median(shards):
    allv = sorted(v for s in shards for v in s)
    return allv[(len(allv) - 1) // 2]

def check_sorted_split(shards, results):
    assert all(isinstance(r, list) for r in results), "every rank must return a list"
    flat = [v for r in results for v in r]
    assert flat == sorted(v for s in shards for v in s), "concatenating the results must give every element once, sorted"
"""

TASK = {
    "title": "Distributed Mode, Median and Sort",
    "difficulty": "Hard",
    "version": 1,
    "function_name": "ClusterStats",
    "description_en": r"""Build `ClusterStats`, whose methods run on each of ten ranks at once and compute the mode, the median and a sorted split of data spread across them, using only messages.

The requirement arrives in parts. Each part keeps every earlier behavior, so one `ClusterStats` class passes all parts at the end. Pass every test of the current part to reveal the next one.

**Rules for every part:**
- There are `P = 10` ranks, numbered `0` to `9`. Each rank creates its own `ClusterStats()` and calls the same method with `(rank, shard, send, recv, barrier)`.
- `shard` is that rank's list of integers, possibly empty and with duplicates. At least one shard is non-empty.
- `send(dst, obj)` delivers `obj` to rank `dst` and returns at once. `recv(src)` waits for the next object rank `src` sent to this rank; objects from one sender arrive in the order sent. `barrier()` waits until all ten ranks have called it.
- Network cost is `len(pickle.dumps(obj))` bytes per `send`. Local work on your own shard is free.
- Every rank must return; only rank `0`'s return value is checked unless a part says otherwise.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why this shows up in interviews:** gathering everything on one rank always works; the work is a protocol whose traffic stays bounded as the data grows, and each later part adds one requirement.

**Where it is used:** distributed training and data pipelines compute global statistics, quantiles and shuffles over shards with collectives such as all-to-all and all-reduce.

Adapted from the distributed mode and median question in Schuture/Anthropic-Interview-Notes (CC BY-NC 4.0), reworded, on one class. Traffic bounds are checked against byte counts with generous constants.""",
    "parts": [
        {
            "title": "Global mode",
            "description_en": r"""**Signature:** `ClusterStats()`, `global_mode(rank, shard, send, recv, barrier) -> int | None`

- Rank `0` returns the value with the highest total count over all shards; on a tie, the smallest such value.
- No rank may receive much more than its share. With `D` the number of distinct values summed over the shards, each rank receives `O(D / P + P)` bytes: do not send every count to one rank.

**Example**, ranks `2`, `3`, `5` and `7` hold `[11, 4]`, `[7, 11]`, `[4, 5]` and `[7]`, the others hold nothing:
- the counts are `11: 2`, `4: 2`, `7: 2`, `5: 1`
- `4`, `7` and `11` tie, so rank `0` returns `4`""",
        },
        {
            "title": "Global median",
            "description_en": r"""Keep Part 1 and add the median.

**Signature:** `global_median(rank, shard, send, recv, barrier) -> int | None`

- With `N` elements in total and `k = (N - 1) // 2`, rank `0` returns the element at index `k` of all elements sorted: the lower middle when `N` is even.
- Total bytes sent by all ranks together must be `O(P * (log2(R) + 1))`, where `R` is one plus the largest value minus the smallest. It must not grow with `N`, so send counts and bounds, never the elements.

**Example**, ranks `0`, `2`, `5` and `9` hold `[12, 4]`, `[9]`, `[1, 15, 7]` and `[20]`:
- sorted: `[1, 4, 7, 9, 12, 15, 20]`, `N = 7`, `k = 3`, so rank `0` returns `9`
- if rank `9` held `[20, 3]` instead, `N = 8`, `k = 3`, and the answer is `7`""",
        },
        {
            "title": "Distributed sort",
            "description_en": r"""Keep Parts 1–2 and add a sort that leaves the data spread out.

**Signature:** `sample_sort(rank, shard, send, recv, barrier) -> list[int]`

- Every rank's return value counts here. Each rank returns a sorted list, and concatenating ranks `0` to `9` in order gives every element of every shard exactly once, in non-decreasing order.
- On data that is not heavily skewed, no rank sends plus receives more than a small constant times `1 / P` of the data's bytes. Skewed data, such as one repeated value, must still be sorted correctly.

**Example**, ranks `0` and `4` hold `[6, 3]` and `[9, 3, 7, 1]`:
- all elements sorted are `[1, 3, 3, 6, 7, 9]`
- rank `0` returning `[1, 3, 3]`, rank `5` returning `[6, 7, 9]` and every other rank `[]` meets the rule, and so does any other split into ten sorted, consecutive pieces""",
        },
    ],
    "hints": [
        {"level": 1, "kind": "questions", "content": "If every rank sends its full count table to rank 0, how much does rank 0 receive? Could each value instead have one owner rank that collects only its counts, and what would each owner then send to rank 0?"},
        {"level": 2, "kind": "analysis", "content": "Count locally with Counter(shard). Send rank value % P the counts of the values it owns, one dict per destination, every rank to every rank. Each rank sums what it receives and sends its best (value, count), smallest value on ties, to rank 0. Rank 0 picks the best of the ten; everyone returns."},
    ],
    "model_connections": [
        "Distributed training shards data across ranks and combines statistics with collectives such as all-reduce and all-to-all.",
        "Data pipelines compute global quantiles and shuffle-sort shards across workers, as in MapReduce and Spark.",
    ],
    "pro_con_analysis": {
        "pros": [
            "Hash-partitioning values to owner ranks spreads the counting so no single rank receives everything.",
            "Binary search on the value range needs only counts, so median traffic grows with log R, not with N.",
            "Sampling splitters from every shard keeps each rank's slice of the sort close to N / P on typical data.",
        ],
        "cons": [
            "One very frequent value lands on a single owner rank, and on a single sort bucket.",
            "The median search makes about log R sequential rounds of messages, each paying latency.",
            "All-to-all messaging needs P squared messages, which grows quickly with the number of ranks.",
        ],
    },
    "tests": [
        {"name": "Part 1: the worked example", "part": 1, "behavior": "state.invariant", "code": _HELPERS + r"""
shards = [[], [], [11, 4], [7, 11], [], [4, 5], [], [7], [], []]
assert Cluster(shards).run("global_mode")[0] == 4
"""},
        {"name": "Part 1: random shards", "part": 1, "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "On random shards, rank 0 did not return the most frequent value (smallest on ties), or some rank failed to return.",
         "code": _HELPERS + r"""
rng = random.Random(1)
for trial in range(25):
    vals = [rng.randrange(-20, 20) for _ in range(rng.randint(1, 60))]
    shards = spread(vals, rng)
    assert Cluster(shards).run("global_mode")[0] == model_mode(shards), (trial, shards)
assert Cluster([[-5]] + [[]] * 9).run("global_mode")[0] == -5
assert Cluster([[]] * 9 + [[4, -4]]).run("global_mode")[0] == -4
"""},
        {"name": "Part 1: no rank receives everything", "part": 1, "visibility": "unshown", "behavior": "budget.enforcement",
         "failure_message": "With 30,000+ distinct values, one rank received close to all the counts; give each value an owner rank so the counts are split ten ways.",
         "code": _HELPERS + r"""
rng = random.Random(2)
shards = [[rng.randrange(r * 2000, r * 2000 + 6000) for _ in range(5000)] for r in range(P)]
cluster = Cluster(shards)
assert cluster.run("global_mode")[0] == model_mode(shards)
everything = sum(len(pickle.dumps(dict(Counter(s)))) for s in shards)
assert max(cluster.got) <= everything / 4 + 5000, f"one rank received {max(cluster.got)} bytes; all counts together are {everything}"
"""},
        {"name": "Part 2: the worked example", "part": 2, "behavior": "state.invariant", "code": _HELPERS + r"""
shards = [[12, 4], [], [9], [], [], [1, 15, 7], [], [], [], [20]]
assert Cluster(shards).run("global_median")[0] == 9
shards[9] = [20, 3]
assert Cluster(shards).run("global_median")[0] == 7
"""},
        {"name": "Part 2: random shards", "part": 2, "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "On random shards, rank 0 did not return the element at index (N - 1) // 2 of the sorted data, or some rank failed to return.",
         "code": _HELPERS + r"""
rng = random.Random(3)
for trial in range(25):
    vals = [rng.randrange(-30, 30) for _ in range(rng.randint(1, 50))]
    shards = spread(vals, rng)
    assert Cluster(shards).run("global_median")[0] == model_median(shards), (trial, shards)
assert Cluster([[7, 7, 7]] + [[]] * 9).run("global_median")[0] == 7
assert Cluster([[]] * 4 + [[-10 ** 9, 10 ** 9]] + [[]] * 5).run("global_median")[0] == -10 ** 9
"""},
        {"name": "Part 2: traffic does not grow with N", "part": 2, "visibility": "unshown", "behavior": "budget.enforcement",
         "failure_message": "Finding the median of 200,000 values sent far more than O(P log R) bytes; send counts for candidate values, never the elements themselves.",
         "code": _HELPERS + r"""
import math
rng = random.Random(4)
shards = [[rng.randrange(-10 ** 6, 10 ** 6) for _ in range(20000)] for _ in range(P)]
cluster = Cluster(shards)
assert cluster.run("global_median")[0] == model_median(shards)
R = max(map(max, shards)) - min(map(min, shards)) + 1
limit = 150 * P * (math.log2(R) + 2)
assert sum(cluster.sent) <= limit, f"sent {sum(cluster.sent)} bytes in total; the limit is {limit:.0f}"
"""},
        {"name": "Part 3: the worked example", "part": 3, "behavior": "state.invariant", "code": _HELPERS + r"""
shards = [[6, 3], [], [], [], [9, 3, 7, 1], [], [], [], [], []]
check_sorted_split(shards, Cluster(shards).run("sample_sort"))
"""},
        {"name": "Part 3: random and skewed shards", "part": 3, "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "The ten returned lists, concatenated in rank order, were not exactly the sorted data, including for skewed data such as one repeated value.",
         "code": _HELPERS + r"""
rng = random.Random(5)
for trial in range(20):
    vals = [rng.randrange(-40, 40) for _ in range(rng.randint(1, 80))]
    shards = spread(vals, rng)
    check_sorted_split(shards, Cluster(shards).run("sample_sort"))
for shards in ([[3] * 50] + [[]] * 9, [[1]] + [[]] * 9, [[9, 8, 7, 6, 5, 4, 3, 2, 1, 0]] * 10):
    check_sorted_split(shards, Cluster(shards).run("sample_sort"))
"""},
        {"name": "Part 3: traffic is balanced", "part": 3, "visibility": "unshown", "behavior": "budget.enforcement",
         "failure_message": "Sorting 40,000 uniform values sent or received far more than a fair share on one rank; pick splitters from samples and send each element straight to its bucket's rank.",
         "code": _HELPERS + r"""
rng = random.Random(6)
shards = [[rng.randrange(0, 10 ** 6) for _ in range(4000)] for _ in range(P)]
cluster = Cluster(shards)
check_sorted_split(shards, cluster.run("sample_sort"))
total = len(pickle.dumps([v for s in shards for v in s]))
worst = max(a + b for a, b in zip(cluster.sent, cluster.got))
assert worst <= 3 * total / P + 20000, f"one rank moved {worst} bytes; the data is {total} bytes"
"""},
    ],
    "solution": r'''# Adapted from Schuture/Anthropic-Interview-Notes (code under the MIT License).
from bisect import bisect_right
from collections import Counter

P = 10


class ClusterStats:
    def global_mode(self, rank, shard, send, recv, barrier):
        # each value has an owner rank, so its counts meet in one place and no rank gets everything
        buckets = [{} for _ in range(P)]
        for value, count in Counter(shard).items():
            buckets[value % P][value] = count
        for dst in range(P):
            send(dst, buckets[dst])
        totals = Counter()
        for src in range(P):
            totals.update(recv(src))
        best = min(totals.items(), key=lambda kv: (-kv[1], kv[0])) if totals else None
        send(0, best)  # one candidate per owner
        if rank != 0:
            return None
        candidates = [c for c in (recv(src) for src in range(P)) if c is not None]
        return min(candidates, key=lambda kv: (-kv[1], kv[0]))[0]

    def global_median(self, rank, shard, send, recv, barrier):
        local = sorted(shard)
        # rank 0 learns N and the value range, then binary-searches the value with counts only
        send(0, (len(local), local[0] if local else None, local[-1] if local else None))
        if rank == 0:
            stats = [recv(src) for src in range(P)]
            n = sum(s[0] for s in stats)
            lo = min(s[1] for s in stats if s[0])
            hi = max(s[2] for s in stats if s[0])
            k = (n - 1) // 2
            while lo < hi:  # smallest v with at least k + 1 elements <= v
                mid = (lo + hi) // 2
                for dst in range(1, P):
                    send(dst, mid)
                below = bisect_right(local, mid) + sum(recv(src) for src in range(1, P))
                if below >= k + 1:
                    hi = mid
                else:
                    lo = mid + 1
            for dst in range(1, P):
                send(dst, None)  # done
            return lo
        while True:
            mid = recv(0)
            if mid is None:
                break
            send(0, bisect_right(local, mid))
        return None

    def sample_sort(self, rank, shard, send, recv, barrier):
        local = sorted(shard)
        # P - 1 evenly spaced samples per rank pick P - 1 splitters on rank 0
        samples = [local[(i * len(local)) // P] for i in range(1, P)] if local else []
        send(0, samples)
        if rank == 0:
            pool = sorted(x for src in range(P) for x in recv(src))
            splitters = [pool[(i * len(pool)) // P] for i in range(1, P)] if pool else []
            for dst in range(P):
                send(dst, splitters)
        splitters = recv(0)
        start = 0
        for dst in range(P):
            end = bisect_right(local, splitters[dst]) if dst < len(splitters) else len(local)
            send(dst, local[start:end])
            start = end
        out = []
        for src in range(P):
            out.extend(recv(src))
        return sorted(out)
''',
    "interview_questions": interview(
        concept=[
            "Why does sending every count to rank 0 break the per-rank traffic bound?",
            "How does giving each value an owner rank keep the result correct and the traffic spread?",
        ],
        deep_dive=[
            "How many messages and bytes does your mode protocol send in total, and how much does the busiest rank receive?",
        ],
        tradeoffs=[
            "How do you find the median by exchanging counts only, and how many rounds does it take?",
            "How do you choose splitters for a sample sort, and what happens when one value dominates?",
            "Where would barriers or a fixed message order matter to avoid deadlock?",
            "How would these protocols change with collectives such as all-reduce and all-to-all, or with many more ranks?",
        ],
    ),
}
