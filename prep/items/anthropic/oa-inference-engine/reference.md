Worth confirming up front: whether the decode step of an already-running sequence should ever be skipped for lack of budget (assumed here never, matching a real scheduler sized so the budget can always cover one token per running sequence, which is why `token_budget >= max_seqs` is given); and whether a preempted request keeps its place in the queue or goes to the back (here the front, so a sequence under repeated memory pressure is retried before anything that has not started yet, rather than risking indefinite starvation).

### Level 1

`add_request` only has to create a `Sequence` and enqueue its id; every later level's work goes into `step`. Prefilling reserves `ceil((len(token_ids) + 1) / BLOCK_SIZE)` blocks — not `ceil(len(token_ids) / BLOCK_SIZE)`, which is one short whenever a prompt's length is itself a multiple of `BLOCK_SIZE`, since prefill is about to append a token that the already-reserved blocks then have no room for.

```python
class Engine:
    def __init__(self, num_blocks: int):
        self.allocator = BlockAllocator(num_blocks)
        self.queue = RequestQueue()
        self.sequences: dict[int, Sequence] = {}
        self.running: dict[int, Sequence] = {}   # insertion order == admission order, used from Level 5
        self._next_id = 0

    def add_request(self, prompt_token_ids: list[int], max_tokens: int) -> int:
        request_id = self._next_id
        self._next_id += 1
        seq = Sequence(request_id, prompt_token_ids, max_tokens)
        self.sequences[request_id] = seq
        self.queue.push_back(request_id)
        return request_id

    def _prefill(self, seq: Sequence) -> None:
        # NOTE: +1 -- room for the token this call is about to produce, not just for token_ids as it is now
        needed = (len(seq.token_ids) + 1 + BLOCK_SIZE - 1) // BLOCK_SIZE
        seq.block_ids.extend(self.allocator.allocate(needed - len(seq.block_ids)))
        seq.append_token(next_token(seq.token_ids))
        seq.status = SeqStatus.RUNNING
        self.running[seq.request_id] = seq

    def step(self) -> None:
        while len(self.queue) > 0:
            self._prefill(self.sequences[self.queue.pop_front()])
```

`add_request` is $O(1)$; `step` is $O(r)$ for $r$ currently waiting requests, since no block is ever evicted yet and each prefill's own allocation is $O(1)$.

### Level 2

Decode gives every running sequence one token, unconditionally, so `used` starts at the number of running sequences (one token each, already spent by the decode loop about to run) and only grows from there as admissions are approved; recomputing it from `len(self.running)` *after* the decode loop, instead, would double-count nothing here yet, but stops being correct once Level 5 lets a sequence disappear from `running` mid-loop.

```python
class Engine(Engine):
    def __init__(self, num_blocks: int, token_budget: int = 10**9, max_seqs: int = 10**9):
        super().__init__(num_blocks)
        self.token_budget = token_budget
        self.max_seqs = max_seqs

    def _decode_step(self, seq: Sequence) -> None:
        if len(seq.token_ids) % BLOCK_SIZE == 0:            # its last block is exactly full
            seq.block_ids.extend(self.allocator.allocate(1))
        seq.append_token(next_token(seq.token_ids))

    def step(self) -> None:
        used = len(self.running)                             # NOTE: one token per running sequence, spent up front
        for request_id in list(self.running):                # NOTE: snapshot -- see Level 5
            seq = self.running.get(request_id)
            if seq is not None:
                self._decode_step(seq)

        while len(self.running) < self.max_seqs and len(self.queue) > 0:
            seq = self.sequences[self.queue.peek_front()]
            if used + len(seq.token_ids) > self.token_budget:   # NOTE: FIFO -- stop, do not skip to a smaller one
                break
            self.queue.pop_front()
            used += len(seq.token_ids)
            self._prefill(seq)
```

`token_budget` and `max_seqs` default to effectively unlimited so that Level 1's examples, which construct `Engine(num_blocks)` alone, keep working unchanged against this and every later level. `step` costs $O(m)$ per call, in the number of sequences decoded (bounded by `max_seqs`) plus the requests newly admitted this round.

### Level 3

A sequence's stop check has to run after *every* call that appends a token — prefill's and decode's alike — so it belongs in one place both call into, not duplicated in each.

```python
class Engine(Engine):
    def _finish_if_stopped(self, seq: Sequence) -> bool:
        if not seq.is_stopped():
            return False
        self.allocator.free(seq.block_ids)
        seq.block_ids = []
        seq.status = SeqStatus.FINISHED
        self.running.pop(seq.request_id, None)
        return True

    def _prefill(self, seq: Sequence) -> None:
        needed = (len(seq.token_ids) + 1 + BLOCK_SIZE - 1) // BLOCK_SIZE
        seq.block_ids.extend(self.allocator.allocate(needed - len(seq.block_ids)))
        seq.append_token(next_token(seq.token_ids))
        if not self._finish_if_stopped(seq):                # NOTE: only reaches RUNNING if it did not just stop
            seq.status = SeqStatus.RUNNING
            self.running[seq.request_id] = seq

    def _decode_step(self, seq: Sequence) -> None:
        if len(seq.token_ids) % BLOCK_SIZE == 0:
            seq.block_ids.extend(self.allocator.allocate(1))
        seq.append_token(next_token(seq.token_ids))
        self._finish_if_stopped(seq)
```

`step` is unchanged: it already only iterates `self.running`, which `_finish_if_stopped` now keeps current. `_finish_if_stopped` itself adds only $O(1)$ amortized work, freeing at most a handful of blocks, so `step`'s cost stays $O(m)$, as in Level 2.

### Level 4

Two different sequences can legitimately produce the same block content — two requests sharing a prompt, most simply — and `register`'s contract (stated in its docstring) is to keep only the first registration when that happens. Calling it for every block that just became full, and letting it decide, is enough; nothing here has to detect the collision itself. `full_block_hashes` is given, so the only new logic is *where* to look each hash up and *when* to stop looking.

```python
class Engine(Engine):
    def _register_if_newly_full(self, seq: Sequence) -> None:
        if len(seq.token_ids) % BLOCK_SIZE == 0:
            block_index = len(seq.token_ids) // BLOCK_SIZE - 1
            block_hash_ = full_block_hashes(seq, block_index + 1)[-1]
            self.allocator.register(seq.block_ids[block_index], block_hash_)

    def _prefill(self, seq: Sequence) -> None:
        num_full = len(seq.token_ids) // BLOCK_SIZE
        hashes = full_block_hashes(seq, num_full)
        hits = 0
        while hits < num_full:                                # NOTE: stop at the first miss -- see Follow-ups
            cached_id = self.allocator.lookup_cached(hashes[hits])
            if cached_id is None:
                break
            seq.block_ids.append(cached_id)                    # NOTE: lookup_cached already increfs -- no incref here
            hits += 1
        needed = (len(seq.token_ids) + 1 + BLOCK_SIZE - 1) // BLOCK_SIZE
        fresh = self.allocator.allocate(needed - hits)
        seq.block_ids.extend(fresh)
        for offset, block_id in enumerate(fresh):
            if hits + offset < num_full:                       # NOTE: never register the trailing partial block
                self.allocator.register(block_id, hashes[hits + offset])
        seq.append_token(next_token(seq.token_ids))
        self._register_if_newly_full(seq)                      # NOTE: the prompt's own tail can complete right here
        if not self._finish_if_stopped(seq):
            seq.status = SeqStatus.RUNNING
            self.running[seq.request_id] = seq

    def _decode_step(self, seq: Sequence) -> None:
        if len(seq.token_ids) % BLOCK_SIZE == 0:
            seq.block_ids.extend(self.allocator.allocate(1))
        seq.append_token(next_token(seq.token_ids))
        self._register_if_newly_full(seq)
        self._finish_if_stopped(seq)
```

Caching only ever changes *which physical block* a position's KV cache lives in, never a sequence's own `token_ids`: `lookup_cached` and `allocate` both return an id whose slot is guaranteed to hold exactly the right content (by construction for a hit, freshly written for a miss), so nothing past this point needs to know which happened.

Prefill's cache lookups cost $O(f)$ in a sequence's number of full blocks $f$; `_register_if_newly_full` recomputes the whole hash chain up to the new block on every call, also $O(f)$, rather than the $O(1)$ an incrementally maintained running hash would give.

### Level 5

`_allocate_or_preempt` retries until the allocator can satisfy the request, each time discarding the running sequence that was admitted most recently — `self.running` is a plain `dict`, and insertion order is admission order, so that sequence is `next(reversed(self.running))`. The loop always terminates: each failed attempt removes one sequence from `running`, so after at most `len(self.running)` rounds either an attempt succeeds or `running` is empty, and an empty engine's whole pool is free, which Level 1's sizing guarantee — no request's total need ever exceeds `num_blocks` — ensures is always enough for one more sequence's reservation.

```python
class Engine(Engine):
    def _allocate_or_preempt(self, count: int) -> list[int]:
        while True:
            try:
                return self.allocator.allocate(count)
            except OutOfBlocks:
                self._preempt(next(reversed(self.running)))   # NOTE: may be the very sequence asking for room

    def _preempt(self, request_id: int) -> None:
        seq = self.running.pop(request_id)
        self.allocator.free(seq.block_ids)
        seq.block_ids = []
        seq.status = SeqStatus.WAITING
        self.queue.push_front(request_id)

    def _prefill(self, seq: Sequence) -> None:
        num_full = len(seq.token_ids) // BLOCK_SIZE
        hashes = full_block_hashes(seq, num_full)
        hits = 0
        while hits < num_full:
            cached_id = self.allocator.lookup_cached(hashes[hits])
            if cached_id is None:
                break
            seq.block_ids.append(cached_id)
            hits += 1
        needed = (len(seq.token_ids) + 1 + BLOCK_SIZE - 1) // BLOCK_SIZE
        fresh = self._allocate_or_preempt(needed - hits)   # NOTE: seq is never its own victim here -- it is not
        seq.block_ids.extend(fresh)                          #  in self.running until this call returns
        for offset, block_id in enumerate(fresh):
            if hits + offset < num_full:
                self.allocator.register(block_id, hashes[hits + offset])
        seq.append_token(next_token(seq.token_ids))
        self._register_if_newly_full(seq)
        if not self._finish_if_stopped(seq):
            seq.status = SeqStatus.RUNNING
            self.running[seq.request_id] = seq

    def _decode_step(self, seq: Sequence) -> None:
        if len(seq.token_ids) % BLOCK_SIZE == 0:
            fresh = self._allocate_or_preempt(1)
            if seq.request_id not in self.running:      # NOTE: seq preempted itself while making its own room
                self.allocator.free(fresh)                #  -- it is WAITING now; hand the block straight back
                return
            seq.block_ids.append(fresh[0])
        seq.append_token(next_token(seq.token_ids))
        self._register_if_newly_full(seq)
        self._finish_if_stopped(seq)
```

`_decode_step` can be asked to extend a sequence that turns out to be the one evicted to make room for itself — the most recently admitted running sequence is a well-defined choice even when only one sequence is running. Skipping the append and returning the spare block leaves no trace: the sequence is already back in the queue, `token_ids` and `num_generated` are untouched, and its next attempt goes through `_prefill` like any other admission. `step`, `_finish_if_stopped` and the admission loop in `step`'s second half are all unchanged from Level 3: a preempted sequence disappearing from `self.running` mid-decode is exactly the case the snapshot-plus-`.get` pattern in `step` was already written to tolerate.

Each call to `_allocate_or_preempt` costs $O(v)$ in the number of sequences $v$ it evicts before its own allocation succeeds, plus that allocation's own $O(1)$ amortized cost.

### Follow-ups

- Chunked prefill: this problem only admits a request once its whole prompt fits the current step's remaining budget. A production engine instead prefills a long prompt a bounded chunk of tokens at a time across several steps, so one very long prompt cannot delay every other sequence's decode step until it is admitted in full.
- Swap instead of recompute: preemption here always discards the victim's blocks and regenerates them later from `token_ids` alone. Swapping copies a preempted sequence's block contents out to host memory and back instead, trading a transfer for the recomputation it avoids — worth it once a sequence's prefix is long enough that recomputing it costs more than copying it.
- A different victim policy: preempting the most recently admitted running sequence bounds how long any one sequence can be starved, since the sequence that has been running longest is never chosen while a newer one remains. Preempting whichever running sequence holds the most blocks instead frees more room per preemption, at the cost of being harder for a caller to predict.
- Partial-block caching: a cache entry only ever covers a full block, so two prompts that agree on everything except their last few tokens still share every earlier block but never the final, partial one. Caching less than a full block would mean handing a sequence a block that a second, diverging sequence could still write into.
- Concurrency: `step()` here runs to completion before anything else touches the engine. A server that accepts new requests while a step is in flight would need a lock around `queue`, `running` and the allocator together, since admitting a request or preempting a sequence changes state that the same step's own loop is still in the middle of reading.

```python
# ---- Level 1 example ----
eng = Engine(num_blocks=10)
assert eng.add_request([2, 9, 4, 7, 1], max_tokens=3) == 0
assert eng.add_request([5, 3, 8], max_tokens=3) == 1
eng.step()
assert eng.sequences[0].block_ids == [0, 1] and eng.sequences[0].generated_token_ids == [15]
assert eng.sequences[1].block_ids == [2] and eng.sequences[1].generated_token_ids == [3]

# ---- Level 2 example ----
eng = Engine(num_blocks=20, token_budget=6, max_seqs=2)
a = eng.add_request([1, 2, 3, 4], max_tokens=5)
b = eng.add_request([5, 6, 7], max_tokens=5)
c = eng.add_request([8, 9], max_tokens=5)
eng.step()
assert eng.sequences[a].generated_token_ids == [14]
assert eng.sequences[b].generated_token_ids == [] and eng.sequences[c].generated_token_ids == []
eng.step()
assert eng.sequences[a].generated_token_ids == [14, 13]
assert eng.sequences[b].generated_token_ids == [4]
assert eng.sequences[c].generated_token_ids == []
eng.step()
assert eng.sequences[a].generated_token_ids == [14, 13, 0]
assert eng.sequences[b].generated_token_ids == [4, 2]
assert eng.sequences[c].generated_token_ids == []

# ---- Level 3 example ----
eng = Engine(num_blocks=8, token_budget=20, max_seqs=5)
a = eng.add_request([2, 9, 4, 7, 1], max_tokens=2)
b = eng.add_request([0, 3], max_tokens=5)
c = eng.add_request([6, 6, 6], max_tokens=4)
eng.step()
assert eng.sequences[a].status == SeqStatus.RUNNING and eng.sequences[a].block_ids == [0, 1]
assert eng.sequences[a].generated_token_ids == [15]
assert eng.sequences[b].status == SeqStatus.FINISHED and eng.sequences[b].block_ids == []
assert eng.sequences[b].generated_token_ids == [16]
assert eng.sequences[c].status == SeqStatus.RUNNING and eng.sequences[c].block_ids == [2]
assert eng.sequences[c].generated_token_ids == [11]
assert eng.allocator.num_free_or_evictable() == 5
eng.step()
assert eng.sequences[a].status == SeqStatus.FINISHED and eng.sequences[a].block_ids == []
assert eng.sequences[a].generated_token_ids == [15, 9]
assert eng.sequences[c].block_ids == [2, 1] and eng.sequences[c].generated_token_ids == [11, 8]
assert eng.allocator.num_free_or_evictable() == 6
eng.step(); eng.step()
assert eng.sequences[c].status == SeqStatus.FINISHED
assert eng.sequences[c].generated_token_ids == [11, 8, 3, 6]
assert eng.allocator.num_free_or_evictable() == 8

# ---- Level 4 example ----
eng = Engine(num_blocks=4, token_budget=40, max_seqs=5)
a = eng.add_request([5, 5, 5, 5, 2], max_tokens=2)
b = eng.add_request([5, 5, 5, 5, 9], max_tokens=4)
eng.step()
assert eng.sequences[a].block_ids == [0, 1] and eng.sequences[a].generated_token_ids == [12]
assert eng.sequences[b].block_ids == [0, 2] and eng.sequences[b].generated_token_ids == [2]
assert eng.allocator.blocks[0].ref_count == 2
eng.step()
assert eng.sequences[a].status == SeqStatus.FINISHED and eng.sequences[a].generated_token_ids == [12, 4]
assert eng.sequences[b].status == SeqStatus.RUNNING and eng.sequences[b].block_ids == [0, 2]
assert eng.sequences[b].generated_token_ids == [2, 10]
assert eng.allocator.blocks[0].ref_count == 1
eng.step(); eng.step()
assert eng.sequences[b].status == SeqStatus.FINISHED
assert eng.sequences[b].generated_token_ids == [2, 10, 12, 4]
assert eng.allocator.blocks[0].ref_count == 0 and 0 in eng.allocator._lru
shared_hash = block_hash(0, [5, 5, 5, 5])
assert eng.allocator._cached_by_hash[shared_hash] == 0

c = eng.add_request([1, 1, 1, 1, 1, 1, 1, 1], max_tokens=1)
eng.step()
assert eng.sequences[c].status == SeqStatus.FINISHED and eng.sequences[c].generated_token_ids == [12]
assert shared_hash not in eng.allocator._cached_by_hash        # evicted to make room for C's 3 blocks

d = eng.add_request([5, 5, 5, 5, 7], max_tokens=1)
eng.step()
assert eng.sequences[d].generated_token_ids == [0]              # a miss, not a hit: the old hash is gone
assert shared_hash in eng.allocator._cached_by_hash             # D's own first block re-registers it fresh

# ---- Level 5 example ----
eng = Engine(num_blocks=3, token_budget=30, max_seqs=5)
p = eng.add_request([6, 6], max_tokens=5)
q = eng.add_request([2, 9, 4, 7], max_tokens=5)
eng.step(); eng.step()
assert eng.sequences[p].status == SeqStatus.RUNNING and eng.sequences[p].block_ids == [0]
assert eng.sequences[p].generated_token_ids == [6, 11]
assert eng.sequences[q].status == SeqStatus.RUNNING and eng.sequences[q].block_ids == [1, 2]
assert eng.sequences[q].generated_token_ids == [11, 8]
assert eng.allocator.num_free_or_evictable() == 0
eng.step()
assert eng.sequences[p].status == SeqStatus.FINISHED
assert eng.sequences[p].generated_token_ids == [6, 11, 8, 3, 6]
assert eng.sequences[q].status == SeqStatus.WAITING
assert eng.sequences[q].generated_token_ids == [11, 8, 3, 6]
eng.step()
assert eng.sequences[q].status == SeqStatus.FINISHED
assert eng.sequences[q].generated_token_ids == [11, 8, 3, 6, 11]
assert eng.allocator.num_free_or_evictable() == 3
print("all five levels' worked examples replayed exactly")
```

```python
# ---- independent reference model + randomized cross-validation ----
import random


def naive_generate(prompt_token_ids: list[int], max_tokens: int) -> list[int]:
    """What one request generates run entirely on its own -- no queueing, no shared blocks, no
    eviction, nothing but next_token called in a loop. The invariant the whole problem rests on:
    batching, caching and preemption change only which physical blocks are used and when, never
    this."""
    token_ids = list(prompt_token_ids)
    generated: list[int] = []
    while len(generated) < max_tokens and (not generated or generated[-1] != EOS_TOKEN):
        token = next_token(token_ids)
        token_ids.append(token)
        generated.append(token)
    return generated


def _check_invariants(eng: "Engine") -> None:
    allocator = eng.allocator
    assert len(allocator._free) == len(set(allocator._free))   # no block ever freed twice
    free_set, lru_set = set(allocator._free), set(allocator._lru)
    assert not (free_set & lru_set)
    live_count = 0
    for block_id, block in allocator.blocks.items():
        if block.ref_count > 0:
            assert block_id not in free_set and block_id not in lru_set
            live_count += 1
        elif block.content_hash is not None:
            assert block_id in lru_set and allocator._cached_by_hash.get(block.content_hash) == block_id
        else:
            assert block_id in free_set
    assert len(free_set) + len(lru_set) + live_count == len(allocator.blocks)   # no block ever leaked


def _run_trial(seed: int) -> None:
    rng = random.Random(seed)
    max_prompt, max_gen = 6, 6
    min_num_blocks = -(-(max_prompt + max_gen) // BLOCK_SIZE)   # else even an empty pool could not fit one request
    num_blocks = rng.randint(min_num_blocks, min_num_blocks + 4)
    max_seqs = rng.randint(1, 4)
    token_budget = max(max_prompt + max_gen, max_seqs, rng.randint(4, 16))

    eng = Engine(num_blocks=num_blocks, token_budget=token_budget, max_seqs=max_seqs)
    expected = {}
    for _ in range(rng.randint(3, 12)):
        prompt = [rng.randrange(VOCAB_SIZE) for _ in range(rng.randint(1, max_prompt))]
        max_tokens = rng.randint(1, max_gen)
        request_id = eng.add_request(prompt, max_tokens)
        expected[request_id] = naive_generate(prompt, max_tokens)

    steps = 0
    while (len(eng.queue) or eng.running) and steps < 500:
        eng.step()
        _check_invariants(eng)
        steps += 1
    assert steps < 500, seed   # every trial is sized to finish; hitting the cap would mean a scheduling bug

    for request_id, want in expected.items():
        assert eng.sequences[request_id].generated_token_ids == want, (seed, request_id)
    _check_invariants(eng)


for seed in range(1000):
    _run_trial(seed)
print("cross-validated 1000 random workloads against an independent single-request reference, with no block leaks")
print("all checks passed")
```
