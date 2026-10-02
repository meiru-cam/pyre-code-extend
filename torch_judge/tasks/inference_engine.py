"""A small LLM inference engine around provided model, allocator and sequence code: prefill, a budgeted decode loop, stopping, prefix caching and preemption."""

from ._interview import interview

_SOLUTION = r'''# Adapted from Schuture/Anthropic-Interview-Notes (code under the MIT License).
# Provided code: use it as it is. Write the Engine class at the bottom.
from collections import OrderedDict, deque
from enum import Enum

VOCAB_SIZE = 19  # token ids are 0 .. 18
EOS_TOKEN = 18   # generating this token ends a sequence
BLOCK_SIZE = 4   # tokens held by one KV-cache block


def next_token(token_ids):
    """A stand-in for the model: a fixed hash of every token so far. token_ids must not be empty."""
    h = 0
    for t in token_ids:
        h = (h * 31 + t + 7) % VOCAB_SIZE
    return h


class OutOfBlocks(Exception):
    """allocate() could not find enough blocks, even after evicting every cached one. Nothing changed."""


class Block:
    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0        # how many sequences use this block
        self.content_hash = None  # set by register(), cleared on eviction


class BlockAllocator:
    """Every block is free (ref_count 0, no hash), cached (ref_count 0, hash kept for reuse) or live (ref_count >= 1)."""

    def __init__(self, num_blocks):
        self.blocks = [Block(i) for i in range(num_blocks)]
        self._free = list(range(num_blocks - 1, -1, -1))  # pop() hands out the lowest id first
        self._by_hash = {}            # content hash -> block id, for every registered block
        self._cached = OrderedDict()  # cached block ids, the one unused longest first

    def num_free_or_evictable(self):
        return len(self._free) + len(self._cached)

    def allocate(self, count):
        """count block ids, each with ref_count 1; evicts the longest-unused cached blocks if the free list is short."""
        if count > self.num_free_or_evictable():
            raise OutOfBlocks(f"need {count}, have {self.num_free_or_evictable()}")
        while len(self._free) < count:
            block_id, _ = self._cached.popitem(last=False)
            del self._by_hash[self.blocks[block_id].content_hash]
            self.blocks[block_id].content_hash = None
            self._free.append(block_id)
        ids = [self._free.pop() for _ in range(count)]
        for block_id in ids:
            self.blocks[block_id].ref_count = 1
        return ids

    def free(self, block_ids):
        """Drops one reference from each block. At 0 a hashed block becomes cached and any other block free."""
        for block_id in block_ids:
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                if block.content_hash is None:
                    self._free.append(block_id)
                else:
                    self._cached[block_id] = None

    def lookup_cached(self, content_hash):
        """The id of the block registered under content_hash, after adding one reference to it, or None."""
        block_id = self._by_hash.get(content_hash)
        if block_id is not None:
            block = self.blocks[block_id]
            if block.ref_count == 0:
                del self._cached[block_id]
            block.ref_count += 1
        return block_id

    def register(self, block_id, content_hash):
        """Lets lookup_cached find this live block; does nothing if another block already has content_hash."""
        if content_hash not in self._by_hash:
            self.blocks[block_id].content_hash = content_hash
            self._by_hash[content_hash] = block_id


def block_hash(prev_hash, tokens):
    h = prev_hash
    for t in tokens:
        h = (h * 1_000_033 + t + 1) & 0xFFFFFFFF
    return h


def full_block_hashes(token_ids, count):
    """The chained hashes of the first count full blocks of token_ids: equal hashes mean equal tokens from the start."""
    hashes, h = [], 0
    for i in range(count):
        h = block_hash(h, token_ids[i * BLOCK_SIZE:(i + 1) * BLOCK_SIZE])
        hashes.append(h)
    return hashes


class Status(Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"


class Sequence:
    def __init__(self, request_id, prompt, max_tokens):
        self.request_id = request_id
        self.prompt = list(prompt)
        self.max_tokens = max_tokens
        self.token_ids = list(prompt)  # the prompt, then every generated token
        self.status = Status.WAITING
        self.block_ids = []

    @property
    def generated(self):
        return self.token_ids[len(self.prompt):]

    def is_stopped(self):
        """True once max_tokens tokens were generated or the last one is EOS_TOKEN."""
        return len(self.generated) >= self.max_tokens or self.token_ids[-1] == EOS_TOKEN


class Engine:
    def __init__(self, num_blocks, token_budget=None, max_seqs=None):
        self.allocator = BlockAllocator(num_blocks)
        self.sequences = []      # sequences[request_id]
        self.waiting = deque()   # request ids, front first
        self.running = []        # running sequences, oldest admission first
        self.token_budget = token_budget
        self.max_seqs = max_seqs
        self.preempted = False   # whether this step has preempted anyone

    def add_request(self, prompt, max_tokens):
        seq = Sequence(len(self.sequences), prompt, max_tokens)
        self.sequences.append(seq)
        self.waiting.append(seq.request_id)
        return seq.request_id

    def step(self):
        self.preempted = False
        decoded = 0
        for seq in list(self.running):
            if seq.status is Status.RUNNING and self._decode(seq):  # an earlier decode may have preempted it
                decoded += 1
        used = decoded
        while self.waiting and not self.preempted:
            if self.max_seqs is not None and len(self.running) >= self.max_seqs:
                break
            seq = self.sequences[self.waiting[0]]
            if self.token_budget is not None and used + len(seq.token_ids) > self.token_budget:
                break  # nothing is admitted out of order
            self.waiting.popleft()
            used += len(seq.token_ids)
            self._prefill(seq)

    def _prefill(self, seq):
        full = len(seq.token_ids) // BLOCK_SIZE
        hits = []
        for content_hash in full_block_hashes(seq.token_ids, full):
            block_id = self.allocator.lookup_cached(content_hash)
            if block_id is None:
                break  # reuse only an unbroken run of leading blocks
            hits.append(block_id)
        need = -(-(len(seq.token_ids) + 1) // BLOCK_SIZE)
        seq.block_ids = hits + self._allocate(need - len(hits))  # seq is not running yet, so never its own victim
        hashes = full_block_hashes(seq.token_ids, full)
        for i in range(len(hits), full):
            self.allocator.register(seq.block_ids[i], hashes[i])
        seq.status = Status.RUNNING
        self.running.append(seq)
        self._generate(seq)

    def _decode(self, seq):
        """Adds one token; False if making room preempted seq itself."""
        if len(seq.token_ids) + 1 > len(seq.block_ids) * BLOCK_SIZE:
            fresh = self._allocate(1, seq)
            if fresh is None:
                return False
            seq.block_ids += fresh
        self._generate(seq)
        return True

    def _allocate(self, count, requester=None):
        """Preempts the most recent admission until count blocks fit; None if that preempted requester."""
        while count > self.allocator.num_free_or_evictable():
            victim = self.running[-1]
            self._preempt(victim)
            if victim is requester:
                return None
        return self.allocator.allocate(count)

    def _preempt(self, seq):
        self._release(seq, Status.WAITING)
        self.waiting.appendleft(seq.request_id)
        self.preempted = True

    def _release(self, seq, status):
        self.allocator.free(seq.block_ids)
        seq.block_ids = []
        seq.status = status
        self.running.remove(seq)

    def _generate(self, seq):
        seq.token_ids.append(next_token(seq.token_ids))
        if len(seq.token_ids) % BLOCK_SIZE == 0:  # the last block just became full
            index = len(seq.token_ids) // BLOCK_SIZE - 1
            self.allocator.register(seq.block_ids[index], full_block_hashes(seq.token_ids, index + 1)[index])
        if seq.is_stopped():
            self._release(seq, Status.FINISHED)
'''

# Every hidden test runs the reference engine side by side with the learner's and compares their state.
_HARNESS = "REF_SRC = " + repr(_SOLUTION) + "\n" + r'''
import random
REF = {}
exec(REF_SRC, REF)
Status = REF["Status"]
nt = REF["next_token"]
EOS = REF["EOS_TOKEN"]

def snap(engine, ids=True):
    seqs = []
    for s in engine.sequences:
        blocks = tuple(s.block_ids) if ids else len(s.block_ids)
        seqs.append((s.status.value, blocks, tuple(s.generated)))
    out = [seqs, engine.allocator.num_free_or_evictable()]
    if ids:
        out.append([b.ref_count for b in engine.allocator.blocks])
    return out

def continuation(prompt, k):
    t, out = list(prompt), []
    for _ in range(k):
        x = nt(t)
        t.append(x)
        out.append(x)
    return out

def run(fn, workload, steps, ids=True, allow_preempt=True, label=None):
    """workload: (num_blocks, budget, max_seqs, {step: [(prompt, max_tokens)]}). Returns False if the model preempted and that is not allowed."""
    num_blocks, budget, max_seqs, arrivals = workload
    got, want = fn(num_blocks, budget, max_seqs), REF["Engine"](num_blocks, budget, max_seqs)
    trace = []
    for step in range(steps):
        for prompt, max_tokens in arrivals.get(step, []):
            a, b = got.add_request(list(prompt), max_tokens), want.add_request(list(prompt), max_tokens)
            assert a == b, (label, "request id", a, b)
        want.step()
        if want.preempted and not allow_preempt:
            return False  # checked before the learner's step, which may not handle running out of blocks yet
        got.step()
        trace.append(step)
        g, w = snap(got, ids), snap(want, ids)
        assert g == w, (label, "after step", step + 1, "got", g, "want", w, "arrivals", arrivals)
    return True
'''

TESTS = [
    {"name": "Part 1: the worked example", "part": 1, "behavior": "contract.signature", "code": r"""
e = {fn}(8)
assert e.add_request([3, 1, 4, 1, 5], 4) == 0
assert e.add_request([2, 7], 4) == 1
e.step()
a, b = e.sequences
assert a.status.value == "running" and a.block_ids == [0, 1] and a.generated == [6]
assert b.status.value == "running" and b.block_ids == [2] and b.generated == [8]
assert e.allocator.num_free_or_evictable() == 5
"""},
    {"name": "Part 1: random requests, one step", "part": 1, "visibility": "unshown", "behavior": "state.invariant",
     "failure_message": "After one step over random waiting requests, a sequence's status, block ids or generated tokens, or the allocator's counts, differed from prefilling each request in queue order with one allocate call for ceil((len(token_ids) + 1) / BLOCK_SIZE) blocks.",
     "code": _HARNESS + r"""
for seed in range(200):
    rng = random.Random(seed)
    reqs = []
    for first in rng.sample(range(18), rng.randint(1, 5)):
        prompt = [first] + [rng.randrange(18) for _ in range(rng.randint(0, 7))]
        if continuation(prompt, 1)[0] != EOS:
            reqs.append((prompt, rng.randint(2, 6)))
    need = sum(-(-(len(p) + 1) // 4) for p, _ in reqs)
    run({fn}, (need + 1 + rng.randint(0, 3), None, None, {0: reqs}), 1, label=seed)
"""},
    {"name": "Part 2: the worked example", "part": 2, "behavior": "budget.enforcement", "code": r"""
e = {fn}(20, 6, 3)
for prompt in ([6, 2, 6], [4, 4, 1, 9], [5], [8, 3]):
    e.add_request(prompt, 6)
e.step()
assert [s.generated for s in e.sequences] == [[17], [], [], []]
e.step()
assert [s.generated for s in e.sequences] == [[17, 0], [13], [12], []]
assert e.sequences[0].block_ids == [0, 1] and e.sequences[1].block_ids == [2, 3] and e.sequences[2].block_ids == [4]
e.step()
assert [s.generated for s in e.sequences] == [[17, 0, 7], [13, 5], [12, 11], []]
"""},
    {"name": "Part 2: random arrivals under a budget", "part": 2, "visibility": "unshown", "behavior": "budget.enforcement",
     "failure_message": "Over several steps, a result differed from decoding every running sequence oldest-admitted first, allocating one block only when the new token needs it, then admitting from the queue front while fewer than max_seqs run and the tokens of this step stay within token_budget, stopping at the first request that does not fit.",
     "code": _HARNESS + r"""
for seed in range(250):
    rng = random.Random(100 + seed)
    arrivals, longest = {}, 1
    for first in rng.sample(range(18), rng.randint(1, 6)):
        prompt = [first] + [rng.randrange(18) for _ in range(rng.randint(0, 6))]
        if EOS in continuation(prompt, 8):
            continue
        arrivals.setdefault(rng.randint(0, 4), []).append((prompt, 50))
        longest = max(longest, len(prompt))
    max_seqs = rng.choice([None, 1, 2, 3])
    budget = rng.choice([None, longest, longest + 2, longest + 5])
    if budget is not None and max_seqs is not None:
        budget = max(budget, max_seqs)
    run({fn}, (64, budget, max_seqs, arrivals), 8, label=seed)
"""},
    {"name": "Part 3: the worked example", "part": 3, "behavior": "edge.empty_or_boundary", "code": r"""
e = {fn}(3, 30, 4)
e.add_request([5, 0], 4)
e.add_request([9, 9, 9], 2)
e.step()
a, b = e.sequences
assert a.status.value == "finished" and a.block_ids == [] and a.generated == [18]
assert b.status.value == "running" and len(b.block_ids) == 1 and b.generated == [4]
assert e.allocator.num_free_or_evictable() == 2
e.step()
assert b.status.value == "finished" and b.block_ids == [] and b.generated == [4, 2]
assert e.allocator.num_free_or_evictable() == 3
e.add_request([1] * 8, 2)
e.step()
c = e.sequences[2]
assert c.status.value == "running" and len(c.block_ids) == 3 and c.generated == [9]
e.step()
assert c.status.value == "finished" and c.generated == [9, 10]
assert e.allocator.num_free_or_evictable() == 3
"""},
    {"name": "Part 3: random workloads with stops", "part": 3, "visibility": "unshown", "behavior": "state.invariant",
     "failure_message": "With requests that stop by max_tokens or by EOS_TOKEN, also during prefill, a status, generated list, number of blocks held or free count differed: a stopped sequence must free all its blocks at once, finish, and stop counting against max_seqs.",
     "code": _HARNESS + r"""
for seed in range(250):
    rng = random.Random(400 + seed)
    arrivals, longest, total = {}, 1, 0
    for first in rng.sample(range(18), rng.randint(1, 7)):
        prompt = [first] + [rng.randrange(18) for _ in range(rng.randint(0, 6))]
        max_tokens = rng.randint(1, 6)
        arrivals.setdefault(rng.randint(0, 6), []).append((prompt, max_tokens))
        longest = max(longest, len(prompt))
    max_seqs = rng.choice([None, 2, 3])
    budget = rng.choice([None, longest + 3, longest + 8])
    run({fn}, (64, budget, max_seqs, arrivals), 14, ids=False, label=seed)
"""},
    {"name": "Part 4: the worked example", "part": 4, "behavior": "attention.cache", "code": r"""
e = {fn}(6, 40, 4)
e.add_request([3] * 8 + [1], 2)
e.add_request([3] * 8 + [2], 2)
e.add_request([3] * 4 + [7] * 4, 2)
e.step()
assert [s.block_ids for s in e.sequences] == [[0, 1, 2], [0, 1, 3], [0, 4, 5]]
assert [b.ref_count for b in e.allocator.blocks] == [3, 2, 1, 1, 1, 1]
assert [s.generated for s in e.sequences] == [[10], [11], [13]]
e.step()
assert all(s.status.value == "finished" for s in e.sequences)
assert e.allocator.num_free_or_evictable() == 6
e.add_request([9] * 12, 1)
e.step()
e.add_request([3] * 8 + [5], 3)
e.step()
d = e.sequences[4]
assert d.block_ids == [0, 4, 2] and d.generated == [14]
"""},
    {"name": "Part 4: random shared prefixes", "part": 4, "visibility": "unshown", "behavior": "attention.cache",
     "failure_message": "With prompts that share leading blocks, a block id, reference count or free count differed: reuse only the unbroken run of leading full blocks found by lookup_cached, allocate the rest at once, register every fresh block as soon as it is full, and leave eviction to the allocator.",
     "code": _HARNESS + r"""
prefixes = [[], [3] * 4, [3] * 8, [3, 3, 3, 3, 7, 7, 7, 7], [4, 4, 4, 4]]
checked = 0
for seed in range(600):
    if checked == 200:
        break
    rng = random.Random(800 + seed)
    num_blocks = rng.randint(6, 14)
    arrivals = {}
    for _ in range(rng.randint(2, 7)):
        prompt = rng.choice(prefixes) + [rng.choice([1, 2, 3]) for _ in range(rng.randint(1, 5))]
        max_tokens = rng.randint(1, 6)
        if -(-(len(prompt) + max_tokens) // 4) <= num_blocks:
            arrivals.setdefault(rng.randint(0, 8), []).append((prompt, max_tokens))
    if run({fn}, (num_blocks, None, rng.choice([None, 3]), arrivals), 16, allow_preempt=False, label=seed):
        checked += 1
assert checked == 200, checked
"""},
    {"name": "Part 4: a later block survives an earlier one", "part": 4, "visibility": "unshown", "behavior": "attention.cache",
     "failure_message": "When an earlier block of a prefix was evicted but a later one is still cached, the later one must not be reused: reuse stops at the first miss.",
     "code": r"""
e = {fn}(5)
e.add_request([3] * 12 + [1], 1)
e.step()
e.add_request([9] * 13, 1)
e.step()
e.add_request([3] * 12 + [2], 2)
e.step()
assert e.sequences[2].block_ids == [0, 1, 2, 4], e.sequences[2].block_ids
assert [b.ref_count for b in e.allocator.blocks] == [1, 1, 1, 0, 1]
"""},
    {"name": "Part 5: the worked example", "part": 5, "behavior": "scheduler.concurrency", "code": r"""
e = {fn}(4, 30, 4)
for prompt in ([1] * 3, [2] * 3, [5] * 3):
    e.add_request(prompt, 5)
e.step()
assert [s.block_ids for s in e.sequences] == [[0], [1], [2]]
e.step()
a, b, c = e.sequences
assert a.block_ids == [0, 3] and a.generated == [2, 14]
assert b.block_ids == [1, 2] and b.generated == [7, 3]
assert c.status.value == "waiting" and c.block_ids == [] and c.generated == [3]
e.step()
assert a.status.value == "finished" and a.generated == [2, 14, 18]
assert c.status.value == "running" and c.block_ids == [0, 3] and c.generated == [3, 8]
for _ in range(3):
    e.step()
assert [s.generated for s in e.sequences] == [[2, 14, 18], [7, 3, 8, 16, 6], [3, 8, 16, 6, 9]]
assert e.allocator.num_free_or_evictable() == 4
"""},
    {"name": "Part 5: random workloads under memory pressure", "part": 5, "visibility": "unshown", "behavior": "scheduler.concurrency",
     "failure_message": "With too few blocks, a result differed from preempting the most recently admitted running sequence, freeing its blocks and putting it at the queue front, until the allocation fits; a sequence may be its own victim; nothing more is admitted in a step after a preemption; every request must still finish with the tokens it would generate alone.",
     "code": _HARNESS + r"""
prefixes = [[], [3] * 4, [6, 6, 6, 6]]
for seed in range(300):
    rng = random.Random(2000 + seed)
    num_blocks = rng.randint(3, 6)
    arrivals, reqs, longest = {}, [], 1
    for _ in range(rng.randint(2, 7)):
        prompt = rng.choice(prefixes) + [rng.choice([1, 2, 5]) for _ in range(rng.randint(1, 4))]
        max_tokens = rng.randint(1, 7)
        if -(-(len(prompt) + max_tokens) // 4) <= num_blocks:
            arrivals.setdefault(rng.randint(0, 5), []).append((prompt, max_tokens))
            reqs.append((prompt, max_tokens))
            longest = max(longest, len(prompt) + max_tokens)
    budget = rng.choice([None, longest, longest + 4])
    run({fn}, (num_blocks, budget, rng.choice([None, 2, 3]), arrivals), 40, label=seed)
"""},
    {"name": "Part 5: tokens do not depend on the workload", "part": 5, "visibility": "unshown", "behavior": "state.invariant",
     "failure_message": "Under memory pressure, some request ended with different generated tokens than it produces alone, did not finish, or left blocks allocated: preemption must keep token_ids and only drop blocks.",
     "code": _HARNESS + r"""
for seed in range(150):
    rng = random.Random(5000 + seed)
    e = {fn}(4, 40, None)
    reqs = []
    for _ in range(6):
        prompt = [rng.choice([1, 3, 3]) for _ in range(rng.randint(1, 7))]
        max_tokens = rng.randint(2, 8)
        if -(-(len(prompt) + max_tokens) // 4) <= 4:
            reqs.append((prompt, max_tokens))
            e.add_request(prompt, max_tokens)
    for _ in range(200):
        if all(s.status.value == "finished" for s in e.sequences):
            break
        e.step()
    assert reqs and len(e.sequences) == len(reqs), (seed, len(e.sequences))
    for s, (prompt, max_tokens) in zip(e.sequences, reqs):
        alone = continuation(prompt, max_tokens)
        if EOS in alone:
            alone = alone[:alone.index(EOS) + 1]
        assert s.status.value == "finished" and s.generated == alone, (seed, prompt, max_tokens, s.generated, alone)
    assert all(b.ref_count == 0 for b in e.allocator.blocks), seed
    assert e.allocator.num_free_or_evictable() == 4
"""},
]

TASK = {
    "title": "Small LLM Inference Engine",
    "difficulty": "Hard",
    "version": 1,
    "function_name": "Engine",
    "description_en": r"""Build `Engine`, the scheduler of a small LLM inference engine, on top of provided model, block allocator and sequence code: prefill, a budgeted decode loop, stopping, prefix caching and preemption.

The requirement arrives in parts. Each part keeps every earlier rule, so one `Engine` class passes all parts at the end. Pass every test of the current part to reveal the next one.

**Rules for every part:**
- The starter's provided code stays as it is: `next_token`, `BlockAllocator`, `block_hash`, `full_block_hashes`, `Status` and `Sequence`. Read it first; what each call assumes is part of the task.
- `Engine(num_blocks, token_budget=None, max_seqs=None)` keeps `self.allocator`, a `BlockAllocator(num_blocks)`, and `self.sequences`, where `sequences[r]` is the `Sequence` of request `r`. `None` means no limit.
- `add_request(prompt, max_tokens) -> int` creates a `WAITING` sequence, puts it at the back of the queue and returns `0`, `1`, `2`, and so on.
- Before any token is generated, the sequence must hold `ceil((len(token_ids) + 1) / BLOCK_SIZE)` blocks; then append `next_token(token_ids)` to `token_ids`.
- Prefill admits a waiting sequence: it reserves the blocks it is missing with one `allocate` call, marks it `RUNNING` and generates one token. Decode generates one more token for a running sequence, with one `allocate(1)` call when the new token needs another block.
- Blocks a sequence gives back go in one `free(block_ids)` call, in block order. Every request fits in `num_blocks` blocks on its own.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why this shows up in interviews:** it is a reading task as much as a writing one: the allocator's contract decides what the engine must do, and every later rule touches the same few lines. Each later part adds one requirement: a step budget, freeing on stop, block reuse across requests, and preemption when memory runs out.

**Where it is used:** vLLM, SGLang and TensorRT-LLM schedule requests the same way, with paged KV-cache blocks, continuous batching, prefix caching keyed by chained block hashes, and recompute-style preemption.

Adapted from the inference engine online assessment in Schuture/Anthropic-Interview-Notes (CC BY-NC 4.0), reworded. The provided code is rewritten with new constants and hashes, a deque instead of `RequestQueue`, and shorter `Sequence` fields (`prompt`, `generated`, `Status`); `token_budget` and `max_seqs` are optional from the start. Admission counts `len(token_ids)`, so a preempted sequence pays for every token it re-prefills, nothing more is admitted in a step after a preemption, and the Part 3 tests compare block counts instead of block ids.""",
    "parts": [
        {
            "title": "Admission and prefill",
            "description_en": r"""**Signatures:** `Engine(num_blocks)`, `add_request(prompt, max_tokens) -> int`, `step()`

- `step()` prefills every waiting request, in queue order, and marks each `RUNNING`.
- The tests here call `step()` once.

**Example:** `e = Engine(8)`, then `add_request([3, 1, 4, 1, 5], 4)` is `0` and `add_request([2, 7], 4)` is `1`:
- after `step()`, request `0` holds blocks `[0, 1]`, since 6 tokens need two blocks, and has generated `[6]`
- request `1` holds `[2]` and has generated `[8]`; `5` blocks remain free""",
        },
        {
            "title": "A budgeted decode loop",
            "description_en": r"""Keep Part 1. `step()` is now called once per round and does two things in order:

- First it decodes every `RUNNING` sequence, the earliest admitted first.
- Then it admits from the front of the queue while fewer than `max_seqs` sequences run and the tokens of this step stay within `token_budget`. Tokens of a step: one per sequence decoded, plus `len(token_ids)` of each sequence admitted.
- The first waiting request that does not fit ends admission for this step, even if a later one would fit.
- `token_budget` is at least every prompt's length and at least `max_seqs`.

**Example:** `Engine(20, 6, 3)` with prompts `[6, 2, 6]`, `[4, 4, 1, 9]`, `[5]` and `[8, 3]`, each with `max_tokens = 6`:
- step 1 admits only the first: `3 + 4` would pass `6`, so the 1-token prompt waits too
- step 2 decodes the first (1 token), then admits the second (4) and the third (1); the fourth waits, as `3` sequences run
- step 3 decodes all three and admits nothing""",
        },
        {
            "title": "Stopping",
            "description_en": r"""Keep Parts 1–2. A sequence stops as soon as `is_stopped()` is true after any token it generates, in prefill or in decode.

- On stopping, free all its blocks, empty `block_ids` and mark it `FINISHED` before handling the next sequence.
- A finished sequence is never decoded again and no longer counts against `max_seqs`.

**Example:** `Engine(3, 30, 4)` with `[5, 0]` (`max_tokens = 4`) and `[9, 9, 9]` (`max_tokens = 2`):
- step 1: the first generates `[18]`, the end token, and finishes at once; the second runs on one block, with `2` blocks free
- step 2: the second reaches two tokens and finishes; all `3` blocks are free
- a request with an 8-token prompt can now take all three blocks""",
        },
        {
            "title": "Prefix caching",
            "description_en": r"""Keep Parts 1–3. Prefill now reuses blocks that already hold the same leading tokens.

- Before allocating, take the hashes of the sequence's full blocks from `full_block_hashes` and call `lookup_cached` on them from block `0`, stopping at the first miss. Allocate only the blocks after the hits.
- Register every block that is not a hit under its hash, with `register`, as soon as it holds `BLOCK_SIZE` tokens: at prefill for blocks the existing tokens already fill, later right after the token that fills it.
- Eviction is the allocator's business: a cached block can vanish when blocks are allocated.

**Example:** `Engine(6, 40, 4)` with `[3]*8 + [1]`, `[3]*8 + [2]` and `[3]*4 + [7]*4`, each with `max_tokens = 2`:
- after step 1 they hold `[0, 1, 2]`, `[0, 1, 3]` and `[0, 4, 5]`, so block `0` has 3 references and block `1` has 2
- all finish in step 2; then a 12-token request evicts the cached block `1`, the one unused longest
- `[3]*8 + [5]` then reuses block `0` only and gets `[0, 4, 2]`""",
        },
        {
            "title": "Preemption",
            "description_en": r"""Keep Parts 1–4. When an allocation, in prefill or decode, needs more blocks than `num_free_or_evictable()`, make room first.

- Preempt the `RUNNING` sequence admitted most recently: free its blocks, mark it `WAITING` and put it at the front of the queue. Repeat until the allocation fits. Its tokens stay; it later re-prefills all of them.
- The victim may be the sequence asking for the block. Then it gets no token in this step.
- After a preemption, `step()` admits nothing more until the next step.
- From here on, `token_budget` is at least `len(prompt) + max_tokens` of every request.

**Example:** `Engine(4, 30, 4)` with `[1]*3`, `[2]*3` and `[5]*3`, each with `max_tokens = 5`:
- step 1 admits all three on blocks `0`, `1` and `2`
- step 2: the first takes block `3`; the second needs a block, so the third, admitted last, is preempted and the second gets block `2`
- step 3: the first ends with `18`, and the third comes back on blocks `[0, 3]`, re-prefilling its 4 tokens""",
        },
    ],
    "hints": [
        {"level": 1, "kind": "questions", "content": "Before next_token runs, how many tokens must the sequence's blocks be able to hold? Which allocator call gives a sequence its blocks, and in what order should waiting requests get them?"},
        {"level": 2, "kind": "analysis", "content": "Keep a deque of waiting request ids. add_request builds a Sequence with the next id, appends it to self.sequences and to the deque. step pops each waiting id in order, calls allocate(ceil((len(token_ids) + 1) / BLOCK_SIZE)) once, stores the ids in block_ids, sets the status to RUNNING and appends next_token(token_ids)."},
    ],
    "model_connections": [
        "vLLM-style serving keeps the KV cache in fixed-size blocks so that memory is allocated per block instead of per maximum sequence length.",
        "Prefix caching lets requests that share a system prompt or few-shot examples skip recomputing those blocks.",
    ],
    "pro_con_analysis": {
        "pros": [
            "Fixed-size blocks avoid fragmentation and let a sequence grow one block at a time.",
            "A per-step token budget keeps each step's latency bounded while new requests join running ones.",
            "Chained block hashes make prefix reuse safe: equal hashes mean equal tokens from the very start.",
        ],
        "cons": [
            "Recompute preemption throws away KV work that must be redone when the sequence returns.",
            "Strict queue order can leave budget unused when a long request blocks shorter ones behind it.",
            "Only full blocks are shared, so a common prefix shorter than a block gains nothing.",
        ],
    },
    "tests": TESTS,
    "solution": _SOLUTION,
    "interview_questions": interview(
        concept=[
            "Why must a sequence hold room for len(token_ids) + 1 tokens before next_token is called?",
            "How many blocks does a 7-token prompt need at prefill, and why reserve them in one allocate call?",
        ],
        deep_dive=[
            "Which state does the engine keep for itself, and which already lives in the provided Sequence and BlockAllocator?",
        ],
        tradeoffs=[
            "Why does admission stop at the first request that does not fit instead of skipping to a shorter one?",
            "What goes wrong if a stopped sequence keeps its blocks until the end of the step?",
            "Why chain block hashes from the start of the sequence, and why stop reusing at the first miss?",
            "Why preempt the most recently admitted sequence, and when would swapping blocks out beat recomputing them?",
        ],
    ),
}
