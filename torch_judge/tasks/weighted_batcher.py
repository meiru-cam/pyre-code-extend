"""Weighted batches from several endless datasets, with checkpoint and resume, then uneven quotas."""

from ._interview import interview

# A registry of endless datasets that counts what the batcher opens and pulls, and a slow model
# written straight from the statement: it rebuilds each batch from per-dataset counts.
_HELPERS = r"""
import json, random

class Registry:
    def __init__(self, lengths):
        self.lengths, self.pulled, self.opened = dict(lengths), 0, []
    def get_iterator(self, name, offset=0):
        self.opened.append((name, offset))
        return self._run(name, offset)
    def _run(self, name, i):
        length = self.lengths[name]
        while True:
            self.pulled += 1
            yield f"{name}-{i % length}"
            i += 1

class Model:
    def __init__(self, lengths, weights, batch_size, allocation="deterministic", seed=0):
        self.lengths, self.weights, self.batch_size = lengths, weights, batch_size
        self.names, self.total = sorted(weights), sum(weights.values())
        self.count, self.allocation, self.rng = {n: 0 for n in self.names}, allocation, random.Random(seed)
    def take(self, name):
        self.count[name] += 1
        return f"{name}-{(self.count[name] - 1) % self.lengths[name]}"
    def pick(self):
        if self.allocation == "stochastic":
            draw, running = self.rng.randrange(self.total), 0
            for name in self.names:
                running += self.weights[name]
                if draw < running:
                    return name
        return min(self.names, key=lambda n: (self.count[n] / self.weights[n], n))
    def next_batch(self):
        if self.batch_size % self.total == 0:
            return [self.take(n) for n in self.names for _ in range(self.batch_size * self.weights[n] // self.total)]
        return [self.take(self.pick()) for _ in range(self.batch_size)]

def random_setup(rng, divisible):
    names = rng.sample(["alpha", "beta", "delta", "gamma", "omega"], rng.randint(1, 4))
    weights = {n: rng.randint(1, 5) for n in names}
    total = sum(weights.values())
    if divisible:
        batch_size = total * rng.randint(1, 3)
    else:
        batch_size = rng.choice([b for b in range(1, 3 * total + 2) if b % total] or [total])
    lengths = {n: rng.randint(1, 12) for n in names}
    return lengths, weights, batch_size

def raises_value_error(make):
    try:
        make()
    except ValueError:
        return True
    return False
"""

TASK = {
    "title": "Weighted Data Batcher",
    "difficulty": "Hard",
    "version": 1,
    "function_name": "DataBatcher",
    "description_en": r"""Build `DataBatcher(registry, weights, batch_size)`, which mixes examples from several named datasets into training batches, in proportion to a weight per dataset.

The requirement arrives in parts. Each part keeps every earlier behavior, so one `DataBatcher` class passes all parts at the end. Pass every test of the current part to reveal the next one.

**Rules for every part:**
- `registry.get_iterator(name, offset=0)` returns a new iterator over dataset `name` that starts at example number `offset`. It never ends: after a dataset's last example it starts again at the first. Opening it at any `offset` is fast.
- `weights` is a `dict` from dataset name (`str`) to a positive `int`. `W` is the sum of the weights. `batch_size` is a positive `int`.
- An empty `weights`, a weight that is not positive, or a `batch_size` that is not positive raises `ValueError` in the constructor.
- `next_batch()` returns a `list` of exactly `batch_size` examples. Each dataset continues where its previous example left off.
- Datasets are always handled in ascending order of name. Call them `d_1 < d_2 < … < d_K`, with weights `w_1 … w_K`.
- Every example the batcher pulls from an iterator ends up in a batch it returns: it never pulls an example to skip over it.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why this shows up in interviews:** the first batch is a few lines; the work is keeping the position small enough to save and exact enough to resume, and each later part adds one requirement.

**Where it is used:** pretraining mixes web, code and books by weight, and a run that resumes from a checkpoint must continue the same data stream rather than restart or skip it.

Adapted from the weighted data batcher question in Schuture/Anthropic-Interview-Notes (CC BY-NC 4.0), reworded, on one class. The open choice of deterministic rule is fixed to the smallest count-to-weight rule its reference code uses.""",
    "parts": [
        {
            "title": "Exact quotas",
            "description_en": r"""**Signature:** `DataBatcher(registry, weights, batch_size)`, `next_batch() -> list`

- In this part, `batch_size` is a multiple of `W`.
- Dataset `d_k` gets exactly `batch_size * w_k / W` examples per batch, one after another, in the order its iterator yields them.
- The batch is `d_1`'s examples, then `d_2`'s, and so on up to `d_K`.

**Example**, with each dataset's examples named `"<name>-<index>"`, `weights = {"wiki": 2, "arxiv": 3, "forum": 1}` (so `W = 6`) and `batch_size = 6`:
- the first `next_batch()` is `["arxiv-0", "arxiv-1", "arxiv-2", "forum-0", "wiki-0", "wiki-1"]`
- the second is `["arxiv-3", "arxiv-4", "arxiv-5", "forum-1", "wiki-2", "wiki-3"]`""",
        },
        {
            "title": "Checkpoint and resume",
            "description_en": r"""Keep Part 1. The batcher can now save its position and resume from it.

**Signature:** `state_dict() -> dict`, `load_state_dict(state) -> None`

- `state_dict()` returns a `dict` that survives `json.loads(json.dumps(state))` unchanged. Its size depends only on `K`, never on how many batches were produced.
- The returned `dict` is a snapshot: changing it afterwards does not change the batcher.
- `load_state_dict(state)` may be called on any batcher built with the same `registry` contents, `weights` and `batch_size`, whatever that batcher did before. Its next batches are then exactly the batches the saved batcher would have returned next.
- Restoring must not walk through examples already produced: reopen each dataset with `offset` instead.

**Example**, same setup as Part 1:
- after one `next_batch()`, `state = b.state_dict()`
- a new `b2 = DataBatcher(registry, weights, 6)` runs `b2.load_state_dict(state)`
- `b2.next_batch()` is `["arxiv-3", "arxiv-4", "arxiv-5", "forum-1", "wiki-2", "wiki-3"]`, the same as `b.next_batch()`""",
        },
        {
            "title": "Uneven quotas",
            "description_en": r"""Keep Parts 1–2. `batch_size` no longer has to be a multiple of `W`.

**Signature:** `DataBatcher(registry, weights, batch_size, allocation="deterministic", seed=0)`

- `allocation` is `"deterministic"` or `"stochastic"`; anything else raises `ValueError`. `seed` is only used by `"stochastic"`.
- When `batch_size` is a multiple of `W`, `next_batch()` follows Part 1 exactly, whatever `allocation` is.
- Otherwise the batch is filled one slot at a time: each slot picks a dataset and takes its next example. The batch lists the examples in the order they were picked.
- `count_k` is how many examples have been taken from `d_k` in total, across every batch so far, including batches before a restored checkpoint.
- `"deterministic"`: pick the dataset with the smallest `count_k / w_k`. On a tie, pick the first name in order.
- `"stochastic"`: the batcher owns one `random.Random(seed)`. Each slot draws `r = rng.randrange(W)` exactly once and picks the first `d_k` with `w_1 + … + w_k > r`. Nothing else may use this generator.
- Checkpoints must resume both allocations exactly. A stochastic batcher restored from a state continues the same draws, even if it was built with a different `seed`.

**Example**, same `weights` (`W = 6`) and `batch_size = 4`, deterministic:
- the first `next_batch()` is `["arxiv-0", "forum-0", "wiki-0", "arxiv-1"]`
- the second is `["wiki-1", "arxiv-2", "arxiv-3", "forum-1"]`""",
        },
    ],
    "hints": [
        {"level": 1, "kind": "questions", "content": "Each dataset is read at its own pace. What single number per dataset says where its next example is? Where does that dataset's iterator live between two next_batch calls, and what happens if you open a new one at offset 0 every batch?"},
        {"level": 2, "kind": "analysis", "content": "Sort the names once. Keep offsets = {name: 0} and a dict of open iterators. For each name in order, quota = batch_size * weights[name] // W; open the iterator at offsets[name] if it is not open yet, pull quota examples with next(), and add quota to offsets[name]. Validate weights and batch_size in __init__."},
    ],
    "model_connections": [
        "Pretraining data loaders mix sources such as web, code and math by weight, and keep that mixture the same across restarts.",
        "Resuming a training run from a checkpoint must restore the data loader's position, or the model sees some examples twice and others never.",
    ],
    "pro_con_analysis": {
        "pros": [
            "One integer offset per dataset is the whole position, so a checkpoint stays tiny and is plain JSON.",
            "Reopening at an offset makes a restore cost one open per dataset, independent of how far the run got.",
            "The smallest count-to-weight rule keeps every dataset within a constant of its ideal share at every slot, not only on average.",
        ],
        "cons": [
            "The deterministic rule repeats the same pattern inside every batch, which a stochastic mix avoids.",
            "A stochastic mix only matches the weights on average, and its checkpoint must carry the generator's internal state.",
            "Wrapping datasets repeat examples silently; a dataset shorter than its quota repeats inside one batch.",
        ],
    },
    "tests": [
        {"name": "Part 1: the worked example", "part": 1, "behavior": "state.invariant", "code": _HELPERS + r"""
weights = {"wiki": 2, "arxiv": 3, "forum": 1}
b = {fn}(Registry({"wiki": 50, "arxiv": 50, "forum": 50}), weights, 6)
assert b.next_batch() == ["arxiv-0", "arxiv-1", "arxiv-2", "forum-0", "wiki-0", "wiki-1"]
assert b.next_batch() == ["arxiv-3", "arxiv-4", "arxiv-5", "forum-1", "wiki-2", "wiki-3"]
"""},
        {"name": "Part 1: random weights and invalid input", "part": 1, "visibility": "unshown", "behavior": "contract.signature",
         "failure_message": "On random weights with batch_size a multiple of W, a batch differed from each dataset's batch_size * w / W next examples in ascending name order; or invalid weights or batch_size did not raise ValueError.",
         "code": _HELPERS + r"""
for seed in range(200):
    rng = random.Random(seed)
    lengths, weights, batch_size = random_setup(rng, divisible=True)
    b, model = {fn}(Registry(lengths), weights, batch_size), Model(lengths, weights, batch_size)
    for _ in range(rng.randint(1, 5)):
        assert b.next_batch() == model.next_batch(), (seed, weights, batch_size)
reg = Registry({"a": 3})
for bad in [({}, 2), ({"a": 0}, 2), ({"a": -2}, 2), ({"a": 1}, 0), ({"a": 1}, -1)]:
    assert raises_value_error(lambda: {fn}(reg, *bad)), f"DataBatcher(registry, {bad[0]}, {bad[1]}) should raise ValueError"
"""},
        {"name": "Part 1: no example is pulled and dropped", "part": 1, "visibility": "unshown", "behavior": "budget.enforcement",
         "failure_message": "The batcher pulled more examples from the registry than it returned; keep each dataset's iterator, or reopen it at its offset, instead of skipping forward.",
         "code": _HELPERS + r"""
reg = Registry({"a": 7, "b": 5, "c": 3})
b = {fn}(reg, {"a": 1, "b": 2, "c": 3}, 12)
returned = 0
for _ in range(50):
    returned += len(b.next_batch())
    assert reg.pulled == returned, f"pulled {reg.pulled} examples to return {returned}"
"""},
        {"name": "Part 2: the worked example", "part": 2, "behavior": "checkpoint.recovery", "code": _HELPERS + r"""
weights, lengths = {"wiki": 2, "arxiv": 3, "forum": 1}, {"wiki": 50, "arxiv": 50, "forum": 50}
b = {fn}(Registry(lengths), weights, 6)
b.next_batch()
state = b.state_dict()
b2 = {fn}(Registry(lengths), weights, 6)
b2.load_state_dict(state)
assert b2.next_batch() == ["arxiv-3", "arxiv-4", "arxiv-5", "forum-1", "wiki-2", "wiki-3"]
assert b.next_batch() == ["arxiv-3", "arxiv-4", "arxiv-5", "forum-1", "wiki-2", "wiki-3"]
"""},
        {"name": "Part 2: resume at every batch", "part": 2, "visibility": "unshown", "behavior": "checkpoint.recovery",
         "failure_message": "A batcher restored from a JSON round trip of state_dict() did not continue exactly like the batcher it came from, or changing the returned dict changed the batcher.",
         "code": _HELPERS + r"""
for seed in range(60):
    rng = random.Random(seed)
    lengths, weights, batch_size = random_setup(rng, divisible=True)
    n = rng.randint(2, 5)
    full = {fn}(Registry(lengths), weights, batch_size)
    expected = [full.next_batch() for _ in range(n)]
    for k in range(n + 1):
        first = {fn}(Registry(lengths), weights, batch_size)
        for _ in range(k):
            first.next_batch()
        state = first.state_dict()
        saved = json.loads(json.dumps(state))
        assert saved == state, "state_dict() must survive a JSON round trip unchanged"
        second = {fn}(Registry(lengths), weights, batch_size)
        for _ in range(rng.randint(0, 3)):  # whatever it did before
            second.next_batch()
        second.load_state_dict(saved)
        assert [second.next_batch() for _ in range(n - k)] == expected[k:], (seed, k)
b = {fn}(Registry({"a": 4, "b": 4}), {"a": 1, "b": 1}, 2)
b.next_batch()
snapshot = b.state_dict()
before = json.dumps(snapshot, sort_keys=True)
b.next_batch()
assert json.dumps(snapshot, sort_keys=True) == before, "a state returned earlier changed when the batcher moved on"
for value in snapshot.values():
    if isinstance(value, dict):
        for key in value:
            value[key] = 999
    elif isinstance(value, list):
        value.clear()
assert b.next_batch() == ["a-2", "b-2"], "changing a returned state changed the batcher"
"""},
        {"name": "Part 2: restore opens, never replays", "part": 2, "visibility": "unshown", "behavior": "budget.enforcement",
         "failure_message": "After many batches, the state grew with the run or restoring pulled examples it did not return; save one offset per dataset and reopen each iterator at it.",
         "code": _HELPERS + r"""
lengths, weights = {"a": 10 ** 9, "b": 10 ** 9, "c": 10 ** 9}, {"a": 1, "b": 1, "c": 2}
b = {fn}(Registry(lengths), weights, 4)
early = len(json.dumps(b.state_dict()))
for _ in range(20000):
    b.next_batch()
state = json.loads(json.dumps(b.state_dict()))
assert len(json.dumps(state)) <= early + 30, f"the state grew from {early} to {len(json.dumps(state))} characters"
reg = Registry(lengths)
resumed = {fn}(reg, weights, 4)
resumed.load_state_dict(state)
assert resumed.next_batch() == ["a-20000", "b-20000", "c-40000", "c-40001"]
assert reg.pulled == 4, f"restoring pulled {reg.pulled} examples to return 4"
"""},
        {"name": "Part 3: the worked example", "part": 3, "behavior": "state.invariant", "code": _HELPERS + r"""
weights = {"wiki": 2, "arxiv": 3, "forum": 1}
b = {fn}(Registry({"wiki": 50, "arxiv": 50, "forum": 50}), weights, 4, allocation="deterministic")
assert b.next_batch() == ["arxiv-0", "forum-0", "wiki-0", "arxiv-1"]
assert b.next_batch() == ["wiki-1", "arxiv-2", "arxiv-3", "forum-1"]
"""},
        {"name": "Part 3: deterministic picks and resume", "part": 3, "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "With deterministic allocation, a pick differed from the smallest total count / weight with ties to the first name, a restored batcher did not continue exactly, or an unknown allocation did not raise ValueError.",
         "code": _HELPERS + r"""
for seed in range(150):
    rng = random.Random(seed)
    lengths, weights, batch_size = random_setup(rng, divisible=seed % 4 == 0)
    b = {fn}(Registry(lengths), weights, batch_size, allocation="deterministic")
    model = Model(lengths, weights, batch_size)
    for _ in range(rng.randint(1, 6)):
        if rng.random() < 0.4:
            state = json.loads(json.dumps(b.state_dict()))
            b = {fn}(Registry(lengths), weights, batch_size, allocation="deterministic")
            b.load_state_dict(state)
        assert b.next_batch() == model.next_batch(), (seed, weights, batch_size)
lengths, weights = {"x": 9, "y": 9}, {"x": 2, "y": 1}
same = {fn}(Registry(lengths), weights, 6, allocation="stochastic", seed=4)
assert same.next_batch() == ["x-0", "x-1", "x-2", "x-3", "y-0", "y-1"], "a multiple of W follows Part 1 for both allocations"
assert raises_value_error(lambda: {fn}(Registry(lengths), weights, 5, allocation="uniform"))
"""},
        {"name": "Part 3: stochastic draws and resume", "part": 3, "visibility": "unshown", "behavior": "checkpoint.recovery",
         "failure_message": "With stochastic allocation, a pick differed from one rng.randrange(W) per slot looked up in the running weight sums, or a restored batcher (built with another seed) did not continue the same draws.",
         "code": _HELPERS + r"""
for seed in range(150):
    rng = random.Random(seed)
    lengths, weights, batch_size = random_setup(rng, divisible=seed % 4 == 0)
    b = {fn}(Registry(lengths), weights, batch_size, allocation="stochastic", seed=seed)
    model = Model(lengths, weights, batch_size, "stochastic", seed)
    for _ in range(rng.randint(1, 6)):
        if rng.random() < 0.4:
            state = json.loads(json.dumps(b.state_dict()))
            b = {fn}(Registry(lengths), weights, batch_size, allocation="stochastic", seed=seed + 1000)
            b.load_state_dict(state)
        assert b.next_batch() == model.next_batch(), (seed, weights, batch_size)
"""},
    ],
    "solution": r'''# Adapted from Schuture/Anthropic-Interview-Notes (code under the MIT License).
import random
from bisect import bisect_right


class DataBatcher:
    def __init__(self, registry, weights, batch_size, allocation="deterministic", seed=0):
        if not weights or any(w <= 0 for w in weights.values()):
            raise ValueError("weights must be a non-empty dict of positive integers")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if allocation not in ("deterministic", "stochastic"):
            raise ValueError(f"unknown allocation {allocation!r}")
        self._registry = registry
        self._names = sorted(weights)  # the one order used everywhere
        self._weights = dict(weights)
        self._total = sum(weights.values())
        self._batch_size = batch_size
        self._allocation = allocation
        self._offsets = {name: 0 for name in self._names}  # examples taken so far, per dataset
        self._iters = {}  # live iterators, opened lazily at the current offset
        self._prefix, running = [], 0
        for name in self._names:
            running += self._weights[name]
            self._prefix.append(running)
        self._rng = random.Random(seed)

    def _take(self, name, count):
        it = self._iters.get(name)
        if it is None:
            it = self._iters[name] = self._registry.get_iterator(name, self._offsets[name])
        self._offsets[name] += count
        return [next(it) for _ in range(count)]

    def next_batch(self):
        if self._batch_size % self._total == 0:
            batch = []
            for name in self._names:
                batch += self._take(name, self._batch_size * self._weights[name] // self._total)
            return batch
        pick = self._pick_deterministic if self._allocation == "deterministic" else self._pick_stochastic
        return [self._take(pick(), 1)[0] for _ in range(self._batch_size)]

    def _pick_deterministic(self):
        best = self._names[0]
        for name in self._names[1:]:
            # offsets[name] / weights[name] < offsets[best] / weights[best], in integers;
            # names are visited in order, so a tie keeps the earlier name
            if self._offsets[name] * self._weights[best] < self._offsets[best] * self._weights[name]:
                best = name
        return best

    def _pick_stochastic(self):
        return self._names[bisect_right(self._prefix, self._rng.randrange(self._total))]

    def state_dict(self):
        state = {"offsets": dict(self._offsets)}
        if self._allocation == "stochastic":
            version, internal, gauss = self._rng.getstate()
            state["rng"] = [version, list(internal), gauss]  # lists, so the state is plain JSON
        return state

    def load_state_dict(self, state):
        self._offsets = dict(state["offsets"])
        self._iters = {}  # reopened lazily at the restored offsets, never replayed
        if self._allocation == "stochastic":
            version, internal, gauss = state["rng"]
            self._rng.setstate((version, tuple(internal), gauss))
''',
    "interview_questions": interview(
        concept=[
            "What does the batcher need to remember per dataset between two calls, and why one value per dataset?",
            "Why is it a bug to open each dataset at offset 0 and skip forward to where it was?",
        ],
        deep_dive=[
            "What does one next_batch call cost, and what does the batcher keep between calls?",
        ],
        tradeoffs=[
            "Why are the offsets enough to resume, and why not save the open iterators too?",
            "Why must a stochastic checkpoint carry the generator's current state rather than the seed?",
            "With the smallest count-to-weight rule, how far can a dataset's count drift from its ideal share, and why?",
            "How would you split the datasets across data-parallel ranks so no two ranks read the same example?",
        ],
    ),
}
