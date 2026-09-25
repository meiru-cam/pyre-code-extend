"""Consume completed agent rollouts from a buffer without waiting for the slowest sample."""

TASK = {
    "title": "Fully-Async Rollout Buffer",
    "difficulty": "Hard",
    "version": 1,
    "function_name": "fully_async_rollout_buffer",
    "description_en": r"""Stream completed rollouts into training batches while slower samples remain in flight.

**Signature:** `async fully_async_rollout_buffer(rollout_fn, train_fn, items, max_concurrency, train_batch_size, timeout) -> dict`

`rollout_fn(item)` is an async trajectory producer and `train_fn(batch)` is
an async training callback. Launch one rollout per item, with at most
`max_concurrency` producers in flight. As soon as `train_batch_size`
successful trajectories are available, call `train_fn(batch)` immediately;
do not wait for every item in the input list. A batch is a list of dictionaries
with `item`, `status` equal to `"ok"`, and `value`. Keep items in their
original order within each emitted batch. At the end, train one final partial
batch if successful trajectories remain.

Return exactly `{"trained_batches": batches, "failures": failures}`. A failure
entry has `item`, `status` equal to `"timeout"` or `"error"`, and
`value: None`; failed items never enter a training batch. Raise
`ValueError` for non-positive concurrency, batch size, or timeout.

────────────────

**Background — context only.** A normal async batch still waits for the
slowest trajectory before training. Fully-async rollout keeps a warm set of
in-flight trajectories and uses a Data Buffer as the handoff: completed
samples can trigger training while other samples continue generating. This
improves tail latency but introduces partial batches, failed samples, and
weight-staleness questions for the next exercise. The callback is the seam
where an actual Slime or actor-learner trainer would consume the batch.""",
    "advisory_prerequisites": ["async_agent_rollout", "rollout_train_boundary", "grpo_train_step"],
    "hints": [
        {"level": 1, "kind": "questions", "content": "What event should trigger train_fn: all tasks finishing or the buffer reaching its batch size? Which records are eligible for training? How do you prove the first batch was trained before the slowest rollout completed?"},
        {"level": 2, "kind": "analysis", "content": "Create bounded producer tasks and consume them with asyncio.as_completed. Put only successful records into a buffer; whenever its length reaches train_batch_size, remove one batch, sort that batch by original index, and await train_fn. After all producers finish, flush a final partial batch. Keep failures separate and never let one exception cancel siblings."},
    ],
    "model_connections": [
        "Slime's fully_async_rollout keeps a background worker warm and redirects completed groups through its Data Buffer instead of waiting for the slowest group.",
        "The training callback stands in for an actor/critic update; the exercise deliberately leaves weight synchronization and staleness to a separate question.",
    ],
    "sources": [{
        "kind": "code",
        "url": "https://github.com/THUDM/slime",
        "commit": "5bae5bb7928d65906e24e99f5b5d3f99a4daaf93",
        "path": "slime/rollout/fully_async_rollout.py",
        "symbol": "generate_rollout_fully_async",
        "license": "Apache-2.0",
        "adapted": "The exercise preserves the ready-sample buffer and early batch emission boundary.",
        "simplifications": "Uses asyncio tasks and a callback instead of Ray actors, SGLang engines, Sample groups, and weight-update abort signals."
    }],
    "tests": [
        {"name": "Trains a ready batch before the slowest rollout finishes", "behavior": "scheduler.concurrency", "code": r"""
import asyncio
events = []
async def rollout(item):
    await asyncio.sleep(0.06 if item == 0 else 0.005)
    events.append(("done", item))
    return item * 10
async def train(batch):
    events.append(("train", [x["item"] for x in batch]))
out = asyncio.run({fn}(rollout, train, [0, 1, 2, 3], 3, 2, 1.0))
assert out["failures"] == []
assert [x["item"] for x in out["trained_batches"][0]] == [1, 2], out
assert events.index(("train", [1, 2])) < events.index(("done", 0)), events
assert sorted(x["item"] for batch in out["trained_batches"] for x in batch) == [0, 1, 2, 3]
"""},
        {"name": "Never exceeds the producer concurrency cap", "visibility": "unshown", "behavior": "scheduler.concurrency", "failure_message": "The rollout buffer allowed more than max_concurrency producers in flight.", "code": r"""
import asyncio
active = 0; peak = 0
async def rollout(item):
    global active, peak
    active += 1; peak = max(peak, active)
    await asyncio.sleep(0.01)
    active -= 1
    return item
async def train(batch): pass
out = asyncio.run({fn}(rollout, train, list(range(8)), 2, 3, 1.0))
assert peak <= 2
"""},
        {"name": "Training releases a rollout still waiting for its batch", "visibility": "unshown", "behavior": "scheduler.concurrency", "failure_message": "A ready training batch must run before all rollouts finish; otherwise the slow rollout cannot complete.", "code": r"""
import asyncio
async def scenario():
    trained = asyncio.Event()
    batches = []
    async def rollout(item):
        if item == 0:
            await trained.wait()
        return item
    async def train(batch):
        batches.append([entry["item"] for entry in batch])
        trained.set()
    result = await asyncio.wait_for({fn}(rollout, train, [0, 1, 2], 3, 2, 1.0), 0.25)
    assert batches == [[1, 2], [0]] and result["failures"] == []
asyncio.run(scenario())
"""},
        {"name": "Flushes a final partial batch", "visibility": "unshown", "behavior": "state.invariant", "failure_message": "Successful samples left in the buffer were dropped when fewer than train_batch_size remained.", "code": r"""
import asyncio
async def rollout(item): return item
async def train(batch): pass
out = asyncio.run({fn}(rollout, train, [1, 2, 3], 3, 2, 1.0))
assert [[x["item"] for x in batch] for batch in out["trained_batches"]] == [[1, 2], [3]], out
"""},
        {"name": "Seeded producer counts conserve successful items across batches", "visibility": "unshown", "behavior": "state.invariant", "failure_message": "Every successful sample must be trained exactly once, including a final partial batch.", "code": r"""
import asyncio, random
for seed in (11, 23, 37):
    count=random.Random(seed).randrange(4, 8)
    calls=[]
    async def rollout(item):
        await asyncio.sleep(0.002 * (count - item))
        return item * 7
    async def train(batch): calls.append(batch)
    got=asyncio.run({fn}(rollout, train, list(range(count)), 3, 2, 1.0))
    assert got["failures"] == []
    assert got["trained_batches"] == calls
    flattened=[entry for batch in calls for entry in batch]
    assert sorted(entry["item"] for entry in flattened) == list(range(count))
    assert all(entry == {"item":entry["item"],"status":"ok","value":entry["item"]*7}
               for entry in flattened)
    assert all(len(batch) == 2 for batch in calls[:-1])
    assert len(calls[-1]) == (1 if count % 2 else 2)
"""},
        {"name": "Separates timeout and ordinary failures", "visibility": "unshown", "behavior": "retry.classification", "failure_message": "Failed rollouts must be reported separately and must never be sent to train_fn.", "code": r"""
import asyncio
async def rollout(item):
    if item == "slow": await asyncio.sleep(0.05)
    if item == "bad": raise RuntimeError("boom")
    return item
trained = []
async def train(batch): trained.extend(x["item"] for x in batch)
out = asyncio.run({fn}(rollout, train, ["ok", "slow", "bad"], 3, 2, 0.01))
assert [x["item"] for x in out["failures"]] == ["slow", "bad"], out
assert trained == ["ok"], trained
"""},
        {"name": "Rejects invalid buffer settings", "visibility": "unshown", "behavior": "contract.signature", "failure_message": "The buffer accepted a non-positive concurrency, batch size, or timeout.", "code": r"""
import asyncio
async def rollout(item): return item
async def train(batch): pass
for kwargs in (
    {"max_concurrency": 0, "train_batch_size": 1, "timeout": 1},
    {"max_concurrency": 1, "train_batch_size": 0, "timeout": 1},
    {"max_concurrency": 1, "train_batch_size": 1, "timeout": 0},
):
    try: asyncio.run({fn}(rollout, train, [1], **kwargs))
    except ValueError: pass
    else: raise AssertionError("invalid setting accepted")
"""},
    ],
    "solution": '''async def fully_async_rollout_buffer(rollout_fn, train_fn, items, max_concurrency, train_batch_size, timeout):
    import asyncio
    if not isinstance(max_concurrency, int) or isinstance(max_concurrency, bool) or max_concurrency <= 0:
        raise ValueError("max_concurrency must be positive")
    if not isinstance(train_batch_size, int) or isinstance(train_batch_size, bool) or train_batch_size <= 0:
        raise ValueError("train_batch_size must be positive")
    if timeout <= 0:
        raise ValueError("timeout must be positive")

    async def produce(index, item, semaphore):
        async with semaphore:
            try:
                value = await asyncio.wait_for(rollout_fn(item), timeout=timeout)
            except asyncio.TimeoutError:
                return index, {"item": item, "status": "timeout", "value": None}
            except Exception:
                return index, {"item": item, "status": "error", "value": None}
            return index, {"item": item, "status": "ok", "value": value}

    semaphore = asyncio.Semaphore(max_concurrency)
    tasks = [
        asyncio.create_task(produce(index, item, semaphore))
        for index, item in enumerate(items)
    ]
    buffer, failures, trained_batches = [], [], []

    async def emit_ready(force=False):
        while len(buffer) >= train_batch_size or (force and buffer):
            batch = sorted(buffer[:train_batch_size], key=lambda entry: entry[0])
            del buffer[:train_batch_size]
            public_batch = [entry[1] for entry in batch]
            await train_fn(public_batch)
            trained_batches.append(public_batch)

    for completed in asyncio.as_completed(tasks):
        index, result = await completed
        if result["status"] == "ok":
            buffer.append((index, result))
            await emit_ready()
        else:
            failures.append((index, result))

    await emit_ready(force=True)
    failures = [result for _, result in sorted(failures)]
    return {"trained_batches": trained_batches, "failures": failures}
''',
}
