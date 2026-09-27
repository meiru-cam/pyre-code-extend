"""Schedule asynchronous rollouts with bounded concurrency and per-sample timeouts."""

from ._interview import interview

TASK = {
    "title": "Async Agent Rollout Scheduler",
    "difficulty": "Hard",
    "version": 1,
    "function_name": "async_agent_rollout",
    "description_en": r"""Run independent asynchronous rollouts with bounded concurrency.

**Signature:** `async async_agent_rollout(rollout_fn, items, max_concurrency, timeout) -> list[dict]`

`rollout_fn(item)` is an async function. Run one task per input item, allow at
most `max_concurrency` calls in flight, and wrap each call in a timeout in
seconds. Return one dictionary per item, in the original input order. A
successful result is `{"item": item, "status": "ok", "value": value}`;
a timeout is `{"item": item, "status": "timeout", "value": None}`; any
ordinary exception is `{"item": item, "status": "error", "value": None}`.
Do not cancel or reorder successful siblings when one rollout is slow.

Raise `ValueError` for a non-positive concurrency or timeout.

────────────────

**Background — context only.** Fully asynchronous rollout keeps a warm pool
of long-running agent trajectories while training consumes completed groups.
This is different from asynchronous optimizer steps: the trajectory still must
be a complete, labelled sample before it enters the trainer. A semaphore is
the backpressure boundary; per-item timeout prevents one hung environment from
holding the whole batch forever.""",
    "advisory_prerequisites": ["agentic_rollout_loop", "rl_eval_loop"],
    "hints": [
        {"level": 1, "kind": "questions", "content": "Where is the concurrency cap enforced? What should happen to a timeout while other tasks are still running? How will you recover the original input order after tasks finish out of order?"},
        {"level": 2, "kind": "analysis", "content": "Create one coroutine per item, guard the rollout call with an asyncio.Semaphore, and use asyncio.wait_for inside the guard. Gather all wrappers with return_exceptions avoided because each wrapper classifies its own timeout and ordinary exception. Sort or gather by input index, not completion order."},
    ],
    "model_connections": [
        "Slime's fully-async rollout keeps background trajectories in flight and returns completed samples to a data buffer.",
        "VERL's agent workers also schedule independent AgentLoop coroutines; the exercise isolates the same backpressure and timeout boundary.",
    ],
    "sources": [{
        "kind": "code",
        "url": "https://github.com/THUDM/slime",
        "commit": "5bae5bb7928d65906e24e99f5b5d3f99a4daaf93",
        "path": "slime/rollout/fully_async_rollout.py",
        "symbol": "AsyncRolloutWorker",
        "license": "Apache-2.0",
        "adapted": "The scheduler keeps independent trajectories in flight with bounded concurrency and returns completed samples.",
        "simplifications": "Uses asyncio tasks instead of Ray actors and classifies timeout/error results locally rather than requeueing into a Data Buffer."
    }],
    "tests": [
        {"name": "Preserves input order while running concurrently", "behavior": "scheduler.concurrency", "code": r"""
import asyncio
async def rollout(item):
    await asyncio.sleep(0.02 * (3 - item))
    return item * 10
out = asyncio.run({fn}(rollout, [0, 1, 2], max_concurrency=3, timeout=1.0))
assert out == [
    {"item": 0, "status": "ok", "value": 0},
    {"item": 1, "status": "ok", "value": 10},
    {"item": 2, "status": "ok", "value": 20},
]
"""},
        {"name": "Enforces the concurrency cap", "visibility": "unshown", "behavior": "scheduler.concurrency", "failure_message": "More than max_concurrency rollouts were in flight at once.", "code": r"""
import asyncio
active = 0; peak = 0
async def rollout(item):
    global active, peak
    active += 1; peak = max(peak, active)
    await asyncio.sleep(0.01)
    active -= 1
    return item
out = asyncio.run({fn}(rollout, list(range(8)), max_concurrency=2, timeout=1.0))
assert peak <= 2 and all(x["status"] == "ok" for x in out)
"""},
        {"name": "Seeded item counts preserve result order under bounded overlap", "visibility": "unshown", "behavior": "scheduler.concurrency", "failure_message": "A different input length must still preserve all item identities and the concurrency bound.", "code": r"""
import asyncio, random
for seed in (11, 23, 37):
    count=random.Random(seed).randrange(4, 8)
    active=0; peak=0
    async def rollout(item):
        global active, peak
        active += 1; peak=max(peak, active)
        await asyncio.sleep(0.001 * (count - item))
        active -= 1
        return item * 7
    got=asyncio.run({fn}(rollout, list(range(count)), 2, 1.0))
    expected=[{"item":i,"status":"ok","value":i*7} for i in range(count)]
    assert got == expected, (seed, got)
    assert 1 <= peak <= 2, (seed, peak)
"""},
        {"name": "Classifies timeout and ordinary errors", "visibility": "unshown", "behavior": "retry.classification", "failure_message": "Timeouts and ordinary rollout failures must be represented per item instead of aborting siblings.", "code": r"""
import asyncio
async def rollout(item):
    if item == "slow": await asyncio.sleep(0.05)
    if item == "bad": raise RuntimeError("boom")
    return item
out = asyncio.run({fn}(rollout, ["ok", "slow", "bad"], max_concurrency=3, timeout=0.01))
assert [x["status"] for x in out] == ["ok", "timeout", "error"]
assert out[1]["value"] is None and out[2]["value"] is None
"""},
        {"name": "Rejects invalid scheduler settings", "visibility": "unshown", "behavior": "contract.signature", "failure_message": "The scheduler accepted a non-positive concurrency or timeout.", "code": r"""
import asyncio
async def rollout(item): return item
for kwargs in ({"max_concurrency": 0, "timeout": 1}, {"max_concurrency": 1, "timeout": 0}):
    try: asyncio.run({fn}(rollout, [1], **kwargs))
    except ValueError: pass
    else: raise AssertionError("invalid setting accepted")
"""},
    ],
    "solution": '''async def async_agent_rollout(rollout_fn, items, max_concurrency, timeout):
    import asyncio
    if not isinstance(max_concurrency, int) or isinstance(max_concurrency, bool) or max_concurrency <= 0:
        raise ValueError("max_concurrency must be positive")
    if timeout <= 0:
        raise ValueError("timeout must be positive")

    async def run_one(index, item, semaphore):
        async with semaphore:
            try:
                value = await asyncio.wait_for(rollout_fn(item), timeout=timeout)
            except asyncio.TimeoutError:
                return index, {"item": item, "status": "timeout", "value": None}
            except Exception:
                return index, {"item": item, "status": "error", "value": None}
            return index, {"item": item, "status": "ok", "value": value}

    async def run_all():
        semaphore = asyncio.Semaphore(max_concurrency)
        results = await asyncio.gather(
            *(run_one(index, item, semaphore) for index, item in enumerate(items))
        )
        return [entry for _, entry in sorted(results)]

    return await run_all()
''',
    "interview_questions": interview(
        concept=[
            'Why run agent rollouts asynchronously, and what is the bottleneck in synchronous rollout?',
            'What does bounded concurrency protect?',
        ],
        deep_dive=[
            'How do you bound concurrency with asyncio, for example a semaphore?',
            'How do you apply a per-call timeout without cancelling siblings?',
            'How do you return results in input order when they finish out of order?',
        ],
        tradeoffs=[
            'Timeouts turn slow samples into failures. What bias does that add to training data?',
            'Async rollouts versus batched synchronous generation: what are the throughput and reproducibility trade-offs?',
        ],
    ),
}
