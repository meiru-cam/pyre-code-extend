"""Per-file async serialization of agent file edits, with cancellation safety."""

from ._interview import interview

TASK = {
    "title": "Per-File Mutation Queue",
    "difficulty": "Hard",
    "version": 1,
    "function_name": "FileMutationQueue",
    "description_en": r"""When an agent runs several tool calls in parallel, two edits to the same file must not interleave, but edits to different files should still run concurrently. Implement the queue that enforces this.

**Signature:** `class FileMutationQueue` with `async run(path, fn)` and `active_keys() -> set[str]`.

**`run(path, fn)`** — `fn` is a zero-argument async function. Return its result or propagate its exception.
- The queue key is `os.path.realpath(os.path.abspath(path))`, so relative paths and symlinks that reach the same file share one key.
- Calls with the same key run `fn` strictly one at a time, in the order `run` was called. Calls with different keys do not wait for each other.
- A failing `fn` releases the key for the next caller.
- If a call is cancelled while it is still waiting its turn, it never runs `fn`, and the calls queued behind it still wait for every earlier call to finish; they must neither deadlock nor start early.

**`active_keys()`** — the set of keys that currently have a running or waiting call. Once every call for a key has finished, the key is gone.

**Constraints:** Use only `asyncio` and the standard library. Everything runs on one event loop.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why serialize per file.** An edit tool reads a file, applies replacements and writes it back. Two concurrent edits to the same file each read the old version, and the second write silently discards the first. Serializing by resolved path keeps parallel tool execution safe.

**Why resolve symlinks.** `src/app.py` and `./link/app.py` can name the same inode; keying by the raw string would let them race.

**The cancellation trap.** Pi is written in TypeScript, where promises cannot be cancelled, so a simple chain of promises is enough there. An `asyncio` task can be cancelled while waiting; if its slot in the chain is released at that moment, the next caller starts while an earlier edit is still running, and if it is never released, every later caller hangs.""",
    "advisory_prerequisites": ["async_agent_rollout"],
    "hints": [
        {"level": 1, "kind": "questions", "content": "What does each caller need to wait for, and what does it need to signal? If a waiting caller is cancelled, when is it safe to signal the caller behind it? How do you know you are the last caller for a key?"},
        {"level": 2, "kind": "analysis", "content": "Keep key -> the Event of the most recent caller. Each run records prev = tails.get(key), stores its own Event as the new tail, awaits prev.wait(), runs fn in try/finally and sets its Event. If cancelled while waiting, schedule a task that waits for prev and then sets its own Event. Delete the key only if the tail is still your Event."},
    ],
    "model_connections": [
        "Pi's withFileMutationQueue guards its write and edit tools when tool calls run in parallel; Claude Code and Codex similarly serialize conflicting file operations.",
    ],
    "pro_con_analysis": {
        "pros": ["Keeps parallel tool execution fast while making same-file edits sequential and deterministic."],
        "cons": ["It only protects writes that go through the queue; a shell command can still modify the file concurrently, and one slow edit blocks every later edit to that file."],
    },
    "sources": [
        {"kind": "code", "url": "https://github.com/badlogic/pi-mono", "commit": "6f7551516b84278eb9da1c340c8e7bc66be1a6ba", "path": "packages/coding-agent/src/core/tools/file-mutation-queue.ts", "symbol": "withFileMutationQueue and getMutationQueueKey", "license": "MIT", "adapted": "Realpath-keyed per-file chaining, release in finally, and deleting the key only when this caller is still the tail.", "simplifications": "Adds asyncio cancellation handling that the promise-based original does not need; no async key registration queue because realpath is synchronous here."},
    ],
    "tests": [
        {"name": "Same file serializes, different files overlap", "behavior": "scheduler.concurrency", "code": r"""
import asyncio, os, tempfile, time
async def main():
    q = {fn}()
    d = tempfile.mkdtemp()
    a, b = os.path.join(d, "a.txt"), os.path.join(d, "b.txt")
    log = []
    async def work(tag):
        log.append(tag + "+"); await asyncio.sleep(0.1); log.append(tag + "-"); return tag
    start = time.perf_counter()
    res = await asyncio.wait_for(asyncio.gather(q.run(a, lambda: work("a1")), q.run(a, lambda: work("a2")), q.run(b, lambda: work("b1"))), timeout=5)
    elapsed = time.perf_counter() - start
    assert res == ["a1", "a2", "b1"], res
    assert log.index("a1-") < log.index("a2+"), log
    assert log.index("b1+") < log.index("a1-"), log
    assert elapsed < 0.3, elapsed
    assert q.active_keys() == set()
asyncio.run(main())
"""},
        {"name": "Symlinks and relative paths share one key", "visibility": "unshown", "behavior": "scheduler.concurrency", "failure_message": "Resolve the path with realpath so every alias of one file uses the same queue.", "code": r"""
import asyncio, os, tempfile
async def main():
    q = {fn}()
    d = os.path.realpath(tempfile.mkdtemp())
    target = os.path.join(d, "real.py")
    open(target, "w").close()
    link = os.path.join(d, "alias.py")
    os.symlink(target, link)
    running, peak = [0], [0]
    async def work():
        running[0] += 1; peak[0] = max(peak[0], running[0])
        await asyncio.sleep(0.03)
        running[0] -= 1
    cwd = os.getcwd()
    os.chdir(d)
    try:
        await asyncio.wait_for(asyncio.gather(q.run(target, work), q.run(link, work), q.run("real.py", work), q.run(os.path.join(d, ".", "real.py"), work)), timeout=5)
    finally:
        os.chdir(cwd)
    assert peak[0] == 1, peak
    assert q.active_keys() == set()
asyncio.run(main())
"""},
        {"name": "Order, failures and key cleanup under a seeded load", "visibility": "unshown", "behavior": "state.invariant", "failure_message": "Run same-key calls in call order, release the key after a failure, and forget keys once idle.", "code": r"""
import asyncio, os, random, tempfile
async def main():
    for seed in (1, 12, 45):
        rng = random.Random(seed)
        q = {fn}()
        d = tempfile.mkdtemp()
        paths = [os.path.join(d, f"f{i}") for i in range(3)]
        started = {p: [] for p in paths}
        active = {p: 0 for p in paths}
        async def job(p, i, fail):
            active[p] += 1
            assert active[p] == 1, "two calls ran at once for one key"
            started[p].append(i)
            await asyncio.sleep(rng.random() * 0.01)
            active[p] -= 1
            if fail:
                raise RuntimeError(i)
            return i
        calls = []
        for i in range(30):
            p = rng.choice(paths)
            fail = rng.random() < 0.2
            calls.append((p, i, fail))
        results = await asyncio.wait_for(asyncio.gather(*(q.run(p, (lambda p=p, i=i, f=fail: job(p, i, f))) for p, i, fail in calls), return_exceptions=True), timeout=5)
        for (p, i, fail), r in zip(calls, results):
            assert (isinstance(r, RuntimeError) and r.args == (i,)) if fail else r == i, (seed, i, r)
        for p in paths:
            assert started[p] == [i for pp, i, _ in calls if pp == p], (seed, started[p])
        assert q.active_keys() == set(), q.active_keys()
    q = {fn}()
    path = os.path.join(tempfile.mkdtemp(), "late")
    log = []
    async def step(tag, delay):
        log.append(tag + "+"); await asyncio.sleep(delay); log.append(tag + "-")
    t1 = asyncio.create_task(q.run(path, lambda: step("a", 0.03)))
    await asyncio.sleep(0)
    t2 = asyncio.create_task(q.run(path, lambda: step("b", 0.1)))
    await asyncio.sleep(0.06)
    t3 = asyncio.create_task(q.run(path, lambda: step("c", 0.0)))
    await asyncio.wait_for(asyncio.gather(t1, t2, t3), timeout=5)
    assert log == ["a+", "a-", "b+", "b-", "c+", "c-"], log
asyncio.run(main())
"""},
        {"name": "Cancelling a waiting call neither deadlocks nor lets the next call jump ahead", "visibility": "unshown", "behavior": "scheduler.concurrency", "failure_message": "A caller cancelled while waiting must hand its turn on only after every earlier call finishes.", "code": r"""
import asyncio, os, tempfile
async def main():
    q = {fn}()
    path = os.path.join(tempfile.mkdtemp(), "x")
    log = []
    first_started = asyncio.Event()
    async def first():
        log.append("first+"); first_started.set(); await asyncio.sleep(0.1); log.append("first-")
    async def second():
        log.append("second ran")
    async def third():
        log.append("third+")
    t1 = asyncio.create_task(q.run(path, first))
    await first_started.wait()
    t2 = asyncio.create_task(q.run(path, second))
    await asyncio.sleep(0.01)
    t3 = asyncio.create_task(q.run(path, third))
    await asyncio.sleep(0.01)
    t2.cancel()
    await asyncio.sleep(0.01)
    assert "third+" not in log, log
    await asyncio.wait_for(asyncio.gather(t1, t3), timeout=1.0)
    assert log == ["first+", "first-", "third+"], log
    assert t2.cancelled()
    await asyncio.sleep(0.01)
    assert q.active_keys() == set(), q.active_keys()
asyncio.run(main())
"""},
    ],
    "solution": '''import asyncio
import os


class FileMutationQueue:
    def __init__(self):
        self._tails = {}
        self._background = set()

    def active_keys(self):
        return set(self._tails)

    def _release(self, key, done):
        done.set()
        if self._tails.get(key) is done:
            del self._tails[key]

    async def run(self, path, fn):
        key = os.path.realpath(os.path.abspath(path))
        prev = self._tails.get(key)
        done = asyncio.Event()
        self._tails[key] = done
        if prev is not None:
            try:
                await prev.wait()
            except asyncio.CancelledError:
                async def hand_over():
                    await prev.wait()
                    self._release(key, done)

                task = asyncio.ensure_future(hand_over())
                self._background.add(task)
                task.add_done_callback(self._background.discard)
                raise
        try:
            return await fn()
        finally:
            self._release(key, done)
''',
    "interview_questions": interview(
        concept=[
            "Why do parallel tool calls in a coding agent need a per-file lock, and what goes wrong without one?",
            "Why key the queue by the resolved real path rather than the path string the model sent?",
        ],
        deep_dive=[
            "Walk through how each caller waits for the previous one and signals the next. Why does the key get deleted only if you are still the tail?",
            "What happens in asyncio if a caller is cancelled while waiting, and how do you hand its turn on safely?",
            "Why must a failing edit still release the key?",
        ],
        tradeoffs=[
            "A per-file queue versus one global lock versus optimistic concurrency that re-reads and retries on conflict: throughput and correctness?",
            "Should read operations also go through the queue? What anomalies can a model observe if they do not?",
        ],
    ),
}
