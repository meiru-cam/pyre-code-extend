"""Hand streamed outputs from the engine to a slower consumer by merging instead of queueing."""

from ._interview import interview

TASK = {
    "title": "Streaming Output Collector",
    "difficulty": "Medium",
    "version": 1,
    "function_name": "OutputCollector",
    "description_en": r"""In vLLM's async server, the engine loop produces an output for every request at every step, while each HTTP response is written by its own task that may fall behind. Instead of an unbounded queue, each request has a one-slot collector: when the consumer is slow, new outputs are merged into the pending one. Implement it.

**Signature:** `OutputCollector(aggregate: bool)` with `put(item)`, `get_nowait()` and `async get()`.

**Outputs** are dicts `{"finished": bool, "outputs": [completion, ...]}`, where a completion is `{"index": int, "text": str, "token_ids": list[int], "finish_reason": str | None}`. An item passed to `put` is an output or an `Exception`.

**`put(item)`** never blocks and never modifies the dicts it is given.
- If nothing is pending, or `item` is an Exception, `item` (a deep copy if it is an output) becomes the pending item and the consumer is woken.
- If an output is pending and `item` is an output, merge `item` into the pending output: `finished` becomes `pending or new`. For each new completion, find the pending completion with the same `index`. If `aggregate`, append its `text` and `token_ids` to that completion and take its `finish_reason`; otherwise replace that completion with a copy of the new one. A new completion with an index not yet pending is appended.
- If an Exception is pending and `item` is an output, drop `item`.

**`get_nowait()`** returns the pending output and clears the slot, or returns None if nothing is pending. If the pending item is an Exception, clear the slot and raise it.

**`async get()`** waits until something is pending, then behaves like `get_nowait()`.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why merge instead of queue.** The engine must never block on a slow client: every other request shares the engine loop. With deltas, merging keeps memory constant per request and lets a slow client receive fewer, larger chunks with the same final text.

**Aggregate versus replace.** In delta mode each output carries only new text, so merges append. In cumulative mode each output already carries the full text so far, so the newest one simply replaces the older.

**Why merge by index.** With n > 1 samples per request, outputs for different samples arrive interleaved; merging by position would splice one sample's text into another.

**Why an exception wins.** An engine error must reach the client even if outputs are pending; the partial output is useless once the request has failed.""",
    "advisory_prerequisites": ["chat_stream_accumulator"],
    "hints": [
        {"level": 1, "kind": "questions", "content": "What should happen to memory if the engine produces 1000 outputs while the client is stalled? How does merging differ between delta and cumulative outputs? How does an async get wait without busy-looping?"},
        {"level": 2, "kind": "analysis", "content": "Keep one pending slot and an asyncio.Event. put stores a deep copy when the slot is empty or the item is an exception, else merges completions by index. get loops: while the slot is empty, await the event; then take the slot, clear the event, and raise if it holds an exception."},
    ],
    "model_connections": [
        "vLLM's AsyncLLM gives every request a RequestOutputCollector; the output handler loop puts, and the request's generate() coroutine gets.",
        "SGLang's tokenizer manager and TGI's router use similar per-request channels between the scheduler and the HTTP layer.",
    ],
    "pro_con_analysis": {
        "pros": ["Constant memory per request, the producer never blocks, and a slow client still receives complete text."],
        "cons": ["Chunks lose their timing, so per-token latency metrics and token-level streaming effects are hidden from a slow client."],
    },
    "sources": [
        {"kind": "code", "url": "https://github.com/vllm-project/vllm", "commit": "953f90d25fea2240965878a1aca66cec519685b0", "path": "vllm/v1/engine/output_processor.py", "symbol": "RequestOutputCollector.put, get and get_nowait", "license": "Apache-2.0", "adapted": "One-slot hand-off woken by an event, exceptions overriding pending output, and merging when the consumer falls behind.", "simplifications": "Outputs are dicts; no pooling outputs or input-stream task; put copies its input instead of mutating the first output."},
        {"kind": "code", "url": "https://github.com/vllm-project/vllm", "commit": "953f90d25fea2240965878a1aca66cec519685b0", "path": "vllm/outputs.py", "symbol": "RequestOutput.add", "license": "Apache-2.0", "adapted": "OR of finished and per-index aggregate or replace of completions.", "simplifications": "Only text, token_ids and finish_reason are merged; no logprobs, stop_reason or transfer parameters."},
    ],
    "tests": [
        {"name": "Merges deltas while the consumer is behind", "behavior": "scheduler.concurrency", "code": r"""
def out(text, ids, finished=False, reason=None, index=0):
    return {"finished": finished, "outputs": [{"index": index, "text": text, "token_ids": ids, "finish_reason": reason}]}
c = {fn}(aggregate=True)
assert c.get_nowait() is None
c.put(out("Hel", [1])); c.put(out("lo", [2])); c.put(out("!", [3], True, "stop"))
assert c.get_nowait() == out("Hello!", [1, 2, 3], True, "stop")
assert c.get_nowait() is None
"""},
        {"name": "Replace mode, indexes, exceptions and copies", "visibility": "unshown", "behavior": "protocol.validation", "failure_message": "Replace completions in cumulative mode, merge by index, let an exception override and drop later outputs, keep finished once set, and never mutate the caller's dicts.", "code": r"""
import copy
def out(text, ids, finished=False, reason=None, index=0):
    return {"finished": finished, "outputs": [{"index": index, "text": text, "token_ids": ids, "finish_reason": reason}]}
c = {fn}(aggregate=False)
c.put(out("He", [1])); c.put(out("Hello", [1, 2]))
assert c.get_nowait() == out("Hello", [1, 2])
c = {fn}(aggregate=True)
first, second, third = out("a", [1], index=1), out("x", [9], index=0), out("b", [2], True, "length", index=1)
snap = copy.deepcopy([first, second, third])
c.put(first); c.put(second); c.put(third)
got = c.get_nowait()
assert got == {"finished": True, "outputs": [{"index": 1, "text": "ab", "token_ids": [1, 2], "finish_reason": "length"},
                                             {"index": 0, "text": "x", "token_ids": [9], "finish_reason": None}]}, got
assert [first, second, third] == snap, "put modified its input"
c.put(out("late", [5])); got["outputs"][0]["text"] = "zz"
assert c.get_nowait() == out("late", [5])
c = {fn}(aggregate=True)
c.put(out("a", [1], True, "stop")); c.put(out("", []))
assert c.get_nowait()["finished"] is True
c.put(out("a", [1])); c.put(RuntimeError("engine died")); c.put(out("b", [2]))
try:
    c.get_nowait()
except RuntimeError as e:
    assert str(e) == "engine died"
else:
    raise AssertionError("exception not raised")
assert c.get_nowait() is None
"""},
        {"name": "Async consumer receives everything in order", "visibility": "unshown", "behavior": "scheduler.concurrency", "failure_message": "get must wait for a put without busy-waiting, clear the slot, and together with merging deliver the exact concatenated text.", "code": r"""
import asyncio, random
async def main(seed):
    rng = random.Random(seed)
    c = {fn}(aggregate=True)
    pieces = ["t%d " % i for i in range(60)]
    async def producer():
        for i, p in enumerate(pieces):
            c.put({"finished": i == len(pieces) - 1, "outputs": [{"index": 0, "text": p, "token_ids": [i], "finish_reason": "stop" if i == len(pieces) - 1 else None}]})
            if rng.random() < 0.5:
                await asyncio.sleep(0)
    async def consumer():
        text, ids, gets = "", [], 0
        while True:
            o = await c.get()
            gets += 1
            text += o["outputs"][0]["text"]; ids += o["outputs"][0]["token_ids"]
            if o["finished"]:
                return text, ids, gets
            for _ in range(rng.randint(0, 3)):
                await asyncio.sleep(0)
    (text, ids, gets), _ = await asyncio.wait_for(asyncio.gather(consumer(), producer()), timeout=5)
    assert text == "".join(pieces) and ids == list(range(60)), (text, ids)
    assert gets < 60, gets
    waiter = asyncio.ensure_future(c.get())
    await asyncio.sleep(0.01)
    assert not waiter.done(), "get returned without a put"
    c.put(ValueError("x"))
    try:
        await asyncio.wait_for(waiter, timeout=5)
    except ValueError:
        pass
    else:
        raise AssertionError("get did not raise the pending exception")
for seed in (7, 33, 61):
    asyncio.run(main(seed))
"""},
    ],
    "solution": '''import asyncio
import copy


class OutputCollector:
    def __init__(self, aggregate):
        self.aggregate = aggregate
        self.output = None
        self.ready = asyncio.Event()

    def put(self, item):
        if self.output is None or isinstance(item, Exception):
            self.output = item if isinstance(item, Exception) else copy.deepcopy(item)
            self.ready.set()
        elif not isinstance(self.output, Exception):
            self._merge(self.output, item)

    def _merge(self, pending, new):
        pending["finished"] = pending["finished"] or new["finished"]
        for completion in new["outputs"]:
            for i, current in enumerate(pending["outputs"]):
                if current["index"] == completion["index"]:
                    if self.aggregate:
                        current["text"] += completion["text"]
                        current["token_ids"].extend(completion["token_ids"])
                        current["finish_reason"] = completion["finish_reason"]
                    else:
                        pending["outputs"][i] = copy.deepcopy(completion)
                    break
            else:
                pending["outputs"].append(copy.deepcopy(completion))

    def get_nowait(self):
        output = self.output
        if output is not None:
            self.output = None
            self.ready.clear()
        if isinstance(output, Exception):
            raise output
        return output

    async def get(self):
        while self.output is None:
            await self.ready.wait()
        return self.get_nowait()
''',
    "interview_questions": interview(
        concept=[
            "In an async LLM server, what sits between the engine loop that produces tokens and the HTTP handler that streams them? Why not call the handler directly?",
            "What is backpressure, and what happens to an LLM server if one client reads its stream very slowly?",
        ],
        deep_dive=[
            "Why does the collector merge outputs instead of queueing them? What stays the same and what changes for a slow client?",
            "Delta mode appends and cumulative mode replaces. Walk through three puts in each mode before one get.",
            "Why must merging match completions by index when n > 1, and why does an exception override pending output?",
        ],
        tradeoffs=[
            "One-slot merging versus a bounded queue that blocks the producer versus an unbounded queue: memory, fairness and latency?",
            "Detecting client disconnects and aborting the request versus finishing generation anyway: GPU cost, caching and idempotency?",
        ],
    ),
}
