"""Full-duplex micro-turns: append each 200 ms chunk to a persistent session instead of re-prefilling."""

from ._interview import interview

TASK = {
    "title": "Full-Duplex Streaming Session",
    "difficulty": "Medium",
    "version": 1,
    "function_name": "StreamingSession",
    "description_en": r"""Thinking Machines' interaction models run in micro-turns: every 200 ms the client sends one chunk of input and the model generates up to 200 ms of output, all in one growing token sequence. Each chunk is a separate request, so the server keeps the session's history and KV cache between requests instead of prefilling the whole conversation again. Implement the session bookkeeping, following SGLang's streaming sessions.

**Signature:** `StreamingSession(bos_id=None)` with `begin(input_ids, max_new_tokens) -> dict`, `finish(output_ids)` and `abort()`.

**State.** The committed history (a token list, initially empty), the KV length (tokens whose KV the session holds, initially 0), and at most one request in flight.

**`begin(input_ids, max_new_tokens)`:**
- Raise `RuntimeError` if a request is in flight.
- If the history is non-empty, `bos_id` is not None and `input_ids` starts with `bos_id`, drop that first token.
- The request's sequence is the history followed by the (possibly trimmed) input. Raise `ValueError` if it is empty.
- Mark the request in flight and return `{"input_ids": sequence, "cached_len": kv_len}`. The server prefills only the tokens after `cached_len`.
- Neither the history nor the KV length changes yet.

**`finish(output_ids)`** — the in-flight request completed (raise `RuntimeError` if none is in flight).
- `out = output_ids[:max_new_tokens]`.
- The history becomes the request's sequence followed by `out`.
- The KV length becomes `len(sequence) + max(len(out) - 1, 0)`: the last sampled token was never run through the model.
- Nothing is in flight afterwards.

**`abort()`** — the in-flight request failed or was cancelled. Nothing is in flight afterwards and the history and KV length are unchanged. With nothing in flight it does nothing.

Returned and stored lists must be copies: later changes to the caller's lists must not affect the session.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Micro-turns.** Instead of waiting for the user to finish speaking, the model sees 200 ms of input, produces 200 ms of output (possibly silence), and repeats. Overlap, interruptions and pauses stay in the context, so the model can backchannel or cut in, and it has a sense of elapsed time.

**Why a session.** Five requests per second per user, each resending the whole conversation, would spend almost all compute on prefill. Appending to a persistent sequence makes each request cost only its new tokens.

**Why commit only on finish.** An aborted chunk may have appended tokens and allocated KV that never completed. Keeping the last committed point lets the next chunk start from a consistent state.

**Why one request in flight.** A chunk depends on the previous chunk's output, so requests in a session are strictly ordered; a second concurrent request would race on the same KV.

**Why drop the BOS.** Tokenizers add a beginning-of-sequence token to every encoded chunk, but only the first one belongs in the middle of an ongoing sequence.""",
    "advisory_prerequisites": ["kv_cache", "request_output_collector"],
    "hints": [
        {"level": 1, "kind": "questions", "content": "What must survive between two chunk requests, and when may it change? After a request generated three tokens, how many of them have KV entries? What should an aborted chunk leave behind?"},
        {"level": 2, "kind": "analysis", "content": "Keep history, kv_len and an in-flight record (sequence, max_new_tokens). begin checks the flag, trims a BOS when history exists, builds history + input, and returns copies. finish truncates the output, sets history and kv_len, and clears the flag; abort only clears the flag."},
    ],
    "model_connections": [
        "Thinking Machines' TML-Interaction-Small streams 200 ms micro-turns and upstreamed streaming sessions to SGLang.",
        "SGLang's Session keeps one in-flight request per streaming session and rolls back to committed token lengths when a turn aborts; OpenAI's Realtime API and Moshi keep similar per-connection state for full-duplex speech.",
    ],
    "pro_con_analysis": {
        "pros": ["Each 200 ms chunk costs only its new tokens, and aborted chunks cannot corrupt the session."],
        "cons": ["The session pins GPU memory for its whole lifetime and ties the user to one server, which complicates load balancing and failover."],
    },
    "sources": [
        {"kind": "code", "url": "https://github.com/sgl-project/sglang", "commit": "2e7e0802f4f3edc94702933675a71d6c7ef4f24f", "path": "python/sglang/srt/session/session_controller.py", "symbol": "Session.create_req, Session._share_token_arrays, Session._strip_bos_token, Session.finish_req and Session.abort_req", "license": "Apache-2.0", "adapted": "One in-flight request per streaming session, history of previous input plus output truncated to max_new_tokens, BOS stripping on appended turns, and committing only on finish so aborts roll back.", "simplifications": "Token lists instead of shared arrays; no replace, offset or drop_previous_output modes, multimodal offsets, timeouts or radix-cache locks; the KV length uses the len(origin_input_ids) + max(len(output_ids) - 1, 0) formula that SGLang falls back to in mem_cache/storage/flexkv/flexkv_radix_cache.py instead of reading it from the pool."},
        {"kind": "paper", "url": "https://thinkingmachines.ai/blog/interaction-models/", "section": "System overview and The interaction model (micro-turns and streaming sessions)"},
    ],
    "tests": [
        {"name": "Appends chunks and reuses KV", "behavior": "state.invariant", "code": r"""
s = {fn}(bos_id=1)
assert s.begin([1, 10, 11], max_new_tokens=4) == {"input_ids": [1, 10, 11], "cached_len": 0}
s.finish([20, 21])
assert s.begin([1, 12], max_new_tokens=4) == {"input_ids": [1, 10, 11, 20, 21, 12], "cached_len": 4}
s.finish([])
assert s.begin([13], max_new_tokens=4) == {"input_ids": [1, 10, 11, 20, 21, 12, 13], "cached_len": 6}
"""},
        {"name": "Aborts roll back and only one request runs", "visibility": "unshown", "behavior": "checkpoint.recovery", "failure_message": "Commit history only in finish, keep it on abort, reject a second begin while one is in flight, and truncate output to max_new_tokens.", "code": r"""
s = {fn}()
s.begin([5, 6], max_new_tokens=2)
try:
    s.begin([7], max_new_tokens=2)
except RuntimeError:
    pass
else:
    raise AssertionError("second request accepted while one is in flight")
s.finish([8, 9, 10, 11])
assert s.begin([12], max_new_tokens=1) == {"input_ids": [5, 6, 8, 9, 12], "cached_len": 3}
s.abort()
assert s.begin([13], max_new_tokens=1) == {"input_ids": [5, 6, 8, 9, 13], "cached_len": 3}
s.abort(); s.abort()
try:
    s.finish([1])
except RuntimeError:
    pass
else:
    raise AssertionError("finish accepted with nothing in flight")
try:
    {fn}().begin([], max_new_tokens=3)
except ValueError:
    pass
else:
    raise AssertionError("empty request accepted")
f = {fn}(bos_id=0)
assert f.begin([0, 4], max_new_tokens=2)["input_ids"] == [0, 4]
f.finish([0])
assert f.begin([0, 0], max_new_tokens=2) == {"input_ids": [0, 4, 0, 0], "cached_len": 2}
"""},
        {"name": "Seeded micro-turn streams match an oracle", "visibility": "unshown", "behavior": "state.invariant", "failure_message": "Across finishes, aborts and BOS-prefixed chunks, the sequence and cached_len must follow the committed history exactly, and returned lists must be copies.", "code": r"""
import random
for seed in (10, 46, 88):
    rng = random.Random(seed)
    bos = rng.choice([None, 1])
    s = {fn}(bos_id=bos)
    hist, kv = [], 0
    for turn in range(60):
        chunk = [rng.randint(1, 9) for _ in range(rng.randint(0, 4))]
        if bos is not None and rng.random() < 0.5:
            chunk = [bos] + chunk
        mnt = rng.randint(1, 4)
        trimmed = chunk[1:] if hist and bos is not None and chunk[:1] == [bos] else chunk
        seq = hist + trimmed
        if not seq:
            try:
                s.begin(chunk, mnt)
            except ValueError:
                continue
            raise AssertionError("empty request accepted")
        got = s.begin(chunk, mnt)
        assert got == {"input_ids": seq, "cached_len": kv}, (seed, turn, got, seq, kv)
        got["input_ids"].append(-1); chunk.append(-2)
        if rng.random() < 0.25:
            s.abort()
            continue
        out = [rng.randint(1, 9) for _ in range(rng.randint(0, 6))]
        s.finish(out)
        out.append(-3)
        kept = out[:-1][:mnt]
        hist, kv = seq + kept, len(seq) + max(len(kept) - 1, 0)
"""},
    ],
    "solution": '''class StreamingSession:
    def __init__(self, bos_id=None):
        self.bos_id = bos_id
        self.history = []
        self.kv_len = 0
        self._inflight = None

    def begin(self, input_ids, max_new_tokens):
        if self._inflight is not None:
            raise RuntimeError("streaming session already has an active request")
        new_ids = list(input_ids)
        if self.history and self.bos_id is not None and new_ids[:1] == [self.bos_id]:
            new_ids = new_ids[1:]
        sequence = self.history + new_ids
        if not sequence:
            raise ValueError("a session request must contain input tokens")
        self._inflight = (sequence, max_new_tokens)
        return {"input_ids": list(sequence), "cached_len": self.kv_len}

    def finish(self, output_ids):
        if self._inflight is None:
            raise RuntimeError("no request in flight")
        sequence, max_new_tokens = self._inflight
        out = list(output_ids[:max_new_tokens])
        self.history = sequence + out
        self.kv_len = len(sequence) + max(len(out) - 1, 0)
        self._inflight = None

    def abort(self):
        self._inflight = None
''',
    "interview_questions": interview(
        concept=[
            "What is a full-duplex interaction model, and how do 200 ms micro-turns differ from turn-based voice pipelines (ASR, LLM, TTS)?",
            "Why would sending every 200 ms chunk as an independent request be too expensive, and what does a streaming session keep on the server?",
        ],
        deep_dive=[
            "After a request with a 6-token sequence generates 3 tokens, what is the KV length and why is it not 9?",
            "Why commit history only when a request finishes? Walk through an aborted chunk followed by the next chunk.",
            "Why does a streaming session allow only one request in flight, and why strip the BOS token from appended chunks?",
        ],
        tradeoffs=[
            "One interleaved sequence of micro-turns versus separate input and output streams (as in Moshi): model simplicity, latency and context growth?",
            "Sticky sessions that pin KV on one GPU versus stateless requests with prefix caching: latency, load balancing and failure recovery?",
        ],
    ),
}
