"""Decode a server-sent event stream incrementally, the way LLM API clients read token streams."""

from ._interview import interview

TASK = {
    "title": "SSE Stream Decoder",
    "difficulty": "Medium",
    "version": 1,
    "function_name": "SSEDecoder",
    "description_en": r"""Streaming LLM APIs send tokens as server-sent events (SSE) over HTTP. The client receives arbitrary byte chunks and must turn them into events as soon as each one is complete. Implement the decoder.

**Signature:** `SSEDecoder()` with `feed(chunk: bytes) -> list[dict]` and a `done` attribute.

**Events** are dicts `{"event": str | None, "data": str, "id": str | None, "retry": int | None}`.

**Lines.**
- A line ends at `\r\n`, `\r` or `\n`. A `\r` at the end of one chunk followed by `\n` at the start of the next is one line ending.
- Decode a line as UTF-8 only once it is complete; a multi-byte character may be split across chunks.
- Bytes after the last line ending wait for the next chunk.

**Fields.** For each complete line:
- An empty line dispatches an event if at least one field below was applied since the last dispatch; otherwise it does nothing.
- A line starting with `:` is a comment and is ignored.
- Otherwise split at the first `:` into field and value (no colon: the whole line is the field and the value is `""`), and remove one leading space from the value.
- `event` sets the event name. `data` appends the value to the data lines. `id` sets the last event id, unless the value contains `"\0"`, in which case the line is ignored. `retry` sets retry to `int(value)` if the value is non-empty and only ASCII digits; otherwise the line is ignored. Other fields are ignored.

**Dispatch.** The event gets the current name, the data lines joined with `"\n"`, the last event id and retry. Then name, data lines and retry reset; the last event id persists into later events.

**End of stream.** If a dispatched event's data starts with `"[DONE]"`, do not return it, set `done = True`, and ignore the rest of that chunk and every later chunk. `done` starts False.

`feed` returns the events completed by that chunk, in order.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why SSE.** It is plain HTTP, one direction, text framed by blank lines. Browsers, proxies and load balancers already handle it, so OpenAI-compatible servers (vLLM, SGLang, llama.cpp) all stream this way.

**Why chunks do not align with events.** TCP and HTTP chunked encoding cut the body wherever buffers fill. One chunk can carry half a line, three events, or half a UTF-8 character; decoding each chunk on its own corrupts CJK text and emoji.

**Why the id persists.** After a dropped connection the client reconnects with `Last-Event-ID`, so the server can resume from the last event it knows was received.

**Why `[DONE]`.** The OpenAI protocol ends the stream with a sentinel data line rather than relying on the connection closing, so a client can tell a finished stream from a cut one.""",
    "advisory_prerequisites": [],
    "hints": [
        {"level": 1, "kind": "questions", "content": "What state must survive between two feed calls: bytes of an unfinished line, a pending carriage return, fields of an unfinished event? When is it safe to decode bytes as UTF-8?"},
        {"level": 2, "kind": "analysis", "content": "Walk the chunk byte by byte. Keep a bytearray for the current line and a skip_lf flag set after a carriage return; if the next byte is a newline, drop it. On each line ending, decode the buffer and pass the line to a field parser that keeps event, data lines, id, retry and a pending flag."},
    ],
    "model_connections": [
        "The openai-python client's SSEDecoder reads every chat completion stream; vLLM, SGLang and TGI serve the same format.",
        "Anthropic's Messages API streams typed SSE events (message_start, content_block_delta, message_stop) and uses the event field to tell them apart.",
    ],
    "pro_con_analysis": {
        "pros": ["Works through ordinary HTTP infrastructure, needs no handshake, and is easy to debug with curl."],
        "cons": ["One direction only and text only; interruption or client-to-server audio needs a second channel or WebSockets."],
    },
    "sources": [
        {"kind": "code", "url": "https://github.com/openai/openai-python", "commit": "a380cf256abc2aa51cbf5d49437ae039b9aa9b12", "path": "src/openai/_streaming.py", "symbol": "_SSELineDecoder, SSEDecoder.decode and Stream.__stream__", "license": "Apache-2.0", "adapted": "CR, LF and CRLF line splitting with a carriage return carried across chunks, decoding only complete lines, field parsing, a last event id that persists, and stopping at a [DONE] data line.", "simplifications": "A pending flag decides dispatch instead of checking the fields' truthiness, and retry accepts ASCII digits only."},
        {"kind": "paper", "url": "https://html.spec.whatwg.org/multipage/server-sent-events.html", "section": "9.2.6 Interpreting an event stream"},
    ],
    "tests": [
        {"name": "Decodes events split across chunks", "behavior": "protocol.validation", "code": r"""
dec = {fn}()
assert dec.feed(b'data: {"t": "Hel') == []
assert dec.feed(b'lo"}\n\nevent: ping\ndata: a\ndata: b\n') == [
    {"event": None, "data": '{"t": "Hello"}', "id": None, "retry": None}]
assert dec.feed(b'\ndata: [DONE]\n\ndata: late\n\n') == [{"event": "ping", "data": "a\nb", "id": None, "retry": None}]
assert dec.done is True and dec.feed(b'data: x\n\n') == []
"""},
        {"name": "Line endings and UTF-8 across chunk boundaries", "visibility": "unshown", "behavior": "edge.empty_or_boundary", "failure_message": "A carriage return at the end of a chunk plus a newline at the start of the next is one line ending, two carriage returns are two, and characters split across chunks must decode intact.", "code": r"""
dec = {fn}()
assert dec.feed(b"data: x\r") == []
assert dec.feed(b"\n\r\n") == [{"event": None, "data": "x", "id": None, "retry": None}]
dec = {fn}()
assert dec.feed(b"data: a\r") == []
assert dec.feed(b"\rdata: b\r\r") == [{"event": None, "data": "a", "id": None, "retry": None},
                                      {"event": None, "data": "b", "id": None, "retry": None}]
dec = {fn}()
raw = "data: 中文😀\n\n".encode()
out = []
for i in range(len(raw)):
    out += dec.feed(raw[i:i + 1])
assert out == [{"event": None, "data": "中文😀", "id": None, "retry": None}], out
"""},
        {"name": "Field rules", "visibility": "unshown", "behavior": "protocol.validation", "failure_message": "Ignore comments, strip only one leading space, keep the last id across events, reject ids with NUL and non-digit retry values, and dispatch only when a field was applied.", "code": r"""
dec = {fn}()
got = dec.feed(b": keepalive\n\n\ndata:  two spaces\nid: 7\nretry: 3000\n\n"
               b"data\n\n"
               b"id: bad\0id\nretry: 12a\nretry: -5\nevent\ndata: z\n\n"
               b"retry: 5\n\n"
               b"id\n\n")
assert got == [
    {"event": None, "data": " two spaces", "id": "7", "retry": 3000},
    {"event": None, "data": "", "id": "7", "retry": None},
    {"event": "", "data": "z", "id": "7", "retry": None},
    {"event": None, "data": "", "id": "7", "retry": 5},
    {"event": None, "data": "", "id": "", "retry": None},
], got
"""},
        {"name": "Seeded streams match a whole-body oracle", "visibility": "unshown", "behavior": "protocol.validation", "failure_message": "Chunking must not change the decoded events: carry partial lines, pending carriage returns and event fields between feed calls.", "code": r"""
import random, re
def oracle(body):
    lines = re.split(rb"\r\n|\r|\n", body)[:-1]
    out, ev, data, rid, retry, pending = [], None, [], None, None, False
    for raw in lines:
        line = raw.decode()
        if line == "":
            if pending:
                e = {"event": ev, "data": "\n".join(data), "id": rid, "retry": retry}
                if e["data"].startswith("[DONE]"):
                    return out
                out.append(e)
            ev, data, retry, pending = None, [], None, False
            continue
        if line.startswith(":"):
            continue
        f, _, v = line.partition(":")
        if v.startswith(" "):
            v = v[1:]
        if f == "event":
            ev, pending = v, True
        elif f == "data":
            data.append(v); pending = True
        elif f == "id" and "\0" not in v:
            rid, pending = v, True
        elif f == "retry" and v and all(c in "0123456789" for c in v):
            retry, pending = int(v), True
    return out
TEXT = ["tok", " ", "é", "中", "😀", ":", "  ", "{}", "[DONE]", "x"]
for seed in (4, 29, 77):
    rng = random.Random(seed)
    for trial in range(120):
        parts = []
        for _ in range(rng.randint(0, 8)):
            kind = rng.choice(["data", "data", "event", "id", "retry", "comment", "blank", "blank", "junk", "done"])
            val = "".join(rng.choice(TEXT) for _ in range(rng.randint(0, 3)))
            if kind == "done":
                line = "data: [DONE]"
            elif kind == "comment":
                line = ":" + val
            elif kind == "blank":
                line = ""
            elif kind == "retry":
                line = "retry: " + rng.choice(["10", "x1", "", "250"])
            elif kind == "junk":
                line = rng.choice(["foo: bar", "data", "event", "idx: 1"])
            else:
                line = kind + rng.choice([": ", ":", ":  "]) + val
            parts.append(line + rng.choice(["\n", "\r", "\r\n"]))
        body = "".join(parts).encode()
        dec, got = {fn}(), []
        i = 0
        while i < len(body):
            j = min(len(body), i + rng.randint(1, 6))
            got += dec.feed(body[i:j])
            i = j
        assert got == oracle(body), (seed, trial, body, got)
"""},
    ],
    "solution": '''class SSEDecoder:
    def __init__(self):
        self._line = bytearray()
        self._skip_lf = False
        self._event = None
        self._data = []
        self._last_id = None
        self._retry = None
        self._pending = False
        self.done = False

    def feed(self, chunk):
        events = []
        for byte in chunk:
            if self.done:
                break
            if self._skip_lf:
                self._skip_lf = False
                if byte == 10:
                    continue
            if byte != 10 and byte != 13:
                self._line.append(byte)
                continue
            self._skip_lf = byte == 13
            line = self._line.decode("utf-8")
            self._line.clear()
            event = self._decode(line)
            if event is None:
                continue
            if event["data"].startswith("[DONE]"):
                self.done = True
            else:
                events.append(event)
        return events

    def _decode(self, line):
        if not line:
            if not self._pending:
                return None
            event = {"event": self._event, "data": "\\n".join(self._data), "id": self._last_id, "retry": self._retry}
            self._event, self._data, self._retry, self._pending = None, [], None, False
            return event
        if line.startswith(":"):
            return None
        field, _, value = line.partition(":")
        if value.startswith(" "):
            value = value[1:]
        if field == "event":
            self._event = value
        elif field == "data":
            self._data.append(value)
        elif field == "id":
            if "\\0" in value:
                return None
            self._last_id = value
        elif field == "retry":
            if not value or any(c not in "0123456789" for c in value):
                return None
            self._retry = int(value)
        else:
            return None
        self._pending = True
        return None
''',
    "interview_questions": interview(
        concept=[
            "How does a chat completion API stream tokens to the client? What does one SSE event look like on the wire?",
            "Why can't a client decode each network chunk as UTF-8 and split it on newlines?",
        ],
        deep_dive=[
            "Walk through what state the decoder keeps between chunks. What goes wrong if a CRLF is split across two chunks and you forget the pending carriage return?",
            "Why does the last event id persist across events while the event name and data reset? How is it used on reconnect?",
            "Why does the OpenAI protocol end with a [DONE] sentinel instead of just closing the connection?",
        ],
        tradeoffs=[
            "SSE versus WebSockets versus gRPC streaming for serving LLM tokens: proxies, bidirectionality, backpressure and client support?",
            "Streaming one event per token versus batching several tokens per event: latency, overhead and user-visible smoothness?",
        ],
    ),
}
