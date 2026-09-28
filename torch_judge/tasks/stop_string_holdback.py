"""Stop generation at a stop string without ever streaming part of it, as vLLM's detokenizer does."""

from ._interview import interview

TASK = {
    "title": "Streaming Stop Strings",
    "difficulty": "Medium",
    "version": 1,
    "function_name": "StopStringStream",
    "description_en": r"""A request can ask generation to end at stop strings such as `"\nUser:"`, and by default the stop string is not part of the output. When text is streamed, a stop string may arrive split over several tokens, so the server must hold back just enough text never to stream a piece of it. Implement vLLM's stop-string handling.

**Signature:** `StopStringStream(stop, include_stop_str_in_output=False, min_tokens=0)` with `update(new_texts: list[str]) -> str | None` and `get_next_output_text(finished: bool) -> str`. It keeps `output_text`, starting at `""`.

**Hold-back length.** `max(len(s) for s in stop) - 1` if `stop` is non-empty and stop strings are excluded from the output; otherwise 0.

**`update(new_texts)`** — each item is the text of one new token.
1. If `new_texts` is empty, return None.
2. Let `check_from = len(output_text)`. For each text: append it to `output_text` and count the token; if `min_tokens > 0` and the token count is now `<= min_tokens`, set `check_from = len(output_text)`.
3. If `stop` is non-empty and the token count is `> min_tokens`, search, then return the match or None.

**Search.** Let `new_chars = len(output_text) - check_from`. If it is 0, there is no match. For each stop string `s`, find its first occurrence at or after index `len(output_text) - new_chars - len(s) + 1` (clamped to 0). Among the stop strings found, the match is the one whose occurrence *ends* earliest; ties go to the earlier string in `stop`. On a match, truncate `output_text` to the end of the occurrence if stop strings are included, else to its start.

**`get_next_output_text(finished)`** returns the text not yet returned, up to `len(output_text) - hold`, where `hold` is 0 if `finished` and the hold-back length otherwise. It returns `""` if there is nothing new.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why hold back one character less than the longest stop string.** If the tail of the text is a proper prefix of a stop string, the next token may complete it. A prefix is at most `len - 1` characters, so holding that many is enough, and holding less could stream part of a stop string the client then cannot take back.

**Why search only the new text plus an overlap.** Earlier text was already searched. A stop string completed by the new text can start up to `len(s) - 1` characters before it, so the search begins there instead of rescanning the whole output.

**Why the earliest end.** Speculative decoding can append several tokens in one step. Picking the stop string that completes first gives the same result as appending the tokens one at a time.

**Why `min_tokens` moves the search start.** Text from the first `min_tokens` tokens cannot trigger a stop, but a stop string that ends after that point may still begin inside it.""",
    "advisory_prerequisites": ["incremental_detokenize"],
    "hints": [
        {"level": 1, "kind": "questions", "content": "What is the longest piece of a stop string that could be sitting at the end of the text right now? Where must the search start so that it catches a stop string that straddles old and new text? Which match wins if two stop strings appear in the same update?"},
        {"level": 2, "kind": "analysis", "content": "Keep output_text, a token count and the offset of text already returned. In update, append texts while moving check_from past the first min_tokens tokens, then str.find each stop string from the computed start and keep the one with the smallest end. In get_next_output_text, compute the limit, slice from the last offset, and advance it."},
    ],
    "model_connections": [
        "vLLM's BaseIncrementalDetokenizer.update and check_stop_strings implement stop strings for the OpenAI-compatible server, including include_stop_str_in_output and min_tokens.",
        "The same hold-back appears in SGLang and TGI, and in client libraries that apply stop sequences on their side.",
    ],
    "pro_con_analysis": {
        "pros": ["Never streams a character it must retract, and the search cost per step depends on the new text, not the whole output."],
        "cons": ["Adds up to one stop-string length of latency to every streamed chunk, and long stop strings make streaming noticeably jumpier."],
    },
    "sources": [
        {"kind": "code", "url": "https://github.com/vllm-project/vllm", "commit": "953f90d25fea2240965878a1aca66cec519685b0", "path": "vllm/v1/engine/detokenizer.py", "symbol": "BaseIncrementalDetokenizer.update, BaseIncrementalDetokenizer.get_next_output_text and check_stop_strings", "license": "Apache-2.0", "adapted": "The hold-back length, the min_tokens search offset, searching new text plus an overlap, the earliest-ending match with list-order ties, truncation for both output modes, and delta output.", "simplifications": "Takes token texts instead of detokenizing ids, drops the stop-token skip and the cumulative output mode."},
    ],
    "tests": [
        {"name": "Holds back a partial stop string", "behavior": "protocol.validation", "code": r"""
s = {fn}(["\nUser:"])
assert s.update(["Hi", " there", "\nUs"]) is None
assert s.get_next_output_text(False) == "Hi the"
assert s.update(["er:", " more"]) == "\nUser:"
assert s.output_text == "Hi there"
assert s.get_next_output_text(True) == "re"
assert s.get_next_output_text(True) == ""
"""},
        {"name": "Match selection, inclusion and min_tokens", "visibility": "unshown", "behavior": "protocol.validation", "failure_message": "Prefer the stop string that ends first (list order on ties), truncate after it when included, and ignore stops that end inside the first min_tokens tokens.", "code": r"""
s = {fn}(["cde", "bc"])
assert s.update(["abcdef"]) == "bc" and s.output_text == "a"
s = {fn}(["xy", "y"], include_stop_str_in_output=True)
assert s.update(["axyz"]) == "xy" and s.output_text == "axy"
assert s.get_next_output_text(False) == "axy"
s = {fn}(["b", "ab"])
assert s.update(["ab"]) == "b" and s.output_text == "a"
s = {fn}(["STOP"], min_tokens=2)
assert s.update(["ST", "OP"]) is None and s.output_text == "STOP"
assert s.update(["x"]) is None
s = {fn}(["STOP"], min_tokens=1)
assert s.update(["ST", "OP"]) == "STOP" and s.output_text == ""
s = {fn}(["ab"], min_tokens=1)
assert s.update(["ab"]) is None and s.update(["c"]) is None and s.output_text == "abc"
s = {fn}([])
assert s.update(["abc"]) is None and s.get_next_output_text(False) == "abc"
assert s.update([]) is None
"""},
        {"name": "Search starts at the overlap", "visibility": "unshown", "behavior": "protocol.validation", "failure_message": "Search from len(output_text) - new_chars - len(s) + 1 so already-searched text is not matched again.", "code": r"""
s = {fn}(["aa"], include_stop_str_in_output=True, min_tokens=1)
assert s.update(["aaa"]) is None
assert s.update(["b"]) is None and s.output_text == "aaab"
s = {fn}(["aba"], include_stop_str_in_output=True, min_tokens=1)
assert s.update(["ab"]) is None
assert s.update(["a"]) == "aba" and s.output_text == "aba"
"""},
        {"name": "Seeded streams never leak a stop string", "visibility": "unshown", "behavior": "protocol.validation", "failure_message": "The deltas must match the reference exactly: hold back len(longest) - 1 characters until finished, and never return text past the truncation point.", "code": r"""
import random
class Ref:
    def __init__(self, stop, inc, mt):
        self.stop, self.inc, self.mt = stop, inc, mt
        self.hold = max(len(x) for x in stop) - 1 if stop and not inc else 0
        self.text, self.n, self.off = "", 0, 0
    def update(self, texts):
        if not texts:
            return None
        start = len(self.text)
        for t in texts:
            self.text += t; self.n += 1
            if self.mt and self.n <= self.mt:
                start = len(self.text)
        if not self.stop or self.n <= self.mt:
            return None
        new = len(self.text) - start
        if not new:
            return None
        best = None
        for x in self.stop:
            i = self.text.find(x, max(0, len(self.text) - new - len(x) + 1))
            if i != -1 and (best is None or i + len(x) < best[1] + len(best[0])):
                best = (x, i)
        if best is None:
            return None
        x, i = best
        self.text = self.text[:i + len(x)] if self.inc else self.text[:i]
        return x
    def out(self, fin):
        lim = len(self.text) - (0 if fin else self.hold)
        if self.off < lim:
            r = self.text[self.off:lim]; self.off = lim; return r
        return ""
ALPH = "abc\n:"
for seed in (5, 43, 96):
    rng = random.Random(seed)
    for trial in range(200):
        stop = ["".join(rng.choice(ALPH) for _ in range(rng.randint(1, 4))) for _ in range(rng.randint(0, 3))]
        inc, mt = rng.random() < 0.3, rng.choice([0, 0, 1, 3])
        s, r = {fn}(stop, include_stop_str_in_output=inc, min_tokens=mt), Ref(stop, inc, mt)
        streamed = ""
        for step in range(rng.randint(1, 10)):
            texts = ["".join(rng.choice(ALPH) for _ in range(rng.randint(0, 3))) for _ in range(rng.randint(0, 3))]
            got, want = s.update(texts), r.update(texts)
            assert got == want and s.output_text == r.text, (seed, trial, stop, texts, got, want, s.output_text, r.text)
            fin = got is not None or rng.random() < 0.1
            a, b = s.get_next_output_text(fin), r.out(fin)
            assert a == b, (seed, trial, stop, a, b)
            streamed += a
            if fin:
                break
        if stop and not inc and not mt:
            assert not any(x in streamed for x in stop), (stop, streamed)
"""},
    ],
    "solution": '''class StopStringStream:
    def __init__(self, stop, include_stop_str_in_output=False, min_tokens=0):
        self.stop = list(stop)
        self.include = include_stop_str_in_output
        self.min_tokens = min_tokens
        self.hold = max(len(s) for s in self.stop) - 1 if self.stop and not self.include else 0
        self.output_text = ""
        self.num_tokens = 0
        self._returned = 0

    def update(self, new_texts):
        if not new_texts:
            return None
        check_from = len(self.output_text)
        for text in new_texts:
            self.output_text += text
            self.num_tokens += 1
            if self.min_tokens and self.num_tokens <= self.min_tokens:
                check_from = len(self.output_text)
        if not self.stop or self.num_tokens <= self.min_tokens:
            return None

        new_chars = len(self.output_text) - check_from
        if not new_chars:
            return None
        best, best_index, best_end = None, 0, None
        for s in self.stop:
            index = self.output_text.find(s, max(0, len(self.output_text) - new_chars - len(s) + 1))
            if index == -1:
                continue
            end = index + len(s)
            if best_end is None or end < best_end:
                best, best_index, best_end = s, index, end
        if best is None:
            return None
        self.output_text = self.output_text[:best_end] if self.include else self.output_text[:best_index]
        return best

    def get_next_output_text(self, finished):
        limit = len(self.output_text) - (0 if finished else self.hold)
        if self._returned < limit:
            text = self.output_text[self._returned:limit]
            self._returned = limit
            return text
        return ""
''',
    "interview_questions": interview(
        concept=[
            "What are stop strings in an LLM API, and why can't the server just check each new token against them?",
            "Why must a streaming server hold back some text before sending it when stop strings are excluded from the output?",
        ],
        deep_dive=[
            "Why is the hold-back exactly the longest stop string's length minus one? Show a case where holding one character less leaks part of a stop string.",
            "Where does the search for a stop string start after an update, and why is the overlap len(s) - 1 characters?",
            "Speculative decoding appends several tokens at once and two stop strings appear. Which one should win and why?",
        ],
        tradeoffs=[
            "Stop strings checked on detokenized text versus stop token ids checked in the sampler: exactness, tokenization ambiguity and cost?",
            "Holding back text for stop strings versus streaming immediately and letting the client trim: latency versus correctness guarantees?",
        ],
    ),
}
