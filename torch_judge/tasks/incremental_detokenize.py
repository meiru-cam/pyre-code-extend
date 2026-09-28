"""Turn a stream of generated tokens into text deltas with a sliding prefix window, as vLLM does."""

from ._interview import interview

TASK = {
    "title": "Incremental Detokenization",
    "difficulty": "Medium",
    "version": 1,
    "function_name": "IncrementalDetokenizer",
    "description_en": r"""An inference server streams text, not token ids. Decoding each new token on its own gives wrong text: tokenizers drop a leading space at the start of a string, and one UTF-8 character can be spread over several byte tokens. Implement vLLM's incremental detokenizer, which decodes a small window of recent tokens and returns only the new part.

**Signature:** `IncrementalDetokenizer(convert, prompt_tokens, initial_offset=5)` with `step(token: str) -> str`.

**Parameters:**
- `convert(tokens: list[str]) -> str` — the tokenizer's tokens-to-string function. Treat it as a black box; it may drop a leading space or produce `"�"` for incomplete bytes.
- `prompt_tokens` — list of prompt token strings.

**State.** A token list and two offsets into it, `prefix_offset <= read_offset`.
- Start with the last `initial_offset + 2` prompt tokens (fewer if the prompt is shorter). `read_offset` is the length of that list and `prefix_offset = max(read_offset - initial_offset, 0)`.

**`step(token)`:**
1. Append `token` to the token list.
2. `prefix_text = convert(tokens[prefix_offset:read_offset])` and `new_text = convert(tokens[prefix_offset:])`.
3. If `len(new_text) <= len(prefix_text)` or `new_text` ends with `"�"`, return `""` and leave both offsets unchanged.
4. Otherwise return `new_text[len(prefix_text):]`, then set `prefix_offset = read_offset` and `read_offset = len(tokens)`.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why a prefix.** SentencePiece tokenizers write a space as the marker `▁` and strip the space at the start of a decoded string. Decoding `▁world` alone gives `world`; decoding `▁Hello ▁world` and subtracting the decoding of `▁Hello` gives ` world`.

**Why hold back on the replacement character.** Byte-fallback tokenizers split rare characters such as CJK or emoji into byte tokens like `<0xE4>`. Until the last byte arrives, the decoded string ends with the replacement character; emitting it would show garbage that later turns into the right character. A replacement character in the middle is a real invalid sequence and is emitted.

**Why a sliding window.** Decoding the whole output every step costs time proportional to its length, so a long generation would be quadratic. The window moves forward whenever text is emitted and stays a few tokens long.

**Why start from the prompt.** The first generated token needs context too: without the prompt's last tokens, a leading space on the first output token would be stripped.""",
    "advisory_prerequisites": ["bpe"],
    "hints": [
        {"level": 1, "kind": "questions", "content": "Why is decode(token) for one token not the text that token adds? When should the detokenizer emit nothing and wait? After emitting, which tokens still need to be in the window?"},
        {"level": 2, "kind": "analysis", "content": "Keep tokens, prefix_offset and read_offset. Each step, decode tokens[prefix:read] and tokens[prefix:]. If the second is not longer or ends with the replacement character, return ''. Otherwise return the extra suffix and move prefix to read and read to the end."},
    ],
    "model_connections": [
        "vLLM's SlowIncrementalDetokenizer calls detokenize_incrementally for every generated token; the offsets come from Hugging Face text-generation-inference.",
        "Hugging Face TextStreamer solves the same problem differently, by buffering until a space or a CJK character.",
    ],
    "pro_con_analysis": {
        "pros": ["Works with any tokenizer through its decode function, costs a few tokens of decoding per step, and never emits text it would later change."],
        "cons": ["Holding back on a trailing replacement character delays output for byte tokens, and a stream that ends mid-character never emits those bytes."],
    },
    "sources": [
        {"kind": "code", "url": "https://github.com/vllm-project/vllm", "commit": "953f90d25fea2240965878a1aca66cec519685b0", "path": "vllm/tokenizers/detokenizer_utils.py", "symbol": "detokenize_incrementally, convert_prompt_ids_to_tokens and INITIAL_INCREMENTAL_DETOKENIZATION_OFFSET", "license": "Apache-2.0", "adapted": "Prompt-seeded token window, prefix and read offsets, holding back on a shorter or replacement-terminated decode, and advancing the window after emitting.", "simplifications": "Tokens arrive as strings through a given convert function; no ids, special-token skipping, added-vocabulary path or out-of-vocabulary ids."},
    ],
    "tests": [
        {"name": "Keeps spaces and waits for complete characters", "behavior": "protocol.validation", "code": r"""
def convert(tokens):
    out = bytearray()
    for t in tokens:
        out += bytes([int(t[3:-1], 16)]) if t.startswith("<0x") else t.replace("▁", " ").encode()
    s = out.decode("utf-8", errors="replace")
    return s[1:] if s.startswith(" ") else s
d = {fn}(convert, ["▁Say", "▁hi", ":"])
out = [d.step(t) for t in ["▁Hello", "▁world", "▁", "<0xE4>", "<0xB8>", "<0xAD>", "!"]]
assert out == [" Hello", " world", " ", "", "", "中", "!"], out
"""},
        {"name": "Offsets and window rules", "visibility": "unshown", "behavior": "state.invariant", "failure_message": "Seed the window from the last initial_offset + 2 prompt tokens, keep offsets fixed while holding back, and slide prefix to read and read to the end after emitting.", "code": r"""
calls = []
def convert(tokens):
    calls.append(tuple(tokens))
    s = "".join(t.replace("▁", " ") for t in tokens)
    return s[1:] if s.startswith(" ") else s
d = {fn}(convert, ["p%d" % i for i in range(10)], initial_offset=3)
calls.clear()
assert d.step("▁x") == " x"
assert calls == [("p7", "p8", "p9"), ("p7", "p8", "p9", "▁x")], calls
calls.clear()
assert d.step("y") == "y"
assert calls == [("▁x",), ("▁x", "y")], calls
d = {fn}(convert, [])
assert d.step("▁a") == "a" and d.step("▁b") == " b"
def shrink(tokens):
    return "".join(t for t in tokens if t != "DEL")
d = {fn}(shrink, ["a"])
assert d.step("DEL") == "" and d.step("b") == "b"
"""},
        {"name": "Seeded streams match the reference window", "visibility": "unshown", "behavior": "protocol.validation", "failure_message": "Emit exactly the text the prefix window adds, hold back while the decode ends in a replacement character, and advance the offsets only after emitting.", "code": r"""
import random
def convert(tokens):
    out = bytearray()
    for t in tokens:
        out += bytes([int(t[3:-1], 16)]) if t.startswith("<0x") else t.replace("▁", " ").encode()
    s = out.decode("utf-8", errors="replace")
    return s[1:] if s.startswith(" ") else s
def reference(prompt, outs, k):
    toks = list(prompt[-(k + 2):]) if prompt else []
    r = len(toks); p = max(r - k, 0); res = []
    for t in outs:
        toks.append(t)
        a, b = convert(toks[p:r]), convert(toks[p:])
        if len(b) <= len(a) or b.endswith("�"):
            res.append(""); continue
        res.append(b[len(a):]); p, r = r, len(toks)
    return res
WORDS = ["▁the", "▁cat", "ing", "▁", "s", ".", "▁中"]
BYTES = [["<0xE4>", "<0xB8>", "<0xAD>"], ["<0xF0>", "<0x9F>", "<0x98>", "<0x80>"], ["<0xC3>", "<0xA9>"]]
for seed in (2, 51, 84):
    rng = random.Random(seed)
    for trial in range(150):
        prompt = [rng.choice(WORDS) for _ in range(rng.randint(0, 9))]
        outs = []
        while len(outs) < rng.randint(1, 16):
            r = rng.random()
            if r < 0.25:
                outs += rng.choice(BYTES)
            elif r < 0.3:
                outs.append("<0xFF>")
            else:
                outs.append(rng.choice(WORDS))
        k = rng.randint(1, 6)
        d = {fn}(convert, prompt, initial_offset=k)
        got = [d.step(t) for t in outs]
        assert got == reference(prompt, outs, k), (seed, trial, prompt, outs, k, got)
        if not convert(outs).endswith("�") and not prompt:
            assert "".join(got) == convert(outs), (prompt, outs, got)
"""},
    ],
    "solution": '''class IncrementalDetokenizer:
    def __init__(self, convert, prompt_tokens, initial_offset=5):
        self.convert = convert
        self.tokens = list(prompt_tokens[-(initial_offset + 2):]) if prompt_tokens else []
        self.read_offset = len(self.tokens)
        self.prefix_offset = max(self.read_offset - initial_offset, 0)

    def step(self, token):
        self.tokens.append(token)
        prefix_text = self.convert(self.tokens[self.prefix_offset:self.read_offset])
        new_text = self.convert(self.tokens[self.prefix_offset:])
        if len(new_text) <= len(prefix_text) or new_text.endswith("\\ufffd"):
            return ""
        self.prefix_offset = self.read_offset
        self.read_offset = len(self.tokens)
        return new_text[len(prefix_text):]
''',
    "interview_questions": interview(
        concept=[
            "Why can't a streaming server decode each generated token id on its own and send the result?",
            "What is byte-fallback tokenization, and what does a client see if the server streams a partial UTF-8 character?",
        ],
        deep_dive=[
            "Walk through prefix_offset and read_offset over three steps where the second token is an incomplete byte. What is decoded at each step and what is returned?",
            "Why is the window seeded with the last few prompt tokens instead of starting empty?",
            "Why does the detokenizer compare lengths before slicing, and when can the new decode be shorter than the prefix?",
        ],
        tradeoffs=[
            "Window-based incremental decoding (vLLM) versus buffering until a word boundary (HF TextStreamer): latency, correctness and CJK handling?",
            "Detokenizing in the engine process versus in a separate API server process: throughput, GIL contention and failure isolation?",
        ],
    ),
}
