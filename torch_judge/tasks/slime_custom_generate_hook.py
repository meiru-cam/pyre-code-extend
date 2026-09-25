"""Implement a Slime custom generation hook's Sample contract."""

TASK = {
    "title": "Slime Custom Generate Hook",
    "difficulty": "Medium",
    "version": 1,
    "function_name": "slime_custom_generate_hook",
    "description_en": r"""Fill a Slime-style Sample from an asynchronous generation result.

**Signature:** `async slime_custom_generate_hook(args, sample, sampling_params) -> Sample`

The hook is invoked per sample. For this CPU exercise, `args.generate(prompt,
sampling_params)` is an async stand-in for the SGLang router and returns a
dictionary with `token_ids` (new response token ids), `text`, and
`finish_reason`: `"stop"`, `"length"`, or `"abort"`. The supplied sample
already has its prompt token ids in `sample.tokens`, as well as fields
`response`, `response_length`, `loss_mask`, and `status`. Its
`sample.Status` enum defines COMPLETED, TRUNCATED, and ABORTED.

Await generation once, append response tokens to a new `sample.tokens`
list, set `response_length` to the number of response tokens, set
`loss_mask` to one per response token, and copy the response text.
Set status from finish_reason using the three enum values above. Return the
same sample object. Raise ValueError for an unknown finish reason or when
`token_ids` is not a list of integers. Do not silently score aborted or
truncated samples as completed.

────────────────────────────────

**Background — context only.** Slime registers this shape of hook through
`--custom-generate-function-path`; its default rollout loop still handles
batching and reward. The real hook calls a router and may add tool-observation
tokens with loss mask zero. This exercise isolates the Sample contract,
status mapping and token alignment that an interviewer can ask you to write
without a GPU or live SGLang server.""",
    "advisory_prerequisites": ["agent_env_adapter", "response_token_mask"],
    "hints": [
        {"level": 1, "kind": "questions", "content": "Does response_length count prompt tokens? Which finish reason means a context limit rather than success? How long must loss_mask be?"},
        {"level": 2, "kind": "analysis", "content": "Await args.generate, validate token ids and finish reason, append ids to a fresh copy of sample.tokens, then assign response_length, loss_mask, response and the corresponding sample.Status enum value."},
    ],
    "model_connections": [
        "Slime calls a per-sample async custom generate function from its default sglang_rollout outer loop.",
        "Sample.response_length and Sample.loss_mask refer to response positions, while Sample.tokens includes the prompt.",
    ],
    "sources": [
        {"kind": "code", "url": "https://github.com/THUDM/slime", "commit": "8ee9c1e1c8871ccd6dc8ec812edfaefa3dd1156b", "path": "slime/rollout/sglang_rollout.py", "symbol": "generate", "license": "Apache-2.0", "adapted": "The async per-sample generation hook and status mapping.", "simplifications": "Replaces SGLang HTTP and tokenizer calls with an injected async args.generate stand-in."},
        {"kind": "code", "url": "https://github.com/THUDM/slime", "commit": "8ee9c1e1c8871ccd6dc8ec812edfaefa3dd1156b", "path": "slime/utils/types.py", "symbol": "Sample", "license": "Apache-2.0", "adapted": "The Sample fields tokens, response_length, loss_mask, response and status.", "simplifications": "Tests use a tiny local object with the same relevant fields and enum names."},
    ],
    "tests": [
        {"name": "Fills prompt-plus-response tokens and completed status", "behavior": "rl.rollout_assembly", "code": r"""
import asyncio
from types import SimpleNamespace
class Sample:
    class Status:
        COMPLETED="completed"; TRUNCATED="truncated"; ABORTED="aborted"
    def __init__(self):
        self.prompt="go"; self.tokens=[10,11]; self.response=""; self.response_length=0
        self.loss_mask=None; self.status=None
async def generate(prompt, params):
    assert prompt=="go" and params=={"temperature":0.7}
    return {"token_ids":[20,21,22],"text":"done","finish_reason":"stop"}
s=Sample(); old=s.tokens
out=asyncio.run({fn}(SimpleNamespace(generate=generate),s,{"temperature":0.7}))
assert out is s and s.tokens==[10,11,20,21,22] and old==[10,11]
assert s.response_length==3 and s.loss_mask==[1,1,1]
assert s.response=="done" and s.status==Sample.Status.COMPLETED
""" },
        {"name": "Keeps length and abort distinct from success", "visibility": "unshown", "behavior": "state.invariant", "failure_message": "A length stop or abort must not be marked COMPLETED.", "code": r"""
import asyncio
from types import SimpleNamespace
class Sample:
    class Status:
        COMPLETED="completed"; TRUNCATED="truncated"; ABORTED="aborted"
    def __init__(self):
        self.prompt="go"; self.tokens=[1]; self.response=""; self.response_length=0
        self.loss_mask=None; self.status=None
for reason,want in [("length","truncated"),("abort","aborted")]:
    async def generate(prompt,params): return {"token_ids":[3],"text":"x","finish_reason":reason}
    s=Sample()
    asyncio.run({fn}(SimpleNamespace(generate=generate),s,{}))
    assert s.status==want and s.response_length==1 and s.loss_mask==[1]
""" },
        {"name": "Seeded prompt and response lengths preserve the Sample contract", "visibility": "unshown", "behavior": "rl.rollout_assembly", "failure_message": "Only generated tokens extend the prompt and response-only fields must stay aligned.", "code": r"""
import asyncio, random
from types import SimpleNamespace
class Sample:
    class Status:
        COMPLETED="completed"; TRUNCATED="truncated"; ABORTED="aborted"
    def __init__(self, prompt, tokens):
        self.prompt=prompt; self.tokens=tokens; self.response=""
        self.response_length=0; self.loss_mask=None; self.status=None
for seed in (11, 23, 37):
    rng = random.Random(seed)
    prompt_ids = [rng.randrange(100) for _ in range(rng.randrange(1, 5))]
    reply_ids = [rng.randrange(100, 200) for _ in range(rng.randrange(0, 5))]
    reason = rng.choice(("stop", "length", "abort"))
    sample = Sample(f"prompt-{seed}", prompt_ids[:])
    old_tokens = sample.tokens
    calls = []
    async def generate(prompt, params):
        calls.append((prompt, params))
        return {"token_ids": reply_ids, "text": f"reply-{seed}", "finish_reason": reason}
    result = asyncio.run({fn}(SimpleNamespace(generate=generate), sample, {"seed": seed}))
    assert result is sample and calls == [(f"prompt-{seed}", {"seed": seed})]
    assert sample.tokens == prompt_ids + reply_ids and old_tokens == prompt_ids
    assert sample.tokens is not old_tokens
    assert sample.response_length == len(reply_ids) and sample.loss_mask == [1] * len(reply_ids)
    assert sample.response == f"reply-{seed}"
    assert sample.status == {"stop":"completed", "length":"truncated", "abort":"aborted"}[reason]
""" },
        {"name": "Rejects malformed generation", "visibility": "unshown", "behavior": "contract.signature", "failure_message": "Unknown finish reasons or invalid token ids must raise ValueError.", "code": r"""
import asyncio
from types import SimpleNamespace
class Sample:
    class Status:
        COMPLETED="completed"; TRUNCATED="truncated"; ABORTED="aborted"
    prompt="p"; tokens=[1]; response=""; response_length=0; loss_mask=None; status=None
for output in ({"token_ids":[2],"text":"x","finish_reason":"unknown"},
               {"token_ids":["2"],"text":"x","finish_reason":"stop"}):
    async def generate(prompt,params): return output
    try: asyncio.run({fn}(SimpleNamespace(generate=generate),Sample(),{}))
    except ValueError: pass
    else: raise AssertionError("malformed generation accepted")
""" },
    ],
    "solution": '''async def slime_custom_generate_hook(args, sample, sampling_params):
    output = await args.generate(sample.prompt, sampling_params)
    ids = output["token_ids"]
    reason = output["finish_reason"]
    if not isinstance(ids, list) or any(not isinstance(x, int) or isinstance(x, bool) for x in ids):
        raise ValueError("token_ids must be a list of integers")
    statuses = {
        "stop": sample.Status.COMPLETED,
        "length": sample.Status.TRUNCATED,
        "abort": sample.Status.ABORTED,
    }
    if reason not in statuses:
        raise ValueError("unknown finish reason")
    sample.tokens = list(sample.tokens) + ids
    sample.response_length = len(ids)
    sample.loss_mask = [1] * len(ids)
    sample.response = output["text"]
    sample.status = statuses[reason]
    return sample
''',
}
