"""Cross-provider context handoff: rewrite a transcript so another model can continue it."""

from ._interview import interview

TASK = {
    "title": "Cross-Model Transcript Handoff",
    "difficulty": "Hard",
    "version": 1,
    "function_name": "transform_messages",
    "description_en": r"""Rewrite a conversation so a different model, possibly from another provider, can continue it. This is what the Pi agent does every time the user switches models mid-session.

**Signature:** `transform_messages(messages, model, normalize_tool_call_id=None) -> list[dict]`

**Messages** are dicts with a `role`:
- `system` and `user` — `content` is a string or a list of blocks `{"type": "text", "text"}` / `{"type": "image", ...}`.
- `assistant` — `provider`, `api`, `model`, `stop_reason`, and `content`: a list of `{"type": "thinking", "thinking", "signature"?, "redacted"?}`, `{"type": "text", "text", ...}` and `{"type": "toolCall", "id", "name", "arguments", "thought_signature"?}` blocks.
- `toolResult` — `tool_call_id`, `tool_name`, `is_error`, and `content`: a list of text and image blocks.

`model` is a dict with `provider`, `api`, `id` and `input`, a list such as `["text", "image"]`. An assistant message is from the *same model* when its `provider`, `api` and `model` equal the target's `provider`, `api` and `id`.

**Pass 1 — rewrite each message in order.**
- A message whose `content` is `None` gets `content = []`.
- If `"image"` is not in `model["input"]`: in a `user` message whose content is a list and in every `toolResult`, replace each image with a text block, `"(image omitted: model does not support images)"` for users and `"(tool image omitted: model does not support images)"` for tool results. Do not add a placeholder when the previous block is already that same placeholder text.
- `assistant` blocks, when the message is from another model:
  - `thinking` with `redacted` true is dropped; other thinking with empty or whitespace-only text is dropped; the rest becomes `{"type": "text", "text": thinking}`;
  - `text` becomes `{"type": "text", "text": text}`, dropping every other key;
  - `toolCall` loses its `thought_signature` key, and if `normalize_tool_call_id` is given its `id` becomes `normalize_tool_call_id(id, model, message)`; remember the mapping from old to new id when they differ.
- `assistant` blocks, when the message is from the same model: a thinking block that has a non-empty `signature` or is `redacted` is kept as is; otherwise a thinking block with empty or whitespace-only text is dropped; all other blocks are kept.
- A `toolResult` whose `tool_call_id` has a remembered mapping gets the new id.

**Pass 2 — repair tool-call pairing.** Walk the rewritten messages with a list of *pending* tool calls, the set of ids answered since they were requested, and a list of *held* system messages. To *close* pending calls: append, for every pending call not yet answered, `{"role": "toolResult", "tool_call_id": id, "tool_name": name, "content": [{"type": "text", "text": "No result provided"}], "is_error": True}`; clear the pending calls and answered ids; then append the held system messages and clear them.
- `assistant`: close pending calls. Then, if its `stop_reason` is `"error"` or `"aborted"`, skip the message. Otherwise append it, and if it has tool calls, they become the pending calls with an empty answered set.
- `toolResult`: add its id to the answered set and append it.
- `system`: hold it if calls are pending, otherwise append it.
- `user`: close pending calls, then append it.
- After the last message, close pending calls.

**Constraints:** Do not modify `messages` or anything inside it.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why thinking becomes text.** Reasoning blocks carry provider-specific signatures that only the model that produced them can verify. Another provider would reject them, so their content is kept as plain text, while redacted reasoning, which is opaque ciphertext, is dropped.

**Why tool-call ids are rewritten.** OpenAI Responses ids can be hundreds of characters long and contain `|`, while Anthropic requires ids that match `^[a-zA-Z0-9_-]+$` with at most 64 characters. The result that answers a call must follow the call's new id.

**Why synthetic results.** Every provider rejects an assistant tool call without a result. A user can interrupt mid-tool-call, and an errored or aborted assistant turn should not be replayed at all, so the transcript is repaired before it is sent.""",
    "advisory_prerequisites": ["tool_call_stream_parser"],
    "hints": [
        {"level": 1, "kind": "questions", "content": "Which fields decide whether an assistant message is from the same model? Why must the id mapping be built before the tool results that use it are rewritten? When is a pending tool call considered orphaned?"},
        {"level": 2, "kind": "analysis", "content": "Deep-copy nothing; build new dicts with {**msg, ...}. Pass 1 is a single loop that fills an id map as it rewrites assistant messages, so later tool results find their new ids. Pass 2 is a small state machine with pending, answered and held lists and one close() helper called on assistant, user and at the end."},
    ],
    "model_connections": [
        "Pi's transformMessages runs before every provider request, which lets a session hop between Anthropic, OpenAI, Google and local models mid-conversation.",
        "LiteLLM and OpenRouter face the same problem when routing one conversation across providers.",
    ],
    "pro_con_analysis": {
        "pros": ["Users can switch models mid-task without losing context, and malformed histories are repaired instead of failing the request."],
        "cons": ["Converted reasoning loses its signature, so the new model sees it as ordinary text, and synthetic results hide what really happened to interrupted calls."],
    },
    "sources": [
        {"kind": "code", "url": "https://github.com/badlogic/pi-mono", "commit": "6f7551516b84278eb9da1c340c8e7bc66be1a6ba", "path": "packages/ai/src/api/transform-messages.ts", "symbol": "transformMessages, downgradeUnsupportedImages and replaceImagesWithPlaceholder", "license": "MIT", "adapted": "Same-model detection, thinking and text conversion, signature stripping, id normalization with result remapping, image placeholders, orphaned-call repair, held system messages and skipping errored turns.", "simplifications": "Snake_case field names, no timestamps on synthetic results, and plain dicts instead of typed messages."},
    ],
    "tests": [
        {"name": "Switching providers converts thinking and repairs an interrupted call", "behavior": "protocol.validation", "code": r"""
target = {"provider": "anthropic", "api": "messages", "id": "claude", "input": ["text", "image"]}
msgs = [
    {"role": "user", "content": "fix the bug"},
    {"role": "assistant", "provider": "openai", "api": "responses", "model": "gpt", "stop_reason": "toolUse", "content": [
        {"type": "thinking", "thinking": "look at main.py", "signature": "sig"},
        {"type": "thinking", "thinking": "enc", "redacted": True},
        {"type": "text", "text": "Reading.", "signature": "t"},
        {"type": "toolCall", "id": "call|1", "name": "read", "arguments": {"path": "main.py"}, "thought_signature": "x"}]},
    {"role": "user", "content": "stop, use utils.py"},
]
out = {fn}(msgs, target, lambda i, m, src: i.replace("|", "_"))
assert out == [
    {"role": "user", "content": "fix the bug"},
    {"role": "assistant", "provider": "openai", "api": "responses", "model": "gpt", "stop_reason": "toolUse", "content": [
        {"type": "text", "text": "look at main.py"}, {"type": "text", "text": "Reading."},
        {"type": "toolCall", "id": "call_1", "name": "read", "arguments": {"path": "main.py"}}]},
    {"role": "toolResult", "tool_call_id": "call_1", "tool_name": "read", "content": [{"type": "text", "text": "No result provided"}], "is_error": True},
    {"role": "user", "content": "stop, use utils.py"},
], out
"""},
        {"name": "Same-model messages keep signed reasoning", "visibility": "unshown", "behavior": "protocol.validation", "failure_message": "For the same model keep signed or redacted thinking and every other block; drop only unsigned empty thinking.", "code": r"""
target = {"provider": "openai", "api": "responses", "id": "gpt", "input": ["text"]}
blocks = [{"type": "thinking", "thinking": "", "signature": "enc"}, {"type": "thinking", "thinking": "  "},
          {"type": "thinking", "thinking": "x", "redacted": True}, {"type": "thinking", "thinking": "plan"},
          {"type": "text", "text": "hi", "signature": "s"},
          {"type": "toolCall", "id": "c|1", "name": "ls", "arguments": {}, "thought_signature": "keep"}]
msg = {"role": "assistant", "provider": "openai", "api": "responses", "model": "gpt", "stop_reason": "toolUse", "content": blocks}
res = {"role": "toolResult", "tool_call_id": "c|1", "tool_name": "ls", "is_error": False, "content": [{"type": "text", "text": "a"}]}
out = {fn}([msg, res], target, lambda i, m, s: "renamed")
assert out[0]["content"] == [blocks[0], blocks[2], blocks[3], blocks[4], blocks[5]], out[0]
assert out[1] == res and len(out) == 2, out
other = dict(msg, api="completions")
out = {fn}([other, res], target, lambda i, m, s: "renamed")
assert out[0]["content"] == [{"type": "text", "text": "plan"}, {"type": "text", "text": "hi"}, {"type": "toolCall", "id": "renamed", "name": "ls", "arguments": {}}], out
assert out[1]["tool_call_id"] == "renamed", out
assert other["content"] is blocks and msg["content"] is blocks and len(blocks) == 6, "input messages were modified"
"""},
        {"name": "Matches a seeded pairing oracle", "visibility": "unshown", "behavior": "protocol.validation", "failure_message": "Close pending calls on assistant, user and at the end; hold system messages while calls are pending; skip errored or aborted assistants.", "code": r"""
import copy, random
NO = {"type": "text", "text": "No result provided"}
def pairing(msgs):
    out, pending, answered, held = [], [], set(), []
    def close():
        nonlocal pending, answered
        for c in pending:
            if c["id"] not in answered:
                out.append({"role": "toolResult", "tool_call_id": c["id"], "tool_name": c["name"], "content": [NO], "is_error": True})
        pending, answered = [], set()
        out.extend(held); held.clear()
    for m in msgs:
        if m["role"] == "assistant":
            close()
            if m["stop_reason"] in ("error", "aborted"):
                continue
            calls = [b for b in m["content"] if b["type"] == "toolCall"]
            if calls:
                pending, answered = calls, set()
            out.append(m)
        elif m["role"] == "toolResult":
            answered.add(m["tool_call_id"]); out.append(m)
        elif m["role"] == "system":
            (held if pending else out).append(m)
        else:
            close(); out.append(m)
    close()
    return out
target = {"provider": "p", "api": "a", "id": "m", "input": ["text", "image"]}
for seed in (2, 40, 77):
    rng = random.Random(seed)
    for trial in range(50):
        msgs, n = [], 0
        for _ in range(rng.randint(1, 10)):
            kind = rng.choice(["user", "system", "assistant", "assistant", "toolResult"])
            if kind == "assistant":
                calls = []
                for _ in range(rng.randint(0, 3)):
                    n += 1; calls.append({"type": "toolCall", "id": f"t{n}", "name": rng.choice(["a", "b"]), "arguments": {}})
                msgs.append({"role": "assistant", "provider": "p", "api": "a", "model": "m", "content": [{"type": "text", "text": "ok"}] + calls,
                             "stop_reason": rng.choice(["stop", "toolUse", "toolUse", "error", "aborted"])})
            elif kind == "toolResult":
                ids = [b["id"] for m in msgs if m["role"] == "assistant" for b in m["content"] if b["type"] == "toolCall"] or ["ghost"]
                cid = rng.choice(ids)
                msgs.append({"role": "toolResult", "tool_call_id": cid, "tool_name": "a", "is_error": False, "content": [{"type": "text", "text": cid}]})
            else:
                msgs.append({"role": kind, "content": f"{kind}-{len(msgs)}"})
        frozen = copy.deepcopy(msgs)
        got = {fn}(msgs, target)
        assert msgs == frozen, "input was modified"
        assert got == pairing(copy.deepcopy(msgs)), (seed, trial, msgs, got)
"""},
        {"name": "Images downgrade to one placeholder per run", "visibility": "unshown", "behavior": "protocol.validation", "failure_message": "For a text-only model replace images in user lists and tool results with the matching placeholder, never twice in a row.", "code": r"""
U = "(image omitted: model does not support images)"
T = "(tool image omitted: model does not support images)"
img = {"type": "image", "data": "..."}
target = {"provider": "x", "api": "y", "id": "z", "input": ["text"]}
msgs = [
    {"role": "user", "content": [img, img, {"type": "text", "text": "see"}, img, {"type": "text", "text": U}, img]},
    {"role": "user", "content": "plain string"},
    {"role": "user", "content": None},
    {"role": "toolResult", "tool_call_id": "q", "tool_name": "shot", "is_error": False, "content": [{"type": "text", "text": "a"}, img, img]},
]
out = {fn}(msgs, target)
assert out[0]["content"] == [{"type": "text", "text": U}, {"type": "text", "text": "see"}, {"type": "text", "text": U}, {"type": "text", "text": U}], out[0]
assert out[1] == msgs[1] and out[2] == {"role": "user", "content": []}, out
assert out[3]["content"] == [{"type": "text", "text": "a"}, {"type": "text", "text": T}], out[3]
vision = dict(target, input=["text", "image"])
assert {fn}(msgs[:1], vision)[0]["content"] == msgs[0]["content"]
"""},
    ],
    "solution": '''NON_VISION_USER = "(image omitted: model does not support images)"
NON_VISION_TOOL = "(tool image omitted: model does not support images)"
NO_RESULT = "No result provided"


def _replace_images(content, placeholder):
    result = []
    previous_was_placeholder = False
    for block in content:
        if block["type"] == "image":
            if not previous_was_placeholder:
                result.append({"type": "text", "text": placeholder})
            previous_was_placeholder = True
            continue
        result.append(block)
        previous_was_placeholder = block.get("text") == placeholder
    return result


def transform_messages(messages, model, normalize_tool_call_id=None):
    supports_images = "image" in model["input"]
    id_map = {}
    transformed = []
    for msg in messages:
        if msg.get("content") is None:
            msg = {**msg, "content": []}
        role = msg["role"]
        if not supports_images:
            if role == "user" and isinstance(msg["content"], list):
                msg = {**msg, "content": _replace_images(msg["content"], NON_VISION_USER)}
            elif role == "toolResult":
                msg = {**msg, "content": _replace_images(msg["content"], NON_VISION_TOOL)}
        if role == "toolResult":
            new_id = id_map.get(msg["tool_call_id"])
            if new_id is not None and new_id != msg["tool_call_id"]:
                msg = {**msg, "tool_call_id": new_id}
        elif role == "assistant":
            same = (msg["provider"] == model["provider"] and msg["api"] == model["api"]
                    and msg["model"] == model["id"])
            content = []
            for block in msg["content"]:
                kind = block["type"]
                if kind == "thinking":
                    if block.get("redacted"):
                        if same:
                            content.append(block)
                        continue
                    if same and block.get("signature"):
                        content.append(block)
                        continue
                    if not block.get("thinking", "").strip():
                        continue
                    content.append(block if same else {"type": "text", "text": block["thinking"]})
                elif kind == "text":
                    content.append(block if same else {"type": "text", "text": block["text"]})
                elif kind == "toolCall":
                    call = block
                    if not same and "thought_signature" in call:
                        call = {k: v for k, v in call.items() if k != "thought_signature"}
                    if not same and normalize_tool_call_id is not None:
                        new_id = normalize_tool_call_id(block["id"], model, msg)
                        if new_id != block["id"]:
                            id_map[block["id"]] = new_id
                            call = {**call, "id": new_id}
                    content.append(call)
                else:
                    content.append(block)
            msg = {**msg, "content": content}
        transformed.append(msg)

    result = []
    pending = []
    answered = set()
    held = []

    def close():
        nonlocal pending, answered
        for call in pending:
            if call["id"] not in answered:
                result.append({"role": "toolResult", "tool_call_id": call["id"], "tool_name": call["name"],
                               "content": [{"type": "text", "text": NO_RESULT}], "is_error": True})
        pending, answered = [], set()
        result.extend(held)
        held.clear()

    for msg in transformed:
        role = msg["role"]
        if role == "assistant":
            close()
            if msg.get("stop_reason") in ("error", "aborted"):
                continue
            calls = [b for b in msg["content"] if b["type"] == "toolCall"]
            if calls:
                pending, answered = calls, set()
            result.append(msg)
        elif role == "toolResult":
            answered.add(msg["tool_call_id"])
            result.append(msg)
        elif role == "system":
            (held if pending else result).append(msg)
        elif role == "user":
            close()
            result.append(msg)
        else:
            result.append(msg)
    close()
    return result
''',
    "interview_questions": interview(
        concept=[
            "What has to change in a conversation history when you switch from one LLM provider to another mid-session?",
            "Why can reasoning blocks from one model not simply be replayed to another?",
        ],
        deep_dive=[
            "Walk through how tool-call ids are normalized and how the matching tool results follow them. Why must the mapping be built while scanning in order?",
            "When is a tool call orphaned, and what does the repair insert? Why are system messages held back while calls are pending?",
            "Why skip errored or aborted assistant messages entirely instead of repairing them?",
        ],
        tradeoffs=[
            "Converting foreign reasoning to text versus dropping it: context quality, token cost and the risk of the new model imitating another model's style?",
            "Repairing malformed histories silently versus failing loudly: when does each help or hurt debugging?",
        ],
    ),
}
