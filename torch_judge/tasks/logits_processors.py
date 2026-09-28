"""Sampling-time logits processors: repetition penalty, temperature, top-k and top-p."""

from ._interview import interview

TASK = {
    "title": "Logits Processor Pipeline",
    "difficulty": "Medium",
    "version": 1,
    "function_name": "process_logits",
    "description_en": r"""Apply the standard decoding-time logits processors in a fixed order.

**Signature:** `process_logits(logits, generated_ids, temperature=1.0, top_k=0, top_p=1.0, repetition_penalty=1.0) -> Tensor`

**Parameters:**
- `logits` — float tensor `(B, V)` for the next token.
- `generated_ids` — list of B lists of token ids already in each sequence (prompt plus generated).
- `temperature` — positive float.
- `top_k` — integer; 0 disables it.
- `top_p` — float in `(0, 1]`; 1.0 disables it.
- `repetition_penalty` — float `>= 1`; 1.0 disables it.

**Returns:** a new float tensor `(B, V)`. Removed tokens are `-inf`. Apply these steps in order:

- **Repetition penalty.** For each distinct token id in `generated_ids[b]`, divide its logit by `repetition_penalty` if the logit is positive, otherwise multiply by it.
- **Temperature.** Divide all logits by `temperature`.
- **Top-k.** Keep every token whose logit is `>=` the k-th largest logit in its row, so tied tokens are all kept.
- **Top-p.** Over the surviving tokens, sort by probability in descending order and keep the smallest prefix whose cumulative probability reaches `top_p`. Always keep the most likely token.

**Constraints:**
- Do not modify `logits` in place.
- Each row is processed independently.
- Raise `ValueError` for `temperature <= 0`, `top_k < 0`, `top_p` outside `(0, 1]`, `repetition_penalty < 1`, or `len(generated_ids) != B`.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why the order matters.** Temperature changes the probabilities that top-p thresholds, so top-p after temperature keeps a different set than top-p before it. The order above matches Hugging Face `generate` and most serving engines.

**Why the sign rule in the repetition penalty.** Dividing a negative logit by a number greater than one would raise it and make the token more likely. Multiplying negatives and dividing positives always lowers the token's score.""",
    "advisory_prerequisites": ["topk_sampling", "softmax"],
    "hints": [
        {"level": 1, "kind": "questions", "content": "Why does the repetition penalty treat positive and negative logits differently? In top-p, is the token that pushes the cumulative sum past p kept or removed?"},
        {"level": 2, "kind": "analysis", "content": "Clone the logits. Loop over rows for the penalty using a set of ids. Divide by temperature. For top-k compare against torch.topk(x, k).values[:, -1:]. For top-p sort descending, softmax, cumsum, and remove positions where cumsum minus the token's own probability is already >= top_p; scatter the removal mask back to vocabulary order."},
    ],
    "model_connections": [
        "Hugging Face's LogitsProcessorList, vLLM's sampler and llama.cpp's sampler chain all apply these processors, with configurable order in llama.cpp.",
    ],
    "pro_con_analysis": {
        "pros": ["Cheap, model-agnostic control over diversity and repetition at inference time."],
        "cons": ["The knobs interact, and the repetition penalty also punishes tokens that should repeat, such as code syntax or names."],
    },
    "tests": [
        {"name": "Each processor on a small row", "behavior": "tensor.shape", "code": r"""
import math, torch
logits = torch.tensor([[2.0, -1.0, 1.0, 0.5]])
out = {fn}(logits, [[0, 1]], repetition_penalty=2.0)
assert torch.allclose(out, torch.tensor([[1.0, -2.0, 1.0, 0.5]])), out
out = {fn}(logits, [[]], temperature=0.5, top_k=2)
assert out[0, 0] == 4.0 and out[0, 2] == 2.0 and math.isinf(out[0, 1]) and math.isinf(out[0, 3]), out
out = {fn}(torch.log(torch.tensor([[0.5, 0.3, 0.15, 0.05]])), [[]], top_p=0.7)
assert torch.isfinite(out[0, :2]).all() and torch.isinf(out[0, 2:]).all(), out
"""},
        {"name": "Matches a seeded loop-based oracle", "visibility": "unshown", "behavior": "tensor.shape", "failure_message": "Apply penalty, temperature, top-k with ties kept, then top-p over the survivors, in that order.", "code": r"""
import math, random, torch
def oracle(row, ids, temperature, top_k, top_p, penalty):
    row = list(row)
    for i in set(ids):
        row[i] = row[i] / penalty if row[i] > 0 else row[i] * penalty
    row = [x / temperature for x in row]
    if top_k:
        kth = sorted(row, reverse=True)[top_k - 1]
        row = [x if x >= kth else -math.inf for x in row]
    if top_p < 1.0:
        m = max(row); exps = [math.exp(x - m) for x in row]; z = sum(exps)
        order = sorted(range(len(row)), key=lambda i: -row[i])
        keep, cum = set(), 0.0
        for i in order:
            keep.add(i); cum += exps[i] / z
            if cum >= top_p:
                break
        row = [x if i in keep else -math.inf for i, x in enumerate(row)]
    return row
for seed in (9, 26, 83):
    rng = random.Random(seed)
    B, V = rng.randint(2, 4), rng.randint(6, 12)
    rows = [[rng.gauss(0, 2) for _ in range(V)] for _ in range(B)]
    ids = [[rng.randrange(V) for _ in range(rng.randint(0, 5))] for _ in range(B)]
    for temperature, top_k, top_p, penalty in ((0.7, 0, 0.8, 1.3), (1.5, 4, 1.0, 1.0), (1.0, 5, 0.6, 2.0), (0.4, 0, 1.0, 1.7)):
        x = torch.tensor(rows, dtype=torch.float64)
        out = {fn}(x, ids, temperature=temperature, top_k=top_k, top_p=top_p, repetition_penalty=penalty)
        assert torch.equal(x, torch.tensor(rows, dtype=torch.float64)), "modified logits in place"
        for b in range(B):
            want = oracle(rows[b], ids[b], temperature, top_k, top_p, penalty)
            for v in range(V):
                got = out[b, v].item()
                assert (math.isinf(got) and math.isinf(want[v])) or abs(got - want[v]) < 1e-9, (seed, b, v, got, want[v])
"""},
        {"name": "Top-k keeps ties and top-p keeps the best token", "visibility": "unshown", "behavior": "edge.empty_or_boundary", "failure_message": "Tokens tied with the k-th logit stay; top-p always keeps at least the most likely token.", "code": r"""
import torch
out = {fn}(torch.tensor([[3.0, 1.0, 1.0, 0.0]]), [[]], top_k=2)
assert torch.isfinite(out[0, :3]).all() and torch.isinf(out[0, 3]), out
out = {fn}(torch.tensor([[5.0, 0.0, -1.0]]), [[]], top_p=0.01)
assert torch.isfinite(out[0, 0]) and torch.isinf(out[0, 1:]).all(), out
"""},
        {"name": "Rejects invalid settings", "visibility": "unshown", "behavior": "contract.signature", "failure_message": "Out-of-range settings and a generated_ids list of the wrong length must raise ValueError.", "code": r"""
import torch
x = torch.zeros(2, 5)
cases = [dict(temperature=0.0), dict(top_k=-1), dict(top_p=0.0), dict(top_p=1.5), dict(repetition_penalty=0.5)]
for kwargs in cases:
    try:
        {fn}(x, [[], []], **kwargs)
    except ValueError:
        pass
    else:
        raise AssertionError(f"accepted {kwargs}")
try:
    {fn}(x, [[]])
except ValueError:
    pass
else:
    raise AssertionError("accepted one id list for two rows")
"""},
    ],
    "solution": '''import torch

def process_logits(logits, generated_ids, temperature=1.0, top_k=0, top_p=1.0, repetition_penalty=1.0):
    if temperature <= 0 or top_k < 0 or not 0 < top_p <= 1 or repetition_penalty < 1:
        raise ValueError("invalid sampling settings")
    if len(generated_ids) != logits.shape[0]:
        raise ValueError("need one id list per row")
    out = logits.clone()
    if repetition_penalty != 1.0:
        for b, ids in enumerate(generated_ids):
            if not ids:
                continue
            idx = torch.tensor(sorted(set(ids)), dtype=torch.long)
            seen = out[b, idx]
            out[b, idx] = torch.where(seen > 0, seen / repetition_penalty, seen * repetition_penalty)
    out = out / temperature
    if top_k:
        kth = torch.topk(out, min(top_k, out.shape[-1]), dim=-1).values[:, -1:]
        out = out.masked_fill(out < kth, float("-inf"))
    if top_p < 1.0:
        sorted_logits, order = torch.sort(out, dim=-1, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        before = torch.cumsum(probs, dim=-1) - probs
        remove_sorted = before >= top_p
        remove_sorted[:, 0] = False
        remove = torch.zeros_like(remove_sorted).scatter(-1, order, remove_sorted)
        out = out.masked_fill(remove, float("-inf"))
    return out
''',
    "interview_questions": interview(
        concept=[
            "What do temperature, top-k and top-p each do to the next-token distribution?",
            "Why does the repetition penalty divide positive logits but multiply negative ones?",
        ],
        deep_dive=[
            "Walk through implementing top-p on a batch. Is the token that crosses the threshold kept, and how do you map the mask back from sorted order?",
            "Why does the order of temperature and top-p matter? Give a concrete example.",
            "How should top-k treat ties, and what does torch.topk do with them?",
        ],
        tradeoffs=[
            "Top-p versus top-k versus min-p for creative writing and for code: which would you use and why?",
            "Repetition penalty versus frequency and presence penalties: what does each punish, and when do they hurt quality?",
        ],
    ),
}
