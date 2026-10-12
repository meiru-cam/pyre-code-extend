"""Coconut curriculum sample: replace the first k reasoning steps with k * c continuous thoughts."""

from ._interview import interview

TASK = {
    "title": "Coconut Curriculum Stage Sample",
    "difficulty": "Medium",
    "version": 1,
    "function_name": "build_stage_sample",
    "description_en": r"""Coconut teaches a language model to reason in latent space with a multi-stage curriculum. At stage `k`, the first `k` written reasoning steps of a chain-of-thought sample are removed and `k * c` latent placeholder tokens take their place between a `<bot>` and an `<eot>` marker. Build one training sample for a given stage.

**Signature:** `build_stage_sample(question_ids, step_ids, answer_ids, stage, c_thought, max_latent_stage, pad_latent_to_max, special_ids, no_special_marker=False) -> dict`

**Parameters:**
- `question_ids` — list of int token ids for the question.
- `step_ids` — list of reasoning steps, each a list of int token ids. It may be empty, and a step may be empty.
- `answer_ids` — list of int token ids for the answer.
- `stage` — int `>= 0`, the curriculum stage `k` of this sample.
- `c_thought` — int `>= 1`, latent tokens per replaced step (`c`).
- `max_latent_stage` — int `>= 0`, the last stage that still replaces steps one at a time.
- `pad_latent_to_max` — bool, only used when `stage > max_latent_stage`.
- `special_ids` — dict with int entries `"bot"`, `"latent"` and `"eot"`.
- `no_special_marker` — bool. When True, leave out the `<bot>` and `<eot>` tokens.

**Returns:** a dict with exactly the keys `"input_ids"`, `"labels"` and `"position_ids"`. Each value is a 1-D `torch.long` tensor, and all three have the same length.

**How many steps to skip and how many latent groups to insert.**
- When `stage <= max_latent_stage`: skip the first `stage` steps and insert `stage` groups. This holds even when `stage` is larger than the number of steps; then every step is skipped and there are still `stage` groups.
- When `stage > max_latent_stage`: skip every step. Insert `max_latent_stage` groups if `pad_latent_to_max` is True, otherwise `min(len(step_ids), max_latent_stage)` groups.
- Each group is `c_thought` copies of `special_ids["latent"]`.

**Layout:**

    input_ids = question + [bot] + [latent] * (groups * c_thought) + [eot] + remaining steps (concatenated, in order) + answer

With `no_special_marker`, the `[bot]` and `[eot]` entries are absent.

**Constraints:**
- `labels` is `-100` at every question, marker and latent position, and equals `input_ids` at every position after them (remaining steps and answer). Do not shift the labels; the model shifts them.
- `position_ids` is `0, 1, ..., len(input_ids) - 1`.
- Stage 0 still has the `<bot><eot>` pair, with an empty latent span between them.
- Do not modify the input lists.
- Raise `ValueError` when `stage` or `max_latent_stage` is not an int `>= 0`, or `c_thought` is not an int `>= 1` (bools are rejected).

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Continuous thoughts.** Coconut (Chain of Continuous Thought) lets the model reason without writing words: after `<bot>`, the last hidden state at each step is fed back as the next input embedding instead of being decoded to a token. The latent placeholder ids in this sample only reserve those positions; at training time their embeddings are overwritten by the fed-back hidden states (the `coconut_latent_forward` idea).

**Why a curriculum.** Training directly with no language reasoning fails, so Coconut starts from a chain-of-thought model and, stage by stage, swaps the first written steps for continuous thoughts. The loss stays on the remaining written steps and the answer, so the latent thoughts are trained only through their effect on what comes after them.

**GSM8k settings.** The official config uses `c_thought = 2`, `max_latent_stage = 3` and `pad_latent_to_max = True`, with three epochs per stage. After stage 3, every written step is removed and the model keeps six latent thoughts. Without padding, a short problem gets fewer thoughts than a long one in that final stage.

**Upstream extras left out.** The official builder can also draw a random stage per sample with probability `uniform_prob` (0 for GSM8k) and has a `no_cot` ablation that removes every step and every latent.""",
    "advisory_prerequisites": ["response_token_mask"],
    "hints": [
        {"level": 1, "kind": "questions", "content": "Which two numbers decide a sample: how many steps are skipped and how many latent groups appear? Do they move together at every stage, or only up to max_latent_stage? Once the input sequence is assembled, how long is the prefix that the loss must ignore, and what does it contain besides the question?"},
        {"level": 2, "kind": "analysis", "content": "First validate the three integers. Then branch on stage versus max_latent_stage to get the number of skipped steps and the number of groups; only the branch above the maximum looks at pad_latent_to_max, and it skips every step. Multiply groups by c_thought. Build the prefix (question, optional bot, latents, optional eot) and the tail (remaining steps flattened, then the answer) as separate lists, so the label mask is just the prefix length. Pitfalls: forgetting the two marker positions in the mask, slicing steps from the end, and pre-shifting labels."},
    ],
    "model_connections": [
        "Coconut (Hao et al. 2024, Meta) trains GPT-2 on GSM8k with c = 2 thoughts per step, three curriculum stages and a final stage with all steps removed.",
        "The same 'replace written reasoning step by step' curriculum appears in implicit chain-of-thought work (iCoT, stepwise internalization), which removes tokens without adding latent slots.",
        "Pause tokens also insert inert placeholders before the answer, but they carry no fed-back hidden state and need no curriculum.",
    ],
    "pro_con_analysis": {
        "pros": ["A simple data transform that turns any chain-of-thought dataset into a latent-reasoning curriculum, with the loss kept on the language the model still writes."],
        "cons": ["The schedule (c, number of stages, epochs per stage, padding) is hand-tuned per dataset, and each stage needs a full pass with sequential latent forward passes."],
    },
    "sources": [
        {
            "kind": "code",
            "url": "https://github.com/facebookresearch/coconut/blob/27273cb8cca4bb763c041a63b036d0c3b7cbbb48/dataset.py",
            "commit": "27273cb8cca4bb763c041a63b036d0c3b7cbbb48",
            "path": "dataset.py",
            "symbol": "get_cot_latent_dataset.process_dataset (lines 243-302)",
            "license": "MIT",
            "adapted": "Same skip and latent counts per stage, same pad_latent_to_max rule, same token layout and the same -100 label prefix of question + latents + markers.",
            "simplifications": "Runs uniform_prob = 0 only (no random stage draw) and drops the no_cot ablation; returns torch.long tensors instead of lists and drops "
                               "attention_mask and idx; upstream's 'skip all' sentinel (slice from 10000) becomes slicing past the last step; adds argument validation.",
        },
        {
            "kind": "code",
            "url": "https://github.com/facebookresearch/coconut/blob/27273cb8cca4bb763c041a63b036d0c3b7cbbb48/args/gsm_coconut.yaml",
            "commit": "27273cb8cca4bb763c041a63b036d0c3b7cbbb48",
            "path": "args/gsm_coconut.yaml",
            "symbol": "c_thought, max_latent_stage, pad_latent_to_max, uniform_prob",
            "license": "MIT",
            "adapted": "The GSM8k values (c_thought 2, max_latent_stage 3, pad_latent_to_max True, uniform_prob 0) used in the schedule test.",
            "simplifications": "Only the four keys the sample builder reads are used.",
        },
        {
            "kind": "paper",
            "url": "https://arxiv.org/abs/2412.06769",
            "section": "Hao et al. 2024 (Coconut), Section 3 (Training Procedure, Figure 2) and Section 4.1 (Experimental Setup, GSM8k c = 2)",
            "note": "Multi-stage curriculum: stage k replaces the first k language steps with k * c continuous thoughts and masks the loss on questions and latent thoughts.",
        },
    ],
    "tests": [
        {"name": "Stage 0 keeps every step behind an empty latent span", "behavior": "rl.rollout_assembly", "code": r"""
import torch
ids = {"bot": 90, "latent": 91, "eot": 92}
out = {fn}([1, 2], [[10, 11], [20], [30, 31]], [40, 41], 0, 2, 3, True, ids)
assert set(out) == {"input_ids", "labels", "position_ids"}, set(out)
assert out["input_ids"].tolist() == [1, 2, 90, 92, 10, 11, 20, 30, 31, 40, 41], out["input_ids"]
assert out["labels"].tolist() == [-100, -100, -100, -100, 10, 11, 20, 30, 31, 40, 41], out["labels"]
assert out["position_ids"].tolist() == list(range(11)), out["position_ids"]
"""},
        {"name": "Stage 2 replaces two steps with 2 * c latents", "behavior": "rl.rollout_assembly", "code": r"""
import torch
ids = {"bot": 90, "latent": 91, "eot": 92}
out = {fn}([1, 2], [[10, 11], [20], [30, 31]], [40, 41], 2, 2, 3, True, ids)
assert out["input_ids"].tolist() == [1, 2, 90, 91, 91, 91, 91, 92, 30, 31, 40, 41], out["input_ids"]
assert out["labels"].tolist() == [-100] * 8 + [30, 31, 40, 41], out["labels"]
assert out["input_ids"].dtype == torch.long
"""},
        {"name": "Follows the GSM8k schedule at every stage", "visibility": "unshown", "behavior": "rl.rollout_assembly", "failure_message": "Up to max_latent_stage, skip the first `stage` steps and insert stage * c_thought latents; above it, skip all steps and insert max_latent_stage * c_thought latents.", "code": r"""
ids = {"bot": 7, "latent": 8, "eot": 9}
q, steps, a = [100, 101, 102], [[200, 201], [210], [220, 221, 222], [230]], [300, 301]
want = {
    0: [200, 201, 210, 220, 221, 222, 230],
    1: [210, 220, 221, 222, 230],
    2: [220, 221, 222, 230],
    3: [230],
    4: [],
    5: [],
}
for stage, rest in want.items():
    n_latent = 2 * min(stage, 3)
    out = {fn}(q, steps, a, stage, 2, 3, True, ids)
    expected = q + [7] + [8] * n_latent + [9] + rest + a
    assert out["input_ids"].tolist() == expected, (stage, out["input_ids"].tolist(), expected)
"""},
        {"name": "Above max_latent_stage every step is skipped", "visibility": "unshown", "behavior": "rl.rollout_assembly", "failure_message": "Once stage > max_latent_stage, drop all reasoning steps, not just the first max_latent_stage of them, and keep max_latent_stage groups.", "code": r"""
ids = {"bot": 7, "latent": 8, "eot": 9}
steps = [[11], [12], [13], [14], [15]]
for stage in (3, 4, 9):
    out = {fn}([1], steps, [50], stage, 3, 2, True, ids)
    assert out["input_ids"].tolist() == [1, 7] + [8] * 6 + [9, 50], (stage, out["input_ids"].tolist())
"""},
        {"name": "pad_latent_to_max decides the final-stage group count", "visibility": "unshown", "behavior": "edge.empty_or_boundary", "failure_message": "Above max_latent_stage, use max_latent_stage groups when pad_latent_to_max is True, else min(len(step_ids), max_latent_stage).", "code": r"""
ids = {"bot": 7, "latent": 8, "eot": 9}
steps = [[11, 12], [13]]
padded = {fn}([1], steps, [50], 4, 2, 3, True, ids)
assert padded["input_ids"].tolist() == [1, 7] + [8] * 6 + [9, 50], padded["input_ids"].tolist()
short = {fn}([1], steps, [50], 4, 2, 3, False, ids)
assert short["input_ids"].tolist() == [1, 7] + [8] * 4 + [9, 50], short["input_ids"].tolist()
long_steps = [[11], [12], [13], [14]]
unpadded = {fn}([1], long_steps, [50], 4, 2, 3, False, ids)
assert unpadded["input_ids"].tolist() == [1, 7] + [8] * 6 + [9, 50], unpadded["input_ids"].tolist()
"""},
        {"name": "Within max, the group count is the stage even past the last step", "visibility": "unshown", "behavior": "edge.empty_or_boundary", "failure_message": "When stage <= max_latent_stage, insert exactly stage * c_thought latents, even if the sample has fewer than `stage` steps and pad_latent_to_max is False.", "code": r"""
ids = {"bot": 7, "latent": 8, "eot": 9}
for pad in (True, False):
    out = {fn}([1, 2], [[11]], [50], 3, 2, 3, pad, ids)
    assert out["input_ids"].tolist() == [1, 2, 7] + [8] * 6 + [9, 50], (pad, out["input_ids"].tolist())
    assert out["labels"].tolist() == [-100] * 10 + [50], (pad, out["labels"].tolist())
"""},
        {"name": "Labels mask the question, both markers and the latents", "visibility": "unshown", "behavior": "rl.masking", "failure_message": "Set labels to -100 for len(question) + 2 + n_latent positions (the markers count) and copy input_ids, unshifted, for the rest.", "code": r"""
ids = {"bot": 7, "latent": 8, "eot": 9}
out = {fn}([1, 2, 3], [[11, 12], [13], [14]], [50, 51], 1, 3, 3, True, ids)
inp, lab = out["input_ids"].tolist(), out["labels"].tolist()
assert inp == [1, 2, 3, 7, 8, 8, 8, 9, 13, 14, 50, 51], inp
assert lab == [-100] * 8 + [13, 14, 50, 51], lab
"""},
        {"name": "no_special_marker drops the markers and their mask", "visibility": "unshown", "behavior": "rl.masking", "failure_message": "With no_special_marker=True, emit no bot/eot tokens and mask only len(question) + n_latent positions.", "code": r"""
ids = {"bot": 7, "latent": 8, "eot": 9}
out = {fn}([1, 2], [[11], [12], [13]], [50], 2, 2, 3, True, ids, no_special_marker=True)
assert out["input_ids"].tolist() == [1, 2, 8, 8, 8, 8, 13, 50], out["input_ids"].tolist()
assert out["labels"].tolist() == [-100] * 6 + [13, 50], out["labels"].tolist()
assert out["position_ids"].tolist() == list(range(8))
plain = {fn}([1, 2], [[11], [12]], [50], 0, 2, 3, True, ids, no_special_marker=True)
assert plain["input_ids"].tolist() == [1, 2, 11, 12, 50] and plain["labels"].tolist() == [-100, -100, 11, 12, 50]
"""},
        {"name": "Handles empty steps and max_latent_stage 0", "visibility": "unshown", "behavior": "edge.empty_or_boundary", "failure_message": "An empty step list or empty steps are fine; with max_latent_stage 0, any stage >= 1 skips all steps and inserts no latents.", "code": r"""
ids = {"bot": 7, "latent": 8, "eot": 9}
out = {fn}([1], [], [50], 0, 2, 3, False, ids)
assert out["input_ids"].tolist() == [1, 7, 9, 50] and out["labels"].tolist() == [-100, -100, -100, 50]
out = {fn}([1], [], [50], 5, 2, 3, False, ids)
assert out["input_ids"].tolist() == [1, 7, 9, 50], out["input_ids"].tolist()
out = {fn}([1], [[], [12]], [50], 1, 2, 3, True, ids)
assert out["input_ids"].tolist() == [1, 7, 8, 8, 9, 12, 50], out["input_ids"].tolist()
out = {fn}([1], [[11], [12]], [50], 1, 2, 0, True, ids)
assert out["input_ids"].tolist() == [1, 7, 9, 50], out["input_ids"].tolist()
out = {fn}([1], [[11], [12]], [50], 0, 2, 0, True, ids)
assert out["input_ids"].tolist() == [1, 7, 9, 11, 12, 50], out["input_ids"].tolist()
"""},
        {"name": "Seeded random samples match the specification", "visibility": "unshown", "behavior": "rl.rollout_assembly", "failure_message": "Check the skip count, the latent count and the label prefix length for every combination of stage, c_thought, max_latent_stage, padding and markers.", "code": r"""
import random
def spec(q, steps, a, stage, c, m, pad, ids, nsm):
    if stage > m:
        skip, groups = len(steps), (m if pad else min(len(steps), m))
    else:
        skip, groups = stage, stage
    head = q + ([] if nsm else [ids["bot"]]) + [ids["latent"]] * (groups * c) + ([] if nsm else [ids["eot"]])
    tail = [t for s in steps[skip:] for t in s] + a
    return head + tail, [-100] * len(head) + tail
ids = {"bot": 1001, "latent": 1002, "eot": 1003}
for seed in range(6):
    rng = random.Random(seed)
    for _ in range(40):
        q = [rng.randrange(1000) for _ in range(rng.randint(0, 5))]
        steps = [[rng.randrange(1000) for _ in range(rng.randint(0, 4))] for _ in range(rng.randint(0, 6))]
        a = [rng.randrange(1000) for _ in range(rng.randint(1, 3))]
        stage, c, m = rng.randint(0, 8), rng.randint(1, 3), rng.randint(0, 4)
        pad, nsm = rng.random() < 0.5, rng.random() < 0.3
        out = {fn}(q, steps, a, stage, c, m, pad, ids, no_special_marker=nsm)
        inp, lab = spec(q, steps, a, stage, c, m, pad, ids, nsm)
        assert out["input_ids"].tolist() == inp, (seed, stage, c, m, pad, nsm)
        assert out["labels"].tolist() == lab, (seed, stage, c, m, pad, nsm)
        assert out["position_ids"].tolist() == list(range(len(inp)))
"""},
        {"name": "Returns 1-D long tensors of equal length", "visibility": "unshown", "behavior": "tensor.dtype_device", "failure_message": "Return exactly input_ids, labels and position_ids as 1-D torch.long tensors of the same length, position_ids starting at 0.", "code": r"""
import torch
ids = {"bot": 7, "latent": 8, "eot": 9}
for args in (([1, 2], [[3], [4]], [5], 1, 2, 3, True), ([], [], [5], 0, 1, 0, False)):
    out = {fn}(*args, ids)
    assert set(out) == {"input_ids", "labels", "position_ids"}, set(out)
    n = out["input_ids"].shape[0]
    for key, value in out.items():
        assert torch.is_tensor(value) and value.dtype == torch.long and value.ndim == 1, (key, value)
        assert value.shape[0] == n, (key, value.shape)
    assert out["position_ids"].tolist() == list(range(n))
"""},
        {"name": "Does not modify the input lists", "visibility": "unshown", "behavior": "state.invariant", "failure_message": "Build new lists; never pop, extend or slice-assign question_ids, step_ids or answer_ids.", "code": r"""
import copy
ids = {"bot": 7, "latent": 8, "eot": 9}
q, steps, a = [1, 2], [[11, 12], [13], [14]], [50]
before = copy.deepcopy((q, steps, a, ids))
for stage in (0, 2, 5):
    {fn}(q, steps, a, stage, 2, 3, False, ids)
    assert (q, steps, a, ids) == before, (stage, q, steps, a, ids)
"""},
        {"name": "Rejects invalid stage, c_thought and max_latent_stage", "visibility": "unshown", "behavior": "contract.signature", "failure_message": "Raise ValueError for a negative or non-int stage or max_latent_stage, a c_thought below 1, and bools.", "code": r"""
ids = {"bot": 7, "latent": 8, "eot": 9}
bad = [(-1, 2, 3), (1.0, 2, 3), (True, 2, 3), (1, 0, 3), (1, 1.5, 3), (1, True, 3), (1, 2, -1), (1, 2, 2.0)]
for stage, c, m in bad:
    try:
        {fn}([1], [[2]], [3], stage, c, m, True, ids)
    except ValueError:
        pass
    else:
        raise AssertionError(f"accepted stage={stage!r} c_thought={c!r} max_latent_stage={m!r}")
"""},
    ],
    "solution": '''import torch


def build_stage_sample(question_ids, step_ids, answer_ids, stage, c_thought, max_latent_stage,
                       pad_latent_to_max, special_ids, no_special_marker=False):
    for name, value, low in (("stage", stage, 0), ("c_thought", c_thought, 1),
                             ("max_latent_stage", max_latent_stage, 0)):
        if isinstance(value, bool) or not isinstance(value, int) or value < low:
            raise ValueError(f"{name} must be an int >= {low}")

    if stage > max_latent_stage:
        n_skip = len(step_ids)
        if pad_latent_to_max:
            n_groups = max_latent_stage
        else:
            n_groups = min(len(step_ids), max_latent_stage)
    else:
        n_skip, n_groups = stage, stage
    n_latent = n_groups * c_thought

    bot = [] if no_special_marker else [special_ids["bot"]]
    eot = [] if no_special_marker else [special_ids["eot"]]
    prefix = list(question_ids) + bot + [special_ids["latent"]] * n_latent + eot
    tail = [token for step in step_ids[n_skip:] for token in step] + list(answer_ids)

    input_ids = prefix + tail
    labels = [-100] * len(prefix) + tail
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "position_ids": torch.arange(len(input_ids), dtype=torch.long),
    }
''',
    "interview_questions": interview(
        concept=[
            "In Coconut's curriculum, what changes in a training sample from stage k to stage k + 1?",
            "Why is the loss masked on the latent positions, and how do the continuous thoughts still get a training signal?",
        ],
        deep_dive=[
            "Walk through a GSM8k sample with four reasoning steps at stages 0 through 4 using c = 2 and max_latent_stage = 3. How many latent tokens and written steps does each stage have?",
            "What does pad_latent_to_max change, at which stages does it matter, and why might padding help or hurt?",
            "The label mask has the length of the question plus the latents plus two. What breaks in training if the two marker positions are left unmasked?",
        ],
        tradeoffs=[
            "A hand-scheduled curriculum versus training with all steps replaced by latent thoughts from the start?",
            "Choosing c, the number of latent thoughts per removed step: accuracy, training cost and inference latency?",
        ],
    ),
}
