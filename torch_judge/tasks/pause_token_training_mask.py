"""Pause-token finetuning example: append pauses to the prefix and mask the loss the paper ignores."""

from ._interview import interview

TASK = {
    "title": "Pause-Token Training Mask",
    "difficulty": "Easy",
    "version": 1,
    "function_name": "insert_pauses",
    "description_en": r"""Build one pause-finetuning example: append `<pause>` tokens to the prefix, then build causal-LM labels that train only on the answer and never on predicting a pause.

**Signature:** `insert_pauses(prefix_ids, target_ids, pause_id, n_pause) -> (input_ids, labels, answer_start)`

**Parameters:**
- `prefix_ids` — 1-D integer tensor of shape `(N,)` with `N >= 1`. The prompt.
- `target_ids` — 1-D integer tensor of shape `(T,)` with `T >= 0`. The answer.
- `pause_id` — integer id of the `<pause>` token.
- `n_pause` — integer `M >= 0`. The number of pause tokens to append after the prefix.

**Returns:** a tuple of three values.
- `input_ids` — `torch.long` tensor of shape `(N + M + T,)`: the prefix, then `M` copies of `pause_id`, then the target.
- `labels` — `torch.long` tensor of the same shape, a new tensor that does not share memory with `input_ids`.
- `answer_start` — Python `int` equal to `N + M`, the index of the first target token (equal to the length when `T = 0`).

**Label convention.** Labels are aligned with `input_ids`, as Hugging Face causal-LM labels are: `labels[j]` is the token the model must predict at index `j`, and the model shifts internally so the logits at `j - 1` are scored against `labels[j]`. They are not pre-shifted.

**The rule.** Write `S_ignore = {k : input_ids[k + 1] == pause_id}`, the paper's set of positions whose next token is `<pause>`, re-indexed from 0. In this aligned convention the loss term for position `k` sits at `labels[k + 1]`, so:

    labels[j] = -100            if j < answer_start               (prefix and pauses)
    labels[j] = -100            if input_ids[j] == pause_id      (j - 1 is in S_ignore)
    labels[j] = input_ids[j]    otherwise

**Constraints:**
- Append the pauses after the prefix, never before it.
- Keep `labels[answer_start]`: the first answer token is predicted from the last pause and must be trained.
- A target token equal to `pause_id` gets label -100 too, because the paper never trains the model to emit a pause.
- `answer_start` is always `N + M`. Do not search for the last pause: `pause_id` may also appear in the prefix or the target.
- With `n_pause = 0` the result is the plain supervised-finetuning example.
- Accept any integer dtype for the inputs and always return `torch.long` on the input device. Do not modify the inputs.
- Raise `ValueError` when either tensor is not 1-D or not an integer dtype, when the prefix is empty, or when `n_pause` is not a non-negative `int` (a `bool` is rejected).

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**The idea.** A decoder computes one vector per input token in each layer before it emits the next token. Appending `M` inert `<pause>` tokens gives it `M` more vectors to work with before its first answer token, at the cost of `M` extra positions. The paper ignores the model's outputs until the last pause, at training and at inference time.

**Why the pause predictions are masked.** The paper's pause-pretraining loss sums cross-entropy over every position except those in `S_ignore`. Asking the model to predict the pauses would teach it to produce a token whose only job is to delay it.

**Where the off-by-one hides.** `S_ignore` names the position that predicts, and HF labels are indexed by the token predicted. In the 0-based form above, the loss term of position `k` sits at `labels[k + 1]`. Writing -100 at `labels[k]` for each `k` in `S_ignore` instead masks the token before each pause and leaves the last pause trained, and on a pause in the target it masks the answer token before the pause.

**Finetuning versus pretraining.** In pause-finetuning (Algorithm 2) the loss covers only target tokens, so every pause in the prefix block is already masked. The `S_ignore` rule from pretraining (Equation 3) only adds something when a target token is itself the pause id, which is why this exercise states both rules.""",
    "advisory_prerequisites": ["response_token_mask", "cross_entropy"],
    "hints": [
        {
            "level": 1,
            "kind": "questions",
            "content": "With prefix [5, 6], two pauses and target [7, 8], which index holds each token, and which index is answer_start? In HF-style labels, which index's label is scored against the logits of the last pause? The paper's S_ignore names the position that predicts a pause. Which label index does that loss term live at once labels are aligned with the input? What should happen to a target token that happens to equal pause_id, and why must answer_start not be found by searching for pauses?",
        },
        {
            "level": 2,
            "kind": "analysis",
            "content": "Validate first: both inputs 1-D integer tensors, the prefix non-empty, n_pause a plain non-negative int. Build the pause block with torch.full on the prefix's device, cast all three pieces to long and concatenate them in the order prefix, pauses, target. answer_start is the prefix length plus n_pause. Clone the concatenated ids for the labels, so writing -100 does not corrupt input_ids. Write -100 on every index below answer_start, then on every index whose token equals pause_id. Do not shift the labels: the model does that. Check the empty-target case returns all -100 and answer_start equal to the length.",
        },
    ],
    "model_connections": [
        "Hugging Face causal-LM models take labels aligned with input_ids and shift them inside the loss, so any -100 written here is read one position later than it is stored.",
        "The supervised-finetuning collators in TRL mask the prompt the same way; pause-finetuning is that mask with M extra masked positions between prompt and answer.",
        "Coconut's staged curriculum replaces reasoning steps with latent thoughts and gives those positions label -100 in the same way, so pause tokens are the simplest member of that family.",
        "Pause pretraining in the paper inserts pauses at random positions in the middle of a document, where the next-token-is-pause rule is the whole mask rather than an extra case.",
    ],
    "pro_con_analysis": {
        "pros": [
            "Adds compute before the answer with almost no new parameters: one extra embedding for the pause token.",
            "The data change is tiny: a fixed number of appended tokens and a mask, with no change to the model or the loss function.",
            "The aligned-label convention matches standard training code, so the example drops into an existing finetuning loop.",
        ],
        "cons": [
            "The paper finds gains mainly when the model is also pause-pretrained; pause-finetuning a standard model gives mixed results.",
            "M is fixed per task and should roughly match at inference; the paper finds robustness to a range of inference counts, but zero or very few pauses after pause-training hurts.",
            "Each pause costs a full position of attention and memory with no direct supervision on what the pause computes.",
        ],
    },
    "sources": [
        {
            "kind": "paper",
            "url": "https://arxiv.org/abs/2310.02226",
            "section": "Goyal et al. 2024 (arXiv v3), Section 3.1 Learning and inference with the pause token: 'Pretraining with the pause token' (S_ignore and Equation 3) and 'Finetuning with the pause token'; Appendix Algorithms 1 and 2",
            "equation": "3",
            "note": "The ORACLE for this exercise is the paper itself: no official code was released. The paper's definitions are transcribed into "
                    "torch_judge/harness/reasoning/vendored/pause_tokens_paper.py in the paper's own 1-based, predicting-position indexing, and the "
                    "oracle test converts the reference's aligned labels into those loss terms. The exercise applies Equation 3's S_ignore during "
                    "finetuning too, which differs from Algorithm 2 alone only when a target token equals the pause id.",
        },
    ],
    "tests": [
        {
            "name": "Two pauses between prefix and answer",
            "behavior": "edge.empty_or_boundary",
            "code": r"""
import torch
P = 99
input_ids, labels, answer_start = {fn}(torch.tensor([5, 6, 7]), torch.tensor([8, 9]), P, 2)
assert input_ids.tolist() == [5, 6, 7, P, P, 8, 9], input_ids
# Aligned labels: index 5 holds the first answer token, scored against the last pause's logits.
assert labels.tolist() == [-100, -100, -100, -100, -100, 8, 9], labels
assert answer_start == 5 and isinstance(answer_start, int), answer_start
assert input_ids.dtype == torch.long and labels.dtype == torch.long
""",
        },
        {
            "name": "A pause token inside the target is not trained",
            "behavior": "edge.empty_or_boundary",
            "code": r"""
import torch
P = 0
input_ids, labels, answer_start = {fn}(torch.tensor([4, 4]), torch.tensor([3, P, 2]), P, 1)
assert input_ids.tolist() == [4, 4, P, 3, P, 2], input_ids
assert labels.tolist() == [-100, -100, -100, 3, -100, 2], labels
assert answer_start == 3, answer_start
""",
        },
        {
            "name": "First answer token keeps its label",
            "visibility": "unshown",
            "behavior": "edge.empty_or_boundary",
            "failure_message": "labels[answer_start] was masked. In aligned labels the first answer token sits at answer_start and is predicted from the last pause, so it must keep its id.",
            "code": r"""
import torch
torch.manual_seed(0)
for n_pause in (1, 3, 6):
    prefix = torch.randint(1, 50, (4,))
    target = torch.randint(1, 50, (5,))
    input_ids, labels, s = {fn}(prefix, target, 77, n_pause)
    assert s == 4 + n_pause, (n_pause, s)
    assert int(labels[s]) == int(target[0]), (n_pause, labels)
    assert labels[s:].tolist() == target.tolist(), (n_pause, labels)
""",
        },
        {
            "name": "No pause is ever a training target",
            "visibility": "unshown",
            "behavior": "state.invariant",
            "failure_message": "A label equals pause_id. Every index holding a pause, in the pause block or in the target, must have label -100.",
            "code": r"""
import torch
torch.manual_seed(1)
P = 3
for _ in range(5):
    prefix = torch.randint(0, 6, (int(torch.randint(1, 6, ()).item()),))
    target = torch.randint(0, 6, (int(torch.randint(0, 8, ()).item()),))
    m = int(torch.randint(0, 5, ()).item())
    input_ids, labels, s = {fn}(prefix, target, P, m)
    assert not (labels == P).any(), (prefix, target, m, labels)
    assert (labels[input_ids == P] == -100).all(), labels
""",
        },
        {
            "name": "Pauses are appended after the prefix",
            "visibility": "unshown",
            "behavior": "tensor.shape",
            "failure_message": "input_ids is not prefix, then the pauses, then the target. The pauses go between the prompt and the answer, not before the prompt.",
            "code": r"""
import torch
prefix = torch.tensor([11, 12, 13, 14])
target = torch.tensor([21, 22, 23])
input_ids, labels, s = {fn}(prefix, target, 5, 3)
assert input_ids.shape == (10,), input_ids.shape
assert input_ids[:4].tolist() == [11, 12, 13, 14], input_ids
assert input_ids[4:7].tolist() == [5, 5, 5], input_ids
assert input_ids[7:].tolist() == [21, 22, 23], input_ids
assert s == 7 and labels.tolist() == [-100] * 7 + [21, 22, 23], labels
""",
        },
        {
            "name": "Matches the paper's loss terms on random examples",
            "visibility": "unshown",
            "behavior": "state.invariant",
            "failure_message": "The labels do not match the paper's rule. Train on index j exactly when j >= N + M and input_ids[j] != pause_id; everything else is -100.",
            "code": r"""
import torch
g = torch.Generator().manual_seed(2)
for _ in range(8):
    N = int(torch.randint(1, 7, (), generator=g))
    T = int(torch.randint(0, 9, (), generator=g))
    M = int(torch.randint(0, 5, (), generator=g))
    P = int(torch.randint(0, 4, (), generator=g))
    prefix = torch.randint(0, 4, (N,), generator=g)
    target = torch.randint(0, 4, (T,), generator=g)
    input_ids, labels, s = {fn}(prefix, target, P, M)
    seq = prefix.tolist() + [P] * M + target.tolist()
    want = [(-100 if (j < N + M or seq[j] == P) else seq[j]) for j in range(len(seq))]
    assert input_ids.tolist() == seq, (seq, input_ids)
    assert labels.tolist() == want, (seq, want, labels)
    assert s == N + M
""",
        },
        {
            "name": "Zero pauses is plain supervised finetuning",
            "visibility": "unshown",
            "behavior": "edge.empty_or_boundary",
            "failure_message": "With n_pause = 0 the example must be prefix plus target, the prefix masked and every target label kept (unless it equals pause_id).",
            "code": r"""
import torch
input_ids, labels, s = {fn}(torch.tensor([1, 2, 3]), torch.tensor([4, 5]), 9, 0)
assert input_ids.tolist() == [1, 2, 3, 4, 5], input_ids
assert labels.tolist() == [-100, -100, -100, 4, 5], labels
assert s == 3
""",
        },
        {
            "name": "answer_start ignores pause ids outside the pause block",
            "visibility": "unshown",
            "behavior": "edge.empty_or_boundary",
            "failure_message": "answer_start must be len(prefix) + n_pause. Do not locate it by searching for the pause id, which may also appear in the prefix or the target.",
            "code": r"""
import torch
P = 7
# A pause id in the prefix with no pause block, and one in the target after the pause block.
_, labels, s = {fn}(torch.tensor([1, P, 2]), torch.tensor([3, 4]), P, 0)
assert s == 3, s
assert labels.tolist() == [-100, -100, -100, 3, 4], labels
_, labels, s = {fn}(torch.tensor([1]), torch.tensor([3, P, 4]), P, 2)
assert s == 3, s
assert labels.tolist() == [-100, -100, -100, 3, -100, 4], labels
""",
        },
        {
            "name": "Empty target",
            "visibility": "unshown",
            "behavior": "edge.empty_or_boundary",
            "failure_message": "With an empty target, input_ids is prefix plus pauses, every label is -100 and answer_start equals the length.",
            "code": r"""
import torch
input_ids, labels, s = {fn}(torch.tensor([1, 2]), torch.tensor([], dtype=torch.long), 0, 3)
assert input_ids.tolist() == [1, 2, 0, 0, 0], input_ids
assert labels.tolist() == [-100] * 5, labels
assert s == 5 and isinstance(s, int), s
""",
        },
        {
            "name": "Returns long tensors and leaves inputs untouched",
            "visibility": "unshown",
            "behavior": "tensor.dtype_device",
            "failure_message": "Return torch.long input_ids and labels that do not share memory, and do not modify prefix_ids or target_ids.",
            "code": r"""
import torch
prefix = torch.tensor([3, 4, 5], dtype=torch.int32)
target = torch.tensor([6, 7], dtype=torch.int32)
input_ids, labels, s = {fn}(prefix, target, 1, 2)
assert input_ids.dtype == torch.long and labels.dtype == torch.long, (input_ids.dtype, labels.dtype)
assert input_ids.device == prefix.device and labels.device == prefix.device
assert input_ids.tolist() == [3, 4, 5, 1, 1, 6, 7], input_ids
assert labels.data_ptr() != input_ids.data_ptr()
labels[-1] = 123
assert input_ids[-1].item() == 7, 'labels share storage with input_ids'
assert prefix.tolist() == [3, 4, 5] and target.tolist() == [6, 7]
assert prefix.dtype == torch.int32 and target.dtype == torch.int32
""",
        },
        {
            "name": "Rejects invalid inputs",
            "visibility": "unshown",
            "behavior": "contract.signature",
            "failure_message": "An invalid input was accepted. Raise ValueError for a non 1-D or non-integer tensor, an empty prefix, or an n_pause that is negative, a float or a bool.",
            "code": r"""
import torch
ok_p, ok_t = torch.tensor([1, 2]), torch.tensor([3])
cases = [
    (torch.tensor([[1, 2]]), ok_t, 0, 1),
    (ok_p, torch.tensor([[3]]), 0, 1),
    (torch.tensor([1.0, 2.0]), ok_t, 0, 1),
    (ok_p, torch.tensor([3.0]), 0, 1),
    (torch.tensor([], dtype=torch.long), ok_t, 0, 1),
    (ok_p, ok_t, 0, -1),
    (ok_p, ok_t, 0, 1.0),
    (ok_p, ok_t, 0, True),
]
for i, (p, t, pid, m) in enumerate(cases):
    try:
        {fn}(p, t, pid, m)
    except ValueError:
        continue
    raise AssertionError(f'case {i} should raise ValueError')
""",
        },
    ],
    "solution": '''import torch


def _check_ids(name, ids):
    if not torch.is_tensor(ids) or ids.ndim != 1:
        raise ValueError(f"{name} must be a 1-D tensor")
    if ids.is_floating_point() or ids.is_complex() or ids.dtype == torch.bool:
        raise ValueError(f"{name} must have an integer dtype, got {ids.dtype}")


def insert_pauses(prefix_ids, target_ids, pause_id, n_pause):
    _check_ids("prefix_ids", prefix_ids)
    _check_ids("target_ids", target_ids)
    if prefix_ids.numel() == 0:
        raise ValueError("prefix_ids must not be empty")
    if isinstance(n_pause, bool) or not isinstance(n_pause, int) or n_pause < 0:
        raise ValueError(f"n_pause must be a non-negative int, got {n_pause!r}")

    device = prefix_ids.device
    pauses = torch.full((n_pause,), int(pause_id), dtype=torch.long, device=device)
    input_ids = torch.cat([prefix_ids.long(), pauses, target_ids.long().to(device)])
    answer_start = prefix_ids.numel() + n_pause

    # Labels are aligned with input_ids; the model shifts them. The paper ignores the
    # loss at position k when token k + 1 is a pause, which here is labels[k + 1].
    labels = input_ids.clone()
    labels[:answer_start] = -100
    labels[input_ids == int(pause_id)] = -100
    return input_ids, labels, answer_start
''',
    "interview_questions": interview(
        concept=[
            "What do pause tokens give a decoder-only model, and where in the sequence are they placed during finetuning?",
            "Why does the pause-training loss skip the positions whose next token is a pause?",
        ],
        deep_dive=[
            "The paper defines S_ignore by the position that predicts a pause. How do you translate that into Hugging Face labels that are aligned with input_ids?",
            "Which label carries the first answer token, and which position's hidden state is scored against it?",
            "What edge cases does the mask have to handle: zero pauses, an empty target, and the pause id appearing in the prefix or the target?",
        ],
        tradeoffs=[
            "Pause tokens versus chain-of-thought: what does each add in compute and in supervision, and when would you choose one over the other?",
            "Why does the paper find that pause-finetuning alone helps less than pause-pretraining plus pause-finetuning, and what does that imply for adopting it on an existing model?",
        ],
    ),
}
