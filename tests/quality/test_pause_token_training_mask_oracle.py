"""pause_token_training_mask's reference solution, checked against the paper.

No official code exists for pause tokens, so the oracle is the paper's own
definitions (S_ignore, Equation 3 and Algorithm 2 of arXiv 2310.02226v3),
transcribed in the paper's 1-based, predicting-position indexing. The reference
returns Hugging Face-style labels aligned with input_ids; ``hf_loss_terms``
applies the shift a causal-LM model applies internally, so both sides are
compared as the same (position, target token) loss terms.
"""

from __future__ import annotations

import pytest
import torch

from torch_judge.harness.reasoning.vendored.pause_tokens_paper import (
    pause_finetune_loss_terms,
    pause_finetune_sequence,
    pause_pretrain_loss_terms,
    s_ignore,
)
from torch_judge.tasks import get_task

TASK_ID = "pause_token_training_mask"


def reference():
    task = get_task(TASK_ID)
    assert task is not None, f"unknown task {TASK_ID!r}"
    namespace: dict = {}
    exec(task["solution"], namespace)
    return namespace[task["function_name"]]


def hf_loss_terms(labels: torch.Tensor) -> list[tuple[int, int]]:
    """Aligned labels to paper terms: 0-based labels[j] is predicted by 1-based position j."""
    return [(j, int(labels[j])) for j in range(1, labels.numel()) if int(labels[j]) != -100]


def random_example(seed: int, vocab: int):
    g = torch.Generator().manual_seed(seed)
    n = int(torch.randint(1, 9, (), generator=g))
    t = int(torch.randint(0, 9, (), generator=g))
    m = int(torch.randint(0, 6, (), generator=g))
    pause = int(torch.randint(0, vocab, (), generator=g))
    prefix = torch.randint(0, vocab, (n,), generator=g)
    target = torch.randint(0, vocab, (t,), generator=g)
    return prefix, target, pause, m


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("vocab", [3, 50])
def test_matches_paper_loss_terms(seed, vocab):
    # vocab=3 makes pause ids in the prefix and the target common; vocab=50 makes them rare.
    prefix, target, pause, m = random_example(seed, vocab)
    input_ids, labels, answer_start = reference()(prefix, target, pause, m)
    p, t = prefix.tolist(), target.tolist()
    assert input_ids.tolist() == pause_finetune_sequence(p, t, pause, m)
    assert answer_start == len(p) + m
    assert hf_loss_terms(labels) == pause_finetune_loss_terms(p, t, pause, m)
    # Index 0 is never scored under the shift; the reference must not rely on it.
    assert int(labels[0]) == -100


@pytest.mark.parametrize(
    "prefix,target,pause,m",
    [
        ([4], [], 0, 0),          # minimal prefix, empty target
        ([4, 5], [], 9, 3),       # empty target after pauses
        ([1, 2, 3], [4, 5], 9, 0),  # n_pause = 0 is plain SFT
        ([7, 1, 7], [2, 3], 7, 0),  # pause id in the prefix, no pause block
        ([1], [3, 7, 4], 7, 2),   # pause id in the target
        ([7], [7, 7], 7, 1),      # every token is the pause id
    ],
)
def test_matches_paper_on_edge_cases(prefix, target, pause, m):
    input_ids, labels, answer_start = reference()(torch.tensor(prefix), torch.tensor(target, dtype=torch.long), pause, m)
    assert input_ids.tolist() == pause_finetune_sequence(prefix, target, pause, m)
    assert answer_start == len(prefix) + m
    assert hf_loss_terms(labels) == pause_finetune_loss_terms(prefix, target, pause, m)


def test_last_pause_predicts_the_first_answer_token():
    """The paper ignores outputs until the last pause; that position's term is t_1."""
    prefix, target, pause, m = [5, 6], [8, 9], 0, 3
    _, labels, s = reference()(torch.tensor(prefix), torch.tensor(target), pause, m)
    last_pause = len(prefix) + m  # 1-based position of the last pause
    assert last_pause not in s_ignore(pause_finetune_sequence(prefix, target, pause, m), pause)
    assert (last_pause, 8) in hf_loss_terms(labels)
    assert int(labels[s]) == 8


def test_pins_divergence_from_algorithm_2_on_a_pause_in_the_target():
    """Algorithm 2 alone trains every target token; the exercise also applies Equation 3's S_ignore."""
    prefix, target, pause, m = [1, 2], [3, 0, 4], 0, 1
    _, labels, _ = reference()(torch.tensor(prefix), torch.tensor(target), pause, m)
    algorithm_2_only = [(len(prefix) + m + j, target[j]) for j in range(len(target))]
    ours = hf_loss_terms(labels)
    assert ours == pause_finetune_loss_terms(prefix, target, pause, m)
    assert ours == [(3, 3), (5, 4)]
    assert algorithm_2_only == [(3, 3), (4, 0), (5, 4)]


def test_restricted_to_the_answer_equals_pretraining_rule():
    """On the answer span, the exercise's terms are exactly Equation 3's pause-pretraining terms."""
    for seed in range(5):
        prefix, target, pause, m = random_example(100 + seed, 3)
        p, t = prefix.tolist(), target.tolist()
        _, labels, s = reference()(prefix, target, pause, m)
        seq = pause_finetune_sequence(p, t, pause, m)
        pretrain = [(k, tok) for k, tok in pause_pretrain_loss_terms(seq, pause) if k >= s]
        assert hf_loss_terms(labels) == pretrain
