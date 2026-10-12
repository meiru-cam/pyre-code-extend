"""coconut_stage_sample's reference solution, checked against Coconut's process_dataset.

Upstream runs with uniform_prob = 0 (the GSM8k value) and no_cot = False, which the
exercise matches exactly. The two upstream extras the exercise leaves out (the random
stage draw and the no_cot ablation) are pinned below as divergences.
"""

from __future__ import annotations

import random

import pytest

from torch_judge.harness.reasoning.vendored.coconut_dataset import (
    GSM_COCONUT_CONFIG,
    make_configs,
    process_sample,
)
from torch_judge.tasks import get_task

TASK_ID = "coconut_stage_sample"
START_ID, LATENT_ID, END_ID = 50257, 50259, 50258  # run.py adds start, end, latent in that order
SPECIAL = {"bot": START_ID, "latent": LATENT_ID, "eot": END_ID}


def reference():
    task = get_task(TASK_ID)
    assert task is not None, f"unknown task {TASK_ID!r}"
    namespace: dict = {}
    exec(task["solution"], namespace)
    return namespace[task["function_name"]]


def random_sample(rng: random.Random) -> dict:
    return {
        "question_tokenized": [rng.randrange(50257) for _ in range(rng.randint(0, 12))],
        "steps_tokenized": [
            [rng.randrange(50257) for _ in range(rng.randint(0, 6))] for _ in range(rng.randint(0, 7))
        ],
        "answer_tokenized": [rng.randrange(50257) for _ in range(rng.randint(1, 4))],
        "idx": rng.randrange(1000),
    }


def compare(sample, stage, configs, no_special_marker=False):
    want = process_sample(sample, stage, configs, START_ID, LATENT_ID, END_ID, no_special_marker=no_special_marker)
    got = reference()(
        sample["question_tokenized"], sample["steps_tokenized"], sample["answer_tokenized"],
        stage, configs.c_thought, configs.max_latent_stage, configs.pad_latent_to_max, SPECIAL,
        no_special_marker=no_special_marker,
    )
    assert got["input_ids"].tolist() == want["input_ids"]
    assert got["labels"].tolist() == want["labels"]
    assert got["position_ids"].tolist() == want["position_ids"]
    assert len(want["attention_mask"]) == len(want["input_ids"])


def test_gsm_config_matches_yaml():
    assert GSM_COCONUT_CONFIG == {
        "c_thought": 2, "max_latent_stage": 3, "pad_latent_to_max": True, "uniform_prob": 0.0, "no_cot": False,
    }


@pytest.mark.parametrize("seed", range(6))
def test_matches_upstream_on_gsm_schedule(seed):
    rng = random.Random(seed)
    configs = make_configs()
    for _ in range(25):
        sample = random_sample(rng)
        for stage in range(configs.max_latent_stage + 3):
            compare(sample, stage, configs)


@pytest.mark.parametrize("seed", range(10, 16))
def test_matches_upstream_on_random_settings(seed):
    rng = random.Random(seed)
    for _ in range(60):
        configs = make_configs(
            c_thought=rng.randint(1, 4),
            max_latent_stage=rng.randint(0, 5),
            pad_latent_to_max=rng.random() < 0.5,
        )
        compare(random_sample(rng), rng.randint(0, 9), configs, no_special_marker=rng.random() < 0.3)


@pytest.mark.parametrize(
    "steps,stage,pad",
    [
        ([], 0, True),
        ([], 4, False),
        ([], 4, True),
        ([[1]], 3, False),          # stage <= max but past the last step: still stage * c latents
        ([[1], [], [2, 3]], 2, True),
        ([[1], [2], [3], [4], [5]], 4, False),
        ([[1], [2]], 9, False),
    ],
)
def test_matches_upstream_on_edges(steps, stage, pad):
    sample = {"question_tokenized": [11, 12], "steps_tokenized": steps, "answer_tokenized": [99], "idx": 0}
    for no_special_marker in (False, True):
        compare(sample, stage, make_configs(pad_latent_to_max=pad), no_special_marker=no_special_marker)


def test_max_latent_stage_zero_matches_upstream():
    sample = {"question_tokenized": [5], "steps_tokenized": [[6], [7]], "answer_tokenized": [8], "idx": 0}
    for stage in (0, 1, 2):
        for pad in (True, False):
            compare(sample, stage, make_configs(max_latent_stage=0, pad_latent_to_max=pad))


def test_divergence_no_cot_is_not_modelled():
    # Upstream's no_cot ablation removes every step and every latent; the exercise has no such flag.
    sample = {"question_tokenized": [5], "steps_tokenized": [[6], [7]], "answer_tokenized": [8], "idx": 0}
    upstream = process_sample(sample, 1, make_configs(no_cot=True), START_ID, LATENT_ID, END_ID)
    assert upstream["input_ids"] == [5, START_ID, END_ID, 8]
    ours = reference()([5], [[6], [7]], [8], 1, 2, 3, True, SPECIAL)
    assert ours["input_ids"].tolist() == [5, START_ID, LATENT_ID, LATENT_ID, END_ID, 7, 8]


def test_divergence_uniform_prob_draws_a_random_stage():
    # With uniform_prob = 1 upstream ignores the scheduled stage; the exercise always uses it.
    sample = {"question_tokenized": [5], "steps_tokenized": [[6], [7], [9]], "answer_tokenized": [8], "idx": 0}
    state = random.getstate()
    try:
        random.seed(0)
        drawn = {len(process_sample(sample, 0, make_configs(uniform_prob=1.0), START_ID, LATENT_ID, END_ID)["input_ids"])
                 for _ in range(50)}
    finally:
        random.setstate(state)
    assert len(drawn) > 1
    ours = reference()([5], [[6], [7], [9]], [8], 0, 2, 3, True, SPECIAL)
    assert ours["input_ids"].tolist() == [5, START_ID, END_ID, 6, 7, 9, 8]
