"""Mutation gate for the DreamerV3 RSSM KL balancing exercise."""

from __future__ import annotations

import pytest

from torch_judge.tasks import get_task
from torch_judge.tasks._schema import validate_task
from tests.quality.mutation_runner import Mutation, assert_mutations_rejected

TASK_ID = "rssm_kl_balance"

DYN_LINE = "    dyn = _kl(log_post.detach(), log_prior)   # trains the prior toward the posterior\n"
REP_LINE = "    rep = _kl(log_post, log_prior.detach())   # trains the posterior toward the prior\n"
CLAMP_LINES = (
    "    dyn = torch.clamp(dyn, min=free_nats)\n"
    "    rep = torch.clamp(rep, min=free_nats)\n"
)
MIX_LINES = (
    "    probs = torch.softmax(logits, dim=-1)\n"
    "    probs = (1 - unimix) * probs + unimix / logits.shape[-1]\n"
    "    return torch.log(probs)\n"
)
GROUP_SUM = "    return per_group.sum(dim=-1)\n"


def _mutation(name: str, *edits: tuple[str, str]) -> Mutation:
    task = get_task(TASK_ID)
    assert task is not None, TASK_ID
    solution = task["solution"]
    for old, new in edits:
        assert old in solution, f"mutation target drifted: {name}"
        solution = solution.replace(old, new, 1)
    return Mutation(name, solution)


MUTATIONS = [
    _mutation(
        "free bits applied per group before summing",
        (GROUP_SUM, "    return per_group.clamp(min=FREE[0]).sum(dim=-1)\n"),
        ("def _kl(log_p, log_q):", "FREE = [0.0]\n\n\ndef _kl(log_p, log_q):"),
        ("    log_post = _log_probs(", "    FREE[0] = free_nats\n    log_post = _log_probs("),
        (CLAMP_LINES, ""),
    ),
    _mutation(
        "stop-gradient on the wrong side",
        (DYN_LINE, "    dyn = _kl(log_post, log_prior.detach())\n"),
        (REP_LINE, "    rep = _kl(log_post.detach(), log_prior)\n"),
    ),
    _mutation(
        "no stop-gradient at all",
        (DYN_LINE, "    dyn = _kl(log_post, log_prior)\n"),
        (REP_LINE, "    rep = _kl(log_post, log_prior)\n"),
    ),
    _mutation(
        "unimix applied to the logits instead of the probabilities",
        (MIX_LINES, "    return torch.log_softmax((1 - unimix) * logits + unimix / logits.shape[-1], dim=-1)\n"),
    ),
    _mutation(
        "uniform share not divided by the class count",
        (MIX_LINES, MIX_LINES.replace("unimix / logits.shape[-1]", "unimix")),
    ),
    _mutation(
        "unimix ignored",
        ("    if unimix == 0:\n", "    if True:\n"),
    ),
    _mutation(
        "log of softmax without unimix underflows",
        ("    if unimix == 0:\n        return torch.log_softmax(logits, dim=-1)\n", ""),
    ),
    _mutation(
        "KL direction reversed",
        ("    per_group = (log_p.exp() * (log_p - log_q)).sum(dim=-1)\n",
         "    per_group = (log_q.exp() * (log_q - log_p)).sum(dim=-1)\n"),
    ),
    _mutation(
        "free bits pass the gradient through",
        (CLAMP_LINES,
         "    dyn = dyn + (free_nats - dyn).clamp(min=0).detach()\n"
         "    rep = rep + (free_nats - rep).clamp(min=0).detach()\n"),
    ),
    _mutation(
        "free bits on the batch mean",
        (CLAMP_LINES, ""),
        ("    loss = (beta_dyn * dyn + beta_rep * rep).mean()\n",
         "    loss = beta_dyn * dyn.mean().clamp(min=free_nats) + beta_rep * rep.mean().clamp(min=free_nats)\n"),
    ),
    _mutation(
        "betas swapped",
        ("    loss = (beta_dyn * dyn + beta_rep * rep).mean()\n", "    loss = (beta_rep * dyn + beta_dyn * rep).mean()\n"),
    ),
    _mutation(
        "loss summed instead of averaged",
        ("    loss = (beta_dyn * dyn + beta_rep * rep).mean()\n", "    loss = (beta_dyn * dyn + beta_rep * rep).sum()\n"),
    ),
    _mutation(
        "groups averaged instead of summed",
        (GROUP_SUM, "    return per_group.mean(dim=-1)\n"),
    ),
    _mutation(
        "computes in float32",
        ("    log_post = _log_probs(", "    post_logits, prior_logits = post_logits.float(), prior_logits.float()\n    log_post = _log_probs("),
    ),
    _mutation(
        "accepts mismatched shapes",
        ("    if post_logits.shape != prior_logits.shape:\n", "    if False:\n"),
    ),
]


def test_rssm_kl_balance_metadata_is_valid():
    task = get_task(TASK_ID)
    assert task is not None
    validate_task(TASK_ID, task)
    assert not any(key.endswith("_zh") for key in task), "new exercises are English only"
    assert any(t.get("visibility", "visible") == "visible" for t in task["tests"])


@pytest.mark.parametrize("_repeat", range(3))
def test_rssm_kl_balance_reference_and_mutations(_repeat):
    assert set(assert_mutations_rejected(TASK_ID, MUTATIONS, require_unshown=True)) == {m.name for m in MUTATIONS}
