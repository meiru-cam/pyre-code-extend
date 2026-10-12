"""Mutation gate for the ACT online temporal-ensembling exercise."""

from __future__ import annotations

import pytest

from torch_judge.tasks import get_task
from torch_judge.tasks._schema import validate_task
from tests.quality.mutation_runner import Mutation, assert_mutations_rejected

TASK_ID = "act_temporal_ensemble"

MERGE_LINE = (
    "merged = (self.avg * cum[n - 1].view(1, -1, 1) + actions[:, :-1] * w[n].view(1, -1, 1))"
    " / cum[n].view(1, -1, 1)"
)


def _mutation(name: str, *edits: tuple[str, str]) -> Mutation:
    task = get_task(TASK_ID)
    assert task is not None, TASK_ID
    solution = task["solution"]
    for old, new in edits:
        assert old in solution, f"mutation target drifted: {name}"
        solution = solution.replace(old, new, 1)
    return Mutation(name, solution)


# Offline form with an explicit history, so the weight order can be flipped directly.
NEWEST_GETS_W0 = '''import torch


class TemporalEnsembler:
    def __init__(self, coeff, chunk_size):
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
            raise ValueError("chunk_size must be an integer >= 1")
        self.coeff = float(coeff)
        self.chunk_size = chunk_size
        self.reset()

    def reset(self):
        self.history = []

    def update(self, actions):
        K = self.chunk_size
        if not torch.is_tensor(actions) or actions.ndim != 3 or actions.shape[1] != K:
            raise ValueError("bad shape")
        if not actions.is_floating_point():
            raise ValueError("bad dtype")
        if self.history and (self.history[0].shape[0] != actions.shape[0] or self.history[0].shape[2] != actions.shape[2]):
            raise ValueError("shape changed")
        self.history.append(actions.clone())
        t = len(self.history) - 1
        covering = list(range(max(0, t - K + 1), t + 1))
        num, den = 0.0, 0.0
        for i, s in enumerate(reversed(covering)):
            w = torch.exp(torch.tensor(-self.coeff * i, dtype=actions.dtype))
            num = num + w * self.history[s][:, t - s]
            den = den + w
        return num / den
'''


# ACT's imitate_episodes.py quirk: a prediction with any zero entry is treated as unpopulated.
DROPS_ZERO_ACTIONS = '''import torch


class TemporalEnsembler:
    def __init__(self, coeff, chunk_size):
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
            raise ValueError("chunk_size must be an integer >= 1")
        self.coeff = float(coeff)
        self.chunk_size = chunk_size
        self.reset()

    def reset(self):
        self.history = []

    def update(self, actions):
        K = self.chunk_size
        if not torch.is_tensor(actions) or actions.ndim != 3 or actions.shape[1] != K:
            raise ValueError("bad shape")
        if not actions.is_floating_point():
            raise ValueError("bad dtype")
        if self.history and (self.history[0].shape[0] != actions.shape[0] or self.history[0].shape[2] != actions.shape[2]):
            raise ValueError("shape changed")
        self.history.append(actions.clone())
        t = len(self.history) - 1
        rows = []
        for b in range(actions.shape[0]):
            preds = torch.stack([self.history[s][b, t - s] for s in range(max(0, t - K + 1), t + 1)])
            preds = preds[torch.all(preds != 0, dim=1)]
            w = torch.exp(-self.coeff * torch.arange(len(preds), dtype=actions.dtype))
            w = w / w.sum()
            rows.append((preds * w[:, None]).sum(dim=0))
        return torch.stack(rows)
'''


# Upstream-style in-place arithmetic on a buffer that aliases the caller's tensor.
ALIASES_INPUT = '''import torch


class TemporalEnsembler:
    def __init__(self, coeff, chunk_size):
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
            raise ValueError("chunk_size must be an integer >= 1")
        self.coeff = float(coeff)
        self.chunk_size = chunk_size
        self.reset()

    def reset(self):
        self.avg = None
        self.count = None

    def update(self, actions):
        K = self.chunk_size
        if not torch.is_tensor(actions) or actions.ndim != 3 or actions.shape[1] != K:
            raise ValueError("bad shape")
        if not actions.is_floating_point():
            raise ValueError("bad dtype")
        w = torch.exp(-self.coeff * torch.arange(K, dtype=actions.dtype))
        cum = torch.cumsum(w, dim=0)
        if self.avg is None:
            self.avg = actions
            self.count = torch.ones(K, 1, dtype=torch.long)
        else:
            if self.avg.shape[0] != actions.shape[0] or self.avg.shape[2] != actions.shape[2]:
                raise ValueError("shape changed")
            self.avg *= cum[self.count - 1]
            self.avg += actions[:, :-1] * w[self.count]
            self.avg /= cum[self.count]
            self.count = self.count + 1
            self.avg = torch.cat([self.avg, actions[:, -1:]], dim=1)
            self.count = torch.cat([self.count, torch.ones_like(self.count[-1:])])
        action, self.avg, self.count = self.avg[:, 0], self.avg[:, 1:], self.count[1:]
        return action
'''


# Keeps the consumed step in the buffer, so every merge lands one step too early.
NEVER_POPS = '''import torch


class TemporalEnsembler:
    def __init__(self, coeff, chunk_size):
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
            raise ValueError("chunk_size must be an integer >= 1")
        self.coeff = float(coeff)
        self.chunk_size = chunk_size
        self.reset()

    def reset(self):
        self.avg = None
        self.count = None

    def update(self, actions):
        K = self.chunk_size
        if not torch.is_tensor(actions) or actions.ndim != 3 or actions.shape[1] != K:
            raise ValueError("bad shape")
        if not actions.is_floating_point():
            raise ValueError("bad dtype")
        w = torch.exp(-self.coeff * torch.arange(K, dtype=actions.dtype))
        cum = torch.cumsum(w, dim=0)
        if self.avg is None:
            self.avg = actions.clone()
            self.count = torch.ones(K, dtype=torch.long)
        else:
            if self.avg.shape[0] != actions.shape[0] or self.avg.shape[2] != actions.shape[2]:
                raise ValueError("shape changed")
            n = self.count.clamp(max=K - 1)
            self.avg = (self.avg * cum[n - 1].view(1, -1, 1) + actions * w[n].view(1, -1, 1)) / cum[n].view(1, -1, 1)
            self.count = self.count + 1
        return self.avg[:, 0].clone()
'''


MUTATIONS = [
    Mutation("newest prediction gets w_0", NEWEST_GETS_W0),
    Mutation("drops exact-zero actions like the original ACT loop", DROPS_ZERO_ACTIONS),
    Mutation("aliases and mutates the input", ALIASES_INPUT),
    Mutation("never pops the consumed step", NEVER_POPS),
    _mutation(
        "divides by the count instead of the cumulative weights",
        (MERGE_LINE, "merged = (self.avg * n.view(1, -1, 1) + actions[:, :-1] * w[n].view(1, -1, 1)) / (n + 1).view(1, -1, 1)"),
    ),
    _mutation(
        "new prediction weighted one index too early",
        (MERGE_LINE, "merged = (self.avg * cum[n - 1].view(1, -1, 1) + actions[:, :-1] * w[n - 1].view(1, -1, 1)) / cum[n].view(1, -1, 1)"),
    ),
    _mutation(
        "normalizes by the full chunk weight sum",
        (MERGE_LINE, "merged = (self.avg * cum[n - 1].view(1, -1, 1) + actions[:, :-1] * w[n].view(1, -1, 1)) / cum[-1]"),
    ),
    _mutation(
        "new chunk overwrites the running average",
        (MERGE_LINE, "merged = actions[:, :-1]"),
    ),
    _mutation(
        "reset keeps the history",
        ("        self.avg = None    # (B, K - 1, D) running average for each future step\n"
         "        self.count = None  # (K - 1,) predictions merged into each future step\n",
         "        if not hasattr(self, 'avg'):\n"
         "            self.avg = None\n"
         "            self.count = None\n"),
    ),
    _mutation(
        "ensembles across the batch",
        ("        return avg[:, 0]\n", "        return avg[:, 0].mean(dim=0, keepdim=True).expand_as(avg[:, 0])\n"),
    ),
    _mutation(
        "casts to float32",
        ("        w = torch.exp(", "        actions = actions.float()\n        w = torch.exp("),
    ),
    _mutation(
        "accepts a wrong chunk length",
        ("actions.ndim != 3 or actions.shape[1] != K", "actions.ndim != 3"),
    ),
    _mutation(
        "accepts a batch change without reset",
        ('                raise ValueError("batch size or action dim changed without reset()")', "                pass"),
    ),
]


def test_act_temporal_ensemble_metadata_is_valid():
    task = get_task(TASK_ID)
    assert task is not None
    validate_task(TASK_ID, task)
    assert not any(key.endswith("_zh") for key in task), "new exercises are English only"
    assert any(t.get("visibility", "visible") == "visible" for t in task["tests"])


@pytest.mark.parametrize("_repeat", range(3))
def test_act_temporal_ensemble_reference_and_mutations(_repeat):
    assert set(assert_mutations_rejected(TASK_ID, MUTATIONS, require_unshown=True)) == {m.name for m in MUTATIONS}
