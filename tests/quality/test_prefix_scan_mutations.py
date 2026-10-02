"""Mutation gate for the multi-part prefix products and parallel scan exercise."""

from __future__ import annotations

import pytest

from torch_judge.tasks import get_task
from torch_judge.tasks._schema import validate_task
from tests.quality.parts_gate import assert_first_fails_in_part, assert_part_mutations_rejected

TASK_ID = "prefix_scan"

MUTATIONS = [
    ("written in place", 1, [("        out = [W[0]]\n        for i in range(1, W.shape[0]):\n            out.append(out[-1] @ W[i])  # new tensors only, so autograd keeps every input it saved\n        return torch.stack(out)\n",
                              "        P = torch.empty_like(W)\n        P[0] = W[0]\n        for i in range(1, W.shape[0]):\n            P[i] = P[i - 1] @ W[i]\n        return P\n")]),
    ("multiplied right to left", 1, [("out.append(out[-1] @ W[i])", "out.append(W[i] @ out[-1])")]),
    ("loss gradient dropped", 2, [("            carry = carry + G[i]\n", "            carry = G[i] if i == W.shape[0] - 1 else carry\n")]),
    ("transpose missing", 2, [("dW.append(carry if i == 0 else P[i - 1].transpose(-1, -2) @ carry)", "dW.append(carry if i == 0 else P[i - 1] @ carry)")]),
    ("autograd inside backward", 2, [("        dW = []\n        carry = torch.zeros_like(W[0])",
                                      "        with torch.enable_grad():\n            Wg = W.detach().clone().requires_grad_(True)\n            return torch.autograd.grad((self.prefix_products(Wg) * G).sum(), Wg)[0]\n        dW = []\n        carry = torch.zeros_like(W[0])")]),
    ("scan is the loop", 3, [("        return self._rounds(W)[0]\n", "        return self.prefix_products(W)\n")]),
    ("operands swapped", 3, [("X = torch.cat([X[:stride], X[:-stride] @ X[stride:]])", "X = torch.cat([X[:stride], X[stride:] @ X[:-stride]])")]),
    ("one more round than needed", 3, [("        while stride < W.shape[0]:\n", "        while stride <= W.shape[0]:\n")]),
    ("backward is the loop", 4, [("        H = G\n        for stride, X in reversed(self._rounds(W)[1]):", "        return self.backward(W, self.prefix_products(W), G)\n        H = G\n        for stride, X in reversed(self._rounds(W)[1]):")]),
    ("left-factor term dropped", 4, [("dX = torch.cat([H[:stride], right]) + torch.cat([left, torch.zeros_like(H[:stride])])", "dX = torch.cat([H[:stride], right])")]),
    ("left-factor term shifted", 4, [("torch.cat([left, torch.zeros_like(H[:stride])])", "torch.cat([torch.zeros_like(H[:stride]), left])")]),
]


def test_mutations_rejected():
    assert_part_mutations_rejected(TASK_ID, MUTATIONS)


@pytest.mark.parametrize(("name", "part", "edits"), MUTATIONS, ids=[m[0] for m in MUTATIONS])
def test_each_mutation_first_fails_in_its_part(name, part, edits):
    assert_first_fails_in_part(TASK_ID, part, edits)


def test_task_metadata_is_valid():
    task = get_task(TASK_ID)
    validate_task(TASK_ID, task)
    assert not {"title_zh", "description_zh", "hint_zh"} & set(task)
