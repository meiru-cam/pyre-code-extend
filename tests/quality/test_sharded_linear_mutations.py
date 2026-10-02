"""Mutation gate for the multi-part sharded linear exercise."""

from __future__ import annotations

import pytest

from torch_judge.tasks import get_task
from torch_judge.tasks._schema import validate_task
from tests.quality.parts_gate import assert_first_fails_in_part, assert_part_mutations_rejected

TASK_ID = "sharded_linear"

MUTATIONS = [
    ("remainder to the last shards", 1, [("        return [np.array(piece) for piece in np.array_split(A, n, axis=axis)]", "        return [np.array(piece) for piece in np.array_split(A[::-1] if axis == 0 else A[:, ::-1], n, axis=axis)][::-1]")]),
    ("gather ignores the axis", 1, [("full = np.concatenate(shards, axis=axis)", "full = np.concatenate(shards, axis=0)")]),
    ("devices share one array", 1, [("return [full.copy() for _ in shards]", "return [full for _ in shards]")]),
    ("reduce shares one array", 1, [("return [total.copy() for _ in shards]", "return [total for _ in shards]")]),
    ("reduce averages", 1, [("total = np.sum(shards, axis=0)", "total = np.mean(shards, axis=0)")]),
    ("column dW transposed", 2, [("dW_shards = [X.T @ dYk for dYk in dY_shards]", "dW_shards = [(dYk.T @ X).T for dYk in dY_shards[::-1]]")]),
    ("column dX from one device", 2, [("return dW_shards, self.all_reduce(partial)", "return dW_shards, [partial[0].copy() for _ in partial]")]),
    ("column forward gathers", 2, [("        return [X @ Wk for Wk in W_shards]  # no", "        self.all_gather(W_shards, 1)\n        return [X @ Wk for Wk in W_shards]  # no")]),
    ("row forward not reduced", 3, [("return self.all_reduce([Xk @ Wk for Xk, Wk in zip(X_shards, W_shards)])", "return [Xk @ Wk for Xk, Wk in zip(X_shards, W_shards)]")]),
    ("row dX reduced", 3, [("dX_shards = [dY @ Wk.T for Wk in W_shards]", "dX_shards = [dY @ Wk.T for Wk in W_shards]\n        self.all_reduce([dY for _ in W_shards])")]),
    ("row dW from the wrong shard", 3, [("dW_shards = [Xk.T @ dY for Xk in X_shards]", "dW_shards = [Xk.T @ dY for Xk in X_shards[:1] * len(X_shards)]")]),
    ("tanh derivative from Z", 4, [("H_shards = [np.tanh(Zk) for Zk in self.column_forward(X, W1_shards)]", "Z_shards = self.column_forward(X, W1_shards)\n        H_shards = [np.tanh(Zk) for Zk in Z_shards]"),
                                   ("dZ_shards = [dHk * (1 - Hk ** 2) for dHk, Hk in zip(dH_shards, H_shards)]", "dZ_shards = [dHk * (1 - Zk ** 2) for dHk, Zk in zip(dH_shards, Z_shards)]")]),
    ("loss gradient not averaged", 4, [("dY = 2 * diff / diff.size", "dY = 2 * diff / diff.shape[0]")]),
    ("no tanh derivative", 4, [("dZ_shards = [dHk * (1 - Hk ** 2) for", "dZ_shards = [dHk * (1 + 0 * Hk) for")]),
    ("loss summed", 4, [("loss = float(np.mean(diff ** 2))", "loss = float(np.sum(diff ** 2))")]),
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
