"""Mutation gate for the multi-part nearest neighbour as a layer exercise."""

from __future__ import annotations

import pytest

from torch_judge.tasks import get_task
from torch_judge.tasks._schema import validate_task
from tests.quality.parts_gate import assert_first_fails_in_part, assert_part_mutations_rejected

TASK_ID = "nearest_neighbor"

MUTATIONS = [
    ("ties to the largest index", 1, [("nearest = np.argmin(self._sq_dists(X_train, X_query), axis=1)",
                                       "D = self._sq_dists(X_train, X_query)\n        nearest = (D.shape[1] - 1) - np.argmin(D[:, ::-1], axis=1)")]),
    ("an (m, n, d) array", 1, [("        q2 = (X_query ** 2).sum(axis=1)[:, None]\n",
                                "        return ((X_query[:, None, :] - X_train[None, :, :]) ** 2).sum(axis=2)\n        q2 = (X_query ** 2).sum(axis=1)[:, None]\n")]),
    ("cross term not doubled", 1, [("return q2 - 2.0 * X_query @ X_train.T + x2", "return q2 - X_query @ X_train.T + x2")]),
    ("training norms dropped", 1, [("return q2 - 2.0 * X_query @ X_train.T + x2", "return q2 - 2.0 * X_query @ X_train.T")]),
    ("softmax without the max shift", 2, [("z = np.exp(logits - logits.max(axis=1, keepdims=True))", "z = np.exp(logits)")]),
    ("bias sign flipped", 2, [("return 2.0 * X_train.T, -(X_train ** 2).sum(axis=1)", "return 2.0 * X_train.T, (X_train ** 2).sum(axis=1)")]),
    ("weights not doubled", 2, [("return 2.0 * X_train.T, -(X_train ** 2).sum(axis=1)", "return X_train.T, -(X_train ** 2).sum(axis=1)")]),
    ("probabilities not normalised", 2, [("probs = z / z.sum(axis=1, keepdims=True)", "probs = z")]),
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
