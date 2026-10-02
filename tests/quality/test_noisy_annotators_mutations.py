"""Mutation gate for the multi-part noisy annotator cleanup exercise."""

from __future__ import annotations

import pytest

from torch_judge.tasks import get_task
from torch_judge.tasks._schema import validate_task
from tests.quality.parts_gate import assert_first_fails_in_part, assert_part_mutations_rejected

TASK_ID = "noisy_annotators"

MUTATIONS = [
    ("ties to the larger class", 1, [("return np.argmax(scores, axis=1), scores.sum(axis=1) > 0",
                                      "return (n_classes - 1) - np.argmax(scores[:, ::-1], axis=1), scores.sum(axis=1) > 0")]),
    ("weights ignored", 1, [("scores[voted, annotations[voted, j]] += weights[j]", "scores[voted, annotations[voted, j]] += 1")]),
    ("has_label from votes, not weight", 1, [("scores.sum(axis=1) > 0  #", "(annotations != -1).any(axis=1)  #")]),
    ("own vote included", 1, [("self.aggregate(np.delete(annotations, j, axis=1), n_classes)", "self.aggregate(annotations, n_classes)")]),
    ("unlabeled consensus counted", 1, [("counted = (annotations[:, j] != -1) & has_others", "counted = annotations[:, j] != -1")]),
    ("sample standard deviation", 1, [("return r < r.mean() - r.std()", "return r < r.mean() - r.std(ddof=1)")]),
    ("flags not applied", 2, [("keep = ~self.flag(self.reliability(annotations, n_classes))", "keep = np.ones(annotations.shape[1], dtype=bool)")]),
    ("unlabeled rows in centroids", 2, [("members = X_train[has_label & (labels == c)]", "members = X_train[labels == c]")]),
    ("empty class predicted", 2, [("dist = np.full((len(X_test), n_classes), np.inf)", "dist = np.zeros((len(X_test), n_classes))")]),
    ("no clipping", 3, [("r = np.clip(np.asarray(r, dtype=float), 0.01, 0.99)", "r = np.asarray(r, dtype=float)")]),
    ("negative weights kept", 3, [("return np.clip(np.log(r * (n_classes - 1) / (1 - r)), 0.0, None)", "return np.log(r * (n_classes - 1) / (1 - r))")]),
    ("class count ignored", 3, [("np.log(r * (n_classes - 1) / (1 - r))", "np.log(r / (1 - r))")]),
    ("weighted is filtered", 3, [("        w = self.reliability_weights(self.reliability(annotations, n_classes), n_classes)\n        return self.aggregate(annotations, n_classes, weights=w)\n",
                                  "        return self.filtered_labels(annotations, n_classes)\n")]),
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
