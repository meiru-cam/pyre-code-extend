"""Mutation gate for the multi-part cloud storage exercise."""

from __future__ import annotations

import pytest

from torch_judge.tasks import get_task
from torch_judge.tasks._schema import validate_task
from tests.quality.parts_gate import assert_first_fails_in_part, assert_part_mutations_rejected

TASK_ID = "cloud_storage"

MUTATIONS = [
    ("add overwrites", 1, [("        if name in self._files:\n            return False\n        self._create(name, size, None)",
                            "        self._create(name, size, None)")]),
    ("copy overwrites", 1, [("if source not in self._files or destination in self._files:",
                             "if source not in self._files:")]),
    ("names fold case", 1, [("        entry = self._files.get(name)\n", "        entry = self._files.get(name) or next((v for k, v in self._files.items() if k.lower() == name.lower()), None)\n")]),
    ("sorted smallest first", 2, [("key=lambda hit: (-hit[0], hit[1])", "key=lambda hit: (hit[0], hit[1])")]),
    ("ties by reversed name", 2, [("key=lambda hit: (-hit[0], hit[1])", "key=lambda hit: (-hit[0], hit[1][::-1])")]),
    ("suffix matched anywhere", 2, [("name.endswith(suffix)", "suffix in name")]),
    ("copy ignores the quota", 3, [("        if owner is not None and size > self._left(owner):\n            return False  # the copy counts against the source's owner\n", "")]),
    ("copy has no owner", 3, [("        self._create(destination, size, owner)", "        self._create(destination, size, None)")]),
    ("exact fit rejected", 3, [("name in self._files or size > self._left(user_id):", "name in self._files or size >= self._left(user_id):")]),
    ("merge keeps source capacity", 3, [("self._capacity[target] += self._capacity.pop(source)", "self._capacity.pop(source)")]),
    ("merge with self allowed", 3, [("if target == source or target not in", "if target not in")]),
    ("compress rounds down", 4, [("(size + 1) // 2", "size // 2")]),
    ("decompress ignores the quota", 4, [("if plain in self._files or size > self._left(user_id):", "if plain in self._files:")]),
    ("merge keeps the source backup", 4, [("        self._backups.pop(source, None)  # the id no longer names this account\n", "")]),
    ("restore overwrites other owners", 4, [("            return None  # a file someone else holds blocks the restore\n", "            pass\n")]),
    ("restore keeps new files", 4, [("        for name in drop:\n            self._remove(name)\n", "")]),
    ("restore keeps the capacity", 4, [("        self._capacity[user_id] = capacity\n        return self._left(user_id)", "        return self._left(user_id)")]),
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
