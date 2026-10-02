"""Mutation gate for the multi-part in-memory file system exercise."""

from __future__ import annotations

import pytest

from torch_judge.tasks import get_task
from torch_judge.tasks._schema import validate_task
from tests.quality.parts_gate import assert_first_fails_in_part, assert_part_mutations_rejected

TASK_ID = "memory_fs"

MUTATIONS = [
    ("dot segments allowed", 1, [('    return None if any(p in (".", "..") for p in parts) else parts\n', "    return parts\n")]),
    ("write over a directory", 1, [("if not parts or self._blocked(parts) or isinstance(self._get(parts), dict):",
                                    "if not parts or self._blocked(parts):")]),
    ("mkdir through a file", 1, [("            if isinstance(node.get(p), _File):\n                return False\n", "")]),
    ("ls unsorted", 1, [("return sorted(node) if isinstance(node, dict) else None", "return list(node) if isinstance(node, dict) else None")]),
    ("move into own subtree", 2, [("        if sp[:shorter] == dp[:shorter]:\n", "        if sp == dp:\n")]),
    ("copy shares nodes", 2, [("moved = copy.deepcopy(node) if keep_source else node", "moved = node")]),
    ("kinds may replace each other", 2, [("if target is not None and isinstance(target, _File) != isinstance(node, _File):",
                                          "if False:")]),
    ("move keeps the source", 2, [("            del self._get(sp[:-1])[sp[-1]]\n        for owner", "            pass\n        for owner")]),
    ("size counts files", 2, [("return sum(len(f.content) for _, f in _files(node, \"/\"))", "return sum(1 for _, f in _files(node, \"/\"))")]),
    ("overwrite changes owner", 3, [("        if old is not None and old.owner != owner:\n", "        if False:\n")]),
    ("old bytes still counted", 3, [("self._used[owner] - freed + len(content) > self._capacity[owner]", "self._used[owner] + len(content) > self._capacity[owner]")]),
    ("copy ignores the quota", 3, [("if owner != ADMIN and change > 0 and self._used[owner] + change > self._capacity[owner]:", "if False:")]),
    ("move over another owner", 3, [("        if isinstance(target, _File) and target.owner != node.owner:\n", "        if False:\n")]),
    ("smallest evicted first", 3, [("key=lambda item: (-len(item[1].content), item[0])", "key=lambda item: (len(item[1].content), item[0])")]),
    ("ties by reversed path", 3, [("key=lambda item: (-len(item[1].content), item[0])", "key=lambda item: (-len(item[1].content), item[0][::-1])")]),
    ("rm frees nothing", 3, [("        for _, f in _files(node, \"/\"):\n            self._bill(f.owner, -len(f.content))\n        del", "        del")]),
    ("admin can be added", 3, [("if user_id == ADMIN or user_id in self._capacity or capacity < 0:", "if user_id in self._capacity or capacity < 0:")]),
    ("snapshot shares state", 4, [("self._snapshots[snapshot_id] = copy.deepcopy((self._root, self._capacity, self._used))",
                                   "self._snapshots[snapshot_id] = (self._root, self._capacity, self._used)")]),
    ("restore shares the snapshot", 4, [("= copy.deepcopy(self._snapshots[snapshot_id])", "= self._snapshots[snapshot_id]")]),
    ("snapshot id reused", 4, [("        if snapshot_id in self._snapshots:\n            return False\n        self._snapshots", "        self._snapshots")]),
    ("restore keeps capacities", 4, [("self._root, self._capacity, self._used = copy.deepcopy(", "self._root, _, self._used = copy.deepcopy(")]),
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
