"""Mutation gate for the multi-part cd destination exercise."""

from __future__ import annotations

import pytest

from torch_judge.tasks import get_task
from torch_judge.tasks._schema import validate_task
from tests.quality.parts_gate import assert_first_fails_in_part, assert_part_mutations_rejected

TASK_ID = "change_directory"

MUTATIONS = [
    ("never back to the root", 1, [("            if stack:  # at the root, .. stays at the root\n", "            if len(stack) > 1:\n")]),
    ("dot kept as a name", 1, [('        if piece in ("", "."):\n', '        if piece == "":\n')]),
    ("empty pieces kept", 1, [('        if piece in ("", "."):\n', '        if piece == ".":\n')]),
    ("root printed empty", 1, [('    return "/" + "/".join(stack)\n', '    return "/".join([""] + stack)\n')]),
    ("absolute treated as relative", 2, [('    if destination.startswith("/"):\n        return "/", destination\n', "")]),
    ("~name accepted", 2, [('        raise ValueError(f"~user is not supported: {destination!r}")\n', "        return cwd, destination\n")]),
    ("only a bare ~", 2, [('if destination == "~" or destination.startswith("~/"):', 'if destination == "~":')]),
    ("relative target from cwd", 3, [("        if target.startswith(\"/\"):\n            stack = []\n", "        stack = [] if target.startswith(\"/\") else _components(cwd)\n")]),
    ("absolute target appended", 3, [("        if target.startswith(\"/\"):\n            stack = []\n", "")]),
    ("cap off by one", 3, [("if expansions > MAX_EXPANSIONS:", "if expansions >= MAX_EXPANSIONS:")]),
    ("target after the rest", 3, [('pending.extendleft(reversed(target.split("/")))', 'pending.extend(target.split("/"))')]),
    ("repeat treated as a loop", 3, [("    expansions = 0\n", "    expansions = 0\n    seen = set()\n"),
                                     ("        expansions += 1\n", "        expansions += 1\n        if (tuple(stack), piece) in seen:\n            raise LinkLoopError(\"repeat\")\n        seen.add((tuple(stack), piece))\n")]),
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
