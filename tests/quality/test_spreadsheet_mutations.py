"""Mutation gate for the multi-part spreadsheet exercise."""

from __future__ import annotations

import pytest

from torch_judge.tasks import get_task
from torch_judge.tasks._schema import validate_task
from tests.quality.parts_gate import (
    assert_first_fails_in_part,
    assert_part_mutations_rejected,
    first_failing_part,
)

TASK_ID = "spreadsheet"

_LAZY_GET = '''    def get(self, name):
        self._check_name(name)
        return self._evaluate(name, set())

    def _evaluate(self, name, path):
        if name in path:
            raise CycleError(name)
        if name in self._literal:
            return self._literal[name]
        if name not in self._terms:
            return 0
        path = path | {name}
        return sum(sign * (self._evaluate(t, path) if isinstance(t, str) else t) for sign, t in self._terms[name])
'''

MUTATIONS = [
    ("lowercase names accepted", 1, [('_NAME = re.compile(r"[A-Z]+[0-9]+")', '_NAME = re.compile(r"[A-Za-z]+[0-9]+")')]),
    ("literals longer than 18 digits", 1, [('[0-9]{1,18}', '[0-9]+')]),
    ("leading operator accepted", 1, [('rf"=\\s*{_TERM}', 'rf"=\\s*[+-]?\\s*{_TERM}')]),
    ("minus read as plus", 1, [('sign = 1 if token == "+" else -1', 'sign = 1')]),
    ("no cycle check", 1, [('            raise CycleError(f"{name} would depend on itself")\n', '            pass\n')]),
    ("only the changed cell is recomputed", 1, [("        visit(start)\n        for cell in reversed(order):", "        order = [start]\n        for cell in reversed(order):")]),
    ("get evaluates every time", 2, [(
        "    def get(self, name):\n        self._check_name(name)\n        return self._values.get(name, 0)\n",
        _LAZY_GET,
    )]),
    ("a rejected set rewires first", 2, [
        ('''        if any(self._reaches(cell, name) for cell in reads):
            raise CycleError(f"{name} would depend on itself")
        for cell in self._reads.pop(name, ()):
            self._readers[cell].discard(name)
        for cell in reads:
            self._readers.setdefault(cell, set()).add(name)
''', '''        for cell in self._reads.pop(name, ()):
            self._readers[cell].discard(name)
        for cell in reads:
            self._readers.setdefault(cell, set()).add(name)
        self._reads[name] = reads
        if any(self._reaches(cell, name) for cell in reads if cell != name) or name in reads:
            raise CycleError(f"{name} would depend on itself")
'''),
    ]),
    ("old dependencies kept", 3, [(
        "        for cell in self._reads.pop(name, ()):\n            self._readers[cell].discard(name)\n",
        "",
    )]),
]


def test_mutations_rejected():
    assert_part_mutations_rejected(TASK_ID, MUTATIONS)


@pytest.mark.parametrize(("name", "part", "edits"), MUTATIONS, ids=[m[0] for m in MUTATIONS])
def test_each_mutation_first_fails_in_its_part(name, part, edits):
    assert_first_fails_in_part(TASK_ID, part, edits)


def test_a_lazy_part_1_solution_passes_part_1_only():
    """Part 1 lets get report a stored cycle, which is what keeps the parts cumulative."""
    task = get_task(TASK_ID)
    lazy = task["solution"].replace(
        '''        if any(self._reaches(cell, name) for cell in reads):
            raise CycleError(f"{name} would depend on itself")
''', "").replace(
        "    def get(self, name):\n        self._check_name(name)\n        return self._values.get(name, 0)\n",
        _LAZY_GET,
    )
    assert lazy != task["solution"]
    assert first_failing_part(task, lazy) == 2


def test_task_metadata_is_valid():
    task = get_task(TASK_ID)
    validate_task(TASK_ID, task)
    assert not {"title_zh", "description_zh", "hint_zh"} & set(task)
