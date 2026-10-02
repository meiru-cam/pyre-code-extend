"""Mutation gate for the multi-part generic type inference exercise."""

from __future__ import annotations

import pytest

from torch_judge.tasks import get_task
from torch_judge.tasks._schema import validate_task
from tests.quality.parts_gate import assert_first_fails_in_part, assert_part_mutations_rejected

TASK_ID = "type_inference"

MUTATIONS = [
    ("space after comma", 1, [('return "[" + ",".join(self.format_type(c) for c in t) + "]"', 'return "[" + ", ".join(self.format_type(c) for c in t) + "]"')]),
    ("arrow without spaces", 1, [('+ ") -> " + self.format_type(ret)', '+ ")->" + self.format_type(ret)')]),
    ("leftover text accepted", 1, [('        if end != len(text):\n            raise ValueError(f"unexpected text at {end}: {text!r}")\n', "")]),
    ("empty tuple rejected", 1, [('            if text[i:i + 1] == "]":\n                return items, i + 1\n            while True:', "            while True:")]),
    ("any name accepted", 1, [('_NAME = re.compile(r"int|float|str|bool|char|T[0-9]+")', '_NAME = re.compile(r"[A-Za-z][A-Za-z0-9]*")')]),
    ("arity unchecked", 2, [('        if len(args) != len(params):\n            raise ArityError(f"expected {len(params)} arguments, got {len(args)}")\n', "")]),
    ("conflicts ignored", 2, [("if expected in bindings and bindings[expected] != actual:", "if False:")]),
    ("tuple lengths not compared", 2, [("isinstance(actual, list) and len(expected) == len(actual):", "isinstance(actual, list):")]),
    ("generics bind names only", 2, [("if isinstance(expected, str) and expected not in PRIMITIVES:", "if isinstance(expected, str) and expected not in PRIMITIVES and isinstance(actual, str):")]),
    ("mismatch reported as conflict", 2, [('            raise TypeMismatchError(f"expected', '            raise GenericConflictError(f"expected')]),
    ("result aliases the arguments", 2, [("        return self._substitute(bindings[t], {})\n", "        return bindings[t]\n")]),
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
