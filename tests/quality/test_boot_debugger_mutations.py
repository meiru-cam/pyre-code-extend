"""Mutation gate for the multi-part boot program debugger exercise."""

from __future__ import annotations

import pytest

from torch_judge.tasks import get_task
from torch_judge.tasks._schema import validate_task
from tests.quality.parts_gate import assert_first_fails_in_part, assert_part_mutations_rejected

TASK_ID = "boot_debugger"

_BRUTE = '''        for i, (op, value) in enumerate(program):
            if op in _SWAP:
                fixed = list(program)
                fixed[i] = (_SWAP[op], value)
                try:
                    done, acc, _ = self.run(fixed)
                except ProgramError:
                    continue
                if done:
                    return i, acc
'''

MUTATIONS = [
    ("past the end ends normally", 1, [("            if pc == n:\n                return True, acc, None\n", "            if pc >= n:\n                return True, acc, None\n")]),
    ("skip uses its value", 1, [('    return i + value if op == "goto" else i + 1', '    return i + value if op != "add" else i + 1')]),
    ("loop line off by one", 1, [("                return False, acc, pc\n", "                return False, acc, pc - 1\n")]),
    ("visited as a list", 1, [("        n, pc, acc, seen = len(program), 0, 0, set()", "        n, pc, acc, seen = len(program), 0, 0, []"),
                              ("            seen.add(pc)\n            op, value = program[pc]\n            if op == \"add\":", "            seen.append(pc)\n            op, value = program[pc]\n            if op == \"add\":")]),
    ("brute-force repair", 2, [("        n = len(program)\n        # Every line", _BRUTE + "        n = len(program)\n        # Every line")]),
    ("out-of-range switch accepted", 2, [("                if other in ends:\n", "                if other in ends or not 0 <= other <= n:\n")]),
    ("wrong alternative for goto", 2, [('other = pc + 1 if op == "goto" else pc + value', 'other = pc + value if op == "goto" else pc + 1')]),
    ("comments not stripped", 3, [('tokens = line.split("#", 1)[0].split()', "tokens = line.split()")]),
    ("unsigned values accepted", 3, [('_VALUE = re.compile(r"[+-][0-9]+")', '_VALUE = re.compile(r"[+-]?[0-9]+")')]),
    ("instruction index as line number", 3, [("raise ParseError(number,", "raise ParseError(len(program) + 1,")]),
    ("extra tokens ignored", 3, [("if len(tokens) != 2 or", "if len(tokens) < 2 or")]),
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
