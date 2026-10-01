"""Mutation gate for the multi-part text editor buffer exercise."""

from __future__ import annotations

import pytest

from torch_judge.tasks import get_task
from torch_judge.tasks._schema import validate_task
from tests.quality.parts_gate import assert_first_fails_in_part, assert_part_mutations_rejected

TASK_ID = "text_editor"

_CHECK = "if not 0 <= start <= end <= len(self._chars):"
_INSERT = '''        self._check(pos, pos)
        self._insert(pos, text)
'''
_REPLAY = '''        if (kind == "insert") == forward:
            self._insert(pos, text)
        else:
            self._delete(pos, pos + len(text))
'''

MUTATIONS = [
    ("insert at the end rejected", 1, [(_CHECK, "if not 0 <= start <= end < len(self._chars) and not (start == end == 0):")]),
    ("negative positions accepted", 1, [(_CHECK, "if not start <= end <= len(self._chars):")]),
    ("reversed range accepted", 1, [(_CHECK, "if not (0 <= start and end <= len(self._chars)):")]),
    ("delete returns a list", 1, [('        removed = "".join(self._chars[start:end])\n', "        removed = self._chars[start:end]\n")]),
    ("insert checks too late", 1, [(_INSERT, "        self._insert(pos, text)\n        self._check(pos, pos)\n")]),
    ("empty insert recorded", 2, [("        if text:\n            self._undo.append", "        if True:\n            self._undo.append")]),
    ("redo kept after a new edit", 2, [("            self._undo.append((\"insert\", pos, text))\n            self._redo.clear()\n",
                                        "            self._undo.append((\"insert\", pos, text))\n")]),
    ("undone delete goes back to 0", 2, [('self._undo.append(("delete", start, removed))', 'self._undo.append(("delete", 0, removed))')]),
    ("undo recorded as an edit", 2, [(_REPLAY, _REPLAY.replace("self._insert(pos, text)", "self.insert(pos, text)").replace("self._delete(pos, pos + len(text))", "self.delete(pos, pos + len(text))"))]),
    ("undo on empty returns None", 2, [("        if not self._undo:\n            return False\n", "        if not self._undo:\n            return None\n")]),
    ("case-insensitive words", 3, [("Counter(_WORD.findall(self.text()))", "Counter(_WORD.findall(self.text().lower()))")]),
    ("digits and underscores in words", 3, [('_WORD = re.compile(r"[A-Za-z]+")', '_WORD = re.compile(r"[A-Za-z0-9_]+")')]),
    ("ties ignore case", 3, [("        found.sort()  # most frequent first, then string order\n", "        found.sort(key=lambda f: (f[0], f[1].lower()))\n")]),
    ("negative k allowed", 3, [("        if k < 0:\n            raise ValueError", "        if False:\n            raise ValueError")]),
    ("frequency ignored", 3, [("found.append((-node.count, word))", "found.append((0, word))")]),
    ("repeated insert applied twice", 4, [("if (site, seq) not in self._inserted:", "if True:")]),
    ("delete before its insert lost", 4, [("            self._deleted.add((op[1], op[2]))\n", "            if (op[1], op[2]) in self._inserted:\n                self._deleted.add((op[1], op[2]))\n")]),
    ("multi-character insert accepted", 4, [("        if not isinstance(ch, str) or len(ch) != 1:\n            raise ValueError", "        if False:\n            raise ValueError")]),
    ("midpoint positions", 4, [("Fraction(random.randint(1, 2 ** 32 - 1), 2 ** 32)", "Fraction(1, 2)")]),
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
