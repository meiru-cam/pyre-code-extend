"""Mutation gate for the multi-part record store exercise."""

from __future__ import annotations

import pytest

from torch_judge.tasks import get_task
from torch_judge.tasks._schema import validate_task
from tests.quality.parts_gate import assert_first_fails_in_part, assert_part_mutations_rejected

TASK_ID = "record_store"

MUTATIONS = [
    ("delete always true", 1, [('            del fields[field]\n            return "true"\n        return "false"\n',
                                '            del fields[field]\n        return "true"\n')]),
    ("missing get is None", 1, [('        return entry[0] if self._alive(entry, timestamp) else ""\n',
                                 '        return entry[0] if self._alive(entry, timestamp) else None\n')]),
    ("set keeps the old value", 1, [("self._records.setdefault(key, {})[field] = (value, None)",
                                     "self._records.setdefault(key, {}).setdefault(field, (value, None))")]),
    ("scan in insertion order", 2, [("        return self._format(live, sorted(live))\n", "        return self._format(live, list(live))\n")]),
    ("prefix matched anywhere", 2, [("if f.startswith(prefix)", "if prefix in f")]),
    ("joined without a space", 2, [('return ", ".join(', 'return ",".join(')]),
    ("expiry inclusive", 3, [("now < entry[1])", "now <= entry[1])")]),
    ("plain calls move the clock back", 3, [("        self._clock = max(self._clock, timestamp)", "        self._clock = timestamp")]),
    ("set_at keeps the ttl", 3, [("        self._records.setdefault(key, {})[field] = (value, None)  # a write without a ttl clears any expiry\n",
                                  "        old = self._records.get(key, {}).get(field)\n        self._records.setdefault(key, {})[field] = (value, old[1] if self._alive(old, timestamp) else None)\n")]),
    ("ttl extends the old window", 3, [("(value, timestamp + ttl)  # a new window, never extended",
                                        "(value, max(timestamp, (self._records.get(key, {}).get(field) or (0, 0))[1] or 0) + ttl)")]),
    ("expired field deleted", 3, [("        if self._alive(fields.get(field), timestamp):\n", "        if field in fields:\n")]),
    ("backup keeps absolute expiry", 4, [("(v, None if end is None else end - timestamp)", "(v, end)"),
                                         ("(v, None if left is None else timestamp + left)", "(v, left)")]),
    ("earliest backup chosen", 4, [("chosen = max(t for t in self._backups if t <= timestamp_to_restore)",
                                    "chosen = min(self._backups)")]),
    ("restore shares the backup", 4, [("        self._records = {key: {f: (v, None if left is None else timestamp + left) for f, (v, left) in fields.items()}\n                         for key, fields in self._backups[chosen].items()}  # new dicts, so the backup stays intact\n",
                                       "        self._records = self._backups[chosen]\n        for fields in self._records.values():\n            for f, (v, left) in fields.items():\n                fields[f] = (v, None if left is None else timestamp + left)\n")]),
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
