"""Mutation gate for the multi-part package version search exercise."""

from __future__ import annotations

import pytest

from torch_judge.tasks import get_task
from torch_judge.tasks._schema import validate_task
from tests.quality.parts_gate import assert_first_fails_in_part, assert_part_mutations_rejected

TASK_ID = "version_search"

_MONOTONE = '''        ordered = sorted(versions, key=_key)
        i = _first_true(len(ordered), lambda j: is_supported(ordered[j]))
'''

MUTATIONS = [
    ("string order", 1, [(_MONOTONE, _MONOTONE.replace("sorted(versions, key=_key)", "sorted(versions)"))]),
    ("any True", 1, [("            found, hi = mid, mid - 1\n", "            return mid\n")]),
    ("scans every version", 1, [(_MONOTONE + "        return None if i is None else ordered[i]\n",
                                 "        return self.earliest_supported_any(versions, is_supported)\n")]),
    ("version normalised", 1, [("        return None if i is None else ordered[i]\n",
                                '        return None if i is None else ".".join(map(str, _key(ordered[i])))\n')]),
    ("bisects regressions", 2, [("        for version in sorted(versions, key=_key):  # the first True in order is the earliest\n            if is_supported(version):\n                return version\n        return None\n",
                                 "        return self.earliest_supported(versions, is_supported)\n")]),
    ("bisects the flat list", 3, [("        cache = {}\n", "        return self.earliest_supported(versions, is_supported)\n        cache = {}\n")]),
    ("earliest patch as representative", 3, [("lambda k: probe(groups[minors[k]][-1])", "lambda k: probe(groups[minors[k]][0])")]),
    ("probes repeated", 3, [("            if version not in cache:\n                cache[version] = is_supported(version)\n            return cache[version]\n",
                             "            return is_supported(version)\n")]),
    ("minors scanned one by one", 3, [("        j = _first_true(len(minors), lambda k: probe(groups[minors[k]][-1]))\n",
                                       "        j = next(k for k in range(len(minors)) if probe(groups[minors[k]][-1]))\n")]),
    ("cycles accepted", 4, [("            return order if len(order) == len(chosen) else None\n", "            return order\n")]),
    ("no backtracking", 4, [("                    if found is not None:\n                        return found\n", "                    return found\n")]),
    ("fixed version not rechecked", 4, [("return search(chosen, rest) if _satisfies(chosen[name], constraint) else None", "return search(chosen, rest)")]),
    ("> before >=", 4, [('_OPS = [("==", operator.eq), (">=", operator.ge), ("<=", operator.le), (">", operator.gt), ("<", operator.lt)]',
                         '_OPS = [("==", operator.eq), (">", operator.gt), ("<", operator.lt), (">=", operator.ge), ("<=", operator.le)]')]),
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
