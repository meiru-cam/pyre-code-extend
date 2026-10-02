"""Mutation gate for the multi-part DNS resolver exercise."""

from __future__ import annotations

import pytest

from torch_judge.tasks import get_task
from torch_judge.tasks._schema import validate_task
from tests.quality.parts_gate import assert_first_fails_in_part, assert_part_mutations_rejected

TASK_ID = "dns_resolver"

MUTATIONS = [
    ("case kept", 1, [('return name.lower().rstrip(".") + "."', 'return name.rstrip(".") + "."')]),
    ("trailing dot doubled", 1, [('return name.lower().rstrip(".") + "."', 'return name.lower() + "."')]),
    ("zone list shared", 1, [("                return list(value), ttl\n", "                return value, ttl\n"),
                             ("            return list(addresses)\n", "            return addresses\n")]),
    ("missing names the query", 2, [("                raise NameNotFoundError(name)\n", "                raise NameNotFoundError(start)\n")]),
    ("cycle not detected", 2, [("            if target in seen:\n", "            if False:\n")]),
    ("length checked first", 2, [("            if target in seen:\n                raise ResolutionCycleError(target)  # a cycle wins over the length limit\n            seen.add(target)\n            chain += 1\n            if chain > self._max:\n                raise ChainTooLongError(start)\n",
                                  "            chain += 1\n            if chain > self._max:\n                raise ChainTooLongError(start)\n            if target in seen:\n                raise ResolutionCycleError(target)\n            seen.add(target)\n")]),
    ("chain limit off by one", 2, [("            if chain > self._max:\n", "            if chain >= self._max:\n")]),
    ("fallback on any error", 3, [("        except NameNotFoundError:\n            if self._fallback is None:", "        except DNSError:\n            if self._fallback is None:")]),
    ("largest ttl kept", 4, [("ttl = record_ttl if ttl is None else min(ttl, record_ttl)", "ttl = record_ttl if ttl is None else max(ttl, record_ttl)")]),
    ("expiry inclusive", 4, [("if hit is not None and self._clock < hit[1]:", "if hit is not None and self._clock <= hit[1]:")]),
    ("cache list shared", 1, [("                return list(hit[0])\n", "                return hit[0]\n")]),
    ("errors cached", 4, [("            flight.error = error\n            raise\n",
                           "            flight.error = error\n            with self._lock:\n                self._cache[name] = ([], self._clock + 60)\n            raise\n")]),
    ("no single flight", 5, [("            leader = flight is None\n", "            leader = True\n"),
                             ("                del self._flights[name]\n", "                self._flights.pop(name, None)\n")]),
    ("one global lookup lock", 5, [("import threading\n", "import threading\n_SERIAL = threading.Lock()\n"),
                                   ("            addresses, ttl = self._lookup(name)\n", "            with _SERIAL:\n                addresses, ttl = self._lookup(name)\n")]),
    ("followers swallow errors", 5, [("                raise flight.error\n", "                return []\n")]),
    ("followers share the list", 5, [("            return list(flight.result)\n", "            return flight.result\n")]),
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
