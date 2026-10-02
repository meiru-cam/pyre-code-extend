"""Mutation gate for the multi-part three-card poker exercise."""

from __future__ import annotations

import pytest

from torch_judge.tasks import get_task
from torch_judge.tasks._schema import validate_task
from tests.quality.parts_gate import assert_first_fails_in_part, assert_part_mutations_rejected

TASK_ID = "poker_hands"

_TRIPS = "    if ranks[0] == ranks[2]:\n        return (3, ranks[0])\n"
_FLUSH = "    if len({card[1] for card in cards}) == 1:\n        return (2, *ranks)\n"

MUTATIONS = [
    ("trips checked after pairs", 1, [(_TRIPS, ""), ("    return (0, *ranks)\n", _TRIPS + "    return (0, *ranks)\n"),
                                      ("    if ranks[0] == ranks[1]:", "    if ranks[0] == ranks[1] != ranks[2]:")]),
    ("pair compared by top card", 1, [("        return (1, ranks[1], ranks[0])\n", "        return (1, *ranks)\n")]),
    ("pair kicker ignored", 1, [("        return (1, ranks[0], ranks[2])\n", "        return (1, ranks[0])\n")]),
    ("high cards compared lowest first", 1, [("    return (0, *ranks)\n", "    return (0, *sorted(ranks))\n")]),
    ("suit breaks ties", 1, [("        ka, kb = _key(a), _key(b)\n",
                              "        ka, kb = (_key(a), sorted(c[1] for c in a)), (_key(b), sorted(c[1] for c in b))\n")]),
    ("only the first winner", 1, [("        return [i for i, k in enumerate(keys) if k == best]\n", "        return [keys.index(best)]\n")]),
    ("quadratic ranking", 1, [("        best = max(keys)\n        return [i for i, k in enumerate(keys) if k == best]\n",
                               "        return [i for i, k in enumerate(keys) if k == max(keys)]\n")]),
    ("no flush", 2, [(_FLUSH, "")]),
    ("flush tie-break on top card only", 2, [("        return (2, *ranks)\n", "        return (2, ranks[0])\n")]),
    ("flush ranked below pair", 2, [("        return (2, *ranks)\n", "        return (0.5, *ranks)\n"),
                                    ("TYPE_NAMES[_key(cards)[0]]", "TYPE_NAMES[2 if _key(cards)[0] == 0.5 else _key(cards)[0]]")]),
    ("fold consumes a card", 3, [('            self._folded[player] = True\n',
                                  '            self._folded[player] = True\n            self._next_card += 1\n')]),
    ("fold rule sees the next card", 3, [("self._strategies[player](list(self._hands[player]))",
                                          "self._strategies[player](self._hands[player] + self._deck[self._next_card:self._next_card + 1])")]),
    ("fold rule asked every pass", 3, [("if round_no == 2 and", "if round_no >= 1 and")]),
    ("folded players can win", 3, [("        alive = [p for p, folded in enumerate(self._folded) if not folded]\n",
                                    "        alive = [p for p, folded in enumerate(self._folded) if not folded] and list(range(len(self._folded)))\n")]),
    ("tie listed with and", 3, [('", ".join(', '" and ".join(')]),
    ("winner indexed among survivors", 3, [("            return f\"Winner: Player {alive[best[0]]}\"\n",
                                            "            return f\"Winner: Player {best[0]}\"\n")]),
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
