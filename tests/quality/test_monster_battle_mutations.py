"""Mutation gate for the multi-part monster battle exercise."""

from __future__ import annotations

import pytest

from torch_judge.tasks import get_task
from torch_judge.tasks._schema import validate_task
from tests.quality.parts_gate import assert_first_fails_in_part, assert_part_mutations_rejected

TASK_ID = "monster_battle"

MUTATIONS = [
    ("team B attacks first", 1, [("    attacking, defending = Team(*team_a), Team(*team_b)\n    log = [f\"Match start: {attacking.name} takes on {defending.name}\"]",
                                  "    attacking, defending = Team(*team_b), Team(*team_a)\n    log = [f\"Match start: {defending.name} takes on {attacking.name}\"]")]),
    ("alive at 0 HP", 1, [("return self.hp > 0", "return self.hp >= 0")]),
    ("last alive attacks", 1, [("            return alive[0]\n", "            return alive[-1]\n")]),
    ("winner is the loser", 1, [('log.append(f"Match over: {attacking.name} wins!")', 'log.append(f"Match over: {defending.name} wins!")')]),
    ("advantage reversed", 2, [("        if BEATS[self.element] == defender.element:\n            return 2\n        if BEATS[defender.element] == self.element:\n            return 0.5\n",
                                "        if BEATS[self.element] == defender.element:\n            return 0.5\n        if BEATS[defender.element] == self.element:\n            return 2\n")]),
    ("damage rounded up", 2, [("max(1, int(self.attack * (self.multiplier(defender) or 1)))",
                               "max(1, int(-(-self.attack * (self.multiplier(defender) or 1) // 1)))")]),
    ("no minimum damage", 2, [("max(1, int(self.attack", "max(0, int(self.attack")]),
    ("label printed as a float", 2, [('f" ({multiplier:g}x)"', 'f" ({float(multiplier)}x)"')]),
    ("unknown element accepted", 2, [("if element is not None and element not in ELEMENTS:", "if False:")]),
    ("tie goes to the last listed", 3, [("return max(alive, key=lambda m: m.damage_to(defender))", "return max(reversed(alive), key=lambda m: m.damage_to(defender))")]),
    ("smart picks the highest attack stat", 3, [("return max(alive, key=lambda m: m.damage_to(defender))", "return max(alive, key=lambda m: m.attack)")]),
    ("smart ignored", 3, [("        if not smart:\n", "        if True:\n")]),
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
