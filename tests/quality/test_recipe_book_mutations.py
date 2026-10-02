"""Mutation gate for the multi-part recipe book exercise."""

from __future__ import annotations

import pytest

from torch_judge.tasks import get_task
from torch_judge.tasks._schema import validate_task
from tests.quality.parts_gate import assert_first_fails_in_part, assert_part_mutations_rejected

TASK_ID = "recipe_book"

MUTATIONS = [
    ("names compared with lower", 1, [("holder = self._names.get(name.casefold())", "holder = self._names.get(name.lower())"),
                                      ("self._names[name.casefold()] = recipe_id\n        return True\n\n    def _change",
                                       "self._names[name.lower()] = recipe_id\n        return True\n\n    def _change"),
                                      ('del self._names[recipe["versions"][-1][0].casefold()]\n        recipe["versions"].append',
                                       'del self._names[recipe["versions"][-1][0].lower()]\n        recipe["versions"].append'),
                                      ("        self._names[name.casefold()] = recipe_id\n        return True\n\n    def _drop",
                                       "        self._names[name.lower()] = recipe_id\n        return True\n\n    def _drop"),
                                      ('        del self._names[recipe["versions"][-1][0].casefold()]\n\n    def add_recipe',
                                       '        del self._names[recipe["versions"][-1][0].lower()]\n\n    def add_recipe')]),
    ("clashes with itself", 1, [("return holder is None or holder == recipe_id", "return holder is None")]),
    ("old name kept on rename", 1, [('        del self._names[recipe["versions"][-1][0].casefold()]\n        recipe["versions"].append', '        recipe["versions"].append')]),
    ("ingredients not copied in", 1, [('recipe["versions"].append((name, list(ingredients)))', 'recipe["versions"].append((name, ingredients))'),
                                      ('"versions": [(name, list(ingredients))]', '"versions": [(name, ingredients)]')]),
    ("ingredients not copied out", 1, [('return {"id": recipe_id, "name": name, "ingredients": list(ingredients)}', 'return {"id": recipe_id, "name": name, "ingredients": ingredients}')]),
    ("ties by id", 2, [('(len(self._recipes[r]["versions"][-1][1]), self._recipes[r]["seq"])', '(len(self._recipes[r]["versions"][-1][1]), r)')]),
    ("ingredient substring match", 2, [("any(i.casefold() == want for i in", "any(want in i.casefold() for i in")]),
    ("prefix case-sensitive", 2, [('r["name"].casefold().startswith(want)', 'r["name"].startswith(prefix)')]),
    ("editors may remove", 3, [('''        if self._owner(recipe_id) is None or self._owner(recipe_id) != user_id:
            return False  # editors may edit but never delete''', '''        if not self._can_edit(user_id, recipe_id):
            return False''')]),
    ("anyone may grant", 3, [("if self._owner(recipe_id) is None or self._owner(recipe_id) != owner_id or user_id not in self._users:",
                             "if self._owner(recipe_id) is None or user_id not in self._users:")]),
    ("unregistered owner accepted", 3, [("return owner_id in self._users and self._create(", "return self._create(")]),
    ("ownerless recipes editable", 3, [('return bool(recipe and recipe["owner"] is not None\n                    and (user_id == recipe["owner"] or user_id in recipe["editors"]))',
                                        'return bool(recipe and (recipe["owner"] is None or user_id == recipe["owner"] or user_id in recipe["editors"]))')]),
    ("rollback truncates", 4, [('return self._change(recipe_id, old["name"], old["ingredients"])',
                                'ok = self._change(recipe_id, old["name"], old["ingredients"])\n        if ok:\n            del self._recipes[recipe_id]["versions"][version:-1]\n        return ok')]),
    ("rollback skips permission", 4, [("if old is None or not self._can_edit(user_id, recipe_id):", "if old is None:")]),
    ("version zero allowed", 4, [("if history is None or not 1 <= version <= len(history):", "if history is None or not 0 <= version <= len(history):")]),
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
