Worth confirming up front, since the statement leaves them open: whether a granted editor may also delete a recipe (here, no — only the owner may delete or grant further editors, which is why `grant_editor` and `delete_recipe` both check ownership specifically, not just edit permission); and whether `rollback_recipe` needs the same authorization as `edit_recipe` or something stricter (here, the same, since a rollback only changes a recipe's current name and ingredients, exactly like a normal edit).

### Level 1

`_Recipe` carries every field any later level will need — `owner_id`, `editors` and `versions` cost nothing to keep from the start, so Level 3 and Level 4 only have to start reading and appending to them, never to add them. `_name_index` maps a lookup key to the one `recipe_id` currently holding it, turning every collision check into a single dict lookup instead of a scan of every recipe; `_name_taken` is shared by every method that changes a name, from `add_recipe` here through `rollback_recipe` in Level 4, with `exclude_id` implementing the "not by a different recipe" half of the rule in one place. `_rename` keeps `_name_index` and a recipe's own `.name` in sync and is reused the same way.

```python
def _key(name: str) -> str:
    return name.casefold()


class _Recipe:
    def __init__(self, recipe_id: str, name: str, ingredients: list[str]):
        self.id = recipe_id
        self.name = name
        self.ingredients = list(ingredients)             # NOTE: a fresh copy -- never alias the caller's list
        self.owner_id: str | None = None                  # set by Level 3's add_recipe
        self.editors: set[str] = set()                     # grown by Level 3's grant_editor (membership only)
        self.versions: list[tuple[str, list[str]]] = [(name, list(ingredients))]   # read from Level 4 on

    def public(self) -> dict:
        return {"id": self.id, "name": self.name, "ingredients": list(self.ingredients)}   # NOTE: copy out too


class RecipeManager:
    def __init__(self) -> None:
        self._recipes: dict[str, _Recipe] = {}      # recipe_id -> _Recipe, in creation order
        self._name_index: dict[str, str] = {}         # lookup key -> the recipe_id currently holding it

    def _name_taken(self, name: str, exclude_id: str | None = None) -> bool:
        holder = self._name_index.get(_key(name))
        return holder is not None and holder != exclude_id

    def _rename(self, recipe: _Recipe, name: str) -> None:
        del self._name_index[_key(recipe.name)]
        recipe.name = name
        self._name_index[_key(name)] = recipe.id

    def add_recipe(self, recipe_id: str, name: str, ingredients: list[str]) -> bool:
        if recipe_id in self._recipes or self._name_taken(name):
            return False
        self._recipes[recipe_id] = _Recipe(recipe_id, name, ingredients)
        self._name_index[_key(name)] = recipe_id
        return True

    def get_recipe(self, recipe_id: str) -> dict | None:
        recipe = self._recipes.get(recipe_id)
        return recipe.public() if recipe is not None else None

    def update_recipe(self, recipe_id: str, name: str, ingredients: list[str]) -> bool:
        recipe = self._recipes.get(recipe_id)
        if recipe is None or self._name_taken(name, exclude_id=recipe_id):
            return False
        self._rename(recipe, name)
        recipe.ingredients = list(ingredients)           # NOTE: copy again -- do not alias the caller's list
        return True

    def delete_recipe(self, recipe_id: str) -> bool:
        recipe = self._recipes.pop(recipe_id, None)
        if recipe is None:
            return False
        del self._name_index[_key(recipe.name)]          # NOTE: frees the lookup key for reuse
        return True
```

Each method is $O(1)$. Replaying the level's example:

```python
mgr = RecipeManager()
assert mgr.add_recipe("r1", "Miso Soup", ["miso paste", "tofu", "scallion"]) is True
assert mgr.add_recipe("r2", "MISO SOUP", ["dashi", "kombu"]) is False
assert mgr.add_recipe("r1", "Ramen", ["noodles", "broth"]) is False
assert mgr.get_recipe("r1") == {"id": "r1", "name": "Miso Soup", "ingredients": ["miso paste", "tofu", "scallion"]}
assert mgr.get_recipe("r9") is None
assert mgr.update_recipe("r1", "miso soup", ["miso paste", "tofu"]) is True
assert mgr.get_recipe("r1")["name"] == "miso soup"
assert mgr.add_recipe("r2", "Chicken Ramen", ["noodles", "broth", "chicken"]) is True
assert mgr.update_recipe("r2", "Miso Soup", ["dashi"]) is False
assert mgr.delete_recipe("r2") is True
assert mgr.delete_recipe("r2") is False
assert mgr.get_recipe("r2") is None
print("Level 1 example replayed")
```

### Level 2

`list_recipes`, `find_by_ingredient` and `search_by_name_prefix` share one `_sorted` helper. It sorts by ingredient count alone, with no second key for the tie-break, and that is enough only because `self._recipes.values()` already yields recipes in creation order — a `dict` keeps the order its keys were first inserted in, `del` plus a later `[...] =` moves a reused id to the end, and changing `recipe.name`/`.ingredients` in place never touches that order — and because `sorted` is a *stable* sort, which leaves recipes that compare equal (the same ingredient count) in the relative order they arrived in. Building the candidate list from anything that does not preserve that order first, such as a `set` of matches, would leave ties in whatever order the string-hash seed happens to produce.

```python
class RecipeManager(RecipeManager):
    def list_recipes(self) -> list[dict]:
        return self._sorted(self._recipes.values())

    def find_by_ingredient(self, ingredient: str) -> list[dict]:
        key = _key(ingredient)
        return self._sorted(r for r in self._recipes.values() if any(_key(i) == key for i in r.ingredients))

    def search_by_name_prefix(self, prefix: str) -> list[dict]:
        key = _key(prefix)
        return self._sorted(r for r in self._recipes.values() if _key(r.name).startswith(key))

    @staticmethod
    def _sorted(recipes) -> list[dict]:
        return [r.public() for r in sorted(recipes, key=lambda r: len(r.ingredients))]   # NOTE: stable -- see above
```

`list_recipes` is $O(n \log n)$ in the number of recipes $n$. `find_by_ingredient` and `search_by_name_prefix` are $O(n \cdot m + k \log k)$, where $m$ is the average ingredient-list length and $k \le n$ the number of matches: every recipe's ingredients or name must be examined once before its matches can be sorted, the same as a full scan without an index. Replaying the level's example:

```python
mgr = RecipeManager()
mgr.add_recipe("r1", "Ramen", ["noodles", "broth", "egg"])
mgr.add_recipe("r2", "Fried Rice", ["rice", "egg", "soy sauce"])
mgr.add_recipe("r3", "Toast", ["bread"])
mgr.add_recipe("r4", "Congee", ["rice", "water"])
assert [r["id"] for r in mgr.list_recipes()] == ["r3", "r4", "r1", "r2"]
assert [r["id"] for r in mgr.find_by_ingredient("EGG")] == ["r1", "r2"]
assert [r["id"] for r in mgr.search_by_name_prefix("CO")] == ["r4"]
assert [r["id"] for r in mgr.search_by_name_prefix("")] == ["r3", "r4", "r1", "r2"]
print("Level 2 example replayed")
```

### Level 3

`add_recipe` calls Level 1's own `add_recipe` through `super()` for the id/name checks and the object itself, and only stamps `owner_id` on afterwards, so the collision rules stay written in exactly one place. `_authorized` is one function shared by `edit_recipe`, `delete_recipe` and, from Level 4, `rollback_recipe`; `allow_editor` is the one thing that tells `delete_recipe` apart from the other two — an owner always passes, a granted editor only when `allow_editor` is true. It does not check `self._users` itself: `add_recipe` already validates `owner_id` before storing it, and `grant_editor` below already validates `user_id` before adding it to `editors`, so an unregistered id can never equal a stored `owner_id` or already sit in an `editors` set — there is nothing left for `_authorized` to check on that account. `grant_editor` checks ownership directly rather than through `_authorized`, since `allow_editor=True` would let an editor grant further editors, which is not the rule.

```python
class RecipeManager(RecipeManager):
    def __init__(self) -> None:
        super().__init__()
        self._users: set[str] = set()   # membership only, like _Recipe.editors -- never iterated for output

    def add_user(self, user_id: str, username: str) -> bool:
        if user_id in self._users:
            return False
        self._users.add(user_id)
        return True

    def add_recipe(self, recipe_id: str, name: str, ingredients: list[str], owner_id: str) -> bool:
        if owner_id not in self._users:
            return False
        if not super().add_recipe(recipe_id, name, ingredients):
            return False
        self._recipes[recipe_id].owner_id = owner_id
        return True

    def _authorized(self, user_id: str, recipe: _Recipe | None, allow_editor: bool) -> bool:
        if recipe is None:
            return False
        return user_id == recipe.owner_id or (allow_editor and user_id in recipe.editors)

    def grant_editor(self, owner_id: str, recipe_id: str, user_id: str) -> bool:
        recipe = self._recipes.get(recipe_id)
        if recipe is None or owner_id != recipe.owner_id or user_id not in self._users:
            return False
        recipe.editors.add(user_id)
        return True

    def edit_recipe(self, user_id: str, recipe_id: str, name: str, ingredients: list[str]) -> bool:
        recipe = self._recipes.get(recipe_id)
        if not self._authorized(user_id, recipe, allow_editor=True) or self._name_taken(name, exclude_id=recipe_id):
            return False
        self._rename(recipe, name)
        recipe.ingredients = list(ingredients)          # NOTE: copy again -- do not alias the caller's list
        return True

    def delete_recipe(self, user_id: str, recipe_id: str) -> bool:
        recipe = self._recipes.get(recipe_id)
        if not self._authorized(user_id, recipe, allow_editor=False):
            return False
        return super().delete_recipe(recipe_id)         # NOTE: reuses Level 1's removal, not reimplemented
```

Every Level 3 method is $O(1)$. Replaying the level's example:

```python
mgr = RecipeManager()
assert mgr.add_user("u1", "Alice") is True
assert mgr.add_user("u2", "Bob") is True
assert mgr.add_user("u1", "Alice2") is False
assert mgr.add_recipe("r1", "Congee", ["rice", "water"], "u1") is True
assert mgr.add_recipe("r2", "Toast", ["bread"], "nope") is False
assert mgr.edit_recipe("u2", "r1", "Rice Congee", ["rice", "water", "ginger"]) is False
assert mgr.grant_editor("u2", "r1", "u2") is False
assert mgr.grant_editor("u1", "r1", "u2") is True
assert mgr.edit_recipe("u2", "r1", "Rice Congee", ["rice", "water", "ginger"]) is True
assert mgr.delete_recipe("u2", "r1") is False
assert mgr.delete_recipe("u1", "r1") is True
print("Level 3 example replayed")
```

### Level 4

`edit_recipe` is overridden once more, but only to add the append: `super().edit_recipe(...)` already does Level 3's authorization check, collision check, rename and ingredient assignment, so if it returns `False` nothing happened and this override returns `False` too, without touching `versions`; if it returns `True`, `recipe.name`/`recipe.ingredients` already hold the new state, and appending a fresh copy of them is the only work left. `rollback_recipe` reads the target version and hands it straight to `edit_recipe`: authorization, the collision check ("the old name is now used by a different recipe") and the append are then exactly `edit_recipe`'s, not reimplemented — a rollback is nothing but an edit whose new name and ingredients happen to come from history instead of from the caller.

```python
class RecipeManager(RecipeManager):
    def edit_recipe(self, user_id: str, recipe_id: str, name: str, ingredients: list[str]) -> bool:
        if not super().edit_recipe(user_id, recipe_id, name, ingredients):
            return False
        recipe = self._recipes[recipe_id]
        recipe.versions.append((recipe.name, list(recipe.ingredients)))   # NOTE: its own copy, aliasing nothing
        return True

    def get_recipe_history(self, recipe_id: str) -> list[dict] | None:
        recipe = self._recipes.get(recipe_id)
        if recipe is None:
            return None
        return [{"version": i, "name": n, "ingredients": list(ing)}
                for i, (n, ing) in enumerate(recipe.versions, start=1)]

    def get_version(self, recipe_id: str, version: int) -> dict | None:
        recipe = self._recipes.get(recipe_id)
        if recipe is None or not 1 <= version <= len(recipe.versions):
            return None
        name, ingredients = recipe.versions[version - 1]     # NOTE: O(1) -- history is 1-based, never shrinks
        return {"version": version, "name": name, "ingredients": list(ingredients)}

    def rollback_recipe(self, user_id: str, recipe_id: str, version: int) -> bool:
        recipe = self._recipes.get(recipe_id)
        if recipe is None or not 1 <= version <= len(recipe.versions):
            return False
        name, ingredients = recipe.versions[version - 1]
        return self.edit_recipe(user_id, recipe_id, name, ingredients)   # NOTE: same auth, same collision rule
```

`get_version` is $O(1)$ beyond copying the one version's ingredient list; `get_recipe_history` is $O(v)$ in the number of versions $v$, since it copies every one. `edit_recipe` and `rollback_recipe` stay $O(1)$. Replaying the level's example:

```python
mgr = RecipeManager()
mgr.add_user("u1", "Alice")
mgr.add_recipe("r1", "Congee", ["rice", "water"], "u1")
mgr.edit_recipe("u1", "r1", "Rice Porridge", ["rice", "water", "salt"])
mgr.edit_recipe("u1", "r1", "Jook", ["rice", "water", "ginger"])
assert mgr.get_recipe_history("r1") == [
    {"version": 1, "name": "Congee", "ingredients": ["rice", "water"]},
    {"version": 2, "name": "Rice Porridge", "ingredients": ["rice", "water", "salt"]},
    {"version": 3, "name": "Jook", "ingredients": ["rice", "water", "ginger"]},
]
assert mgr.get_version("r1", 1) == {"version": 1, "name": "Congee", "ingredients": ["rice", "water"]}
assert mgr.get_version("r1", 5) is None

assert mgr.add_recipe("r2", "Congee", ["rice"], "u1") is True
assert mgr.rollback_recipe("u1", "r1", 1) is False
assert mgr.grant_editor("u1", "r1", "u2") is False

mgr.add_user("u2", "Bob")
assert mgr.grant_editor("u1", "r1", "u2") is True
assert mgr.rollback_recipe("u2", "r1", 2) is True
assert mgr.get_recipe("r1") == {"id": "r1", "name": "Rice Porridge", "ingredients": ["rice", "water", "salt"]}
assert len(mgr.get_recipe_history("r1")) == 4
assert mgr.edit_recipe("u2", "r1", "Jook", ["rice", "water", "ginger"]) is True
assert mgr.delete_recipe("u2", "r1") is False
assert mgr.delete_recipe("u1", "r1") is True
print("Level 4 example replayed")
```

### Follow-ups

- Revoking an editor: not required here; a symmetric `revoke_editor(owner_id, recipe_id, user_id)` would remove one id from `editors` in $O(1)$ and would not need to touch `versions` at all, since permissions live on the recipe, not on any one version.
- Concurrent calls: nothing here locks, so two threads calling `edit_recipe` on the same `recipe_id` at once could both pass the collision check before either renames. A lock per `recipe_id`, held for the whole authorize-check-mutate sequence (the way `edit_recipe` and `rollback_recipe` already group that sequence into one method), would serialize exactly the calls that touch shared state without blocking edits to unrelated recipes.
- A recipe edited thousands of times: `get_recipe_history` copies every version on every call, so its cost grows with the recipe's whole edit history, not with what the caller actually wants. A paginated form, or one that takes a version range, would avoid re-copying versions nobody asked for.
- Transferring ownership: this problem never lets a recipe's owner change after `add_recipe`. Adding a transfer would have to decide whether it counts as a version of its own (here, no, since it changes neither name nor ingredients) and whether the current owner, an editor, or only an external admin may initiate it.
- Non-ASCII names: `casefold()`, not `lower()`, is what the lookup key uses, because `casefold` is meant for caseless matching across scripts — `"STRASSE".casefold() == "straße".casefold()` is `True`, while the same comparison with `lower()` is `False`.

```python
# ---- edge cases the level examples do not reach ----

# mutating a list after passing it in, or a dict/list a method returned, must never reach the store
mgr = RecipeManager()
mgr.add_user("u1", "Alice")
ingredients_in = ["rice", "water"]
mgr.add_recipe("r1", "Congee", ingredients_in, "u1")
ingredients_in.append("salt")
assert mgr.get_recipe("r1")["ingredients"] == ["rice", "water"]

fetched = mgr.get_recipe("r1")
fetched["ingredients"].append("intruder")
assert mgr.get_recipe("r1")["ingredients"] == ["rice", "water"]

mgr.edit_recipe("u1", "r1", "Rice Porridge", ["rice", "water", "salt"])
old_version = mgr.get_version("r1", 1)
old_version["ingredients"].append("intruder")
assert mgr.get_version("r1", 1)["ingredients"] == ["rice", "water"]          # history unaffected

# an empty store
empty = RecipeManager()
assert empty.list_recipes() == [] and empty.find_by_ingredient("rice") == [] and empty.search_by_name_prefix("") == []
assert empty.get_recipe("r1") is None and empty.get_recipe_history("r1") is None and empty.get_version("r1", 1) is None

# unknown recipe_id, unknown user_id, and version 0 are all in-range failures, not crashes
assert mgr.edit_recipe("u1", "ghost", "X", []) is False
assert mgr.delete_recipe("u1", "ghost") is False
assert mgr.grant_editor("u1", "ghost", "u1") is False
assert mgr.rollback_recipe("u1", "ghost", 1) is False
assert mgr.get_version("r1", 0) is None

# editing a recipe never moves it in the tie-break order, even though it is the most recently touched
mgr2 = RecipeManager()
mgr2.add_user("u1", "Alice")
mgr2.add_recipe("r1", "A-recipe", ["x", "y"], "u1")     # 2 ingredients, created 1st
mgr2.add_recipe("r2", "B-recipe", ["x"], "u1")          # 1 ingredient,  created 2nd
assert [r["id"] for r in mgr2.list_recipes()] == ["r2", "r1"]
mgr2.edit_recipe("u1", "r2", "B-recipe", ["x", "y"])    # now tied with r1 on count, but r1 was created first
assert [r["id"] for r in mgr2.list_recipes()] == ["r1", "r2"]

# rollback to the current latest version appends an identical copy rather than failing
mgr3 = RecipeManager()
mgr3.add_user("u1", "Alice")
mgr3.add_recipe("r1", "Soup", ["water"], "u1")
assert mgr3.rollback_recipe("u1", "r1", 1) is True
assert mgr3.get_recipe_history("r1") == [
    {"version": 1, "name": "Soup", "ingredients": ["water"]},
    {"version": 2, "name": "Soup", "ingredients": ["water"]},
]

# deleting a recipe and reusing its id starts an unrelated recipe: history and permissions do not carry over
mgr4 = RecipeManager()
mgr4.add_user("u1", "Alice")
mgr4.add_user("u2", "Bob")
mgr4.add_recipe("r1", "Old", ["x"], "u1")
mgr4.edit_recipe("u1", "r1", "Old Two", ["x", "y"])
mgr4.grant_editor("u1", "r1", "u2")
assert mgr4.delete_recipe("u1", "r1") is True
assert mgr4.add_recipe("r1", "New", ["z"], "u2") is True
assert mgr4.get_recipe_history("r1") == [{"version": 1, "name": "New", "ingredients": ["z"]}]
assert mgr4.edit_recipe("u1", "r1", "New Two", ["z"]) is False   # u1 owned the old r1, not this one
assert mgr4.edit_recipe("u2", "r1", "New Two", ["z"]) is True    # u2 is this recipe's actual owner
print("edge cases OK")
```

```python
# ---- independent reference model, straight from the statement ----
import random
from collections import Counter


class _NaiveRecipeManager:
    """Recipes and users are flat containers; every uniqueness or permission check re-scans them from
    scratch, and rollback is written out on its own rather than calling edit_recipe."""

    def __init__(self) -> None:
        self.recipes: dict[str, dict] = {}   # recipe_id -> {name, ingredients, owner, editors, versions}
        self.users: set[str] = set()

    def _name_used(self, name: str, exclude_id: str | None = None) -> bool:
        key = name.casefold()
        return any(rid != exclude_id and r["name"].casefold() == key for rid, r in self.recipes.items())

    def add_user(self, user_id, username):
        if user_id in self.users:
            return False
        self.users.add(user_id)
        return True

    def add_recipe(self, recipe_id, name, ingredients, owner_id):
        if recipe_id in self.recipes or self._name_used(name) or owner_id not in self.users:
            return False
        self.recipes[recipe_id] = {"name": name, "ingredients": list(ingredients), "owner": owner_id,
                                    "editors": set(), "versions": [(name, list(ingredients))]}
        return True

    def get_recipe(self, recipe_id):
        r = self.recipes.get(recipe_id)
        return None if r is None else {"id": recipe_id, "name": r["name"], "ingredients": list(r["ingredients"])}

    def _can_edit(self, user_id, r):
        return r is not None and (user_id == r["owner"] or user_id in r["editors"])

    def grant_editor(self, owner_id, recipe_id, user_id):
        r = self.recipes.get(recipe_id)
        if r is None or owner_id != r["owner"] or user_id not in self.users:
            return False
        r["editors"].add(user_id)
        return True

    def edit_recipe(self, user_id, recipe_id, name, ingredients):
        r = self.recipes.get(recipe_id)
        if not self._can_edit(user_id, r) or self._name_used(name, exclude_id=recipe_id):
            return False
        r["name"], r["ingredients"] = name, list(ingredients)
        r["versions"].append((name, list(ingredients)))
        return True

    def delete_recipe(self, user_id, recipe_id):
        r = self.recipes.get(recipe_id)
        if r is None or user_id != r["owner"]:
            return False
        del self.recipes[recipe_id]
        return True

    def _ordered(self, pairs):
        return sorted(pairs, key=lambda kv: len(kv[1]["ingredients"]))

    def list_recipes(self):
        return [{"id": rid, "name": r["name"], "ingredients": list(r["ingredients"])}
                for rid, r in self._ordered(self.recipes.items())]

    def find_by_ingredient(self, ingredient):
        key = ingredient.casefold()
        matches = [(rid, r) for rid, r in self.recipes.items() if any(i.casefold() == key for i in r["ingredients"])]
        return [{"id": rid, "name": r["name"], "ingredients": list(r["ingredients"])}
                for rid, r in self._ordered(matches)]

    def search_by_name_prefix(self, prefix):
        key = prefix.casefold()
        matches = [(rid, r) for rid, r in self.recipes.items() if r["name"].casefold().startswith(key)]
        return [{"id": rid, "name": r["name"], "ingredients": list(r["ingredients"])}
                for rid, r in self._ordered(matches)]

    def get_recipe_history(self, recipe_id):
        r = self.recipes.get(recipe_id)
        if r is None:
            return None
        return [{"version": i, "name": n, "ingredients": list(ing)} for i, (n, ing) in enumerate(r["versions"], 1)]

    def get_version(self, recipe_id, version):
        r = self.recipes.get(recipe_id)
        if r is None or not 1 <= version <= len(r["versions"]):
            return None
        n, ing = r["versions"][version - 1]
        return {"version": version, "name": n, "ingredients": list(ing)}

    def rollback_recipe(self, user_id, recipe_id, version):
        r = self.recipes.get(recipe_id)
        if not self._can_edit(user_id, r) or not 1 <= version <= len(r["versions"]):
            return False
        n, ing = r["versions"][version - 1]
        if self._name_used(n, exclude_id=recipe_id):
            return False
        r["name"], r["ingredients"] = n, list(ing)
        r["versions"].append((n, list(ing)))
        return True


def _apply(mgr, op):
    return getattr(mgr, op[0])(*op[1:])


def _random_ops(rng, n, recipe_ids, user_ids, names, ingredient_pool):
    ops = []
    for _ in range(n):
        kind = rng.choices(
            ["add_user", "add_recipe", "get_recipe", "edit_recipe", "delete_recipe", "grant_editor",
             "list_recipes", "find_by_ingredient", "search_by_name_prefix",
             "get_recipe_history", "get_version", "rollback_recipe"],
            weights=[2, 4, 3, 4, 2, 3, 2, 3, 3, 2, 3, 3])[0]
        rid, uid, name = rng.choice(recipe_ids), rng.choice(user_ids), rng.choice(names)
        ingredients = rng.sample(ingredient_pool, rng.randint(1, 4))
        if kind == "add_user":
            ops.append(("add_user", uid, f"name-{uid}"))
        elif kind == "add_recipe":
            ops.append(("add_recipe", rid, name, ingredients, uid))
        elif kind == "get_recipe":
            ops.append(("get_recipe", rid))
        elif kind == "edit_recipe":
            ops.append(("edit_recipe", uid, rid, name, ingredients))
        elif kind == "delete_recipe":
            ops.append(("delete_recipe", uid, rid))
        elif kind == "grant_editor":
            ops.append(("grant_editor", uid, rid, rng.choice(user_ids)))
        elif kind == "list_recipes":
            ops.append(("list_recipes",))
        elif kind == "find_by_ingredient":
            ops.append(("find_by_ingredient", rng.choice(ingredient_pool)))
        elif kind == "search_by_name_prefix":
            ops.append(("search_by_name_prefix", name[:rng.randint(0, len(name))]))
        elif kind == "get_recipe_history":
            ops.append(("get_recipe_history", rid))
        elif kind == "get_version":
            ops.append(("get_version", rid, rng.randint(0, 5)))
        else:
            ops.append(("rollback_recipe", uid, rid, rng.randint(0, 5)))
    return ops


def _agree(seed, n_ops):
    rng = random.Random(seed)
    recipe_ids = ["r1", "r2", "r3"]
    user_ids = ["u1", "u2", "u3", "ghost"]           # small pool -- ids collide and get reused across ops
    names = ["Congee", "congee", "Toast", "Jook"]     # "Congee"/"congee" deliberately share a lookup key
    ingredient_pool = ["rice", "water", "egg", "bread", "salt"]
    real, naive = RecipeManager(), _NaiveRecipeManager()
    counts = Counter()
    for op in _random_ops(rng, n_ops, recipe_ids, user_ids, names, ingredient_pool):
        got, want = _apply(real, op), _apply(naive, op)
        assert got == want, (seed, op, got, want)
        if op[0] in ("add_recipe", "edit_recipe", "delete_recipe", "grant_editor", "rollback_recipe"):
            counts[(op[0], bool(want))] += 1
    return counts


totals = Counter()
for seed in range(400):
    totals += _agree(seed, n_ops=40 + seed % 40)
assert min(totals.values()) > 10, totals   # every mutator, both succeeding and blocked, fired repeatedly
print(f"cross-validated 400 random operation sequences against an independent reference model: {dict(totals)}")
print("all checks passed")
```
