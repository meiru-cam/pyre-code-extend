"""A recipe book with case-insensitive unique names, search, per-recipe permissions and version history."""

from ._interview import interview

# A slow model that rebuilds every answer from a log of versions, and a random call generator.
_MODEL = r"""
import random

class Model:
    def __init__(self):
        self.rows, self.users, self.made = {}, set(), 0
    def taken(self, name, rid=None):
        return any(r != rid and row["v"][-1][0].casefold() == name.casefold() for r, row in self.rows.items())
    def add(self, rid, name, ing, owner=None):
        if rid in self.rows or self.taken(name):
            return False
        self.made += 1
        self.rows[rid] = {"v": [(name, list(ing))], "made": self.made, "owner": owner, "eds": set()}
        return True
    def change(self, rid, name, ing):
        if rid not in self.rows or self.taken(name, rid):
            return False
        self.rows[rid]["v"].append((name, list(ing)))
        return True
    def view(self, rid):
        name, ing = self.rows[rid]["v"][-1]
        return {"id": rid, "name": name, "ingredients": list(ing)}
    def listing(self):
        return [self.view(r) for r in sorted(self.rows, key=lambda r: (len(self.rows[r]["v"][-1][1]), self.rows[r]["made"]))]

def random_ops(rng, n, levels):
    ids, names = ["a", "b", "c", "d"], ["Stew", "stew", "STEW ", "Pie", "pie crust", "Soup", "Salad"]
    foods = ["Cumin", "cumin", "leek", "Leek", "dill", "kale"]
    users = ["k1", "k2", "k3"]
    model, out = Model(), []
    for _ in range(n):
        ops = ["add", "add", "get", "update", "delete"]
        if levels >= 2:
            ops += ["list", "find", "prefix"]
        if levels >= 3:
            ops += ["user", "owned", "owned", "grant", "edit", "edit", "remove"]
        if levels >= 4:
            ops += ["history", "version", "rollback", "rollback"]
        op = rng.choice(ops)
        rid, name, user = rng.choice(ids), rng.choice(names), rng.choice(users)
        ing = [rng.choice(foods) for _ in range(rng.randint(0, 4))]
        row = model.rows.get(rid)
        can = row is not None and row["owner"] is not None and (user == row["owner"] or user in row["eds"])
        if op == "add":
            args, want = (rid, name, ing), model.add(rid, name, ing)
        elif op == "get":
            args, want = (rid,), model.view(rid) if row else None
        elif op == "update":
            args, want = (rid, name, ing), model.change(rid, name, ing)
        elif op == "delete":
            args, want = (rid,), model.rows.pop(rid, None) is not None
        elif op == "list":
            args, want = (), model.listing()
        elif op == "find":
            food = rng.choice(foods + ["cum"])
            args, want = (food,), [r for r in model.listing() if any(i.casefold() == food.casefold() for i in r["ingredients"])]
        elif op == "prefix":
            pre = rng.choice(["", "s", "ST", "pie", "Pie c", "x"])
            args, want = (pre,), [r for r in model.listing() if r["name"].casefold().startswith(pre.casefold())]
        elif op == "user":
            args, want = (user,), user not in model.users
            model.users.add(user)
        elif op == "owned":
            args = (user, rid, name, ing)
            want = user in model.users and model.add(rid, name, ing, user)
        elif op == "grant":
            other = rng.choice(users + ["nobody"])
            args = (user, rid, other)
            want = row is not None and row["owner"] is not None and row["owner"] == user and other in model.users
            if want:
                row["eds"].add(other)
        elif op == "edit":
            args, want = (user, rid, name, ing), can and model.change(rid, name, ing)
        elif op == "remove":
            args, want = (user, rid), row is not None and row["owner"] is not None and row["owner"] == user
            if want:
                del model.rows[rid]
        elif op == "history":
            args = (rid,)
            want = None if row is None else [{"version": i + 1, "name": n, "ingredients": list(g)} for i, (n, g) in enumerate(row["v"])]
        elif op == "version":
            k = rng.randint(-1, 5)
            args = (rid, k)
            want = None if row is None or not 1 <= k <= len(row["v"]) else {"version": k, "name": row["v"][k - 1][0], "ingredients": list(row["v"][k - 1][1])}
        else:
            k = rng.randint(0, 5)
            args = (user, rid, k)
            want = can and 1 <= k <= len(row["v"]) and model.change(rid, *row["v"][k - 1])
        out.append((op, args, want))
    return out

METHOD = {"add": "add_recipe", "get": "get_recipe", "update": "update_recipe", "delete": "delete_recipe",
          "list": "list_recipes", "find": "find_by_ingredient", "prefix": "search_by_name_prefix",
          "user": "add_user", "owned": "add_owned_recipe", "grant": "grant_editor", "edit": "edit_recipe",
          "remove": "remove_recipe", "history": "get_recipe_history", "version": "get_version", "rollback": "rollback_recipe"}

def replay(book, ops, label):
    for i, (op, args, want) in enumerate(ops):
        got = getattr(book, METHOD[op])(*[list(a) if isinstance(a, list) else a for a in args])
        assert got == want, (label, i, op, args, got, want)
"""

TASK = {
    "title": "Recipe Book With Permissions and History",
    "difficulty": "Medium",
    "version": 1,
    "function_name": "RecipeBook",
    "description_en": r"""Build `RecipeBook`, which stores recipes with unique names, searches them, controls who may change them, and keeps every version.

The requirement arrives in parts. Each part keeps every earlier behavior, so one `RecipeBook` class passes all parts at the end. Pass every test of the current part to reveal the next one.

**Rules for every part:**
- A recipe has an id, a name and a list of ingredient strings. Ids and user ids are non-empty strings chosen by the caller.
- Ingredients keep the order and duplicates they were given in.
- A recipe dict is `{"id": ..., "name": ..., "ingredients": [...]}`. Names come back exactly as stored.
- Two names clash when their `casefold()` values are equal: `"Pie"` and `"PIE"` clash, `"Pie"` and `"Pie "` do not. No two recipes may hold clashing names at once.
- Lists go in and out as copies. Changing a list you passed in, or a list you got back, never changes the book.
- A call that fails changes nothing. No method raises.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why this shows up in interviews:** it is a timed online assessment about keeping several indexes consistent: an id map, a name index and a creation order. Each later part adds one requirement, and a name index that misses one update path shows up as a wrong clash later.

**Where it is used:** content management systems, wikis with page history and access control, and any CRUD service with unique slugs.

Adapted from the recipe manager online assessment in Schuture/Anthropic-Interview-Notes (CC BY-NC 4.0), reworded, on one class. The source replaces `add_recipe` and `delete_recipe` with owner-aware forms in its third level; here the owner-aware forms are new methods, `add_owned_recipe` and `remove_recipe`, so the earlier ones keep working: `update_recipe` and `delete_recipe` stay callable with no permission check, and in Part 4 `update_recipe` also adds a version. A recipe made by plain `add_recipe` has no owner, so edit, remove, grant and rollback refuse it. Users take only an id, and `add_owned_recipe` takes the owner first.""",
    "parts": [
        {
            "title": "Add, read, update and delete",
            "description_en": r"""**Signatures:**
- `RecipeBook().add_recipe(recipe_id, name, ingredients) -> bool` creates the recipe and returns `True`. It returns `False` if the id exists or the name clashes with any recipe.
- `get_recipe(recipe_id) -> dict | None` returns the recipe dict, or `None`.
- `update_recipe(recipe_id, name, ingredients) -> bool` replaces the name and ingredients. It returns `False` if the id does not exist or the name clashes with a different recipe. A recipe never clashes with itself, so changing only the case of its own name is allowed.
- `delete_recipe(recipe_id) -> bool` removes the recipe and returns `True`, or returns `False` if there is none. Its id and name are free again.

**Example:**
- `add_recipe("d1", "Lentil Stew", ["lentils", "onion", "cumin"])` and `add_recipe("d3", "Flatbread", ["flour", "water"])` are `True`
- `update_recipe("d3", "LENTIL stew", [])` is `False`: the name clashes with `d1`
- `delete_recipe("d1")` is `True`, which frees the name: `add_recipe("d2", "lentil STEW", ["lentils"])` is `True`
- `update_recipe("d2", "Lentil Stew", ["lentils", "leek"])` is `True`: only the case of its own name changes, and `get_recipe("d2")` is `{"id": "d2", "name": "Lentil Stew", "ingredients": ["lentils", "leek"]}`
- `get_recipe("d1")` is `None`, a second `delete_recipe("d1")` is `False`, and `add_recipe("d3", "Pita", [])` is `False`: `d3` exists""",
        },
        {
            "title": "List and search",
            "description_en": r"""Keep Part 1 and add three read methods. Each returns a list of recipe dicts in the same order.

**The order:** fewer ingredients first. Ties go by creation: the recipe added first comes first. `update_recipe` keeps a recipe's place; deleting a recipe and adding one with the same id counts as a new, latest creation.

**Signatures:**
- `list_recipes() -> list[dict]` returns every recipe.
- `find_by_ingredient(ingredient) -> list[dict]` returns the recipes with an ingredient equal to `ingredient` after `casefold()`. Part of an ingredient name does not match.
- `search_by_name_prefix(prefix) -> list[dict]` returns the recipes whose casefolded name starts with the casefolded `prefix`. An empty prefix matches all.

**Example:** add `d1` `"Pesto Pasta"` `["pasta", "pesto"]`, `d2` `"Garlic Bread"` `["bread", "Garlic", "butter"]`, `d3` `"Grilled Corn"` `["corn", "butter"]`, `d4` `"Tomato Soup"` `["tomato", "garlic", "stock", "cream"]`:
- `list_recipes()` ids are `["d1", "d3", "d2", "d4"]`
- `find_by_ingredient("GARLIC")` ids are `["d2", "d4"]`; `find_by_ingredient("garl")` is `[]`
- `search_by_name_prefix("g")` ids are `["d3", "d2"]`
- after `delete_recipe("d1")` and `add_recipe("d1", "Pesto Pasta", ["pasta", "pesto"])`, `list_recipes()` ids are `["d3", "d1", "d2", "d4"]`""",
        },
        {
            "title": "Users and permissions",
            "description_en": r"""Keep Parts 1–2 and add owners and editors. Part 1's methods keep working unchanged.

**Signatures:**
- `add_user(user_id) -> bool` registers a user, or returns `False` if the id is already registered.
- `add_owned_recipe(owner_id, recipe_id, name, ingredients) -> bool` works like `add_recipe` and makes `owner_id` the owner. It returns `False` if `owner_id` is not registered.
- `grant_editor(owner_id, recipe_id, user_id) -> bool` lets `user_id` edit the recipe. It returns `False` if the recipe does not exist, `owner_id` is not its owner, or `user_id` is not registered. Granting twice returns `True`.
- `edit_recipe(user_id, recipe_id, name, ingredients) -> bool` works like `update_recipe`, but only for the owner or a granted editor.
- `remove_recipe(user_id, recipe_id) -> bool` works like `delete_recipe`, but only for the owner. Editors may not remove.
- A recipe made by `add_recipe` has no owner: `edit_recipe` and `remove_recipe` refuse it for every user.
- `update_recipe` and `delete_recipe` keep working on every recipe with no permission check. Reading and searching stay open to everyone.
- Removing a recipe, by `remove_recipe` or `delete_recipe`, drops its owner and editors: a new recipe with the same id starts with none.

**Example:**
- `add_recipe("d3", "Crispbread", ["rye"])` and `add_user("chef-a")` are `True`; `edit_recipe("chef-a", "d3", "Crispbread", [])` is `False`: `d3` has no owner
- `add_owned_recipe("chef-a", "d1", "Pea Soup", ["peas", "mint"])` is `True`; `add_owned_recipe("chef-z", "d2", "Oatcake", ["oats"])` is `False`
- `add_user("chef-b")` is `True`, a second time `False`
- `grant_editor("chef-a", "d1", "chef-b")` is `True`; then `remove_recipe("chef-b", "d1")` is `False` and `edit_recipe("chef-b", "d1", "Pea and Mint Soup", ["peas", "mint", "stock"])` is `True`
- `grant_editor("chef-a", "d1", "chef-z")` is `False`: not a user; `remove_recipe("chef-a", "d1")` is `True`""",
        },
        {
            "title": "Version history and rollback",
            "description_en": r"""Keep Parts 1–3. Every successful change now adds a version instead of overwriting.

- Creating a recipe makes version `1`. Each successful `update_recipe`, `edit_recipe` or `rollback_recipe` appends the next version. Versions are numbered from `1` and never change or disappear while the recipe exists.
- `get_recipe` and the Part 2 methods use the latest version. A new recipe under a deleted id starts again at version `1`.

**Signatures:**
- `get_recipe_history(recipe_id) -> list[dict] | None` returns `{"version": k, "name": ..., "ingredients": [...]}` for every version, oldest first, or `None` if there is no such recipe.
- `get_version(recipe_id, version) -> dict | None` returns one of those entries, or `None` if the recipe or version does not exist.
- `rollback_recipe(user_id, recipe_id, version) -> bool` appends a copy of that version as the newest one. It needs the same permission as `edit_recipe`. It returns `False` if the permission is missing, the version does not exist, or the old name clashes with a different recipe now. Rolling back to the latest version appends an identical copy.

**Example:** `chef-a` owns `d1`, made as `"Chili"` `["beans", "chili"]`, then edited to `"Chili"` `["beans", "chili", "corn"]` and to `"Smoky Chili"` `["beans", "chipotle"]`:
- `rollback_recipe("chef-a", "d1", 1)` is `True`: version `4` is `"Chili"` `["beans", "chili"]`, and `get_recipe("d1")["name"]` is `"Chili"`
- `get_version("d1", 3)["name"]` is `"Smoky Chili"`; `get_version("d1", 9)` is `None`
- `add_recipe("d2", "smoky chili", ["beans"])` is `True`, so `rollback_recipe("chef-a", "d1", 3)` is now `False`""",
        },
    ],
    "hints": [
        {"level": 1, "kind": "questions", "content": "Checking a new name against every stored recipe is O(n). Which dict would answer \"is this name taken, and by whom?\" in O(1), and which key makes \"Pie\" and \"PIE\" land on the same entry?"},
        {"level": 2, "kind": "analysis", "content": "Keep recipes: id -> (name, ingredients) and names: name.casefold() -> id. add checks both dicts, then writes both. update allows names.get(key) in (None, recipe_id), deletes the old name's key and writes the new one. delete removes both entries. Copy ingredient lists in and out with list(...)."},
    ],
    "model_connections": [
        "Model registries keep every version of a model with an owner and a list of people allowed to promote or roll it back.",
        "Prompt and dataset versioning tools append versions and roll back by copying an old version forward, as rollback_recipe does here.",
    ],
    "pro_con_analysis": {
        "pros": [
            "A casefold name index makes the clash check O(1) on every write.",
            "A creation counter on each recipe gives a stable tie-break that survives updates.",
            "Appending versions keeps history immutable, so rollback is just another append.",
        ],
        "cons": [
            "Every write path must keep the name index in sync, and one missed path causes false clashes or duplicate names.",
            "Listing and searching sort every recipe on each call, O(n log n).",
            "Storing each version in full costs memory proportional to the number of edits.",
        ],
    },
    "tests": [
        {"name": "Part 1: the worked example", "part": 1, "behavior": "state.invariant", "code": r"""
book = {fn}()
assert book.add_recipe("d1", "Lentil Stew", ["lentils", "onion", "cumin"]) is True
assert book.add_recipe("d3", "Flatbread", ["flour", "water"]) is True
assert book.update_recipe("d3", "LENTIL stew", []) is False
assert book.delete_recipe("d1") is True
assert book.add_recipe("d2", "lentil STEW", ["lentils"]) is True
assert book.update_recipe("d2", "Lentil Stew", ["lentils", "leek"]) is True
assert book.get_recipe("d2") == {"id": "d2", "name": "Lentil Stew", "ingredients": ["lentils", "leek"]}
assert book.get_recipe("d1") is None
assert book.delete_recipe("d1") is False
assert book.add_recipe("d3", "Pita", []) is False
assert book.get_recipe("d3")["name"] == "Flatbread"
"""},
        {"name": "Part 1: copies, clashes and freed names", "part": 1, "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "Ingredient lists must be copied in and out; names clash only by casefold (a trailing space is a different name); a failed call changes nothing; renaming frees the old name and deleting frees both id and name.",
         "code": r"""
book = {fn}()
mine = ["dill", "dill", "kale"]
assert book.add_recipe("a", "Pie", mine)
mine.append("oops")
got = book.get_recipe("a")
assert got["ingredients"] == ["dill", "dill", "kale"], "order and duplicates kept, input list copied"
got["ingredients"].clear()
assert book.get_recipe("a")["ingredients"] == ["dill", "dill", "kale"], "returned list must be a copy"
assert book.add_recipe("b", "Pie ", ["x"]) is True, "a trailing space makes a different name"
assert book.add_recipe("c", "GROSS", []) is True
assert book.add_recipe("d", "groß", []) is False, "casefold, not lower: groß and GROSS clash"
assert book.update_recipe("a", "pie ", ["y"]) is False and book.get_recipe("a")["name"] == "Pie"
assert book.update_recipe("missing", "Tart", []) is False
assert book.update_recipe("a", "Tart", ["z"]) is True
assert book.add_recipe("e", "PIE", []) is True, "the old name is free after a rename"
assert book.add_recipe("f", "tart", []) is False
assert book.delete_recipe("a") is True
assert book.add_recipe("a", "TART", ["n"]) is True, "id and name are free after delete"
"""},
        {"name": "Part 1: random calls", "part": 1, "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "On a random sequence of add, get, update and delete calls, a return value differed from a simple model.",
         "code": _MODEL + r"""
for seed in range(150):
    replay({fn}(), random_ops(random.Random(seed), 60, 1), seed)
"""},
        {"name": "Part 2: the worked example", "part": 2, "behavior": "state.invariant", "code": r"""
book = {fn}()
book.add_recipe("d1", "Pesto Pasta", ["pasta", "pesto"])
book.add_recipe("d2", "Garlic Bread", ["bread", "Garlic", "butter"])
book.add_recipe("d3", "Grilled Corn", ["corn", "butter"])
book.add_recipe("d4", "Tomato Soup", ["tomato", "garlic", "stock", "cream"])
ids = lambda rows: [r["id"] for r in rows]
assert ids(book.list_recipes()) == ["d1", "d3", "d2", "d4"]
assert ids(book.find_by_ingredient("GARLIC")) == ["d2", "d4"]
assert book.find_by_ingredient("garl") == []
assert ids(book.search_by_name_prefix("g")) == ["d3", "d2"]
book.delete_recipe("d1")
book.add_recipe("d1", "Pesto Pasta", ["pasta", "pesto"])
assert ids(book.list_recipes()) == ["d3", "d1", "d2", "d4"]
assert book.list_recipes()[0] == {"id": "d3", "name": "Grilled Corn", "ingredients": ["corn", "butter"]}
"""},
        {"name": "Part 2: order after updates and empty books", "part": 2, "visibility": "unshown", "behavior": "edge.empty_or_boundary",
         "failure_message": "An empty book lists nothing; update keeps a recipe's creation place among ties while its new ingredient count moves it; an empty prefix matches all; returned dicts are copies.",
         "code": r"""
book = {fn}()
assert book.list_recipes() == [] and book.find_by_ingredient("x") == [] and book.search_by_name_prefix("") == []
for rid in "abc":
    book.add_recipe(rid, "Dish " + rid, ["x"])
book.update_recipe("a", "Dish A2", ["y"])
assert [r["id"] for r in book.list_recipes()] == ["a", "b", "c"], "an update keeps the creation place"
book.update_recipe("a", "Dish A3", ["y", "z"])
assert [r["id"] for r in book.list_recipes()] == ["b", "c", "a"]
assert [r["id"] for r in book.search_by_name_prefix("")] == ["b", "c", "a"]
assert [r["id"] for r in book.search_by_name_prefix("DISH A")] == ["a"]
book.list_recipes()[0]["ingredients"].append("junk")
assert book.get_recipe("b")["ingredients"] == ["x"]
"""},
        {"name": "Part 2: random calls", "part": 2, "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "On a random sequence of calls including list_recipes, find_by_ingredient and search_by_name_prefix, a return value differed from a simple model.",
         "code": _MODEL + r"""
for seed in range(150):
    replay({fn}(), random_ops(random.Random(seed), 60, 2), seed)
"""},
        {"name": "Part 3: the worked example", "part": 3, "behavior": "protocol.validation", "code": r"""
book = {fn}()
assert book.add_recipe("d3", "Crispbread", ["rye"]) is True
assert book.add_user("chef-a") is True
assert book.edit_recipe("chef-a", "d3", "Crispbread", []) is False
assert book.add_owned_recipe("chef-a", "d1", "Pea Soup", ["peas", "mint"]) is True
assert book.add_owned_recipe("chef-z", "d2", "Oatcake", ["oats"]) is False and book.get_recipe("d2") is None
assert book.add_user("chef-b") is True and book.add_user("chef-b") is False
assert book.grant_editor("chef-a", "d1", "chef-b") is True
assert book.remove_recipe("chef-b", "d1") is False
assert book.edit_recipe("chef-b", "d1", "Pea and Mint Soup", ["peas", "mint", "stock"]) is True
assert book.get_recipe("d1")["name"] == "Pea and Mint Soup"
assert book.grant_editor("chef-a", "d1", "chef-z") is False
assert book.remove_recipe("chef-a", "d1") is True
"""},
        {"name": "Part 3: who may do what", "part": 3, "visibility": "unshown", "behavior": "protocol.validation",
         "failure_message": "Only the owner may grant or remove; the owner or a granted editor may edit; unregistered users and missing recipes always get False; a recipe without an owner refuses edit and remove; editors of one recipe have no rights on another.",
         "code": r"""
book = {fn}()
for u in ["o", "e", "x"]:
    assert book.add_user(u)
assert book.add_owned_recipe("o", "r", "Pho", ["noodles"])
assert book.add_owned_recipe("x", "s", "Bun", ["noodles"])
assert book.add_owned_recipe("o", "t", "pho", []) is False, "the owned form still checks names"
assert book.edit_recipe("e", "r", "Pho 2", []) is False
assert book.grant_editor("e", "r", "e") is False, "only the owner may grant"
assert book.grant_editor("o", "r", "e") and book.grant_editor("o", "r", "e"), "granting twice is fine"
assert book.grant_editor("o", "missing", "e") is False
assert book.edit_recipe("e", "r", "BUN", []) is False, "the name clash still applies"
assert book.edit_recipe("e", "r", "Pho 2", ["noodles", "basil"]) is True
assert book.edit_recipe("e", "s", "Bun 2", []) is False, "rights are per recipe"
assert book.edit_recipe("nobody", "r", "Pho 3", []) is False
assert book.remove_recipe("e", "r") is False and book.remove_recipe("nobody", "r") is False
assert book.remove_recipe("o", "missing") is False
book.add_recipe("free", "Plain", [])
assert book.edit_recipe("o", "free", "Plain 2", []) is False and book.remove_recipe("o", "free") is False
assert book.update_recipe("r", "Pho 3", []) is True, "the unchecked Part 1 methods keep working"
assert book.remove_recipe("o", "r") is True
assert book.get_recipe("r") is None
assert book.add_owned_recipe("x", "r", "Pho", [])
assert book.edit_recipe("e", "r", "Pho 4", []) is False, "a new recipe with an old id has no old editors"
assert book.add_owned_recipe("o", "z", "Laksa", []) and book.grant_editor("o", "z", "e")
assert book.delete_recipe("z") is True
assert book.add_owned_recipe("x", "z", "Laksa", [])
assert book.edit_recipe("e", "z", "Laksa 2", []) is False, "delete_recipe drops the editors too"
"""},
        {"name": "Part 3: random calls", "part": 3, "visibility": "unshown", "behavior": "protocol.validation",
         "failure_message": "On a random sequence of calls with users, owners and editors, a return value differed from a simple model.",
         "code": _MODEL + r"""
for seed in range(150):
    replay({fn}(), random_ops(random.Random(seed), 80, 3), seed)
"""},
        {"name": "Part 4: the worked example", "part": 4, "behavior": "state.invariant", "code": r"""
book = {fn}()
book.add_user("chef-a")
book.add_owned_recipe("chef-a", "d1", "Chili", ["beans", "chili"])
assert book.edit_recipe("chef-a", "d1", "Chili", ["beans", "chili", "corn"])
assert book.edit_recipe("chef-a", "d1", "Smoky Chili", ["beans", "chipotle"])
assert book.rollback_recipe("chef-a", "d1", 1) is True
assert book.get_version("d1", 4) == {"version": 4, "name": "Chili", "ingredients": ["beans", "chili"]}
assert book.get_recipe("d1")["name"] == "Chili"
assert book.get_version("d1", 3)["name"] == "Smoky Chili"
assert book.get_version("d1", 9) is None
assert len(book.get_recipe_history("d1")) == 4
assert book.add_recipe("d2", "smoky chili", ["beans"]) is True
assert book.rollback_recipe("chef-a", "d1", 3) is False
"""},
        {"name": "Part 4: history edges", "part": 4, "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "Every successful update, edit or rollback appends one version and failed ones append none; versions start at 1; out-of-range versions and missing recipes give None or False; history entries are copies; a recipe's own old names never clash with it; deleting a recipe drops its history.",
         "code": r"""
book = {fn}()
book.add_user("o")
book.add_user("e")
assert book.get_recipe_history("none") is None and book.get_version("none", 1) is None
book.add_recipe("p", "Plain", ["a"])
assert book.update_recipe("p", "PLAIN", ["a", "b"])
assert book.update_recipe("p", "Taken", []) and book.update_recipe("p", "Plain", ["c"])
assert [h["version"] for h in book.get_recipe_history("p")] == [1, 2, 3, 4]
assert book.get_version("p", 0) is None and book.get_version("p", -1) is None
assert book.rollback_recipe("o", "p", 1) is False, "no owner, so no rollback"
book.add_owned_recipe("o", "r", "Risotto", ["arborio"])
book.add_recipe("q", "Taken", [])
assert book.edit_recipe("o", "r", "TAKEN", []) is False
assert len(book.get_recipe_history("r")) == 1, "a failed edit adds no version"
assert book.rollback_recipe("e", "r", 1) is False, "no permission"
book.grant_editor("o", "r", "e")
assert book.rollback_recipe("e", "r", 1) is True, "rolling back to the latest version appends a copy"
assert book.rollback_recipe("e", "r", 9) is False and book.rollback_recipe("e", "r", 0) is False
assert book.get_recipe_history("r") == [{"version": 1, "name": "Risotto", "ingredients": ["arborio"]},
                                         {"version": 2, "name": "Risotto", "ingredients": ["arborio"]}]
book.get_recipe_history("r")[0]["ingredients"].append("junk")
book.get_version("r", 1)["ingredients"].append("junk")
assert book.get_version("r", 1)["ingredients"] == ["arborio"]
assert book.edit_recipe("o", "r", "rISOTTO", ["x"]) and book.rollback_recipe("o", "r", 1), "its own old name is no clash"
assert book.remove_recipe("o", "r")
assert book.add_owned_recipe("o", "r", "New", []) and len(book.get_recipe_history("r")) == 1
"""},
        {"name": "Part 4: random calls", "part": 4, "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "On a random sequence of calls with versions and rollbacks, a return value differed from a simple model that appends one version per successful change.",
         "code": _MODEL + r"""
for seed in range(150):
    replay({fn}(), random_ops(random.Random(seed), 80, 4), seed)
"""},
    ],
    "solution": r'''# Adapted from Schuture/Anthropic-Interview-Notes (code under the MIT License).
class RecipeBook:
    def __init__(self):
        self._recipes = {}  # recipe_id -> {"versions": [(name, ingredients)], "seq": n, "owner": id or None, "editors": set}
        self._names = {}    # name.casefold() -> recipe_id
        self._seq = 0       # creation counter for the listing order
        self._users = set()

    def _view(self, recipe_id):
        name, ingredients = self._recipes[recipe_id]["versions"][-1]
        return {"id": recipe_id, "name": name, "ingredients": list(ingredients)}  # copies out

    def _name_free(self, name, recipe_id=None):
        holder = self._names.get(name.casefold())
        return holder is None or holder == recipe_id  # a recipe never collides with itself

    def _create(self, recipe_id, name, ingredients, owner):
        if recipe_id in self._recipes or not self._name_free(name):
            return False
        self._seq += 1
        self._recipes[recipe_id] = {"versions": [(name, list(ingredients))], "seq": self._seq,
                                    "owner": owner, "editors": set()}
        self._names[name.casefold()] = recipe_id
        return True

    def _change(self, recipe_id, name, ingredients):
        if not self._name_free(name, recipe_id):
            return False
        recipe = self._recipes[recipe_id]
        del self._names[recipe["versions"][-1][0].casefold()]
        recipe["versions"].append((name, list(ingredients)))  # a new version; history is never rewritten
        self._names[name.casefold()] = recipe_id
        return True

    def _drop(self, recipe_id):
        recipe = self._recipes.pop(recipe_id)
        del self._names[recipe["versions"][-1][0].casefold()]

    def add_recipe(self, recipe_id, name, ingredients):
        return self._create(recipe_id, name, ingredients, None)

    def get_recipe(self, recipe_id):
        return self._view(recipe_id) if recipe_id in self._recipes else None

    def update_recipe(self, recipe_id, name, ingredients):
        return recipe_id in self._recipes and self._change(recipe_id, name, ingredients)

    def delete_recipe(self, recipe_id):
        if recipe_id not in self._recipes:
            return False
        self._drop(recipe_id)
        return True

    def list_recipes(self):
        order = sorted(self._recipes, key=lambda r: (len(self._recipes[r]["versions"][-1][1]), self._recipes[r]["seq"]))
        return [self._view(r) for r in order]

    def find_by_ingredient(self, ingredient):
        want = ingredient.casefold()
        return [r for r in self.list_recipes() if any(i.casefold() == want for i in r["ingredients"])]

    def search_by_name_prefix(self, prefix):
        want = prefix.casefold()
        return [r for r in self.list_recipes() if r["name"].casefold().startswith(want)]

    def add_user(self, user_id):
        if user_id in self._users:
            return False
        self._users.add(user_id)
        return True

    def add_owned_recipe(self, owner_id, recipe_id, name, ingredients):
        return owner_id in self._users and self._create(recipe_id, name, ingredients, owner_id)

    def _owner(self, recipe_id):
        recipe = self._recipes.get(recipe_id)
        return recipe and recipe["owner"]

    def _can_edit(self, user_id, recipe_id):
        recipe = self._recipes.get(recipe_id)
        return bool(recipe and recipe["owner"] is not None
                    and (user_id == recipe["owner"] or user_id in recipe["editors"]))

    def grant_editor(self, owner_id, recipe_id, user_id):
        if self._owner(recipe_id) != owner_id or user_id not in self._users:
            return False
        self._recipes[recipe_id]["editors"].add(user_id)
        return True

    def edit_recipe(self, user_id, recipe_id, name, ingredients):
        return self._can_edit(user_id, recipe_id) and self._change(recipe_id, name, ingredients)

    def remove_recipe(self, user_id, recipe_id):
        if self._owner(recipe_id) != user_id:
            return False  # editors may edit but never delete
        self._drop(recipe_id)
        return True

    def get_recipe_history(self, recipe_id):
        if recipe_id not in self._recipes:
            return None
        return [{"version": i, "name": n, "ingredients": list(ing)}
                for i, (n, ing) in enumerate(self._recipes[recipe_id]["versions"], start=1)]

    def get_version(self, recipe_id, version):
        history = self.get_recipe_history(recipe_id)
        if history is None or not 1 <= version <= len(history):
            return None
        return history[version - 1]

    def rollback_recipe(self, user_id, recipe_id, version):
        old = self.get_version(recipe_id, version)
        if old is None or not self._can_edit(user_id, recipe_id):
            return False
        return self._change(recipe_id, old["name"], old["ingredients"])
''',
    "interview_questions": interview(
        concept=[
            "Why does the name check need its own index keyed by casefold, and what does casefold handle that lower does not?",
            "Why must ingredient lists be copied when they come in and when they go out?",
        ],
        deep_dive=[
            "When a recipe is renamed, which entries of the name index must change, and what bug appears if the old one is left behind?",
        ],
        tradeoffs=[
            "How would you keep list_recipes and the searches fast if the book held millions of recipes?",
            "Why add new owner-aware methods instead of changing the signatures of add_recipe and delete_recipe?",
            "Why should rollback append a copy instead of truncating history back to the old version?",
            "How would you store versions more cheaply than a full copy of each one?",
        ],
    ),
}
