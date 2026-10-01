"""An in-memory table store queried by method calls, growing from projection to indexes."""

from ._interview import interview

_EMPLOYEES = r"""
ROWS = [
    {"id": 101, "name": "Ana", "dept": "eng", "salary": 95000},
    {"id": 102, "name": "Bo", "dept": "sales", "salary": 71000},
    {"id": 103, "name": "Cy", "dept": "eng", "salary": 88000},
    {"id": 104, "name": "Dee", "dept": "ops", "salary": 60000},
    {"id": 105, "name": "Eli", "dept": "eng", "salary": 88000},
    {"id": 106, "name": "Fo", "dept": "sales", "salary": 71000},
    {"id": 107, "name": "Gia", "dept": "ops", "salary": None},
    {"id": 108, "name": "Hu", "dept": "ops", "salary": None},
]

def employees(db):
    db.create_table("employees", ["id", "name", "dept", "salary"])
    for row in ROWS:
        db.insert("employees", row)
    return db

def raises(kind, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except kind:
        return True
    return False
"""

# A brute-force reading of the statement, used to check random queries.
_ORACLE = r"""
import random, functools

def rank(v):
    return (0, v) if isinstance(v, int) else (1, v)

def holds(a, op, b):
    if a is None or b is None:
        return False
    a, b = rank(a), rank(b)
    return {"=": a == b, "!=": a != b, "<": a < b, "<=": a <= b, ">": a > b, ">=": a >= b}[op]

def oracle_select(rows, columns, where, order_by):
    kept = [r for r in rows if all(holds(r[c], op, v) for c, op, v in (where or []))]
    def compare(x, y):
        for column, ascending in order_by or []:
            kx = (0,) if x[column] is None else (1,) + rank(x[column])
            ky = (0,) if y[column] is None else (1,) + rank(y[column])
            if kx != ky:
                return (-1 if kx < ky else 1) * (1 if ascending else -1)
        return 0
    kept = sorted(kept, key=functools.cmp_to_key(compare))
    return [{c: r[c] for c in columns} for r in kept]

def random_rows(rng, n):
    values = [None, 0, 1, 2, 5, 10, -3, "", "a", "b", "ab", "z", "0"]
    return [{"k": i, "x": rng.choice(values), "y": rng.choice(values), "s": rng.choice(values)} for i in range(n)]

def random_where(rng):
    ops = ["=", "!=", "<", "<=", ">", ">="]
    values = [None, 0, 1, 2, 5, "", "a", "ab", "0"]
    return [(rng.choice("xys"), rng.choice(ops), rng.choice(values)) for _ in range(rng.randint(0, 2))]

def random_order(rng):
    return [(c, rng.random() < 0.5) for c in rng.sample("xys", rng.randint(0, 3))]
"""

TASK = {
    "title": "In-Memory Database",
    "difficulty": "Hard",
    "version": 1,
    "function_name": "Database",
    "description_en": r"""Build `Database`, which keeps tables in memory and answers queries through method calls. There is no SQL text to parse.

The requirement arrives in parts. Each part keeps every earlier behavior, so one `Database` class passes all parts at the end. Pass every test of the current part to reveal the next one.

**Rules for every part:**
- A table has a fixed, ordered list of column names, set when it is created.
- A row is a `dict` with exactly the table's columns as keys. A value is an `int`, a `str`, or `None` for no value.
- Any method given a table or column that does not exist raises `KeyError`. This includes columns named in a query, even on an empty table.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why this shows up in interviews:** each part is small, but they interact. Keeping each concern in its own function is what lets a later part go in without rewriting an earlier one.

**Where it is used:** the query layer of every relational database, from SQLite to Postgres.

Adapted from the in-memory database question in Schuture/OpenAI-Interview-Notes (CC BY-NC 4.0), reworded.""",
    "parts": [
        {
            "title": "Tables, insertion, projection",
            "description_en": r"""**Signature:** `create_table(table, columns) -> None`, `insert(table, row) -> None`, `select(table, columns=None) -> list[dict]`

- `create_table` makes an empty table. It raises `KeyError` if the table already exists.
- `insert` appends a copy of `row`. It raises `KeyError` if `row` has a key that is not a column, checked first, then `ValueError` if a column is missing from `row`.
- `select` returns every row in insertion order, keeping only `columns`, or every column when `columns` is `None`. The rows returned are copies.

**Example:**
- `create_table("employees", ["id", "name", "dept", "salary"])`, then insert eight rows
- `select("employees", ["id", "name"])` returns `[{"id": 101, "name": "Ana"}, …]` in insertion order""",
        },
        {
            "title": "Filtering with WHERE",
            "description_en": r"""Keep Part 1 and add a `where` parameter:

**Signature:** `select(table, columns=None, where=None)`, where `where` is a list of `(column, operator, value)` that must all hold

- `operator` is one of `=`, `!=`, `<`, `<=`, `>`, `>=`. `None` or `[]` means no filtering.
- A condition is false whenever the row's value or `value` is `None`, for every operator, `!=` included.
- Values of one type use Python's own comparison. Across types, ints and strs are never equal, and any int ranks below any str.

**Example:**
- `where=[("dept", "=", "eng"), ("salary", ">=", 90000)]` keeps only Ana
- `where=[("salary", "<", 70000)]` keeps only Dee: rows with `None` salary never match""",
        },
        {
            "title": "Sorting with ORDER BY",
            "description_en": r"""Keep Parts 1–2 and add an `order_by` parameter:

**Signature:** `select(table, columns=None, where=None, order_by=None)`, where `order_by` is a list of `(column, ascending)`, most significant first

- Values compare as in Part 2, and `None` is smaller than every `int` and `str`. So `None` comes first in an ascending column and last in a descending one.
- Rows that tie on every `order_by` column keep their insertion order.
- `None` or `[]` keeps insertion order.

**Example:**
- `order_by=[("salary", True)]` gives Gia, Hu, Dee, Bo, Fo, Cy, Eli, Ana
- `order_by=[("dept", True), ("salary", False)]` gives Ana, Cy, Eli, Dee, Gia, Hu, Bo, Fo""",
        },
        {
            "title": "Indexes",
            "description_en": r"""Keep Parts 1–3 and add:

**Signature:** `create_index(table, column, kind) -> None`, with `kind` either `"hash"` or `"sorted"`

- `create_index` builds the index from the rows already in the table. Creating an index that exists rebuilds it. Every later `insert` keeps every index of the table up to date.
- A `"hash"` index serves `=` conditions. A `"sorted"` index serves `<`, `<=`, `>` and `>=`.
- `select` returns the same rows in the same order with or without an index.
- If any condition names an indexed column with an operator its index serves, `select` must not look at every row of the table, even if other conditions in the same `where` have no index. `!=` and unindexed conditions are checked row by row on the rows the index returned.""",
        },
    ],
    "hints": [
        {"level": 1, "kind": "questions", "content": "What does a table need besides its rows to check an insert? In what order must insert run its two checks? What does select hand back if the caller later edits the dicts it returned, or the dict it inserted?"},
        {"level": 2, "kind": "analysis", "content": "Store each table as its column list plus a list of row dicts. In insert, look for unknown keys before missing ones, then append a new dict with the keys in column order. In select, validate every requested column against the schema before touching rows, then build a fresh dict per row with only those keys, so no caller can reach the stored rows."},
    ],
    "model_connections": [
        "Relational engines such as SQLite plan each query by choosing between a full scan and an index, then apply the remaining predicates to the candidate rows, which is part 4.",
        "SQL defines comparisons with NULL as unknown, so NULL never satisfies a WHERE condition; part 2 adopts the same rule for None.",
    ],
    "pro_con_analysis": {
        "pros": [
            "A hash index turns an equality lookup from O(n) into O(1) plus the matching rows.",
            "A sorted index answers range conditions with two binary searches.",
            "Python's stable sort makes multi-column ORDER BY a sequence of single-column sorts.",
        ],
        "cons": [
            "Every index slows every insert and costs memory proportional to the table.",
            "Index candidates must be re-sorted into insertion order before ORDER BY, which costs O(k log k).",
            "Mixed int and str values need a type-aware comparison key everywhere values are compared.",
        ],
    },
    "tests": [
        {"name": "Part 1: tables, insertion, projection", "part": 1, "behavior": "contract.signature", "code": _EMPLOYEES + r"""
db = employees({fn}())
got = db.select("employees", ["id", "name"])
assert got == [{"id": r["id"], "name": r["name"]} for r in ROWS]
assert db.select("employees") == ROWS
assert list(db.select("employees")[0]) == ["id", "name", "dept", "salary"]
"""},
        {"name": "Part 1: missing tables and columns", "part": 1, "visibility": "unshown", "behavior": "protocol.validation",
         "failure_message": "Raise KeyError for an unknown table or column, a duplicate table, or an unknown row key (checked before missing ones, which raise ValueError).",
         "code": _EMPLOYEES + r"""
db = {fn}()
db.create_table("t", ["a", "b"])
assert raises(KeyError, db.create_table, "t", ["c"])
assert raises(KeyError, db.select, "nope")
assert raises(KeyError, db.insert, "nope", {"a": 1, "b": 2})
assert raises(KeyError, db.select, "t", ["a", "zz"]), "an unknown column must raise even on an empty table"
assert raises(KeyError, db.insert, "t", {"a": 1, "b": 2, "c": 3})
assert raises(KeyError, db.insert, "t", {"a": 1, "c": 3}), "an unknown key is checked before a missing one"
assert raises(ValueError, db.insert, "t", {"a": 1})
assert db.select("t") == []
"""},
        {"name": "Part 1: stored rows are private copies", "part": 1, "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "insert must store a copy and select must return copies, so no caller can change the table by editing a dict.",
         "code": r"""
db = {fn}()
db.create_table("t", ["a", "b"])
row = {"a": 1, "b": "x"}
db.insert("t", row)
row["a"] = 99
out = db.select("t")
out[0]["b"] = "changed"
assert db.select("t") == [{"a": 1, "b": "x"}]
"""},
        {"name": "Part 2: filtering with WHERE", "part": 2, "behavior": "state.invariant", "code": _EMPLOYEES + r"""
db = employees({fn}())
assert db.select("employees", ["name"], where=[("dept", "=", "eng"), ("salary", ">=", 90000)]) == [{"name": "Ana"}]
assert db.select("employees", ["name"], where=[("salary", "<", 70000)]) == [{"name": "Dee"}]
assert db.select("employees", ["name"], where=[]) == db.select("employees", ["name"])
"""},
        {"name": "Part 2: None and mixed types", "part": 2, "visibility": "unshown", "behavior": "edge.empty_or_boundary",
         "failure_message": "A condition with None on either side is false for every operator, an int never equals a str, and every int is smaller than every str.",
         "code": _EMPLOYEES + r"""
db = employees({fn}())
names = lambda where: [r["name"] for r in db.select("employees", ["name"], where=where)]
assert names([("salary", "!=", 60000)]) == ["Ana", "Bo", "Cy", "Eli", "Fo"]
assert names([("salary", "=", None)]) == []
assert names([("salary", "!=", None)]) == []
assert names([("salary", "<", "0")]) == ["Ana", "Bo", "Cy", "Dee", "Eli", "Fo"]
assert names([("id", "=", "101")]) == []
assert names([("id", "!=", "101")]) == [r["name"] for r in ROWS]
assert names([("name", ">", 5)]) == [r["name"] for r in ROWS]
assert raises(KeyError, db.select, "employees", where=[("bonus", "=", 1)])
db.create_table("empty", ["a"])
assert raises(KeyError, db.select, "empty", where=[("b", "=", 1)]), "an unknown where column must raise even on an empty table"
"""},
        {"name": "Part 2: random filters match a reference", "part": 2, "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "A random WHERE returned different rows than the comparison rules give.",
         "code": _ORACLE + r"""
for seed in range(150):
    rng = random.Random(seed)
    rows = random_rows(rng, 25)
    db = {fn}()
    db.create_table("t", ["k", "x", "y", "s"])
    for r in rows:
        db.insert("t", r)
    where = random_where(rng)
    assert db.select("t", ["k"], where=where) == oracle_select(rows, ["k"], where, []), (seed, where)
"""},
        {"name": "Part 3: sorting with ORDER BY", "part": 3, "behavior": "events.ordering", "code": _EMPLOYEES + r"""
db = employees({fn}())
names = lambda order: [r["name"] for r in db.select("employees", order_by=order)]
assert names([("salary", True)]) == ["Gia", "Hu", "Dee", "Bo", "Fo", "Cy", "Eli", "Ana"]
assert names([("dept", True), ("salary", False)]) == ["Ana", "Cy", "Eli", "Dee", "Gia", "Hu", "Bo", "Fo"]
assert names([]) == [r["name"] for r in ROWS]
"""},
        {"name": "Part 3: stable ties and None placement", "part": 3, "visibility": "unshown", "behavior": "events.ordering",
         "failure_message": "Ties must keep insertion order, None sorts below every value (first ascending, last descending), and ints sort below strs.",
         "code": _EMPLOYEES + r"""
db = {fn}()
db.create_table("t", ["k", "v"])
for k, v in enumerate([3, None, "b", 3, "a", None, -1, "b"]):
    db.insert("t", {"k": k, "v": v})
keys = lambda order: [r["k"] for r in db.select("t", ["k"], order_by=order)]
assert keys([("v", True)]) == [1, 5, 6, 0, 3, 4, 2, 7]
assert keys([("v", False)]) == [2, 7, 4, 0, 3, 6, 1, 5]
assert raises(KeyError, db.select, "t", order_by=[("w", True)])
db.create_table("empty", ["a"])
assert raises(KeyError, db.select, "empty", order_by=[("b", True)]), "an unknown order_by column must raise even on an empty table"
"""},
        {"name": "Part 3: random queries match a reference", "part": 3, "visibility": "unshown", "behavior": "events.ordering",
         "failure_message": "A random WHERE plus ORDER BY returned different rows or a different order than the rules give.",
         "code": _ORACLE + r"""
for seed in range(150):
    rng = random.Random(seed)
    rows = random_rows(rng, 25)
    db = {fn}()
    db.create_table("t", ["k", "x", "y", "s"])
    for r in rows:
        db.insert("t", r)
    where, order = random_where(rng), random_order(rng)
    assert db.select("t", ["k", "x"], where=where, order_by=order) == oracle_select(rows, ["k", "x"], where, order), (seed, where, order)
"""},
        {"name": "Part 4: an index changes nothing visible", "part": 4, "behavior": "state.invariant", "code": _EMPLOYEES + r"""
db = employees({fn}())
queries = [[("dept", "=", "eng")], [("salary", ">", 70000)], [("salary", "<=", 88000), ("dept", "!=", "ops")]]
before = [db.select("employees", ["name"], where=w) for w in queries]
db.create_index("employees", "dept", "hash")
db.create_index("employees", "salary", "sorted")
assert [db.select("employees", ["name"], where=w) for w in queries] == before
assert raises(KeyError, db.create_index, "nope", "dept", "hash")
assert raises(KeyError, db.create_index, "employees", "bonus", "sorted")
db.insert("employees", {"id": 109, "name": "Ivy", "dept": "eng", "salary": 70500})
assert db.select("employees", ["name"], where=[("dept", "=", "eng")]) == [{"name": n} for n in ("Ana", "Cy", "Eli", "Ivy")]
assert db.select("employees", ["name"], where=[("salary", ">", 70000), ("salary", "<", 71000)]) == [{"name": "Ivy"}]
"""},
        {"name": "Part 4: an index avoids a full scan", "part": 4, "visibility": "unshown", "behavior": "performance.complexity",
         "failure_message": "With a matching index, select still compared values in every row; read the candidate rows from the index instead.",
         "code": r"""
class Counted(str):
    compares = 0
    def _count(self):
        Counted.compares += 1
    def __eq__(self, other): self._count(); return str.__eq__(self, other)
    def __ne__(self, other): self._count(); return str.__ne__(self, other)
    def __lt__(self, other): self._count(); return str.__lt__(self, other)
    def __le__(self, other): self._count(); return str.__le__(self, other)
    def __gt__(self, other): self._count(); return str.__gt__(self, other)
    def __ge__(self, other): self._count(); return str.__ge__(self, other)
    __hash__ = str.__hash__

n = 2000
db = {fn}()
db.create_table("t", ["k", "tag", "code"])
for i in range(n):
    db.insert("t", {"k": i, "tag": Counted(f"t{i % 200:03d}"), "code": Counted(f"c{i:05d}")})
db.create_index("t", "tag", "hash")
db.create_index("t", "code", "sorted")
for where, expected in (
    ([("tag", "=", "t007")], [i for i in range(n) if i % 200 == 7]),
    ([("code", ">=", "c01990")], list(range(1990, n))),
    ([("tag", "=", "t007"), ("k", "!=", 7)], [i for i in range(n) if i % 200 == 7 and i != 7]),
):
    Counted.compares = 0
    got = [r["k"] for r in db.select("t", ["k"], where=where)]
    assert got == expected, where
    assert Counted.compares < n // 4, f"{where}: {Counted.compares} comparisons for {n} rows"
"""},
        {"name": "Part 4: both kinds of index on one column", "part": 4, "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "Building one kind of index must not touch another index on the same column; each row must still come back once.",
         "code": r"""
db = {fn}()
db.create_table("t", ["k", "x"])
for k, x in enumerate([3, 1, 2, None, 2]):
    db.insert("t", {"k": k, "x": x})
db.create_index("t", "x", "sorted")
db.create_index("t", "x", "hash")
db.insert("t", {"k": 5, "x": 2})
assert [r["k"] for r in db.select("t", ["k"], where=[("x", ">=", 2)])] == [0, 2, 4, 5]
assert [r["k"] for r in db.select("t", ["k"], where=[("x", "=", 2)])] == [2, 4, 5]
db.create_index("t", "x", "sorted")
assert [r["k"] for r in db.select("t", ["k"], where=[("x", "=", 2)])] == [2, 4, 5]
assert [r["k"] for r in db.select("t", ["k"], where=[("x", "<", 3)])] == [1, 2, 4, 5]
"""},
        {"name": "Part 4: random queries with indexes match a reference", "part": 4, "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "With indexes in place, a random query returned different rows or a different order than without them.",
         "code": _ORACLE + r"""
for seed in range(150):
    rng = random.Random(seed)
    rows = random_rows(rng, 30)
    db = {fn}()
    db.create_table("t", ["k", "x", "y", "s"])
    for r in rows[:15]:
        db.insert("t", r)
    for column in rng.sample("xys", 2):
        db.create_index("t", column, rng.choice(["hash", "sorted"]))
    if rng.random() < 0.5:
        db.create_index("t", "x", "sorted")
    for r in rows[15:]:
        db.insert("t", r)
    where, order = random_where(rng), random_order(rng)
    assert db.select("t", ["k"], where=where, order_by=order) == oracle_select(rows, ["k"], where, order), (seed, where, order)
"""},
    ],
    "solution": r'''import bisect

_OPS = {
    "=": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
}


def _rank(value):
    # Every int is smaller than every str, and an int never equals a str.
    return (0, value) if isinstance(value, int) else (1, value)


def _holds(a, op, b):
    return a is not None and b is not None and _OPS[op](_rank(a), _rank(b))


class _Table:
    def __init__(self, columns):
        self.columns = list(columns)
        self.rows = []
        self.hash_indexes = {}    # column -> {rank: [row ids]}
        self.sorted_indexes = {}  # column -> sorted [(rank, row id)]

    def check(self, column):
        if column not in self.columns:
            raise KeyError(column)

    def _add(self, row_id, column, kind):
        value = self.rows[row_id][column]
        if value is None:
            return
        if kind == "hash":
            self.hash_indexes[column].setdefault(_rank(value), []).append(row_id)
        else:
            bisect.insort(self.sorted_indexes[column], (_rank(value), row_id))

    def index_row(self, row_id):
        for column in self.hash_indexes:
            self._add(row_id, column, "hash")
        for column in self.sorted_indexes:
            self._add(row_id, column, "sorted")

    def build_index(self, column, kind):
        if kind == "hash":
            self.hash_indexes[column] = {}
        elif kind == "sorted":
            self.sorted_indexes[column] = []
        else:
            raise ValueError(kind)
        for row_id in range(len(self.rows)):
            self._add(row_id, column, kind)

    def candidates(self, where):
        """Row ids an index can narrow to, or None when a full scan is needed."""
        for column, op, value in where:
            if value is None:
                return []
            key = _rank(value)
            if op == "=" and column in self.hash_indexes:
                return list(self.hash_indexes[column].get(key, []))
            if op in ("<", "<=", ">", ">=") and column in self.sorted_indexes:
                entries = self.sorted_indexes[column]
                low = bisect.bisect_left(entries, (key,))
                high = bisect.bisect_right(entries, (key, float("inf")))
                chosen = {"<": entries[:low], "<=": entries[:high], ">": entries[high:], ">=": entries[low:]}[op]
                return sorted(row_id for _, row_id in chosen)
        return None


class Database:
    def __init__(self):
        self._tables = {}

    def _table(self, name):
        if name not in self._tables:
            raise KeyError(name)
        return self._tables[name]

    def create_table(self, table, columns):
        if table in self._tables:
            raise KeyError(table)
        self._tables[table] = _Table(columns)

    def insert(self, table, row):
        t = self._table(table)
        for key in row:
            t.check(key)
        missing = [c for c in t.columns if c not in row]
        if missing:
            raise ValueError(f"missing columns: {missing}")
        t.rows.append({c: row[c] for c in t.columns})
        t.index_row(len(t.rows) - 1)

    def create_index(self, table, column, kind):
        t = self._table(table)
        t.check(column)
        t.build_index(column, kind)

    def select(self, table, columns=None, where=None, order_by=None):
        t = self._table(table)
        columns = t.columns if columns is None else list(columns)
        where, order_by = list(where or []), list(order_by or [])
        for column in columns:
            t.check(column)
        for column, _, _ in where:
            t.check(column)
        for column, _ in order_by:
            t.check(column)
        ids = t.candidates(where)
        rows = t.rows if ids is None else [t.rows[i] for i in ids]
        kept = [r for r in rows if all(_holds(r[c], op, v) for c, op, v in where)]
        # Stable sorts from the least significant column up; None ranks below everything.
        for column, ascending in reversed(order_by):
            kept.sort(key=lambda r: (0,) if r[column] is None else (1,) + _rank(r[column]), reverse=not ascending)
        return [{c: r[c] for c in columns} for r in kept]
''',
    "interview_questions": interview(
        concept=[
            "What does a table store besides its rows, and which checks does insert run, in which order?",
            "Why should insert store a copy of the row and select return copies?",
        ],
        deep_dive=[
            "How does select validate the requested columns before it reads any row, and what happens on an empty table?",
        ],
        tradeoffs=[
            "How do you compare values so that None never matches and every int sorts below every str?",
            "How do you sort by several columns, each with its own direction, and keep ties in insertion order?",
            "When does a hash index beat a sorted index, and what does every index cost on insert?",
        ],
    ),
}
