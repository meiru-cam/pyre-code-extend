Implement a `Database` class that keeps its tables in memory and answers every query through a direct method call: there is no SQL text to parse anywhere in this problem, arguments are passed straight to Python methods instead. A *table* has a fixed, ordered list of column names, chosen once when it is created with `create_table`; every row inserted into it afterwards is a Python `dict` with exactly those keys, and every value in a row is a Python `int`, a `str`, or `None` (meaning "this row has no value in this column").

Every method below that takes a table name or a column name raises `KeyError` if that table, or that column of that table, does not exist; this includes a column named in `select`'s `columns`, `where` or `order_by`, even on an empty table. `create_table` also raises `KeyError` if a table with that name already exists. `insert` checks its `row` argument in this order: first that every key of `row` is one of the table's columns (`KeyError` if not), then that every column of the table is a key of `row` (`ValueError` if not) — so a row that is both missing a column and carries an unknown one raises `KeyError`, not `ValueError`.

### Part 1 — Tables, insertion, projection

```py
class Database:
    def create_table(self, table: str, columns: list[str]) -> None:
        """Creates an empty table with these columns, in this order."""

    def insert(self, table: str, row: dict[str, int | str | None]) -> None:
        """Appends a copy of row to table."""

    def select(self, table: str, columns: list[str] | None = None) -> list[dict[str, int | str | None]]:
        """Returns every row of table, projected to columns (all of the table's columns, in schema
        order, if columns is None), as a list of dicts, in the order the rows were inserted."""
```

For example:

```text
db = Database()
db.create_table("employees", ["id", "name", "dept", "salary"])
for row in [
    {"id": 101, "name": "Ana", "dept": "eng", "salary": 95000},
    {"id": 102, "name": "Bo", "dept": "sales", "salary": 71000},
    {"id": 103, "name": "Cy", "dept": "eng", "salary": 88000},
    {"id": 104, "name": "Dee", "dept": "ops", "salary": 60000},
    {"id": 105, "name": "Eli", "dept": "eng", "salary": 88000},
    {"id": 106, "name": "Fo", "dept": "sales", "salary": 71000},
    {"id": 107, "name": "Gia", "dept": "ops", "salary": None},
    {"id": 108, "name": "Hu", "dept": "ops", "salary": None},
]:
    db.insert("employees", row)

db.select("employees", ["id", "name"])
# [{"id": 101, "name": "Ana"}, {"id": 102, "name": "Bo"}, {"id": 103, "name": "Cy"}, {"id": 104, "name": "Dee"},
#  {"id": 105, "name": "Eli"}, {"id": 106, "name": "Fo"}, {"id": 107, "name": "Gia"}, {"id": 108, "name": "Hu"}]
```

### Part 2 — Filtering with WHERE

`select` grows a `where` parameter: a list of `(column, operator, value)` conditions, all of which must hold (combined with AND). `operator` is one of `"="`, `"!="`, `"<"`, `"<="`, `">"`, `">="`. A condition is false whenever the row's value in `column`, or `value` itself, is `None` — this holds for every operator, including `"="`, so a row with no value in a column never matches a condition on that column, not even `("salary", "!=", 60000)`. When neither side is `None`, two values of the same type compare as Python compares them, while an `int` never equals a `str` and every `int` is smaller than every `str` (so `("salary", "<", "0")` holds for every row whose salary is an `int`).

```py
def select(self, table: str, columns: list[str] | None = None,
           where: list[tuple[str, str, int | str | None]] | None = None) -> list[dict[str, int | str | None]]:
    """Same as Part 1, but keeps only the rows for which every (column, operator, value) condition
    of where holds; where=None (or []) means no filtering."""
```

Continuing the table of Part 1:

```text
db.select("employees", ["name"], where=[("dept", "=", "eng"), ("salary", ">=", 90000)])
# [{"name": "Ana"}]

db.select("employees", ["name"], where=[("salary", "<", 70000)])
# [{"name": "Dee"}]   -- Gia's and Hu's salary is None, so "<" never matches for them
```

### Part 3 — Sorting with ORDER BY

`select` grows an `order_by` parameter: a list of `(column, ascending)` pairs, the most significant column first. Rows that tie on every `order_by` column keep their relative insertion order. Within one column, values are ordered as in Part 2 and `None` counts as smaller than every `int` and every `str`, so combined with that column's direction, `None` values come first when it is ascending and last when it is descending.

```py
def select(self, table, columns=None, where=None,
           order_by: list[tuple[str, bool]] | None = None) -> list[dict[str, int | str | None]]:
    """Same as Part 2, but sorts the returned rows by order_by; order_by=None (or []) keeps
    insertion order."""
```

Sorting by a single column:

```text
[r["name"] for r in db.select("employees", order_by=[("salary", True)])]
# ["Gia", "Hu", "Dee", "Bo", "Fo", "Cy", "Eli", "Ana"]     -- None first, ascending
```

Sorting by department ascending, and within each department by salary descending:

```text
[r["name"] for r in db.select("employees", order_by=[("dept", True), ("salary", False)])]
# ["Ana", "Cy", "Eli", "Dee", "Gia", "Hu", "Bo", "Fo"]
# dept "eng": 95000, 88000, 88000 (Cy before Eli: insertion order)
# dept "ops": 60000, then None, None (Gia before Hu; None sorts last because the column is descending)
# dept "sales": 71000, 71000 (Bo before Fo: insertion order)
```

### Part 4 — Indexes

`create_index(table, column, kind)` builds an index on `column`, from the rows already in `table`; every `insert` afterwards keeps every index of that table up to date. `kind` is `"hash"`, which serves `"="` conditions, or `"sorted"`, which serves `"<"`, `"<="`, `">"` and `">="` conditions. `select` returns the same rows, in the same order, whether or not a matching index exists — only how much work it does can change: if at least one condition of `where` names an indexed column with a matching operator, `select` must not scan every row of the table, even when the other conditions of the same `where` are not backed by an index. `"!="`, and a condition on a column without a matching index, are always checked row by row.

```py
def create_index(self, table: str, column: str, kind: str) -> None:
    """kind is "hash" or "sorted". Rebuilds the index if one already exists for this (column, kind)."""
```

```text
db.create_index("employees", "dept", "hash")
db.create_index("employees", "salary", "sorted")

db.select("employees", ["name"], where=[("dept", "=", "eng")])
# [{"name": "Ana"}, {"name": "Cy"}, {"name": "Eli"}]     -- same rows as before create_index
```
