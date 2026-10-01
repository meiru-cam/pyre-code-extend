A *cell name* is one or more uppercase letters `A`–`Z` (the column) followed by one or more digits `0`–`9` (the row) and nothing else, such as `A1` or `AB12`; `a1`, `A` and `A1B` are not cell names. A cell holds either a Python `int` or a *formula*: a string made of `=` followed by one or more *terms*, with exactly one `+` or `-` between consecutive terms, for example `=R1 + 20 - S3` or `=5`. A term is a cell name or a non-negative integer literal of at most 18 digits. Spaces may appear before or after any term or operator, but not inside a term. Nothing else is allowed: no `*`, `/` or parentheses, and no operator before the first term or after the last, so `=-R1`, `=R1 - -S3` and `=R1 2` are all invalid. Reading a cell name that was never passed to `set`, directly or through a formula, gives `0`. Both `set` and `get` raise `FormulaError` when given a name that is not a cell name; `set` also raises it for a value that is neither an `int` nor a valid formula.

### Part 1 — Evaluate on demand

Implement `LazySpreadsheet`. `set(name, value)` stores `value` — an `int`, or a formula string — for `name`, replacing whatever `name` held before. `get(name)` returns the current integer value of `name`: the literal itself, or the formula evaluated by substituting, for every cell it references, that cell's own current value, recursively. Cells form a *circular dependency* when a cell's formula reads that same cell, directly or through other cells' formulas. `set` stores a formula even if it creates one; `get(name)` must raise `CycleError` if evaluating `name` needs the value of a cell on a circular dependency, and return normally otherwise. You may assume that following references from any cell, without visiting a cell twice, never passes through more than 200 cells.

```py
class LazySpreadsheet:
    def set(self, name: str, value: int | str) -> None:
        """Stores value for name. Raises FormulaError if name or value is not of the shape above."""

    def get(self, name: str) -> int:
        """Evaluates name, recursively reading the cells its formula references.
        Raises CycleError if that needs a cell on a circular dependency."""
```

For example, after setting `P4 = 6`, `Q2 = 17`, `M3 = "=Q2 - P4"` and `M8 = "=M3 - P4 + 30"`, `get("M8")` is `35`. Setting `P4 = 14` afterwards and reading `M8` again gives `19`, since `M3` is now `3`. Setting `U1 = "=U2"` and then `U2 = "=U1"` both succeed; `get("U1")` then raises `CycleError`, and so does `get` on any cell whose formula reads `U1`.

### Part 2 — Update eagerly

Implement `Spreadsheet` with the same `set`/`get` contract, except that `set` keeps every cell's value up to date immediately, so that `get` only ever reads an already-computed value and runs in $O(1)$. Unlike `LazySpreadsheet.set`, if the new `value` would create a circular dependency, `set` must reject it and raise `CycleError` instead of storing anything, leaving the cell's previous content, the dependency graph and every cached value exactly as they were.

```py
class Spreadsheet:
    def set(self, name: str, value: int | str) -> None:
        """Same contract as LazySpreadsheet.set, but recomputes every affected cell immediately, and
        raises CycleError (without changing anything) if value would create a circular dependency."""

    def get(self, name: str) -> int:
        """O(1): returns the already-computed value of name."""
```

With the same sequence of `set` calls as the Part 1 example, `get("M8")` again returns `35`, then `19` after `P4` is updated — but now every `get` is a plain cache lookup, because `set` already recomputed `M8` (and everything between it and `P4`) when `P4` changed. Calling `set("U1", "=U2")` and then `set("U2", "=U1")` raises `CycleError` on the second call, and neither `U1`, `U2`, nor any other cell is affected by the rejected call.

### Part 3 — Edge cases

Implement and pass tests for the following behaviors of `Spreadsheet` (and, where noted, of `LazySpreadsheet`):

- Overwriting a cell that held a formula, with a new formula or a literal, drops its old dependency edges: a cell it used to read no longer triggers its recomputation.
- A direct self-reference (`set("H2", "=H2")`) and a cycle that runs through several cells are both rejected by `Spreadsheet.set`; the same two cases are only caught by `LazySpreadsheet.get`.
- A formula may reference a cell that has not been `set` yet (it reads as `0`); setting that cell afterwards updates every cell that transitively depends on it.
- In a "diamond" — two cells that both read a common cell, and a fourth cell that reads both of them — a single `set` on the common cell recomputes the fourth cell exactly once, not once per path to it.
- Overwriting a cell that held a literal with a formula adds its new dependency edges immediately: the cell's own cached value and every cell that transitively reads it reflect the formula from that same `set` call, not a stale literal.
