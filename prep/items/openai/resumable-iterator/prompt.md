A *resumable iterator* produces items one at a time like an ordinary Python iterator, but can also export its position with `get_state()` and later use `set_state(state)` to continue from that position, exactly as if it had never stopped. Implement the following four parts.

Every iterator below walks data nested $d$ levels deep ($d = 1$ for a plain list): what $d$ indexing steps reach, $\text{items}[i_1][i_2] \cdots [i_d]$, is an item, and items are produced in lexicographic order of their *index tuples* $(i_1, \ldots, i_d)$ — for a matrix, row by row. States are such tuples:

- `get_state()` returns a `tuple` of $d$ non-negative `int`s: the index tuple of the item the next `__next__` call will produce. Once no item remains (the iterator is *exhausted*), it returns $(\text{len}(\text{items}), 0, \ldots, 0)$ instead.
- A tuple is *canonical* for given data if it is that exhausted tuple, or if at every level $k$, $0 \le i_k < \text{len}(L_k)$, where $L_k$ is the list reached by indexing $i_1, \ldots, i_{k-1}$ ($L_1$ is `items`). These are exactly the tuples `get_state()` can return. The second condition rules out pointing into an empty list and pointing at the end of a non-empty one: in both cases the position has already moved on to the next item. For example, with rows `[[5, 7], [2]]` the position after 7 is `(1, 0)`, so `(0, 2)` is not canonical; with rows `[[], [4]]`, `(0, 0)` is not canonical because row 0 is empty.

`set_state(state)` checks, in this order: `TypeError` if `state` is not a tuple or a list, its length is not $d$, or an element is not an `int` (a `bool` does not count as an `int`); otherwise `ValueError` if `state` is not canonical for this iterator's own data. A rejected state leaves the iterator where it was. Otherwise the next `__next__` call produces the item the tuple points to, or raises `StopIteration` for the exhausted tuple. A list is accepted like a tuple, because a state that goes through `json.dumps` / `json.loads` comes back as a list. `set_state` may move backward or forward, including after `StopIteration` has been raised.

Because `set_state` judges a state only against the receiving iterator's own data, a state saved on one instance can be given to another. On data of the same *shape* — every corresponding list, at every level, has the same length; the items may differ — iteration continues exactly as it would have on the original. On data of another shape, the state is accepted only if it is canonical there, and iteration then continues from the item it points to in that data. Modifying a list after constructing an iterator over it is outside this contract.

### Part 1 — Abstract interface

Define `ResumableIterator`, an abstract base class that every iterator below inherits from, declaring `__iter__`, `__next__`, `get_state`, and `set_state` as abstract methods with the contract above.

```py
class ResumableIterator(ABC):
    @abstractmethod
    def __iter__(self) -> "ResumableIterator": ...

    @abstractmethod
    def __next__(self):
        """Returns the next item. Raises StopIteration once none remain."""

    @abstractmethod
    def get_state(self) -> tuple:
        """Returns a copyable, JSON-serializable tuple describing the current position."""

    @abstractmethod
    def set_state(self, state) -> None:
        """Moves to the position state describes. Raises TypeError or ValueError as specified above."""
```

### Part 2 — 1D list

`items` is a plain Python list of any length, possibly empty. Implement `ResumableListIterator(items)`; it produces `items[0], items[1], ...` in order.

```py
class ResumableListIterator(ResumableIterator):
    def __init__(self, items: list) -> None:
        """items may be empty."""
```

Example:

```text
items = [68, 71, 74, 70, 73, 69]

next -> 68
next -> 71
next -> 74
state = get_state()      # (3,)
next -> 70
set_state(state)
next -> 70                # produced again
next -> 73
next -> 69
next -> StopIteration
```

### Part 3 — 2D list (matrix)

`items` is a list of lists (*rows*); any row may be empty, including the first, the last, or all of them. Implement `Resumable2DIterator(items)`; it produces the elements of `items[0]`, then `items[1]`, and so on, skipping every empty row.

```py
class Resumable2DIterator(ResumableIterator):
    def __init__(self, items: list) -> None:
        """items is a list of lists; a row may be empty."""
```

Example:

```text
items = [[7, 2], [], [9], [], [3, 5, 1]]

next -> 7
next -> 2
state = get_state()      # (2, 0) -- already past the empty row 1, pointing at the 9 in row 2
next -> 9
set_state(state)
next -> 9                 # produced again
next -> 3
next -> 5
next -> 1
next -> StopIteration
get_state()               # (5, 0) -- exhausted
```

`Resumable2DIterator([[], []])` produces nothing; its `get_state()` is `(2, 0)` from the start.

### Part 4 — nested lists of arbitrary depth

`items` is nested `depth` levels deep ($\text{depth} = d \ge 1$): whatever `depth` indexing steps reach is an item, even if it is itself a list. A list at any level may be empty. Implement `ResumableNestedIterator(items, depth)`; with `depth=1` or `depth=2` it behaves exactly like Part 2's or Part 3's iterator.

```py
class ResumableNestedIterator(ResumableIterator):
    def __init__(self, items: list, depth: int) -> None:
        """items is nested depth levels deep, depth >= 1; a list at any level may be empty."""
```

Example with `depth=3`:

```text
items = [
    [[5, 3], []],
    [],
    [[], [9, 1]],
    [[6]],
]

next -> 5
next -> 3
state = get_state()      # (2, 1, 0) -- skipped items[0][1], items[1] and items[2][0], all empty
next -> 9
set_state(state)
next -> 9                 # produced again
next -> 1
next -> 6
next -> StopIteration
```
