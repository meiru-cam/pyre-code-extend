`board` is a grid of $N$ rows and $M$ columns holding integers (negative values allowed), indexed by row `i` and column `j`, both starting at 0. A *path* starts at `(0, p)` for a given starting column `p`, and from `(i, j)` it either takes a *step* to `(i+1, j-1)`, `(i+1, j)`, or `(i+1, j+1)` — whichever of these has a column inside `[0, M)` — or, using one of a budget of at most `K` uses over the whole path, takes the *special move* to `(i+2, j)`. Any move whose destination falls outside the grid is illegal — in particular, the special move can never be taken from row `N - 2`, since that would put `i + 2` at `N`, past the last row. The path stops the first time it reaches row `N - 1`. When `N = 1` that is the starting cell itself, and the path takes no move at all. Write $v_0, v_1, \dots, v_L$ for the values of the cells visited, in the order they are visited ($v_0 = board[0][p]$, $v_L$ is the value at row `N - 1`). The *base score* of a path is $\sum_{t=0}^{L} v_t$.

$N$ and $M$ are each at most $200$, $K$ satisfies $0 \le K \le N$, and every cell of `board` holds an integer with absolute value at most $10^9$.

Implement the following four parts.

### Part 1 — Maximum score

Return the largest base score over all paths from `(0, p)` to row `N - 1`.

```py
def max_score(board: list[list[int]], p: int, K: int) -> int:
    """1 <= N, M <= 200, |board[i][j]| <= 10**9, 0 <= p < M, 0 <= K <= N. Returns the largest base score."""
```

Example: with

```text
board = [[-3,  1,  4],
         [-1,  4,  4],
         [-3, -2, -3],
         [ 1,  0, -1],
         [ 5,  5,  3]]
```

`p = 0`, `K = 1`: `max_score(board, 0, 1) == 6`, attained e.g. by `(0,0) -> (1,1) -> (3,1) -> (4,0)` (values `-3, 4, 0, 5`), which uses the special move once, from row 1 to row 3.

### Part 2 — Recovering a path

Return one path attaining the maximum base score, as the list of cells it visits, `[(0, p), ..., (N - 1, ·)]`. When several paths attain the maximum, compare their visited-cell lists with the usual lexicographic order — compare the first cell as a `(row, column)` pair, then the second, and so on — and return the smallest one.

```py
def optimal_path(board: list[list[int]], p: int, K: int) -> list[tuple[int, int]]:
    """Same input as max_score. Returns the lexicographically smallest cell list among the
    paths that attain the maximum base score."""
```

Example: on the `board` above with `p = 0`, `K = 1`, two paths attain the maximum score 6: `(0,0) -> (1,1) -> (3,1) -> (4,0)` and `(0,0) -> (1,1) -> (3,1) -> (4,1)`. They agree on every cell except the last, where `(4,0) < (4,1)`, so `optimal_path` returns the first one.

### Part 3 — Counting maximum-score paths

Two paths are *different* if their visited-cell lists differ. In particular, a path that uses the special move from `(i,j)` to `(i+2,j)` is different from one that steps through `(i+1,j)` on the way from `(i,j)` to `(i+2,j)`, even though the two agree on every cell before and after that segment. Return the number of different paths attaining the maximum base score, modulo `10^9 + 7`.

```py
def count_optimal_paths(board: list[list[int]], p: int, K: int) -> int:
    """Same input as max_score. Returns the number of distinct maximum-base-score paths,
    modulo 10**9 + 7."""
```

Example: with

```text
board = [[ 2,  5],
         [ 0, -9],
         [ 3,  1]]
```

`p = 0`, `K = 1`: the maximum base score is `5`, attained by exactly two paths — `(0,0) -> (1,0) -> (2,0)` (values `2, 0, 3`) and `(0,0) -> (2,0)` (values `2, 3`, using the special move) — so `count_optimal_paths(board, 0, 1) == 2`.

### Part 4 — Bonuses

Two bonus rules apply on top of the base score, both stated in terms of the order cells are *visited* — a special move still makes its two endpoints visited consecutively, exactly like a step. Given integers `X` and `Y`, each between `0` and `10^9`:

- For every `t` with `1 <= t <= L` such that $v_{t-1} = v_t$, add `X`.
- For every `t` with `2 <= t <= L` such that $v_{t-2} < v_{t-1} < v_t$, add `Y`.

Return the largest score (base score plus both bonuses) over all paths from `(0, p)` to row `N - 1`.

```py
def max_score_with_bonuses(board: list[list[int]], p: int, K: int, X: int, Y: int) -> int:
    """Same input as max_score, plus bonus amounts 0 <= X, Y <= 10**9. Returns the largest
    base score plus bonuses over all paths."""
```

Example: with

```text
board = [[ 3,  1],
         [-2,  5],
         [ 1,  2],
         [ 6,  6],
         [ 5,  6]]
```

`p = 0`, `K = 1`, `X = 3`, `Y = 6`: without bonuses the maximum score is `22`. With bonuses, `max_score_with_bonuses(board, 0, 1, 3, 6) == 29`, attained only by `(0,0) -> (1,1) -> (3,1) -> (4,1)` (values `3, 5, 6, 6`, using the special move from row 1 to row 3): the last two values are equal (`+X`), and the first three, `3 < 5 < 6`, are strictly increasing even though the special move puts two rows between the second and third (`+Y`).
