`grid` is a list of $R$ rows, each a list of $C$ integers. `grid[r][c]` is the state of the cell in row $r$ and column $c$: `0` means *healthy* and `1` means *infected*. The *neighbours* of a cell are the cells directly above, below, left and right of it that lie inside the grid. Diagonal cells are not neighbours.

Time advances in whole days, and day 0 is the input grid. The grid of day $t + 1$ is computed from the grid of day $t$ by applying the rule to all cells at the same time (*synchronous update*). A cell that becomes infected on day $t + 1$ therefore starts infecting its neighbours in the step from day $t + 1$ to day $t + 2$.

The problem has five parts, and each part adds a rule to the previous one.

### Part 1 — Spread

A healthy cell becomes infected on day $t + 1$ if at least one of its neighbours is infected on day $t$. Infected cells stay infected. Return the smallest $t$ such that no cell is healthy on day $t$, or `-1` if there is no such day. If no cell is healthy on day 0, which includes the empty grid, return `0`.

```py
def days_until_all_infected(grid: list[list[int]]) -> int:
    """grid[r][c] is 0 (healthy) or 1 (infected); from Part 2 on also 2 (immune). Returns the day, or -1."""
```

Example: two cells are infected on day 0, every cell is infected on day 2, and the answer is `2`.

```text
day 0        day 1        day 2
1 0 0 0      1 1 0 0      1 1 1 1
0 0 0 0      1 0 0 1      1 1 1 1
0 0 0 1      0 0 1 1      1 1 1 1
```

### Part 2 — Immune cells

The grid may also contain `2`, which means *immune*. An immune cell never changes state and does not count as an infected neighbour, so the infection cannot pass through it. The task is the same as in Part 1. Return `-1` if some healthy cell can never be infected.

In the left grid the infection has to go around the immune cells and reaches the top-right cell on day
- In the right grid the right-hand column is cut off, and the answer is `-1`.

```text
1 2 0                     1 2 0
0 2 0      -> 6           2 2 0      -> -1
0 0 0
```

### Part 3 — Recovery after D days

An integer $D \ge 1$ is given. A cell that has been infected for $D$ days recovers: it becomes immune and stops infecting others. For every infected cell let $t_0$ be the day on which it became infected, with $t_0 = 0$ for the cells that are infected in the input. The step from day $t - 1$ to day $t$ has two phases, in this order:

- Recovery: every infected cell with $t - t_0 \ge D$ becomes immune.
- Spread: every healthy cell with at least one neighbour that is still infected after phase 1 becomes infected, with $t_0 = t$.

Return the smallest $t$ such that no cell is infected on day $t$ (`0` if no cell is infected on day 0). Healthy cells may remain.

```py
def days_until_outbreak_ends(grid: list[list[int]], D: int) -> int:
    """grid holds 0, 1 or 2, and D >= 1. Returns the first day on which no cell is infected."""
```

Example: the single row `1 0 0` with $D = 2$, cells numbered 0, 1, 2 from the left. The answer is `4`. With $D = 1$ cell 0 recovers on day 1 before it infects anyone, and the answer is `1`.

```text
day 0:  1 0 0
day 1:  1 1 0    cell 0: 1 - 0 < 2, no recovery; it infects cell 1 (t0 = 1)
day 2:  2 1 1    cell 0: 2 - 0 >= 2, recovers; cell 1 infects cell 2 (t0 = 2)
day 3:  2 2 1    cell 1: 3 - 1 >= 2, recovers; cell 2 has no healthy neighbour
day 4:  2 2 2    cell 2: 4 - 2 >= 2, recovers; no cell is infected
```

### Part 4 — Thresholds and deaths

This part has three versions.

**Version A, spread threshold.** An integer $K \ge 1$ is given. In phase 2 a healthy cell becomes infected only if at least $K$ of its neighbours are infected. $K = 1$ is Part 3. The return value is the same as in Part 3. Example with $K = 2$ and $D = 2$: the outbreak ends on day 4, and three cells are never infected because they never have two infected neighbours on the same day.

```text
day 0      day 1      day 2      day 3      day 4
1 0 1      1 1 1      2 1 2      2 2 2      2 2 2
0 0 0      1 0 0      1 1 0      2 1 0      2 2 0
1 0 0      1 0 0      2 0 0      2 0 0      2 0 0
```

**Version B, deaths.** A new state `3` means *dead*. A dead cell never changes state and never infects others. A healthy cell becomes infected as in Part 3. If it has at least $K$ infected neighbours at the moment it becomes infected, then at the end of its $D$ days it dies instead of becoming immune. This $K$ is a threshold of its own and need not agree with the one of version A. The count is taken on that day only: an infected cell is never looked at again, so neighbours infected later neither doom it nor restart its $D$ days. Return the day on which the outbreak ends and the number of dead cells.

**Version C.** Immune cells, recovery, a spread threshold and deaths combined in one simulation.

```py
def simulate(grid: list[list[int]], recover_after: int, spread_threshold: int = 1,
             death_threshold: int | None = None) -> tuple[int, int]:
    """recover_after is D. spread_threshold is the K of version A. death_threshold is the K of
    version B, or None when nobody dies. Returns (day on which the outbreak ends, number of dead cells)."""
```

### Part 5 — Intervention

On day 0, before the first step, you may burn one whole row, one whole column, or nothing at all. Every cell of the burnt line becomes dead whatever its state was: it is never infected, it never infects, and it counts as a death. The rules of Part 4 version C then run to the end. Return the smallest total number of deaths that can be reached.

```py
def min_deaths(grid: list[list[int]], recover_after: int, death_threshold: int,
               spread_threshold: int = 1) -> int:
    """Returns the smallest number of dead cells over the R + C + 1 choices."""
```

Example with $D = 2$ and both thresholds 1, columns numbered 0 to 3 from the left: burning nothing costs 9 deaths, burning column 3 costs 11, burning column 2 costs 4, and the answer is `4`.

```text
0 0 0 1
0 0 0 1
0 0 1 0
```

Everything else is yours to fix and to state: whether the burn may be delayed to a later day, whether more than one line may be burnt, and whether a part of a line may be burnt.
