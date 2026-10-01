Confirm with the interviewer before coding: four neighbours or eight; how cells are encoded (integers or characters, immune as `2` or `-1`); in Part 3, whether recovery comes before the spread of the same day and whether the test is $\ge$ or $>$; in Part 4, whether $K$ is a threshold for infection, for death, or for both. With five parts in 60 minutes, Parts 1 and 2 should be written quickly and kept simple.

### Parts 1–2: multi-source BFS

Start a breadth-first search from all infected cells at once, that is, put all of them into the queue with distance 0. The BFS distance of a cell is then its distance to the nearest infected cell, which is exactly the day on which it becomes infected. One BFS level is one day, and the answer is the number of the last level. Keep a count of the healthy cells not yet reached: stop when it is 0, and return `-1` if the queue runs empty first. Immune cells are never entered, which is all that Part 2 adds. Every cell enters the queue at most once, so time and memory are $O(R \cdot C)$. Recomputing the whole grid for every day is easier to write but costs $O((R \cdot C)^2)$ in the worst case, because a corridor of immune cells can make the infection take on the order of $R \cdot C$ days.

```python
from collections import Counter, deque

HEALTHY, INFECTED, IMMUNE, DEAD = 0, 1, 2, 3
STEPS = ((1, 0), (-1, 0), (0, 1), (0, -1))      # NOTE: add the four diagonals for the 8-neighbour variant


def days_until_all_infected(grid):
    if not grid or not grid[0]:
        return 0
    rows, cols = len(grid), len(grid[0])
    reached = [[cell != HEALTHY for cell in row] for row in grid]      # NOTE: the input grid is not modified
    frontier = deque((r, c) for r in range(rows) for c in range(cols) if grid[r][c] == INFECTED)
    healthy = sum(cell == HEALTHY for row in grid for cell in row)

    days = 0
    while frontier and healthy:
        days += 1
        for _ in range(len(frontier)):          # NOTE: exactly one BFS level per day
            r, c = frontier.popleft()
            for dr, dc in STEPS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and not reached[nr][nc]:
                    reached[nr][nc] = True      # immune cells start as reached, so they are never entered
                    healthy -= 1
                    frontier.append((nr, nc))
    return days if healthy == 0 else -1         # NOTE: healthy cells left over are walled off -> -1
```

### Parts 3–4: day-by-day simulation

With timers, BFS levels are no longer enough. Store $t_0$ for every infected cell and simulate the two phases of the statement day by day. Each cell is infected for at most $D$ days and costs constant work on each of them, so the simulation takes $O(R \cdot C \cdot D)$. Part 4 uses the same loop with two thresholds as parameters.

For Part 3 alone there is a closed form. With threshold 1 and $D \ge 2$, a cell infects all its healthy neighbours on the day after it was infected, so the infection advances exactly as in Part 2. If $d_{\max}$ is the largest BFS distance among the cells that are reached, the last cell is infected on day $d_{\max}$ and recovers on day $d_{\max} + D$, which is the answer. For $D = 1$ the answer is 1, because the initial cells recover before they spread. With a threshold $K > 1$ the closed form fails, since a cell may collect its $K$ infected neighbours over several days.

```python
def simulate(grid, recover_after, spread_threshold=1, death_threshold=None):
    """Returns (day on which the outbreak ends, number of dead cells)."""
    if not grid or not grid[0]:
        return 0, 0
    rows, cols = len(grid), len(grid[0])
    state = [row[:] for row in grid]
    # NOTE: the cells infected in the input carry a timer too, with t0 = 0; they recover on day D
    infected_on = {(r, c): 0 for r in range(rows) for c in range(cols) if state[r][c] == INFECTED}
    doomed = set()
    day = deaths = 0

    while infected_on:
        day += 1
        # Phase 1: recovery.  NOTE: it comes BEFORE the spread of the same day, and the test is >=, not >
        for cell in [cell for cell, since in infected_on.items() if day - since >= recover_after]:
            del infected_on[cell]
            r, c = cell
            state[r][c] = DEAD if cell in doomed else IMMUNE
            deaths += cell in doomed

        # Phase 2: spread.  NOTE: count first, apply afterwards. Writing into `state` while counting
        # would let a cell infected today infect its neighbours today.
        pressure = Counter()                    # healthy cell -> number of infected neighbours
        for r, c in infected_on:                # NOTE: only infected cells count; immune and dead ones do not
            for dr, dc in STEPS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and state[nr][nc] == HEALTHY:
                    pressure[(nr, nc)] += 1
        for (r, c), count in pressure.items():
            if count >= spread_threshold:
                state[r][c] = INFECTED
                infected_on[(r, c)] = day
                if death_threshold is not None and count >= death_threshold:
                    doomed.add((r, c))          # decided now, carried out when the D days are over
    return day, deaths


def days_until_outbreak_ends(grid, D):          # Part 3 is the engine with threshold 1 and no deaths
    return simulate(grid, D)[0]
```

### Part 5

The intervention is chosen once, so the whole decision space is $R + C + 1$ options: burn nothing, or burn one of the $R$ rows or $C$ columns. Each option is one run of the Part 4 engine on a copy of the grid with that line set to `3`, and the engine needs no change, because a dead cell is neither healthy nor infected and so already plays no part. The minimum over the options is therefore exact, at $O((R + C) \cdot R \cdot C \cdot D)$. A line costs its whole length in deaths the moment it is burnt, so it pays only when it seals the infection into a small part of the grid. Burning the line that holds the most infected cells is the wrong rule: it spends that length where the infection already is instead of where it is going, and on the grid of the statement that line is column 3, which ends up worse than burning nothing. If the burn may also be delayed or repeated, the options multiply by $R + C + 1$ per day and the search has to be narrowed to a few candidate lines per day.

```python
def min_deaths(grid, recover_after, death_threshold, spread_threshold=1):
    """Part 5: the best of burning nothing and burning one whole row or column on day 0."""
    if not grid or not grid[0]:
        return 0
    rows, cols = len(grid), len(grid[0])
    lines = [[(r, c) for c in range(cols)] for r in range(rows)]
    lines += [[(r, c) for r in range(rows)] for c in range(cols)]
    best = simulate(grid, recover_after, spread_threshold, death_threshold)[1]   # burning nothing
    for line in lines:
        burnt = [row[:] for row in grid]
        for r, c in line:
            burnt[r][c] = DEAD                  # NOTE: a burnt cell counts as a death whatever it was
        rest = simulate(burnt, recover_after, spread_threshold, death_threshold)[1]
        best = min(best, len(line) + rest)      # NOTE: the line itself already costs len(line) deaths
    return best
```

### Follow-ups

- The two thresholds of version B are independent. A death threshold at or below the spread threshold kills every cell infected after day 0; one above 4 kills nobody, since a cell has at most four neighbours.
- Characters instead of integers (`.`, `X`, `I`): convert on input and keep the core numeric.
- Very large or sparse grids: store only the set of infected cells and the frontier, or split the grid into tiles that exchange their borders after every day.
- Counting infected neighbours for all cells at once is a 2D convolution with a plus-shaped kernel.

```python
a = [[1, 0, 0, 0],
     [0, 0, 0, 0],
     [0, 0, 0, 1]]
b = [[1, 2, 0],
     [0, 2, 0],
     [0, 0, 0]]
d = [[1, 0, 1],
     [0, 0, 0],
     [1, 0, 0]]
e = [[0, 0, 0, 1],
     [0, 0, 0, 1],
     [0, 0, 1, 0]]

# Parts 1-2: the examples of the statement and the edge cases
assert days_until_all_infected(a) == 2
assert days_until_all_infected(b) == 6
assert days_until_all_infected([[1, 2, 0], [2, 2, 0]]) == -1
assert days_until_all_infected([[1]]) == 0 and days_until_all_infected([[0]]) == -1
assert days_until_all_infected([]) == 0 and days_until_all_infected([[2, 2]]) == 0

# Part 3
assert days_until_outbreak_ends([[1, 0, 0]], 2) == 4
assert days_until_outbreak_ends([[1, 0, 0]], 1) == 1
assert days_until_outbreak_ends([[0, 0, 0]], 2) == 0
assert days_until_outbreak_ends(a, 3) == 2 + 3          # d_max + D

# Part 4: variant A (the traced example) and variant B
assert simulate(d, 2, spread_threshold=2) == (4, 0)
assert simulate(d, 2, death_threshold=2) == (4, 4)
# a death threshold at or below the spread threshold, then one that no cell can ever reach
assert simulate(d, 2, spread_threshold=2, death_threshold=1)[1] == 3    # all three cells infected later die
assert simulate(d, 2, death_threshold=5)[1] == 0                        # nobody has five neighbours


def burn(grid, line, index):
    return [[DEAD if (r if line == "row" else c) == index else v for c, v in enumerate(row)]
            for r, row in enumerate(grid)]


# Part 5: the example of the statement, and why "burn the most infected line" is the wrong rule
assert simulate(e, 2, death_threshold=1)[1] == 9                        # burning nothing
assert 3 + simulate(burn(e, "col", 3), 2, death_threshold=1)[1] == 11   # the column with two infected cells
assert 3 + simulate(burn(e, "col", 2), 2, death_threshold=1)[1] == 4    # the column that seals the rest off
assert min_deaths(e, 2, 1) == 4
assert min_deaths([[0, 0], [0, 0]], 2, 1) == 0 and min_deaths([], 2, 1) == 0


def naive_min_deaths(grid, D, k_death, k_spread=1):
    """Straight from the statement: a whole-grid rewrite per day, sharing nothing with simulate()."""
    rows, cols = len(grid), len(grid[0])
    options = [[]] + [[(r, c) for c in range(cols)] for r in range(rows)] \
                   + [[(r, c) for r in range(rows)] for c in range(cols)]
    totals = []
    for line in options:
        state = [row[:] for row in grid]
        for r, c in line:
            state[r][c] = DEAD
        age = {(r, c): 0 for r in range(rows) for c in range(cols) if state[r][c] == INFECTED}
        fatal, deaths = set(), len(line)
        while age:
            for cell in list(age):
                age[cell] += 1
                if age[cell] >= D:
                    del age[cell]
                    state[cell[0]][cell[1]] = DEAD if cell in fatal else IMMUNE
                    deaths += cell in fatal
            before = [row[:] for row in state]
            for r in range(rows):
                for c in range(cols):
                    if before[r][c] != HEALTHY:
                        continue
                    n = sum(before[r + dr][c + dc] == INFECTED for dr, dc in STEPS
                            if 0 <= r + dr < rows and 0 <= c + dc < cols)
                    if n >= k_spread:
                        state[r][c], age[(r, c)] = INFECTED, 0
                        if n >= k_death:
                            fatal.add((r, c))
        totals.append(deaths)
    return min(totals)


# every 3x4 grid with two infected cells, against the naive version
cases = 0
for i in range(12):
    for j in range(i + 1, 12):
        g = [[HEALTHY] * 4 for _ in range(3)]
        g[i // 4][i % 4] = g[j // 4][j % 4] = INFECTED
        for days in (1, 2, 3):
            for threshold in (1, 2):
                assert min_deaths(g, days, threshold) == naive_min_deaths(g, days, threshold)
                cases += 1
assert cases == 396
```
