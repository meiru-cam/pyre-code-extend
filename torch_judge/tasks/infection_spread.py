"""An infection spreading on a grid, with one rule added per part."""

from ._interview import interview

# A slow model written straight from the statement: rewrite the whole grid once per day.
# It shares no code with the reference, so random grids can be checked against it.
_HELPERS = r"""
import random

STEPS = ((1, 0), (-1, 0), (0, 1), (0, -1))

def infected_around(g, r, c):
    return sum(g[r + dr][c + dc] == 1 for dr, dc in STEPS
               if 0 <= r + dr < len(g) and 0 <= c + dc < len(g[0]))

def slow_all_infected(grid):
    g, day = [row[:] for row in grid], 0
    while any(v == 0 for row in g for v in row):
        new = [[1 if v == 0 and infected_around(g, r, c) else v for c, v in enumerate(row)]
               for r, row in enumerate(g)]
        if new == g:
            return -1
        g, day = new, day + 1
    return day

def slow_simulate(grid, recover_after, spread=1, death=None, burnt=()):
    g = [row[:] for row in grid]
    for r, c in burnt:
        g[r][c] = 3
    age = {(r, c): 0 for r, row in enumerate(g) for c, v in enumerate(row) if v == 1}
    fatal, deaths, day = set(), len(burnt), 0
    while age:
        day += 1
        for cell in list(age):
            age[cell] += 1
            if age[cell] >= recover_after:
                del age[cell]
                g[cell[0]][cell[1]] = 3 if cell in fatal else 2
                deaths += cell in fatal
        before = [row[:] for row in g]
        for r, row in enumerate(before):
            for c, v in enumerate(row):
                n = infected_around(before, r, c) if v == 0 else 0
                if v == 0 and n >= spread:
                    g[r][c], age[(r, c)] = 1, 0
                    if death is not None and n >= death:
                        fatal.add((r, c))
    return day, deaths

def slow_min_deaths(grid, recover_after, death, spread=1):
    if not grid:
        return 0
    rows, cols = len(grid), len(grid[0])
    lines = [[]] + [[(r, c) for c in range(cols)] for r in range(rows)]
    lines += [[(r, c) for r in range(rows)] for c in range(cols)]
    return min(slow_simulate(grid, recover_after, spread, death, line)[1] for line in lines)

def random_grid(rng, states):
    rows, cols = rng.randint(1, 5), rng.randint(1, 5)
    return [[rng.choice(states) for _ in range(cols)] for _ in range(rows)]

def frozen(grid):
    return tuple(tuple(row) for row in grid)
"""

TASK = {
    "title": "Infection Spread on a Grid",
    "difficulty": "Medium",
    "version": 1,
    "function_name": "Outbreak",
    "description_en": r"""Build `Outbreak`, which simulates an infection spreading across a grid of cells, day by day.

The requirement arrives in parts. Each part keeps every earlier behavior, so one `Outbreak` class passes all parts at the end. Pass every test of the current part to reveal the next one.

**Rules for every part:**
- `Outbreak(grid)` takes `grid`, a list of rows of equal length holding integers. `0` is a healthy cell and `1` an infected one; later parts add more states.
- Each cell has up to four neighbours: the cells that share an edge with it. Cells touching only at a corner are not neighbours.
- Day 0 is the input grid. Day `t + 1` is computed from day `t` all at once: a cell infected on day `t + 1` spreads nothing until the step to day `t + 2`.
- The grid may be empty (`[]`). No method may modify `grid`.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why this shows up in interviews:** the first part is a short graph search, and each later part adds one requirement.

**Where it is used:** epidemic models, wildfire spread and other cellular automata.

Adapted from the infection spread question in Schuture/OpenAI-Interview-Notes (CC BY-NC 4.0), reworded, on one class, with the order of phases and the moment deaths are counted spelled out.""",
    "parts": [
        {
            "title": "Spread",
            "description_en": r"""**Signature:** `days_until_all_infected() -> int`

- Any healthy cell with an infected neighbour on day `t` is itself infected on day `t + 1`, and stays so.
- Return the first day on which no cell is healthy.
- Return `0` when no cell is healthy on day 0, which includes the empty grid.
- Return `-1` when that day never comes.

**Example:**
- `Outbreak([[1, 0, 0, 0, 0], [0, 0, 0, 0, 0]]).days_until_all_infected()` is `5`
- `Outbreak([[0, 0]]).days_until_all_infected()` is `-1`""",
        },
        {
            "title": "Immune cells",
            "description_en": r"""Keep Part 1 and add a third state:

- `2` marks an immune cell. Its state never changes, and it never infects anything, so it blocks the spread like a wall.
- `days_until_all_infected` keeps its meaning, so it returns `-1` when an immune wall shuts some healthy cell off.
- Each cell is processed a constant number of times: grids of 10,000 cells must finish within a second, even when the infection takes thousands of days to wind through them.

**Example:**
- `[[0, 2, 1], [0, 2, 0], [0, 0, 0]]` gives `6`: the infection goes around the immune column
- `[[1, 2, 0]]` gives `-1`""",
        },
        {
            "title": "Recovery",
            "description_en": r"""**Signature:** `days_until_outbreak_ends(recover_after) -> int`

Keep Parts 1–2 and add recovery. `recover_after` is an integer `>= 1`. Let `t0` be the day a cell became infected, with `t0 = 0` for cells infected in the input. The step to day `t` runs two phases in order:

- Recovery first: every infected cell with `t - t0 >= recover_after` becomes immune (`2`).
- Spread second: every healthy cell with a neighbour still infected after the recovery phase becomes infected, with `t0 = t`.
- Return the earliest day with zero infected cells, or `0` if the input has none. Some cells may stay healthy forever.

**Example:**
- `Outbreak([[0, 0, 1]]).days_until_outbreak_ends(2)` is `4`
- With `recover_after = 1` it is `1`: the first cell recovers before it can spread""",
        },
        {
            "title": "Thresholds and deaths",
            "description_en": r"""**Signature:** `simulate(recover_after, spread_threshold=1, death_threshold=None) -> tuple[int, int]`

Keep Parts 1–3. `simulate` runs the Part 3 rules with two more:

- Spread threshold: in the spread phase a healthy cell becomes infected only when at least `spread_threshold` of its neighbours are infected. With `1` this is Part 3.
- Deaths: a new state `3` is a dead cell. It never changes and never spreads. The input grid never contains `3`.
- When `death_threshold` is not `None`, a cell is doomed if the count that infected it, the same one the spread threshold checks (neighbours still infected after the recovery phase, not cells infected in this same step), is at least `death_threshold`: when its `recover_after` days are up it dies instead of becoming immune.
- That count is taken only on the day the cell is infected. Neighbours infected later change nothing.
- Return `(first day with no infected cell, number of cells that died)`.

**Example:** for `[[1, 0, 1], [0, 0, 0], [0, 0, 1]]`:
- `simulate(2, spread_threshold=2)` is `(4, 0)`
- `simulate(2, death_threshold=2)` is `(4, 4)`""",
        },
        {
            "title": "One intervention",
            "description_en": r"""**Signature:** `min_deaths(recover_after, death_threshold, spread_threshold=1) -> int`

Keep Parts 1–4. Before day 0 you get one optional intervention: burn a single row or a single column completely, or do nothing:

- Every cell of a burnt line becomes dead (`3`), whatever it held, and counts as a death.
- The Part 4 rules then run to the end with the given thresholds.
- Return the smallest total number of deaths over all these choices.

**Example:** for `[[1, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0]]` with `recover_after = 2` and `death_threshold = 1`:
- Burning nothing costs `9` deaths
- Burning column `1` costs `4`, which is the answer""",
        },
    ],
    "hints": [
        {"level": 1, "kind": "questions", "content": "If you push every infected cell into one queue at the start, what does a cell's distance from that queue tell you? How do you know one day ends and the next begins? How do you tell, when the queue empties, that some healthy cell was never reached?"},
        {"level": 2, "kind": "analysis", "content": "Run one breadth-first search that starts from all infected cells at once, and process the queue one level per day. Mark cells as reached when you enqueue them, on a copy, and count the healthy cells left: stop at 0 and return the day, or return -1 if the queue empties first."},
    ],
    "model_connections": [
        "Epidemic models such as SIR track susceptible, infected and recovered populations; this grid is a cellular-automaton version of that model.",
        "Counting infected neighbours for every cell at once is a 2D convolution with a plus-shaped kernel, which is how GPU simulations of such grids run.",
    ],
    "pro_con_analysis": {
        "pros": [
            "A multi-source breadth-first search visits each cell once, so plain spreading costs O(rows x cols).",
            "A day-by-day simulation with per-cell timers follows the statement directly, so new rules slot in as new phases.",
            "Trying every row, every column and doing nothing makes the intervention search exact.",
        ],
        "cons": [
            "Timers and thresholds break the breadth-first shortcut, so the simulation costs O(cells x days).",
            "The exact intervention search reruns the whole simulation once per row and column.",
            "Burning on later days or burning several lines grows the search space exponentially.",
        ],
    },
    "tests": [
        {"name": "Part 1: the worked example", "part": 1, "behavior": "state.invariant", "code": r"""
assert {fn}([[1, 0, 0, 0, 0], [0, 0, 0, 0, 0]]).days_until_all_infected() == 5
assert {fn}([[0, 0]]).days_until_all_infected() == -1
assert {fn}([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]]).days_until_all_infected() == 2
"""},
        {"name": "Part 1: empty and finished grids", "part": 1, "visibility": "unshown", "behavior": "edge.empty_or_boundary",
         "failure_message": "A grid with no healthy cell, including the empty grid, is finished on day 0; a grid with no infected cell and a healthy one never finishes.",
         "code": r"""
assert {fn}([]).days_until_all_infected() == 0
assert {fn}([[1]]).days_until_all_infected() == 0
assert {fn}([[1, 1], [1, 1]]).days_until_all_infected() == 0
assert {fn}([[0]]).days_until_all_infected() == -1
assert {fn}([[1], [0], [0]]).days_until_all_infected() == 2
"""},
        {"name": "Part 1: the grid is left untouched", "part": 1, "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "The method changed the grid it was given; work on a copy or on separate bookkeeping.",
         "code": _HELPERS + r"""
grid = [[1, 0, 0], [0, 0, 0]]
before = frozen(grid)
outbreak = {fn}(grid)
assert outbreak.days_until_all_infected() == 3
assert frozen(grid) == before
assert outbreak.days_until_all_infected() == 3, "a second call must give the same answer"
"""},
        {"name": "Part 1: random grids match a reference", "part": 1, "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "On a random grid of healthy and infected cells, the day differed from a day-by-day simulation.",
         "code": _HELPERS + r"""
for seed in range(300):
    rng = random.Random(seed)
    grid = random_grid(rng, [0, 0, 0, 1])
    assert {fn}(grid).days_until_all_infected() == slow_all_infected(grid), (seed, grid)
"""},
        {"name": "Part 2: immune cells block the spread", "part": 2, "behavior": "state.invariant", "code": r"""
assert {fn}([[0, 2, 1], [0, 2, 0], [0, 0, 0]]).days_until_all_infected() == 6
assert {fn}([[1, 2, 0]]).days_until_all_infected() == -1
assert {fn}([[2, 2], [2, 1]]).days_until_all_infected() == 0
"""},
        {"name": "Part 2: random grids with immune cells match a reference", "part": 2, "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "On a random grid with immune cells, the day differed from a day-by-day simulation.",
         "code": _HELPERS + r"""
for seed in range(300):
    rng = random.Random(seed)
    grid = random_grid(rng, [0, 0, 1, 2])
    assert {fn}(grid).days_until_all_infected() == slow_all_infected(grid), (seed, grid)
"""},
        {"name": "Part 2: a long winding corridor", "part": 2, "visibility": "unshown", "behavior": "performance.complexity",
         "failure_message": "A 101 x 101 grid whose corridor takes thousands of days was too slow; visit each cell a constant number of times instead of rescanning the grid every day.",
         "code": r"""
import time
size = 101
grid = [[0] * size for _ in range(size)]
for r in range(1, size, 2):
    gap = size - 1 if r % 4 == 1 else 0
    for c in range(size):
        if c != gap:
            grid[r][c] = 2
grid[0][0] = 1
start = time.perf_counter()
days = {fn}(grid).days_until_all_infected()
elapsed = time.perf_counter() - start
assert days == 5200, days
assert elapsed < 1.0, f"took {elapsed:.2f}s"
"""},
        {"name": "Part 3: recovery comes before the spread", "part": 3, "behavior": "state.invariant", "code": r"""
assert {fn}([[0, 0, 1]]).days_until_outbreak_ends(2) == 4
assert {fn}([[0, 0, 1]]).days_until_outbreak_ends(1) == 1
assert {fn}([[0, 0, 0]]).days_until_outbreak_ends(3) == 0
assert {fn}([]).days_until_outbreak_ends(2) == 0
"""},
        {"name": "Part 3: random grids match a reference", "part": 3, "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "On a random grid, the day the outbreak ends differed from a day-by-day simulation of recovery then spread.",
         "code": _HELPERS + r"""
for seed in range(300):
    rng = random.Random(seed)
    grid = random_grid(rng, [0, 0, 0, 1, 2])
    before = frozen(grid)
    recover_after = rng.randint(1, 4)
    expected = slow_simulate(grid, recover_after)[0]
    assert {fn}(grid).days_until_outbreak_ends(recover_after) == expected, (seed, grid, recover_after)
    assert frozen(grid) == before
"""},
        {"name": "Part 4: the worked example", "part": 4, "behavior": "state.invariant", "code": r"""
outbreak = {fn}([[1, 0, 1], [0, 0, 0], [0, 0, 1]])
assert outbreak.simulate(2, spread_threshold=2) == (4, 0)
assert outbreak.simulate(2, death_threshold=2) == (4, 4)
assert outbreak.simulate(2) == outbreak.simulate(2, spread_threshold=1, death_threshold=None)
assert outbreak.simulate(2)[0] == outbreak.days_until_outbreak_ends(2)
"""},
        {"name": "Part 4: the death count is taken once", "part": 4, "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "Whether a cell dies depends only on its infected neighbours on the day it is infected, not on neighbours infected later.",
         "code": r"""
# Cell (0, 1) is infected on day 1 by one neighbour; its other neighbour, (0, 2), is infected on day 2.
outbreak = {fn}([[1, 0, 0, 0]])
assert outbreak.simulate(3, death_threshold=2) == (6, 0)
assert outbreak.simulate(3, death_threshold=1) == (6, 3)
assert outbreak.simulate(2, death_threshold=5) == (5, 0), "nobody has five neighbours"
assert outbreak.simulate(2, spread_threshold=5) == (2, 0)
# One cell has three infected neighbours before the recovery phase but only two after it.
late = {fn}([[0, 1, 0, 1], [1, 0, 0, 0], [0, 2, 0, 0], [2, 1, 0, 1]])
assert late.simulate(4, spread_threshold=2, death_threshold=3) == (8, 0), "count neighbours after recovery"
"""},
        {"name": "Part 4: random grids match a reference", "part": 4, "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "On a random grid, (day the outbreak ends, deaths) differed from a day-by-day simulation with the two thresholds.",
         "code": _HELPERS + r"""
for seed in range(400):
    rng = random.Random(seed)
    grid = random_grid(rng, [0, 0, 0, 1, 2])
    recover_after = rng.randint(1, 4)
    spread = rng.randint(1, 3)
    death = rng.choice([None, 1, 2, 3])
    expected = slow_simulate(grid, recover_after, spread, death)
    before = frozen(grid)
    got = {fn}(grid).simulate(recover_after, spread_threshold=spread, death_threshold=death)
    assert tuple(got) == expected, (seed, grid, recover_after, spread, death, got, expected)
    assert frozen(grid) == before, "simulate must not change the input grid"
"""},
        {"name": "Part 5: the worked example", "part": 5, "behavior": "state.invariant", "code": r"""
outbreak = {fn}([[1, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0]])
assert outbreak.simulate(2, death_threshold=1) == (5, 9)
assert outbreak.min_deaths(2, 1) == 4
"""},
        {"name": "Part 5: doing nothing and burnt cells count", "part": 5, "visibility": "unshown", "behavior": "edge.empty_or_boundary",
         "failure_message": "Doing nothing must be one of the choices, and every cell of a burnt line counts as a death, immune and infected cells included.",
         "code": _HELPERS + r"""
assert {fn}([]).min_deaths(2, 1) == 0
assert {fn}([[0, 0], [0, 0]]).min_deaths(2, 1) == 0
assert {fn}([[1, 0, 0]]).min_deaths(2, 2) == 0
assert {fn}([[1, 0, 0, 0, 0]]).min_deaths(2, 1) == 1, "burning the column of the infected cell costs 1"
assert {fn}([[2, 1], [2, 0]]).min_deaths(3, 1) == 1
grid = [[1, 0, 0], [0, 0, 0]]
before = frozen(grid)
{fn}(grid).min_deaths(2, 1)
assert frozen(grid) == before, "a burn must not change the input grid"
"""},
        {"name": "Part 5: random grids match a reference", "part": 5, "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "On a random grid, the smallest number of deaths differed from trying every row, every column and doing nothing.",
         "code": _HELPERS + r"""
for seed in range(150):
    rng = random.Random(seed)
    grid = random_grid(rng, [0, 0, 0, 1, 2])
    recover_after = rng.randint(1, 3)
    spread = rng.randint(1, 2)
    death = rng.randint(1, 3)
    expected = slow_min_deaths(grid, recover_after, death, spread)
    assert {fn}(grid).min_deaths(recover_after, death, spread_threshold=spread) == expected, (seed, grid)
"""},
    ],
    "solution": r'''# Adapted from Schuture/OpenAI-Interview-Notes (code under the MIT License).
from collections import Counter, deque

HEALTHY, INFECTED, IMMUNE, DEAD = 0, 1, 2, 3
_STEPS = ((1, 0), (-1, 0), (0, 1), (0, -1))


class Outbreak:
    def __init__(self, grid):
        self.grid = [list(row) for row in grid]
        self.rows = len(self.grid)
        self.cols = len(self.grid[0]) if self.grid else 0

    def _neighbours(self, r, c):
        for dr, dc in _STEPS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                yield nr, nc

    def days_until_all_infected(self):
        # Multi-source BFS: one queue level per day, each cell enqueued at most once.
        reached = [[cell != HEALTHY for cell in row] for row in self.grid]
        frontier = deque((r, c) for r in range(self.rows) for c in range(self.cols)
                         if self.grid[r][c] == INFECTED)
        healthy = sum(cell == HEALTHY for row in self.grid for cell in row)
        days = 0
        while frontier and healthy:
            days += 1
            for _ in range(len(frontier)):
                r, c = frontier.popleft()
                for nr, nc in self._neighbours(r, c):
                    if not reached[nr][nc]:
                        reached[nr][nc] = True
                        healthy -= 1
                        frontier.append((nr, nc))
        return days if healthy == 0 else -1

    def days_until_outbreak_ends(self, recover_after):
        return self.simulate(recover_after)[0]

    def simulate(self, recover_after, spread_threshold=1, death_threshold=None):
        return self._run([row[:] for row in self.grid], recover_after, spread_threshold, death_threshold)

    def min_deaths(self, recover_after, death_threshold, spread_threshold=1):
        lines = [[(r, c) for c in range(self.cols)] for r in range(self.rows)]
        lines += [[(r, c) for r in range(self.rows)] for c in range(self.cols)]
        best = self.simulate(recover_after, spread_threshold, death_threshold)[1]
        for line in lines:
            state = [row[:] for row in self.grid]
            for r, c in line:
                state[r][c] = DEAD
            deaths = self._run(state, recover_after, spread_threshold, death_threshold)[1]
            best = min(best, len(line) + deaths)
        return best

    def _run(self, state, recover_after, spread_threshold, death_threshold):
        infected_on = {(r, c): 0 for r in range(self.rows) for c in range(self.cols)
                       if state[r][c] == INFECTED}
        doomed = set()
        day = deaths = 0
        while infected_on:
            day += 1
            for cell in [cell for cell, since in infected_on.items() if day - since >= recover_after]:
                del infected_on[cell]
                state[cell[0]][cell[1]] = DEAD if cell in doomed else IMMUNE
                deaths += cell in doomed
            # Count every healthy cell's infected neighbours first, then apply, so a cell
            # infected today cannot spread today.
            pressure = Counter()
            for r, c in infected_on:
                for nr, nc in self._neighbours(r, c):
                    if state[nr][nc] == HEALTHY:
                        pressure[(nr, nc)] += 1
            for (r, c), count in pressure.items():
                if count >= spread_threshold:
                    state[r][c] = INFECTED
                    infected_on[(r, c)] = day
                    if death_threshold is not None and count >= death_threshold:
                        doomed.add((r, c))
        return day, deaths
''',
    "interview_questions": interview(
        concept=[
            "Why does a breadth-first search started from every infected cell at once give each cell its infection day?",
            "How do you tell from the search that some healthy cell can never be infected?",
        ],
        deep_dive=[
            "Why should a cell be marked as reached when it enters the queue rather than when it leaves it?",
        ],
        tradeoffs=[
            "Once cells recover after a fixed number of days, why does a plain breadth-first search stop being enough?",
            "Why must the spread phase count every cell's infected neighbours before it infects any of them?",
            "Is burning the line with the most infected cells a good rule, and what does the exact search over lines cost?",
        ],
    ),
}
