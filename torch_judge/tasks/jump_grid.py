"""Best path down a grid with a limited number of two-row jumps: its score, one path, how many there are, and bonuses."""

from ._interview import interview

# Enumerates every path of a small board, independent of the reference's dynamic programming.
_HELPERS = r"""
import random, time

def all_paths(board, p, k):
    n, m = len(board), len(board[0])
    out = []
    def walk(i, j, left, cells):
        if i == n - 1:
            out.append(list(cells)); return
        for c in (j - 1, j, j + 1):
            if 0 <= c < m and i + 1 < n:
                walk(i + 1, c, left, cells + [(i + 1, c)])
        if left and i + 2 < n:
            walk(i + 2, j, left - 1, cells + [(i + 2, j)])
    walk(0, p, k, [(0, p)])
    return out

def scored(board, path, x=0, y=0):
    v = [board[i][j] for i, j in path]
    s = sum(v)
    s += sum(x for t in range(1, len(v)) if v[t - 1] == v[t])
    s += sum(y for t in range(2, len(v)) if v[t - 2] < v[t - 1] < v[t])
    return s

def brute(board, p, k, x=0, y=0):
    paths = all_paths(board, p, k)
    base = [scored(board, q) for q in paths]
    top = max(base)
    best = [q for q, s in zip(paths, base) if s == top]
    return top, min(best), len(best), max(scored(board, q, x, y) for q in paths)

def random_board(rng, n, m, low=-3, high=3):
    return [[rng.randint(low, high) for _ in range(m)] for _ in range(n)]

def zero_paths(n, m, p, k):
    # How many paths an all-zero board has: every one of them scores 0.
    memo = {}
    def go(i, j, left):
        if i == n - 1:
            return 1
        key = (i, j, left)
        if key not in memo:
            total = sum(go(i + 1, c, left) for c in (j - 1, j, j + 1) if 0 <= c < m)
            if left and i + 2 < n:
                total += go(i + 2, j, left - 1)
            memo[key] = total
        return memo[key]
    return go(0, p, k)

def big_board(seed, n=60, m=60):
    rng = random.Random(seed)
    return [[rng.randint(-10**9, 10**9) for _ in range(m)] for _ in range(n)]
"""

P1 = [[-2, 4, 5], [-5, -5, -5], [-5, 4, 0], [-1, -4, 3]]
P2 = [[-4, -2, 5], [5, 4, -5], [4, 4, 1], [-5, -2, -5]]
P3 = [[-3, -1], [0, -2], [-1, 4], [0, 0]]
P4 = [[-2, -3, 6], [-2, -4, 4], [-1, 3, 2], [0, 6, 2]]

TESTS = [
    {"name": "Part 1: the worked example", "part": 1, "behavior": "contract.signature", "code": f"""
g = {{fn}}({P1!r})
assert g.max_score(1, 1) == 11
assert g.max_score(1, 0) == 6
"""},
    {"name": "Part 1: edges of the grid and the budget", "part": 1, "visibility": "unshown", "behavior": "edge.empty_or_boundary",
     "failure_message": "One row means the path is the start cell; one column allows only straight steps; a jump from the second-to-last row is illegal; k = 0 means steps only; k larger than useful changes nothing; sums of values up to 10**9 stay exact; the board is never changed.",
     "code": _HELPERS + r"""
assert {fn}([[7, -2]]).max_score(1, 3) == -2
assert {fn}([[1], [-5], [2], [-9], [4]]).max_score(0, 0) == -7
assert {fn}([[1], [-5], [2], [-9], [4]]).max_score(0, 2) == 7
assert {fn}([[1], [-5], [2], [-9], [4]]).max_score(0, 1) == 2, "step to the 2, then jump over the -9"
assert {fn}([[0, 0], [9, 9]]).max_score(0, 5) == 9, "a jump past the last row is not allowed"
assert {fn}([[0], [-1], [0]]).max_score(0, 1) == 0 and {fn}([[0], [-1], [0]]).max_score(0, 0) == -1
board = [[10**9] * 3 for _ in range(5)]
assert {fn}(board).max_score(2, 0) == 5 * 10**9
board = [[-2, 4, 5], [-5, -5, -5], [-5, 4, 0], [-1, -4, 3]]
before = [row[:] for row in board]
{fn}(board).max_score(0, 2)
assert board == before, "the board must not change"
"""},
    {"name": "Part 1: random small boards", "part": 1, "visibility": "unshown", "behavior": "state.invariant",
     "failure_message": "On random small boards, max_score differed from trying every path.",
     "code": _HELPERS + r"""
rng = random.Random(31)
for trial in range(400):
    n, m = rng.randint(1, 6), rng.randint(1, 4)
    board = random_board(rng, n, m)
    p, k = rng.randrange(m), rng.randint(0, 3)
    assert {fn}(board).max_score(p, k) == brute(board, p, k)[0], (board, p, k)
"""},
    {"name": "Part 1: a 60 by 60 board", "part": 1, "visibility": "unshown", "behavior": "performance.complexity",
     "failure_message": "A 60 by 60 board with k = 60 took too long: the number of paths is astronomically large, so reuse the best score from each (row, column, jumps left) instead of trying paths.",
     "code": _HELPERS + r"""
board = big_board(1)
g = {fn}(board)
start = time.perf_counter()
first = g.max_score(7, 60)
elapsed = time.perf_counter() - start
assert first >= g.max_score(7, 0) and g.max_score(7, 60) == first
assert elapsed < 4.0, f"{elapsed:.2f}s for one 60 x 60 board"
"""},
    {"name": "Part 2: the worked example", "part": 2, "behavior": "metrics.ties", "code": f"""
g = {{fn}}({P2!r})
assert g.max_score(1, 1) == 5
assert g.optimal_path(1, 1) == [(0, 1), (1, 0), (2, 0), (3, 1)]
assert g.optimal_path(1, 0) == [(0, 1), (1, 0), (2, 0), (3, 1)]
"""},
    {"name": "Part 2: random small boards", "part": 2, "visibility": "unshown", "behavior": "metrics.ties",
     "failure_message": "On random small boards with many ties, optimal_path must return the smallest cell list, as a list of (row, column) tuples, among the paths with the best score.",
     "code": _HELPERS + r"""
rng = random.Random(32)
for trial in range(400):
    n, m = rng.randint(1, 6), rng.randint(1, 4)
    board = random_board(rng, n, m, -1, 1)
    p, k = rng.randrange(m), rng.randint(0, 3)
    got = {fn}(board).optimal_path(p, k)
    assert got == brute(board, p, k)[1], (board, p, k, got)
    assert all(type(cell) is tuple for cell in got)
board = big_board(2)
start = time.perf_counter()
path = {fn}(board).optimal_path(3, 60)
elapsed = time.perf_counter() - start
assert path[0] == (0, 3) and path[-1][0] == 59 and sum(board[i][j] for i, j in path) == {fn}(board).max_score(3, 60)
assert elapsed < 4.0, f"{elapsed:.2f}s for one 60 x 60 path"
"""},
    {"name": "Part 3: the worked example", "part": 3, "behavior": "metrics.ties", "code": f"""
g = {{fn}}({P3!r})
assert g.max_score(1, 2) == 3
assert g.count_optimal_paths(1, 2) == 4
assert g.count_optimal_paths(1, 0) == 2
"""},
    {"name": "Part 3: counts, the modulus and size", "part": 3, "visibility": "unshown", "behavior": "metrics.ties",
     "failure_message": "Counts must match every path that reaches the best score on random boards, a jump and the two steps it skips count as different paths, the count is taken modulo 10**9 + 7, and a 60 by 60 board must count in a few seconds.",
     "code": _HELPERS + r"""
rng = random.Random(33)
for trial in range(400):
    n, m = rng.randint(1, 6), rng.randint(1, 4)
    board = random_board(rng, n, m, -1, 1)
    p, k = rng.randrange(m), rng.randint(0, 3)
    assert {fn}(board).count_optimal_paths(p, k) == brute(board, p, k)[2], (board, p, k)
assert {fn}([[0], [0], [0]]).count_optimal_paths(0, 1) == 2
for n, m, k in [(30, 4, 5), (60, 60, 60)]:
    start = time.perf_counter()
    got = {fn}([[0] * m for _ in range(n)]).count_optimal_paths(0, k)
    elapsed = time.perf_counter() - start
    assert got == zero_paths(n, m, 0, k) % (10**9 + 7), (n, m, k, got)
assert elapsed < 4.0, f"{elapsed:.2f}s to count a 60 x 60 board"
"""},
    {"name": "Part 4: the worked example", "part": 4, "behavior": "contract.signature", "code": f"""
g = {{fn}}({P4!r})
assert g.max_score(1, 1) == 10
assert g.max_score_with_bonuses(1, 1, 2, 5) == 14
assert g.max_score_with_bonuses(1, 1, 0, 0) == 10
"""},
    {"name": "Part 4: bonuses across jumps and ties", "part": 4, "visibility": "unshown", "behavior": "state.invariant",
     "failure_message": "Bonuses look at the values in visiting order, so a jump's two ends count as neighbours; equal neighbours add x; three strictly rising values in a row add y, with no y for the first two cells; the result must match trying every path, and a 60 by 60 board must finish in a few seconds.",
     "code": _HELPERS + r"""
assert {fn}([[1], [9], [1]]).max_score_with_bonuses(0, 1, 10, 0) == 12, "the jump makes 1 and 1 neighbours"
assert {fn}([[1], [2], [3]]).max_score_with_bonuses(0, 0, 0, 7) == 13
assert {fn}([[1], [2], [2]]).max_score_with_bonuses(0, 0, 0, 7) == 5 + 0, "y needs strict rises"
assert {fn}([[1], [2], [2]]).max_score_with_bonuses(0, 0, 4, 7) == 9
rng = random.Random(34)
for trial in range(400):
    n, m = rng.randint(1, 6), rng.randint(1, 4)
    board = random_board(rng, n, m)
    p, k = rng.randrange(m), rng.randint(0, 3)
    x, y = rng.randint(0, 5), rng.randint(0, 5)
    assert {fn}(board).max_score_with_bonuses(p, k, x, y) == brute(board, p, k, x, y)[3], (board, p, k, x, y)
board = big_board(4)
start = time.perf_counter()
got = {fn}(board).max_score_with_bonuses(9, 60, 10**9, 10**9)
elapsed = time.perf_counter() - start
assert got >= {fn}(board).max_score(9, 60)
assert elapsed < 4.0, f"{elapsed:.2f}s with bonuses on a 60 x 60 board"
"""},
]

TASK = {
    "title": "Grid Path with Limited Jumps",
    "difficulty": "Hard",
    "version": 1,
    "function_name": "JumpGrid",
    "description_en": r"""Build `JumpGrid`, which finds the best-scoring path down a grid where a few moves may skip a row: its score, the path itself, how many best paths there are, and the score with bonuses.

The requirement arrives in parts. Each part keeps every earlier behavior and adds one method, so one `JumpGrid` class passes all parts at the end. Pass every test of the current part to reveal the next one.

**Rules for every part:**
- `JumpGrid(board)` takes `N` rows of `M` integers, with `1 <= N, M <= 60` and every value between `-10**9` and `10**9`. Never change `board`.
- Every method must finish in a few seconds on a 60 by 60 board with `k = 60`.
- A path begins at `(0, p)` and stops on reaching row `N - 1`, so a one-row board gives a path of one cell.
- From `(i, j)`, a step goes to `(i + 1, j - 1)`, `(i + 1, j)` or `(i + 1, j + 1)`, if that column exists. A jump goes to `(i + 2, j)`, if that row exists. A path may use at most `k` jumps; `k >= 0`, and it may be larger than any path can use.
- A path's score is the sum of the values of the cells it visits, start and end included.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why this shows up in interviews:** it is a grid dynamic program with one extra dimension, the jumps left. Each later part adds one requirement: rebuilding a path, counting optimal paths, then a score that depends on the cells visited just before.

**Where it is used:** beam and lattice searches in decoding, sequence alignment with limited gaps, seam carving for image resizing, and any planner that must count or rank equally good plans.

Adapted from the grid path question in Schuture/OpenAI-Interview-Notes (CC BY-NC 4.0), reworded, as methods of one class that holds the board, with the special move renamed a jump and `K`, `X` and `Y` written `k`, `x` and `y`. The size limit is 60 instead of 200, so a plain Python table is fast enough, and the rule that `board` stays unchanged is new.""",
    "parts": [
        {
            "title": "Best score",
            "description_en": r"""**Signature:** `JumpGrid(board).max_score(p, k) -> int`

- Return the largest score over all paths from `(0, p)` with at most `k` jumps. `0 <= p < M`.

**Example:** `board = [[-2, 4, 5], [-5, -5, -5], [-5, 4, 0], [-1, -4, 3]]` and `p = 1`:
- `max_score(1, 1)` is `11`: the path jumps from `(0, 1)` over the row of `-5`s to `(2, 1)`, then steps to `(3, 2)`, for `4 + 4 + 3`
- `max_score(1, 0)` is `6`: without a jump the path must take a `-5`""",
        },
        {
            "title": "One best path",
            "description_en": r"""Keep Part 1 and add a method.

**Signature:** `optimal_path(p, k) -> list[tuple[int, int]]`

- Return the cells of a path with the best score, from `(0, p)` to the last row, as `(row, column)` tuples.
- When several paths tie, return the smallest list in Python's list order: compare the first cells, then the second ones, and so on.

**Example:** `board = [[-4, -2, 5], [5, 4, -5], [4, 4, 1], [-5, -2, -5]]`, `p = 1`, `k = 1`:
- the best score is `5`, reached by `[(0, 1), (1, 0), (2, 0), (3, 1)]` and `[(0, 1), (1, 0), (2, 1), (3, 1)]`
- they first differ at the third cell, where `(2, 0) < (2, 1)`, so `optimal_path(1, 1)` is the first""",
        },
        {
            "title": "Counting best paths",
            "description_en": r"""Keep Parts 1–2 and add a method.

**Signature:** `count_optimal_paths(p, k) -> int`

- Return how many different paths reach the best score, modulo `10**9 + 7`. Two paths differ if their cell lists differ, so a jump over `(i + 1, j)` and the two steps through it are different paths.
- Every path may tie, and the count can be far larger than `10**9`.

**Example:** `board = [[-3, -1], [0, -2], [-1, 4], [0, 0]]` and `p = 1`:
- with `k = 2` the best score is `3`. The path either steps through `(1, 0)` or jumps to `(2, 1)`, then ends at `(3, 0)` or `(3, 1)`, so `count_optimal_paths(1, 2)` is `4`
- with `k = 0` only the stepping paths remain: `count_optimal_paths(1, 0)` is `2`""",
        },
        {
            "title": "Bonuses",
            "description_en": r"""Keep Parts 1–3 and add a method.

**Signature:** `max_score_with_bonuses(p, k, x, y) -> int`, with `0 <= x, y <= 10**9`

- Bonuses look at the cell values in visiting order, `v0` first and `vL` last. The two ends of a jump are consecutive in that order, the same as the two ends of a step.
- Add `x` for every `t >= 1` with `v(t-1) == vt`, and `y` for every `t >= 2` with `v(t-2) < v(t-1) < vt`.
- Return the largest score plus bonuses over all paths. The best path here may differ from Part 1's.

**Example:** `board = [[-2, -3, 6], [-2, -4, 4], [-1, 3, 2], [0, 6, 2]]`, `p = 1`, `k = 1`:
- `max_score(1, 1)` is `10`, from the values `-3, 4, 3, 6`, which earn no bonus
- `max_score_with_bonuses(1, 1, 2, 5)` is `14`: the values `-3, -2, 3, 6` sum to `4` and rise strictly twice, adding `2 * 5`""",
        },
    ],
    "hints": [
        {"level": 1, "kind": "questions", "content": "Once a path reaches (i, j) with b jumps left, does the best way to finish depend on how it got there? How many such (i, j, b) situations are there, and in what order can you fill in their best finishes?"},
        {"level": 2, "kind": "analysis", "content": "Let best[i][j][b] be the largest score from (i, j) to the last row with b jumps left, counting board[i][j]. On the last row it is board[i][j]. Above it, add board[i][j] to the largest best over the legal steps (same b) and the jump (b - 1, if b > 0 and row i + 2 exists). Fill rows from the bottom up and return best[0][p][k]: O(N * M * k) time."},
    ],
    "model_connections": [
        "Beam search and lattice decoding keep the best score per state and step, and alignment models add a limited number of skip moves, the same table with a budget dimension.",
        "Counting the best paths modulo a prime is how decoders and parsers report ambiguity without enumerating every derivation.",
    ],
    "pro_con_analysis": {
        "pros": [
            "Reusing the best finish per (row, column, jumps left) turns an exponential search into O(N * M * k).",
            "The same table answers the score, rebuilds a path greedily and drives the count.",
            "Adding the previous value to the state handles bonuses without changing the overall method.",
        ],
        "cons": [
            "The table has N * M * (k + 1) entries, so memory grows with the jump budget.",
            "Bonuses that depend on more history need a larger state for each extra cell remembered.",
            "Exact integer scores are needed to find ties; with floating-point scores, ties and counts become unreliable.",
        ],
    },
    "tests": TESTS,
    "solution": r'''# Adapted from Schuture/OpenAI-Interview-Notes (code under the MIT License).
from functools import lru_cache

MOD = 10**9 + 7


class JumpGrid:
    def __init__(self, board):
        self._board = [list(row) for row in board]  # a private copy, so later changes to board do not leak in
        self._n = len(board)
        self._m = len(board[0])

    def _moves(self, i, j, jumps):
        """(next row, next column, jumps left) for every legal move from (i, j), in cell order."""
        if i + 1 < self._n:
            for c in (j - 1, j, j + 1):
                if 0 <= c < self._m:
                    yield i + 1, c, jumps
        if jumps and i + 2 < self._n:
            yield i + 2, j, jumps - 1

    def _table(self, k):
        """best[i][j][b]: the largest base score from (i, j) to the last row with b jumps left."""
        n, m, board = self._n, self._m, self._board
        best = [[[0] * (k + 1) for _ in range(m)] for _ in range(n)]
        for i in range(n - 1, -1, -1):
            for j in range(m):
                for b in range(k + 1):
                    if i == n - 1:
                        best[i][j][b] = board[i][j]
                    else:  # every row above the last has at least one step down
                        best[i][j][b] = board[i][j] + max(best[r][c][left] for r, c, left in self._moves(i, j, b))
        return best

    def max_score(self, p, k):
        return self._table(k)[0][p][k]

    def optimal_path(self, p, k):
        best = self._table(k)
        i, j, b = 0, p, k
        path = [(0, p)]
        while i < self._n - 1:
            want = best[i][j][b] - self._board[i][j]
            # moves come in (row, column) order, so the first optimal one keeps the path smallest
            i, j, b = next(move for move in self._moves(i, j, b) if best[move[0]][move[1]][move[2]] == want)
            path.append((i, j))
        return path

    def count_optimal_paths(self, p, k):
        best = self._table(k)
        n, m = self._n, self._m
        count = [[[1] * (k + 1) for _ in range(m)] for _ in range(n)]
        for i in range(n - 2, -1, -1):
            for j in range(m):
                for b in range(k + 1):
                    want = best[i][j][b] - self._board[i][j]
                    count[i][j][b] = sum(count[r][c][left] for r, c, left in self._moves(i, j, b)
                                         if best[r][c][left] == want) % MOD
        return count[0][p][k]

    def max_score_with_bonuses(self, p, k, x, y):
        board, last = self._board, self._n - 1

        @lru_cache(maxsize=None)
        def score(i, j, b, before):
            """Best score from (i, j) on, counting board[i][j]; before is the value visited just before it."""
            here = board[i][j]
            if i == last:
                return here
            options = []
            for r, c, left in self._moves(i, j, b):
                nxt = board[r][c]
                bonus = (x if here == nxt else 0) + (y if before is not None and before < here < nxt else 0)
                options.append(bonus + score(r, c, left, here))
            return here + max(options)

        result = score(0, p, k, None)
        score.cache_clear()
        return result
''',
    "interview_questions": interview(
        concept=[
            "Why does the best way to finish from (i, j) depend only on i, j and the jumps left, and not on how the path got there?",
            "Why must a jump be refused from the second-to-last row, and what would the path look like if it were allowed?",
        ],
        deep_dive=[
            "In what order must the table be filled so that every value it reads is already known, and what are the time and memory costs?",
        ],
        tradeoffs=[
            "How do you rebuild the lexicographically smallest best path from the table without storing every path?",
            "Why can counting best paths reuse the score table, and why must you only add counts from moves that reach the best score?",
            "What extra state do the bonuses need, and how much larger does the table become?",
            "How would you cut memory if only the score were needed, and why does that no longer work for rebuilding the path?",
        ],
    ),
}
