### Part 1 — One secret, one guess per call

There is a hidden integer `secret` in the closed interval `[1, n]`. The only way to learn anything about it is `check(x)`, whose answer is delayed by exactly one call:

- `x` must lie in `[1, n]` and must never repeat a value passed to an earlier `check` call in this game.
- The very first call to `check` returns `None` — there is no earlier guess yet to compare.
- Every call after that returns the comparison of the **previous** call's argument against `secret`: `-1` if that previous `x` was less than `secret`, `0` if it was equal, `1` if it was greater. The comparison for the `x` you are passing *right now* only comes back on the *next* call.
- As soon as the comparisons revealed so far leave a single possible `secret`, `check` must not be called again; with `n = 1` that is already the case before the first call.
- Because of the delay, there is always at most one call whose comparison you have not yet seen. You may spend one further call — with any `x` in `[1, n]` that has not been used yet — purely to retrieve it.

```py
def check(x: int) -> int | None:
    """Provided. x must lie in [1, n] and must not repeat an earlier call's x in this game.
    Returns None on the very first call. On every later call, returns -1, 0, or 1: the
    comparison of the PREVIOUS call's argument against the hidden secret, never of x itself."""

def find_secret(n: int, check) -> int:
    """Returns secret, calling check at most 2 * ceil(log2(n)) + 1 times."""
```

For example, with `n = 7` and `secret = 5`:

```text
check(3) -> None    # no earlier guess to compare yet
check(6) -> -1      # compares 3 to secret: 3 < 5
check(4) -> 1       # compares 6 to secret: 6 > 5, so secret is in {4, 5}
check(7) -> -1      # compares 4 to secret: 4 < 5, so secret = 5
```

At this point `secret = 5` is already the only possibility, so calling `check` again (with `2`, or any other unused number) would be rejected.

### Part 2 — Two guesses per call, same secret

`check_batch` replaces `check` with the same one-call delay, now applied to a whole list at once:

- Every `x` used across every call in the game — in any batch — must be distinct and lie in `[1, n]`, and every call must carry at least one guess.
- The first call returns `None`.
- Every later call returns a list the same length as the **previous** call's list, holding the comparison of each of those `x`'s against `secret`, in the same order.
- The same "one more call to retrieve the last batch" allowance and the same "stop once `secret` is determined" rule apply.

```py
def check_batch(xs: list[int]) -> list[int] | None:
    """Provided. Every x used across every call in the game must be distinct and lie in [1, n],
    and xs must be non-empty. Returns None on the first call. On every later call, returns a list
    the same length as the PREVIOUS call's xs, comparing each of those guesses to secret (same
    -1/0/1 sense as check), in the same order."""

def find_secret_batched(n: int, check_batch) -> int:
    """Returns secret, calling check_batch at most 2 * ceil(log(n, 3)) + 1 times."""
```

For example, with `n = 13` and `secret = 9`:

```text
check_batch([4, 10]) -> None       # no earlier batch to compare yet
check_batch([2])     -> [-1, 1]    # compares 4, then 10, to secret: 4 < 9, 10 > 9
```

### Part 3 — Many independent games, minimize the number of rounds

There are several independent games running behind the *same* channel, each identified by a game id, each with its own bound `n[g]` and its own hidden `secret[g]`. Each round you submit a dict mapping some subset of the ids — possibly none of them — to one guess for that game; `check_round` answers by echoing back the **previous round's whole submission**, comparison by comparison — regardless of which ids the current round itself contains:

- Within one game id, every guess ever submitted for it must be distinct and lie in `[1, n[g]]`.
- The first round returns `None`.
- Every later round returns a dict with exactly the keys that were submitted in the **previous** round, each mapped to the comparison of that guess against its own game's secret.
- A game id must not be guessed again once its own secret is already determined.
- What is to be minimized now is the total number of rounds (calls to `check_round`), not the total number of individual guesses.

```py
def check_round(guesses: dict[str, int]) -> dict[str, int] | None:
    """Provided. guesses maps a subset of game ids to one guess each, guesses[g] in [1, n[g]];
    within one id, guesses must never repeat. Returns None on the first call. On every later
    call, returns a dict with exactly the keys submitted in the PREVIOUS call, each mapped to the
    -1/0/1 comparison of that guess against ITS OWN game's secret -- independent of which ids the
    CURRENT call itself contains."""

def solve_games(bounds: dict[str, int], check_round) -> dict[str, int]:
    """Returns every game's secret. Calls check_round at most
    2 * ceil(log2(max(bounds.values()))) + 1 times, regardless of how many games there are."""
```

For example, with games `"A"` (`n = 5`, `secret = 4`) and `"B"` (`n = 20`, `secret = 15`):

```text
check_round({"A": 2, "B": 8}) -> None                 # no earlier round to compare yet
check_round({})               -> {"A": -1, "B": -1}   # echoes round 1's guesses, though this round asks nothing new
```
