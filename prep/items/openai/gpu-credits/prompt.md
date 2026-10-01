A *GPU credit ledger* issues credit in *grants* and spends it over time.

`add_credit(credit_id, amount, timestamp, expiration)` issues a new grant of `amount` credits, identified by `credit_id` (unique across every grant ever issued). `expiration` is a duration, not an absolute time: the grant is valid on the closed interval `[timestamp, timestamp + expiration]`, including both endpoints.

`subtract(amount, timestamp)` spends `amount` credits at time `timestamp`. It drains the grants that are valid at `timestamp`, starting from the one with the smallest `timestamp + expiration` — the one that expires soonest — and moving on to the next once the current one is exhausted, so a single call may drain more than one grant. Credit a grant loses this way never comes back.

If `amount` is larger than everything valid at `timestamp`, `subtract` still never raises: the shortfall becomes an outstanding *debt*. Debt is not attached to any one grant. It is repaid by grants with a later `timestamp`: when a grant activates, its `amount` first pays down whatever debt is still outstanding, and only what is left over, if anything, becomes the grant's credit, which a later `subtract` can drain. The part that went to the debt is spent for good, exactly like credit drained by `subtract`: the debt does not come back when that grant expires. A ledger can therefore stay in debt for a long time, until enough new grants have activated to clear it.

`get_balance(timestamp)` reports the state of the ledger at `timestamp`, as if every `add_credit` and `subtract` made so far had been replayed in increasing order of its own `timestamp` — not the order the calls were actually made in — keeping only those whose `timestamp` is at most the one being queried. Let $v$ be the sum of the remaining credit of every grant valid at that instant, minus the debt outstanding at that instant. `get_balance` returns `None` when $v$ is negative, and returns $v$ itself otherwise. In particular, when no grant is valid and no debt is outstanding — before the first grant activates, or after the last one has expired — $v = 0$ and the return value is the integer `0`, not `None`.

Amounts, timestamps and durations are non-negative integers. Across `add_credit` and `subtract` together, no two calls ever carry the same `timestamp`; two different grants may still expire at the same instant. Across the whole call sequence, $U$ (calls to `add_credit`/`subtract`) and $Q$ (calls to `get_balance`) are each at most $2 \times 10^4$; every `timestamp` and `expiration` is at most $10^9$, `amount` is between $1$ and $10^9$, and a running total can exceed the range of a 32-bit integer.

### Part 1 — In-order arrival

For this part, `add_credit`, `subtract` and `get_balance` are always called in non-decreasing order of `timestamp`: the sequence of calls matches the order in which the events actually happen.

```py
class GPUCreditLedger:
    def add_credit(self, credit_id: str, amount: int, timestamp: int, expiration: int) -> None: ...
    def subtract(self, amount: int, timestamp: int) -> None: ...
    def get_balance(self, timestamp: int) -> int | None: ...
```

Example:

```text
add_credit('r1', 6, 10, 20)   # valid on [10, 30]
get_balance(10)  -> 6         # r1's window starts exactly at 10
add_credit('r2', 5, 14, 6)    # valid on [14, 20]
subtract(3, 16)               # drains r2 first, since it expires sooner: r2 5 -> 2
add_credit('r3', 4, 18, 8)    # valid on [18, 26]
subtract(9, 19)               # earliest-expiring first: r2 (2), then r3 (4), then r1 (3 of 6)
get_balance(19)  -> 3         # only r1 is left, holding 3 credits
get_balance(20)  -> 3         # 20 is the last instant of r2's window [14, 20], but r2 is already empty
add_credit('r4', 2, 24, 4)    # valid on [24, 28]
get_balance(28)  -> 5         # r1 (3) and r4 (2) are both still valid
get_balance(29)  -> 3         # r4's window [24, 28] has just ended; only r1 remains
get_balance(31)  -> 0         # r1's window [10, 30] has ended too: nothing is valid, and the answer is 0
```

### Part 2 — Out-of-order arrival

Real callers cannot guarantee this: `add_credit` and `subtract` may now be called in any order, independent of their `timestamp` — a `subtract` for `timestamp = 42` may be called before the `add_credit` for `timestamp = 35` that is meant to fund it. `get_balance(timestamp)` must still behave exactly as defined above, as if every call made so far had been replayed in increasing order of `timestamp`.

```py
class GPUCreditLedgerAnyOrder:
    def add_credit(self, credit_id: str, amount: int, timestamp: int, expiration: int) -> None: ...
    def subtract(self, amount: int, timestamp: int) -> None: ...
    def get_balance(self, timestamp: int) -> int | None: ...
```

Example (calls are listed in the order they are made, which is not the order of their `timestamp`):

```text
subtract(5, 42)                 # called before any add_credit at all
get_balance(8)  -> 0            # nothing has a timestamp <= 8 yet
add_credit('s1', 7, 35, 25)     # valid on [35, 60]
get_balance(35) -> 7            # the subtract's timestamp (42) is still in the future of this query
get_balance(42) -> 2            # replayed in timestamp order: 7 - 5 = 2
subtract(10, 48)                # only 2 credits are valid at 48; the other 8 become debt
get_balance(48) -> None         # v = 0 - 8 = -8
add_credit('s2', 5, 52, 10)     # valid on [52, 62]; all 5 go to the debt, 8 -> 3, and s2 holds nothing
get_balance(52) -> None         # v = 0 - 3 = -3
add_credit('s3', 3, 57, 10)     # valid on [57, 67]; clears exactly the remaining debt
get_balance(57) -> 0            # v = 0 - 0, returned as the integer 0, not None
get_balance(70) -> 0            # s2 and s3 have expired; the debt they repaid does not come back
```

### Part 3 — Answering many queries quickly

Implement `GPUCreditLedgerFast`, with the same behavior as `GPUCreditLedgerAnyOrder`, so that a long call sequence dominated by `get_balance` is fast. When the $Q$ calls to `get_balance` have non-decreasing timestamps, each made after every `add_credit`/`subtract` whose `timestamp` is at most its own, the time spent inside all $Q$ of them together must be $O(U \log U + Q)$ — against the $O(Q \cdot U \log U)$ of replaying every call from scratch on each query. Other arrival patterns (a query out of that order, or an `add_credit`/`subtract` landing at or behind a timestamp already queried) may fall back to a slower path, but `get_balance` must still return the value defined above.

```py
class GPUCreditLedgerFast:
    def add_credit(self, credit_id: str, amount: int, timestamp: int, expiration: int) -> None: ...
    def subtract(self, amount: int, timestamp: int) -> None: ...
    def get_balance(self, timestamp: int) -> int | None: ...
```
