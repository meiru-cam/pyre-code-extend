Implement the four levels below in order: a level's tests must pass before the next level's tests run, and every level extends the same class rather than replacing it. Every method's first argument is `timestamp`, a non-negative integer; across the whole sequence of calls made to one instance, timestamps never decrease. An `account_id` is a non-empty string, and every `amount` is a positive integer. An `account_id` identifies at most one *live* account at a time: `create_account` can make it live, `merge_accounts` (Level 4) can make it stop being live, and no other call changes this.

### Level 1 — Accounts, deposits and withdrawals

```py
class Bank:
    def create_account(self, timestamp: int, account_id: str) -> bool:
        """Opens a new account with balance 0. Returns True, or False, without effect, if
        account_id is already live."""

    def deposit(self, timestamp: int, account_id: str, amount: int) -> int | None:
        """Adds amount to account_id's balance. Returns the new balance, or None, without
        effect, if account_id is not live."""

    def withdraw(self, timestamp: int, account_id: str, amount: int) -> int | None:
        """Subtracts amount from account_id's balance. Returns the new balance. Returns None,
        without effect, if account_id is not live, or if amount is greater than its balance."""
```

Example:

```text
create_account(0, "A")     # -> True
create_account(1, "A")     # -> False    -- "A" is already live
deposit(2, "A", 100)       # -> 100
withdraw(3, "A", 40)       # -> 60
withdraw(4, "A", 1000)     # -> None     -- only 60 is available
deposit(5, "B", 10)        # -> None     -- "B" was never created
withdraw(6, "B", 1)        # -> None
```

### Level 2 — Ranking by outgoing spend

Every live account tracks an *outgoing total*: the sum of every `amount` that a successful `withdraw` call has taken from it (a call that returns `None` leaves it unchanged). A freshly created account starts at an outgoing total of 0.

```py
class Bank:
    def top_spenders(self, timestamp: int, n: int) -> list[str]:
        """Returns the n live accounts with the largest outgoing total, ranked highest first;
        accounts tied on outgoing total are ordered by account_id ascending. Each entry is the
        string f"{account_id}({outgoing_total})", e.g. "A(120)". Returns every live account, in
        the same order, if fewer than n are live. n is a positive integer."""
```

Example:

```text
create_account(0, "A"); create_account(0, "B"); create_account(0, "C")
deposit(1, "A", 100); deposit(2, "B", 100); deposit(3, "C", 100)
withdraw(4, "A", 30)              # -> 70     A's outgoing total is now 30
withdraw(5, "B", 30)              # -> 70     B's outgoing total is now 30, tied with A
top_spenders(6, 2)                # -> ["A(30)", "B(30)"]          tied on 30: "A" before "B"
top_spenders(6, 5)                # -> ["A(30)", "B(30)", "C(0)"]  only 3 accounts exist
withdraw(7, "A", 1000)            # -> None   fails: A's outgoing total stays 30
top_spenders(8, 1)                # -> ["A(30)"]
```

### Level 3 — Two-step transfers

From this level on, wherever this problem says an operation is allowed only if `amount` is at most an account's balance, it means the account's *available balance*: its balance minus the sum of `amount` over every transfer, started by `transfer` below, that is still pending with that account as its source. Deposits, and the balance itself, are unaffected by this; only what `withdraw` and `transfer` may take is. A transfer created at timestamp $t_0$ is *pending* for a call timestamped $t$ while $t_0 \le t \le t_0 + 1{,}000$, both ends inclusive, and *expired* for every later $t$; an expired transfer's hold stops counting against its source's available balance, the same as a completed one's, but nothing about the transfer itself changes retroactively — expiry is not a completion.

```py
class Bank:
    def transfer(self, timestamp: int, source_id: str, target_id: str, amount: int) -> str | None:
        """Starts a transfer of amount out of source_id, addressed to target_id, and returns its
        transfer_id: the string "transfer1", "transfer2", ... in the order transfers are created
        (one counter shared by every account, advanced only when transfer succeeds). The
        transfer is pending until accept_transfer completes or expires it; neither account's
        balance changes yet. Returns None, without effect, if any of: source_id equals
        target_id; source_id or target_id is not live; amount is greater than source_id's
        available balance."""

    def accept_transfer(self, timestamp: int, account_id: str, transfer_id: str) -> bool:
        """Completes a pending transfer: source_id's balance decreases by amount and target_id's
        balance increases by amount, in one step. Only target_id may complete a transfer, and
        only while it is pending. Returns True on success. Returns False, without effect, if
        any of: transfer_id was never returned by transfer; that transfer has already been
        completed; account_id is not its target_id; the transfer is expired."""
```

Example:

```text
create_account(0, "A"); create_account(0, "B"); create_account(0, "C")
deposit(1, "A", 500)                          # -> 500
transfer(10, "A", "B", 200)                   # -> "transfer1"   holds 200; window closes at 10+1000=1010
transfer(15, "A", "C", 250)                   # -> "transfer2"   holds 250; window closes at 15+1000=1015
withdraw(20, "A", 100)                        # -> None          available = 500-200-250 = 50 < 100
accept_transfer(1010, "B", "transfer1")       # -> True          last acceptable instant: 1010 <= 1010
                                               #    A debited 200 -> balance 300; B credited 200 -> balance 200
accept_transfer(1010, "B", "transfer1")       # -> False         already completed
accept_transfer(1011, "A", "transfer2")       # -> False         A is not transfer2's target ("C" is)
accept_transfer(1016, "C", "transfer2")       # -> False         1016 > 1015: one past the window, expired
withdraw(1300, "A", 300)                      # -> 0             transfer2's hold no longer counts: available = 300
```

### Level 4 — Merging accounts

`merge_accounts(timestamp, a, b)` merges `b` into `a`. Returns `False`, without effect, if `a` equals `b`, or if `a` or `b` is not live. Otherwise it has all of the following effects, and returns `True`:

- `a`'s balance increases by `b`'s balance.
- `a`'s outgoing total increases by `b`'s outgoing total.
- Every transfer still pending with `b` as its `source_id` has `a` as its `source_id` from now on (its hold counts against `a`, not `b`); every transfer still pending with `b` as its `target_id` has `a` as its `target_id` from now on (`a`, not `b`, may complete it). A transfer whose `source_id` and `target_id` are both `a` after this stays pending and may be completed like any other; completing it debits and credits the same account, so it only releases the hold.
- `b` stops being live.

`account_id` may identify a sequence of accounts over time, one *generation* at a time: `create_account` starts a new generation of `account_id` whenever `account_id` is not currently live, including right after `merge_accounts` has just made its previous generation stop being live. A generation's balance starts at 0 and is entirely its own: `merge_accounts` is the only way one generation's balance can affect another's, and it affects only whichever generation of `a` is live at the moment of the merge, never a later generation that reuses the id `b`.

```py
class Bank:
    def get_balance(self, timestamp: int, account_id: str, time_at: int) -> int | None:
        """Returns the balance that account_id's current generation had at time_at: its balance
        immediately after the last deposit, withdrawal, transfer completion or merge credit on
        that generation timestamped at most time_at (a completed transfer counts at the
        timestamp accept_transfer completed it, not the one transfer created it; a merge counts
        at the timestamp merge_accounts was called). Returns None if account_id is not live, or
        if time_at is before the current generation's own create_account call. time_at is a
        non-negative integer with no ordering constraint of its own -- unlike every method's
        leading timestamp, it may repeat or go backward from one call to the next."""
```

Example:

```text
create_account(0, "A"); create_account(0, "B")
deposit(1, "A", 100)                           # -> 100
deposit(2, "B", 50)                            # -> 50
transfer(3, "B", "A", 20)                      # -> "transfer1"   B holds 20, pending, target "A"
merge_accounts(5, "A", "B")                    # -> True          A: balance 100+50=150; transfer1's source
                                                #    becomes "A" (target was already "A"); "B" stops being live
get_balance(6, "B", 4)                         # -> None          "B" is not live, at any time_at
get_balance(6, "A", 4)                         # -> 100           A's own balance just before the merge
get_balance(6, "A", 5)                         # -> 150           the merge itself, at its own timestamp
accept_transfer(10, "A", "transfer1")          # -> True          source and target are both "A": debits then
                                                #    credits the same account, net 0; balance stays 150
get_balance(11, "A", 10)                       # -> 150
create_account(12, "B")                        # -> True          a fresh generation of "B"
get_balance(13, "B", 4)                        # -> None          4 predates this generation's own creation (12);
                                                #    the merged-away generation's balance at 4 (which was 50)
                                                #    does not carry over
get_balance(13, "B", 12)                       # -> 0
deposit(14, "B", 9)                            # -> 9
get_balance(15, "B", 14)                       # -> 9
merge_accounts(16, "A", "A")                   # -> False         a and b are the same account
merge_accounts(17, "A", "nope")                # -> False         "nope" is not live
```
