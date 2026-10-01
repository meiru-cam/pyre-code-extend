"""A bank ledger whose operations arrive one part at a time, as in an online assessment."""

from ._interview import interview

# A deliberately slow model written straight from the statement: no indexes, every query
# rescans. Adapted from the MIT-licensed cross-check in Schuture/Anthropic-Interview-Notes.
_ORACLE = r"""
import random

class SlowBank:
    def __init__(self):
        self.accounts, self.transfers, self.seq = {}, {}, 0
    def create_account(self, ts, aid):
        if aid in self.accounts:
            return False
        self.accounts[aid] = {"balance": 0, "outgoing": 0, "events": [(ts, 0)]}
        return True
    def _record(self, aid, ts):
        self.accounts[aid]["events"].append((ts, self.accounts[aid]["balance"]))
    def deposit(self, ts, aid, amount):
        if aid not in self.accounts:
            return None
        self.accounts[aid]["balance"] += amount
        self._record(aid, ts)
        return self.accounts[aid]["balance"]
    def _held(self, aid, ts):
        return sum(t["amount"] for t in self.transfers.values()
                   if t["source"] == aid and not t["accepted"] and ts <= t["created_at"] + 1000)
    def withdraw(self, ts, aid, amount):
        if aid not in self.accounts or amount > self.accounts[aid]["balance"] - self._held(aid, ts):
            return None
        acc = self.accounts[aid]
        acc["balance"] -= amount
        acc["outgoing"] += amount
        self._record(aid, ts)
        return acc["balance"]
    def top_spenders(self, ts, n):
        ranked = sorted(self.accounts.items(), key=lambda item: (-item[1]["outgoing"], item[0]))
        return [f"{aid}({acc['outgoing']})" for aid, acc in ranked[:n]]
    def transfer(self, ts, source, target, amount):
        if source == target or source not in self.accounts or target not in self.accounts:
            return None
        if amount > self.accounts[source]["balance"] - self._held(source, ts):
            return None
        self.seq += 1
        tid = f"transfer{self.seq}"
        self.transfers[tid] = {"source": source, "target": target, "amount": amount,
                               "created_at": ts, "accepted": False}
        return tid
    def accept_transfer(self, ts, aid, tid):
        t = self.transfers.get(tid)
        if t is None or t["accepted"] or t["target"] != aid or ts > t["created_at"] + 1000:
            return False
        t["accepted"] = True
        self.accounts[t["source"]]["balance"] -= t["amount"]
        self._record(t["source"], ts)
        self.accounts[t["target"]]["balance"] += t["amount"]
        self._record(t["target"], ts)
        return True
    def merge_accounts(self, ts, a, b):
        if a == b or a not in self.accounts or b not in self.accounts:
            return False
        self.accounts[a]["balance"] += self.accounts[b]["balance"]
        self._record(a, ts)
        self.accounts[a]["outgoing"] += self.accounts[b]["outgoing"]
        for t in self.transfers.values():
            if t["source"] == b:
                t["source"] = a
            if t["target"] == b:
                t["target"] = a
        del self.accounts[b]
        return True
    def get_balance(self, ts, aid, time_at):
        if aid not in self.accounts:
            return None
        value = None
        for event_ts, balance in self.accounts[aid]["events"]:
            if event_ts > time_at:
                break
            value = balance
        return value

def random_ops(rng, n, kinds):
    ids = ["A", "B", "C", "D"]
    ops, ts, transfers = [("create_account", 0, aid) for aid in ids[:2]], 0, 0
    weights = {"create": 3, "deposit": 4, "withdraw": 4, "top": 2, "transfer": 4,
               "accept": 4, "merge": 2, "balance": 3}
    for _ in range(n):
        ts += rng.choice([0, 0, 1, 2, 5, 900, 1400])
        aid, other = rng.choice(ids), rng.choice(ids)
        kind = rng.choices(kinds, weights=[weights[k] for k in kinds])[0]
        if kind == "create":
            ops.append(("create_account", ts, aid))
        elif kind == "deposit":
            ops.append(("deposit", ts, aid, rng.randint(1, 50)))
        elif kind == "withdraw":
            ops.append(("withdraw", ts, aid, rng.randint(1, 60)))
        elif kind == "top":
            ops.append(("top_spenders", ts, rng.randint(1, len(ids) + 1)))
        elif kind == "transfer":
            ops.append(("transfer", ts, aid, other, rng.randint(1, 40)))
            transfers += 1
        elif kind == "accept":
            ops.append(("accept_transfer", ts, aid, f"transfer{rng.randint(1, max(transfers, 1))}"))
        elif kind == "merge":
            ops.append(("merge_accounts", ts, aid, other))
        else:
            ops.append(("get_balance", ts, aid, rng.randint(0, max(ts, 1))))
    return ops

def agrees_with_oracle(make_bank, kinds, seeds):
    for seed in seeds:
        rng = random.Random(seed)
        bank, slow = make_bank(), SlowBank()
        for op in random_ops(rng, 25 + seed % 30, kinds):
            got = getattr(bank, op[0])(*op[1:])
            want = getattr(slow, op[0])(*op[1:])
            assert got == want, f"seed {seed}: {op} returned {got!r}, expected {want!r}"
"""

TASK = {
    "title": "Bank Ledger",
    "difficulty": "Hard",
    "version": 1,
    "function_name": "Bank",
    "description_en": r"""Build `Bank`, an in-memory ledger of accounts. It has the shape of a timed online assessment: each part adds methods to the same class.

The requirement arrives in parts. Each part keeps every earlier behavior, so one `Bank` class passes all parts at the end. Pass every test of the current part to reveal the next one.

**Rules for every method:**
- The first argument is `timestamp`, a non-negative `int`. Across all calls on one `Bank`, timestamps never decrease.
- An `account_id` is a non-empty `str`. An `amount` is a positive `int`.
- A failed call returns its failure value and changes nothing.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why this shows up in interviews:** online assessments grade each stage against cases you cannot see, on a clock. The skill is keeping the first data model flexible enough that later stages extend it instead of rewriting it.

**Where it is used:** account ledgers in payment and banking systems.

Adapted from the bank system online-assessment question in Schuture/Anthropic-Interview-Notes (CC BY-NC 4.0), reworded.""",
    "parts": [
        {
            "title": "Accounts, deposits and withdrawals",
            "description_en": r"""**Signature:** `create_account(timestamp, account_id) -> bool`, `deposit(timestamp, account_id, amount) -> int | None`, `withdraw(timestamp, account_id, amount) -> int | None`

- `create_account` opens an account with balance 0 and returns `True`. It returns `False` if the account already exists.
- `deposit` adds `amount` and returns the new balance, or `None` if the account does not exist.
- `withdraw` subtracts `amount` and returns the new balance. It returns `None` if the account does not exist or `amount` is greater than the balance.

**Example:**
- `create_account(0, "A")` is `True`; `create_account(1, "A")` is `False`
- `deposit(2, "A", 100)` is `100`; `withdraw(3, "A", 40)` is `60`
- `withdraw(4, "A", 1000)` is `None`; `deposit(5, "B", 10)` is `None`""",
        },
        {
            "title": "Top spenders",
            "description_en": r"""Each account tracks its outgoing total: the sum of every successful `withdraw` from it, starting at 0. Keep Part 1 and add:

**Signature:** `top_spenders(timestamp, n) -> list[str]`

- Return the `n` accounts with the largest outgoing total, highest first, as strings like `"A(30)"`.
- Ties are ordered by `account_id` ascending.
- If fewer than `n` accounts exist, return all of them in that order.

**Example** (A, B and C each deposited 100):
- `withdraw(4, "A", 30)`, `withdraw(5, "B", 30)`
- `top_spenders(6, 2)` is `["A(30)", "B(30)"]`
- `top_spenders(6, 5)` is `["A(30)", "B(30)", "C(0)"]`""",
        },
        {
            "title": "Two-step transfers",
            "description_en": r"""A transfer holds money first and moves it only when the target accepts. Keep Parts 1–2 and add:

**Signature:** `transfer(timestamp, source_id, target_id, amount) -> str | None`, `accept_transfer(timestamp, account_id, transfer_id) -> bool`

- An account's available balance is its balance minus the amounts of its pending outgoing transfers. From now on `withdraw` and `transfer` check `amount` against the available balance.
- A transfer created at time `t0` is pending while `t0 <= t <= t0 + 1000`, and expired after that. An expired transfer no longer holds money and can never be accepted.
- `transfer` returns `"transfer1"`, `"transfer2"`, … from one counter that advances only on success. Neither balance changes yet. It returns `None` if the two ids are equal, either account does not exist, or `amount` exceeds the source's available balance.
- `accept_transfer` succeeds only for the transfer's target, while it is pending and not yet accepted: it debits the source and credits the target, then returns `True`. Otherwise it returns `False`.
- Outgoing totals count withdrawals only, not transfers.

**Example** (A has 500):
- `transfer(10, "A", "B", 200)` is `"transfer1"`; `transfer(15, "A", "C", 250)` is `"transfer2"`
- `withdraw(20, "A", 100)` is `None`: only 50 is available
- `accept_transfer(1010, "B", "transfer1")` is `True`; doing it again is `False`
- `accept_transfer(1016, "C", "transfer2")` is `False`: it expired after 1015""",
        },
        {
            "title": "Merges and balance history",
            "description_en": r"""Keep Parts 1–3 and add:

**Signature:** `merge_accounts(timestamp, a, b) -> bool`, `get_balance(timestamp, account_id, time_at) -> int | None`

- `merge_accounts` folds `b` into `a` and returns `True`. It returns `False` if `a == b` or either does not exist.
- A merge adds `b`'s balance and outgoing total to `a`, moves `b`'s pending transfers to `a` on whichever side `b` was, and closes `b`.
- A transfer that ends up from `a` to `a` stays pending and can be accepted; accepting it releases the hold and leaves the balance unchanged.
- After a merge, `create_account(t, b)` opens a new account with balance 0 and its own history.
- `get_balance` returns the account's balance just after its last balance change timestamped at or before `time_at`: a deposit, withdrawal, accepted transfer or merge credit. An accepted transfer counts at its accept time.
- `get_balance` returns `None` if the account does not exist, or if `time_at` is before this account's own `create_account`. `time_at` may be any non-negative `int`.

**Example:**
- A has 100, B has 50, and `transfer(3, "B", "A", 20)` is pending
- `merge_accounts(5, "A", "B")` is `True`; `get_balance(6, "A", 4)` is `100` and `get_balance(6, "A", 5)` is `150`
- `accept_transfer(10, "A", "transfer1")` is `True` and A stays at 150
- `create_account(12, "B")`, then `get_balance(13, "B", 4)` is `None` and `get_balance(13, "B", 12)` is `0`""",
        },
    ],
    "hints": [
        {"level": 1, "kind": "questions", "content": "What does each account need to store, and which method creates that record? Which comparison decides that a withdrawal is too large, and is withdrawing the whole balance allowed? What should a call do first, so that a refused call changes nothing?"},
        {"level": 2, "kind": "analysis", "content": "Keep one small record per account id in a dict. Route every balance change through one helper, so that each later requirement changes one place. In each method, validate first and mutate last, returning the failure value before touching anything."},
    ],
    "model_connections": [
        "Card networks place an authorization hold that reduces the available balance until the merchant captures or the hold expires, which is the part 3 transfer.",
        "Point-in-time balance queries in ledgers read an append-only history with binary search, as get_balance does.",
    ],
    "pro_con_analysis": {
        "pros": [
            "Holds kept apart from the settled balance let a pending transfer reserve money without moving it.",
            "An append-only balance history answers any past timestamp in O(log n) with bisect.",
            "Validating before mutating keeps every failed call free of side effects.",
        ],
        "cons": [
            "Recomputing the available balance by scanning transfers grows with history unless expired holds are pruned.",
            "A merge deletes b, so nothing records which of a's transfers came from b; undoing a merge needs a snapshot.",
            "Ranking by sorting every account on each top_spenders call is O(n log n) per query.",
        ],
    },
    "tests": [
        {"name": "Part 1: the worked example", "part": 1, "behavior": "state.invariant", "code": r"""
bank = {fn}()
assert bank.create_account(0, "A") is True
assert bank.create_account(1, "A") is False
assert bank.deposit(2, "A", 100) == 100
assert bank.withdraw(3, "A", 40) == 60
assert bank.withdraw(4, "A", 1000) is None
assert bank.deposit(5, "B", 10) is None
assert bank.withdraw(6, "B", 1) is None
"""},
        {"name": "Part 1: withdrawing the whole balance", "part": 1, "visibility": "unshown", "behavior": "edge.empty_or_boundary",
         "failure_message": "withdraw must allow an amount equal to the balance and refuse one greater, leaving the balance unchanged on refusal.",
         "code": r"""
bank = {fn}()
bank.create_account(0, "A")
bank.deposit(1, "A", 30)
assert bank.withdraw(2, "A", 31) is None
assert bank.withdraw(3, "A", 30) == 0
assert bank.withdraw(4, "A", 1) is None
assert bank.deposit(5, "A", 7) == 7
"""},
        {"name": "Part 1: random sequences match a reference", "part": 1, "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "A random sequence of creates, deposits and withdrawals returned a different value than the rules give.",
         "code": _ORACLE + r"""
agrees_with_oracle({fn}, ["create", "deposit", "withdraw"], range(150))
"""},
        {"name": "Part 2: ranking with ties", "part": 2, "behavior": "metrics.ties", "code": r"""
bank = {fn}()
for aid in ("A", "B", "C"):
    bank.create_account(0, aid)
for i, aid in enumerate(("A", "B", "C")):
    bank.deposit(1 + i, aid, 100)
assert bank.withdraw(4, "A", 30) == 70
assert bank.withdraw(5, "B", 30) == 70
assert bank.top_spenders(6, 2) == ["A(30)", "B(30)"]
assert bank.top_spenders(6, 5) == ["A(30)", "B(30)", "C(0)"]
assert bank.withdraw(7, "A", 1000) is None
assert bank.top_spenders(8, 1) == ["A(30)"]
"""},
        {"name": "Part 2: failed withdrawals do not count", "part": 2, "visibility": "unshown", "behavior": "metrics.ties",
         "failure_message": "Only successful withdrawals add to the outgoing total, and ties sort by account_id ascending.",
         "code": r"""
bank = {fn}()
for aid in ("zed", "amy", "kim"):
    bank.create_account(0, aid)
bank.deposit(1, "zed", 10)
bank.deposit(1, "kim", 10)
assert bank.withdraw(2, "zed", 50) is None
assert bank.withdraw(3, "kim", 4) == 6
assert bank.withdraw(3, "zed", 4) == 6
assert bank.top_spenders(4, 3) == ["kim(4)", "zed(4)", "amy(0)"]
"""},
        {"name": "Part 2: random sequences match a reference", "part": 2, "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "A random sequence including top_spenders returned a different value than the rules give.",
         "code": _ORACLE + r"""
agrees_with_oracle({fn}, ["create", "deposit", "withdraw", "top"], range(150))
"""},
        {"name": "Part 3: holds, acceptance and expiry", "part": 3, "behavior": "budget.enforcement", "code": r"""
bank = {fn}()
for aid in ("A", "B", "C"):
    bank.create_account(0, aid)
bank.deposit(1, "A", 500)
assert bank.transfer(10, "A", "B", 200) == "transfer1"
assert bank.transfer(15, "A", "C", 250) == "transfer2"
assert bank.withdraw(20, "A", 100) is None
assert bank.accept_transfer(1010, "B", "transfer1") is True
assert bank.accept_transfer(1010, "B", "transfer1") is False
assert bank.accept_transfer(1011, "A", "transfer2") is False
assert bank.accept_transfer(1016, "C", "transfer2") is False
assert bank.withdraw(1300, "A", 300) == 0
"""},
        {"name": "Part 3: transfer validation and the shared counter", "part": 3, "visibility": "unshown", "behavior": "protocol.validation",
         "failure_message": "transfer must reject a self-transfer, a missing account or too large an amount without advancing the transfer counter.",
         "code": r"""
bank = {fn}()
bank.create_account(0, "A"); bank.create_account(0, "B")
bank.deposit(1, "A", 50)
assert bank.transfer(2, "A", "A", 5) is None
assert bank.transfer(2, "A", "Z", 5) is None
assert bank.transfer(2, "Z", "A", 5) is None
assert bank.transfer(2, "A", "B", 51) is None
assert bank.transfer(3, "A", "B", 50) == "transfer1"
assert bank.transfer(3, "A", "B", 1) is None
assert bank.accept_transfer(4, "B", "transfer9") is False
assert bank.deposit(5, "B", 1) == 1
assert bank.transfer(6, "B", "A", 1) == "transfer2"
assert bank.top_spenders(7, 2) == ["A(0)", "B(0)"], "transfers are not withdrawals"
"""},
        {"name": "Part 3: the window boundary", "part": 3, "visibility": "unshown", "behavior": "edge.empty_or_boundary",
         "failure_message": "A transfer is pending through t0 + 1000 inclusive and expired after; an expired hold stops reducing the available balance.",
         "code": r"""
bank = {fn}()
bank.create_account(0, "A"); bank.create_account(0, "B")
bank.deposit(0, "A", 100)
assert bank.transfer(100, "A", "B", 100) == "transfer1"
assert bank.withdraw(1100, "A", 1) is None, "still pending at t0 + 1000"
assert bank.transfer(1101, "A", "B", 100) == "transfer2", "expired one tick later"
assert bank.accept_transfer(1101, "B", "transfer1") is False
assert bank.accept_transfer(2101, "B", "transfer2") is True
assert bank.withdraw(2102, "A", 1) is None
assert bank.deposit(2103, "B", 5) == 105
"""},
        {"name": "Part 3: random sequences match a reference", "part": 3, "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "A random sequence including transfers returned a different value than the rules give.",
         "code": _ORACLE + r"""
agrees_with_oracle({fn}, ["create", "deposit", "withdraw", "top", "transfer", "accept"], range(200))
"""},
        {"name": "Part 4: merges and history", "part": 4, "behavior": "state.invariant", "code": r"""
bank = {fn}()
bank.create_account(0, "A"); bank.create_account(0, "B")
assert bank.deposit(1, "A", 100) == 100
assert bank.deposit(2, "B", 50) == 50
assert bank.transfer(3, "B", "A", 20) == "transfer1"
assert bank.merge_accounts(5, "A", "B") is True
assert bank.get_balance(6, "B", 4) is None
assert bank.get_balance(6, "A", 4) == 100
assert bank.get_balance(6, "A", 5) == 150
assert bank.accept_transfer(10, "A", "transfer1") is True
assert bank.get_balance(11, "A", 10) == 150
assert bank.create_account(12, "B") is True
assert bank.get_balance(13, "B", 4) is None
assert bank.get_balance(13, "B", 12) == 0
assert bank.deposit(14, "B", 9) == 9
assert bank.get_balance(15, "B", 14) == 9
assert bank.merge_accounts(16, "A", "A") is False
assert bank.merge_accounts(17, "A", "nope") is False
"""},
        {"name": "Part 4: transfers follow a merge on both sides", "part": 4, "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "A merge must move b's pending transfers to a, whether b was the source or the target.",
         "code": r"""
bank = {fn}()
for aid in ("A", "B", "C", "D"):
    bank.create_account(0, aid)
bank.deposit(1, "B", 100)
bank.deposit(1, "D", 100)
assert bank.transfer(2, "B", "C", 40) == "transfer1"
assert bank.transfer(3, "D", "B", 25) == "transfer2"
assert bank.merge_accounts(4, "A", "B") is True
assert bank.withdraw(4, "A", 61) is None, "the 40 hold now counts against A"
assert bank.accept_transfer(5, "C", "transfer1") is True
assert bank.get_balance(6, "A", 5) == 60
assert bank.accept_transfer(7, "B", "transfer2") is False
assert bank.accept_transfer(7, "A", "transfer2") is True
assert bank.get_balance(8, "A", 7) == 85
"""},
        {"name": "Part 4: chained merges and outgoing totals", "part": 4, "visibility": "unshown", "behavior": "metrics.ties",
         "failure_message": "A merge must add b's balance and outgoing total to a, and close b.",
         "code": r"""
bank = {fn}()
for aid in ("A", "B", "C"):
    bank.create_account(0, aid)
bank.deposit(1, "A", 10); bank.deposit(1, "B", 20); bank.deposit(1, "C", 30)
bank.withdraw(2, "B", 5); bank.withdraw(2, "C", 7)
assert bank.merge_accounts(3, "A", "B") is True
assert bank.merge_accounts(4, "A", "C") is True
assert bank.get_balance(5, "A", 4) == 10 + 15 + 23
assert bank.top_spenders(5, 3) == ["A(12)"]
assert bank.deposit(6, "B", 1) is None
assert bank.get_balance(7, "A", 10 ** 9) == 48
"""},
        {"name": "Part 4: random sequences match a reference", "part": 4, "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "A random sequence including merges and balance queries returned a different value than the rules give.",
         "code": _ORACLE + r"""
agrees_with_oracle({fn}, ["create", "deposit", "withdraw", "top", "transfer", "accept", "merge", "balance"], range(300))
"""},
    ],
    "solution": r'''import bisect

WINDOW = 1000


class _Account:
    def __init__(self, timestamp):
        self.balance = 0
        self.outgoing = 0
        self.times = [timestamp]
        self.balances = [0]

    def change(self, timestamp, delta):
        self.balance += delta
        self.times.append(timestamp)
        self.balances.append(self.balance)


class _Transfer:
    def __init__(self, source, target, amount, created):
        self.source = source
        self.target = target
        self.amount = amount
        self.created = created
        self.accepted = False

    def pending(self, timestamp):
        return not self.accepted and timestamp <= self.created + WINDOW


class Bank:
    def __init__(self):
        self._live = {}
        self._transfers = {}

    def _available(self, account_id, timestamp):
        held = sum(t.amount for t in self._transfers.values()
                   if t.source == account_id and t.pending(timestamp))
        return self._live[account_id].balance - held

    def create_account(self, timestamp, account_id):
        if account_id in self._live:
            return False
        self._live[account_id] = _Account(timestamp)
        return True

    def deposit(self, timestamp, account_id, amount):
        account = self._live.get(account_id)
        if account is None:
            return None
        account.change(timestamp, amount)
        return account.balance

    def withdraw(self, timestamp, account_id, amount):
        account = self._live.get(account_id)
        if account is None or amount > self._available(account_id, timestamp):
            return None
        account.change(timestamp, -amount)
        account.outgoing += amount
        return account.balance

    def top_spenders(self, timestamp, n):
        ranked = sorted(self._live.items(), key=lambda item: (-item[1].outgoing, item[0]))
        return [f"{account_id}({account.outgoing})" for account_id, account in ranked[:n]]

    def transfer(self, timestamp, source_id, target_id, amount):
        if source_id == target_id or source_id not in self._live or target_id not in self._live:
            return None
        if amount > self._available(source_id, timestamp):
            return None
        transfer_id = f"transfer{len(self._transfers) + 1}"
        self._transfers[transfer_id] = _Transfer(source_id, target_id, amount, timestamp)
        return transfer_id

    def accept_transfer(self, timestamp, account_id, transfer_id):
        t = self._transfers.get(transfer_id)
        if t is None or t.target != account_id or not t.pending(timestamp):
            return False
        t.accepted = True
        self._live[t.source].change(timestamp, -t.amount)
        self._live[t.target].change(timestamp, t.amount)
        return True

    def merge_accounts(self, timestamp, a, b):
        if a == b or a not in self._live or b not in self._live:
            return False
        keep, gone = self._live[a], self._live.pop(b)
        keep.change(timestamp, gone.balance)
        keep.outgoing += gone.outgoing
        for t in self._transfers.values():
            if t.pending(timestamp):
                if t.source == b:
                    t.source = a
                if t.target == b:
                    t.target = a
        return True

    def get_balance(self, timestamp, account_id, time_at):
        account = self._live.get(account_id)
        if account is None:
            return None
        index = bisect.bisect_right(account.times, time_at)
        return account.balances[index - 1] if index else None
''',
    "interview_questions": interview(
        concept=[
            "What does each account need to store for these three methods, and why should a failed call change nothing?",
            "Is withdrawing exactly the whole balance allowed, and what comparison expresses that?",
        ],
        deep_dive=[
            "How do you keep the validate-then-mutate order in every method, so a refused call never half-applies?",
        ],
        tradeoffs=[
            "How would you rank accounts by outgoing total, and what does sorting on every call cost compared with keeping a sorted structure?",
            "Why hold money for a pending transfer instead of moving it at once, and what makes the available balance cheap to compute?",
            "What history must an account keep so get_balance can answer any past time, and how does a merge fit into it?",
        ],
    ),
}
