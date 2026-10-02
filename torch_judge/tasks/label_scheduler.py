"""Schedules of (task, model, human) reviews: coverage and uniqueness, then balance by task, then a greedy check."""

from ._interview import interview

# A checker written straight from the statement: it replays a schedule prefix by prefix.
_HELPERS = r"""
import random

def contract(t, m, h, k):
    if t <= 0 or m <= 0 or h <= 0 or k > t:
        return None
    return [] if k == 0 else "list"

def check(schedule, t, m, h, k, task_balance=False):
    want = contract(t, m, h, k)
    if want != "list":
        assert schedule == want, f"({t}, {m}, {h}, {k}) should return {want!r}, got {schedule!r}"
        return
    assert isinstance(schedule, list) and len(schedule) == h * k, f"({t}, {m}, {h}, {k}): expected {h * k} assignments"
    pairs, per_human, per_task = set(), [0] * h, [[0] * m for _ in range(t)]
    for p, a in enumerate(schedule):
        task, model, human = a
        assert 0 <= task < t and 0 <= model < m and 0 <= human < h, f"assignment {a} is out of range"
        assert (task, human) not in pairs, f"human {human} reviews task {task} twice"
        pairs.add((task, human))
        per_human[human] += 1
        per_task[task][model] += 1
        if task_balance:
            row = per_task[task]
            assert max(row) - min(row) <= 1, f"task {task} is unbalanced after {p + 1} assignments: {row}"
    assert all(c >= k for c in per_human), f"some human has fewer than {k} assignments"

class Model:
    def greedy(self, t, m, h, k):
        if contract(t, m, h, k) != "list":
            return contract(t, m, h, k)
        tc, hc, out = [[0] * m for _ in range(t)], [[0] * m for _ in range(h)], []
        for r in range(k):
            for u in range(h):
                x = (u + r) % t
                i = min(range(m), key=lambda i: (tc[x][i], hc[u][i], i))
                tc[x][i] += 1
                hc[u][i] += 1
                out.append((x, i, u))
        return out
    def imbalance(self, schedule, m):
        for p in range(1, len(schedule) + 1):
            for u in {a[2] for a in schedule[:p]}:
                row = [sum(1 for a in schedule[:p] if a[2] == u and a[1] == i) for i in range(m)]
                if max(row) - min(row) > 1:
                    return p
        return None

SMALL = [(t, m, h, k) for t in range(-1, 5) for m in range(-1, 4) for h in range(-1, 5) for k in range(0, 6)]
"""
TASK = {
    "title": "Labeling Task Scheduler",
    "difficulty": "Medium",
    "version": 1,
    "function_name": "LabelScheduler",
    "description_en": r"""Build `LabelScheduler`, which plans which human rates which model's output on which task.

The requirement arrives in parts. Each part keeps every earlier behavior, so one `LabelScheduler` class passes all parts at the end. Pass every test of the current part to reveal the next one.

**Rules for every part:**
- There are `t` tasks, `m` models and `h` humans, numbered from `0`. An assignment is a tuple `(task, model, human)`: that human rates that model's output on that task.
- A schedule is a `list` of assignments in the order they are handed out. A prefix is its first `p` assignments, for any `p`.
- Coverage: every human appears in at least `k` assignments. Uniqueness: no two assignments share both `task` and `human`.
- `k` is an `int` of `0` or more. Every builder returns `None` when `t`, `m` or `h` is `0` or less, whatever `k` is. Otherwise it returns `[]` when `k == 0`, and `None` when `k > t`.
- Otherwise it returns a schedule of exactly `h * k` assignments with coverage and uniqueness. That is the fewest possible.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why this shows up in interviews:** the first schedule is a short loop; the work is a rule that holds at every prefix rather than only at the end, and each later part adds one requirement.

**Where it is used:** human preference data for comparing models is collected by showing raters model outputs, and a rater who sees one model far more often than others skews the comparison.

Adapted from the labeling task scheduler question in Schuture/OpenAI-Interview-Notes (CC BY-NC 4.0), reworded, on one class. The open-ended last part is narrowed to one fixed greedy rule plus a checker that finds where it fails.""",
    "parts": [
        {
            "title": "Coverage and uniqueness",
            "description_en": r"""**Signature:** `LabelScheduler()`, `basic_schedule(t, m, h, k) -> list[tuple[int, int, int]] | None`

- Follow the rules above. Any `model` from `0` to `m - 1` is allowed in this part.
- Any schedule that meets the rules is accepted; the order is up to you.

**Example**, `t = 3`, `m = 2`, `h = 2`, `k = 2`. One valid answer:
- `[(0, 0, 0), (1, 0, 1), (1, 0, 0), (2, 0, 1)]`
- human `0` rates tasks `0` and `1`, human `1` rates tasks `1` and `2`: each human twice, never the same task twice
- `basic_schedule(3, 2, 2, 4)` is `None`: a human can rate at most `3` different tasks""",
        },
        {
            "title": "Balanced by task",
            "description_en": r"""Keep Part 1 and add a builder whose models are spread evenly over each task.

**Signature:** `balanced_schedule(t, m, h, k) -> list[tuple[int, int, int]] | None`

- Same contract and length as `basic_schedule`.
- For a task `x`, count how often each of the `m` models has been used on `x` so far. In every prefix, for every task, the largest and smallest of those `m` counts differ by at most `1`.

**Example**, `t = 3`, `m = 2`, `h = 4`, `k = 2`. One valid answer:
- `[(0, 0, 0), (1, 0, 1), (2, 0, 2), (0, 1, 3), (1, 1, 0), (2, 1, 1), (0, 0, 2), (1, 0, 3)]`
- task `0` uses models `0`, `1`, `0` in that order: its counts go `(1, 0)`, `(1, 1)`, `(2, 1)` and never differ by more than `1`""",
        },
        {
            "title": "A greedy rule and its check",
            "description_en": r"""Keep Parts 1–2. Add one fixed greedy builder, and a checker for balance by human.

**Signature:** `greedy_schedule(t, m, h, k) -> list[tuple[int, int, int]] | None`, `first_human_imbalance(schedule, m) -> int | None`

- `greedy_schedule` has the same contract and length, and hands out assignments in this order: for `r` from `0` to `k - 1`, for `u` from `0` to `h - 1`, human `u` gets task `(u + r) % t`.
- Its model is the one with the smallest count on that task so far. Ties go to the smallest count for that human so far, then to the smallest model number.
- Balance by human is balance by task with humans in place of tasks: for each human, the counts of the `m` models in that human's assignments.
- `first_human_imbalance(schedule, m)` returns the smallest `p` such that the prefix of length `p` is not balanced by human, or `None` if every prefix is.

**Example:**
- `greedy_schedule(3, 2, 4, 2)` is `[(0, 0, 0), (1, 0, 1), (2, 0, 2), (0, 1, 3), (1, 1, 0), (2, 1, 1), (0, 1, 2), (1, 0, 3)]`
- `first_human_imbalance` of that schedule with `m = 2` is `None`
- `first_human_imbalance([(0, 1, 0), (2, 0, 1), (1, 1, 0)], 2)` is `3`: human `0` now has counts `(0, 2)`""",
        },
    ],
    "hints": [
        {"level": 1, "kind": "questions", "content": "Uniqueness means a human can rate each task once. How many assignments can one human get at most, and what does that say about k > t? If every human gets exactly k, how long is the schedule, and can any valid schedule be shorter?"},
        {"level": 2, "kind": "analysis", "content": "Validate first: None for a non-positive t, m or h; [] for k == 0; None for k > t. Then loop r over range(k) and u over range(h) and append ((u + r) % t, 0, u). For one human, r takes k <= t different values, so (u + r) % t never repeats a task."},
    ],
    "model_connections": [
        "Preference data for RLHF and model evals is collected by showing raters model outputs, and the mix of models per rater and per prompt affects how fair the comparison is.",
        "Arena-style evaluations assign prompts and model pairs to raters so that each model is seen about equally often.",
    ],
    "pro_con_analysis": {
        "pros": [
            "Rotating tasks by round gives every human k different tasks with a simple modular rule.",
            "Round-robin over models per task keeps task balance at every prefix, not only at the end.",
            "A prefix checker turns a claim about a greedy rule into something you can test and find a counterexample for.",
        ],
        "cons": [
            "Balancing tasks first can leave a human with the same model twice in a row, and no tie-break fixes that.",
            "The fixed round order ignores rater availability; a real platform hands out work as raters arrive.",
            "Exact balance at every prefix is stricter than many studies need and limits other scheduling goals.",
        ],
    },
    "tests": [
        {"name": "Part 1: the worked example", "part": 1, "behavior": "contract.signature", "code": _HELPERS + r"""
s = {fn}()
check(s.basic_schedule(3, 2, 2, 2), 3, 2, 2, 2)
assert s.basic_schedule(3, 2, 2, 4) is None
assert s.basic_schedule(3, 2, 2, 0) == []
assert s.basic_schedule(0, 2, 2, 1) is None
"""},
        {"name": "Part 1: every small case", "part": 1, "visibility": "unshown", "behavior": "edge.empty_or_boundary",
         "failure_message": "For some small t, m, h, k the result broke the contract (None, [] or exactly h * k assignments) or a schedule repeated a (task, human) pair or left a human below k.",
         "code": _HELPERS + r"""
s = {fn}()
for t, m, h, k in SMALL:
    check(s.basic_schedule(t, m, h, k), t, m, h, k)
assert s.basic_schedule(2, 0, 3, 0) is None, "an invalid dimension returns None even when k == 0"
"""},
        {"name": "Part 1: a large schedule", "part": 1, "visibility": "unshown", "behavior": "performance.complexity",
         "failure_message": "A schedule of 240,000 assignments was wrong or too slow; build it directly instead of searching.",
         "code": _HELPERS + r"""
check({fn}().basic_schedule(400, 5, 3000, 80), 400, 5, 3000, 80)
"""},
        {"name": "Part 2: the worked example", "part": 2, "behavior": "state.invariant", "code": _HELPERS + r"""
s = {fn}()
check(s.balanced_schedule(3, 2, 4, 2), 3, 2, 4, 2, task_balance=True)
assert s.balanced_schedule(3, 2, 4, 5) is None
"""},
        {"name": "Part 2: balanced at every prefix", "part": 2, "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "Some prefix had a task whose model counts differed by more than 1, or the schedule broke the Part 1 rules.",
         "code": _HELPERS + r"""
s = {fn}()
for t, m, h, k in SMALL:
    check(s.balanced_schedule(t, m, h, k), t, m, h, k, task_balance=True)
rng = random.Random(3)
for _ in range(40):
    t, m, h = rng.randint(1, 9), rng.randint(1, 6), rng.randint(1, 30)
    k = rng.randint(0, t)
    check(s.balanced_schedule(t, m, h, k), t, m, h, k, task_balance=True)
check(s.balanced_schedule(300, 7, 2000, 60), 300, 7, 2000, 60, task_balance=True)
"""},
        {"name": "Part 3: the worked example", "part": 3, "behavior": "state.invariant", "code": _HELPERS + r"""
s = {fn}()
g = s.greedy_schedule(3, 2, 4, 2)
assert g == [(0, 0, 0), (1, 0, 1), (2, 0, 2), (0, 1, 3), (1, 1, 0), (2, 1, 1), (0, 1, 2), (1, 0, 3)], g
assert s.first_human_imbalance(g, 2) is None
assert s.first_human_imbalance([(0, 1, 0), (2, 0, 1), (1, 1, 0)], 2) == 3
"""},
        {"name": "Part 3: greedy picks and the checker", "part": 3, "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "greedy_schedule differed from the stated order and model rule (task count, then human count, then model number), or first_human_imbalance returned the wrong prefix length.",
         "code": _HELPERS + r"""
s, model = {fn}(), Model()
for t, m, h, k in SMALL:
    assert s.greedy_schedule(t, m, h, k) == model.greedy(t, m, h, k), (t, m, h, k)
for t, m, h, k in [(2, 3, 7, 2), (3, 3, 8, 2), (4, 3, 9, 3), (5, 4, 11, 4)]:
    g = s.greedy_schedule(t, m, h, k)
    assert g == model.greedy(t, m, h, k), (t, m, h, k)
    assert s.first_human_imbalance(g, m) == model.imbalance(g, m), (t, m, h, k)
rng = random.Random(5)
for _ in range(300):
    m, n = rng.randint(1, 4), rng.randint(0, 12)
    sched = [(rng.randint(0, 3), rng.randint(0, m - 1), rng.randint(0, 3)) for _ in range(n)]
    assert s.first_human_imbalance(sched, m) == model.imbalance(sched, m), (sched, m)
assert s.first_human_imbalance([], 3) is None
assert s.first_human_imbalance([(0, 0, 5)], 3) is None, "one assignment gives counts like (1, 0, 0): a difference of 1"
"""},
    ],
    "solution": r'''# Adapted from Schuture/OpenAI-Interview-Notes (code under the MIT License).
class LabelScheduler:
    def _invalid(self, t, m, h, k):
        if t <= 0 or m <= 0 or h <= 0:
            return None
        if k == 0:
            return []
        if k > t:  # a human can see each task once, so at most t assignments each
            return None
        return False

    def basic_schedule(self, t, m, h, k):
        early = self._invalid(t, m, h, k)
        if early is not False:
            return early
        # round r gives human u task (u + r) % t; k <= t rounds never repeat a task for one human
        return [((u + r) % t, 0, u) for r in range(k) for u in range(h)]

    def balanced_schedule(self, t, m, h, k):
        early = self._invalid(t, m, h, k)
        if early is not False:
            return early
        seen = [0] * t  # occurrences of each task so far; its c-th occurrence uses model c % m
        schedule = []
        for task, _, u in self.basic_schedule(t, m, h, k):
            schedule.append((task, seen[task] % m, u))
            seen[task] += 1
        return schedule

    def greedy_schedule(self, t, m, h, k):
        early = self._invalid(t, m, h, k)
        if early is not False:
            return early
        task_count = [[0] * m for _ in range(t)]
        human_count = [[0] * m for _ in range(h)]
        schedule = []
        for task, _, u in self.basic_schedule(t, m, h, k):
            model = min(range(m), key=lambda i: (task_count[task][i], human_count[u][i], i))
            task_count[task][model] += 1
            human_count[u][model] += 1
            schedule.append((task, model, u))
        return schedule

    def first_human_imbalance(self, schedule, m):
        counts = {}
        for p, (_, model, u) in enumerate(schedule, start=1):
            row = counts.setdefault(u, [0] * m)
            row[model] += 1
            if max(row) - min(row) > 1:  # only human u changed, so only u can newly break balance
                return p
        return None
''',
    "interview_questions": interview(
        concept=[
            "Why can no schedule meet coverage when k is larger than t?",
            "Why is h * k the fewest assignments any valid schedule can have?",
        ],
        deep_dive=[
            "How does rotating each human's task by round guarantee no repeated (task, human) pair?",
        ],
        tradeoffs=[
            "Why does giving a task's c-th occurrence model c % m keep that task balanced at every prefix?",
            "Why does always picking a model with the smallest count keep the counts within 1 of each other?",
            "Can the greedy rule that balances tasks first also guarantee balance by human, and how would you find out?",
            "How would you schedule if raters arrive one at a time instead of in fixed rounds?",
        ],
    ),
}
