There are $t$ tasks, indexed $0, 1, \dots, t-1$; $m$ models, indexed $0, \dots, m-1$; and $h$ human labelers, indexed $0, \dots, h-1$. Labeling a task means showing a human labeler one model's output on that task and asking for a rating. An *assignment* is a triple $(\mathrm{task}, \mathrm{model}, \mathrm{human})$ recording one such review. A *schedule* is an ordered list $S = (a_0, a_1, \dots, a_{L-1})$ of assignments, $a_j = (\mathrm{task}_j, \mathrm{model}_j, \mathrm{human}_j)$, in the order the platform hands them out. The order matters: every rule below is stated for each *prefix* of $S$ — for $0 \le p \le L$, the prefix of length $p$ is $S[:p] = (a_0, \dots, a_{p-1})$.

An integer $k \ge 0$ is also given. Every part below shares two base rules:

- **Coverage.** For every human $u$, at least $k$ assignments of $S$ have $\mathrm{human}_j = u$.
- **Uniqueness.** No two assignments share both the same task and the same human: for $i \ne j$, $(\mathrm{task}_i, \mathrm{human}_i) \ne (\mathrm{task}_j, \mathrm{human}_j)$.

Every part also asks for a schedule with the fewest possible assignments.

### Part 1 — Coverage and uniqueness

Implement `build_basic_schedule`. Model choice is unconstrained in this part.

```py
from typing import List, Optional, Tuple
Assignment = Tuple[int, int, int]  # (task, model, human)

def build_basic_schedule(t: int, m: int, h: int, k: int) -> Optional[List[Assignment]]:
    """Returns None if t <= 0, m <= 0, or h <= 0 (invalid dimensions, regardless of k). Otherwise
    returns [] if k == 0, or None if k > t (infeasible). Otherwise
    returns a schedule of exactly h * k assignments satisfying coverage and uniqueness."""
```

Example, with $t = 2$, $m = 2$, $h = 3$, $k = 2$: one valid schedule is

```text
(0, 0, 0), (1, 0, 1), (0, 0, 2), (1, 0, 0), (0, 0, 1), (1, 0, 2)
```

Each human appears exactly twice, once on each task; the model is always 0, which this part allows.

### Part 2 — Balanced by task

For a prefix $S[:p]$, a task $x$, and a model $i$, let

$$c_p(x, i) = \sum_{j=0}^{p-1} [\mathrm{task}_j = x \text{ and } \mathrm{model}_j = i]$$

count how many times model $i$ has been used on task $x$ within that prefix. A schedule is *balanced by task* if for every prefix $p$ and every task $x$,

$$\max_i c_p(x, i) - \min_i c_p(x, i) \le 1 ,$$

the maximum and minimum ranging over all $m$ models. Implement `build_balanced_schedule`, returning a schedule with coverage, uniqueness, and balance by task.

```py
def build_balanced_schedule(t: int, m: int, h: int, k: int) -> Optional[List[Assignment]]:
    """Same feasibility contract as build_basic_schedule. The returned schedule is also balanced
    by task at every prefix, with exactly h * k assignments."""
```

Example, same $t, m, h, k$ as Part 1: one minimal balanced schedule is

```text
(0, 0, 0), (1, 0, 1), (0, 1, 2), (1, 1, 0), (0, 0, 1), (1, 0, 2)
```

Task 0 appears at positions 0, 2 and 4; its per-model counts evolve $(1, 0) \to (1, 1) \to (2, 1)$, so the difference never exceeds 1. Task 1 behaves identically.

### Part 3 — Also balanced by human

Define $c_p(u, i)$ for a human $u$ the same way as $c_p(x, i)$ above, counting assignments with $\mathrm{human}_j = u$ and $\mathrm{model}_j = i$ instead. A schedule is *balanced by human* if $\max_i c_p(u, i) - \min_i c_p(u, i) \le 1$ for every prefix $p$ and every human $u$.

Implement `build_doubly_balanced_schedule`, which must keep coverage, uniqueness, and balance by task exactly as in Part 2, and additionally tries to also achieve balance by human, using a greedy rule of your choice: at each step, pick the model for the next assignment from the running counts. Then decide whether your rule actually guarantees balance by human at every prefix, for every valid $t, m, h, k$ — prove it, or find the smallest counterexample you can: parameters, and the prefix at which it fails.

```py
def build_doubly_balanced_schedule(t: int, m: int, h: int, k: int) -> Optional[List[Assignment]]:
    """Same feasibility contract and minimal length h * k as build_balanced_schedule, and always
    balanced by task. Attempts, via a greedy rule, to also be balanced by human."""
```

Example, with $t = 2$, $m = 3$, $h = 4$, $k = 2$: one schedule that is balanced both by task and by human is

```text
(0, 0, 0), (1, 0, 1), (0, 1, 2), (1, 1, 3), (1, 2, 0), (0, 2, 1), (1, 0, 2), (0, 0, 3)
```

Task 0 appears four times, at positions 0, 2, 5 and 7, using models 0, 1, 2, 0 in turn; every human's two models differ from each other.
