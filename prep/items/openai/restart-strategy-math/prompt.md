An LLM inference service needs a random time $T \ge 0$, measured in minutes, to complete one request. The distribution of $T$ is unknown, and only its mean $E[T] = 1$ is given.

A running request may be abandoned and *restarted*. Every attempt draws a new completion time from the same distribution, independently of all earlier attempts, and restarting itself takes no time. An attempt with *timeout* $t$ runs for at most $t$ minutes: it *succeeds* if its completion time satisfies $T \le t$, and otherwise it is abandoned after $t$ minutes. A *schedule* is a list of timeouts $(t_1, \dots, t_m)$ with $t_1 + \dots + t_m = 10$, the total budget in minutes. Attempt $i$ runs with timeout $t_i$ and starts when attempt $i - 1$ is abandoned. The schedule succeeds if one of its attempts succeeds.

Example: the schedule $(2, 3, 5)$, when the three attempts draw the completion times 2.6, 3.4 and 4.1.

```text
attempt 1: minutes 0 to 2     2.6 > 2, abandoned
attempt 2: minutes 2 to 5     3.4 > 3, abandoned
attempt 3: minutes 5 to 10    4.1 <= 5, succeeds at minute 5 + 4.1 = 9.1
```

The *guarantee* of a schedule is the largest number $g$ such that the schedule succeeds with probability at least $g$ under every distribution with $T \ge 0$ and $E[T] = 1$.

Each of the six parts asks for a derivation. There is nothing to implement.

### Part 1 — A tail bound

What is the best upper bound on $P(T > 5)$ that holds for every such distribution? Is there a distribution for which it holds with equality?

### Part 2 — One restart

Find the guarantee of the schedule $(5, 5)$ and compare it with that of the single attempt $(10)$.

### Part 3 — Several restarts, budget split equally

With $N$ restarts there are $N + 1$ attempts. The budget is divided equally among them, so every timeout is $10 / (N + 1)$. Find the guarantee for every $N \ge 0$.

### Part 4 — The best number of equal attempts

Which number of attempts $m = N + 1$ gives the schedule of Part 3 its largest guarantee, and how large is that guarantee?

### Part 5 — A distribution for which restarting changes nothing

Suppose for this part that the distribution of $T$ is known. Is there a distribution with $E[T] = 1$ under which all schedules, including the single attempt $(10)$, succeed with the same probability?

### Part 6 — Unequal timeouts

The distribution is again unknown apart from $E[T] = 1$, and the timeouts of a schedule may differ. Is there a schedule with a larger guarantee than the best schedule of Part 4? Find the best schedule with two attempts, then describe what happens with $m$ attempts. Finally, say how the choice of a schedule changes when the distribution is known.
