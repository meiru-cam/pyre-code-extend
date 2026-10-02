An agent framework runs $A$ agents at once, agent $a$ numbered $0, \dots, A - 1$. Agent $a$ issues a fixed batch of *tool calls* — writing code, running the test suite, running a linter, and so on — numbered $0, \dots, T_a - 1$ within that agent. Every tool call has an integer *duration* between 1 and $10^9$ and a (possibly empty) list of *dependencies*: other tool calls' ids, always belonging to that same agent, that must run to completion first. Those dependencies may contain a cycle: if any tool call can never run because it depends, directly or transitively, on one, every function below raises `CycleError` instead of returning a schedule.

Time advances over the integers. A schedule assigns every tool call a *start time* under the following rules:

- A tool call occupies one *slot* of the available concurrency from its start time up to (but not including) `start_time + duration`; while it holds a slot, nothing else may use that slot.
- A call may start at an instant $t$ only if every one of its dependencies has `start_time + duration` at most $t$: a dependency that finishes at $t$ unlocks its dependents at $t$, not before.
- At any instant where some calls finish and others could start, every finishing call is retired first — freeing its slot and checking whether that unlocks a dependent — and only once every finishing call has been retired are new calls started into the freed slots.
- Once started, a call runs to completion without interruption: there is no preemption.
- No slot is ever left idle while eligible work exists: at no instant may a slot sit free while some call whose dependencies are all satisfied has not started yet — every such slot is filled at that very instant (the schedule is *work-conserving*).
- Whenever slots run out before every eligible call can start, or several calls simply start together, ties are broken by ascending `(agent_id, tool_id)`.

Every part below returns the calls that were started, as `(start_time, agent_id, tool_id)` triples ordered by `start_time` and then by `(agent_id, tool_id)`, together with the *makespan*: the largest `start_time + duration` over every call (0 when there are no calls at all).

```py
from typing import NamedTuple, Sequence

class ToolCall(NamedTuple):
    tool_id: int
    duration: int              # 1 .. 10**9
    deps: Sequence[int] = ()   # other tool_id's belonging to the SAME agent, all required first

class CycleError(Exception):
    """Some tool calls can never run: their dependencies form a cycle."""
```

### Part 1 — One agent's own tool calls

A single agent submits a list of `ToolCall`s in which every value `0, ..., len(calls) - 1` occurs exactly once as some call's `tool_id` (a call's position inside the list need not match its `tool_id`). The agent's own concurrency cap `capacity` is a positive integer; a cap above `len(calls)` only leaves slots permanently unused. Implement `schedule_agent`.

```py
from typing import List, Tuple

def schedule_agent(calls: List[ToolCall], capacity: int) -> Tuple[List[Tuple[int, int, int]], int]:
    """Every triple in the returned list has agent_id == 0. Returns ([], 0) for an empty `calls`."""
```

Example, with `capacity = 1` and three calls — `ToolCall(0, 2)`, `ToolCall(1, 1)`, and `ToolCall(2, 1, deps=(0,))`:

```text
schedule_agent(calls, 1) == ([(0, 0, 0), (2, 0, 1), (3, 0, 2)], 4)
```

Call 1 has no dependency and is ready from time 0, but the single slot is held by call 0 until time 2, so call 1 only starts then; call 2 waits on call 0 and starts right after it, at time 3.

### Part 2 — Several agents, each under its own cap

Now $A$ agents each submit their own list of `ToolCall`s. Agent $a$'s `tool_id`s are local to agent $a$ — the same `tool_id` in two different agents names two different calls. Agent $a$ has its own positive cap `capacities[a]`, and no agent's calls ever compete with another agent's for a slot. Implement `schedule_agents_independently`.

```py
def schedule_agents_independently(
    agents: List[List[ToolCall]], capacities: List[int]
) -> Tuple[List[Tuple[int, int, int]], int]:
    """agents[a] is agent a's own calls, scheduled exactly as schedule_agent would schedule them on
    their own under capacities[a]. Returns every started call across every agent, and the largest of
    the per-agent makespans."""
```

Example, with agent 0's calls `ToolCall(0, 2)`, `ToolCall(1, 1, deps=(0,))` under `capacities[0] = 1`, and agent 1's calls `ToolCall(0, 1)`, `ToolCall(1, 1)` under `capacities[1] = 2`:

```text
schedule_agents_independently(agents, [1, 2]) == ([(0, 0, 0), (0, 1, 0), (0, 1, 1), (2, 0, 1)], 3)
```

Agent 1's two calls have no dependencies, and its cap of 2 lets both start at time 0; agent 0's own makespan of 3 is what sets the overall makespan, even though agent 1 finishes earlier.

### Part 3 — One cap shared by every agent

The same $A$ agents as in Part 2 now draw from a single positive global `capacity` — a call belonging to any agent may take any free slot, so agents genuinely compete for slots. Implement `schedule_agents_global`.

```py
def schedule_agents_global(
    agents: List[List[ToolCall]], capacity: int
) -> Tuple[List[Tuple[int, int, int]], int]:
    """Same `agents` shape as schedule_agents_independently, but one `capacity` shared by all of
    them."""
```

Constraints: $1 \le A \le 10^5$; the total number of calls $N = \sum_a \text{len(agents[a])}$ satisfies $N \le 2 \times 10^5$; the total number of dependency edges $E$ satisfies $E \le 5 \times 10^5$; time must be $O((N + E) \log N)$ and space $O(N + E)$.

Example, with three agents and `capacity = 2`:

- Agent 0: `ToolCall(0, 2)`, `ToolCall(1, 3, deps=(0,))`, `ToolCall(2, 1, deps=(0,))`
- Agent 1: `ToolCall(0, 4)`, `ToolCall(1, 2, deps=(0,))`
- Agent 2: `ToolCall(0, 2)`

Writing $(a, i)$ for agent $a$'s call $i$, and stepping through it:

- $t = 0$: $(0,0)$, $(1,0)$ and $(2,0)$ all have no dependency and are ready, but only 2 slots exist. The tie-break starts $(0,0)$ and $(1,0)$; $(2,0)$ stays ready, waiting for a slot. $(0,0)$ finishes at 2, $(1,0)$ at 4.
- $t = 2$: $(0,0)$ finishes, freeing one slot and unlocking both $(0,1)$ and $(0,2)$ — each depended only on it. Ready now: $(0,1)$, $(0,2)$, and the still-waiting $(2,0)$; only one slot is free ($(1,0)$ still runs). The tie-break starts $(0,1)$; it finishes at 5.
- $t = 4$: $(1,0)$ finishes, unlocking $(1,1)$. Ready now: $(0,2)$, $(1,1)$, $(2,0)$; one slot is free ($(0,1)$ still runs). $(0,2)$ wins the tie-break and starts, finishing at 5.
- $t = 5$: $(0,1)$ and $(0,2)$ both finish, freeing both slots; neither unlocks anything else. Ready: $(1,1)$ and $(2,0)$ — with two free slots, both start, in that order. Both finish at 7.
- $t = 7$: everything has finished.

```text
schedule_agents_global(agents, 2) == (
    [(0, 0, 0), (0, 1, 0), (2, 0, 1), (4, 0, 2), (5, 1, 1), (5, 2, 0)],
    7,
)
```

Note that $(2,0)$ became ready at time 0 but is outranked by lower `agent_id`s at every contested slot until time 5: the tie-break decides who gets a slot when several calls compete for it, not only how simultaneous starts get printed.
