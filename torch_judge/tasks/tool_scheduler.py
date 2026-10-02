"""Schedule agents' tool calls with dependencies on limited slots: one agent, agents apart, agents sharing a cap, then both caps."""

from ._interview import interview

# A per-tick brute force written from the rules, independent of the event-driven reference.
_HELPERS = r"""
import random, time

def raises(kind, call):
    try:
        call()
    except Exception as e:
        assert type(e).__name__ == kind, f"expected {kind}, got {type(e).__name__}: {e}"
        return e
    raise AssertionError(f"expected {kind}, nothing was raised")

def brute(agents, capacity, caps=None):
    flat = [(a, c) for a, calls in enumerate(agents) for c in calls]
    done = set()
    started = set()
    finish_at = {}
    running = {a: 0 for a in range(len(agents))}
    starts, makespan, t = [], 0, 0
    while len(started) < len(flat):
        for key in [k for k, f in finish_at.items() if f == t]:
            del finish_at[key]
            done.add(key)
            running[key[0]] -= 1
        waiting = sorted((a, c[0]) for a, c in flat if (a, c[0]) not in started
                         and all((a, d) in done for d in c[2]))
        by_key = {(a, c[0]): c for a, c in flat}
        for a, tool in waiting:
            if len(finish_at) >= capacity:
                break
            if caps is not None and running[a] >= caps[a]:
                continue
            c = by_key[(a, tool)]
            starts.append((t, a, tool))
            started.add((a, tool))
            finish_at[(a, tool)] = t + c[1]
            running[a] += 1
            makespan = max(makespan, t + c[1])
        if len(started) == len(flat):
            break
        if not finish_at:
            return "CycleError"
        t += 1
    return starts, makespan

def random_agents(rng, agents=4, calls=6, longest=4, cycle=0.0):
    out = []
    for _ in range(rng.randint(1, agents)):
        n = rng.randint(0, calls)
        made = []
        for tool in range(n):
            deps = rng.sample(range(tool), rng.randint(0, min(2, tool))) if tool else []
            made.append((tool, rng.randint(1, longest), deps))
        if n >= 2 and rng.random() < cycle:
            a, b = rng.sample(range(n), 2)
            made[min(a, b)] = (min(a, b), made[min(a, b)][1], made[min(a, b)][2] + [max(a, b)])
        rng.shuffle(made)
        out.append(made)
    return out

def check(got, want, label):
    if want == "CycleError":
        raises("CycleError", got)
    else:
        result = got()
        assert result == want, (label, result, want)
"""

TESTS = [
    {"name": "Part 1: the worked example", "part": 1, "behavior": "contract.signature", "code": r"""
s = {fn}()
calls = [(2, 3, [0]), (0, 4, []), (1, 1, []), (3, 2, [1])]
assert s.run_one(calls, 2) == ([(0, 0, 0), (0, 0, 1), (1, 0, 3), (4, 0, 2)], 7)
assert s.run_one([], 3) == ([], 0)
"""},
    {"name": "Part 1: instants, ties and cycles", "part": 1, "visibility": "unshown", "behavior": "events.ordering",
     "failure_message": "A dependency finishing at t unlocks its dependents at t; finishing calls free their slots before new calls start; ties go to the smaller tool_id whatever the list order; no slot stays free while a call is ready; any call that can never start raises CycleError.",
     "code": _HELPERS + r"""
s = {fn}()
assert s.run_one([(1, 5, [0]), (0, 5, [])], 1) == ([(0, 0, 0), (5, 0, 1)], 10)
assert s.run_one([(0, 2, []), (1, 3, []), (2, 1, [])], 1) == ([(0, 0, 0), (2, 0, 1), (5, 0, 2)], 6)
assert s.run_one([(3, 1, []), (2, 1, []), (1, 1, []), (0, 1, [])], 2) == ([(0, 0, 0), (0, 0, 1), (1, 0, 2), (1, 0, 3)], 2)
assert s.run_one([(0, 1, []), (1, 1, [])], 10) == ([(0, 0, 0), (0, 0, 1)], 1)
assert s.run_one([(0, 10**9, []), (1, 10**9, [0])], 1) == ([(0, 0, 0), (10**9, 0, 1)], 2 * 10**9)
raises("CycleError", lambda: s.run_one([(0, 1, [0])], 1))
raises("CycleError", lambda: s.run_one([(0, 1, []), (1, 1, [2]), (2, 1, [1])], 4))
raises("CycleError", lambda: s.run_one([(0, 1, []), (1, 1, [0, 3]), (2, 1, [1]), (3, 1, [2])], 1))
"""},
    {"name": "Part 1: random calls", "part": 1, "visibility": "unshown", "behavior": "state.invariant",
     "failure_message": "On random dependency lists, capacities and some cycles, the starts or makespan differed from stepping time one unit at a time and filling free slots in (agent_id, tool_id) order.",
     "code": _HELPERS + r"""
s = {fn}()
rng = random.Random(21)
for trial in range(300):
    calls = random_agents(rng, agents=1, calls=8, cycle=0.15)[0]
    capacity = rng.randint(1, 4)
    check(lambda: s.run_one(calls, capacity), brute([calls], capacity), (trial, calls, capacity))
"""},
    {"name": "Part 2: the worked example", "part": 2, "behavior": "contract.signature", "code": r"""
s = {fn}()
agents = [[(0, 3, []), (1, 2, [])], [(0, 1, []), (1, 1, [0])]]
assert s.run_each(agents, [1, 1]) == ([(0, 0, 0), (0, 1, 0), (1, 1, 1), (3, 0, 1)], 5)
assert s.run_each([], []) == ([], 0) and s.run_each([[], []], [1, 1]) == ([], 0)
"""},
    {"name": "Part 2: random agents", "part": 2, "visibility": "unshown", "behavior": "state.invariant",
     "failure_message": "Each agent must be scheduled alone under its own cap, the starts merged by (start_time, agent_id, tool_id), the makespan the largest over agents, and a cycle in any agent raises CycleError.",
     "code": _HELPERS + r"""
s = {fn}()
rng = random.Random(22)
for trial in range(300):
    agents = random_agents(rng, cycle=0.05)
    caps = [rng.randint(1, 3) for _ in agents]
    want = [brute([calls], caps[a]) for a, calls in enumerate(agents)]
    if "CycleError" in want:
        raises("CycleError", lambda: s.run_each(agents, caps))
        continue
    merged = sorted((t, a, tool) for a, (starts, _) in enumerate(want) for t, _, tool in starts)
    assert s.run_each(agents, caps) == (merged, max([m for _, m in want], default=0)), (trial, agents, caps)
"""},
    {"name": "Part 3: the worked example", "part": 3, "behavior": "contract.signature", "code": r"""
s = {fn}()
agents = [[(0, 1, []), (1, 2, [0])], [(0, 3, [])], [(0, 2, []), (1, 1, [0])]]
assert s.run_shared(agents, 2) == ([(0, 0, 0), (0, 1, 0), (1, 0, 1), (3, 2, 0), (5, 2, 1)], 6)
"""},
    {"name": "Part 3: random agents sharing slots", "part": 3, "visibility": "unshown", "behavior": "scheduler.concurrency",
     "failure_message": "With one cap shared by every agent, the starts or makespan differed from stepping time one unit at a time and giving each free slot to the ready call with the smallest (agent_id, tool_id).",
     "code": _HELPERS + r"""
s = {fn}()
rng = random.Random(23)
for trial in range(300):
    agents = random_agents(rng, cycle=0.05)
    capacity = rng.randint(1, 5)
    check(lambda: s.run_shared(agents, capacity), brute(agents, capacity), (trial, agents, capacity))
"""},
    {"name": "Part 3: many calls and long durations", "part": 3, "visibility": "unshown", "behavior": "performance.complexity",
     "failure_message": "About 40,000 calls with durations up to 10**9 took too long: jump to the next finish time instead of stepping time, and keep ready and running calls in heaps rather than rescanning lists.",
     "code": r"""
import random, time
rng = random.Random(24)
agents = []
for a in range(400):
    calls = []
    for tool in range(100):
        deps = [tool - 1] if tool % 4 else []
        calls.append((tool, rng.randint(1, 10**9), deps))
    rng.shuffle(calls)
    agents.append(calls)
s = {fn}()
start = time.perf_counter()
starts, makespan = s.run_shared(agents, 64)
elapsed = time.perf_counter() - start
assert len(starts) == 40000 and len({(a, t) for _, a, t in starts}) == 40000
assert starts == sorted(starts)
assert elapsed < 6.0, f"{elapsed:.2f}s for 40,000 calls"
"""},
    {"name": "Part 4: the worked example", "part": 4, "behavior": "contract.signature", "code": r"""
s = {fn}()
agents = [[(0, 2, []), (1, 2, [])], [(0, 1, []), (1, 1, []), (2, 1, [])]]
assert s.run_capped(agents, [1, 2], 2) == ([(0, 0, 0), (0, 1, 0), (1, 1, 1), (2, 0, 1), (2, 1, 2)], 4)
"""},
    {"name": "Part 4: random agents with both caps", "part": 4, "visibility": "unshown", "behavior": "budget.enforcement",
     "failure_message": "With a cap per agent and a shared cap, each free slot must go to the smallest (agent_id, tool_id) among ready calls whose agent is below its own cap; with very large caps on one side the result must match run_each or run_shared.",
     "code": _HELPERS + r"""
s = {fn}()
rng = random.Random(25)
for trial in range(300):
    agents = random_agents(rng, cycle=0.05)
    caps = [rng.randint(1, 3) for _ in agents]
    capacity = rng.randint(1, 5)
    check(lambda: s.run_capped(agents, caps, capacity), brute(agents, capacity, caps), (trial, agents, caps, capacity))
for trial in range(40):
    agents = random_agents(rng)
    caps = [rng.randint(1, 3) for _ in agents]
    big = 10**6
    assert s.run_capped(agents, caps, big) == s.run_each(agents, caps)
    assert s.run_capped(agents, [big] * len(agents), 3) == s.run_shared(agents, 3)
"""},
    {"name": "Part 4: many agents held at their cap", "part": 4, "visibility": "unshown", "behavior": "performance.complexity",
     "failure_message": "Many ready calls whose agents sit at their own cap took too long: skip a capped agent as a whole rather than setting aside and re-queuing its calls one by one.",
     "code": r"""
import random, time
rng = random.Random(26)
agents = [[(tool, rng.randint(1, 10**6), []) for tool in range(150)] for _ in range(250)]
s = {fn}()
start = time.perf_counter()
starts, makespan = s.run_capped(agents, [1] * 250, 200)
elapsed = time.perf_counter() - start
assert len(starts) == 37500
first = {a: t for t, a, tool in starts if tool == 0}
assert all(first[a] == 0 for a in range(200)) and all(first[a] > 0 for a in range(200, 250))
assert elapsed < 6.0, f"{elapsed:.2f}s for 37,500 calls"
"""},
]

TASK = {
    "title": "Agent Tool-Call Scheduler",
    "difficulty": "Hard",
    "version": 1,
    "function_name": "ToolScheduler",
    "description_en": r"""Build `ToolScheduler`, which plans when each agent's tool calls start, given their durations, their dependencies and a limited number of slots.

The requirement arrives in parts. Each part keeps every earlier behavior and adds one method, so one `ToolScheduler` class passes all parts at the end. Pass every test of the current part to reveal the next one.

**Rules for every part:**
- A call is a tuple `(tool_id, duration, deps)`. An agent's `tool_id`s are exactly `0` to `n - 1` in some order, `duration` is an integer from `1` to `10**9`, and `deps` lists `tool_id`s of the same agent that must finish first.
- Time is an integer and starts at `0`. A call holds one slot from its start `s` until `s + duration`, and runs without interruption.
- A call is ready at time `t` once every dependency has `start + duration <= t`. At each `t`, first release every call that finishes at `t`, then fill free slots with ready calls.
- A slot is never left free while a ready call waits. When there are more ready calls than free slots, the smaller `(agent_id, tool_id)` starts first.
- Each method returns `(starts, makespan)`. `starts` holds one `(start_time, agent_id, tool_id)` per call, sorted. `makespan` is the largest `start + duration`, or `0` with no calls.
- If some call can never start because of a dependency cycle, raise `CycleError`, a class you define.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why this shows up in interviews:** it is list scheduling with dependencies, the core of every task runner. Each later part adds one requirement, and durations up to `10**9` rule out stepping through time one unit at a time.

**Where it is used:** agent frameworks running tool calls in parallel, build systems such as Bazel and make `-j`, workflow engines such as Airflow, and GPU job queues with per-team limits.

Adapted from the agent tool scheduler question in Schuture/OpenAI-Interview-Notes (CC BY-NC 4.0), reworded, on one class whose methods replace the three functions. Calls are plain tuples. Part 4, a cap per agent under a shared cap, comes from the source's follow-ups.""",
    "parts": [
        {
            "title": "One agent",
            "description_en": r"""**Signature:** `ToolScheduler().run_one(calls, capacity) -> (starts, makespan)`

- `calls` belong to agent `0`, and `capacity` is a positive number of slots.

**Example:** `calls = [(2, 3, [0]), (0, 4, []), (1, 1, []), (3, 2, [1])]` and `capacity = 2`:
- at `0`, calls `0` and `1` start; call `1` finishes at `1`, which makes call `3` ready, so it starts at `1`
- call `2` waits for call `0`, which finishes at `4`
- the result is `([(0, 0, 0), (0, 0, 1), (1, 0, 3), (4, 0, 2)], 7)`, and `run_one([], 3)` is `([], 0)`""",
        },
        {
            "title": "Agents on their own caps",
            "description_en": r"""Keep Part 1 and add a method.

**Signature:** `run_each(agents, capacities) -> (starts, makespan)`

- `agents[a]` is the list of agent `a`'s calls. `tool_id`s are local, so the same id in two agents names two different calls.
- Agent `a` has `capacities[a]` slots of its own and never competes with other agents. Each agent's starts are exactly what `run_one` would give it alone, with its own `agent_id`.
- Return every start across agents, sorted, and the largest makespan.

**Example:** `agents = [[(0, 3, []), (1, 2, [])], [(0, 1, []), (1, 1, [0])]]` and `capacities = [1, 1]`:
- agent `0` runs its calls at `0` and `3`; agent `1` at `0` and `1`
- the result is `([(0, 0, 0), (0, 1, 0), (1, 1, 1), (3, 0, 1)], 5)`""",
        },
        {
            "title": "One shared cap",
            "description_en": r"""Keep Parts 1–2 and add a method.

**Signature:** `run_shared(agents, capacity) -> (starts, makespan)`

- Every agent's calls draw from the same `capacity` slots, so agents compete for each free slot under the tie rule.
- Up to `40,000` calls with durations up to `10**9` must finish in a few seconds: aim for `O((N + E) log N)` for `N` calls and `E` dependencies.

**Example:** `agents = [[(0, 1, []), (1, 2, [0])], [(0, 3, [])], [(0, 2, []), (1, 1, [0])]]` and `capacity = 2`:
- at `0`, three calls are ready and `(0, 0)` and `(1, 0)` win; at `1`, `(0, 1)` beats `(2, 0)` for the freed slot
- at `3`, both running calls finish and `(2, 0)` starts; `(2, 1)` follows at `5`
- the result is `([(0, 0, 0), (0, 1, 0), (1, 0, 1), (3, 2, 0), (5, 2, 1)], 6)`""",
        },
        {
            "title": "Both caps at once",
            "description_en": r"""Keep Parts 1–3 and add a method.

**Signature:** `run_capped(agents, capacities, capacity) -> (starts, makespan)`

- Agent `a` may hold at most `capacities[a]` slots at once, and all agents together at most `capacity`.
- A free slot goes to the smallest `(agent_id, tool_id)` among ready calls whose agent is below its own cap. A capped agent's ready calls wait without blocking other agents.
- With every per-agent cap large, this is `run_shared`; with `capacity` large, it is `run_each`.
- Many agents may sit at their cap with many ready calls: skip such an agent as a whole rather than one call at a time.

**Example:** `agents = [[(0, 2, []), (1, 2, [])], [(0, 1, []), (1, 1, []), (2, 1, [])]]`, `capacities = [1, 2]`, `capacity = 2`:
- at `0`, agent `0` takes one slot and `(1, 0)` the other, since `(0, 1)` is held by agent `0`'s cap
- at `1`, `(1, 1)` starts; at `2`, `(0, 1)` and `(1, 2)` start
- the result is `([(0, 0, 0), (0, 1, 0), (1, 1, 1), (2, 0, 1), (2, 1, 2)], 4)`""",
        },
    ],
    "hints": [
        {"level": 1, "kind": "questions", "content": "Between two moments when a call finishes, can anything about the schedule change? Which ready call should get a slot when several want it, and which data structure hands you that call quickly? How can you tell that the calls left over can never start?"},
        {"level": 2, "kind": "analysis", "content": "Count unfinished dependencies per call and keep a heap of ready calls keyed by (agent_id, tool_id) and a heap of running calls keyed by finish time. Loop: start ready calls while slots are free; then jump to the earliest finish time, release every call ending then, and push dependents whose count reaches 0. If nothing runs and calls remain, they wait on a cycle. Each call is pushed and popped once, so the run is O((N + E) log N)."},
    ],
    "model_connections": [
        "Agent frameworks run independent tool calls in parallel and hold dependent ones until their inputs exist, under a limit on concurrent sandboxes.",
        "Training and evaluation clusters schedule jobs with dependencies on a shared pool of GPUs with per-team quotas, the same two caps as Part 4.",
    ],
    "pro_con_analysis": {
        "pros": [
            "Jumping between finish times makes the work depend on the number of calls, not on the durations.",
            "Dependency counts give the ready set and the cycle check in one pass, as in Kahn's algorithm.",
            "A fixed tie rule makes the schedule deterministic and easy to test.",
        ],
        "cons": [
            "The tie rule always favours small agent ids, so a high id can wait for a long time.",
            "Greedy list scheduling is not optimal: starting a short call first can delay a long chain and lengthen the makespan.",
            "Real tool calls do not have known durations, so a live executor can only react to completions, not plan ahead.",
        ],
    },
    "tests": TESTS,
    "solution": r'''# Adapted from Schuture/OpenAI-Interview-Notes (code under the MIT License).
import heapq


class CycleError(Exception):
    pass


class ToolScheduler:
    def _run(self, agents, caps, capacity):
        """Event-driven list scheduling; caps[a] is agent a's own limit, capacity the shared one."""
        index = {}  # (agent_id, tool_id) -> (duration, dependents)
        waiting = {}  # (agent_id, tool_id) -> unfinished dependencies
        for a, calls in enumerate(agents):
            for tool, duration, deps in calls:
                index[(a, tool)] = (duration, [])
                waiting[(a, tool)] = len(deps)
        for a, calls in enumerate(agents):
            for tool, _, deps in calls:
                for dep in deps:
                    index[(a, dep)][1].append(tool)
        ready = [[] for _ in agents]  # per agent: heap of ready tool_ids
        for (a, tool), count in waiting.items():
            if count == 0:
                ready[a].append(tool)
        for heap in ready:
            heapq.heapify(heap)
        holding = [0] * len(agents)
        eligible = [a for a in range(len(agents)) if ready[a]]  # heap of agents that may start a call
        queued = set(eligible)
        running = []  # heap of (finish_time, agent_id, tool_id)
        starts, now, makespan, free = [], 0, 0, capacity
        while len(starts) < len(index):
            while free and eligible:  # fill every free slot before time moves
                a = heapq.heappop(eligible)  # only agents below their cap with a ready call are queued
                queued.discard(a)
                tool = heapq.heappop(ready[a])
                finish = now + index[(a, tool)][0]
                starts.append((now, a, tool))
                heapq.heappush(running, (finish, a, tool))
                makespan = max(makespan, finish)
                holding[a] += 1
                free -= 1
                if ready[a] and holding[a] < caps[a]:
                    heapq.heappush(eligible, a)
                    queued.add(a)
            if len(starts) == len(index):
                break
            if not running:
                raise CycleError("some calls wait on a dependency cycle")
            now = running[0][0]
            touched = set()
            while running and running[0][0] == now:  # release everything finishing now first
                _, a, tool = heapq.heappop(running)
                holding[a] -= 1
                free += 1
                touched.add(a)
                for nxt in index[(a, tool)][1]:
                    waiting[(a, nxt)] -= 1
                    if waiting[(a, nxt)] == 0:
                        heapq.heappush(ready[a], nxt)
            for a in touched:
                if a not in queued and ready[a] and holding[a] < caps[a]:
                    heapq.heappush(eligible, a)
                    queued.add(a)
        return sorted(starts), makespan

    def run_one(self, calls, capacity):
        return self.run_shared([calls], capacity)

    def run_each(self, agents, capacities):
        return self.run_capped(agents, capacities, sum(capacities))

    def run_shared(self, agents, capacity):
        return self.run_capped(agents, [capacity] * len(agents), capacity)

    def run_capped(self, agents, capacities, capacity):
        return self._run(agents, capacities, capacity)
''',
    "interview_questions": interview(
        concept=[
            "Why can the scheduler jump from one finish time to the next instead of stepping through every time unit?",
            "Why must every call finishing at time t be released before any new call starts at t?",
        ],
        deep_dive=[
            "How do dependency counts find the ready calls and also detect a cycle, and what is the total cost?",
        ],
        tradeoffs=[
            "Why is scheduling each agent alone and merging wrong once agents share one cap?",
            "With a shared cap, what does the fixed tie rule do to agents with large ids, and how would you make it fairer?",
            "With both caps, why can one ready heap across all agents become slow, and what replaces it?",
            "Real tool calls have unknown durations. How would a live executor use the same dependency counts?",
        ],
    ),
}
