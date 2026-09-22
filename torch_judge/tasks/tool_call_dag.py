"""Dependency-aware scheduling of parallel tool calls, with failure propagation."""

from torch_judge.tasks._schema import build_design_note_rubric

TASK = {
    "title": "Tool-Call DAG Scheduler",
    "difficulty": "Hard",
    "version": 1,
    "function_name": "schedule_tool_calls",
    "description_en": r"""Run a batch of tool calls whose arguments depend on each other's results, in parallel rounds.

**Signature:** `schedule_tool_calls(calls, execute, max_parallel) -> dict`

**Parameters:**
- `calls` — a list of dictionaries, each with a string `id` and a list `deps` of the ids it depends on. Any other keys belong to the caller; pass the dictionary through untouched.
- `execute` — `execute(call, dep_results) -> result`. Runs one call. `dep_results` maps each id in that call's `deps` to its result — those ids and no others. It may raise any `Exception`.
- `max_parallel` — positive integer. The most calls one round may run.

**Returns:** a dictionary with exactly four keys:

- `results` — maps the id of every call that succeeded to its result.
- `errors` — maps the id of every call that raised to the exception it raised.
- `skipped` — ids of calls that never ran because a dependency failed or was itself skipped, in input order.
- `rounds` — a list of lists: the ids run in each round, in the order they ran.

**The schedule, per round:**

1. A call is ready when every one of its dependencies succeeded in an **earlier** round.
2. Run the ready calls in input order, at most `max_parallel` of them. Ready calls beyond the cap wait for the next round; they are not lost.
3. When a call raises, record the exception, then mark every call that depends on it — directly or through any chain — as skipped. Never run a skipped call.
4. Stop when no call is ready. Every call ends in exactly one of `results`, `errors`, or `skipped`.

**Constraints:**
- A failure stops only what depends on it. Independent branches keep running.
- Validate the whole graph before running anything. Raise a `ValueError`, without calling `execute`, when: `max_parallel` is not a positive integer; an id is not a string or appears twice; `deps` is not a list; a dependency names an unknown id; a call depends on itself; or the dependencies contain a cycle.
- Only `Exception` subclasses count as call failures. Do not catch anything broader.
- Do not mutate `calls` or any dictionary in it.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Where this shows up.** A model that emits several tool calls in one turn often emits calls that need each other: search, then fetch the top result, then summarize two fetches together. Running them one at a time wastes the latency the model bought by planning ahead; running them all at once feeds a call a result that does not exist yet. The scheduler in between is a topological sort that runs in waves.

**Rounds make the schedule observable.** A real executor runs a call the moment its last dependency lands. Grouping into rounds — the superstep model — gives up a little latency in exchange for a schedule you can assert on, log, and replay. That trade is why this exercise can be graded deterministically, and it is the same shape as a bulk-synchronous graph engine.

**Why validate up front.** A cycle or an unknown dependency leaves some call never ready. A loop that polls for ready work without checking for progress spins forever, and a scheduler that discovers the problem halfway through has already paid for side effects it cannot take back. Kahn's algorithm finds both before anything runs.

**What skipping buys.** A failed call has no result to hand its dependents. Running them anyway means inventing a value; waiting for them means waiting forever. Skipping them — transitively — and reporting which ones were skipped keeps every call accounted for, so the caller can tell the model exactly which parts of its plan did not happen.""",
    "advisory_prerequisites": ["budgeted_agent_loop"],
    "design_note_rubric": build_design_note_rubric(),
    "hints": [
        {
            "level": 1,
            "kind": "questions",
            "content": "Before scheduling anything, what could make some call never become ready, and how would you detect it without running a single call? Once the graph is known to be sound: what exactly makes a call ready — is 'its dependencies have been started' enough, or does something stronger have to be true? When a call fails, which calls does that affect: only the ones that list it directly, or more? And which calls must it not affect? Last, if six calls are ready and the cap is four, where do the other two go, and in what order do they run?",
        },
        {
            "level": 2,
            "kind": "analysis",
            "content": "Validate first: check max_parallel, then collect ids into a dict while checking for non-strings and duplicates, then check every dependency is known and not the call itself. For cycles, run Kahn's algorithm on in-degrees — if fewer calls are emitted than exist, there is a cycle; raise before touching execute. Then schedule with a status per id. Each round, compute the ready list by walking calls in input order and keeping those still pending whose every dependency has status 'succeeded' as of the start of the round; take the first max_parallel. Computing readiness once per round, before running anything in it, is what stops a dependent from running in the same round as its dependency — updating status mid-round and re-scanning is the easy way to get that wrong. On a failure, walk the reverse edges from the failed id with a stack and mark every pending descendant skipped. Marking only direct dependents leaves grandchildren pending forever, so they end up in no bucket at all. Build dep_results from the call's deps only, and pass the original call dictionary through without copying fields into it.",
        },
    ],
    "model_connections": [
        "LLMCompiler's TaskFetchingUnit schedules a planner's function calls by dependency, substituting each dependency's observation into the arguments that reference it, which is the dep_results mapping here.",
        "Its scheduling loop polls for executable tasks until every task's done event is set, with no cycle check, no concurrency cap and no failure path: a task that raises never sets its event, so the loop waits on it forever. This exercise requires all three.",
        "Parallel function calling in model APIs returns several calls per turn and leaves ordering to the client; clients that need one call's output as another's input end up building this scheduler themselves.",
        "Graph runtimes that execute agent workflows in supersteps run every node whose inputs are ready, then synchronize, which is the round structure used here.",
    ],
    "pro_con_analysis": {
        "pros": [
            "Independent calls run concurrently while dependent ones still see real results, so latency tracks the depth of the plan rather than its size.",
            "Up-front validation means a malformed plan costs nothing: no side effect runs before the cycle or unknown dependency is found.",
            "Transitive skipping keeps every call in exactly one bucket, so the model can be told precisely what did not happen and why.",
        ],
        "cons": [
            "Round barriers add latency: a fast call's dependent waits for the slowest call of the round before it can start.",
            "Skipping is all-or-nothing; a dependent that could have run with a partial or default input is skipped anyway.",
            "A static graph cannot express calls whose dependencies are only known after an earlier result arrives; that needs re-planning, which is a different loop.",
        ],
    },
    "sources": [
        {
            "kind": "code",
            "url": "https://github.com/SqueezeAILab/LLMCompiler",
            "commit": "a00c9d35507507da70e8c637eee64efc8c1857ae",
            "path": "src/llm_compiler/task_fetching_unit.py",
            "symbol": "TaskFetchingUnit.schedule",
            "license": "MIT",
            "adapted": "Dependency-driven execution of a planner's function calls: a task becomes executable when all its dependencies are done, and each dependency's observation is made available to the tasks that reference it.",
            "simplifications": "Replaced the asyncio polling loop with deterministic rounds, and the ${n} argument-string substitution with a dep_results mapping. Added what upstream lacks: up-front cycle and unknown-dependency detection, a concurrency cap, deterministic input-order scheduling instead of iterating a set, and a failure path that skips dependents transitively instead of leaving the loop waiting on a done event that is never set.",
        },
    ],
    "tests": [
        {
            "name": "A diamond runs in three rounds with each call seeing only its own dependencies",
            "code": r"""
seen = {}
def execute(call, dep_results):
    seen[call['id']] = dict(dep_results)
    return call['id'].upper() + ''.join(sorted(dep_results.values()))

calls = [
    {'id': 'a', 'deps': []},
    {'id': 'b', 'deps': ['a']},
    {'id': 'c', 'deps': ['a']},
    {'id': 'd', 'deps': ['b', 'c']},
]
out = {fn}(calls, execute, 4)
assert out['rounds'] == [['a'], ['b', 'c'], ['d']], out['rounds']
assert seen['a'] == {}, seen['a']
assert seen['b'] == {'a': 'A'}, seen['b']
assert seen['d'] == {'b': 'BA', 'c': 'CA'}, seen['d']
assert out['results'] == {'a': 'A', 'b': 'BA', 'c': 'CA', 'd': 'DBACA'}, out['results']
assert out['errors'] == {} and out['skipped'] == [], out
""",
        },
        {
            "name": "A failure skips its dependents and nothing else",
            "code": r"""
ran = []
def execute(call, dep_results):
    ran.append(call['id'])
    if call['id'] == 'b':
        raise RuntimeError('boom')
    return call['id']

calls = [
    {'id': 'a', 'deps': []},
    {'id': 'b', 'deps': ['a']},
    {'id': 'c', 'deps': ['b']},
    {'id': 'x', 'deps': []},
    {'id': 'y', 'deps': ['x']},
    {'id': 'z', 'deps': ['y']},
]
out = {fn}(calls, execute, 4)
assert isinstance(out['errors'].get('b'), RuntimeError), out['errors']
assert out['skipped'] == ['c'], out['skipped']
# z is still waiting when b fails; the failure must not reach it.
assert out['results'] == {'a': 'a', 'x': 'x', 'y': 'y', 'z': 'z'}, out['results']
assert 'c' not in ran, ran
""",
        },
        {
            "name": "A dependent never runs in the same round as its dependency",
            "visibility": "unshown",
            "behavior": "events.ordering",
            "failure_message": "A call ran in the same round as one of its dependencies. Readiness is decided at the start of a round, from calls that succeeded in earlier rounds only.",
            "code": r"""
def execute(call, dep_results):
    for dep in call['deps']:
        assert dep in dep_results, f"{call['id']} ran before {dep} had a result"
    return call['id']

chain = [{'id': f'n{i}', 'deps': [f'n{i-1}'] if i else []} for i in range(5)]
out = {fn}(chain, execute, 8)
assert out['rounds'] == [[f'n{i}'] for i in range(5)], out['rounds']

where = {}
wide = [
    {'id': 'p', 'deps': []}, {'id': 'q', 'deps': ['p']},
    {'id': 'r', 'deps': []}, {'id': 's', 'deps': ['r', 'q']},
]
out = {fn}(wide, execute, 8)
for index, ids in enumerate(out['rounds']):
    for i in ids:
        where[i] = index
for call in wide:
    for dep in call['deps']:
        assert where[dep] < where[call['id']], out['rounds']
""",
        },
        {
            "name": "Skips propagate through every chain of dependents",
            "visibility": "unshown",
            "behavior": "state.invariant",
            "failure_message": "A call downstream of a failure was left pending or was run. Skipping is transitive: a dependent of a skipped call is skipped too, and every call ends in exactly one bucket.",
            "code": r"""
ran = []
def execute(call, dep_results):
    ran.append(call['id'])
    if call['id'] == 'root':
        raise ValueError('bad input')
    return 1

calls = [
    {'id': 'root', 'deps': []},
    {'id': 'fetch', 'deps': ['root']},
    {'id': 'cite', 'deps': ['fetch']},
    {'id': 'audit', 'deps': ['cite', 'other']},
    {'id': 'other', 'deps': []},
]
out = {fn}(calls, execute, 2)
# Input order, which is deliberately not alphabetical.
assert out['skipped'] == ['fetch', 'cite', 'audit'], out['skipped']
assert set(ran) == {'root', 'other'}, ran
ids = [c['id'] for c in calls]
buckets = list(out['results']) + list(out['errors']) + out['skipped']
assert sorted(buckets) == sorted(ids), buckets
""",
        },
        {
            "name": "The concurrency cap is respected and deferred calls still run",
            "visibility": "unshown",
            "behavior": "scheduler.concurrency",
            "failure_message": "A round ran more than max_parallel calls, or a ready call beyond the cap was dropped or reordered. Excess ready calls wait for the next round, in input order.",
            "code": r"""
def execute(call, dep_results):
    return call['id']

calls = [{'id': f't{i}', 'deps': []} for i in range(7)] + [{'id': 'join', 'deps': ['t0', 't6']}]
out = {fn}(calls, execute, 3)
assert all(len(r) <= 3 for r in out['rounds']), out['rounds']
assert out['rounds'][:3] == [['t0', 't1', 't2'], ['t3', 't4', 't5'], ['t6']], out['rounds']
assert out['rounds'][3] == ['join'], out['rounds']
assert len(out['results']) == 8, out['results']

one = {fn}(calls, execute, 1)
assert [r for r in one['rounds']] == [[c['id']] for c in calls], one['rounds']
""",
        },
        {
            "name": "An invalid graph is rejected before anything runs",
            "visibility": "unshown",
            "behavior": "protocol.validation",
            "failure_message": "A cycle, unknown dependency, self-dependency, duplicate id or bad cap was not rejected with ValueError before execute ran. Validate the whole graph up front.",
            "code": r"""
ran = []
def execute(call, dep_results):
    ran.append(call['id'])
    return 0

ok = [{'id': 'a', 'deps': []}]
cases = (
    ([{'id': 'a', 'deps': ['b']}, {'id': 'b', 'deps': ['a']}], 2),
    ([{'id': 'x', 'deps': []}, {'id': 'a', 'deps': ['c']}, {'id': 'b', 'deps': ['a']}, {'id': 'c', 'deps': ['b']}], 2),
    ([{'id': 'a', 'deps': ['a']}], 2),
    ([{'id': 'a', 'deps': ['ghost']}], 2),
    ([{'id': 'a', 'deps': []}, {'id': 'a', 'deps': []}], 2),
    ([{'id': 3, 'deps': []}], 2),
    ([{'id': 'a', 'deps': 'b'}], 2),
    (ok, 0),
    (ok, -1),
    (ok, True),
    (ok, 1.5),
)
for calls, cap in cases:
    try:
        {fn}(calls, execute, cap)
    except ValueError:
        assert ran == [], f'execute ran before validation failed: {ran}'
        continue
    raise AssertionError(f'{calls!r} with cap {cap!r} should raise ValueError')
""",
        },
        {
            "name": "Calls are passed through untouched and never mutated",
            "visibility": "unshown",
            "behavior": "state.invariant",
            "failure_message": "The caller's call dictionaries were mutated or replaced. Pass each call to execute exactly as given, and leave the input list unchanged.",
            "code": r"""
import copy
received = []
def execute(call, dep_results):
    received.append(call)
    return len(dep_results)

calls = [
    {'id': 'a', 'deps': [], 'tool': 'search', 'args': {'q': 'x'}},
    {'id': 'b', 'deps': ['a'], 'tool': 'fetch', 'args': {'url': '$a'}},
]
before = copy.deepcopy(calls)
{fn}(calls, execute, 2)
assert calls == before, 'input was mutated'
assert received[0] is calls[0] and received[1] is calls[1], 'execute must receive the caller dictionaries'
""",
        },
        {
            "name": "Only Exception subclasses are treated as call failures",
            "visibility": "unshown",
            "behavior": "retry.classification",
            "failure_message": "A BaseException such as KeyboardInterrupt was swallowed as a call failure. Catch Exception, not everything.",
            "code": r"""
def execute(call, dep_results):
    if call['id'] == 'stop':
        raise KeyboardInterrupt()
    return 1

calls = [{'id': 'ok', 'deps': []}, {'id': 'stop', 'deps': []}]
try:
    {fn}(calls, execute, 2)
except KeyboardInterrupt:
    pass
else:
    raise AssertionError('KeyboardInterrupt must propagate, not become a call error')
""",
        },
        {
            "name": "Empty and fully failed plans are accounted for",
            "visibility": "unshown",
            "behavior": "edge.empty_or_boundary",
            "failure_message": "An empty plan, or a plan whose every root fails, produced the wrong buckets or rounds.",
            "code": r"""
def fail(call, dep_results):
    raise RuntimeError(call['id'])

empty = {fn}([], fail, 3)
assert empty == {'results': {}, 'errors': {}, 'skipped': [], 'rounds': []}, empty

calls = [{'id': 'a', 'deps': []}, {'id': 'b', 'deps': []}, {'id': 'c', 'deps': ['a', 'b']}]
out = {fn}(calls, fail, 5)
assert out['rounds'] == [['a', 'b']], out['rounds']
assert sorted(out['errors']) == ['a', 'b'], out['errors']
assert out['skipped'] == ['c'] and out['results'] == {}, out
""",
        },
    ],
    "solution": '''def schedule_tool_calls(calls, execute, max_parallel):
    if not isinstance(max_parallel, int) or isinstance(max_parallel, bool) or max_parallel < 1:
        raise ValueError("max_parallel must be a positive integer")

    by_id = {}
    for call in calls:
        call_id = call.get("id")
        if not isinstance(call_id, str):
            raise ValueError("every call needs a string id")
        if call_id in by_id:
            raise ValueError(f"duplicate id {call_id!r}")
        if not isinstance(call.get("deps"), list):
            raise ValueError(f"deps of {call_id!r} must be a list")
        by_id[call_id] = call

    dependents = {call_id: [] for call_id in by_id}
    indegree = {call_id: 0 for call_id in by_id}
    for call_id, call in by_id.items():
        for dep in call["deps"]:
            if dep == call_id:
                raise ValueError(f"{call_id!r} depends on itself")
            if dep not in by_id:
                raise ValueError(f"{call_id!r} depends on unknown id {dep!r}")
            dependents[dep].append(call_id)
            indegree[call_id] += 1

    frontier = [call_id for call_id, degree in indegree.items() if degree == 0]
    emitted = 0
    remaining = dict(indegree)
    while frontier:
        current = frontier.pop()
        emitted += 1
        for child in dependents[current]:
            remaining[child] -= 1
            if remaining[child] == 0:
                frontier.append(child)
    if emitted != len(by_id):
        raise ValueError("the dependencies contain a cycle")

    status = {call_id: "pending" for call_id in by_id}
    results, errors, rounds = {}, {}, []

    while True:
        ready = [
            call["id"] for call in calls
            if status[call["id"]] == "pending"
            and all(status[dep] == "succeeded" for dep in call["deps"])
        ][:max_parallel]
        if not ready:
            break
        rounds.append(ready)
        for call_id in ready:
            call = by_id[call_id]
            dep_results = {dep: results[dep] for dep in call["deps"]}
            try:
                results[call_id] = execute(call, dep_results)
                status[call_id] = "running"
            except Exception as error:
                errors[call_id] = error
                status[call_id] = "failed"
                stack = list(dependents[call_id])
                while stack:
                    child = stack.pop()
                    if status[child] == "pending":
                        status[child] = "skipped"
                        stack.extend(dependents[child])
        for call_id in ready:
            if status[call_id] == "running":
                status[call_id] = "succeeded"

    skipped = [call["id"] for call in calls if status[call["id"]] == "skipped"]
    return {"results": results, "errors": errors, "skipped": skipped, "rounds": rounds}
''',
}
