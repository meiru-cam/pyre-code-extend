"""Count and map a tree of machines by message passing, then survive lost and repeated messages."""

from ._interview import interview

# The simulated networks. Network and UnreliableNetwork are adapted from the MIT-licensed code in
# Schuture/OpenAI-Interview-Notes; Scripted drops or repeats chosen messages so a case can check
# exactly what a node sends.
_HELPERS = r"""
import random

class Network:
    def __init__(self, seed=0):
        self.nodes, self._queue, self._rng = {}, [], random.Random(seed)
        self.sent = []
    def register(self, node):
        self.nodes[node.node_id] = node
    def send_async_message(self, from_id, to_id, message):
        self.sent.append((from_id, to_id, message))
        self._queue.append((to_id, from_id, message))
    def deliver_round(self):
        batch, self._queue = self._queue, []
        self._rng.shuffle(batch)
        for to_id, from_id, message in batch:
            self.nodes[to_id].receive_message(from_id, message)
    def run(self, max_rounds=10_000):
        rounds = 0
        while self._queue and rounds < max_rounds:
            self.deliver_round()
            rounds += 1
        return rounds

class TickingNetwork(Network):
    # After every round, quiet or not, calls on_tick() on every node; stops after 30 quiet rounds.
    def run(self, max_rounds=10_000, idle_rounds_to_stop=30):
        idle = rounds = 0
        while rounds < max_rounds:
            if self._queue:
                self.deliver_round()
                idle = 0
            else:
                idle += 1
            for node in list(self.nodes.values()):
                node.on_tick()
            rounds += 1
            if idle >= idle_rounds_to_stop and not self._queue:
                break
        return rounds

class UnreliableNetwork(TickingNetwork):
    def __init__(self, drop_prob=0.0, duplicate_prob=0.0, seed=0):
        super().__init__(seed=seed)
        self.drop_prob, self.duplicate_prob = drop_prob, duplicate_prob
    def send_async_message(self, from_id, to_id, message):
        self.sent.append((from_id, to_id, message))
        if self._rng.random() < self.drop_prob:
            return
        self._queue.append((to_id, from_id, message))
        if self._rng.random() < self.duplicate_prob:
            self._queue.append((to_id, from_id, message))

class Scripted(TickingNetwork):
    # drop / repeat: sets of (from_id, to_id, n), meaning the n-th message on that link, from 1.
    def __init__(self, drop=(), repeat=(), seed=0):
        super().__init__(seed=seed)
        self.drop, self.repeat, self.counts = set(drop), set(repeat), {}
    def send_async_message(self, from_id, to_id, message):
        self.sent.append((from_id, to_id, message))
        n = self.counts[(from_id, to_id)] = self.counts.get((from_id, to_id), 0) + 1
        if (from_id, to_id, n) in self.drop:
            return
        self._queue.append((to_id, from_id, message))
        if (from_id, to_id, n) in self.repeat:
            self._queue.append((to_id, from_id, message))
    def count(self, from_id, to_id):
        return sum(1 for f, t, _ in self.sent if (f, t) == (from_id, to_id))

EXAMPLE = {"A": ["B", "C", "D"], "B": ["E", "F"], "D": ["G"], "G": ["H"]}

def build(cls, network, children, root="A"):
    parent = {c: p for p, cs in children.items() for c in cs}
    ids = [root] + [c for cs in children.values() for c in cs]
    nodes = {i: cls(i, parent.get(i), list(children.get(i, [])), network) for i in ids}
    return nodes[root]

def random_tree(rng, n):
    children = {}
    for i in range(1, n):
        children.setdefault(f"n{rng.randrange(i)}", []).append(f"n{i}")
    return children

def shape(children, node):
    return (node, [shape(children, c) for c in children.get(node, [])])
"""

TASK = {
    "title": "Cluster Count and Topology",
    "difficulty": "Medium",
    "version": 1,
    "function_name": "ClusterNode",
    "description_en": r"""Build `ClusterNode`, one machine in a tree-shaped cluster, so that the root can count the machines and map the tree by passing messages.

The requirement arrives in parts. Each part keeps every earlier behavior, so one `ClusterNode` class passes all parts at the end. Pass every test of the current part to reveal the next one.

**Rules for every part:**
- `ClusterNode(node_id, parent_id, children_ids, network)` is one machine. `parent_id` is `None` only for the root. The constructor calls `network.register(self)`.
- A machine talks only to its parent and its children, with `network.send_async_message(self.node_id, to_id, message)`. A message can be any Python value you choose.
- Sending only queues the message. The network later calls `receive_message(from_id, message)` on the target, in rounds, in a random order within each round. Messages sent during a round are delivered in a later round.
- A query starts when the caller invokes `root.receive_message(None, ("request", query_type, request_id))`. When the network is done, the root holds the answer in `root.results[request_id]`.
- Several queries with different `request_id`s may run at the same time.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why this shows up in interviews:** the first part is a recursive sum written as messages, and each later part adds one requirement.

**Where it is used:** cluster membership and monitoring, and distributed aggregation over a spanning tree, as in sensor networks and gossip protocols.

Adapted from the cluster count and topology question in Schuture/OpenAI-Interview-Notes (CC BY-NC 4.0), reworded, on one class whose queries all carry a `request_id`, with the resend window fixed at 3 to 10 ticks so it can be tested.""",
    "parts": [
        {
            "title": "Count the machines",
            "description_en": r"""**Signature:** `ClusterNode(node_id, parent_id, children_ids, network)`, `receive_message(from_id, message) -> None`

- For `query_type == "count"`, the root's result is the number of machines in the tree, the root included.
- A machine without children answers at once. A machine with children asks every child, waits until all of them have answered, in any order, and then reports its own total.
- No machine may read another machine's attributes; everything goes through messages.

**Example:** in the tree where `A` has children `B`, `C`, `D`, `B` has `E` and `F`, `D` has `G`, and `G` has `H`:
- `A.receive_message(None, ("request", "count", "q1"))`, then `network.run()`
- `A.results["q1"]` is `8`""",
        },
        {
            "title": "Map the tree",
            "description_en": r"""Keep Part 1 and add a second query type.

- For `query_type == "topology"`, the result describes the subtree under each machine as a tuple `(node_id, [subtrees of its children])`, children in the order of `children_ids`. A machine without children is `(node_id, [])`.
- Answers still arrive in any order, so put them back in `children_ids` order yourself.

**Example:** in the same tree, the result of a topology query is `("A", [("B", [("E", []), ("F", [])]), ("C", []), ("D", [("G", [("H", [])])])])`.""",
        },
        {
            "title": "Lost and repeated messages",
            "description_en": r"""**Signature:** `on_tick() -> None`

Keep Parts 1–2. The network may now drop any message or deliver it twice. After every round it calls `on_tick()` on every machine, and it stops once 30 rounds in a row deliver nothing.

- A request for a `request_id` that a machine is still working on is ignored.
- A request for a `request_id` that a machine has finished is answered again with the saved result, without asking its children again.
- A child's answer counts once per `request_id`, however many copies arrive.
- When a child has not answered a request, ask that child again, and only that child, at least 3 and at most 10 ticks after asking it last.

**Example:** with each message dropped with probability `0.2` and duplicated with probability `0.2`, `A.results["q1"]` is `8` for every network seed.""",
        },
    ],
    "hints": [
        {"level": 1, "kind": "questions", "content": "What does a machine need to remember between the request arriving and its last child answering? How does a leaf know it is a leaf? Where does the root put its total, given that it has no parent to send to?"},
        {"level": 2, "kind": "analysis", "content": "Keep, per request_id, the set of children still pending and a dict of answers by child id. On a request, a leaf answers at once; otherwise forward it to every child. On an answer, record it and drop that child from pending; when pending is empty, report 1 plus the sum to the parent, or store it in results at the root."},
    ],
    "model_connections": [
        "Distributed systems aggregate over a spanning tree, as in the convergecast step of sensor networks and in cluster monitors that roll up per-node metrics.",
        "Real RPC layers retry on timeout and deduplicate by request id, because networks lose and repeat messages; this is the at-least-once delivery that idempotent handlers make safe.",
    ],
    "pro_con_analysis": {
        "pros": [
            "Broadcast down and aggregate up visits every machine once and needs no global view.",
            "Keying answers by child id makes repeated answers harmless and keeps child order for the topology.",
            "Caching each finished result per request id turns a repeated request into a cheap replay.",
        ],
        "cons": [
            "A machine waits for its slowest child, so one slow subtree delays the whole answer.",
            "Per-request state kept forever grows without bound; a real system expires it.",
            "Timeouts trade latency for extra traffic: short ones resend needlessly, long ones wait after a loss.",
        ],
    },
    "tests": [
        {"name": "Part 1: the worked example", "part": 1, "behavior": "state.invariant", "code": _HELPERS + r"""
network = Network(seed=0)
root = build({fn}, network, EXAMPLE)
root.receive_message(None, ("request", "count", "q1"))
network.run()
assert root.results["q1"] == 8
"""},
        {"name": "Part 1: any delivery order, tiny trees and parallel queries", "part": 1, "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "The count must not depend on delivery order, a lone root counts 1, and two queries with different request ids must not mix.",
         "code": _HELPERS + r"""
for seed in range(50):
    network = Network(seed=seed)
    root = build({fn}, network, EXAMPLE)
    root.receive_message(None, ("request", "count", "a"))
    root.receive_message(None, ("request", "count", "b"))
    network.run()
    assert root.results["a"] == root.results["b"] == 8, seed
network = Network()
lone = {fn}("solo", None, [], network)
lone.receive_message(None, ("request", "count", "x"))
network.run()
assert lone.results["x"] == 1
"""},
        {"name": "Part 1: only messages between neighbours", "part": 1, "visibility": "unshown", "behavior": "protocol.validation",
         "failure_message": "Every message must go from a machine to its parent or one of its children, and every machine must send at least one.",
         "code": _HELPERS + r"""
network = Network(seed=3)
root = build({fn}, network, EXAMPLE)
root.receive_message(None, ("request", "count", "q"))
network.run()
parent = {c: p for p, cs in EXAMPLE.items() for c in cs}
for from_id, to_id, _ in network.sent:
    assert parent.get(to_id) == from_id or parent.get(from_id) == to_id, (from_id, to_id)
assert {f for f, _, _ in network.sent} == set("ABCDEFGH")
"""},
        {"name": "Part 1: random trees", "part": 1, "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "On a random tree the count differed from the number of machines.",
         "code": _HELPERS + r"""
for seed in range(60):
    rng = random.Random(seed)
    n = rng.randint(1, 40)
    network = Network(seed=seed)
    root = build({fn}, network, random_tree(rng, n), root="n0")
    root.receive_message(None, ("request", "count", seed))
    network.run()
    assert root.results[seed] == n, (seed, n)
"""},
        {"name": "Part 2: the worked example", "part": 2, "behavior": "state.invariant", "code": _HELPERS + r"""
network = Network(seed=1)
root = build({fn}, network, EXAMPLE)
root.receive_message(None, ("request", "topology", "t1"))
network.run()
assert root.results["t1"] == ("A", [("B", [("E", []), ("F", [])]), ("C", []), ("D", [("G", [("H", [])])])])
"""},
        {"name": "Part 2: child order and mixed queries", "part": 2, "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "Subtrees must follow children_ids order whatever order answers arrive in, and a count and a topology query may run together.",
         "code": _HELPERS + r"""
tree = {"r": ["z", "a", "m"], "a": ["y", "b"]}
for seed in range(40):
    network = Network(seed=seed)
    root = build({fn}, network, tree, root="r")
    root.receive_message(None, ("request", "topology", "t"))
    root.receive_message(None, ("request", "count", "c"))
    network.run()
    assert root.results["t"] == ("r", [("z", []), ("a", [("y", []), ("b", [])]), ("m", [])]), seed
    assert root.results["c"] == 6
"""},
        {"name": "Part 2: random trees", "part": 2, "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "On a random tree the topology differed from the tree's actual shape.",
         "code": _HELPERS + r"""
for seed in range(60):
    rng = random.Random(100 + seed)
    tree = random_tree(rng, rng.randint(1, 30))
    network = Network(seed=seed)
    root = build({fn}, network, tree, root="n0")
    root.receive_message(None, ("request", "topology", "t"))
    network.run()
    assert root.results["t"] == shape(tree, "n0"), seed
"""},
        {"name": "Part 3: an unreliable network", "part": 3, "behavior": "retry.classification", "code": _HELPERS + r"""
for seed in range(150):
    network = UnreliableNetwork(drop_prob=0.2, duplicate_prob=0.2, seed=seed)
    root = build({fn}, network, EXAMPLE)
    root.receive_message(None, ("request", "count", "q1"))
    root.receive_message(None, ("request", "topology", "t1"))
    network.run()
    assert root.results["q1"] == 8, seed
    assert root.results["t1"] == ("A", [("B", [("E", []), ("F", [])]), ("C", []), ("D", [("G", [("H", [])])])]), seed
"""},
        {"name": "Part 3: duplicates start nothing new", "part": 3, "visibility": "unshown", "behavior": "effects.idempotency",
         "failure_message": "A repeated request must not make a machine ask its children twice, and a repeated answer must not be counted twice.",
         "code": _HELPERS + r"""
# B asks its leaves in round 1 and hears back in round 3, before any timeout can fire.
network = Scripted(repeat={("A", "B", 1), ("B", "E", 1), ("E", "B", 1)})
root = build({fn}, network, {"A": ["B"], "B": ["E", "F"]})
root.receive_message(None, ("request", "count", "q"))
network.run()
assert root.results["q"] == 4, "an answer delivered twice was counted twice"
assert network.count("B", "E") == 1 and network.count("B", "F") == 1, "B asked a child twice"
before = [network.count("A", "B"), network.count("B", "E")]
root.receive_message(None, ("request", "count", "q"))
network.run()
assert root.results["q"] == 4
assert [network.count("A", "B"), network.count("B", "E")] == before, "a finished query must be answered from the saved result"
"""},
        {"name": "Part 3: only the silent child is asked again", "part": 3, "visibility": "unshown", "behavior": "retry.backoff",
         "failure_message": "After a lost message, the machine must ask again only the child that has not answered, and a finished child must reply from its saved result without asking its own children.",
         "code": _HELPERS + r"""
network = Scripted(drop={("B", "E", 1)})
root = build({fn}, network, {"B": ["E", "F"]}, root="B")
root.receive_message(None, ("request", "count", "q"))
network.run()
assert root.results["q"] == 3
assert network.count("B", "E") == 2 and network.count("B", "F") == 1, (network.count("B", "E"), network.count("B", "F"))
network = Scripted(drop={("D", "A", 1)})
root = build({fn}, network, {"A": ["D"], "D": ["G"]})
root.receive_message(None, ("request", "topology", "t"))
network.run()
assert root.results["t"] == ("A", [("D", [("G", [])])])
assert network.count("D", "A") >= 2
assert network.count("D", "G") == 1, "D had already finished, so it must not ask G again"
"""},
        {"name": "Part 3: resend timing", "part": 3, "visibility": "unshown", "behavior": "retry.backoff",
         "failure_message": "A request to a silent child must be repeated no sooner than 3 ticks and no later than 10 ticks after the last one.",
         "code": _HELPERS + r"""
class Clocked(Scripted):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ticks, self.stamps = 0, []
    def send_async_message(self, from_id, to_id, message):
        if (from_id, to_id) == ("C", "X"):
            self.stamps.append(self.ticks)
        super().send_async_message(from_id, to_id, message)
    def run(self, **kwargs):
        nodes = self.nodes
        original = {i: n.on_tick for i, n in nodes.items()}
        def counting_tick():
            self.ticks += 1
            original["C"]()
        nodes["C"].on_tick = counting_tick
        return super().run(**kwargs)
network = Clocked(drop={("C", "X", 1), ("C", "X", 2)})
root = build({fn}, network, {"C": ["X"]}, root="C")
root.receive_message(None, ("request", "count", "q"))
network.run()
assert root.results["q"] == 2
gaps = [b - a for a, b in zip(network.stamps, network.stamps[1:])]
assert len(network.stamps) == 3 and all(3 <= g <= 10 for g in gaps), network.stamps
"""},
    ],
    "solution": r'''# Adapted from Schuture/OpenAI-Interview-Notes (code under the MIT License).

QUERIES = {
    # query type -> (value of a machine without children, combine(node, answers in children_ids order))
    "count": (lambda node: 1, lambda node, answers: 1 + sum(answers)),
    "topology": (lambda node: (node.node_id, []), lambda node, answers: (node.node_id, answers)),
}


class ClusterNode:
    TIMEOUT = 3  # ticks to wait for a child before asking it again

    def __init__(self, node_id, parent_id, children_ids, network):
        self.node_id = node_id
        self.parent_id = parent_id
        self.children_ids = list(children_ids)
        self.network = network
        self.results = {}   # request_id -> final value, on the root only
        self._state = {}    # request_id -> progress; kept so repeated requests are answered from it
        self._tick = 0
        network.register(self)

    def _send(self, to_id, message):
        self.network.send_async_message(self.node_id, to_id, message)

    def receive_message(self, from_id, message):
        if message[0] == "request":
            _, query_type, request_id = message
            state = self._state.get(request_id)
            if state is not None:
                if state["done"]:
                    self._reply(request_id)
                return
            self._state[request_id] = {
                "query_type": query_type,
                "pending": set(self.children_ids),
                "answers": {},  # child id -> value, so a repeated answer overwrites, never adds
                "done": False,
                "value": None,
                "asked_at": {child: self._tick for child in self.children_ids},
            }
            if not self.children_ids:
                self._finish(request_id, QUERIES[query_type][0](self))
            for child in self.children_ids:
                self._send(child, ("request", query_type, request_id))
        else:
            _, request_id, value = message
            state = self._state.get(request_id)
            if state is None or from_id not in state["pending"]:
                return
            state["answers"][from_id] = value
            state["pending"].discard(from_id)
            if not state["pending"]:
                combine = QUERIES[state["query_type"]][1]
                self._finish(request_id, combine(self, [state["answers"][c] for c in self.children_ids]))

    def _finish(self, request_id, value):
        state = self._state[request_id]
        state["done"], state["value"] = True, value
        self._reply(request_id)

    def _reply(self, request_id):
        value = self._state[request_id]["value"]
        if self.parent_id is None:
            self.results[request_id] = value
        else:
            self._send(self.parent_id, ("response", request_id, value))

    def on_tick(self):
        self._tick += 1
        for request_id, state in self._state.items():
            for child in self.children_ids:
                if child in state["pending"] and self._tick - state["asked_at"][child] >= self.TIMEOUT:
                    self._send(child, ("request", state["query_type"], request_id))
                    state["asked_at"][child] = self._tick
''',
    "interview_questions": interview(
        concept=[
            "What does a machine store between forwarding a request and hearing from its last child?",
            "How does a machine know that every one of its children has answered?",
        ],
        deep_dive=[
            "Two queries with different request ids reach the same machine at once: what keeps their state apart?",
        ],
        tradeoffs=[
            "Why record each child's answer by child id instead of keeping a running total, both for ordering subtrees and for repeated answers?",
            "Which messages can be repeated or lost, and what state makes each case harmless?",
            "How do you choose the resend timeout, and what goes wrong if it is too short or too long?",
        ],
    ),
}
