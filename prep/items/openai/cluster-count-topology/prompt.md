A cluster's machines form a rooted tree. Each machine can send a message only to its *parent* or to one of its *children*; there is no other channel between machines, and in particular two siblings cannot talk directly. Communication is asynchronous: sending a message only queues it, and the receiving machine's `receive_message` runs at some later, unpredictable time. The exercise runs on the following simulated network, given here in full and used by every part below.

```python
import random


class Network:
    """A minimal asynchronous network. send_async_message only enqueues a message; nothing is
    delivered synchronously. run() repeatedly takes every message currently queued, shuffles it
    with the network's own random generator (delivery order has nothing to do with send order),
    and calls receive_message on each target in that shuffled order. A handler invoked while a
    round is being delivered may itself call send_async_message; those new messages join the queue
    for the NEXT round, never the one being delivered, so a message can never be processed before
    it is sent. run() stops once the queue is empty, or after max_rounds as a safety net against a
    handler that keeps sending forever."""

    def __init__(self, seed=0):
        self.nodes = {}
        self._queue = []                 # list of (to_id, from_id, message), not yet delivered
        self._rng = random.Random(seed)  # NOTE: seeded, so a run is reproducible for a fixed seed

    def register(self, node):
        self.nodes[node.node_id] = node

    def send_async_message(self, from_id, to_id, message):
        self._queue.append((to_id, from_id, message))

    def run(self, max_rounds=10_000):
        rounds = 0
        while self._queue and rounds < max_rounds:
            batch, self._queue = self._queue, []       # NOTE: freeze this round's batch first, so
            self._rng.shuffle(batch)                    #       sends made while delivering it land
            for to_id, from_id, message in batch:       #       in self._queue for the NEXT round
                self.nodes[to_id].receive_message(from_id, message)
            rounds += 1
        return rounds


class Node:
    """Given skeleton, subclassed for every part. node_id is this machine's id, parent_id its
    parent's id (None only for the root), children_ids the ids of its children, and network the
    Network instance to send through."""

    def __init__(self, node_id, parent_id, children_ids, network):
        self.node_id = node_id
        self.parent_id = parent_id
        self.children_ids = list(children_ids)
        self.network = network
        network.register(self)

    def send(self, to_id, message):
        self.network.send_async_message(self.node_id, to_id, message)

    def receive_message(self, from_id, message):
        raise NotImplementedError

    def on_tick(self):
        """Called once per round by UnreliableNetwork.run() (Part 3); unused before that, and the
        base Network above never calls it."""
```

A `message` can be any Python object you choose — a string, a tuple, a small dict; this simulation never inspects or serialises it, so its shape is entirely up to you. The process that starts a query calls `receive_message(None, ...)` directly on the root, exactly as if it were a message "from outside the cluster"; `from_id` is `None` only for that one call, never for a message from another machine.

The tree used in the examples below has 8 machines:

```text
        A
     /  |  \
    B   C   D
   / \      |
  E   F     G
            |
            H
```

### Part 1 — Counting the cluster

Implement `receive_message` on a `Node` subclass so that, once `Network.run()` returns, the root's own `result` attribute holds the number of machines in the cluster, including the root itself.

- The count starts when the root receives the initial `receive_message(None, ...)` call.
- A machine with no children answers immediately, without contacting anyone.
- A machine with children forwards the request to all of them and waits until every one of them has answered before adding 1 for itself and reporting the total to its own parent — or, if it is the root, storing it in `result` instead of sending it anywhere.
- Children may answer in any order, and the code must not assume a particular one.
- If the root itself has no children, the count is 1.

```py
class CountingNode(Node):
    def receive_message(self, from_id: str | None, message) -> None:
        """After Network.run() returns, self.result on the ROOT holds the number of machines in
        the whole cluster."""
```

On the tree above: `A` forwards to `B`, `C`, `D`. `C` is a leaf and answers `1` right away. `B` forwards to `E` and `F`, both leaves, and once both have answered — in whichever order they arrive — reports `1 + 1 + 1 = 3` to `A`. `D` forwards to `G`, which forwards to `H`; `H` answers `1`, `G` reports `1 + 1 = 2`, and `D` reports `1 + 2 = 3`. Once `A` has heard from `B`, `C` and `D`, in whatever order, it computes `1 + 3 + 1 + 3 = 8`, which is `A.result`.

### Part 2 — Mapping the topology

Do the same downward-request, upward-aggregation as Part 1, but gather the shape of the whole tree instead of a count. Represent the subtree rooted at a machine with id `x` and children subtrees `t_1, ..., t_k`, listed in the same order as `children_ids`, as the nested tuple `(x, [t_1, ..., t_k])`; a leaf is `(x, [])`. Once `Network.run()` returns, the root's `result` attribute must hold this tuple for the whole cluster.

```py
class TopologyNode(Node):
    def receive_message(self, from_id: str | None, message) -> None:
        """After Network.run() returns, self.result on the ROOT holds (node_id, [child topologies])
        for the whole cluster, children listed in the order of children_ids."""
```

On the tree above, the result is `("A", [("B", [("E", []), ("F", [])]), ("C", []), ("D", [("G", [("H", [])])])])`.

### Part 3 — Duplicated and lost messages

Requests and responses can now be delivered more than once, or not at all. `Network` gains two independent per-message probabilities and a logical clock, given here in full:

```python
class UnreliableNetwork(Network):
    """A message is independently dropped with probability drop_prob (never delivered at all) and,
    if not dropped, duplicated with probability duplicate_prob (a second, unrelated copy is queued,
    so receive_message runs twice for that one send). After every round -- including one where the
    queue happened to be empty -- on_tick() is called once on every registered node; this is the
    model's only notion of time, and it is what a node uses to notice that a child has been silent
    for too long and resend to it. run() stops once the queue has been empty for
    idle_rounds_to_stop consecutive rounds, or after max_rounds, whichever comes first."""

    def __init__(self, drop_prob=0.0, duplicate_prob=0.0, seed=0):
        super().__init__(seed=seed)
        self.drop_prob = drop_prob
        self.duplicate_prob = duplicate_prob

    def send_async_message(self, from_id, to_id, message):
        if self._rng.random() < self.drop_prob:
            return                                        # NOTE: silently discarded, never queued
        super().send_async_message(from_id, to_id, message)
        if self._rng.random() < self.duplicate_prob:
            super().send_async_message(from_id, to_id, message)   # a second, independent delivery

    def run(self, max_rounds=10_000, idle_rounds_to_stop=30):
        idle, rounds = 0, 0
        while rounds < max_rounds:
            if self._queue:
                batch, self._queue = self._queue, []
                self._rng.shuffle(batch)
                for to_id, from_id, message in batch:
                    self.nodes[to_id].receive_message(from_id, message)
                idle = 0
            else:
                idle += 1
            for node in self.nodes.values():
                node.on_tick()                            # NOTE: ticks even on an idle round, so a
            rounds += 1                                    #       pending timeout still fires
            if idle >= idle_rounds_to_stop and not self._queue:
                break
        return rounds
```

`run()` cannot tell whether a query has finished: it treats `idle_rounds_to_stop` consecutive rounds with an empty queue as the end. A dropped message leaves nothing in the queue, so a machine that is still waiting for an answer has to resend well within that many ticks, or `run()` returns before the result exists.

Redo Part 1 and Part 2 so the final result is still correct when running on an `UnreliableNetwork`. Every query now carries a `request_id`, unique to that one invocation, chosen by whoever starts it (the caller that invokes `receive_message(None, ...)` on the root).

- A machine that receives a request whose `request_id` it has already finished must resend its cached answer for that request, not recompute it.
- A machine that receives a request whose `request_id` it is still working on must ignore the duplicate rather than starting a second aggregation.
- A machine must not count the same child's answer twice: two responses from the same child for the same `request_id` must not both be folded into the total.
- If a machine has not heard back from one of its children for a given `request_id` within some number of ticks, it must resend the request to that child alone, not to every child again.

```py
class RobustNode(Node):
    def receive_message(self, from_id: str | None, message) -> None:
        """Same task as Parts 1-2, run over an UnreliableNetwork. Once the query for a given
        request_id has finished, its value is stored in self.results[request_id] on the ROOT;
        correct even when requests or responses are duplicated or dropped."""

    def on_tick(self) -> None:
        """Called once per round by UnreliableNetwork.run(). Resend a request to any child that has
        been pending too long for some request_id."""
```

On the tree above, with `drop_prob = duplicate_prob = 0.2`, after a start call such as `root.receive_message(None, ("request", "count", "q1"))` followed by `network.run()`, `root.results["q1"]` must equal `8`, exactly as in Part 1, for every network seed from 0 to 999.
