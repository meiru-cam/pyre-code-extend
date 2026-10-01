"""A follow graph with frozen snapshots, growing one query per part."""

from ._interview import interview

# A slow model written straight from the statement: every snapshot is a deep copy, and every
# query scans it. It shares no code with the reference.
_HELPERS = r"""
import random, time
from collections import Counter

def raises_value_error(fn, *args):
    try:
        fn(*args)
    except ValueError:
        return True
    return False

class CopyNetwork:
    def __init__(self):
        self.following = {}
    def add_user(self, u):
        if u in self.following:
            return "error"
        self.following[u] = set()
    def follow(self, a, b):
        if a not in self.following or b not in self.following:
            return "error"
        if a != b:
            self.following[a].add(b)
    def unfollow(self, a, b):
        if a not in self.following or b not in self.following:
            return "error"
        self.following[a].discard(b)
    def create_snapshot(self):
        return CopySnapshot({u: set(fs) for u, fs in self.following.items()})

class CopySnapshot:
    def __init__(self, following):
        self.following = following
    def is_following(self, a, b):
        return b in self.following[a]
    def get_following(self, u):
        return sorted(self.following[u])
    def get_followers(self, u):
        return sorted(a for a, fs in self.following.items() if u in fs)
    def recommend(self, u, k):
        direct = self.following[u]
        score = Counter(c for f in direct for c in self.following[f] if c != u and c not in direct)
        return sorted(score, key=lambda c: (-score[c], c))[:k]

def mirrored(rng, steps, users, unfollows):
    # Applies the same random calls to the learner's network and to CopyNetwork, and returns
    # both lists of snapshots. A call the model rejects must raise ValueError in the learner's.
    net, model = {fn}(), CopyNetwork()
    pairs = []
    for _ in range(steps):
        roll = rng.random()
        if roll < 0.15:
            u = rng.choice(users)
            op, args = "add_user", (u,)
        elif roll < 0.25:
            pairs.append((net.create_snapshot(), model.create_snapshot()))
            continue
        else:
            op = "unfollow" if unfollows and rng.random() < 0.3 else "follow"
            args = (rng.choice(users), rng.choice(users))
        expected_error = getattr(model, op)(*args) == "error"
        assert raises_value_error(getattr(net, op), *args) == expected_error, (op, args)
    pairs.append((net.create_snapshot(), model.create_snapshot()))
    return pairs

def best_of_three(fn):
    best = float("inf")
    for _ in range(3):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best
"""

TASK = {
    "title": "Follow Graph with Snapshots",
    "difficulty": "Medium",
    "version": 1,
    "function_name": "SocialNetwork",
    "description_en": r"""Build `SocialNetwork`, a directed follow graph whose snapshots keep answering as of the moment they were taken.

The requirement arrives in parts. Each part keeps every earlier behavior, so one `SocialNetwork` class passes all parts at the end. Pass every test of the current part to reveal the next one.

**Rules for every part:**
- User ids are strings. Following is directed: `a` following `b` says nothing about `b` following `a`.
- A snapshot is the object `create_snapshot()` returns. Calls made on the network afterwards never change what an existing snapshot answers.
- Every method that takes a user id raises `ValueError` when that user does not exist, on the network or as of the snapshot.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why this shows up in interviews:** the first part looks like a dictionary of sets, and each later part adds one requirement.

**Where it is used:** social graphs, and any store that serves reads as of a fixed version while writes continue.

Adapted from the social network question in Schuture/OpenAI-Interview-Notes (CC BY-NC 4.0), reworded, with a cost requirement added to the lists and the history part rebuilt on the same class as cheap snapshots.""",
    "parts": [
        {
            "title": "Users, follows and snapshots",
            "description_en": r"""**Signature:** `add_user(user_id) -> None`, `follow(follower, followee) -> None`, `create_snapshot() -> snapshot`, and on the snapshot `is_following(follower, followee) -> bool`

- `add_user` registers a user and raises `ValueError` if the id already exists.
- `follow` makes `follower` follow `followee`. Following yourself, or following someone you already follow, changes nothing.
- `create_snapshot` freezes the current users and follows.
- `is_following` answers as of the snapshot. A user added after the snapshot does not exist for it.

**Example:**
- Add `ana`, `ben`, `cy`, `dee`; `follow("ana", "ben")`, `follow("ana", "cy")`; `s1 = create_snapshot()`
- Then `follow("ben", "dee")`
- `s1.is_following("ana", "cy")` is `True`; `s1.is_following("ben", "dee")` is `False`""",
        },
        {
            "title": "Following and follower lists",
            "description_en": r"""**Signature:** on the snapshot, `get_following(user_id) -> list[str]` and `get_followers(user_id) -> list[str]`

Keep Part 1 and add:

- `get_following` lists the users `user_id` follows, and `get_followers` the users who follow `user_id`, both as of the snapshot and sorted ascending.
- Each call costs about the size of its own answer, not the size of the whole graph: on a snapshot of 20,000 users, a list of two names must come back about as fast as on a snapshot of twenty.

**Example:** after `cy` also follows `ana`, `ben` and `dee`, a new snapshot `s2` gives:
- `s2.get_following("cy")` is `["ana", "ben", "dee"]`
- `s2.get_followers("dee")` is `["ben", "cy"]`""",
        },
        {
            "title": "Two-hop recommendations",
            "description_en": r"""**Signature:** on the snapshot, `recommend(user_id, k) -> list[str]`

Keep Parts 1–2 and add:

- A candidate is a user followed by one of the users `user_id` follows, except `user_id` itself and anyone `user_id` already follows.
- A candidate's score is the number of different users `user_id` follows that lead to it.
- Return the top `k` candidates by score, highest first, ties broken by user id ascending. Return fewer when fewer exist.

**Example:** with `s2`, where `ana` follows `ben` and `cy`, `ben` follows `dee`, and `cy` follows `ana`, `ben` and `dee`:
- `s2.recommend("ana", 5)` is `["dee"]`: `dee` scores 2, and `ben` is already followed""",
        },
        {
            "title": "Unfollow and cheap snapshots",
            "description_en": r"""**Signature:** `unfollow(follower, followee) -> None`

Keep Parts 1–3 and add:

- `unfollow` ends a follow. Unfollowing someone you do not follow, or yourself, changes nothing; unknown users raise `ValueError`.
- `create_snapshot` must not copy the graph: taking a snapshot of a network with 40,000 follows must be about as fast as of one with ten.
- Every older snapshot still answers every query as of its own moment, including follows that ended later.

**Example:**
- `follow("ana", "ben")`, `s3 = create_snapshot()`, `unfollow("ana", "ben")`, `s4 = create_snapshot()`
- `s3.is_following("ana", "ben")` is `True`; `s4.is_following("ana", "ben")` is `False`""",
        },
    ],
    "hints": [
        {"level": 1, "kind": "questions", "content": "What structure gives you a user's followees with O(1) membership and no duplicates? If a snapshot stores the network's dictionary, what happens to it on the next follow? What exactly does copying the outer dictionary copy?"},
        {"level": 2, "kind": "analysis", "content": "Keep a dict from each user to the set of users they follow. The simplest correct snapshot holds its own copy of the users and of every inner set, so later follows cannot reach it. Check both users exist before any change, and ignore self-follows."},
    ],
    "model_connections": [
        "Social graphs such as Twitter's follow graph serve timelines and \"who to follow\" from a frozen version while new follows keep arriving.",
        "Multi-version stores like PostgreSQL's MVCC let a reader see the database as of its own snapshot, by recording when each row version starts and ends instead of copying tables.",
    ],
    "pro_con_analysis": {
        "pros": [
            "A dict of sets makes follow, unfollow and membership O(1).",
            "A reverse index makes the follower list cost the size of its answer.",
            "Recording when each follow starts and ends makes a snapshot O(1) and keeps every old version answerable.",
        ],
        "cons": [
            "Deep-copying the graph per snapshot costs O(users + follows) every time.",
            "A versioned history answers is_following with a binary search instead of a set lookup.",
            "History grows with every follow and unfollow, so a long-lived network needs compaction of versions nobody holds.",
        ],
    },
    "tests": [
        {"name": "Part 1: the worked example", "part": 1, "behavior": "state.invariant", "code": r"""
net = {fn}()
for user in ("ana", "ben", "cy", "dee"):
    net.add_user(user)
net.follow("ana", "ben")
net.follow("ana", "cy")
s1 = net.create_snapshot()
net.follow("ben", "dee")
assert s1.is_following("ana", "cy") is True
assert s1.is_following("ben", "dee") is False
assert net.create_snapshot().is_following("ben", "dee") is True
assert s1.is_following("cy", "ana") is False, "following is directed"
"""},
        {"name": "Part 1: no-ops and errors", "part": 1, "visibility": "unshown", "behavior": "protocol.validation",
         "failure_message": "Following yourself or repeating a follow must change nothing; a duplicate user or an unknown id must raise ValueError.",
         "code": _HELPERS + r"""
net = {fn}()
net.add_user("ana")
net.add_user("ben")
assert raises_value_error(net.add_user, "ana")
assert raises_value_error(net.follow, "ana", "zed")
assert raises_value_error(net.follow, "zed", "ana")
net.follow("ana", "ana")
net.follow("ana", "ben")
net.follow("ana", "ben")
snap = net.create_snapshot()
assert snap.is_following("ana", "ana") is False
assert snap.is_following("ana", "ben") is True, "a repeated follow must not undo the first"
assert raises_value_error(snap.is_following, "ana", "zed")
"""},
        {"name": "Part 1: snapshots are frozen", "part": 1, "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "A snapshot changed after it was taken: it saw a later follow or a user added later.",
         "code": _HELPERS + r"""
net = {fn}()
net.add_user("ana")
net.add_user("ben")
before = net.create_snapshot()
net.follow("ana", "ben")
net.add_user("cy")
net.follow("cy", "ana")
after = net.create_snapshot()
assert before.is_following("ana", "ben") is False
assert raises_value_error(before.is_following, "cy", "ana"), "cy did not exist yet"
assert after.is_following("ana", "ben") and after.is_following("cy", "ana")
"""},
        {"name": "Part 1: random calls match a reference", "part": 1, "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "Under a random mix of add_user, follow and create_snapshot, some snapshot answered is_following differently from a deep copy.",
         "code": _HELPERS + r"""
for seed in range(60):
    rng = random.Random(seed)
    users = [f"u{i}" for i in range(6)]
    for snap, model in mirrored(rng, 60, users, unfollows=False):
        for a in model.following:
            for b in model.following:
                assert snap.is_following(a, b) == model.is_following(a, b), (seed, a, b)
"""},
        {"name": "Part 2: the worked example", "part": 2, "behavior": "state.invariant", "code": r"""
net = {fn}()
for user in ("ana", "ben", "cy", "dee"):
    net.add_user(user)
for a, b in [("ana", "ben"), ("ana", "cy"), ("ben", "dee"), ("cy", "ana"), ("cy", "ben"), ("cy", "dee")]:
    net.follow(a, b)
s2 = net.create_snapshot()
assert s2.get_following("cy") == ["ana", "ben", "dee"]
assert s2.get_followers("dee") == ["ben", "cy"]
assert s2.get_followers("cy") == ["ana"]
assert s2.get_following("dee") == []
"""},
        {"name": "Part 2: lists of every snapshot match a reference", "part": 2, "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "A following or follower list differed from a deep copy, was not sorted, or an unknown id did not raise ValueError.",
         "code": _HELPERS + r"""
for seed in range(60):
    rng = random.Random(seed)
    users = [f"u{i}" for i in range(7)]
    for snap, model in mirrored(rng, 60, users, unfollows=False):
        for u in users:
            if u in model.following:
                assert snap.get_following(u) == model.get_following(u), (seed, u)
                assert snap.get_followers(u) == model.get_followers(u), (seed, u)
            else:
                assert raises_value_error(snap.get_following, u)
                assert raises_value_error(snap.get_followers, u)
"""},
        {"name": "Part 2: lists cost the size of their answer", "part": 2, "visibility": "unshown", "behavior": "performance.complexity",
         "failure_message": "get_followers on a 20,000-user snapshot was much slower than on a 20-user one; keep a reverse index instead of scanning every user.",
         "code": _HELPERS + r"""
def ring(n):
    net = {fn}()
    for i in range(n):
        net.add_user(f"u{i}")
    for i in range(n):
        net.follow(f"u{i}", f"u{(i + 1) % n}")
        net.follow(f"u{i}", f"u{(i + 2) % n}")
    return net.create_snapshot()
small, big = ring(20), ring(20000)
assert big.get_followers("u5") == ["u3", "u4"]
def lists(snap):
    return lambda: [(snap.get_followers(f"u{i % 20}"), snap.get_following(f"u{i % 20}")) for i in range(1000)]
ratio = best_of_three(lists(big)) / best_of_three(lists(small))
assert ratio < 20, f"lists on the big snapshot took {ratio:.0f}x as long"
"""},
        {"name": "Part 3: the worked example", "part": 3, "behavior": "state.invariant", "code": r"""
net = {fn}()
for user in ("ana", "ben", "cy", "dee"):
    net.add_user(user)
for a, b in [("ana", "ben"), ("ana", "cy"), ("ben", "dee"), ("cy", "ana"), ("cy", "ben"), ("cy", "dee")]:
    net.follow(a, b)
s2 = net.create_snapshot()
assert s2.recommend("ana", 5) == ["dee"]
assert s2.recommend("ana", 0) == []
assert s2.recommend("dee", 3) == []
"""},
        {"name": "Part 3: ranking and ties", "part": 3, "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "recommend must rank by score descending, break ties by user id ascending, cut at k, and leave out the user and anyone already followed.",
         "code": _HELPERS + r"""
net = {fn}()
for user in ("me", "f1", "f2", "f3", "a", "b", "c", "d"):
    net.add_user(user)
for followee in ("f1", "f2", "f3"):
    net.follow("me", followee)
for f, cs in {"f1": ["d", "c", "me", "f2"], "f2": ["d", "b"], "f3": ["d", "c", "a"]}.items():
    for c in cs:
        net.follow(f, c)
snap = net.create_snapshot()
assert snap.recommend("me", 10) == ["d", "c", "a", "b"]
assert snap.recommend("me", 2) == ["d", "c"]
assert raises_value_error(snap.recommend, "ghost", 1)
"""},
        {"name": "Part 3: random graphs match a reference", "part": 3, "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "On a random graph, recommend differed from counting two-hop paths directly.",
         "code": _HELPERS + r"""
for seed in range(60):
    rng = random.Random(seed)
    users = [f"u{i}" for i in range(8)]
    for snap, model in mirrored(rng, 80, users, unfollows=False):
        for u in model.following:
            k = rng.randint(0, 6)
            assert snap.recommend(u, k) == model.recommend(u, k), (seed, u, k)
"""},
        {"name": "Part 4: unfollow and older snapshots", "part": 4, "behavior": "state.invariant", "code": _HELPERS + r"""
net = {fn}()
for user in ("ana", "ben", "cy"):
    net.add_user(user)
net.follow("ana", "ben")
s3 = net.create_snapshot()
net.unfollow("ana", "ben")
s4 = net.create_snapshot()
net.follow("ana", "ben")
s5 = net.create_snapshot()
assert [s.is_following("ana", "ben") for s in (s3, s4, s5)] == [True, False, True]
assert s4.get_followers("ben") == [] and s3.get_followers("ben") == ["ana"]
net.unfollow("ana", "cy")
net.unfollow("ana", "ana")
assert raises_value_error(net.unfollow, "ana", "zed")
assert net.create_snapshot().is_following("ana", "cy") is False
"""},
        {"name": "Part 4: random calls with unfollow match a reference", "part": 4, "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "With unfollow in the mix, some older snapshot answered differently from a deep copy taken at that moment.",
         "code": _HELPERS + r"""
for seed in range(80):
    rng = random.Random(seed)
    users = [f"u{i}" for i in range(6)]
    for snap, model in mirrored(rng, 80, users, unfollows=True):
        for u in model.following:
            assert snap.get_following(u) == model.get_following(u), (seed, u)
            assert snap.get_followers(u) == model.get_followers(u), (seed, u)
            assert snap.recommend(u, 3) == model.recommend(u, 3), (seed, u)
            for v in model.following:
                assert snap.is_following(u, v) == model.is_following(u, v), (seed, u, v)
"""},
        {"name": "Part 4: snapshots do not copy the graph", "part": 4, "visibility": "unshown", "behavior": "performance.complexity",
         "failure_message": "create_snapshot on a network with 40,000 follows was much slower than on one with ten; record when follows start and end instead of copying.",
         "code": _HELPERS + r"""
def network(n):
    net = {fn}()
    for i in range(n):
        net.add_user(f"u{i}")
    for i in range(n):
        net.follow(f"u{i}", f"u{(i + 1) % n}")
        net.follow(f"u{i}", f"u{(i + 7) % n}")
    return net
small, big = network(5), network(20000)
assert big.create_snapshot().get_following("u0") == ["u1", "u7"]
def snapshots(net):
    return lambda: [net.create_snapshot() for _ in range(300)]
ratio = best_of_three(snapshots(big)) / best_of_three(snapshots(small))
assert ratio < 20, f"snapshots of the big network took {ratio:.0f}x as long"
"""},
    ],
    "solution": r'''# Adapted from Schuture/OpenAI-Interview-Notes (code under the MIT License).
import bisect
from collections import Counter


class SocialNetwork:
    """Every change gets a version number; a snapshot is just the version it was taken at."""

    def __init__(self):
        self._version = 0
        self._joined = {}  # user -> version that added them
        self._events = {}  # (follower, followee) -> versions at which that follow started or ended
        self._out = {}     # user -> everyone they have ever followed
        self._in = {}      # user -> everyone who has ever followed them

    def add_user(self, user_id):
        if user_id in self._joined:
            raise ValueError(f"user already exists: {user_id!r}")
        self._version += 1
        self._joined[user_id] = self._version
        self._out[user_id] = set()
        self._in[user_id] = set()

    def follow(self, follower, followee):
        self._set(follower, followee, True)

    def unfollow(self, follower, followee):
        self._set(follower, followee, False)

    def create_snapshot(self):
        return Snapshot(self, self._version)

    def _set(self, follower, followee, following):
        for user in (follower, followee):
            if user not in self._joined:
                raise ValueError(f"unknown user: {user!r}")
        if follower == followee or self._following_at(follower, followee, self._version) == following:
            return
        self._version += 1
        self._events.setdefault((follower, followee), []).append(self._version)
        self._out[follower].add(followee)
        self._in[followee].add(follower)

    def _following_at(self, follower, followee, version):
        # The events alternate start, end, start, ...: an odd count up to version means following.
        events = self._events.get((follower, followee), ())
        return bisect.bisect_right(events, version) % 2 == 1


class Snapshot:
    def __init__(self, network, version):
        self._net = network
        self._version = version

    def _check(self, *users):
        for user in users:
            if self._net._joined.get(user, self._version + 1) > self._version:
                raise ValueError(f"unknown user: {user!r}")

    def _follows(self, follower, followee):
        return self._net._following_at(follower, followee, self._version)

    def is_following(self, follower, followee):
        self._check(follower, followee)
        return self._follows(follower, followee)

    def get_following(self, user_id):
        self._check(user_id)
        return sorted(v for v in self._net._out[user_id] if self._follows(user_id, v))

    def get_followers(self, user_id):
        self._check(user_id)
        return sorted(u for u in self._net._in[user_id] if self._follows(u, user_id))

    def recommend(self, user_id, k):
        direct = set(self.get_following(user_id))
        score = Counter()
        for friend in direct:
            for candidate in self.get_following(friend):
                if candidate != user_id and candidate not in direct:
                    score[candidate] += 1
        return sorted(score, key=lambda c: (-score[c], c))[:k]
''',
    "interview_questions": interview(
        concept=[
            "Which structure do you use for each user's followees, and what does it give follow and is_following?",
            "What must a snapshot hold so that later calls cannot change its answers, and why is copying only the outer dictionary not enough?",
        ],
        deep_dive=[
            "What happens to a snapshot that stored a reference to the live network's dictionary, and how would a test catch it?",
        ],
        tradeoffs=[
            "How do you make the follower list cost the size of its answer instead of the size of the graph?",
            "How do you count each candidate's score so that one followed user adds at most one point?",
            "How can a snapshot cost O(1) to take and still answer for its own moment after later follows and unfollows?",
        ],
    ),
}
