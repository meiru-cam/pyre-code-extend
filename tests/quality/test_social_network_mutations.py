"""Mutation gate for the multi-part social network exercise."""

from __future__ import annotations

import pytest

from torch_judge.tasks import get_task
from torch_judge.tasks._schema import validate_task
from tests.quality.parts_gate import (
    assert_first_fails_in_part,
    assert_part_mutations_rejected,
    first_failing_part,
)

TASK_ID = "social_network"

# The textbook answer to parts 1-3: every snapshot deep-copies the graph and builds a reverse index.
_COPYING = '''from collections import Counter


class SocialNetwork:
    def __init__(self):
        self._following = {}

    def add_user(self, user_id):
        if user_id in self._following:
            raise ValueError(user_id)
        self._following[user_id] = set()

    def _check(self, *users):
        for user in users:
            if user not in self._following:
                raise ValueError(user)

    def follow(self, follower, followee):
        self._check(follower, followee)
        if follower != followee:
            self._following[follower].add(followee)

    def unfollow(self, follower, followee):
        self._check(follower, followee)
        self._following[follower].discard(followee)

    def create_snapshot(self):
        return Snapshot(self._following)


class Snapshot:
    def __init__(self, following):
        self._following = {u: set(fs) for u, fs in following.items()}
        self._followers = {u: set() for u in following}
        for u, fs in self._following.items():
            for v in fs:
                self._followers[v].add(u)

    def _check(self, *users):
        for user in users:
            if user not in self._following:
                raise ValueError(user)

    def is_following(self, follower, followee):
        self._check(follower, followee)
        return followee in self._following[follower]

    def get_following(self, user_id):
        self._check(user_id)
        return sorted(self._following[user_id])

    def get_followers(self, user_id):
        self._check(user_id)
        return sorted(self._followers[user_id])

    def recommend(self, user_id, k):
        self._check(user_id)
        direct = self._following[user_id]
        score = Counter(c for f in direct for c in self._following[f] if c != user_id and c not in direct)
        return sorted(score, key=lambda c: (-score[c], c))[:k]
'''

_SCANNING = _COPYING.replace(
    "        return sorted(self._followers[user_id])\n",
    "        return sorted(u for u, fs in self._following.items() if user_id in fs)\n",
)

MUTATIONS = [
    ("snapshot reads the live graph", 1, [("return Snapshot(self, self._version)", "return Snapshot(self, float('inf'))")]),
    ("duplicate user accepted", 1, [('            raise ValueError(f"user already exists: {user_id!r}")\n', "            return\n")]),
    ("unknown users accepted", 1, [('                raise ValueError(f"unknown user: {user!r}")\n        if follower == followee',
                                     '                return\n        if follower == followee')]),
    ("self-follow recorded", 1, [("if follower == followee or self._following_at", "if self._following_at")]),
    ("repeated follow toggles", 1, [(" or self._following_at(follower, followee, self._version) == following:", ":")]),
    ("snapshot sees later users", 1, [("self._net._joined.get(user, self._version + 1) > self._version",
                                       "user not in self._net._joined")]),
    ("following sorted descending", 2, [("return sorted(v for v in self._net._out[user_id] if self._follows(user_id, v))",
                                         "return sorted((v for v in self._net._out[user_id] if self._follows(user_id, v)), reverse=True)")]),
    ("followers are the following", 2, [("return sorted(u for u in self._net._in[user_id] if self._follows(u, user_id))",
                                         "return sorted(v for v in self._net._out[user_id] if self._follows(user_id, v))")]),
    ("followers scan every user", 2, [("return sorted(u for u in self._net._in[user_id] if self._follows(u, user_id))",
                                       "return sorted(u for u in self._net._out if self._follows(u, user_id))")]),
    ("already-followed recommended", 3, [("if candidate != user_id and candidate not in direct:", "if candidate != user_id:")]),
    ("self recommended", 3, [("if candidate != user_id and candidate not in direct:", "if candidate not in direct:")]),
    ("ties broken descending", 3, [("key=lambda c: (-score[c], c))[:k]", "key=lambda c: (score[c], c), reverse=True)[:k]")]),
    ("k ignored", 3, [("key=lambda c: (-score[c], c))[:k]", "key=lambda c: (-score[c], c))")]),
    ("unfollow when not following starts one", 4, [("self._following_at(follower, followee, self._version) == following:",
                                                     "following and self._following_at(follower, followee, self._version):")]),
    ("unfollow skips the user check", 4, [(
        "    def unfollow(self, follower, followee):\n        self._set(follower, followee, False)\n",
        "    def unfollow(self, follower, followee):\n        if follower in self._joined and followee in self._joined:\n            self._set(follower, followee, False)\n",
    )]),
]


def test_mutations_rejected():
    assert_part_mutations_rejected(TASK_ID, MUTATIONS)


@pytest.mark.parametrize(("name", "part", "edits"), MUTATIONS, ids=[m[0] for m in MUTATIONS])
def test_each_mutation_first_fails_in_its_part(name, part, edits):
    assert_first_fails_in_part(TASK_ID, part, edits)


def test_a_copying_solution_passes_parts_1_to_3():
    """Part 4's O(1) snapshot is the only thing that rules out the deep copy."""
    assert first_failing_part(get_task(TASK_ID), _COPYING) == 4


def test_a_scanning_follower_list_fails_part_2():
    assert first_failing_part(get_task(TASK_ID), _SCANNING) == 2


def test_task_metadata_is_valid():
    task = get_task(TASK_ID)
    validate_task(TASK_ID, task)
    assert not {"title_zh", "description_zh", "hint_zh"} & set(task)
