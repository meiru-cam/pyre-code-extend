A social network holds a set of users and a directed *follow* relation between them: if `A` follows `B` that does not imply `B` follows `A`. Implement the following four parts.

### Part 1 — Users, follows, and snapshots

`SocialNetwork` holds the live, mutable state. A `Snapshot` freezes that state at one instant: no matter what happens to the `SocialNetwork` afterwards, a `Snapshot` that has already been returned keeps answering as of the moment it was created.

```py
class SocialNetwork:
    def add_user(self, user_id: str) -> None:
        """Registers a new user. Raises ValueError if user_id already exists."""

    def follow(self, follower: str, followee: str) -> None:
        """follower starts following followee. Raises ValueError if follower or followee is
        not a registered user. follower == followee is a no-op, and so is a call for a pair
        that already follows."""

    def create_snapshot(self) -> "Snapshot":
        """Freezes the current users and follow edges into a Snapshot. Any add_user() or
        follow() call made on this SocialNetwork afterwards must not change a Snapshot that
        has already been returned."""


class Snapshot:
    def is_following(self, follower: str, followee: str) -> bool:
        """Raises ValueError if follower or followee was not a user of this snapshot."""
```

Example: register `alice`, `bob`, `carol`, `dave`; call `follow("alice", "bob")` and `follow("alice", "carol")`; take a snapshot `s1`. Afterwards, `follow("bob", "dave")` is called on the live network. `s1.is_following("alice", "carol")` is `True`, but `s1.is_following("bob", "dave")` is `False` — that edge did not exist yet when `s1` was created. A snapshot taken after the second call would return `True` for the same query. (A repeated `follow("alice", "bob")`, or a `follow("alice", "alice")` call, at any point would not change any of this.)

### Part 2 — Following and follower lists

Extend `Snapshot` with the two lists below, each returned sorted by user id, ascending.

```py
class Snapshot:
    def get_following(self, user_id: str) -> list[str]:
        """Users user_id follows, as of this snapshot, sorted by user id ascending.
        Raises ValueError if user_id is unknown to this snapshot."""

    def get_followers(self, user_id: str) -> list[str]:
        """Users who follow user_id, as of this snapshot, sorted by user id ascending.
        Raises ValueError if user_id is unknown to this snapshot."""
```

Example: continuing Part 1, suppose that by the time a second snapshot `s2` is taken, `bob` additionally follows `erin`, and `carol` follows `dave`, `frank`, `bob`, and `alice`. `s2.get_following("carol")` is `["alice", "bob", "dave", "frank"]`, and `s2.get_followers("dave")` is `["bob", "carol"]`.

### Part 3 — Two-hop recommendations

A *candidate* for `user_id` is any user reached by following one of `user_id`'s own followees, excluding `user_id` itself and anyone `user_id` already follows directly. A candidate's *score* is the number of distinct followees of `user_id` through which it is reached.

```py
class Snapshot:
    def recommend(self, user_id: str, k: int) -> list[str]:
        """Top-k two-hop recommendations for user_id, ranked by score descending, ties
        broken by user id ascending. Returns fewer than k entries if fewer candidates
        exist. Raises ValueError if user_id is unknown to this snapshot."""
```

Example: using `s2` from Part 2 (`alice` follows `bob`, `carol`; `bob` follows `dave`, `erin`; `carol` follows `dave`, `frank`, `bob`, `alice`), `s2.recommend("alice", 2)` is `["dave", "erin"]` — `dave` is reached through both `bob` and `carol` (score 2), `erin` only through `bob` (score 1, ahead of `frank`'s score 1 by user id). `bob` is reached through `carol` too, but is excluded because `alice` already follows him. `s2.recommend("alice", 5)` is `["dave", "erin", "frank"]`, since only three candidates exist.

### Part 4 — Point-in-time queries

Instead of only answering queries at the instants where `create_snapshot()` was explicitly called, extend the network so any past instant can be queried directly. Every `follow`/`unfollow` call now carries the time `t` at which it happens; calls to `follow()`/`unfollow()` on the same `FollowTimeline` arrive with non-decreasing `t`, though `is_following()` may be called with any `t`, not only the latest one. This class needs no `add_user`: a user id exists the moment it first appears in a call, and `is_following` is `False` for any pair that has never had a recorded event.

```py
class FollowTimeline:
    def follow(self, follower: str, followee: str, t: int) -> None:
        """follower starts following followee at time t. follower == followee is a no-op,
        and so is a call that would repeat the pair's current state."""

    def unfollow(self, follower: str, followee: str, t: int) -> None:
        """follower stops following followee at time t. Same no-op rules as follow()."""

    def is_following(self, follower: str, followee: str, t: int) -> bool:
        """Whether follower was following followee at time t. If follow() and unfollow()
        were both called with the same t for this pair, the call made later decides the
        state at that t."""
```

Example: `follow("x", "y", 10)`, then `unfollow("x", "y", 20)`, then `follow("x", "y", 30)`. `is_following("x", "y", t)` is `False` for `t < 10`, `True` for `10 <= t < 20`, `False` for `20 <= t < 30`, and `True` for `t >= 30`.
