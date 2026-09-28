"""RTFM-style world streaming: pick context frames for the next frame from a spatial memory of posed frames."""

from ._interview import interview

TASK = {
    "title": "Posed Frame Context Retrieval",
    "difficulty": "Medium",
    "version": 1,
    "function_name": "SpatialFrameMemory",
    "description_en": r"""World Labs' RTFM generates video frames in real time as the user moves through a world. Every generated frame has a camera pose, and to render the next frame the model conditions on nearby frames retrieved from this spatial memory instead of on the whole history. Implement the memory and its context selection. The blog does not publish the exact rule; this exercise uses the one below.

**Signature:** `SpatialFrameMemory()` with `add(frame_id, position, yaw)` and `context(position, yaw, k, recent=2, max_angle=90.0, angle_weight=1.0) -> list`.

**`add`** stores a frame with its camera `position` (a tuple of 3 floats) and `yaw` in degrees. Raise `ValueError` if `frame_id` was already added.

**Angular difference** between yaws `a` and `b`: `abs((a - b + 180) % 360 - 180)`, in degrees from 0 to 180.

**`context(position, yaw, k, ...)`** — the frames to condition on for a camera at `position` facing `yaw`:
- Raise `ValueError` unless `k >= 1` and `0 <= recent <= k`.
- Always take the `recent` most recently added frames (all of them if fewer are stored).
- Fill the remaining slots, up to `k` frames in total, from the other frames whose angular difference to `yaw` is at most `max_angle`, best score first. Score = Euclidean distance between positions + `angle_weight * radians(angular difference)`; lower is better, and on equal scores the more recently added frame wins.
- Return the chosen ids in the order they were added.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Frames as the world model.** RTFM has no explicit 3D representation. An autoregressive diffusion transformer predicts the next frame from previous frames, so the frames themselves, with their poses, are the memory of the world.

**Why retrieve by pose.** Attending to every frame ever generated would grow cost without bound. Frames taken from nearby cameras facing a similar way show the same surfaces, so they carry what the next frame needs; World Labs calls assembling this custom context "context juggling".

**Why always keep recent frames.** They carry motion and lighting continuity, and they keep the stream smooth even when the camera moves into a region with no stored frames.

**Why this gives persistence.** When the user walks away and comes back, the old frames of that place are retrieved again, so the world looks the same instead of being re-imagined.""",
    "advisory_prerequisites": [],
    "hints": [
        {"level": 1, "kind": "questions", "content": "Which frames are included no matter where the camera is? Why filter by viewing direction before ranking by distance? How do you break ties so the result is deterministic?"},
        {"level": 2, "kind": "analysis", "content": "Keep a list of (id, position, yaw) in insertion order and a set of ids. For context, take the last `recent` indices, filter the rest by angular difference, sort by (score, -index), take what fits, then sort all chosen indices and map them to ids."},
    ],
    "model_connections": [
        "World Labs' RTFM retrieves nearby posed frames to form each frame's context and runs at interactive frame rates on a single H100.",
        "Retrieval-conditioned world models and video generators with memory banks, such as WorldMem, pick past frames by camera pose or field-of-view overlap in the same way.",
    ],
    "pro_con_analysis": {
        "pros": ["Bounded context per frame no matter how long the session runs, and revisited places stay consistent."],
        "cons": ["A pose heuristic can miss relevant frames, for example through occlusion or seen through a window, and the memory itself still grows with the session."],
    },
    "sources": [
        {"kind": "paper", "url": "https://www.worldlabs.ai/blog/rtfm", "section": "Posed frames and spatial memory (context juggling)"},
    ],
    "tests": [
        {"name": "Keeps recent frames and adds nearby ones facing the same way", "behavior": "routing.selection", "code": r"""
m = {fn}()
m.add("a", (0.0, 0.0, 0.0), 0.0)
m.add("b", (10.0, 0.0, 0.0), 0.0)
m.add("c", (1.0, 0.0, 0.0), 180.0)
m.add("d", (20.0, 0.0, 0.0), 90.0)
m.add("e", (30.0, 0.0, 0.0), 90.0)
assert m.context((0.5, 0.0, 0.0), 10.0, k=3) == ["a", "d", "e"]
assert m.context((9.0, 0.0, 0.0), 350.0, k=4, recent=1) == ["a", "b", "e"]
"""},
        {"name": "Angles wrap, ties prefer recent frames, and arguments are checked", "visibility": "unshown", "behavior": "routing.selection", "failure_message": "Wrap yaw differences across 0/360, rank by distance plus weighted angle with recency breaking ties, return insertion order, and validate ids and k.", "code": r"""
m = {fn}()
for i in range(4):
    m.add(i, (0.0, 0.0, 0.0), 0.0)
m.add("far", (5.0, 0.0, 0.0), 0.0)
m.add("x", (100.0, 0.0, 0.0), 0.0)
assert m.context((0.0, 0.0, 0.0), 359.0, k=3, recent=1) == [2, 3, "x"]
assert m.context((0.0, 0.0, 0.0), 0.0, k=2, recent=0) == [2, 3]
n = {fn}()
n.add("near_side", (0.0, 0.0, 1.0), 80.0)
n.add("far_front", (0.0, 0.0, 2.0), 0.0)
n.add("r", (50.0, 0.0, 0.0), 0.0)
assert n.context((0.0, 0.0, 0.0), 0.0, k=2, recent=1, angle_weight=1.0) == ["far_front", "r"]
assert n.context((0.0, 0.0, 0.0), 0.0, k=2, recent=1, angle_weight=0.0) == ["near_side", "r"]
assert n.context((0.0, 0.0, 0.0), 0.0, k=3, recent=1, max_angle=45.0) == ["far_front", "r"]
assert n.context((0.0, 0.0, 0.0), 0.0, k=3, recent=1, max_angle=80.0) == ["near_side", "far_front", "r"]
for bad in ({"k": 0}, {"k": 2, "recent": 3}, {"k": 2, "recent": -1}):
    try:
        n.context((0.0, 0.0, 0.0), 0.0, **bad)
    except ValueError:
        pass
    else:
        raise AssertionError(f"accepted {bad}")
try:
    n.add("r", (0.0, 0.0, 0.0), 0.0)
except ValueError:
    pass
else:
    raise AssertionError("duplicate id accepted")
assert {fn}().context((0.0, 0.0, 0.0), 0.0, k=3) == []
"""},
        {"name": "Seeded camera paths match an oracle", "visibility": "unshown", "behavior": "routing.selection", "failure_message": "The chosen frames must equal the recent frames plus the best-scoring direction-compatible others, returned in insertion order.", "code": r"""
import math, random
def oracle(frames, pos, yaw, k, recent, max_angle, w):
    n = len(frames)
    chosen = set(range(max(0, n - recent), n))
    def ad(a, b):
        return abs((a - b + 180) % 360 - 180)
    rest = [i for i in range(n) if i not in chosen and ad(frames[i][2], yaw) <= max_angle]
    rest.sort(key=lambda i: (math.dist(frames[i][1], pos) + w * math.radians(ad(frames[i][2], yaw)), -i))
    chosen |= set(rest[:max(0, k - len(chosen))])
    return [frames[i][0] for i in sorted(chosen)]
for seed in (1, 36, 70):
    rng = random.Random(seed)
    m, frames = {fn}(), []
    x = y = 0.0
    heading = 0.0
    for step in range(120):
        heading = (heading + rng.choice([0, 0, 15, -15, 90, -90, 180])) % 360
        x += math.cos(math.radians(heading)) * rng.choice([0.0, 1.0, 1.0, 2.0])
        y += math.sin(math.radians(heading)) * rng.choice([0.0, 1.0])
        pos = (round(x, 3), round(y, 3), 1.5)
        yaw = heading + rng.choice([0.0, 360.0, -360.0])
        if step % 3 == 0 or rng.random() < 0.3:
            k = rng.randint(1, 8)
            recent = rng.randint(0, k)
            max_angle = rng.choice([30.0, 60.0, 90.0, 180.0])
            w = rng.choice([0.0, 1.0, 5.0])
            got = m.context(pos, yaw, k, recent=recent, max_angle=max_angle, angle_weight=w)
            assert got == oracle(frames, pos, yaw, k, recent, max_angle, w), (seed, step, got)
        m.add(f"f{step}", pos, yaw)
        frames.append((f"f{step}", pos, yaw))
"""},
    ],
    "solution": '''import math


def _angle(a, b):
    return abs((a - b + 180) % 360 - 180)


class SpatialFrameMemory:
    def __init__(self):
        self.frames = []
        self.ids = set()

    def add(self, frame_id, position, yaw):
        if frame_id in self.ids:
            raise ValueError(f"frame {frame_id!r} already added")
        self.ids.add(frame_id)
        self.frames.append((frame_id, tuple(position), yaw))

    def context(self, position, yaw, k, recent=2, max_angle=90.0, angle_weight=1.0):
        if k < 1 or not 0 <= recent <= k:
            raise ValueError("need k >= 1 and 0 <= recent <= k")
        n = len(self.frames)
        chosen = set(range(max(0, n - recent), n))

        def score(i):
            _, pos, frame_yaw = self.frames[i]
            return math.dist(pos, position) + angle_weight * math.radians(_angle(frame_yaw, yaw))

        others = [i for i in range(n) if i not in chosen and _angle(self.frames[i][2], yaw) <= max_angle]
        others.sort(key=lambda i: (score(i), -i))
        chosen.update(others[:k - len(chosen)])
        return [self.frames[i][0] for i in sorted(chosen)]
''',
    "interview_questions": interview(
        concept=[
            "How can a video model act as a world model without any explicit 3D representation? What plays the role of memory?",
            "Why can't a real-time world model condition every new frame on all frames generated so far?",
        ],
        deep_dive=[
            "Walk through choosing context for a camera that turns around in place. Why filter by viewing direction before ranking by distance?",
            "Why always include the most recent frames, even when they are far from the query pose in some sense?",
            "How does pose-based retrieval make the world persistent when the user leaves an area and returns?",
        ],
        tradeoffs=[
            "Retrieval by pose heuristics versus learned retrieval over frame embeddings versus an explicit 3D representation (Gaussian splats): consistency, cost and generality?",
            "Spending the real-time frame budget on more context frames versus more diffusion steps per frame: quality, latency and memory?",
        ],
    ),
}
