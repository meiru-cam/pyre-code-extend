"""Run named step pipelines over numpy images: exact steps, one load per image, a thread pool, then shared prefixes computed once per image."""

from ._interview import interview

# An independent model of every step, written with plain loops on small images.
_MODEL = r"""
import random, threading, time, weakref
import numpy as np

def m_gray(a):
    if a.ndim == 2:
        return a.copy()
    h, w, _ = a.shape
    out = np.zeros((h, w), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            r, g, b = (int(x) for x in a[i, j])
            out[i, j] = (299 * r + 587 * g + 114 * b + 500) // 1000
    return out

def m_scale(a, f):
    h, w = a.shape[:2]
    nh, nw = max(1, round(h * f)), max(1, round(w * f))
    out = np.zeros((nh, nw) + a.shape[2:], dtype=np.uint8)
    for i in range(nh):
        for j in range(nw):
            out[i, j] = a[(2 * i + 1) * h // (2 * nh), (2 * j + 1) * w // (2 * nw)]
    return out

def m_blur(a, r):
    h, w = a.shape[:2]
    out = np.zeros_like(a)
    for i in range(h):
        for j in range(w):
            win = a[max(0, i - r):i + r + 1, max(0, j - r):j + r + 1].astype(np.int64)
            n = win.shape[0] * win.shape[1]
            out[i, j] = (win.sum(axis=(0, 1)) + n // 2) // n
    return out

def m_rot_once(a):  # 90 degrees counter-clockwise
    h, w = a.shape[:2]
    out = np.zeros((w, h) + a.shape[2:], dtype=np.uint8)
    for i in range(w):
        for j in range(h):
            out[i, j] = a[j, w - 1 - i]
    return out

def m_step(a, step):
    t = step["type"]
    if t == "grayscale":
        return m_gray(a)
    if t == "flip_horizontal":
        return np.array([row[::-1] for row in a], dtype=np.uint8)
    if t == "flip_vertical":
        return np.array(list(a)[::-1], dtype=np.uint8)
    if t == "scale":
        return m_scale(a, step["factor"])
    if t == "blur":
        return m_blur(a, step["radius"])
    if t == "rotate":
        for _ in range((step["angle"] // 90) % 4):
            a = m_rot_once(a)
        return a.copy()
    raise AssertionError(t)

def m_run(a, steps):
    for s in steps:
        a = m_step(a, s)
    return a

def rand_image(rng, gray=False):
    h, w = rng.randint(1, 6), rng.randint(1, 6)
    shape = (h, w) if gray else (h, w, 3)
    return np.array([rng.randrange(256) for _ in range(int(np.prod(shape)))], dtype=np.uint8).reshape(shape)

def rand_step(rng):
    t = rng.choice(["grayscale", "flip_horizontal", "flip_vertical", "scale", "blur", "rotate"])
    if t == "scale":
        return {"type": t, "factor": rng.choice([0.25, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0])}
    if t == "blur":
        return {"type": t, "radius": rng.randint(0, 3)}
    if t == "rotate":
        return {"type": t, "angle": rng.choice([-270, -180, -90, 0, 90, 180, 270, 360, 450])}
    return {"type": t}

class Store:
    # load and save callbacks over a dict of images; save keeps a copy, so it holds no reference to the result
    def __init__(self, images, load_delay=0.0):
        self.images, self.load_delay = images, load_delay
        self.saved, self.loads = {}, []
        self.lock = threading.Lock()
        self.active = self.peak = 0
    def load(self, name):
        with self.lock:
            self.loads.append(name)
            self.active += 1
            self.peak = max(self.peak, self.active)
        time.sleep(self.load_delay)
        with self.lock:
            self.active -= 1
        img = self.images[name].copy()
        img.setflags(write=False)  # a step that writes to its input fails loudly
        return img
    def save(self, name, pipeline, image):
        assert isinstance(image, np.ndarray) and image.dtype == np.uint8, (name, pipeline, type(image), getattr(image, "dtype", None))
        with self.lock:
            assert (name, pipeline) not in self.saved, ("saved twice", name, pipeline)
            self.saved[(name, pipeline)] = image.copy()

def check_saved(store, names, pipelines, label):
    want = {(n, p): m_run(store.images[n], steps) for n in names for p, steps in pipelines.items()}
    assert set(store.saved) == set(want), (label, sorted(set(want) - set(store.saved))[:5], sorted(set(store.saved) - set(want))[:5])
    for key, w in want.items():
        got = store.saved[key]
        assert got.shape == w.shape and np.array_equal(got, w), (label, key, got.shape, w.shape)

def rand_case(rng, n_images, n_pipes):
    images = {f"img{i}": rand_image(rng, gray=rng.random() < 0.2) for i in range(n_images)}
    pipes = {f"p{k}": [rand_step(rng) for _ in range(rng.randint(0, 4))] for k in range(n_pipes)}
    return images, pipes

def run_bounded(target, seconds=6):
    # run target on a daemon thread so a deadlock fails the test instead of hanging it
    box = {}
    def body():
        try:
            target()
        except BaseException as e:
            box["error"] = e
    t = threading.Thread(target=body, daemon=True)
    t.start()
    t.join(seconds)
    assert not t.is_alive(), f"did not return within {seconds} seconds"
    return box.get("error")

EXAMPLE = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
                    [[10, 10, 10], [200, 100, 0], [0, 0, 0]]], dtype=np.uint8)
EXAMPLE_PIPES = {"mono": [{"type": "flip_horizontal"}, {"type": "grayscale"}],
                 "big": [{"type": "scale", "factor": 2.0}, {"type": "rotate", "angle": 270}]}
"""

_SPY = r"""
def spy_of(base):
    # counts every apply_step call and how many of its results are still alive when the next call starts
    class Spy(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.calls, self.refs, self.max_alive = 0, [], 0
            self.spy_lock = threading.Lock()
        def apply_step(self, image, step):
            with self.spy_lock:
                self.calls += 1
                self.refs = [r for r in self.refs if r() is not None]
                self.max_alive = max(self.max_alive, len(self.refs))
            out = super().apply_step(image, step)
            with self.spy_lock:
                self.refs.append(weakref.ref(out))
            return out
    return Spy

def prefixes(pipelines):
    seen = set()
    for steps in pipelines.values():
        key = ()
        for s in steps:
            key += (tuple(sorted(s.items())),)
            seen.add(key)
    return len(seen)
"""

TESTS = [
    {"name": "Part 1: the worked example", "part": 1, "behavior": "state.invariant", "code": _MODEL + r"""
store = Store({"photo": EXAMPLE})
{fn}(store.load, store.save).run(["photo"], EXAMPLE_PIPES)
assert store.loads == ["photo"], store.loads
assert store.saved["photo", "mono"].tolist() == [[29, 150, 76], [0, 119, 10]], store.saved["photo", "mono"].tolist()
big = store.saved["photo", "big"]
assert big.shape == (6, 4, 3), big.shape
assert big[0, 0].tolist() == [10, 10, 10], big[0, 0].tolist()
"""},
    {"name": "Part 1: every step matches its definition", "part": 1, "visibility": "unshown", "behavior": "state.invariant",
     "failure_message": "On random small RGB and grayscale images, a step returned the wrong shape or pixels, was not uint8, wrote to or shared memory with its input, or an unknown type did not raise ValueError.",
     "code": _MODEL + r"""
p = {fn}(None, None)
for seed in range(300):
    rng = random.Random(seed)
    img = rand_image(rng, gray=seed % 4 == 0)
    img.setflags(write=False)
    step = rand_step(rng)
    got = p.apply_step(img, step)
    want = m_step(img, step)
    assert isinstance(got, np.ndarray) and got.dtype == np.uint8, (seed, step, type(got), getattr(got, "dtype", None))
    assert got.shape == want.shape and np.array_equal(got, want), (seed, step, img.shape, got.shape, want.shape)
    assert not np.shares_memory(got, img), (seed, step, "the result shares memory with the input")
try:
    p.apply_step(EXAMPLE, {"type": "sepia"})
except Exception as e:
    assert type(e).__name__ == "ValueError", type(e).__name__
else:
    raise AssertionError("an unknown step type must raise ValueError")
"""},
    {"name": "Part 1: random pipelines, one load per image", "part": 1, "visibility": "unshown", "behavior": "effects.idempotency",
     "failure_message": "Running random pipelines (some empty) over several images saved a wrong or missing result, saved a pair twice, or loaded an image more than once or out of order.",
     "code": _MODEL + r"""
for seed in range(40):
    rng = random.Random(1000 + seed)
    images, pipes = rand_case(rng, rng.randint(1, 3), rng.randint(1, 4))
    store = Store(images)
    names = list(images)
    {fn}(store.load, store.save).run(names, pipes)
    assert store.loads == names, (seed, store.loads)
    check_saved(store, names, pipes, seed)
"""},
    {"name": "Part 2: the worked example", "part": 2, "behavior": "scheduler.concurrency", "code": _MODEL + r"""
images = {f"photo{i}": EXAMPLE for i in range(4)}
store = Store(images, load_delay=0.2)
assert run_bounded(lambda: {fn}(store.load, store.save).run_parallel(list(images), EXAMPLE_PIPES, max_workers=2)) is None
assert sorted(store.loads) == sorted(images), store.loads
assert store.peak == 2, f"expected 2 images loading at once, saw {store.peak}"
check_saved(store, list(images), EXAMPLE_PIPES, "example")
"""},
    {"name": "Part 2: worker bounds, random pipelines and errors", "part": 2, "visibility": "unshown", "behavior": "scheduler.concurrency",
     "failure_message": "run_parallel ran more images at once than max_workers, ran fewer than 2 at once when it could, loaded an image twice, saved a wrong result, or swallowed an error from load.",
     "code": _MODEL + r"""
for workers, n, lo, hi in ((1, 3, 1, 1), (3, 8, 2, 3), (None, 6, 2, 6)):
    rng = random.Random(workers or 0)
    images, pipes = rand_case(rng, n, 3)
    store = Store(images, load_delay=0.15)
    assert run_bounded(lambda: {fn}(store.load, store.save).run_parallel(list(images), pipes, max_workers=workers)) is None
    assert lo <= store.peak <= hi, (workers, store.peak)
    assert sorted(store.loads) == sorted(images), (workers, store.loads)
    check_saved(store, list(images), pipes, workers)
store = Store({"a": EXAMPLE})
err = run_bounded(lambda: {fn}(store.load, store.save).run_parallel(["a", "missing"], EXAMPLE_PIPES, max_workers=2))
assert err is not None and type(err).__name__ == "KeyError", f"load raised KeyError for 'missing', run_parallel raised {type(err).__name__ if err else None}"
"""},
    {"name": "Part 3: the worked example", "part": 3, "behavior": "performance.complexity", "code": _MODEL + _SPY + r"""
pipes = {
    "a": [{"type": "flip_vertical"}, {"type": "blur", "radius": 1}, {"type": "grayscale"}],
    "b": [{"type": "flip_vertical"}, {"type": "blur", "radius": 1}],
    "c": [{"type": "flip_vertical"}, {"radius": 1, "type": "blur"}, {"type": "rotate", "angle": 180}],
    "d": [],
    "e": [{"type": "grayscale"}],
}
store = Store({"photo": EXAMPLE})
p = spy_of({fn})(store.load, store.save)
assert run_bounded(lambda: p.run_shared(["photo"], pipes)) is None
assert p.calls == 5, f"expected 5 step applications, saw {p.calls}"
assert store.loads == ["photo"], store.loads
check_saved(store, ["photo"], pipes, "example")
"""},
    {"name": "Part 3: shared prefixes on random pipelines", "part": 3, "visibility": "unshown", "behavior": "performance.complexity",
     "failure_message": "On random pipelines with repeated prefixes, run_shared made a step application count other than the number of distinct prefixes times the number of images, saved a wrong result, or loaded an image twice.",
     "code": _MODEL + _SPY + r"""
for seed in range(40):
    rng = random.Random(2000 + seed)
    pool = [rand_step(rng) for _ in range(3)]
    images = {f"img{i}": rand_image(rng, gray=rng.random() < 0.2) for i in range(rng.randint(1, 3))}
    pipes = {f"p{k}": [dict(rng.choice(pool)) for _ in range(rng.randint(0, 4))] for k in range(rng.randint(1, 6))}
    store = Store(images)
    p = spy_of({fn})(store.load, store.save)
    assert run_bounded(lambda: p.run_shared(list(images), pipes, max_workers=2)) is None
    assert p.calls == prefixes(pipes) * len(images), (seed, p.calls, prefixes(pipes), len(images))
    assert sorted(store.loads) == sorted(images), (seed, store.loads)
    check_saved(store, list(images), pipes, seed)
"""},
    {"name": "Part 3: parallel, and few intermediates alive", "part": 3, "visibility": "unshown", "behavior": "budget.enforcement",
     "failure_message": "run_shared ran fewer than 2 images at once with max_workers=3, or, with max_workers=1, kept more step results alive at once than the longest pipeline has steps.",
     "code": _MODEL + _SPY + r"""
images = {f"img{i}": EXAMPLE for i in range(6)}
store = Store(images, load_delay=0.15)
p = spy_of({fn})(store.load, store.save)
assert run_bounded(lambda: p.run_shared(list(images), EXAMPLE_PIPES, max_workers=3)) is None
assert 2 <= store.peak <= 3, store.peak
check_saved(store, list(images), EXAMPLE_PIPES, "parallel")
root = {"type": "flip_horizontal"}
pipes = {f"b{k}": [root, {"type": "blur", "radius": k}, {"type": "grayscale"}] for k in range(5)}
images = {f"img{i}": EXAMPLE for i in range(2)}
store = Store(images)
p = spy_of({fn})(store.load, store.save)
assert run_bounded(lambda: p.run_shared(list(images), pipes, max_workers=1)) is None
assert p.calls == 11 * 2, p.calls
assert p.max_alive <= 3, f"{p.max_alive} step results alive at once; the longest pipeline has 3 steps"
check_saved(store, list(images), pipes, "tree")
"""},
]

TASK = {
    "title": "Image Processing Pipeline",
    "difficulty": "Medium",
    "version": 1,
    "function_name": "ImagePipeline",
    "description_en": r"""Write `ImagePipeline`, which runs named pipelines of image steps over a batch of images and saves every result. Images are numpy arrays, and the steps are defined exactly below, so every result has one right answer.

The requirement arrives in parts. Each part keeps every earlier behavior, so one `ImagePipeline` passes all parts at the end. Pass every test of the current part to reveal the next one.

**Rules for every part:**
- `ImagePipeline(load, save)`: `load(image_name)` returns a `uint8` array of shape `(H, W, 3)` (RGB) or `(H, W)` (gray). `save(image_name, pipeline_name, image)` stores one result.
- A step is a dict with a `"type"` and that type's parameters. `apply_step(image, step)` returns a new `uint8` array that shares no memory with `image`, and never writes to `image`. An unknown type raises `ValueError`.
- `grayscale`: each pixel becomes `(299*R + 587*G + 114*B + 500) // 1000`, giving a 2-D array. A 2-D image comes back unchanged.
- `flip_horizontal` mirrors left to right; `flip_vertical` mirrors top to bottom.
- `scale` with `"factor"` `f > 0`: the new height is `max(1, round(H * f))` with Python's `round`, and likewise the width. Output pixel `(i, j)` copies source pixel `((2*i + 1) * H // (2 * newH), (2*j + 1) * W // (2 * newW))`.
- `blur` with `"radius"` `r >= 0`: each value becomes the mean of the window of rows `i-r..i+r` and columns `j-r..j+r`, clipped to the image, per channel. With `n` pixels in the clipped window, the mean is `(total + n // 2) // n`. `r = 0` changes nothing.
- `rotate` with `"angle"`, a multiple of `90`: rotates counter-clockwise by that many degrees, so a negative angle turns clockwise.
- `apply_step` never calls itself, so each applied step is exactly one `apply_step` call.
- `pipelines` is a dict from pipeline name to a list of steps, applied in order. An empty list saves the loaded image unchanged.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why this shows up in interviews:** the round starts as plain looping over images and pipelines, then asks where the time goes. Each later part adds one requirement: a worker pool that loads each image once, then reuse of the work two pipelines share.

**Where it is used:** data loaders for vision models apply chains of augmentations to every image, and preprocessing jobs fan out over many images while sharing decoded inputs and common prefixes of work.

Adapted from the image processing pipeline in Schuture/Anthropic-Interview-Notes (CC BY-NC 4.0), reworded. Pillow images become numpy arrays with exact step definitions: bilinear scaling becomes nearest-neighbour, the Gaussian blur becomes a box blur, and rotation is limited to multiples of 90 degrees. Folders and output paths become the `load` and `save` callbacks, `process_images` and its two variants become `run`, `run_parallel` and `run_shared` on one class, images come in the order given rather than directory order, worker processes become threads, and the rule that only file paths may cross a process boundary is left out.""",
    "parts": [
        {
            "title": "Steps and a sequential run",
            "description_en": r"""**Signature:** `ImagePipeline(load, save)` with `apply_step(image, step)` and `run(image_names, pipelines) -> None`

- `run` handles the images in the given order. It loads each image once, runs every pipeline on it in dict order, and calls `save(image_name, pipeline_name, result)` for each.
- Each pipeline starts from the loaded image; one pipeline never sees another's work.

**Example:** `photo` is a `(2, 3, 3)` image with rows `[red, green, blue]` and `[(10,10,10), (200,100,0), black]`, where red is `(255,0,0)`.
- pipeline `mono` = `[flip_horizontal, grayscale]` saves `[[29, 150, 76], [0, 119, 10]]`
- pipeline `big` = `[scale 2.0, rotate 270]` saves an array of shape `(6, 4, 3)` whose pixel `[0, 0]` is `(10, 10, 10)`
- `load("photo")` is called exactly once""",
        },
        {
            "title": "Images in parallel",
            "description_en": r"""Keep Part 1. Add `run_parallel(image_names, pipelines, max_workers=None) -> None`.

- It makes the same `save` calls as `run`, in any order, with images handled on up to `max_workers` threads at once. `None` means the thread pool's default size.
- Each image is still loaded exactly once, and `load` may be slow, so several images must load at the same time.
- It returns only after every save is done. An exception raised by `load`, `save` or a step is raised again from `run_parallel`.

**Example:** four copies of `photo` under the names `photo0` to `photo3`, the Part 1 pipelines, and a `load` that takes `0.2` seconds:
- `run_parallel(names, pipelines, max_workers=2)` makes the same 8 saves as `run`
- exactly 2 images are loading at the busiest moment, and each name is loaded once""",
        },
        {
            "title": "Shared prefixes",
            "description_en": r"""Keep Parts 1–2. Add `run_shared(image_names, pipelines, max_workers=None) -> None`.

- It makes the same saves as `run_parallel`, under the same rules, but for each image every distinct prefix of steps is computed only once. Two steps are the same when their types and parameters are equal; the order of keys in a step dict does not matter.
- Every step applied goes through `self.apply_step`, one call per application. Work is never shared between images.
- While an image is processed, at most as many `apply_step` results are kept alive as the longest pipeline has steps. `save` keeps its own copy, so a saved result still counts while your code holds a reference to it.

**Example:** one image `photo` and five pipelines:
- `a` = `[flip_vertical, blur 1, grayscale]`, `b` = `[flip_vertical, blur 1]`, `c` = `[flip_vertical, blur 1, rotate 180]` with the blur written as `{"radius": 1, "type": "blur"}`
- `d` = `[]` saves the image as loaded; `e` = `[grayscale]`
- running each pipeline alone costs 9 step applications; `run_shared` makes 5: `flip_vertical`, then `blur 1` on it, then `grayscale` and `rotate 180` on that, plus `grayscale` on the original
- `load("photo")` is called once""",
        },
    ],
    "hints": [
        {"level": 1, "kind": "questions", "content": "Which steps can be written with array slicing alone, and which need arithmetic on a wider integer type before going back to uint8? For scale, which source row does output row 0 copy when an image of height 3 doubles to height 6?"},
        {"level": 2, "kind": "analysis", "content": "Flips and rotations are views like a[:, ::-1] and np.rot90, so copy them to get a new array. Grayscale and blur need int64 sums so 255 * 299 cannot overflow, then a cast to uint8. Scale builds index arrays with the given formula and indexes rows, then columns. Load once per image outside the pipeline loop, and reassign the image after each step instead of writing into it."},
    ],
    "model_connections": [
        "Vision data loaders apply chains of crops, flips, resizes and colour changes to every training image, often on a pool of workers.",
        "Sharing the common prefix of augmentation or preprocessing chains is the same idea as reusing a KV cache for prompts that share a prefix.",
    ],
    "pro_con_analysis": {
        "pros": [
            "Exact integer definitions make every result reproducible and testable bit for bit.",
            "Loading each image once and running its pipelines together removes repeated decode work.",
            "A prefix tree per image computes each shared step once, and a depth-first walk keeps only one branch of results in memory.",
        ],
        "cons": [
            "Threads only speed up numpy work that releases the GIL; pure Python steps need processes, which cost copying the arrays.",
            "Handling an image's pipelines together keeps its source and branch results in memory until the image is done.",
            "A prefix tree only pays off when pipelines really share leading steps; otherwise it is extra bookkeeping.",
        ],
    },
    "tests": TESTS,
    "solution": r'''# Adapted from Schuture/Anthropic-Interview-Notes (code under the MIT License).
from concurrent.futures import ThreadPoolExecutor

import numpy as np


class ImagePipeline:
    def __init__(self, load, save):
        self.load = load  # image_name -> uint8 array
        self.save = save  # (image_name, pipeline_name, image) -> None

    # -- part 1: steps and a sequential run
    def apply_step(self, image, step):
        """A new array; image itself is never written to."""
        kind = step["type"]
        if kind == "grayscale":
            if image.ndim == 2:
                return image.copy()
            rgb = image.astype(np.int64)
            return ((299 * rgb[..., 0] + 587 * rgb[..., 1] + 114 * rgb[..., 2] + 500) // 1000).astype(np.uint8)
        if kind == "flip_horizontal":
            return image[:, ::-1].copy()
        if kind == "flip_vertical":
            return image[::-1].copy()
        if kind == "scale":
            h, w = image.shape[:2]
            nh, nw = max(1, round(h * step["factor"])), max(1, round(w * step["factor"]))
            rows = (2 * np.arange(nh) + 1) * h // (2 * nh)  # the source pixel whose centre is nearest
            cols = (2 * np.arange(nw) + 1) * w // (2 * nw)
            return image[rows][:, cols]
        if kind == "blur":
            return self._box_blur(image, step["radius"])
        if kind == "rotate":
            return np.rot90(image, (step["angle"] // 90) % 4).copy()  # counter-clockwise
        raise ValueError(f"unknown step type {kind!r}")

    @staticmethod
    def _box_blur(image, r):
        if r == 0:
            return image.copy()
        h, w = image.shape[:2]
        # summed-area table with a zero row and column in front
        table = np.zeros((h + 1, w + 1) + image.shape[2:], dtype=np.int64)
        table[1:, 1:] = image.astype(np.int64).cumsum(0).cumsum(1)
        top, bottom = np.clip(np.arange(h) - r, 0, h), np.clip(np.arange(h) + r + 1, 0, h)
        left, right = np.clip(np.arange(w) - r, 0, w), np.clip(np.arange(w) + r + 1, 0, w)
        total = (table[bottom][:, right] - table[top][:, right]
                 - table[bottom][:, left] + table[top][:, left])
        count = np.outer(bottom - top, right - left)
        if image.ndim == 3:
            count = count[..., None]
        return ((total + count // 2) // count).astype(np.uint8)

    def run_pipeline(self, image, steps):
        for step in steps:
            image = self.apply_step(image, step)
        return image

    def run(self, image_names, pipelines):
        for name in image_names:
            source = self.load(name)  # once per image, shared by every pipeline
            for pipeline_name, steps in pipelines.items():
                self.save(name, pipeline_name, self.run_pipeline(source, steps))

    # -- part 2: images in parallel, each loaded once
    def run_parallel(self, image_names, pipelines, max_workers=None):
        self._fan_out(image_names, pipelines, max_workers, self._one_image)

    def _one_image(self, name, pipelines):
        source = self.load(name)
        for pipeline_name, steps in pipelines.items():
            self.save(name, pipeline_name, self.run_pipeline(source, steps))

    def _fan_out(self, image_names, pipelines, max_workers, work):
        with ThreadPoolExecutor(max_workers=max_workers) as pool:  # None means os.cpu_count()
            for future in [pool.submit(work, name, pipelines) for name in image_names]:
                future.result()  # re-raise a worker's error here

    # -- part 3: each shared prefix computed once per image
    def run_shared(self, image_names, pipelines, max_workers=None):
        self._fan_out(image_names, pipelines, max_workers, self._one_image_shared)

    def _one_image_shared(self, name, pipelines):
        root = {}  # step key -> [step, children, pipeline names ending here]
        ends_at_root = []
        for pipeline_name, steps in pipelines.items():
            children, node = root, None
            for step in steps:
                key = tuple(sorted(step.items()))  # the key order inside a step does not matter
                node = children.setdefault(key, [step, {}, []])
                children = node[1]
            (node[2] if node else ends_at_root).append(pipeline_name)
        source = self.load(name)
        for pipeline_name in ends_at_root:
            self.save(name, pipeline_name, source)
        self._walk(name, root, source)

    def _walk(self, name, children, image):
        """Depth first, so only the images along the current branch are alive."""
        for step, grandchildren, ending in children.values():
            result = self.apply_step(image, step)
            for pipeline_name in ending:
                self.save(name, pipeline_name, result)
            self._walk(name, grandchildren, result)
''',
    "interview_questions": interview(
        concept=[
            "Why must apply_step return a new array instead of changing its input, and what goes wrong when two pipelines start from the same loaded image?",
            "Why does grayscale or blur need a wider integer type before casting back to uint8?",
        ],
        deep_dive=[
            "How does the scale index formula pick the source pixel whose centre is nearest, and what happens when the factor rounds a dimension down to zero?",
        ],
        tradeoffs=[
            "When do threads speed up this work, and when would you need processes instead? What then crosses the process boundary?",
            "How do you make sure each image is loaded once while several pipelines and workers use it?",
            "How do you find the prefixes that pipelines share, and why does the key order inside a step not matter?",
            "Why does a depth-first walk of the prefix tree keep fewer results alive than computing it level by level?",
        ],
    ),
}
