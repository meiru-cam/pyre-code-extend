"""A first-fit memory allocator, first with a linear scan, then in logarithmic time."""

from ._interview import interview

# A slow model written straight from the statement: one flag per address, scanned on every call.
_HELPERS = r"""
import random, time

class ByteMap:
    def __init__(self, capacity):
        self.used = [False] * capacity
        self.blocks = {}
    def allocate(self, size):
        run = 0
        for address, used in enumerate(self.used):
            run = 0 if used else run + 1
            if run == size:
                start = address - size + 1
                self.used[start:address + 1] = [True] * size
                self.blocks[start] = size
                return start
        return "MemoryError"
    def free(self, address, size):
        if self.blocks.get(address) != size:
            return "ValueError"
        del self.blocks[address]
        self.used[address:address + size] = [False] * size
    def free_blocks(self):
        out, start = [], None
        for address, used in enumerate(self.used + [True]):
            if not used and start is None:
                start = address
            elif used and start is not None:
                out.append((start, address - start))
                start = None
        return out

def outcome(fn, *args):
    try:
        return fn(*args)
    except (ValueError, MemoryError) as error:
        return type(error).__name__

def blocks(allocator):
    return [tuple(block) for block in allocator.free_blocks()]

def fragmented(make, n):
    # n free single bytes at even addresses, then one free block of 10 bytes at the end.
    allocator = make(2 * n + 10)
    for _ in range(2 * n):
        allocator.allocate(1)
    for address in range(0, 2 * n, 2):
        allocator.free(address, 1)
    return allocator

def best_of_three(fn):
    best = float("inf")
    for _ in range(3):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best
"""

TASK = {
    "title": "First-Fit Memory Allocator",
    "difficulty": "Hard",
    "version": 1,
    "function_name": "MemoryAllocator",
    "description_en": r"""Build `MemoryAllocator`, which hands out and takes back ranges of a fixed address space using first fit.

The requirement arrives in parts. Each part keeps every earlier behavior, so one `MemoryAllocator` class passes all parts at the end. Pass every test of the current part to reveal the next one.

**Rules for every part:**
- The address space is the integers `0` to `capacity - 1`, and at any moment every one of them is in use or available.
- A block is a run of addresses that are all allocated or all free and cannot be extended. Two free blocks never touch: freed space always joins the free blocks next to it.
- Errors are the built-in `ValueError` and `MemoryError`. A call that raises changes nothing.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why this shows up in interviews:** the first version is a careful list walk, and each later part adds one requirement.

**Where it is used:** heap allocators such as glibc malloc, GPU memory pools, and any system that carves a large buffer into pieces.

Adapted from the memory allocator question in Schuture/OpenAI-Interview-Notes (CC BY-NC 4.0), reworded, on one class, with a `free_blocks` method added so tests can see the free space.""",
    "parts": [
        {
            "title": "First fit",
            "description_en": r"""**Signature:** `MemoryAllocator(capacity)`, `allocate(size) -> int`, `free(address, size) -> None`, `free_blocks() -> list[tuple[int, int]]`

- `MemoryAllocator(capacity)` starts with everything free and raises `ValueError` if `capacity <= 0`.
- `allocate(size)` picks the free block with the lowest start address among those holding at least `size` bytes, takes `size` bytes from its low end, and returns their start address. What is left of that block stays free.
- A non-positive `size` is a `ValueError`; a request that no free block can hold is a `MemoryError`.
- `free(address, size)` releases a block that `allocate(size)` returned and that is still allocated. Anything else raises `ValueError`: an address never returned, a second free, a wrong size, or `size <= 0`.
- A released block joins a free block that touches it on the left, on the right, or both.
- `free_blocks()` returns every free block as `(start, size)`, in address order.

**Example:** with `capacity = 20`, five calls to `allocate(4)` return `0, 4, 8, 12, 16`. Then:
- `free(4, 4)`, `free(12, 4)` leave `[(4, 4), (12, 4)]`
- `free(8, 4)` joins both sides: `[(4, 12)]`
- `allocate(3)` returns `4`, the lowest block that fits""",
        },
        {
            "title": "Logarithmic time",
            "description_en": r"""Keep Part 1 and make it fast. `n` is the number of free blocks.

- `allocate` and `free` run in `O(log n)`, with the same results and the same errors as before. Placement stays first fit: the lowest address that fits, not the tightest fit.
- `capacity` can be up to `10**9`, so nothing may cost time per byte.
- With 20,000 scattered free bytes below it, allocating and freeing a block takes about as long as with 200.

**Example:** with `capacity = 40`, `allocate(10), allocate(4), allocate(8), allocate(18)` return `0, 10, 14, 22`. After `free(0, 10)` and `free(14, 8)`, `allocate(7)` returns `0`, though `(14, 8)` fits more tightly.""",
        },
    ],
    "hints": [
        {"level": 1, "kind": "questions", "content": "What do you store for each free block, and in what order? Which blocks can a released block possibly touch, and how do you find them? What extra record lets free tell a real allocation from a made-up one?"},
        {"level": 2, "kind": "analysis", "content": "Keep the free blocks as a list of [start, size] sorted by start, and a dict from allocated address to size. allocate walks the list and takes the first block that is big enough; free checks the dict, finds the first block starting after the address, and merges with it and with the one before it when they touch."},
    ],
    "model_connections": [
        "glibc malloc and similar heap allocators keep free chunks in bins and merge neighbours on free to fight fragmentation.",
        "Deep learning frameworks such as PyTorch run caching allocators that carve large GPU buffers into blocks and coalesce them when they are released.",
    ],
    "pro_con_analysis": {
        "pros": [
            "First fit is simple and tends to keep large free blocks at high addresses.",
            "Merging on free keeps the number of free blocks low, so free space stays usable.",
            "A balanced tree ordered by address, augmented with each subtree's largest block, answers first fit and neighbour lookups in O(log n).",
        ],
        "cons": [
            "First fit leaves small fragments near the low addresses that every later search must step over.",
            "An augmented balanced tree is much more code than a list, and Python's constant factors make it slower for small n.",
            "Recording each allocation's size costs memory per block; real allocators store it in a header next to the block.",
        ],
    },
    "tests": [
        {"name": "Part 1: the worked example", "part": 1, "behavior": "state.invariant", "code": _HELPERS + r"""
allocator = {fn}(20)
assert [allocator.allocate(4) for _ in range(5)] == [0, 4, 8, 12, 16]
assert blocks(allocator) == []
allocator.free(4, 4)
allocator.free(12, 4)
assert blocks(allocator) == [(4, 4), (12, 4)]
allocator.free(8, 4)
assert blocks(allocator) == [(4, 12)]
assert allocator.allocate(3) == 4
assert blocks(allocator) == [(7, 9)]
"""},
        {"name": "Part 1: the four merge cases", "part": 1, "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "A released block must join the free block on its left, on its right, both, or neither, leaving no two free blocks touching.",
         "code": _HELPERS + r"""
allocator = {fn}(25)
assert [allocator.allocate(5) for _ in range(5)] == [0, 5, 10, 15, 20]
allocator.free(5, 5)
assert blocks(allocator) == [(5, 5)]
allocator.free(10, 5)
assert blocks(allocator) == [(5, 10)]
allocator.free(20, 5)
assert blocks(allocator) == [(5, 10), (20, 5)]
allocator.free(15, 5)
assert blocks(allocator) == [(5, 20)]
allocator.free(0, 5)
assert blocks(allocator) == [(0, 25)]
"""},
        {"name": "Part 1: errors", "part": 1, "visibility": "unshown", "behavior": "protocol.validation",
         "failure_message": "Bad capacities and sizes raise ValueError, a request nothing fits raises MemoryError, and a free that does not match a live allocation raises ValueError and changes nothing.",
         "code": _HELPERS + r"""
assert outcome({fn}, 0) == "ValueError" and outcome({fn}, -4) == "ValueError"
allocator = {fn}(10)
assert outcome(allocator.allocate, 0) == "ValueError"
assert outcome(allocator.allocate, -1) == "ValueError"
assert outcome(allocator.allocate, 11) == "MemoryError"
assert allocator.allocate(10) == 0
assert outcome(allocator.allocate, 1) == "MemoryError"
assert outcome(allocator.free, 0, 4) == "ValueError", "wrong size"
assert outcome(allocator.free, 3, 7) == "ValueError", "never returned"
assert outcome(allocator.free, 0, 0) == "ValueError"
assert blocks(allocator) == []
allocator.free(0, 10)
assert outcome(allocator.free, 0, 10) == "ValueError", "second free"
assert blocks(allocator) == [(0, 10)]
"""},
        {"name": "Part 1: random calls match a byte map", "part": 1, "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "Under random allocate and free calls, an address, an error or the free blocks differed from tracking every byte.",
         "code": _HELPERS + r"""
for seed in range(150):
    rng = random.Random(seed)
    capacity = rng.randint(1, 40)
    allocator, model = {fn}(capacity), ByteMap(capacity)
    live = []
    for _ in range(40):
        if live and rng.random() < 0.45:
            address, size = rng.choice(live) if rng.random() < 0.8 else (rng.randrange(capacity), rng.randint(1, 5))
            expected = model.free(address, size)
            assert outcome(allocator.free, address, size) == expected, (seed, address, size)
            if expected is None:
                live.remove((address, size))
        else:
            size = rng.randint(1, 8)
            expected = model.allocate(size)
            assert outcome(allocator.allocate, size) == expected, (seed, size)
            if expected != "MemoryError":
                live.append((expected, size))
        assert blocks(allocator) == model.free_blocks(), seed
"""},
        {"name": "Part 2: first fit, not best fit", "part": 2, "behavior": "state.invariant", "code": _HELPERS + r"""
allocator = {fn}(40)
assert (allocator.allocate(10), allocator.allocate(4), allocator.allocate(8), allocator.allocate(18)) == (0, 10, 14, 22)
allocator.free(0, 10)
allocator.free(14, 8)
assert blocks(allocator) == [(0, 10), (14, 8)]
assert allocator.allocate(7) == 0
# Kept to 10**7 so a per-byte design fails on time instead of exhausting the grader's memory.
big = {fn}(10**7)
assert big.allocate(10**7 - 1) == 0
assert blocks(big) == [(10**7 - 1, 1)]
"""},
        {"name": "Part 2: allocate and free do not scan the free blocks", "part": 2, "visibility": "unshown", "behavior": "performance.complexity",
         "failure_message": "Allocating and freeing past 20,000 small free blocks was much slower than past 200; find the first fit and the neighbours with a tree search, not a walk.",
         "code": _HELPERS + r"""
def churn(allocator, n):
    def run():
        for _ in range(300):
            address = allocator.allocate(5)
            assert address == 2 * n
            allocator.free(address, 5)
    return run
small, big = fragmented({fn}, 200), fragmented({fn}, 20000)
ratio = best_of_three(churn(big, 20000)) / best_of_three(churn(small, 200))
assert ratio < 15, f"allocate past 20,000 free blocks took {ratio:.0f}x as long as past 200"
assert len(blocks(big)) == 20001
"""},
        {"name": "Part 2: long random runs match a byte map", "part": 2, "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "Over a long random run, an address, an error or the free blocks differed from tracking every byte.",
         "code": _HELPERS + r"""
for seed in range(20):
    rng = random.Random(1000 + seed)
    capacity = rng.randint(100, 400)
    allocator, model = {fn}(capacity), ByteMap(capacity)
    live = []
    for step in range(400):
        if live and rng.random() < 0.5:
            address, size = live.pop(rng.randrange(len(live)))
            assert outcome(allocator.free, address, size) == model.free(address, size)
        else:
            size = rng.randint(1, 20)
            expected = model.allocate(size)
            assert outcome(allocator.allocate, size) == expected, (seed, step, size)
            if expected != "MemoryError":
                live.append((expected, size))
        if step % 20 == 0:
            assert blocks(allocator) == model.free_blocks(), (seed, step)
    assert blocks(allocator) == model.free_blocks(), seed
"""},
    ],
    "solution": r'''# Adapted from Schuture/OpenAI-Interview-Notes (code under the MIT License).
import random


class _Node:
    """A treap node: one free block, keyed by start address, ordered as a heap by priority."""

    __slots__ = ("start", "size", "priority", "left", "right", "max_size")

    def __init__(self, start, size):
        self.start, self.size, self.max_size = start, size, size
        self.priority = random.random()
        self.left = self.right = None


def _pull(node):
    node.max_size = max(node.size, node.left.max_size if node.left else 0,
                        node.right.max_size if node.right else 0)


def _merge(a, b):
    # Every start in a is below every start in b.
    if a is None or b is None:
        return a or b
    if a.priority > b.priority:
        a.right = _merge(a.right, b)
        _pull(a)
        return a
    b.left = _merge(a, b.left)
    _pull(b)
    return b


def _split(node, key):
    # Returns (blocks starting at or below key, blocks starting above it).
    if node is None:
        return None, None
    if node.start <= key:
        low, high = _split(node.right, key)
        node.right = low
        _pull(node)
        return node, high
    low, high = _split(node.left, key)
    node.left = high
    _pull(node)
    return low, node


def _insert(root, start, size):
    low, high = _split(root, start)
    return _merge(_merge(low, _Node(start, size)), high)


def _delete(root, start):
    low, high = _split(root, start)
    low, _ = _split(low, start - 1)
    return _merge(low, high)


def _first_fit(node, size):
    # The lowest-address block holding at least size bytes; each step discards a subtree.
    while node is not None and node.max_size >= size:
        if node.left is not None and node.left.max_size >= size:
            node = node.left
        elif node.size >= size:
            return node
        else:
            node = node.right
    return None


def _neighbours(node, address):
    before = after = None
    while node is not None:
        if node.start < address:
            before, node = node, node.right
        else:
            after, node = node, node.left
    return before, after


class MemoryAllocator:
    def __init__(self, capacity):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._free = _Node(0, capacity)
        self._allocated = {}  # address -> size of every live allocation

    def allocate(self, size):
        if size <= 0:
            raise ValueError("size must be positive")
        node = _first_fit(self._free, size)
        if node is None:
            raise MemoryError(f"no free block holds {size} bytes")
        start = node.start
        self._free = _delete(self._free, start)
        if node.size > size:
            self._free = _insert(self._free, start + size, node.size - size)
        self._allocated[start] = size
        return start

    def free(self, address, size):
        if size <= 0 or self._allocated.get(address) != size:
            raise ValueError(f"no live allocation of {size} bytes at {address}")
        del self._allocated[address]
        before, after = _neighbours(self._free, address)
        start, total = address, size
        if before is not None and before.start + before.size == address:
            start, total = before.start, before.size + total
            self._free = _delete(self._free, before.start)
        if after is not None and address + size == after.start:
            total += after.size
            self._free = _delete(self._free, after.start)
        self._free = _insert(self._free, start, total)

    def free_blocks(self):
        out, stack, node = [], [], self._free
        while stack or node is not None:
            while node is not None:
                stack.append(node)
                node = node.left
            node = stack.pop()
            out.append((node.start, node.size))
            node = node.right
        return out
''',
    "interview_questions": interview(
        concept=[
            "How do you represent the free space so that first fit and merging are both straightforward?",
            "How does free decide whether a call matches a live allocation?",
        ],
        deep_dive=[
            "When a block is released, which free blocks can it touch, and how many merge cases are there?",
        ],
        tradeoffs=[
            "Why does a tree ordered by size answer best fit but not first fit?",
            "How does storing each subtree's largest block let one tree answer first fit in O(log n)?",
            "Why is a sorted Python list with bisect still O(n) per insertion, and when is that acceptable anyway?",
        ],
    ),
}
