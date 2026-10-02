"""An iterator over IPv4 addresses that grows a direction, a block, a step and batch reads."""

from ._interview import interview

# An independent oracle built on the standard library's ipaddress module.
_ORACLE = r"""
import ipaddress, itertools, random, time

MAX_ADDR = 2 ** 32 - 1

def bounds(s):
    ip_part, slash, prefix = s.partition("/")
    start = int(ipaddress.IPv4Address(ip_part))
    if not slash:
        return start, 0, MAX_ADDR
    net = ipaddress.IPv4Network(f"{ip_part}/{prefix}", strict=False)
    return start, int(net.network_address), int(net.broadcast_address)

def expected(s, reverse=False, step=1, count=10 ** 6):
    start, lo, hi = bounds(s)
    out, cur = [], start
    while len(out) < count and lo <= cur <= hi:
        out.append(str(ipaddress.IPv4Address(cur)))
        cur += -step if reverse else step
    return out

def addr(value):
    return str(ipaddress.IPv4Address(value))

def raises(make, error=ValueError):
    try:
        make()
    except error:
        return True
    return False

def drained_twice(it):
    for _ in range(3):
        assert raises(lambda: next(it), StopIteration), "an exhausted iterator must keep raising StopIteration"
"""

TASK = {
    "title": "IPv4 Address Iterator",
    "difficulty": "Easy",
    "version": 1,
    "function_name": "IPV4Iterator",
    "description_en": r"""Build `IPV4Iterator(address)`, an iterator that walks IPv4 addresses one at a time, starting from `address`.

The requirement arrives in parts. Each part keeps every earlier behavior, so one `IPV4Iterator` class passes all parts at the end. Pass every test of the current part to reveal the next one.

**Rules for every part:**
- An address is written `"a.b.c.d"`: exactly four segments separated by dots. Each segment is decimal digits `0`–`9` with a value from `0` to `255`, and has no leading zero unless it is exactly `0`. So `"0.0.0.0"` and `"10.0.0.255"` are valid; `"7.08.9.10"`, `"7.8.00.10"`, `"7.8.9.256"` and `" 1.2.3.4"` are not.
- The address `"a.b.c.d"` stands for the 32-bit number `a·2²⁴ + b·2¹⁶ + c·2⁸ + d`. Addresses are ordered by that number, from `0.0.0.0` up to `255.255.255.255`.
- Every invalid argument makes the constructor raise `ValueError`.
- Each `next()` does constant work. The iterator must not build the whole range up front: starting at `0.0.0.0` must be as quick as starting anywhere else.
- Once the iterator raises `StopIteration`, every later `next()` raises it too.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why this shows up in interviews:** turning dotted text into one integer makes the walking trivial, so the work is in strict parsing and in off-by-one boundaries. Each later part adds one requirement.

**Where it is used:** network scanners, firewall rule checkers and IP address managers all walk address ranges.

Adapted from the IPv4 iterator question in Schuture/OpenAI-Interview-Notes (CC BY-NC 4.0), reworded.""",
    "parts": [
        {
            "title": "Forward iteration",
            "description_en": r"""**Signature:** `IPV4Iterator(address)`, `iter(it) -> it`, `next(it) -> str`

- `address` is an address `"a.b.c.d"` as described above.
- `next()` returns addresses in increasing order, starting with `address` itself and ending with `255.255.255.255`. After that it raises `StopIteration`.

**Example**, from `"172.16.9.253"`:
- `172.16.9.253`, `172.16.9.254`, `172.16.9.255`, `172.16.10.0`: after `.255` the third segment goes up by one and the last starts again at `0`
- `"255.255.255.255"` is the highest address, so an iterator started there returns just it""",
        },
        {
            "title": "Reverse iteration",
            "description_en": r"""Keep Part 1 and add a direction.

**Signature:** `IPV4Iterator(address, reverse=False)`

- `reverse=False` keeps Part 1's behavior.
- `reverse=True` returns addresses in decreasing order, starting with `address` and ending with `0.0.0.0`, then raises `StopIteration`.

**Example**, from `"172.16.10.1"` with `reverse=True`:
- `172.16.10.1`, `172.16.10.0`, `172.16.9.255`, `172.16.9.254`: below `.0` the third segment goes down by one and the last restarts at `255`
- `"0.0.0.0"` is the lowest address, so a reverse iterator started there returns just it""",
        },
        {
            "title": "Confined to a block",
            "description_en": r"""Keep Parts 1–2. `address` may now also be `"a.b.c.d/prefix"`, a block of addresses.

- `prefix` is an integer from `0` to `32`, written with the same rules as a segment: digits only, no leading zero unless it is exactly `0`. Anything else raises `ValueError`.
- The block's first address is the number of `a.b.c.d` with its lowest `32 - prefix` bits set to 0. Its last address is the same number with those bits set to 1. The block is every address between them, both included.
- Iteration still starts at `a.b.c.d` itself, which may be anywhere in the block. Forward iteration stops after the block's last address, and reverse iteration after its first.
- Without `/prefix`, the range is still `0.0.0.0` to `255.255.255.255`.

**Example**, `"100.64.7.45/27"` keeps the top 27 bits, so its block runs from `100.64.7.32` to `100.64.7.63`:
- forward gives 19 addresses, `100.64.7.45` through `100.64.7.63`
- reverse gives 14 addresses, `100.64.7.45` down to `100.64.7.32`
- `"192.168.40.9/32"` is a block of one address; `"192.168.40.6/31"` gives `192.168.40.6` and `192.168.40.7`
- `"77.1.2.3/0"` is the whole address space: forward, it begins at `77.1.2.3` and ends at `255.255.255.255`""",
        },
        {
            "title": "Step size",
            "description_en": r"""Keep Parts 1–3 and add a step.

**Signature:** `IPV4Iterator(address, reverse=False, step=1)`

- `step` is a positive integer; `step <= 0` raises `ValueError`.
- The first address returned is still `address`. Each later `next()` moves `step` addresses in the chosen direction.
- Before returning an address, check that it is inside the range (the block, or the whole address space). The first one outside ends the iteration, whether or not some step hit the last address exactly.

**Example:**
- `"100.64.7.32/27"` with `step=5`: `.32`, `.37`, `.42`, `.47`, `.52`, `.57`, `.62`; then `.67` is past `.63`, so it ends
- `"100.64.7.61/27"` with `reverse=True, step=6`: `.61`, `.55`, `.49`, `.43`, `.37`; then `.31` is below `.32`, so it ends
- `IPV4Iterator("100.64.7.32/27", step=0)` raises `ValueError`""",
        },
        {
            "title": "Batch reads",
            "description_en": r"""Keep Parts 1–4 and add a batch read.

**Signature:** `next_batch(size) -> list[str]`

- `next_batch(size)` returns what `size` calls to `next()` would return, in the same order, and moves the iterator just as far.
- It never raises `StopIteration`: near the end it returns fewer than `size` addresses, and once nothing is left it returns `[]`.
- `size` is a non-negative integer; `size < 0` raises `ValueError`. `size == 0` returns `[]` and moves nothing, even on an exhausted iterator.

**Example**, `it = IPV4Iterator("100.64.7.32/28")`, a block of 16 addresses:
- `it.next_batch(6)` returns `100.64.7.32` through `100.64.7.37`
- `it.next_batch(50)` returns the other 10, the last being `100.64.7.47`
- `it.next_batch(2)` returns `[]`""",
        },
    ],
    "hints": [
        {"level": 1, "kind": "questions", "content": "If each address were a single number, what would \"the next address\" be, and would you still need to handle a segment passing 255 yourself? Which strings does int() accept that the rules forbid, such as \" 7\", \"+7\", \"007\" or other scripts' digits?"},
        {"level": 2, "kind": "analysis", "content": "Parse once: split on \".\", require exactly four parts, check each against a strict pattern such as 0 or a 1-9 followed by up to two digits, then range-check and combine with value = (value << 8) | segment. Keep one integer cursor and an upper bound. next() checks cursor > bound, formats the cursor with shifts 24, 16, 8 and 0, then adds 1."},
    ],
    "model_connections": [
        "Cluster schedulers and Kubernetes network plugins hand out pod and node IPs by walking CIDR blocks.",
        "Distributed training jobs build their rendezvous host lists from address ranges, where an off-by-one drops or duplicates a worker.",
    ],
    "pro_con_analysis": {
        "pros": [
            "Treating an address as one 32-bit integer turns carry and borrow into plain addition and subtraction.",
            "Masking the low bits gives a block's first and last address in O(1).",
            "A cursor and two bounds use O(1) memory, however large the range.",
        ],
        "cons": [
            "Building the whole range as strings up front would take hundreds of gigabytes for the full space.",
            "Lenient parsing such as int() on each segment accepts spaces, signs and other scripts' digits.",
            "A shared cursor is not thread-safe; concurrent callers need a lock or separate sub-ranges.",
        ],
    },
    "tests": [
        {"name": "Part 1: the worked example", "part": 1, "behavior": "state.invariant", "code": r"""
import itertools
it = {fn}("172.16.9.253")
assert iter(it) is it
assert [next(it) for _ in range(4)] == ["172.16.9.253", "172.16.9.254", "172.16.9.255", "172.16.10.0"]
assert list({fn}("255.255.255.255")) == ["255.255.255.255"]
assert list(itertools.islice({fn}("1.255.255.254"), 3)) == ["1.255.255.254", "1.255.255.255", "2.0.0.0"]
"""},
        {"name": "Part 1: strict parsing", "part": 1, "visibility": "unshown", "behavior": "protocol.validation",
         "failure_message": "The constructor must raise ValueError for anything but four dot-separated decimal segments from 0 to 255 without leading zeros, and accept every valid address.",
         "code": _ORACLE + r"""
BAD = ["", "1.2.3", "1.2.3.4.5", "1.2.3.256", "1.2.3.1000", "01.2.3.4", "1.2.3.04", "1.2.03.4", "1.2.00.4",
       "1.2.3.-4", "1.2.3.+4", "1.2.3.a", "1..3.4", ".1.2.3", "1.2.3.", " 1.2.3.4", "1.2.3.4 ", "1.2.3.4\n",
       "1.2.3.٣", "1.2.3.４", "1.2.3.4_0", "1.2.3.0x1", "256.0.0.0", "1,2,3,4"]
for s in BAD:
    assert raises(lambda: {fn}(s)), f"{s!r} should raise ValueError"
for s in ["0.0.0.0", "10.0.0.255", "255.255.255.255", "100.20.3.0", "9.99.199.249"]:
    assert next({fn}(s)) == s
"""},
        {"name": "Part 1: random starts and the last address", "part": 1, "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "From a random start the addresses must come in increasing numeric order, carrying across segments, and stop right after 255.255.255.255.",
         "code": _ORACLE + r"""
rng = random.Random(1)
for _ in range(500):
    seed = addr(rng.getrandbits(32))
    assert list(itertools.islice({fn}(seed), 40)) == expected(seed, count=40), seed
for gap in range(0, 300, 7):
    seed = addr(MAX_ADDR - gap)
    it = {fn}(seed)
    assert list(it) == expected(seed), seed
    drained_twice(it)
"""},
        {"name": "Part 1: nothing is built up front", "part": 1, "visibility": "unshown", "behavior": "performance.complexity",
         "failure_message": "Starting at 0.0.0.0 and reading a few addresses must be instant; keep an integer cursor instead of building the range.",
         "code": _ORACLE + r"""
start = time.perf_counter()
it = {fn}("0.0.0.0")
assert [next(it) for _ in range(3)] == ["0.0.0.0", "0.0.0.1", "0.0.0.2"]
elapsed = time.perf_counter() - start
assert elapsed < 2, f"took {elapsed:.2f}s"
start = time.perf_counter()
it = {fn}("10.0.0.0")
for _ in range(200000):
    next(it)
assert next(it) == "10.3.13.64"
elapsed = time.perf_counter() - start
assert elapsed < 8, f"200,000 calls took {elapsed:.2f}s"
"""},
        {"name": "Part 2: the worked example", "part": 2, "behavior": "state.invariant", "code": r"""
it = {fn}("172.16.10.1", reverse=True)
assert [next(it) for _ in range(4)] == ["172.16.10.1", "172.16.10.0", "172.16.9.255", "172.16.9.254"]
assert list({fn}("0.0.0.0", reverse=True)) == ["0.0.0.0"]
assert list({fn}("0.0.0.2", reverse=True)) == ["0.0.0.2", "0.0.0.1", "0.0.0.0"]
"""},
        {"name": "Part 2: both directions against an oracle", "part": 2, "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "In reverse the addresses must come in decreasing order, borrowing across segments, and stop right after 0.0.0.0; reverse=False must not change Part 1.",
         "code": _ORACLE + r"""
rng = random.Random(2)
for _ in range(400):
    seed = addr(rng.getrandbits(32))
    reverse = rng.random() < 0.5
    assert list(itertools.islice({fn}(seed, reverse=reverse), 30)) == expected(seed, reverse, count=30), (seed, reverse)
for gap in range(0, 300, 7):
    it = {fn}(addr(gap), reverse=True)
    assert list(it) == expected(addr(gap), True), gap
    drained_twice(it)
assert list({fn}("255.255.255.255", reverse=False)) == ["255.255.255.255"]
assert list(itertools.islice({fn}("3.0.0.0", reverse=True), 2)) == ["3.0.0.0", "2.255.255.255"]
"""},
        {"name": "Part 3: the worked example", "part": 3, "behavior": "state.invariant", "code": r"""
forward = list({fn}("100.64.7.45/27"))
assert forward[0] == "100.64.7.45" and forward[-1] == "100.64.7.63" and len(forward) == 19
backward = list({fn}("100.64.7.45/27", reverse=True))
assert backward[0] == "100.64.7.45" and backward[-1] == "100.64.7.32" and len(backward) == 14
assert list({fn}("192.168.40.9/32")) == ["192.168.40.9"]
assert list({fn}("192.168.40.6/31")) == ["192.168.40.6", "192.168.40.7"]
it = {fn}("77.1.2.3/0")
assert [next(it) for _ in range(3)] == ["77.1.2.3", "77.1.2.4", "77.1.2.5"]
"""},
        {"name": "Part 3: prefix parsing", "part": 3, "visibility": "unshown", "behavior": "protocol.validation",
         "failure_message": "The prefix must be 0 to 32 in plain decimal without a leading zero; anything else, including a netmask or a second slash, raises ValueError.",
         "code": _ORACLE + r"""
BAD = ["1.2.3.4/", "1.2.3.4/33", "1.2.3.4/100", "1.2.3.4/-1", "1.2.3.4/+8", "1.2.3.4/1/2", "1.2.3.4/8\n",
       "1.2.3.4/ 8", "1.2.3.4/08", "1.2.3.4/00", "1.2.3.4/255.0.0.0", "1.2.3.4/٣", "/8", "1.2.3/8", "01.2.3.4/8"]
for s in BAD:
    assert raises(lambda: {fn}(s)), f"{s!r} should raise ValueError"
for prefix in range(33):
    s = f"1.2.3.4/{prefix}"
    assert next({fn}(s)) == "1.2.3.4", s
"""},
        {"name": "Part 3: every prefix against an oracle", "part": 3, "visibility": "unshown", "behavior": "edge.empty_or_boundary",
         "failure_message": "Inside a block, iteration must start at the given address and stop right after the block's last address forward, or its first address in reverse, for every prefix from /0 to /32.",
         "code": _ORACLE + r"""
rng = random.Random(3)
for _ in range(600):
    suffix = rng.choice(["", f"/{rng.randint(0, 32)}", f"/{rng.randint(24, 32)}"])
    reverse = rng.random() < 0.5
    _, lo, hi = bounds(addr(rng.getrandbits(32)) + suffix)
    gap = rng.randint(0, min(40, hi - lo))
    seed = addr(lo + gap if reverse else hi - gap) + suffix
    it = {fn}(seed, reverse=reverse)
    assert list(it) == expected(seed, reverse), (seed, reverse)
    drained_twice(it)
    seed = addr(rng.getrandbits(32)) + suffix
    assert list(itertools.islice({fn}(seed, reverse=reverse), 5)) == expected(seed, reverse, count=5), (seed, reverse)
"""},
        {"name": "Part 4: the worked example", "part": 4, "behavior": "state.invariant", "code": r"""
assert list({fn}("100.64.7.32/27", step=5)) == [f"100.64.7.{n}" for n in (32, 37, 42, 47, 52, 57, 62)]
assert list({fn}("100.64.7.61/27", reverse=True, step=6)) == [f"100.64.7.{n}" for n in (61, 55, 49, 43, 37)]
try:
    {fn}("100.64.7.32/27", step=0)
    raise AssertionError("step=0 should raise ValueError")
except ValueError:
    pass
"""},
        {"name": "Part 4: steps that jump past the end", "part": 4, "visibility": "unshown", "behavior": "edge.empty_or_boundary",
         "failure_message": "A step must stop iteration as soon as the next address would leave the range, even when it jumps over the last address; step <= 0 raises ValueError.",
         "code": _ORACLE + r"""
for bad in (0, -1, -100):
    assert raises(lambda: {fn}("1.2.3.4", step=bad)), bad
rng = random.Random(4)
for _ in range(600):
    suffix = rng.choice(["", f"/{rng.randint(0, 32)}", f"/{rng.randint(22, 32)}"])
    reverse, step = rng.random() < 0.5, rng.choice([1, 2, 3, 7, 256, 1000])
    _, lo, hi = bounds(addr(rng.getrandbits(32)) + suffix)
    gap = rng.randint(0, min(30 * step, hi - lo))
    seed = addr(lo + gap if reverse else hi - gap) + suffix
    it = {fn}(seed, reverse=reverse, step=step)
    assert list(it) == expected(seed, reverse, step), (seed, reverse, step)
    drained_twice(it)
assert list({fn}("8.8.8.8", step=2 ** 40)) == ["8.8.8.8"]
assert list({fn}("8.8.8.8", reverse=True, step=2 ** 40)) == ["8.8.8.8"]
"""},
        {"name": "Part 5: the worked example", "part": 5, "behavior": "state.invariant", "code": r"""
it = {fn}("100.64.7.32/28")
assert it.next_batch(6) == [f"100.64.7.{n}" for n in range(32, 38)]
rest = it.next_batch(50)
assert len(rest) == 10 and rest[-1] == "100.64.7.47"
assert it.next_batch(2) == []
"""},
        {"name": "Part 5: batches mixed with next()", "part": 5, "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "next_batch must return exactly what the same number of next() calls would, never raise StopIteration, return [] once exhausted, treat size 0 as a no-op and reject a negative size.",
         "code": _ORACLE + r"""
assert {fn}("1.2.3.4").next_batch(0) == []
it = {fn}("1.2.3.4")
it.next_batch(0)
assert next(it) == "1.2.3.4", "size 0 must not move the iterator"
assert raises(lambda: {fn}("1.2.3.4").next_batch(-1))
rng = random.Random(5)
for _ in range(500):
    seed = f"{addr(rng.getrandbits(32))}/{rng.randint(25, 32)}"
    reverse, step = rng.random() < 0.5, rng.randint(1, 4)
    it, got = {fn}(seed, reverse=reverse, step=step), []
    while True:
        if rng.random() < 0.3:
            try:
                got.append(next(it))
            except StopIteration:
                break
        else:
            size = rng.randint(0, 10)
            batch = it.next_batch(size)
            assert isinstance(batch, list)
            got += batch
            if size and len(batch) < size:
                break
    assert got == expected(seed, reverse, step), (seed, reverse, step)
    assert it.next_batch(0) == [] and it.next_batch(3) == []
    assert raises(lambda: it.next_batch(-1))
    drained_twice(it)
"""},
    ],
    "solution": r'''# Adapted from Schuture/OpenAI-Interview-Notes (code under the MIT License).
import re

_SEGMENT = re.compile(r"\A(0|[1-9][0-9]{0,2})\Z")  # \Z, not $: "$" also matches before a final newline
_PREFIX = re.compile(r"\A(0|[1-9][0-9]?)\Z")
MAX_IP = (1 << 32) - 1


def _parse_ip(text):
    segments = text.split(".")
    if len(segments) != 4:
        raise ValueError(f"expected 4 dot-separated segments, got {text!r}")
    value = 0
    for segment in segments:
        if not _SEGMENT.match(segment) or int(segment) > 255:  # a pattern, not int(): int(" 7") is 7
            raise ValueError(f"bad segment {segment!r} in {text!r}")
        value = (value << 8) | int(segment)  # the leftmost segment is the most significant byte
    return value


def _parse(text):
    """Returns (start, first, last) for "a.b.c.d" or "a.b.c.d/prefix"."""
    ip_part, slash, prefix_part = text.partition("/")
    start = _parse_ip(ip_part)
    if not slash:
        return start, 0, MAX_IP
    if not _PREFIX.match(prefix_part) or int(prefix_part) > 32:
        raise ValueError(f"bad prefix {prefix_part!r}")
    host_bits = 32 - int(prefix_part)
    first = start >> host_bits << host_bits  # /0 shifts by 32, which a Python int handles fine
    return start, first, first + (1 << host_bits) - 1


def _format(value):
    return ".".join(str((value >> shift) & 0xFF) for shift in (24, 16, 8, 0))


class IPV4Iterator:
    def __init__(self, address, reverse=False, step=1):
        self._cursor, self._lo, self._hi = _parse(address)
        if step <= 0:
            raise ValueError(f"step must be positive, got {step}")
        self._delta = -step if reverse else step

    def __iter__(self):
        return self

    def __next__(self):
        if not self._lo <= self._cursor <= self._hi:  # a step may jump past an end without landing on it
            raise StopIteration
        result = _format(self._cursor)
        self._cursor += self._delta
        return result

    def next_batch(self, size):
        if size < 0:
            raise ValueError(f"size must be non-negative, got {size}")
        out = []
        for _ in range(size):
            try:
                out.append(next(self))
            except StopIteration:
                break
        return out
''',
    "interview_questions": interview(
        concept=[
            "Why turn an address into one 32-bit integer instead of incrementing the four segments?",
            "Which inputs does a lenient parser such as int() on each segment wrongly accept?",
        ],
        deep_dive=[
            "Why must the iterator not build the list of addresses up front, and how much memory would the full space take?",
        ],
        tradeoffs=[
            "How do you get a block's first and last address from a prefix, and what happens at /0 in a language with 32-bit integers?",
            "With a step larger than 1, why is checking for equality with the last address not enough to stop?",
            "How should a batch read behave near the end, and what does a size of 0 mean?",
            "What goes wrong if two threads share one iterator, and how would you fix it?",
        ],
    ),
}
