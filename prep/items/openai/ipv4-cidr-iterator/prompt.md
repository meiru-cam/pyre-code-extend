An IPv4 address is written as four decimal segments separated by dots, `"a.b.c.d"`. Each segment is between `0` and `255`, and is written without a leading zero unless the segment is exactly `0` — so `"1.2.3.4"` and `"0.0.0.0"` are valid, `"1.2.03.4"` and `"1.2.00.4"` are not. Such an address corresponds to the 32-bit integer $a \cdot 2^{24} + b \cdot 2^{16} + c \cdot 2^{8} + d$, and addresses are ordered, and iterated, by this integer: `0.0.0.0` is the smallest possible address and `255.255.255.255` is the largest. Implement the following five parts as one class, `IPV4Iterator`, where each part extends the constructor and the iteration behaviour of the one before it. If the constructor's argument `ip_or_cidr` does not split into exactly four such segments, or a segment is not written in the form above, or a segment's value exceeds `255`, the constructor raises `ValueError`. Every invalid argument in all five parts raises `ValueError`, so an input that breaks several rules at once may be rejected by whichever check runs first.

### Part 1 — Forward iteration

`ip_or_cidr` is a bare address (no `/`, that is Part 3's addition). Iterating produces addresses in increasing numeric order, one per call, starting from (and including) `ip_or_cidr`, up to and including `255.255.255.255`; after that, further calls raise `StopIteration`. If `ip_or_cidr` is already `"255.255.255.255"`, exactly that one address is produced.

```py
class IPV4Iterator:
    def __init__(self, ip_or_cidr: str) -> None:
        """ip_or_cidr must be a bare address "a.b.c.d". Raises ValueError if it is not a valid
        address."""

    def __iter__(self) -> "IPV4Iterator": ...

    def __next__(self) -> str:
        """Returns the next address, starting from (and including) ip_or_cidr, in increasing
        numeric order, up to and including 255.255.255.255. Raises StopIteration afterwards."""
```

Example, starting from `192.0.2.254`:

```text
call 1: 192.0.2.254
call 2: 192.0.2.255
call 3: 192.0.3.0    # the last segment overflowed 255, carrying into the segment before it
call 4: 192.0.3.1
```

### Part 2 — Reverse iteration

Add `reverse: bool = False`. With `reverse=True`, iterating produces addresses in decreasing numeric order, starting from (and including) `ip_or_cidr`, down to and including `0.0.0.0`; after that, further calls raise `StopIteration`. If `ip_or_cidr` is already `"0.0.0.0"`, exactly that one address is produced.

```py
class IPV4Iterator:
    def __init__(self, ip_or_cidr: str, reverse: bool = False) -> None:
        """reverse=False keeps Part 1's behaviour. reverse=True walks backward from (and
        including) ip_or_cidr, in decreasing numeric order, down to and including 0.0.0.0."""
```

Example, starting from `192.0.2.1` with `reverse=True`:

```text
call 1: 192.0.2.1
call 2: 192.0.2.0
call 3: 192.0.1.255   # borrowed from the segment before it
call 4: 192.0.1.254
```

### Part 3 — Confined to a CIDR block

`ip_or_cidr` may now be `"a.b.c.d/prefix"`, a *CIDR block*: `prefix` is an integer from `0` to `32`, written with the same no-leading-zero rule as a segment; if it is not written in this form, or its value is outside `0`-`32`, the constructor raises `ValueError`. Given the address's 32-bit integer and `prefix`, the block's *network address* is that integer with its low `32 - prefix` bits cleared, and its *broadcast address* is the network address with those same low bits set; the block is every address from the network address to the broadcast address, inclusive of both. The network address need not be `ip_or_cidr` itself — since it is computed by masking `ip_or_cidr`, the given address always lies inside the resulting block. Iteration is confined to the block: forward stops after the broadcast address (inclusive), reverse stops after the network address (inclusive), and iteration still starts at `ip_or_cidr` itself, which need not be either end. A `/32` block holds exactly one address (`ip_or_cidr` itself, which is then both ends). A `/0` block is the entire address space, `0.0.0.0` through `255.255.255.255`, but iteration still starts at `ip_or_cidr`.

Example, block `203.0.113.96`–`203.0.113.111` (`203.0.113.96/28`), starting from `203.0.113.100`:

```text
forward: 203.0.113.100, 203.0.113.101, ..., 203.0.113.111   (12 addresses, ends at the broadcast address)
reverse: 203.0.113.100, 203.0.113.99, ..., 203.0.113.96      (5 addresses, ends at the network address)
```

Special prefixes: `IPV4Iterator("198.51.100.77/32")` produces only `"198.51.100.77"`. `IPV4Iterator("198.51.100.16/31")` produces `"198.51.100.16"` then `"198.51.100.17"`. `IPV4Iterator("10.20.30.40/0")` starts at `"10.20.30.40"` and, forward, does not stop until `"255.255.255.255"`.

### Part 4 — Step size

Add `step: int = 1`. `step` must be a positive integer; `step <= 0` raises `ValueError`. Each call to `__next__` moves the current position `step` addresses in the direction `reverse` selects; the first address produced is still `ip_or_cidr` itself, unaffected by `step`. Iteration stops as soon as the *next* address to produce would fall outside the allowed range (`0.0.0.0`–`255.255.255.255`, or the CIDR block from Part 3) — a `step` greater than `1` can jump past that boundary without ever landing on it exactly, and no address beyond it is ever produced.

```py
class IPV4Iterator:
    def __init__(self, ip_or_cidr: str, reverse: bool = False, step: int = 1) -> None:
        """step is the number of addresses __next__ advances by on each call, in the direction
        reverse selects. Raises ValueError if step <= 0."""
```

Example, block `203.0.113.16/28` (network `203.0.113.16`, broadcast `203.0.113.31`), `step=3`:

```text
203.0.113.16, 203.0.113.19, 203.0.113.22, 203.0.113.25, 203.0.113.28, 203.0.113.31
# the next candidate would be 203.0.113.34, past the broadcast address -> StopIteration
```

Same block, starting from `203.0.113.30` with `reverse=True, step=4`:

```text
203.0.113.30, 203.0.113.26, 203.0.113.22, 203.0.113.18
# the next candidate would be 203.0.113.14, below the network address 203.0.113.16 -> StopIteration
```

`IPV4Iterator("203.0.113.16/28", step=0)` raises `ValueError`.

### Part 5 — Batch reads

Add `next_batch(size: int) -> list[str]`. It returns up to `size` addresses starting at the current position, equivalent to `size` consecutive `__next__` calls collected into a list, except that it never raises `StopIteration` itself: once the iterator would be exhausted it simply returns fewer than `size` addresses, and an empty list once no addresses remain at all. `size` must be a non-negative integer; `size < 0` raises `ValueError`. `size == 0` always returns `[]` without raising and without moving the current position, regardless of whether the iterator is exhausted.

```py
class IPV4Iterator:
    def next_batch(self, size: int) -> list[str]:
        """Returns up to size addresses from the current position, in the same order __next__
        would produce them. Returns fewer than size once fewer remain, [] once none remain.
        Raises ValueError if size < 0."""
```

Example, block `203.0.113.16/28` (16 addresses, `203.0.113.16`–`203.0.113.31`):

```text
it = IPV4Iterator("203.0.113.16/28")
it.next_batch(5)    -> ["203.0.113.16", "203.0.113.17", "203.0.113.18", "203.0.113.19", "203.0.113.20"]
it.next_batch(100)  -> the remaining 11 addresses, ending with "203.0.113.31"
it.next_batch(5)    -> []
```
