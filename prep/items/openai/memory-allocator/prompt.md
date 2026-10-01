An address space of `capacity` bytes is the integer range `[0, capacity)`; every address in it is, at any moment, either allocated or free. A *block* is a maximal contiguous range of addresses that is entirely allocated or entirely free. Because `free` always merges a newly released block with a touching free block, two free blocks are never adjacent to each other. Implement the following two parts in Python, using only the standard library. Also write your own tests for both classes, and be ready to explain how you would design them.

### Part 1 — First fit, linear scan

```py
class MemoryAllocator:
    def __init__(self, capacity: int) -> None: ...
    def allocate(self, size: int) -> int: ...
    def free(self, address: int, size: int) -> None: ...
```

`__init__(capacity)` starts with the whole address space free. Raises `ValueError` if `capacity <= 0`.

`allocate(size)` creates a new allocated block of `size` bytes, placed by *first fit*: among the free blocks of size at least `size`, it uses the one with the lowest start address, taking `size` bytes from its low end — if that free block is larger than `size`, the remaining bytes stay free, now starting at `address + size` — and returns the new block's start address. Raises `ValueError` if `size <= 0`, and `MemoryError` if no free block of size at least `size` exists.

`free(address, size)` releases the block that starts at `address`; `size` must equal the value passed to the `allocate` call that returned `address`. The released bytes become free and merge with a touching free block on the left, on the right, or both. Raises `ValueError` if `size <= 0`, or if `(address, size)` does not match a block that is currently allocated — one condition that covers an address `allocate` never returned, a second free of the same address, and a `size` that does not match the recorded allocation.

Example, `capacity = 25`:

```text
allocate(5) -> 0, allocate(5) -> 5, allocate(5) -> 10, allocate(5) -> 15, allocate(5) -> 20   (fully allocated)

free(5, 5)    # both neighbours allocated                            -> free: [5, 10)
free(10, 5)   # [5, 10) touches at address 10                        -> free: [5, 15)
free(20, 5)   # no neighbour on either side is free                  -> free: [5, 15), [20, 25)
free(15, 5)   # touches [5, 15) on the left AND [20, 25) on the right -> free: [5, 25)
free(0, 5)    # touches [5, 25) on the right                         -> free: [0, 25)
```

### Part 2 — O(log n) allocation and release

Implement `MemoryAllocatorLog`, with the same constructor and the same two methods as `MemoryAllocator` above, returning the same values and raising the same errors for every input — but with `allocate` and `free` both running in $O(\log n)$, where $n$ is the number of free blocks currently tracked. Across the whole call sequence there are at most $2 \times 10^5$ calls to `allocate` and `free` combined, and `capacity` can be as large as $10^9$.

```py
class MemoryAllocatorLog:
    def __init__(self, capacity: int) -> None: ...
    def allocate(self, size: int) -> int: ...
    def free(self, address: int, size: int) -> None: ...
```

Example distinguishing first fit from a placement rule that only looks at size: `capacity = 30`; four calls `allocate(12), allocate(3), allocate(6), allocate(9)` return `0, 12, 15, 21`; then `free(0, 12)` and `free(15, 6)` each merge with nothing, since their neighbours (at address 12 and address 21) are still allocated, leaving two free blocks, `[0, 12)` and `[15, 21)`. `allocate(5)` must return `0`, the lower address, even though `[15, 21)` is the tighter fit for a request of size 5.
