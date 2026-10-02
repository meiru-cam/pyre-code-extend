Design a backend that fills in crossword puzzles: given a board and a dictionary, place a word in every slot so that every pair of crossing slots agrees on their shared letter, or determine that no such placement exists. A puzzle arrives as an explicit list of slots — position, direction (across or down), and length; the solver never sees the black-square grid that produced them, only the slots and how they cross. A word may be used at most once across the whole grid; the same word may not fill two different slots. Any clue text attached to a slot is metadata the solver ignores — any word of the right length and letters is an acceptable fill.

Scale for this design:

- A board is about 50×50 cells with about 100 slots to fill; slot lengths range from 3 to 15 cells.
- The dictionary holds about 1,000,000 English words, grouped by length.
- 20,000 solve requests arrive per day on average, with submissions reaching 5x that average during a recurring one-hour peak (a nightly batch of newly authored puzzles, say).
- A solve attempt runs for at most 5 minutes (300 s) of wall-clock time; if no filling has been found by then, the service reports that it gave up, which is a different outcome from having proven that no filling exists.
- The service runs on a shared, elastic worker pool; a single puzzle may claim at most 64 workers at once, so many puzzles make progress concurrently without one puzzle starving the rest.

In scope: turning a puzzle's slots and the dictionary into one valid filling or a definitive "no filling exists" / "gave up" result; splitting the search across the worker pool; pruning; rebalancing load between workers; stopping every worker once any one of them finds a filling; tolerating a worker crash or a coordinator failover without losing search progress or double-counting it; determining, reliably, when no filling exists. Out of scope: generating the black-square layout itself; clue authoring or mapping a clue's text to candidate answers; a UI for manually editing a filling; ranking multiple valid fillings by quality (any valid filling is an acceptable answer).

Produce:

- A requirements and scale estimate: why filling the board by brute force is infeasible on one machine, dictionary and index memory, worker pool size, and task-queue throughput.
- A data model (puzzle, slot, task, worker) and 3-5 core APIs.
- An architecture diagram and a walk-through of one puzzle along it.
- Deep dives into: splitting the search into tasks and balancing load across workers; pruning and slot ordering, and how they interact with splitting; and fault tolerance — what happens when a worker or the coordinator crashes, and why the "no filling exists" conclusion stays correct. For each, compare at least two alternatives, say which you would pick, and state the cost.
