Design the backend for a browser-based IDE, in the style of Replit or GitHub Codespaces: a user opens a tab, and without installing anything locally, edits files, opens a terminal, runs shell commands, and watches stdout/stderr appear as the command produces it. Each user's workspace runs inside its own isolated sandbox — a container or a lightweight virtual machine — with its own filesystem, process tree and resource limits; nothing in one workspace's sandbox is visible to, or can affect, another workspace's. A workspace goes through a lifecycle: it is created once, the sandbox behind it is started (or resumed) whenever the user reopens it, put to sleep (hibernated) after the user stops using it, and eventually deleted. Files a user writes, and packages they install with a package manager inside the sandbox, are part of the workspace's persisted state and survive hibernation and resume, same as source files; they are lost only when the workspace itself is deleted. If the browser tab loses its connection mid-session — a flaky network, a laptop going to sleep — reconnecting must reattach to the same running terminal rather than starting a new one, and any command still running keeps running while the tab is away.

Scale for this design:

- 2,000,000 registered users.
- 36,000 workspaces holding a live sandbox at once during peak hours; an interactive session, from opening a workspace to closing its last browser tab, lasts about 20 minutes on average.
- Each workspace gets a fixed quota: 2 vCPUs, 4 GiB RAM, and 10 GiB of persistent file storage.
- Opening a workspace: a live sandbox ready in under 2 seconds at the 95th percentile on the normal path; under 7 seconds at the 95th percentile on the rare path where no pre-started sandbox is available and one has to be provisioned from scratch.
- Terminal output: a line printed by a running command must reach the browser within 150 ms at the 95th percentile.
- A workspace with no open browser connection and no running foreground process hibernates after 15 minutes in that idle state.

In scope: workspace lifecycle management (create, start, hibernate, resume, delete); sandbox isolation and resource limits; file (and installed-package) persistence; the terminal streaming and reconnection path; and capacity and warm-pool sizing for the sandbox fleet. Out of scope: the code editor itself — syntax highlighting, autocomplete and language servers are assumed to run client-side or against a separate, already-designed service; billing and plan management; and real-time collaborative editing (two people typing in the same file at once) — sharing here only grants another user view or edit access to the same workspace, one active editor at a time.

Produce:

- A requirements and scale estimate: concurrently running sandboxes and how many hosts the sandbox fleet needs, warm-pool size, terminal-output bandwidth, and total persistent storage.
- A data model (workspace, sandbox assignment, file, process) and the core APIs — REST for workspace lifecycle and file management, WebSocket for terminal I/O and file-change events — listed separately.
- An architecture diagram and a walk-through of opening a workspace and running a command along it.
- Deep dives into: sandbox isolation; terminal streaming and reconnection; and workspace lifecycle and cost. For each, compare at least two alternatives, say which you would pick, and state the cost of that choice.
