Design the execution harness for a multi-step, tool-calling agent, together with the offline evaluation framework built on top of it. An *agent run* pursues a stated goal by alternating between calling a language model and calling external tools (a code-execution sandbox, an internal search API, a ticketing system, a file store, ...) across many steps, until it finishes or is stopped. The harness is shared by several product teams inside the company, each building a different agent (customer support, an internal coding assistant, an ops-automation agent) on the same execution and tool-calling machinery.

Out of scope: training or fine-tuning the underlying language model, and the design of the model's own inference-serving stack — assume a hosted completion API with a given latency distribution is available, and focus on everything around it: how a run is orchestrated, how tools are called safely, how the full trajectory of a run is recorded, and how a fixed suite of tasks is scored to catch regressions before they reach production.

Scale for this design:

- 20 runs/second on average across every agent sharing the harness (about 1.7 million runs/day), with a 3x peak-to-average ratio at busy hours.
- A run takes 12 steps on average; a step is one model call, optionally followed by one tool call (about 70% of steps call a tool).
- The hosted completion API averages 900 ms per call. Tool calls average 500 ms blended across a mix where about one in five goes through an isolated code-execution sandbox (averaging 1.9 s including sandbox setup and teardown) and the rest are fast, scoped internal-API or search calls (averaging 150 ms).
- A fixed evaluation suite of 2,000 tasks runs nightly against the current harness and model version; every pull request that touches the harness, a tool definition, or a system prompt runs a 200-task smoke subset before it can merge.

In scope: the run orchestrator and its core abstractions (task, context, tool, trajectory, state, termination condition); the tool-calling interface, including permission scoping, timeouts, retries, and how a tool's errors reach the model; the recording and replay of complete execution traces, including what is sampled for long-term storage; and the offline evaluation framework — datasets and grading, the metrics reported per run, how the evaluation set is kept from leaking into anything the model or its prompts can see, and how evaluation results gate a release through CI and a nightly regression dashboard. Out of scope beyond model training and serving: the tool backends themselves (the code sandbox, the search index, the ticketing system) are given, fixed services, and so is the UI a human uses to read a trace or grade a run.

Produce:

- Requirements and a scale estimate: concurrent runs in flight, tool-call rate, sandbox pool size, and trajectory storage volume.
- A data model (task, run, step, tool definition, evaluation result) and 4-5 core APIs.
- An architecture diagram covering both the production run path and the evaluation path, and a walk-through of one run along that path.
- Deep dives into: tool-call isolation, timeouts, retries and error propagation; trajectory recording and replay, including sampling and its storage cost; and the evaluation framework — datasets and grading, multi-dimensional metrics, preventing evaluation-set leakage and overfitting, and wiring regression tests into CI. For each, compare at least two alternatives, say which you would pick, and state the cost of that choice.
