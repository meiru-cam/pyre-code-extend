Design the backend and client-facing API for an AI-assisted rewrite feature inside a collaborative document editor. A user selects a passage of text and picks an instruction — one of four presets (*Improve writing*, *Make shorter*, *Make longer*, *Fix grammar*) or a free-form instruction they type themselves — and the assistant streams a suggested rewrite into a panel next to the selection as it is generated; the user can accept it (replacing the selection), discard it, or try again. The generated text must appear incrementally, token by token, rather than only once the full rewrite is ready, and the feature must hold up under a large number of concurrent users.

The document editor itself — its collaborative sync, storage, and search — is out of scope, and so is the language model: it runs behind an internal streaming completion service, reachable as a GPU-bound unit of work with these given characteristics:

- time to first token: 400 ms on average, for the prompt lengths this feature sends;
- steady-state generation rate once decoding starts: 35 tokens/sec per stream;
- concurrency: each GPU replica serves up to 20 concurrent streams before per-stream generation rate degrades;
- cancellation: when the caller closes a streaming call, the service stops that stream at its next decode step, and it no longer counts toward the replica's 20 concurrent streams.

Content moderation of the selection and of the model's output happens inside that service and is not designed further here.

Scale for this design:

- 5,000,000 daily active users of the editor; 6% of them use the rewrite feature at least once on a given day, averaging 4 invocations each, for 1,200,000 rewrite requests/day.
- Usage concentrates in business hours across time zones; the peak submission rate runs at 5x the daily average.
- An average request carries 220 input tokens (the selection, surrounding-paragraph context, and the instruction) and produces 180 output tokens.
- The model-serving GPU time is charged back internally at $0.05 per 1,000 input tokens and $0.15 per 1,000 output tokens.

In scope: the client-facing request/response API and its streaming transport; admission control, queueing, and overload behavior in front of the model-serving pool; caching and cost controls; the client's incremental rendering and its recovery from a dropped stream or a request error; monitoring, logging, and a production debugging path; and how the design changes at 10x the request rate. Out of scope: the document editor and the model itself, as above, and authentication (assume a valid, already-verified user identity arrives with every request).

Produce:

- A requirements and scale estimate: average and peak request rate, concurrent streams in flight (Little's law), GPU replicas needed, egress bandwidth, and daily token cost.
- A data model and the core client-facing API (start a rewrite, stream it, cancel it, check its status).
- An architecture diagram, a walk-through of one request along it, and what you would monitor and how you would debug a slow or failed request.
- Deep dives into: the streaming transport and how a client's cancellation stops generation on the model-serving side; admission control, queueing, overload degradation, and cost control; and the frontend's incremental rendering and its recovery from an interrupted stream or a failed request. For each, compare at least two alternatives, say which you would pick, and state the cost.
- How the design changes at 10x the request rate.
