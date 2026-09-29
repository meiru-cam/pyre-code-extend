Design the public API and *dispatcher* that sit in front of a pool of GPU model replicas serving a single large language model. A *replica* is one running copy of the model, bound to a fixed set of GPUs, that can execute at most one *batch* — a group of requests processed together in a single forward pass — at a time. Every replica in the base design exposes the same fixed contract:

```py
def run_batch(prompts: list[str], max_output_tokens: int) -> list[str]:
    """Runs one inference batch on this replica and blocks until it returns."""
```

- `1 <= len(prompts) <= 64`; the call returns exactly one completion string per input, in the same order.
- The call takes a fixed **100 ms**, regardless of how many of the 64 slots are filled.
- `max_output_tokens` is capped at 512 per call. A request whose completion would run longer than that must be resubmitted with the partial output folded back in as context, continuing where the previous call left off, and pays the fixed 100 ms again for the continuation.
- A replica accepts exactly one `run_batch` call at a time; a second call blocks until the first returns. The base design gives one GPU to one replica; the follow-up in part (e) below changes that.

The dispatcher is everything between the client and the replica pool: it forms batches out of individual requests (bounded by a maximum batch size and a maximum wait), routes each batch to a replica, and returns each request's slice of the batch's output list to the caller that sent it.

Traffic is a mix of two classes. *Real-time* requests are interactive — a person is waiting, the call is synchronous or streamed, and the target is a 95th-percentile (p95) end-to-end latency of **500 ms**. *Offline* requests are submitted as part of a batch job with no per-request latency target, only a job-level completion SLA measured in minutes; they exist to keep GPUs busy with throughput work between real-time bursts. At peak the system receives **10,000 requests/second**, of which 80% (8,000/s) are real-time and 20% (2,000/s) are offline. An average request, of either class, carries **300 input tokens** and generates **200 output tokens**. Every replica serves one `(model, model_version)` pair; the pool holds several models and versions concurrently, and a replica whose model has gone cold can be reloaded for another model at the cost of a load time on the order of tens of seconds.

In scope: the dispatcher's batching policy, load balancing and admission control in front of the replica pool, the memory accounting for a KV cache and an optional prefix cache at the level the dispatcher can see (bytes per token, hit rate, eviction — not the attention kernel itself), isolating real-time from offline traffic, and a follow-up that fixes a separate 8-GPU pool shared by a large model (one batch needs all 8 GPUs) and a small model (one batch needs 1 GPU). Out of scope: the model's internals (attention, sampling, tokenization — `run_batch` is a black box that respects the contract above), authentication itself (assume every request arrives with a verified `tenant_id` and a rate-limit tier already attached), and training.

Produce:

- A requirements and scale estimate: replicas needed at peak — a theoretical floor assuming every batch is full, and a provisioned fleet split across the real-time and offline pools — the KV-cache bytes per token for a stated model configuration, and the queueing-delay budget inside the 500 ms SLO.
- A data model and the core API: a synchronous call, a streamed call, and an asynchronous batch-job submission.
- An architecture diagram and a walk-through of one request along it.
- Deep dives into: (a) the batching policy and the gap between theoretical maximum and achieved throughput — batch fill, queueing, the max-wait trade-off, and continuous, token-level batching as the next step; (b) load-balancing signals and admission control — rate limits per tenant, bounded queues, load shedding, retries with idempotency keys, cancellation and timeouts, and how streaming works against the fixed-batch contract; (c) the KV cache and a prefix cache — when it pays off, the memory arithmetic per sequence, and eviction; (d) isolating real-time from offline traffic — SLOs, capacity planning, observability; (e) a follow-up fixing a separate 8-GPU pool shared by a large model whose batch needs all 8 GPUs and a small model whose batch needs 1, at equal batch latency — the queueing and dispatch policy, and its utilisation cost. For each, compare at least two alternatives, say which you would pick, and state the cost.
