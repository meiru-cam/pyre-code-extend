Design the telemetry platform behind a company's own deployed client software — a desktop application, a mobile app, and an SDK embedded in other internal products — each instance of which continuously emits *telemetry events*: discrete records of something that happened (a feature was used, a job finished, an error occurred) alongside periodic numeric measurements (a duration, a queue depth). Every event names the logical metric it reports as a `telemetry_name` (`sync_completed`, `search_performed`, ...), carries a set of *dimensions* — typed tags such as `platform` or `client_version` used to filter and group results — and, for a measurement, one or more numeric *measures*. Internal teams (product analytics, on-call SRE, customer support, security/compliance) read the data back through live dashboards, ad hoc SQL-style queries, and alerts that watch a rolling aggregate and page someone on a threshold breach; a team's access to a given event's fields depends on how sensitive each field is.

Scale for this design:

- 24,000,000 deployed client instances, each emitting 180 events/day on average — 4,320,000,000 events/day platform-wide — at an average serialized size of 500 bytes/event (a compact envelope plus payload).
- Traffic has roughly a 4x peak-to-average ratio: the installed base concentrates in a few timezones' business hours, and clients flush their local buffer on a fixed interval, which synchronizes bursts across large cohorts of clients.
- Retention: the raw, as-received event store is kept 30 days; curated, per-event-type tables are kept 400 days (13 months) for trend analysis. A field classified sensitive (below) is purged or hashed out of curated storage after 30 days regardless of the table's own retention.
- Roughly 4,000 distinct canonical telemetry names are registered across the product surface, and 2,000 alert rules are active at once.
- A field is classified sensitive when its value identifies or could be attributed to one person or account (an email address that ends up in a free-text error message, a raw device identifier) rather than describing aggregate product behaviour.

Every event a client sends carries a fixed envelope:

```py
class TelemetryEvent:
    event_id: str        # client-generated; identical across a retried send of the same logical event
    client_id: str        # stable per installed client instance
    client_version: str
    schema_version: int   # the registered schema version this payload's shape conforms to
    telemetry_name: str   # as sent by the client -- may be an old, non-canonical variant
    event_time: int        # unix ms, the client's own clock, when the event happened
    dimensions: dict[str, str | int | float | bool]
    measures: dict[str, float]
```

- `event_id` is chosen once by the client and resent unchanged on every retry of the same logical event (a network retry, or an offline client replaying its buffer after reconnecting); two events sharing an `event_id` are the same occurrence, counted once.
- `event_time` is the client's own clock and is not trusted: it can be skewed or wrong outright. The platform separately stamps every event with `receive_time`, its own clock reading when ingestion first accepts it; `receive_time` is monotonic per ingestion node and always close to now.
- `schema_version` names the version of `telemetry_name`'s registered field list the client believes its payload matches; it does not by itself guarantee `telemetry_name` is a name the registry currently recognizes as *canonical* (the registry's one standing name for a metric, with every other spelling a *variant* of it) — that is exactly the follow-up below.
- Clients batch events locally and flush at a fixed interval or size threshold, whichever comes first; a batch send is at-least-once — the client retries a batch it never received an acknowledgement for.

In scope: the ingestion API and its authentication; the durable log and stream processing (validation, deduplication, enrichment, late and out-of-order handling); the raw and curated storage tiers, their partitioning and retention; the query engine's access-control layer and PII handling; alerting on streaming aggregates; the schema registry, its compatibility rules, and the follow-up below. Out of scope: the dashboard or BI tool's own user interface (assume it calls the query engine's API); anomaly detection or other ML-based alerting; billing or cost allocation across teams; the client SDK's transport and retry logic beyond the contract stated above.

**Follow-up.** A meaningful share of installed clients are on old versions that cannot be upgraded soon. Different client versions report what is logically the same event inconsistently: a background-sync completion, for instance, reaches the ingestion API as `sync_completed` from current clients but as `SyncCompleted`, `sync-completed`, or `evt_sync_done` from three different older versions, and some versions that do send `telemetry_name: sync_completed` correctly still name its duration measure `duration_ms` (milliseconds) rather than the current schema's `duration_seconds`. Design how the schema evolves and how downstream data — both events arriving now and the history already written — stays correct under one name, one field, and one unit, covering:

- incoming events under a recognized variant are folded into a single canonical metric rather than fragmenting it into several;
- data already written under a variant can be corrected once that variant is recognized, without reprocessing everything;
- the name (and any unit conversion) as originally sent is kept alongside the canonical record, so a mapping decision can be audited or reversed;
- two telemetry names that are only superficially similar are never silently folded into one metric;
- mapping rules can be added, edited, and deprecated over time, with a stated effect on queries already running against live data versus on data already stored.

Produce:

- A requirements and scale estimate: ingest throughput and bandwidth, the durable log's partitioning, the stream-processing tier's size, storage for each of the raw, curated, and rolled-up tiers, and the cost of evaluating the active alert rules.
- A data model (the event envelope as stored, the schema registry, the name-mapping table, alert rules, and rolled-up aggregates) and 5-6 core APIs.
- An architecture diagram separating ingestion and buffering from stream processing and storage from query and alerting, and a walk-through of one event along it.
- Deep dives into: (a) ingestion, the durable log's partitioning, and stream processing — deduplication under at-least-once delivery, and late and out-of-order events; (b) the raw-versus-curated storage split, partitioning and compaction, the query engine, and alerting on streaming aggregates, including high-cardinality dimensions; (c) the schema registry, compatibility rules, and the naming-variant follow-up above; (d) access control by field classification, PII handling, and retention. For each, compare at least two alternatives, say which you would pick, and state the cost.
