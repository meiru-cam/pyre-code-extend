Design the webhook delivery system behind a multi-tenant SaaS platform. Each tenant registers one or more *endpoints* — an HTTPS callback URL plus the subset of event types it wants — and the platform must deliver every matching event to every one of that tenant's active endpoints as an HTTP POST, with **at-least-once** delivery: a retried delivery is preferable to a lost one. Events are produced entirely inside the platform (a job finishes, an order changes status, ...) and each is scoped to exactly one tenant; an event is never delivered to another tenant's endpoints.

Scale for this design:

- 216,000,000 events/day platform-wide, averaging 3 subscribed endpoints per event, for 648,000,000 baseline delivery attempts/day before any retries.
- Traffic has roughly a 3x peak-to-average ratio (tenants' business hours overlap enough that the day is not flat).
- Endpoint response times split into three tiers, by traffic share: 90% of deliveries go to endpoints that answer in roughly 200 ms; 8% go to endpoints that are consistently slow, taking roughly 3 s; 2% go to endpoints that are unreachable at that moment, so the call runs all the way to the platform's own 10 s request timeout. Which endpoints fall in which tier changes over time (a slow deploy, an outage), but this is the mix at any instant.
- Delivery latency target: 95% of deliveries must have their first attempt start within 2 s of the event being accepted. For an endpoint that is actually reachable, a delivery must reach a terminal outcome — delivered, or dead-lettered (given up on and kept for the tenant to inspect and replay) — within 90 minutes of its first attempt.
- 150,000 tenants, averaging 2 registered endpoints each (300,000 endpoints total).

In scope: the endpoint registration API; accepting an event and fanning it out to the matching subscriptions; the delivery workers, their retry policy and backoff; isolating a slow or dead endpoint from the rest of the fleet; dead-lettering and manual replay; request signing and protection against server-side request forgery (SSRF) on the outbound call. Out of scope: the internal services that decide an event happened (assume one arrives at the ingestion API already carrying a tenant id, an event type, and a JSON payload); the tenant-facing dashboard; billing or metering of webhook volume; delivery channels other than HTTP.

Produce:

- A requirements and scale estimate: baseline and peak delivery throughput; the number of HTTP requests in flight at once (via Little's law), broken out by response-time tier; the number of delivery workers that takes; the extra traffic retries add; a storage estimate for events and delivery records.
- A data model (endpoints, events, deliveries) and 3-5 core APIs.
- An architecture diagram that separates the synchronous registration-and-ingestion path from the asynchronous fan-out-and-delivery path, and a walk-through of one event along that path.
- Deep dives into: (a) at-least-once delivery and idempotency, including which HTTP response codes are retried and which are not; (b) the retry policy and how a slow or permanently dead endpoint is kept from starving delivery capacity that other endpoints need; (c) delivery ordering guarantees and the security of the outbound call (payload signing and SSRF protection).
