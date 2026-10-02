Design a service that turns a long URL into a short code and, when a client requests that code, redirects them to the original URL. A create call is served by a general HTTP API used by client applications and a lightweight web form; a redirect is served by following a link of the form `https://s.example/{code}` from anywhere on the web — a browser, a chat app's link preview, a scanned QR code. A caller may request a custom alias instead of a generated code; a custom alias and a generated code share one namespace, and if the requested alias is already taken the create call fails outright, never falling back to a different code. A code is redirectable the instant its create call returns success, from any region a client happens to be in.

A link has no expiration by default, though a caller may set one at creation time; once a link is expired or explicitly deleted, requests for its code stop redirecting for good, and the code is never reassigned to a different destination later. An authenticated caller (one that carries an `owner_id`) who submits a URL that normalizes to one they already own (same scheme and host, case-insensitively, same path and query, no fragment) gets back their existing code unless they ask for a fresh one; anonymous creates (no `owner_id`) are never deduplicated against anything. Every successful redirect counts toward that code's click count; the count is approximate, a bounded amount of lag is acceptable, and nothing else about the click (referrer, device, location) is recorded.

Scale for this design:

- 1,200,000 new links created per day (about 14/s on average).
- A 100:1 ratio of redirects to creates.
- Traffic is bursty around link shares and campaigns: plan for a 5x peak-to-average ratio on both creates and redirects.
- Capacity-plan for 5 years of accumulation at the above creation rate.
- Redirect latency target: p50 under 20 ms, p99 under 150 ms. Redirect availability target: 99.95%.

In scope: the create and redirect APIs; short-code generation and collision handling; the durable mapping store; the caching layer behind the redirect path; multi-region serving of redirects; an owner's paginated list of their own links; the approximate click count. Out of scope: the identity layer that produces `owner_id` for an authenticated call; a moderation policy beyond a pass/fail malicious-URL check at creation time; branded/custom domains; analytics beyond the raw click count.

Produce:

- A scale estimate: short-code length, storage footprint, redirect QPS (average and peak), and how many popular links the cache needs to hold.
- A data model for the link record and its secondary indexes, and 3-5 core APIs.
- An architecture diagram that separates the write path from the read path, and a walk-through of one redirect along that path.
- Deep dives into: short-code generation; the read path; and storage and scale. For each, compare at least two alternatives, say which you would pick, and state the cost of that choice.
