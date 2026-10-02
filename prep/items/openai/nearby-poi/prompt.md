Design a service that finds points of interest (POIs) — restaurants, shops, cafes, and other businesses — near a user's location. A client supplies its coordinates and either a search radius or a desired count $K$, plus an optional category filter, and the service returns the matching POIs ordered by real-world distance from the client. Each POI has a name, address, category, rating, and a set of weekly opening hours; the owning business can edit any of these at any time (hours, category, a temporary or permanent closure). Reads vastly outnumber writes.

Scale for this design:

- $2\times10^8$ (200 million) POIs worldwide.
- 864,000,000 nearby-search requests per day across all regions, concentrated in each region's local daytime and evening hours.
- Target latency for a single search request: p50 under 60 ms, p95 under 150 ms.
- On an average day, about 0.5% of POIs are created, edited (most often just the hours), or closed by their owners; a single owner-facing bulk edit can touch up to 10,000 POIs at once.
- An edit or a closure must stop, or start, being reflected in search results within 2 minutes.

In scope: the primary POI store and its write path; the spatial index — how it is built, kept in sync with the primary store, and either replicated or sharded across query-serving nodes; the radius-search and exact-$K$-nearest query paths, both with an optional category filter; caching of popular queries. Out of scope: ranking beyond distance and rating (personalization, ads, sponsored placement); rendering map tiles and the CDN that serves them; turning a free-text address into coordinates; photo and review storage; authenticating business owners.

Produce:

- Requirements and a scale estimate: bytes per POI record; spatial-index memory, and whether it fits on one server; peak QPS and the number of query-serving nodes it implies; the size of a query-result cache.
- A data model (the POI record and the spatial index's entries) and 3-5 core APIs.
- An architecture diagram that separates the write path from the read path, and a walk-through of one search request along that path.
- Deep dives into: the choice of spatial index (Geohash vs. QuadTree) and how the boundary problem is handled; how an exact radius search and an exact $K$-nearest search are computed from that index; and how the index is replicated or sharded, including how a dense city's POIs are kept from overloading a single shard.
