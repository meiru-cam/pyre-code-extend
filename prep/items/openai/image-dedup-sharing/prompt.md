Design a service that stores images and lets users share them. A user uploads an image, and afterwards views or downloads it by a user-facing `image_id`, or deletes it. Any request that supplies a currently valid `image_id` may view or download that image; only its owner may delete it. Two uploads whose bytes are byte-for-byte identical must end up sharing a single stored copy of the content; the identity of a piece of content is its SHA-256 hash.

The object store holds arbitrary byte blobs under string keys and gives three guarantees: a *create-if-absent* (conditional) write, which creates a key only if it does not already exist and otherwise fails without changing anything; *multipart upload*, so a large object can be sent as several parts and becomes readable only once the upload is explicitly completed; and atomicity of a write to a single key — no reader ever observes a partially written object, whether it arrived as one PUT or as a completed multipart upload. The metadata store is a transactional relational database: several statements against it can be grouped into one all-or-nothing commit.

Scale for this design:

- 40,000,000 registered users; 400,000 image uploads per day.
- Images average 3 MiB; the largest image the service accepts is 100 MiB, and any upload at or above 8 MiB goes through multipart.
- About 30% of daily uploads are byte-identical to content the service already stores (screenshots, memes, and other images that get reshared widely).
- 4,000,000 views/downloads per day — 10 per upload.
- Download latency target: 150 ms at the median, served from a CDN edge cache, and 900 ms at the 95th percentile, a cache miss served from the object store.

In scope: the upload, view/download and delete lifecycle of a single image; computing and verifying the SHA-256 identity of uploaded content; keeping the metadata store and the object store consistent across the multi-step upload and delete paths, including a process dying in the middle of either one; two users uploading identical content at the same time; and what happens to shared content when one user's reference to it is deleted while other users still hold references to the same content. Out of scope: generating thumbnails or transcoding images to other formats; near-duplicate detection — finding images that are visually similar but not byte-identical is a different problem from this exact-content deduplication, though it can come up as an extension; and the authentication system itself (assume every request already carries a verified user id).

Produce:

- Requirements and a scale estimate: bytes uploaded per day before and after deduplication, storage growth per year, and download bandwidth under the CDN hit-rate assumption you choose.
- A data model for stored content and for a user's reference to it, and 3-5 core APIs.
- An architecture diagram, with a walk-through of one upload and one download along it.
- Deep dives into: (a) the upload path as a multi-step commit spanning the object store and the metadata store, and what a crash leaves behind at each step; (b) deletion and reference counting, including the race between the last reference to some content being deleted and a new upload of that same content; (c) the read path at scale — caching, content shared by many images, and access after deletion.
