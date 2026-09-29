Design the retrieval-augmented generation (RAG) system behind an enterprise chat assistant. Each enterprise customer (a *tenant*) connects its internal data — either by uploading files directly or by linking a source system (a wiki, a ticketing system, a chat archive, ...) through a connector that keeps a copy in sync — and every employee at that company can then ask the assistant questions in a chat interface. Every answer must be grounded in the connected documents, must carry citations to the specific passages it draws on, and must only draw on documents the asking user is allowed to see in the source system. A single tenant's documents, retrieval results, and conversations must never be visible to another tenant. Conversations are multi-turn: a follow-up question can refer back to earlier turns in the same session.

Scale for this design:

- 3,000 tenants, averaging 10,000 documents each — 30,000,000 documents in total.
- Documents average 1,200 tokens (roughly 3-4 pages); some are much shorter (chat threads, tickets) and some much longer (contracts, design docs).
- 3,000,000 questions per day across all tenants (about 1,000 per tenant per day on average), with usage concentrated in each tenant's business hours.
- A newly uploaded or edited document must become searchable within 5 minutes; a deleted document, or one whose permissions were just narrowed, must stop being retrievable within the same 5 minutes.
- Target latency for an answer: time to the first generated token under 800 ms at the median and under 1.8 s at the 95th percentile.

In scope: the ingestion pipeline (parsing, chunking, embedding, indexing) for documents arriving through connectors or direct upload; the retrieval, ranking and generation path that turns a question into a cited answer; multi-turn session handling; per-document access control enforcement; multi-tenant isolation; citation tracking; and an evaluation plan for retrieval and answer quality. Out of scope: training or serving the underlying language model itself — assume it is available as a hosted completion API with given latency and throughput characteristics; the identity provider (assume a directory service can be asked, for a given user, which groups and documents they can see); OCR and format-specific parsing for exotic file types; the chat UI.

Produce:

- Requirements and a scale estimate: chunk count, vector index memory, write throughput, query QPS.
- A data model (documents, chunks, sessions, messages) and 3-5 core APIs.
- An architecture diagram that separates the offline ingestion pipeline from the online question-answering path, and a walk-through of one question along that path.
- Deep dives into: tenant isolation and permissions; retrieval quality (hybrid retrieval and reranking); and the trustworthiness and evaluation of answers. For each, compare at least two alternatives, say which you would pick, and state the cost of that choice.
