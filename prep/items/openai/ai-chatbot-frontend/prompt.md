Design the web application behind a simple AI chatbot: a single chat page, similar to a typical ChatGPT-style main screen. A user must sign in before using it; once signed in, they type a text message, send it, and watch the assistant's reply appear on screen token by token as it is generated, rather than all at once. Replies are produced by an existing chat completion API: it takes an ordered list of `{role, content}` messages (`role` is `system`, `user`, or `assistant`) and streams the reply back as a sequence of text deltas, followed by one final event carrying a finish reason (`stop` or `length` on success, `error` on failure). Before streaming starts it may instead refuse a request with HTTP `429` (rate-limited) or `503` (overloaded). It bills per input and output token, and closing a streaming request's connection early stops generation. Assume the API itself, its latency, and its throughput are already provisioned and out of scope.

Three rules fix the shape of the design:

- The backend must never persist a user's messages or the assistant's replies: no database row, log line, or cache entry may retain conversation content past the single request that produced it.
- All conversation state — every message, in order, with its role and content — lives in the browser page's own JavaScript memory, not in `sessionStorage`, `localStorage`, or any other persistent browser storage.
- Reloading the page discards that in-memory state: a refresh clears the conversation, and the next message the user sends starts a brand new one.

Scale for this design:

- 150,000 daily active users, opening an average of 1.5 conversations per day each — 225,000 new conversations per day.
- Each conversation averages 8 user turns; a user turn averages 80 tokens, an assistant reply averages 300 tokens, and the completion API streams at roughly 40 tokens/second per reply.
- Usage is concentrated in a roughly 16-hour waking window shared by most users' time zones, and the busiest hour runs at about 5x the day's average request rate.

In scope: the browser-side chat application (rendering, conversation state, streaming updates); the login flow and how the session credential is carried on later requests; a thin backend that authenticates the user, forwards a message to the completion API, and relays the stream back; error handling and retry for a dropped connection or a failed completion request. Out of scope: the completion API's own implementation; storing or searching past conversations; multi-device or multi-tab sync of a single conversation; file or image attachments; rate limiting or billing beyond stating where a limiter would sit.

Produce:

- A requirements and scale estimate: peak concurrent streaming replies, the resulting egress bandwidth, and the memory footprint of one conversation in the browser compared with a typical browser storage quota.
- A data model for the browser-side conversation state and the backend's session state, and 3-5 core APIs (login, sending a message and streaming its reply, and canceling an in-flight reply).
- An architecture diagram and a walk-through of one message along it.
- Deep dives into streaming rendering, login and credential handling, and error handling and retry. For each, compare at least two alternatives, say which you would pick, and state the cost of that choice.
