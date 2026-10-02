Design a website that lets a user experiment interactively with a hosted text-completion model: type a prompt, adjust a few sampling parameters, watch the completion stream in token by token, and save a prompt-and-parameters combination as a *preset* to reload and rerun later. You are the only engineer on this project, starting from an empty repository — every technology choice, from the wire protocol to the storage engine, is yours to make and justify.

The completion model itself is not something you build: it is reachable through an existing HTTP endpoint, `POST /v1/completions`, with a JSON body `{model, prompt, max_tokens, temperature, top_p, stop, stream}`. With `stream: false` it returns one JSON object once generation finishes. With `stream: true` it instead returns a chunked `text/event-stream` response: each event's `data` field is a JSON object `{text, finish_reason}`, where `text` is a few newly generated tokens of text and `finish_reason` is `null` until the last event, then one of `"stop"` (a stop sequence was hit) or `"length"` (`max_tokens` was reached). The endpoint bills the owner of the API key per generated token, and closing a streaming request's connection early stops generation.

The page has a prompt box, a run button, a panel of the four parameters below it (temperature, maximum length, top-p, and up to four stop sequences), and a sidebar listing the user's saved presets. Submitting a prompt appends the streamed completion right after the prompt text, in the same box, rendered so the original prompt reads as plain text and the appended completion is visually distinguished from it (for example, a highlighted background).

Scale for this design:

- 60,000 registered users, of whom about 6,000 are active on a given day.
- An active user runs about 8 completions a day on average, for roughly 48,000 runs/day system-wide.
- A registered user has 8 saved presets on average (many have none, a few have dozens).
- Usage is heaviest during a combined US/Europe working-hours window, roughly 8 hours a day, which carries 60% of a day's runs.
- A completion streams for about 5 seconds on average (a model generating around 40 tokens/second, and an average completion length of 200 tokens).

In scope: the prompt-and-parameters editor and its interaction with the completion endpoint above; streaming the completion into the browser; cancelling a run in progress; the full lifecycle of a preset (create, list, load, edit, delete); per-user rate limiting on how many completions can be requested. Login is out of scope beyond assuming an existing identity service authenticates the user and attaches a verified user id to every request that reaches your backend — you do not need to design sign-up, sessions or password handling. Also out of scope: billing or usage-based cost tracking, sharing a preset with other users or a team, and any change to the completion model itself (fine-tuning, or choosing among model variants beyond passing a `model` string through unchanged).

Produce:

- Requirements and a scale estimate: peak concurrent streams and preset storage volume.
- A data model (users, presets, and whatever tracks a run in progress) and the core APIs: preset CRUD, starting a completion, streaming it back, and cancelling it.
- An architecture diagram covering the browser, your backend, the completion API, and preset storage, with a walk-through of one run along that path.
- Deep dives into: streaming delivery to the browser; front-end state management; and saving a preset safely under concurrent edits.
