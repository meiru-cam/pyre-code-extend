# AI Help Review Reliability Design

## Problem

`Ask question` and short synthetic `Review my code` requests can succeed, while
real code-review requests can return a generic failure. The current server route
maps an invalid JSON response or a missing, empty
`choices[0].message.content` value to `malformed_response` without recovery.
The client then collapses most server errors into `AI request failed`, and the
server logs do not identify the request mode or failure stage.

The browser sends exactly one request per explicit click. This repair must not
introduce browser retries, automatic code sharing, or any connection between AI
feedback and deterministic exercise completion.

## Goals

- Recover from one transient provider failure without requiring another learner
  click.
- Give code-review requests enough time for their larger prompt while retaining
  a bounded total duration.
- Preserve the provider-neutral OpenAI-compatible interface.
- Surface stable, actionable error categories in the UI.
- Add useful diagnostics without logging prompts, learner code, response
  content, provider URLs, credentials, or environment values.
- Keep all regression tests deterministic and offline through a mocked provider.

## Non-goals

- Changing deterministic grading, solved status, hints, or submission history.
- Persisting AI Help prompts or responses.
- Displaying provider chain-of-thought or `reasoning_content`.
- Adding a DeepSeek-specific SDK.
- Retrying authentication failures, local validation failures, secret-boundary
  failures, or rate limits.

## Server Design

Move the upstream request behavior behind a server-only AI Help provider module.
Its interface accepts the validated request, provider configuration, and an
optional fetch dependency for tests. It returns either advisory guidance or a
typed, sanitized failure.

Use mode-aware per-attempt timeouts:

- `question`: 20 seconds
- `review_code`: 30 seconds

Allow at most two attempts. Retry only:

- transport failures and provider timeouts;
- HTTP 408, 500, 502, 503, and 504;
- a successful HTTP response with invalid JSON;
- a successful HTTP response without a non-empty
  `choices[0].message.content`.

Do not retry HTTP 400, 401, 403, 404, or 429. Preserve the existing local
request rate limit; an internal retry counts as part of the same learner
request.

The provider module must accept only string `message.content`. It must not use
`reasoning_content` as learner-facing guidance. After retry exhaustion, return
the most specific stable failure code:

- `authentication_failed`
- `rate_limited`
- `provider_timeout`
- `provider_unavailable`
- `malformed_response`

Emit one sanitized warning only after final failure. The warning may contain the
request mode, learner-code character count, attempt count, upstream HTTP status,
and stable failure code. It must not contain learner text, code, response
payloads, provider configuration, URLs, or credentials.

## Client Design

Keep the existing explicit `Ask question` and `Review my code` actions. The
browser continues to send learner code only for the explicit review action and
never retries.

Map stable server failure codes to specific learner-facing messages:

- timeout: the provider took too long;
- malformed response: the provider returned an incomplete response;
- rate limit: try again later;
- authentication or missing configuration: server configuration needs
  attention;
- unavailable or unknown: the provider is temporarily unavailable.

Do not display raw upstream error bodies.

## Testing

Follow red-green-refactor.

Server tests must prove:

- review mode retries one malformed response and returns the second valid
  response;
- review mode retries one transient 5xx response;
- timeout behavior is bounded and retryable without real timers;
- authentication and rate-limit failures are not retried;
- two malformed responses return `malformed_response`;
- sanitized diagnostics contain metadata but no prompt, code, response body,
  URL, model configuration, or API key;
- question mode still omits learner code.

Client tests must prove:

- one click still produces exactly one browser POST;
- review mode includes learner code only after explicit consent;
- each stable server failure code maps to the intended message;
- raw provider details are never rendered.

Run the focused Vitest suites first, then all frontend tests, TypeScript
validation, and the production build. The final runtime check uses a synthetic,
non-repository review payload and records only status, stable error code, and
duration.

## Acceptance Criteria

- A transient malformed or 5xx provider response can recover within the same
  explicit Review request.
- Permanent failures return a specific sanitized error and never leak external
  provider details.
- The browser sends one POST per learner action and performs no hidden retry.
- `web/.env` remains ignored, untracked, unread, and uncommitted.
- Existing deterministic evaluation and solved-state behavior are unchanged.
