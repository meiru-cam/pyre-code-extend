# Design Notes and Optional DeepSeek Review (Plan 6 of 6)

**Goal:** Persist structured system-design notes separately from code, provide optional advisory DeepSeek review, harden the external-AI data boundary, and verify the complete 12-exercise vertical slice.

**Base:** New isolated `feature/design-notes-ai-review` worktree from merged Plan 5 `origin/main`. Deterministic grading alone controls Solved status.

## Task 1: Design-note domain and SQLite migration

**Files:** modify `grading_service/main.py`; create `tests/test_design_notes_api.py`; modify frontend types.

- RED API tests for `PUT/GET /design-notes/{task_id}` with `session_token`, `contract_version`, and structured fields: API boundaries, state/ownership, failure/recovery, backpressure/concurrency, durability/idempotency, observability, security, and tradeoffs.
- Add additive `design_notes` table keyed by user/task/version. Submission history records whether a note was present but Solved remains evaluator-only. Migration is serialized and preserves all existing data.
- Reject unknown task IDs, invalid versions/fields, oversized notes, and cross-session access.

## Task 2: Local draft editor and separate statuses

**Files:** create design-note local-storage helper/tests and editor component; modify problem workspace/result UI.

- RED jsdom interaction tests: each rubric field drafts/restores independently; submit persists; implementation status, note status, and AI-review status never overwrite one another; hint levels remain independently revealable.
- Reference example and rubric are optional to open, not prerequisites.

## Task 3: Provider-neutral design review

**Files:** add a provider adapter and `/api/design-review`; test request minimization and response/error mapping.

- `DesignReviewRequest` sends only exercise ID/title/description, visible rubric, and design note. It must never contain learner code, reference solution, unshown test code, repository files, environment values, or submission history.
- Default optional provider is DeepSeek's OpenAI-compatible endpoint/model configured only through `web/.env`; no credential is stored in browser state, SQLite, logs, fixtures, snapshots, or generated files.
- Distinguish missing config, authentication, rate limit, timeout/unavailable, and malformed response. AI review is advisory transient UI state and cannot alter progress.

## Task 4: External-AI secret boundary

**Files:** create/test a local secret-pattern scanner and explicit AI-sharing UI copy; harden existing AI Help.

- Ask-a-question mode sends query plus minimal exercise context and no code by default.
- Review-my-code mode includes current code only after the learner explicitly chooses it.
- Block likely API keys, bearer tokens, private keys, credential assignments, and `.env`-like payloads before any external fetch. Test false-positive-safe ordinary code and design prose.
- Server constructs provider credentials from environment and strips them from all returned errors.

## Task 5: Full vertical-slice quality and integration gate

- Reference solutions pass all 12 exercises; every required mutation is rejected; seeded suites pass three consecutive times; agent/guardrail failure scenarios remain offline.
- Verify schemas, exports, starters, paths, four separate path pages, hint interactions, behavior summaries, versioned progress, design-note API/UI, AI request minimization, and no effect of AI on Solved.
- Run full Python tests, deterministic exporter/solution builds with clean diff, TypeScript, all Vitest/jsdom tests, Next production build, and manual browser smoke on frontend port 4001/backend 8000 without disturbing existing services.
- Run a secret-pattern scan of staged paths/names only (never print or read `web/.env`), confirm no env file is tracked/staged, review/fix, PR, merge, and audit that Plans 2–6 are all present on `origin/main`.
