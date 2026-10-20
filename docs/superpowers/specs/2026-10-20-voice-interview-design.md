# Voice Interview Design

## Problem

The interview gate asks real interview questions, but learners answer them by
typing, and nobody watches them code. In a live ML coding interview, candidates
explain their reasoning out loud while writing code, and an interviewer probes
shapes, edge cases and failing tests. Typing answers does not practice that
skill.

## Goals

- Let the learner answer the Opening and Deep dive questions aloud, with the
  spoken answers counting toward the interview gate under its existing rules.
- Let the learner code while talking. The interviewer sees their code and Run
  results and asks follow-up questions.
- Ask the Tradeoffs questions aloud after Submit.
- Keep every exercise fully usable offline and without provider keys.

## Non-goals

- Grading spoken answers. Solved status stays deterministic.
- Persisting transcripts or audio.
- Sending images. The SDK supports multimodal messages, but code is sent as
  text because text is exact and cheaper.

## Data boundary

This is the one deliberate exception to "no automatic code sharing" in
`2026-07-25-ai-help-review-reliability-design.md`. The exception is scoped:

- The learner opens a consent panel and presses **Allow mic and start** for each
  session. Nothing is sent before that.
- While the session is connected, the browser sends:
  - microphone audio;
  - a code snapshot after `SNAPSHOT_IDLE_MS` (3 s) of editor quiet, only when
    the code has changed, at most `MAX_SNAPSHOT_CHARS` characters;
  - the editor contents and a short result summary after each Run or Submit.
- Code and Run/Submit error text are run through `scanForExternalAiSecrets`
  before they are sent. A matching snapshot is replaced by a "withheld"
  notice; a matching result keeps only its pass counts.
- Provider error text is never shown in the UI; any connection error shows
  "The voice interview could not start."
- **End**, leaving the exercise, or closing the tab ends the session, and
  nothing more is sent.
- The ElevenLabs API key stays on the server. The browser receives only a
  short-lived signed URL from `POST /api/voice-interview/session`.

## Flow

1. `GET /api/voice-interview/status` reports whether `ELEVENLABS_API_KEY` and
   `ELEVENLABS_AGENT_ID` are set. When they are not, the topbar shows no voice
   control.
2. On start, the browser checks microphone permission, fetches a signed URL,
   and opens the conversation. It passes the per-exercise system prompt and
   first message from `buildVoiceInterviewSession` as overrides. The prompt
   lives in the repo, not in the dashboard, so it is versioned with the
   exercises.
3. Gate phase: the agent asks the questions in index order. The browser buffers
   the learner's final transcript turns. When the agent calls the client tool
   `record_answer({ question_index })`, the buffered speech is appended to that
   answer in the existing `InterviewRecord`, under the same localStorage key.
   The agent asks only the questions the interview level requires. Once each
   of them reaches `MIN_ANSWER_LENGTH`, the record becomes `answered` at that
   level, just as **Start coding** does, and the editor unlocks. The
   tool result tells the agent what happened ("too short", "still open: 1",
   "editor is open"), so the gate rules stay in the browser.
4. Coding phase: when the editor opens, the buffer is cleared and stops
   collecting, so coding talk is never stored as an answer. Code snapshots and
   Run summaries arrive as contextual updates, which do not interrupt the
   agent.
5. Wrap-up: Submit sends the result plus "move on to the Tradeoffs questions",
   clears the buffer and starts collecting again. The prompt holds Tradeoffs
   until then. Tradeoff answers are recorded with the same tool.
6. Leaving the exercise while the session is still loading cancels the start,
   so no session runs without an End button.
7. When a connected session in which the learner spoke at least once ends
   (End, or leaving the exercise), the record gets `voice: true` and its level
   is stamped, as **Start coding** does. Connecting and hanging up without a
   word marks nothing. Recording a spoken answer later keeps the mark. A voice interview counts as the interview: the gate
   stays open, and the panel folds the typed questions away under "Voice
   interview" as optional. It is marked at the end, not the start, so the
   editor stays closed while the opening questions are asked.

## Transcript and silence

- The whole conversation shows in a scrollable, collapsible panel that stays
  after End until closed, so a hint is still readable after the next turn.
  Voice-style tags such as `[calm]` are stripped. Nothing is stored.
- The agent's `turn.turn_timeout` is 60 s with `turn_eagerness: patient`
  (the default is 7 s, which interrupted thinking with "still there?"). The
  prompt also says silence is thinking time.

## Agent setup (one time)

Put `ELEVENLABS_API_KEY=...` in `web/.env` and run `npm run voice:agent`.
`scripts/create-voice-agent.js` creates the `record_answer` client tool and a
private agent with prompt and first-message overrides enabled, then writes
`ELEVENLABS_AGENT_ID` and the key it used to `web/.env` (or the file given with
`-- --env <path>`), so a key passed only through the environment also lands
where the app reads it. It refuses to replace an existing agent ID unless you
pass `-- --force`. Restart the web server afterward.

The same setup by hand in the dashboard:

1. Create an agent. Any voice. The LLM is your choice.
2. **Security**: enable overrides for **System prompt** and **First message**.
   Without this, the session is rejected.
3. **Security**: enable authentication (private agent), so only signed URLs can
   connect.
4. **Tools**: add a client tool:
   - Name: `record_answer`
   - Description: "Record the candidate's spoken answer to one interview
     question. Call it after the candidate finishes answering."
   - Parameter: `question_index`, a number, required. The index of the question
     in the interview list.
   - Wait for response: on.
5. Put the agent ID and an API key in `web/.env` as `ELEVENLABS_AGENT_ID` and
   `ELEVENLABS_API_KEY`, then restart `next dev`.

## Code

- `web/src/lib/voiceInterview.ts`: prompt builder, transcript buffer, answer
  recording, snapshot and result text. Pure functions, unit tested.
- `web/src/lib/server/voiceInterviewProvider.ts`: environment config and the
  signed URL request.
- `web/src/app/api/voice-interview/{status,session}/route.ts`
- `web/src/components/workspace/VoiceInterview.tsx`: topbar control, consent
  panel, live line, store subscriptions. Loaded with `ssr: false`.
- SDK: `@elevenlabs/react` 1.16.0, pinned exactly.

## Verification

- `npx vitest run src/lib/voiceInterview.test.ts src/app/api/voice-interview`
- Manual, with a microphone and a configured agent:
  1. Answer Opening and Deep dive aloud. The answers appear in the panel and
     the editor unlocks.
  2. Write code and pause. The interviewer refers to it.
  3. Run a failing test. The interviewer asks about it.
  4. Submit. The Tradeoffs questions follow.
  5. Remove the environment variables. The voice control disappears, and the
     typed gate works as before.
