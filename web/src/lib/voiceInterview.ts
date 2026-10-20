import {
  INTERVIEW_STAGE_LABELS,
  MIN_ANSWER_LENGTH,
  missingGatedAnswers,
  requiredQuestionIndices,
  type InterviewLevel,
  type InterviewRecord,
} from '@/lib/interviewAnswer';
import { scanForExternalAiSecrets } from '@/lib/secretBoundary';
import type { InterviewQuestion, Problem, SubmissionResult } from '@/lib/types';

/**
 * Client tool the agent calls once the learner has finished answering one question.
 * scripts/create-voice-agent.js registers the same name with the agent; change both together.
 */
export const RECORD_ANSWER_TOOL = 'record_answer';

/** Longest code snapshot sent to the interviewer; longer code is cut with a marker. */
export const MAX_SNAPSHOT_CHARS = 12_000;

/** Quiet time in the editor before the current code is sent. */
export const SNAPSHOT_IDLE_MS = 3_000;

/** Turns kept in the on-screen transcript. */
export const MAX_TRANSCRIPT_LINES = 200;

const MAX_DESCRIPTION_CHARS = 6_000;
const MAX_FAILURES_LISTED = 5;

export interface VoiceInterviewSession {
  prompt: string;
  firstMessage: string;
}

function numberedQuestions(questions: readonly InterviewQuestion[]): string {
  return questions
    .map((question, index) => `${index}. [${INTERVIEW_STAGE_LABELS[question.stage]}] ${question.question}`)
    .join('\n');
}

/** The agent's system prompt and opening line for one exercise. */
export function buildVoiceInterviewSession(
  problem: Problem,
  questions: readonly InterviewQuestion[],
  gateOpen: boolean,
  level: InterviewLevel = 'hard',
): VoiceInterviewSession {
  const required = requiredQuestionIndices(questions, level);
  const description = problem.descriptionEn.length > MAX_DESCRIPTION_CHARS
    ? `${problem.descriptionEn.slice(0, MAX_DESCRIPTION_CHARS)}\n[description truncated]`
    : problem.descriptionEn;
  const prompt = [
    'You are a calm, senior ML engineer running a live coding interview. The candidate talks while they code.',
    '',
    `Exercise: ${problem.title} (function \`${problem.functionName}\`)`,
    '',
    'Exercise description:',
    description,
    '',
    'Interview questions, by index:',
    numberedQuestions(questions),
    '',
    'How the interview runs:',
    gateOpen
      ? '1. The opening questions are already done. Skip to step 2.'
      : `1. Ask questions ${required.join(', ')} one at a time, in index order; skip the other Opening and Deep dive questions. Let the candidate finish; ask one short follow-up if the answer is thin.`,
    `2. After the candidate finishes answering any question, call the \`${RECORD_ANSWER_TOOL}\` tool with that question_index. Its result tells you whether the answer counted and whether the editor is open. If it says the answer is too short, ask the candidate to say more, then call the tool again.`,
    '3. Once the editor is open, the candidate codes. You receive their code and Run results as context updates. Stay mostly quiet: say little while they make progress, ask "why" about a choice, an edge case or a tensor shape after they finish a thought aloud or a test fails.',
    '4. Only after you are told the candidate has submitted, ask the Tradeoffs questions one at a time and record each answer the same way. If they say they are done before that, ask them to press Submit. After the last answer, thank them and say they can press End.',
    '',
    'Silence means the candidate is thinking or typing. Never ask whether they are still there; wait.',
    'Rules: never dictate code or the full solution; give at most a nudge when the candidate is stuck for a while. Keep every turn under three sentences. Speak the language the candidate speaks.',
  ].join('\n');
  const firstMessage = gateOpen
    ? `Hi. Let's work through ${problem.title}. Talk me through your approach as you code.`
    : `Hi. Let's work through ${problem.title}. ${questions[required[0] ?? 0]?.question ?? 'Tell me how you would approach it.'}`;
  return { prompt, firstMessage };
}

/** A turn as shown on screen: voice-style tags such as "[calm]" are spoken cues, not text. */
export function transcriptText(message: string): string {
  return message.replace(/\[[a-z][a-z ]*\]\s*/gi, '').replace(/\s+/g, ' ').trim();
}

/** Spoken learner turns since the last recorded answer. */
export class TranscriptBuffer {
  private lines: string[] = [];

  push(text: string) {
    const trimmed = text.trim();
    if (trimmed) this.lines.push(trimmed);
  }

  /** Returns the buffered speech and empties the buffer. */
  take(): string {
    const text = this.lines.join(' ');
    this.lines = [];
    return text;
  }

  clear() {
    this.lines = [];
  }
}

export interface RecordAnswerOutcome {
  record: InterviewRecord;
  /** Tool result spoken back to the agent. */
  message: string;
}

/**
 * Adds spoken text to one answer. When every answer the level requires is long enough the
 * record is committed as answered, the same as pressing "Start coding" with them filled in.
 */
export function recordSpokenAnswer(
  questions: readonly InterviewQuestion[],
  record: InterviewRecord,
  questionIndex: unknown,
  spoken: string,
  now = new Date(),
  level: InterviewLevel = 'hard',
): RecordAnswerOutcome {
  const index = typeof questionIndex === 'number' ? questionIndex : Number(questionIndex);
  if (!Number.isInteger(index) || index < 0 || index >= questions.length) {
    return { record, message: `No question has index ${String(questionIndex)}. Use 0 to ${questions.length - 1}.` };
  }
  const existing = record.answers[index] ?? '';
  const answer = [existing.trim(), spoken.trim()].filter(Boolean).join('\n');
  const answers = record.answers.map((value, i) => (i === index ? answer : value));
  const missing = missingGatedAnswers(questions, answers, level);
  // Learn has no gate, so speaking there never commits the record.
  const committing = record.status === 'draft' && level !== 'learn' && missing.length === 0;
  const status = committing ? 'answered' : record.status;
  const lockedLevel = committing ? level : record.level;
  const next: InterviewRecord = {
    status, answers, updatedAt: now.toISOString(), ...(lockedLevel ? { level: lockedLevel } : {}),
    // A voice mark from an earlier session keeps the gate open.
    ...(record.voice ? { voice: true as const } : {}),
  };

  const gated = requiredQuestionIndices(questions, level).includes(index);
  if (gated && answer.length < MIN_ANSWER_LENGTH) {
    return { record: next, message: `Answer ${index} is too short to count. Ask the candidate to say more.` };
  }
  if (status !== 'draft' || level === 'learn') {
    return {
      record: next,
      message: committing
        ? `Answer ${index} recorded. The editor is open now; let the candidate start coding.`
        : `Answer ${index} recorded.`,
    };
  }
  return { record: next, message: `Answer ${index} recorded. Still open: questions ${missing.join(', ')}.` };
}

/** Context update for the current code, or null when nothing new should be sent. */
export function codeSnapshotUpdate(code: string, lastSent: string | null): string | null {
  if (code === lastSent) return null;
  if (scanForExternalAiSecrets(code).blocked) {
    return 'Code snapshot withheld: the editor appears to contain a credential.';
  }
  const body = code.length > MAX_SNAPSHOT_CHARS
    ? `${code.slice(0, MAX_SNAPSHOT_CHARS)}\n# [snapshot truncated]`
    : code;
  return `Candidate's current code:\n\`\`\`python\n${body}\n\`\`\``;
}

/** Short context update for a Run or Submit result. */
export function resultUpdate(result: SubmissionResult, kind: 'run' | 'submit'): string {
  const label = kind === 'submit' ? 'Submitted' : 'Ran tests';
  const next = kind === 'submit' ? ['The candidate has submitted; move on to the Tradeoffs questions.'] : [];
  const failures = (result.results ?? []).filter((test) => !test.passed);
  const lines = result.error
    ? [`${label}: error before tests ran: ${result.error.slice(0, 300)}`]
    : [`${label}: ${result.passed}/${result.total} passed.`];
  if (!result.error) {
    for (const test of failures.slice(0, MAX_FAILURES_LISTED)) {
      const where = test.errorLine ? ` (line ${test.errorLine})` : '';
      const why = (test.behavior ?? test.error ?? '').split('\n')[0].slice(0, 200);
      lines.push(`- ${test.name} failed${where}${why ? `: ${why}` : ''}`);
    }
    if (failures.length > MAX_FAILURES_LISTED) lines.push(`- and ${failures.length - MAX_FAILURES_LISTED} more`);
  }
  // Error text can echo the learner's code, so it passes the same secret check as a snapshot.
  if (scanForExternalAiSecrets(lines).blocked) {
    const counts = result.error ? 'error before tests ran' : `${result.passed}/${result.total} passed`;
    return [`${label}: ${counts}. Details withheld: they appear to contain a credential.`, ...next].join('\n');
  }
  return [...lines, ...next].join('\n');
}
