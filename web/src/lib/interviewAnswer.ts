import type { InterviewQuestion, InterviewStage, Problem } from '@/lib/types';

const ANSWER_SCHEMA_VERSION = 2;
const ANSWER_KEY_PREFIX = 'pyre-code-interview-answer:v2:';

/** Shortest trimmed answer accepted per gated question, so a keystroke cannot unlock. */
export const MIN_ANSWER_LENGTH = 20;

/** Answering every question in these stages unlocks help; tradeoffs come after coding. */
export const GATED_STAGES: readonly InterviewStage[] = ['concept', 'deep_dive'];

export const INTERVIEW_STAGE_LABELS: Record<InterviewStage, string> = {
  concept: 'Opening',
  deep_dive: 'Deep dive',
  tradeoffs: 'Tradeoffs',
};

/** Used until a task authors its own questions. */
export const FALLBACK_INTERVIEW_QUESTIONS: readonly InterviewQuestion[] = [
  {
    stage: 'concept',
    question: 'What is this, what problem does it solve, and where is it used?',
    hint: 'Name the input and output, the failure of the simpler approach it fixes, and one model or system that uses it.',
  },
  {
    stage: 'deep_dive',
    question: 'How would you implement it? Walk through the steps, tensor shapes, and edge cases.',
    hint: 'Go step by step from input to output, writing the shape after each step, then name one edge case such as empty input or masking.',
  },
  {
    stage: 'tradeoffs',
    question: 'What does it buy you, what does it cost, and what would you compare it against?',
    hint: 'Pick one alternative and compare on quality, compute or memory, and implementation complexity.',
  },
];

export function interviewQuestionsFor(problem: Problem): readonly InterviewQuestion[] {
  return problem.interviewQuestions?.length
    ? problem.interviewQuestions
    : FALLBACK_INTERVIEW_QUESTIONS;
}

/**
 * `draft`: typed but not yet committed; help stays locked.
 * `answered`: the learner answered the gated questions before seeing any help.
 * `skipped`: the learner unlocked help without answering; kept so it stays visible.
 */
export type InterviewStatus = 'draft' | 'answered' | 'skipped';

export interface InterviewRecord {
  status: InterviewStatus;
  /** One answer per question, by index into the task's question list. */
  answers: string[];
  updatedAt: string;
}

const STATUSES: readonly InterviewStatus[] = ['draft', 'answered', 'skipped'];

/** Hints, reference tradeoffs, solution, AI help and the editor open only after this. */
export function isInterviewUnlocked(record: InterviewRecord | null): boolean {
  return record?.status === 'answered' || record?.status === 'skipped';
}

export function emptyInterviewRecord(questions: readonly InterviewQuestion[]): InterviewRecord {
  return { status: 'draft', answers: questions.map(() => ''), updatedAt: '' };
}

/** Indices of gated questions whose answer is still too short. */
export function missingGatedAnswers(
  questions: readonly InterviewQuestion[],
  answers: readonly string[],
): number[] {
  return questions
    .map((question, index) => ({ question, index }))
    .filter(({ question, index }) => (
      GATED_STAGES.includes(question.stage)
      && (answers[index] ?? '').trim().length < MIN_ANSWER_LENGTH
    ))
    .map(({ index }) => index);
}

function answerKey(taskId: string, contractVersion: number) {
  return `${ANSWER_KEY_PREFIX}${taskId}:${contractVersion}`;
}

export function loadInterviewRecord(
  taskId: string,
  contractVersion: number,
  questions: readonly InterviewQuestion[],
): InterviewRecord | null {
  if (typeof window === 'undefined') return null;
  try {
    const raw = window.localStorage.getItem(answerKey(taskId, contractVersion));
    if (!raw) return null;
    const parsed = JSON.parse(raw) as {
      schemaVersion?: number;
      status?: unknown;
      answers?: unknown;
      updatedAt?: unknown;
    };
    if (
      parsed.schemaVersion !== ANSWER_SCHEMA_VERSION
      || !STATUSES.includes(parsed.status as InterviewStatus)
      || !Array.isArray(parsed.answers)
      || parsed.answers.some((answer) => typeof answer !== 'string')
    ) {
      return null;
    }
    // Authored questions can be added or removed later; keep answers aligned by index.
    const stored = parsed.answers as string[];
    const answers = questions.map((_, index) => stored[index] ?? '');
    const status = parsed.status as InterviewStatus;
    return {
      // An answer that no longer covers the gated questions goes back to draft.
      status: status === 'answered' && missingGatedAnswers(questions, answers).length > 0
        ? 'draft'
        : status,
      answers,
      updatedAt: typeof parsed.updatedAt === 'string' ? parsed.updatedAt : '',
    };
  } catch {
    return null;
  }
}

export function saveInterviewRecord(
  taskId: string,
  contractVersion: number,
  record: InterviewRecord,
) {
  if (typeof window === 'undefined') return;
  try {
    window.localStorage.setItem(
      answerKey(taskId, contractVersion),
      JSON.stringify({ schemaVersion: ANSWER_SCHEMA_VERSION, ...record }),
    );
  } catch {
    // Storage is best-effort; the gate stays unlocked for this session regardless.
  }
}
