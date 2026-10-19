import type { InterviewQuestion, InterviewStage, Problem } from '@/lib/types';

const ANSWER_SCHEMA_VERSION = 2;
const ANSWER_KEY_PREFIX = 'pyre-code-interview-answer:v2:';

/** Shortest trimmed answer accepted per gated question, so a keystroke cannot unlock. */
export const MIN_ANSWER_LENGTH = 20;

/** Questions in these stages can gate help; tradeoffs come after coding. */
export const GATED_STAGES: readonly InterviewStage[] = ['concept', 'deep_dive'];

/**
 * How much interview the gate asks for, browser-wide. `learn` has no gate and opens the
 * explanations first; the others require that many gated questions, `hard` all of them.
 */
export type InterviewLevel = 'learn' | 'easy' | 'medium' | 'hard';

export const INTERVIEW_LEVELS: readonly InterviewLevel[] = ['learn', 'easy', 'medium', 'hard'];

export const DEFAULT_INTERVIEW_LEVEL: InterviewLevel = 'easy';

const REQUIRED_QUESTION_COUNT: Record<InterviewLevel, number> = {
  learn: 0,
  easy: 2,
  medium: 4,
  hard: Infinity,
};

/**
 * Indices of the gated questions a level requires, in question order. Opening and deep-dive
 * questions are taken in turn, so even easy asks one of each.
 */
export function requiredQuestionIndices(
  questions: readonly InterviewQuestion[],
  level: InterviewLevel,
): number[] {
  const byStage = GATED_STAGES.map((stage) => questions
    .map((question, index) => ({ question, index }))
    .filter(({ question }) => question.stage === stage)
    .map(({ index }) => index));
  const interleaved: number[] = [];
  for (let round = 0; byStage.some((indices) => round < indices.length); round += 1) {
    for (const indices of byStage) {
      if (round < indices.length) interleaved.push(indices[round]);
    }
  }
  return interleaved.slice(0, REQUIRED_QUESTION_COUNT[level]).sort((a, b) => a - b);
}

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
  /** The level the answers were locked in at; records from before levels count as hard. */
  level?: InterviewLevel;
}

const STATUSES: readonly InterviewStatus[] = ['draft', 'answered', 'skipped'];

/** Hints, reference tradeoffs, solution, AI help and the editor open only after this. */
export function isInterviewUnlocked(record: InterviewRecord | null): boolean {
  return record?.status === 'answered' || record?.status === 'skipped';
}

export function emptyInterviewRecord(questions: readonly InterviewQuestion[]): InterviewRecord {
  return { status: 'draft', answers: questions.map(() => ''), updatedAt: '' };
}

/** Indices of questions the level requires whose answer is still too short. */
export function missingGatedAnswers(
  questions: readonly InterviewQuestion[],
  answers: readonly string[],
  level: InterviewLevel = 'hard',
): number[] {
  return requiredQuestionIndices(questions, level)
    .filter((index) => (answers[index] ?? '').trim().length < MIN_ANSWER_LENGTH);
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
      level?: unknown;
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
    const level = INTERVIEW_LEVELS.includes(parsed.level as InterviewLevel)
      ? parsed.level as InterviewLevel
      : undefined;
    return {
      // An answer that no longer covers the questions its level required goes back to draft.
      // Changing the level later never relocks an exercise already answered.
      status: status === 'answered' && missingGatedAnswers(questions, answers, level).length > 0
        ? 'draft'
        : status,
      answers,
      updatedAt: typeof parsed.updatedAt === 'string' ? parsed.updatedAt : '',
      ...(level ? { level } : {}),
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

const INTERVIEW_LEVEL_KEY = 'pyre-interview-level';
/** Practice mode was the on/off predecessor of learn; an old "on" carries over. */
const LEGACY_PRACTICE_MODE_KEY = 'pyre-practice-mode';

export function loadInterviewLevel(): InterviewLevel {
  if (typeof window === 'undefined') return DEFAULT_INTERVIEW_LEVEL;
  try {
    const stored = window.localStorage.getItem(INTERVIEW_LEVEL_KEY);
    if (INTERVIEW_LEVELS.includes(stored as InterviewLevel)) return stored as InterviewLevel;
    return window.localStorage.getItem(LEGACY_PRACTICE_MODE_KEY) === '1'
      ? 'learn'
      : DEFAULT_INTERVIEW_LEVEL;
  } catch {
    return DEFAULT_INTERVIEW_LEVEL;
  }
}

export function saveInterviewLevel(level: InterviewLevel) {
  if (typeof window === 'undefined') return;
  try {
    window.localStorage.setItem(INTERVIEW_LEVEL_KEY, level);
  } catch {
    // Storage is best-effort; the level still applies for this session.
  }
}
