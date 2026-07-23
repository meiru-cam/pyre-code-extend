import type { Locale } from '@/lib/i18n';
import type { Problem } from '@/lib/types';

export type AiHelpMode = 'question' | 'review_code';

export interface AiHelpExerciseContext {
  id: string;
  title: string;
  functionName: string;
  description: string;
}

export type AiHelpRequest =
  | {
      mode: 'question';
      query: string;
      exercise: AiHelpExerciseContext;
    }
  | {
      mode: 'review_code';
      query: string;
      exercise: AiHelpExerciseContext;
      learnerCode: string;
    };

interface BuildAiHelpRequestInput {
  problem: Problem;
  mode: AiHelpMode;
  query: string;
  currentCode: string;
  locale: Locale;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === 'object' && !Array.isArray(value);
}

function requireExactKeys(
  value: Record<string, unknown>,
  expected: readonly string[],
  context: string,
) {
  const actual = Object.keys(value);
  const unexpected = actual.filter((key) => !expected.includes(key));
  const missing = expected.filter((key) => !actual.includes(key));
  if (unexpected.length || missing.length) {
    throw new Error(
      `${context} has unknown field(s) or missing field(s): `
      + [...unexpected, ...missing].join(', '),
    );
  }
}

function boundedString(
  value: unknown,
  context: string,
  maxLength: number,
): string {
  if (typeof value !== 'string' || !value.trim() || value.length > maxLength) {
    throw new Error(`${context} must be a non-empty bounded string`);
  }
  return value;
}

export function buildAiHelpRequest({
  problem,
  mode,
  query,
  currentCode,
  locale,
}: BuildAiHelpRequestInput): AiHelpRequest {
  const exercise = {
    id: problem.id,
    title: locale === 'zh' ? problem.titleZh : problem.title,
    functionName: problem.functionName,
    description: locale === 'zh'
      ? problem.descriptionZh
      : problem.descriptionEn,
  };
  if (mode === 'question') return { mode, query, exercise };
  return { mode, query, exercise, learnerCode: currentCode };
}

export function parseAiHelpRequest(value: unknown): AiHelpRequest {
  if (!isRecord(value)) throw new Error('request must be an object');
  if (value.mode !== 'question' && value.mode !== 'review_code') {
    throw new Error('mode must be question or review_code');
  }
  const expected = value.mode === 'question'
    ? ['mode', 'query', 'exercise']
    : ['mode', 'query', 'exercise', 'learnerCode'];
  try {
    requireExactKeys(value, expected, 'request');
  } catch (error) {
    if (value.mode === 'question' && 'learnerCode' in value) {
      throw new Error('learnerCode is not allowed in question mode');
    }
    throw error;
  }
  if (!isRecord(value.exercise)) throw new Error('exercise must be an object');
  requireExactKeys(
    value.exercise,
    ['id', 'title', 'functionName', 'description'],
    'exercise',
  );
  const exercise = {
    id: boundedString(value.exercise.id, 'exercise id', 200),
    title: boundedString(value.exercise.title, 'exercise title', 500),
    functionName: boundedString(
      value.exercise.functionName,
      'exercise function name',
      200,
    ),
    description: boundedString(
      value.exercise.description,
      'exercise description',
      20_000,
    ),
  };
  const query = boundedString(value.query, 'query', 4_000);
  if (value.mode === 'question') return { mode: 'question', query, exercise };
  return {
    mode: 'review_code',
    query,
    exercise,
    learnerCode: boundedString(value.learnerCode, 'learner code', 50_000),
  };
}
