import type {
  DesignNoteFieldName,
  DesignNoteFields,
  Problem,
} from '@/lib/types';
import type { Locale } from '@/lib/i18n';

const DESIGN_NOTE_FIELDS: readonly DesignNoteFieldName[] = [
  'api_boundaries',
  'state_ownership',
  'failure_recovery',
  'backpressure_concurrency',
  'durability_idempotency',
  'observability',
  'security',
  'tradeoffs',
];

export interface DesignReviewRequest {
  exercise: {
    id: string;
    title: string;
    description: string;
    rubric: Array<{ field: DesignNoteFieldName; label: string }>;
  };
  note: DesignNoteFields;
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
  const unknown = actual.filter((key) => !expected.includes(key));
  const missing = expected.filter((key) => !actual.includes(key));
  if (unknown.length || missing.length) {
    throw new Error(
      `${context} has unknown field(s) or missing field(s): `
      + [...unknown, ...missing].join(', '),
    );
  }
}

function requireBoundedString(
  value: unknown,
  context: string,
  maxLength: number,
): string {
  if (typeof value !== 'string' || !value.trim() || value.length > maxLength) {
    throw new Error(`${context} must be a non-empty bounded string`);
  }
  return value;
}

export function buildDesignReviewRequest(
  problem: Problem,
  note: DesignNoteFields,
  locale: Locale,
): DesignReviewRequest {
  return {
    exercise: {
      id: problem.id,
      title: locale === 'zh' ? problem.titleZh : problem.title,
      description: locale === 'zh' ? problem.descriptionZh : problem.descriptionEn,
      rubric: (problem.designNoteRubric ?? []).map(({ field, label }) => ({
        field,
        label,
      })),
    },
    note: Object.fromEntries(
      DESIGN_NOTE_FIELDS.map((field) => [field, note[field] ?? '']),
    ) as DesignNoteFields,
  };
}

export function parseDesignReviewRequest(value: unknown): DesignReviewRequest {
  if (!isRecord(value)) throw new Error('request must be an object');
  requireExactKeys(value, ['exercise', 'note'], 'request');
  if (!isRecord(value.exercise)) throw new Error('exercise must be an object');
  requireExactKeys(
    value.exercise,
    ['id', 'title', 'description', 'rubric'],
    'exercise',
  );
  const exercise = value.exercise;
  if (!Array.isArray(exercise.rubric) || exercise.rubric.length > 8) {
    throw new Error('rubric must be an array of at most eight fields');
  }
  const seen = new Set<string>();
  const rubric = exercise.rubric.map((entry, index) => {
    if (!isRecord(entry)) throw new Error(`rubric ${index} must be an object`);
    requireExactKeys(entry, ['field', 'label'], `rubric ${index}`);
    if (
      typeof entry.field !== 'string'
      || !DESIGN_NOTE_FIELDS.includes(entry.field as DesignNoteFieldName)
      || seen.has(entry.field)
    ) {
      throw new Error(`rubric ${index} has an invalid field`);
    }
    seen.add(entry.field);
    return {
      field: entry.field as DesignNoteFieldName,
      label: requireBoundedString(entry.label, `rubric ${index} label`, 200),
    };
  });

  if (!isRecord(value.note)) throw new Error('note must be an object');
  const noteRecord = value.note;
  requireExactKeys(noteRecord, DESIGN_NOTE_FIELDS, 'note');
  let totalNoteLength = 0;
  const note = Object.fromEntries(DESIGN_NOTE_FIELDS.map((field) => {
    const fieldValue = noteRecord[field];
    if (typeof fieldValue !== 'string' || fieldValue.length > 4_000) {
      throw new Error(`note field ${field} must be a bounded string`);
    }
    totalNoteLength += fieldValue.length;
    return [field, fieldValue];
  })) as DesignNoteFields;
  if (totalNoteLength > 16_000) throw new Error('note is too large');

  return {
    exercise: {
      id: requireBoundedString(exercise.id, 'exercise id', 200),
      title: requireBoundedString(exercise.title, 'exercise title', 500),
      description: requireBoundedString(
        exercise.description,
        'exercise description',
        20_000,
      ),
      rubric,
    },
    note,
  };
}
