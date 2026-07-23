import type {
  DesignNoteFieldName,
  DesignNoteFields,
  Problem,
} from '@/lib/types';

const DRAFT_SCHEMA_VERSION = 1;
const DRAFT_KEY_PREFIX = 'pyre-code-design-note:v1:';

type DesignNoteRubric = readonly NonNullable<Problem['designNoteRubric']>[number][];

function draftKey(taskId: string, contractVersion: number) {
  return `${DRAFT_KEY_PREFIX}${taskId}:${contractVersion}`;
}

export function emptyDesignNoteFields(
  rubric: readonly { field: DesignNoteFieldName }[],
): DesignNoteFields {
  return Object.fromEntries(
    rubric.map(({ field }) => [field, '']),
  ) as DesignNoteFields;
}

export function hasDesignNoteContent(
  fields: Partial<DesignNoteFields>,
): boolean {
  return Object.values(fields).some(
    (value) => typeof value === 'string' && value.trim().length > 0,
  );
}

export function loadDesignNoteDraft(
  taskId: string,
  contractVersion: number,
  rubric: DesignNoteRubric,
): DesignNoteFields | null {
  if (typeof window === 'undefined') return null;
  try {
    const raw = window.localStorage.getItem(draftKey(taskId, contractVersion));
    if (!raw) return null;
    const parsed = JSON.parse(raw) as {
      schemaVersion?: number;
      fields?: Record<string, unknown>;
    };
    if (parsed.schemaVersion !== DRAFT_SCHEMA_VERSION || !parsed.fields) {
      return null;
    }
    const expected = rubric.map(({ field }) => field);
    if (
      Object.keys(parsed.fields).length !== expected.length
      || expected.some((field) => typeof parsed.fields?.[field] !== 'string')
    ) {
      return null;
    }
    return Object.fromEntries(
      expected.map((field) => [field, parsed.fields?.[field] as string]),
    ) as DesignNoteFields;
  } catch {
    return null;
  }
}

export function saveDesignNoteDraft(
  taskId: string,
  contractVersion: number,
  fields: DesignNoteFields,
) {
  if (typeof window === 'undefined') return;
  try {
    window.localStorage.setItem(
      draftKey(taskId, contractVersion),
      JSON.stringify({ schemaVersion: DRAFT_SCHEMA_VERSION, fields }),
    );
  } catch {
    // A draft is best-effort when storage is unavailable or full.
  }
}
