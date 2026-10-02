const DRAFT_SCHEMA_VERSION = 1;
const DRAFT_KEY_PREFIX = 'pyre-code-prep:v1:';

/** A learner's practice answer to one prep item, kept only in this browser. */
export interface PrepDraft {
  answer: string;
  /** Rubric points the learner has ticked as covered. */
  covered: string[];
}

export const EMPTY_PREP_DRAFT: PrepDraft = { answer: '', covered: [] };

export function hasPrepDraftContent(draft: PrepDraft | null): boolean {
  return draft !== null && draft.answer.trim().length > 0;
}

export function loadPrepDraft(itemId: string): PrepDraft | null {
  if (typeof window === 'undefined') return null;
  try {
    const raw = window.localStorage.getItem(`${DRAFT_KEY_PREFIX}${itemId}`);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as { schemaVersion?: number } & Partial<PrepDraft>;
    if (
      parsed.schemaVersion !== DRAFT_SCHEMA_VERSION
      || typeof parsed.answer !== 'string'
      || !Array.isArray(parsed.covered)
      || parsed.covered.some((point) => typeof point !== 'string')
    ) {
      return null;
    }
    return { answer: parsed.answer, covered: parsed.covered };
  } catch {
    return null;
  }
}

export function savePrepDraft(itemId: string, draft: PrepDraft) {
  if (typeof window === 'undefined') return;
  try {
    window.localStorage.setItem(
      `${DRAFT_KEY_PREFIX}${itemId}`,
      JSON.stringify({ schemaVersion: DRAFT_SCHEMA_VERSION, ...draft }),
    );
  } catch {
    // A draft is best-effort when storage is unavailable or full.
  }
}
