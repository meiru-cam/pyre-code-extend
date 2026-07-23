// @vitest-environment jsdom

import { beforeEach, describe, expect, it } from 'vitest';

import {
  emptyDesignNoteFields,
  hasDesignNoteContent,
  loadDesignNoteDraft,
  saveDesignNoteDraft,
} from '@/lib/designNoteDraft';

const rubric = [
  { field: 'api_boundaries', label: 'API boundaries and class responsibilities' },
  { field: 'state_ownership', label: 'State and ownership' },
] as const;

describe('design-note local drafts', () => {
  beforeEach(() => window.localStorage.clear());

  it('restores each structured field independently by task and contract version', () => {
    const fields = {
      ...emptyDesignNoteFields(rubric),
      api_boundaries: 'Gateway owns validation.',
      state_ownership: 'Scheduler owns run state.',
    };
    saveDesignNoteDraft('tool_registry', 1, fields);

    expect(loadDesignNoteDraft('tool_registry', 1, rubric)).toEqual(fields);
    expect(loadDesignNoteDraft('tool_registry', 2, rubric)).toBeNull();
    expect(loadDesignNoteDraft('budgeted_agent_loop', 1, rubric)).toBeNull();
  });

  it('ignores malformed or schema-incompatible stored values', () => {
    window.localStorage.setItem(
      'pyre-code-design-note:v1:tool_registry:1',
      JSON.stringify({
        schemaVersion: 999,
        fields: { api_boundaries: 'stale', state_ownership: 'stale' },
      }),
    );
    expect(loadDesignNoteDraft('tool_registry', 1, rubric)).toBeNull();
  });

  it('treats whitespace-only drafts as empty', () => {
    expect(hasDesignNoteContent({
      api_boundaries: ' ',
      state_ownership: '\n',
    })).toBe(false);
    expect(hasDesignNoteContent({
      api_boundaries: '',
      state_ownership: 'A real tradeoff',
    })).toBe(true);
  });
});
