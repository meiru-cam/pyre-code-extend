import { describe, expect, it } from 'vitest';

import {
  buildDesignReviewRequest,
  parseDesignReviewRequest,
} from '@/lib/designReview';
import type { DesignNoteFields, Problem } from '@/lib/types';

const note: DesignNoteFields = {
  api_boundaries: 'Gateway validates input.',
  state_ownership: 'Scheduler owns state.',
  failure_recovery: 'Checkpoint before effects.',
  backpressure_concurrency: 'Bound the ready queue.',
  durability_idempotency: 'Persist idempotency keys.',
  observability: 'Trace terminal outcomes.',
  security: 'Use exact capabilities.',
  tradeoffs: 'Durability adds latency.',
};

const problem = {
  id: 'tool_registry',
  title: 'Validated Tool Registry',
  titleZh: 'Validated Tool Registry',
  difficulty: 'Medium',
  functionName: 'ToolRegistry',
  hint: 'HINT_MUST_NOT_LEAK',
  hintZh: '',
  descriptionEn: 'Implement a validated registry.',
  descriptionZh: 'DESCRIPTION_ZH_MUST_NOT_LEAK',
  tests: [
    {
      name: 'hidden',
      visibility: 'unshown',
      code: 'UNSHOWN_TEST_MUST_NOT_LEAK',
    },
  ],
  designNoteRubric: [
    { field: 'api_boundaries', label: 'API boundaries' },
    { field: 'state_ownership', label: 'State and ownership' },
  ],
  starterCode: 'LEARNER_CODE_MUST_NOT_LEAK',
  solutionCode: 'REFERENCE_SOLUTION_MUST_NOT_LEAK',
} as Problem & { starterCode: string; solutionCode: string };

describe('design-review request boundary', () => {
  it('constructs an explicit allowlist with only visible exercise context and note', () => {
    const request = buildDesignReviewRequest(problem, note, 'en');
    expect(request).toEqual({
      exercise: {
        id: 'tool_registry',
        title: 'Validated Tool Registry',
        description: 'Implement a validated registry.',
        rubric: [
          { field: 'api_boundaries', label: 'API boundaries' },
          { field: 'state_ownership', label: 'State and ownership' },
        ],
      },
      note,
    });

    const serialized = JSON.stringify(request);
    for (const forbidden of [
      'LEARNER_CODE_MUST_NOT_LEAK',
      'REFERENCE_SOLUTION_MUST_NOT_LEAK',
      'UNSHOWN_TEST_MUST_NOT_LEAK',
      'HINT_MUST_NOT_LEAK',
      'DESCRIPTION_ZH_MUST_NOT_LEAK',
      'submission',
      'environment',
    ]) {
      expect(serialized).not.toContain(forbidden);
    }
  });

  it('rejects unknown top-level, exercise, rubric, and note fields', () => {
    const valid = buildDesignReviewRequest(problem, note, 'en');
    expect(() => parseDesignReviewRequest({ ...valid, learnerCode: 'x' })).toThrow(
      /unknown field/i,
    );
    expect(() => parseDesignReviewRequest({
      ...valid,
      exercise: { ...valid.exercise, solution: 'x' },
    })).toThrow(/unknown field/i);
    expect(() => parseDesignReviewRequest({
      ...valid,
      exercise: {
        ...valid.exercise,
        rubric: [...valid.exercise.rubric, { field: 'unknown', label: 'x' }],
      },
    })).toThrow(/rubric/i);
    expect(() => parseDesignReviewRequest({
      ...valid,
      note: { ...valid.note, unknown: 'x' },
    })).toThrow(/note/i);
  });
});
