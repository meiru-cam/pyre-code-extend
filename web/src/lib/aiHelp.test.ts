import { describe, expect, it } from 'vitest';

import { buildAiHelpRequest, parseAiHelpRequest } from '@/lib/aiHelp';
import type { Problem } from '@/lib/types';

const problem = {
  id: 'qk_norm',
  title: 'QK Norm',
  titleZh: 'QK Norm ZH',
  difficulty: 'Medium',
  functionName: 'qk_norm',
  hint: 'HINT_MUST_NOT_LEAK',
  hintZh: '',
  descriptionEn: 'Normalize queries and keys.',
  descriptionZh: 'ZH_DESCRIPTION',
  tests: [{ name: 'hidden', code: 'UNSHOWN_TEST', visibility: 'unshown' }],
  starterCode: 'STARTER_MUST_NOT_LEAK',
  solutionCode: 'SOLUTION_MUST_NOT_LEAK',
} as Problem & { starterCode: string; solutionCode: string };

describe('AI Help request boundary', () => {
  it('question mode sends the learner query and minimal context without code', () => {
    const request = buildAiHelpRequest({
      problem,
      mode: 'question',
      query: 'How do I use softmax from F?',
      currentCode: 'CURRENT_CODE_MUST_NOT_LEAK',
      locale: 'en',
    });
    expect(request).toEqual({
      mode: 'question',
      query: 'How do I use softmax from F?',
      exercise: {
        id: 'qk_norm',
        title: 'QK Norm',
        functionName: 'qk_norm',
        description: 'Normalize queries and keys.',
      },
    });
    const serialized = JSON.stringify(request);
    for (const forbidden of [
      'CURRENT_CODE_MUST_NOT_LEAK',
      'STARTER_MUST_NOT_LEAK',
      'SOLUTION_MUST_NOT_LEAK',
      'UNSHOWN_TEST',
      'HINT_MUST_NOT_LEAK',
      'ZH_DESCRIPTION',
    ]) {
      expect(serialized).not.toContain(forbidden);
    }
  });

  it('review-code mode includes code only after that mode is explicit', () => {
    const request = buildAiHelpRequest({
      problem,
      mode: 'review_code',
      query: 'Why is the shape wrong?',
      currentCode: 'def qk_norm(): pass',
      locale: 'en',
    });
    expect(request.mode).toBe('review_code');
    if (request.mode !== 'review_code') throw new Error('expected review mode');
    expect(request.learnerCode).toBe('def qk_norm(): pass');
  });

  it('rejects over-sharing and code attached to question mode', () => {
    const valid = buildAiHelpRequest({
      problem,
      mode: 'question',
      query: 'Explain the reduction dimension.',
      currentCode: 'not sent',
      locale: 'en',
    });
    expect(() => parseAiHelpRequest({ ...valid, solutionCode: 'x' })).toThrow(
      /unknown field/i,
    );
    expect(() => parseAiHelpRequest({ ...valid, learnerCode: 'x' })).toThrow(
      /question mode/i,
    );
    expect(() => parseAiHelpRequest({
      ...valid,
      exercise: { ...valid.exercise, sampleTests: [] },
    })).toThrow(/unknown field/i);
  });
});
