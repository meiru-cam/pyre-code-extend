import { describe, expect, it } from 'vitest';

import { hasProblemInformation, sourceHref, sourceLocator } from '@/lib/problemInformation';
import type { Problem } from '@/lib/types';

const problem: Problem = {
  id: 'qk_norm',
  title: 'QK Norm',
  titleZh: 'QK Norm',
  difficulty: 'Medium',
  functionName: 'qk_norm',
  hint: '',
  hintZh: '',
  descriptionEn: 'Description',
  descriptionZh: 'Description',
  tests: [],
  modelConnections: ['Qwen3 applies QK norm before RoPE.'],
  proConAnalysis: { pros: ['Stable logits'], cons: ['Extra reductions'] },
  sources: [
    {
      kind: 'code',
      url: 'https://github.com/example/model',
      commit: 'a'.repeat(40),
      path: 'model.py',
      symbol: 'Attention.forward',
      license: 'Apache-2.0',
      adapted: 'Mask semantics.',
      simplifications: 'No cache.',
    },
    {
      kind: 'paper',
      url: 'https://arxiv.org/abs/1234.5678',
      section: '2.1 Architecture',
    },
  ],
};

describe('ProblemInformation', () => {
  it('detects model connections and pros/cons as renderable information', () => {
    expect(hasProblemInformation(problem)).toBe(true);
    expect(hasProblemInformation({ ...problem, modelConnections: undefined, proConAnalysis: undefined, sources: undefined })).toBe(false);
  });

  it('builds immutable code links and precise paper locators', () => {
    expect(sourceHref(problem.sources![0])).toBe(
      'https://github.com/example/model/blob/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/model.py',
    );
    expect(sourceLocator(problem.sources![0])).toBe('model.py · Attention.forward');
    expect(sourceLocator(problem.sources![1])).toBe('2.1 Architecture');
  });
});
