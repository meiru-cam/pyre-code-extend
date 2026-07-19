import { describe, expect, it } from 'vitest';
import { getSampleTests } from '@/lib/problemContext';
import type { Problem } from '@/lib/types';

const problem: Problem = {
  id: 'x',
  title: 'X',
  titleZh: 'X',
  difficulty: 'Easy',
  functionName: 'f',
  hint: '',
  hintZh: '',
  descriptionEn: '',
  descriptionZh: '',
  tests: [
    { name: 'unshown', visibility: 'unshown', behavior: 'tensor.shape' },
    { name: 'visible one', code: 'assert {fn}(1) == 1' },
    { name: 'visible two', code: 'assert {fn}(2) == 2' },
  ],
};

describe('getSampleTests', () => {
  it('filters unshown cases before applying the sample limit', () => {
    expect(getSampleTests(problem, 1)).toEqual([
      { name: 'visible one', code: 'assert f(1) == 1' },
    ]);
  });
});
