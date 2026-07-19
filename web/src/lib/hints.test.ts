import { describe, expect, it } from 'vitest';
import { getHintLevels, resultTestIndex, visibleTestIndices } from '@/lib/hints';
import type { Problem, Test, TestResult } from '@/lib/types';

const base: Problem = {
  id: 'x',
  title: 'X',
  titleZh: 'X',
  difficulty: 'Easy',
  functionName: 'f',
  hint: 'legacy',
  hintZh: '',
  descriptionEn: '',
  descriptionZh: '',
  tests: [],
};

describe('getHintLevels', () => {
  it('returns empty for legacy problems', () => {
    expect(getHintLevels(base)).toEqual([]);
  });

  it('returns hints sorted by level', () => {
    const problem: Problem = {
      ...base,
      hints: [
        { level: 2, kind: 'analysis', content: 'b' },
        { level: 1, kind: 'questions', content: 'a' },
      ],
    };
    expect(getHintLevels(problem).map((hint) => hint.level)).toEqual([1, 2]);
  });
});

describe('visibleTestIndices', () => {
  it('keeps original indices and drops unshown cases', () => {
    const tests: Test[] = [
      { name: 'a', code: 'x' },
      { name: 'b', visibility: 'unshown', behavior: 'tensor.shape' },
      { name: 'c', code: 'y' },
    ];
    expect(visibleTestIndices(tests)).toEqual([0, 2]);
  });
});

describe('resultTestIndex', () => {
  it('uses the original task-test index returned by the grader', () => {
    const result = {
      name: 'visible two',
      passed: true,
      execTimeMs: 1,
      testIndex: 2,
    } satisfies TestResult;
    expect(resultTestIndex(result, 0)).toBe(2);
  });

  it('falls back for responses produced before testIndex was added', () => {
    const legacy = { name: 'legacy', passed: true, execTimeMs: 1 } satisfies TestResult;
    expect(resultTestIndex(legacy, 1)).toBe(1);
  });
});
