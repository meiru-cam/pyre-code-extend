import type { HintLevel, Problem, Test, TestResult } from '@/lib/types';

export function getHintLevels(problem: Problem): HintLevel[] {
  if (!problem.hints || problem.hints.length === 0) return [];
  return [...problem.hints].sort((a, b) => a.level - b.level);
}

export function visibleTestIndices(tests: Test[]): number[] {
  return tests
    .map((test, index) => (test.visibility === 'unshown' ? -1 : index))
    .filter((index) => index >= 0);
}

export function resultTestIndex(result: TestResult, fallback: number): number {
  return result.testIndex ?? fallback;
}
