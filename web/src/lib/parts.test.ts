// @vitest-environment jsdom

import { beforeEach, describe, expect, it } from 'vitest';

import {
  highestPassedPart,
  loadUnlockedPart,
  resultThroughPart,
  saveUnlockedPart,
  unlockedAfter,
} from '@/lib/parts';
import type { SubmissionResult, Test } from '@/lib/types';

const tests: Test[] = [
  { name: 'a', part: 1 },
  { name: 'b', part: 1, visibility: 'unshown' },
  { name: 'c', part: 2 },
  { name: 'd', part: 3 },
];

function result(passed: Record<number, boolean>): SubmissionResult {
  const results = Object.entries(passed).map(([index, ok]) => ({
    name: tests[Number(index)].name, passed: ok, execTimeMs: 0, testIndex: Number(index),
  }));
  return { passed: 0, total: results.length, allPassed: false, results, totalTimeMs: 0 };
}

describe('part grading', () => {
  it('shows only the cases of unlocked parts and recounts them', () => {
    const shown = resultThroughPart(tests, result({ 0: true, 1: true, 2: false, 3: false }), 2);
    expect(shown.results.map((r) => r.testIndex)).toEqual([0, 1, 2]);
    expect([shown.passed, shown.total]).toEqual([2, 3]);
  });

  it('unlocks straight to the last part when a full solution passes everything', () => {
    const all = result({ 0: true, 1: true, 2: true, 3: true });
    expect(unlockedAfter(tests, all, 3, 1)).toBe(3);
    expect(resultThroughPart(tests, all, 3).results).toHaveLength(4);
  });

  it('counts a part as passed only when all its cases passed', () => {
    expect(highestPassedPart(tests, result({ 0: true, 1: false }), 3)).toBe(0);
    expect(highestPassedPart(tests, result({ 0: true, 1: true }), 3)).toBe(1);
    expect(highestPassedPart(tests, result({ 0: true, 1: true, 2: true, 3: true }), 3)).toBe(3);
  });

  it('stops at the first failing part even if a later one passed', () => {
    expect(highestPassedPart(tests, result({ 0: true, 1: true, 2: false, 3: true }), 3)).toBe(1);
  });

  it('unlocks one part past the highest passing part and never locks again', () => {
    expect(unlockedAfter(tests, result({ 0: true, 1: true }), 3, 1)).toBe(2);
    expect(unlockedAfter(tests, result({ 0: true, 1: true, 2: true, 3: true }), 3, 1)).toBe(3);
    expect(unlockedAfter(tests, result({ 0: false }), 3, 2)).toBe(2);
  });
});

describe('unlock storage', () => {
  beforeEach(() => window.localStorage.clear());

  it('restores per problem and version and rejects out-of-range values', () => {
    saveUnlockedPart('time_map', 1, 3);
    expect(loadUnlockedPart('time_map', 1, 4)).toBe(3);
    expect(loadUnlockedPart('time_map', 2, 4)).toBe(1);
    expect(loadUnlockedPart('time_map', 1, 2)).toBe(1);
  });
});
