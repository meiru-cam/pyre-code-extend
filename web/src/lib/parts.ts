import type { Problem, SubmissionResult, Test } from '@/lib/types';

const UNLOCK_KEY_PREFIX = 'pyre-code-parts:v1:';

export function partCount(problem: Pick<Problem, 'parts'>): number {
  return problem.parts?.length ?? 0;
}

/** Every case of parts 1..part, visible or not, since parts are graded together. */
export function testIndicesThroughPart(tests: Test[], part: number): number[] {
  return tests
    .map((test, index) => ((test.part ?? 1) <= part ? index : -1))
    .filter((index) => index >= 0);
}

/**
 * The highest part k such that every case of parts 1..k was graded in `result` and passed,
 * or 0 when part 1 is not fully passing.
 */
export function highestPassedPart(tests: Test[], result: SubmissionResult, parts: number): number {
  const passed = new Set(
    result.results.filter((r) => r.passed && r.testIndex !== undefined).map((r) => r.testIndex),
  );
  let highest = 0;
  for (let part = 1; part <= parts; part += 1) {
    const indices = tests.map((test, index) => (test.part === part ? index : -1)).filter((i) => i >= 0);
    if (!indices.every((index) => passed.has(index))) break;
    highest = part;
  }
  return highest;
}

/** The part a learner may see after `result`: one past the highest fully passing part. */
export function unlockedAfter(tests: Test[], result: SubmissionResult, parts: number, current: number): number {
  return Math.max(current, Math.min(parts, highestPassedPart(tests, result, parts) + 1));
}

function unlockKey(problemId: string, version: number) {
  return `${UNLOCK_KEY_PREFIX}${problemId}:${version}`;
}

export function loadUnlockedPart(problemId: string, version: number, parts: number): number {
  if (typeof window === 'undefined') return 1;
  try {
    const stored = Number(window.localStorage.getItem(unlockKey(problemId, version)));
    return Number.isInteger(stored) && stored >= 1 && stored <= parts ? stored : 1;
  } catch {
    return 1;
  }
}

export function saveUnlockedPart(problemId: string, version: number, part: number) {
  if (typeof window === 'undefined') return;
  try {
    window.localStorage.setItem(unlockKey(problemId, version), String(part));
  } catch {
    // Unlocking is best-effort when storage is unavailable; passing again re-unlocks.
  }
}
