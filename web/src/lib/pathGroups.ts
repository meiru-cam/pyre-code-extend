import type { LearningPath, PathGroup } from '@/lib/types';

/** Display order of the groups on /paths. */
export const PATH_GROUPS: PathGroup[] = ['foundations', 'architectures', 'training', 'systems', 'agents'];

const LAST_PATH_KEY = 'pyre-code:last-path';

type Progress = { solved: number; total: number };

/** Paths bucketed by group, in PATH_GROUPS order; each group keeps the file order. Empty groups are dropped. */
export function groupPaths<T extends Pick<LearningPath, 'group'>>(paths: T[]): { group: PathGroup; paths: T[] }[] {
  return PATH_GROUPS
    .map((group) => ({ group, paths: paths.filter((p) => p.group === group) }))
    .filter((entry) => entry.paths.length > 0);
}

/** The path to offer as "continue": the last one opened unless it is finished, else the unfinished one with the most solved (first wins ties). */
export function pickContinuePath<T extends Pick<LearningPath, 'id'> & Progress>(
  paths: T[],
  lastPathId: string | null,
): T | null {
  const unfinished = (p: T) => p.solved < p.total;
  const last = paths.find((p) => p.id === lastPathId);
  if (last && unfinished(last)) return last;
  let best: T | null = null;
  for (const p of paths) {
    if (p.solved > 0 && unfinished(p) && (!best || p.solved > best.solved)) best = p;
  }
  return best;
}

export function loadLastPath(): string | null {
  try {
    return window.localStorage.getItem(LAST_PATH_KEY);
  } catch {
    return null;
  }
}

export function saveLastPath(pathId: string): void {
  try {
    window.localStorage.setItem(LAST_PATH_KEY, pathId);
  } catch {
    // Remembering the last path is a convenience; storage may be unavailable.
  }
}
