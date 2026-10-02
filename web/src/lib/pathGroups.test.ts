// @vitest-environment jsdom

import { beforeEach, describe, expect, it } from 'vitest';

import pathsData from '@/lib/paths.json';
import { PATH_GROUPS, groupPaths, loadLastPath, pickContinuePath, saveLastPath } from '@/lib/pathGroups';
import type { LearningPath } from '@/lib/types';

const paths = pathsData.paths as LearningPath[];

describe('path groups', () => {
  it('puts every path in exactly one known group', () => {
    for (const path of paths) expect(PATH_GROUPS).toContain(path.group);
    const grouped = groupPaths(paths).flatMap((entry) => entry.paths.map((p) => p.id));
    expect(grouped.sort()).toEqual(paths.map((p) => p.id).sort());
  });

  it('only names prerequisites that exist', () => {
    const ids = new Set(paths.map((p) => p.id));
    for (const path of paths) for (const id of path.prerequisites) expect(ids).toContain(id);
  });

  it('keeps group order and drops empty groups', () => {
    const grouped = groupPaths([
      { id: 'b', group: 'agents' as const },
      { id: 'a', group: 'foundations' as const },
    ]);
    expect(grouped.map((entry) => entry.group)).toEqual(['foundations', 'agents']);
  });
});

describe('continue path', () => {
  const list = [
    { id: 'done', solved: 3, total: 3 },
    { id: 'fresh', solved: 0, total: 4 },
    { id: 'started', solved: 1, total: 4 },
    { id: 'half', solved: 2, total: 4 },
    { id: 'also-half', solved: 2, total: 9 },
  ];

  it('prefers the last opened path while it is unfinished', () => {
    expect(pickContinuePath(list, 'fresh')?.id).toBe('fresh');
  });

  it('falls back to the unfinished path with the most solved when the last one is finished or unknown', () => {
    expect(pickContinuePath(list, 'done')?.id).toBe('half');
    expect(pickContinuePath(list, 'gone')?.id).toBe('half');
    expect(pickContinuePath(list, null)?.id).toBe('half');
  });

  it('offers nothing when no path is under way', () => {
    expect(pickContinuePath([{ id: 'fresh', solved: 0, total: 4 }], null)).toBeNull();
  });
});

describe('last path storage', () => {
  beforeEach(() => window.localStorage.clear());

  it('round-trips the last opened path', () => {
    expect(loadLastPath()).toBeNull();
    saveLastPath('peft');
    expect(loadLastPath()).toBe('peft');
  });
});
