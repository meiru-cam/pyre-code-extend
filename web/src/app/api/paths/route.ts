import { NextResponse } from 'next/server';
import pathsData from '@/lib/paths.json';
import { uniqueProgress } from '@/lib/pathGroups';
import { fetchProgressMap } from '@/lib/server/progress';
import type { LearningPath } from '@/lib/types';

export async function GET() {
  const progressMap = await fetchProgressMap();
  const isSolved = (id: string) => progressMap[id]?.status === 'solved';
  const learningPaths = pathsData.paths as LearningPath[];
  const paths = learningPaths.map((path) => ({
    ...path,
    solved: path.problems.filter(isSolved).length,
    total: path.problems.length,
  }));

  return NextResponse.json({ paths, ...uniqueProgress(learningPaths, isSolved) });
}
