import { NextResponse } from 'next/server';
import pathsData from '@/lib/paths.json';
import { fetchProgressMap } from '@/lib/server/progress';
import type { LearningPath } from '@/lib/types';

export async function GET() {
  const progressMap = await fetchProgressMap();
  const paths = (pathsData.paths as LearningPath[]).map((path) => {
    const solved = path.problems.filter((id) => progressMap[id]?.status === 'solved').length;
    return { ...path, solved, total: path.problems.length };
  });

  return NextResponse.json({ paths });
}
