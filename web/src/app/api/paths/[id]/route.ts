import { NextResponse } from 'next/server';
import pathsData from '@/lib/paths.json';
import { fetchProgressMap, summarizeExercises } from '@/lib/server/progress';
import type { LearningPath } from '@/lib/types';

export async function GET(_request: Request, { params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const path = (pathsData.paths as LearningPath[]).find((p) => p.id === id);
  if (!path) return NextResponse.json({ error: 'Not found' }, { status: 404 });

  const problems = summarizeExercises(path.problems, await fetchProgressMap());
  const solved = problems.filter((p) => p.status === 'solved').length;

  return NextResponse.json({ ...path, problems, solved, total: path.problems.length });
}
