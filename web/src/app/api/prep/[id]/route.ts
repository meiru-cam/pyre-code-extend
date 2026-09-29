import { NextResponse } from 'next/server';
import prepData from '@/lib/prep.json';
import problemsData from '@/lib/problems.json';
import { fetchProgressMap } from '@/lib/server/progress';
import type { LearningPathProblemSummary, PrepItem, PrepItemDetail } from '@/lib/types';

type ProblemMeta = { id: string; title: string; titleZh: string; difficulty: string };

export async function GET(_request: Request, { params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const item = (prepData.items as PrepItem[]).find((entry) => entry.id === id);
  if (!item) return NextResponse.json({ error: 'Not found' }, { status: 404 });

  const progressMap = await fetchProgressMap();
  const problems = (problemsData as { problems: ProblemMeta[] }).problems;
  const exerciseSteps: LearningPathProblemSummary[] = item.exercises.map((exerciseId) => {
    const problem = problems.find((p) => p.id === exerciseId);
    return {
      id: exerciseId,
      title: problem?.title ?? exerciseId,
      titleZh: problem?.titleZh ?? exerciseId,
      difficulty: (problem?.difficulty ?? 'Easy') as LearningPathProblemSummary['difficulty'],
      status: (progressMap[exerciseId]?.status ?? 'todo') as LearningPathProblemSummary['status'],
    };
  });

  const detail: PrepItemDetail = { ...item, exerciseSteps };
  return NextResponse.json(detail);
}
