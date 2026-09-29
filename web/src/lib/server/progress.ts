import { cookies } from 'next/headers';
import problemsData from '@/lib/problems.json';
import { GRADING_SERVICE_URL } from '@/lib/constants';
import type { LearningPathProblemSummary } from '@/lib/types';

export type ProgressMap = Record<string, { status: string }>;

type ProblemMeta = { id: string; title: string; titleZh: string; difficulty: string };

/** The signed-in learner's per-problem progress, or an empty map when unavailable. */
export async function fetchProgressMap(): Promise<ProgressMap> {
  const cookieStore = await cookies();
  const sessionToken = cookieStore.get('session_token')?.value;
  if (!sessionToken) return {};
  try {
    const userRes = await fetch(`${GRADING_SERVICE_URL}/users`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sessionToken }),
    });
    if (!userRes.ok) return {};
    const { userId } = await userRes.json();
    const progressRes = await fetch(`${GRADING_SERVICE_URL}/progress/${userId}`);
    return progressRes.ok ? await progressRes.json() : {};
  } catch {
    return {};
  }
}

/** Title, difficulty and the learner's status for each exercise id, in order. */
export function summarizeExercises(
  exerciseIds: string[],
  progressMap: ProgressMap,
): LearningPathProblemSummary[] {
  const problems = (problemsData as { problems: ProblemMeta[] }).problems;
  return exerciseIds.map((exerciseId) => {
    const problem = problems.find((p) => p.id === exerciseId);
    return {
      id: exerciseId,
      title: problem?.title ?? exerciseId,
      titleZh: problem?.titleZh ?? exerciseId,
      difficulty: (problem?.difficulty ?? 'Easy') as LearningPathProblemSummary['difficulty'],
      status: (progressMap[exerciseId]?.status ?? 'todo') as LearningPathProblemSummary['status'],
    };
  });
}
