import { cookies } from 'next/headers';
import { GRADING_SERVICE_URL } from '@/lib/constants';

export type ProgressMap = Record<string, { status: string }>;

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
