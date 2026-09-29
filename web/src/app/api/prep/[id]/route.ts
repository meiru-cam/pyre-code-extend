import { NextResponse } from 'next/server';
import prepData from '@/lib/prep.json';
import { fetchProgressMap, summarizeExercises } from '@/lib/server/progress';
import type { PrepItem, PrepItemDetail } from '@/lib/types';

export async function GET(_request: Request, { params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const item = (prepData.items as PrepItem[]).find((entry) => entry.id === id);
  if (!item) return NextResponse.json({ error: 'Not found' }, { status: 404 });

  const detail: PrepItemDetail = {
    ...item,
    exerciseSteps: summarizeExercises(item.exercises, await fetchProgressMap()),
  };
  return NextResponse.json(detail);
}
