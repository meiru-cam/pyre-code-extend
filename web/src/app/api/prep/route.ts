import { NextResponse } from 'next/server';
import prepData from '@/lib/prep.json';
import { fetchProgressMap } from '@/lib/server/progress';
import type { PrepItem, PrepItemSummary, PrepLink } from '@/lib/types';

export async function GET() {
  const progressMap = await fetchProgressMap();
  const items: PrepItemSummary[] = (prepData.items as PrepItem[]).map(
    ({ prompt: _prompt, reference: _reference, ...item }) => ({
      ...item,
      exercisesSolved: item.exercises.filter((id) => progressMap[id]?.status === 'solved').length,
    }),
  );
  return NextResponse.json({ items, links: prepData.links as PrepLink[] });
}
