import { describe, expect, it } from 'vitest';

import { companyName, groupByRound, prepTab } from '@/lib/prep';

describe('prep grouping', () => {
  it('sends companies without their own tab to other', () => {
    expect(prepTab('openai')).toBe('openai');
    expect(prepTab('stripe')).toBe('other');
  });

  it('groups a tab by round in interview order and drops empty rounds', () => {
    const entries = [
      { id: 'a', company: 'openai', round: 'behavioral' as const },
      { id: 'b', company: 'openai', round: 'ml-coding' as const },
      { id: 'c', company: 'stripe', round: 'coding' as const },
    ];
    expect(groupByRound(entries, 'openai').map((g) => [g.round, g.entries.map((e) => e.id)]))
      .toEqual([['ml-coding', ['b']], ['behavioral', ['a']]]);
    expect(groupByRound(entries, 'other').map((g) => g.round)).toEqual(['coding']);
  });

  it('keeps brand casing for known companies', () => {
    expect(companyName('xai')).toBe('xAI');
    expect(companyName('databricks')).toBe('Databricks');
  });
});
