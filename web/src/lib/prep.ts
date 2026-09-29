import type { TranslationKey } from '@/lib/i18n';
import type { PrepRound } from '@/lib/types';

/** Companies with their own tab; every other company shares the last tab. */
export const PREP_TABS = ['openai', 'anthropic', 'other'] as const;
export type PrepTab = (typeof PREP_TABS)[number];

export const PREP_ROUND_ORDER: PrepRound[] = [
  'ml-coding', 'coding', 'system-design', 'behavioral', 'take-home',
];

export const PREP_ROUND_LABEL: Record<PrepRound, TranslationKey> = {
  'ml-coding': 'prepRoundMlCoding',
  coding: 'prepRoundCoding',
  'system-design': 'prepRoundSystemDesign',
  behavioral: 'prepRoundBehavioral',
  'take-home': 'prepRoundTakeHome',
};

const COMPANY_NAMES: Record<string, string> = {
  openai: 'OpenAI',
  anthropic: 'Anthropic',
  xai: 'xAI',
};

export function companyName(company: string): string {
  return COMPANY_NAMES[company] ?? company.charAt(0).toUpperCase() + company.slice(1);
}

export function prepTab(company: string): PrepTab {
  return (PREP_TABS as readonly string[]).includes(company) ? (company as PrepTab) : 'other';
}

/** Entries of one tab grouped by round, in interview order, skipping empty rounds. */
export function groupByRound<T extends { company: string; round: PrepRound }>(
  entries: T[],
  tab: PrepTab,
): { round: PrepRound; entries: T[] }[] {
  const inTab = entries.filter((entry) => prepTab(entry.company) === tab);
  return PREP_ROUND_ORDER
    .map((round) => ({ round, entries: inTab.filter((entry) => entry.round === round) }))
    .filter((group) => group.entries.length > 0);
}
