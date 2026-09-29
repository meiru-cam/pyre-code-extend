'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import { ArrowRight, ExternalLink } from 'lucide-react';
import { TopNav } from '@/components/layout/TopNav';
import { Footer } from '@/components/layout/Footer';
import { Badge } from '@/components/ui/Badge';
import { useLocale } from '@/context/LocaleContext';
import {
  PREP_ROUND_LABEL,
  PREP_TABS,
  companyName,
  difficultyVariant,
  groupByRound,
  type PrepTab,
} from '@/lib/prep';
import { hasPrepDraftContent, loadPrepDraft } from '@/lib/prepDraft';
import { cn } from '@/lib/utils';
import type { PrepItemSummary, PrepLink } from '@/lib/types';

type Entry = (PrepItemSummary & { entryType: 'item' }) | (PrepLink & { entryType: 'link' });

const TAB_STORAGE_KEY = 'pyre-code-prep-tab';

function StatusChip({ label, tone }: { label: string; tone: 'solved' | 'drafted' }) {
  const color = tone === 'solved' ? 'var(--easy)' : 'var(--accent)';
  return (
    <span
      className="mono text-[11px] px-[7px] py-[2px] rounded-[5px] tracking-[0.04em] flex-shrink-0"
      style={{ color, border: `1px solid color-mix(in oklab, ${color} 30%, var(--line))` }}
    >
      {label}
    </span>
  );
}

export default function PrepPage() {
  const { t } = useLocale();
  const [items, setItems] = useState<PrepItemSummary[]>([]);
  const [links, setLinks] = useState<PrepLink[]>([]);
  const [drafted, setDrafted] = useState<Set<string>>(new Set());
  const [tab, setTab] = useState<PrepTab>('openai');

  useEffect(() => {
    try {
      const saved = window.localStorage.getItem(TAB_STORAGE_KEY);
      if (saved && (PREP_TABS as readonly string[]).includes(saved)) setTab(saved as PrepTab);
    } catch {
      // Remembering the tab is a convenience only.
    }
    fetch('/api/prep')
      .then((r) => r.json())
      .then((d: { items: PrepItemSummary[]; links: PrepLink[] }) => {
        setItems(d.items ?? []);
        setLinks(d.links ?? []);
        setDrafted(new Set((d.items ?? []).filter((i) => hasPrepDraftContent(loadPrepDraft(i.id))).map((i) => i.id)));
      });
  }, []);

  const chooseTab = (next: PrepTab) => {
    setTab(next);
    try {
      window.localStorage.setItem(TAB_STORAGE_KEY, next);
    } catch {
      // Remembering the tab is a convenience only.
    }
  };

  const entries: Entry[] = [
    ...items.map((item) => ({ ...item, entryType: 'item' as const })),
    ...links.map((link) => ({ ...link, entryType: 'link' as const })),
  ];
  const groups = groupByRound(entries, tab);
  const tabCount = (candidate: PrepTab) => groupByRound(entries, candidate).reduce((n, g) => n + g.entries.length, 0);

  return (
    <div className="min-h-screen bg-bg">
      <TopNav />
      <main className="max-w-[1280px] mx-auto px-7 max-[600px]:px-4 pt-10 pb-20">
        <div className="mb-8 pb-7 max-w-[780px]" style={{ borderBottom: '1px solid var(--line)' }}>
          <div className="eyebrow mb-2.5">{t('prep')}</div>
          <h1 className="text-[clamp(32px,4vw,52px)] font-semibold tracking-[-0.032em] leading-[1.05] mb-3.5">
            {t('prepHero')}
          </h1>
          <p className="text-base text-text-2 leading-relaxed max-w-[58ch]">{t('prepSubtitle')}</p>
        </div>

        <div className="flex gap-1.5 mb-8 flex-wrap" role="tablist">
          {PREP_TABS.map((candidate) => (
            <button
              key={candidate}
              role="tab"
              aria-selected={tab === candidate}
              onClick={() => chooseTab(candidate)}
              className={cn(
                'px-3.5 py-1.5 rounded-[8px] text-sm font-medium transition-colors',
                tab === candidate ? 'text-accent' : 'text-text-2 hover:text-text',
              )}
              style={
                tab === candidate
                  ? { background: 'var(--accent-wash)', border: '1px solid var(--accent-line)' }
                  : { border: '1px solid var(--line)' }
              }
            >
              {candidate === 'other' ? t('prepOther') : companyName(candidate)}
              <span className="mono text-[11px] text-text-3 ml-2 tabular-nums">{tabCount(candidate)}</span>
            </button>
          ))}
        </div>

        <div className="flex flex-col gap-10">
          {groups.map((group) => (
            <section key={group.round}>
              <div className="flex items-baseline justify-between mb-3">
                <h2 className="text-[20px] font-semibold tracking-[-0.02em]">{t(PREP_ROUND_LABEL[group.round])}</h2>
                <span className="mono text-xs text-text-3 tabular-nums">{group.entries.length}</span>
              </div>
              <div className="rounded-[12px] overflow-hidden" style={{ border: '1px solid var(--line)' }}>
                {group.entries.map((entry, i) => {
                  const rowClass = 'group flex items-center gap-4 px-5 py-4 transition-colors hover:bg-[color-mix(in_oklab,var(--text)_3%,transparent)]';
                  const rowStyle = { background: 'var(--bg-elev)', borderTop: i === 0 ? undefined : '1px solid var(--line)' };
                  if (entry.entryType === 'link') {
                    return (
                      <a key={entry.id} href={entry.url} target="_blank" rel="noopener noreferrer" className={rowClass} style={rowStyle}>
                        <div className="flex-1 min-w-0">
                          <div className="font-medium text-sm">{entry.title}</div>
                          <div className="mono text-[11.5px] text-text-3 mt-0.5">
                            {tab === 'other' ? `${companyName(entry.company)} · ` : ''}{t('prepExternalSource')}
                          </div>
                        </div>
                        <Badge>{t('prepExternal')}</Badge>
                        <ExternalLink className="w-4 h-4 text-text-3 group-hover:text-accent transition-colors flex-shrink-0" />
                      </a>
                    );
                  }
                  const solved = entry.exercises.length > 0 && entry.exercisesSolved === entry.exercises.length;
                  return (
                    <Link key={entry.id} href={`/prep/${entry.id}`} className={rowClass} style={rowStyle}>
                      <div className="flex-1 min-w-0">
                        <div className="font-medium text-sm">{entry.title}</div>
                        <div className="text-[13px] text-text-2 mt-1 leading-snug line-clamp-2 max-w-[90ch]">{entry.summary}</div>
                        <div className="mono text-[11.5px] text-text-3 mt-1.5">
                          {[
                            entry.format,
                            entry.frequency && t('prepFrequency', { value: entry.frequency }),
                            entry.exercises.length > 0
                              && t('prepExercisesSolved', { solved: entry.exercisesSolved, total: entry.exercises.length }),
                          ].filter(Boolean).join(' · ')}
                        </div>
                      </div>
                      {solved && <StatusChip label={t('prepSolved').toUpperCase()} tone="solved" />}
                      {!solved && drafted.has(entry.id) && <StatusChip label={t('prepDrafted').toUpperCase()} tone="drafted" />}
                      {entry.difficulty && (
                        <Badge variant={difficultyVariant(entry.difficulty)} className="max-[600px]:hidden">
                          {entry.difficulty.toUpperCase()}
                        </Badge>
                      )}
                      <ArrowRight className="w-4 h-4 text-text-3 group-hover:text-accent transition-colors flex-shrink-0" />
                    </Link>
                  );
                })}
              </div>
            </section>
          ))}
        </div>

        <p className="text-xs text-text-3 mt-12 max-w-[80ch] leading-relaxed">{t('prepAttribution')}</p>
      </main>
      <Footer />
    </div>
  );
}
