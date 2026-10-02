'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import { ArrowRight } from 'lucide-react';
import { TopNav } from '@/components/layout/TopNav';
import { Footer } from '@/components/layout/Footer';
import { useLocale } from '@/context/LocaleContext';
import { PathGroupCards, ProgressBar, percent, type PathsResponse } from '@/components/paths/PathGroupCards';
import { loadLastPath, pickContinuePath } from '@/lib/pathGroups';
import type { LearningPath } from '@/lib/types';

export default function PathsPage() {
  const { locale, t } = useLocale();
  const [data, setData] = useState<PathsResponse | null>(null);
  const [lastPathId, setLastPathId] = useState<string | null>(null);

  useEffect(() => {
    setLastPathId(loadLastPath());
    fetch('/api/paths')
      .then((r) => r.json())
      .then((d) => setData(d));
  }, []);

  const paths = data?.paths ?? [];
  const titleOf = (path: LearningPath) => (locale === 'zh' ? path.titleZh : path.titleEn);
  const continuePath = pickContinuePath(paths, lastPathId);

  return (
    <div className="min-h-screen bg-bg">
      <TopNav />
      <main className="max-w-[1080px] mx-auto px-7 max-[640px]:px-4 pt-10 pb-20">
        <div className="mb-8">
          <div className="eyebrow mb-2.5">{t('paths')}</div>
          <h1 className="text-[clamp(28px,3.4vw,40px)] font-semibold tracking-[-0.03em] leading-[1.1] mb-2.5">
            {t('pathsHero')}
          </h1>
          <p className="text-base text-text-2 leading-relaxed max-w-[58ch]">{t('pathsSubtitle')}</p>
        </div>

        {data?.overall && (
          <div
            className="flex items-center gap-6 max-[640px]:flex-col max-[640px]:items-stretch max-[640px]:gap-4 rounded-[12px] px-5 py-4 mb-6"
            style={{ border: '1px solid var(--line)', background: 'var(--bg-elev)' }}
          >
            <div className="flex-1 min-w-0">
              <div className="flex items-baseline justify-between mb-2">
                <span className="text-sm font-semibold">{t('pathOverall')}</span>
                <span className="mono text-xs text-text-2 tabular-nums">
                  {t('pathOverallCount', data.overall)} · {percent(data.overall)}%
                </span>
              </div>
              <ProgressBar progress={data.overall} thick />
            </div>
            {continuePath && (
              <Link
                href={`/paths/${continuePath.id}`}
                className="flex items-center gap-3 rounded-[10px] px-4 py-2.5 flex-shrink-0 min-w-0 max-w-[360px] max-[640px]:max-w-none group"
                style={{ background: 'var(--accent-wash)', border: '1px solid var(--accent)' }}
              >
                <span className="mono text-[11px] tracking-[0.14em] uppercase text-accent flex-shrink-0">
                  {t('pathContinue')}
                </span>
                <span className="text-sm font-semibold truncate min-w-0">{titleOf(continuePath)}</span>
                <ArrowRight className="w-4 h-4 text-accent flex-shrink-0 transition-transform duration-150 group-hover:translate-x-[3px]" />
              </Link>
            )}
          </div>
        )}

        <PathGroupCards data={data} />
      </main>
      <Footer />
    </div>
  );
}
