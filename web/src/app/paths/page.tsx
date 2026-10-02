'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import { ArrowRight, Check } from 'lucide-react';
import { TopNav } from '@/components/layout/TopNav';
import { Footer } from '@/components/layout/Footer';
import { useLocale } from '@/context/LocaleContext';
import { groupPaths, loadLastPath, pickContinuePath } from '@/lib/pathGroups';
import type { TranslationKey } from '@/lib/i18n';
import type { LearningPath, PathGroup } from '@/lib/types';

type PathWithProgress = LearningPath & { solved: number; total: number };

const GROUP_LABEL: Record<PathGroup, TranslationKey> = {
  foundations: 'pathGroupFoundations',
  architectures: 'pathGroupArchitectures',
  training: 'pathGroupTraining',
  systems: 'pathGroupSystems',
  agents: 'pathGroupAgents',
};

function percent(path: PathWithProgress) {
  return path.total > 0 ? Math.round((path.solved / path.total) * 100) : 0;
}

function ProgressBar({ pct }: { pct: number }) {
  return (
    <div className="h-[3px] rounded-pill relative w-full" style={{ background: 'var(--line)' }}>
      <div className="absolute inset-y-0 left-0 rounded-pill" style={{ width: `${pct}%`, background: 'var(--accent)' }} />
    </div>
  );
}

export default function PathsPage() {
  const { locale, t } = useLocale();
  const [paths, setPaths] = useState<PathWithProgress[]>([]);
  const [lastPathId, setLastPathId] = useState<string | null>(null);

  useEffect(() => {
    setLastPathId(loadLastPath());
    fetch('/api/paths')
      .then((r) => r.json())
      .then((d) => setPaths(d.paths ?? []));
  }, []);

  const titleOf = (path: LearningPath) => (locale === 'zh' ? path.titleZh : path.titleEn);
  const continuePath = pickContinuePath(paths, lastPathId);

  return (
    <div className="min-h-screen bg-bg">
      <TopNav />
      <main className="max-w-[960px] mx-auto px-7 max-[640px]:px-4 pt-10 pb-20">
        <div className="mb-8">
          <div className="eyebrow mb-2.5">{t('paths')}</div>
          <h1 className="text-[clamp(28px,3.4vw,40px)] font-semibold tracking-[-0.03em] leading-[1.1] mb-2.5">
            {t('pathsHero')}
          </h1>
          <p className="text-base text-text-2 leading-relaxed max-w-[58ch]">{t('pathsSubtitle')}</p>
        </div>

        {continuePath && (
          <Link
            href={`/paths/${continuePath.id}`}
            className="flex items-center gap-4 rounded-[12px] px-5 py-4 mb-10 group"
            style={{ background: 'var(--accent-wash)', border: '1px solid var(--accent)' }}
          >
            <span className="mono text-[11px] tracking-[0.14em] uppercase text-accent flex-shrink-0">{t('pathContinue')}</span>
            <span className="font-semibold truncate flex-1 min-w-0">{titleOf(continuePath)}</span>
            <span className="mono text-xs text-text-2 tabular-nums flex-shrink-0">
              {continuePath.solved}/{continuePath.total}
            </span>
            <ArrowRight className="w-4 h-4 text-accent flex-shrink-0 transition-transform duration-150 group-hover:translate-x-[3px]" />
          </Link>
        )}

        <div className="flex flex-col gap-9">
          {groupPaths(paths).map(({ group, paths: members }) => (
            <section key={group}>
              <h2 className="text-sm font-semibold tracking-[-0.01em] mb-2.5 px-1">{t(GROUP_LABEL[group])}</h2>
              <ul
                className="rounded-[12px] overflow-hidden"
                style={{ border: '1px solid var(--line)', background: 'var(--bg-elev)' }}
              >
                {members.map((path, idx) => {
                  const pct = percent(path);
                  const done = path.total > 0 && path.solved === path.total;
                  return (
                    <li key={path.id} style={idx > 0 ? { borderTop: '1px solid var(--line)' } : undefined}>
                      <Link
                        href={`/paths/${path.id}`}
                        className="grid grid-cols-[minmax(0,1fr)_140px_auto] max-[640px]:grid-cols-[minmax(0,1fr)_auto] items-center gap-5 max-[640px]:gap-3 px-5 max-[640px]:px-4 py-3.5 group transition-colors duration-150 hover:bg-[color-mix(in_oklab,var(--accent)_4%,var(--bg-elev))]"
                      >
                        <div className="min-w-0">
                          <div className="flex items-center gap-2">
                            {done && <Check className="w-4 h-4 text-easy flex-shrink-0" />}
                            <span className="font-medium truncate">{titleOf(path)}</span>
                          </div>
                          <p className="text-[13px] text-text-3 truncate mt-0.5">
                            {locale === 'zh' ? path.descriptionZh : path.descriptionEn}
                          </p>
                        </div>
                        <div className="flex items-center gap-2.5 max-[640px]:hidden">
                          <ProgressBar pct={pct} />
                        </div>
                        <div className="flex items-center gap-2 mono text-xs text-text-2 tabular-nums">
                          <span className="min-w-[5ch] text-right">{path.solved}/{path.total}</span>
                          <ArrowRight className="w-4 h-4 text-text-3 transition-[color,transform] duration-150 group-hover:text-accent group-hover:translate-x-[3px]" />
                        </div>
                      </Link>
                    </li>
                  );
                })}
              </ul>
            </section>
          ))}
        </div>
      </main>
      <Footer />
    </div>
  );
}
