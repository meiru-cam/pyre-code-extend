'use client';

import Link from 'next/link';
import { Check } from 'lucide-react';
import { useLocale } from '@/context/LocaleContext';
import { groupPaths, type GroupProgress } from '@/lib/pathGroups';
import type { TranslationKey } from '@/lib/i18n';
import type { LearningPath, PathGroup } from '@/lib/types';

export type PathWithProgress = LearningPath & { solved: number; total: number };

/** What /api/paths returns. */
export type PathsResponse = {
  paths: PathWithProgress[];
  overall: GroupProgress;
  groups: Partial<Record<PathGroup, GroupProgress>>;
};

const GROUP_LABEL: Record<PathGroup, TranslationKey> = {
  foundations: 'pathGroupFoundations',
  architectures: 'pathGroupArchitectures',
  training: 'pathGroupTraining',
  systems: 'pathGroupSystems',
  agents: 'pathGroupAgents',
};

export function percent({ solved, total }: GroupProgress) {
  return total > 0 ? Math.round((solved / total) * 100) : 0;
}

export function ProgressBar({ progress, thick = false }: { progress: GroupProgress; thick?: boolean }) {
  return (
    <div
      className={`${thick ? 'h-[6px]' : 'h-[3px]'} rounded-pill relative w-full`}
      style={{ background: 'var(--line)' }}
    >
      <div
        className="absolute inset-y-0 left-0 rounded-pill"
        style={{ width: `${percent(progress)}%`, background: 'var(--accent)' }}
      />
    </div>
  );
}

/** One card per path group: the group's progress, then each path with its own count. */
export function PathGroupCards({ data }: { data: PathsResponse | null }) {
  const { locale, t } = useLocale();
  const paths = data?.paths ?? [];

  return (
    <div className="grid grid-cols-3 max-[900px]:grid-cols-2 max-[600px]:grid-cols-1 gap-4">
      {groupPaths(paths).map(({ group, paths: members }) => {
        const progress = data?.groups?.[group];
        return (
          <section
            key={group}
            className="rounded-[12px] px-5 pt-4 pb-3 min-w-0"
            style={{ border: '1px solid var(--line)', background: 'var(--bg-elev)' }}
          >
            <div className="flex items-baseline justify-between gap-2 mb-2">
              <h2 className="text-[15px] font-semibold leading-snug">{t(GROUP_LABEL[group])}</h2>
              {progress && (
                <span className="mono text-xs text-text-3 tabular-nums flex-shrink-0">
                  {progress.solved}/{progress.total}
                </span>
              )}
            </div>
            {progress && <ProgressBar progress={progress} />}
            <ul className="mt-3 -mx-2">
              {members.map((path) => {
                const done = path.total > 0 && path.solved === path.total;
                return (
                  <li key={path.id}>
                    <Link
                      href={`/paths/${path.id}`}
                      title={locale === 'zh' ? path.descriptionZh : path.descriptionEn}
                      className="flex items-center gap-2 rounded-[8px] px-2 py-1.5 text-[13px] text-text-2 transition-colors duration-150 hover:text-accent hover:bg-[color-mix(in_oklab,var(--accent)_5%,transparent)]"
                    >
                      <span className="flex-1 min-w-0 leading-snug">
                        {locale === 'zh' ? path.titleZh : path.titleEn}
                      </span>
                      {done && <Check className="w-3.5 h-3.5 text-easy flex-shrink-0" />}
                      <span className="mono text-[11px] text-text-3 tabular-nums flex-shrink-0">
                        {path.solved}/{path.total}
                      </span>
                    </Link>
                  </li>
                );
              })}
            </ul>
          </section>
        );
      })}
    </div>
  );
}
