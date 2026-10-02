'use client';

import { useEffect, useRef, useState } from 'react';
import { useParams } from 'next/navigation';
import Link from 'next/link';
import { ArrowRight, Check, ExternalLink } from 'lucide-react';
import 'katex/dist/katex.min.css';
import { TopNav } from '@/components/layout/TopNav';
import { Footer } from '@/components/layout/Footer';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import { MarkdownContent } from '@/components/workspace/MarkdownContent';
import { useLocale } from '@/context/LocaleContext';
import { PREP_ROUND_LABEL, companyName, difficultyVariant } from '@/lib/prep';
import {
  EMPTY_PREP_DRAFT,
  loadPrepDraft,
  savePrepDraft,
  type PrepDraft,
} from '@/lib/prepDraft';
import type { PrepItemDetail } from '@/lib/types';

const SAVE_DELAY_MS = 400;

function Section({ title, children }: { title?: string; children: React.ReactNode }) {
  return (
    <section className="pt-8 mt-8" style={{ borderTop: '1px solid var(--line)' }}>
      {title && <h2 className="text-[18px] font-semibold tracking-[-0.02em] mb-4">{title}</h2>}
      {children}
    </section>
  );
}

export default function PrepItemPage() {
  const { id } = useParams<{ id: string }>();
  const { locale, t } = useLocale();
  const [item, setItem] = useState<PrepItemDetail | null>(null);
  const [notFound, setNotFound] = useState(false);
  const [draft, setDraft] = useState<PrepDraft>(EMPTY_PREP_DRAFT);
  const [showReference, setShowReference] = useState(false);
  const saveTimer = useRef<ReturnType<typeof setTimeout>>();
  const pendingSave = useRef<(() => void) | null>(null);

  useEffect(() => {
    setItem(null);
    setNotFound(false);
    fetch(`/api/prep/${id}`)
      .then((r) => (r.ok ? r.json() : null))
      .then((d: PrepItemDetail | null) => (d ? setItem(d) : setNotFound(true)))
      .catch(() => setNotFound(true));
    setDraft(loadPrepDraft(id) ?? EMPTY_PREP_DRAFT);
  }, [id]);

  // Write out a debounced save that is still waiting when the learner leaves the page.
  useEffect(() => () => {
    clearTimeout(saveTimer.current);
    pendingSave.current?.();
  }, [id]);

  const updateDraft = (next: PrepDraft) => {
    setDraft(next);
    clearTimeout(saveTimer.current);
    pendingSave.current = () => {
      savePrepDraft(id, next);
      pendingSave.current = null;
    };
    saveTimer.current = setTimeout(() => pendingSave.current?.(), SAVE_DELAY_MS);
  };

  const toggleCovered = (point: string) => {
    const covered = draft.covered.includes(point)
      ? draft.covered.filter((p) => p !== point)
      : [...draft.covered, point];
    updateDraft({ ...draft, covered });
  };

  if (!item) {
    return (
      <div className="min-h-screen bg-bg flex flex-col items-center justify-center gap-3">
        <p className="text-sm text-text-3">{notFound ? t('prepNotFound') : t('loading')}</p>
        {notFound && <Link href="/prep" className="text-sm text-accent">{t('prep')}</Link>}
      </div>
    );
  }

  const meta = [
    item.kind,
    item.format,
    item.frequency && t('prepFrequency', { value: item.frequency }),
  ].filter(Boolean);

  return (
    <div className="min-h-screen bg-bg">
      <TopNav />
      <main className="max-w-[860px] mx-auto px-7 max-[600px]:px-4 pt-8 pb-20">
        <div className="mono text-xs text-text-3 flex items-center gap-1.5 mb-5 flex-wrap">
          <Link href="/prep" className="hover:text-text transition-colors">{t('prep')}</Link>
          <span className="opacity-60">/</span>
          <span>{companyName(item.company)}</span>
          <span className="opacity-60">/</span>
          <span className="text-text font-medium">{item.title}</span>
        </div>

        <div className="eyebrow mb-2">{t(PREP_ROUND_LABEL[item.round])}</div>
        <h1 className="text-[clamp(28px,3.4vw,42px)] font-semibold tracking-[-0.03em] leading-[1.1] mb-3">{item.title}</h1>
        <div className="flex items-center gap-2 flex-wrap mb-4">
          {item.difficulty && <Badge variant={difficultyVariant(item.difficulty)}>{item.difficulty.toUpperCase()}</Badge>}
          <span className="mono text-xs text-text-3">{meta.join(' · ')}</span>
        </div>
        <p className="text-base text-text-2 leading-relaxed mb-3">{item.summary}</p>
        <a
          href={item.source}
          target="_blank"
          rel="noopener noreferrer"
          className="inline-flex items-center gap-1.5 mono text-xs text-text-3 hover:text-accent transition-colors"
        >
          {t('prepSource')} <ExternalLink className="w-3 h-3" />
        </a>

        <Section>
          <MarkdownContent content={item.prompt} extended />
        </Section>

        {item.exerciseSteps.length > 0 && (
          <Section title={t('prepPracticeWith')}>
            <div className="rounded-[12px] overflow-hidden" style={{ border: '1px solid var(--line)' }}>
              {item.exerciseSteps.map((step, i) => (
                <Link
                  key={step.id}
                  href={`/problems/${step.id}`}
                  className="group flex items-center gap-4 px-5 py-3.5 transition-colors hover:bg-[color-mix(in_oklab,var(--text)_3%,transparent)]"
                  style={{ background: 'var(--bg-elev)', borderTop: i === 0 ? undefined : '1px solid var(--line)' }}
                >
                  <div className="flex-1 min-w-0">
                    <div className="font-medium text-sm">{locale === 'zh' ? step.titleZh : step.title}</div>
                    <div className="mono text-[11.5px] text-text-3 mt-0.5">{step.id}.py</div>
                  </div>
                  {step.status === 'solved' && <Check className="w-4 h-4 text-easy flex-shrink-0" />}
                  <Badge variant={difficultyVariant(step.difficulty)}>
                    {step.difficulty.toUpperCase()}
                  </Badge>
                  <ArrowRight className="w-4 h-4 text-text-3 group-hover:text-accent transition-colors flex-shrink-0" />
                </Link>
              ))}
            </div>
          </Section>
        )}

        {item.rubric.length > 0 && (
          <Section title={t('prepCoverThese')}>
            <ul className="flex flex-col gap-2 mb-6">
              {item.rubric.map((point) => (
                <li key={point}>
                  <label className="flex items-start gap-2.5 text-sm text-text-2 cursor-pointer">
                    <input
                      type="checkbox"
                      className="mt-[3px] accent-[var(--accent)]"
                      checked={draft.covered.includes(point)}
                      onChange={() => toggleCovered(point)}
                    />
                    <span>{point}</span>
                  </label>
                </li>
              ))}
            </ul>
            <label htmlFor="prep-answer" className="block text-sm font-medium mb-2">{t('prepYourAnswer')}</label>
            <textarea
              id="prep-answer"
              value={draft.answer}
              onChange={(e) => updateDraft({ ...draft, answer: e.target.value })}
              placeholder={t('prepAnswerPlaceholder')}
              rows={14}
              className="w-full rounded-[10px] p-4 text-sm leading-relaxed text-text resize-y focus:outline-none"
              style={{ background: 'var(--bg-sunken)', border: '1px solid var(--line)' }}
            />
          </Section>
        )}

        {item.reference && (
          <Section>
            <Button variant="secondary" onClick={() => setShowReference((v) => !v)}>
              {showReference ? t('prepHideReference') : t('prepShowReference')}
            </Button>
            {showReference && (
              <div className="mt-6">
                <MarkdownContent content={item.reference} extended />
              </div>
            )}
          </Section>
        )}
      </main>
      <Footer />
    </div>
  );
}
