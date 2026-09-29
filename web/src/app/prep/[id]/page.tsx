'use client';

import { useEffect, useRef, useState } from 'react';
import { useParams } from 'next/navigation';
import Link from 'next/link';
import { ArrowRight, Check, Copy, ExternalLink } from 'lucide-react';
import 'katex/dist/katex.min.css';
import { TopNav } from '@/components/layout/TopNav';
import { Footer } from '@/components/layout/Footer';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import { MarkdownContent } from '@/components/workspace/MarkdownContent';
import { useLocale } from '@/context/LocaleContext';
import { PREP_ROUND_LABEL, companyName } from '@/lib/prep';
import {
  EMPTY_PREP_DRAFT,
  formatPrepForGrading,
  loadPrepDraft,
  savePrepDraft,
  type PrepDraft,
} from '@/lib/prepDraft';
import type { PrepItemDetail } from '@/lib/types';

const SAVE_DELAY_MS = 400;

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="pt-8 mt-8" style={{ borderTop: '1px solid var(--line)' }}>
      <h2 className="text-[18px] font-semibold tracking-[-0.02em] mb-4">{title}</h2>
      {children}
    </section>
  );
}

export default function PrepItemPage() {
  const { id } = useParams<{ id: string }>();
  const { t } = useLocale();
  const [item, setItem] = useState<PrepItemDetail | null>(null);
  const [draft, setDraft] = useState<PrepDraft>(EMPTY_PREP_DRAFT);
  const [showReference, setShowReference] = useState(false);
  const [copied, setCopied] = useState(false);
  const saveTimer = useRef<ReturnType<typeof setTimeout>>();

  useEffect(() => {
    fetch(`/api/prep/${id}`)
      .then((r) => r.json())
      .then((d: PrepItemDetail) => setItem(d));
    setDraft(loadPrepDraft(id) ?? EMPTY_PREP_DRAFT);
  }, [id]);

  const updateDraft = (next: PrepDraft) => {
    setDraft(next);
    clearTimeout(saveTimer.current);
    saveTimer.current = setTimeout(() => savePrepDraft(id, next), SAVE_DELAY_MS);
  };

  const toggleCovered = (point: string) => {
    const covered = draft.covered.includes(point)
      ? draft.covered.filter((p) => p !== point)
      : [...draft.covered, point];
    updateDraft({ ...draft, covered });
  };

  const copyForGrading = async () => {
    if (!item) return;
    try {
      await navigator.clipboard.writeText(formatPrepForGrading(item, draft.answer));
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      // Clipboard access can be denied; the answer is still on the page to copy by hand.
    }
  };

  if (!item) {
    return (
      <div className="min-h-screen bg-bg flex items-center justify-center">
        <p className="text-sm text-text-3">{t('loading')}</p>
      </div>
    );
  }

  const meta = [item.kind, item.format, item.frequency && `freq: ${item.frequency}`].filter(Boolean);

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
          {item.difficulty && <Badge>{item.difficulty.toUpperCase()}</Badge>}
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

        <div className="pt-8 mt-8" style={{ borderTop: '1px solid var(--line)' }}>
          <MarkdownContent content={item.prompt} extended />
        </div>

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
                    <div className="font-medium text-sm">{step.title}</div>
                    <div className="mono text-[11.5px] text-text-3 mt-0.5">{step.id}.py</div>
                  </div>
                  {step.status === 'solved' && <Check className="w-4 h-4 text-easy flex-shrink-0" />}
                  <Badge variant={step.difficulty.toLowerCase() as 'easy' | 'medium' | 'hard'}>
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
            <div className="flex justify-end mt-3">
              <Button variant="secondary" onClick={copyForGrading} disabled={!draft.answer.trim()}>
                {copied ? <Check className="w-3.5 h-3.5" /> : <Copy className="w-3.5 h-3.5" />}
                {copied ? t('prepCopied') : t('prepCopyForGrading')}
              </Button>
            </div>
          </Section>
        )}

        {item.reference && (
          <div className="pt-8 mt-8" style={{ borderTop: '1px solid var(--line)' }}>
            <Button variant="secondary" onClick={() => setShowReference((v) => !v)}>
              {showReference ? t('prepHideReference') : t('prepShowReference')}
            </Button>
            {showReference && (
              <div className="mt-6">
                <MarkdownContent content={item.reference} extended />
              </div>
            )}
          </div>
        )}
      </main>
      <Footer />
    </div>
  );
}
