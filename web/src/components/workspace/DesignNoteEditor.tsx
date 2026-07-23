'use client';

import { useEffect, useRef, useState } from 'react';
import { Check, CircleDashed, Save, Sparkles } from 'lucide-react';

import { Button } from '@/components/ui/Button';
import { MarkdownContent } from '@/components/workspace/MarkdownContent';
import {
  emptyDesignNoteFields,
  hasDesignNoteContent,
  loadDesignNoteDraft,
  saveDesignNoteDraft,
} from '@/lib/designNoteDraft';
import { buildDesignReviewRequest } from '@/lib/designReview';
import type { Locale } from '@/lib/i18n';
import { scanForExternalAiSecrets } from '@/lib/secretBoundary';
import type {
  DesignNoteFieldName,
  DesignNoteFields,
  DesignNoteResponse,
  Problem,
  ProblemProgress,
} from '@/lib/types';

type NoteStatus = 'empty' | 'draft' | 'saving' | 'saved' | 'error';
type AiReviewStatus = 'not_requested' | 'reviewing' | 'ready' | 'error';

interface DesignNoteEditorProps {
  problem: Problem;
  implementationStatus: ProblemProgress['status'];
  locale?: Locale;
}

const STATUS_LABELS: Record<ProblemProgress['status'], string> = {
  todo: 'Todo',
  attempted: 'Attempted',
  solved: 'Solved',
};

const NOTE_LABELS: Record<NoteStatus, string> = {
  empty: 'Empty',
  draft: 'Draft',
  saving: 'Saving',
  saved: 'Saved',
  error: 'Save failed',
};

const AI_REVIEW_LABELS: Record<AiReviewStatus, string> = {
  not_requested: 'Not requested',
  reviewing: 'Reviewing',
  ready: 'Ready',
  error: 'Review failed',
};

export function DesignNoteEditor(props: DesignNoteEditorProps) {
  const contractVersion = props.problem.version ?? 1;
  if (!props.problem.designNoteRubric?.length) return null;
  return (
    <DesignNoteEditorVersion
      key={`${props.problem.id}:${contractVersion}`}
      {...props}
    />
  );
}

function DesignNoteEditorVersion({
  problem,
  implementationStatus,
  locale = 'en',
}: DesignNoteEditorProps) {
  const rubric = problem.designNoteRubric ?? [];
  const contractVersion = problem.version ?? 1;
  const hadLocalDraft = useRef(false);
  const [fields, setFields] = useState<DesignNoteFields>(() => {
    const draft = loadDesignNoteDraft(problem.id, contractVersion, rubric);
    hadLocalDraft.current = draft !== null;
    return draft ?? emptyDesignNoteFields(rubric);
  });
  const [noteStatus, setNoteStatus] = useState<NoteStatus>(() => (
    hasDesignNoteContent(fields) ? 'draft' : 'empty'
  ));
  const [aiReviewStatus, setAiReviewStatus] = useState<AiReviewStatus>('not_requested');
  const [review, setReview] = useState<string | null>(null);
  const [reviewError, setReviewError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    fetch(
      `/api/design-notes/${encodeURIComponent(problem.id)}`
      + `?contractVersion=${contractVersion}`,
    )
      .then(async (response) => {
        if (!response.ok) return null;
        return response.json() as Promise<DesignNoteResponse>;
      })
      .then((saved) => {
        if (!active || !saved || hadLocalDraft.current) return;
        setFields(saved.fields);
        saveDesignNoteDraft(problem.id, contractVersion, saved.fields);
        setNoteStatus(hasDesignNoteContent(saved.fields) ? 'saved' : 'empty');
      })
      .catch(() => {
        // Local drafting remains available when persistence is unavailable.
      });
    return () => {
      active = false;
    };
  }, [contractVersion, problem.id]);

  const updateField = (field: DesignNoteFieldName, value: string) => {
    setFields((previous) => {
      const next = { ...previous, [field]: value };
      saveDesignNoteDraft(problem.id, contractVersion, next);
      setNoteStatus(hasDesignNoteContent(next) ? 'draft' : 'empty');
      return next;
    });
  };

  const save = async () => {
    setNoteStatus('saving');
    try {
      const response = await fetch(`/api/design-notes/${encodeURIComponent(problem.id)}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ contractVersion, fields }),
      });
      if (!response.ok) throw new Error('save failed');
      const saved = await response.json() as DesignNoteResponse;
      setFields(saved.fields);
      saveDesignNoteDraft(problem.id, contractVersion, saved.fields);
      setNoteStatus('saved');
    } catch {
      setNoteStatus('error');
    }
  };

  const requestReview = async () => {
    const request = buildDesignReviewRequest(problem, fields, locale);
    if (scanForExternalAiSecrets(request).blocked) {
      setReview(null);
      setReviewError(
        'Remove likely credentials or private keys before AI review.',
      );
      setAiReviewStatus('error');
      return;
    }
    setAiReviewStatus('reviewing');
    setReview(null);
    setReviewError(null);
    try {
      const response = await fetch('/api/design-review', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(request),
      });
      if (!response.ok) throw new Error('review failed');
      const result = await response.json() as { review?: unknown };
      if (typeof result.review !== 'string' || !result.review.trim()) {
        throw new Error('review failed');
      }
      setReview(result.review);
      setAiReviewStatus('ready');
    } catch {
      setReviewError(
        'AI review is unavailable. Your saved note and implementation status were not changed.',
      );
      setAiReviewStatus('error');
    }
  };

  const completed = rubric.filter(({ field }) => fields[field]?.trim()).length;

  return (
    <section
      className="rounded-xl p-4 space-y-4"
      style={{ border: '1px solid var(--line)', background: 'var(--bg-elev)' }}
    >
      <div className="flex items-start justify-between gap-4">
        <div>
          <h2 className="text-sm font-semibold text-text">System-design note</h2>
          <p className="mt-1 text-xs leading-relaxed text-text-3">
            Capture the runtime decisions behind your implementation. Notes do not affect Solved status.
          </p>
        </div>
        <span className="mono whitespace-nowrap text-[10.5px] text-text-3">
          {completed}/{rubric.length} axes
        </span>
      </div>

      <div
        className="grid gap-2 rounded-lg p-2.5 text-[10.5px] text-text-3 sm:grid-cols-3"
        style={{ border: '1px solid var(--line)', background: 'var(--bg-sunken)' }}
      >
        <span>Implementation: {STATUS_LABELS[implementationStatus]}</span>
        <span>Design note: {NOTE_LABELS[noteStatus]}</span>
        <span>AI review: {AI_REVIEW_LABELS[aiReviewStatus]}</span>
      </div>

      <div className="grid grid-cols-8 gap-1" aria-label="Design-note completeness">
        {rubric.map(({ field, label }) => (
          <span
            key={field}
            title={label}
            className="h-1.5 rounded-full"
            style={{
              background: fields[field]?.trim()
                ? 'var(--accent)'
                : 'var(--line-strong)',
            }}
          />
        ))}
      </div>

      <details className="text-xs text-text-3">
        <summary className="cursor-pointer font-medium text-text-2">
          Rubric and example
        </summary>
        <div className="mt-2 space-y-2 leading-relaxed">
          <p>
            Address only the dimensions that materially affect this design; concise notes are valid.
          </p>
          <p>
            Example: the API gateway validates requests, while a run-scoped scheduler owns mutable state and emits durable checkpoints before external effects.
          </p>
        </div>
      </details>

      <div className="space-y-3">
        {rubric.map(({ field, label }) => (
          <label key={field} className="block space-y-1.5">
            <span className="flex items-center gap-1.5 text-xs font-medium text-text-2">
              {fields[field]?.trim()
                ? <Check className="h-3 w-3 text-accent" />
                : <CircleDashed className="h-3 w-3 text-text-3" />}
              {label}
            </span>
            <textarea
              aria-label={label}
              value={fields[field] ?? ''}
              onChange={(event) => updateField(field, event.target.value)}
              rows={3}
              maxLength={4_000}
              className="w-full resize-y rounded-lg px-3 py-2 text-sm leading-relaxed text-text outline-none focus:border-accent"
              style={{
                border: '1px solid var(--line)',
                background: 'var(--bg-sunken)',
              }}
            />
          </label>
        ))}
      </div>

      <div className="flex items-center justify-between gap-3">
        <p className="text-[11px] text-text-3">
          Saved notes are versioned separately from code submissions.
        </p>
        <div className="flex flex-wrap justify-end gap-2">
          <Button
            onClick={save}
            disabled={noteStatus === 'saving'}
            className="whitespace-nowrap"
          >
            <Save className="h-3.5 w-3.5" />
            Save design note
          </Button>
          <Button
            onClick={requestReview}
            disabled={
              aiReviewStatus === 'reviewing'
              || !hasDesignNoteContent(fields)
            }
            className="whitespace-nowrap"
          >
            <Sparkles className="h-3.5 w-3.5" />
            Review design note
          </Button>
        </div>
      </div>

      {review ? (
        <div
          className="rounded-lg p-3 text-sm text-text-2"
          style={{ border: '1px solid var(--accent-line)', background: 'var(--accent-wash)' }}
        >
          <div className="mono mb-2 text-[10.5px] uppercase tracking-[0.1em] text-accent">
            Advisory AI review
          </div>
          <MarkdownContent content={review} />
        </div>
      ) : reviewError ? (
        <p className="text-xs text-hard">
          {reviewError}
        </p>
      ) : null}
    </section>
  );
}
