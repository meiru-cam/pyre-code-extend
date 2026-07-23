'use client';

import { useEffect, useState } from 'react';
import { Code2, Loader2, ServerCog, Sparkles } from 'lucide-react';

import { Button } from '@/components/ui/Button';
import { MarkdownContent } from '@/components/workspace/MarkdownContent';
import { useLocale } from '@/context/LocaleContext';
import {
  buildAiHelpRequest,
  type AiHelpMode,
} from '@/lib/aiHelp';
import { scanForExternalAiSecrets } from '@/lib/secretBoundary';
import type { AiHelpResponse, Problem } from '@/lib/types';
import { useProblemStore } from '@/store/problemStore';

interface AIHelpTabProps {
  problem: Problem;
}

export function AIHelpTab({ problem }: AIHelpTabProps) {
  const { locale, t } = useLocale();
  const [serverConfigured, setServerConfigured] = useState<boolean | null>(null);
  const {
    currentCode,
    aiHelpQuery,
    setAiHelpQuery,
    aiHelpResponse,
    setAiHelpResponse,
    aiHelpError,
    setAiHelpError,
    aiHelpLoading,
    setAiHelpLoading,
  } = useProblemStore();

  useEffect(() => {
    let active = true;
    fetch('/api/ai-help/status')
      .then((response) => response.json())
      .then((data: { configured?: unknown }) => {
        if (active) setServerConfigured(data.configured === true);
      })
      .catch(() => {
        if (active) setServerConfigured(false);
      });
    return () => {
      active = false;
    };
  }, []);

  async function requestHelp(mode: AiHelpMode) {
    if (!serverConfigured) {
      setAiHelpError(t('aiHelpMissingConfig'));
      return;
    }
    const body = buildAiHelpRequest({
      problem,
      mode,
      query: aiHelpQuery,
      currentCode,
      locale,
    });
    if (scanForExternalAiSecrets(body).blocked) {
      setAiHelpError(t('aiHelpSecretDetected'));
      return;
    }

    setAiHelpLoading(true);
    setAiHelpError(null);
    setAiHelpResponse(null);
    try {
      const response = await fetch('/api/ai-help', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      const data = await response.json() as Partial<AiHelpResponse> & {
        code?: string;
      };
      if (!response.ok || typeof data.guidance !== 'string') {
        if (data.code === 'secret_detected') {
          setAiHelpError(t('aiHelpSecretDetected'));
        } else if (data.code === 'missing_configuration') {
          setAiHelpError(t('aiHelpMissingConfig'));
        } else {
          setAiHelpError(t('aiHelpRequestFailed'));
        }
        return;
      }
      setAiHelpResponse(data.guidance);
    } catch {
      setAiHelpError(t('aiHelpRequestFailed'));
    } finally {
      setAiHelpLoading(false);
    }
  }

  const disabled = aiHelpLoading || !aiHelpQuery.trim();

  return (
    <div className="p-4 space-y-4 overflow-y-auto h-full">
      <div className="rounded-xl p-3 space-y-2 bg-bg-elev border border-line">
        <div className="text-xs text-text-2 flex items-center gap-1.5">
          <Sparkles className="w-3.5 h-3.5 text-accent" />
          {t('aiHelpSafetyNote')}
        </div>
        <p className="text-xs leading-relaxed text-text-3">
          {t('aiHelpDataBoundary')}
        </p>
      </div>

      <label className="block space-y-1.5">
        <span className="text-xs font-medium text-text-2">
          {t('aiHelpQuestion')}
        </span>
        <textarea
          value={aiHelpQuery}
          onChange={(event) => setAiHelpQuery(event.target.value)}
          className="w-full rounded-lg px-3 py-2.5 text-sm text-text outline-none resize-y bg-bg-elev border border-line focus:border-line-strong"
          rows={4}
          maxLength={4_000}
          placeholder={t('aiHelpQuestionPlaceholder')}
        />
      </label>

      <div className="grid grid-cols-2 gap-2">
        <Button
          onClick={() => requestHelp('question')}
          disabled={disabled}
        >
          {aiHelpLoading
            ? <Loader2 className="w-3.5 h-3.5 animate-spin" />
            : <Sparkles className="w-3.5 h-3.5" />}
          {aiHelpLoading ? t('aiHelpGenerating') : t('aiHelpAskQuestion')}
        </Button>
        <Button
          variant="secondary"
          onClick={() => requestHelp('review_code')}
          disabled={disabled || !currentCode.trim()}
        >
          <Code2 className="w-3.5 h-3.5" />
          {t('aiHelpReviewCode')}
        </Button>
      </div>

      {serverConfigured && (
        <div className="flex items-center gap-1.5 text-xs text-text-3">
          <ServerCog className="w-3.5 h-3.5" />
          {t('aiHelpServerConfigured')}
        </div>
      )}

      {serverConfigured === false && (
        <div className="rounded-xl px-4 py-3 text-sm text-hard border border-line bg-bg-elev">
          {t('aiHelpMissingConfig')}
        </div>
      )}

      {aiHelpError && (
        <div
          role="alert"
          className="rounded-xl px-4 py-3 text-sm text-hard whitespace-pre-wrap border border-line bg-bg-elev"
        >
          {aiHelpError}
        </div>
      )}

      {aiHelpResponse ? (
        <div className="rounded-xl px-4 py-4 border border-line bg-bg-elev">
          <div className="mb-2 text-xs font-medium uppercase tracking-wide text-text-3">
            {t('aiHelpResponseTitle')}
          </div>
          <div className="text-sm text-text-2">
            <MarkdownContent content={aiHelpResponse} />
          </div>
        </div>
      ) : !aiHelpError && (
        <div className="rounded-xl px-4 py-6 text-sm text-text-3 border border-dashed border-line">
          {t('aiHelpEmpty')}
        </div>
      )}
    </div>
  );
}
