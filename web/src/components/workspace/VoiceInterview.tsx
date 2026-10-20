'use client';

import { useEffect, useRef, useState } from 'react';
import { ConversationProvider, useConversation } from '@elevenlabs/react';
import { Mic, PhoneOff } from 'lucide-react';

import { Button } from '@/components/ui/Button';
import { useLocale } from '@/context/LocaleContext';
import type { InterviewRecord } from '@/lib/interviewAnswer';
import type { InterviewQuestion, Problem } from '@/lib/types';
import {
  RECORD_ANSWER_TOOL,
  SNAPSHOT_IDLE_MS,
  TranscriptBuffer,
  buildVoiceInterviewSession,
  codeSnapshotUpdate,
  recordSpokenAnswer,
  resultUpdate,
} from '@/lib/voiceInterview';
import { useProblemStore } from '@/store/problemStore';

interface VoiceInterviewProps {
  problem: Problem;
  questions: readonly InterviewQuestion[];
  record: InterviewRecord;
  onRecordChange: (record: InterviewRecord) => void;
  /** True once the editor is open, from the gate or practice mode. */
  editorOpen: boolean;
}

/** Topbar control for a spoken interview. Renders nothing unless the server is configured. */
export function VoiceInterview(props: VoiceInterviewProps) {
  const [configured, setConfigured] = useState(false);

  useEffect(() => {
    fetch('/api/voice-interview/status')
      .then((r) => r.json())
      .then((d: { configured?: boolean }) => setConfigured(Boolean(d.configured)))
      .catch(() => setConfigured(false));
  }, []);

  if (!configured) return null;
  return (
    <ConversationProvider>
      <VoiceInterviewControl {...props} />
    </ConversationProvider>
  );
}

type Line = { role: 'user' | 'agent'; text: string };

function VoiceInterviewControl({ problem, questions, record, onRecordChange, editorOpen }: VoiceInterviewProps) {
  const { t } = useLocale();
  const [consentOpen, setConsentOpen] = useState(false);
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastLine, setLastLine] = useState<Line | null>(null);

  // Client tools and callbacks outlive renders, so they read the latest values through refs.
  const bufferRef = useRef(new TranscriptBuffer());
  // Speech counts toward an answer only before the editor opens and after Submit.
  const answeringRef = useRef(!editorOpen);
  const recordRef = useRef(record);
  const onRecordChangeRef = useRef(onRecordChange);
  const lastSnapshotRef = useRef<string | null>(null);
  const mountedRef = useRef(true);

  const conversation = useConversation({
    onMessage: ({ role, message }) => {
      if (role === 'user' && answeringRef.current) bufferRef.current.push(message);
      setLastLine({ role, text: message });
    },
    // Provider error text is never shown; it can carry upstream details.
    onError: () => setError(t('voiceStartFailed')),
  });
  const live = conversation.status === 'connected';
  const conversationRef = useRef(conversation);

  useEffect(() => {
    recordRef.current = record;
    onRecordChangeRef.current = onRecordChange;
    conversationRef.current = conversation;
  });

  useEffect(() => {
    mountedRef.current = true;
    return () => { mountedRef.current = false; };
  }, []);

  const sendSnapshot = (code: string) => {
    const update = codeSnapshotUpdate(code, lastSnapshotRef.current);
    if (!update) return;
    lastSnapshotRef.current = code;
    conversationRef.current.sendContextualUpdate(update);
  };

  const start = async () => {
    setConsentOpen(false);
    setError(null);
    setStarting(true);
    try {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        stream.getTracks().forEach((track) => track.stop());
      } catch {
        setError(t('voiceMicDenied'));
        return;
      }
      const res = await fetch('/api/voice-interview/session', { method: 'POST' });
      const data = await res.json().catch(() => ({})) as { signedUrl?: string; error?: string };
      if (!res.ok || !data.signedUrl) {
        setError(data.error ?? t('voiceStartFailed'));
        return;
      }
      // The learner left the exercise while this was loading; a session now would have no End button.
      if (!mountedRef.current) return;
      const session = buildVoiceInterviewSession(problem, questions, editorOpen);
      bufferRef.current.clear();
      answeringRef.current = !editorOpen;
      lastSnapshotRef.current = null;
      // Connection failures after this point arrive through onError, not as a rejection.
      conversation.startSession({
        signedUrl: data.signedUrl,
        overrides: { agent: { prompt: { prompt: session.prompt }, firstMessage: session.firstMessage } },
        clientTools: {
          [RECORD_ANSWER_TOOL]: ({ question_index }: { question_index?: unknown }) => {
            const outcome = recordSpokenAnswer(
              questions, recordRef.current, question_index, bufferRef.current.take(),
            );
            recordRef.current = outcome.record;
            onRecordChangeRef.current(outcome.record);
            return outcome.message;
          },
        },
      });
    } catch {
      setError(t('voiceStartFailed'));
    } finally {
      setStarting(false);
    }
  };

  // Code: one snapshot after the editor goes quiet, only once the editor is open.
  const currentCode = useProblemStore((s) => s.currentCode);
  useEffect(() => {
    if (!live || !editorOpen) return;
    const timer = setTimeout(() => sendSnapshot(currentCode), SNAPSHOT_IDLE_MS);
    return () => clearTimeout(timer);
  }, [live, editorOpen, currentCode]);

  // Results: only a result that just arrived, with the graded code first so they read together.
  useEffect(() => {
    if (!live) return;
    return useProblemStore.subscribe((state, prev) => {
      const submitted = state.submissionResult !== null && state.submissionResult !== prev.submissionResult;
      const ran = state.runResult !== null && state.runResult !== prev.runResult
        && state.runResult !== state.submissionResult;
      const result = submitted ? state.submissionResult : ran ? state.runResult : null;
      if (!result) return;
      sendSnapshot(state.currentCode);
      // Tradeoff answers start fresh rather than with talk from the coding phase.
      if (submitted) {
        bufferRef.current.clear();
        answeringRef.current = true;
      }
      conversationRef.current.sendContextualUpdate(resultUpdate(result, submitted ? 'submit' : 'run'));
    });
  }, [live]);

  // Talk while coding is not an answer to anything; stop collecting it when the gate opens.
  useEffect(() => {
    if (!editorOpen) return;
    bufferRef.current.clear();
    answeringRef.current = false;
  }, [editorOpen]);

  const liveStyle = { border: '1px solid var(--accent-line)', background: 'var(--accent-wash)', color: 'var(--accent)' };
  const idleStyle = { border: '1px solid var(--line)', background: 'var(--bg-elev)', color: 'var(--text-2)' };

  return (
    <div className="relative">
      {live ? (
        <button
          onClick={() => conversation.endSession()}
          title={t('voiceEnd')}
          className="h-[30px] px-3 inline-flex items-center gap-1.5 rounded-lg text-[13px] cursor-pointer transition-[color,border-color] duration-150"
          style={liveStyle}
        >
          <span className={`w-1.5 h-1.5 rounded-full bg-accent ${conversation.isSpeaking ? 'animate-pulse' : ''}`} />
          {conversation.isSpeaking ? t('voiceSpeaking') : t('voiceListening')}
          <PhoneOff className="w-3.5 h-3.5 ml-1" />
        </button>
      ) : (
        <button
          onClick={() => setConsentOpen((open) => !open)}
          disabled={starting || conversation.status === 'connecting'}
          aria-expanded={consentOpen}
          title={t('voiceInterviewTitle')}
          className="h-[30px] px-3 inline-flex items-center gap-1.5 rounded-lg text-[13px] cursor-pointer hover:border-line-strong disabled:opacity-60 transition-[color,border-color] duration-150"
          style={idleStyle}
        >
          <Mic className="w-3.5 h-3.5" />
          {starting || conversation.status === 'connecting' ? t('voiceConnecting') : t('voiceInterview')}
        </button>
      )}

      {consentOpen && !live && (
        <div
          role="dialog"
          aria-label={t('voiceConsentTitle')}
          className="absolute right-0 top-[38px] z-30 w-[320px] space-y-3 rounded-lg p-4 shadow-lg"
          style={{ border: '1px solid var(--line)', background: 'var(--bg-elev)' }}
        >
          <h2 className="text-sm font-semibold text-text">{t('voiceConsentTitle')}</h2>
          <p className="text-[13px] leading-relaxed text-text-2">{t('voiceConsentBody')}</p>
          <div className="flex items-center gap-2">
            <Button size="sm" onClick={start}>{t('voiceConsentStart')}</Button>
            <Button size="sm" variant="ghost" onClick={() => setConsentOpen(false)}>{t('voiceConsentCancel')}</Button>
          </div>
        </div>
      )}

      {error && !live && (
        <p className="absolute right-0 top-[38px] z-20 w-[260px] rounded-md px-2.5 py-1.5 text-xs text-text-2" style={{ background: 'var(--bg-sunken)', border: '1px solid var(--line)' }}>
          {error}
        </p>
      )}

      {live && lastLine && (
        <div
          className="fixed bottom-4 left-4 z-30 max-w-[420px] rounded-lg px-3 py-2 text-[13px] leading-relaxed shadow-lg"
          style={{ border: '1px solid var(--line)', background: 'var(--bg-elev)' }}
        >
          <span className="mono mr-1.5 text-[10.5px] uppercase tracking-[0.1em] text-text-2">
            {lastLine.role === 'agent' ? t('voiceInterviewer') : t('voiceYou')}
          </span>
          <span className="text-text">{lastLine.text}</span>
        </div>
      )}
    </div>
  );
}
