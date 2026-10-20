'use client';

import { Lightbulb, Lock, MessageSquareQuote } from 'lucide-react';

import { Button } from '@/components/ui/Button';
import {
  GATED_STAGES,
  INTERVIEW_STAGE_LABELS,
  MIN_ANSWER_LENGTH,
  isInterviewUnlocked,
  missingGatedAnswers,
  requiredQuestionIndices,
  type InterviewLevel,
  type InterviewRecord,
} from '@/lib/interviewAnswer';
import type { InterviewQuestion, InterviewStage } from '@/lib/types';

interface InterviewPanelProps {
  questions: readonly InterviewQuestion[];
  record: InterviewRecord;
  onChange: (record: InterviewRecord) => void;
  /** False hides the tradeoffs stage, e.g. while a multi-part exercise still has locked parts. */
  showTradeoffs?: boolean;
  /** Decides which questions the gate requires; learn has no gate and opens every hint. */
  level?: InterviewLevel;
}

const STAGE_ORDER: readonly InterviewStage[] = ['concept', 'deep_dive', 'tradeoffs'];

export function InterviewPanel({
  questions, record, onChange, showTradeoffs = true, level = 'hard',
}: InterviewPanelProps) {
  const learn = level === 'learn';
  const answered = isInterviewUnlocked(record);
  const unlocked = answered || learn;
  const required = requiredQuestionIndices(questions, level);
  const missing = missingGatedAnswers(questions, record.answers, level);
  // Once unlocked, the questions the learner was asked are the ones from the level they locked in at.
  // An answer from before levels existed covered every question, so it counts as hard.
  const asked = requiredQuestionIndices(
    questions,
    record.level ?? (record.status === 'answered' ? 'hard' : level),
  );
  const optionalAfter = (askedIndices: readonly number[]) => questions
    .map((question, index) => ({ question, index }))
    .filter(({ question, index }) => GATED_STAGES.includes(question.stage) && !askedIndices.includes(index))
    .map(({ index }) => index);
  const optionalFold = (indices: readonly number[]) => indices.length > 0 && (
    <details className="rounded-lg px-3 py-2" style={{ border: '1px solid var(--line)' }}>
      <summary className="cursor-pointer select-none text-[13px] text-text-2">
        More questions (optional) · {indices.length}
      </summary>
      <div className="mt-3">{questionsView(indices, GATED_STAGES)}</div>
    </details>
  );

  const update = (answers: string[], status = record.status) => {
    onChange({
      status,
      answers,
      updatedAt: new Date().toISOString(),
      // The level is stamped once, when the learner starts coding; later edits keep it, so
      // changing the level afterwards never relocks the exercise.
      ...(status !== record.status ? { level } : record.level ? { level: record.level } : {}),
    });
  };
  const updateAnswer = (index: number, value: string) => {
    update(record.answers.map((answer, i) => (i === index ? value : answer)));
  };

  const questionsView = (indices: readonly number[], stages: readonly InterviewStage[]) => (
    <div className="space-y-5">
      {stages.map((stage) => {
        const entries = indices
          .map((index) => ({ question: questions[index], index }))
          .filter(({ question }) => question.stage === stage);
        if (entries.length === 0) return null;
        return (
          <div key={stage} className="space-y-3">
            <h3 className="mono text-[11px] font-semibold uppercase tracking-[0.1em] text-text-2">
              {INTERVIEW_STAGE_LABELS[stage]}
            </h3>
            {entries.map(({ question, index }) => {
              const answer = record.answers[index] ?? '';
              const short = !unlocked && missing.includes(index) && answer.length > 0;
              return (
                <div key={index} className="space-y-1.5">
                  <span className="block text-sm leading-relaxed text-text">{question.question}</span>
                  {question.hint && (
                    <details open={learn} className="group text-[13px] leading-relaxed text-text-2">
                      <summary className="inline-flex cursor-pointer select-none items-center gap-1 text-accent">
                        <Lightbulb className="h-3.5 w-3.5" />
                        <span className="group-open:hidden">Show hint</span>
                        <span className="hidden group-open:inline">Hide hint</span>
                      </summary>
                      <p className="mt-1 rounded-md px-2.5 py-1.5" style={{ background: 'var(--bg-sunken)' }}>
                        {question.hint}
                      </p>
                    </details>
                  )}
                  <textarea
                    aria-label={question.question}
                    value={answer}
                    onChange={(event) => updateAnswer(index, event.target.value)}
                    rows={3}
                    maxLength={4_000}
                    className="w-full resize-y rounded-lg px-3 py-2 text-sm leading-relaxed text-text outline-none focus:border-accent"
                    style={{ border: '1px solid var(--line)', background: 'var(--bg-sunken)' }}
                  />
                  {short && (
                    <span className="block text-xs text-text-2">
                      At least {MIN_ANSWER_LENGTH} characters.
                    </span>
                  )}
                </div>
              );
            })}
          </div>
        );
      })}
    </div>
  );

  if (unlocked) {
    // Tradeoff questions follow the code, as they would in an interview.
    const stages = showTradeoffs ? STAGE_ORDER : GATED_STAGES;
    return (
      <details
        open={answered || learn}
        className="rounded-lg p-3.5"
        style={{ border: '1px solid var(--line)', background: 'var(--bg-elev)' }}
      >
        <summary className="flex cursor-pointer items-center gap-2 text-sm font-medium text-text">
          <MessageSquareQuote className="h-4 w-4" />
          Interview
          <span className="mono ml-auto text-[10.5px] uppercase tracking-[0.1em] text-text-2">
            {!answered ? 'Optional · Learn' : record.status === 'answered' ? 'Answered' : 'Skipped'}
          </span>
        </summary>
        <p className="mt-2 mb-4 text-[13px] text-text-2">
          {learn && !answered
            ? 'Questions to think through while you read and code; each hint names what to cover. Answers are optional and save automatically.'
            : showTradeoffs
              ? 'Tradeoff questions are open now. Compare with the reference pros and cons below. Edits save automatically.'
              : 'Tradeoff questions open once every part is unlocked. Edits save automatically.'}
        </p>
        {learn ? (
          questionsView(questions.map((_, index) => index), stages)
        ) : (
          <div className="space-y-5">
            {questionsView(
              questions
                .map((question, index) => ({ question, index }))
                .filter(({ question, index }) => asked.includes(index) || question.stage === 'tradeoffs')
                .map(({ index }) => index),
              stages,
            )}
            {optionalFold(optionalAfter(asked))}
          </div>
        )}
      </details>
    );
  }

  return (
    <section
      className="space-y-4 rounded-lg p-4"
      style={{
        border: '1px solid var(--accent-line)',
        background: 'color-mix(in oklab, var(--accent) 4%, var(--bg))',
      }}
    >
      <div className="flex items-center gap-2">
        <Lock className="h-4 w-4 text-accent" />
        <h2 className="text-sm font-semibold text-text">
          Interview first · {required.length} {required.length === 1 ? 'question' : 'questions'}
        </h2>
      </div>
      <p className="text-[13px] leading-relaxed text-text-2">
        Answer out loud or in writing; each question has a hint if you are stuck. Coding hints, reference pros and cons, the solution, AI help and the editor unlock when you start coding.
      </p>
      {questionsView(required, GATED_STAGES)}
      {optionalFold(optionalAfter(required))}
      <div className="flex items-center gap-3">
        {/* One way in: with the required answers it counts as answered, without them as a skip. */}
        <Button
          size="sm"
          onClick={() => update(record.answers, missing.length === 0 ? 'answered' : 'skipped')}
        >
          Start coding
        </Button>
        {missing.length > 0 && (
          <span className="text-xs text-text-2">Unanswered questions are recorded as skipped.</span>
        )}
      </div>
    </section>
  );
}

export function InterviewLockedNotice() {
  return (
    <div className="flex h-full min-h-[160px] flex-col items-center justify-center gap-2 p-6 text-center">
      <Lock className="h-5 w-5 text-text-2" />
      <p className="text-sm text-text-2">Locked until you answer the interview questions.</p>
      <p className="text-[13px] text-text-2">Open the Description tab and press Start coding.</p>
    </div>
  );
}
