// @vitest-environment jsdom

import { beforeEach, describe, expect, it } from 'vitest';

import {
  FALLBACK_INTERVIEW_QUESTIONS,
  emptyInterviewRecord,
  interviewQuestionsFor,
  isInterviewUnlocked,
  loadInterviewRecord,
  missingGatedAnswers,
  saveInterviewRecord,
} from '@/lib/interviewAnswer';
import type { InterviewQuestion, Problem } from '@/lib/types';

const questions: InterviewQuestion[] = [
  { stage: 'concept', question: 'Why divide logits by T?' },
  { stage: 'deep_dive', question: 'How do you stop teacher gradients?' },
  { stage: 'tradeoffs', question: 'What does teacher forcing miss?' },
];

const long = 'An answer comfortably past the minimum.';

describe('interview answers', () => {
  beforeEach(() => window.localStorage.clear());

  it('uses authored questions and falls back to the generic ones', () => {
    const base = { id: 'x' } as Problem;
    expect(interviewQuestionsFor(base)).toBe(FALLBACK_INTERVIEW_QUESTIONS);
    expect(interviewQuestionsFor({ ...base, interviewQuestions: questions })).toBe(questions);
  });

  it('stays locked with no record or only a draft', () => {
    expect(isInterviewUnlocked(null)).toBe(false);
    expect(isInterviewUnlocked(emptyInterviewRecord(questions))).toBe(false);
  });

  it('unlocks after an answer or an explicit skip', () => {
    expect(isInterviewUnlocked({ status: 'answered', answers: [], updatedAt: '' })).toBe(true);
    expect(isInterviewUnlocked({ status: 'skipped', answers: [], updatedAt: '' })).toBe(true);
  });

  it('gates on opening and deep-dive answers but not tradeoffs', () => {
    expect(missingGatedAnswers(questions, [long, long, ''])).toEqual([]);
    expect(missingGatedAnswers(questions, ['short', '   ', long])).toEqual([0, 1]);
  });

  it('round-trips by task and contract version', () => {
    const record = { status: 'answered' as const, answers: [long, long, ''], updatedAt: 'now' };
    saveInterviewRecord('masked_kd_kl', 1, record);
    expect(loadInterviewRecord('masked_kd_kl', 1, questions)).toEqual(record);
    expect(loadInterviewRecord('masked_kd_kl', 2, questions)).toBeNull();
    expect(loadInterviewRecord('dpo_loss', 1, questions)).toBeNull();
  });

  it('realigns answers when questions are added and relocks uncovered gates', () => {
    saveInterviewRecord('t', 1, { status: 'answered', answers: [long, long], updatedAt: '' });
    const grown: InterviewQuestion[] = [
      ...questions.slice(0, 2),
      { stage: 'deep_dive', question: 'New gated question?' },
      questions[2],
    ];
    const loaded = loadInterviewRecord('t', 1, grown);
    expect(loaded?.answers).toEqual([long, long, '', '']);
    expect(loaded?.status).toBe('draft');
  });

  it('ignores malformed storage', () => {
    window.localStorage.setItem('pyre-code-interview-answer:v2:t:1', '{not json');
    expect(loadInterviewRecord('t', 1, questions)).toBeNull();
  });
});
