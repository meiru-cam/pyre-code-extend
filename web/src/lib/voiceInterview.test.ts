import { describe, expect, it } from 'vitest';

import { emptyInterviewRecord, type InterviewRecord } from '@/lib/interviewAnswer';
import type { InterviewQuestion, Problem, SubmissionResult } from '@/lib/types';
import {
  MAX_SNAPSHOT_CHARS,
  RECORD_ANSWER_TOOL,
  TranscriptBuffer,
  buildVoiceInterviewSession,
  codeSnapshotUpdate,
  recordSpokenAnswer,
  resultUpdate,
  transcriptText,
} from '@/lib/voiceInterview';

const questions: InterviewQuestion[] = [
  { stage: 'concept', question: 'What is QK norm for?' },
  { stage: 'deep_dive', question: 'Which axis do you normalize over?' },
  { stage: 'tradeoffs', question: 'QK norm versus logit soft-capping?' },
];

const problem = {
  id: 'qk_norm',
  title: 'QK Norm',
  functionName: 'qk_norm',
  descriptionEn: 'Normalize queries and keys before the dot product.',
} as Problem;

const LONG = 'It keeps attention logits bounded so training stays stable.';

describe('buildVoiceInterviewSession', () => {
  it('lists every question by index and names the record tool', () => {
    const { prompt, firstMessage } = buildVoiceInterviewSession(problem, questions, false);
    expect(prompt).toContain('0. [Opening] What is QK norm for?');
    expect(prompt).toContain('2. [Tradeoffs] QK norm versus logit soft-capping?');
    expect(prompt).toContain(RECORD_ANSWER_TOOL);
    expect(prompt).toContain('Normalize queries and keys');
    expect(firstMessage).toContain('What is QK norm for?');
  });

  it('skips the opening questions when the editor is already open', () => {
    const { prompt, firstMessage } = buildVoiceInterviewSession(problem, questions, true);
    expect(prompt).toContain('already done');
    expect(firstMessage).not.toContain('What is QK norm for?');
  });
});

describe('transcriptText', () => {
  it('drops voice-style tags and keeps the words', () => {
    expect(transcriptText('[calm] Good start.  [warmly] Go on.')).toBe('Good start. Go on.');
    expect(transcriptText('x[0] stays')).toBe('x[0] stays');
    expect(transcriptText('[calm] ')).toBe('');
  });
});

describe('TranscriptBuffer', () => {
  it('joins spoken turns and empties on take', () => {
    const buffer = new TranscriptBuffer();
    buffer.push(' first part ');
    buffer.push('');
    buffer.push('second part');
    expect(buffer.take()).toBe('first part second part');
    expect(buffer.take()).toBe('');
  });
});

describe('recordSpokenAnswer', () => {
  const draft = (): InterviewRecord => emptyInterviewRecord(questions);
  const now = new Date('2026-10-20T10:00:00Z');

  it('fills one answer and reports the questions still open', () => {
    const outcome = recordSpokenAnswer(questions, draft(), 0, LONG, now);
    expect(outcome.record.answers[0]).toBe(LONG);
    expect(outcome.record.status).toBe('draft');
    expect(outcome.record.updatedAt).toBe(now.toISOString());
    expect(outcome.message).toContain('Still open: questions 1');
  });

  it('commits the record once every gated answer is long enough', () => {
    const first = recordSpokenAnswer(questions, draft(), 0, LONG, now).record;
    const outcome = recordSpokenAnswer(questions, first, 1, 'Over the head dimension, per head.', now);
    expect(outcome.record.status).toBe('answered');
    expect(outcome.message).toContain('editor is open');
  });

  describe('with interview levels', () => {
    const five: InterviewQuestion[] = [
      { stage: 'concept', question: 'Opening one?' },
      { stage: 'concept', question: 'Opening two?' },
      { stage: 'deep_dive', question: 'Deep one?' },
      { stage: 'deep_dive', question: 'Deep two?' },
      { stage: 'tradeoffs', question: 'Tradeoff?' },
    ];

    it('tells the interviewer to ask only the questions easy requires', () => {
      const { prompt, firstMessage } = buildVoiceInterviewSession(problem, five, false, 'easy');
      expect(prompt).toContain('Ask questions 0, 2 one at a time');
      expect(firstMessage).toContain('Opening one?');
    });

    it('opens the editor once the easy questions are answered and records the level', () => {
      const first = recordSpokenAnswer(five, emptyInterviewRecord(five), 0, LONG, now, 'easy').record;
      const outcome = recordSpokenAnswer(five, first, 2, LONG, now, 'easy');
      expect(outcome.record.status).toBe('answered');
      expect(outcome.record.level).toBe('easy');
      expect(outcome.message).toContain('editor is open');
    });

    it('never commits a spoken answer in learn, which has no gate', () => {
      const outcome = recordSpokenAnswer(five, emptyInterviewRecord(five), 4, LONG, now, 'learn');
      expect(outcome.record.status).toBe('draft');
      expect(outcome.record.answers[4]).toBe(LONG);
      expect(outcome.record.level).toBeUndefined();
      expect(outcome.message).toBe('Answer 4 recorded.');
    });

    it('does not treat a question outside the level as gated', () => {
      const outcome = recordSpokenAnswer(five, emptyInterviewRecord(five), 1, 'Short.', now, 'easy');
      expect(outcome.message).not.toContain('too short');
    });
  });

  it('asks for more when a gated answer is too short', () => {
    const outcome = recordSpokenAnswer(questions, draft(), 0, 'Stability.', now);
    expect(outcome.record.status).toBe('draft');
    expect(outcome.message).toContain('too short');
  });

  it('appends to an answer the learner already typed', () => {
    const typed = { ...draft(), answers: ['Typed start.', '', ''] };
    const outcome = recordSpokenAnswer(questions, typed, '0', 'Then spoken detail.', now);
    expect(outcome.record.answers[0]).toBe('Typed start.\nThen spoken detail.');
  });

  it('records tradeoff answers without changing a committed status', () => {
    const answered: InterviewRecord = { status: 'answered', answers: [LONG, LONG, ''], updatedAt: '' };
    const outcome = recordSpokenAnswer(questions, answered, 2, 'Soft-capping is cheaper.', now);
    expect(outcome.record.status).toBe('answered');
    expect(outcome.record.answers[2]).toBe('Soft-capping is cheaper.');
  });

  it('keeps a voice mark from an earlier session, so the gate stays open', () => {
    const voiced: InterviewRecord = { status: 'draft', answers: ['', '', ''], updatedAt: '', voice: true };
    const outcome = recordSpokenAnswer(questions, voiced, 2, 'Soft-capping is cheaper.', now);
    expect(outcome.record.voice).toBe(true);
    expect(outcome.record.answers[2]).toBe('Soft-capping is cheaper.');
  });

  it('keeps a skip as a skip', () => {
    const skipped: InterviewRecord = { status: 'skipped', answers: ['', '', ''], updatedAt: '' };
    expect(recordSpokenAnswer(questions, skipped, 0, LONG, now).record.status).toBe('skipped');
  });

  it('rejects an index outside the question list', () => {
    const record = draft();
    const outcome = recordSpokenAnswer(questions, record, 7, LONG, now);
    expect(outcome.record).toBe(record);
    expect(outcome.message).toContain('No question has index 7');
  });
});

describe('codeSnapshotUpdate', () => {
  it('sends changed code and skips unchanged code', () => {
    expect(codeSnapshotUpdate('x = 1', null)).toContain('x = 1');
    expect(codeSnapshotUpdate('x = 1', 'x = 1')).toBeNull();
  });

  it('truncates long code', () => {
    const update = codeSnapshotUpdate('a'.repeat(MAX_SNAPSHOT_CHARS + 50), null)!;
    expect(update).toContain('[snapshot truncated]');
    expect(update.length).toBeLessThan(MAX_SNAPSHOT_CHARS + 200);
  });

  it('withholds code that looks like it holds a credential', () => {
    const update = codeSnapshotUpdate(`key = "sk-${'a'.repeat(30)}"`, null)!;
    expect(update).toContain('withheld');
    expect(update).not.toContain('sk-');
  });
});

describe('resultUpdate', () => {
  const result: SubmissionResult = {
    passed: 1,
    total: 3,
    allPassed: false,
    totalTimeMs: 12,
    results: [
      { name: 'shape', passed: true, execTimeMs: 1 },
      { name: 'unit norm', passed: false, execTimeMs: 1, errorLine: 4, error: 'AssertionError: norm 2.0\ntrace' },
      { name: 'grad', passed: false, execTimeMs: 1, behavior: 'Gradient is None' },
    ],
  };

  it('summarizes a run with failing test names and first error lines', () => {
    const text = resultUpdate(result, 'run');
    expect(text).toContain('Ran tests: 1/3 passed.');
    expect(text).toContain('- unit norm failed (line 4): AssertionError: norm 2.0');
    expect(text).not.toContain('trace');
    expect(text).toContain('- grad failed: Gradient is None');
  });

  it('moves the interviewer to tradeoffs after a submit', () => {
    expect(resultUpdate(result, 'submit')).toContain('Tradeoffs');
  });

  it('reports an error that stopped the tests', () => {
    expect(resultUpdate({ ...result, error: 'SyntaxError' }, 'run')).toContain('SyntaxError');
  });

  it('withholds error text that echoes a credential but keeps the counts', () => {
    const leaky = { ...result, error: `NameError near api_key = "${'k'.repeat(30)}"` };
    const text = resultUpdate(leaky, 'submit');
    expect(text).toContain('withheld');
    expect(text).not.toContain('kkkk');
    expect(text).toContain('Tradeoffs');
  });
});
