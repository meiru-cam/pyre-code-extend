// @vitest-environment jsdom

import React from 'react';
import { cleanup, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';

vi.mock('@/context/LocaleContext', () => ({
  useLocale: () => ({
    locale: 'en',
    t: (key: string) => key,
  }),
}));

vi.mock('@/components/workspace/DesignNoteEditor', () => ({
  DesignNoteEditor: () => null,
}));

import { DescriptionTab } from '@/components/workspace/DescriptionTab';
import type { Problem } from '@/lib/types';

const problem: Problem = {
  id: 'two_hints',
  title: 'Two Hints',
  titleZh: 'Two Hints',
  difficulty: 'Medium',
  functionName: 'f',
  hint: '',
  hintZh: '',
  descriptionEn: 'Try it.',
  descriptionZh: 'Try it.',
  tests: [],
  hints: [
    { level: 1, kind: 'questions', content: 'Question-only nudge' },
    { level: 2, kind: 'analysis', content: 'Deeper analysis' },
  ],
};

describe('DescriptionTab hints', () => {
  afterEach(cleanup);

  it('reveals level 1 without revealing level 2, then keeps both independent', async () => {
    const user = userEvent.setup();
    render(React.createElement(DescriptionTab, { problem }));

    expect(screen.queryByText('Question-only nudge')).toBeNull();
    expect(screen.queryByText('Deeper analysis')).toBeNull();

    await user.click(screen.getByText('hint 1 · Guiding questions'));
    expect(screen.getByText('Question-only nudge')).toBeTruthy();
    expect(screen.queryByText('Deeper analysis')).toBeNull();

    await user.click(screen.getByText('hint 2 · Analysis'));
    expect(screen.getByText('Question-only nudge')).toBeTruthy();
    expect(screen.getByText('Deeper analysis')).toBeTruthy();

    await user.click(screen.getByText('hint 1 · Guiding questions'));
    expect(screen.queryByText('Question-only nudge')).toBeNull();
    expect(screen.getByText('Deeper analysis')).toBeTruthy();
  });
});

describe('DescriptionTab interview gate', () => {
  afterEach(cleanup);

  const withTradeoffs: Problem = {
    ...problem,
    proConAnalysis: { pros: ['Reference pro'], cons: ['Reference con'] },
  };

  function renderWithGate(status: 'draft' | 'answered' | 'skipped') {
    const onChange = vi.fn();
    const record = {
      status,
      answers: ['', '', ''],
      updatedAt: '',
    };
    render(React.createElement(DescriptionTab, {
      problem: withTradeoffs,
      interview: { record, onChange },
    }));
    return onChange;
  }

  it('hides hints and reference tradeoffs until the learner answers', () => {
    renderWithGate('draft');
    expect(screen.getByText('Interview first')).toBeTruthy();
    expect(screen.queryByText('hint 1 · Guiding questions')).toBeNull();
    expect(screen.queryByText('Reference pro')).toBeNull();
    expect((screen.getByText('Lock in answers') as HTMLButtonElement).disabled).toBe(true);
  });

  it('records a skip explicitly', async () => {
    const user = userEvent.setup();
    const onChange = renderWithGate('draft');
    await user.click(screen.getByText('Skip (recorded)'));
    expect(onChange).toHaveBeenCalledWith(expect.objectContaining({ status: 'skipped' }));
  });

  it('shows authored questions and holds tradeoffs back until unlock', () => {
    render(React.createElement(DescriptionTab, {
      problem: {
        ...withTradeoffs,
        interviewQuestions: [
          { stage: 'concept', question: 'Why divide by T?' },
          { stage: 'deep_dive', question: 'How do you detach the teacher?' },
          { stage: 'tradeoffs', question: 'What does teacher forcing miss?' },
        ],
      },
      interview: { record: { status: 'draft', answers: ['', '', ''], updatedAt: '' }, onChange: vi.fn() },
    }));
    expect(screen.getByText('Why divide by T?')).toBeTruthy();
    expect(screen.getByText('How do you detach the teacher?')).toBeTruthy();
    expect(screen.queryByText('What does teacher forcing miss?')).toBeNull();
  });

  it('shows hints and reference tradeoffs once unlocked', () => {
    renderWithGate('answered');
    expect(screen.getByText('hint 1 · Guiding questions')).toBeTruthy();
    expect(screen.getByText('Reference pro')).toBeTruthy();
    expect(screen.getByText('Interview')).toBeTruthy();
  });
});
