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

import { DescriptionTab, splitAtDivider } from '@/components/workspace/DescriptionTab';
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
    expect(screen.getByText(/^Interview first/)).toBeTruthy();
    expect(screen.queryByText('hint 1 · Guiding questions')).toBeNull();
    expect(screen.queryByText('Reference pro')).toBeNull();
    expect(screen.getByText('Unanswered questions are recorded as skipped.')).toBeTruthy();
  });

  it('records starting without the answers as a skip', async () => {
    const user = userEvent.setup();
    const onChange = renderWithGate('draft');
    await user.click(screen.getByText('Start coding'));
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

describe('DescriptionTab interview levels', () => {
  afterEach(cleanup);

  const fiveQuestions: Problem = {
    ...problem,
    proConAnalysis: { pros: ['Reference pro'], cons: [] },
    interviewQuestions: [
      { stage: 'concept', question: 'Opening one?' },
      { stage: 'concept', question: 'Opening two?' },
      { stage: 'deep_dive', question: 'Deep one?' },
      { stage: 'deep_dive', question: 'Deep two?' },
      { stage: 'deep_dive', question: 'Deep three?' },
      { stage: 'tradeoffs', question: 'Tradeoff?' },
    ],
  };
  const long = 'An answer comfortably past the minimum.';

  it('asks easy for one opening and one deep dive, the rest folded as optional', async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    render(React.createElement(DescriptionTab, {
      problem: fiveQuestions,
      interview: {
        record: { status: 'draft', answers: [long, '', long, '', '', ''], updatedAt: '' },
        onChange,
        level: 'easy',
      },
    }));

    expect(screen.getByText('Interview first · 2 questions')).toBeTruthy();
    expect(screen.getByText('More questions (optional) · 3')).toBeTruthy();
    await user.click(screen.getByText('Start coding'));
    expect(onChange).toHaveBeenCalledWith(expect.objectContaining({ status: 'answered', level: 'easy' }));
  });

  it('folds the interview questions by default, before and after starting to code', () => {
    const locked = render(React.createElement(DescriptionTab, {
      problem: fiveQuestions,
      interview: { record: { status: 'draft', answers: [], updatedAt: '' }, onChange: vi.fn(), level: 'easy' },
    }));
    expect(screen.getByText('Opening one?').closest('details')!.open).toBe(false);
    expect(screen.getByText('Start coding')).toBeTruthy();
    locked.unmount();

    render(React.createElement(DescriptionTab, {
      problem: fiveQuestions,
      interview: {
        record: { status: 'answered', answers: [long, '', long, '', '', ''], updatedAt: '', level: 'easy' },
        onChange: vi.fn(),
        level: 'easy',
      },
    }));
    expect(screen.getByText('Opening one?').closest('details')!.open).toBe(false);
  });

  it('keeps the unasked questions folded after starting to code at easy', () => {
    const { container } = render(React.createElement(DescriptionTab, {
      problem: fiveQuestions,
      interview: {
        record: {
          status: 'answered', answers: [long, '', long, '', '', ''], updatedAt: '', level: 'easy',
        },
        onChange: vi.fn(),
        level: 'easy',
      },
    }));

    const fold = screen.getByText('More questions (optional) · 3').closest('details')!;
    expect(fold.open).toBe(false);
    for (const unasked of ['Opening two?', 'Deep two?', 'Deep three?']) {
      expect(fold.contains(screen.getByText(unasked))).toBe(true);
    }
    for (const shown of ['Opening one?', 'Deep one?', 'Tradeoff?']) {
      expect(fold.contains(screen.getByText(shown))).toBe(false);
    }
    expect(container.textContent).toContain('Answered');
  });

  it('keeps the locked-in level when an answer is edited at another level', async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    render(React.createElement(DescriptionTab, {
      problem: fiveQuestions,
      interview: {
        record: { status: 'answered', answers: [long, '', long, '', '', ''], updatedAt: '', level: 'easy' },
        onChange,
        level: 'hard',
      },
    }));
    await user.type(screen.getByLabelText('Opening one?'), '!');
    expect(onChange).toHaveBeenLastCalledWith(expect.objectContaining({ status: 'answered', level: 'easy' }));
  });

  it('shows every question for an answer from before levels existed', () => {
    render(React.createElement(DescriptionTab, {
      problem: fiveQuestions,
      interview: {
        record: { status: 'answered', answers: [long, long, long, long, long, ''], updatedAt: '' },
        onChange: vi.fn(),
        level: 'easy',
      },
    }));
    expect(screen.queryByText(/More questions \(optional\)/)).toBeNull();
    expect(screen.getByText('Deep three?')).toBeTruthy();
  });

  it('lists every question in learn even after a skip', () => {
    render(React.createElement(DescriptionTab, {
      problem: fiveQuestions,
      interview: {
        record: { status: 'skipped', answers: ['', '', '', '', '', ''], updatedAt: '' },
        onChange: vi.fn(),
        level: 'learn',
      },
    }));
    expect(screen.queryByText(/More questions \(optional\)/)).toBeNull();
    expect(screen.getByText('Opening two?')).toBeTruthy();
  });

  it('opens learn with hints and reference material already showing', () => {
    render(React.createElement(DescriptionTab, {
      problem: fiveQuestions,
      interview: {
        record: { status: 'draft', answers: ['', '', '', '', '', ''], updatedAt: '' },
        onChange: vi.fn(),
        level: 'learn',
      },
    }));

    expect(screen.queryByText(/^Interview first/)).toBeNull();
    expect(screen.getByText('Question-only nudge')).toBeTruthy();
    expect(screen.getByText('Deeper analysis')).toBeTruthy();
    expect(screen.getByText('Reference pro')).toBeTruthy();
    expect(screen.getByText('Optional · Learn')).toBeTruthy();
  });
});


const parted: Problem = {
  ...problem,
  id: 'parted',
  descriptionEn: 'Build it.\n\n────\n\n**Background** only context.',
  parts: [
    { title: 'Basic store', descriptionEn: 'Store values.' },
    { title: 'Injectable clock', descriptionEn: 'Default to the clock.' },
    { title: 'Concurrent callers', descriptionEn: 'Add a lock.' },
  ],
};

describe('DescriptionTab parts', () => {
  afterEach(cleanup);

  it('shows unlocked parts only, between the requirement and the background', () => {
    const { container } = render(React.createElement(DescriptionTab, { problem: parted, unlockedPart: 2 }));
    const text = container.textContent ?? '';

    expect(screen.getByText('Basic store')).toBeTruthy();
    expect(screen.getByText('Injectable clock')).toBeTruthy();
    expect(screen.queryByText('Concurrent callers')).toBeNull();
    expect(screen.queryByText('Add a lock.')).toBeNull();
    expect(screen.getByText('partLocked')).toBeTruthy();
    expect(text.indexOf('Build it.')).toBeLessThan(text.indexOf('Basic store'));
    expect(text.indexOf('Injectable clock')).toBeLessThan(text.indexOf('only context.'));
  });

  it('drops the locked notice once every part is shown', () => {
    render(React.createElement(DescriptionTab, { problem: parted, unlockedPart: 3 }));
    expect(screen.getByText('Concurrent callers')).toBeTruthy();
    expect(screen.queryByText('partLocked')).toBeNull();
  });

  it('splits at the divider and keeps it with the background', () => {
    expect(splitAtDivider('a\n\n────\nb')).toEqual(['a', '────\nb']);
    expect(splitAtDivider('no divider')).toEqual(['no divider', '']);
  });
});


describe('DescriptionTab tradeoffs on multi-part exercises', () => {
  afterEach(cleanup);

  const withInterview: Problem = {
    ...parted,
    interviewQuestions: [
      { stage: 'concept', question: 'What structure per key?' },
      { stage: 'deep_dive', question: 'Which bisect variant?' },
      { stage: 'tradeoffs', question: 'One lock or one per key?' },
    ],
  };
  const answered = {
    record: { status: 'skipped' as const, answers: ['', '', ''], updatedAt: '' },
    onChange: () => {},
  };

  it('hides tradeoff questions while a part is still locked', () => {
    render(React.createElement(DescriptionTab, { problem: withInterview, unlockedPart: 2, interview: answered }));
    expect(screen.queryByText('One lock or one per key?')).toBeNull();
    expect(screen.getByText(/open once every part is unlocked/)).toBeTruthy();
  });

  it('shows them once every part is unlocked', () => {
    render(React.createElement(DescriptionTab, { problem: withInterview, unlockedPart: 3, interview: answered }));
    expect(screen.getByText('One lock or one per key?')).toBeTruthy();
  });
});
