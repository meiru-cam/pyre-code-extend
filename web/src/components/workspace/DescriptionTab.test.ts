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
