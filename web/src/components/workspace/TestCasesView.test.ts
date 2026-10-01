// @vitest-environment jsdom

import React from 'react';
import { cleanup, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

vi.mock('@/context/LocaleContext', () => ({
  useLocale: () => ({ locale: 'en', t: (key: string) => key }),
}));

import { TestCasesView } from '@/components/workspace/TestCasesView';
import type { Test } from '@/lib/types';

const tests: Test[] = [
  { name: 'part one', code: 'assert part_one', part: 1 },
  { name: 'part two', code: 'assert part_two', part: 2 },
];

describe('TestCasesView parts', () => {
  afterEach(cleanup);

  it('samples only unlocked parts', () => {
    const { container } = render(React.createElement(TestCasesView, { tests, functionName: 'f', unlockedPart: 1 }));
    expect(screen.queryByText('Case 2')).toBeNull();
    expect(container.textContent).not.toContain('part_two');
  });

  it('samples every part once they are unlocked, and ignores parts on plain exercises', () => {
    render(React.createElement(TestCasesView, { tests, functionName: 'f', unlockedPart: 2 }));
    expect(screen.getByText('Case 2')).toBeTruthy();
    cleanup();
    render(React.createElement(TestCasesView, { tests, functionName: 'f' }));
    expect(screen.getByText('Case 2')).toBeTruthy();
  });
});
