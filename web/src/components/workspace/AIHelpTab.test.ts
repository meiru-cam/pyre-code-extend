// @vitest-environment jsdom

import React from 'react';
import {
  cleanup,
  render,
  screen,
  waitFor,
} from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import {
  afterEach,
  beforeEach,
  describe,
  expect,
  it,
  vi,
} from 'vitest';

vi.mock('@/context/LocaleContext', () => ({
  useLocale: () => ({
    locale: 'en',
    t: (key: string) => ({
      aiHelpSafetyNote: 'Advisory AI; solved status stays deterministic.',
      aiHelpQuestion: 'Question',
      aiHelpQuestionPlaceholder: 'Ask about an API, concept, shape, or failure.',
      aiHelpAskQuestion: 'Ask question',
      aiHelpReviewCode: 'Review my code',
      aiHelpGenerating: 'Generating...',
      aiHelpEmpty: 'Ask a question or explicitly request code review.',
      aiHelpResponseTitle: 'Guidance',
      aiHelpRequestFailed: 'AI request failed.',
      aiHelpMissingConfig: 'AI Help is not configured on the server.',
      aiHelpSecretDetected: 'Remove likely credentials or private keys first.',
      aiHelpServerConfigured: 'Using the server-configured AI endpoint.',
    }[key] ?? key),
  }),
}));

import { AIHelpTab } from '@/components/workspace/AIHelpTab';
import type { Problem } from '@/lib/types';
import { useProblemStore } from '@/store/problemStore';

const problem: Problem = {
  id: 'qk_norm',
  title: 'QK Norm',
  titleZh: 'QK Norm',
  difficulty: 'Medium',
  functionName: 'qk_norm',
  hint: 'HINT_MUST_NOT_LEAK',
  hintZh: '',
  descriptionEn: 'Normalize queries and keys.',
  descriptionZh: 'ZH_DESCRIPTION',
  tests: [{ name: 'hidden', code: 'UNSHOWN_TEST', visibility: 'unshown' }],
};

function jsonResponse(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  });
}

describe('AIHelpTab external-AI boundary', () => {
  beforeEach(() => {
    window.localStorage.clear();
    useProblemStore.setState({
      currentCode: 'CURRENT_CODE_MUST_NOT_LEAK',
      aiHelpQuery: '',
      aiHelpResponse: null,
      aiHelpError: null,
      aiHelpLoading: false,
    });
    vi.stubGlobal('fetch', vi.fn(async (
      input: RequestInfo | URL,
      init?: RequestInit,
    ) => {
      if (String(input) === '/api/ai-help/status') {
        return jsonResponse({ configured: true });
      }
      if (init?.method === 'POST') {
        return jsonResponse({
          guidance: '## Guidance\nUse dim=-1.',
          advisory: true,
        });
      }
      return jsonResponse({}, 404);
    }));
  });

  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it('asks a question without sending learner code or hidden material', async () => {
    const user = userEvent.setup();
    render(React.createElement(AIHelpTab, { problem }));
    await user.type(
      screen.getByLabelText('Question'),
      'How do I use softmax from F?',
    );
    await user.click(screen.getByRole('button', { name: 'Ask question' }));

    await waitFor(() => expect(screen.getByText('Use dim=-1.')).toBeTruthy());
    const post = vi.mocked(fetch).mock.calls.find(
      ([, init]) => init?.method === 'POST',
    );
    const body = JSON.parse(String(post?.[1]?.body));
    expect(body).toEqual({
      mode: 'question',
      query: 'How do I use softmax from F?',
      exercise: {
        id: 'qk_norm',
        title: 'QK Norm',
        functionName: 'qk_norm',
        description: 'Normalize queries and keys.',
      },
    });
    expect(JSON.stringify(body)).not.toContain('CURRENT_CODE_MUST_NOT_LEAK');
    expect(JSON.stringify(body)).not.toContain('UNSHOWN_TEST');
    expect(JSON.stringify(body)).not.toContain('HINT_MUST_NOT_LEAK');
  });

  it('sends learner code only after explicit review-code action', async () => {
    const user = userEvent.setup();
    useProblemStore.setState({ currentCode: 'def qk_norm(): pass' });
    render(React.createElement(AIHelpTab, { problem }));
    await user.type(screen.getByLabelText('Question'), 'Why is the shape wrong?');
    await user.click(screen.getByRole('button', { name: 'Review my code' }));

    await waitFor(() => expect(vi.mocked(fetch)).toHaveBeenCalledTimes(2));
    const post = vi.mocked(fetch).mock.calls.find(
      ([, init]) => init?.method === 'POST',
    );
    expect(JSON.parse(String(post?.[1]?.body))).toMatchObject({
      mode: 'review_code',
      learnerCode: 'def qk_norm(): pass',
    });
  });

  it('blocks likely secrets in the browser before POST', async () => {
    const user = userEvent.setup();
    render(React.createElement(AIHelpTab, { problem }));
    await user.type(
      screen.getByLabelText('Question'),
      `Please inspect sk-${'A'.repeat(32)}`,
    );
    await user.click(screen.getByRole('button', { name: 'Ask question' }));

    expect(
      screen.getByText('Remove likely credentials or private keys first.'),
    ).toBeTruthy();
    expect(
      vi.mocked(fetch).mock.calls.filter(([, init]) => init?.method === 'POST'),
    ).toHaveLength(0);
  });

  it('does not render or persist browser-side provider configuration', async () => {
    const getItem = vi.spyOn(Storage.prototype, 'getItem');
    const setItem = vi.spyOn(Storage.prototype, 'setItem');
    render(React.createElement(AIHelpTab, { problem }));

    await waitFor(() => expect(
      screen.getByText('Using the server-configured AI endpoint.'),
    ).toBeTruthy());
    expect(screen.queryByLabelText(/API Key/i)).toBeNull();
    expect(screen.queryByLabelText(/Base URL/i)).toBeNull();
    expect(getItem).not.toHaveBeenCalled();
    expect(setItem).not.toHaveBeenCalled();
  });
});
