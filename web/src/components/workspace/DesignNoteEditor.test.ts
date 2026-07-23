// @vitest-environment jsdom

import React from 'react';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { DesignNoteEditor } from '@/components/workspace/DesignNoteEditor';
import type { Problem } from '@/lib/types';

const rubric = [
  { field: 'api_boundaries', label: 'API boundaries and class responsibilities' },
  { field: 'state_ownership', label: 'State and ownership' },
  { field: 'failure_recovery', label: 'Failure behavior and recovery' },
  { field: 'backpressure_concurrency', label: 'Backpressure and concurrency' },
  { field: 'durability_idempotency', label: 'Durability and idempotency' },
  { field: 'observability', label: 'Observability' },
  { field: 'security', label: 'Security' },
  { field: 'tradeoffs', label: 'Explicit tradeoffs' },
] as const;

const problem: Problem = {
  id: 'tool_registry',
  title: 'Validated Tool Registry',
  titleZh: 'Validated Tool Registry',
  difficulty: 'Medium',
  functionName: 'ToolRegistry',
  hint: '',
  hintZh: '',
  descriptionEn: 'Implement a registry.',
  descriptionZh: 'Implement a registry.',
  tests: [],
  version: 1,
  designNoteRubric: [...rubric],
};

function response(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  });
}

describe('DesignNoteEditor', () => {
  beforeEach(() => {
    window.localStorage.clear();
    vi.stubGlobal('fetch', vi.fn(async (_input: RequestInfo | URL, init?: RequestInit) => {
      if (!init?.method || init.method === 'GET') return response({}, 404);
      if (String(_input) === '/api/design-review') {
        return response({
          review: '## Advisory review\nClarify the state owner.',
          model: 'review-model',
          advisory: true,
        });
      }
      return response({
        taskId: problem.id,
        contractVersion: 1,
        fields: JSON.parse(String(init.body)).fields,
        updatedAt: '2026-07-23T12:00:00Z',
      });
    }));
  });

  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it('keeps implementation, note, and AI-review statuses independent', async () => {
    const user = userEvent.setup();
    render(React.createElement(DesignNoteEditor, {
      problem,
      implementationStatus: 'attempted',
    }));

    expect(screen.getByText('Implementation: Attempted')).toBeTruthy();
    expect(screen.getByText('Design note: Empty')).toBeTruthy();
    expect(screen.getByText('AI review: Not requested')).toBeTruthy();

    await user.type(
      screen.getByLabelText('API boundaries and class responsibilities'),
      'Gateway validates requests.',
    );
    expect(screen.getByText('Implementation: Attempted')).toBeTruthy();
    expect(screen.getByText('Design note: Draft')).toBeTruthy();
    expect(screen.getByText('AI review: Not requested')).toBeTruthy();

    await user.click(screen.getByRole('button', { name: 'Save design note' }));
    await waitFor(() => expect(screen.getByText('Design note: Saved')).toBeTruthy());
    expect(screen.getByText('Implementation: Attempted')).toBeTruthy();
    expect(screen.getByText('AI review: Not requested')).toBeTruthy();

    await user.click(screen.getByRole('button', { name: 'Review design note' }));
    await waitFor(() => expect(screen.getByText('AI review: Ready')).toBeTruthy());
    expect(screen.getByText('Implementation: Attempted')).toBeTruthy();
    expect(screen.getByText('Design note: Saved')).toBeTruthy();
    expect(screen.getByText('Clarify the state owner.')).toBeTruthy();

    const reviewCall = vi.mocked(fetch).mock.calls.find(
      ([input]) => String(input) === '/api/design-review',
    );
    const reviewBody = JSON.parse(String(reviewCall?.[1]?.body));
    expect(Object.keys(reviewBody)).toEqual(['exercise', 'note']);
    expect(JSON.stringify(reviewBody)).not.toContain('currentCode');
    expect(JSON.stringify(reviewBody)).not.toContain('solution');
  });

  it('persists the exact structured fields and restores a local draft', async () => {
    const user = userEvent.setup();
    const first = render(React.createElement(DesignNoteEditor, {
      problem,
      implementationStatus: 'todo',
    }));
    await user.type(
      screen.getByLabelText('API boundaries and class responsibilities'),
      'Gateway boundary',
    );
    await user.type(
      screen.getByLabelText('State and ownership'),
      'Scheduler state',
    );
    first.unmount();

    render(React.createElement(DesignNoteEditor, {
      problem,
      implementationStatus: 'todo',
    }));
    expect(
      (screen.getByLabelText(
        'API boundaries and class responsibilities',
      ) as HTMLTextAreaElement).value,
    ).toBe('Gateway boundary');
    expect(
      (screen.getByLabelText('State and ownership') as HTMLTextAreaElement).value,
    ).toBe('Scheduler state');

    await user.click(screen.getByRole('button', { name: 'Save design note' }));
    await waitFor(() => expect(fetch).toHaveBeenCalledWith(
      '/api/design-notes/tool_registry',
      expect.objectContaining({ method: 'PUT' }),
    ));
    const putCall = vi.mocked(fetch).mock.calls.find(([, init]) => init?.method === 'PUT');
    const body = JSON.parse(String(putCall?.[1]?.body));
    expect(body.contractVersion).toBe(1);
    expect(body.fields.api_boundaries).toBe('Gateway boundary');
    expect(body.fields.state_ownership).toBe('Scheduler state');
    expect(Object.keys(body.fields)).toEqual(rubric.map((item) => item.field));
  });

  it('keeps rubric guidance and reference example optional to open', () => {
    render(React.createElement(DesignNoteEditor, {
      problem,
      implementationStatus: 'solved',
    }));
    expect(screen.queryByText('Example: the API gateway validates')).toBeNull();

    fireEvent.click(screen.getByText('Rubric and example'));
    expect(screen.getByText(/Example: the API gateway validates/)).toBeTruthy();
  });

  it('blocks likely secrets locally before requesting external AI review', async () => {
    const user = userEvent.setup();
    render(React.createElement(DesignNoteEditor, {
      problem,
      implementationStatus: 'todo',
    }));
    await user.type(
      screen.getByLabelText('Security'),
      `Never log sk-${'A'.repeat(32)}`,
    );
    await user.click(screen.getByRole('button', { name: 'Review design note' }));

    expect(
      screen.getByText('Remove likely credentials or private keys before AI review.'),
    ).toBeTruthy();
    expect(
      vi.mocked(fetch).mock.calls.filter(
        ([input]) => String(input) === '/api/design-review',
      ),
    ).toHaveLength(0);
  });

  it('isolates draft state when navigation changes task or contract version', async () => {
    const user = userEvent.setup();
    const view = render(React.createElement(DesignNoteEditor, {
      problem,
      implementationStatus: 'todo',
    }));
    await user.type(
      screen.getByLabelText('State and ownership'),
      'Registry-owned state',
    );

    const otherProblem: Problem = {
      ...problem,
      id: 'budgeted_agent_loop',
      title: 'Budgeted Agent Loop',
    };
    view.rerender(React.createElement(DesignNoteEditor, {
      problem: otherProblem,
      implementationStatus: 'todo',
    }));
    expect(
      (screen.getByLabelText('State and ownership') as HTMLTextAreaElement).value,
    ).toBe('');

    view.rerender(React.createElement(DesignNoteEditor, {
      problem,
      implementationStatus: 'todo',
    }));
    expect(
      (screen.getByLabelText('State and ownership') as HTMLTextAreaElement).value,
    ).toBe('Registry-owned state');
  });
});
