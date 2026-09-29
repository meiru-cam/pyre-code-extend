// @vitest-environment jsdom

import { beforeEach, describe, expect, it } from 'vitest';

import {
  formatPrepForFeedback,
  hasPrepDraftContent,
  loadPrepDraft,
  savePrepDraft,
} from '@/lib/prepDraft';

describe('prep local drafts', () => {
  beforeEach(() => window.localStorage.clear());

  it('restores the answer and covered points per item', () => {
    const draft = { answer: 'Shard by user id.', covered: ['Architecture'] };
    savePrepDraft('openai-webhook-delivery', draft);

    expect(loadPrepDraft('openai-webhook-delivery')).toEqual(draft);
    expect(loadPrepDraft('anthropic-prompt-playground')).toBeNull();
  });

  it('ignores stored values from another schema', () => {
    window.localStorage.setItem(
      'pyre-code-prep:v1:openai-webhook-delivery',
      JSON.stringify({ schemaVersion: 999, answer: 'stale', covered: [] }),
    );
    expect(loadPrepDraft('openai-webhook-delivery')).toBeNull();
  });

  it('treats a whitespace-only answer as no draft', () => {
    expect(hasPrepDraftContent({ answer: ' \n', covered: ['Architecture'] })).toBe(false);
    expect(hasPrepDraftContent({ answer: 'Queue per tenant', covered: [] })).toBe(true);
    expect(hasPrepDraftContent(null)).toBe(false);
  });
});

describe('feedback text', () => {
  it('lists the question, every rubric point and the trimmed answer', () => {
    const text = formatPrepForFeedback(
      { title: 'Webhook delivery', prompt: 'Design it.', rubric: ['Architecture', 'Deep dives'] },
      '  At-least-once with idempotency keys.\n',
    );
    expect(text).toContain('Interview question: Webhook delivery');
    expect(text).toContain('- Architecture\n- Deep dives');
    expect(text.endsWith('My answer:\nAt-least-once with idempotency keys.')).toBe(true);
  });
});
