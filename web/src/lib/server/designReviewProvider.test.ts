import { afterEach, describe, expect, it, vi } from 'vitest';

import type { DesignReviewRequest } from '@/lib/designReview';
import {
  DesignReviewProviderError,
  OpenAICompatibleDesignReviewProvider,
  designReviewProviderFromEnv,
} from '@/lib/server/designReviewProvider';

const request: DesignReviewRequest = {
  exercise: {
    id: 'tool_registry',
    title: 'Validated Tool Registry',
    description: 'Implement a registry.',
    rubric: [{ field: 'api_boundaries', label: 'API boundaries' }],
  },
  note: {
    api_boundaries: 'Gateway validates.',
    state_ownership: '',
    failure_recovery: '',
    backpressure_concurrency: '',
    durability_idempotency: '',
    observability: '',
    security: '',
    tradeoffs: '',
  },
};

function response(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  });
}

function provider(fetchImpl: typeof fetch) {
  return new OpenAICompatibleDesignReviewProvider({
    baseUrl: 'https://provider.invalid/v1',
    credential: 'placeholder',
    model: 'review-model',
    fetchImpl,
    timeoutMs: 100,
  });
}

describe('OpenAI-compatible design-review provider', () => {
  afterEach(() => {
    vi.unstubAllEnvs();
  });

  it('returns advisory Markdown from a valid provider response', async () => {
    const fetchImpl = vi.fn(async () => response({
      model: 'review-model',
      choices: [{ message: { content: '## Review\nStrengthen ownership.' } }],
    }));

    await expect(provider(fetchImpl).review(request)).resolves.toEqual({
      review: '## Review\nStrengthen ownership.',
      model: 'review-model',
    });
    expect(fetchImpl).toHaveBeenCalledWith(
      'https://provider.invalid/v1/chat/completions',
      expect.objectContaining({ method: 'POST' }),
    );
  });

  it.each([
    [401, 'authentication_failed'],
    [403, 'authentication_failed'],
    [429, 'rate_limited'],
    [500, 'provider_unavailable'],
  ])('maps HTTP %s to %s without returning upstream text', async (status, code) => {
    const fetchImpl = vi.fn(async () => response({
      error: { message: 'UPSTREAM_DETAIL_MUST_NOT_LEAK' },
    }, status));
    await expect(provider(fetchImpl).review(request)).rejects.toMatchObject({
      code,
    });
    await provider(fetchImpl).review(request).catch((error) => {
      expect(String(error)).not.toContain('UPSTREAM_DETAIL_MUST_NOT_LEAK');
    });
  });

  it('distinguishes timeout, network failure, and malformed success', async () => {
    const timeout = vi.fn(async () => {
      throw new DOMException('aborted', 'AbortError');
    });
    await expect(provider(timeout).review(request)).rejects.toMatchObject({
      code: 'provider_timeout',
    });

    const unavailable = vi.fn(async () => {
      throw new Error('socket failed');
    });
    await expect(provider(unavailable).review(request)).rejects.toMatchObject({
      code: 'provider_unavailable',
    });

    const malformed = vi.fn(async () => response({ choices: [] }));
    await expect(provider(malformed).review(request)).rejects.toMatchObject({
      code: 'malformed_response',
    });
  });

  it('requires server environment configuration', () => {
    vi.stubEnv('AI_HELP_BASE_URL', '');
    vi.stubEnv('AI_HELP_API_KEY', '');
    vi.stubEnv('AI_HELP_MODEL', '');
    expect(() => designReviewProviderFromEnv()).toThrow(
      expect.objectContaining({ code: 'missing_configuration' }),
    );
  });

  it('uses a typed stable provider error', () => {
    const error = new DesignReviewProviderError(
      'rate_limited',
      'Provider rate limit reached.',
      429,
    );
    expect(error).toMatchObject({
      code: 'rate_limited',
      status: 429,
      message: 'Provider rate limit reached.',
    });
  });
});
