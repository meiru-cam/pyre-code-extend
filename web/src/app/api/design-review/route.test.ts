import { afterEach, describe, expect, it, vi } from 'vitest';

import type { DesignReviewRequest } from '@/lib/designReview';
import { POST } from '@/app/api/design-review/route';

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

describe('/api/design-review', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.unstubAllEnvs();
  });

  it('calls only the provider endpoint and returns advisory state, not progress', async () => {
    vi.stubEnv('AI_HELP_BASE_URL', 'https://provider.invalid/v1');
    vi.stubEnv('AI_HELP_API_KEY', 'placeholder');
    vi.stubEnv('AI_HELP_MODEL', 'review-model');
    const fetchMock = vi.fn<typeof fetch>(async () => new Response(JSON.stringify({
      model: 'review-model',
      choices: [{ message: { content: '## Advisory review' } }],
    }), { status: 200 }));
    vi.stubGlobal('fetch', fetchMock);

    const response = await POST(new Request('http://localhost/api/design-review', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    }));
    const body = await response.json();

    expect(response.status).toBe(200);
    expect(body).toEqual({
      review: '## Advisory review',
      model: 'review-model',
      advisory: true,
    });
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(String(fetchMock.mock.calls[0][0])).toBe(
      'https://provider.invalid/v1/chat/completions',
    );
    expect(JSON.stringify(body)).not.toContain('solved');
    expect(JSON.stringify(body)).not.toContain('progress');
  });

  it('rejects over-shared requests before an external fetch', async () => {
    vi.stubEnv('AI_HELP_BASE_URL', 'https://provider.invalid/v1');
    vi.stubEnv('AI_HELP_API_KEY', 'placeholder');
    vi.stubEnv('AI_HELP_MODEL', 'review-model');
    const fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock);

    const response = await POST(new Request('http://localhost/api/design-review', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ...request, learnerCode: 'do not send' }),
    }));

    expect(response.status).toBe(400);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('blocks likely secrets in a note before an external fetch', async () => {
    vi.stubEnv('AI_HELP_BASE_URL', 'https://provider.invalid/v1');
    vi.stubEnv('AI_HELP_API_KEY', 'placeholder');
    vi.stubEnv('AI_HELP_MODEL', 'review-model');
    const fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock);

    const response = await POST(new Request('http://localhost/api/design-review', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        ...request,
        note: {
          ...request.note,
          security: `Do not log sk-${'A'.repeat(32)}`,
        },
      }),
    }));

    expect(response.status).toBe(400);
    expect(await response.json()).toMatchObject({ code: 'secret_detected' });
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it.each([
    ['', '', '', 503, 'missing_configuration'],
    ['https://provider.invalid/v1', 'placeholder', 'review-model', 401, 'authentication_failed'],
  ])(
    'maps stable errors without exposing configuration',
    async (baseUrl, credential, model, expectedStatus, expectedCode) => {
      vi.stubEnv('AI_HELP_BASE_URL', baseUrl);
      vi.stubEnv('AI_HELP_API_KEY', credential);
      vi.stubEnv('AI_HELP_MODEL', model);
      const fetchMock = vi.fn<typeof fetch>(async () => new Response(JSON.stringify({
        error: { message: 'PRIVATE_PROVIDER_DETAIL' },
      }), { status: 401 }));
      vi.stubGlobal('fetch', fetchMock);

      const response = await POST(new Request('http://localhost/api/design-review', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(request),
      }));
      const body = await response.json();

      expect(response.status).toBe(expectedStatus);
      expect(body.code).toBe(expectedCode);
      expect(JSON.stringify(body)).not.toContain('PRIVATE_PROVIDER_DETAIL');
      expect(JSON.stringify(body)).not.toContain('placeholder');
    },
  );
});
