import { afterEach, describe, expect, it, vi } from 'vitest';

import type { AiHelpRequest } from '@/lib/aiHelp';
import { POST } from '@/app/api/ai-help/route';

const question: AiHelpRequest = {
  mode: 'question',
  query: 'How do I use softmax from F?',
  exercise: {
    id: 'qk_norm',
    title: 'QK Norm',
    functionName: 'qk_norm',
    description: 'Normalize queries and keys.',
  },
};

function request(body: unknown) {
  return new Request('http://localhost/api/ai-help', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
}

function providerResponse(content = '## Guidance\nUse dim=-1.', status = 200) {
  return new Response(JSON.stringify({
    model: 'helper-model',
    choices: [{ message: { content } }],
    error: { message: 'PRIVATE_UPSTREAM_DETAIL' },
  }), { status });
}

describe('/api/ai-help', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.unstubAllEnvs();
  });

  function configure() {
    vi.stubEnv('AI_HELP_BASE_URL', 'https://provider.invalid/v1');
    vi.stubEnv('AI_HELP_API_KEY', 'placeholder');
    vi.stubEnv('AI_HELP_MODEL', 'helper-model');
  }

  it('sends question mode with no learner code or reference material', async () => {
    configure();
    const fetchMock = vi.fn<typeof fetch>(async () => providerResponse());
    vi.stubGlobal('fetch', fetchMock);

    const response = await POST(request(question));
    const body = await response.json();
    const upstream = JSON.parse(String(fetchMock.mock.calls[0][1]?.body));
    const prompt = JSON.stringify(upstream.messages);

    expect(response.status).toBe(200);
    expect(body.guidance).toContain('Use dim=-1');
    expect(prompt).toContain('How do I use softmax from F?');
    expect(prompt).not.toContain('learnerCode');
    expect(prompt).not.toContain('Reference Solution');
    expect(prompt).not.toContain('Sample Test');
  });

  it('includes learner code only in explicit review-code mode', async () => {
    configure();
    const fetchMock = vi.fn<typeof fetch>(async () => providerResponse());
    vi.stubGlobal('fetch', fetchMock);
    const review: AiHelpRequest = {
      ...question,
      mode: 'review_code',
      learnerCode: 'def qk_norm(): pass',
    };

    const response = await POST(request(review));
    const upstream = JSON.parse(String(fetchMock.mock.calls[0][1]?.body));
    expect(response.status).toBe(200);
    expect(JSON.stringify(upstream.messages)).toContain('def qk_norm(): pass');
  });

  it('blocks over-sharing and likely secrets before any provider fetch', async () => {
    configure();
    const fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock);

    const overShared = await POST(request({ ...question, solutionCode: 'x' }));
    expect(overShared.status).toBe(400);
    expect(fetchMock).not.toHaveBeenCalled();

    const secret = await POST(request({
      ...question,
      query: `Please inspect sk-${'A'.repeat(32)}`,
    }));
    expect(secret.status).toBe(400);
    expect(await secret.json()).toMatchObject({ code: 'secret_detected' });
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('requires server configuration and never accepts browser config', async () => {
    vi.stubEnv('AI_HELP_BASE_URL', '');
    vi.stubEnv('AI_HELP_API_KEY', '');
    vi.stubEnv('AI_HELP_MODEL', '');
    const fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock);

    const response = await POST(request({
      ...question,
      config: {
        baseUrl: 'https://attacker.invalid',
        apiKey: 'browser-value',
        model: 'x',
      },
    }));
    expect(response.status).toBe(400);
    expect(fetchMock).not.toHaveBeenCalled();

    const missing = await POST(request(question));
    expect(missing.status).toBe(503);
    expect(await missing.json()).toMatchObject({ code: 'missing_configuration' });
  });

  it('sanitizes authentication, rate-limit, and unavailable errors', async () => {
    configure();
    for (const [status, expectedCode] of [
      [401, 'authentication_failed'],
      [429, 'rate_limited'],
      [500, 'provider_unavailable'],
    ] as const) {
      const fetchMock = vi.fn<typeof fetch>(
        async () => providerResponse('', status),
      );
      vi.stubGlobal('fetch', fetchMock);
      const response = await POST(request(question));
      const body = await response.json();
      expect(body.code).toBe(expectedCode);
      expect(JSON.stringify(body)).not.toContain('PRIVATE_UPSTREAM_DETAIL');
      expect(JSON.stringify(body)).not.toContain('placeholder');
    }
  });
});
