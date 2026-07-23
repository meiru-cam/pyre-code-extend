import { beforeEach, describe, expect, it, vi } from 'vitest';

const cookieState = vi.hoisted(() => ({ value: undefined as string | undefined }));

vi.mock('next/headers', () => ({
  cookies: async () => ({
    get: () => cookieState.value ? { value: cookieState.value } : undefined,
  }),
}));

import { GET, PUT } from '@/app/api/design-notes/[taskId]/route';

function context(taskId = 'tool_registry') {
  return { params: Promise.resolve({ taskId }) };
}

function upstream(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  });
}

describe('/api/design-notes/[taskId]', () => {
  beforeEach(() => {
    cookieState.value = undefined;
    vi.restoreAllMocks();
  });

  it('does not create a session or call the backend for an anonymous GET', async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock);

    const response = await GET(
      new Request('http://localhost/api/design-notes/tool_registry?contractVersion=1'),
      context(),
    );

    expect(response.status).toBe(404);
    expect(fetchMock).not.toHaveBeenCalled();
    vi.unstubAllGlobals();
  });

  it('keeps the session token server-side and maps the backend response', async () => {
    cookieState.value = 'private-session';
    const fetchMock = vi.fn<typeof fetch>(async () => upstream({
      task_id: 'tool_registry',
      contract_version: 1,
      fields: { api_boundaries: 'Gateway' },
      updated_at: '2026-07-23 12:00:00',
    }));
    vi.stubGlobal('fetch', fetchMock);

    const response = await GET(
      new Request('http://localhost/api/design-notes/tool_registry?contractVersion=1'),
      context(),
    );
    const body = await response.json();

    expect(response.status).toBe(200);
    expect(body.taskId).toBe('tool_registry');
    expect(body.contractVersion).toBe(1);
    expect(body.updatedAt).toBe('2026-07-23 12:00:00');
    expect(fetchMock).toHaveBeenCalledWith(
      expect.stringContaining('/design-notes/tool_registry?contract_version=1'),
      expect.objectContaining({
        cache: 'no-store',
        headers: { 'X-Session-Token': 'private-session' },
      }),
    );
    expect(String(fetchMock.mock.calls[0][0])).not.toContain('private-session');
    vi.unstubAllGlobals();
  });

  it('creates an HTTP-only session on PUT and sends snake-case backend fields', async () => {
    const fetchMock = vi.fn<typeof fetch>(async (_url, init) => {
      const sent = JSON.parse(String(init?.body));
      return upstream({
        task_id: 'tool_registry',
        contract_version: sent.contract_version,
        fields: sent.fields,
        updated_at: '2026-07-23 12:00:00',
      });
    });
    vi.stubGlobal('fetch', fetchMock);
    const fields = {
      api_boundaries: 'Gateway',
      state_ownership: '',
      failure_recovery: '',
      backpressure_concurrency: '',
      durability_idempotency: '',
      observability: '',
      security: '',
      tradeoffs: '',
    };

    const response = await PUT(
      new Request('http://localhost/api/design-notes/tool_registry', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ contractVersion: 1, fields }),
      }),
      context(),
    );
    const [, init] = fetchMock.mock.calls[0];
    const sent = JSON.parse(String(init?.body));

    expect(sent.contract_version).toBe(1);
    expect(sent.fields).toEqual(fields);
    expect(sent.session_token).toEqual(expect.any(String));
    expect(sent.session_token.length).toBeGreaterThan(10);
    expect(response.headers.get('set-cookie')).toContain('session_token=');
    expect(response.headers.get('set-cookie')).toContain('HttpOnly');
    vi.unstubAllGlobals();
  });

  it('sanitizes backend network failures for GET and PUT', async () => {
    cookieState.value = 'private-session';
    vi.stubGlobal('fetch', vi.fn(async () => {
      throw new Error('PRIVATE_NETWORK_DETAIL');
    }));

    const getResponse = await GET(
      new Request('http://localhost/api/design-notes/tool_registry?contractVersion=1'),
      context(),
    );
    const putResponse = await PUT(
      new Request('http://localhost/api/design-notes/tool_registry', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ contractVersion: 1, fields: {} }),
      }),
      context(),
    );

    expect(getResponse.status).toBe(502);
    expect(putResponse.status).toBe(502);
    expect(JSON.stringify(await getResponse.json())).not.toContain(
      'PRIVATE_NETWORK_DETAIL',
    );
    expect(JSON.stringify(await putResponse.json())).not.toContain(
      'PRIVATE_NETWORK_DETAIL',
    );
    vi.unstubAllGlobals();
  });
});
