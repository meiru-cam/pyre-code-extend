import { afterEach, describe, expect, it, vi } from 'vitest';

import { POST } from '@/app/api/voice-interview/session/route';
import { GET } from '@/app/api/voice-interview/status/route';

function request(ip: string) {
  return new Request('http://localhost/api/voice-interview/session', {
    method: 'POST',
    headers: { 'x-forwarded-for': ip },
  });
}

describe('/api/voice-interview', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.unstubAllEnvs();
  });

  function configure() {
    vi.stubEnv('ELEVENLABS_API_KEY', 'placeholder-key');
    vi.stubEnv('ELEVENLABS_AGENT_ID', 'agent_test');
  }

  it('reports configuration only when both variables are set', async () => {
    vi.stubEnv('ELEVENLABS_API_KEY', '');
    vi.stubEnv('ELEVENLABS_AGENT_ID', 'agent_test');
    expect(await (await GET()).json()).toEqual({ configured: false });
    configure();
    expect(await (await GET()).json()).toEqual({ configured: true });
  });

  it('returns 503 without configuration and never calls the provider', async () => {
    vi.stubEnv('ELEVENLABS_API_KEY', '');
    vi.stubEnv('ELEVENLABS_AGENT_ID', '');
    const fetchMock = vi.fn<typeof fetch>();
    vi.stubGlobal('fetch', fetchMock);

    const response = await POST(request('10.0.0.1'));
    expect(response.status).toBe(503);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('returns the signed URL and keeps the key server-side', async () => {
    configure();
    const fetchMock = vi.fn<typeof fetch>(async () => new Response(
      JSON.stringify({ signed_url: 'wss://example.invalid/convai?token=t' }),
    ));
    vi.stubGlobal('fetch', fetchMock);

    const response = await POST(request('10.0.0.2'));
    const body = await response.json();

    expect(response.status).toBe(200);
    expect(body).toEqual({ signedUrl: 'wss://example.invalid/convai?token=t' });
    expect(JSON.stringify(body)).not.toContain('placeholder-key');
    const [url, init] = fetchMock.mock.calls[0];
    expect(String(url)).toContain('get-signed-url?agent_id=agent_test');
    expect((init?.headers as Record<string, string>)['xi-api-key']).toBe('placeholder-key');
  });

  it('maps a provider failure to 502 without leaking the upstream body', async () => {
    configure();
    vi.stubGlobal('fetch', vi.fn<typeof fetch>(async () => new Response('PRIVATE_DETAIL', { status: 401 })));

    const response = await POST(request('10.0.0.3'));
    const body = await response.json();
    expect(response.status).toBe(502);
    expect(JSON.stringify(body)).not.toContain('PRIVATE_DETAIL');
  });

  it('rate limits repeated starts from one address', async () => {
    configure();
    vi.stubGlobal('fetch', vi.fn<typeof fetch>(async () => new Response(JSON.stringify({ signed_url: 'wss://x' }))));

    const statuses = [];
    for (let i = 0; i < 6; i += 1) statuses.push((await POST(request('10.0.0.4'))).status);
    expect(statuses.slice(0, 5)).toEqual([200, 200, 200, 200, 200]);
    expect(statuses[5]).toBe(429);
  });
});
