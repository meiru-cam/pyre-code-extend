import { NextResponse } from 'next/server';

import { fetchSignedUrl, voiceInterviewConfig } from '@/lib/server/voiceInterviewProvider';

const RATE_WINDOW_MS = 60_000;
const RATE_MAX = 5;
const PROVIDER_TIMEOUT_MS = 10_000;
const hits = new Map<string, number[]>();

function isRateLimited(ip: string): boolean {
  const now = Date.now();
  const timestamps = (hits.get(ip) ?? [])
    .filter((timestamp) => now - timestamp < RATE_WINDOW_MS);
  if (timestamps.length >= RATE_MAX) {
    hits.set(ip, timestamps);
    return true;
  }
  timestamps.push(now);
  hits.set(ip, timestamps);
  return false;
}

function errorResponse(code: string, message: string, status: number) {
  return NextResponse.json({ code, error: message }, { status });
}

/** Starts one voice interview: returns a signed conversation URL. */
export async function POST(request: Request) {
  const ip = request.headers.get('x-forwarded-for')?.split(',')[0]?.trim()
    || 'unknown';
  if (isRateLimited(ip)) {
    return errorResponse('rate_limited', 'Too many voice interviews started. Please try again later.', 429);
  }

  const config = voiceInterviewConfig();
  if (!config) {
    return errorResponse('missing_configuration', 'Optional voice interview is not configured on the server.', 503);
  }

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), PROVIDER_TIMEOUT_MS);
  try {
    const signedUrl = await fetchSignedUrl(config, controller.signal);
    if (!signedUrl) {
      return errorResponse('provider_error', 'The voice provider did not start a session.', 502);
    }
    return NextResponse.json({ signedUrl });
  } catch {
    return errorResponse('provider_unavailable', 'The voice provider could not be reached.', 502);
  } finally {
    clearTimeout(timer);
  }
}
