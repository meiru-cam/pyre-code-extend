import { NextResponse } from 'next/server';

import {
  parseAiHelpRequest,
  type AiHelpRequest,
} from '@/lib/aiHelp';
import { scanForExternalAiSecrets } from '@/lib/secretBoundary';

const RATE_WINDOW_MS = 60_000;
const RATE_MAX = 10;
const PROVIDER_TIMEOUT_MS = 20_000;
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

function buildPrompt(request: AiHelpRequest): string {
  return [
    `Exercise ID: ${request.exercise.id}`,
    `Exercise Title: ${request.exercise.title}`,
    `Target Function: ${request.exercise.functionName}`,
    '',
    'Exercise Description:',
    request.exercise.description,
    '',
    'Learner Question:',
    request.query,
    ...(request.mode === 'review_code'
      ? ['', 'Learner Current Code:', request.learnerCode]
      : []),
    '',
    request.mode === 'review_code'
      ? 'Analyze the learner code first. Give concrete debugging questions and small local suggestions.'
      : 'Answer the learner question without assuming access to their current code.',
    'Use concise Markdown. Do not provide a full submit-ready implementation.',
  ].join('\n');
}

function extractContent(data: unknown): { guidance: string; model?: string } | null {
  if (!data || typeof data !== 'object') return null;
  const candidate = data as {
    model?: unknown;
    choices?: Array<{ message?: { content?: unknown } }>;
  };
  const content = candidate.choices?.[0]?.message?.content;
  if (typeof content !== 'string' || !content.trim()) return null;
  return {
    guidance: content.trim(),
    ...(typeof candidate.model === 'string' && candidate.model
      ? { model: candidate.model }
      : {}),
  };
}

function errorResponse(code: string, message: string, status: number) {
  return NextResponse.json({ code, error: message }, { status });
}

export async function POST(request: Request) {
  const ip = request.headers.get('x-forwarded-for')?.split(',')[0]?.trim()
    || 'unknown';
  if (isRateLimited(ip)) {
    return errorResponse(
      'rate_limited',
      'Too many AI Help requests. Please try again later.',
      429,
    );
  }

  let body: AiHelpRequest;
  try {
    body = parseAiHelpRequest(await request.json());
  } catch {
    return errorResponse('invalid_request', 'Invalid AI Help request.', 400);
  }

  if (scanForExternalAiSecrets(body).blocked) {
    return errorResponse(
      'secret_detected',
      'Remove likely credentials or private keys before using AI Help.',
      400,
    );
  }

  const baseUrl = process.env.AI_HELP_BASE_URL?.trim().replace(/\/+$/, '') ?? '';
  const apiKey = process.env.AI_HELP_API_KEY?.trim() ?? '';
  const model = process.env.AI_HELP_MODEL?.trim() ?? '';
  if (!baseUrl || !apiKey || !model) {
    return errorResponse(
      'missing_configuration',
      'Optional AI Help is not configured on the server.',
      503,
    );
  }

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), PROVIDER_TIMEOUT_MS);
  try {
    const upstream = await fetch(`${baseUrl}/chat/completions`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${apiKey}`,
      },
      signal: controller.signal,
      body: JSON.stringify({
        model,
        stream: false,
        temperature: 0.3,
        messages: [
          {
            role: 'system',
            content: [
              'You are an advisory tutor for PyTorch and AI systems exercises.',
              'Give explanations, debugging questions, and next steps.',
              'Never decide whether an exercise is solved.',
              'Do not provide a complete submit-ready solution.',
            ].join(' '),
          },
          { role: 'user', content: buildPrompt(body) },
        ],
      }),
    });

    if (upstream.status === 401 || upstream.status === 403) {
      return errorResponse(
        'authentication_failed',
        'The AI Help provider rejected its server configuration.',
        401,
      );
    }
    if (upstream.status === 429) {
      return errorResponse(
        'rate_limited',
        'The AI Help provider rate limit was reached.',
        429,
      );
    }
    if (!upstream.ok) {
      return errorResponse(
        'provider_unavailable',
        'The AI Help provider is unavailable.',
        503,
      );
    }

    let data: unknown;
    try {
      data = await upstream.json();
    } catch {
      return errorResponse(
        'malformed_response',
        'The AI Help provider returned an invalid response.',
        502,
      );
    }
    const result = extractContent(data);
    if (!result) {
      return errorResponse(
        'malformed_response',
        'The AI Help provider returned an invalid response.',
        502,
      );
    }
    return NextResponse.json({ ...result, advisory: true });
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') {
      return errorResponse(
        'provider_timeout',
        'The AI Help provider timed out.',
        504,
      );
    }
    return errorResponse(
      'provider_unavailable',
      'The AI Help provider is unavailable.',
      503,
    );
  } finally {
    clearTimeout(timer);
  }
}
