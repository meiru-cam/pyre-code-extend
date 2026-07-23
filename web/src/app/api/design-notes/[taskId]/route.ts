import { cookies } from 'next/headers';
import { NextResponse } from 'next/server';

import { GRADING_SERVICE_URL } from '@/lib/constants';

interface RouteContext {
  params: Promise<{ taskId: string }>;
}

interface BackendDesignNote {
  task_id: string;
  contract_version: number;
  fields: Record<string, string>;
  updated_at: string;
}

function clientNote(note: BackendDesignNote) {
  return {
    taskId: note.task_id,
    contractVersion: note.contract_version,
    fields: note.fields,
    updatedAt: note.updated_at,
  };
}

function invalidVersion() {
  return NextResponse.json(
    { error: 'A positive contract version is required.' },
    { status: 400 },
  );
}

function unavailable(message: string) {
  return NextResponse.json({ error: message }, { status: 502 });
}

export async function GET(request: Request, { params }: RouteContext) {
  const { taskId } = await params;
  const contractVersion = Number(
    new URL(request.url).searchParams.get('contractVersion'),
  );
  if (!Number.isInteger(contractVersion) || contractVersion < 1) {
    return invalidVersion();
  }

  const cookieStore = await cookies();
  const sessionToken = cookieStore.get('session_token')?.value;
  if (!sessionToken) {
    return NextResponse.json({ error: 'Design note not found.' }, { status: 404 });
  }

  const query = new URLSearchParams({
    contract_version: String(contractVersion),
  });
  let upstream: Response;
  try {
    upstream = await fetch(
      `${GRADING_SERVICE_URL}/design-notes/${encodeURIComponent(taskId)}?${query}`,
      {
        cache: 'no-store',
        headers: { 'X-Session-Token': sessionToken },
      },
    );
  } catch {
    return unavailable('Design note could not be loaded.');
  }
  if (!upstream.ok) {
    return NextResponse.json(
      { error: upstream.status === 404 ? 'Design note not found.' : 'Design note could not be loaded.' },
      { status: upstream.status === 404 ? 404 : 502 },
    );
  }
  try {
    return NextResponse.json(clientNote(await upstream.json()));
  } catch {
    return unavailable('Design note could not be loaded.');
  }
}

export async function PUT(request: Request, { params }: RouteContext) {
  const { taskId } = await params;
  let body: {
    contractVersion?: unknown;
    fields?: unknown;
  };
  try {
    body = await request.json() as typeof body;
  } catch {
    return NextResponse.json(
      { error: 'Invalid design-note request.' },
      { status: 400 },
    );
  }
  if (
    typeof body.contractVersion !== 'number'
    || !Number.isInteger(body.contractVersion)
    || body.contractVersion < 1
  ) {
    return invalidVersion();
  }

  const cookieStore = await cookies();
  const existingToken = cookieStore.get('session_token')?.value;
  const sessionToken = existingToken ?? crypto.randomUUID();
  let upstream: Response;
  try {
    upstream = await fetch(
      `${GRADING_SERVICE_URL}/design-notes/${encodeURIComponent(taskId)}`,
      {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_token: sessionToken,
          contract_version: body.contractVersion,
          fields: body.fields,
        }),
      },
    );
  } catch {
    return unavailable('Design note could not be saved.');
  }
  if (!upstream.ok) {
    return NextResponse.json(
      { error: 'Design note could not be saved.' },
      { status: upstream.status >= 400 && upstream.status < 500 ? upstream.status : 502 },
    );
  }

  let response: NextResponse;
  try {
    response = NextResponse.json(clientNote(await upstream.json()));
  } catch {
    return unavailable('Design note could not be saved.');
  }
  if (!existingToken) {
    response.cookies.set('session_token', sessionToken, {
      httpOnly: true,
      sameSite: 'lax',
      secure: process.env.NODE_ENV === 'production',
      maxAge: 60 * 60 * 24 * 30,
    });
  }
  return response;
}
