import { NextResponse } from 'next/server';

import { parseDesignReviewRequest } from '@/lib/designReview';
import {
  DesignReviewProviderError,
  designReviewProviderFromEnv,
} from '@/lib/server/designReviewProvider';
import { scanForExternalAiSecrets } from '@/lib/secretBoundary';

export async function POST(request: Request) {
  let reviewRequest;
  try {
    reviewRequest = parseDesignReviewRequest(await request.json());
  } catch {
    return NextResponse.json(
      { code: 'invalid_request', error: 'Invalid design-review request.' },
      { status: 400 },
    );
  }

  if (scanForExternalAiSecrets(reviewRequest).blocked) {
    return NextResponse.json(
      {
        code: 'secret_detected',
        error: 'Remove likely credentials or private keys before AI review.',
      },
      { status: 400 },
    );
  }

  try {
    const result = await designReviewProviderFromEnv().review(reviewRequest);
    return NextResponse.json({ ...result, advisory: true });
  } catch (error) {
    if (error instanceof DesignReviewProviderError) {
      return NextResponse.json(
        { code: error.code, error: error.message },
        { status: error.status },
      );
    }
    return NextResponse.json(
      { code: 'provider_unavailable', error: 'The review provider is unavailable.' },
      { status: 503 },
    );
  }
}
