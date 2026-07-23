import type { DesignReviewRequest } from '@/lib/designReview';

export type DesignReviewProviderErrorCode =
  | 'missing_configuration'
  | 'authentication_failed'
  | 'rate_limited'
  | 'provider_timeout'
  | 'provider_unavailable'
  | 'malformed_response';

export class DesignReviewProviderError extends Error {
  constructor(
    public readonly code: DesignReviewProviderErrorCode,
    message: string,
    public readonly status: number,
  ) {
    super(message);
    this.name = 'DesignReviewProviderError';
  }
}

interface ProviderConfig {
  baseUrl: string;
  credential: string;
  model: string;
  fetchImpl?: typeof fetch;
  timeoutMs?: number;
}

export interface DesignReviewResult {
  review: string;
  model: string;
}

export interface DesignReviewProvider {
  review(request: DesignReviewRequest): Promise<DesignReviewResult>;
}

export class OpenAICompatibleDesignReviewProvider
implements DesignReviewProvider {
  private readonly baseUrl: string;
  private readonly credential: string;
  private readonly model: string;
  private readonly fetchImpl: typeof fetch;
  private readonly timeoutMs: number;

  constructor(config: ProviderConfig) {
    this.baseUrl = config.baseUrl.trim().replace(/\/+$/, '');
    this.credential = config.credential;
    this.model = config.model;
    this.fetchImpl = config.fetchImpl ?? fetch;
    this.timeoutMs = config.timeoutMs ?? 20_000;
  }

  async review(request: DesignReviewRequest): Promise<DesignReviewResult> {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeoutMs);
    let response: Response;
    try {
      response = await this.fetchImpl(`${this.baseUrl}/chat/completions`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${this.credential}`,
        },
        signal: controller.signal,
        body: JSON.stringify({
          model: this.model,
          stream: false,
          temperature: 0.2,
          messages: [
            {
              role: 'system',
              content: [
                'You review system-design notes for an implementation exercise.',
                'Return concise advisory Markdown with strengths, risks, and concrete next questions.',
                'Do not decide whether the exercise is solved.',
                'Use only the exercise context and note supplied by the user message.',
              ].join(' '),
            },
            {
              role: 'user',
              content: JSON.stringify(request),
            },
          ],
        }),
      });
    } catch (error) {
      if (error instanceof DOMException && error.name === 'AbortError') {
        throw new DesignReviewProviderError(
          'provider_timeout',
          'The review provider timed out.',
          504,
        );
      }
      throw new DesignReviewProviderError(
        'provider_unavailable',
        'The review provider is unavailable.',
        503,
      );
    } finally {
      clearTimeout(timer);
    }

    if (response.status === 401 || response.status === 403) {
      throw new DesignReviewProviderError(
        'authentication_failed',
        'The review provider rejected its server configuration.',
        401,
      );
    }
    if (response.status === 429) {
      throw new DesignReviewProviderError(
        'rate_limited',
        'The review provider rate limit was reached.',
        429,
      );
    }
    if (!response.ok) {
      throw new DesignReviewProviderError(
        'provider_unavailable',
        'The review provider is unavailable.',
        503,
      );
    }

    let data: unknown;
    try {
      data = await response.json();
    } catch {
      throw new DesignReviewProviderError(
        'malformed_response',
        'The review provider returned an invalid response.',
        502,
      );
    }
    const candidate = data as {
      model?: unknown;
      choices?: Array<{ message?: { content?: unknown } }>;
    };
    const content = candidate.choices?.[0]?.message?.content;
    if (typeof content !== 'string' || !content.trim()) {
      throw new DesignReviewProviderError(
        'malformed_response',
        'The review provider returned an invalid response.',
        502,
      );
    }
    return {
      review: content.trim(),
      model: typeof candidate.model === 'string' && candidate.model
        ? candidate.model
        : this.model,
    };
  }
}

export function designReviewProviderFromEnv(
  environment: NodeJS.ProcessEnv = process.env,
  fetchImpl: typeof fetch = fetch,
): DesignReviewProvider {
  const baseUrl = environment.AI_HELP_BASE_URL?.trim() ?? '';
  const credential = environment.AI_HELP_API_KEY?.trim() ?? '';
  const model = environment.AI_HELP_MODEL?.trim() ?? '';
  if (!baseUrl || !credential || !model) {
    throw new DesignReviewProviderError(
      'missing_configuration',
      'Optional AI review is not configured.',
      503,
    );
  }
  return new OpenAICompatibleDesignReviewProvider({
    baseUrl,
    credential,
    model,
    fetchImpl,
  });
}
