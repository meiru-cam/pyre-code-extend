export type ExternalAiSecretCategory =
  | 'api_key'
  | 'bearer_token'
  | 'private_key'
  | 'credential_assignment'
  | 'env_payload';

export interface ExternalAiSecretScan {
  blocked: boolean;
  categories: ExternalAiSecretCategory[];
}

const DETECTORS: ReadonlyArray<{
  category: ExternalAiSecretCategory;
  pattern: RegExp;
}> = [
  {
    category: 'api_key',
    pattern: /\bsk-[A-Za-z0-9_-]{20,}\b/,
  },
  {
    category: 'bearer_token',
    pattern: /\bAuthorization\s*:\s*Bearer\s+[A-Za-z0-9._~+/=-]{20,}/i,
  },
  {
    category: 'private_key',
    pattern: /-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----/,
  },
  {
    category: 'credential_assignment',
    pattern: /\b(?:api[_-]?key|access[_-]?token|auth[_-]?token|password|secret)\s*[:=]\s*["']?[A-Za-z0-9._~+/=-]{20,}["']?/i,
  },
  {
    category: 'env_payload',
    pattern: /\b[A-Z][A-Z0-9_]*(?:API_KEY|TOKEN|SECRET|PASSWORD)\s*=\s*[A-Za-z0-9._~+/=-]{20,}/,
  },
];

function collectStrings(value: unknown, output: string[]): void {
  if (typeof value === 'string') {
    output.push(value);
    return;
  }
  if (Array.isArray(value)) {
    for (const item of value) collectStrings(item, output);
    return;
  }
  if (value && typeof value === 'object') {
    for (const item of Object.values(value as Record<string, unknown>)) {
      collectStrings(item, output);
    }
  }
}

export function scanForExternalAiSecrets(value: unknown): ExternalAiSecretScan {
  const strings: string[] = [];
  collectStrings(value, strings);

  const categories = DETECTORS
    .filter(({ pattern }) => strings.some((text) => pattern.test(text)))
    .map(({ category }) => category);

  return {
    blocked: categories.length > 0,
    categories,
  };
}
