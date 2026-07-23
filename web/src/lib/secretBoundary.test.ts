import { describe, expect, it } from 'vitest';

import { scanForExternalAiSecrets } from '@/lib/secretBoundary';

describe('external-AI secret scanner', () => {
  it.each([
    [{ text: `sk-${'A'.repeat(32)}` }, 'api_key'],
    [{ text: `Authorization: Bearer ${'b'.repeat(32)}` }, 'bearer_token'],
    [{ text: `-----BEGIN ${'PRIVATE'} KEY-----` }, 'private_key'],
    [{ text: `api_key = "${'c'.repeat(24)}"` }, 'credential_assignment'],
    [{ text: `SERVICE_TOKEN=${'d'.repeat(24)}` }, 'env_payload'],
  ])('blocks likely secret material as %s', (payload, category) => {
    expect(scanForExternalAiSecrets(payload)).toEqual({
      blocked: true,
      categories: [category],
    });
  });

  it('deduplicates categories across nested request values', () => {
    expect(scanForExternalAiSecrets({
      query: `sk-${'A'.repeat(32)}`,
      code: [`sk-${'B'.repeat(32)}`],
    })).toEqual({ blocked: true, categories: ['api_key'] });
  });

  it.each([
    'How do I use softmax from torch.nn.functional?',
    'The API key remains server-side and rotates every 30 days.',
    'api_key = os.getenv("API_KEY")',
    'token_count = attention_mask.sum()',
    'password_hash = hash_password(password)',
    'Use Authorization headers without logging them.',
    'SERVICE_TOKEN=your_token_here',
    'sk-...',
  ])('allows ordinary code and design prose: %s', (text) => {
    expect(scanForExternalAiSecrets({ text })).toEqual({
      blocked: false,
      categories: [],
    });
  });
});
