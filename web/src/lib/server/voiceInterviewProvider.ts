const SIGNED_URL_ENDPOINT = 'https://api.elevenlabs.io/v1/convai/conversation/get-signed-url';

export interface VoiceInterviewConfig {
  apiKey: string;
  agentId: string;
}

export function voiceInterviewConfig(): VoiceInterviewConfig | null {
  const apiKey = process.env.ELEVENLABS_API_KEY?.trim() ?? '';
  const agentId = process.env.ELEVENLABS_AGENT_ID?.trim() ?? '';
  return apiKey && agentId ? { apiKey, agentId } : null;
}

/** A short-lived URL the browser uses to open one conversation; the API key never leaves the server. */
export async function fetchSignedUrl(
  config: VoiceInterviewConfig,
  signal?: AbortSignal,
): Promise<string | null> {
  const url = `${SIGNED_URL_ENDPOINT}?agent_id=${encodeURIComponent(config.agentId)}`;
  const response = await fetch(url, { headers: { 'xi-api-key': config.apiKey }, signal });
  if (!response.ok) return null;
  const data = await response.json().catch(() => null) as { signed_url?: unknown } | null;
  return typeof data?.signed_url === 'string' && data.signed_url ? data.signed_url : null;
}
