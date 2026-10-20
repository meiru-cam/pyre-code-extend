#!/usr/bin/env node

// Creates the ElevenLabs agent the voice interview talks to, then writes its ID to web/.env.
// Needs only ELEVENLABS_API_KEY, from the environment or web/.env.
// Usage: npm run voice:agent [-- --force] [-- --env /path/to/web/.env]

const fs = require('fs');
const path = require('path');

const API = 'https://api.elevenlabs.io/v1/convai';
const envFlag = process.argv.indexOf('--env');
const envArg = envFlag === -1 ? null : process.argv[envFlag + 1];
if (envFlag !== -1 && (!envArg || envArg.startsWith('--'))) {
  console.error('--env needs a path, for example --env ../pyre-code/web/.env');
  process.exit(1);
}
const envPath = envArg ? path.resolve(envArg) : path.resolve(__dirname, '..', 'web', '.env');

// The real prompt and first message are sent per exercise by the app as overrides.
const PLACEHOLDER_PROMPT = 'You are a calm, senior ML engineer running a live coding interview.';

// The name must match RECORD_ANSWER_TOOL in web/src/lib/voiceInterview.ts.
const RECORD_ANSWER_TOOL = {
  type: 'client',
  name: 'record_answer',
  description: "Record the candidate's spoken answer to one interview question. Call it after the candidate finishes answering.",
  parameters: {
    type: 'object',
    properties: {
      question_index: {
        type: 'number',
        description: 'Index of the question in the interview list.',
      },
    },
    required: ['question_index'],
  },
  expects_response: true,
  response_timeout_secs: 10,
};

function readEnvFile() {
  return fs.existsSync(envPath) ? fs.readFileSync(envPath, 'utf8') : '';
}

function envValue(text, name) {
  const match = text.match(new RegExp(`^${name}=(.*)$`, 'm'));
  return match ? match[1].trim() : '';
}

function setEnvValue(text, name, value) {
  const line = `${name}=${value}`;
  const pattern = new RegExp(`^${name}=.*$`, 'm');
  if (pattern.test(text)) return text.replace(pattern, line);
  return `${text}${text && !text.endsWith('\n') ? '\n' : ''}${line}\n`;
}

async function post(apiKey, url, body) {
  const response = await fetch(url, {
    method: 'POST',
    headers: { 'xi-api-key': apiKey, 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(`${url} returned ${response.status}: ${JSON.stringify(data.detail ?? data).slice(0, 500)}`);
  }
  return data;
}

async function main() {
  const force = process.argv.includes('--force');
  let envText = readEnvFile();
  const apiKey = process.env.ELEVENLABS_API_KEY?.trim() || envValue(envText, 'ELEVENLABS_API_KEY');
  if (!apiKey) {
    console.error(`Set ELEVENLABS_API_KEY in ${envPath} or the environment first.`);
    process.exit(1);
  }
  const existing = envValue(envText, 'ELEVENLABS_AGENT_ID');
  if (existing && !force) {
    console.error(`${envPath} already has ELEVENLABS_AGENT_ID=${existing}. Pass --force to create a new agent.`);
    process.exit(1);
  }

  const tool = await post(apiKey, `${API}/tools`, { tool_config: RECORD_ANSWER_TOOL });
  const agent = await post(apiKey, `${API}/agents/create`, {
    name: 'Pyre Code interviewer',
    conversation_config: {
      agent: {
        first_message: 'Hi. Let us start the interview.',
        prompt: { prompt: PLACEHOLDER_PROMPT, tool_ids: [tool.id] },
      },
    },
    platform_settings: {
      overrides: {
        conversation_config_override: {
          agent: { first_message: true, prompt: { prompt: true } },
        },
      },
      // Only signed URLs from /api/voice-interview/session can connect.
      auth: { enable_auth: true },
    },
  });

  envText = setEnvValue(envText, 'ELEVENLABS_API_KEY', apiKey);
  envText = setEnvValue(envText, 'ELEVENLABS_AGENT_ID', agent.agent_id);
  fs.writeFileSync(envPath, envText);
  console.log(`Created agent ${agent.agent_id} with tool ${tool.id}; wrote ELEVENLABS_API_KEY and ELEVENLABS_AGENT_ID to ${envPath}.`);
  console.log('Restart the web server so it reads the new values.');
}

main().catch((error) => {
  console.error(error.message);
  process.exit(1);
});
