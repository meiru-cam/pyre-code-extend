/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'standalone',
  reactStrictMode: true,
  // SWC's minifier mangles the text \\ud800 inside the bundled problem JSON (durable_kv), which breaks the build.
  experimental: { serverMinification: false },
  turbopack: {
    root: __dirname,
  },
  // Paths merged into others keep working as links.
  async redirects() {
    return [
      { source: '/paths/advanced-attention', destination: '/paths/attention-position', permanent: true },
      { source: '/paths/agent-guardrails-security', destination: '/paths/agent-runtime-system-design', permanent: true },
      { source: '/paths/diffusion-transformer', destination: '/paths/vision-transformer', permanent: true },
    ];
  },
};

module.exports = nextConfig;
