/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'standalone',
  reactStrictMode: true,
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
