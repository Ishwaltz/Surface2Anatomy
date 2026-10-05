/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  env: {
    NEXT_PUBLIC_INFERENCE_API_URL: process.env.NEXT_PUBLIC_INFERENCE_API_URL || "https://sharonmelhi-surface2anatomy-backend.hf.space",
  },
};

export default nextConfig;
