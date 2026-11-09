import type { NextConfig } from "next";
import { PHASE_DEVELOPMENT_SERVER } from "next/constants";

export default function config(phase: string): NextConfig {
  const development = phase === PHASE_DEVELOPMENT_SERVER;
  const engine = process.env.ROBOGOGOGO_API_URL ?? "http://127.0.0.1:8765";
  return {
    output: development ? undefined : "export",
    reactStrictMode: true,
    trailingSlash: true,
    ...(development
      ? {
          async rewrites() {
            return [
              { source: "/api/:path*", destination: `${engine}/api/:path*` },
              { source: "/docs", destination: `${engine}/docs` },
              {
                source: "/openapi.json",
                destination: `${engine}/openapi.json`,
              },
            ];
          },
        }
      : {}),
  };
}
