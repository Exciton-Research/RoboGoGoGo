// Package the Next.js export with the Python distribution. This directory is generated.
import { cp, rm, access } from "node:fs/promises";
import { fileURLToPath } from "node:url";
const source = fileURLToPath(new URL("../out/", import.meta.url));
const destination = fileURLToPath(
  new URL("../../src/robogogogo/web/", import.meta.url),
);
await access(`${source}/index.html`);
await rm(destination, { recursive: true, force: true });
await cp(source, destination, { recursive: true });
console.log(`Next.js Studio packaged into ${destination}`);
