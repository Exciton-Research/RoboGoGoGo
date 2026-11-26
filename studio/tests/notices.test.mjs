import test from "node:test";
import assert from "node:assert/strict";
import {
  cp,
  mkdir,
  mkdtemp,
  readFile,
  rm,
  symlink,
  writeFile,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

test("notice check rejects stale text and lockfile changes without rewriting", async () => {
  const directory = await mkdtemp(path.join(tmpdir(), "robogogogo-notices-"));
  const studio = fileURLToPath(new URL("../", import.meta.url));
  try {
    await mkdir(path.join(directory, "studio/scripts"), { recursive: true });
    await cp(
      path.join(studio, "scripts/generate-notices.mjs"),
      path.join(directory, "studio/scripts/generate-notices.mjs"),
    );
    await cp(
      path.join(studio, "package-lock.json"),
      path.join(directory, "studio/package-lock.json"),
    );
    await cp(
      path.join(studio, "package.json"),
      path.join(directory, "studio/package.json"),
    );
    await symlink(
      path.join(studio, "node_modules"),
      path.join(directory, "studio/node_modules"),
      "dir",
    );
    const destination = path.join(directory, "THIRD_PARTY_NOTICES.md");
    await cp(
      new URL("../../THIRD_PARTY_NOTICES.md", import.meta.url),
      destination,
    );
    const check = () =>
      spawnSync(
        process.execPath,
        [
          path.join(directory, "studio/scripts/generate-notices.mjs"),
          "--check",
        ],
        { encoding: "utf8" },
      );
    const valid = check();
    assert.equal(valid.status, 0, valid.stderr);
    await writeFile(destination, "stale notices\n");
    const stale = check();
    assert.notEqual(stale.status, 0);
    assert.match(stale.stderr, /missing or stale/);
    assert.equal(await readFile(destination, "utf8"), "stale notices\n");
    await cp(
      new URL("../../THIRD_PARTY_NOTICES.md", import.meta.url),
      destination,
    );
    const lockPath = path.join(directory, "studio/package-lock.json");
    const lock = JSON.parse(await readFile(lockPath, "utf8"));
    lock.packages[""].version += "-fixture";
    await writeFile(lockPath, JSON.stringify(lock));
    const changedLock = check();
    assert.notEqual(changedLock.status, 0);
    assert.match(changedLock.stderr, /missing or stale/);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
