// Use a disposable Studio workspace: this creates and saves one demo episode.
import { mkdir } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";

const base = process.env.STUDIO_URL ?? "http://127.0.0.1:8765";
const output = fileURLToPath(new URL("../../docs/images/", import.meta.url));
await mkdir(output, { recursive: true });
const browser = await chromium.launch({
  headless: true,
  ...(process.env.BROWSER_CHANNEL
    ? { channel: process.env.BROWSER_CHANNEL }
    : {}),
});
try {
  const page = await browser.newPage({
    viewport: { width: 1600, height: 1100 },
    deviceScaleFactor: 1,
    colorScheme: "light",
    reducedMotion: "reduce",
  });
  const failures = [];
  page.on("pageerror", (error) => failures.push(error.message));
  async function screenshot(name, fullPage = false) {
    await page.evaluate(() => document.fonts.ready);
    await page.screenshot({
      path: `${output}${name}.png`,
      fullPage,
      animations: "disabled",
    });
    console.log(`Captured docs/images/${name}.png`);
  }
  await page.goto(`${base}/simulation/`);
  await page.getByText("Python engine connected", { exact: true }).waitFor();
  await page.getByRole("button", { name: "Episode", exact: true }).click();
  await page
    .getByRole("button", { name: "Start new episode", exact: true })
    .click();
  const executed = page.waitForResponse(
    (r) => r.url().endsWith("/api/vla/act") && r.request().method() === "POST",
  );
  await page
    .getByRole("button", { name: "Execute + capture next RGB", exact: true })
    .click();
  const action = await executed;
  if (!action.ok())
    throw new Error(`Demo action failed: ${await action.text()}`);
  const download = page.waitForEvent("download");
  await page
    .getByRole("button", { name: "Save & export episode", exact: true })
    .click();
  await download;
  await page.locator(".inspector-scroll").evaluate((node) => {
    node.scrollTop = 0;
  });
  await screenshot("episode-setup");
  await page.locator(".episode-stats").evaluate((node) => {
    node.scrollIntoView({ block: "start" });
  });
  await screenshot("episode");
  await page.getByRole("button", { name: "Robot", exact: true }).click();
  await screenshot("simulation");

  await page.goto(`${base}/`);
  await page
    .getByRole("button", { name: "Validate dataset", exact: true })
    .first()
    .click();
  await page
    .getByRole("status")
    .filter({ hasText: "aligned transitions" })
    .waitFor();
  await screenshot("workflow", true);
  await page.goto(`${base}/hardware/`);
  await page
    .getByRole("heading", { name: /telemetry/i })
    .first()
    .waitFor();
  await screenshot("hardware", true);
  await page.goto(`${base}/execution/`);
  await page
    .getByRole("button", { name: "Arm manual control", exact: true })
    .waitFor();
  await screenshot("command-test", true);
  if (failures.length)
    throw new Error(`Browser errors: ${failures.join("; ")}`);
} finally {
  await browser.close();
}
