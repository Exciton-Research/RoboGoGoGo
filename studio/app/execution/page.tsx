"use client";

import { useEffect, useRef, useState } from "react";
import { request } from "../../src/api";
import styles from "../workflow.module.css";
import controls from "./execution.module.css";

type Status = {
  backend: "test" | null;
  mode: string;
  reason: string;
  epoch?: number;
  owner?: string | null;
  stop_confirmed?: boolean;
  lease_remaining_s?: number;
  next_sequence?: number;
  active_command?: string | null;
  feedback?: {
    sequence: number;
    left: number[];
    right: number[];
    grippers: number[];
    moving: boolean;
  };
  events?: {
    time: number;
    kind: string;
    epoch: number;
    reason?: string;
    command_id?: string;
  }[];
};
type Lease = { token: string; epoch: number; snapshot: Status };

export default function ExecutionPage() {
  const [status, setStatus] = useState<Status | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [heartbeats, setHeartbeats] = useState(true);
  const lease = useRef<Lease | null>(null);
  const sendingHeartbeats = useRef(true);
  const generation = useRef(0);
  const mounted = useRef(false);

  function receive(next: Status) {
    if (!mounted.current) return;
    setStatus((previous) => {
      if ((previous?.epoch ?? -1) > (next.epoch ?? -1)) return previous;
      if (
        previous?.epoch === next.epoch &&
        (previous?.feedback?.sequence ?? 0) > (next.feedback?.sequence ?? 0)
      )
        return previous;
      return next;
    });
    if (lease.current && (next.epoch ?? 0) > lease.current.epoch)
      lease.current = null;
  }

  useEffect(() => {
    mounted.current = true;
    const poll = async () => {
      try {
        receive(await request<Status>("/execution"));
      } catch (failure) {
        if (mounted.current) setError(String(failure));
      }
    };
    const beat = async () => {
      const owned = lease.current;
      if (!owned || !sendingHeartbeats.current) return;
      try {
        receive(
          await request<Status>("/execution/heartbeat", { token: owned.token }),
        );
      } catch (failure) {
        if (lease.current?.token === owned.token) {
          lease.current = null;
          if (mounted.current) setError(String(failure));
        }
      }
    };
    const release = () => {
      if (!lease.current) return;
      generation.current++;
      lease.current = null;
      void fetch("/api/execution/stop", { method: "POST", keepalive: true });
    };
    const visibility = () => {
      if (document.hidden) release();
    };
    void poll();
    const polling = setInterval(() => void poll(), 200);
    const heartbeat = setInterval(() => void beat(), 250);
    document.addEventListener("visibilitychange", visibility);
    return () => {
      mounted.current = false;
      clearInterval(polling);
      clearInterval(heartbeat);
      document.removeEventListener("visibilitychange", visibility);
      release();
    };
  }, []);

  async function operate(
    kind: "manual" | "policy" | "takeover" | "reset" | "pose" | "neutral",
  ) {
    const operation = ++generation.current;
    setBusy(true);
    setError("");
    try {
      const fresh = await request<Status>("/execution");
      if (operation !== generation.current) return;
      if (kind === "manual" || kind === "policy" || kind === "takeover") {
        const acquired = await request<Lease>(
          kind === "takeover" ? "/execution/takeover" : "/execution/arm",
          {
            expected_epoch: fresh.epoch,
            ...(kind === "takeover" ? {} : { owner: kind }),
          },
        );
        if (operation !== generation.current) return;
        lease.current = acquired;
        sendingHeartbeats.current = true;
        setHeartbeats(true);
        receive(acquired.snapshot);
      } else if (kind === "reset") {
        receive(await request<Status>("/execution/reset", {}));
      } else {
        const owned = lease.current;
        if (!owned || !fresh.feedback)
          throw new Error("Acquire a control lease first");
        const neutral = kind === "neutral";
        const result = await request<{ snapshot: Status }>(
          "/execution/command",
          {
            token: owned.token,
            sequence: fresh.next_sequence,
            observation_sequence: fresh.feedback.sequence,
            left: [neutral ? 0 : 0.2, 0, 0, 0, 0, 0],
            right: [neutral ? 0 : -0.2, 0, 0, 0, 0, 0],
            left_gripper: { width: neutral ? 0.06 : 0.02, force_limit: 5 },
            right_gripper: { width: neutral ? 0.06 : 0.04, force_limit: 5 },
            duration_s: 2,
          },
        );
        receive(result.snapshot);
      }
    } catch (failure) {
      if (mounted.current && operation === generation.current)
        setError(String(failure));
    } finally {
      if (mounted.current && operation === generation.current) setBusy(false);
    }
  }

  async function stop() {
    generation.current++;
    lease.current = null;
    setBusy(false);
    try {
      receive(await request<Status>("/execution/stop", {}));
      setError("");
    } catch (failure) {
      setError(
        `Stop request failed: ${String(failure)}. Stop is not confirmed.`,
      );
    }
  }

  const enabled = status?.backend === "test";
  const canArm = enabled && status?.mode === "stopped" && status.stop_confirmed;
  return (
    <main className={styles.page}>
      <header className={styles.header}>
        <a href="/" className={styles.brand}>
          robogogogo<span>.</span>
        </a>
        <nav>
          <a href="/">Workflow</a>
          <a href="/hardware/">Hardware telemetry</a>
        </nav>
      </header>
      <section className={styles.intro}>
        <p className={styles.eyebrow}>COMMAND EXECUTION · TEST BACKEND</p>
        <h1>
          Two arms.
          <br />
          One control owner.
        </h1>
        <p>
          Exercise coordinated arm and gripper commands, heartbeat expiry, stop
          and manual takeover. No physical robot is connected to these controls.
        </p>
      </section>
      {error && (
        <p role="alert" className={styles.error}>
          {error}
        </p>
      )}
      {!enabled && (
        <div className={styles.empty}>
          {status?.reason ?? "Loading execution service…"}
        </div>
      )}
      {enabled && (
        <>
          <section className={styles.robot} aria-label="Execution status">
            <div>
              <p className={styles.eyebrow}>EXECUTION STATE</p>
              <h2>{status.mode}</h2>
              <p>{status.reason}</p>
            </div>
            <div>
              <strong>Owner: {status.owner ?? "none"}</strong>
              <p>
                Control epoch {status.epoch} · Lease{" "}
                {(status.lease_remaining_s ?? 0).toFixed(2)} s
              </p>
              <p>
                {status.stop_confirmed
                  ? "Stationary stop acknowledged"
                  : "No current stop acknowledgement"}
              </p>
            </div>
            <button className={controls.stop} onClick={() => void stop()}>
              Stop all test motion
            </button>
          </section>
          <section className={controls.panel} aria-label="Ownership controls">
            <h2>Control ownership</h2>
            <div className={controls.buttons}>
              <button
                disabled={busy || !canArm}
                onClick={() => void operate("manual")}
              >
                Arm manual control
              </button>
              <button
                disabled={busy || !canArm}
                onClick={() => void operate("policy")}
              >
                Arm policy test lease
              </button>
              <button
                disabled={
                  busy || status.mode === "fault" || status.mode === "closed"
                }
                onClick={() => void operate("takeover")}
              >
                Take over manually
              </button>
              <button
                disabled={busy || !["stopped", "fault"].includes(status.mode)}
                onClick={() => void operate("reset")}
              >
                Reset fault
              </button>
            </div>
            <p>
              Takeover stops the current group before granting a new manual
              lease. Previous-owner commands are rejected. The policy lease is
              an ownership test, not a learned policy.
            </p>
            <label className={controls.check}>
              <input
                type="checkbox"
                checked={heartbeats}
                onChange={(event) => {
                  setHeartbeats(event.target.checked);
                  sendingHeartbeats.current = event.target.checked;
                }}
              />
              Send owner heartbeats
            </label>
            <p>
              Turn heartbeats off to test the watchdog. Changing tabs or leaving
              this console releases this page’s lease and requests a stop.
            </p>
          </section>
          <section className={controls.panel} aria-label="Coordinated commands">
            <h2>Arms and grippers</h2>
            <div className={controls.buttons}>
              <button
                disabled={busy || !lease.current || status.mode !== "armed"}
                onClick={() => void operate("pose")}
              >
                Send coordinated test pose
              </button>
              <button
                disabled={busy || !lease.current || status.mode !== "armed"}
                onClick={() => void operate("neutral")}
              >
                Return to neutral
              </button>
            </div>
            <p>
              Test pose: left joint 1 +0.2 rad, right joint 1 −0.2 rad; gripper
              widths 20 / 40 mm. Neutral: all arm joints zero, both widths 60
              mm. Each command lasts 2 seconds, with a 5 N gripper force cap.
            </p>
            <div className={controls.readings}>
              <div>
                <h3>Left arm · rad</h3>
                <output>
                  {status.feedback?.left.map((q) => q.toFixed(3)).join(" · ")}
                </output>
                <p>
                  Gripper:{" "}
                  {((status.feedback?.grippers[0] ?? 0) * 1000).toFixed(1)} mm
                </p>
              </div>
              <div>
                <h3>Right arm · rad</h3>
                <output>
                  {status.feedback?.right.map((q) => q.toFixed(3)).join(" · ")}
                </output>
                <p>
                  Gripper:{" "}
                  {((status.feedback?.grippers[1] ?? 0) * 1000).toFixed(1)} mm
                </p>
              </div>
            </div>
            <p>
              These values come from a deterministic test backend. It does not
              model contact or measure force; a force cap is a requested limit,
              not force feedback.
            </p>
          </section>
          <section className={controls.panel}>
            <h2>Recent execution events</h2>
            <p>Bounded in-memory diagnostics for this service session.</p>
            <ol className={controls.events}>
              {status.events
                ?.slice(-12)
                .reverse()
                .map((event, index) => (
                  <li key={`${event.epoch}-${event.time}-${index}`}>
                    <strong>{event.kind}</strong> · epoch {event.epoch}
                    {event.reason && ` · ${event.reason}`}
                    {event.command_id && <code>{event.command_id}</code>}
                  </li>
                ))}
            </ol>
          </section>
        </>
      )}
    </main>
  );
}
