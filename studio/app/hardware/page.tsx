"use client";

import { useEffect, useRef, useState } from "react";
import { request } from "../../src/api";
import styles from "./hardware.module.css";

type Stream = {
  timestamp: number;
  frame_id: string;
  age_s: number;
  stale: boolean;
  revision: number;
};
type Camera = Stream & { name: string; media_type: string };
type Telemetry = {
  backends: string[];
  backend: string | null;
  status: string;
  error: string | null;
  session: number;
  joints:
    | (Stream & {
        names: string[];
        position: number[];
        velocity: number[];
        effort: number[];
      })
    | null;
  cameras: Camera[];
};

function CameraView({
  camera,
  session,
  stale,
}: {
  camera: Camera;
  session: number;
  stale: boolean;
}) {
  const [failed, setFailed] = useState(false);
  const url = `/api/hardware/cameras/${encodeURIComponent(camera.name)}?session=${session}&revision=${camera.revision}`;
  useEffect(() => setFailed(false), [url]);
  return (
    <article className={styles.camera}>
      <div className={styles.cardTitle}>
        <strong>{camera.name}</strong>
        <span className={stale || failed ? styles.warning : styles.live}>
          {failed
            ? "Image unavailable"
            : stale
              ? "Stale · last received image"
              : "Receiving"}
        </span>
      </div>
      <img
        src={url}
        alt={`${camera.name} robot camera`}
        onError={() => setFailed(true)}
      />
      <p>
        Frame: {camera.frame_id || "not supplied"} · Acquisition{" "}
        {camera.timestamp.toFixed(3)} s
      </p>
    </article>
  );
}

export default function HardwarePage() {
  const [data, setData] = useState<Telemetry | null>(null);
  const [backend, setBackend] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [received, setReceived] = useState(0);
  const [now, setNow] = useState(Date.now());
  const generation = useRef(0);
  useEffect(() => {
    let stopped = false;
    let timer: ReturnType<typeof setTimeout>;
    const ticker = setInterval(() => setNow(Date.now()), 500);
    const poll = async () => {
      const version = generation.current;
      try {
        const response = await fetch("/api/hardware", {
          cache: "no-store",
          signal: AbortSignal.timeout(3000),
        });
        if (!response.ok)
          throw new Error(`Hardware monitor unavailable (${response.status})`);
        const snapshot: Telemetry = await response.json();
        if (!stopped && version === generation.current) {
          setData(snapshot);
          setReceived(Date.now());
          setBackend((value) => value || snapshot.backends[0] || "");
        }
      } catch {
        // Local freshness timer marks retained values stale during network loss.
      } finally {
        if (!stopped) timer = setTimeout(poll, 500);
      }
    };
    void poll();
    return () => {
      stopped = true;
      clearTimeout(timer);
      clearInterval(ticker);
    };
  }, []);
  async function change(connect: boolean) {
    generation.current += 1;
    setBusy(true);
    setError("");
    try {
      const snapshot = await request<Telemetry>(
        connect ? "/hardware/connect" : "/hardware/disconnect",
        connect ? { backend } : {},
      );
      generation.current += 1;
      setData(snapshot);
      setReceived(Date.now());
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : String(failure));
    } finally {
      setBusy(false);
    }
  }
  const offline = now - received > 2500;
  const stale = (stream: Stream) =>
    offline ||
    stream.stale ||
    stream.age_s + Math.max(0, now - received) / 1000 > 2;
  return (
    <main className={styles.page}>
      <header>
        <a href="/">← robogogogo Studio</a>
        <span className={styles.badge}>READ-ONLY HARDWARE</span>
      </header>
      <h1>Robot telemetry</h1>
      <p className={styles.intro}>
        Observe physical joint states and head or wrist cameras. Connecting here
        does not enable robot movement.
      </p>
      <section className={styles.connection} aria-label="Hardware connection">
        <label>
          Registered backend
          <select
            aria-label="Hardware backend"
            value={backend}
            disabled={busy || !data?.backends.length}
            onChange={(event) => setBackend(event.target.value)}
          >
            {!data?.backends.length && (
              <option value="">No backend configured</option>
            )}
            {data?.backends.map((name) => (
              <option key={name} value={name}>
                {name}
              </option>
            ))}
          </select>
        </label>
        <button
          disabled={busy || !backend || offline}
          onClick={() => void change(true)}
        >
          {busy ? "Updating…" : "Connect / reconnect"}
        </button>
        <button
          disabled={busy || !data?.backend || offline}
          onClick={() => void change(false)}
        >
          Disconnect
        </button>
        <strong
          role="status"
          className={
            offline || data?.status !== "streaming"
              ? styles.warning
              : styles.live
          }
        >
          {offline ? "Studio connection unavailable" : data?.status}
        </strong>
      </section>
      {(error || data?.error) && (
        <p className={styles.error} role="alert">
          {error || data?.error}
        </p>
      )}
      {data && !data.backends.length && (
        <p className={styles.empty}>
          Start Studio with a ROS 2 topic configuration, or register a
          manufacturer SDK telemetry adapter in Python. Backends are configured
          on the server.
        </p>
      )}
      <section className={styles.joints}>
        <div className={styles.cardTitle}>
          <h2>Joint states</h2>
          <span
            className={
              data?.joints && !stale(data.joints) ? styles.live : styles.warning
            }
          >
            {data?.joints
              ? stale(data.joints)
                ? "Stale · last received values"
                : "Receiving"
              : "Waiting for joint telemetry"}
          </span>
        </div>
        {data?.joints ? (
          <>
            <p>
              Acquisition {data.joints.timestamp.toFixed(3)} s · Frame{" "}
              {data.joints.frame_id || "not supplied"}
            </p>
            <div className={styles.scroll}>
              <table>
                <thead>
                  <tr>
                    <th>Joint</th>
                    <th>Position (rad / m)</th>
                    <th>Velocity (rad/s / m/s)</th>
                    <th>Effort (N·m / N)</th>
                  </tr>
                </thead>
                <tbody>
                  {data.joints.names.map((name, i) => (
                    <tr key={name}>
                      <td>{name}</td>
                      <td>{data.joints!.position[i].toFixed(4)}</td>
                      <td>
                        {data.joints!.velocity[i]?.toFixed(4) ?? "Not supplied"}
                      </td>
                      <td>
                        {data.joints!.effort[i]?.toFixed(3) ?? "Not supplied"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        ) : (
          <p>
            Named sensor measurements appear here after the backend receives
            them.
          </p>
        )}
      </section>
      <h2>
        Camera feeds <small>({data?.cameras.length ?? 0})</small>
      </h2>
      <div className={styles.cameras}>
        {data?.cameras.map((camera) => (
          <CameraView
            key={`${data.session}:${camera.name}`}
            session={data.session}
            camera={camera}
            stale={stale(camera)}
          />
        ))}
      </div>
      {!data?.cameras.length && (
        <p className={styles.empty}>
          Waiting for camera frames. Configure one or more PNG/JPEG streams.
        </p>
      )}
      <footer>
        Telemetry is separate from the simulation. Acquisition timestamps may
        use a robot clock; reception freshness does not establish camera/joint
        synchronization. Camera calibration and robot motion are not enabled on
        this page.
      </footer>
    </main>
  );
}
