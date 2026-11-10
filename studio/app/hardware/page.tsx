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
  perception?: {
    state: string;
    reason: string;
    calibration_revision?: string;
    calibration_digest?: string;
    world_frame?: string;
    clock_domain?: string;
    age_s?: number;
    bundles?: number;
    skew_s?: number;
    joint_gap_s?: number;
    dropped?: number;
    rejected?: number;
    last_rejection?: string;
    scene_memory?: {
      surfaces?: number;
      remembered?: number;
      max_uncertainty_m?: number;
    };
    cameras?: {
      name: string;
      frame_id: string;
      mount_link: string;
      width: number;
      height: number;
      reprojection_error_px: number;
    }[];
    rejected_pairs?: { pair: string[]; reason: string }[];
  };
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
  const perceptionStale =
    offline ||
    (data?.perception?.age_s != null &&
      data.perception.age_s + Math.max(0, now - received) / 1000 > 0.5);
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
      <section
        className={styles.joints}
        aria-label="Calibration and synchronization"
      >
        <div className={styles.cardTitle}>
          <h2>Calibration & synchronization</h2>
          <span
            className={
              data?.perception?.state === "observing" && !perceptionStale
                ? styles.live
                : styles.warning
            }
          >
            {offline
              ? "Unavailable"
              : (data?.perception?.state?.replaceAll("_", " ") ??
                "Not configured")}
          </span>
        </div>
        <p>
          {data?.perception?.reason ??
            "Load a measured hardware calibration profile on the server to enable synchronized scene memory."}
        </p>
        {data?.perception?.calibration_revision ? (
          <>
            <p>
              Calibration{" "}
              <strong>{data.perception.calibration_revision}</strong> · World{" "}
              {data.perception.world_frame} · Clock{" "}
              {data.perception.clock_domain}
            </p>
            <p className={styles.digest}>
              SHA-256: {data.perception.calibration_digest}
            </p>
            <div className={styles.metrics}>
              <div>
                <small>Accepted bundles</small>
                <strong>{data.perception.bundles ?? 0}</strong>
              </div>
              <div>
                <small>Camera skew</small>
                <strong>
                  {data.perception.skew_s == null
                    ? "—"
                    : `${(data.perception.skew_s * 1000).toFixed(2)} ms`}
                </strong>
              </div>
              <div>
                <small>Joint bracket</small>
                <strong>
                  {data.perception.joint_gap_s == null
                    ? "—"
                    : `${(data.perception.joint_gap_s * 1000).toFixed(2)} ms`}
                </strong>
              </div>
              <div>
                <small>Memory samples</small>
                <strong>{data.perception.scene_memory?.surfaces ?? 0}</strong>
              </div>
              <div>
                <small>Remembered samples</small>
                <strong>{data.perception.scene_memory?.remembered ?? 0}</strong>
              </div>
              <div>
                <small>Max uncertainty</small>
                <strong>
                  {(
                    (data.perception.scene_memory?.max_uncertainty_m ?? 0) *
                    1000
                  ).toFixed(1)}{" "}
                  mm
                </strong>
              </div>
            </div>
            <p>
              Dropped buffered samples: {data.perception.dropped ?? 0} ·
              Rejected samples: {data.perception.rejected ?? 0}
            </p>
            {data.perception.last_rejection && (
              <p className={styles.warning}>
                Last rejection: {data.perception.last_rejection}
              </p>
            )}
            {data.perception.rejected_pairs?.map((pair) => (
              <p className={styles.warning} key={pair.pair.join("/")}>
                {pair.pair.join(" / ")}: {pair.reason}
              </p>
            ))}
            <div className={styles.scroll}>
              <table>
                <thead>
                  <tr>
                    <th>Camera</th>
                    <th>Optical frame</th>
                    <th>Mount link</th>
                    <th>Resolution</th>
                    <th>Calibration RMS</th>
                  </tr>
                </thead>
                <tbody>
                  {data.perception.cameras?.map((camera) => (
                    <tr key={camera.name}>
                      <td>{camera.name}</td>
                      <td>{camera.frame_id}</td>
                      <td>{camera.mount_link}</td>
                      <td>
                        {camera.width} × {camera.height}
                      </td>
                      <td>{camera.reprojection_error_px.toFixed(3)} px</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {!offline &&
              data.perception.age_s != null &&
              data.perception.age_s + Math.max(0, now - received) / 1000 <=
                0.5 &&
              data.perception.state !== "fault" && (
                <p>
                  <a
                    href="/api/hardware/scene"
                    target="_blank"
                    rel="noreferrer"
                  >
                    Inspect current hardware scene memory (JSON)
                  </a>
                </p>
              )}
          </>
        ) : (
          <p>
            Provide measured intrinsics, distortion, optical-to-mount
            transforms, a hardware URDF, and a verified shared clock.
            Uncalibrated preview images never enter scene memory.
          </p>
        )}
      </section>
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
        synchronization. Only the calibrated pipeline combines acquisition times
        and measured robot poses. Robot motion remains disabled.
      </footer>
    </main>
  );
}
