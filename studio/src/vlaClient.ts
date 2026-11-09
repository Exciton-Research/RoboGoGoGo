import { useEffect, useRef, useState, type RefObject } from "react";
import type { StereoReport } from "./components/PerceptionPanel";
import { request } from "./api";
import type { CameraConfig, DatasetEpisode, Snapshot } from "./types";
import type { ViewportHandle } from "./Viewport";

export function useVla(options: {
  viewport: RefObject<ViewportHandle | null>;
  accept: (snapshot: Snapshot) => void;
  setBusy: (busy: boolean) => void;
  setError: (error: string) => void;
  setNotice: (notice: string) => void;
}) {
  const [stereo, setStereo] = useState<StereoReport | null>(null);
  const [taskRunning, setTaskRunning] = useState(false);
  const [collecting, setCollecting] = useState(false);
  const [episodes, setEpisodes] = useState<DatasetEpisode[]>([]);
  const stop = useRef(false);
  useEffect(
    () => () => {
      stop.current = true;
    },
    [],
  );
  const handle = async <T>(work: () => Promise<T>) => {
    options.setBusy(true);
    options.setError("");
    options.setNotice("");
    try {
      return await work();
    } catch (error) {
      options.setError((error as Error).message);
      return null;
    } finally {
      options.setBusy(false);
    }
  };
  const refresh = async () => {
    try {
      setEpisodes(await request<DatasetEpisode[]>("/vla/episodes"));
    } catch (error) {
      options.setError((error as Error).message);
    }
  };
  const captureState = async (snapshot: Snapshot) => {
    if (!options.viewport.current)
      throw new Error("Observation camera is not available");
    const frames = options.viewport.current.capture(snapshot);
    const result = await request<{ observation: number; snapshot: Snapshot }>(
      "/vla/observe",
      { ...frames, sequence: snapshot.sequence },
    );
    options.accept(result.snapshot);
    return result;
  };
  const pausedState = async () => {
    let snapshot = await request<Snapshot>("/snapshot");
    if (snapshot.state.running)
      snapshot = await request<Snapshot>("/command", { op: "pause" });
    return snapshot;
  };
  const capture = () => handle(async () => captureState(await pausedState()));
  const start = (
    instruction: string,
    variant: string,
    cameras?: CameraConfig[],
  ) =>
    handle(async () => {
      const configured = cameras ?? (await pausedState()).episode?.cameras;
      const payload = configured?.map(
        ({
          name,
          width,
          height,
          mount,
          position,
          look_at,
          up,
          rpy,
          vertical_fov_degrees,
          near,
          far,
        }) => ({
          name,
          width,
          height,
          mount,
          position,
          look_at,
          up,
          rpy,
          vertical_fov_degrees,
          near,
          far,
        }),
      );
      const next = await request<Snapshot>("/vla/reset", {
        instruction,
        variant,
        cameras: payload,
      });
      options.accept(next);
      return next;
    });
  const execute = async (
    observation: number,
    delta: number[],
    source: string,
  ) => {
    const next = await request<Snapshot>("/vla/act", {
      delta,
      observation,
      control_ticks: 10,
      source,
    });
    options.accept(next);
    return captureState(next);
  };
  const act = (delta: number[]) =>
    handle(async () => {
      if (!delta.every(Number.isFinite))
        throw new Error("Action values must be finite numbers");
      const observation = await captureState(await pausedState());
      return execute(observation.observation, delta, "manual");
    });
  const collect = async () => {
    setCollecting(true);
    stop.current = false;
    await handle(async () => {
      let observation = await captureState(await pausedState());
      for (
        let i = 0;
        i < 80 && !stop.current && !observation.snapshot.episode?.done;
        i++
      ) {
        const action = await request<{ delta: number[]; source: string }>(
          "/control/rule-based-action",
        );
        observation = await execute(
          observation.observation,
          action.delta,
          action.source,
        );
      }
      options.setNotice(
        `Collected ${observation.snapshot.episode?.actions ?? 0} actions from the rule-based reach controller (simulator target state).`,
      );
    });
    setCollecting(false);
  };
  const analyzeStereo = (left: string, right: string) =>
    handle(async () => {
      const captured = await captureState(await pausedState());
      setStereo(
        await request<StereoReport>("/perception/stereo", {
          left,
          right,
          observation: captured.observation,
        }),
      );
    });
  const planTask = (
    kind: "inspection" | "handover" | "material_grasp" = "inspection",
    grip?: import("./types").GripParameters,
  ) =>
    handle(async () => {
      await pausedState();
      const next = await request<Snapshot>("/tasks/plan", { kind, ...grip });
      options.accept(next);
      options.setNotice(
        "Task planned with collision-checked paths. Run it to record observations, actions, and task events.",
      );
    });
  const runTask = async () => {
    setTaskRunning(true);
    stop.current = false;
    await handle(async () => {
      let snapshot = await pausedState();
      if (snapshot.coordination?.status === "paused") {
        snapshot = await request<Snapshot>("/tasks/resume", {});
        options.accept(snapshot);
      }
      let captured = await captureState(snapshot);
      for (
        let i = 0;
        i < 450 &&
        !stop.current &&
        ["ready", "running"].includes(
          captured.snapshot.coordination?.status ?? "",
        );
        i++
      ) {
        const next = await request<Snapshot>("/tasks/step", {
          observation: captured.observation,
        });
        options.accept(next);
        captured =
          next.coordination?.status === "blocked" &&
          next.episode?.actions === captured.snapshot.episode?.actions
            ? { ...captured, snapshot: next }
            : await captureState(next);
      }
      if (
        ["ready", "running"].includes(
          captured.snapshot.coordination?.status ?? "",
        )
      ) {
        options.accept(await request<Snapshot>("/tasks/pause", {}));
        options.setNotice(
          "Coordinated task paused. Resume to continue the current stage.",
        );
      } else {
        options.setNotice(
          `Coordinated task ${captured.snapshot.coordination?.status}: ${captured.snapshot.episode?.actions ?? 0} aligned actions recorded.`,
        );
      }
    });
    setTaskRunning(false);
  };
  const save = () =>
    handle(async () => {
      const result = await request<DatasetEpisode>("/vla/save", {});
      await refresh();
      const link = document.createElement("a");
      link.href = `/api/vla/episodes/${result.id}`;
      link.download = `robogogogo-episode-${result.id.slice(0, 8)}.zip`;
      link.click();
      options.setNotice(
        "Episode saved locally and exported with aligned RGB, instructions, state, and actions.",
      );
      return result;
    });
  return {
    planTask,
    runTask,
    taskRunning,
    stereo,
    analyzeStereo,
    capture,
    start,
    act,
    collect,
    save,
    refresh,
    episodes,
    collecting,
    stop: () => {
      stop.current = true;
    },
  };
}
