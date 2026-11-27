"""One simulation actor: explicit stepping, reference playback, recording, and replay."""

import json
import queue
import threading
import time
import zipfile
from concurrent.futures import Future
from pathlib import Path
from uuid import UUID, uuid4

import numpy as np

from .controller import ReferenceController
from .environment import PandaEnv
from .recording import Episode, Recording


class PandaSession:
    def __init__(self, assets, directory):
        self.assets, self.directory = assets, Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.requests = queue.Queue()
        self.closed = threading.Event()
        self.thread = threading.Thread(target=self._run, name="panda-environment", daemon=True)
        self.thread.start()

    def call(self, method, **kwargs):
        if self.closed.is_set():
            raise getattr(self, "startup_error", RuntimeError("Environment session is closed"))
        future = Future()
        self.requests.put((method, kwargs, future))
        return future.result(timeout=120)

    def _run(self):
        env = None
        try:
            env = self.env = PandaEnv(self.assets)
            self.status, self.mode, self.message = (
                "ready",
                "manual",
                "Ready for manual control or the reference task.",
            )
            self.seed, self.revision, self.controller = 0, 0, None
            self.recording, self.replay, self.backup = None, None, None
            self.frame, self.playing, self.next_tick = 0, False, time.perf_counter()
            self.last_saved = None
            self.episode_id = str(uuid4())
            self.session_id = str(uuid4())
            self.library = self._list_episodes()
            while not self.closed.is_set():
                now = time.perf_counter()
                if self.playing and now >= self.next_tick:
                    try:
                        self._tick()
                    except Exception as error:
                        self.status, self.message, self.playing = "failed", str(error), False
                    self.next_tick = max(self.next_tick + env.dt, time.perf_counter())
                    continue
                try:
                    method, kwargs, future = self.requests.get(
                        timeout=max(0, self.next_tick - now) if self.playing else 0.1
                    )
                except queue.Empty:
                    continue
                try:
                    result = getattr(self, "_" + method)(**kwargs)
                    future.set_result(result)
                except Exception as error:
                    future.set_exception(error)
        except Exception as error:
            self.startup_error = error
        finally:
            self.closed.set()
            if getattr(self, "recording", None) is not None:
                # Preserve any finished transitions on ordinary shutdown.
                try:
                    if self.recording.actions:
                        self.recording.save(env, "shutdown")
                except Exception as error:
                    self.shutdown_error = error
                finally:
                    self.recording.close()
            if env is not None:
                env.close()
            while not self.requests.empty():
                _, _, future = self.requests.get_nowait()
                future.set_exception(getattr(self, "startup_error", RuntimeError("Session closed")))

    def _snapshot(self):
        controller = self.controller
        return {
            **self.env.snapshot(),
            "revision": self.revision,
            "session_id": self.session_id,
            "episode_id": self.episode_id,
            "seed": self.replay.manifest["seed"] if self.replay else self.seed,
            "ended": bool(self.env.terminated or self.env.truncated),
            "status": self.status,
            "mode": self.mode,
            "message": self.message,
            "phase": controller.phase if controller is not None else 0,
            "phases": list(ReferenceController.labels),
            "progress": (
                (
                    controller.phase
                    + min(1, controller.elapsed / controller.phases[controller.phase][0])
                )
                / 9
            )
            if controller is not None
            else 0,
            "recording": None
            if self.recording is None
            else {
                "id": self.recording.id,
                "transitions": len(self.recording.actions),
                "rgb": self.recording.images,
            },
            "episodes": self.library,
            "last_saved": self.last_saved,
            "replay": None
            if self.replay is None
            else {
                "id": self.replay.manifest["id"],
                "frame": self.frame,
                "frames": self.replay.count + 1,
                "playing": self.playing,
                "rgb": self.replay.manifest["rgb"],
                "outcome": self.replay.manifest["outcome"],
            },
        }

    def _scene(self):
        return {**self.env.geometry(), "state": self._snapshot()}

    def _guard_live(self, expected_revision=None):
        if self.replay is not None:
            raise ValueError("Replay is read-only. Return to live control first.")
        if self.playing:
            raise ValueError("Pause the reference task before manual control")
        if self.env.terminated or self.env.truncated:
            raise ValueError("Episode ended; reset before stepping")
        if expected_revision is not None and expected_revision != self.revision:
            raise ValueError("State changed in another request; refresh before acting")

    def _advance(self, action, source):
        result = self.env.step(action)
        self.revision += 1
        if self.recording is not None:
            self.recording.append(self.env, action, result, source)
        if result[2] or result[3]:
            self.playing = False
            self.status = "complete" if result[4]["success"] else "failed"
            self.message = (
                "Block placed, released, and settled."
                if result[4]["success"]
                else "Episode ended: " + str(result[4]["reason"])
            )
            if self.recording is not None:
                self._finish_recording("environment_ended")
        return result

    def _step(self, action, expected_revision=None):
        self._guard_live(expected_revision)
        action = np.asarray(action, dtype=float)
        if not self.env.action_space.contains(action):
            raise ValueError("Action must contain seven joint targets in rad and opening in m")
        self.controller, self.mode, self.status = None, "manual", "paused"
        self._advance(action, "api")
        return self._snapshot()

    def _jog(self, delta, opening=None, expected_revision=None):
        self._guard_live(expected_revision)
        delta = np.asarray(delta, dtype=float)
        if (
            delta.shape != (3,)
            or not np.all(np.isfinite(delta))
            or np.linalg.norm(delta) > 0.020001
        ):
            raise ValueError("Manual translation must be a finite vector of at most 20 mm")
        target = self.env.data.site_xpos[self.env.site] + delta
        if np.any(target < [0.12, -0.38, 0.755]) or np.any(target > [0.75, 0.38, 1.35]):
            raise ValueError("Manual target is outside the tabletop workspace")
        gap = float(self.env.previous_action[7] if opening is None else opening)
        if not np.isfinite(gap) or not 0 <= gap <= 0.08:
            raise ValueError("Gripper opening must be in [0,0.08] m")
        start = self.env.data.qpos[:7].copy()
        goal = self.env.solve(target, start)
        self.controller, self.mode, self.status = None, "manual", "paused"
        self.message = "Manual control · simulation advances only when you act."
        count = 10 if opening is not None else 6
        for index in range(1, count + 1):
            t = index / count
            blend = t * t * t * (10 + t * (-15 + 6 * t))
            self._advance(np.r_[start + blend * (goal - start), gap], "manual")
            if self.env.terminated or self.env.truncated:
                break
        return self._snapshot()

    def _reset(self, seed=0):
        if self.replay is not None:
            raise ValueError("Return to live control before resetting")
        if self.recording is not None:
            raise ValueError("Save or discard the current recording before resetting")
        self.playing = False
        self.env.reset(seed=seed)
        self.seed, self.controller, self.mode, self.status = seed, None, "manual", "ready"
        self.episode_id = str(uuid4())
        self.message = "Reset complete. Ready for manual control or the reference task."
        self.revision += 1
        return self._snapshot()

    def _command(self, command, seed=0):
        if command == "reset":
            return self._reset(seed)
        if command == "pause":
            self.playing = False
            self.status = "paused"
        elif command == "run":
            self._guard_live()
            if self.controller is None:
                self.controller = ReferenceController(self.env)
            self.mode, self.status, self.playing = "script", "running", True
            self.next_tick = time.perf_counter()
        else:
            raise ValueError("Unknown command")
        self.revision += 1
        return self._snapshot()

    def _tick(self):
        if self.replay is not None:
            if self.frame < self.replay.count:
                self._seek(self.frame + 1)
            else:
                self.playing, self.status = False, "paused"
            return
        action = self.controller.action(self.env)
        self._advance(action, "scripted")
        if not self.env.terminated and not self.env.truncated:
            self.controller.advance(self.env)
            self.message = self.controller.labels[self.controller.phase]

    def _record(self, command, images=True):
        if self.replay is not None:
            raise ValueError("Return to live control before recording")
        if command == "start":
            if self.recording is not None:
                raise ValueError("A recording is already active")
            if self.env.terminated or self.env.truncated:
                raise ValueError("Reset before starting a new recording")
            self.recording = Recording(self.directory, self.env, seed=self.seed, images=images)
        elif command == "save":
            self._finish_recording("user_stopped")
        elif command == "discard":
            if self.recording is not None:
                self.recording.close()
                self.recording = None
        else:
            raise ValueError("Unknown recording command")
        self.revision += 1
        return self._snapshot()

    def _finish_recording(self, reason):
        if self.recording is None:
            raise ValueError("No recording is active")
        self.last_saved = self.recording.save(self.env, reason)
        self.recording = None
        self.library = self._list_episodes()

    def _list_episodes(self):
        episodes = []
        for path in sorted(
            self.directory.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True
        )[:64]:
            try:
                with zipfile.ZipFile(path) as archive:
                    if archive.getinfo("episode.json").file_size > 100_000:
                        continue
                    row = json.loads(archive.read("episode.json"))
                if row.get("schema") == "panda-episode-v1" and str(UUID(row["id"])) == path.stem:
                    episodes.append(
                        {
                            key: row[key]
                            for key in (
                                "id",
                                "created",
                                "seed",
                                "transitions",
                                "duration",
                                "outcome",
                                "rgb",
                            )
                        }
                    )
            except (ValueError, KeyError, OSError, zipfile.BadZipFile):
                continue
        return episodes

    def episode_path(self, identifier):
        return self.directory / f"{UUID(str(identifier))}.zip"

    def _load_replay(self, identifier):
        if self.recording is not None:
            raise ValueError("Save or discard the current recording before replay")
        episode = Episode(self.episode_path(identifier), self.env)
        if self.replay is None:
            self.backup = (self.env.capture_state(), self.status, self.mode, self.message)
        self.playing, self.replay, self.mode, self.status = False, episode, "replay", "paused"
        return self._seek(0)

    def _seek(self, frame):
        if self.replay is None:
            raise ValueError("No replay is loaded")
        self.env.restore_state(self.replay.state(frame))
        self.frame = frame
        self.message = "Recorded state · return to live control to act."
        self.revision += 1
        return self._snapshot()

    def _replay_command(self, command, frame=None):
        if self.replay is None:
            raise ValueError("No replay is loaded")
        if command == "seek":
            self.playing = False
            self._seek(frame)
        elif command == "play":
            if self.frame == self.replay.count:
                self._seek(0)
            self.playing = True
            self.next_tick = time.perf_counter()
        elif command == "pause":
            self.playing = False
        elif command == "exit":
            self.playing = False
            state, status, mode, message = self.backup
            self.env.restore_state(state)
            self.status, self.mode, self.message = (
                "paused" if status == "running" else status,
                mode,
                message,
            )
            self.replay = self.backup = None
        else:
            raise ValueError("Unknown replay command")
        self.revision += 1
        return self._snapshot()

    def close(self):
        self.closed.set()
        self.thread.join(timeout=5)
