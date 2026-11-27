# Panda grasp-and-place environment

One Franka Emika Panda grasps a 50 mm, 80 g block, carries it to a marked target,
releases it, and withdraws. The environment provides seeded reset, explicit
stepping, manual control, local episode recording, and read-only replay.
MuJoCo simulates the arm, gripper, block, and table contacts. Three.js displays
the simulated poses using the official robot meshes.

The reference controller uses scripted Jacobian IK with smooth joint targets.
It is not a learned policy or a real-robot result. The block has a free joint
throughout the task; it is never attached or welded to the hand. Only reset
writes its initial position.

![The Panda environment replaying a recorded grasp with overhead and wrist RGB](images/panda-grasp-place.jpg)

## Start

```sh
bash scripts/setup.sh
.venv/bin/robogogogo-demo
```

Open [http://127.0.0.1:8765](http://127.0.0.1:8765). For an existing installation:

```sh
.venv/bin/pip install -e '.[studio,demo]'
.venv/bin/python -m robogogogo.demo --setup
npm run build --prefix studio
.venv/bin/python -m robogogogo.demo
```

Setup explicitly downloads pinned official Panda assets into the Git-ignored
`runs/demo/assets/` cache. Requests do not download assets. The standalone server
binds to localhost and serves only the Panda environment. The original RG Mobile
Duo workspace remains available through `robogogogo-studio` at `/simulation/`
and `/workflow/`.

## Control one task

- **Reset:** apply the chosen seed and restore the robot and block. Seed 0 uses
  the original placement; other seeds jitter the initial block position within
  a bounded tabletop region. Save or discard an active recording before reset.
- **Run reference task:** execute the scripted grasp-and-place motion. Pause
  freezes simulation and allows manual takeover. Manual action replaces the
  reference controller; its next run plans from the current pose.
- **Manual:** jog the tool by 5, 10, or 20 mm in world coordinates; choose a total
  finger opening from 0 to 80 mm; or hold the current targets for one 50 ms step.
  Each jog spans six control steps (0.3 s); changing the gripper spans ten
  (0.5 s). IK rejects unreachable targets.
- **Record episode:** capture manual, API, or scripted actions. RGB is enabled
  by default. Save retains partial runs; discard deletes the active recording.
  Environment success or termination automatically saves an active recording.
- **Episodes:** replay a saved run, seek to an exact frame, or download its ZIP.
  Replay is read-only and shows stored states rather than re-running actions.
  **Return to live control** restores the pre-replay physical state. A reference
  task that was running before replay returns paused.

Simulation does not advance while idle in manual mode. The reference task and
replay advance only while playing. One server is one shared session across tabs;
manual requests with a stale state revision are rejected.

Drag to orbit, scroll to zoom, and use the frame button to restore the view.
Live overhead and wrist previews use Three.js with camera poses and fields of
view from MuJoCo. Recorded RGB comes from MuJoCo's native renderer, so its
materials and lighting differ from the browser preview. RGB replay displays the
saved PNGs alongside the recorded physical state.

## Python environment

```python
from robogogogo.demo import PandaEnv

with PandaEnv() as env:
    observation, info = env.reset(seed=7)
    action = observation["previous_action"].copy()
    observation, reward, terminated, truncated, info = env.step(action)
```

`PandaEnv` implements the Gymnasium API. Each action is eight absolute position
targets: seven arm joints in radians, followed by total finger opening in meters
(`[0, 0.08]`). Joint bounds come from `env.action_space`; invalid targets are
rejected. Each `step` runs 25 physics steps of 2 ms, giving a 20 Hz control rate.
The default time limit is 800 control steps (40 simulated seconds). Terminated
or truncated episodes require a reset before another action.

The state observation contains `proprio` (25 values: nine joint positions, nine
velocities, tool world XYZ, and tool quaternion XYZW), `previous_action` (8), and
`time` (1, seconds since reset). Block/target coordinates and success diagnostics
are in `info`, not the agent observation. `observation_mode="rgb"` additionally
returns `overhead` and `wrist` uint8 RGB arrays of shape `(144, 240, 3)`.
`render_mode="rgb_array"` supports `env.render()` for the overhead image.

State-only stepping needs no native OpenGL context. RGB observations and
recording require a working MuJoCo offscreen renderer; disable RGB for state-only
recording on systems without one. Browser previews require WebGL. The environment
and renderer belong to a single thread; the server owns them on a dedicated
worker. MuJoCo and Gymnasium are optional `demo` dependencies; the existing JAX
workspace remains separate.

The browser uses `/api/demo/reset`, `/step`, `/manual`, `/command`, `/record`, and
`/replay`. A direct step request contains `{"action": [eight targets]}` and may
include `expected_revision` from `/api/demo/state` to reject stale commands.
These routes control the same live environment, not separate per-client instances.

## Recordings

Archives stay under the Git-ignored `runs/demo/episodes/` directory. Select a
stable location with `robogogogo-demo --recordings /path/to/episodes`.

Each `panda-episode-v1` ZIP contains:

- `episode.json`: task instruction, simulated origin, seed, outcome, model hash,
  pinned model revision, MuJoCo version, timing, action units/bounds, observation
  ordering, camera intrinsics, and camera pose convention.
- `trajectory.npz`: H actions, rewards, termination/truncation flags, and action
  sources; H+1 observations, timestamps, joint states, camera-to-world matrices,
  and native integration states with task bookkeeping. Loading uses no pickle.
- Optional `rgb/overhead/00000.png` and `rgb/wrist/00000.png` sequences: H+1
  frames per camera, captured at the same boundaries as the observations.

Recordings can start partway through a live run. The first observation is the
state at recording start; times retain their offset from environment reset.
Manual sources are labeled `manual`, the reference `scripted`, and direct step
requests `api`. Partial and unsuccessful runs are retained when saved. Replay
requires a matching model hash and MuJoCo version. It is state playback, not a
claim of cross-version deterministic re-simulation.

## What completion means

Success requires all of the following for six consecutive control steps:

- block center within 35 mm of the destination;
- block height within 8 mm of its resting height on the table;
- translational speed below 20 mm/s;
- open gripper with no finger contact;
- an earlier lift of at least 95 mm above the resting block center;
- tool center more than 120 mm clear of the block.

Reward is 1 on success and 0 otherwise. Success or a block falling below the
workspace terminates the episode; reaching the time limit truncates it. The
reference controller also checks bilateral contact before lift and elevation
after lift/carry. A reference failure pauses its motion and displays the reason;
manual recovery remains possible if the environment has not ended.

The UI reports measured lift, finger opening, and target distance. The motion
uses fixed waypoints: this is one bounded task, without general collision-free
planning, arbitrary object grasping, or a trained recovery policy.

## Model provenance

The robot comes from [MuJoCo Menagerie's Panda model](https://github.com/google-deepmind/mujoco_menagerie/tree/fb36886b3023db08c198264e1e9956ab202c3e3b/franka_emika_panda),
under its upstream Apache 2.0 license. The ignored asset cache retains the source
license, README, pinned revision, and file digests. The browser links to the source
and `/api/demo/license` serves its license. The workbench, block, controllers,
environment API, and UI are project additions. These assets do not imply that we
designed or manufactured the Panda robot.
