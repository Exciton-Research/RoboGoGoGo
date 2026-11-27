# Panda grasp-and-place demo

The home page shows one task: a Franka Emika Panda picks up a 50 mm, 80 g block,
carries it to the marked destination, opens its gripper, and withdraws. MuJoCo
simulates the arm, gripper, block, and table contacts. Three.js renders the actual
simulated poses using the official robot meshes.

The controller is scripted Jacobian IK with smooth joint targets. This is a
simulation demonstration, not a learned policy or a real-robot result. The block
has a free joint throughout the task. It is never attached or welded to the hand;
only reset writes its initial position.

![The actual demo paused while the Panda holds the block above the workbench](images/panda-grasp-place.jpg)

## Start

The checkout setup builds the UI and downloads the pinned official Panda assets
into the Git-ignored `runs/demo/assets/` cache:

```sh
bash scripts/setup.sh
.venv/bin/robogogogo-demo
```

Open [http://127.0.0.1:8765](http://127.0.0.1:8765). Select **Run grasp & place**.
Drag the scene to orbit; use the frame button to restore the initial camera.
Pause/resume freezes and continues simulation. Reset restores the task.
**Try a new starting position** moves the block within a small declared workspace
at reset. The overhead and virtual wrist views show the same live scene.

For an existing environment, install `.[studio,demo]`, rebuild the UI, and fetch
the assets once:

```sh
.venv/bin/pip install -e '.[studio,demo]'
.venv/bin/python -m robogogogo.demo --setup
npm run build --prefix studio
.venv/bin/python -m robogogogo.demo
```

Asset downloading is explicit; HTTP requests do not fetch files from the internet.
The first load compiles the upstream meshes. Subsequent scene requests reuse
cached compiled geometry and transfer packed buffers. Browser WebGL is required;
native offscreen OpenGL, a GPU compute stack, and JAX startup are not needed for
the standalone demo server. The default server binds only to localhost.

The standalone server serves the demo API only. Start `robogogogo-studio` to use
the original RG Mobile Duo workspace, hardware interfaces, and the workflow now
at `/workflow/`. That server also serves the Panda demo if its dependencies and
cached assets are available.

## What completion means

The controller must establish contact on both fingers before lifting. It checks
that the block remains elevated after lift and carry. Final success requires:

- the block center within 35 mm of the destination center;
- its height within 8 mm of its resting height on the tabletop;
- translational speed below 20 mm/s after settling;
- an open gripper with no finger contact;
- an earlier lift of at least 95 mm above the resting block center.

The UI reports measured lift, finger opening, and horizontal target distance.
A missed grasp or slipped block stops the task and is displayed as a failure.
The motion uses fixed waypoints and does not provide general collision-free
planning, arbitrary object grasping, or recovery.

## Model provenance

The robot comes from [MuJoCo Menagerie's Panda model](https://github.com/google-deepmind/mujoco_menagerie/tree/fb36886b3023db08c198264e1e9956ab202c3e3b/franka_emika_panda),
under its upstream Apache 2.0 license. The cache retains the source license,
README, pinned revision, and file digests. The browser links to the source and
the server provides the license at `/api/demo/license`. The workbench, block,
controller, and UI are project additions. These assets are not a claim that we
designed or manufactured the Panda robot.
