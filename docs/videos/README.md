# Panda simulation video

[Watch the full MP4](panda-grasp-place.mp4) · [Animated preview](panda-grasp-place.gif)

The video shows one complete scripted collection of a lemon from an eight-object
table: approach, finger alignment, grasp, lift, carry, lower, release, and
withdrawal. The main view and overhead/wrist cameras render the same simulation
state. Playback runs at normal simulation speed, with one-second initial and
two-second final holds. The final placement meets the demo environment's success
criterion. This is one demonstration, not a benchmark or a learned-policy result.

The MP4 uses H.264, 1280 × 720 pixels, 20 fps, no audio, and fast-start metadata.
The README uses a looping 640-pixel animated preview linked to the full MP4.
[panda-grasp-place.json](panda-grasp-place.json) records the seed, phases, model
hash, frame counts, timing, outcome, and media checksums.

## Regenerate

From the repository root, using the local environment created by `scripts/setup.sh`:

```sh
.venv/bin/robogogogo-demo --setup
.venv/bin/python -m pip install imageio-ffmpeg==0.6.0 'pillow>=10.1'
.venv/bin/python scripts/render_demo_video.py
```

A system FFmpeg executable can be used instead of `imageio-ffmpeg`. On a
headless NVIDIA Linux server, set `MUJOCO_GL=egl` before the rendering command.
On desktop macOS/Linux, use MuJoCo's default rendering backend. The script uses
existing assets and never downloads them implicitly.

Optional arguments: `--assets`, `--output`, `--work`, and `--seed`. Intermediate
files stay under ignored `runs/demo/video-render/`. Regeneration replaces the
three generated media/metadata files in this directory. The renderer stops on
a failed demonstration rather than labeling an incomplete motion as success.

## Asset attribution

- Panda robot meshes: Google DeepMind, [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/fb36886b3023db08c198264e1e9956ab202c3e3b/franka_emika_panda),
  pinned revision `fb36886b3023db08c198264e1e9956ab202c3e3b`, Apache-2.0.
- Textured objects: Calli et al., [YCB Object and Model Set, Google 16k scans](https://ycb-benchmarks.s3.amazonaws.com/index.html),
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
  Scans are centered/rotated and textures resized by the demo import pipeline;
  the video renders these processed assets in a simulated scene.
- Rendering/export script and original layout: project contributors, Apache-2.0.

Third-party assets retain their own licenses. See the repository [NOTICE](../../NOTICE).
