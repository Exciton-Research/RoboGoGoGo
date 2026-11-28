"""Render a complete scripted Panda collection as MP4 plus an animated preview.

Requires the demo dependencies and an FFmpeg executable (or imageio-ffmpeg).
Assets must already be installed with robogogogo-demo --setup. On a headless
NVIDIA server, set MUJOCO_GL=egl before starting Python. No browser is required.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

import mujoco
from PIL import Image, ImageDraw, ImageFont

from robogogogo.demo.controller import ReferenceController
from robogogogo.demo.environment import PandaEnv
from robogogogo.demo.objects import CREDIT, DEFAULT_CLUTTER, LICENSE, SOURCE

WIDTH, HEIGHT, FPS = 1280, 720, 20
PAPER, INK, MUTED, GREEN = "#f7f6f2", "#202922", "#66736b", "#345348"
PHASES = ("Approach", "Align", "Grasp", "Lift", "Carry", "Lower", "Release", "Withdraw", "Settle")


@lru_cache
def font(size):
    return ImageFont.load_default(size=size)


def ffmpeg_path():
    if installed := shutil.which("ffmpeg"):
        return installed
    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise RuntimeError(
            "Install FFmpeg or run python -m pip install imageio-ffmpeg==0.6.0"
        ) from exc
    return imageio_ffmpeg.get_ffmpeg_exe()


def compose(main, overhead, wrist, *, phase, seconds, success=False):
    frame = Image.new("RGB", (WIDTH, HEIGHT), PAPER)
    draw = ImageDraw.Draw(frame)
    draw.text((28, 22), "robogogogo.", font=font(32), fill=INK)
    draw.text((265, 34), "PANDA / GRASP & PLACE", font=font(16), fill=MUTED)
    draw.text((985, 30), "SIMULATION  /  1x", font=font(18), fill=GREEN)
    draw.line((24, 78, 1256, 78), fill="#d7dcd5", width=1)
    frame.paste(Image.fromarray(main), (24, 96))
    draw.text(
        (24, 650), "Scripted controller  /  MuJoCo contact dynamics", font=font(18), fill=MUTED
    )
    draw.text((956, 95), "LIVE CAMERA VIEWS", font=font(15), fill=MUTED)
    frame.paste(Image.fromarray(overhead), (936, 124))
    draw.text((936, 312), "01  OVERHEAD", font=font(16), fill=MUTED)
    frame.paste(Image.fromarray(wrist), (936, 350))
    draw.text((936, 538), "02  WRIST", font=font(16), fill=MUTED)
    draw.rounded_rectangle((936, 576, 1256, 634), radius=8, fill=GREEN)
    label = "Placed & released" if success else PHASES[phase]
    draw.text((954, 592), label, font=font(24), fill="white")
    draw.text((1132, 655), f"{seconds:4.1f} s", font=font(19), fill=INK)
    draw.rounded_rectangle((38, 108, 453, 143), radius=5, fill=PAPER)
    draw.text((50, 117), "TARGET  Lemon  >  marked collection cell", font=font(17), fill=INK)
    for i in range(len(PHASES)):
        x = 24 + i * 100
        draw.rounded_rectangle(
            (x, 688, x + 90, 695), radius=3, fill=GREEN if success or i <= phase else "#d7dcd5"
        )
    draw.text((936, 688), "YCB scans / Calli et al.", font=font(14), fill=MUTED)
    return frame


def render(args):
    executable = ffmpeg_path()
    args.output.mkdir(parents=True, exist_ok=True)
    args.work.mkdir(parents=True, exist_ok=True)
    movie = args.output / "panda-grasp-place.mp4"
    preview = args.output / "panda-grasp-place.gif"
    with tempfile.TemporaryDirectory(prefix="render-", dir=args.work) as work:
        pending = Path(work) / movie.name
        with PandaEnv(args.assets, object_id="lemon", object_ids=list(DEFAULT_CLUTTER)) as env:
            env.reset(seed=args.seed)
            env.model.vis.global_.offwidth = 1024
            env.model.vis.global_.offheight = 576
            options = mujoco.MjvOption()
            options.geomgroup[3] = 0
            camera = mujoco.MjvCamera()
            camera.lookat[:] = [0.35, 0.0, 0.92]
            camera.distance, camera.azimuth, camera.elevation = 1.65, 135, -28
            controller = ReferenceController(env)
            process = subprocess.Popen(
                [
                    executable,
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pixel_format",
                    "rgb24",
                    "-video_size",
                    f"{WIDTH}x{HEIGHT}",
                    "-framerate",
                    str(FPS),
                    "-i",
                    "pipe:0",
                    "-an",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "medium",
                    "-crf",
                    "22",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(pending),
                ],
                stdin=subprocess.PIPE,
            )
            frames, phases = 0, []
            try:
                with (
                    mujoco.Renderer(env.model, height=540, width=900) as main_renderer,
                    mujoco.Renderer(env.model, height=180, width=320) as cameras,
                ):

                    def capture(success=False):
                        nonlocal frames
                        main_renderer.update_scene(env.data, camera=camera, scene_option=options)
                        main = main_renderer.render().copy()
                        cameras.update_scene(env.data, camera="overhead", scene_option=options)
                        overhead = cameras.render().copy()
                        cameras.update_scene(env.data, camera="wrist", scene_option=options)
                        wrist = cameras.render().copy()
                        frame = compose(
                            main,
                            overhead,
                            wrist,
                            phase=controller.phase,
                            seconds=env.steps * env.dt,
                            success=success,
                        )
                        process.stdin.write(frame.tobytes())
                        if frames == 0:
                            frame.save(args.work / "first-frame.png")
                        frames += 1
                        return frame

                    initial = capture()
                    for _ in range(FPS - 1):
                        process.stdin.write(initial.tobytes())
                        frames += 1
                    for _ in range(env.max_steps):
                        if not phases or phases[-1]["phase"] != controller.phase:
                            phases.append(
                                dict(
                                    phase=controller.phase,
                                    label=PHASES[controller.phase],
                                    simulation_seconds=env.steps * env.dt,
                                )
                            )
                            print(phases[-1], flush=True)
                        _, _, ended, truncated, info = env.step(controller.action(env))
                        capture(success=info["success"])
                        if ended or truncated:
                            if not info["success"]:
                                raise RuntimeError(
                                    f"Demonstration did not complete: {info['reason']}"
                                )
                            break
                        controller.advance(env)
                    else:
                        raise RuntimeError("Demonstration exceeded the recording limit")
                    final = capture(success=True)
                    final.save(args.work / "last-frame.png")
                    for _ in range(2 * FPS - 1):
                        process.stdin.write(final.tobytes())
                        frames += 1
                process.stdin.close()
                if process.wait() != 0:
                    raise RuntimeError("FFmpeg could not encode the demonstration")
            except BaseException:
                process.stdin.close()
                process.terminate()
                process.wait()
                raise
            metadata = dict(
                description="Complete scripted lemon collection from an eight-object tabletop",
                origin="simulation",
                controller="ReferenceController; simulator-pose reference",
                seed=args.seed,
                mujoco=mujoco.__version__,
                model_sha256=env.model_hash,
                dimensions=[WIDTH, HEIGHT],
                fps=FPS,
                frames=frames,
                playback_seconds=frames / FPS,
                simulation_seconds=env.steps * env.dt,
                playback_speed=1,
                success=bool(info["success"]),
                phases=phases,
                objects=[item["id"] for item in env.objects],
                asset_credit=CREDIT,
                asset_source=SOURCE,
                asset_license=LICENSE,
                note="One demonstration, not a benchmark or learned-policy performance claim. "
                "Includes one-second initial and two-second final holds. Camera feeds are simultaneous.",
            )
        pending.replace(movie)
    subprocess.run(
        [
            executable,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(movie),
            "-filter_complex",
            "[0:v]fps=8,scale=640:-1:flags=lanczos,split[a][b];[a]palettegen=stats_mode=diff[p];"
            "[b][p]paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle",
            "-loop",
            "0",
            str(preview),
        ],
        check=True,
    )
    metadata["sha256"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (movie, preview)
    }
    (args.output / "panda-grasp-place.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, default=Path("runs/demo/assets"))
    parser.add_argument("--output", type=Path, default=Path("docs/videos"))
    parser.add_argument("--work", type=Path, default=Path("runs/demo/video-render"))
    parser.add_argument("--seed", type=int, default=0)
    render(parser.parse_args())


if __name__ == "__main__":
    main()
