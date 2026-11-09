"""Stereo evaluation of one captured, synchronized observation."""

import base64

import numpy as np

from ..perception.stereo import estimate_stereo, evaluate_depth, opencv


def analyze_stereo(workspace, request):
    cv = opencv()
    with workspace.lock:
        episode = workspace.episode
        if not episode or not 0 <= request.observation < len(episode.observations):
            raise ValueError("Capture an aligned observation first")
        configs = {c["name"]: c for c in episode.cameras}
        if request.left == request.right or any(
            n not in configs for n in (request.left, request.right)
        ):
            raise ValueError("Choose two different configured cameras")
        index = request.observation
        frames = episode.images[index]
        poses = episode.observations[index]["camera_poses"]
        depths = episode.depths[index]
        episode_id, timestamp = episode.id, episode.observations[index]["time"]
        images = []
        for name in (request.left, request.right):
            decoded = cv.imdecode(np.frombuffer(frames[name], np.uint8), cv.IMREAD_COLOR)
            if decoded is None:
                raise ValueError("Captured PNG could not be decoded")
            images.append(cv.cvtColor(decoded, cv.COLOR_BGR2RGB))
        left, right = images
    disparities = min(64, ((left.shape[1] // 2) // 16) * 16)
    result = estimate_stereo(
        left,
        right,
        configs[request.left],
        configs[request.right],
        poses[request.left],
        poses[request.right],
        ground_truth=depths.get(request.left),
        num_disparities=disparities,
    )
    truth = result.ground_truth
    finite = result.depth[np.isfinite(result.depth)]
    scale = max(0.1, float(np.percentile(finite, 95))) if finite.size else 5.0
    if truth is not None and np.isfinite(truth).any():
        scale = max(0.1, float(np.nanpercentile(truth, 95)))

    def preview(values, maximum):
        valid = np.isfinite(values) & (values > 0)
        normalized = np.uint8(np.clip(np.nan_to_num(values, nan=0) / maximum, 0, 1) * 255)
        color = cv.applyColorMap(normalized, cv.COLORMAP_TURBO)
        color[~valid] = [20, 20, 20]
        ok, png = cv.imencode(".png", color)
        if not ok:
            raise ValueError("Could not encode depth preview")
        return "data:image/png;base64," + base64.b64encode(png).decode()

    return dict(
        episode_id=episode_id,
        observation=index,
        time=timestamp,
        left=request.left,
        right=request.right,
        baseline_m=result.baseline_m,
        focal_px=result.focal_px,
        valid_fraction=float(result.valid.mean()),
        depth_scale_m=scale,
        estimated=preview(result.depth, scale),
        ground_truth=preview(truth, scale) if truth is not None else None,
        disparity=preview(result.disparity, disparities),
        metrics=evaluate_depth(result.depth, truth) if truth is not None else None,
        convention="Rectified left-camera optical Z, meters; dark pixels are invalid",
        method="StereoSGBM + left/right consistency + texture mask",
    )
