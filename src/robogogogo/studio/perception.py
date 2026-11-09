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


def update_recovery_observation(workspace, observation_index):
    """Camera-mode task input: RGB stereo only; renderer ground-truth depth is excluded."""
    from ..perception.recovery import RecoveryObservation, backproject, detect

    m, session, episode = workspace.task_service.manipulation, workspace.session, workspace.episode
    planner = m.recovery.planner
    observation = episode.observations[observation_index]
    configs = {c["name"]: c for c in episode.cameras}
    timestamp = observation["time"]
    try:
        names = ("head_left", "head_right")
        if any(name not in configs for name in names):
            raise ValueError("Camera recovery requires calibrated head_left/head_right RGB frames")
        cv = opencv()
        images = []
        for name in names:
            decoded = cv.imdecode(
                np.frombuffer(episode.images[observation_index][name], np.uint8), cv.IMREAD_COLOR
            )
            if decoded is None:
                raise ValueError("Invalid captured RGB frame")
            images.append(cv.cvtColor(decoded, cv.COLOR_BGR2RGB))
        poses = observation["camera_poses"]
        result = estimate_stereo(
            *images,
            *(configs[n] for n in names),
            *(poses[n] for n in names),
            num_disparities=min(64, (images[0].shape[1] // 32) * 16),
        )
        # One-pixel disparity uncertainty; do not advertise distant, uncertain depth
        # as millimeter geometry. Calibration/systematic errors require separate validation.
        uncertainty = result.depth**2 / (result.focal_px * result.baseline_m)
        valid = np.isfinite(result.depth) & (uncertainty <= 0.02)
        depth = np.where(valid, result.depth, np.nan)
        sigma = max(0.003, float(uncertainty[valid].max())) if valid.any() else 0.02
        cloud = backproject(
            depth,
            result.rectified_left,
            result.intrinsics,
            result.position,
            result.rotation,
            timestamp=timestamp,
            uncertainty_m=sigma,
            source="calibrated_rgb_stereo",
        )
        holders = [name for name, grip in m.grippers.items() if grip["holding"]]
        seed = None
        if holders:
            # Proprioception supplies only an object-selection cue near the fingers.
            # No object pose, structural nodes, material color or support registry is read.
            pose = session.poses()
            body = m.arms[holders[0]].body
            seed = np.asarray(pose.position[body]) + 0.2 * np.asarray(pose.rotation[body])[:, 0]
        planner.observation = detect(cloud, seed=seed)
    except ValueError as error:
        planner.observation = RecoveryObservation(
            timestamp, "calibrated_rgb_stereo", 0.02, (), (), None, dict(error=str(error))
        )
    # Store the exact camera-derived decision inputs alongside the captured observation.
    observation["recovery_perception"] = dict(
        **planner.observation.summary(),
        grasps=list(planner.observation.grasps),
        supports=list(planner.observation.supports),
    )
