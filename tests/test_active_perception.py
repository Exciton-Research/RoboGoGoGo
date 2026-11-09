import importlib.util
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from robogogogo.active_perception import ActivePerception
from robogogogo.perception import RecoveryObservation
from robogogogo.planning import SphereObstacle
from robogogogo.planning.motion import PlanningError
from robogogogo.studio.schemas import PlanTask

for name in ("camera_recovery", "recovery_search", "active_perception_demo"):
    filename = "active_perception" if name == "active_perception_demo" else name
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).parents[1] / "examples" / (filename + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
DEMO = sys.modules["active_perception_demo"]
RIG = sys.modules["recovery_search"]


def setup(extra=()):
    session, m, world = RIG.fixture(extra)
    m.grippers["left"]["holding"] = True
    m.enable_recovery(camera_based=True, active_cameras=[DEMO.wrist_camera(session)])
    m.recovery.world = world
    return session, m, world


def test_view_plan_freezes_retainer_and_checks_swept_path():
    session, m, world = setup()
    view = m.recovery.view
    view.begin(m, session, world, "searching_helper")
    np.testing.assert_array_equal(view.goal[:3], session.state.q[:3])
    assert not np.array_equal(view.goal[3:], session.state.q[3:])
    assert world.edge_free(
        session.state.q, view.goal, object_start=m.object_segment(), object_end=m.object_segment()
    )


def test_blocked_camera_and_closed_helper_cannot_move():
    session, m, world = setup([SphereObstacle("obstruction", (0.2, 0.25, 0.8), 0.1)])
    with pytest.raises(PlanningError, match="No reachable camera"):
        m.recovery.view.begin(m, session, world, "searching_helper")
    session, m, world = setup()
    m.grippers["right"]["command"] = "closed"
    with pytest.raises(PlanningError, match="retaining grasp"):
        m.recovery.view.begin(m, session, world, "searching_helper")


def test_settled_capture_rejects_stale_images_then_resumes_from_fresh_evidence():
    session, m, world = setup()
    recovery = m.recovery
    recovery.phase = "searching_helper"
    old = RecoveryObservation(0.0, "test", 0.002, (), (), None)
    recovery.planner.observation = old
    assert recovery.seek_view(m, session, "searching_helper")
    session.state = session.state._replace(q=jnp.asarray(recovery.view.goal))
    recovery.view.command = recovery.view.goal.copy()
    recovery.view.dwell = 0.2
    session.elapsed = 1.0
    recovery.prepare(m, session, world)
    assert recovery.phase == "view_capture"
    recovery.prepare(m, session, world)
    assert recovery.phase == "view_capture"
    recovery.planner.observation = RecoveryObservation(
        1.0, "test", 0.002, ({"position": [0.2, 0.09, 0.5]},), (), np.array([0, 1, 0])
    )
    recovery.prepare(m, session, world)
    assert recovery.phase == "searching_helper"


def test_capture_timeout_and_view_budget_do_not_authorize_release():
    session, m, world = setup()
    r = m.recovery
    r.phase = "searching_helper"
    r.seek_view(m, session, "searching_helper")
    r.phase, r.changed = "view_capture", 0.0
    r.view.capture_after = 0.0
    session.elapsed = 2.1
    r.prepare(m, session, world)
    assert r.phase == "blocked"
    assert m.grippers["left"]["command"] == "closed"
    r.view.attempts = r.view.max_views
    with pytest.raises(PlanningError, match="view budget"):
        r.view.begin(m, session, world, "searching_helper")


def test_active_perception_requires_camera_recovery():
    with pytest.raises(ValueError, match="camera-based"):
        PlanTask(active_perception=True)


def test_wrist_motion_reveals_missing_grasp_flank_and_completes_recovery():
    result = DEMO.run()
    assert result["initial_grasps"] == 0
    assert result["failure"] is None
    assert result["recovery"]["phase"] == "secured"
    assert result["recovery"]["active_perception"]["attempts"] > 0
    events = {e["event"] for e in result["events"]}
    assert {"recovery_view_moving", "recovery_view_capture", "recovery_helper_closing"} <= events


def test_motorized_head_moves_with_native_dynamics_and_preserves_arm_joints():
    from robogogogo import models
    from robogogogo.manipulation import Manipulation
    from robogogogo.planning import Capsule, CollisionWorld
    from robogogogo.session import Session
    from robogogogo.vla.cameras import mounted_rig

    model, _ = models.with_pan_tilt_head(models.arm6(), [])
    session = Session(model, arms=[dict(name="arm", tool="tool")])
    arm = session.arms[0]
    poses = session.poses()
    cue = np.asarray(poses.position[arm.body]) + 0.2 * np.asarray(poses.rotation[arm.body])[:, 0]
    m = Manipulation(model, session.arms, cue, [[0.5, 0, 0.2, 0.2, 0.2]])
    m.enable_adaptive_grasp()
    m.grippers["arm"].update(holding=True, command="closed")
    m.enable_recovery(
        camera_based=True, active_cameras=mounted_rig(model, session.arms, wrists=False)
    )
    head = model.names.index("head")
    world = CollisionWorld(
        model,
        [
            Capsule("head", head, (0, 0, 0), (0.08, 0, 0), 0.05),
            Capsule("payload", arm.body, (0.2, -0.12, 0), (0.2, 0.12, 0), 0.03),
        ],
    )
    view = m.recovery.view
    before = np.asarray(session.state.q).copy()
    view.begin(m, session, world, "searching_helper")
    assert view.selected_camera.startswith("head_")
    np.testing.assert_array_equal(view.goal[:6], before[:6])
    for _ in range(500):
        target, settled = view.advance(session, float(model.dt) * session.substeps)
        session.target_q = jnp.asarray(target)
        session.advance(collision_world=world)
        if settled:
            break
    assert settled
    np.testing.assert_allclose(session.state.q[:6], before[:6], atol=1e-6)
    assert np.linalg.norm(np.asarray(session.state.q)[-2:] - before[-2:]) > 0.1


def test_motorized_studio_rig_keeps_two_arm_groups(tmp_path):
    from robogogogo.studio.workspace import Studio

    workspace = Studio(tmp_path, scene="dual_arm_head")
    assert len(workspace.session.arms) == 2
    assert workspace.session.model.home.shape == (14,)
    assert all(max(arm.joint_indices) < 12 for arm in workspace.session.arms)
    assert len(workspace.episode.cameras) == 4


def test_camera_motion_reveals_support_behind_a_real_rendered_occluder():
    from robogogogo.perception import detect
    from robogogogo.planning.collision import BoxObstacle
    from robogogogo.vla.cameras import world_camera_pose

    box = BoxObstacle("occluding_panel", (0.19, 0.15, 0.7), (0.14, 0.02, 0.2))
    session, m, world = setup([box])
    # Match this isolated sensor scene rather than the contact benchmark's tables.
    world.obstacles = [box, BoxObstacle("observed_table", (0.32, 0, 0.375), (0.2, 0.2, 0.05))]
    camera = DEMO.wrist_camera(session)
    boxes = [
        ([0.32, 0, 0.375], [0.2, 0.2, 0.05], [80, 110, 80]),
        (box.center, box.size, [90, 90, 90]),
    ]

    def capture():
        pose = session.poses()
        optical = world_camera_pose(
            camera, dict(positions=np.asarray(pose.position), rotations=np.asarray(pose.rotation))
        )
        position = np.asarray(optical["position"])
        target = position + np.asarray(optical["rotation"])[:, 2]
        cloud = sys.modules["camera_recovery"].render_camera(
            position, target, boxes, timestamp=session.elapsed
        )
        return detect(cloud, target_mask=np.zeros(len(cloud.points), bool))

    assert not capture().supports
    view = m.recovery.view
    view.begin(m, session, world, "searching_support")
    # This test isolates native camera motion with a static retained payload;
    # the end-to-end grasp test separately exercises coupled contact dynamics.
    session.contact_system = None
    for _ in range(600):
        target, settled = view.advance(session, float(session.model.dt) * session.substeps)
        session.target_q = jnp.asarray(target)
        session.advance(collision_world=world)
        if settled:
            break
    assert settled
    observation = capture()
    assert observation.supports
    assert all(abs(p["position"][2] - 0.4) < 0.003 for p in observation.supports)
    recovery = m.recovery
    recovery.phase = "view_capture"
    recovery.view.capture_after = session.elapsed
    recovery.target = session.state.q
    recovery.planner.observation = observation
    recovery.prepare(m, session, world)
    assert recovery.phase == "searching_support"


def test_invalid_camera_configuration_preserves_existing_controller():
    session, m, _ = setup()
    previous = m.recovery
    with pytest.raises(ValueError, match="calibrated cameras"):
        m.enable_recovery(camera_based=True, active_cameras=[])
    assert m.recovery is previous
    camera = DEMO.wrist_camera(session)
    with pytest.raises(ValueError, match="unique"):
        ActivePerception(session.model, [camera, camera])
    with pytest.raises(ValueError, match="mount"):
        ActivePerception(session.model, [dict(camera, mount="missing_head")])
