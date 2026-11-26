"""Internal numerical regression fixture; not a supported robot workflow."""

import numpy as np

from robogogogo import spatial as sp
from robogogogo.manipulation import Manipulation
from robogogogo.planning import Capsule, CollisionWorld
from robogogogo.planning.collision import BoxObstacle
from robogogogo.session import Session


def fixture(extra_obstacles=(), *, supports=None, clearance_m=0.015, finger_tip=0.23):
    b = sp.ModelBuilder()
    rotation = np.asarray(sp.rpy_matrix(np.array([np.pi / 2, np.pi / 2, 0.0])))
    for name, origin in [("left", (0.2, -0.09, 0.7)), ("right", (0.2, 0.25, 0.8))]:
        parent = -1
        for i in range(3):
            # Columns of R.T are the local directions of world translations.
            parent = b.add_link(
                f"{name}_{i}",
                parent=parent,
                joint="prismatic",
                axis=rotation.T[:, i],
                origin=origin if i == 0 else (0, 0, 0),
                rpy=(np.pi / 2, np.pi / 2, 0.0) if i == 0 else (0, 0, 0),
                mass=1.0,
                inertia=np.eye(3) * 0.01,
                limits=(-0.5, 0.5),
                effort=150.0,
            )
    s = Session(b.build(), arms=[dict(name=n, tool=f"{n}_2") for n in ("left", "right")])
    if supports is None:
        supports = [[0.2, 0, 0.03, 0.17, 0.48], [0.48, 0, 0.03, 0.17, 0.45]]
    m = Manipulation(
        s.model, s.arms, [0.2, 0, 0.5], supports, material="soft_elastomer", structural=True
    )
    m.enable_adaptive_grasp(initial_force_n=16, damage_force_n=50)
    s.contact_system = m
    capsules = []
    for arm in s.arms:
        capsules.append(
            Capsule(arm.name + ":palm", arm.body, (0.0, -0.06, 0.0), (0.0, 0.06, 0.0), 0.018)
        )
        for side in (-1, 1):
            capsules.append(
                Capsule(
                    arm.name + ":finger:" + str(side),
                    arm.body,
                    (0.02, side * 0.057, 0.0),
                    (finger_tip, side * 0.057, 0.0),
                    0.008,
                )
            )
    pose = s.poses()
    center = rotation.T @ (np.array([0.2, 0, 0.5]) - np.asarray(pose.position)[s.arms[0].body])
    axis = rotation.T @ np.array([0.0, 1.0, 0.0])
    capsules.append(
        Capsule(
            "payload",
            s.arms[0].body,
            tuple(center - 0.12 * axis),
            tuple(center + 0.12 * axis),
            0.03,
        )
    )
    obstacles = [
        BoxObstacle("support_" + str(i), (cx, cy, z / 2), (2 * hx, 2 * hy, z))
        for i, (cx, cy, hx, hy, z) in enumerate(supports)
    ]
    world = CollisionWorld(
        s.model,
        capsules,
        margin=clearance_m,
        obstacles=[*obstacles, *extra_obstacles],
        excluded_capsule_pairs=[
            ("payload", c.name) for c in capsules if c.name.startswith("left:finger:")
        ],
        support_contacts=[("payload", "support_0")],
    )
    m.grasp("left", s)
    return s, m, world


def run(scenario="helper", camera_provider=None, fixture_factory=fixture):
    s, m, world = fixture_factory()
    enabled = False
    lifted = 0.0
    traveled = 0.0
    for i in range(5000):
        if m.owner and not m.motion_hold and not enabled:
            if scenario == "support":
                lifted = min(0.10, lifted + 0.0003)
                s.target_q = s.target_q.at[2].set(lifted)
                if lifted >= 0.10 and i > 650:
                    traveled = min(0.15, traveled + 0.0003)
                    s.target_q = s.target_q.at[0].set(traveled)
                if traveled >= 0.15 and i > 1250:
                    m.enable_recovery()
                    enabled = True
                    nodes = m.contact.structure
                    # Known pre-existing damage tests preservation during transport.
                    m.contact = m.contact._replace(
                        structure=nodes._replace(damage=nodes.damage.at[0].set(0.02))
                    )
            elif i > 250:
                m.enable_recovery()
                enabled = True
                # Request recovery using a sustained measured slip warning, caused below.
                m.configure_gripper("left", mu_static=0.04, mu_dynamic=0.02)
        if enabled and scenario == "helper" and m.recovery.attempts == 0:
            s.target_q = s.target_q.at[1].add(0.0003)
        if enabled and camera_provider is not None:
            m.recovery.planner.vision_required = True
            if m.recovery.planner.observation is None or i % 25 == 0:
                m.recovery.planner.observation = camera_provider(s, m)
        s.advance(collision_world=world)
        if m.failure or (m.recovery and (m.recovery.placed or m.recovery.secured)):
            break
    return dict(time_s=s.elapsed, **m.summary())
