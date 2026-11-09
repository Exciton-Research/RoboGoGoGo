"""Contact-driven manipulation with independent object and jaw dynamics."""

from dataclasses import dataclass, replace
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from . import control
from . import spatial as sp
from .adaptive_grasp import AdaptiveGrip, GripSafety
from .simulation.grasp_contact import ContactState, contact_step
from .simulation.materials import MATERIALS, Material, initial_material_state, material_summary
from .simulation.structural_contact import envelope, structural_contact_step
from .simulation.structure import CohesiveBar


class ObjectState(NamedTuple):
    position: jax.Array
    rotation: jax.Array
    velocity: jax.Array
    angular_velocity: jax.Array


@dataclass(frozen=True)
class GraspObject:
    name: str = "transfer_bar"
    mass: float = 0.25
    radius: float = 0.03
    half_length: float = 0.12

    def __post_init__(self):
        if (
            not np.isfinite([self.mass, self.radius, self.half_length]).all()
            or min(self.mass, self.radius, self.half_length) <= 0
        ):
            raise ValueError("Object mass and dimensions must be positive and finite")

    @property
    def inertia(self):
        # Cylinder with hemispherical ends, longitudinal axis +Y.
        r, length = self.radius, 2 * self.half_length
        cylinder_volume = np.pi * r * r * length
        sphere_volume = 4 * np.pi * r**3 / 3
        mc = self.mass * cylinder_volume / (cylinder_volume + sphere_volume)
        ms = self.mass - mc
        axial = 0.5 * mc * r * r + 0.4 * ms * r * r
        transverse = mc * (3 * r * r + length * length) / 12 + ms * (
            0.4 * r * r + length * length / 4 + 3 * length * r / 8
        )
        return np.diag([transverse, axial, transverse])


@jax.jit
def free_object_step(state, dt, gravity, radius, half_length, supports):
    """Free flight plus inelastic support-plane contact for a capsule.

    Supports are [cx,cy,half_x,half_y,top_z]; contact uses the COM footprint.
    No robot contact impulses, rolling friction, or multi-point tipping model.
    """
    velocity = state.velocity + dt * gravity
    position = state.position + dt * velocity
    speed = jnp.linalg.norm(state.angular_velocity)
    rotation = (
        sp.rotation(state.angular_velocity / jnp.maximum(speed, 1e-12), speed * dt) @ state.rotation
    )
    extent = radius + half_length * jnp.abs(rotation[2, 1])
    inside = (jnp.abs(position[0] - supports[:, 0]) <= supports[:, 2]) & (
        jnp.abs(position[1] - supports[:, 1]) <= supports[:, 3]
    )
    height = jnp.maximum(
        0.0, jnp.max(jnp.where(inside, supports[:, 4], -jnp.inf), initial=-jnp.inf)
    )
    contact = (position[2] - extent <= height) & (velocity[2] <= 0)
    position = position.at[2].set(jnp.where(contact, height + extent, position[2]))
    velocity = jnp.where(contact, velocity * jnp.array([0.8, 0.8, 0.0]), velocity)
    angular = jnp.where(contact, state.angular_velocity * 0.8, state.angular_velocity)
    return ObjectState(position, rotation, velocity, angular)


def payload_model(model, body, mass, offset, inertia):
    """Exactly combine welded payload/link inertias using the parallel-axis theorem."""
    offset = jnp.asarray(offset)
    inertia = jnp.asarray(inertia)
    old_mass = model.mass[body]
    total = old_mass + mass
    center = (old_mass * model.com[body] + mass * offset) / total

    def shift(mass, delta):
        return mass * (jnp.dot(delta, delta) * jnp.eye(3) - jnp.outer(delta, delta))

    combined = (
        model.inertia[body]
        + inertia
        + shift(old_mass, model.com[body] - center)
        + shift(mass, offset - center)
    )
    return replace(
        model,
        mass=model.mass.at[body].set(total),
        com=model.com.at[body].set(center),
        inertia=model.inertia.at[body].set(combined),
    )


class Manipulation:
    """A free rigid object grasped by force-limited, finite-mass parallel jaws.

    ``owner`` is task telemetry, never a constraint or a source of forces.
    Every jaw participates in contact, including during overlapping handover.
    """

    def __init__(
        self,
        model,
        arms,
        position,
        supports,
        *,
        contact_dt=0.0002,
        material="rigid_reference",
        structural=False,
    ):
        self.model, self.arms = model, {a.name: a for a in arms}
        self.names = tuple(self.arms)
        if isinstance(material, str):
            if material != "rigid_reference" and material not in MATERIALS:
                raise ValueError("Unknown object material")
            material = None if material == "rigid_reference" else MATERIALS[material]
        if material is not None and not isinstance(material, Material):
            raise ValueError("Expected a Material or preset name")
        if structural and material is None:
            raise ValueError("Structural dynamics require a deformable material")
        self.structural = None
        if structural:
            contact_dt = min(contact_dt, 0.0001)
        self.material = material
        self.object = GraspObject(mass=material.mass_kg if material else 0.25)
        self.supports = np.asarray(supports, dtype=float).reshape(-1, 5)
        if np.shape(position) != (3,) or not np.isfinite(position).all():
            raise ValueError("Object position must be a finite 3-vector")
        if not np.isfinite(self.supports).all() or (self.supports[:, 2:4] <= 0).any():
            raise ValueError("Invalid support geometry")
        if not self.names or not 0 < contact_dt <= 0.0002:
            raise ValueError("Contacts require arms and a timestep in (0, 0.0002]")
        self.microsteps = int(np.ceil(float(model.dt) / contact_dt))
        self.contact_dt = float(model.dt) / self.microsteps
        dtype = model.home.dtype
        n = len(arms)
        self.contact = ContactState(
            jnp.asarray(position, dtype=dtype),
            jnp.eye(3, dtype=dtype),
            jnp.zeros(3, dtype=dtype),
            jnp.zeros(3, dtype=dtype),
            jnp.full(n, 0.10, dtype=dtype),
            jnp.zeros(n, dtype=dtype),
            jnp.zeros((n, 2, 6, 3), dtype=dtype),
            jnp.zeros((3 * (len(self.supports) + 1), 3), dtype=dtype),
            jnp.full(n, 0.10, dtype=dtype),
        )
        if material is not None:
            self.contact = self.contact._replace(material_state=initial_material_state(dtype))
        if structural:
            self.structural = CohesiveBar(
                material.name,
                grippers=n,
                dt=self.contact_dt,
                pad_half_extent=0.0225,
                pad_half_height=0.009,
            )
            nodes = self.structural.initial(center=position)._replace(
                jaw_width=self.contact.width, drive_width=self.contact.drive_width
            )
            self.contact = self.contact._replace(structure=nodes)
        self.grippers = {
            name: dict(
                width=0.10,
                command="open",
                holding=False,
                force_limit_n=40.0,
                mu_static=0.8,
                mu_dynamic=0.6,
                normal_force_n=[0.0, 0.0],
                friction_force_n=[0.0, 0.0],
                slip_speed_m_s=0.0,
                slipping=False,
                contact=False,
            )
            for name in self.names
        }
        self.adaptive = {}
        self.recovery = None
        self.damage_limit_exceeded = False
        self.hold_target = None
        self.owner = None
        self.events = []
        self.pending = None
        self.transfer_share = 1.0
        self.load_verified_time = 0.0
        self.stable = {name: 0.0 for name in self.names}
        self.last_reaction = jnp.zeros_like(model.home)
        self.report = None
        self.failure = None
        self._tick = jax.jit(self._coupled_tick, static_argnames=("substeps",))

    @property
    def state(self):
        return ObjectState(*self.contact[:4])

    @state.setter
    def state(self, state):
        if self.structural is not None:
            transform = state.rotation @ self.contact.rotation.T
            relative = (self.contact.structure.position - self.contact.position) @ transform.T
            nodes = self.contact.structure._replace(
                position=relative + state.position,
                velocity=state.velocity + jnp.cross(state.angular_velocity, relative),
            )
            self.contact = self.contact._replace(structure=nodes)
        self.contact = self.contact._replace(
            position=state.position,
            rotation=state.rotation,
            velocity=state.velocity,
            angular_velocity=state.angular_velocity,
        )

    def object_segment(self):
        if self.structural is not None:
            return np.asarray(envelope(self.contact.structure))
        axis = np.asarray(self.state.rotation)[:, 1] * self.object.half_length
        return np.array([self.state.position - axis, self.state.position + axis])

    def summary(self):
        return dict(
            object=dict(
                name=self.object.name,
                shape="capsule",
                mass=self.object.mass,
                radius=self.object.radius,
                half_length=self.object.half_length,
                position=np.asarray(self.state.position).tolist(),
                rotation=np.asarray(self.state.rotation).tolist(),
                velocity=np.asarray(self.state.velocity).tolist(),
                angular_velocity=np.asarray(self.state.angular_velocity).tolist(),
                owner=self.owner,
                damage_limit_exceeded=self.damage_limit_exceeded,
                damage_monitoring_enabled=bool(self.adaptive),
                inertia=self.object.inertia.tolist(),
                material=self.material_status(),
                structure=self.structural.summary(self.contact.structure)
                if self.structural
                else None,
            ),
            grippers={name: dict(g) for name, g in self.grippers.items()},
            events=list(self.events),
            grasp_model="structural_frictional_contact"
            if self.structural
            else "compliant_frictional_contact",
            payload_dynamics=True,
            contact_dt=self.contact_dt,
            reaction_torque=np.asarray(self.last_reaction).tolist(),
            maximum_penetration_m=0.0
            if self.report is None
            else float(np.max(self.report.penetration)),
            failure=self.failure,
            adaptive_grasp={name: c.summary() for name, c in self.adaptive.items()},
            motion_hold=self.motion_hold,
            recovery=self.recovery.summary() if self.recovery else None,
            support_load_n=np.asarray(self.report.support_load).tolist()
            if self.report is not None and self.report.support_load is not None
            else [],
        )

    def material_status(self):
        result = material_summary(self.material, self.contact.material_state)
        if self.structural:
            damage = np.asarray(self.contact.structure.damage)
            result.update(
                model="xpbd_cohesive_cells_v1",
                damaged=bool(np.any(damage > 0)),
                fractured=bool(np.any(np.all(damage.reshape(-1, 4) >= 1, axis=1))),
                damage_fraction=float(damage.max()),
                peak_pressure_pa=float(self.contact.structure.peak_pressure.max()),
                risk=0.0 if self.report is None else float(self.report.material_risk),
            )
        return result

    def configure_gripper(self, name, *, force_limit_n=None, mu_static=None, mu_dynamic=None):
        if name not in self.grippers:
            raise ValueError("Unknown gripper")
        config = dict(self.grippers[name])
        for key, value in dict(
            force_limit_n=force_limit_n, mu_static=mu_static, mu_dynamic=mu_dynamic
        ).items():
            if value is not None:
                if not np.isfinite(value) or value < 0:
                    raise ValueError("Contact parameters must be finite and nonnegative")
                config[key] = float(value)
        if config["mu_dynamic"] > config["mu_static"] or config["force_limit_n"] > 80:
            raise ValueError("Require dynamic friction <= static friction and force <= 80 N")
        self.grippers[name] = config

    def enable_adaptive_grasp(self, *, initial_force_n=8.0, damage_force_n=55.0):
        safety = GripSafety(initial_force_n=initial_force_n, damage_force_n=damage_force_n)
        self.adaptive = {name: AdaptiveGrip(safety) for name in self.names}

    def enable_recovery(self, *, camera_based=False, active_cameras=None):
        from .recovery import GraspRecovery

        if not self.adaptive:
            raise ValueError("Recovery requires adaptive force feedback")
        if active_cameras is not None and not camera_based:
            raise ValueError("Active perception requires camera-based recovery")
        self.recovery = GraspRecovery(self.model, tuple(self.arms.values()))
        self.recovery.planner.vision_required = bool(camera_based)
        if active_cameras is not None:
            from .active_perception import ActivePerception

            self.recovery.view = ActivePerception(self.model, active_cameras)

    def observe_recovery(self, cloud, *, seed=None, target_mask=None):
        """Supply calibrated visible surfaces; no object-state segmentation fallback."""
        from .perception.recovery import detect

        if not self.recovery or not self.recovery.planner.vision_required:
            raise ValueError("Enable camera-based recovery before supplying observations")
        observation = detect(cloud, seed=seed, target_mask=target_mask)
        self.recovery.planner.observation = observation
        return observation

    @property
    def motion_hold(self):
        if self.recovery and (
            self.recovery.active or self.recovery.placed or self.recovery.secured
        ):
            return True
        return any(c.motion_hold for c in self.adaptive.values())

    def _command(self, name, command, session):
        if name not in self.grippers:
            raise ValueError("Unknown gripper")
        if self.grippers[name]["command"] != command:
            # Start the new servo trajectory at the measured opening. Otherwise a
            # stale closing drive can squeeze again when an open command arrives.
            index = self.names.index(name)
            drive = self.contact.drive_width.at[index].set(self.contact.width[index])
            self.contact = self.contact._replace(drive_width=drive)
            if self.contact.structure is not None:
                self.contact = self.contact._replace(
                    structure=self.contact.structure._replace(drive_width=drive)
                )
        self.grippers[name]["command"] = command
        self.events.append(dict(event=f"{command}_command", arm=name, time=session.elapsed))

    def grasp(self, name, session):
        if name not in self.grippers:
            raise ValueError("Unknown gripper")
        if self.owner is not None or self.pending:
            raise ValueError("Grasp operation already active")
        self.pending = ("grasp", name, None, session.elapsed)
        self._command(name, "closed", session)

    def transfer(self, receiver, session):
        if receiver not in self.grippers:
            raise ValueError("Unknown gripper")
        if self.owner is None or self.owner == receiver or self.pending:
            raise ValueError("Transfer requires a held object and a different receiver")
        self.pending = ("transfer", receiver, self.owner, session.elapsed)
        self._command(receiver, "closed", session)

    def release(self, name, session):
        if self.owner != name or self.pending:
            raise ValueError("Only the current holder may start a release")
        self.pending = ("release", name, None, session.elapsed)
        self._command(name, "open", session)

    def _parameters(self):
        values = [self.grippers[n] for n in self.names]
        forces = []
        for name, gripper in zip(self.names, values):
            limit = gripper["force_limit_n"]
            controller = self.adaptive.get(name)
            if controller:
                requested = (
                    controller.force
                    if gripper["command"] == "closed"
                    else (controller.safety.damage_force_n * controller.safety.force_reserve)
                )
                limit = min(limit, requested)
            if (
                self.structural
                and self.pending
                and self.pending[0] == "load_transfer"
                and name == self.pending[2]
            ):
                limit *= self.transfer_share
            if self.recovery and self.recovery.active:
                limit *= self.recovery.force_scale(name)
            forces.append(limit)
        return tuple(
            jnp.asarray(v, dtype=self.model.home.dtype)
            for v in (
                [0.015 if g["command"] == "closed" else 0.10 for g in values],
                forces,
                [g["mu_static"] for g in values],
                [g["mu_dynamic"] for g in values],
                [
                    self.adaptive[n].safety.damage_force_n * 0.9 if n in self.adaptive else 1e12
                    for n in self.names
                ],
            )
        )

    def predict_tick(self, session):
        return self._tick(
            session.state,
            self.contact,
            session.target_q,
            self.last_reaction,
            *self._parameters(),
            protection=jnp.asarray(bool(self.adaptive)),
            substeps=session.substeps,
        )

    def _coupled_tick(
        self,
        robot,
        contact,
        goal,
        previous_reaction,
        targets,
        limits,
        mus,
        mud,
        ceilings,
        *,
        protection=False,
        substeps=5,
    ):
        model = self.model
        bodies = jnp.array([a.body for a in self.arms.values()])

        # Geometry/Jacobians and joint inertia are refreshed each native physics step.
        # Contact reactions are subcycled; no payload mass is merged into any link.
        def outer(carry, _):
            robot, contact, previous_reaction = carry

            def tools(q):
                p = sp.forward_kinematics(model, q)
                return p.position[bodies], p.rotation[bodies]

            position, orientation = tools(robot.q)
            jv, dr = jax.jacfwd(tools)(robot.q)
            skew = jnp.einsum("aijn,akj->aikn", dr, orientation)
            jw = jnp.stack((skew[:, 2, 1], skew[:, 0, 2], skew[:, 1, 0]), axis=1)
            h = self.contact_dt
            predicted = robot.q + h * robot.qd
            low, high = predicted < model.lower, predicted > model.upper
            damping = model.damping + model.limit_damping * (
                (low & (robot.qd < 0)) | (high & (robot.qd > 0))
            )
            stiffness = model.limit_stiffness * (low | high)
            inverse_mass = jnp.linalg.inv(
                sp.mass_matrix(model, robot.q) + jnp.diag(h * damping + h * h * stiffness)
            )
            # Measured contact load feedforward; actuator limits still apply in acceleration.
            torque = control.computed_torque(model, robot, goal) - previous_reaction
            free_acceleration = sp.acceleration(model, robot, torque, implicit_dt=self.contact_dt)
            qd = robot.qd

            def micro(carry, _):
                dq, qd, contact = carry
                turn = jnp.einsum("aij,j->ai", jw, dq)
                angle = jnp.linalg.norm(turn, axis=1)
                rot = (
                    jax.vmap(sp.rotation)(turn / jnp.maximum(angle[:, None], 1e-12), angle)
                    @ orientation
                )
                pos = position + jnp.einsum("aij,j->ai", jv, dq)
                vel, omega = (jnp.einsum("aij,j->ai", j, qd) for j in (jv, jw))
                if self.structural is not None:
                    contact, report = structural_contact_step(
                        self.structural,
                        contact,
                        pos,
                        rot,
                        vel,
                        omega,
                        targets,
                        limits,
                        mus,
                        mud,
                        model.gravity,
                        jnp.asarray(self.supports),
                        ceilings,
                        protection,
                    )
                else:
                    contact, report = contact_step(
                        contact,
                        pos,
                        rot,
                        vel,
                        omega,
                        targets,
                        limits,
                        mus,
                        mud,
                        self.object.mass,
                        jnp.asarray(self.object.inertia),
                        self.object.radius,
                        self.object.half_length,
                        model.gravity,
                        jnp.asarray(self.supports),
                        self.contact_dt,
                        force_ceiling=ceilings,
                        material=self.material,
                        protect_material=protection,
                    )
                reaction = jnp.einsum("aij,ai->j", jv, report.force) + jnp.einsum(
                    "aij,ai->j", jw, report.moment
                )
                qd = qd + self.contact_dt * (free_acceleration + inverse_mass @ reaction)
                return (dq + self.contact_dt * qd, qd, contact), (report, reaction)

            (dq, qd, contact), (reports, reactions) = jax.lax.scan(
                micro, (jnp.zeros_like(robot.q), qd, contact), None, length=self.microsteps
            )
            robot = sp.State(robot.q + dq, qd)
            report = jax.tree.map(lambda a: a[-1], reports)
            report = report._replace(
                peak_normal=jnp.max(reports.normal, axis=0),
                material_risk=jnp.max(reports.material_risk),
            )
            reaction = jnp.mean(reactions, axis=0)
            axis = contact.rotation[:, 1] * self.object.half_length
            segment = (
                envelope(contact.structure)
                if self.structural is not None
                else jnp.stack((contact.position - axis, contact.position + axis))
            )
            return (robot, contact, reaction), (
                robot.q,
                segment,
                report,
                jnp.clip(torque, -model.effort, model.effort),
            )

        following, traces = jax.lax.scan(
            outer, (robot, contact, previous_reaction), None, length=substeps
        )
        return following, traces

    def commit_tick(self, following, traces, session):
        _, self.contact, self.last_reaction = following
        self.report = jax.tree.map(lambda a: a[-1], traces[2])
        dt = float(self.model.dt) * session.substeps
        for i, name in enumerate(self.names):
            g = self.grippers[name]
            normal = np.asarray(self.report.normal[i])
            speed = float(np.max(self.report.slip_speed[i]))
            sliding = bool(np.any(self.report.sliding[i])) and speed > 0.01
            bilateral = bool(np.min(normal) > 0.3)
            stable = (
                bilateral and speed < 0.008 and abs(float(self.contact.width_velocity[i])) < 0.02
            )
            self.stable[name] = self.stable[name] + dt if stable else 0.0
            holding = bilateral and g["command"] == "closed"
            if sliding and not g["slipping"]:
                self.events.append(dict(event="slip", arm=name, time=session.elapsed))
            if g["holding"] and not holding and g["command"] == "closed":
                self.events.append(dict(event="contact_lost", arm=name, time=session.elapsed))
            g.update(
                width=float(self.contact.width[i]),
                normal_force_n=normal.tolist(),
                friction_force_n=np.asarray(self.report.friction[i]).tolist(),
                slip_speed_m_s=speed,
                slipping=sliding,
                contact=bilateral,
                holding=holding,
            )
        material = self.material_status()
        if self.recovery:
            self.recovery.observe(self, session, dt, material)
        recovering = self.recovery is not None and self.recovery.active
        if material["damaged"]:
            event = "material_fracture" if material["fractured"] else "material_damage"
            if not any(e["event"] == event for e in self.events):
                self.events.append(
                    dict(
                        event=event,
                        time=session.elapsed,
                        permanent_m=material["maximum_permanent_m"],
                        damage=material["damage_fraction"],
                    )
                )
            if material["fractured"] or not recovering:
                self.failure = (
                    "Object fractured" if material["fractured"] else "Object material damaged"
                )
        for i, name in enumerate(self.names):
            if name not in self.adaptive:
                continue
            controller = self.adaptive[name]
            if recovering:
                peak = float(np.max(traces[2].peak_normal[:, i]))
                controller.peak_force = max(controller.peak_force, peak)
                if peak > controller.safety.damage_force_n or controller.fault:
                    self.damage_limit_exceeded |= peak > controller.safety.damage_force_n
                    self.recovery.block(
                        self, session, controller.fault or "Hard force limit exceeded"
                    )
                    recovering = False
                    break
                self.recovery.feedback(self, name, dt, float(np.max(traces[2].material_risk)))
                continue
            if (
                self.structural
                and self.pending
                and self.pending[0] == "load_transfer"
                and name == self.pending[2]
                and not material["damaged"]
                and float(np.max(traces[2].material_risk)) < 0.55
                and float(np.max(traces[2].peak_normal[:, i])) < controller.safety.damage_force_n
            ):
                # Intentional donor unloading must not fight the load-transfer ramp.
                controller.status = "transferring"
                continue
            previous = controller.status
            g = self.grippers[name]
            controller.update(
                closed=g["command"] == "closed",
                normals=g["normal_force_n"],
                slip_speed=g["slip_speed_m_s"],
                peak_force=float(np.max(traces[2].peak_normal[:, i])),
                hardware_limit=g["force_limit_n"],
                dt=dt,
                material_risk=float(np.max(traces[2].material_risk)),
                material_damaged=material["damaged"],
            )
            if controller.status != previous:
                self.events.append(
                    dict(
                        event="adaptive_" + controller.status,
                        arm=name,
                        time=session.elapsed,
                        force_setpoint_n=controller.force,
                    )
                )
            if controller.peak_force > controller.safety.damage_force_n:
                self.damage_limit_exceeded = True
            if controller.fault and self.failure is None:
                self.failure = controller.fault
        if recovering:
            self.recovery.after_tick(self, session, dt, material)
            return
        if self.recovery and (self.recovery.placed or self.recovery.secured):
            session.target_q = self.recovery.target
            return
        if self.motion_hold:
            # Hold immediately between feedback ticks, not after the full action chunk.
            if self.hold_target is None:
                self.hold_target = session.state.q
            session.target_q = self.hold_target
        else:
            self.hold_target = None
        if self.failure:
            return
        if self.pending:
            operation, name, giver, started = self.pending
            if session.elapsed - started > (8.0 if self.structural else 3.0):
                self.failure = f"{operation} failed: no sustained stable contact for {name}"
                return
            secure = (
                not self.motion_hold
                and self.stable[name] >= 0.12
                and min(self.grippers[name]["normal_force_n"])
                >= min(
                    2.0,
                    min(
                        self.grippers[name]["force_limit_n"],
                        self.adaptive[name].force
                        if name in self.adaptive
                        else self.grippers[name]["force_limit_n"],
                    )
                    * 0.5,
                )
            )
            if operation == "grasp" and secure:
                self.owner = name
                self.pending = None
                self.events.append(dict(event="grasp", arm=name, time=session.elapsed))
            elif operation == "transfer" and secure and self.structural:
                self.transfer_share = 1.0
                self.load_verified_time = 0.0
                self.pending = ("load_transfer", name, giver, started)
                self.events.append(
                    dict(event="load_transfer_started", arm=name, time=session.elapsed)
                )
            elif operation == "load_transfer":
                if self.motion_hold or not self.grippers[name]["holding"]:
                    self.transfer_share = min(1.0, self.transfer_share + 4 * dt)
                    self.stable[name] = 0.0
                else:
                    self.transfer_share = max(0.0, self.transfer_share - dt)
                unloaded = (
                    self.transfer_share == 0
                    and secure
                    and max(self.grippers[giver]["normal_force_n"]) < 0.2
                )
                self.load_verified_time = self.load_verified_time + dt if unloaded else 0.0
                if self.load_verified_time >= 0.25:
                    self.events.append(
                        dict(event="receiver_load_verified", arm=name, time=session.elapsed)
                    )
                    self._command(giver, "open", session)
                    self.pending = ("donor_release", name, giver, started)
            elif operation == "transfer" and secure:
                self.events.append(dict(event="receiver_secured", arm=name, time=session.elapsed))
                self._command(giver, "open", session)
                self.pending = ("donor_release", name, giver, started)
            elif (
                operation == "donor_release"
                and self.grippers[giver]["width"] > 0.09
                and max(self.grippers[giver]["normal_force_n"]) < 0.2
            ):
                if self.motion_hold:
                    return
                if not self.grippers[name]["holding"]:
                    self.failure = "Receiver lost contact during donor release"
                    return
                self.owner = name
                self.pending = None
                self.events.append(
                    dict(event="handover", giver=giver, receiver=name, time=session.elapsed)
                )
            elif (
                operation == "release"
                and self.grippers[name]["width"] > 0.09
                and max(self.grippers[name]["normal_force_n"]) < 0.2
            ):
                self.owner = None
                self.pending = None
                self.events.append(dict(event="release", arm=name, time=session.elapsed))
        if self.owner and not self.pending and not self.grippers[self.owner]["holding"]:
            self.failure = f"Grasp lost: {self.owner}"
            self.owner = None
