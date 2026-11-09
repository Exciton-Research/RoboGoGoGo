"""Bounded tactile recovery, with physical support verification and no pose welds.

Recovery samples helper grasps and nearby supports, preserving held-tool poses
and checking every planned and executed edge against the collision world.
"""

import jax
import jax.numpy as jnp
import numpy as np

from .control import inverse_kinematics_poses
from .planning.motion import PlanningError
from .recovery_planning import RecoveryPlanner, contact_world


class GraspRecovery:
    ACTIVE = {
        "searching_helper",
        "approaching_helper",
        "helper_closing",
        "searching_support",
        "transporting",
        "stabilizing",
        "unloading",
        "opening",
        "reclosing",
        "verifying",
        "lowering",
        "seating",
        "releasing",
        "settling",
    }

    def __init__(self, model, arms):
        self.planner = RecoveryPlanner(model, {a.name: a for a in arms})
        self.world = None
        self.route = None
        self.route_index = 1
        self.route_alpha = 0.0
        self.new_helper = False
        self.extended = False
        self.support_searches = 0
        self.initial_damage = 0.0
        self.risk_hold = False
        self.phase = "idle"
        self.reason = None
        self.primary = self.helper = None
        self.support = None
        self.attempts = 0
        self.warning_time = self.dwell = 0.0
        self.verified_support_s = 0.0
        self.started = self.changed = self.cooldown_until = 0.0
        self.target = None
        self.scale = 1.0
        self.records = []
        self.saved_pending = None
        self.saved_goal = None
        self.positions = self.rotations = self.moving = None
        bodies = tuple(a.body for a in arms)
        self.solve = jax.jit(
            lambda q, p, r: inverse_kinematics_poses(model, p, r, bodies, q, iterations=20)
        )

    @property
    def active(self):
        return self.phase in self.ACTIVE

    @property
    def placed(self):
        return self.phase == "placed"

    @property
    def secured(self):
        return self.phase == "secured"

    def summary(self):
        return dict(
            risk_hold=self.risk_hold,
            search_checks=self.planner.checked,
            rejected_candidates=list(self.planner.rejections[-12:]),
            route_waypoints=len(self.route.joints) if self.route else 0,
            route_progress=self.route_index if self.route else 0,
            selected_grasp=self.route.grasp_position if self.route else None,
            phase=self.phase,
            active=self.active,
            reason=self.reason,
            primary=self.primary,
            helper=self.helper,
            support_index=self.support,
            attempts=self.attempts,
            load_fraction=self.scale,
            support_verified_s=self.verified_support_s if self.placed else self.dwell,
        )

    def transition(self, m, session, phase):
        self.phase, self.changed, self.dwell = phase, session.elapsed, 0.0
        m.events.append(
            dict(
                event="recovery_" + phase,
                time=session.elapsed,
                reason=self.reason,
                primary=self.primary,
                helper=self.helper,
                support_index=self.support,
            )
        )

    def block(self, m, session, reason):
        self.reason = reason
        self.transition(m, session, "blocked")
        m.failure = "Recovery blocked: " + reason
        session.target_q = session.state.q

    def observe(self, m, session, dt, material):
        if (
            self.active
            or self.placed
            or self.secured
            or self.phase == "blocked"
            or session.elapsed < self.cooldown_until
        ):
            return
        if material["fractured"] or not m.adaptive:
            return
        warnings = [
            n
            for n, c in m.adaptive.items()
            if m.grippers[n]["command"] == "closed"
            and (
                c.status in ("recovering", "force_limited", "unloading")
                or m.grippers[n]["slipping"]
            )
        ]
        at_risk = material.get("risk", 0.0) > 0.55 or any(
            c.status == "unloading" for c in m.adaptive.values()
        )
        self.warning_time = self.warning_time + dt if warnings or at_risk else 0.0
        if not material["damaged"] and self.warning_time < 0.12:
            return
        if not m.owner and not any(g["holding"] for g in m.grippers.values()):
            return
        if self.attempts >= 2:
            self.block(m, session, "Recovery attempt limit reached")
            return
        self.attempts += 1
        self.reason = (
            "material_damage"
            if material["damaged"]
            else ("material_risk" if at_risk else "slipping")
        )
        self.primary = warnings[0] if warnings else m.owner
        self.primary = self.primary or next(n for n, g in m.grippers.items() if g["holding"])
        self.started = session.elapsed
        self.initial_damage = material.get("damage_fraction", 0.0)
        self.support_searches = 0
        self.risk_hold = False
        self.saved_pending, self.saved_goal = m.pending, session.target_q
        m.pending = None
        self.target = session.state.q
        m.hold_target = self.target
        self.scale = 1.0
        helpers = [
            n
            for n, g in m.grippers.items()
            if n != self.primary
            and g["holding"]
            and m.stable[n] >= 0.15
            and min(g["normal_force_n"]) * 2 * g["mu_dynamic"]
            >= 1.5 * m.object.mass * np.linalg.norm(m.model.gravity)
        ]
        self.helper = helpers[0] if helpers and self.reason == "slipping" else None
        if self.helper:
            self.transition(m, session, "stabilizing")
        elif self.reason == "slipping" and self.world is not None:
            self.extended = True
            self.transition(m, session, "searching_helper")
        else:
            self.begin_lowering(m, session)

    def begin_lowering(self, m, session):
        center = np.asarray(m.state.position)
        candidates = [
            i
            for i, s in enumerate(m.supports)
            if np.all(np.abs(center[:2] - s[:2]) <= s[2:4] - 0.005) and s[4] <= center[2] + 0.003
        ]
        if not candidates:
            if self.world is not None:
                self.extended = True
                self.transition(m, session, "searching_support")
            else:
                self.block(m, session, "No registered support directly below the object")
            return
        self.support = max(candidates, key=lambda i: m.supports[i, 4])
        pose = session.poses()
        bodies = [a.body for a in m.arms.values()]
        self.positions = np.asarray(pose.position)[bodies].copy()
        self.rotations = np.asarray(pose.rotation)[bodies].copy()
        self.moving = np.array([g["holding"] for g in m.grippers.values()])
        if not self.moving.any():
            self.block(m, session, "No gripper retains the object")
            return
        self.target = session.state.q
        self.scale = 1.0
        self.transition(m, session, "lowering")

    def guarded_world(self, world, m):
        world = (
            world if world is not None else (self.world if self.active or self.secured else None)
        )
        if not (self.active or self.placed or self.secured):
            return world
        helper = None
        if self.helper and (
            m.grippers[self.helper]["holding"]
            or self.phase
            in (
                "helper_closing",
                "stabilizing",
                "unloading",
                "opening",
                "reclosing",
                "verifying",
                "secured",
            )
        ):
            helper = self.helper
        if (
            self.phase == "approaching_helper"
            and self.route.contact_start is not None
            and self.route_index >= self.route.contact_start
        ):
            helper = self.helper
        support = (
            self.support
            if self.phase in ("lowering", "seating", "releasing", "settling", "placed")
            else None
        )
        result = contact_world(world, m, helper, support)
        for name, grip in m.grippers.items():
            if grip["holding"]:
                result = contact_world(result, m, name)
        return result

    def prepare(self, m, session, world=None):
        if world is not None:
            self.world = world
        if not self.active:
            return
        if self.phase in ("searching_helper", "searching_support"):
            try:
                if self.phase == "searching_helper":
                    self.route = self.planner.helper_path(m, session, self.world, self.primary)
                    self.helper = self.route.helper
                    self.new_helper = True
                    phase = "approaching_helper"
                else:
                    self.support_searches += 1
                    if self.support_searches > 2:
                        raise PlanningError("Support replanning limit reached")
                    self.route = self.planner.support_path(m, session, self.world)
                    self.support = self.route.support
                    phase = "transporting"
                self.route_index, self.route_alpha = 1, 0.0
                self.transition(m, session, phase)
            except PlanningError as error:
                self.planner.rejections.append(dict(reason=str(error)))
                if self.phase == "searching_helper":
                    self.helper = None
                    self.begin_lowering(m, session)
                else:
                    self.block(m, session, str(error))
                return
        dt = float(session.model.dt) * session.substeps
        if self.phase in ("approaching_helper", "transporting"):
            if not any(g["holding"] for g in m.grippers.values()):
                self.block(m, session, "Lost the retaining grasp during recovery transport")
                return
            # Reject a stale grasp/path if slipping changed the object-to-holder transform.
            carrier = next(i for i, n in enumerate(m.names) if m.grippers[n]["holding"])
            pose = session.poses()
            actual_tool = np.asarray(pose.position)[tuple(m.arms.values())[carrier].body]
            expected = self.route.centers[0] + actual_tool - self.route.positions[0][carrier]
            if np.linalg.norm(np.asarray(m.state.position) - expected) > 0.015:
                self.block(m, session, "Object slipped outside the planned recovery envelope")
                return
            material = m.material_status()
            if material.get("damage_fraction", 0.0) > self.initial_damage + 1e-4:
                self.block(
                    m, session, "Material damage increased during recovery approach/transport"
                )
                return
            risk = material.get("risk", 0.0)
            self.risk_hold = risk > (0.45 if self.risk_hold else 0.55)
            if self.risk_hold:
                self.target = session.state.q
                session.target_q = self.target
                m.hold_target = self.target
                return
            a, b = self.route.joints[self.route_index - 1 : self.route_index + 1]
            pa, pb = self.route.positions[self.route_index - 1 : self.route_index + 1]
            increment = min(
                dt * 0.04 / max(np.max(np.linalg.norm(pb - pa, axis=1)), 1e-9),
                dt * 0.15 / max(np.max(np.abs(b - a)), 1e-9),
            )
            self.route_alpha = min(1.0, self.route_alpha + increment)
            self.target = jnp.asarray(a + (b - a) * self.route_alpha)
            if (
                self.route_alpha == 1.0
                and np.max(np.abs(np.asarray(session.state.q) - b)) < 0.002
                and np.max(np.abs(np.asarray(session.state.qd))) < 0.03
            ):
                if self.route_index < len(self.route.joints) - 1:
                    self.route_index += 1
                    self.route_alpha = 0.0
                elif self.phase == "approaching_helper":
                    m._command(self.helper, "closed", session)
                    self.transition(m, session, "helper_closing")
                else:
                    # Re-evaluate the measured footprint before the guarded final descent.
                    self.begin_lowering(m, session)
        if self.phase == "lowering":
            self.positions[self.moving, 2] -= 0.015 * dt
            q, errors, angles = self.solve(
                session.state.q, jnp.asarray(self.positions), jnp.asarray(self.rotations)
            )
            if np.max(errors) > 0.002 or np.max(angles) > 0.03:
                self.block(
                    m, session, "Descent is not reachable with the current grasp orientation"
                )
                return
            self.target = q
        session.target_q = self.target
        self.records.append(
            dict(time=session.elapsed, phase=self.phase, target_q=np.asarray(self.target).tolist())
        )
        self.records = self.records[-10000:]

    def feedback(self, m, name, dt, risk):
        c, g = m.adaptive[name], m.grippers[name]
        ceiling = min(g["force_limit_n"], c.safety.damage_force_n * c.safety.force_reserve)
        c.force = min(c.force, ceiling)
        if risk > 0.55:
            c.force *= np.exp(-3 * dt)
        elif g["holding"] and g["slip_speed_m_s"] > 0.004 and self.reason != "material_damage":
            c.force = min(ceiling, c.force + 20 * dt)
        c.status = "recovery_" + self.phase

    def force_scale(self, name):
        if self.phase in ("seating", "releasing", "settling"):
            return self.scale
        if self.phase == "unloading" and name == self.primary:
            return self.scale
        return 1.0

    def after_tick(self, m, session, dt, material):
        if not self.active:
            return
        if material["fractured"]:
            self.block(m, session, "Object has separated into fragments")
            return
        if session.elapsed - self.started > (45 if self.extended else 18):
            self.block(m, session, "Recovery timed out without verified support")
            return
        loads = (
            np.asarray(m.report.support_load)
            if m.report.support_load is not None
            else np.zeros(len(m.supports) + 1)
        )
        load = loads[self.support + 1] if self.support is not None else 0.0
        weight = m.object.mass * np.linalg.norm(m.model.gravity)
        quiet = (
            np.linalg.norm(m.state.velocity) < 0.01
            and np.linalg.norm(m.state.angular_velocity) < 0.05
        )
        if m.structural:
            quiet = (
                quiet
                and np.max(np.linalg.norm(np.asarray(m.contact.structure.velocity), axis=1)) < 0.02
            )
        risk = material.get("risk", 0.0)
        age = session.elapsed - self.changed
        if self.phase in ("stabilizing", "unloading", "opening", "reclosing", "verifying"):
            if material["damaged"] or risk > 0.55 or not m.grippers[self.helper]["holding"]:
                m._command(self.primary, "closed", session)
                self.begin_lowering(m, session)
                return
        if self.phase == "helper_closing":
            helper = m.grippers[self.helper]
            capacity = min(helper["normal_force_n"]) * 2 * helper["mu_dynamic"]
            if (
                helper["holding"]
                and m.stable[self.helper] > 0.2
                and capacity > 1.5 * weight
                and risk < 0.45
            ):
                self.transition(m, session, "stabilizing")
            elif age > 3 or material["damaged"]:
                m._command(self.helper, "open", session)
                self.begin_lowering(m, session)
        elif self.phase == "stabilizing" and age > 0.2:
            self.transition(m, session, "unloading")
        elif self.phase == "unloading":
            self.scale = max(0.0, self.scale - dt * 2)
            if self.scale == 0.0 and quiet and m.stable[self.helper] > 0.2:
                m._command(self.primary, "open", session)
                self.scale = 1.0
                self.transition(m, session, "opening")
        elif self.phase == "opening" and m.grippers[self.primary]["width"] > 0.09:
            m._command(self.primary, "closed", session)
            self.transition(m, session, "reclosing")
        elif self.phase == "reclosing":
            if m.stable[self.primary] > 0.2:
                self.transition(m, session, "verifying")
            elif age > 3:
                self.begin_lowering(m, session)
        elif self.phase == "verifying":
            good = (
                quiet
                and risk < 0.45
                and all(
                    m.grippers[n]["holding"] and m.stable[n] > 0.2
                    for n in (self.primary, self.helper)
                )
            )
            self.dwell = self.dwell + dt if good else 0.0
            if self.dwell >= 0.25:
                if self.new_helper:
                    # A newly contacted arm changes the task's contact configuration.
                    # Keep both grips and finish recovery for replanning, never replay stale waypoints.
                    self.target = session.state.q
                    session.target_q = m.hold_target = self.target
                    m.pending = None
                    self.transition(m, session, "secured")
                    return
                if self.saved_pending:
                    op, n, g, t = self.saved_pending
                    m.pending = (op, n, g, t + session.elapsed - self.started)
                session.target_q = self.saved_goal
                m.hold_target = None
                for c in m.adaptive.values():
                    c.status = "stable"
                    c.recovery_time = c.slip_time = 0.0
                self.cooldown_until = session.elapsed + 1.0
                self.warning_time = 0.0
                self.transition(m, session, "regrasped")
        elif self.phase == "lowering":
            center = np.asarray(m.state.position)
            support = m.supports[self.support]
            if np.any(np.abs(center[:2] - support[:2]) > support[2:4] - 0.002):
                self.block(m, session, "Object drifted outside the selected support")
            elif load > 0.3 * weight:
                self.target = session.state.q
                self.transition(m, session, "seating")
            elif not any(g["holding"] for g in m.grippers.values()):
                self.block(m, session, "Lost all grasps before reaching the support")
        elif self.phase == "seating":
            self.scale = max(0.0, self.scale - 2 * dt)
            good = self.scale == 0 and load > 0.8 * weight and quiet
            self.dwell = self.dwell + dt if good else 0.0
            if self.dwell >= 0.25:
                for name in m.names:
                    m._command(name, "open", session)
                self.scale = 1.0
                self.transition(m, session, "releasing")
            elif age > 3:
                self.block(m, session, "Support did not take the object weight")
        elif self.phase == "releasing" and all(g["width"] > 0.09 for g in m.grippers.values()):
            self.transition(m, session, "settling")
        elif self.phase == "settling":
            good = (
                load > 0.8 * weight
                and quiet
                and all(max(g["normal_force_n"]) < 0.2 for g in m.grippers.values())
            )
            self.dwell = self.dwell + dt if good else 0.0
            if self.dwell >= 0.3:
                m.owner = m.pending = None
                self.verified_support_s = self.dwell
                self.transition(m, session, "placed")
        if self.active:
            session.target_q = self.target
            m.hold_target = self.target
