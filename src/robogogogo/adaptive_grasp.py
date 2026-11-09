"""Bounded tactile feedback controller; object force limits are user-supplied envelopes."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class GripSafety:
    initial_force_n: float = 8.0
    damage_force_n: float = 55.0
    force_reserve: float = 0.75
    increase_rate_n_s: float = 80.0
    slip_on_m_s: float = 0.004
    slip_off_m_s: float = 0.001
    recovery_timeout_s: float = 1.5

    def __post_init__(self):
        values = vars(self)
        if not all(math.isfinite(v) and v > 0 for v in values.values()):
            raise ValueError("Grip safety values must be finite and positive")
        if not self.force_reserve < 1 or self.slip_off_m_s >= self.slip_on_m_s:
            raise ValueError("Require force reserve < 1 and slip hysteresis")
        if self.initial_force_n > self.damage_force_n * self.force_reserve:
            raise ValueError("Initial force exceeds reserved object force envelope")


class AdaptiveGrip:
    """100 Hz feedback using measured pad loads and relative contact speed.

    No coefficient-of-friction oracle or owner-dependent force. A recovered grip
    retains its learned force; opening resets the next acquisition to initial force.
    """

    def __init__(self, safety=GripSafety()):
        self.safety = safety
        self.force = safety.initial_force_n
        self.status = "idle"
        self.filtered_speed = 0.0
        self.slip_time = 0.0
        self.stable_time = 0.0
        self.recovery_time = 0.0
        self.peak_force = 0.0
        self.fault = None

    def update(
        self,
        *,
        closed,
        normals,
        slip_speed,
        peak_force,
        hardware_limit,
        dt,
        material_risk=0.0,
        material_damaged=False,
    ):
        if (
            not all(
                math.isfinite(x)
                for x in (*normals, slip_speed, peak_force, hardware_limit, dt, material_risk)
            )
            or dt <= 0
        ):
            raise ValueError("Invalid tactile feedback")
        if min(*normals, slip_speed, peak_force, hardware_limit, material_risk) < 0:
            raise ValueError("Tactile loads and limits must be nonnegative")
        s = self.safety
        ceiling = min(hardware_limit, s.damage_force_n * s.force_reserve)
        self.peak_force = max(self.peak_force, peak_force)
        if peak_force > s.damage_force_n:
            self.fault = "Object damage force threshold exceeded"
        if material_damaged:
            self.fault = "Object material damage detected"
        if self.fault:
            self.status = "fault"
            self.force = 0.0
            return
        if not closed:
            self.force = min(s.initial_force_n, ceiling)
            self.status = "idle"
            self.filtered_speed = self.slip_time = self.stable_time = self.recovery_time = 0.0
            return
        self.force = min(self.force, ceiling)
        self.filtered_speed += (1 - math.exp(-dt / 0.02)) * (slip_speed - self.filtered_speed)
        bilateral = min(normals) > 0.3
        if peak_force > 0.9 * s.damage_force_n or material_risk > 0.55:
            reduction = 2 * s.increase_rate_n_s * dt
            if material_risk > 0.55:
                reduction = min(reduction, self.force * (1 - math.exp(-10 * dt)))
            self.force = max(0.0, self.force - reduction)
            self.status = "unloading"
            self.stable_time = 0.0
            self.recovery_time += dt
        elif self.status == "unloading":
            # Do not treat unloading as slip recovery and immediately squeeze again.
            self.recovery_time += dt
            if material_risk >= 0.45:
                self.force *= math.exp(-3 * dt)
            self.stable_time = (
                self.stable_time + dt
                if bilateral and material_risk < 0.45 and self.filtered_speed < s.slip_off_m_s
                else 0.0
            )
            if self.stable_time >= 0.12:
                self.status = "stable"
                self.recovery_time = self.slip_time = 0.0
        elif not bilateral:
            self.status = "acquiring"
            # Never squeeze harder on empty space or on a single fingertip contact.
            self.slip_time = self.stable_time = 0.0
        else:
            self.slip_time = self.slip_time + dt if self.filtered_speed > s.slip_on_m_s else 0.0
            if self.slip_time >= 0.02 or self.status in (
                "recovering",
                "force_limited",
            ):
                self.recovery_time += dt
                self.stable_time = (
                    self.stable_time + dt if self.filtered_speed < s.slip_off_m_s else 0.0
                )
                if self.stable_time >= 0.12:
                    self.status = "stable"
                    self.recovery_time = self.slip_time = 0.0
                else:
                    self.force = min(ceiling, self.force + s.increase_rate_n_s * dt)
                    self.status = "force_limited" if self.force >= ceiling - 1e-9 else "recovering"
            else:
                self.status = "stable"
        if self.recovery_time > s.recovery_timeout_s:
            self.fault = "Grip cannot stabilize within the force and material limits"
            self.status = "fault"

    @property
    def motion_hold(self):
        return self.status in ("recovering", "force_limited", "unloading", "fault")

    def summary(self):
        return dict(
            status=self.status,
            force_setpoint_n=self.force,
            filtered_slip_speed_m_s=self.filtered_speed,
            peak_force_n=self.peak_force,
            damage_force_n=self.safety.damage_force_n,
            fault=self.fault,
            motion_hold=self.motion_hold,
        )
