"""Reproducible contact-driven structural specimens; independent of arm planning."""

from dataclasses import asdict

import jax
import jax.numpy as jnp
import numpy as np

from .structure import CohesiveBar


class StructuralLab:
    """Two finite-mass jaws per tool, no pose constraints on the specimen.

    Prescribed tool trajectories are bench fixtures, not articulated robot dynamics.
    Destructive recipes intentionally disable the existing robot damage guard.
    """

    def __init__(self, material="soft_foam", recipe="bend"):
        if recipe not in ("bend", "tear", "crush"):
            raise ValueError("Unknown structural recipe")
        self.recipe = recipe
        self.model = CohesiveBar(
            material,
            grippers=2 if recipe == "tear" else 1,
            dt=0.0001,
            pad_half_extent=0.025 if recipe == "crush" else 0.035,
        )
        self.state = self.model.initial()
        self.time = 0.0
        self.report = None
        self.history = []
        self.initial_state = self.model.summary(self.state)
        self.centers = self.targets(0)[0]
        self.advance = jax.jit(self._advance)

    def targets(self, t):
        count = self.model.grippers
        y = (
            jnp.array([-0.10, 0.10])
            if count == 2
            else jnp.array([0.0 if self.recipe == "crush" else -0.10])
        )
        lift = jnp.clip(t - 0.65, 0, 1) * 0.08
        pull = jnp.clip(t - 1.8, 0, 2) * 0.05 if self.recipe == "tear" else 0.0
        y = y + jnp.sign(y) * pull
        centers = jnp.stack((jnp.zeros(count), y, jnp.full(count, 0.022 + lift)), axis=1)
        velocity = jnp.zeros((count, 3)).at[:, 2].set(jnp.where((t > 0.65) & (t < 1.65), 0.08, 0))
        if self.recipe == "tear":
            velocity = velocity.at[:, 1].set(
                jnp.where((t > 1.8) & (t < 3.8), jnp.sign(y) * 0.05, 0)
            )
        rotation = jnp.tile(
            jnp.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])[None], (count, 1, 1)
        )
        limit = jnp.full(count, 10.0 if self.recipe == "bend" else 60.0)
        if self.recipe == "crush":
            limit = jnp.full(count, jnp.where(t > 1.8, 400.0, 10.0))
        return centers, rotation, velocity, jnp.zeros((count, 3)), jnp.full(count, 0.004), limit

    def _advance(self, state, time):
        def step(carry, i):
            state, _ = carry
            return self.model.step(state, *self.targets(time + i * self.model.dt)), None

        report = self.model.step(state, *self.targets(time))[1]
        return jax.lax.scan(step, (state, report), jnp.arange(200))[0]

    def step(self):
        if self.time >= 6:
            return self.snapshot()
        state, report = self.advance(self.state, self.time)
        if not np.isfinite(np.asarray(state.position)).all():
            raise ValueError("Structural integration became non-finite; reset the specimen")
        self.state, self.report = state, report
        self.time += 0.02
        self.centers = self.targets(self.time)[0]
        result = self.snapshot()
        self.history.append(result)
        return result

    def snapshot(self):
        return {
            **self.model.summary(self.state, self.report),
            "time_s": self.time,
            "recipe": self.recipe,
            "complete": self.time >= 6,
            "gripper_centers": np.asarray(self.centers).tolist(),
            "commanded_gap_m": np.asarray(self.targets(self.time)[4]).tolist(),
            "force_limit_n": np.asarray(self.targets(self.time)[5]).tolist(),
        }

    def export(self):
        return {
            "schema": "robogogogo.structural-lab.v1",
            "material": asdict(self.model.material),
            "dt_s": self.model.dt,
            "recipe": self.recipe,
            "cells": self.model.cells,
            "length_m": self.model.length,
            "width_m": self.model.width,
            "pad_half_extent_m": self.model.pad_half_extent,
            "solver_iterations": self.model.iterations,
            "initial_state": self.initial_state,
            "limitations": "Calibrated strut cells; predefined cohesive interfaces; prescribed tool motion; nodal self-contact.",
            "frames": self.history,
        }
