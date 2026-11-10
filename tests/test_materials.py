"""Constitutive, history, frame and damage-feedback regressions."""

from dataclasses import replace
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from robogogogo.adaptive_grasp import AdaptiveGrip
from robogogogo.simulation.materials import (
    MATERIALS,
    initial_material_state,
    material_response,
    material_summary,
    surface_indices,
)


@partial(jax.jit, static_argnames=("material", "steps", "ramp_steps"))
def press(state, material, overlap, steps=1000, dt=0.0002, ramp_steps=0):
    def step(state, index):
        force, following = material_response(
            state,
            material,
            jnp.array([37]),
            jnp.array([overlap * jnp.minimum((index + 1) / max(ramp_steps, 1), 1.0)]),
            jnp.array([0.0]),
            jnp.array([24e-6]),
            jnp.array([4000.0]),
            jnp.array([3.0]),
            dt,
        )
        return following, force[0]

    return jax.lax.scan(step, state, jnp.arange(steps))


def test_soft_material_creeps_and_recovers_without_damage_at_small_strain():
    material = MATERIALS["soft_foam"]
    held, forces = press(initial_material_state(), material, 0.002, steps=3000, ramp_steps=1000)
    assert float(forces[-1]) < float(forces.max())
    assert 0.001 < float(held.indentation[37]) < 0.002
    assert float(held.permanent.max()) == 0
    assert not material_summary(material, held)["damaged"]
    released, _ = press(held, material, -0.01, steps=3000)
    assert float(released.indentation.max()) < 1e-8
    assert float(released.pressure.max()) == 0
    assert float(released.peak_pressure.max()) >= float(held.peak_pressure.max())


def test_plastic_dent_and_damage_survive_release_and_contact_relocation():
    material = MATERIALS["soft_foam"]
    held, _ = press(initial_material_state(), material, 0.009, steps=2000)
    assert float(held.permanent[37]) > 0.001
    assert material_summary(material, held)["damaged"]
    released, _ = press(held, material, -0.01, steps=5000)
    assert float(released.permanent[37]) >= float(held.permanent[37])
    np.testing.assert_allclose(released.indentation[37], released.permanent[37], atol=1e-8)
    assert np.all(np.asarray(released.damage) >= np.asarray(held.damage))
    assert np.count_nonzero(np.asarray(released.permanent)) == 1


def test_fragile_pressure_failure_latches_and_reduces_load_bearing_stiffness():
    material = MATERIALS["fragile_shell"]
    state, _ = press(initial_material_state(), material, 0.003, steps=100)
    assert material_summary(material, state)["fractured"]
    assert float(state.peak_pressure.max()) > material.strength_pa
    intact = initial_material_state()
    damaged = intact._replace(damage=intact.damage.at[37].set(1))
    _, intact_force = press(intact, replace(material, strength_pa=1e10, failure_strain=0.9), 0.0001)
    _, damaged_force = press(
        damaged, replace(material, strength_pa=1e10, failure_strain=0.9), 0.0001
    )
    assert float(damaged_force[-1]) < float(intact_force[-1])


def test_material_refinement_and_validation():
    material = MATERIALS["soft_foam"]
    coarse, _ = press(initial_material_state(), material, 0.002, steps=1000)
    fine, _ = press(initial_material_state(), material, 0.002, steps=2000, dt=0.0001)
    np.testing.assert_allclose(coarse.indentation, fine.indentation, atol=1e-6)
    for kwargs in ({"modulus_pa": -1}, {"relaxation_s": float("nan")}, {"yield_strain": 0.9}):
        with pytest.raises(ValueError):
            replace(material, **kwargs)


def test_material_mapping_is_body_local_and_wraps_azimuth():
    points = jnp.array([[0.03, 0, 1e-10], [0.03, 0, -1e-10], [0, -0.14, 0.02]])
    indices = surface_indices(points, 0.12, 0.03)
    assert int(indices[0]) // 24 == int(indices[1]) // 24
    assert int(indices[0]) % 24 == 0 and int(indices[1]) % 24 == 23
    assert int(indices[2]) // 24 < 2


def test_material_risk_overrides_slip_tightening_and_damage_latches_fault():
    c = AdaptiveGrip()
    for _ in range(10):
        c.update(
            closed=True,
            normals=[8, 8],
            slip_speed=0.05,
            peak_force=8,
            hardware_limit=40,
            dt=0.01,
            material_risk=0.6,
        )
    assert c.force < 8 and c.status == "unloading" and c.motion_hold
    c.update(
        closed=True,
        normals=[2, 2],
        slip_speed=0,
        peak_force=2,
        hardware_limit=40,
        dt=0.01,
        material_damaged=True,
    )
    assert c.fault and c.force == 0


def test_coupled_soft_grasp_deforms_without_damage_and_fragile_overgrip_stops():
    import runpy
    from pathlib import Path

    run = runpy.run_path(str(Path(__file__).parent / "fixtures/material_grasp.py"))["run"]
    soft = run("soft_foam", duration=2.0)
    assert soft["failure"] is None
    material = soft["object"]["material"]
    assert 0.0005 < material["maximum_indentation_m"] < 0.003
    assert material["maximum_permanent_m"] == 0 and not material["damaged"]
    assert soft["grippers"]["arm"]["holding"]
    assert soft["object"]["owner"] == "arm"
    assert soft["object"]["position"][2] > 0.53
    damaged = run("fragile_shell", force=40, adaptive=False, duration=1.0)
    assert damaged["failure"] == "Object material damaged"
    assert damaged["object"]["material"]["damaged"]
    assert any(e["event"] == "material_damage" for e in damaged["events"])
    safe = run("fragile_shell", duration=2.0)
    assert safe["failure"] is None
    assert not safe["object"]["material"]["damaged"]
    assert safe["grippers"]["arm"]["holding"]
    assert safe["object"]["position"][2] > 0.53


def test_unloading_crosses_hysteresis_band_without_tightening_again():
    c = AdaptiveGrip()
    c.update(
        closed=True,
        normals=[8, 8],
        slip_speed=0,
        peak_force=8,
        hardware_limit=40,
        dt=0.01,
        material_risk=0.6,
    )
    force = c.force
    for _ in range(20):
        c.update(
            closed=True,
            normals=[6, 6],
            slip_speed=0,
            peak_force=6,
            hardware_limit=40,
            dt=0.01,
            material_risk=0.5,
        )
    assert c.force < force and c.motion_hold
    force = c.force
    for _ in range(20):
        c.update(
            closed=True,
            normals=[4, 4],
            slip_speed=0,
            peak_force=4,
            hardware_limit=40,
            dt=0.01,
            material_risk=0.3,
        )
    assert c.force == force and c.status == "stable" and c.fault is None


def test_material_protection_reduces_overgrip_independently_of_force_threshold():
    import runpy
    from pathlib import Path

    run = runpy.run_path(str(Path(__file__).parent / "fixtures/material_grasp.py"))["run"]
    protected = run("fragile_shell", force=20, force_threshold=55)
    assert protected["failure"] is None
    assert protected["object"]["owner"] == "arm"
    assert not protected["object"]["material"]["damaged"]
    assert protected["feedback"]["arm"]["force_setpoint_n"] < 20
    assert any(e["event"] == "adaptive_unloading" for e in protected["events"])


def test_material_interpolation_is_continuous_across_cell_boundaries_and_seam():
    from robogogogo.simulation.materials import ANGULAR_CELLS, AXIAL_CELLS, surface_stencil

    for angle in (0.0, 2 * np.pi / ANGULAR_CELLS):
        angles = jnp.array([angle - 1e-8, angle + 1e-8])
        points = jnp.stack((0.03 * jnp.cos(angles), jnp.zeros(2), 0.03 * jnp.sin(angles)), axis=1)
        indices, weights = surface_stencil(points, 0.12, 0.03)
        field = jnp.tile(
            jnp.cos((jnp.arange(ANGULAR_CELLS) + 0.5) * 2 * jnp.pi / ANGULAR_CELLS), AXIAL_CELLS
        )
        values = jnp.sum(field[indices] * weights, axis=1)
        np.testing.assert_allclose(weights.sum(axis=1), 1)
        np.testing.assert_allclose(values[0], values[1], atol=1e-7)
