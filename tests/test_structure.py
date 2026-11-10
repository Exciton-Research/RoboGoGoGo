"""Physical regression tests for the structural specimen solver."""

import jax.numpy as jnp
import numpy as np
import pytest

from robogogogo.simulation.structural_lab import StructuralLab
from robogogogo.simulation.structure import CohesiveBar, StructuralMaterial


def free_step(model, state, **kwargs):
    return model.step(
        state,
        jnp.array([[2.0, 0.0, 2.0]]),
        jnp.eye(3)[None],
        jnp.zeros((1, 3)),
        jnp.zeros((1, 3)),
        jnp.array([0.075]),
        jnp.array([10.0]),
        **kwargs,
    )


def test_free_translation_and_gravity_conserve_internal_momentum():
    model = CohesiveBar(cells=3)
    state = model.initial(center=(0, 0, 1))._replace(
        velocity=jnp.tile(jnp.array([0.2, -0.1, 0.3]), (24, 1))
    )
    following, _ = free_step(model, state)
    expected = np.array([0.2, -0.1, 0.3]) + model.dt * np.array([0, 0, -9.81])
    np.testing.assert_allclose(following.velocity.mean(axis=0), expected, atol=1e-9)
    assert float(following.damage.max()) == 0


def test_broken_interfaces_do_not_heal_or_weld_fragments():
    model = CohesiveBar(cells=3)
    state = model.initial(center=(0, 0, 1))
    state = state._replace(
        damage=jnp.ones_like(state.damage),
        velocity=state.velocity.at[:8, 1].set(-0.2).at[-8:, 1].set(0.2),
    )
    following, _ = free_step(model, state)
    np.testing.assert_array_equal(following.damage, 1)
    assert len(model.fragments(following)) == 3
    assert sum(f["mass_kg"] for f in model.fragments(following)) == pytest.approx(
        model.material.mass
    )
    assert float(following.velocity[:8, 1].mean()) == pytest.approx(-0.2, abs=1e-7)
    assert float(following.velocity[-8:, 1].mean()) == pytest.approx(0.2, abs=1e-7)


def test_invalid_fracture_energy_rejected():
    with pytest.raises(ValueError, match="Fracture energy"):
        StructuralMaterial("invalid", 1000, 1, 1000, 1000, 0.00001)


@pytest.mark.parametrize(
    "material,recipe", [("soft_foam", "bend"), ("soft_foam", "tear"), ("fragile_shell", "crush")]
)
def test_contact_driven_recipes(material, recipe):
    lab = StructuralLab(material, recipe)
    previous_damage = np.asarray(lab.state.damage)
    intact = None
    for i in range(140):
        frame = lab.step()
        damage = np.asarray(lab.state.damage)
        assert np.all(damage >= previous_damage)
        assert np.isfinite(np.asarray(lab.state.velocity)).all()
        previous_damage = damage
        if i == 79:
            intact = frame
    assert intact["maximum_damage"] == 0
    assert np.mean(np.asarray(intact["positions"])[:, :, 2]) > 0.05
    if recipe == "bend":
        assert frame["bend_m"] > 0.0005
        assert frame["maximum_damage"] == 0
        assert len(frame["fragments"]) == 1
    else:
        assert frame["broken_bonds"] >= 4
        assert len(frame["fragments"]) > 1
    exported = lab.export()
    assert len(exported["frames"]) == 140
    assert exported["dt_s"] == 0.0001


def test_standalone_structural_api_is_removed():
    from robogogogo.studio.api import create_app

    app = create_app()
    assert not any(route.path.startswith("/api/structure") for route in app.routes)
