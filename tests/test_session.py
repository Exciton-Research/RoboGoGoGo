import json

import jax.numpy as jnp
import numpy as np
import pytest

from fixtures import models
from robogogogo import spatial as sp
from robogogogo.session import Session


def test_export_can_replay_applied_torques(tmp_path):
    model = models.slider()
    session = Session(model, scene="drop")
    for _ in range(20):
        session.advance()
    path = session.export(tmp_path / "recording.npz")
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
        assert metadata["format"] == "robogogogo-trajectory-v1"
        assert data["q"].shape == (21, 1)
        assert data["time"][-1] == pytest.approx(0.1)
        state = sp.State(jnp.asarray(data["q"][0]), jnp.asarray(data["qd"][0]))
        for torque in data["torque"][1:]:
            state = sp.step(model, state, jnp.asarray(torque), substeps=metadata["substeps"])
        np.testing.assert_allclose(state.q, data["q"][-1], atol=1e-12)
        assert data["model_inertia"].shape == (1, 3, 3)
    session.reset()
    assert session.elapsed == 0 and len(session.history) == 1
