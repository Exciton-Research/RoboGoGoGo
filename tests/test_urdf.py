from importlib.resources import files

import jax.numpy as jnp
import numpy as np
import pytest

from robogogogo import models, urdf
from robogogogo import spatial as sp


def fixture():
    return files("robogogogo").joinpath("assets/arm6.urdf").read_text()


def test_load_bundled_robot_matches_native_dynamics():
    robot = urdf.loads(fixture())
    expected = models.arm6()
    assert robot.name == "robogogogo_arm6"
    assert len(robot.joint_names) == 6
    assert len(robot.visuals) == 8
    q = jnp.array([0.2, -0.8, 1.2, 0.1, -0.4, 0.0])
    v = jnp.array([0.1, -0.2, 0.3, 0.0, 0.2, -0.1])
    np.testing.assert_allclose(
        sp.mass_matrix(robot.model, q), sp.mass_matrix(expected, q), atol=1e-7
    )
    np.testing.assert_allclose(
        sp.inverse_dynamics(robot.model, q, v, v), sp.inverse_dynamics(expected, q, v, v), atol=1e-6
    )


def test_inertial_frame_rotation_and_prismatic_joint():
    robot = urdf.loads("""<robot name="slider">
      <link name="base"/>
      <link name="moving"><inertial><origin xyz=".1 .2 .3" rpy="0 0 1.5707963267948966"/>
        <mass value="2"/><inertia ixx=".02" iyy=".03" izz=".04" ixy="0" ixz="0" iyz="0"/>
      </inertial><collision><origin xyz="0 0 0"/><geometry><sphere radius=".1"/></geometry></collision></link>
      <joint name="slide" type="prismatic"><parent link="base"/><child link="moving"/>
        <origin xyz="0 0 .5"/><axis xyz="0 0 1"/><limit lower="-.2" upper=".5" effort="20" velocity="2"/>
      </joint></robot>""")
    np.testing.assert_allclose(robot.model.inertia[1], np.diag([0.03, 0.02, 0.04]), atol=1e-9)
    np.testing.assert_allclose(sp.mass_matrix(robot.model, jnp.array([0.1])), [[2.0]], atol=1e-10)
    np.testing.assert_allclose(sp.point_position(robot.model, jnp.array([0.1])), [0, 0, 0.6])
    assert robot.model.sphere_radius.shape == (1,)


@pytest.mark.parametrize(
    "change",
    [
        lambda s: s.replace('type="revolute"', 'type="floating"', 1),
        lambda s: s.replace('<axis xyz="0.0 0.0 1.0" />', '<axis xyz="0 0 0" />', 1),
        lambda s: s.replace('<parent link="pedestal" />', '<parent link="missing" />', 1),
        lambda s: s.replace("</joint>", '<mimic joint="other"/></joint>', 1),
        lambda s: s.replace(
            "</link>", '<collision><geometry><box size="1 1 1"/></geometry></collision></link>', 1
        ),
        lambda s: s.replace("</robot>", '<link name="disconnected"/></robot>'),
    ],
)
def test_unsupported_or_invalid_urdf_is_rejected(change):
    original = fixture()
    modified = change(original)
    assert modified != original
    with pytest.raises(ValueError):
        urdf.loads(modified)
