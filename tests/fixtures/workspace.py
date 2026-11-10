"""Legacy numerical regression workspaces, excluded from the installed product."""

import time
import xml.etree.ElementTree as ET
from dataclasses import replace
from importlib.resources import files

import jax.numpy as jnp
import numpy as np

from robogogogo import urdf
from robogogogo.robot.cameras import mounted_rig
from robogogogo.session import Session
from robogogogo.studio.schemas import CameraSpec
from robogogogo.studio.workspace import Studio as ProductStudio

from . import models


class Studio(ProductStudio):
    def _replace_session(self, scene, xml=None, dt=0.002, substeps=5, arms=None):
        from robogogogo.robot.mobile_design import (
            DESIGN,
            mobile_arm_definitions,
            mobile_dual_arm_urdf,
            mobile_home,
        )

        if scene == "mobile_dual_arm":
            xml, arms = mobile_dual_arm_urdf(), mobile_arm_definitions()
        if scene in ("dual_arm", "dual_arm_head"):
            xml = models.multi_arm_urdf(2)
            arms = [
                {"name": "left_arm", "tool": "left_tool"},
                {"name": "right_arm", "tool": "right_tool"},
            ]
        if scene in ("urdf", "dual_arm", "dual_arm_head", "mobile_dual_arm"):
            if not xml:
                raise ValueError("URDF content is required")
            try:
                robot = urdf.loads(xml, dt=dt)
            except (ET.ParseError, KeyError, TypeError) as error:
                raise ValueError(f"Invalid URDF: {error}") from error
            model, label = robot.model, robot.name
            visuals = list(robot.visuals)
            if scene == "mobile_dual_arm":
                model = replace(model, home=jnp.asarray(mobile_home(model)))
            if scene in ("dual_arm", "dual_arm_head"):
                model = replace(model, home=jnp.tile(models.arm6(dt=dt).home, 2))
        elif scene == "arm":
            arms = arms or [{"name": "arm", "tool": "tool"}]
            model, label = models.arm6(dt=dt), "RG-6 Research Arm"
            visuals = list(
                urdf.loads(
                    files("robogogogo").joinpath("assets/mobile_arm_chain.urdf").read_text()
                ).visuals
            )
        elif scene == "drop":
            model, label, visuals = models.slider(dt=dt), "Guided sphere", []
        else:
            raise ValueError("Unknown scene")
        if scene == "dual_arm_head":
            model, visuals = models.with_pan_tilt_head(model, visuals)
            label += " + Pan/Tilt Head"
        elif scene in ("arm", "dual_arm"):
            model, visuals = models.with_camera_mast(model, visuals)
        if model.home.size > 48 or len(model.parents) > 128:
            raise ValueError("Studio currently supports up to 48 moving joints and 128 links")
        candidate = Session(
            model,
            scene="drop" if scene == "drop" else "arm",
            substeps=substeps,
            arms=arms,
            mobile=scene == "mobile_dual_arm",
        )
        self.session, self.scene, self.xml = candidate, scene, xml
        self.episode = None
        self.task = None
        self.running = False
        self.epoch += 1
        self.error = None
        moving = [(i, index) for i, index in enumerate(model.indices) if index >= 0]

        def finite(value):
            return float(value) if np.isfinite(float(value)) else None

        self.model_info = dict(
            name=label,
            mobile_design=DESIGN if scene == "mobile_dual_arm" else None,
            scene=scene,
            dof=int(model.home.size),
            arms=[arm.to_dict() for arm in candidate.arms],
            camera_presets={
                key: [
                    CameraSpec(**config).model_dump(mode="json")
                    for config in mounted_rig(model, candidate.arms, wrists=wrists)
                ]
                for key, wrists in [("head_wrists", True), ("head", False)]
            }
            if "head" in model.names and len(candidate.arms) <= 6
            else {},
            names=list(model.names),
            parents=list(model.parents),
            kinds=list(model.kinds),
            indices=list(model.indices),
            masses=np.asarray(model.mass).tolist(),
            axes=np.asarray(model.axis).tolist(),
            joints=[
                dict(
                    name=model.names[i],
                    body=i,
                    kind=model.kinds[i],
                    lower=finite(model.lower[j]),
                    upper=finite(model.upper[j]),
                    effort=float(model.effort[j]),
                    home=float(model.home[j]),
                )
                for i, j in moving
            ],
            dt=float(model.dt),
            substeps=substeps,
            visuals=visuals,
            spheres=[
                dict(body=int(b), center=np.asarray(c).tolist(), radius=float(r))
                for b, c, r in zip(model.sphere_body, model.sphere_center, model.sphere_radius)
            ],
            backend="JAX / CPU"
            if candidate.state.q.device.platform == "cpu"
            else f"JAX / {candidate.state.q.device.platform.upper()}",
            dtype=str(candidate.state.q.dtype),
            max_samples=self.max_samples,
        )
        self._run_wall, self._run_sim = time.perf_counter(), 0.0
        self._refresh()
