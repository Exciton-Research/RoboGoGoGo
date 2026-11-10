"""robogogogo: a dual-arm robotics workflow with independent deployment and simulation layers."""

from importlib import import_module

_MODULES = {"control", "env", "env3d", "models", "spatial", "urdf"}
_PHYSICS = {
    "Model",
    "State",
    "acceleration",
    "energy",
    "kinematics",
    "make_model",
    "mass_matrix",
    "rollout",
    "step",
}


def __getattr__(name):
    if name in _MODULES:
        value = import_module(f".{name}", __name__)
    elif name in _PHYSICS:
        value = getattr(import_module(".physics", __name__), name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


__all__ = [
    "Model",
    "State",
    "make_model",
    "kinematics",
    "mass_matrix",
    "acceleration",
    "energy",
    "step",
    "rollout",
    "env",
    "spatial",
    "models",
    "control",
    "env3d",
    "urdf",
]
