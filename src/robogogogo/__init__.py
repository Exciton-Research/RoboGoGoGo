"""Robogogogo: the RG Mobile Duo robot workflow and native 3D engine."""

from importlib import import_module

_MODULES = {"control", "models", "spatial", "urdf", "env3d"}


def __getattr__(name):
    if name not in _MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = import_module(f".{name}", __name__)
    globals()[name] = value
    return value


__all__ = sorted(_MODULES)
