"""Panda manipulation environment and local workbench."""


def __getattr__(name):
    if name in {"PandaPolicy", "step_policy"}:
        from . import policy

        return getattr(policy, name)
    if name == "PandaEnv":
        from .environment import PandaEnv

        return PandaEnv
    raise AttributeError(name)


__all__ = ["PandaEnv", "PandaPolicy", "step_policy"]
