"""Panda manipulation environment and local workbench."""


def __getattr__(name):
    if name == "PandaEnv":
        from .environment import PandaEnv

        return PandaEnv
    raise AttributeError(name)


__all__ = ["PandaEnv"]
