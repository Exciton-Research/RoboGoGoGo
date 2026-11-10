"""Optional classical stereo perception; the physics engine needs no OpenCV."""

from .memory import SceneMemory
from .recovery import RecoveryObservation, SurfaceCloud, backproject, detect, fuse
from .stereo import StereoResult, estimate_stereo, evaluate_depth

__all__ = [
    "SceneMemory",
    "StereoResult",
    "estimate_stereo",
    "evaluate_depth",
    "RecoveryObservation",
    "SurfaceCloud",
    "backproject",
    "detect",
    "fuse",
]
