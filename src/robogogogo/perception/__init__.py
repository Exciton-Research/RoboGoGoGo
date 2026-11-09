"""Optional classical stereo perception; the physics engine needs no OpenCV."""

from .stereo import StereoResult, estimate_stereo, evaluate_depth

__all__ = ["StereoResult", "estimate_stereo", "evaluate_depth"]
