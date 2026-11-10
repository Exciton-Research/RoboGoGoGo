"""Compatibility import; use robogogogo.robot.ros2."""

import importlib
import sys

if __name__ == "__main__":
    import runpy

    runpy.run_module("robogogogo.robot.ros2", run_name="__main__")
else:
    sys.modules[__name__] = importlib.import_module("robogogogo.robot.ros2")
