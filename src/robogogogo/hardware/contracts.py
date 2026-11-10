"""Compatibility import; use robogogogo.robot.contracts."""

import importlib
import sys

if __name__ == "__main__":
    import runpy

    runpy.run_module("robogogogo.robot.contracts", run_name="__main__")
else:
    sys.modules[__name__] = importlib.import_module("robogogogo.robot.contracts")
