"""Compatibility import; use robogogogo.runtime.supervisor."""

import importlib
import sys

if __name__ == "__main__":
    import runpy

    runpy.run_module("robogogogo.runtime.supervisor", run_name="__main__")
else:
    sys.modules[__name__] = importlib.import_module("robogogogo.runtime.supervisor")
