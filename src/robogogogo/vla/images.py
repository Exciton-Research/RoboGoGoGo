"""Compatibility import; use robogogogo.data.images."""

import importlib
import sys

if __name__ == "__main__":
    import runpy

    runpy.run_module("robogogogo.data.images", run_name="__main__")
else:
    sys.modules[__name__] = importlib.import_module("robogogogo.data.images")
