"""Explicit dependency injection for legacy numerical regression tests only."""

from robogogogo.studio.api import create_app as product_app

from .workspace import Studio


def create_app(**kwargs):
    if kwargs.get("initial_scene", "mobile_dual_arm") != "mobile_dual_arm":
        kwargs["_workspace_factory"] = Studio
    return product_app(**kwargs)
