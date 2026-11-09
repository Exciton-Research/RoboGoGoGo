"""Use double precision for numerical regression tests."""

import jax

jax.config.update("jax_enable_x64", True)
