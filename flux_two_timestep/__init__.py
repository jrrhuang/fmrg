"""
Two-timestep FLUX inference pipeline for FlowMap LoRAs.

- `FluxPipelineTwoTimestep`: a `FluxPipeline` subclass that conditions the
  transformer on a pair of (current, target) timesteps.
- `add_dual_time_embedder`: wraps the transformer's time-text embedder so
  it accepts the (t, t_next) pair.

`_diffusers_compat` installs a no-op `cache_context` on
`FluxTransformer2DModel` for diffusers versions that don't ship one.
"""

from . import _diffusers_compat  # noqa: F401  (must run before FluxPipelineTwoTimestep)
from .two_timestep_inference import FluxPipelineTwoTimestep
from .dual_time_embedder import add_dual_time_embedder

__all__ = ["FluxPipelineTwoTimestep", "add_dual_time_embedder"]
