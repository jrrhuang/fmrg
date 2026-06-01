"""
Install a no-op `cache_context` method on `FluxTransformer2DModel` if the
installed `diffusers` version lacks one. When no inference-time caching is
enabled (our case) the upstream `cache_context` is itself a no-op, so this
makes `FluxPipelineTwoTimestep` work against the pinned `diffusers==0.34.0`.
"""

from contextlib import contextmanager

from diffusers.models import FluxTransformer2DModel


if not hasattr(FluxTransformer2DModel, "cache_context"):
    @contextmanager
    def _noop_cache_context(self, name: str):
        yield

    FluxTransformer2DModel.cache_context = _noop_cache_context
