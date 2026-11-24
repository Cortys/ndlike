"""Heterogeneous broadcast tuples with backend-specific operation dispatch.

Import NumpyBroadcastTuple, JaxBroadcastTuple, or TorchBroadcastTuple from the
corresponding numpy, jax, or torch submodule. Optional backends are loaded only
when their submodule is imported.
"""

from ndlike.tuple._common import BroadcastTuple

__all__ = ["BroadcastTuple"]
