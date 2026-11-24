"""Tools for defining custom array types across NumPy, PyTorch, and JAX."""

from ndlike.array_like import ArrayLike, WeakArrayLike
from ndlike.tuple import BroadcastTuple

__all__ = ["ArrayLike", "BroadcastTuple", "WeakArrayLike"]
