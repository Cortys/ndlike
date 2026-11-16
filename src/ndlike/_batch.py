"""Backend-independent indexing of compact broadcast representations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    from ndlike.array_like import ToIndices


def is_basic_index(index: ToIndices, is_scalar_integer: Callable[[object], bool]) -> bool:
    """Recognize basic indices without reading array values or traced integers."""
    indices = index if isinstance(index, tuple) else (index,)
    for item in indices:
        if item is None or item is Ellipsis:
            continue
        if isinstance(item, slice):
            if all(part is None or isinstance(part, (int, np.integer)) for part in (item.start, item.stop, item.step)):
                continue
            return False
        if isinstance(item, (bool, np.bool_)):
            return False
        if isinstance(item, (int, np.integer)) or is_scalar_integer(item):
            continue
        return False
    return True


def normalize_broadcast_index(
    index: ToIndices, value_batch_shape: tuple[int, ...], primary_batch_shape: tuple[int, ...]
) -> tuple[Any, ...]:
    """Project a known-basic logical batch index onto compact storage."""
    index_tuple = index if isinstance(index, tuple) else (index,)
    missing = len(primary_batch_shape) - sum(item is not None and item is not Ellipsis for item in index_tuple)
    ellipses = sum(item is Ellipsis for item in index_tuple)
    if missing < 0 or ellipses > 1:
        msg = "Too many indices or ellipses for protected batch dimensions."
        raise IndexError(msg)

    normalized: list[Any] = []
    for item in index_tuple:
        normalized.extend((slice(None),) * missing if item is Ellipsis else (item,))
    if ellipses == 0:
        normalized.extend((slice(None),) * missing)

    leading = len(primary_batch_shape) - len(value_batch_shape)
    projected: list[Any] = []
    axis = 0
    for item in normalized:
        if item is None:
            projected.append(None)
            continue
        if axis >= leading:
            shared = value_batch_shape[axis - leading] == 1 and primary_batch_shape[axis] != 1
            projected.append((slice(None) if isinstance(item, slice) else 0) if shared else item)
        axis += 1
    return tuple(projected)
