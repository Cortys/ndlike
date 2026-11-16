"""NumPy broadcast tuples with opaque heterogeneous members."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Self, cast

import numpy as np

from ndlike.array_like import NumpyLikeImplementation, to_numpy_like
from ndlike.tuple._common import BroadcastTuple, tuple_function

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from types import ModuleType

    from numpy.typing import DTypeLike

    from ndlike.array_like import AnyNumpyLike, ArrayFlagsLike


_MUTATING_FUNCTIONS = {
    np.copyto: "dst",
    np.put: "a",
    np.putmask: "a",
    np.place: "arr",
    np.fill_diagonal: "a",
    np.put_along_axis: "arr",
}


# Empty inputs cannot reveal return structure by dispatching to a member.
_EMPTY_ARRAY_FUNCTIONS = frozenset(
    (
        np.reshape,
        np.transpose,
        np.matrix_transpose,
        np.swapaxes,
        np.moveaxis,
        np.squeeze,
        np.expand_dims,
        np.broadcast_to,
        np.stack,
        np.concatenate,
        np.copy,
        np.astype,
        np.ravel,
        np.repeat,
        np.take,
        np.take_along_axis,
        np.sum,
        np.prod,
        np.mean,
        np.std,
        np.var,
        np.all,
        np.any,
        np.cumsum,
        np.cumprod,
        np.dot,
    )
)


class NumpyBroadcastTuple[Members: tuple[AnyNumpyLike | np.ndarray | np.generic, ...]](
    BroadcastTuple[Members],
    NumpyLikeImplementation[
        "NumpyBroadcastTuple[Members]",
        object,
        "NumpyBroadcastTuple[Members]",
        object,
        "NumpyBroadcastTuple[Members]",
        "NumpyBroadcastTuple[tuple[Any, ...]]",
    ],
):
    """A collection exposing the broadcast shape of its NumPy-like members.

    Functions and ufuncs pass through to corresponding members after broadcasting.
    Indexing addresses the logical batch; use ``members`` for positional access.
    """

    def __array_like__(self, dtype: DTypeLike | None = None, /, *, copy: bool | None = None) -> Self:
        """Return self by default, forwarding explicit conversion options to members."""
        if dtype is None and not copy:
            return self
        return self._map_members(lambda member: to_numpy_like(member, dtype=dtype, copy=copy))

    def __post_init__(self) -> None:
        """Require native or dispatched members so shaping cannot coerce representations."""
        super().__post_init__()
        for member in self.members:
            if not isinstance(member, (np.ndarray, np.generic)) and not callable(
                getattr(type(member), "__array_function__", None)
            ):
                msg = "NumPy tuple members must be NumPy arrays or implement __array_function__."
                raise TypeError(msg)

    @staticmethod
    def _broadcast_member[V: AnyNumpyLike | np.ndarray | np.generic](member: V, shape: tuple[int, ...]) -> V:
        return cast("V", np.broadcast_to(member, shape))

    @staticmethod
    def _is_scalar_integer(value: object) -> bool:
        return isinstance(value, np.ndarray) and value.ndim == 0 and np.issubdtype(value.dtype, np.integer)

    @property
    def dtype(self) -> np.dtype:
        """Representative dtype, or object dtype when there are no members."""
        return self.members[0].dtype if self.members else np.dtype(object)

    def __array_namespace__(self, /, *, api_version: str | None = None) -> ModuleType:
        """Return the representative namespace, or NumPy when empty."""
        return super().__array_namespace__(api_version=api_version) if self.members else np

    @property
    def flags(self) -> ArrayFlagsLike:
        """Representative flags, or flags of empty NumPy storage when empty."""
        return self.members[0].flags if self.members else np.empty(0).flags

    def __array_ufunc__(self, ufunc: np.ufunc, method: str, *inputs: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        """Pass every ufunc method through to corresponding members."""
        if method != "__call__" and isinstance(kwargs.get("out"), tuple):
            (output,) = kwargs["out"]
            kwargs = {**kwargs, "out": output}
        if method == "at" and not isinstance(inputs[0], BroadcastTuple):
            msg = "ufunc.at requires a broadcast tuple destination."
            raise TypeError(msg)
        return tuple_function(
            type(self),
            getattr(ufunc, method),
            inputs,
            kwargs,
            mutate=0 if method == "at" else None,
            indexed=method == "at",
            nout=ufunc.nout if method == "__call__" else 1,
        )

    def __array_function__(
        self,
        func: Callable,
        types: tuple[type[Any], ...],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:  # noqa: ANN401
        """Pass backend functions through to corresponding members."""
        del types
        if func is np.shape:
            return self.shape
        if func is np.ndim:
            return self.ndim
        if func is np.size:
            axis = args[1] if len(args) > 1 else kwargs.get("axis")
            return self.size if axis is None else self.shape[axis]
        if func is np.concatenate and len(args) > 2:
            kwargs = {**kwargs, "out": args[2]}
            args = args[:2]
        destination = _MUTATING_FUNCTIONS.get(func)
        return tuple_function(
            type(self),
            func,
            args,
            kwargs,
            mutate=(0 if args else destination) if destination is not None else None,
            indexed=destination is not None and func is not np.copyto,
            nout=1 if func in _EMPTY_ARRAY_FUNCTIONS else None,
        )

    def reshape(self, *shape: int | Sequence[int], order: Literal["C", "F", "A"] = "C") -> Self:
        """Reshape the common logical batch, preserving member representations."""
        target = shape[0] if len(shape) == 1 and not isinstance(shape[0], int) else shape
        return cast("Self", np.reshape((self), cast("Any", target), order=order))

    def squeeze(self, axis: int | tuple[int, ...] | None = None) -> Self:
        """Remove singleton logical batch axes."""
        return np.squeeze(cast("Any", self), axis=axis)
