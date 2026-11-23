"""PyTorch broadcast tuples with opaque heterogeneous members."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Self, cast, overload

import numpy as np
import torch

from ndlike.torch_like import TorchLikeImplementation, _broadcast_to_args, to_torch_like
from ndlike.tuple._common import BroadcastTuple, tuple_function

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from types import ModuleType

    from ndlike.torch_like import AnyTorchLike


# Only functions with a fixed, single-array result can operate without members.
_EMPTY_ARRAY_FUNCTIONS = frozenset(
    (
        torch.reshape,
        torch.transpose,
        torch.permute,
        torch.swapaxes,
        torch.movedim,
        torch.squeeze,
        torch.unsqueeze,
        torch.broadcast_to,
        torch.stack,
        torch.cat,
        torch.concatenate,
        torch.clone,
        torch.conj,
        torch.adjoint,
        torch.ravel,
        torch.flatten,
        torch.sum,
        torch.prod,
        torch.mean,
        torch.std,
        torch.var,
        torch.all,
        torch.any,
        torch.add,
        torch.sub,
        torch.mul,
        torch.true_divide,
        torch.floor_divide,
        torch.remainder,
        torch.pow,
        torch.matmul,
        torch.bitwise_and,
        torch.bitwise_or,
        torch.bitwise_xor,
        torch.bitwise_not,
        torch.bitwise_left_shift,
        torch.bitwise_right_shift,
        torch.eq,
        torch.ne,
        torch.lt,
        torch.le,
        torch.gt,
        torch.ge,
        torch.neg,
        torch.positive,
        torch.abs,
    )
)


class TorchBroadcastTuple[Members: tuple[AnyTorchLike | torch.Tensor, ...]](
    BroadcastTuple[Members],
    TorchLikeImplementation[
        "TorchBroadcastTuple[Members]",
        object,
        "TorchBroadcastTuple[Members]",
        object,
        "TorchBroadcastTuple[Members]",
        "TorchBroadcastTuple[tuple[Any, ...]]",
    ],
):
    """A collection exposing the broadcast shape of its Torch-like members.

    Functions pass through to corresponding members, preserving native dispatch
    and gradients. Use ``members`` to access stored representations, and indexing
    to select or update the logical batch.
    """

    def __torch_like__(
        self,
        dtype: torch.dtype | None = None,
        /,
        *,
        device: torch.device | str | None = None,
        copy: bool = False,
    ) -> Self:
        """Return self by default, forwarding explicit conversion options to members."""
        if dtype is None and device is None and not copy:
            return self
        return self._map_members(lambda member: to_torch_like(member, dtype, device=device, copy=copy))

    def __post_init__(self) -> None:
        """Require native or dispatched members so shaping cannot coerce representations."""
        super().__post_init__()
        for member in self.members:
            if not isinstance(member, torch.Tensor) and not callable(getattr(type(member), "__torch_function__", None)):
                msg = "Torch tuple members must be tensors or implement __torch_function__."
                raise TypeError(msg)

    @staticmethod
    def _broadcast_member[V: AnyTorchLike | torch.Tensor](member: V, shape: tuple[int, ...]) -> V:
        return cast("V", torch.broadcast_to(cast("Any", member), shape))

    @staticmethod
    def _is_scalar_integer(value: object) -> bool:
        return (
            isinstance(value, torch.Tensor)
            and value.ndim == 0
            and value.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)
        ) or (isinstance(value, np.ndarray) and value.ndim == 0 and np.issubdtype(value.dtype, np.integer))

    def __array_namespace__(self, /, *, api_version: str | None = None) -> ModuleType:
        """Return the representative member's namespace when available."""
        if not self.members:
            return torch
        namespace = getattr(self.members[0], "__array_namespace__", None)
        if namespace is None:
            msg = "Underlying tensor does not support __array_namespace__."
            raise NotImplementedError(msg)
        return namespace(api_version=api_version)

    @property
    def mT(self) -> Self:  # noqa: N802
        """Swap the last two logical batch axes."""
        if self.members and self.ndim < 2:
            msg = "mT requires at least two dimensions."
            raise ValueError(msg)
        return cast("Self", self.transpose(-2, -1))

    @property
    def mH(self) -> Self:  # noqa: N802
        """Conjugate and transpose the last two logical batch axes."""
        return cast("Self", torch.adjoint(cast("Any", self)))

    def numpy(self, *, force: bool = False) -> np.ndarray:
        """Reject conversion of multiple representations into one array."""
        del force
        return self.__array__()

    def to(self, *args: Any, **kwargs: Any) -> Self:  # noqa: ANN401
        """Move or cast each member through its own implementation."""
        return self._map_members(lambda member: member.to(*args, **kwargs))

    def detach(self) -> Self:
        """Detach each member without expanding compact storage."""
        return self._map_members(lambda member: member.detach())

    def reshape(self, *shape: int | tuple[int, ...]) -> Self:
        """Reshape the common logical batch."""
        target = shape[0] if len(shape) == 1 and isinstance(shape[0], tuple) else shape
        return cast("Self", torch.reshape(cast("Any", self), cast("Any", target)))

    def squeeze(self, dim: int | tuple[int, ...] | list[int] | None = None) -> Self:
        """Remove singleton logical batch dimensions."""
        return (
            cast("Self", torch.squeeze(cast("Any", self), dim=dim))
            if dim is not None
            else cast("Self", torch.squeeze(cast("Any", self)))
        )

    def unsqueeze(self, dim: int) -> Self:
        """Insert a singleton logical batch dimension at ``dim``."""
        return cast("Self", torch.unsqueeze(cast("Any", self), dim))

    @overload
    def movedim(self, source: int, destination: int) -> Self: ...

    @overload
    def movedim(self, source: Sequence[int], destination: Sequence[int]) -> Self: ...

    def movedim(self, source: int | Sequence[int], destination: int | Sequence[int]) -> Self:
        """Move logical batch dimensions from ``source`` to ``destination``."""
        return torch.movedim(self, source, destination)  # ty: ignore[no-matching-overload]

    @overload
    def broadcast_to(self, size: Sequence[int]) -> Self: ...

    @overload
    def broadcast_to(self, dim: int, /, *dims: int) -> Self: ...

    def broadcast_to(self, *args: int | Sequence[int], **kwargs: Sequence[int]) -> Self:
        """Broadcast the common logical batch to ``size``."""
        return torch.broadcast_to(self, *_broadcast_to_args(args), **kwargs)  # ty: ignore[invalid-explicit-return, invalid-argument-type]

    @classmethod
    def __torch_function__(
        cls,
        func: Callable,
        types: tuple[type[Any], ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:  # noqa: ANN401
        """Pass backend functions through to corresponding members."""
        del types
        if func is torch.numel:
            value = args[0] if args else (kwargs or {})["input"]
            return math.prod(value.shape)
        return tuple_function(
            cls,
            func,
            args,
            {} if kwargs is None else kwargs,
            nout=1 if func in _EMPTY_ARRAY_FUNCTIONS else None,
        )
