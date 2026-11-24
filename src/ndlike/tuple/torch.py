"""PyTorch broadcast tuples with opaque heterogeneous members."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Self, cast, overload

import numpy as np
import torch

from ndlike.torch_like import AnyTorchLike, TorchLikeImplementation, to_torch_like
from ndlike.tuple._common import BroadcastTuple, tuple_function

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence
    from types import ModuleType

    from ndlike.array_like import ToIndices
    from ndlike.tuple._common import PreservingMembers


# Only functions with a fixed, single-array result can operate without members.
_EMPTY_ARRAY_FUNCTIONS = frozenset(
    (
        torch.reshape,
        torch.transpose,
        torch.permute,
        torch.swapaxes,
        torch.movedim,
        torch.moveaxis,
        torch.squeeze,
        torch.unsqueeze,
        torch.broadcast_to,
        torch.gather,
        torch.stack,
        torch.cat,
        torch.concat,
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


type TorchMember = AnyTorchLike | torch.Tensor


class TorchBroadcastTuple[Members: tuple[TorchMember, ...]](
    BroadcastTuple[Members],
    TorchLikeImplementation[
        "TorchBroadcastTuple[Members]",
        object,
        "TorchBroadcastTuple[Members]",
        object,
        "TorchBroadcastTuple[tuple[TorchMember, ...]]",
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
        """Require each member to be a native array or support backend dispatch."""
        super().__post_init__()
        for member in self.members:
            if not isinstance(member, torch.Tensor) and not callable(getattr(type(member), "__torch_function__", None)):
                msg = "Torch tuple members must be tensors or implement __torch_function__."
                raise TypeError(msg)

    @staticmethod
    def _broadcast_member(member: TorchMember, shape: tuple[int, ...]) -> TorchMember:
        return torch.broadcast_to(cast("Any", member), shape)

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
    def mT(self) -> TorchBroadcastTuple[tuple[TorchMember, ...]]:  # noqa: N802
        """Swap the last two logical batch axes."""
        if self.members and self.ndim < 2:
            msg = "mT requires at least two dimensions."
            raise ValueError(msg)
        return self.transpose(-2, -1)

    @property
    def mH(self) -> TorchBroadcastTuple[tuple[TorchMember, ...]]:  # noqa: N802
        """Conjugate and transpose the last two logical batch axes."""
        return cast("TorchBroadcastTuple[tuple[TorchMember, ...]]", torch.adjoint(cast("Any", self)))

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

    @overload
    def gather(self: PreservingMembers, dim: int, index: torch.Tensor) -> TorchBroadcastTuple[Members]: ...

    @overload
    def gather(
        self, dim: int, index: torch.Tensor | TorchBroadcastTuple[Any]
    ) -> TorchBroadcastTuple[tuple[TorchMember, ...]]: ...

    def gather(self, dim: int, index: torch.Tensor | TorchBroadcastTuple[Any]) -> TorchBroadcastTuple[Any]:
        """Gather values along a logical batch dimension."""
        return cast("TorchBroadcastTuple[Any]", torch.gather(cast("Any", self), dim, cast("Any", index)))

    if TYPE_CHECKING:

        def _ndlike_tuple_owner(self) -> TorchBroadcastTuple[Members]:
            """Identify the container rebuilt by memberwise shaping."""
            ...

        @overload
        def __getitem__(self: PreservingMembers, index: ToIndices, /) -> TorchBroadcastTuple[Members]: ...
        @overload
        def __getitem__(self, index: ToIndices, /) -> TorchBroadcastTuple[tuple[TorchMember, ...]]: ...
        def __getitem__(self, index: ToIndices, /) -> TorchBroadcastTuple[Any]:
            """Index the logical batch."""
            ...

        @overload
        def __iter__(self: PreservingMembers) -> Iterator[TorchBroadcastTuple[Members]]: ...
        @overload
        def __iter__(self) -> Iterator[TorchBroadcastTuple[tuple[TorchMember, ...]]]: ...
        def __iter__(self) -> Iterator[TorchBroadcastTuple[Any]]:
            """Iterate over the logical batch."""
            ...

        @overload
        def reshape(
            self: PreservingMembers, *shape: int | tuple[int, ...] | list[int]
        ) -> TorchBroadcastTuple[Members]: ...
        @overload
        def reshape(
            self, *shape: int | tuple[int, ...] | list[int]
        ) -> TorchBroadcastTuple[tuple[TorchMember, ...]]: ...
        def reshape(self, *shape: int | tuple[int, ...] | list[int]) -> TorchBroadcastTuple[Any]:
            """Reshape the logical batch."""
            ...

        @overload
        def transpose(self: PreservingMembers, dim0: int, dim1: int) -> TorchBroadcastTuple[Members]: ...
        @overload
        def transpose(self, dim0: int, dim1: int) -> TorchBroadcastTuple[tuple[TorchMember, ...]]: ...
        def transpose(self, dim0: int, dim1: int) -> TorchBroadcastTuple[Any]:
            """Transpose the logical batch axes."""
            ...

        @overload
        def permute(
            self: PreservingMembers, *dims: torch.Size | int | tuple[int] | list[int]
        ) -> TorchBroadcastTuple[Members]: ...
        @overload
        def permute(
            self, *dims: torch.Size | int | tuple[int] | list[int]
        ) -> TorchBroadcastTuple[tuple[TorchMember, ...]]: ...
        def permute(self, *dims: torch.Size | int | tuple[int] | list[int]) -> TorchBroadcastTuple[Any]:
            """Permute the logical batch axes."""
            ...

        @overload
        def squeeze(
            self: PreservingMembers, dim: int | tuple[int, ...] | list[int] | None = None
        ) -> TorchBroadcastTuple[Members]: ...
        @overload
        def squeeze(
            self, dim: int | tuple[int, ...] | list[int] | None = None
        ) -> TorchBroadcastTuple[tuple[TorchMember, ...]]: ...
        def squeeze(self, dim: int | tuple[int, ...] | list[int] | None = None) -> TorchBroadcastTuple[Any]:
            """Remove singleton logical axes."""
            ...

        @overload
        def unsqueeze(self: PreservingMembers, dim: int) -> TorchBroadcastTuple[Members]: ...
        @overload
        def unsqueeze(self, dim: int) -> TorchBroadcastTuple[tuple[TorchMember, ...]]: ...
        def unsqueeze(self, dim: int) -> TorchBroadcastTuple[Any]:
            """Insert a singleton logical axis."""
            ...

        @overload
        def movedim(self: PreservingMembers, source: int, destination: int) -> TorchBroadcastTuple[Members]: ...
        @overload
        def movedim(
            self: PreservingMembers, source: Sequence[int], destination: Sequence[int]
        ) -> TorchBroadcastTuple[Members]: ...
        @overload
        def movedim(self, source: int, destination: int) -> TorchBroadcastTuple[tuple[TorchMember, ...]]: ...
        @overload
        def movedim(
            self, source: Sequence[int], destination: Sequence[int]
        ) -> TorchBroadcastTuple[tuple[TorchMember, ...]]: ...
        def movedim(self, source: int | Sequence[int], destination: int | Sequence[int]) -> TorchBroadcastTuple[Any]:
            """Move logical batch axes."""
            ...

        @overload
        def broadcast_to(self: PreservingMembers, size: Sequence[int]) -> TorchBroadcastTuple[Members]: ...
        @overload
        def broadcast_to(self: PreservingMembers, dim: int, /, *dims: int) -> TorchBroadcastTuple[Members]: ...
        @overload
        def broadcast_to(self, size: Sequence[int]) -> TorchBroadcastTuple[tuple[TorchMember, ...]]: ...
        @overload
        def broadcast_to(self, dim: int, /, *dims: int) -> TorchBroadcastTuple[tuple[TorchMember, ...]]: ...
        def broadcast_to(self, *args: int | Sequence[int], **kwargs: Sequence[int]) -> TorchBroadcastTuple[Any]:
            """Broadcast the logical batch."""
            ...

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
