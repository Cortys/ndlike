"""Shared storage and memberwise operation dispatch for broadcast tuples."""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from contextlib import suppress
from dataclasses import dataclass, replace
from inspect import signature
import operator
from typing import TYPE_CHECKING, Any, Self

import numpy as np

from ndlike._batch import is_basic_index, normalize_broadcast_index

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator
    from types import ModuleType

    import jax
    from jax.sharding import Sharding
    from jax.typing import DTypeLike as JaxDTypeLike
    from numpy.typing import DTypeLike
    import torch

    from ndlike.array_like import ToIndices
    from ndlike.tuple.jax import JaxBroadcastTuple
    from ndlike.tuple.numpy import NumpyBroadcastTuple
    from ndlike.tuple.torch import TorchBroadcastTuple


@dataclass(frozen=True, eq=False, slots=True, weakref_slot=True)
class BroadcastTuple[Members: tuple[Any, ...]](metaclass=ABCMeta):
    """Tuple of opaque array-like members sharing a broadcast shape.

    Access representation members through ``members``. Indexing and iteration
    address the logical batch, not the member positions. The first member is
    representative for dtype, device, and namespace metadata.
    Empty tuples have neutral shape ``()`` and no members to constrain indexing.
    The type parameter describes the entire member tuple: ``tuple[A, B]``
    for fixed positions or ``tuple[T, ...]`` for arbitrary length.
    """

    members: Members

    def __post_init__(self) -> None:
        """Validate structure and shapes without coercing or evaluating members."""
        if not isinstance(self.members, tuple):
            msg = "members must be a tuple of array-like objects."
            raise TypeError(msg)
        for member in self.members:
            if not hasattr(member, "shape") or not hasattr(member, "ndim"):
                msg = "Each member must expose shape and ndim."
                raise TypeError(msg)
        # Check compatibility, but derive shape afresh after pytree transforms.
        np.broadcast_shapes(*(member.shape for member in self.members))

    def with_members(self, members: Members) -> Self:
        """Rebuild the concrete owner, preserving any additional dataclass fields."""
        return replace(self, members=members)

    def __array_like__(
        self, dtype: DTypeLike | None = None, /, *, copy: bool | None = None
    ) -> NumpyBroadcastTuple[Any]:
        """Convert stored members to NumPy-like representations without broadcasting."""
        from ndlike.array_like import to_numpy_like  # noqa: PLC0415 - avoid circular backend imports
        from ndlike.tuple.numpy import NumpyBroadcastTuple  # noqa: PLC0415

        return NumpyBroadcastTuple(
            _rebuild_sequence(
                type(self.members), (to_numpy_like(member, dtype=dtype, copy=copy) for member in self.members)
            )
        )

    def __jax_like__(
        self,
        dtype: JaxDTypeLike | None = None,
        /,
        *,
        device: jax.Device | Sharding | None = None,
        copy: bool = False,
    ) -> JaxBroadcastTuple[Any]:
        """Convert stored members to JAX-like representations, loading JAX on demand."""
        from ndlike.jax_like import to_jax_like  # noqa: PLC0415 - optional backend
        from ndlike.tuple.jax import JaxBroadcastTuple  # noqa: PLC0415

        return JaxBroadcastTuple(
            _rebuild_sequence(
                type(self.members), (to_jax_like(member, dtype, device=device, copy=copy) for member in self.members)
            )
        )

    def __torch_like__(
        self,
        dtype: torch.dtype | None = None,
        /,
        *,
        device: torch.device | str | None = None,
        copy: bool = False,
    ) -> TorchBroadcastTuple[Any]:
        """Convert stored members to Torch-like representations, loading Torch on demand."""
        from ndlike.torch_like import to_torch_like  # noqa: PLC0415 - optional backend
        from ndlike.tuple.torch import TorchBroadcastTuple  # noqa: PLC0415

        return TorchBroadcastTuple(
            _rebuild_sequence(
                type(self.members), (to_torch_like(member, dtype, device=device, copy=copy) for member in self.members)
            )
        )

    @property
    def shape(self) -> tuple[int, ...]:
        """The broadcast of all stored member shapes."""
        return np.broadcast_shapes(*(member.shape for member in self.members))

    @property
    def dtype(self) -> Any:  # noqa: ANN401
        """Data type of the first member; members are never promoted together."""
        return self.members[0].dtype if self.members else None

    @property
    def device(self) -> Any:  # noqa: ANN401
        """Device of the first member, or None when empty."""
        return self.members[0].device if self.members else None

    def __array_namespace__(self, /, *, api_version: str | None = None) -> ModuleType:
        """Return the first member's array namespace."""
        return self.members[0].__array_namespace__(api_version=api_version)

    def __array__(self, dtype: DTypeLike | None = None, /, *, copy: bool | None = None) -> np.ndarray:
        """Reject implicit conversion of heterogeneous representations."""
        del dtype, copy
        msg = "A broadcast tuple cannot be coerced to one array; access .members explicitly."
        raise TypeError(msg)

    @staticmethod
    @abstractmethod
    def _broadcast_member[V](member: V, shape: tuple[int, ...]) -> V:
        """Dispatch a broadcast to a member's own representation."""

    @staticmethod
    @abstractmethod
    def _is_scalar_integer(value: object) -> bool:
        """Recognize backend scalar integer indices without reading their value."""

    def _expanded_members(self) -> Members:
        shape = self.shape
        return _rebuild_sequence(
            type(self.members),
            (
                member if tuple(member.shape) == shape else self._broadcast_member(member, shape)
                for member in self.members
            ),
        )

    def __getitem__(self, index: ToIndices, /) -> Self:
        """Index the logical batch while retaining compact basic-index storage."""
        if not self.members:
            return self.with_members(self.members)
        indices = index if isinstance(index, tuple) else (index,)
        if any(isinstance(item, BroadcastTuple) for item in indices):
            return tuple_function(type(self), operator.getitem, (self, index), {}, nout=1)
        shape = self.shape
        basic = is_basic_index(index, self._is_scalar_integer)
        results: list[Any] = []
        for member in self.members:
            value = member
            member_shape = tuple(value.shape)
            if basic:
                member_index = normalize_broadcast_index(index, member_shape, shape)
            else:
                member_index = index
                if member_shape != shape:
                    value = self._broadcast_member(member, shape)
            result = value[member_index]
            if isinstance(value, np.ndarray) and not hasattr(result, "shape"):
                # Object-array indexing returns the contained Python object.
                # Restore a scalar array without invoking that object's conversion.
                scalar = np.empty((), dtype=value.dtype)
                scalar[()] = result
                result = scalar
            results.append(result)
        return self.with_members(_rebuild_sequence(type(self.members), results))

    def __setitem__(self, index: ToIndices, value: object, /) -> None:
        """Assign corresponding members directly to their existing storage."""
        tuple_function(type(self), operator.setitem, (self, index, value), {}, mutate=0, indexed=True)

    def __len__(self) -> int:
        """Return the length of the first logical batch dimension."""
        if not self.shape:
            msg = "len() of unsized object"
            raise TypeError(msg)
        return self.shape[0]

    def __iter__(self) -> Iterator[Self]:
        """Iterate over the logical batch, retaining all members in each result."""
        for index in range(len(self)):
            yield self[index]

    def _map_members(self, op: Callable[[Any], Any]) -> Self:
        return self.with_members(_rebuild_sequence(type(self.members), (op(member) for member in self.members)))


def _rebuild_sequence[T: tuple[Any, ...] | list[Any]](cls: type[T], values: Iterable[Any]) -> T:
    """Reconstruct using the container's sequence or named-tuple constructor."""
    items = tuple(values)
    if hasattr(cls, "_fields"):
        return cls(*items)
    return cls(items)


def _tuple_operands(value: object) -> Iterator[BroadcastTuple]:
    """Find tuple operands in call arguments without traversing their members."""
    if isinstance(value, BroadcastTuple):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _tuple_operands(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _tuple_operands(item)


def tuple_function(  # noqa: C901, PLR0912, PLR0915
    owner: type[BroadcastTuple],
    func: Callable,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    mutate: int | str | None = None,
    indexed: bool = False,
    nout: int | None = None,
) -> Any:  # noqa: ANN401
    """Pass a backend operation through to corresponding opaque members.

    Argument sequences are traversed to support stack/concatenate. A broadcast
    tuple itself is always opaque; its member representations are never traversed.
    Empty calls require known output arity via nout (or an explicit mutation).
    They never evaluate func to infer an arbitrary result container.
    """
    out = kwargs.get("out")
    out_position = None
    if "out" not in kwargs:
        # Some native backend functions do not expose a Python signature.
        with suppress(TypeError, ValueError):
            parameters = signature(func)
            out = parameters.bind(*args, **kwargs).arguments.get("out")
            positional = [
                name
                for name, param in parameters.parameters.items()
                if param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD)
            ]
            if "out" in positional and positional.index("out") < len(args):
                out_position = positional.index("out")
    outputs = out if isinstance(out, tuple) else (out,)
    if any(value is not None and not isinstance(value, BroadcastTuple) for value in outputs):
        msg = "Output destinations must be broadcast tuples, one per result."
        raise TypeError(msg)
    operands = {id(value): value for value in _tuple_operands((args, kwargs))}
    if not operands or any(type(value) is not owner for value in operands.values()):
        return NotImplemented
    template = next(iter(operands.values()))
    count = len(template.members)
    if any(len(value.members) != count for value in operands.values()):
        msg = "Broadcast tuple operands must have the same number of members."
        raise ValueError(msg)
    members = {}

    def substitute(value: Any, index: int, *, destination: bool = False) -> Any:  # noqa: ANN401
        if isinstance(value, BroadcastTuple):
            if destination:
                return value.members[index]
            if id(value) not in members:
                members[id(value)] = value._expanded_members()  # noqa: SLF001
            return members[id(value)][index]
        if isinstance(value, (tuple, list)):
            return _rebuild_sequence(type(value), (substitute(item, index, destination=destination) for item in value))
        if isinstance(value, dict):
            return {key: substitute(item, index, destination=destination) for key, item in value.items()}
        return value

    target = args[mutate] if isinstance(mutate, int) else kwargs[mutate] if isinstance(mutate, str) else None
    if mutate is not None and not isinstance(target, BroadcastTuple):
        msg = "Mutation requires a broadcast tuple destination."
        raise TypeError(msg)
    if not count and nout is None and mutate is None:
        msg = "This operation has no known result structure for an empty broadcast tuple."
        raise NotImplementedError(msg)
    target_shape = target.shape if indexed and isinstance(target, BroadcastTuple) else None
    results = []
    for index in range(count):
        if target is not None and target_shape is not None and tuple(target.members[index].shape) != target_shape:
            msg = "Indexed writes require fully expanded destination members."
            raise ValueError(msg)
        results.append(
            func(
                *(
                    substitute(value, index, destination=position in (mutate, out_position))
                    for position, value in enumerate(args)
                ),
                **{key: substitute(value, index, destination=key in ("out", mutate)) for key, value in kwargs.items()},
            )
        )
    if out is not None:
        if not isinstance(out, tuple):
            return out
        if all(value is not None for value in out):
            return out[0] if len(out) == 1 else out
    if count:
        result = pack_results(template, tuple(results))
    elif mutate is not None:
        result = None
    else:
        result = (
            template.with_members(template.members)
            if nout == 1
            else tuple(template.with_members(template.members) for _ in range(nout or 0))
        )
    if out is None or result is None:
        return result
    if len(out) == 1:
        return result if out[0] is None else out[0]
    return tuple(value if value is not None else result[index] for index, value in enumerate(out))


def pack_results(template: BroadcastTuple, results: tuple[Any, ...]) -> Any:  # noqa: ANN401
    """Rebuild array results and preserve native multi-result container structure.

    Non-array results remain per-member tuples. Array results must satisfy the
    collection's usual common-shape invariant.
    """
    if all(value is None for value in results):
        return None
    if all(hasattr(value, "shape") and hasattr(value, "ndim") for value in results):
        return template.with_members(_rebuild_sequence(type(template.members), results))
    first = results[0]
    if isinstance(first, (tuple, list)) and all(
        type(value) is type(first) and len(value) == len(first) for value in results
    ):
        items = [pack_results(template, tuple(value[index] for value in results)) for index in range(len(first))]
        return _rebuild_sequence(type(first), items)
    if isinstance(first, dict) and all(isinstance(value, dict) and value.keys() == first.keys() for value in results):
        return {key: pack_results(template, tuple(value[key] for value in results)) for key in first}
    return results
