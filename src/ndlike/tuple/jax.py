"""JAX broadcast tuples with dynamic, opaque array-like members."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Self, overload

import jax
from jax import numpy as jnp
import numpy as np

from ndlike import jax_functions as jf
from ndlike.jax_like import AnyJaxLike, JaxLikeImplementation, _jax_validate_dynamic_field, to_jax_like
from ndlike.tuple._common import BroadcastTuple, _rebuild_sequence, tuple_function

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence
    from types import ModuleType
    from typing import Protocol

    from jax.sharding import Sharding
    from jax.typing import ArrayLike as JTArrayLike, DTypeLike

    from ndlike.array_like import ToIndices
    from ndlike.jax_like import JaxLike
    from ndlike.tuple._common import NestedPreservingTuple, PreservingMembers, ShapePreserving

    # The bound also makes these members valid JaxBroadcastTuple inputs.
    class _JaxPreservingLeaf(JaxLike[Any, Any, Any, Any, Any, Any, Any], ShapePreserving, Protocol):
        """A JAX-like member that preserves its representation under shaping."""

    class _JaxPreservingTuple(JaxLike[Any, Any, Any, Any, Any, Any, Any], NestedPreservingTuple, Protocol):
        """A nested JAX tuple whose member types survive shaping."""

    type JaxShapePreservingMember = _JaxPreservingLeaf | _JaxPreservingTuple


# Known array results only; populated tuples still forward arbitrary functions.
_EMPTY_OUTPUTS = {
    func: 1
    for name in (
        "reshape",
        "transpose",
        "matrix_transpose",
        "swapaxes",
        "moveaxis",
        "squeeze",
        "expand_dims",
        "broadcast_to",
        "stack",
        "concatenate",
        "copy",
        "astype",
        "take_along_axis",
        "sum",
        "mean",
        "std",
        "var",
        "add",
        "subtract",
        "multiply",
        "true_divide",
        "floor_divide",
        "remainder",
        "power",
        "matmul",
        "bitwise_and",
        "bitwise_or",
        "bitwise_xor",
        "left_shift",
        "right_shift",
        "invert",
        "equal",
        "not_equal",
        "less",
        "less_equal",
        "greater",
        "greater_equal",
        "negative",
        "positive",
        "absolute",
        "conj",
    )
    for func in (getattr(jf, f"jax_{name}"), getattr(jnp, name))
}
_EMPTY_OUTPUTS.update(
    {func: 2 for name in ("divmod", "frexp", "modf") for func in (getattr(jf, f"jax_{name}"), getattr(jnp, name))}
)


type JaxMember = AnyJaxLike | jax.Array


class JaxBroadcastTuple[Members: tuple[JaxMember, ...]](
    BroadcastTuple[Members],
    JaxLikeImplementation[
        "JaxBroadcastTuple[Members]",
        object,
        "JaxBroadcastTuple[Members]",
        object,
        "JaxBroadcastTuple[tuple[JaxMember, ...]]",
        "JaxBroadcastTuple[tuple[Any, ...]]",
    ],
):
    """A collection exposing the broadcast shape of its JAX-like members.

    Members remain dynamic pytree children. Native vmap maps stored member axes,
    not implicit logical broadcasts. Operations use ndlike's JAX wrappers and
    pass through to members. Indexed updates delegate through each member's .at.
    """

    pytree_children: ClassVar[tuple[str, ...]] = ("members",)

    def tree_flatten(self) -> tuple[tuple[Any, ...], Any]:
        """Expose members individually, independent of container pytree registration."""
        children, layout = super().tree_flatten()
        if type(self.members) is not tuple and not hasattr(self.members, "_fields"):
            _jax_validate_dynamic_field(tuple(self.members), f"{type(self).__name__}.members")
        index = layout[0].index("members")
        return (
            (*children[:index], *self.members, *children[index + 1 :]),
            (layout, type(self.members), index, len(self.members)),
        )

    @classmethod
    def tree_unflatten(cls, aux_data: Any, children: tuple[Any, ...]) -> Self:  # noqa: ANN401
        """Restore the member container and owner metadata without owner initialization."""
        layout, member_type, index, count = aux_data
        members = _rebuild_sequence(member_type, children[index : index + count])
        return super().tree_unflatten(layout, (*children[:index], members, *children[index + count :]))

    def __jax_like__(
        self,
        dtype: DTypeLike | None = None,
        /,
        *,
        device: jax.Device | Sharding | None = None,
        copy: bool = False,
    ) -> Self:
        """Return self by default, forwarding explicit conversion options to members."""
        if dtype is None and device is None and not copy:
            return self
        return self._map_members(lambda member: to_jax_like(member, dtype, device=device, copy=copy))

    def __post_init__(self) -> None:
        """Require each member to be a native array or support backend dispatch."""
        super().__post_init__()
        for member in self.members:
            if not isinstance(member, jax.Array) and not callable(getattr(type(member), "__jax_function__", None)):
                msg = "JAX tuple members must be JAX arrays or implement __jax_function__."
                raise TypeError(msg)

    @staticmethod
    def _broadcast_member(member: JaxMember, shape: tuple[int, ...]) -> JaxMember:
        return jf.jax_broadcast_to(member, shape)

    @staticmethod
    def _is_scalar_integer(value: object) -> bool:
        return isinstance(value, (jax.Array, np.ndarray)) and value.ndim == 0 and np.issubdtype(value.dtype, np.integer)

    @property
    def weak_type(self) -> bool | None:
        """First member's weak typing flag, or None if unavailable or empty."""
        return getattr(self.members[0], "weak_type", None) if self.members else None

    @property
    def sharding(self) -> Any:  # noqa: ANN401
        """Representative sharding, or None when empty."""
        return self.members[0].sharding if self.members else None

    def devices(self) -> set[jax.Device]:
        """Representative devices, or an empty set when empty."""
        return self.members[0].devices() if self.members else set()

    def __array_namespace__(self, /, *, api_version: str | None = None) -> ModuleType:
        """Return the representative namespace, or JAX NumPy when empty."""
        return super().__array_namespace__(api_version=api_version) if self.members else jnp

    @property
    def at(self) -> JaxTupleIndexUpdateHelper[Self]:
        """Access each member's native indexed reads and updates."""
        return JaxTupleIndexUpdateHelper(self)

    def __setitem__(self, index: ToIndices, value: object, /) -> None:
        """Use out-of-place JAX indexed updates instead of assignment."""
        del index, value
        msg = "JAX arrays are immutable; use .at[index].set(value)."
        raise TypeError(msg)

    def _map_array_fields(self, op: Callable[[Any], Any]) -> Self:
        return self._map_members(op)

    @overload
    def take_along_axis(
        self: PreservingMembers, indices: JTArrayLike, axis: int | None = -1
    ) -> JaxBroadcastTuple[Members]: ...

    @overload
    def take_along_axis(
        self, indices: JTArrayLike | JaxBroadcastTuple[Any], axis: int | None = -1
    ) -> JaxBroadcastTuple[tuple[JaxMember, ...]]: ...

    def take_along_axis(
        self, indices: JTArrayLike | JaxBroadcastTuple[Any], axis: int | None = -1
    ) -> JaxBroadcastTuple[Any]:
        """Gather values along a logical batch axis."""
        return jf.jax_take_along_axis(self, indices, axis)

    if TYPE_CHECKING:

        def _ndlike_tuple_owner(self) -> JaxBroadcastTuple[Members]:
            """Identify the container rebuilt by memberwise shaping."""
            ...

        @overload
        def __getitem__[M: tuple[JaxShapePreservingMember, ...]](
            self: JaxBroadcastTuple[M], index: ToIndices, /
        ) -> JaxBroadcastTuple[M]: ...
        @overload
        def __getitem__(self, index: ToIndices, /) -> JaxBroadcastTuple[tuple[JaxMember, ...]]: ...
        def __getitem__(self, index: ToIndices, /) -> JaxBroadcastTuple[Any]:
            """Index the logical batch."""
            ...

        @overload
        def __iter__(self: PreservingMembers) -> Iterator[JaxBroadcastTuple[Members]]: ...
        @overload
        def __iter__(self) -> Iterator[JaxBroadcastTuple[tuple[JaxMember, ...]]]: ...
        def __iter__(self) -> Iterator[JaxBroadcastTuple[Any]]:
            """Iterate over the logical batch."""
            ...

        @overload
        def reshape(
            self: PreservingMembers, *args: int | Sequence[int], order: str = "C"
        ) -> JaxBroadcastTuple[Members]: ...
        @overload
        def reshape(self, *args: int | Sequence[int], order: str = "C") -> JaxBroadcastTuple[tuple[JaxMember, ...]]: ...
        def reshape(self, *args: int | Sequence[int], order: str = "C") -> JaxBroadcastTuple[Any]:
            """Reshape the logical batch."""
            ...

        @overload
        def transpose(self: PreservingMembers, *args: int | Sequence[int] | None) -> JaxBroadcastTuple[Members]: ...
        @overload
        def transpose(self, *args: int | Sequence[int] | None) -> JaxBroadcastTuple[tuple[JaxMember, ...]]: ...
        def transpose(self, *args: int | Sequence[int] | None) -> JaxBroadcastTuple[Any]:
            """Transpose the logical batch axes."""
            ...

        @overload
        def swapaxes(self: PreservingMembers, axis1: int, axis2: int) -> JaxBroadcastTuple[Members]: ...
        @overload
        def swapaxes(self, axis1: int, axis2: int) -> JaxBroadcastTuple[tuple[JaxMember, ...]]: ...
        def swapaxes(self, axis1: int, axis2: int) -> JaxBroadcastTuple[Any]:
            """Swap two logical axes."""
            ...

        @overload
        def squeeze(self: PreservingMembers, axis: int | Sequence[int] | None = None) -> JaxBroadcastTuple[Members]: ...
        @overload
        def squeeze(self, axis: int | Sequence[int] | None = None) -> JaxBroadcastTuple[tuple[JaxMember, ...]]: ...
        def squeeze(self, axis: int | Sequence[int] | None = None) -> JaxBroadcastTuple[Any]:
            """Remove singleton logical axes."""
            ...

        @overload
        def ravel(self: PreservingMembers, order: str = "C") -> JaxBroadcastTuple[Members]: ...
        @overload
        def ravel(self, order: str = "C") -> JaxBroadcastTuple[tuple[JaxMember, ...]]: ...
        def ravel(self, order: str = "C") -> JaxBroadcastTuple[Any]:
            """Flatten the logical batch."""
            ...

        @overload
        def flatten(self: PreservingMembers, order: str = "C") -> JaxBroadcastTuple[Members]: ...
        @overload
        def flatten(self, order: str = "C") -> JaxBroadcastTuple[tuple[JaxMember, ...]]: ...
        def flatten(self, order: str = "C") -> JaxBroadcastTuple[Any]:
            """Flatten the logical batch."""
            ...

    @classmethod
    def __jax_function__(
        cls,
        func: Callable,
        types: tuple[type[Any], ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:  # noqa: ANN401
        """Pass backend functions through to corresponding members."""
        del types
        kwargs = {} if kwargs is None else kwargs
        metadata_func = getattr(func, "__wrapped__", func)
        if metadata_func in (jnp.shape, jnp.ndim, jnp.size):
            value = args[0] if args else kwargs["a"]
            if metadata_func is jnp.shape:
                return value.shape
            if metadata_func is jnp.ndim:
                return value.ndim
            axis = args[1] if len(args) > 1 else kwargs.get("axis")
            return value.size if axis is None else value.shape[axis]
        return tuple_function(cls, func, args, kwargs, nout=_EMPTY_OUTPUTS.get(func, _EMPTY_OUTPUTS.get(metadata_func)))


@dataclass(frozen=True)
class JaxTupleIndexUpdateHelper[T: JaxBroadcastTuple[Any]]:
    """A tuple owner's native-style indexed update interface."""

    owner: T

    def __getitem__(self, index: ToIndices, /) -> JaxTupleIndexUpdateRef[T]:
        """Bind a logical batch index."""
        return JaxTupleIndexUpdateRef(self.owner, index)


@dataclass(frozen=True)
class JaxTupleIndexUpdateRef[T: JaxBroadcastTuple[Any]]:
    """Forward indexed operations and their options to each JAX-like member."""

    owner: T
    index: ToIndices

    @overload
    def get[M: tuple[JaxShapePreservingMember, ...]](
        self: JaxTupleIndexUpdateRef[JaxBroadcastTuple[M]],
        **kwargs: Any,  # noqa: ANN401
    ) -> JaxBroadcastTuple[M]: ...
    @overload
    def get(self, **kwargs: Any) -> JaxBroadcastTuple[tuple[JaxMember, ...]]: ...  # noqa: ANN401

    def get(self, **kwargs: Any) -> JaxBroadcastTuple[Any]:
        """Read compactly by default, or delegate native get options when supplied."""
        if not kwargs:
            return self.owner[self.index]
        return tuple_function(
            type(self.owner),
            lambda member, index: member.at[index].get(**kwargs),
            (self.owner, self.index),
            {},
            nout=1,
        )

    def _update(self, method: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> T:
        def call(member: Any, index: Any, *values: Any) -> Any:  # noqa: ANN401
            return getattr(member.at[index], method)(*values, **kwargs)

        return tuple_function(
            type(self.owner),
            call,
            (self.owner, self.index, *args),
            {},
            nout=1
            if method in ("set", "add", "subtract", "multiply", "divide", "power", "min", "max", "apply")
            else None,
        )

    def set(self, value: object, **kwargs: Any) -> T:  # noqa: ANN401
        """Pass indexed set through to each member."""
        return self._update("set", (value,), kwargs)

    def add(self, value: object, **kwargs: Any) -> T:  # noqa: ANN401
        """Pass indexed add through to each member."""
        return self._update("add", (value,), kwargs)

    def subtract(self, value: object, **kwargs: Any) -> T:  # noqa: ANN401
        """Pass indexed subtract through to each member."""
        return self._update("subtract", (value,), kwargs)

    def multiply(self, value: object, **kwargs: Any) -> T:  # noqa: ANN401
        """Pass indexed multiply through to each member."""
        return self._update("multiply", (value,), kwargs)

    def divide(self, value: object, **kwargs: Any) -> T:  # noqa: ANN401
        """Pass indexed divide through to each member."""
        return self._update("divide", (value,), kwargs)

    def power(self, value: object, **kwargs: Any) -> T:  # noqa: ANN401
        """Pass indexed power through to each member."""
        return self._update("power", (value,), kwargs)

    def min(self, value: object, **kwargs: Any) -> T:  # noqa: ANN401
        """Pass indexed min through to each member."""
        return self._update("min", (value,), kwargs)

    def max(self, value: object, **kwargs: Any) -> T:  # noqa: ANN401
        """Pass indexed max through to each member."""
        return self._update("max", (value,), kwargs)

    def __getattr__(self, method: str) -> Callable[..., T]:
        """Forward additional native methods, such as apply, without an allowlist."""
        if method.startswith("_"):
            raise AttributeError(method)

        def update(*args: Any, **kwargs: Any) -> T:  # noqa: ANN401
            return self._update(method, args, kwargs)

        return update
