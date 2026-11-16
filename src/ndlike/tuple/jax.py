"""JAX broadcast tuples with dynamic, opaque array-like members."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Self, cast

import jax
from jax import numpy as jnp
import numpy as np

from ndlike import jax_functions as jf
from ndlike.jax_like import JaxLikeImplementation, _jax_validate_dynamic_field, to_jax_like
from ndlike.tuple._common import BroadcastTuple, _rebuild_sequence, tuple_function

if TYPE_CHECKING:
    from collections.abc import Callable

    from jax.sharding import Sharding
    from jax.typing import DTypeLike

    from ndlike.array_like import ToIndices
    from ndlike.jax_like import AnyJaxLike


class JaxBroadcastTuple[Members: tuple[AnyJaxLike | jax.Array, ...]](
    BroadcastTuple[Members],
    JaxLikeImplementation[
        "JaxBroadcastTuple[Members]",
        object,
        "JaxBroadcastTuple[Members]",
        object,
        "JaxBroadcastTuple[Members]",
        "JaxBroadcastTuple[tuple[Any, ...]]",
    ],
):
    """A nonempty collection exposing the broadcast shape of its JAX-like members.

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
        """Require native or dispatched members so shaping cannot coerce representations."""
        super().__post_init__()
        for member in self.members:
            if not isinstance(member, jax.Array) and not callable(getattr(type(member), "__jax_function__", None)):
                msg = "JAX tuple members must be JAX arrays or implement __jax_function__."
                raise TypeError(msg)

    @staticmethod
    def _broadcast_member[V: AnyJaxLike | jax.Array](member: V, shape: tuple[int, ...]) -> V:
        return cast("V", jf.jax_broadcast_to(member, shape))

    @staticmethod
    def _is_scalar_integer(value: object) -> bool:
        return isinstance(value, (jax.Array, np.ndarray)) and value.ndim == 0 and np.issubdtype(value.dtype, np.integer)

    @property
    def sharding(self) -> Any:  # noqa: ANN401
        """Sharding of the representative first member."""
        return self.members[0].sharding

    def devices(self) -> set[jax.Device]:
        """Devices of the representative first member."""
        return self.members[0].devices()

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
        return tuple_function(cls, func, args, kwargs)


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

    def get(self, **kwargs: Any) -> T:  # noqa: ANN401
        """Read compactly by default, or delegate native get options when supplied."""
        if not kwargs:
            return self.owner[self.index]
        return tuple_function(
            type(self.owner),
            lambda member, index: member.at[index].get(**kwargs),
            (self.owner, self.index),
            {},
        )

    def _update(self, method: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> T:
        def call(member: Any, index: Any, *values: Any) -> Any:  # noqa: ANN401
            return getattr(member.at[index], method)(*values, **kwargs)

        return tuple_function(type(self.owner), call, (self.owner, self.index, *args), {})

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
