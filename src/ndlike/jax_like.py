# pyright: reportInvalidTypeForm=false
"""Protocols and base classes for JAX-like array objects.

Array operations use the wrappers in :mod:`ndlike.jax_functions` and
``__jax_function__``. Transformations use explicitly declared pytree children.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import suppress
from dataclasses import fields, is_dataclass
from inspect import isabstract
import math
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, Self, override, runtime_checkable

from flextype import flexdispatch
import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class
import numpy as np

from ndlike.array_like import ArrayLike
from ndlike.jax_functions import (
    handle_jax_function,
    has_jax_function,
    jax_astype,
    jax_conj,
    jax_copy,
    jax_matrix_transpose,
    jax_reshape,
    jax_squeeze,
    jax_transpose,
)
from ndlike.jax_operators import JaxOperatorsMixin

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence
    from types import ModuleType

    from jax.sharding import Sharding
    from jax.typing import DTypeLike
    from numpy.typing import ArrayLike as NpArrayLike, DTypeLike as NpDTypeLike, NDArray

    from ndlike.array_like import ToIndices


@runtime_checkable
class JaxLikeConvertible[DT](ArrayLike[DT], Protocol):
    """Protocol for array-like objects that can be converted to JAX arrays."""

    def __jax_like__(
        self,
        dtype: DTypeLike | None = None,
        /,
        *,
        device: jax.Device | Sharding | None = None,
        copy: bool = False,
    ) -> JaxLike[Any]:
        """Convert to a JaxLike."""


@runtime_checkable
class JaxLike[DT](ArrayLike[DT], Protocol):
    """Protocol for array-like objects that implement JAX-specific APIs.

    Every member declared here also exists on a real ``jax.Array``, otherwise
    ``isinstance(jnp.ones(3), JaxLike)`` would silently be False. That rules out
    ``__jax_function__``, which is declared on :class:`JaxLikeImplementation` instead.

    Dispatched operations may return another JaxLike implementation or a native
    array, rather than preserving the concrete wrapper class.
    """

    @property
    def T(self) -> JaxLike[Any] | jax.Array:  # noqa: N802
        """Return a transposed array."""

    @property
    def mT(self) -> JaxLike[Any] | jax.Array:  # noqa: N802
        """Return an array with its last two axes transposed."""

    @property
    def at(self) -> Any:  # noqa: ANN401
        """Indexed update helper, see ``jax.Array.at``."""

    @property
    def sharding(self) -> Any:  # noqa: ANN401
        """The sharding of the underlying array."""

    def devices(self) -> set[jax.Device]:
        """Return the set of devices the array lives on."""

    def block_until_ready(self) -> Self:
        """Block until the asynchronous computation of the array has finished."""

    def astype(
        self,
        dtype: DTypeLike | None,
        copy: bool = False,
        device: jax.Device | Sharding | None = None,
    ) -> Self:
        """Return a copy of the array cast to the given data type."""

    def copy(self) -> JaxLike[Any] | jax.Array:
        """Return a copy of the array."""

    def reshape(self, *args: int | Sequence[int], order: str = "C") -> JaxLike[Any] | jax.Array:
        """Return a reshaped version of the array."""

    def ravel(self, order: str = "C") -> JaxLike[Any] | jax.Array:
        """Return a flattened version of the array."""

    def flatten(self, order: str = "C") -> JaxLike[Any] | jax.Array:
        """Return a flattened version of the array."""

    def squeeze(self, axis: int | Sequence[int] | None = None) -> JaxLike[Any] | jax.Array:
        """Return the array with axes of length one removed."""

    def transpose(self, *args: int | Sequence[int] | None) -> JaxLike[Any] | jax.Array:
        """Return a transposed version of the array."""

    def view(self, dtype: DTypeLike | None = None, type: None = None) -> Self:  # noqa: A002
        """Return a bit-cast view of the array."""

    def item(self, *args: int) -> bool | int | float | complex:
        """Return the array as a Python scalar."""

    def tolist(self) -> Any:  # noqa: ANN401
        """Return the array as a (nested) Python list."""


class JaxLikeImplementation[DT](JaxOperatorsMixin, JaxLike[DT], ABC):
    """ABC implementation for array-like objects that behave like JAX arrays.

    Subclasses implement ``__jax_function__`` plus the representation-specific members.
    Operators and dispatched methods route through :mod:`ndlike.jax_functions` and thus
    back into ``__jax_function__``. Dataclass subclasses retain inherited array equality
    unless they explicitly define ``__eq__``. Hashing follows ordinary dataclass rules.

    Concrete (non-abstract) subclasses are additionally registered as JAX pytree nodes, which makes
    ``jax.jit``, ``jax.vmap``, ``jax.grad`` and ``jax.tree`` work on them. This does not replace the
    override machinery: ``jnp.sum(obj)`` still raises for a registered pytree.

    Default flattening requires an explicit ``pytree_children`` tuple. Other dataclass
    fields are static metadata, which must be immutable, hashable and meaningfully
    comparable, and must not contain JAX/NumPy arrays. Non-dataclasses can declare all
    stored fields via ``_pytree_field_names`` or provide custom flatten/unflatten methods.

    Default traversal rejects NumPy arrays in dynamic fields, including nested
    containers, rather than silently converting them to JAX data. Mixed representations
    remain usable by supported eager operations and opaque :func:`jax_tree_map` calls.
    JAX's own ``float0`` cotangents are exempt from this NumPy-array restriction.

    Reconstruction bypasses initialization and validation. In particular, ``jax.grad``
    packages storage-coordinate cotangents in the original class, even when their values
    violate primal invariants. Domain-specific methods need not make sense on gradients.

    Native tree utilities traverse the representation structurally. :func:`jax_tree_map`
    instead treats implementations as opaque unless ``tree_map_traverse`` is true.

    ``at``, the JAX stand-in for ``__setitem__``, is left to subclasses: an indexed update has no
    single meaning across fields with different batch semantics.

    Representation-specific indexing, metadata, namespace access, ``at``, ``sharding`` and
    ``devices`` are explicitly abstract rather than inherited as protocol placeholders.
    """

    pytree_children: ClassVar[tuple[str, ...]]
    tree_map_traverse: ClassVar[bool] = False

    @abstractmethod
    def __getitem__(self, index: ToIndices, /) -> Any:  # noqa: ANN401
        """Return values selected by index."""

    @abstractmethod
    def __setitem__(self, index: ToIndices, value: NpArrayLike, /) -> None:
        """Set values selected by index."""

    @abstractmethod
    def __array_namespace__(self, /, *, api_version: str | None = None) -> ModuleType:
        """Return the array namespace."""

    @abstractmethod
    def __len__(self) -> int:
        """Return the length along the first axis."""

    @abstractmethod
    def __iter__(self) -> Iterator[Any]:
        """Iterate over the first axis."""

    @property
    @abstractmethod
    def dtype(self) -> Any:  # noqa: ANN401
        """Data type of the array."""

    @property
    @abstractmethod
    def device(self) -> Any:  # noqa: ANN401
        """Device of the array."""

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return len(self.shape)

    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...]:
        """Shape of the array."""

    @property
    def size(self) -> int:
        """Number of elements in the array."""
        return math.prod(self.shape)

    @property
    @abstractmethod
    def at(self) -> Any:  # noqa: ANN401
        """Return the subclass-specific indexed update helper."""

    @property
    @abstractmethod
    def sharding(self) -> Any:  # noqa: ANN401
        """Return the sharding of the underlying array."""

    @abstractmethod
    def devices(self) -> set[jax.Device]:
        """Return the devices containing the underlying array."""

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Preserve inherited equality and register concrete subclasses as JAX pytree nodes.

        Args:
            **kwargs: Keyword arguments forwarded to ``super().__init_subclass__``.
        """
        super().__init_subclass__(**kwargs)
        if "__eq__" not in cls.__dict__:
            inherited_eq = cls.__eq__
            cls.__eq__ = inherited_eq

        if cls is JaxLikeImplementation or isabstract(cls) or ABC in cls.__bases__:
            return

        # Re-registering the same class raises, which happens on module reloads.
        with suppress(ValueError):
            register_pytree_node_class(cls)

    @classmethod
    def _pytree_field_names(cls) -> tuple[str, ...]:
        """Return all stored field names, including static metadata.

        Non-dataclasses must override this or provide custom flattening.

        Raises:
            TypeError: If the class is not a dataclass.
        """
        if is_dataclass(cls):
            return tuple(field.name for field in fields(cls))
        msg = f"{cls.__name__} must be a dataclass or override _pytree_field_names or tree_flatten/tree_unflatten."
        raise TypeError(msg)

    @classmethod
    def _pytree_child_names(cls) -> tuple[str, ...]:
        """Return explicitly declared dynamic field names in traversal order.

        Raises:
            TypeError: If the declaration is missing or not a tuple of field names.
        """
        names = getattr(cls, "pytree_children", None)
        if names is None:
            msg = f"{cls.__name__} must declare pytree_children or override tree_flatten/tree_unflatten."
            raise TypeError(msg)
        if not isinstance(names, tuple) or not all(isinstance(name, str) and name for name in names):
            msg = f"{cls.__name__}.pytree_children must be a tuple of non-empty field names."
            raise TypeError(msg)
        return names

    def tree_flatten(self) -> tuple[tuple[Any, ...], Any]:
        """Split the object into pytree children and static auxiliary data.

        Only fields declared in ``pytree_children`` become children, regardless of
        their values. All other fields become static auxiliary data. Subclasses
        may override this together with :meth:`tree_unflatten` to provide different
        transformation semantics. NumPy array children are rejected by default,
        including for native tree utilities; opaque eager mapping does not flatten.

        Returns:
            The children and the auxiliary data.

        Raises:
            TypeError: If the schema, dynamic fields, or static metadata is unsupported.
            ValueError: If child field names are repeated or unknown.
        """
        names = type(self)._pytree_field_names()  # noqa: SLF001
        child_names = type(self)._pytree_child_names()  # noqa: SLF001
        child_names_set = set(child_names)
        if len(child_names_set) != len(child_names):
            msg = f"{type(self).__name__}.pytree_children must not repeat field names."
            raise ValueError(msg)
        unknown = child_names_set.difference(names)
        if unknown:
            msg = f"{type(self).__name__}.pytree_children refers to unknown fields: {', '.join(sorted(unknown))}."
            raise ValueError(msg)

        children = tuple(getattr(self, name) for name in child_names)
        for name, value in zip(child_names, children, strict=True):
            _jax_validate_dynamic_field(value, f"{type(self).__name__}.{name}")
        aux: list[tuple[str, Any]] = []
        for name in names:
            if name not in child_names_set:
                value = getattr(self, name)
                _jax_validate_static_metadata(value, f"{type(self).__name__}.{name}")
                aux.append((name, value))

        return children, (child_names, tuple(aux))

    @classmethod
    def tree_unflatten(cls, aux_data: Any, children: tuple[Any, ...]) -> Self:  # noqa: ANN401
        """Rebuild an object from pytree children and auxiliary data.

        Neither ``__init__`` nor ``__post_init__`` is called: JAX may provide
        tracers, placeholders, or cotangents that violate primal-value invariants.

        Args:
            aux_data: The auxiliary data produced by :meth:`tree_flatten`.
            children: The children produced by :meth:`tree_flatten`.

        Returns:
            The reconstructed object.
        """
        child_names, aux = aux_data

        obj = object.__new__(cls)
        for name, value in zip(child_names, children, strict=True):
            object.__setattr__(obj, name, value)
        for name, value in aux:
            object.__setattr__(obj, name, value)
        return obj

    @classmethod
    @abstractmethod
    def __jax_function__(
        cls,
        func: Callable,
        types: tuple[type[Any], ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:  # noqa: ANN401
        """Handle a call to one of the wrappers in :mod:`ndlike.jax_functions`.

        Args:
            func: The wrapper function that was called.
            types: The types of the overriding arguments.
            args: The positional arguments of the call.
            kwargs: The keyword arguments of the call.

        Returns:
            The result of the call, or ``NotImplemented`` if it is not supported.
        """

    @classmethod
    def _array_field_names(cls) -> tuple[str, ...]:
        """Return the representation fields used by array convenience methods.

        Defaults to the explicit child declaration, independently of custom
        flatten/unflatten methods. Override this for different field semantics.
        """
        return cls._pytree_child_names()

    def _map_array_fields(self, op: Callable[[Any], Any]) -> Self:
        """Apply an operation to the declared representation fields.

        ``None`` fields and all other declared fields are preserved. This does not
        flatten the object or require metadata to be valid static JAX data.

        Args:
            op: The operation to apply to each array-valued field.

        Returns:
            A copy with the operation applied to every array-valued field.
        """
        updates: dict[str, Any] = {}
        for name in type(self)._array_field_names():  # noqa: SLF001
            value = getattr(self, name)
            updates[name] = value if value is None else op(value)
        obj = object.__new__(type(self))
        for name in type(self)._pytree_field_names():  # noqa: SLF001
            object.__setattr__(obj, name, getattr(self, name))
        for name, value in updates.items():
            object.__setattr__(obj, name, value)
        return obj

    def stop_gradient(self) -> Self:
        """Return a copy detached from the autodiff graph.

        Subclasses holding NumPy sidecar fields should override this to leave those fields
        untouched.

        Returns:
            A copy through which gradients do not propagate.
        """
        return self._map_array_fields(
            lambda child: (
                child.stop_gradient()
                if isinstance(child, JaxLikeImplementation) and not isinstance(child, jax.Array)
                else jax.lax.stop_gradient(child)
            )
        )

    def block_until_ready(self) -> Self:
        """Block until the asynchronous computation of every array-valued field has finished.

        Returns:
            The object itself.
        """
        for name in type(self)._array_field_names():  # noqa: SLF001
            child = getattr(self, name)
            if child is not None and hasattr(child, "block_until_ready"):
                child.block_until_ready()
        return self

    def astype(
        self,
        dtype: DTypeLike | None,
        copy: bool = False,
        device: jax.Device | Sharding | None = None,
    ) -> Self:
        """Return a copy of the array cast to the given data type.

        Args:
            dtype: The target data type.
            copy: Whether to always return a copy.
            device: The device the result should live on.

        Returns:
            A copy with every array-valued field cast to the given data type.
        """
        return self._map_array_fields(lambda child: jax_astype(child, dtype, copy=copy, device=device))

    @override
    def to_device(self, device: Any, /, *, stream: int | Any | None = None) -> Self:
        """Move the array to the given device.

        Args:
            device: The target device.
            stream: Unsupported by JAX, must be None.

        Returns:
            A copy with every array-valued field on the given device.

        Raises:
            NotImplementedError: If ``stream`` is not None.
        """
        if stream is not None:
            msg = "The stream argument is not supported by JAX."
            raise NotImplementedError(msg)
        return self._map_array_fields(
            lambda child: (
                child.to_device(device) if isinstance(child, JaxLikeImplementation) else jax.device_put(child, device)
            )
        )

    @property
    def T(self) -> JaxLike[Any] | jax.Array:  # noqa: N802
        """Inverts the order of the axes of the underlying array."""
        return jax_transpose(self)

    @property
    def mT(self) -> JaxLike[Any] | jax.Array:  # noqa: N802
        """The version of the underlying array with its last two axes swapped."""
        return jax_matrix_transpose(self)

    @property
    def mH(self) -> JaxLike[Any] | jax.Array:  # noqa: N802
        """The adjoint (conjugate) transposed version of the underlying array."""
        return jax_conj(self.mT)

    def reshape(self, *args: int | Sequence[int], order: str = "C") -> JaxLike[Any] | jax.Array:
        """Return a reshaped version of the array.

        Args:
            *args: The target shape, either as a single sequence or as separate integers.
            order: The read/write element order.

        Returns:
            The reshaped array.
        """
        shape = args[0] if len(args) == 1 and not isinstance(args[0], int) else args
        return jax_reshape(self, shape, order)  # ty: ignore[no-matching-overload]

    def transpose(self, *args: int | Sequence[int] | None) -> JaxLike[Any] | jax.Array:
        """Return a transposed version of the array.

        Args:
            *args: The permutation of the axes, either as a single sequence or as separate
                integers. If omitted, the axes are reversed.

        Returns:
            The transposed array.
        """
        if len(args) == 0:
            return jax_transpose(self)
        axes = args[0] if len(args) == 1 and not isinstance(args[0], int) else args
        return jax_transpose(self, axes)  # ty: ignore[invalid-argument-type]

    def ravel(self, order: str = "C") -> JaxLike[Any] | jax.Array:
        """Return a flattened version of the array.

        Args:
            order: The read/write element order.

        Returns:
            The flattened array.
        """
        return jax_reshape(self, (-1,), order)

    def flatten(self, order: str = "C") -> JaxLike[Any] | jax.Array:
        """Return a flattened version of the array.

        Args:
            order: The read/write element order.

        Returns:
            The flattened array.
        """
        return self.ravel(order)

    def squeeze(self, axis: int | Sequence[int] | None = None) -> JaxLike[Any] | jax.Array:
        """Return the array with axes of length one removed.

        Args:
            axis: The axis or axes to remove. If omitted, all length-one axes are removed.

        Returns:
            The squeezed array.
        """
        return jax_squeeze(self, axis)

    def copy(self) -> JaxLike[Any] | jax.Array:
        """Return a copy of the array."""
        return jax_copy(self)

    def view(self, dtype: DTypeLike | None = None, type: None = None) -> Self:  # noqa: A002
        """Return a bit-cast view of the array.

        Args:
            dtype: The data type to reinterpret the underlying bytes as.
            type: Unsupported by JAX, must be ``None``.

        Returns:
            A copy with every array-valued field bit-cast to the given data type.

        Raises:
            NotImplementedError: If ``type`` is not None.
        """
        if type is not None:
            msg = "The type argument of view() is not supported by JAX."
            raise NotImplementedError(msg)
        return self._map_array_fields(lambda child: child.view(dtype))

    def item(self, *args: int) -> bool | int | float | complex:
        """Return the array as a Python scalar.

        Args:
            *args: Optional index of the element to return.

        Returns:
            The selected element as a Python scalar.
        """
        return np.asarray(self).item(*args)

    def tolist(self) -> Any:  # noqa: ANN401
        """Return the array as a (nested) Python list."""
        return np.asarray(self).tolist()

    @override
    def __array__(self, dtype: NpDTypeLike | None = None, /, *, copy: bool | None = None) -> NDArray[Any]:
        """Convert to a numpy array.

        Args:
            dtype: The desired data type of the output array.
            copy: Whether to return a copy of the data.

        Returns:
            The converted numpy array.
        """
        if has_jax_function((self,)):
            return handle_jax_function(np.asarray, (self,), self, dtype=dtype, copy=copy)
        return np.asarray(jnp.asarray(self), dtype=dtype, copy=copy)


JaxLikeImplementation.register(jax.Array)


def _jax_validate_dynamic_field(value: object, name: str) -> None:
    # Nested implementations validate their own fields when JAX traverses them;
    # treating them as leaves here avoids repeatedly flattening their layouts.
    leaves = jax.tree.leaves(value, is_leaf=lambda node: isinstance(node, JaxLikeImplementation))
    for leaf in leaves:
        if isinstance(leaf, np.ndarray) and leaf.dtype != jax.dtypes.float0:
            msg = (
                f"{name} contains a NumPy array. Default JaxLike pytree traversal requires JAX-backed dynamic fields. "
                "NumPy sidecars are eager-only; use opaque jax_tree_map, explicitly convert to JAX data, "
                "or provide custom flattening with defined transformation semantics."
            )
            raise TypeError(msg)


def _jax_validate_static_metadata(value: object, name: str) -> None:
    msg = (
        f"{name} must be static, hashable metadata without JAX/NumPy arrays. "
        "Declare JAX data in pytree_children, keep NumPy sidecars outside transformed values, "
        "or provide custom flattening with defined transformation semantics."
    )
    if isinstance(value, (jax.Array, np.ndarray)):
        raise TypeError(msg)
    try:
        hash(value)
    except TypeError as error:
        raise TypeError(msg) from error


def jax_tree_map(
    function: Callable[..., Any],
    tree: object,
    *rest: object,
    is_leaf: Callable[[Any], bool] | None = None,
    traverse_jaxlikes: bool | None = None,
) -> Any:  # noqa: ANN401
    """Map a function without exposing custom array representations by default.

    Args:
        function: Function applied to corresponding leaves of the input trees.
        tree: Primary input tree.
        *rest: Additional trees, following ``jax.tree.map``'s structure rules.
        is_leaf: Additional predicate for treating objects as opaque leaves.
        traverse_jaxlikes: Override every implementation's ``tree_map_traverse``
            flag. If None, use the per-class flag (false by default).

    Returns:
        The mapped tree. Opaque implementations are passed to ``function`` as
        whole objects, allowing their custom operators and methods to run.

    Notes:
        This only affects this helper; native ``jax.tree.map`` always follows
        pytree registration. A true ``is_leaf`` result takes precedence over
        either traversal setting. Traversing a representation does not enforce
        its invariants or invoke its custom operators. Default traversal rejects
        NumPy sidecars, while opaque eager mapping leaves them under the control
        of the representation's operations. Opaque mapping does not bypass JAX
        validation when mixed objects are transformation arguments or results.
    """

    def mapping_leaf(value: object) -> bool:
        if is_leaf is not None and is_leaf(value):
            return True
        if isinstance(value, JaxLikeImplementation):
            traverse = (
                getattr(type(value), "tree_map_traverse", False) if traverse_jaxlikes is None else traverse_jaxlikes
            )
            return not traverse
        return False

    return jax.tree.map(function, tree, *rest, is_leaf=mapping_leaf)


@flexdispatch
def to_jax_like[DT](
    data: object,
    /,
    dtype: DTypeLike | None = None,
    *,
    device: jax.Device | Sharding | None = None,
    copy: bool = False,
) -> JaxLike[Any] | jax.Array:
    """Convert an ArrayLike to a JaxLike.

    If possible, use the __jax_like__ method to convert the array, otherwise use jax.numpy.asarray.

    Args:
        data: The data to convert.
        dtype: The desired data type of the output array.
        device: The desired device of the output array.
        copy: Whether to return a copy of the input data.

    Returns:
        The converted array.
    """
    return jnp.asarray(data, dtype=dtype, device=device, copy=copy)


@to_jax_like.register(JaxLikeConvertible)
def _[DT](
    data: JaxLikeConvertible[DT],
    /,
    dtype: DTypeLike | None = None,
    *,
    device: jax.Device | Sharding | None = None,
    copy: bool = False,
) -> JaxLike[Any] | jax.Array:
    return data.__jax_like__(dtype, device=device, copy=copy)


__all__ = ["JaxLike", "JaxLikeConvertible", "JaxLikeImplementation", "jax_tree_map", "to_jax_like"]
