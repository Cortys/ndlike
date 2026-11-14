"""Shared base class and helpers for protected-axis implementations."""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, fields, is_dataclass
from functools import wraps
from inspect import BoundArguments, signature
import math
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, cast, overload

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping


class EffectiveProtectedAxes:
    """Resolve constructor-owned fields once, after dataclass decoration.

    Each subclass gets its own descriptor. Resolution replaces it on that
    concrete class, so subclasses never reuse a parent's resolved layout.
    """

    def __get__(self, instance: object, owner: type) -> dict[str, int]:
        del instance
        cls = cast("Any", owner)
        excluded = {field.name for field in fields(cls) if not field.init} if is_dataclass(cls) else set()
        axes = {name: count for name, count in cls.protected_axes.items() if name not in excluded}
        cls._effective_protected_axes = axes
        return axes


class _AxisProtectedBase[V, R](metaclass=ABCMeta):
    """Read effective fields and logical shapes without validating representations.

    Instances and custom readers must supply fields with sufficient protected
    rank and compatible batch shapes. Only all-broadcast layouts need to inspect
    every field's shape to determine the logical shape.
    """

    protected_axes: ClassVar[dict[str, int]]
    permitted_functions: ClassVar[set[Callable[..., Any]]]
    _effective_protected_axes: ClassVar[EffectiveProtectedAxes] = EffectiveProtectedAxes()

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        cls.protected_axes = inherit_protected_axes(cls)
        # Each subclass resolves its own layout after decorators have completed.
        cls._effective_protected_axes = EffectiveProtectedAxes()

    @abstractmethod
    def with_protected_values(self, values: dict[str, V], func: Callable | None) -> R:
        """Reconstruct a value in the backend's representation family."""
        ...

    def validate(self) -> None:
        """Raise if effective fields have insufficient rank or incompatible batches.

        Call explicitly, or from __post_init__ to check dataclass construction.
        Fields owned by initialization (init=False) are excluded.
        """
        axes = self._effective_protected_axes
        values = self.protected_values()
        if values.keys() != axes.keys():
            msg = "Protected values must contain exactly the effective protected fields."
            raise ValueError(msg)
        validate_batch_sync(values, axes, getattr(type(self), "broadcasted_batch", frozenset()))

    def _postprocess_protected_values(self, values: dict[str, V], func: Callable) -> dict[str, V]:
        """Optionally transform the values exposed to a particular function."""
        del func
        return values

    @overload
    def protected_values(self, func: None = None) -> dict[str, V]: ...

    @overload
    def protected_values(self, func: Callable) -> dict[str, V] | None: ...

    def protected_values(self, func: Callable | None = None) -> dict[str, V] | None:
        """Read effective values as-is, optionally checking function permissions."""
        if func is not None and func not in type(self).permitted_functions:
            return None
        values = {name: getattr(self, name) for name in self._effective_protected_axes}
        return values if func is None else self._postprocess_protected_values(values, func)

    def _protected_batch_shape(self, values: Mapping[str, V] | None = None) -> tuple[int, ...]:
        """Read one shape, reusing supplied values when all fields broadcast."""
        axes = self._effective_protected_axes
        broadcasted = getattr(type(self), "broadcasted_batch", ())
        for name, count in axes.items():
            if name not in broadcasted:
                return batch_shape(value_shape(getattr(self, name)), count)
        if not axes:
            return ()
        if values is None:
            values = self.protected_values()
        return np.broadcast_shapes(*(batch_shape(value_shape(values[name]), count) for name, count in axes.items()))

    @property
    def shape(self) -> tuple[int, ...]:
        """Read the logical shape without collecting fields unless all broadcast."""
        return self._protected_batch_shape()


def require_effective_fields(axes: Mapping[str, int]) -> None:
    """Reject batch operations with no independent fields to carry their result."""
    if not axes:
        msg = "Batch operations require at least one effective protected field."
        raise ValueError(msg)


def inherit_protected_axes(cls: type) -> dict[str, int]:
    """Merge base layouts, requiring local declarations to resolve count conflicts.

    Local declarations may override inherited counts and retain their own order.
    """
    axes: dict[str, int] = {}
    # Bases already include their ancestors, including abstract classes.
    local_axes = vars(cls).get("protected_axes", {})
    if not isinstance(local_axes, dict):
        msg = f"{cls.__name__} must define protected_axes as a non-empty dict[str, int]."
        raise TypeError(msg)
    declarations = [getattr(base, "protected_axes", {}) for base in cls.__bases__]
    declarations.append(local_axes)
    for declared_axes in declarations:
        if not isinstance(declared_axes, dict):
            msg = f"{cls.__name__} must define protected_axes as a non-empty dict[str, int]."
            raise TypeError(msg)
        for name, value in declared_axes.items():
            if not isinstance(name, str) or len(name) == 0:
                msg = f"{cls.__name__}.protected_axes must use non-empty string keys."
                raise TypeError(msg)
            if not isinstance(value, int) or value < 0:
                msg = f"{cls.__name__}.protected_axes[{name}] must be a non-negative integer."
                raise TypeError(msg)
            if name in axes and axes[name] != value and name not in local_axes:
                msg = (
                    f"{cls.__name__}.protected_axes[{name!r}] conflicts with an inherited "
                    f"axis count: {axes[name]} != {value}. Redeclare {name!r} locally to resolve the conflict."
                )
                raise TypeError(msg)
            axes[name] = value
    return {name: count for name, count in axes.items() if name not in local_axes} | local_axes


class AxisProtectedCreator[V, R](Protocol):
    """Rebuild a representation from values in the same backend family."""

    def __call__(self, values: dict[str, V]) -> R:
        """Create an object from updated protected values."""
        ...


@dataclass(frozen=True, slots=True)
class AxisProtectedInternals[V, R]:
    """Internal representation for one protected-axis object."""

    create: AxisProtectedCreator[V, R]
    values: dict[str, V]
    protected_axes: dict[str, int]
    owner_type: type[Any]
    batch_shape: tuple[int, ...]
    broadcasted_batch: frozenset[str] = frozenset()

    @property
    def batch_ndim(self) -> int:
        """Return the logical number of batch dimensions."""
        return len(self.batch_shape)


@dataclass(frozen=True, slots=True)
class ProtectedValueSequenceInternals[V, R]:
    """Extracted internals for sequence-based operations."""

    has_protected: bool
    template: AxisProtectedInternals[V, R] | None
    values_by_field: dict[str, list[object]]


class InternalsExtractor[V, R](Protocol):
    """Backend adapter that supplies the protected-value type at dispatch."""

    def __call__(
        self, obj: object, func: Callable | None = None, *, check_is_permitted: bool = False
    ) -> AxisProtectedInternals[V, R] | None:
        """Extract internals when the object supports protected-axis operations."""
        ...


def extract_axis_protected_internals[V, R](
    obj: _AxisProtectedBase[V, R],
    func: Callable | None = None,
    *,
    check_is_permitted: bool = False,
) -> AxisProtectedInternals[V, R] | None:
    """Extract internals from an object with a backend-typed interface.

    Args:
        obj: Object whose protected-value family is established by the caller.
        func: Triggering function, also passed to reconstruction.
        check_is_permitted: Whether to read values in the function's context.

    Returns:
        Extracted internals, or None for an unsupported layout or function.
    """
    protected_axes = obj._effective_protected_axes  # noqa: SLF001
    if not obj.protected_axes:
        return None
    context = func if check_is_permitted else None
    values = obj.protected_values() if context is None else obj.protected_values(context)
    if values is None:
        return None
    shape = obj._protected_batch_shape(values if context is None else None)  # noqa: SLF001

    def create(values: dict[str, V]) -> R:
        return obj.with_protected_values(values, func)

    broadcasted_batch = frozenset(getattr(obj, "broadcasted_batch", ())).intersection(protected_axes)
    return AxisProtectedInternals(
        create=create,
        values=dict(values),
        protected_axes=dict(protected_axes),
        owner_type=type(obj),
        broadcasted_batch=broadcasted_batch,
        batch_shape=shape,
    )


def primary_protected_name(protected_axes: Mapping[str, int]) -> str:
    """Select the first declared protected field, independently of constructor order."""
    return next(iter(protected_axes))


def validate_batch_sync(
    values: Mapping[str, object],
    protected_axes: Mapping[str, int],
    broadcasted_batch: set[str] | frozenset[str] = frozenset(),
) -> tuple[int, ...]:
    """Resolve and validate the batch shape implied by the protected fields.

    The first unmarked field determines shape without broadcast inference.
    If every field is marked, their batch shapes are broadcast together.
    """
    shapes: dict[str, tuple[int, ...]] = {}
    for name, value in values.items():
        axes = protected_axes[name]
        ndim = value_ndim(value)
        shape = value_shape(value)
        if ndim < axes:
            msg = f"Operation removed protected trailing axes for field {name!r}."
            raise ValueError(msg)
        shapes[name] = batch_shape(shape, axes)

    if not shapes:
        return ()
    witness = next((name for name in shapes if name not in broadcasted_batch), None)
    expected = shapes[witness] if witness is not None else np.broadcast_shapes(*shapes.values())
    for name, current in shapes.items():
        if name in broadcasted_batch:
            if len(current) > len(expected) or any(
                size not in (1, target) for size, target in zip(reversed(current), reversed(expected), strict=False)
            ):
                msg = f"Batch shape {current} of field {name!r} cannot broadcast to logical batch shape {expected}."
                raise ValueError(msg)
        elif current != expected:
            msg = "Operation produced inconsistent batch-shapes across protected fields."
            raise ValueError(msg)
    return expected


def apply_unary[V, R](internals: AxisProtectedInternals[V, R], op: Callable[[str, V, int], V]) -> R:
    """Transform protected fields and rebuild, trusting the operation's layout."""
    results = {name: op(name, value, internals.protected_axes[name]) for name, value in internals.values.items()}
    return internals.create(results)


def extract_protected_value_sequence_internals[V, R](
    values: tuple[object, ...],
    func: Callable | None = None,
    *,
    extract: InternalsExtractor[V, R],
) -> ProtectedValueSequenceInternals[V, R]:
    """Extract and align sequence fields using a backend-typed extractor.

    Args:
        values: Protected objects and unprotected operands to align.
        func: Function context passed to the extractor.
        extract: Backend adapter that determines the protected-value family.

    Returns:
        Aligned fields and the first protected object's internals. Unprotected
        operands preceding the first protected object are skipped.

    Raises:
        ValueError: If protected inputs use different protected-axis layouts.
    """
    template: AxisProtectedInternals[V, R] | None = None
    values_by_field: dict[str, list[object]] = {}
    for value in values:
        internals = extract(value, func)
        if internals is None:
            if template is None:
                continue
            for name in template.protected_axes:
                values_by_field[name].append(value)
            continue

        if template is None:
            template = internals
            require_effective_fields(template.protected_axes)
            values_by_field = {name: [] for name in internals.protected_axes}
        elif internals.protected_axes != template.protected_axes:
            msg = "All protected inputs must share identical protected_axes definitions."
            raise ValueError(msg)
        elif internals.broadcasted_batch != template.broadcasted_batch:
            msg = "All protected inputs must share identical broadcasted_batch definitions."
            raise ValueError(msg)

        for name in template.protected_axes:
            values_by_field[name].append(internals.values[name])

    return ProtectedValueSequenceInternals(template is not None, template, values_by_field)


def map_batch_axes(value: object, protected_axes_count: int, batch_axes: tuple[int, ...]) -> tuple[int, ...]:
    """Append protected trailing axes to a normalized batch permutation."""
    ndim = value_ndim(value)
    batch_ndim = ndim - protected_axes_count
    normalized = normalize_axes(batch_axes, batch_ndim)
    return (*normalized, *range(batch_ndim, ndim))


def normalize_batch_reduction_axes(
    axis: object, batch_ndim: int, *, parameter_name: str = "axis"
) -> int | tuple[int, ...]:
    """Normalize reduction axes within the batch dimensions."""
    if axis is None:
        return tuple(range(batch_ndim))
    if isinstance(axis, int):
        return normalize_axis(axis, batch_ndim)
    if isinstance(axis, (tuple, list)) and all(isinstance(item, int) for item in axis):
        return normalize_axes(tuple(axis), batch_ndim)

    msg = f"reduction {parameter_name} must be None, an int, or a tuple/list of ints."
    raise TypeError(msg)


def has_numpy_protected_value[V, R](internals: AxisProtectedInternals[V, R]) -> bool:
    """Return whether any protected field contains a NumPy array."""
    return any(isinstance(value, np.ndarray) for value in internals.values.values())


def apply_structural_op[V](value: V, backend_op: Callable[[V], V], numpy_op: Callable[[np.ndarray], V]) -> V:
    """Apply a structural operation while preserving NumPy-backed fields."""
    if isinstance(value, np.ndarray):
        return numpy_op(value)
    return backend_op(value)


class FunctionOverride(Protocol):
    """Signature shared by backend function-dispatch overrides."""

    def __call__(
        self, func: Callable, types: tuple[type[Any], ...], args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> Any:  # noqa: ANN401
        """Handle a backend function call."""
        ...


class BoundFunction(Protocol):
    """Function handler receiving signature-bound arguments."""

    def __call__(self, func: Callable, params: BoundArguments) -> Any:  # noqa: ANN401
        """Handle bound arguments."""
        ...


class BoundFunctionWithInternals[V, R](Protocol):
    """Bound function handler receiving backend-typed protected internals."""

    def __call__(self, func: Callable, params: BoundArguments, internals: AxisProtectedInternals[V, R]) -> Any:  # noqa: ANN401
        """Handle bound arguments and extracted internals."""
        ...


def function_override(handler: BoundFunction) -> FunctionOverride:
    """Adapt a signature-bound handler to backend function-dispatch shape."""

    @wraps(handler)
    def wrapper(
        func: Callable,
        types: tuple[type[Any], ...],  # noqa: ARG001
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:  # noqa: ANN401
        params = signature(func).bind(*args, **kwargs)
        params.apply_defaults()
        return handler(func, params)

    return wrapper


def internals_override[V, R](
    parameter_name: str,
    check_is_permitted: bool = False,
    *,
    extract: InternalsExtractor[V, R],
    allow_empty: bool = True,
) -> Callable[[BoundFunctionWithInternals[V, R]], FunctionOverride]:
    """Adapt a bound handler using a backend-typed internals extractor.

    Args:
        parameter_name: Bound parameter containing the protected object.
        check_is_permitted: Whether to read values in the function's context.
        extract: Backend adapter that determines the protected-value family.
        allow_empty: Whether the operation works without independent fields.

    Returns:
        A decorator that binds arguments and extracts protected internals.
    """

    def decorator(handler: BoundFunctionWithInternals[V, R]) -> FunctionOverride:
        @wraps(handler)
        def wrapper(func: Callable, params: BoundArguments) -> Any:  # noqa: ANN401
            internals = extract(params.arguments[parameter_name], func, check_is_permitted=check_is_permitted)
            if internals is None:
                return NotImplemented
            if not allow_empty:
                require_effective_fields(internals.protected_axes)
            return handler(func, params, internals)

        return function_override(wrapper)

    return decorator


def value_ndim(value: object) -> int:
    """Return ``value.ndim`` as an ``int``.

    Args:
        value: Value expected to expose an ``ndim`` attribute.

    Returns:
        The number of dimensions.

    Raises:
        TypeError: If the value does not expose ``ndim`` as an integer.
    """
    ndim = getattr(value, "ndim", None)
    if not isinstance(ndim, int):
        msg = f"Value of type {type(value).__name__} does not expose an integer ndim attribute."
        raise TypeError(msg)
    return ndim


def value_shape(value: object) -> tuple[int, ...]:
    """Return ``value.shape`` as a tuple of integers.

    Args:
        value: Value expected to expose a ``shape`` attribute.

    Returns:
        The shape tuple.

    Raises:
        TypeError: If the value does not expose an iterable integer shape.
    """
    shape = getattr(value, "shape", None)
    if shape is None:
        msg = f"Value of type {type(value).__name__} does not expose a shape attribute."
        raise TypeError(msg)

    try:
        shape_tuple = tuple(int(dim) for dim in shape)
    except TypeError as exc:
        msg = f"Value of type {type(value).__name__} does not expose an iterable integer shape."
        raise TypeError(msg) from exc

    return shape_tuple


def batch_shape(shape: tuple[int, ...], protected_axes: int) -> tuple[int, ...]:
    """Return the batch prefix of a full shape."""
    return shape if protected_axes == 0 else shape[:-protected_axes]


def flattened_batch_shape(shape: tuple[int, ...], protected_axes: int) -> tuple[int, ...]:
    """Collapse batch dimensions to one axis, retaining the protected suffix.

    An unbatched value becomes a batch of length one. Explicit multiplication
    preserves the batch size even when a protected dimension has length zero.
    """
    return (math.prod(batch_shape(shape, protected_axes)), *protected_shape(shape, protected_axes))


def protected_shape(shape: tuple[int, ...], protected_axes: int) -> tuple[int, ...]:
    """Return the protected trailing suffix of a full shape."""
    return () if protected_axes == 0 else shape[-protected_axes:]


def normalize_axis(axis: int, ndim: int, *, allow_endpoint: bool = False) -> int:
    """Normalize a possibly-negative batch axis and validate bounds."""
    bound = ndim + (1 if allow_endpoint else 0)
    normalized = axis + bound if axis < 0 else axis
    if normalized < 0 or normalized >= bound:
        msg = f"axis {axis} is out of bounds for batch dimensions with ndim {ndim}."
        raise ValueError(msg)
    return normalized


def normalize_axes(axes: tuple[int, ...], ndim: int, *, allow_endpoint: bool = False) -> tuple[int, ...]:
    """Normalize and validate a tuple of batch axes."""
    return tuple(normalize_axis(axis, ndim, allow_endpoint=allow_endpoint) for axis in axes)


def coerce_axis_tuple(axis: int | tuple[int, ...] | list[int]) -> tuple[int, ...]:
    """Convert axis input into a tuple form."""
    return (axis,) if isinstance(axis, int) else tuple(axis)
