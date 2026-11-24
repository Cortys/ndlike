"""Shared base class and helpers for protected-axis implementations."""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from collections.abc import Sequence
from copy import copy
from dataclasses import dataclass, field, fields, is_dataclass, replace
from functools import wraps
from inspect import BoundArguments, isabstract, signature
import math
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, Self, SupportsIndex, cast, overload

import numpy as np

from ndlike.tuple import BroadcastTuple

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping


_UNSET = object()


class EffectiveProtectedAxes:
    """Resolve constructor-owned fields once, after dataclass decoration.

    Each subclass gets its own descriptor. Resolution replaces it on that
    concrete class, so subclasses never reuse a parent's resolved layout.
    """

    def __get__(self, instance: object, owner: type) -> dict[str, int]:
        del instance
        cls = cast("Any", owner)
        axes = dict(cls.protected_axes)
        structural_names = None
        if is_dataclass(cls):
            declared = fields(cls)
            init_names = {field.name for field in declared if field.init}
            if axes and axes.keys() <= init_names and cls.__new__ is object.__new__:
                structural_names = frozenset(field.name for field in declared)
            for field in declared:
                if not field.init:
                    axes.pop(field.name, None)
        cls._structural_field_names = structural_names
        cls._effective_protected_axes = axes
        return axes


class _AxisProtectedBase[V, R](metaclass=ABCMeta):
    """Read effective fields and logical shapes without validating representations.

    Instances and custom readers must supply fields with sufficient protected
    rank and compatible batch shapes. Only all-broadcast layouts need to inspect
    every field's shape to determine the logical shape.

    Plain dataclass subclasses rebuild structurally without invoking ``__init__``
    or ``__post_init__`` on synchronized single-object logical batch rearrangements
    unless transforming an empty logical batch into a nonempty one, storing
    undeclared attributes in the instance dictionary, overriding ``__new__`` / ``protected_values``
    / ``with_protected_values``, or setting ``revalidate_on_batch_change = True``.
    Compatible sequences and indexed replacements from validated instances use
    the same structural reconstruction safeguards.
    """

    protected_axes: ClassVar[dict[str, int]]
    broadcasted_batch: ClassVar[set[str]] = set()
    permitted_functions: ClassVar[set[Callable[..., Any]]]
    revalidate_on_batch_change: ClassVar[bool] = False
    _effective_protected_axes: ClassVar[EffectiveProtectedAxes] = EffectiveProtectedAxes()
    _structural_field_names: ClassVar[frozenset[str] | None] = None
    _default_with_protected_values: ClassVar[object] = None

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        cls.protected_axes = inherit_protected_axes(cls)
        cls.broadcasted_batch = inherit_broadcasted_batch(cls)
        # Each subclass resolves its own layout after decorators have completed.
        cls._effective_protected_axes = EffectiveProtectedAxes()
        cls._structural_field_names = None

    @abstractmethod
    def with_protected_values(self, values: dict[str, V], func: Callable | None) -> R:
        """Reconstruct a value in the backend's representation family."""
        ...

    def _with_protected_values_structural(self, values: dict[str, V], func: Callable | None = None) -> R:
        """Rebuild a plain dataclass structurally on synchronized batch rearrangements.

        Copy declared fields by name, preserving non-protected metadata by reference.
        Subclasses that rely on custom attribute storage should set
        ``revalidate_on_batch_change = True`` to use ordinary construction.
        """
        if not _can_rebuild_structurally(type(self), self) or values.keys() != self._effective_protected_axes.keys():
            return self.with_protected_values(values, func)
        return cast("R", self._from_structural_fields(self, values))

    @classmethod
    def _from_structural_fields(cls, source: _AxisProtectedBase[Any, Any], updates: Mapping[str, Any]) -> Self:
        obj = object.__new__(cls)
        for name in cast("frozenset[str]", cls._structural_field_names):
            if name in updates:
                value = updates[name]
            else:
                try:
                    value = object.__getattribute__(source, name)
                except AttributeError:
                    continue  # An init=False field can legitimately be unset.
            object.__setattr__(obj, name, value)
        return obj

    @classmethod
    def _combine_sequence_metadata(cls, items: Sequence[Self], *, func: Callable | None) -> dict[str, Any]:
        """Return overrides for static fields that have a domain-specific combination rule.

        The common ancestor receives all protected inputs. Omitted static fields
        must compare equal and retain the first input's value.
        """
        del items, func
        return {}

    def _compact_assignment_values(
        self,
        prepared: object,
        current: Mapping[str, V],
        candidates: Mapping[str, object],
        field_index: Callable[[int], tuple[Any, ...]],
    ) -> dict[str, V]:
        if not isinstance(prepared, type(self)):
            return {}
        compact = {
            name: value
            for name, value in current.items()
            if name in self.broadcasted_batch
            and candidates[name] is value
            and value_ndim(value) == self._effective_protected_axes[name]
        }
        if compact.keys() == current.keys():
            # Even a no-op scalar update needs a valid logical index.
            name = next(iter(current))
            _ = cast("Any", current[name])[field_index(self._effective_protected_axes[name])]
        return compact

    def _assignment_is_compatible(
        self,
        prepared: object,
        current: Mapping[str, V],
        candidates: Mapping[str, object],
        dtype_of: Callable[[object], Any],
    ) -> bool:
        """Check structural eligibility, destination static fields, and lossless leaf casts."""
        return (
            _can_rebuild_structurally(type(self), self)
            and isinstance(prepared, type(self))
            and self._effective_protected_axes == prepared._effective_protected_axes  # noqa: SLF001
            and _static_fields_equal(self, prepared, self.protected_axes)
            and all(_lossless_assignment(current[name], candidates[name], dtype_of) for name in current)
        )

    def _set_protected_items(
        self,
        prepared: object,
        candidates: Mapping[str, object],
        *,
        field_index: Callable[[int], tuple[Any, ...]],
        broadcast: Callable[[V, int, tuple[int, ...]], V],
        clone: Callable[[V], V],
        dtype_of: Callable[[object], Any],
    ) -> None:
        """Write compatible replacements directly; stage and validate all other assignments."""
        current = self.protected_values()
        logical_shape = self._protected_batch_shape(current)
        compact = self._compact_assignment_values(prepared, current, candidates, field_index)
        direct = self._assignment_is_compatible(prepared, current, candidates, dtype_of)

        targets: dict[str, V] = {}
        for name, axes in self._effective_protected_axes.items():
            if name in compact:
                continue
            original = current[name]
            value = broadcast(original, axes, logical_shape) if name in self.broadcasted_batch else original
            if value is original and isinstance(value, np.ndarray) and not value.flags.writeable:
                msg = f"Assignment destination for field {name!r} is read-only."
                raise ValueError(msg)
            # Direct writes update existing buffers in place and only clone newly expanded broadcasts.
            # Validated writes stage into copies so a failed constructor cannot leak bad values into self.
            targets[name] = value if direct and value is original else clone(value)

        for name, target in targets.items():
            cast("Any", target)[field_index(self._effective_protected_axes[name])] = candidates[name]

        if direct:
            for name, target in targets.items():
                if target is not current[name]:
                    object.__setattr__(self, name, target)
            return

        staged = current | targets
        # A constructor may normalize a shared field in place before raising.
        for name, value in compact.items():
            if not isinstance(value, np.ndarray) or value.flags.writeable:
                staged[name] = clone(value)
        result = self.with_protected_values(staged, None)
        if not isinstance(result, type(self)):
            msg = "Indexed assignment must rebuild an instance of the destination type."
            raise TypeError(msg)
        self._apply_validated_assignment(current, result, compact)

    def _apply_validated_assignment(self, current: Mapping[str, V], result: Self, compact: Mapping[str, V]) -> None:
        """Write validated fields into existing buffers when compatible, else install them."""
        for name, value in result.protected_values().items():
            original = current[name]
            if value is original:
                continue
            same_layout = (
                value_shape(value) == value_shape(original)
                and getattr(value, "dtype", None) == getattr(original, "dtype", None)
                and getattr(value, "device", None) == getattr(original, "device", None)
            )
            if name in compact:
                # Keep the shared object unless construction changed its value.
                if same_layout and _metadata_equal(original, value):
                    continue
            elif same_layout and not (isinstance(original, np.ndarray) and not original.flags.writeable):
                cast("Any", original)[...] = value
                continue
            object.__setattr__(self, name, value)
        # Include constructor-owned slots and any undeclared dictionary state.
        state = _stored_attributes(result)
        if is_dataclass(result):
            stale = _stored_attributes(self).keys() - state.keys()
            for f in fields(result):
                if not f.init and f.name in stale:
                    object.__delattr__(self, f.name)
        for name, value in state.items():
            if name not in current and getattr(self, name, _UNSET) is not value:
                object.__setattr__(self, name, value)

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

    _default_protected_values: ClassVar[object] = protected_values

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


def _stored_attributes(obj: object) -> dict[str, Any]:
    """Collect an instance's dictionary and populated slots."""
    state = cast("dict | tuple[dict | None, dict] | None", object.__getstate__(obj))
    if isinstance(state, tuple):
        dictionary, slots = state
        return (dictionary or {}) | slots
    return state or {}


def _metadata_equal(left: Any, right: Any) -> bool:  # noqa: ANN401
    if left is right:
        return True
    if getattr(left, "shape", None) != getattr(right, "shape", None):
        return False
    try:
        equal = left == right
        return bool(equal.all() if getattr(equal, "ndim", 0) else equal)
    except (TypeError, ValueError, RuntimeError):
        return False


def _static_fields_equal(target: Any, source: Any, protected: Mapping[str, int] | set[str]) -> bool:  # noqa: ANN401
    return all(
        _metadata_equal(getattr(target, f.name, _UNSET), getattr(source, f.name, _UNSET))
        for f in fields(target)
        if f.name not in protected
    )


def _lossless_assignment(target: object, source: object, dtype_of: Callable[[object], Any]) -> bool:
    if target is source:
        return True
    if isinstance(target, _AxisProtectedBase) and isinstance(source, _AxisProtectedBase):
        return target._assignment_is_compatible(source, target.protected_values(), source.protected_values(), dtype_of)  # noqa: SLF001
    if isinstance(target, BroadcastTuple) and isinstance(source, BroadcastTuple) and type(target) is type(source):
        return (
            _static_fields_equal(target, source, {"members"})
            and len(target.members) == len(source.members)
            and all(_lossless_assignment(t, s, dtype_of) for t, s in zip(target.members, source.members, strict=True))
        )
    return _lossless_dtype_cast(dtype_of(source), dtype_of(target))


def _lossless_dtype_cast(src: Any, dst: Any) -> bool:  # noqa: ANN401
    if src is None or dst is None:
        return False
    if src == dst:
        return True
    try:
        src, dst = np.dtype(src), np.dtype(dst)
    except TypeError:
        return False
    if src.kind in "iu" and dst.kind in "fc":
        # NumPy's "safe" casting also accepts int64 -> float64, which rounds large IDs.
        return 8 * src.itemsize - int(src.kind == "i") <= np.finfo(dst).nmant + 1
    return src.kind in "biufc" and dst.kind in "biufc" and bool(np.can_cast(src, dst, casting="safe"))


def _can_rebuild_structurally(cls: type[_AxisProtectedBase], source: _AxisProtectedBase) -> bool:
    """Check the result class's construction hooks and the source's stored state."""
    # Resolve the descriptors before reading their cached structural field names.
    _ = cls._effective_protected_axes, source._effective_protected_axes  # noqa: SLF001
    source_names = type(source)._structural_field_names  # noqa: SLF001
    return (
        cls._structural_field_names is not None
        and not cls.revalidate_on_batch_change
        and cls.with_protected_values is cls._default_with_protected_values
        and cls.protected_values is cls._default_protected_values
        and source_names is not None
        and vars(source).keys() <= source_names
    )


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


def inherit_broadcasted_batch(cls: type[_AxisProtectedBase]) -> set[str]:
    """Merge base and local broadcast markers, validating against the resolved layout."""
    broadcasted: set[str] = set()
    declarations = [getattr(base, "broadcasted_batch", set()) for base in cls.__bases__]
    declarations.append(vars(cls).get("broadcasted_batch", set()))
    for declaration in declarations:
        if not isinstance(declaration, set):
            msg = f"{cls.__name__}.broadcasted_batch must be a set of protected field names."
            raise TypeError(msg)
        broadcasted.update(declaration)
    if not broadcasted <= cls.protected_axes.keys():
        msg = f"{cls.__name__}.broadcasted_batch must be a set of protected field names."
        raise TypeError(msg)
    return broadcasted


class AxisProtectedCreator[V, R](Protocol):
    """Rebuild a representation from values in the same backend family."""

    def __call__(self, values: dict[str, V]) -> R:
        """Create an object from updated protected values."""
        ...


def _logical_batch_is_empty(
    values: Mapping[str, object],
    protected_axes: Mapping[str, int],
    broadcasted_batch: set[str] | frozenset[str],
) -> bool:
    """Return whether the static logical batch shape implied by ``values`` is empty."""
    for name, count in protected_axes.items():
        if name not in broadcasted_batch:
            return 0 in batch_shape(value_shape(values[name]), count)
    return any(0 in batch_shape(value_shape(values[name]), count) for name, count in protected_axes.items())


def rebuild_structural[V, R](
    obj: _AxisProtectedBase[V, R],
    values: dict[str, V],
    source_batch_shape: tuple[int, ...],
    protected_axes: Mapping[str, int],
    broadcasted_batch: set[str] | frozenset[str],
    func: Callable | None = None,
) -> R:
    """Rebuild structurally unless transforming an empty logical batch into a nonempty one."""
    if 0 in source_batch_shape and not _logical_batch_is_empty(values, protected_axes, broadcasted_batch):
        return obj.with_protected_values(values, func)
    return obj._with_protected_values_structural(values, func)  # noqa: SLF001


@dataclass(frozen=True, slots=True)
class AxisProtectedInternals[V, R]:
    """Internal representation for one protected-axis object."""

    owner: _AxisProtectedBase[V, R]
    values: dict[str, V]
    protected_axes: dict[str, int]
    batch_shape: tuple[int, ...]
    broadcasted_batch: frozenset[str] = frozenset()
    func: Callable | None = None

    @property
    def owner_type(self) -> type[Any]:
        """Return the concrete type of the extracted owner."""
        return type(self.owner)

    @property
    def batch_ndim(self) -> int:
        """Return the logical number of batch dimensions."""
        return len(self.batch_shape)

    def create(self, values: dict[str, V]) -> R:
        """Reconstruct via ordinary ``with_protected_values``."""
        return self.owner.with_protected_values(values, self.func)

    def rebuild_structural(self, values: dict[str, V]) -> R:
        """Rebuild structurally unless transforming an empty logical batch into a nonempty one."""
        return rebuild_structural(
            self.owner,
            values,
            self.batch_shape,
            self.protected_axes,
            self.broadcasted_batch,
            self.func,
        )


@dataclass(frozen=True, slots=True)
class ProtectedValueSequenceInternals[V, R]:
    """Extracted internals for sequence-based operations."""

    has_protected: bool
    template: AxisProtectedInternals[V, R] | None
    values_by_field: dict[str, list[object]]
    compact_values: dict[str, V] = field(default_factory=dict)
    creator: AxisProtectedCreator[V, R] | None = None

    def create(self, values: dict[str, V]) -> R:
        if self.creator is None:
            msg = "Cannot rebuild a sequence without protected inputs."
            raise TypeError(msg)
        return self.creator(self.compact_values | values)


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
    broadcasted_batch = frozenset(getattr(obj, "broadcasted_batch", ())).intersection(protected_axes)
    return AxisProtectedInternals(
        owner=obj,
        values=dict(values),
        protected_axes=dict(protected_axes),
        batch_shape=shape,
        broadcasted_batch=broadcasted_batch,
        func=func,
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


def apply_unary[V, R](
    internals: AxisProtectedInternals[V, R],
    op: Callable[[str, V, int], V],
    *,
    structural: bool = False,
) -> R:
    """Transform protected fields and rebuild, trusting the operation's layout."""
    results = {name: op(name, value, internals.protected_axes[name]) for name, value in internals.values.items()}
    return internals.rebuild_structural(results) if structural else internals.create(results)


def prepare_broadcasted_batch[V, R](
    internals: AxisProtectedInternals[V, R],
    *,
    shape_only: bool,
    value_only: bool,
    broadcast: Callable[[V, int, tuple[int, ...]], V],
    reshape: Callable[[V, tuple[int, ...]], V],
) -> AxisProtectedInternals[V, R]:
    """Expand logical batches, preserving singleton storage when a native field carries shape."""
    if not internals.broadcasted_batch or value_only:
        return internals
    values = {}
    for name, value in internals.values.items():
        axes = internals.protected_axes[name]
        if (
            shape_only
            and len(internals.broadcasted_batch) < len(internals.protected_axes)
            and name in internals.broadcasted_batch
            and math.prod(batch_shape(value_shape(value), axes)) == 1
        ):
            suffix = protected_shape(value_shape(value), axes)
            values[name] = value if value_ndim(value) == axes else reshape(value, suffix)
        else:
            values[name] = broadcast(value, axes, internals.batch_shape)
    return replace(internals, values=values)


def apply_shape_only[V, R](
    internals: AxisProtectedInternals[V, R],
    op: Callable[[str, V, int], V],
    *,
    structural: bool = True,
) -> R:
    """Transform batches while retaining normalized singleton broadcast fields."""

    def field_op(name: str, value: V, axes: int) -> V:
        # An empty representation can remain scalar even after explicit broadcasting.
        if (
            (len(internals.broadcasted_batch) < len(internals.protected_axes) or internals.batch_shape != ())
            and name in internals.broadcasted_batch
            and value_ndim(value) == axes
        ):
            return value
        return op(name, value, axes)

    return apply_unary(internals, field_op, structural=structural)


def _common_sequence_class(items: Sequence[_AxisProtectedBase]) -> type[_AxisProtectedBase]:
    classes = tuple(dict.fromkeys(type(item) for item in items))
    cls = next(base for base in classes[0].__mro__ if all(issubclass(t, base) for t in classes))
    if not issubclass(cls, _AxisProtectedBase):
        msg = f"Protected inputs need a common AxisProtected ancestor, got {cls.__name__}."
        raise TypeError(msg)
    if any(
        t.protected_axes != cls.protected_axes or t._effective_protected_axes != cls._effective_protected_axes  # noqa: SLF001
        for t in classes
    ):
        msg = "All protected inputs must share identical protected_axes definitions with their common ancestor."
        raise ValueError(msg)
    if any(t.broadcasted_batch != cls.broadcasted_batch for t in classes):
        msg = "All protected inputs must share identical broadcasted_batch definitions with their common ancestor."
        raise ValueError(msg)
    if isabstract(cls):
        msg = f"Protected inputs need a concrete common ancestor, got abstract class {cls.__name__}."
        raise TypeError(msg)
    return cls


def _sequence_metadata(
    cls: type[_AxisProtectedBase], items: Sequence[_AxisProtectedBase], func: Callable | None
) -> dict[str, Any]:
    if not is_dataclass(cls):
        return {}
    overrides = cls._combine_sequence_metadata(items, func=func)
    static_names = [f.name for f in fields(cls) if f.name not in cls.protected_axes]
    if unknown := overrides.keys() - static_names:
        msg = f"Sequence metadata overrides must name static fields of {cls.__name__}, got {sorted(unknown)}."
        raise ValueError(msg)
    metadata = dict(overrides)
    for name in static_names:
        if name in overrides:
            continue
        first = getattr(items[0], name, _UNSET)
        if any(not _metadata_equal(first, getattr(item, name, _UNSET)) for item in items[1:]):
            msg = f"Static field {name!r} must compare equal across {cls.__name__} inputs."
            raise ValueError(msg)
        if first is not _UNSET:
            metadata[name] = first
    return metadata


def _sequence_creator[V, R](
    inputs: list[AxisProtectedInternals[V, R]], *, structural: bool
) -> AxisProtectedCreator[V, R]:
    first = inputs[0]
    items = [item.owner for item in inputs]
    cls = _common_sequence_class(items)
    metadata = _sequence_metadata(cls, items, first.func)
    source = next((item for item in items if type(item) is cls), None)
    template = items[0] if source is None else source
    can_structural = (
        structural
        and _can_rebuild_structurally(cls, template)
        and all(_can_rebuild_structurally(type(item), item) for item in items)
    )
    if can_structural and (
        source is None or cls._with_protected_values_structural is _AxisProtectedBase._with_protected_values_structural  # noqa: SLF001
    ):
        return lambda values: cast("R", cls._from_structural_fields(template, metadata | values))

    if source is not None:
        receiver = source
        if any(getattr(source, key, _UNSET) is not value for key, value in metadata.items()):
            # Reconstruction hooks read static metadata from their receiver.
            receiver = copy(source)
            for key, value in metadata.items():
                object.__setattr__(receiver, key, value)
        rebuild = receiver._with_protected_values_structural if can_structural else receiver.with_protected_values  # noqa: SLF001
        return lambda values: rebuild(values, first.func)

    init_metadata = (
        {f.name: metadata[f.name] for f in fields(cls) if f.init and f.name in metadata} if is_dataclass(cls) else {}
    )
    return lambda values: cast("R", cls(**init_metadata, **values))


def extract_protected_value_sequence_internals[V, R](
    values: tuple[object, ...],
    func: Callable | None = None,
    *,
    extract: Callable[[object, Callable | None], AxisProtectedInternals[V, R] | None],
    broadcast: Callable[[V, int, tuple[int, ...]], V] | None = None,
    compact: bool = True,
    structural: bool = True,
) -> ProtectedValueSequenceInternals[V, R]:
    """Align raw and protected operands in order, expanding only fields needed by the operation."""
    extracted = [extract(value, func) for value in values]
    inputs = [item for item in extracted if item is not None]
    if not inputs:
        return ProtectedValueSequenceInternals(False, None, {})

    first = inputs[0]
    axes = first.protected_axes
    require_effective_fields(axes)
    all_protected = len(inputs) == len(values)
    create = _sequence_creator(inputs, structural=structural and all_protected)

    shared = {
        name: first.values[name]
        for name in first.broadcasted_batch
        if compact
        and all_protected
        and value_ndim(first.values[name]) == axes[name]
        and all(item.values[name] is first.values[name] for item in inputs)
    }
    if shared.keys() == axes.keys():
        # One physical field must carry the resulting logical batch shape.
        del shared[next(iter(axes))]

    values_by_field = {}
    for name, count in axes.items():
        if name in shared:
            continue
        entries = []
        for raw, item in zip(values, extracted, strict=True):
            if item is None:
                entries.append(raw)
            elif broadcast is not None and name in item.broadcasted_batch:
                entries.append(broadcast(item.values[name], count, item.batch_shape))
            else:
                entries.append(item.values[name])
        values_by_field[name] = entries
    return ProtectedValueSequenceInternals(True, first, values_by_field, shared, create)


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


def numpy_index_field(value: np.ndarray, index: tuple[Any, ...], axes: int) -> Any:  # noqa: ANN401
    """Index a NumPy field while preserving the dtype and contents of scalar cells."""
    if axes == 0 and value.dtype.kind == "O":
        # A scalar index unboxes the Python object, which can itself look like a batch.
        # Retain a temporary trailing axis until NumPy has completed the selection.
        return value[..., None][(*index, slice(None))].squeeze(axis=-1)
    result = value[index]
    if axes == 0 and getattr(result, "dtype", None) != value.dtype:
        # NumPy scalars can lose information such as byte order or fixed string width.
        return np.asarray(result, dtype=value.dtype)
    return result


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


def like_field_shape(name: str, value: object, axes_count: int, batch_target_shape: tuple[int, ...]) -> tuple[int, ...]:
    """Combine a logical batch shape override with a field's protected trailing shape."""
    field_shape = value_shape(value)
    if len(field_shape) < axes_count:
        msg = f"Protected field {name!r} has ndim {len(field_shape)}, expected >= {axes_count}."
        raise ValueError(msg)
    return (*batch_target_shape, *protected_shape(field_shape, axes_count))


def expand_for_protected_axes[T](value: T, protected_axes_count: int) -> T:
    """Append trailing singleton axes so a non-scalar array aligns with logical batch axes.

    Scalars and 0D arrays are returned unchanged so scalar and 0D promotion semantics
    are preserved.
    """
    if protected_axes_count <= 0:
        return value
    ndim = getattr(value, "ndim", None)
    if not isinstance(ndim, int) or ndim == 0:
        return value
    return cast("Any", value)[(..., *(None,) * protected_axes_count)]


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


def normalize_reshape_shape(
    shape: object,
    *,
    array_types: type[Any] | tuple[type[Any], ...],
    error_message: str,
) -> tuple[Any, ...]:
    """Extract logical batch target dimensions without converting array data."""
    if isinstance(shape, (int, np.integer)):
        return (shape,)
    if isinstance(shape, (tuple, list)):
        return tuple(1 if dim is None else dim for dim in shape)
    if isinstance(shape, array_types):
        if shape.ndim == 0:
            return (shape,)
        if shape.ndim == 1:
            return tuple(shape)
        raise TypeError(error_message)
    if isinstance(shape, SupportsIndex):
        return (shape,)
    if isinstance(shape, Sequence) and not isinstance(shape, (str, bytes, bytearray)):
        return tuple(1 if dim is None else dim for dim in shape)
    raise TypeError(error_message)


def _validate_nested_where_parity(
    left_values: Mapping[str, object],
    right_values: Mapping[str, object],
    protected_axes: Mapping[str, int],
) -> bool:
    """Recursively verify that paired protected branches match nested protected layouts."""
    for name in protected_axes:
        left_val = left_values[name]
        right_val = right_values[name]
        if isinstance(left_val, _AxisProtectedBase) != isinstance(right_val, _AxisProtectedBase):
            return False
        if not isinstance(left_val, _AxisProtectedBase) or not isinstance(right_val, _AxisProtectedBase):
            continue
        if type(left_val) is not type(right_val):
            return False
        left_axes = left_val._effective_protected_axes  # noqa: SLF001
        right_axes = right_val._effective_protected_axes  # noqa: SLF001
        if left_axes != right_axes:
            msg = "All protected operands must share identical protected_axes definitions."
            raise ValueError(msg)
        if not _validate_nested_where_parity(left_val.protected_values(), right_val.protected_values(), left_axes):
            return False
    return True


def validate_where_pair_compatibility[V, R](
    template: AxisProtectedInternals[V, R],
    other: AxisProtectedInternals[V, R],
) -> bool:
    """Validate same-type layout, trailing protected shapes, and recursive nested parity."""
    if other.owner_type is not template.owner_type:
        return False
    if other.protected_axes != template.protected_axes:
        msg = "All protected operands must share identical protected_axes definitions."
        raise ValueError(msg)
    for name, axes_count in template.protected_axes.items():
        if protected_shape(value_shape(other.values[name]), axes_count) != protected_shape(
            value_shape(template.values[name]), axes_count
        ):
            msg = f"All protected operands must share identical protected trailing shapes for field {name!r}."
            raise ValueError(msg)
    return _validate_nested_where_parity(template.values, other.values, template.protected_axes)


def validate_where_result[V, R](
    values: Mapping[str, V],
    template: AxisProtectedInternals[V, R],
    expected_batch_shape: tuple[int, ...],
    *,
    allow_broadcasted: bool = False,
) -> None:
    """Validate ``where`` results against expected logical batch and protected trailing shapes."""
    if values.keys() != template.protected_axes.keys():
        msg = "where results must contain every protected field."
        raise ValueError(msg)
    actual_batch_shape = validate_batch_sync(
        values,
        template.protected_axes,
        template.broadcasted_batch if allow_broadcasted else frozenset(),
    )
    if actual_batch_shape != expected_batch_shape:
        msg = f"where produced batch shape {actual_batch_shape}, expected {expected_batch_shape}."
        raise ValueError(msg)
    for name, axes_count in template.protected_axes.items():
        if protected_shape(value_shape(values[name]), axes_count) != protected_shape(
            value_shape(template.values[name]), axes_count
        ):
            msg = f"where modified protected trailing axes for field {name!r}."
            raise ValueError(msg)
