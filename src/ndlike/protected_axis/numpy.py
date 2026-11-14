"""Utilities for representations with protected trailing array axes."""

from __future__ import annotations

from abc import ABC
from dataclasses import replace
from inspect import isabstract
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Never, Self, cast, overload, override

import numpy as np
from numpy.typing import NDArray

from ndlike.array_like import (
    AnyNumpyLike,
    ArrayAPIVersion,
    ArrayFlagsLike,
    NumpyLike,
    NumpyLikeImplementation,
    Order,
    ToIndices,
)
from ndlike.protected_axis._common import (
    _AxisProtectedBase,
    batch_shape,
    normalize_axes,
    normalize_axis,
    primary_protected_name,
    protected_shape,
    require_effective_fields,
    value_ndim,
    value_shape,
)
from ndlike.protected_axis.numpy_functions import (
    numpy_function,
    protected_comparison_ufunc,
    protected_matmul_ufunc,
    validate_ufunc_outputs,
    validate_ufunc_protected_shapes,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping
    from types import ModuleType

    from numpy.typing import DTypeLike


type AnyNumpyAxisProtected = NumpyAxisProtected[Any, Any, Any, Any, Any, Any, Any]  # noqa: BKN001 - broad backend protocol alias
type NumpyProtectedValue = NumpyLike[Any] | np.ndarray


def _validate_field_ndim(name: str, ndim: int, protected_axes: int) -> None:
    if ndim < protected_axes:
        msg = f"Protected field {name!r} has ndim {ndim}, expected >= {protected_axes}."
        raise ValueError(msg)


class NumpyAxisProtected[
    OwnerT: AnyNumpyLike = NumpyLike[Any],
    OpValueT = OwnerT,
    OpResultT = OwnerT,
    SetValueT = Never,
    ShapeResultT = OwnerT,
    CompResultT = NDArray[np.bool_],
    SetResultT = OwnerT,
](
    _AxisProtectedBase[NumpyProtectedValue, OwnerT | OpResultT | ShapeResultT | CompResultT | SetResultT],
    NumpyLikeImplementation[OwnerT, OpValueT, OpResultT, SetValueT, ShapeResultT, CompResultT, SetResultT],
    ABC,
):
    """ABC for representations with one or multiple protected array-like fields."""

    protected_axes: ClassVar[dict[str, int]] = {}
    permitted_ufuncs: ClassVar[dict[np.ufunc, list[str]]] = {}
    permitted_functions: ClassVar[set[Callable[..., Any]]] = set()
    primary_protected_name: ClassVar[str]

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)

        if cls is NumpyAxisProtected or isabstract(cls):
            return

        axes = cls.protected_axes
        if not axes:
            msg = f"{cls.__name__} must define protected_axes as a non-empty dict[str, int]."
            raise TypeError(msg)

        cls.primary_protected_name = primary_protected_name(axes)

        permitted_ufuncs = getattr(cls, "permitted_ufuncs", {})
        if not isinstance(permitted_ufuncs, dict):
            msg = f"{cls.__name__}.permitted_ufuncs must be a dict[np.ufunc, list[str]]."
            raise TypeError(msg)

        normalized_permitted_ufuncs: dict[np.ufunc, list[str]] = {}
        for ufunc, methods in permitted_ufuncs.items():
            if not isinstance(ufunc, np.ufunc):
                msg = f"{cls.__name__}.permitted_ufuncs keys must be numpy ufuncs."
                raise TypeError(msg)
            if not isinstance(methods, list) or not all(isinstance(method, str) for method in methods):
                msg = f"{cls.__name__}.permitted_ufuncs[{ufunc.__name__!r}] must be a list[str]."
                raise TypeError(msg)
            normalized_permitted_ufuncs[ufunc] = list(methods)
        cls.permitted_ufuncs = normalized_permitted_ufuncs

        permitted_functions = getattr(cls, "permitted_functions", set())
        if not isinstance(permitted_functions, set) or not all(callable(func) for func in permitted_functions):
            msg = f"{cls.__name__}.permitted_functions must be a set of callables."
            raise TypeError(msg)
        cls.permitted_functions = set(permitted_functions)

    @overload
    def protected_values(self, func: None = None, method: None = None) -> dict[str, NumpyProtectedValue]: ...

    @overload
    def protected_values(self, func: Callable, method: str | None = None) -> dict[str, NumpyProtectedValue] | None: ...

    def protected_values(
        self, func: Callable | None = None, method: str | None = None
    ) -> dict[str, NumpyProtectedValue] | None:
        """Read effective values, with optional NumPy function or ufunc permissions."""
        if func is None or method is None:
            return super().protected_values(func)
        if method not in type(self).permitted_ufuncs.get(func, ()):
            return None
        return self._postprocess_protected_values(super().protected_values(), func)

    def protected_value(self) -> NumpyProtectedValue:
        """Return the primary protected value."""
        return getattr(self, type(self).primary_protected_name)

    @overload
    def with_protected_values(
        self,
        values: dict[str, NumpyProtectedValue],
    ) -> Self: ...

    @overload
    def with_protected_values(
        self,
        values: dict[str, NumpyProtectedValue],
        func: Callable | None,
    ) -> OwnerT | OpResultT | ShapeResultT | CompResultT | SetResultT: ...

    def with_protected_values(
        self,
        values: dict[str, NumpyProtectedValue],
        func: Callable | None = None,  # noqa: ARG002
    ) -> OwnerT | OpResultT | ShapeResultT | CompResultT | SetResultT:
        """Return a copy with updated protected field values."""
        return replace(self, **values)  # ty:ignore[invalid-argument-type]

    @override
    def astype(
        self,
        dtype: DTypeLike,
        order: Order = "K",
        casting: Literal["no", "equiv", "safe", "same_kind", "unsafe"] = "unsafe",
        subok: bool = True,
        copy: bool = True,
    ) -> OpResultT:
        """Cast each protected field using ndarray.astype's casting and layout options."""
        values = self.protected_values()
        updates = {
            name: value.astype(dtype, order=order, casting=casting, subok=subok, copy=copy)
            for name, value in values.items()
        }
        if all(updates[name] is value for name, value in values.items()):
            return cast("OpResultT", self)
        return cast("OpResultT", self.with_protected_values(updates))

    @override
    def __len__(self) -> int:
        if self.ndim == 0:
            msg = "len() of unsized object"
            raise TypeError(msg)
        return self.shape[0]

    @override
    def __iter__(self) -> Iterator[ShapeResultT]:
        for index in range(len(self)):
            yield self[index]

    @override
    def __array_namespace__(self, /, *, api_version: ArrayAPIVersion | None = None) -> ModuleType:
        return self.protected_value().__array_namespace__(api_version=api_version)

    @override
    def __array__(self, dtype: DTypeLike | None = None, /, *, copy: bool | None = None) -> np.ndarray:
        if len(self._effective_protected_axes) != 1:
            msg = "Cannot convert multi-field protected object to a single numpy array."
            raise TypeError(msg)
        return np.asarray(next(iter(self.protected_values().values())), dtype=dtype, copy=copy)

    @override
    @property
    def dtype(self) -> np.dtype:
        return self.protected_value().dtype

    @override
    @property
    def device(self) -> str:
        return self.protected_value().device

    @property
    def protected_shape(self) -> tuple[int, ...]:
        """Protected trailing shape of the primary field."""
        primary_name = type(self).primary_protected_name
        axes = type(self).protected_axes[primary_name]
        return protected_shape(value_shape(self.protected_value()), axes)

    @override
    @property
    def flags(self) -> ArrayFlagsLike:
        return self.protected_value().flags

    def _index_with_protected_axes(self, index: ToIndices, protected_axes_count: int) -> tuple[Any, ...]:
        index_tuple = index if isinstance(index, tuple) else (index,)
        return (*index_tuple, *(slice(None),) * protected_axes_count)

    def _prepare_assignment_value(self, value: object) -> object:
        """Prepare a replacement value before field extraction and shape validation.

        Subclasses may convert the value to match destination metadata or raise
        if incompatible. Return the original value or a converted replacement
        without mutating the supplied object. Assignment retains destination metadata.
        """
        return value

    def _coerce_assignment_value(self, value: object) -> Mapping[str, object]:
        field_names = tuple(self._effective_protected_axes.keys())

        if isinstance(value, type(self)):
            candidate_values: Mapping[str, object] = value.protected_values()
        elif isinstance(value, tuple):
            if len(value) != len(field_names):
                msg = f"Expected tuple with {len(field_names)} values for assignment."
                raise TypeError(msg)
            candidate_values = dict(zip(field_names, value, strict=True))
        elif len(field_names) == 1:
            candidate_values = {field_names[0]: value}
        else:
            msg = "Assignment to multi-field protected object requires matching instance or value tuple."
            raise TypeError(msg)

        current_values = self.protected_values()
        for name, axes in self._effective_protected_axes.items():
            candidate = candidate_values[name]
            if axes == 0:
                continue

            candidate_ndim = value_ndim(candidate)
            candidate_shape = value_shape(candidate)
            _validate_field_ndim(name, candidate_ndim, axes)

            current_shape = value_shape(current_values[name])
            if protected_shape(candidate_shape, axes) != protected_shape(current_shape, axes):
                msg = f"Assigned value for field {name!r} modifies protected trailing axes."
                raise ValueError(msg)

        return candidate_values

    @override
    def __getitem__(self, index: ToIndices, /) -> ShapeResultT:
        require_effective_fields(self._effective_protected_axes)
        values = self.protected_values()
        indexed: dict[str, NumpyProtectedValue] = {}

        for name, axes in self._effective_protected_axes.items():
            full_index = self._index_with_protected_axes(index, axes)
            result = values[name][full_index]

            if axes == 0 and not hasattr(result, "ndim"):
                result = np.asarray(result)

            result_ndim = value_ndim(result)
            result_shape = value_shape(result)
            _validate_field_ndim(name, result_ndim, axes)

            original_shape = value_shape(values[name])
            if protected_shape(result_shape, axes) != protected_shape(original_shape, axes):
                msg = f"Indexing field {name!r} modified protected trailing axes."
                raise IndexError(msg)

            indexed[name] = result

        return cast("ShapeResultT", self.with_protected_values(indexed))

    @override
    def __setitem__(self, index: ToIndices, value: Self | SetValueT, /) -> None:
        require_effective_fields(self._effective_protected_axes)
        prepared = self._prepare_assignment_value(value)
        values = self.protected_values()
        candidate_values = self._coerce_assignment_value(prepared)

        for name, axes in self._effective_protected_axes.items():
            full_index = self._index_with_protected_axes(index, axes)
            cast("Any", values[name])[full_index] = candidate_values[name]

    def _postprocess_ufunc_result(
        self,
        result: np.ndarray,
        *,
        ufunc: np.ufunc,
        method: str,
    ) -> np.ndarray:
        """Optionally postprocess wrapped ufunc results; comparisons bypass this hook."""
        del ufunc, method
        return result

    def __array_ufunc__(  # noqa: C901, PLR0912, PLR0915
        self,
        ufunc: np.ufunc,
        method: str,
        *inputs: object,
        **kwargs: object,
    ) -> object:
        if ufunc is np.matmul and method == "__call__":
            return protected_matmul_ufunc(self, inputs, kwargs)
        if (
            ufunc in (np.equal, np.not_equal, np.less, np.less_equal, np.greater, np.greater_equal)
            and method == "__call__"
        ):
            return protected_comparison_ufunc(self, ufunc, inputs, kwargs)
        if method not in {"__call__", "reduce", "accumulate", "reduceat"}:
            return NotImplemented

        protected_axes = self._effective_protected_axes
        if method != "__call__":
            require_effective_fields(protected_axes)
        input_values: list[dict[str, NumpyProtectedValue] | None] = []
        for value in inputs:
            if isinstance(value, type(self)):
                value_protected_axes = value._effective_protected_axes  # noqa: SLF001
                if value_protected_axes != protected_axes:
                    msg = "All protected inputs must share identical protected_axes definitions."
                    raise ValueError(msg)
                val = value.protected_values(ufunc, method)
                if val is None:
                    return NotImplemented
                input_values.append(val)
            else:
                input_values.append(None)
        self_values = self.protected_values(ufunc, method)

        if self_values is None:
            return NotImplemented

        out = kwargs.get("out")
        out_items = out if isinstance(out, tuple) else ((out,) if out is not None else ())
        out_values: list[dict[str, NumpyProtectedValue] | None] = []
        for value in out_items:
            if isinstance(value, type(self)):
                value_protected_axes = value._effective_protected_axes  # noqa: SLF001
                if value_protected_axes != protected_axes:
                    msg = "All protected outputs must share identical protected_axes definitions."
                    raise ValueError(msg)
                out_values.append(value.protected_values())
            else:
                out_values.append(None)

        if len(protected_axes) != 1 and any(
            item is not None and values is None for item, values in zip(out_items, out_values, strict=True)
        ):
            msg = "non-protected out is only supported for single-field protected objects."
            raise TypeError(msg)

        validate_ufunc_protected_shapes(protected_axes, self_values, [*input_values, *out_values])

        def map_reduce_axis(axis: object, batch_ndim: int) -> int | tuple[int, ...]:
            if axis is None:
                return tuple(range(batch_ndim))
            if isinstance(axis, int):
                return normalize_axis(axis, batch_ndim)
            if isinstance(axis, (tuple, list)) and all(isinstance(item, int) for item in axis):
                axis_tuple = tuple(axis)
                return normalize_axes(axis_tuple, batch_ndim)
            msg = "reduce axis must be None, an int, or a tuple/list of ints."
            raise TypeError(msg)

        def map_indexed_axis(axis: object, batch_ndim: int, *, method_name: str) -> int:
            if not isinstance(axis, int):
                msg = f"{method_name} axis must be an int."
                raise TypeError(msg)
            return normalize_axis(axis, batch_ndim)

        output_count = ufunc.nout if method == "__call__" else 1
        results: list[dict[str, NumpyProtectedValue]] = [{} for _ in range(output_count)]
        field_calls: dict[str, tuple[list[Any], dict[str, Any]]] = {}
        for name, axes_count in protected_axes.items():
            field_kwargs = dict(kwargs)

            if out is not None:
                mapped_out_items = tuple(
                    field_values[name] if field_values is not None else item
                    for item, field_values in zip(out_items, out_values, strict=True)
                )
                if isinstance(out, tuple):
                    if method == "__call__":
                        field_kwargs["out"] = mapped_out_items
                    else:
                        if len(mapped_out_items) != 1:
                            msg = f"ufunc method {method!r} expects a single out value."
                            raise TypeError(msg)
                        field_kwargs["out"] = mapped_out_items[0]
                else:
                    field_kwargs["out"] = mapped_out_items[0]

            field_inputs = tuple(
                protected[name] if protected is not None else value
                for value, protected in zip(inputs, input_values, strict=True)
            )
            field_input_items = list(field_inputs)

            if method in {"reduce", "accumulate", "reduceat"}:
                batch_ndim = value_ndim(self_values[name]) - axes_count

                if method == "reduce":
                    raw_axis = field_kwargs.get("axis", field_input_items[1] if len(field_input_items) > 1 else 0)
                    field_kwargs["axis"] = map_reduce_axis(raw_axis, batch_ndim)
                    field_input_items = field_input_items[:1]
                elif method == "accumulate":
                    raw_axis = field_kwargs.get("axis", field_input_items[1] if len(field_input_items) > 1 else 0)
                    field_kwargs["axis"] = map_indexed_axis(raw_axis, batch_ndim, method_name="accumulate")
                    field_input_items = field_input_items[:1]
                else:
                    if len(field_input_items) < 2:
                        msg = "reduceat requires indices as its second argument."
                        raise TypeError(msg)
                    raw_axis = field_kwargs.get("axis", field_input_items[2] if len(field_input_items) > 2 else 0)
                    field_kwargs["axis"] = map_indexed_axis(raw_axis, batch_ndim, method_name="reduceat")
                    field_input_items = field_input_items[:2]

            field_calls[name] = (field_input_items, field_kwargs)

        if any(item is not None for item in out_items):
            validate_ufunc_outputs(ufunc, method, field_calls, protected_axes, self_values)

        for name, (field_input_items, field_kwargs) in field_calls.items():
            axes_count = protected_axes[name]
            result = getattr(ufunc, method)(*field_input_items, **field_kwargs)

            result_items = result if output_count > 1 else (result,)
            for index, result_item in enumerate(result_items):
                if out is not None and out_items[index] is not None:
                    continue

                item = result_item
                if axes_count == 0 and not hasattr(item, "ndim"):
                    item = np.asarray(item)

                result_ndim = value_ndim(item)
                result_shape = value_shape(item)
                original_shape = value_shape(self_values[name])
                if result_ndim < axes_count:
                    msg = f"Ufunc operation removed protected trailing axes for field {name!r}."
                    raise ValueError(msg)
                if protected_shape(result_shape, axes_count) != protected_shape(original_shape, axes_count):
                    msg = f"Ufunc operation modified protected trailing axes for field {name!r}."
                    raise ValueError(msg)

                if isinstance(item, np.ndarray):
                    item = self._postprocess_ufunc_result(item, ufunc=ufunc, method=method)

                results[index][name] = item

        outputs: list[object] = []
        for index, values in enumerate(results):
            if out is not None and out_items[index] is not None:
                outputs.append(out_items[index])
                continue

            expected_batch_shape: tuple[int, ...] | None = None
            for name, value in values.items():
                axes_count = protected_axes[name]
                current_batch_shape = batch_shape(value_shape(value), axes_count)
                if expected_batch_shape is None:
                    expected_batch_shape = current_batch_shape
                elif current_batch_shape != expected_batch_shape:
                    msg = "Ufunc operation produced inconsistent batch-shapes across protected fields."
                    raise ValueError(msg)

            outputs.append(self.with_protected_values(values))

        return tuple(outputs) if output_count > 1 else outputs[0]

    def __array_function__(
        self,
        func: Callable,
        types: tuple[type[Any], ...],
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> object:
        return numpy_function(func, types, args, kwargs)

    def reshape(self, *shape: int | tuple[int, ...], order: str = "C", copy: bool | None = None) -> ShapeResultT:
        """Return a copy with reshaped protected values."""
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]

        return np.reshape(self, shape, order=order, copy=copy)  # ty:ignore[no-matching-overload]
