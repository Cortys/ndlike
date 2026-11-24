"""NumPy array-function implementations for protected-axis values."""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING, Any, cast

from flextype import switchdispatch
import numpy as np

from ndlike._batch import normalize_like_shape
from ndlike.array_like import NumpyLike
from ndlike.protected_axis._common import (
    AxisProtectedCreator,
    AxisProtectedInternals,
    BoundFunctionWithInternals,
    FunctionOverride,
    ProtectedValueSequenceInternals,
    _AxisProtectedBase,
    apply_shape_only as _apply_shape_only,
    apply_unary as _apply_unary,
    batch_shape,
    expand_for_protected_axes,
    extract_axis_protected_internals,
    extract_protected_value_sequence_internals,
    flattened_batch_shape,
    function_override as numpy_function_override,
    internals_override,
    like_field_shape,
    map_batch_axes as _map_batch_axes,
    normalize_axes,
    normalize_axis,
    normalize_batch_reduction_axes as _normalize_batch_reduction_axes,
    normalize_reshape_shape,
    prepare_broadcasted_batch,
    protected_shape,
    require_effective_fields,
    validate_batch_sync,
    validate_where_pair_compatibility,
    validate_where_result,
    value_ndim,
    value_shape,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from inspect import BoundArguments

    from ndlike.protected_axis.numpy import AnyNumpyAxisProtected


type NumpyProtectedValue = NumpyLike[Any] | np.ndarray


type NumpyAxisProtectedCreator = AxisProtectedCreator[NumpyProtectedValue, Any]
type NumpyAxisProtectedInternals = AxisProtectedInternals[NumpyProtectedValue, Any]
type _NumpyBoundFunctionWithInternals = BoundFunctionWithInternals[NumpyProtectedValue, Any]


_SHAPE_ONLY = frozenset(
    (
        np.reshape,
        np.transpose,
        np.matrix_transpose,
        np.swapaxes,
        np.moveaxis,
        np.squeeze,
        np.expand_dims,
        np.broadcast_to,
    )
)
_VALUE_ONLY = frozenset((np.copy, np.astype, np.zeros_like, np.ones_like, np.full_like))


def _broadcast_batch(value: NumpyProtectedValue, axes: int, shape: tuple[int, ...]) -> NumpyProtectedValue:
    """Broadcast a field's batch while retaining its protected suffix."""
    target = (*shape, *protected_shape(value_shape(value), axes))
    return value if value_shape(value) == target else np.broadcast_to(value, target)


def _broadcast_values(
    obj: AnyNumpyAxisProtected, values: dict[str, NumpyProtectedValue]
) -> dict[str, NumpyProtectedValue]:
    """Prepare ufunc operands using the same logical batch as structural dispatch."""
    if not obj.broadcasted_batch:
        return values
    shape = obj._protected_batch_shape()  # noqa: SLF001
    return {
        name: _broadcast_batch(value, obj._effective_protected_axes[name], shape)  # noqa: SLF001
        for name, value in values.items()
    }


def numpy_axis_protected_internals(
    obj: object, func: Callable | None = None, *, check_is_permitted: bool = False, broadcast: bool = True
) -> NumpyAxisProtectedInternals | None:
    """Extract protected-axis internals from object."""
    if not isinstance(obj, _AxisProtectedBase):
        return None
    # Dispatch establishes the backend value family.
    internals = extract_axis_protected_internals(obj, func, check_is_permitted=check_is_permitted)
    if internals is None or not broadcast:
        return internals
    return prepare_broadcasted_batch(
        internals,
        shape_only=func in _SHAPE_ONLY,
        value_only=func in _VALUE_ONLY,
        broadcast=_broadcast_batch,
        reshape=np.reshape,
    )


def _extract_protected_value_sequence_internals(
    values: tuple[object, ...], func: Callable | None
) -> ProtectedValueSequenceInternals[NumpyProtectedValue, Any]:
    """Extract and align protected values for sequence operations."""
    return extract_protected_value_sequence_internals(values, func, extract=numpy_axis_protected_internals)


@switchdispatch
def numpy_function(
    func: Callable,
    types: tuple[type[Any], ...],  # noqa: ARG001
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:  # noqa: ANN401
    """Implementation of NumPy array functions for protected-axis objects."""
    del func, args, kwargs
    return NotImplemented


def numpy_internals_override(
    array_param_name: str,
    check_is_permitted: bool = False,
    *,
    allow_empty: bool = False,
) -> Callable[[_NumpyBoundFunctionWithInternals], FunctionOverride]:
    """Decorator for functions that operate on one protected-axis argument."""
    return internals_override(
        array_param_name, check_is_permitted, extract=numpy_axis_protected_internals, allow_empty=allow_empty
    )


@numpy_function.register(np.copy)
@numpy_internals_override("a", allow_empty=True)
def protected_copy_function(
    func: Callable,
    params: BoundArguments,
    internals: NumpyAxisProtectedInternals,
) -> Any:  # noqa: ANN401
    order = params.arguments.get("order", "C")
    subok = params.arguments.get("subok", True)

    if not subok:
        if len(internals.protected_axes) != 1:
            msg = "Cannot copy multi-field protected object with subok=False."
            raise TypeError(msg)
        return func(next(iter(internals.values.values())), order=order, subok=subok)

    return _apply_unary(
        internals,
        lambda _name, value, _axes: func(value, order=order, subok=subok),
        structural=True,
    )


@numpy_function.register(np.astype)
@numpy_internals_override("x", allow_empty=True)
def protected_astype_function(
    func: Callable,
    params: BoundArguments,
    internals: NumpyAxisProtectedInternals,
) -> Any:  # noqa: ANN401
    dtype = params.arguments["dtype"]
    copy = params.arguments.get("copy", True)
    return _apply_unary(internals, lambda _name, value, _axes: func(value, dtype, copy=copy))


@numpy_function.multi_register([np.zeros_like, np.ones_like, np.full_like])
@numpy_internals_override("a", allow_empty=True)
def protected_like_function(
    func: Callable,
    params: BoundArguments,
    internals: NumpyAxisProtectedInternals,
) -> Any:  # noqa: ANN401
    dtype = params.arguments.get("dtype", None)
    order = params.arguments.get("order", "K")
    subok = params.arguments.get("subok", True)
    shape = params.arguments.get("shape", None)
    device = params.arguments.get("device", None)
    batch_target_shape = None if shape is None else normalize_like_shape(shape)

    if batch_target_shape is not None:
        require_effective_fields(internals.protected_axes)

    def build(name: str, value: NumpyProtectedValue, axes_count: int) -> NumpyProtectedValue:
        if batch_target_shape is not None:
            field_shape: tuple[int, ...] | None = like_field_shape(name, value, axes_count, batch_target_shape)
        elif (
            name in internals.broadcasted_batch and batch_shape(value_shape(value), axes_count) != internals.batch_shape
        ):
            field_shape = like_field_shape(name, value, axes_count, internals.batch_shape)
        else:
            field_shape = None
        field_kwargs: dict[str, Any] = {
            "dtype": dtype,
            "order": order,
            "subok": subok,
            "shape": field_shape,
            "device": device,
        }
        if func is np.full_like:
            return func(value, params.arguments["fill_value"], **field_kwargs)
        return func(value, **field_kwargs)

    if not subok:
        if len(internals.protected_axes) != 1:
            func_name = getattr(func, "__name__", repr(func))
            msg = f"Cannot apply {func_name} to a multi-field protected object with subok=False."
            raise TypeError(msg)
        name, axes_count = next(iter(internals.protected_axes.items()))
        return build(name, internals.values[name], axes_count)

    return _apply_unary(internals, build)


def _reduction_mean_values(
    mean: object, internals: NumpyAxisProtectedInternals
) -> dict[str, NumpyProtectedValue] | None:
    mean_internals = numpy_axis_protected_internals(mean)
    if mean_internals is None:
        return None
    if mean_internals.protected_axes != internals.protected_axes:
        msg = "mean must use the same protected_axes layout as input values."
        raise ValueError(msg)
    return mean_internals.values


@numpy_function.multi_register([np.mean, np.sum, np.average, np.std, np.var])
@numpy_internals_override("a", check_is_permitted=True)
def protected_batch_reduction_function(  # noqa: PLR0912
    func: Callable,
    params: BoundArguments,
    internals: NumpyAxisProtectedInternals,
) -> Any:  # noqa: ANN401
    axis = params.arguments.get("axis", None)
    out = params.arguments.get("out", None)
    out_internals = numpy_axis_protected_internals(out, broadcast=False)
    if out_internals is not None and out_internals.protected_axes != internals.protected_axes:
        msg = "out must use the same protected_axes layout as input values."
        raise ValueError(msg)

    if out is not None and out_internals is None and len(internals.protected_axes) != 1:
        msg = "non-protected out is only supported for single-field protected objects."
        raise TypeError(msg)

    mean_values = _reduction_mean_values(params.arguments.get("mean"), internals)
    results: dict[str, NumpyProtectedValue] = {}
    for name, axes_count in internals.protected_axes.items():
        value = internals.values[name]
        batch_ndim = value_ndim(value) - axes_count
        mapped_axis = _normalize_batch_reduction_axes(axis, batch_ndim)

        field_kwargs: dict[str, object] = {}
        for key, field_value in params.arguments.items():
            if key == "a":
                continue
            if key == "axis":
                field_kwargs[key] = mapped_axis
                continue
            if key == "out":
                if out is None:
                    field_kwargs[key] = None
                elif out_internals is not None:
                    field_kwargs[key] = out_internals.values[name]
                else:
                    field_kwargs[key] = out
                continue
            field_kwargs[key] = mean_values[name] if key == "mean" and mean_values is not None else field_value

        result = func(value, **field_kwargs)

        if out is not None:
            continue

        if axes_count == 0 and not hasattr(result, "ndim"):
            result = np.asarray(result)

        original_shape = value_shape(value)
        result_shape = value_shape(result)
        if protected_shape(result_shape, axes_count) != protected_shape(original_shape, axes_count):
            msg = f"Reduction modified protected trailing axes for field {name!r}."
            raise ValueError(msg)

        results[name] = result

    if out is not None:
        return out

    return internals.create(results)


@numpy_function.register(np.transpose)
@numpy_internals_override("a")
def protected_transpose_function(
    func: Callable,
    params: BoundArguments,
    internals: NumpyAxisProtectedInternals,
) -> Any:  # noqa: ANN401
    axes = params.arguments.get("axes", None)

    if axes is None:
        batch_axes = tuple(reversed(range(internals.batch_ndim)))
    else:
        if not isinstance(axes, (tuple, list)) or not all(isinstance(axis, int) for axis in axes):
            msg = "transpose axes must be a tuple/list of integers."
            raise TypeError(msg)
        batch_axes = tuple(axes)
        if len(batch_axes) != internals.batch_ndim:
            msg = "transpose axes must only refer to batch dimensions."
            raise ValueError(msg)

    def op(_name: str, value: NumpyProtectedValue, axes_count: int) -> NumpyProtectedValue:
        full_axes = _map_batch_axes(value, axes_count, batch_axes)
        return func(value, axes=full_axes)

    return _apply_shape_only(internals, op)


@numpy_function.register(np.matrix_transpose)
@numpy_internals_override("x")
def protected_matrix_transpose_function(
    func: Callable,
    params: BoundArguments,  # noqa: ARG001
    internals: NumpyAxisProtectedInternals,
) -> Any:  # noqa: ANN401
    del func

    if internals.batch_ndim < 2:
        msg = "matrix_transpose requires at least 2 batch dimensions."
        raise ValueError(msg)

    batch_axes = list(range(internals.batch_ndim))
    batch_axes[-2], batch_axes[-1] = batch_axes[-1], batch_axes[-2]

    def op(_name: str, value: NumpyProtectedValue, axes_count: int) -> NumpyProtectedValue:
        full_axes = _map_batch_axes(value, axes_count, tuple(batch_axes))
        return np.transpose(value, axes=full_axes)

    return _apply_shape_only(internals, op)


@numpy_function.register(np.reshape)
@numpy_internals_override("a")
def protected_reshape_function(
    func: Callable,
    params: BoundArguments,
    internals: NumpyAxisProtectedInternals,
) -> Any:  # noqa: ANN401
    shape = params.arguments.get("shape", params.arguments.get("newshape", None))
    if shape is None:
        return NotImplemented

    batch_target_shape = normalize_reshape_shape(
        shape,
        array_types=np.ndarray,
        error_message="reshape newshape must be an int, tuple, or list.",
    )

    order = params.arguments.get("order", "C")
    copy = params.arguments.get("copy", None)

    def op(_name: str, value: NumpyProtectedValue, axes_count: int) -> NumpyProtectedValue:
        target_shape = (*batch_target_shape, *protected_shape(value_shape(value), axes_count))
        kwargs: dict[str, object] = {"order": order}
        if copy is not None:
            kwargs["copy"] = copy
        return func(value, target_shape, **kwargs)

    structural = isinstance(order, str) and order.upper() in ("C", "F")
    return _apply_shape_only(internals, op, structural=structural)


@numpy_function.register(np.expand_dims)
@numpy_internals_override("a")
def protected_expand_dims_function(
    func: Callable,
    params: BoundArguments,
    internals: NumpyAxisProtectedInternals,
) -> Any:  # noqa: ANN401
    axis = params.arguments["axis"]
    if isinstance(axis, int):
        axis_tuple = (axis,)
    elif isinstance(axis, (tuple, list)) and all(isinstance(item, int) for item in axis):
        axis_tuple = tuple(axis)
    else:
        msg = "expand_dims axis must be an int or tuple/list of ints."
        raise TypeError(msg)

    def op(_name: str, value: NumpyProtectedValue, axes_count: int) -> NumpyProtectedValue:
        batch_ndim = value_ndim(value) - axes_count
        full_axes = normalize_axes(axis_tuple, batch_ndim + len(axis_tuple) - 1, allow_endpoint=True)
        return func(value, axis=full_axes)

    return _apply_shape_only(internals, op)


@numpy_function.register(np.squeeze)
@numpy_internals_override("a")
def protected_squeeze_function(
    func: Callable,
    params: BoundArguments,
    internals: NumpyAxisProtectedInternals,
) -> Any:  # noqa: ANN401
    axis = params.arguments.get("axis", None)

    def op(_name: str, value: NumpyProtectedValue, axes_count: int) -> NumpyProtectedValue:
        batch_ndim = value_ndim(value) - axes_count
        shape = value_shape(value)

        if batch_ndim == 0 and isinstance(axis, int) and axis in (-1, 0):
            return value
        if axis is None:
            squeeze_axes = tuple(i for i, size in enumerate(shape[:batch_ndim]) if size == 1)
        else:
            if isinstance(axis, int):
                axis_tuple = (axis,)
            elif isinstance(axis, (tuple, list)) and all(isinstance(item, int) for item in axis):
                axis_tuple = tuple(axis)
            else:
                msg = "squeeze axis must be an int or tuple/list of ints."
                raise TypeError(msg)

            squeeze_axes = normalize_axes(axis_tuple, batch_ndim)

        return func(value, axis=squeeze_axes)

    return _apply_shape_only(internals, op)


@numpy_function.register(np.swapaxes)
@numpy_internals_override("a")
def protected_swapaxes_function(
    func: Callable,
    params: BoundArguments,
    internals: NumpyAxisProtectedInternals,
) -> Any:  # noqa: ANN401
    axis1 = params.arguments["axis1"]
    axis2 = params.arguments["axis2"]
    if not isinstance(axis1, int) or not isinstance(axis2, int):
        msg = "swapaxes axis values must be integers."
        raise TypeError(msg)

    def op(_name: str, value: NumpyProtectedValue, axes_count: int) -> NumpyProtectedValue:
        batch_ndim = value_ndim(value) - axes_count
        full_axis1 = normalize_axis(axis1, batch_ndim)
        full_axis2 = normalize_axis(axis2, batch_ndim)
        return func(value, full_axis1, full_axis2)

    return _apply_shape_only(internals, op)


@numpy_function.register(np.moveaxis)
@numpy_internals_override("a")
def protected_moveaxis_function(
    func: Callable,
    params: BoundArguments,
    internals: NumpyAxisProtectedInternals,
) -> Any:  # noqa: ANN401
    source = params.arguments["source"]
    destination = params.arguments["destination"]

    if isinstance(source, int):
        source_tuple = (source,)
        source_was_int = True
    elif isinstance(source, (tuple, list)) and all(isinstance(item, int) for item in source):
        source_tuple = tuple(source)
        source_was_int = False
    else:
        msg = "moveaxis source must be an int or tuple/list of ints."
        raise TypeError(msg)

    if isinstance(destination, int):
        destination_tuple = (destination,)
        destination_was_int = True
    elif isinstance(destination, (tuple, list)) and all(isinstance(item, int) for item in destination):
        destination_tuple = tuple(destination)
        destination_was_int = False
    else:
        msg = "moveaxis destination must be an int or tuple/list of ints."
        raise TypeError(msg)

    def op(_name: str, value: NumpyProtectedValue, axes_count: int) -> NumpyProtectedValue:
        batch_ndim = value_ndim(value) - axes_count
        mapped_source = normalize_axes(source_tuple, batch_ndim)
        mapped_destination = normalize_axes(destination_tuple, batch_ndim)
        source_arg: int | tuple[int, ...] = mapped_source[0] if source_was_int else mapped_source
        destination_arg: int | tuple[int, ...] = mapped_destination[0] if destination_was_int else mapped_destination
        return func(value, source=source_arg, destination=destination_arg)

    return _apply_shape_only(internals, op)


@numpy_function.register(np.concatenate)
def protected_concatenate_function(
    func: Callable,
    types: tuple[type[Any], ...],  # noqa: ARG001
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:  # noqa: ANN401
    values = tuple(args[0])
    axis = kwargs.get("axis", args[1] if len(args) > 1 else 0)
    out = kwargs.get("out", args[2] if len(args) > 2 else None)

    out_internals = numpy_axis_protected_internals(out, broadcast=False)
    sequence = _extract_protected_value_sequence_internals(values, func)
    template = sequence.template if sequence.template is not None else out_internals
    if template is None:
        return NotImplemented

    if out_internals is not None and out_internals.protected_axes != template.protected_axes:
        msg = "out must use the same protected_axes layout as input values."
        raise ValueError(msg)

    if out is not None and out_internals is None and len(template.protected_axes) != 1:
        msg = "non-protected out is only supported for single-field protected objects."
        raise TypeError(msg)

    if axis is not None and not isinstance(axis, int):
        msg = "concatenate axis must be an int or None."
        raise TypeError(msg)

    field_kwargs = {key: value for key, value in kwargs.items() if key not in {"axis", "out"}}
    results: dict[str, NumpyProtectedValue] = {}
    for name, axes_count in template.protected_axes.items():
        if not sequence.has_protected:
            msg = "concatenate with protected out requires at least one protected input."
            raise TypeError(msg)

        field_values = sequence.values_by_field[name]
        if axis is None:
            # NumPy accepts a single concatenation axis, so collapse batch axes
            # first. An explicit size also handles zero-sized protected suffixes.
            flattened = []
            for field_value in field_values:
                shape = (
                    value_shape(field_value) if hasattr(field_value, "shape") else np.shape(cast("Any", field_value))
                )
                flattened.append(np.reshape(cast("Any", field_value), flattened_batch_shape(shape, axes_count)))
            field_values = flattened
            mapped_axis = 0
        else:
            batch_ndim = value_ndim(template.values[name]) - axes_count
            mapped_axis = normalize_axis(axis, batch_ndim)

        out_value = out_internals.values[name] if out_internals is not None else out
        result = func(field_values, axis=mapped_axis, out=out_value, **field_kwargs)
        if out_value is None:
            results[name] = result

    if out is not None:
        return out

    return template.create(results)


@numpy_function.register(np.stack)
@numpy_function_override
def protected_stack_function(
    func: Callable,
    params: BoundArguments,
) -> Any:  # noqa: ANN401
    values = tuple(params.arguments["arrays"])
    axis = params.arguments.get("axis", 0)
    out = params.arguments.get("out", None)

    out_internals = numpy_axis_protected_internals(out, broadcast=False)
    sequence = _extract_protected_value_sequence_internals(values, func)
    template = sequence.template if sequence.template is not None else out_internals
    if template is None:
        return NotImplemented

    if out_internals is not None and out_internals.protected_axes != template.protected_axes:
        msg = "out must use the same protected_axes layout as input values."
        raise ValueError(msg)

    if not isinstance(axis, int):
        msg = "stack axis must be an int."
        raise TypeError(msg)

    results: dict[str, NumpyProtectedValue] = {}
    for name, axes_count in template.protected_axes.items():
        if not sequence.has_protected:
            msg = "stack with protected out requires at least one protected input."
            raise TypeError(msg)

        field_values = sequence.values_by_field[name]
        batch_ndim = value_ndim(template.values[name]) - axes_count
        mapped_axis = normalize_axis(axis, batch_ndim, allow_endpoint=True)

        out_value = out_internals.values[name] if out_internals is not None else None
        result = func(field_values, axis=mapped_axis, out=out_value)
        if out_value is None:
            results[name] = result

    if out is not None:
        return out

    return template.create(results)


_NDARRAY_ARRAY_FUNCTION = getattr(np.ndarray, "__array_function__", None)


def _has_custom_array_function(value: object) -> bool:
    """Return whether ``value`` (or a nested sequence element) overrides ``__array_function__``."""
    if isinstance(value, (list, tuple)):
        return any(_has_custom_array_function(item) for item in value)
    override = getattr(type(value), "__array_function__", None)
    return override is not None and override is not _NDARRAY_ARRAY_FUNCTION


def _extract_where_condition(condition: object) -> tuple[object, tuple[int, ...]] | None:
    """Validate a boolean logical-batch condition for protected ``np.where``."""
    if isinstance(condition, _AxisProtectedBase):
        msg = "where condition must be a boolean array or scalar, not a protected-axis object."
        raise TypeError(msg)
    if _has_custom_array_function(condition):
        return None
    if isinstance(condition, (bool, np.bool_)):
        return condition, ()
    if isinstance(condition, np.ndarray):
        if not np.issubdtype(condition.dtype, np.bool_):
            msg = f"where condition must have boolean dtype, got {condition.dtype}."
            raise TypeError(msg)
        return condition, condition.shape
    if isinstance(condition, (list, tuple)):
        array = np.asarray(condition)
        if not np.issubdtype(array.dtype, np.bool_):
            msg = f"where condition must have boolean dtype, got {array.dtype}."
            raise TypeError(msg)
        return array, array.shape
    msg = "where condition must be a boolean array or scalar."
    raise TypeError(msg)


def _extract_protected_where_branch(
    arg: _AxisProtectedBase[Any, Any],
    func: Callable,
    template: NumpyAxisProtectedInternals | None,
) -> tuple[NumpyAxisProtectedInternals, None, tuple[int, ...]] | None:
    """Extract and validate one protected operand for ``np.where``."""
    if not callable(getattr(type(arg), "__array_function__", None)):
        return None
    internals = numpy_axis_protected_internals(arg, func, check_is_permitted=True)
    if internals is None:
        return None
    require_effective_fields(internals.protected_axes)
    validate_batch_sync(internals.values, internals.protected_axes)
    if template is not None and not validate_where_pair_compatibility(template, internals):
        return None
    return internals, None, internals.batch_shape


def _extract_where_branch(
    arg: object,
    func: Callable,
    template: NumpyAxisProtectedInternals | None,
) -> tuple[NumpyAxisProtectedInternals | None, object, tuple[int, ...]] | None:
    """Extract a protected branch or validate a raw logical-batch array/scalar branch."""
    if isinstance(arg, _AxisProtectedBase):
        return _extract_protected_where_branch(arg, func, template)
    if _has_custom_array_function(arg):
        return None
    if isinstance(arg, (bool, int, float, complex, np.generic)):
        return None, arg, ()
    if isinstance(arg, np.ndarray):
        return None, arg, arg.shape
    if isinstance(arg, (list, tuple)):
        array = np.asarray(arg)
        if array.dtype != object:
            return None, array, array.shape
    return None


@numpy_function.register(np.where)
def protected_where_function(
    func: Callable,
    types: tuple[type[Any], ...],  # noqa: ARG001
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:  # noqa: ANN401
    """Select logical-batch entries across protected fields via three-argument ``np.where``.

    Opt-in through ``permitted_functions`` (consistent with NumPy reductions) because
    selecting across batch entries or introducing a raw branch can violate representation
    invariants. Subclasses may postprocess inputs via ``_postprocess_protected_values``
    and outputs via ``with_protected_values(..., func=np.where)``.
    """
    if kwargs or len(args) != 3:
        if not kwargs and len(args) == 2:
            msg = "either both or neither of x and y should be given"
            raise ValueError(msg)
        return NotImplemented

    condition, x, y = args
    x_branch = _extract_where_branch(x, func, None)
    if x_branch is None:
        return NotImplemented
    x_internals, x_raw, x_shape = x_branch

    y_branch = _extract_where_branch(y, func, x_internals)
    if y_branch is None:
        return NotImplemented
    y_internals, y_raw, y_shape = y_branch

    template = x_internals if x_internals is not None else y_internals
    if template is None:
        return NotImplemented

    extracted_condition = _extract_where_condition(condition)
    if extracted_condition is None:
        return NotImplemented
    cond_value, cond_shape = extracted_condition

    expected_batch_shape = np.broadcast_shapes(cond_shape, x_shape, y_shape)

    results: dict[str, NumpyProtectedValue] = {}
    for name, axes_count in template.protected_axes.items():
        field_cond = expand_for_protected_axes(cond_value, axes_count)
        field_x = x_internals.values[name] if x_internals is not None else expand_for_protected_axes(x_raw, axes_count)
        field_y = y_internals.values[name] if y_internals is not None else expand_for_protected_axes(y_raw, axes_count)
        result = func(field_cond, field_x, field_y)
        if axes_count == 0 and not hasattr(result, "ndim"):
            result = np.asarray(result)
        results[name] = result

    validate_where_result(results, template, expected_batch_shape)
    return template.create(results)


def _matmul_field(args: tuple[Any, ...], counts: tuple[int, ...], options: dict[str, Any]) -> Any:  # noqa: ANN401
    """Use protected coordinates as independent batches of visible matrices."""
    arrays = tuple(arg if hasattr(arg, "ndim") else np.asanyarray(arg) for arg in args)
    ranks = tuple(value_ndim(arg) - count for arg, count in zip(arrays, counts, strict=True))
    if min(ranks) < 1:
        msg = "matmul requires at least one visible dimension per operand."
        raise ValueError(msg)
    count = max(counts)
    batch_ndim = max(max(rank - 2, 0) for rank in ranks)
    normalized = []
    for index, (arg, axes, rank) in enumerate(zip(arrays, counts, ranks, strict=True)):
        value = np.moveaxis(arg, tuple(range(-axes, 0)), tuple(range(axes))) if axes else arg
        if rank == 1:
            value = np.expand_dims(value, -2 if index == 0 else -1)
        shape = value_shape(value)
        prefix = shape[:axes] if axes else (1,) * count
        normalized.append(np.reshape(value, (*prefix, *((1,) * (batch_ndim - max(rank - 2, 0))), *shape[axes:])))
    mutable_options = dict(options)
    out = mutable_options.get("out")
    if out is not None:
        if count:
            out = np.moveaxis(out, tuple(range(-count, 0)), tuple(range(count)))
        if ranks[1] == 1:
            out = np.expand_dims(out, -1)
        if ranks[0] == 1:
            out = np.expand_dims(out, -2)
        mutable_options["out"] = out
    result = np.matmul(*normalized, **mutable_options)
    if ranks[0] == 1:
        result = np.squeeze(result, -2)
    if ranks[1] == 1:
        result = np.squeeze(result, -1)
    return np.moveaxis(result, tuple(range(count)), tuple(range(-count, 0))) if count else result


def validate_ufunc_protected_shapes(
    axes: dict[str, int],
    reference: dict[str, NumpyProtectedValue],
    operands: list[dict[str, NumpyProtectedValue] | None],
) -> None:
    """Require identical protected trailing shapes across inputs and outputs."""
    for values in operands:
        if values is None:
            continue
        for name, count in axes.items():
            if protected_shape(value_shape(values[name]), count) != protected_shape(
                value_shape(reference[name]), count
            ):
                msg = f"All protected operands must share identical protected trailing shapes for field {name!r}."
                raise ValueError(msg)


def _expand_raw_operand(value: object, protected_axes_count: int) -> object:
    """Align a raw NumPy array or Python sequence across logical batch axes."""
    if protected_axes_count <= 0:
        return value
    if isinstance(value, np.ndarray):
        return expand_for_protected_axes(value, protected_axes_count)
    if isinstance(value, (list, tuple)):
        return expand_for_protected_axes(np.asarray(value), protected_axes_count)
    return value


def _operator_inputs(
    owner: AnyNumpyAxisProtected,
    operands: tuple[object, ...],
    reference: dict[str, NumpyProtectedValue],
    func: np.ufunc,
    *,
    has_output: bool = False,
) -> list[dict[str, NumpyProtectedValue] | None] | None:
    """Check permission, concrete type, and protected shapes for every operand."""
    axes = owner._effective_protected_axes  # noqa: SLF001
    inputs: list[dict[str, NumpyProtectedValue] | None] = []
    for index, value in enumerate(operands):
        if not isinstance(value, _AxisProtectedBase):
            inputs.append(None)
            continue
        if type(value) is not type(owner):
            return None
        protected = cast("AnyNumpyAxisProtected", value)
        if protected._effective_protected_axes != axes:  # noqa: SLF001
            msg = "All protected operands must share identical protected_axes definitions."
            raise ValueError(msg)
        values = protected.protected_values(func, "__call__")
        if values is None:
            return None
        if not has_output or index < len(operands) - 1:
            values = _broadcast_values(protected, values)
        for name, count in axes.items():
            if protected_shape(value_shape(values[name]), count) != protected_shape(
                value_shape(reference[name]), count
            ):
                msg = f"All protected operands must share identical protected trailing shapes for field {name!r}."
                raise ValueError(msg)
        inputs.append(values)
    return inputs


def _validate_operator_fields(
    values: dict[str, Any], axes: dict[str, int], reference: dict[str, NumpyProtectedValue], func: np.ufunc
) -> None:
    validate_batch_sync(values, axes)
    for name, count in axes.items():
        if protected_shape(value_shape(values[name]), count) != protected_shape(value_shape(reference[name]), count):
            msg = f"{func.__name__} modified protected trailing axes for field {name!r}."
            raise ValueError(msg)


def _matmul_out_values(out: object, axes: dict[str, int]) -> dict[str, Any]:
    if isinstance(out, _AxisProtectedBase):
        values = out.protected_values()
    elif len(axes) == 1:
        values = dict.fromkeys(axes, out)
    else:
        msg = "non-protected out is only supported for single-field protected objects."
        raise TypeError(msg)
    return values


def protected_comparison_ufunc(
    owner: AnyNumpyAxisProtected, ufunc: np.ufunc, operands: tuple[object, ...], kwargs: dict[str, object]
) -> object:
    """Return one native boolean per visible batch entry; masks apply to that batch."""
    reference = owner.protected_values(ufunc, "__call__")
    if reference is None:
        return NotImplemented
    axes = owner._effective_protected_axes  # noqa: SLF001
    require_effective_fields(axes)
    inputs = _operator_inputs(owner, operands, reference, ufunc)
    if inputs is None:
        return NotImplemented
    options: dict[str, Any] = dict(kwargs)
    out = options.pop("out", None)
    if isinstance(out, tuple):
        (out,) = out
    where = options.pop("where", True)
    align_raw = any(fields is not None for fields in inputs)
    values = {
        name: ufunc(
            *(
                (_expand_raw_operand(arg, axes[name]) if align_raw else arg) if fields is None else fields[name]
                for arg, fields in zip(operands, inputs, strict=True)
            ),
            **options,
        )
        for name in axes
    }
    _validate_operator_fields(values, axes, reference, ufunc)
    reduce = np.any if ufunc is np.not_equal else np.all
    combine = np.logical_or if ufunc is np.not_equal else np.logical_and
    result = None
    for name, count in axes.items():
        field_result = reduce(values[name], axis=tuple(range(-count, 0)))
        result = field_result if result is None else combine(result, field_result)
    result = np.asarray(result, dtype=np.bool_)
    shape = np.broadcast_shapes(result.shape, np.shape(where))
    if out is not None and (not isinstance(out, np.ndarray) or out.shape != shape):
        msg = "comparison out must be a native ndarray with the resulting batch shape."
        raise ValueError(msg)
    # Apply the mask after component aggregation so masked output entries retain
    # their old values, without reducing uninitialized component comparisons.
    masked = np.equal(
        result,
        True,
        out=out,
        where=where,
        casting=options.get("casting", "same_kind"),
        order=options.get("order", "K"),
        subok=options.get("subok", True),
    )
    return masked if out is not None else np.asarray(masked)


def protected_matmul_ufunc(
    owner: AnyNumpyAxisProtected, operands: tuple[object, ...], kwargs: dict[str, object]
) -> object:
    """Apply matmul to visible dimensions and preserve protected output storage."""
    reference = owner.protected_values(np.matmul, "__call__")
    if reference is None:
        return NotImplemented
    axes = owner._effective_protected_axes  # noqa: SLF001
    require_effective_fields(axes)
    if len(operands) != 2:
        msg = "matmul expects two operands."
        raise TypeError(msg)
    options = dict(kwargs)
    out = options.pop("out", None)
    if isinstance(out, tuple):
        if len(out) != 1:
            msg = "matmul expects one output."
            raise ValueError(msg)
        out = out[0]
    if "axes" in options or "axis" in options:
        msg = "matmul core-axis overrides are unsupported for protected arrays; transpose the visible axes instead."
        raise ValueError(msg)
    participants = (*operands, out) if isinstance(out, _AxisProtectedBase) else operands
    inputs = _operator_inputs(owner, participants, reference, np.matmul, has_output=len(participants) > len(operands))
    if inputs is None:
        return NotImplemented
    inputs = inputs[:2]

    def apply(name: str, field_options: dict[str, Any]) -> Any:  # noqa: ANN401
        field_args = tuple(
            arg if values is None else values[name] for arg, values in zip(operands, inputs, strict=True)
        )
        counts = tuple(0 if values is None else axes[name] for values in inputs)
        return _matmul_field(field_args, counts, field_options)

    if out is not None:
        out_values = _matmul_out_values(out, axes)
        for name, value in out_values.items():
            apply(name, {**options, "out": value})
        return out
    results = {name: apply(name, options) for name in axes}
    _validate_operator_fields(results, axes, reference, np.matmul)
    processed = {
        name: owner._postprocess_ufunc_result(value, ufunc=np.matmul, method="__call__")  # noqa: SLF001
        if isinstance(value, np.ndarray)
        else value
        for name, value in results.items()
    }
    _validate_operator_fields(processed, axes, reference, np.matmul)
    return owner.with_protected_values(processed)


@numpy_function.register(np.broadcast_to)
@numpy_internals_override("array")
def protected_broadcast_to_function(
    func: Callable,
    params: BoundArguments,
    internals: NumpyAxisProtectedInternals,
) -> Any:  # noqa: ANN401
    """Broadcast visible batch axes without changing protected suffixes."""
    del func
    shape = tuple(params.arguments["shape"])
    return _apply_shape_only(internals, lambda _name, value, axes: _broadcast_batch(value, axes, shape))


@numpy_function.register(np.take_along_axis)
@numpy_internals_override("arr")
def protected_take_along_axis_function(
    func: Callable,
    params: BoundArguments,
    internals: NumpyAxisProtectedInternals,
) -> Any:  # noqa: ANN401
    """Gather protected entries along a logical batch axis."""
    indices = params.arguments["indices"]
    axis = params.arguments.get("axis", -1)

    if isinstance(indices, _AxisProtectedBase) or _has_custom_array_function(indices):
        return NotImplemented
    if axis is not None:
        axis = operator.index(axis)
    indices_array = np.asanyarray(indices)
    batch_ndim = 1 if axis is None else internals.batch_ndim
    if indices_array.ndim != batch_ndim:
        msg = "take_along_axis indices must have the same ndim as the protected object's batch dimensions."
        raise ValueError(msg)

    results: dict[str, NumpyProtectedValue] = {}
    for name, axes_count in internals.protected_axes.items():
        value = internals.values[name]
        if axis is None:
            value = np.reshape(value, flattened_batch_shape(value_shape(value), axes_count))
            mapped_axis = 0
        else:
            mapped_axis = normalize_axis(axis, value_ndim(value) - axes_count)

        field_index = indices_array
        if axes_count > 0:
            target_shape = (*indices_array.shape, *protected_shape(value_shape(value), axes_count))
            field_index = np.broadcast_to(expand_for_protected_axes(indices_array, axes_count), target_shape)

        results[name] = func(value, field_index, axis=mapped_axis)

    return internals.rebuild_structural(results)
