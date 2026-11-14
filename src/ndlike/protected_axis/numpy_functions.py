"""NumPy array-function implementations for protected-axis values."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from flextype import switchdispatch
import numpy as np

from ndlike.array_like import NumpyLike
from ndlike.protected_axis._common import (
    AxisProtectedCreator,
    AxisProtectedInternals,
    BoundFunctionWithInternals,
    FunctionOverride,
    ProtectedValueSequenceInternals,
    _AxisProtectedBase,
    apply_unary as _apply_unary,
    extract_axis_protected_internals,
    extract_protected_value_sequence_internals,
    flattened_batch_shape,
    function_override as numpy_function_override,
    internals_override,
    map_batch_axes as _map_batch_axes,
    normalize_axes,
    normalize_axis,
    normalize_batch_reduction_axes as _normalize_batch_reduction_axes,
    protected_shape,
    require_effective_fields,
    validate_batch_sync,
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


def numpy_axis_protected_internals(
    obj: object, func: Callable | None = None, *, check_is_permitted: bool = False
) -> NumpyAxisProtectedInternals | None:
    """Extract protected-axis internals from object."""
    if not isinstance(obj, _AxisProtectedBase):
        return None
    # Dispatch establishes the backend value family.
    return extract_axis_protected_internals(
        obj,
        func,
        check_is_permitted=check_is_permitted,
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

    return _apply_unary(internals, lambda _name, value, _axes: func(value, order=order, subok=subok))


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
    out_internals = numpy_axis_protected_internals(out, None)
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

    return _apply_unary(internals, op)


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

    return _apply_unary(internals, op)


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

    if isinstance(shape, int):
        batch_target_shape = (shape,)
    else:
        if not isinstance(shape, (tuple, list)):
            msg = "reshape newshape must be an int, tuple, or list."
            raise TypeError(msg)
        batch_target_shape = tuple(1 if dim is None else dim for dim in shape)

    order = params.arguments.get("order", "C")
    copy = params.arguments.get("copy", None)

    def op(_name: str, value: NumpyProtectedValue, axes_count: int) -> NumpyProtectedValue:
        target_shape = (*batch_target_shape, *protected_shape(value_shape(value), axes_count))
        kwargs: dict[str, object] = {"order": order}
        if copy is not None:
            kwargs["copy"] = copy
        return func(value, target_shape, **kwargs)

    return _apply_unary(internals, op)


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

    return _apply_unary(internals, op)


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

    return _apply_unary(internals, op)


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

    return _apply_unary(internals, op)


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

    return _apply_unary(internals, op)


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

    out_internals = numpy_axis_protected_internals(out, None)
    sequence = _extract_protected_value_sequence_internals(values, None)
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

    out_internals = numpy_axis_protected_internals(out, None)
    sequence = _extract_protected_value_sequence_internals(values, None)
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


def _ufunc_result_shapes(
    ufunc: np.ufunc, method: str, args: list[Any], options: dict[str, Any]
) -> tuple[tuple[int, ...], ...]:
    if method == "__call__":
        if ufunc.signature is not None:
            # Generalized ufuncs can have arbitrary core-axis rules. Matmul uses
            # its dedicated handler; defer other signatures to the native ufunc.
            result = ufunc(*args, **{key: value for key, value in options.items() if key != "out"})
            items = result if ufunc.nout > 1 else (result,)
            return tuple(value_shape(value) for value in items)
        shapes = [np.shape(value) for value in args]
        if "where" in options:
            shapes.append(np.shape(options["where"]))
        return (np.broadcast_shapes(*shapes),) * ufunc.nout

    shape = value_shape(args[0])
    axis = options["axis"]
    if method == "reduce":
        dims = (axis,) if isinstance(axis, int) else axis
        return (
            tuple(
                1 if index in dims else size
                for index, size in enumerate(shape)
                if options.get("keepdims", False) or index not in dims
            ),
        )
    if method == "reduceat":
        shape = (*shape[:axis], len(args[1]), *shape[axis + 1 :])
    return (shape,)


def validate_ufunc_outputs(
    ufunc: np.ufunc,
    method: str,
    calls: dict[str, tuple[list[Any], dict[str, Any]]],
    axes: dict[str, int],
    reference: dict[str, NumpyProtectedValue],
) -> None:
    """Validate every result and destination shape before the first field writes."""
    batch_shapes: dict[int, tuple[int, ...]] = {}
    for name, (args, options) in calls.items():
        shapes = _ufunc_result_shapes(ufunc, method, args, options)
        out = options["out"]
        outputs = out if isinstance(out, tuple) else (out,)
        count = axes[name]
        for index, (shape, output) in enumerate(zip(shapes, outputs, strict=True)):
            if protected_shape(shape, count) != protected_shape(value_shape(reference[name]), count):
                msg = f"Ufunc operation modified protected trailing axes for field {name!r}."
                raise ValueError(msg)
            batch = shape[:-count] if count else shape
            if batch_shapes.setdefault(index, batch) != batch:
                msg = "Ufunc operation produced inconsistent batch-shapes across protected fields."
                raise ValueError(msg)
            if output is not None and value_shape(output) != shape:
                msg = f"out must match the ufunc result shape for field {name!r}."
                raise ValueError(msg)


def _operator_inputs(
    owner: AnyNumpyAxisProtected,
    operands: tuple[object, ...],
    reference: dict[str, NumpyProtectedValue],
    func: np.ufunc,
) -> list[dict[str, NumpyProtectedValue] | None] | None:
    """Check permission, concrete type, and protected shapes for every operand."""
    axes = owner._effective_protected_axes  # noqa: SLF001
    inputs: list[dict[str, NumpyProtectedValue] | None] = []
    for value in operands:
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


def _matmul_out_values(out: object, results: dict[str, Any]) -> dict[str, Any]:
    if isinstance(out, _AxisProtectedBase):
        values = out.protected_values()
    elif len(results) == 1:
        values = dict.fromkeys(results, out)
    else:
        msg = "non-protected out is only supported for single-field protected objects."
        raise TypeError(msg)
    for name, result in results.items():
        value = values[name]
        if value_shape(value) != value_shape(result):
            msg = f"out must match the matmul result shape for field {name!r}."
            raise ValueError(msg)
        if isinstance(value, np.ndarray) and not value.flags.writeable:
            msg = f"out is read-only for field {name!r}."
            raise ValueError(msg)
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
    values = {
        name: ufunc(
            *(arg if fields is None else fields[name] for arg, fields in zip(operands, inputs, strict=True)),
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
    inputs = _operator_inputs(owner, participants, reference, np.matmul)
    if inputs is None:
        return NotImplemented
    inputs = inputs[:2]

    def apply(name: str, field_options: dict[str, Any]) -> Any:  # noqa: ANN401
        field_args = tuple(
            arg if values is None else values[name] for arg, values in zip(operands, inputs, strict=True)
        )
        counts = tuple(0 if values is None else axes[name] for values in inputs)
        return _matmul_field(field_args, counts, field_options)

    results = {name: apply(name, options) for name in axes}
    _validate_operator_fields(results, axes, reference, np.matmul)
    if out is not None:
        out_values = _matmul_out_values(out, results)
        buffers = {
            name: apply(name, {**options, "out": np.copy(value, subok=True)}) for name, value in out_values.items()
        }
        for name, value in out_values.items():
            value[...] = buffers[name]
        return out
    processed = {
        name: owner._postprocess_ufunc_result(value, ufunc=np.matmul, method="__call__")  # noqa: SLF001
        if isinstance(value, np.ndarray)
        else value
        for name, value in results.items()
    }
    _validate_operator_fields(processed, axes, reference, np.matmul)
    return owner.with_protected_values(processed)
