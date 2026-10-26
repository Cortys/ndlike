"""Tests for ``NumpyAxisProtected`` indexing, ufunc, and conversion paths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
import pytest

from ndlike.protected_axis.numpy import NumpyAxisProtected


@dataclass(frozen=True, slots=True)
class _SingleProtected(NumpyAxisProtected[np.ndarray]):
    """Single-field representation with a 1D protected trailing axis."""

    array: np.ndarray
    protected_axes: ClassVar[dict[str, int]] = {"array": 1}
    permitted_ufuncs: ClassVar[dict[np.ufunc, list[str]]] = {
        np.add: ["__call__", "reduce", "accumulate", "reduceat"],
        np.multiply: ["__call__"],
        np.modf: ["__call__"],
        np.frexp: ["__call__"],
        np.divmod: ["__call__"],
    }


@dataclass(frozen=True, slots=True)
class _ScalarProtected(NumpyAxisProtected[np.ndarray]):
    """Single-field representation with no protected trailing axis."""

    array: np.ndarray
    protected_axes: ClassVar[dict[str, int]] = {"array": 0}
    permitted_ufuncs: ClassVar[dict[np.ufunc, list[str]]] = {
        np.add: ["__call__"],
        np.modf: ["__call__"],
        np.frexp: ["__call__"],
        np.divmod: ["__call__"],
    }


@dataclass(frozen=True, slots=True)
class _TwoFieldProtected(NumpyAxisProtected[np.ndarray]):
    """Two-field representation with one protected trailing axis."""

    left: np.ndarray
    right: np.ndarray
    protected_axes: ClassVar[dict[str, int]] = {"left": 1, "right": 1}
    permitted_ufuncs: ClassVar[dict[np.ufunc, list[str]]] = {
        np.add: ["__call__"],
        np.modf: ["__call__"],
        np.frexp: ["__call__"],
        np.divmod: ["__call__"],
    }


# ---------------------------------------------------------------------------
# protected_values: permitted vs unpermitted func / method.
# ---------------------------------------------------------------------------


def test_protected_values_no_args_returns_all_values() -> None:
    """``protected_values()`` without arguments returns all fields."""
    x = _SingleProtected(np.arange(6.0).reshape(2, 3))
    values = x.protected_values()
    assert "array" in values
    np.testing.assert_array_equal(values["array"], x.array)


def test_protected_values_with_permitted_func_returns_dict() -> None:
    """``protected_values(np.mean)`` returns the dict for permitted reductions."""

    @dataclass(frozen=True, slots=True)
    class _Reducer(NumpyAxisProtected[np.ndarray]):
        array: np.ndarray
        protected_axes: ClassVar[dict[str, int]] = {"array": 1}
        permitted_functions: ClassVar[set[Any]] = {np.mean}

    x = _Reducer(np.arange(6.0).reshape(2, 3))
    values = x.protected_values(np.mean)
    assert values is not None
    np.testing.assert_array_equal(values["array"], x.array)


def test_protected_values_with_unpermitted_func_returns_none() -> None:
    """``protected_values(np.sum)`` returns None when the function is not permitted."""
    x = _SingleProtected(np.arange(6.0).reshape(2, 3))
    assert x.protected_values(np.sum) is None


def test_protected_values_with_permitted_ufunc_method_returns_dict() -> None:
    """``protected_values(np.add, '__call__')`` returns the dict when permitted."""
    x = _SingleProtected(np.arange(6.0).reshape(2, 3))
    values = x.protected_values(np.add, "__call__")
    assert values is not None


def test_protected_values_with_unpermitted_ufunc_method_returns_none() -> None:
    """An unpermitted ``method`` returns None."""
    x = _SingleProtected(np.arange(6.0).reshape(2, 3))
    assert x.protected_values(np.multiply, "reduce") is None


def test_protected_values_with_unknown_ufunc_returns_none() -> None:
    """Ufuncs not declared in ``permitted_ufuncs`` return None."""
    x = _SingleProtected(np.arange(6.0).reshape(2, 3))
    assert x.protected_values(np.subtract, "__call__") is None


# ---------------------------------------------------------------------------
# with_protected_values: round-trip.
# ---------------------------------------------------------------------------


def test_with_protected_values_round_trip() -> None:
    """``with_protected_values`` returns an instance carrying the new values."""
    x = _SingleProtected(np.arange(6.0).reshape(2, 3))
    new_array = np.zeros((2, 3))

    y = x.with_protected_values({"array": new_array})

    assert isinstance(y, _SingleProtected)
    np.testing.assert_array_equal(y.array, new_array)
    assert y is not x


# ---------------------------------------------------------------------------
# Indexing: int, slice, ellipsis, tuple, boolean.
# ---------------------------------------------------------------------------


def test_getitem_int_index() -> None:
    """Integer indexing into the leading batch axis works."""
    x = _SingleProtected(np.arange(24.0).reshape(2, 3, 4))
    y = x[0]
    assert isinstance(y, _SingleProtected)
    assert y.array.shape == (3, 4)


def test_getitem_slice_index() -> None:
    """Slicing the leading batch axis preserves the protected axes."""
    x = _SingleProtected(np.arange(24.0).reshape(2, 3, 4))
    y = x[1:2]
    assert isinstance(y, _SingleProtected)
    assert y.array.shape == (1, 3, 4)


def test_getitem_ellipsis_index() -> None:
    """``Ellipsis`` indexing returns the same shape representation."""
    x = _SingleProtected(np.arange(24.0).reshape(2, 3, 4))
    y = x[...]
    assert isinstance(y, _SingleProtected)
    assert y.array.shape == x.array.shape


def test_getitem_tuple_index() -> None:
    """Tuple indexing across multiple batch axes works."""
    x = _SingleProtected(np.arange(24.0).reshape(2, 3, 4))
    y = x[0, 1]
    assert isinstance(y, _SingleProtected)
    assert y.array.shape == (4,)


def test_getitem_boolean_array_index() -> None:
    """Boolean-array indexing collapses to a 1D batch axis."""
    x = _SingleProtected(np.arange(24.0).reshape(2, 3, 4))
    mask = np.array([True, False])
    y = x[mask]
    assert isinstance(y, _SingleProtected)
    assert y.array.shape == (1, 3, 4)


def test_getitem_axes_zero_promotes_scalar_to_array() -> None:
    """For ``axes=0``, scalar results are promoted via ``np.asarray``."""
    x = _ScalarProtected(np.arange(6.0))
    y = x[0]
    assert isinstance(y, _ScalarProtected)
    assert y.array.ndim == 0


# ---------------------------------------------------------------------------
# __array_namespace__ and __array__
# ---------------------------------------------------------------------------


def test_array_namespace_returns_numpy() -> None:
    """``__array_namespace__`` delegates to numpy's own namespace."""
    x = _SingleProtected(np.arange(6.0).reshape(2, 3))
    ns = x.__array_namespace__()
    # numpy's namespace is the numpy module itself.
    assert ns is not None


def test_array_dunder_returns_ndarray_for_single_field() -> None:
    """``__array__`` returns a plain ndarray for single-field objects."""
    x = _SingleProtected(np.arange(6.0).reshape(2, 3))
    arr = np.asarray(x)
    assert isinstance(arr, np.ndarray)
    np.testing.assert_array_equal(arr, x.array)


def test_array_dunder_with_dtype_converts() -> None:
    """``__array__(dtype=...)`` converts dtype."""
    x = _SingleProtected(np.arange(6.0, dtype=np.float64).reshape(2, 3))
    arr = np.asarray(x, dtype=np.float32)
    assert arr.dtype == np.float32


def test_array_dunder_rejects_multi_field() -> None:
    """``__array__`` raises for multi-field representations."""
    x = _TwoFieldProtected(np.zeros((2, 3)), np.ones((2, 3)))
    with pytest.raises(TypeError, match="multi-field"):
        _ = np.asarray(x)


# ---------------------------------------------------------------------------
# Properties.
# ---------------------------------------------------------------------------


def test_size_property_for_zero_dim_returns_one() -> None:
    """``size`` returns 1 for zero-dim protected representations."""
    x = _SingleProtected(np.array([1.0, 2.0, 3.0]))
    assert x.size == 1


def test_dtype_device_flags_delegate_to_primary_field() -> None:
    """dtype, device, and flags delegate to the primary field's underlying ndarray."""
    arr = np.arange(6.0).reshape(2, 3)
    x = _SingleProtected(arr)
    assert x.dtype == arr.dtype
    # device on numpy is "cpu"
    assert x.device == "cpu"
    # The flags object is recreated each access; check semantics rather than identity.
    assert x.flags["C_CONTIGUOUS"] == arr.flags["C_CONTIGUOUS"]


# ---------------------------------------------------------------------------
# Ufunc dispatch: in/out, reduce, accumulate, reduceat.
# ---------------------------------------------------------------------------


def test_ufunc_returns_notimplemented_for_unknown_method() -> None:
    """Unknown ufunc methods (e.g. ``outer``, ``at``) return NotImplemented."""
    x = _SingleProtected(np.arange(6.0).reshape(2, 3))

    # np.add.outer does not match the limited set of accepted methods, so
    # numpy raises a TypeError when our handler returns NotImplemented.
    with pytest.raises(TypeError):
        _ = np.add.outer(x, x)


def test_ufunc_call_returns_protected_result() -> None:
    """Permitted ufunc ``__call__`` returns a protected result."""
    x = _SingleProtected(np.arange(6.0).reshape(2, 3))
    y = np.add(x, 1.0)
    assert isinstance(y, _SingleProtected)
    np.testing.assert_array_equal(y.array, x.array + 1.0)


def test_ufunc_with_protected_out_writes_in_place() -> None:
    """``out=`` with a protected buffer updates and returns the buffer itself."""
    x = _SingleProtected(np.arange(6.0).reshape(2, 3))
    out = _SingleProtected(np.zeros((2, 3)))

    result = np.add(x, 1.0, out=out)  # ty: ignore[no-matching-overload]
    assert result is out
    np.testing.assert_array_equal(out.array, x.array + 1.0)


def test_ufunc_with_raw_out_for_single_field() -> None:
    """``out=`` accepts a raw ndarray for single-field protected objects."""
    x = _SingleProtected(np.arange(6.0).reshape(2, 3))
    raw_out = np.zeros((2, 3))

    result = np.add(x, 1.0, out=raw_out)
    np.testing.assert_array_equal(raw_out, x.array + 1.0)
    assert result is raw_out


def test_ufunc_with_tuple_out_for_single_field() -> None:
    """``out=(buf,)`` (tuple) works for ``__call__`` ufunc method."""
    x = _SingleProtected(np.arange(6.0).reshape(2, 3))
    out = _SingleProtected(np.zeros((2, 3)))

    result = np.add(x, 1.0, out=(out,))  # ty: ignore[no-matching-overload]
    assert result is out
    np.testing.assert_array_equal(out.array, x.array + 1.0)


@pytest.mark.parametrize("cls", [_SingleProtected, _TwoFieldProtected])
@pytest.mark.parametrize("out_spec", [None, (False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize(("ufunc", "extra_inputs"), [(np.modf, ()), (np.frexp, ()), (np.divmod, (0.5,))])
def test_multi_output_ufunc_allocates_missing_outputs(cls, out_spec, ufunc, extra_inputs) -> None:
    """Each missing output is wrapped, while provided output identities are preserved."""
    values = {
        name: np.arange(2 * (3 + i), dtype=float).reshape(2, 3 + i) / 4 - 1.25
        for i, name in enumerate(cls.protected_axes)
    }
    x = cls(**values)
    expected = {name: ufunc(value, *extra_inputs) for name, value in values.items()}
    out = (
        tuple(
            cls(**{name: np.empty_like(items[index]) for name, items in expected.items()}) if provided else None
            for index, provided in enumerate(out_spec)
        )
        if out_spec is not None
        else None
    )

    result = ufunc(x, *extra_inputs) if out is None else ufunc(x, *extra_inputs, out=out)

    assert isinstance(result, tuple)
    assert len(result) == 2
    for index, item in enumerate(result):
        assert isinstance(item, cls)
        if out is not None and out[index] is not None:
            assert item is out[index]
        for name, items in expected.items():
            actual = getattr(item, name)
            np.testing.assert_array_equal(actual, items[index])
            assert actual.dtype == items[index].dtype


@pytest.mark.parametrize("provided_index", [0, 1])
@pytest.mark.parametrize(("ufunc", "extra_inputs"), [(np.modf, ()), (np.frexp, ()), (np.divmod, (0.5,))])
def test_multi_output_ufunc_with_raw_out(provided_index, ufunc, extra_inputs) -> None:
    """Single-field objects accept raw buffers alongside allocated protected outputs."""
    x = _SingleProtected(np.arange(6.0).reshape(2, 3) / 4)
    expected = ufunc(x.array, *extra_inputs)
    raw_out = np.empty_like(expected[provided_index])
    out = (raw_out, None) if provided_index == 0 else (None, raw_out)

    result = ufunc(x, *extra_inputs, out=out)

    assert result[provided_index] is raw_out
    allocated = result[1 - provided_index]
    assert isinstance(allocated, _SingleProtected)
    np.testing.assert_array_equal(raw_out, expected[provided_index])
    np.testing.assert_array_equal(allocated.array, expected[1 - provided_index])
    assert allocated.array.dtype == expected[1 - provided_index].dtype


@pytest.mark.parametrize(("ufunc", "extra_inputs"), [(np.modf, ()), (np.frexp, ()), (np.divmod, (0.5,))])
def test_multi_output_ufunc_promotes_scalars_per_output(ufunc, extra_inputs) -> None:
    """Scalar outputs with no protected axes are wrapped individually."""
    x = _ScalarProtected(np.array(-1.25))

    result = ufunc(x, *extra_inputs, out=(None, None))

    for item, expected in zip(result, ufunc(x.array, *extra_inputs), strict=True):
        assert isinstance(item, _ScalarProtected)
        assert item.array.ndim == 0
        np.testing.assert_array_equal(item.array, expected)
        assert item.array.dtype == expected.dtype


@pytest.mark.parametrize("cls", [_SingleProtected, _TwoFieldProtected])
def test_single_output_ufunc_with_none_out(cls) -> None:
    """A single ``None`` out slot allocates and returns a protected object, not a tuple."""
    x = cls(**{name: np.ones((2, 3)) for name in cls.protected_axes})

    result = np.add(x, 1.0, out=(None,))  # ty:ignore[no-matching-overload]

    assert isinstance(result, cls)
    for name in cls.protected_axes:
        np.testing.assert_array_equal(getattr(result, name), 2.0)


def test_ufunc_with_unprotected_out_for_multi_field_raises() -> None:
    """A raw ``out`` is rejected when the protected object has multiple fields."""
    x = _TwoFieldProtected(np.zeros((2, 3)), np.zeros((2, 3)))
    raw_out = np.zeros((2, 3))

    with pytest.raises(TypeError, match="non-protected out"):
        _ = np.add(x, 1.0, out=raw_out)


def test_ufunc_rejects_unpermitted_method() -> None:
    """A ufunc whose method isn't permitted leads to NotImplemented (raised by numpy)."""
    x = _SingleProtected(np.arange(6.0).reshape(2, 3))

    # multiply only permits __call__, not reduce.
    with pytest.raises(TypeError):
        _ = np.multiply.reduce(x, axis=0)


def test_ufunc_reduce_with_no_axis_kwarg_uses_default_zero() -> None:
    """``np.add.reduce(x)`` reduces along axis 0 by default."""
    x = _SingleProtected(np.arange(24.0).reshape(2, 3, 4))
    y = np.add.reduce(x)
    assert isinstance(y, _SingleProtected)
    np.testing.assert_array_equal(y.array, np.add.reduce(x.array, axis=0))


def test_ufunc_accumulate_default_axis_zero() -> None:
    """``np.add.accumulate(x)`` accumulates along axis 0 by default."""
    x = _SingleProtected(np.arange(24.0).reshape(2, 3, 4))
    y = np.add.accumulate(x)
    assert isinstance(y, _SingleProtected)


def test_ufunc_reduceat_with_indices() -> None:
    """``np.add.reduceat(x, indices)`` reduces blocks along axis 0."""
    x = _SingleProtected(np.arange(24.0).reshape(2, 3, 4))
    y = np.add.reduceat(x, [0, 1])
    assert isinstance(y, _SingleProtected)


def test_ufunc_reduce_rejects_invalid_axis_type() -> None:
    """``axis`` must be None, int, or tuple/list of ints."""
    x = _SingleProtected(np.arange(24.0).reshape(2, 3, 4))

    with pytest.raises(TypeError, match="reduce axis must be"):
        _ = np.add.reduce(x, axis="0")  # ty:ignore[invalid-argument-type]


def test_ufunc_accumulate_rejects_non_int_axis() -> None:
    """``accumulate`` requires an int axis."""
    x = _SingleProtected(np.arange(24.0).reshape(2, 3, 4))

    with pytest.raises(TypeError, match="accumulate axis must be an int"):
        _ = np.add.accumulate(x, axis=(0,))  # ty:ignore[invalid-argument-type]


# ---------------------------------------------------------------------------
# Iter dispatch.
# ---------------------------------------------------------------------------


def test_iteration_yields_indexed_objects() -> None:
    """Iterating walks the leading batch axis."""
    x = _SingleProtected(np.arange(12.0).reshape(3, 4))
    items = list(x)
    assert len(items) == 3
    for i, item in enumerate(items):
        assert isinstance(item, _SingleProtected)
        np.testing.assert_array_equal(item.array, x.array[i])
