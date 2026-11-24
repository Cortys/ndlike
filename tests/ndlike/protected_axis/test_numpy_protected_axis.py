"""Tests for generalized numpy protected-axis behavior."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
import pytest

from ndlike.protected_axis.numpy import NumpyAxisProtected


@dataclass(frozen=True, slots=True)
class SingleArray(NumpyAxisProtected):
    array: np.ndarray
    protected_axes: ClassVar[dict[str, int]] = {"array": 1}
    permitted_ufuncs: ClassVar[dict[np.ufunc, list[str]]] = {np.add: ["__call__"]}


@dataclass(frozen=True, slots=True)
class ReductionArray(NumpyAxisProtected):
    array: np.ndarray
    protected_axes: ClassVar[dict[str, int]] = {"array": 1}
    permitted_functions: ClassVar[set[Any]] = {np.mean, np.sum}
    permitted_ufuncs: ClassVar[dict[np.ufunc, list[str]]] = {np.add: ["reduce", "accumulate", "reduceat"]}


@dataclass(frozen=True, slots=True)
class PairArray(NumpyAxisProtected["PairArray", "PairArray", "PairArray", tuple[np.ndarray, np.ndarray]]):
    first: np.ndarray
    second: np.ndarray
    protected_axes: ClassVar[dict[str, int]] = {"first": 0, "second": 0}


@dataclass(frozen=True, slots=True)
class InnerPair(NumpyAxisProtected):
    left: np.ndarray
    right: np.ndarray
    protected_axes: ClassVar[dict[str, int]] = {"left": 1, "right": 1}


@dataclass(frozen=True, slots=True)
class OuterNested(NumpyAxisProtected):
    inner: InnerPair
    aux: np.ndarray
    protected_axes: ClassVar[dict[str, int]] = {"inner": 1, "aux": 1}


def test_single_field_array_and_ufunc() -> None:
    x = SingleArray(np.arange(6.0).reshape(2, 3))

    as_array = np.asarray(x)
    np.testing.assert_array_equal(as_array, x.array)

    shifted = np.add(x, 1.0)
    assert isinstance(shifted, SingleArray)
    np.testing.assert_array_equal(shifted.array, x.array + 1.0)


def test_unpermitted_reductions_return_notimplemented() -> None:
    x = SingleArray(np.arange(24.0).reshape(2, 3, 4))

    with pytest.raises(TypeError):
        _ = np.sum(x)

    with pytest.raises(TypeError):
        _ = np.mean(x)

    with pytest.raises(TypeError):
        _ = np.add.reduce(x, axis=0)


def test_permitted_sum_and_mean_reduce_only_batch_axes() -> None:
    x = ReductionArray(np.arange(24.0).reshape(2, 3, 4))

    summed = np.sum(x)
    assert isinstance(summed, ReductionArray)
    np.testing.assert_array_equal(summed.array, np.sum(x.array, axis=(0, 1)))
    assert summed.shape == ()
    assert summed.protected_shape == (4,)

    meaned = np.mean(x, axis=1)
    assert isinstance(meaned, ReductionArray)
    np.testing.assert_array_equal(meaned.array, np.mean(x.array, axis=1))
    assert meaned.shape == (2,)
    assert meaned.protected_shape == (4,)


def test_permitted_ufunc_reductions_reduce_only_batch_axes() -> None:
    x = ReductionArray(np.arange(24.0).reshape(2, 3, 4))

    reduced = np.add.reduce(x, axis=None)
    assert isinstance(reduced, ReductionArray)
    np.testing.assert_array_equal(reduced.array, np.add.reduce(x.array, axis=(0, 1)))
    assert reduced.shape == ()
    assert reduced.protected_shape == (4,)

    accumulated = np.add.accumulate(x, axis=0)
    assert isinstance(accumulated, ReductionArray)
    np.testing.assert_array_equal(accumulated.array, np.add.accumulate(x.array, axis=0))
    assert accumulated.protected_shape == (4,)

    reduced_at = np.add.reduceat(x, [0, 2], axis=1)
    assert isinstance(reduced_at, ReductionArray)
    np.testing.assert_array_equal(reduced_at.array, np.add.reduceat(x.array, [0, 2], axis=1))
    assert reduced_at.protected_shape == (4,)


def test_multi_field_metadata_comes_from_first_field() -> None:
    x = PairArray(
        first=np.ones((2, 3), dtype=np.float64),
        second=np.ones((2, 3), dtype=np.float32),
    )

    assert x.shape == (2, 3)
    assert x.ndim == 2
    assert x.size == 6
    assert x.dtype == np.float64


def test_multi_field_array_and_ufunc_raise_by_default() -> None:
    x = PairArray(np.ones((2, 2)), np.ones((2, 2)))

    with pytest.raises(TypeError, match="Cannot convert multi-field"):
        _ = np.asarray(x)

    with pytest.raises(TypeError):
        _ = np.add(x, 1)


def test_shape_functions_apply_to_all_fields() -> None:
    x = PairArray(np.arange(6.0).reshape(2, 3), np.arange(6.0).reshape(2, 3) + 10)

    expanded = np.expand_dims(x, axis=0)
    assert isinstance(expanded, PairArray)
    assert expanded.first.shape == (1, 2, 3)
    assert expanded.second.shape == (1, 2, 3)

    reshaped = np.reshape(x, (3, 2))
    assert isinstance(reshaped, PairArray)
    assert reshaped.first.shape == (3, 2)
    assert reshaped.second.shape == (3, 2)


def test_reshape_method_supports_list_sequence_order_and_copy() -> None:
    x = SingleArray(np.arange(24.0).reshape(2, 3, 4))

    for result in (x.reshape(3, 2), x.reshape((3, 2)), x.reshape([3, 2]), x.reshape(range(3, 1, -1))):
        assert isinstance(result, SingleArray)
        assert result.shape == (3, 2)
        assert result.protected_shape == (4,)
        np.testing.assert_array_equal(result.array, x.array.reshape(3, 2, 4))

    copied = x.reshape([3, 2], copy=True)
    assert isinstance(copied, SingleArray)
    assert not np.shares_memory(copied.array, x.array)
    np.testing.assert_array_equal(copied.array, x.array.reshape(3, 2, 4))

    viewed = x.reshape((3, 2), copy=False)
    assert isinstance(viewed, SingleArray)
    assert np.shares_memory(viewed.array, x.array)

    transposed = x.swapaxes(0, 1)
    assert isinstance(transposed, SingleArray)
    with pytest.raises(ValueError, match="copy"):
        transposed.reshape([6], copy=False)

    f_ordered = x.reshape([3, 2], order="F", copy=True)
    assert isinstance(f_ordered, SingleArray)
    np.testing.assert_array_equal(f_ordered.array, np.reshape(x.array, (3, 2, 4), order="F", copy=True))

    with pytest.raises(ValueError, match="order"):
        x.reshape((3, 2), order="Z")


def test_reshape_method_retains_scalar_ndarray_index_and_legacy_none() -> None:
    class _Dim:
        def __init__(self, value: int) -> None:
            self.value = value

        def __index__(self) -> int:
            return self.value

    x = SingleArray(np.arange(24.0).reshape(2, 3, 4))

    for scalar_shape in (np.int64(6), np.array(6), _Dim(6)):
        reshaped = x.reshape(scalar_shape)  # ty: ignore[invalid-argument-type]
        assert isinstance(reshaped, SingleArray)
        assert reshaped.shape == (6,)
        assert reshaped.protected_shape == (4,)
        np.testing.assert_array_equal(reshaped.array, x.array.reshape(6, 4))

    vec_reshaped = x.reshape(np.array([3, 2]))  # ty: ignore[invalid-argument-type]
    assert isinstance(vec_reshaped, SingleArray)
    assert vec_reshaped.shape == (3, 2)
    assert vec_reshaped.protected_shape == (4,)
    np.testing.assert_array_equal(vec_reshaped.array, x.array.reshape(3, 2, 4))

    with pytest.raises(TypeError, match="reshape newshape"):
        x.reshape(np.array([[3, 2]]))  # ty: ignore[invalid-argument-type]

    for singleton in (SingleArray(np.arange(4.0)), SingleArray(np.arange(4.0).reshape(1, 4))):
        legacy_none = singleton.reshape(None)  # ty: ignore[invalid-argument-type]
        assert isinstance(legacy_none, SingleArray)
        assert legacy_none.shape == (1,)
        assert legacy_none.protected_shape == (4,)
        np.testing.assert_array_equal(legacy_none.array, singleton.array.reshape(1, 4))

    with pytest.raises(ValueError, match="cannot reshape"):
        x.reshape(None)  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize(
    ("axes", "shape"),
    [
        ((), (2, 3)),
        ((0, 1), (1, 1, 2, 3)),
        ((0, 2), (1, 2, 1, 3)),
        ((0, 3), (1, 2, 3, 1)),
        ((1, 3), (2, 1, 3, 1)),
        ((2, 3), (2, 3, 1, 1)),
        ((3, 1), (2, 1, 3, 1)),
        ((0, -1), (1, 2, 3, 1)),
        ((-4, -1), (1, 2, 3, 1)),
    ],
)
def test_expand_dims_uses_positions_in_the_output_batch_shape(axes, shape) -> None:
    x = SingleArray(np.arange(24.0).reshape(2, 3, 4))
    result = np.expand_dims(x, axis=axes)
    assert isinstance(result, SingleArray)
    assert result.shape == shape
    assert result.protected_shape == (4,)
    np.testing.assert_array_equal(result.array, x.array.reshape((*shape, 4)))


@pytest.mark.parametrize("axes", [(0, 4), (0, -5), (0, 0), (0, -4)])
def test_expand_dims_rejects_out_of_range_and_repeated_batch_axes(axes) -> None:
    x = SingleArray(np.zeros((2, 3, 4)))
    with pytest.raises(ValueError, match=r"out of bounds|repeated axis"):
        np.expand_dims(x, axis=axes)


def test_expand_dims_inserts_multiple_axes_into_a_scalar_batch() -> None:
    x = SingleArray(np.arange(4.0))
    result = np.expand_dims(x, axis=(0, 1))
    assert isinstance(result, SingleArray)
    assert result.shape == (1, 1)
    np.testing.assert_array_equal(result.array, x.array[None, None, :])


def test_getitem_and_setitem_for_instance_and_tuple() -> None:
    x = PairArray(np.zeros((2, 2)), np.ones((2, 2)))

    y = x[0]
    assert isinstance(y, PairArray)
    assert y.first.shape == (2,)
    assert y.second.shape == (2,)

    x[0] = PairArray(np.array([5.0, 6.0]), np.array([7.0, 8.0]))
    np.testing.assert_array_equal(x.first[0], np.array([5.0, 6.0]))
    np.testing.assert_array_equal(x.second[0], np.array([7.0, 8.0]))

    x[1] = (np.array([1.0, 2.0]), np.array([3.0, 4.0]))
    np.testing.assert_array_equal(x.first[1], np.array([1.0, 2.0]))
    np.testing.assert_array_equal(x.second[1], np.array([3.0, 4.0]))


def test_iteration_yields_indexed_protected_objects() -> None:
    x = PairArray(np.arange(6.0).reshape(2, 3), np.arange(6.0).reshape(2, 3) + 10)

    items = list(x)

    assert len(items) == 2
    assert all(isinstance(item, PairArray) for item in items)
    np.testing.assert_array_equal(items[0].first, np.array([0.0, 1.0, 2.0]))
    np.testing.assert_array_equal(items[0].second, np.array([10.0, 11.0, 12.0]))
    np.testing.assert_array_equal(items[1].first, np.array([3.0, 4.0, 5.0]))
    np.testing.assert_array_equal(items[1].second, np.array([13.0, 14.0, 15.0]))


def test_nested_multi_field_value_delegates_without_array_casts() -> None:
    inner = InnerPair(
        left=np.arange(24.0).reshape(2, 3, 4),
        right=np.arange(24.0).reshape(2, 3, 4) + 100,
    )
    outer = OuterNested(inner=inner, aux=np.arange(10.0).reshape(2, 5))

    values = outer.protected_values()
    assert values["inner"] is inner

    expanded = np.expand_dims(outer, axis=0)
    assert isinstance(expanded, OuterNested)
    assert isinstance(expanded.inner, InnerPair)
    assert expanded.inner.left.shape == (1, 2, 3, 4)
    assert expanded.inner.right.shape == (1, 2, 3, 4)
    assert expanded.aux.shape == (1, 2, 5)

    stacked = np.stack((outer, outer), axis=0)
    assert isinstance(stacked, OuterNested)
    assert isinstance(stacked.inner, InnerPair)
    assert stacked.inner.left.shape == (2, 2, 3, 4)
    assert stacked.inner.right.shape == (2, 2, 3, 4)
    assert stacked.aux.shape == (2, 2, 5)

    sliced = outer[1]
    assert isinstance(sliced, OuterNested)
    assert isinstance(sliced.inner, InnerPair)
    assert sliced.inner.left.shape == (3, 4)
    assert sliced.inner.right.shape == (3, 4)
    assert sliced.aux.shape == (5,)


def test_swapaxes_and_squeeze_methods_preserve_protected_axes() -> None:
    x = SingleArray(np.arange(24.0).reshape(2, 3, 4))

    swapped = x.swapaxes(0, -1)
    assert isinstance(swapped, SingleArray)
    assert swapped.shape == (3, 2)
    assert swapped.protected_shape == (4,)
    np.testing.assert_array_equal(swapped.array, np.swapaxes(x.array, 0, 1))

    with pytest.raises(TypeError):
        x.swapaxes(axis1=0, axis2=1)  # ty: ignore[positional-only-parameter-as-kwarg]

    with pytest.raises(ValueError, match="out of bounds"):
        x.swapaxes(0, 2)

    singleton = SingleArray(np.arange(24.0).reshape(1, 2, 1, 3, 4))
    squeezed_all = singleton.squeeze()
    assert isinstance(squeezed_all, SingleArray)
    assert squeezed_all.shape == (2, 3)
    assert squeezed_all.protected_shape == (4,)
    assert singleton.squeeze(0).shape == (2, 1, 3)
    assert singleton.squeeze(axis=-2).shape == (1, 2, 3)
    assert singleton.squeeze((0, 2)).shape == (2, 3)
    assert singleton.squeeze(axis=None).shape == (2, 3)

    protected_singleton = SingleArray(np.arange(2.0).reshape(1, 2, 1))
    squeezed = protected_singleton.squeeze()
    assert isinstance(squeezed, SingleArray)
    assert squeezed.shape == (2,)
    assert squeezed.protected_shape == (1,)

    with pytest.raises(ValueError, match="not equal to one"):
        x.squeeze(axis=0)

    scalar = SingleArray(np.array([1.0]))
    for axis in (0, -1):
        result = scalar.squeeze(axis)
        assert isinstance(result, SingleArray)
        assert result.shape == ()
        assert result.array.shape == (1,)


def test_swapaxes_and_squeeze_methods_on_nested_and_broadcasted_layouts() -> None:
    inner = InnerPair(
        left=np.arange(24.0).reshape(1, 2, 3, 4),
        right=np.arange(24.0).reshape(1, 2, 3, 4) + 100,
    )
    outer = OuterNested(inner=inner, aux=np.arange(10.0).reshape(1, 2, 5))

    swapped = outer.swapaxes(0, 1)
    assert isinstance(swapped, OuterNested)
    assert isinstance(swapped.inner, InnerPair)
    assert swapped.shape == (2, 1)
    assert swapped.inner.left.shape == (2, 1, 3, 4)
    assert swapped.aux.shape == (2, 1, 5)

    squeezed = outer.squeeze(axis=0)
    assert isinstance(squeezed, OuterNested)
    assert squeezed.shape == (2,)
    assert squeezed.inner.left.shape == (2, 3, 4)
    assert squeezed.aux.shape == (2, 5)

    @dataclass(frozen=True, slots=True)
    class Shared(NumpyAxisProtected):
        compact: np.ndarray
        data: np.ndarray
        protected_axes: ClassVar[dict[str, int]] = {"compact": 1, "data": 1}
        broadcasted_batch: ClassVar[set[str]] = {"compact"}

    shared = Shared(np.ones(4), np.arange(24.0).reshape(1, 2, 3, 4))
    swapped_shared = shared.swapaxes(1, 2)
    assert isinstance(swapped_shared, Shared)
    assert swapped_shared.shape == (1, 3, 2)
    assert swapped_shared.compact.shape == (4,)
    squeezed_shared = shared.squeeze()
    assert isinstance(squeezed_shared, Shared)
    assert squeezed_shared.shape == (2, 3)
    assert squeezed_shared.compact.shape == (4,)
