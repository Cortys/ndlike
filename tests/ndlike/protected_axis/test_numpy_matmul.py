"""NumPy matrix multiplication over visible dimensions, including ufunc outputs."""

from __future__ import annotations

from dataclasses import dataclass
import operator
from typing import Any, ClassVar

import numpy as np
import pytest

from ndlike.protected_axis.numpy import NumpyAxisProtected

# NumPy stubs only accept native arrays as ufunc outputs.
matmul: Any = np.matmul


@dataclass(frozen=True)
class Matrix(NumpyAxisProtected["Matrix", Any]):
    array: np.ndarray
    label: str = "original"
    protected_axes: ClassVar[dict[str, int]] = {"array": 1}
    permitted_ufuncs: ClassVar[dict[np.ufunc, list[str]]] = {np.matmul: ["__call__"]}


@dataclass(frozen=True)
class Pair(NumpyAxisProtected["Pair", Any]):
    first: np.ndarray
    second: np.ndarray
    protected_axes: ClassVar[dict[str, int]] = {"first": 1, "second": 2}
    permitted_ufuncs: ClassVar[dict[np.ufunc, list[str]]] = {np.matmul: ["__call__"]}


@pytest.mark.parametrize("left_shape", [(3,), (2, 3), (4, 2, 3)])
@pytest.mark.parametrize("right_shape", [(3,), (3, 5), (4, 3, 5)])
def test_matmul_uses_visible_matrix_and_vector_dimensions(left_shape, right_shape):
    x = Matrix(np.arange(np.prod(left_shape) * 2.0).reshape((*left_shape, 2)))
    y = Matrix(np.arange(np.prod(right_shape) * 2.0).reshape((*right_shape, 2)))
    expected = np.stack([x.array[..., i] @ y.array[..., i] for i in range(2)], axis=-1)
    result = x @ y
    assert isinstance(result, Matrix)
    np.testing.assert_array_equal(result.array, expected)
    for protected in (False, True):
        raw = np.empty_like(expected)
        out = Matrix(raw, "destination") if protected else raw
        assert matmul(x, y, out=out) is out
        np.testing.assert_array_equal(raw, expected)
    assert isinstance(matmul(x, y, out=(None,)), Matrix)


@pytest.mark.parametrize("count", [0, 1, 2])
@pytest.mark.parametrize("size", [0, 2])
def test_raw_operands_batched_broadcasting_and_empty_protected_axes(count, size):
    @dataclass(frozen=True)
    class Protected(NumpyAxisProtected["Protected", Any]):
        array: np.ndarray
        protected_axes: ClassVar[dict[str, int]] = {"array": count}
        permitted_ufuncs: ClassVar[dict[np.ufunc, list[str]]] = {np.matmul: ["__call__"]}

    x = Protected(np.ones((2, 3, *((size,) * count))))
    result = x @ np.ones((4, 3, 5))
    assert isinstance(result, Protected)
    assert result.shape == (4, 2, 5)
    np.testing.assert_array_equal(result.array, np.full((4, 2, 5, *((size,) * count)), 3.0))
    out = Protected(np.empty_like(result.array))
    assert matmul(x, np.ones((4, 3, 5)), out=out) is out
    np.testing.assert_array_equal(out.array, result.array)
    reflected = np.ones((4, 6, 2)) @ x
    assert isinstance(reflected, Protected)
    np.testing.assert_array_equal(reflected.array, np.full((4, 6, 3, *((size,) * count)), 2.0))


def test_multifield_inplace_preserves_storage_and_metadata():
    x = Pair(np.ones((2, 3, 4)), np.ones((2, 3, 4, 5)))
    original, first, second = x, x.first, x.second
    x @= np.eye(3) * 2
    assert x is original
    assert x.first is first
    assert x.second is second
    np.testing.assert_array_equal(first, np.full_like(first, 2))
    np.testing.assert_array_equal(second, np.full_like(second, 2))
    one = Matrix(np.ones((2, 3, 4)), "retained")
    assert operator.imatmul(one, np.eye(3)) is one
    assert one.label == "retained"


def test_raw_vectors_and_lists():
    x = Matrix(np.arange(12.0).reshape(3, 4))
    expected = np.stack([np.dot(x.array[:, i], [1, 2, 3]) for i in range(4)])
    for result in (x @ [1, 2, 3], [1, 2, 3] @ x):
        assert isinstance(result, Matrix)
        assert result.shape == ()
        np.testing.assert_array_equal(result.array, expected)


def test_invalid_protected_shapes_output_sizes_and_axes_do_not_mutate():
    x = Matrix(np.ones((2, 3, 1)))
    with pytest.raises(ValueError, match="identical protected trailing shapes"):
        _ = x @ Matrix(np.ones((3, 5, 4)))
    with pytest.raises(ValueError, match="result shape"):
        operator.imatmul(x, np.ones((3, 5)))
    np.testing.assert_array_equal(x.array, np.ones((2, 3, 1)))
    with pytest.raises(ValueError, match="visible dimension"):
        _ = Matrix(np.ones(4)) @ Matrix(np.ones(4))
    with pytest.raises(ValueError, match="core-axis overrides"):
        matmul(x, np.ones((3, 5)), axes=[(-2, -1), (-2, -1), (-2, -1)])


def test_matmul_forwards_dtype_casting_and_validates_every_output_before_writing():
    x = Pair(np.ones((2, 3, 4)), np.ones((2, 3, 4, 5)))
    converted = matmul(x, np.eye(3), dtype=np.float32)
    assert isinstance(converted, Pair)
    assert converted.first.dtype == converted.second.dtype == np.float32
    out = Pair(np.zeros_like(x.first), np.zeros_like(x.second, dtype=np.int64))
    with pytest.raises(TypeError, match="cast"):
        matmul(x, np.eye(3) * 0.5, out=out)
    np.testing.assert_array_equal(out.first, np.zeros_like(x.first))
    np.testing.assert_array_equal(out.second, np.zeros_like(x.second, dtype=np.int64))
    assert matmul(x, np.eye(3) * 0.5, out=out, casting="unsafe") is out
    np.testing.assert_array_equal(out.first, np.full_like(x.first, 0.5))
    np.testing.assert_array_equal(out.second, np.zeros_like(out.second))
    with pytest.raises(TypeError, match="single-field"):
        matmul(x, np.eye(3), out=np.zeros_like(x.first))


def test_aliased_fields_use_original_inputs():
    storage = np.ones((2, 3, 4))
    x = Pair(storage, storage[..., None])
    assert operator.imatmul(x, np.eye(3) * 2) is x
    np.testing.assert_array_equal(x.first, np.full_like(storage, 2))
    np.testing.assert_array_equal(x.second, np.full_like(x.second, 2))


def test_permissions_and_concrete_types():
    @dataclass(frozen=True)
    class Denied(Matrix):
        permitted_ufuncs: ClassVar[dict[np.ufunc, list[str]]] = {}

    x = Matrix(np.ones((2, 3, 4)))
    with pytest.raises(TypeError):
        _ = Denied(x.array) @ np.eye(3)
    with pytest.raises(TypeError):
        _ = x @ Denied(np.ones((3, 2, 4)))
    with pytest.raises(TypeError):
        matmul(x, np.eye(3), out=Denied(np.zeros_like(x.array)))


def test_result_postprocessing_cannot_modify_protected_shapes():
    @dataclass(frozen=True)
    class Broken(Matrix):
        def _postprocess_ufunc_result(self, result, *, ufunc, method):
            assert ufunc is np.matmul
            assert method == "__call__"
            return result[..., :1]

    with pytest.raises(ValueError, match="protected trailing axes"):
        _ = Broken(np.ones((2, 3, 4))) @ np.eye(3)


def test_output_only_dispatch_and_noncontiguous_output():
    storage = np.zeros((4, 2, 3))
    out = Matrix(np.moveaxis(storage, 0, -1))
    left = np.ones((2, 5))
    right = np.ones((5, 3, 4))
    # With no protected input, the raw operands use their native matmul shapes.
    raw_left = np.ones((2, 5))
    raw_right = np.ones((5, 3))
    raw = matmul(raw_left, raw_right)
    np.testing.assert_array_equal(raw, np.full((2, 3), 5.0))
    assert matmul(Matrix(np.repeat(left[..., None], 4, axis=-1)), Matrix(right), out=out) is out
    np.testing.assert_array_equal(out.array, np.full((2, 3, 4), 5.0))
    only_out = Matrix(np.empty((2, 3)))
    assert matmul(raw_left, raw_right, out=only_out) is only_out
    np.testing.assert_array_equal(only_out.array, raw)


def test_readonly_later_output_does_not_mutate_earlier_fields():
    x = Pair(np.ones((2, 3, 4)), np.ones((2, 3, 4, 5)))
    out = Pair(np.zeros_like(x.first), np.zeros_like(x.second))
    out.second.flags.writeable = False
    with pytest.raises(ValueError, match="read-only"):
        matmul(x, np.eye(3), out=out)
    np.testing.assert_array_equal(out.first, np.zeros_like(x.first))
