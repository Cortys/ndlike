"""NumPy comparisons aggregate protected components into native batch masks."""

from __future__ import annotations

from dataclasses import dataclass
import operator
from typing import Any, ClassVar

import numpy as np
import pytest

from ndlike.protected_axis.numpy import NumpyAxisProtected

_COMPARISONS = [
    (operator.eq, np.equal),
    (operator.ne, np.not_equal),
    (operator.lt, np.less),
    (operator.le, np.less_equal),
    (operator.gt, np.greater),
    (operator.ge, np.greater_equal),
]


@dataclass(frozen=True)
class Pair(NumpyAxisProtected["Pair", Any]):
    first: np.ndarray
    second: np.ndarray
    protected_axes: ClassVar[dict[str, int]] = {"first": 1, "second": 2}
    permitted_ufuncs: ClassVar[dict[np.ufunc, list[str]]] = {func: ["__call__"] for _, func in _COMPARISONS}


@pytest.mark.parametrize(("op", "func"), _COMPARISONS)
@pytest.mark.parametrize("kind", ["scalar", "array", "protected"])
@pytest.mark.parametrize("reflected", [False, True])
def test_comparisons_reduce_components_and_fields(op, func, kind, reflected):
    x = Pair(np.array([[1, 2]]), np.array([[[3, 4]]]))
    other = {
        "scalar": 2,
        "array": np.array([2, 3]),
        "protected": Pair(np.array([[1, 2], [1, 3]]), np.array([[[3, 4]], [[4, 5]]])),
    }[kind]
    args = (other, x) if reflected else (x, other)
    reduce = np.any if func is np.not_equal else np.all
    combine = np.logical_or if func is np.not_equal else np.logical_and
    expected = []
    for name, count in Pair.protected_axes.items():
        field_args = tuple(getattr(arg, name) if isinstance(arg, Pair) else arg for arg in args)
        expected.append(reduce(func(*field_args), axis=tuple(range(-count, 0))))
    result = op(*args)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.bool_
    np.testing.assert_array_equal(result, combine(*expected))
    np.testing.assert_array_equal(func(*args), result)
    assert isinstance(func(*args, out=(None,)), np.ndarray)
    out = np.empty_like(result)
    assert func(*args, out=out) is out
    np.testing.assert_array_equal(out, result)


@pytest.mark.parametrize(("op", "func"), _COMPARISONS)
@pytest.mark.parametrize("shape", [(), (2,)])
def test_zero_protected_rank_and_scalar_batch_return_native_ndarrays(op, func, shape):
    del func

    @dataclass(frozen=True)
    class Scalar(NumpyAxisProtected["Scalar", Any]):
        array: np.ndarray
        protected_axes: ClassVar[dict[str, int]] = {"array": 0}
        permitted_ufuncs: ClassVar[dict[np.ufunc, list[str]]] = Pair.permitted_ufuncs

    x = Scalar(np.ones(shape))
    result = op(x, 1)
    assert isinstance(result, np.ndarray)
    assert result.shape == shape
    np.testing.assert_array_equal(result, op(x.array, 1))
    scalar = Pair(np.array([1, 2]), np.array([[3, 4]]))
    result = op(scalar, scalar)
    assert isinstance(result, np.ndarray)
    assert result.shape == ()
    assert result.dtype == np.bool_


@pytest.mark.parametrize(("op", "func"), _COMPARISONS)
def test_empty_protected_axes_use_all_and_any_identities(op, func):
    x = Pair(np.empty((2, 0)), np.empty((2, 0, 3)))
    np.testing.assert_array_equal(op(x, x), np.full(2, func is not np.not_equal))


def test_masks_apply_to_visible_batch_and_preserve_existing_output():
    x = Pair(np.ones((2, 3)), np.ones((2, 2, 3)))
    out = np.zeros(2, dtype=bool)
    assert np.equal(x, x, out=out, where=[False, True]) is out
    np.testing.assert_array_equal(out, [False, True])
    expanded = np.zeros((2, 2), dtype=bool)
    assert np.equal(x, x, out=expanded, where=[[True], [False]]) is expanded
    np.testing.assert_array_equal(expanded, [[True, True], [False, False]])


def test_output_casting_matches_native_comparisons():
    x = Pair(np.ones((2, 3)), np.ones((2, 2, 3)))
    out = np.zeros(2, dtype=np.int64)
    assert np.equal(x, x, out=out) is out
    np.testing.assert_array_equal(out, [1, 1])
    with pytest.raises(TypeError, match="cast"):
        np.equal(x, x, out=out, casting="no")


def test_invalid_outputs_and_operand_shapes_do_not_mutate():
    x = Pair(np.ones((2, 3)), np.ones((2, 2, 3)))
    out = np.zeros((2, 3), dtype=bool)
    with pytest.raises(ValueError, match="resulting batch shape"):
        np.equal(x, x, out=out)
    np.testing.assert_array_equal(out, np.zeros_like(out))
    protected_out: Any = x
    with pytest.raises(ValueError, match="native ndarray"):
        np.equal(x, x, out=protected_out)
    bad = Pair(np.ones((2, 1)), np.ones((2, 2, 3)))
    with pytest.raises(ValueError, match="identical protected trailing shapes"):
        np.equal(x, bad)
    with pytest.raises(ValueError, match="modified protected trailing axes"):
        np.equal(bad, np.ones(3))
    inconsistent = Pair(np.ones((1, 3)), np.ones((1, 2, 3)))
    with pytest.raises(ValueError, match="inconsistent batch-shapes"):
        np.equal(inconsistent, np.ones((2, 3)))


def test_permissions_and_concrete_types():
    @dataclass(frozen=True)
    class Denied(Pair):
        permitted_ufuncs: ClassVar[dict[np.ufunc, list[str]]] = {}

    x = Pair(np.ones((2, 3)), np.ones((2, 2, 3)))
    denied = Denied(x.first, x.second)
    with pytest.raises(TypeError):
        np.equal(denied, denied)
    with pytest.raises(TypeError):
        np.equal(x, denied)


def test_comparisons_bypass_arithmetic_postprocessing_and_reconstruction():
    @dataclass(frozen=True)
    class NoReconstruction(Pair):
        def _postprocess_ufunc_result(self, result, *, ufunc, method):
            del result, ufunc, method
            pytest.fail("Comparisons must bypass arithmetic postprocessing.")

        def with_protected_values(self, values, func=None):
            del values, func
            pytest.fail("Comparisons must return a native mask without reconstruction.")

    x = NoReconstruction(np.ones((2, 3)), np.ones((2, 2, 3)))
    np.testing.assert_array_equal(operator.eq(x, x), [True, True])
