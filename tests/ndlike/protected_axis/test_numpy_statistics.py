"""Variance and standard deviation reduce only visible NumPy batch axes."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import pytest

from ndlike.protected_axis.numpy import NumpyAxisProtected


@dataclass(frozen=True)
class Statistics(NumpyAxisProtected["Statistics"]):
    first: np.ndarray
    second: np.ndarray
    protected_axes: ClassVar[dict[str, int]] = {"first": 1, "second": 2}
    permitted_functions: ClassVar[set[Callable]] = {np.std, np.var, np.mean}


@dataclass(frozen=True)
class Single(NumpyAxisProtected["Single"]):
    array: np.ndarray
    protected_axes: ClassVar[dict[str, int]] = {"array": 1}
    permitted_functions: ClassVar[set[Callable]] = {np.std, np.var}


@pytest.mark.parametrize("func", [np.std, np.var])
@pytest.mark.parametrize(("axis", "mapped"), [(None, (0, 1)), (-1, 1), ((-2, -1), (0, 1)), ((), ())])
@pytest.mark.parametrize("keepdims", [False, True])
def test_statistics_preserve_each_protected_shape(func, axis, mapped, keepdims):
    x = Statistics(np.arange(24.0).reshape(2, 3, 4), np.arange(36.0).reshape(2, 3, 2, 3))
    result = func(x, axis=axis, keepdims=keepdims)
    assert isinstance(result, Statistics)
    np.testing.assert_allclose(result.first, func(x.first, axis=mapped, keepdims=keepdims))
    np.testing.assert_allclose(result.second, func(x.second, axis=mapped, keepdims=keepdims))


@pytest.mark.parametrize("func", [np.std, np.var])
def test_corrections_precomputed_mean_and_output(func):
    x = Statistics(np.arange(24.0).reshape(2, 3, 4), np.arange(36.0).reshape(2, 3, 2, 3))
    mean = np.mean(x, axis=0, keepdims=True)
    result = func(x, axis=0, correction=1, mean=mean)
    expected = [func(value, axis=0, ddof=1) for value in (x.first, x.second)]
    np.testing.assert_allclose(result.first, expected[0])
    np.testing.assert_allclose(result.second, expected[1])
    out = Statistics(np.empty_like(expected[0]), np.empty_like(expected[1]))
    assert func(x, 0, None, out, 1) is out
    np.testing.assert_allclose(out.first, expected[0])
    np.testing.assert_allclose(out.second, expected[1])


@pytest.mark.parametrize("func", [np.std, np.var])
def test_mask_dtype_and_native_output(func):
    x = Single(np.arange(24).reshape(2, 3, 4))
    mask = np.ones_like(x.array, dtype=bool)
    mask[0, 0, 0] = False
    expected = func(x.array, axis=0, dtype=np.float32, where=mask, ddof=0)
    out = np.empty_like(expected)
    assert func(x, axis=0, dtype=np.float32, where=mask, ddof=0, out=out) is out
    np.testing.assert_allclose(out, expected)


@pytest.mark.parametrize("func", [np.std, np.var])
def test_scalar_batch_preserves_components(func):
    x = Single(np.arange(4.0))
    result = func(x, keepdims=True)
    assert result.shape == ()
    np.testing.assert_array_equal(result.array, np.zeros(4))


@pytest.mark.parametrize("func", [np.std, np.var])
def test_permissions_and_protected_axis_rejection(func):
    @dataclass(frozen=True)
    class Denied(Single):
        permitted_functions: ClassVar[set[Callable]] = set()

    with pytest.raises(TypeError):
        func(Denied(np.ones((2, 3))))
    with pytest.raises(ValueError, match="out of bounds"):
        func(Single(np.ones((2, 3))), axis=1)
