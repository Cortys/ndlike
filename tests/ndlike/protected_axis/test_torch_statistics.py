"""Statistical reductions preserve protected tensor components."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar

import pytest

pytest.importorskip("torch")
import torch
from torch.testing import assert_close

from ndlike.protected_axis.torch import TorchAxisProtected
from ndlike.torch_functions import torch_average


@dataclass(frozen=True)
class Statistics(TorchAxisProtected["Statistics"]):
    first: torch.Tensor
    second: torch.Tensor
    protected_axes: ClassVar[dict[str, int]] = {"first": 1, "second": 2}
    permitted_functions: ClassVar[set[Callable]] = {torch.std, torch.var, torch.mean, torch.sum, torch_average}


@dataclass(frozen=True)
class Single(TorchAxisProtected["Single"]):
    array: torch.Tensor
    protected_axes: ClassVar[dict[str, int]] = {"array": 1}
    permitted_functions: ClassVar[set[Callable]] = {torch.std, torch.var, torch.mean, torch.sum, torch_average}


@pytest.mark.parametrize("func", [torch.std, torch.var])
@pytest.mark.parametrize(("dim", "mapped"), [(None, (0, 1)), (-1, 1), ((-2, -1), (0, 1)), ((), (0, 1))])
@pytest.mark.parametrize("keepdim", [False, True])
def test_statistics_preserve_each_protected_shape(func, dim, mapped, keepdim):
    x = Statistics(torch.arange(24.0).reshape(2, 3, 4), torch.arange(36.0).reshape(2, 3, 2, 3))
    result = func(input=x, dim=dim, keepdim=keepdim)
    assert isinstance(result, Statistics)
    assert_close(result.first, func(x.first, dim=mapped, keepdim=keepdim))
    assert_close(result.second, func(x.second, dim=mapped, keepdim=keepdim))


@pytest.mark.parametrize("func", [torch.std, torch.var])
def test_correction_legacy_overloads_and_out(func):
    x = Statistics(torch.arange(24.0).reshape(2, 3, 4), torch.arange(36.0).reshape(2, 3, 2, 3))
    result = func(x, False)
    assert_close(result.first, func(x.first, dim=(0, 1), correction=0))
    assert_close(result.second, func(x.second, dim=(0, 1), correction=0))
    result = func(x, -1, False, True)
    expected = [func(value, dim=1, correction=0, keepdim=True) for value in (x.first, x.second)]
    assert_close(result.first, expected[0])
    assert_close(result.second, expected[1])
    out = Statistics(torch.empty_like(expected[0]), torch.empty_like(expected[1]))
    assert func(x, dim=1, correction=0, keepdim=True, out=out) is out
    assert_close(out.first, expected[0])
    assert_close(out.second, expected[1])


@pytest.mark.parametrize("func", [torch.std, torch.var])
@pytest.mark.parametrize("keepdim", [False, True])
def test_scalar_batch_preserves_components(func, keepdim):
    x = Single(torch.arange(4.0))
    out = Single(torch.empty_like(x.array))
    assert func(x, correction=0, keepdim=keepdim, out=out) is out
    assert out.shape == ()
    assert_close(out.array, torch.zeros(4))
    legacy = func(x, None, False, keepdim)
    assert_close(legacy.array, torch.zeros(4))
    with pytest.warns(UserWarning, match="degrees of freedom"):
        result = func(x)
    assert result.array.shape == (4,)
    assert torch.isnan(result.array).all()


@pytest.mark.parametrize("func", [torch.std, torch.var])
def test_native_output_and_reject_resizing(func):
    x = Single(torch.arange(24.0).reshape(2, 3, 4))
    expected = func(x.array, dim=(0, 1))
    out = torch.empty_like(expected)
    assert func(x, out=out) is out
    assert_close(out, expected)
    bad = Single(torch.zeros(3))
    with pytest.raises(ValueError, match="result shape"):
        func(x, out=bad)
    assert_close(bad.array, torch.zeros(3))


@pytest.mark.parametrize("func", [torch.std, torch.var])
@pytest.mark.parametrize("shape", [(4,), (2, 3, 4)])
@pytest.mark.parametrize("axis", [None, ()])
def test_numpy_style_keyword_aliases(func, shape, axis):
    x = Single(torch.ones(shape))
    expected = func(x, dim=axis, keepdim=True, correction=0)
    out = Single(torch.empty_like(expected.array))
    assert func(x, axis=axis, keepdims=True, correction=0, out=out) is out
    assert_close(out.array, expected.array)

    if len(shape) > 1:
        expected = func(x.array, dim=1, keepdim=True, correction=0)
        raw = torch.empty_like(expected)
        assert func(x, axis=-1, keepdims=True, correction=0, out=raw) is raw
        assert_close(raw, expected)


@pytest.mark.parametrize("func", [torch.std, torch.var])
def test_permissions_and_protected_axis_rejection(func):
    @dataclass(frozen=True)
    class Denied(Single):
        permitted_functions: ClassVar[set[Callable]] = set()

    with pytest.raises(TypeError):
        func(Denied(torch.ones((2, 3))))
    with pytest.raises(ValueError, match="out of bounds"):
        func(Single(torch.ones((2, 3))), dim=1)


@pytest.mark.parametrize("func", [torch.mean, torch.sum, torch_average])
@pytest.mark.parametrize("keepdim", [False, True])
@pytest.mark.parametrize("empty", [False, True])
def test_scalar_batch_mean_sum_average_preserve_fields_and_output(func, keepdim, empty):
    size = 0 if empty else 3
    x = Statistics(torch.arange(float(size)), torch.arange(float(2 * size)).reshape(2, size))
    result = func(x, dim=None, keepdim=keepdim)
    assert result.shape == ()
    assert_close(result.first, x.first)
    assert_close(result.second, x.second)
    default = func(x)
    assert_close(default.first, x.first)
    assert_close(default.second, x.second)
    out = Statistics(torch.empty_like(x.first), torch.empty_like(x.second))
    storage = (out.first, out.second)
    assert func(x, dim=None, keepdim=keepdim, out=out) is out
    assert out.first is storage[0]
    assert out.second is storage[1]
    assert_close(out.first, x.first)
    assert_close(out.second, x.second)


@pytest.mark.parametrize("func", [torch.mean, torch.sum])
def test_scalar_batch_reduction_preserves_native_options_and_autograd(func):
    x = Single(torch.arange(3.0, requires_grad=True))
    result = func(x, None, True, dtype=torch.float64)
    assert_close(result.array, x.array.to(torch.float64))
    result.array.sum().backward()
    assert_close(x.array.grad, torch.ones_like(x.array))
    raw = torch.empty(3, dtype=torch.float64)
    with torch.no_grad():
        assert func(x, axis=(), keepdims=True, dtype=torch.float64, out=raw) is raw
    assert_close(raw, x.array.to(torch.float64))
    ints = Single(torch.arange(3, dtype=torch.int32))
    if func is torch.sum:
        assert func(ints).array.dtype == torch.int64
    else:
        with pytest.raises(RuntimeError, match="dtype"):
            func(ints)


@pytest.mark.parametrize("func", [torch.mean, torch.sum, torch_average])
def test_empty_dimension_sequence_reduces_all_visible_axes(func):
    x = Single(torch.arange(24.0).reshape(2, 3, 4))
    result = func(x, dim=(), keepdim=True)
    assert_close(result.array, func(x.array, dim=(0, 1), keepdim=True))


@pytest.mark.parametrize("func", [torch.mean, torch.sum, torch_average])
@pytest.mark.parametrize("scalar_batch", [False, True])
def test_reduction_rejects_bad_output_shape_before_writing(func, scalar_batch):
    x = Statistics(torch.arange(3.0), torch.arange(6.0).reshape(2, 3))
    if not scalar_batch:
        x = Statistics(x.first.repeat(4, 1), x.second.repeat(4, 1, 1))
    out = Statistics(torch.zeros(3), torch.zeros(2, 2))
    with pytest.raises(ValueError, match="result shape"):
        func(x, dim=None, out=out)
    assert_close(out.first, torch.zeros(3))
    assert_close(out.second, torch.zeros(2, 2))


def test_scalar_batch_weighted_average_preserves_components():
    x = Single(torch.arange(3.0))
    result = torch_average(x, weights=torch.tensor(2.0), keepdim=True)
    assert_close(result.array, x.array)
