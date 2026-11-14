"""Operators through the default protected-axis Torch dispatcher."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import operator
from typing import Any, ClassVar

import numpy as np
import pytest

pytest.importorskip("torch")
import torch
from torch.testing import assert_close

from ndlike.protected_axis.torch import TorchAxisProtected

_BINARY: list[tuple[Callable, Callable, Callable]] = [
    (operator.add, operator.iadd, torch.add),
    (operator.sub, operator.isub, torch.sub),
    (operator.mul, operator.imul, torch.mul),
    (operator.truediv, operator.itruediv, torch.true_divide),
    (operator.floordiv, operator.ifloordiv, torch.floor_divide),
    (operator.mod, operator.imod, torch.remainder),
    (operator.pow, operator.ipow, torch.pow),
    (operator.and_, operator.iand, torch.bitwise_and),
    (operator.or_, operator.ior, torch.bitwise_or),
    (operator.xor, operator.ixor, torch.bitwise_xor),
    (operator.lshift, operator.ilshift, torch.bitwise_left_shift),
    (operator.rshift, operator.irshift, torch.bitwise_right_shift),
]
_UNARY: list[tuple[Callable, Callable]] = [
    (operator.pos, torch.positive),
    (operator.neg, torch.neg),
    (abs, torch.abs),
    (operator.invert, torch.bitwise_not),
]
_COMPARISONS: list[tuple[Callable, Callable]] = [
    (operator.eq, torch.eq),
    (operator.ne, torch.ne),
    (operator.lt, torch.lt),
    (operator.le, torch.le),
    (operator.gt, torch.gt),
    (operator.ge, torch.ge),
]
_PERMITTED: set[Callable] = (
    {func for _, _, func in _BINARY} | {func for _, func in [*_UNARY, *_COMPARISONS]} | {torch.matmul}
)


@dataclass(frozen=True)
class Numeric(TorchAxisProtected["Numeric", Any]):
    array: torch.Tensor
    label: str = "original"
    protected_axes: ClassVar[dict[str, int]] = {"array": 1}
    permitted_functions: ClassVar[set[Callable]] = _PERMITTED


@dataclass(frozen=True)
class Pair(TorchAxisProtected["Pair", Any]):
    first: torch.Tensor
    second: torch.Tensor
    protected_axes: ClassVar[dict[str, int]] = {"first": 1, "second": 2}
    permitted_functions: ClassVar[set[Callable]] = _PERMITTED


def call(func: Callable, *args: object, **kwargs: object):
    """Call an override-aware function without Torch's native-only annotations."""
    return func(*args, **kwargs)


@pytest.mark.parametrize(("op", "inplace", "func"), _BINARY)
@pytest.mark.parametrize("kind", ["scalar", "tensor", "protected"])
@pytest.mark.parametrize("reflected", [False, True])
def test_binary_operations(op, inplace, func, kind, reflected):
    del inplace
    x = Numeric(torch.tensor([[2, 3], [4, 5]]))
    other = {"scalar": 2, "tensor": torch.tensor([1, 2]), "protected": Numeric(torch.tensor([1, 2]))}[kind]
    raw = other.array if isinstance(other, Numeric) else other
    left, right = (other, x) if reflected else (x, other)
    expected = func(raw, x.array) if reflected else func(x.array, raw)
    result = op(left, right)
    assert isinstance(result, Numeric)
    assert_close(result.array, expected)
    assert_close(x.array, torch.tensor([[2, 3], [4, 5]]))


@pytest.mark.parametrize(("op", "func"), _UNARY)
def test_unary_operations(op, func):
    x = Pair(torch.tensor([[-2, 3], [4, -5]]), torch.tensor([[[-2, 3]], [[4, -5]]]))
    result = op(x)
    assert isinstance(result, Pair)
    assert_close(result.first, func(x.first))
    assert_close(result.second, func(x.second))


@pytest.mark.parametrize(("op", "inplace", "func"), _BINARY)
@pytest.mark.parametrize("output", ["inplace", "protected", "tensor"])
def test_outputs_preserve_destination_identity_and_storage(op, inplace, func, output):
    del op
    dtype = torch.float32 if func is torch.true_divide else torch.int64
    x = Numeric(torch.tensor([[4, 6], [8, 10]], dtype=dtype))
    out = x if output == "inplace" else Numeric(torch.zeros_like(x.array), label="destination")
    storage = out.array
    expected = func(x.array, 2)
    result = inplace(x, 2) if output == "inplace" else call(func, x, 2, out=out if output == "protected" else storage)
    assert result is (storage if output == "tensor" else out)
    assert out.array is storage
    assert_close(storage, expected)
    assert out.label == ("original" if output == "inplace" else "destination")


def test_multifield_inplace_and_cross_field_aliasing():
    storage = torch.ones((2, 2))
    x = Pair(storage, storage.unsqueeze(-1))
    original = x
    x += 2
    assert x is original
    assert x.first is storage
    assert_close(x.first, torch.full((2, 2), 3.0))
    assert_close(x.second, torch.full((2, 2, 1), 3.0))


@pytest.mark.parametrize(("op", "func"), _COMPARISONS)
def test_comparisons_reduce_fields_and_broadcast_batch(op, func):
    x = Pair(torch.tensor([[1, 2]]), torch.tensor([[[3, 4]]]))
    y = Pair(torch.tensor([[1, 2], [1, 3]]), torch.tensor([[[3, 4]], [[3, 5]]]))
    reduce = torch.any if func is torch.ne else torch.all
    combine = torch.logical_or if func is torch.ne else torch.logical_and
    expected = combine(reduce(func(x.first, y.first), dim=-1), reduce(func(x.second, y.second), dim=(-1, -2)))
    result = op(x, y)
    assert result.shape == (2,)
    assert result.dtype == torch.bool
    assert_close(result, expected)
    out = torch.empty(2, dtype=torch.bool)
    assert call(func, x, y, out=out) is out
    assert_close(out, expected)


@pytest.mark.parametrize("left_shape", [(3,), (2, 3), (4, 2, 3)])
@pytest.mark.parametrize("right_shape", [(3,), (3, 5), (4, 3, 5)])
def test_matmul_uses_visible_dimensions(left_shape, right_shape):
    x = Numeric(torch.arange(int(np.prod(left_shape)) * 2.0).reshape((*left_shape, 2)))
    y = Numeric(torch.arange(int(np.prod(right_shape)) * 2.0).reshape((*right_shape, 2)))
    expected = torch.stack([x.array[..., i] @ y.array[..., i] for i in range(2)], dim=-1)
    result = x @ y
    assert isinstance(result, Numeric)
    assert_close(result.array, expected)
    out = Numeric(torch.empty_like(expected))
    assert call(torch.matmul, x, y, out=out) is out
    assert_close(out.array, expected)


def test_matmul_raw_operands_multiple_protected_axes_and_inplace():
    x = Pair(torch.ones((2, 3, 4)), torch.ones((2, 3, 4, 5)))
    result = x @ torch.ones((3, 6))
    assert isinstance(result, Pair)
    assert result.shape == (2, 6)
    assert_close(result.first, torch.full((2, 6, 4), 3.0))
    assert_close(result.second, torch.full((2, 6, 4, 5), 3.0))
    reflected = torch.ones((7, 2)) @ x
    assert isinstance(reflected, Pair)
    assert reflected.shape == (7, 3)
    assert_close(reflected.second, torch.full((7, 3, 4, 5), 2.0))
    first, second, original = x.first, x.second, x
    x @= torch.eye(3) * 2
    assert x is original
    assert x.first is first
    assert x.second is second
    assert_close(first, torch.full_like(first, 2))
    assert_close(second, torch.full_like(second, 2))


def test_named_operands_alpha_and_output_only_dispatch():
    x = Numeric(torch.ones((2, 3)))
    result = call(torch.add, input=x, other=2, alpha=3)
    assert_close(result.array, torch.full((2, 3), 7.0))
    result = call(torch.pow, input=x, exponent=2)
    assert_close(result.array, x.array)
    assert call(torch.add, torch.ones((2, 3)), 2, out=x) is x
    assert_close(x.array, torch.full((2, 3), 3.0))
    assert call(torch.add, input=x, other=2, alpha=3, out=x) is x
    assert_close(x.array, torch.full((2, 3), 9.0))


@pytest.mark.parametrize("reflected", [False, True])
def test_divmod_dispatches_each_result(reflected):
    x = Numeric(torch.tensor([[5, 7], [8, 9]]))
    left, right = (20, x) if reflected else (x, 3)
    raw_left, raw_right = (20, x.array) if reflected else (x.array, 3)
    quotient, remainder = divmod(left, right)
    assert isinstance(quotient, Numeric)
    assert isinstance(remainder, Numeric)
    assert_close(quotient.array, torch.floor_divide(raw_left, raw_right))
    assert_close(remainder.array, call(torch.remainder, raw_left, raw_right))


def test_invalid_layouts_and_output_shapes_do_not_mutate():
    x = Numeric(torch.ones((2, 1)))
    with pytest.raises(ValueError, match="identical protected trailing shapes"):
        _ = x + Numeric(torch.ones((2, 3)))
    with pytest.raises(ValueError, match="modified protected trailing axes"):
        _ = x + torch.ones((2, 3))
    x = Pair(torch.ones((1, 3)), torch.ones((1, 2, 3)))
    with pytest.raises(ValueError, match="inconsistent batch-shapes"):
        _ = x + torch.ones((2, 3))
    with pytest.raises(ValueError, match="result shape"):
        call(torch.add, x, Pair(torch.ones((2, 3)), torch.ones((2, 2, 3))), out=x)
    assert_close(x.first, torch.ones((1, 3)))
    assert_close(x.second, torch.ones((1, 2, 3)))
    out = Pair(torch.zeros_like(x.first), torch.zeros((2, 2, 3)))
    with pytest.raises(ValueError, match="result shape"):
        call(torch.add, x, 1, out=out)
    assert_close(out.first, torch.zeros_like(x.first))
    with pytest.raises(TypeError, match="single-field"):
        call(torch.add, x, 1, out=torch.zeros_like(x.first))


def test_native_casting_and_autograd_errors_leave_output_unchanged():
    x = Pair(torch.full((2, 3), 3.0), torch.full((2, 2, 3), 3.0))
    out = Pair(torch.zeros_like(x.first), torch.zeros_like(x.second, dtype=torch.int64))
    with pytest.raises(RuntimeError, match="cast"):
        call(torch.true_divide, x, 2, out=out)
    assert_close(out.first, torch.zeros_like(x.first))
    assert_close(out.second, torch.zeros_like(x.second, dtype=torch.int64))
    leaf = torch.ones((2, 3), requires_grad=True)
    y = Numeric(leaf)
    (y * y).array.sum().backward()
    assert_close(leaf.grad, torch.full_like(leaf, 2))
    with pytest.raises(RuntimeError, match="automatic differentiation"):
        operator.iadd(y, 1)
    assert_close(leaf, torch.ones_like(leaf))
    with torch.no_grad():
        assert operator.iadd(y, 1) is y
    assert_close(leaf, torch.full_like(leaf, 2))


def test_permissions_apply_to_both_inputs_and_destination():
    @dataclass(frozen=True)
    class Veto(Numeric):
        deny: bool = False

        def protected_values(self, func=None):
            if func is not None and self.deny:
                return None
            return super().protected_values() if func is None else super().protected_values(func)

    allowed = Veto(torch.ones((2, 3)))
    denied = Veto(torch.ones((2, 3)), deny=True)
    for left, right in ((allowed, denied), (denied, allowed)):
        with pytest.raises(TypeError):
            _ = left + right
    with pytest.raises(TypeError):
        call(torch.add, allowed, 1, out=denied)
    with pytest.raises(TypeError):
        _ = Numeric(allowed.array) + allowed
    with pytest.raises(TypeError):
        call(torch.add, allowed, object())


def test_numpy_sidecars_are_declined():
    @dataclass(frozen=True)
    class Sidecar(Numeric):
        sidecar: np.ndarray = field(default_factory=lambda: np.ones(2))
        protected_axes: ClassVar[dict[str, int]] = {"sidecar": 0}

    with pytest.raises(TypeError):
        _ = Sidecar(torch.ones((2, 3))) + 1


def test_result_hook_and_reconstruction_receive_context_but_out_retains_metadata():
    calls = []

    @dataclass(frozen=True)
    class Hooked(Numeric):
        def _postprocess_elementwise_result(self, values, *, func, operands):
            assert func is torch.add
            assert operands[0] is self
            calls.append("hook")
            return {"array": values["array"] * 2}

        def with_protected_values(self, values, func=None):
            assert func is torch.add
            calls.append("create")
            return Hooked(values["array"], "created")

    x = Hooked(torch.ones((2, 3)))
    result = x + 2
    assert result.label == "created"
    assert_close(result.array, torch.full((2, 3), 6.0))
    assert calls == ["hook", "create"]
    assert operator.iadd(x, 2) is x
    assert x.label == "original"
    assert_close(x.array, torch.full((2, 3), 3.0))
    assert calls == ["hook", "create"]


def test_result_hook_cannot_change_protected_shape():
    @dataclass(frozen=True)
    class Broken(Numeric):
        def _postprocess_elementwise_result(self, values, *, func, operands):
            del func, operands
            return {"array": values["array"][..., :1]}

    with pytest.raises(ValueError, match="modified protected trailing axes"):
        _ = Broken(torch.ones((2, 3))) + 1


@pytest.mark.parametrize("count", [0, 1, 2])
@pytest.mark.parametrize("size", [0, 2])
def test_matmul_empty_and_unprotected_dimensions(count, size):
    @dataclass(frozen=True)
    class Matrix(TorchAxisProtected["Matrix", Any]):
        array: torch.Tensor
        protected_axes: ClassVar[dict[str, int]] = {"array": count}
        permitted_functions: ClassVar[set[Callable]] = {torch.matmul}

    left = Matrix(torch.ones((2, 3, *((size,) * count))))
    result = left @ torch.ones((4, 3, 5))
    assert isinstance(result, Matrix)
    assert result.shape == (4, 2, 5)
    assert_close(result.array, torch.full((4, 2, 5, *((size,) * count)), 3.0))
    out = Matrix(torch.empty_like(result.array))
    assert call(torch.matmul, left, torch.ones((4, 3, 5)), out=out) is out
    assert_close(out.array, result.array)


def test_nested_protected_fields_support_arithmetic_and_out():
    @dataclass(frozen=True)
    class Outer(TorchAxisProtected["Outer", Any]):
        inner: Numeric
        protected_axes: ClassVar[dict[str, int]] = {"inner": 1}
        permitted_functions: ClassVar[set[Callable]] = {torch.add}

    x = Outer(Numeric(torch.ones((2, 3, 4))))
    result = x + 2
    assert isinstance(result, Outer)
    assert result.shape == (2,)
    assert_close(result.inner.array, torch.full((2, 3, 4), 3.0))
    original, inner, storage = x, x.inner, x.inner.array
    x += 2
    assert x is original
    assert x.inner is inner
    assert x.inner.array is storage
    assert_close(storage, result.inner.array)


def test_out_uses_actual_storage_not_permission_hook_values():
    @dataclass(frozen=True)
    class Scaled(Numeric):
        def _postprocess_protected_values(self, values, func):
            del func
            return {"array": values["array"] * 2}

    x = Scaled(torch.ones((2, 3)))
    out = Scaled(torch.zeros((2, 3)))
    assert call(torch.add, x, 1, out=out) is out
    assert_close(out.array, torch.full((2, 3), 3.0))


def test_matmul_gradients_and_output_shape_validation():
    left = torch.ones((2, 3, 4), requires_grad=True)
    right = torch.ones((3, 2, 4), requires_grad=True)
    (Numeric(left) @ Numeric(right)).array.sum().backward()
    assert_close(left.grad, torch.full_like(left, 2))
    assert_close(right.grad, torch.full_like(right, 2))
    x = Numeric(torch.ones((2, 3, 4)))
    with pytest.raises(ValueError, match="result shape"):
        operator.imatmul(x, torch.ones((3, 5)))
    assert_close(x.array, torch.ones((2, 3, 4)))
    with pytest.raises(ValueError, match="visible dimension"):
        _ = Numeric(torch.ones(4)) @ Numeric(torch.ones(4))


def test_comparison_outputs_reject_wrong_shapes_without_resizing():
    x = Numeric(torch.ones((2, 3)))
    out = torch.zeros(3, dtype=torch.bool)
    with pytest.raises(ValueError, match="batch shape"):
        call(torch.eq, x, x, out=out)
    assert_close(out, torch.zeros(3, dtype=torch.bool))
    with pytest.raises(ValueError, match="batch shape"):
        call(torch.eq, x, x, out=x)
    assert_close(x.array, torch.ones((2, 3)))


@pytest.mark.parametrize("func", [torch.neg, torch.abs, torch.bitwise_not])
def test_unary_out_updates_existing_fields(func):
    x = Numeric(torch.tensor([[-2, 3], [4, -5]]))
    expected = func(x.array)
    assert call(func, x, out=x) is x
    assert_close(x.array, expected)


def test_denied_operations_and_invalid_kwargs_do_not_mutate():
    @dataclass(frozen=True)
    class Denied(Numeric):
        permitted_functions: ClassVar[set[Callable]] = set()

    x = Denied(torch.ones((2, 3)))
    for op in (operator.add, operator.iadd):
        with pytest.raises(TypeError):
            op(x, 1)
    assert_close(x.array, torch.ones((2, 3)))
    allowed = Numeric(torch.ones((2, 3)))
    with pytest.raises(TypeError):
        call(torch.add, allowed, 1, unknown=True, out=allowed)
    assert_close(allowed.array, torch.ones((2, 3)))
