"""Torch operator dispatch and default inheritance by Torch-like implementations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import operator
from types import NotImplementedType
from typing import ClassVar, Never

import numpy as np
import pytest

pytest.importorskip("torch")
import torch
from torch.testing import assert_close

from ndlike.protected_axis.torch import TorchAxisProtected
from ndlike.torch_like import TorchLikeImplementation
from ndlike.torch_operators import TorchOperatorsMixin

_BINARY: list[tuple[Callable, Callable]] = [
    (operator.add, torch.add),
    (operator.sub, torch.sub),
    (operator.mul, torch.mul),
    (operator.truediv, torch.true_divide),
    (operator.floordiv, torch.floor_divide),
    (operator.mod, torch.remainder),
    (operator.pow, torch.pow),
    (operator.and_, torch.bitwise_and),
    (operator.or_, torch.bitwise_or),
    (operator.xor, torch.bitwise_xor),
    (operator.lshift, torch.bitwise_left_shift),
    (operator.rshift, torch.bitwise_right_shift),
]
_COMPARISONS: list[tuple[Callable, Callable]] = [
    (operator.eq, torch.eq),
    (operator.ne, torch.ne),
    (operator.lt, torch.lt),
    (operator.le, torch.le),
    (operator.gt, torch.gt),
    (operator.ge, torch.ge),
]
_UNARY: list[tuple[Callable, Callable]] = [
    (operator.pos, torch.positive),
    (operator.neg, torch.neg),
    (operator.abs, torch.abs),
    (operator.invert, torch.bitwise_not),
]


@dataclass
class _Call:
    func: Callable
    types: tuple[type, ...]
    args: tuple[object, ...]
    kwargs: dict[str, object]


class _Recording(TorchOperatorsMixin):
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None) -> _Call:
        del cls
        return _Call(func, types, args, kwargs or {})

    def __array__(self, *args: object, **kwargs: object) -> Never:
        del args, kwargs
        pytest.fail("Operators must not coerce wrappers through __array__")


@dataclass(frozen=True, eq=False)
class _Tensor(TorchAxisProtected[torch.Tensor]):
    tensor: torch.Tensor
    protected_axes: ClassVar[dict[str, int]] = {"tensor": 1}

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None) -> _Tensor | NotImplementedType:
        if not all(issubclass(t, (cls, torch.Tensor)) for t in types):
            return NotImplemented
        raw = tuple(arg.tensor if isinstance(arg, cls) else arg for arg in args)
        return cls(func(*raw, **(kwargs or {})))


@pytest.mark.parametrize(("op", "func"), [*_BINARY, (operator.matmul, torch.matmul)])
@pytest.mark.parametrize("reflected", [False, True])
@pytest.mark.parametrize("kind", ["torch", "wrapper"])
def test_binary_dispatch_preserves_function_and_operand_order(op, func, reflected, kind):
    x = _Recording()
    other = torch.ones(2) if kind == "torch" else _Recording()
    left, right = (other, x) if reflected else (x, other)
    method = f"__{'r' if reflected else ''}{op.__name__.removesuffix('_')}__"
    result = getattr(x, method)(other)

    assert isinstance(result, _Call)
    assert result.func is func
    assert result.args[0] is left
    assert result.args[1] is right
    assert _Recording in result.types
    assert result.kwargs == {}


@pytest.mark.parametrize(("op", "func"), _COMPARISONS)
def test_comparison_dispatch_preserves_function_and_operands(op, func):
    x, other = _Recording(), torch.ones(2)
    result = op(x, other)
    assert result.func is func
    assert result.args[0] is x
    assert result.args[1] is other


@pytest.mark.parametrize(("op", "func"), _UNARY)
def test_unary_dispatch_preserves_function_and_operand(op, func):
    x = _Recording()
    result = op(x)

    assert result.func is func
    assert result.args[0] is x
    assert result.types == (_Recording,)
    assert result.kwargs == {}


@pytest.mark.parametrize(("op", "func"), [*_BINARY, *_COMPARISONS])
@pytest.mark.parametrize("reflected", [False, True])
@pytest.mark.parametrize("kind", ["scalar", "torch", "wrapper"])
def test_implementation_inherits_working_binary_operators(op, func, reflected, kind):
    del func
    x = _Tensor(torch.tensor([[2, 3], [4, 5]]))
    other = {"scalar": 2, "torch": torch.tensor([1, 2]), "wrapper": _Tensor(torch.tensor([1, 2]))}[kind]
    raw = other.tensor if isinstance(other, _Tensor) else other
    left, right = (other, x) if reflected else (x, other)
    expected = op(raw, x.tensor) if reflected else op(x.tensor, raw)

    result = op(left, right)

    assert isinstance(result, _Tensor)
    assert_close(result.tensor, expected)
    assert_close(x.tensor, torch.tensor([[2, 3], [4, 5]]))


@pytest.mark.parametrize(("op", "func"), _UNARY)
def test_implementation_inherits_working_unary_operators(op, func):
    x = _Tensor(torch.tensor([[-2, 3], [4, -5]]))
    result = op(x)
    assert isinstance(result, _Tensor)
    assert_close(result.tensor, func(x.tensor))


@pytest.mark.parametrize("reflected", [False, True])
def test_divmod_dispatches_both_results(reflected):
    x = _Recording()
    left, right = (20, x) if reflected else (x, 3)
    quotient, remainder = divmod(left, right)

    assert isinstance(quotient, _Call)
    assert isinstance(remainder, _Call)
    assert quotient.func is torch.floor_divide
    assert remainder.func is torch.remainder
    for result in (quotient, remainder):
        assert result.args[0] is left
        assert result.args[1] is right


@pytest.mark.parametrize("reflected", [False, True])
def test_implementation_inherits_divmod(reflected):
    x = _Tensor(torch.tensor([[5, 7], [8, 9]]))
    left, right = (20, x) if reflected else (x, 3)
    quotient, remainder = divmod(left, right)
    assert isinstance(quotient, _Tensor)
    assert isinstance(remainder, _Tensor)
    expected_quotient = 20 // x.tensor if reflected else x.tensor // 3
    expected_remainder = 20 % x.tensor if reflected else x.tensor % 3
    assert_close(quotient.tensor, expected_quotient)
    assert_close(remainder.tensor, expected_remainder)


@pytest.mark.parametrize("reflected", [False, True])
def test_implementation_inherits_matmul(reflected):
    x = _Tensor(torch.arange(6.0).reshape(2, 3))
    other = torch.ones((4, 2)) if reflected else torch.ones((3, 4))
    result = other @ x if reflected else x @ other
    expected = other @ x.tensor if reflected else x.tensor @ other
    assert isinstance(result, _Tensor)
    assert_close(result.tensor, expected)


def test_augmented_assignment_rebinds_without_mutating():
    x = _Tensor(torch.ones((2, 3)))
    original = x
    x += 2
    assert x is not original
    assert_close(original.tensor, torch.ones((2, 3)))
    assert_close(x.tensor, torch.full((2, 3), 3.0))


def test_modular_exponentiation_is_unsupported():
    x = _Recording()
    assert x.__pow__(2, 3) is NotImplemented
    assert x.__rpow__(2, 3) is NotImplemented
    with pytest.raises(TypeError):
        pow(x, 2, 3)


def test_implementation_inherits_mixin_by_default():
    assert issubclass(TorchLikeImplementation, TorchOperatorsMixin)


@pytest.mark.parametrize(("op", "func"), [*_BINARY, *_UNARY, *_COMPARISONS, (operator.matmul, torch.matmul)])
def test_override_errors_propagate_unchanged(op, func):
    error = TypeError("implementation error")

    class _Broken(TorchOperatorsMixin):
        @classmethod
        def __torch_function__(cls, func, types, args=(), kwargs=None) -> Never:
            del cls, func, types, args, kwargs
            raise error

    with pytest.raises(TypeError, match="implementation error") as caught:
        op(_Broken()) if (op, func) in _UNARY else op(_Broken(), torch.ones(2))
    assert caught.value is error


@pytest.mark.parametrize("op", [operator.add, operator.neg, operator.matmul, divmod])
def test_declined_overrides_raise_torch_dispatch_errors(op):
    class _Denied(TorchOperatorsMixin):
        @classmethod
        def __torch_function__(cls, func, types, args=(), kwargs=None) -> NotImplementedType:
            del cls, func, types, args, kwargs
            return NotImplemented

    with pytest.raises(TypeError, match=r"Multiple dispatch failed|no implementation found"):
        op(_Denied()) if op is operator.neg else op(_Denied(), torch.ones(2))


def test_numpy_ufunc_cannot_bypass_overrides():
    with pytest.raises(TypeError):
        np.add(_Recording(), 1)


@pytest.mark.parametrize(("op", "func"), [*_BINARY, *_COMPARISONS, (operator.matmul, torch.matmul)])
@pytest.mark.parametrize("reflected", [False, True])
def test_numpy_operands_are_rejected_without_coercing_wrapper(op, func, reflected):
    del func
    x, other = _Recording(), np.ones(2)
    with pytest.raises(TypeError):
        op(other, x) if reflected else op(x, other)
