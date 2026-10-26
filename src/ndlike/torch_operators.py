"""Python operators expressed through PyTorch's function override protocol."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from collections.abc import Callable


def _binary(func: Callable, *, reflected: bool = False) -> Callable[[Any, object], Any]:
    """Make a binary operator that dispatches through a torch function."""

    def method(self: object, other: object) -> Any:  # noqa: ANN401
        """Return the torch function's result with the appropriate operand order."""
        args = (other, self) if reflected else (self, other)
        return func(*args)

    return method


def _unary(func: Callable) -> Callable[[Any], Any]:
    """Make a unary operator that dispatches through a torch function."""

    def method(self: object) -> Any:  # noqa: ANN401
        """Return the torch function's result."""
        return func(self)

    return method


def _divmod(left: object, right: object) -> tuple[Any, Any]:
    """Dispatch quotient and remainder separately because torch has no divmod."""
    return (
        torch.floor_divide(left, right),  # ty: ignore[invalid-argument-type]
        torch.remainder(left, right),  # ty: ignore[no-matching-overload]
    )


class TorchOperatorsMixin:  # noqa: PLW1641
    """Provide array operators through ``__torch_function__`` implementations.

    Operators call the corresponding torch functions directly, preserving their
    dispatch and error behavior. Implementations control permissions and result
    types. Augmented assignment uses Python's out-of-place operator fallback.
    Subclasses can override comparison methods; dataclasses must use ``eq=False``
    to inherit the mixin's equality operator instead of generating their own.
    """

    # NumPy must defer mixed infix expressions rather than bypass overrides by
    # coercing an operand through __array__.
    __array_priority__ = 1000
    __array_ufunc__ = None

    __add__ = _binary(torch.add)
    __radd__ = _binary(torch.add, reflected=True)
    __sub__ = _binary(torch.sub)
    __rsub__ = _binary(torch.sub, reflected=True)
    __mul__ = _binary(torch.mul)
    __rmul__ = _binary(torch.mul, reflected=True)
    __truediv__ = _binary(torch.true_divide)
    __rtruediv__ = _binary(torch.true_divide, reflected=True)
    __floordiv__ = _binary(torch.floor_divide)
    __rfloordiv__ = _binary(torch.floor_divide, reflected=True)
    __mod__ = _binary(torch.remainder)
    __rmod__ = _binary(torch.remainder, reflected=True)
    __divmod__ = _binary(_divmod)
    __rdivmod__ = _binary(_divmod, reflected=True)
    __matmul__ = _binary(torch.matmul)
    __rmatmul__ = _binary(torch.matmul, reflected=True)
    __and__ = _binary(torch.bitwise_and)
    __rand__ = _binary(torch.bitwise_and, reflected=True)
    __or__ = _binary(torch.bitwise_or)
    __ror__ = _binary(torch.bitwise_or, reflected=True)
    __xor__ = _binary(torch.bitwise_xor)
    __rxor__ = _binary(torch.bitwise_xor, reflected=True)
    __lshift__ = _binary(torch.bitwise_left_shift)
    __rlshift__ = _binary(torch.bitwise_left_shift, reflected=True)
    __rshift__ = _binary(torch.bitwise_right_shift)
    __rrshift__ = _binary(torch.bitwise_right_shift, reflected=True)
    __eq__ = _binary(torch.eq)
    __ne__ = _binary(torch.ne)
    __lt__ = _binary(torch.lt)
    __le__ = _binary(torch.le)
    __gt__ = _binary(torch.gt)
    __ge__ = _binary(torch.ge)
    __pos__ = _unary(torch.positive)
    __neg__ = _unary(torch.neg)
    __abs__ = _unary(torch.abs)
    __invert__ = _unary(torch.bitwise_not)

    def __pow__(self, other: object, modulo: object = None) -> Any:  # noqa: ANN401
        """Dispatch exponentiation; modular exponentiation is unsupported."""
        if modulo is not None:
            return NotImplemented
        return torch.pow(self, other)  # ty: ignore[no-matching-overload]

    def __rpow__(self, other: object, modulo: object = None) -> Any:  # noqa: ANN401
        """Dispatch reflected exponentiation; modular exponentiation is unsupported."""
        if modulo is not None:
            return NotImplemented
        return torch.pow(other, self)  # ty: ignore[no-matching-overload]
