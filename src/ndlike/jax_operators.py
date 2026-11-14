"""Python operators expressed through probly's JAX function override protocol."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import jax
import numpy as np

from ndlike import jax_functions as jf

if TYPE_CHECKING:
    from collections.abc import Callable


def _supports_operand(value: object) -> bool:
    """Recognize JAX inputs and declared JAX protocols without converting them."""
    return (
        isinstance(value, (jax.Array, np.ndarray, np.number, np.bool_, bool, int, float, complex))
        or callable(getattr(type(value), "__jax_function__", None))
        or callable(getattr(type(value), "__jax_array__", None))
    )


def _binary(func: Callable, *, reflected: bool = False) -> Callable[[Any, object], Any]:
    """Make an operator that leaves unsupported operands to Python's fallback."""

    def method(self: object, other: object) -> Any:  # noqa: ANN401
        """Dispatch the binary operation or return NotImplemented.

        Args:
            self: The operand providing this operator.
            other: The other operand.

        Returns:
            The override result, or NotImplemented for unsupported operands.
        """
        if not _supports_operand(other):
            return NotImplemented
        args = (other, self) if reflected else (self, other)
        return jf.try_jax_function(func, args, *args)

    return method


def _unary(func: Callable) -> Callable[[Any], Any]:
    """Make a unary operator, for which Python has no reflected fallback."""

    def method(self: object) -> Any:  # noqa: ANN401
        """Return the unary operation's override result."""
        return jf.handle_jax_function(func, (self,), self)

    return method


class JaxOperatorsMixin[OpValueT = Any, OpResultT = Any, CompResultT = jax.Array]:  # noqa: PLW1641
    """Provide array operators through ``__jax_function__`` implementations.

    Operators use the corresponding ``jax_*`` wrapper as their dispatch key.
    Implementations control permissions and result types. Augmented assignment
    uses Python's ordinary out-of-place operator fallback, matching JAX immutability.
    Subclasses can override comparison methods. ``JaxLikeImplementation`` preserves
    inherited equality for dataclasses; dataclasses using this mixin directly still
    need ``eq=False`` to avoid generating their own equality operator.

    Binary operators decline unsupported operand families before backend dispatch,
    allowing Python's reflected-operator fallback. Backend errors still propagate.

    NumPy ufuncs may coerce through ``__array__`` without preserving the wrapper.
    Subclasses can define ``__array_ufunc__`` to override or disable that behavior.
    """

    # Let NumPy handle its own infix operations without forced deferral.
    __array_priority__ = 0

    __add__ = cast("Callable[[Any, OpValueT], OpResultT]", _binary(jf.jax_add))
    __radd__ = cast("Callable[[Any, OpValueT], OpResultT]", _binary(jf.jax_add, reflected=True))
    __sub__ = cast("Callable[[Any, OpValueT], OpResultT]", _binary(jf.jax_subtract))
    __rsub__ = cast("Callable[[Any, OpValueT], OpResultT]", _binary(jf.jax_subtract, reflected=True))
    __mul__ = cast("Callable[[Any, OpValueT], OpResultT]", _binary(jf.jax_multiply))
    __rmul__ = cast("Callable[[Any, OpValueT], OpResultT]", _binary(jf.jax_multiply, reflected=True))
    __truediv__ = cast("Callable[[Any, OpValueT], OpResultT]", _binary(jf.jax_true_divide))
    __rtruediv__ = cast("Callable[[Any, OpValueT], OpResultT]", _binary(jf.jax_true_divide, reflected=True))
    __floordiv__ = cast("Callable[[Any, OpValueT], OpResultT]", _binary(jf.jax_floor_divide))
    __rfloordiv__ = cast("Callable[[Any, OpValueT], OpResultT]", _binary(jf.jax_floor_divide, reflected=True))
    __mod__ = cast("Callable[[Any, OpValueT], OpResultT]", _binary(jf.jax_remainder))
    __rmod__ = cast("Callable[[Any, OpValueT], OpResultT]", _binary(jf.jax_remainder, reflected=True))
    __divmod__ = cast("Callable[[Any, OpValueT], tuple[OpResultT, OpResultT]]", _binary(jf.jax_divmod))
    __rdivmod__ = cast("Callable[[Any, OpValueT], tuple[OpResultT, OpResultT]]", _binary(jf.jax_divmod, reflected=True))
    __matmul__ = cast("Callable[[Any, OpValueT], OpResultT]", _binary(jf.jax_matmul))
    __rmatmul__ = cast("Callable[[Any, OpValueT], OpResultT]", _binary(jf.jax_matmul, reflected=True))
    __and__ = cast("Callable[[Any, OpValueT], OpResultT]", _binary(jf.jax_bitwise_and))
    __rand__ = cast("Callable[[Any, OpValueT], OpResultT]", _binary(jf.jax_bitwise_and, reflected=True))
    __or__ = cast("Callable[[Any, OpValueT], OpResultT]", _binary(jf.jax_bitwise_or))
    __ror__ = cast("Callable[[Any, OpValueT], OpResultT]", _binary(jf.jax_bitwise_or, reflected=True))
    __xor__ = cast("Callable[[Any, OpValueT], OpResultT]", _binary(jf.jax_bitwise_xor))
    __rxor__ = cast("Callable[[Any, OpValueT], OpResultT]", _binary(jf.jax_bitwise_xor, reflected=True))
    __lshift__ = cast("Callable[[Any, OpValueT], OpResultT]", _binary(jf.jax_left_shift))
    __rlshift__ = cast("Callable[[Any, OpValueT], OpResultT]", _binary(jf.jax_left_shift, reflected=True))
    __rshift__ = cast("Callable[[Any, OpValueT], OpResultT]", _binary(jf.jax_right_shift))
    __rrshift__ = cast("Callable[[Any, OpValueT], OpResultT]", _binary(jf.jax_right_shift, reflected=True))
    __eq__ = cast("Callable[[Any, OpValueT], CompResultT]", _binary(jf.jax_equal))
    __ne__ = cast("Callable[[Any, OpValueT], CompResultT]", _binary(jf.jax_not_equal))
    __lt__ = cast("Callable[[Any, OpValueT], CompResultT]", _binary(jf.jax_less))
    __le__ = cast("Callable[[Any, OpValueT], CompResultT]", _binary(jf.jax_less_equal))
    __gt__ = cast("Callable[[Any, OpValueT], CompResultT]", _binary(jf.jax_greater))
    __ge__ = cast("Callable[[Any, OpValueT], CompResultT]", _binary(jf.jax_greater_equal))
    __pos__ = cast("Callable[[Any], OpResultT]", _unary(jf.jax_positive))
    __neg__ = cast("Callable[[Any], OpResultT]", _unary(jf.jax_negative))
    __abs__ = cast("Callable[[Any], OpResultT]", _unary(jf.jax_absolute))
    __invert__ = cast("Callable[[Any], OpResultT]", _unary(jf.jax_invert))

    def __pow__(self, other: OpValueT, modulo: object = None) -> OpResultT:
        """Dispatch exponentiation; modular exponentiation is unsupported."""
        if modulo is not None or not _supports_operand(other):
            return NotImplemented
        return jf.try_jax_function(jf.jax_power, (self, other), self, other)

    def __rpow__(self, other: OpValueT, modulo: object = None) -> OpResultT:
        """Dispatch reflected exponentiation; modular exponentiation is unsupported."""
        if modulo is not None or not _supports_operand(other):
            return NotImplemented
        return jf.try_jax_function(jf.jax_power, (other, self), other, self)
