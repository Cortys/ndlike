"""Shared static operator contracts; NumPy retains its native mixin dispatch."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, Self, cast

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable


class ArrayOperators[OpValueT, OpResultT, CompResultT](Protocol):  # noqa: PLW1641
    """Operand and result contracts shared by the three array backends."""

    def __add__(self, other: OpValueT, /) -> OpResultT:
        """Apply the add operation."""

    def __radd__(self, other: OpValueT, /) -> OpResultT:
        """Apply the radd operation."""

    def __sub__(self, other: OpValueT, /) -> OpResultT:
        """Apply the sub operation."""

    def __rsub__(self, other: OpValueT, /) -> OpResultT:
        """Apply the rsub operation."""

    def __mul__(self, other: OpValueT, /) -> OpResultT:
        """Apply the mul operation."""

    def __rmul__(self, other: OpValueT, /) -> OpResultT:
        """Apply the rmul operation."""

    def __truediv__(self, other: OpValueT, /) -> OpResultT:
        """Apply the truediv operation."""

    def __rtruediv__(self, other: OpValueT, /) -> OpResultT:
        """Apply the rtruediv operation."""

    def __floordiv__(self, other: OpValueT, /) -> OpResultT:
        """Apply the floordiv operation."""

    def __rfloordiv__(self, other: OpValueT, /) -> OpResultT:
        """Apply the rfloordiv operation."""

    def __mod__(self, other: OpValueT, /) -> OpResultT:
        """Apply the mod operation."""

    def __rmod__(self, other: OpValueT, /) -> OpResultT:
        """Apply the rmod operation."""

    def __matmul__(self, other: OpValueT, /) -> OpResultT:
        """Apply the matmul operation."""

    def __rmatmul__(self, other: OpValueT, /) -> OpResultT:
        """Apply the rmatmul operation."""

    def __and__(self, other: OpValueT, /) -> OpResultT:
        """Apply the and operation."""

    def __rand__(self, other: OpValueT, /) -> OpResultT:
        """Apply the rand operation."""

    def __or__(self, other: OpValueT, /) -> OpResultT:
        """Apply the or operation."""

    def __ror__(self, other: OpValueT, /) -> OpResultT:
        """Apply the ror operation."""

    def __xor__(self, other: OpValueT, /) -> OpResultT:
        """Apply the xor operation."""

    def __rxor__(self, other: OpValueT, /) -> OpResultT:
        """Apply the rxor operation."""

    def __lshift__(self, other: OpValueT, /) -> OpResultT:
        """Apply the lshift operation."""

    def __rlshift__(self, other: OpValueT, /) -> OpResultT:
        """Apply the rlshift operation."""

    def __rshift__(self, other: OpValueT, /) -> OpResultT:
        """Apply the rshift operation."""

    def __rrshift__(self, other: OpValueT, /) -> OpResultT:
        """Apply the rrshift operation."""

    def __pow__(self, other: OpValueT, /) -> OpResultT:
        """Apply the pow operation."""

    def __rpow__(self, other: OpValueT, /) -> OpResultT:
        """Apply the rpow operation."""

    if TYPE_CHECKING:

        def __eq__(self, other: OpValueT, /) -> CompResultT:  # ty: ignore[incompatible-method-override]
            """Apply the eq operation."""

        def __ne__(self, other: OpValueT, /) -> CompResultT:  # ty: ignore[incompatible-method-override]
            """Apply the ne operation."""

    def __lt__(self, other: OpValueT, /) -> CompResultT:
        """Apply the lt operation."""

    def __le__(self, other: OpValueT, /) -> CompResultT:
        """Apply the le operation."""

    def __gt__(self, other: OpValueT, /) -> CompResultT:
        """Apply the gt operation."""

    def __ge__(self, other: OpValueT, /) -> CompResultT:
        """Apply the ge operation."""

    def __pos__(self) -> OpResultT:
        """Apply the pos operation."""

    def __neg__(self) -> OpResultT:
        """Apply the neg operation."""

    def __abs__(self) -> OpResultT:
        """Apply the abs operation."""

    def __invert__(self) -> OpResultT:
        """Apply the invert operation."""


class InplaceArrayOperators[OpValueT](Protocol):
    """Mutating operator contracts shared by NumPy and Torch arrays.

    Native Torch tensors lack ``__imatmul__``, so that method is declared only
    on the NumPy protocol and the implementation mixins.
    """

    def __iadd__(self, other: OpValueT, /) -> Self:
        """Apply add with this array as the output."""

    def __isub__(self, other: OpValueT, /) -> Self:
        """Apply sub with this array as the output."""

    def __imul__(self, other: OpValueT, /) -> Self:
        """Apply mul with this array as the output."""

    def __itruediv__(self, other: OpValueT, /) -> Self:
        """Apply truediv with this array as the output."""

    def __ifloordiv__(self, other: OpValueT, /) -> Self:
        """Apply floordiv with this array as the output."""

    def __imod__(self, other: OpValueT, /) -> Self:
        """Apply mod with this array as the output."""

    def __ipow__(self, other: OpValueT, /) -> Self:
        """Apply pow with this array as the output."""

    def __iand__(self, other: OpValueT, /) -> Self:
        """Apply and with this array as the output."""

    def __ior__(self, other: OpValueT, /) -> Self:
        """Apply or with this array as the output."""

    def __ixor__(self, other: OpValueT, /) -> Self:
        """Apply xor with this array as the output."""

    def __ilshift__(self, other: OpValueT, /) -> Self:
        """Apply lshift with this array as the output."""

    def __irshift__(self, other: OpValueT, /) -> Self:
        """Apply rshift with this array as the output."""


class NumpyOperatorsMixin[OpValueT, OpResultT, CompResultT](np.lib.mixins.NDArrayOperatorsMixin):  # noqa: PLW1641
    """Type NumPy mixin operators without changing their dispatch behavior."""

    __add__ = cast("Callable[[Any, OpValueT], OpResultT]", np.lib.mixins.NDArrayOperatorsMixin.__add__)
    __radd__ = cast("Callable[[Any, OpValueT], OpResultT]", np.lib.mixins.NDArrayOperatorsMixin.__radd__)
    __sub__ = cast("Callable[[Any, OpValueT], OpResultT]", np.lib.mixins.NDArrayOperatorsMixin.__sub__)
    __rsub__ = cast("Callable[[Any, OpValueT], OpResultT]", np.lib.mixins.NDArrayOperatorsMixin.__rsub__)
    __mul__ = cast("Callable[[Any, OpValueT], OpResultT]", np.lib.mixins.NDArrayOperatorsMixin.__mul__)
    __rmul__ = cast("Callable[[Any, OpValueT], OpResultT]", np.lib.mixins.NDArrayOperatorsMixin.__rmul__)
    __truediv__ = cast("Callable[[Any, OpValueT], OpResultT]", np.lib.mixins.NDArrayOperatorsMixin.__truediv__)
    __rtruediv__ = cast("Callable[[Any, OpValueT], OpResultT]", np.lib.mixins.NDArrayOperatorsMixin.__rtruediv__)
    __floordiv__ = cast("Callable[[Any, OpValueT], OpResultT]", np.lib.mixins.NDArrayOperatorsMixin.__floordiv__)
    __rfloordiv__ = cast("Callable[[Any, OpValueT], OpResultT]", np.lib.mixins.NDArrayOperatorsMixin.__rfloordiv__)
    __mod__ = cast("Callable[[Any, OpValueT], OpResultT]", np.lib.mixins.NDArrayOperatorsMixin.__mod__)
    __rmod__ = cast("Callable[[Any, OpValueT], OpResultT]", np.lib.mixins.NDArrayOperatorsMixin.__rmod__)
    __matmul__ = cast("Callable[[Any, OpValueT], OpResultT]", np.lib.mixins.NDArrayOperatorsMixin.__matmul__)
    __rmatmul__ = cast("Callable[[Any, OpValueT], OpResultT]", np.lib.mixins.NDArrayOperatorsMixin.__rmatmul__)
    __and__ = cast("Callable[[Any, OpValueT], OpResultT]", np.lib.mixins.NDArrayOperatorsMixin.__and__)
    __rand__ = cast("Callable[[Any, OpValueT], OpResultT]", np.lib.mixins.NDArrayOperatorsMixin.__rand__)
    __or__ = cast("Callable[[Any, OpValueT], OpResultT]", np.lib.mixins.NDArrayOperatorsMixin.__or__)
    __ror__ = cast("Callable[[Any, OpValueT], OpResultT]", np.lib.mixins.NDArrayOperatorsMixin.__ror__)
    __xor__ = cast("Callable[[Any, OpValueT], OpResultT]", np.lib.mixins.NDArrayOperatorsMixin.__xor__)
    __rxor__ = cast("Callable[[Any, OpValueT], OpResultT]", np.lib.mixins.NDArrayOperatorsMixin.__rxor__)
    __lshift__ = cast("Callable[[Any, OpValueT], OpResultT]", np.lib.mixins.NDArrayOperatorsMixin.__lshift__)
    __rlshift__ = cast("Callable[[Any, OpValueT], OpResultT]", np.lib.mixins.NDArrayOperatorsMixin.__rlshift__)
    __rshift__ = cast("Callable[[Any, OpValueT], OpResultT]", np.lib.mixins.NDArrayOperatorsMixin.__rshift__)
    __rrshift__ = cast("Callable[[Any, OpValueT], OpResultT]", np.lib.mixins.NDArrayOperatorsMixin.__rrshift__)
    __pow__ = cast("Callable[[Any, OpValueT], OpResultT]", np.lib.mixins.NDArrayOperatorsMixin.__pow__)
    __rpow__ = cast("Callable[[Any, OpValueT], OpResultT]", np.lib.mixins.NDArrayOperatorsMixin.__rpow__)
    __eq__ = cast("Callable[[Any, OpValueT], CompResultT]", np.lib.mixins.NDArrayOperatorsMixin.__eq__)
    __ne__ = cast("Callable[[Any, OpValueT], CompResultT]", np.lib.mixins.NDArrayOperatorsMixin.__ne__)
    __lt__ = cast("Callable[[Any, OpValueT], CompResultT]", np.lib.mixins.NDArrayOperatorsMixin.__lt__)
    __le__ = cast("Callable[[Any, OpValueT], CompResultT]", np.lib.mixins.NDArrayOperatorsMixin.__le__)
    __gt__ = cast("Callable[[Any, OpValueT], CompResultT]", np.lib.mixins.NDArrayOperatorsMixin.__gt__)
    __ge__ = cast("Callable[[Any, OpValueT], CompResultT]", np.lib.mixins.NDArrayOperatorsMixin.__ge__)
    __pos__ = cast("Callable[[Any], OpResultT]", np.lib.mixins.NDArrayOperatorsMixin.__pos__)
    __neg__ = cast("Callable[[Any], OpResultT]", np.lib.mixins.NDArrayOperatorsMixin.__neg__)
    __abs__ = cast("Callable[[Any], OpResultT]", np.lib.mixins.NDArrayOperatorsMixin.__abs__)
    __invert__ = cast("Callable[[Any], OpResultT]", np.lib.mixins.NDArrayOperatorsMixin.__invert__)
    __divmod__ = cast(
        "Callable[[Any, OpValueT], tuple[OpResultT, OpResultT]]", np.lib.mixins.NDArrayOperatorsMixin.__divmod__
    )
    __rdivmod__ = cast(
        "Callable[[Any, OpValueT], tuple[OpResultT, OpResultT]]", np.lib.mixins.NDArrayOperatorsMixin.__rdivmod__
    )

    __iadd__ = cast("Callable[[Any, OpValueT], Self]", np.lib.mixins.NDArrayOperatorsMixin.__iadd__)
    __isub__ = cast("Callable[[Any, OpValueT], Self]", np.lib.mixins.NDArrayOperatorsMixin.__isub__)
    __imul__ = cast("Callable[[Any, OpValueT], Self]", np.lib.mixins.NDArrayOperatorsMixin.__imul__)
    __itruediv__ = cast("Callable[[Any, OpValueT], Self]", np.lib.mixins.NDArrayOperatorsMixin.__itruediv__)
    __ifloordiv__ = cast("Callable[[Any, OpValueT], Self]", np.lib.mixins.NDArrayOperatorsMixin.__ifloordiv__)
    __imod__ = cast("Callable[[Any, OpValueT], Self]", np.lib.mixins.NDArrayOperatorsMixin.__imod__)
    __ipow__ = cast("Callable[[Any, OpValueT], Self]", np.lib.mixins.NDArrayOperatorsMixin.__ipow__)
    __imatmul__ = cast("Callable[[Any, OpValueT], Self]", np.lib.mixins.NDArrayOperatorsMixin.__imatmul__)
    __iand__ = cast("Callable[[Any, OpValueT], Self]", np.lib.mixins.NDArrayOperatorsMixin.__iand__)
    __ior__ = cast("Callable[[Any, OpValueT], Self]", np.lib.mixins.NDArrayOperatorsMixin.__ior__)
    __ixor__ = cast("Callable[[Any, OpValueT], Self]", np.lib.mixins.NDArrayOperatorsMixin.__ixor__)
    __ilshift__ = cast("Callable[[Any, OpValueT], Self]", np.lib.mixins.NDArrayOperatorsMixin.__ilshift__)
    __irshift__ = cast("Callable[[Any, OpValueT], Self]", np.lib.mixins.NDArrayOperatorsMixin.__irshift__)
