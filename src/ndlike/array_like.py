"""Protocol for ndarray-like objects."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Sequence
import math
from types import EllipsisType, ModuleType
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Never,
    Protocol,
    Self,
    SupportsIndex,
    cast,
    overload,
    override,
    runtime_checkable,
)

from flextype import flexdispatch
import numpy as np
from numpy.typing import NDArray

from ndlike._operators import ArrayOperators, InplaceArrayOperators, NumpyOperatorsMixin

if TYPE_CHECKING:
    from numpy.typing import DTypeLike


@runtime_checkable
class _SupportsArray[T: np.dtype](Protocol):
    def __array__(self) -> np.ndarray[Any, T]: ...


type _NestedSequence[T] = T | Sequence[_NestedSequence[T]]

type _DualArrayLike[DT: np.dtype, T] = _NestedSequence[_SupportsArray[DT]] | _NestedSequence[T]
type _ArrayLikeInt_co = _DualArrayLike[np.dtype[np.bool | np.integer], int]
type ToIndex = SupportsIndex | slice | EllipsisType | _ArrayLikeInt_co | None
type ToIndices = ToIndex | tuple[ToIndex, ...]
type Order = Literal["C", "F", "A", "K"]
type ArrayAPIVersion = Literal["2022.12", "2023.12", "2024.12", "2025.12"]

_GetFlag = Literal[
    "C",
    "CONTIGUOUS",
    "C_CONTIGUOUS",
    "F",
    "FORTRAN",
    "F_CONTIGUOUS",
    "W",
    "WRITEABLE",
    "B",
    "BEHAVED",
    "O",
    "OWNDATA",
    "A",
    "ALIGNED",
    "X",
    "WRITEBACKIFCOPY",
    "CA",
    "CARRAY",
    "FA",
    "FARRAY",
    "FNC",
    "FORC",
]

_SetFlag = Literal[
    "A",
    "ALIGNED",
    "W",
    "WRITEABLE",
    "X",
    "WRITEBACKIFCOPY",
]


class ArrayFlagsLike(Protocol):
    """Protocol for array flags."""

    aligned: bool
    writeable: bool
    writebackifcopy: bool

    @property
    def behaved(self) -> bool:
        """True if the array is well-behaved, i.e. it is aligned and writeable."""

    @property
    def c_contiguous(self) -> bool:
        """True if the array is C-contiguous."""

    @property
    def carray(self) -> bool:
        """True if the array is C-contiguous."""

    @property
    def contiguous(self) -> bool:
        """True if the array is contiguous."""

    @property
    def f_contiguous(self) -> bool:
        """True if the array is Fortran-contiguous."""

    @property
    def farray(self) -> bool:
        """True if the array is Fortran-contiguous."""

    @property
    def fnc(self) -> bool:
        """True if the array is Fortran-contiguous but not C-contiguous."""

    @property
    def forc(self) -> bool:
        """True if the array is C-contiguous or Fortran-contiguous."""

    @property
    def fortran(self) -> bool:
        """True if the array is Fortran-contiguous."""

    @property
    def owndata(self) -> bool:
        """True if the array owns its own data."""

    def __getitem__(self, key: _GetFlag, /) -> bool:
        """Get the value of a flag."""

    def __setitem__(self, key: _SetFlag, value: bool, /) -> None:
        """Set the value of a flag."""


@runtime_checkable
class WeakArrayLike[
    OwnerT: WeakArrayLike[Any, Any, Any, Any, Any, Any, Any] = WeakArrayLike[Any],
    OpValueT = OwnerT,
    OpResultT = OwnerT,
    SetValueT = Never,
    ShapeResultT = OwnerT,
    CompResultT = WeakArrayLike[Any, Any, Any, Never, Any, Any, Any],
    SetResultT = OwnerT,
](Protocol):
    """Core array protocol shared by NumPy, JAX, and PyTorch.

    This protocol omits the integer ``size`` property, ``__array_namespace__`` and
    ``to_device``, which PyTorch does not expose. Assignment accepts ``Self`` plus
    the additional values described by ``SetValueT``, which defaults to ``Never``
    (no additional types). Index types are backend-specific.
    A matching array may still reject an operation, such as item assignment on
    immutable JAX arrays.

    Unlike `numpy.typing.ArrayLike`, this protocol does not only check
    whether an object can be converted to a numpy array, but rather
    whether it supports core indexing and shape/metadata operations similar to a numpy array.
    This protocol is based in the Python array API standard: https://data-apis.org/array-api
    Unlike the official standard, this protocol does not require support for all numerical operations.
    """

    if not TYPE_CHECKING:

        def __class_getitem__(cls, arguments: object) -> Any:  # noqa: ANN401
            """Substitute dependent defaults before constructing a generic alias."""
            # CPython does not substitute a preceding TypeVar used as a default.
            # Fill those defaults before Generic collects subclass parameters.
            args = list(arguments if isinstance(arguments, tuple) else (arguments,))
            parameters = cls.__parameters__
            for parameter in parameters[len(args) :]:
                if not parameter.has_default():
                    break
                default = parameter.__default__
                for index, preceding in enumerate(parameters[: len(args)]):
                    if default is preceding:
                        default = args[index]
                        break
                args.append(default)
            return super().__class_getitem__(tuple(args))

    @overload
    def __array__(self, dtype: None = None, /) -> np.ndarray: ...

    @overload
    def __array__(self, dtype: np.dtype, /) -> np.ndarray: ...

    def __array__(self, dtype: np.dtype | None = None, /) -> np.ndarray:
        """Convert to a NumPy array using an optional dtype object."""

    def __getitem__(self, index: Any, /) -> ShapeResultT:  # noqa: ANN401
        """Return values selected by a backend-supported index."""

    def __setitem__(self, index: Any, value: Self | SetValueT, /) -> None:  # noqa: ANN401
        """Assign a backend-supported value, if the array permits mutation."""

    @property
    def ndim(self) -> int:
        """Number of dimensions."""

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the array."""

    @property
    def dtype(self) -> Any:  # noqa: ANN401
        """Data type of the array."""

    @property
    def device(self) -> Any:  # noqa: ANN401
        """Device of the array."""

    @property
    def T(self) -> ShapeResultT:  # noqa: N802
        """Transposed array."""

    @property
    def mT(self) -> ShapeResultT:  # noqa: N802
        """Matrix transposed array."""

    def __len__(self) -> int:
        """Length along the first axis."""

    def __iter__(self) -> Iterator[ShapeResultT]:
        """Iterator over the first axis."""


@runtime_checkable
class ArrayLike[
    OwnerT: ArrayLike[Any, Any, Any, Any, Any, Any, Any] = ArrayLike[Any],
    OpValueT = OwnerT,
    OpResultT = OwnerT,
    SetValueT = Never,
    ShapeResultT = OwnerT,
    CompResultT = WeakArrayLike[Any, Any, Any, Never, Any, Any, Any],
    SetResultT = OwnerT,
](
    WeakArrayLike[OwnerT, OpValueT, OpResultT, SetValueT, ShapeResultT, CompResultT, SetResultT],
    Protocol,
):
    """NumPy/JAX array protocol with an integer element-count property.

    Adds namespace access, device transfer and copy-aware NumPy conversion to the
    shared core. Device types are backend-specific: NumPy accepts ``"cpu"``, while
    JAX accepts a device or sharding object.

    Parameters are ``OwnerT``, ``OpValueT``, ``OpResultT``, ``SetValueT``,
    ``ShapeResultT``, ``CompResultT``, and ``SetResultT``. Operands and value,
    shape/indexing, and functional-update results default to the owner type. Explicit
    result arguments replace that default. Copies return OwnerT; iteration yields
    ShapeResultT. Comparisons default to WeakArrayLike.
    Assignment accepts ``Self | SetValueT``; ``SetValueT`` defaults to Never.
    The owner defaults to a broad ArrayLike and must satisfy the array protocol.
    Numerical operators are required only by the backend-specific protocols.
    Those protocols restrict owners and comparison defaults to their backend.
    """

    @overload
    def __array__(self, dtype: None = None, /, *, copy: bool | None = None) -> np.ndarray: ...

    @overload
    def __array__(self, dtype: np.dtype, /, *, copy: bool | None = None) -> np.ndarray: ...

    def __array__(self, dtype: np.dtype | None = None, /, *, copy: bool | None = None) -> np.ndarray:
        """Convert to a NumPy array using an optional dtype object and copy policy."""

    def __array_namespace__(self, /, *, api_version: ArrayAPIVersion | None = None) -> ModuleType:
        """Return the array namespace."""

    def to_device(self, device: Any, /, *, stream: int | Any | None = None) -> Self:  # noqa: ANN401
        """Move the array to a backend-supported device."""

    def __getitem__(self, index: ToIndices, /) -> ShapeResultT:
        """Return values selected by a NumPy-style index."""

    def __setitem__(self, index: ToIndices, value: Self | SetValueT, /) -> None:
        """Assign values, if the array permits mutation."""

    @property
    def size(self) -> int:
        """Number of elements in the array."""


@runtime_checkable
class NumpyLikeConvertible[SetValueT = Never](ArrayLike[Any, Any, Any, SetValueT], Protocol):
    """Protocol for array-like objects that can be converted to numpy arrays."""

    def __array_like__(self, dtype: DTypeLike | None = None, /, *, copy: bool | None = None) -> NumpyLike[Any]:
        """Convert to a NumpyLike."""


type AnyNumpyLike = NumpyLike[Any, Any, Any, Any, Any, Any, Any]  # noqa: BKN001 - broad backend protocol alias


@runtime_checkable
class NumpyLike[
    OwnerT: AnyNumpyLike = NumpyLike[Any],
    OpValueT = OwnerT,
    OpResultT = OwnerT,
    SetValueT = Never,
    ShapeResultT = OwnerT,
    CompResultT = NDArray[np.bool_],
    SetResultT = OwnerT,
](
    ArrayLike[OwnerT, OpValueT, OpResultT, SetValueT, ShapeResultT, CompResultT, SetResultT],
    ArrayOperators[OpValueT, OpResultT, CompResultT],
    InplaceArrayOperators[OpValueT],
    Protocol,
):
    """Protocol for array-like objects that implement NumPy-specific APIs."""

    def __imatmul__(self, other: OpValueT, /) -> Self:
        """Apply matrix multiplication with this array as the output."""

    def to_device(self, device: Literal["cpu"], /, *, stream: int | Any | None = None) -> Self:  # noqa: ANN401
        """Move the array to the CPU."""

    @property
    def dtype(self) -> np.dtype:
        """Data type of the array."""

    def __array_ufunc__(self, ufunc: np.ufunc, method: str, *inputs: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        """Handle numpy ufuncs."""

    def __array_function__(
        self,
        func: Callable,
        types: tuple[type[Any], ...],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:  # noqa: ANN401
        """Handle numpy array functions."""

    @property
    def mT(self) -> ShapeResultT:  # noqa: N802
        """Matrix-transposed view."""

    @property
    def T(self) -> ShapeResultT:  # noqa: N802
        """Transposed view."""

    @overload
    def transpose(self, *axes: int) -> ShapeResultT: ...

    @overload
    def transpose(self, axes: Sequence[int] | None, /) -> ShapeResultT: ...

    def transpose(self, *axes: int | Sequence[int] | None) -> ShapeResultT:
        """Return transposed array."""

    def copy(self, order: Order = "C") -> OwnerT:
        """Return array copy."""

    def astype(
        self,
        dtype: DTypeLike,
        order: Order = "K",
        casting: Literal["no", "equiv", "safe", "same_kind", "unsafe"] = "unsafe",
        subok: bool = True,
        copy: bool = True,
    ) -> OpResultT:
        """Return cast array."""

    @property
    def flags(self) -> ArrayFlagsLike:
        """The flags of the array."""


class NumpyLikeImplementation[
    OwnerT: AnyNumpyLike = NumpyLike[Any],
    OpValueT = OwnerT,
    OpResultT = OwnerT,
    SetValueT = Never,
    ShapeResultT = OwnerT,
    CompResultT = NDArray[np.bool_],
    SetResultT = OwnerT,
](
    NumpyOperatorsMixin[OpValueT, OpResultT, CompResultT],
    ArrayLike[OwnerT, OpValueT, OpResultT, SetValueT, ShapeResultT, CompResultT, SetResultT],
    ABC,
):
    """ABC implementation for array-like objects that behave like NumPy arrays.

    Representation-specific conversion and metadata are explicitly abstract rather
    than inherited as protocol placeholders. Unsupported indexing, iteration and
    scalar conversions retain their concrete ``NotImplementedError`` fallbacks.

    Dataclass subclasses retain inherited array equality unless they explicitly
    define ``__eq__``. Hashing follows the ordinary dataclass rules.
    """

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Preserve inherited equality before a subclass's dataclass decorator runs."""
        super().__init_subclass__(**kwargs)
        if "__eq__" not in cls.__dict__:
            setattr(cls, "__eq__", cls.__eq__)  # noqa: B010 - do not redeclare the typed inherited method

    @abstractmethod
    def __array__(self, dtype: DTypeLike | None = None, /, *, copy: bool | None = None) -> np.ndarray:
        """Convert to a NumPy array."""

    @abstractmethod
    def __array_namespace__(self, /, *, api_version: ArrayAPIVersion | None = None) -> ModuleType:
        """Return the array namespace."""

    @property
    @abstractmethod
    def device(self) -> Any:  # noqa: ANN401
        """Device of the array."""

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return len(self.shape)

    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...]:
        """Shape of the array."""

    @property
    def size(self) -> int:
        """Number of elements in the array."""
        return math.prod(self.shape)

    def __getitem__(self, index: ToIndices, /) -> ShapeResultT:
        """Return a new array containing the indexed values."""
        msg = f"{type(self).__name__} does not support indexing."
        raise NotImplementedError(msg)

    def __setitem__(self, index: ToIndices, value: Self | SetValueT, /) -> None:
        """Set the values at the given indices to the given value."""
        msg = f"{type(self).__name__} does not support item assignment."
        raise NotImplementedError(msg)

    @override
    @property
    @abstractmethod
    def dtype(self) -> np.dtype:
        """Data type of the array."""

    @abstractmethod
    def __array_ufunc__(self, ufunc: np.ufunc, method: str, *inputs: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        """Handle numpy ufuncs.

        Args:
            ufunc: The ufunc to apply.
            method: The method of the ufunc.
            inputs: The input arrays.
            kwargs: Additional keyword arguments.

        Returns:
            The result of applying the ufunc.
        """

    @abstractmethod
    def __array_function__(
        self,
        func: Callable,
        types: tuple[type[Any], ...],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:  # noqa: ANN401
        """Handle numpy array functions.

        Args:
            func: The numpy function to apply.
            types: The types of the input arguments.
            args: The input arguments.
            kwargs: Additional keyword arguments.

        Returns:
            The result of applying the numpy function.
        """

    @property
    def mT(self) -> ShapeResultT:  # noqa: N802
        """The transposed version of the underlying array."""
        return cast("ShapeResultT", np.matrix_transpose(self))

    @property
    def T(self) -> ShapeResultT:  # noqa: N802
        """The transposed version of the underlying array."""
        return cast("ShapeResultT", np.transpose(self))

    def transpose(self, *axes: int | Sequence[int] | None) -> ShapeResultT:
        """Return a transposed version of the NumpySample.

        This method implicitly also provides full axis tracking support for
        - `np.moveaxis`
        - `np.rollaxis`
        Those functions call out to `transpose` methods for custom array types.

        Args:
            axes: The axes to transpose.

        Returns:
            A transposed version of the NumpySample.
        """
        if len(axes) == 0:
            return cast("ShapeResultT", np.transpose(self))
        if len(axes) == 1 and not isinstance(axes[0], int):
            return cast("ShapeResultT", np.transpose(self, axes[0]))
        return np.transpose(self, axes)  # ty:ignore[no-matching-overload]

    def copy(self, order: Order = "C") -> OwnerT:
        """Return a copy of the array."""
        return np.copy(self, order=order, subok=True)  # ty:ignore[no-matching-overload]

    def astype(
        self,
        dtype: DTypeLike,
        order: Order = "K",
        casting: Literal["no", "equiv", "safe", "same_kind", "unsafe"] = "unsafe",
        subok: bool = True,
        copy: bool = True,
    ) -> OpResultT:
        """Copy of the array, cast to a specified type."""
        if order != "K" or casting != "unsafe" or not subok:
            msg = "Non-default ndarray.astype options require a representation-specific implementation."
            raise NotImplementedError(msg)
        return np.astype(self, dtype, copy=copy)  # ty: ignore[no-matching-overload]

    def __index__(self) -> int:
        """Converts 0d integer array to a Python integer."""
        msg = f"{type(self).__name__} does not support conversion to index."
        raise NotImplementedError(msg)

    def __int__(self) -> int:
        """Converts 0d integer array to a Python integer."""
        msg = f"{type(self).__name__} does not support conversion to int."
        raise NotImplementedError(msg)

    def __bool__(self) -> bool:
        """Converts 0d boolean array to a Python boolean."""
        msg = f"{type(self).__name__} does not support conversion to bool."
        raise NotImplementedError(msg)

    def __float__(self) -> float:
        """Converts 0d float array to a Python float."""
        msg = f"{type(self).__name__} does not support conversion to float."
        raise NotImplementedError(msg)

    def __complex__(self) -> complex:
        """Converts 0d complex array to a Python complex."""
        msg = f"{type(self).__name__} does not support conversion to complex."
        raise NotImplementedError(msg)

    def __len__(self) -> int:
        """Returns the length of the array along the first dimension.

        This is not mandated by the array API standard, but numpy, torch and jax all support it.
        """
        msg = f"{type(self).__name__} does not support len()."
        raise NotImplementedError(msg)

    def __iter__(self) -> Iterator[ShapeResultT]:
        """Returns an iterator over the first dimension of the array.

        This is not mandated by the array API standard, but numpy, torch and jax all support it.
        """
        msg = f"{type(self).__name__} does not support iteration."
        raise NotImplementedError(msg)

    @property
    @abstractmethod
    def flags(self) -> ArrayFlagsLike:
        """The flags of the array."""

    @override
    def to_device(self, device: Any, /, *, stream: Any = None) -> Self:
        """Move the array to a device."""
        return self


NumpyLikeImplementation.register(np.ndarray)


@flexdispatch
def to_numpy_like[DT](
    array: object,
    *,
    dtype: DTypeLike | None = None,
    copy: bool | None = None,
) -> NumpyLike[Any] | np.ndarray:
    """Convert an ArrayLike to a NumpyLike.

    If possible, use the __array_like__ method to convert the array, otherwise use np.asanyarray.

    Args:
        array: The array to convert.
        dtype: The desired data type of the output array.
        copy: Whether to return a copy of the input array.

    Returns:
        The converted array.
    """
    return np.asanyarray(array, dtype=dtype, copy=copy)


@to_numpy_like.register(NumpyLikeConvertible)
def _to_numpy_like_convertible(
    array: NumpyLikeConvertible,
    *,
    dtype: DTypeLike | None = None,
    copy: bool | None = None,
) -> NumpyLike[Any] | NDArray[Any]:
    return array.__array_like__(dtype, copy=copy)
