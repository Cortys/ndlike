"""Protocol for ndarray-like objects."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Protocol, Self, overload, override, runtime_checkable

from flextype import flexdispatch
import numpy as np
import torch

from ndlike.array_like import ArrayLikeWithoutSize

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from types import ModuleType

    from numpy.typing import ArrayLike as NpArrayLike, DTypeLike, NDArray

    from ndlike.array_like import ToIndices


@runtime_checkable
class TorchLikeConvertible[DT](ArrayLikeWithoutSize[DT], Protocol):
    """Protocol for array-like objects that can be converted to torch tensors."""

    def __torch_like__(
        self,
        dtype: torch.dtype | None = None,
        /,
        *,
        device: torch.device | str | None = None,
        copy: bool = False,
    ) -> TorchLike[Any]:
        """Convert to a TorchLike."""


@runtime_checkable
class TorchLike[DT](ArrayLikeWithoutSize[DT], Protocol):
    """Protocol for array-like objects that implement torch-specific APIs."""

    @property
    def T(self) -> TorchLike[Any] | torch.Tensor:  # noqa: N802
        """Return a transposed array."""

    @property
    def mT(self) -> TorchLike[Any] | torch.Tensor:  # noqa: N802
        """The transposed version of the underlying array."""

    @property
    def mH(self) -> TorchLike[Any] | torch.Tensor:  # noqa: N802
        """The adjoint (conjugate) transposed version of the underlying array."""

    @overload
    def size(self, dim: int) -> int: ...

    @overload
    def size(self, dim: None = ...) -> torch.Size: ...

    def size(self, dim: int | None = None) -> int | torch.Size:
        """Return the size of the array along the given dimension."""

    @overload
    def to(
        self,
        dtype: torch.dtype,
        non_blocking: bool = False,
        copy: bool = False,
        *,
        memory_format: torch.memory_format = torch.preserve_format,
    ) -> Self: ...

    @overload
    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        non_blocking: bool = False,
        copy: bool = False,
        *,
        memory_format: torch.memory_format = torch.preserve_format,
    ) -> Self: ...

    @overload
    def to(
        self,
        other: torch.Tensor,
        non_blocking: bool = False,
        copy: bool = False,
    ) -> Self: ...

    def to(self, *args: Any, **kwargs: Any) -> Self:
        """Move and/or cast the tensor, mirroring ``torch.Tensor.to``."""

    def __torch_function__(
        self, func: Callable, types: tuple[type[Any], ...], args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> Any:  # noqa: ANN401
        """Handle torch functions."""

    def numpy(self, *, force: bool = False) -> NDArray[Any]:
        """Convert to a numpy array."""

    def detach(self) -> Self:
        """Return a detached version of the array."""


class TorchLikeImplementation[DT](TorchLike[DT], ABC):
    """ABC implementation for array-like objects that behave like torch tensors.

    Representation-specific indexing, metadata and namespace access are explicitly
    abstract so incomplete subclasses cannot inherit silent protocol placeholders.
    """

    @abstractmethod
    def __getitem__(self, index: ToIndices, /) -> Any:  # noqa: ANN401
        """Return values selected by index."""

    @abstractmethod
    def __setitem__(self, index: ToIndices, value: NpArrayLike, /) -> None:
        """Set values selected by index."""

    @abstractmethod
    def __array_namespace__(self, /, *, api_version: str | None = None) -> ModuleType:
        """Return the array namespace."""

    @abstractmethod
    def __len__(self) -> int:
        """Return the length along the first axis."""

    @abstractmethod
    def __iter__(self) -> Iterator[Any]:
        """Iterate over the first axis."""

    @property
    @abstractmethod
    def dtype(self) -> Any:  # noqa: ANN401
        """Data type of the array."""

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
    @abstractmethod
    def mT(self) -> TorchLike[Any] | torch.Tensor:  # noqa: N802
        """The transposed version of the underlying array."""

    @property
    @abstractmethod
    def mH(self) -> TorchLike[Any] | torch.Tensor:  # noqa: N802
        """The adjoint (conjugate) transposed version of the underlying array."""

    @property
    def T(self) -> TorchLike[Any] | torch.Tensor:  # noqa: N802
        """Inverts the order of the dimensions of the underlying array."""
        return self.permute(*reversed(range(self.ndim)))

    @overload
    def size(self, dim: int) -> int: ...

    @overload
    def size(self, dim: None = ...) -> torch.Size: ...

    def size(self, dim: int | None = None) -> int | torch.Size:
        """Return the size of the array along the given dimension."""
        if dim is None:
            return torch.Size(self.shape)

        normalized_dim = dim + self.ndim if dim < 0 else dim
        if normalized_dim < 0 or normalized_dim >= self.ndim:
            msg = f"dim {dim} out of bounds for dimensions with ndim {self.ndim}."
            raise IndexError(msg)

        return self.shape[normalized_dim]

    @overload
    def to(
        self,
        dtype: torch.dtype,
        non_blocking: bool = False,
        copy: bool = False,
        *,
        memory_format: torch.memory_format = torch.preserve_format,
    ) -> Self: ...

    @overload
    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        non_blocking: bool = False,
        copy: bool = False,
        *,
        memory_format: torch.memory_format = torch.preserve_format,
    ) -> Self: ...

    @overload
    def to(
        self,
        other: torch.Tensor,
        non_blocking: bool = False,
        copy: bool = False,
    ) -> Self: ...

    @abstractmethod
    def to(self, *args: Any, **kwargs: Any) -> Self:
        """Move and/or cast the tensor, mirroring ``torch.Tensor.to``."""

    def cpu(self, memory_format: torch.memory_format = torch.preserve_format) -> Self:
        """Move the array to the CPU."""
        return self.to("cpu", memory_format=memory_format)

    def cuda(
        self,
        device: torch.device | str | None = None,
        non_blocking: bool = False,
        memory_format: torch.memory_format = torch.preserve_format,
    ) -> Self:
        """Move the array to the GPU."""
        if device is None:
            device = "cuda"
        return self.to(
            device=device,
            non_blocking=non_blocking,
            memory_format=memory_format,
        )

    @override
    def to_device(self, device: torch.device | str | None = None, /, *, stream: int | Any | None = None) -> Self:
        """Move the array to the given device.

        Args:
            device: Target device, or None to retain the current device.
            stream: Unsupported; must be None.

        Returns:
            The moved array representation.

        Raises:
            NotImplementedError: If a stream is provided.
        """
        if stream is not None:
            msg = "stream argument of array.to_device() is not supported."
            raise NotImplementedError(msg)
        return self.to(device=device)

    def clone(self, *, memory_format: torch.memory_format = torch.preserve_format) -> TorchLike[Any] | torch.Tensor:
        """Return a copy of the array."""
        return torch.clone(self, memory_format=memory_format)  # ty: ignore[invalid-argument-type]

    def transpose(self, dim0: int, dim1: int) -> TorchLike[Any] | torch.Tensor:
        """Return a transposed version of the array."""
        return torch.transpose(self, dim0, dim1)  # ty: ignore[invalid-argument-type]

    def permute(self, *dims: torch.Size | int | tuple[int] | list[int]) -> TorchLike[Any] | torch.Tensor:
        """Return a permuted version of the array."""
        if len(dims) == 1 and isinstance(dims[0], (tuple, list)):
            dims = dims[0]  # ty:ignore[invalid-assignment]

        return torch.permute(self, dims)  # ty: ignore[invalid-argument-type]

    @abstractmethod
    def __torch_function__(
        self, func: Callable, types: tuple[type[Any], ...], args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> Any:  # noqa: ANN401
        """Handle torch functions.

        Args:
            func: The torch function to apply.
            types: The types of the input arguments.
            args: The input arguments.
            kwargs: Additional keyword arguments.

        Returns:
            The result of applying the torch function.
        """

    @abstractmethod
    def numpy(self, *, force: bool = False) -> NDArray[Any]:
        """Convert to a numpy array."""

    @override
    def __array__(self, dtype: DTypeLike | None = None, /, *, copy: bool | None = None) -> NDArray[Any]:
        """Convert to a numpy array."""
        force = copy is not False and (dtype is not None or copy is True)
        return np.asarray(self.numpy(force=force), dtype=dtype, copy=copy)

    @abstractmethod
    def detach(self) -> Self:
        """Return a detached version of the array."""

    def resolve_conj(self) -> TorchLike[Any] | torch.Tensor:
        """Return a version of the array with any conjugate operations resolved."""
        return torch.resolve_conj(self)  # ty: ignore[invalid-argument-type]

    def resolve_neg(self) -> TorchLike[Any] | torch.Tensor:
        """Return a version of the array with any negation operations resolved."""
        return torch.resolve_neg(self)  # ty: ignore[invalid-argument-type]


TorchLikeImplementation.register(torch.Tensor)


@flexdispatch
def to_torch_like[DT](
    data: object,
    /,
    dtype: torch.dtype | None = None,
    *,
    device: torch.device | str | None = None,
    copy: bool = False,
) -> TorchLike[Any] | torch.Tensor:
    """Convert an ArrayLike to a TorchLike.

    If possible, use the __torch_like__ method to convert the array, otherwise use torch.as_tensor.

    Args:
        data: The data to convert.
        dtype: The desired data type of the output tensor.
        device: The desired device of the output tensor.
        copy: Whether to return a copy of the input data.

    Returns:
        The converted tensor.
    """
    res = torch.as_tensor(data, dtype=dtype, device=device)

    if res is data and copy:
        res = res.clone()

    return res


@to_torch_like.register(TorchLikeConvertible)
def _[DT](
    data: TorchLikeConvertible[DT],
    /,
    dtype: torch.dtype | None = None,
    *,
    device: torch.device | str | None = None,
    copy: bool = False,
) -> TorchLike[Any] | torch.Tensor:
    return data.__torch_like__(dtype, device=device, copy=copy)
