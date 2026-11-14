"""Utilities for representations with protected trailing tensor axes."""

from __future__ import annotations

from abc import ABC
from dataclasses import replace
from inspect import isabstract
from typing import TYPE_CHECKING, Any, ClassVar, Never, Self, cast, overload, override

import numpy as np
import torch

from ndlike.protected_axis._common import (
    _AxisProtectedBase,
    primary_protected_name,
    protected_shape,
    require_effective_fields,
    value_ndim,
    value_shape,
)
from ndlike.protected_axis.torch_functions import torch_function
from ndlike.torch_like import AnyTorchLike, TorchLike, TorchLikeImplementation

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping
    from types import ModuleType

    from ndlike.array_like import ToIndices


type AnyTorchAxisProtected = TorchAxisProtected[Any, Any, Any, Any, Any, Any, Any]  # noqa: BKN001 - broad backend protocol alias
type TorchProtectedValue = TorchLike[Any] | torch.Tensor | np.ndarray


def _validate_field_ndim(name: str, ndim: int, protected_axes: int) -> None:
    if ndim < protected_axes:
        msg = f"Protected field {name!r} has ndim {ndim}, expected >= {protected_axes}."
        raise ValueError(msg)


class TorchAxisProtected[
    OwnerT: AnyTorchLike = TorchLike[Any],
    OpValueT = OwnerT,
    OpResultT = OwnerT,
    SetValueT = Never,
    ShapeResultT = OwnerT,
    CompResultT = torch.Tensor,
    SetResultT = OwnerT,
](
    _AxisProtectedBase[TorchProtectedValue, OwnerT | OpResultT | ShapeResultT | CompResultT | SetResultT],
    TorchLikeImplementation[OwnerT, OpValueT, OpResultT, SetValueT, ShapeResultT, CompResultT, SetResultT],
    ABC,
):
    """ABC for representations with protected torch-like fields and optional NumPy sidecars."""

    protected_axes: ClassVar[dict[str, int]] = {}
    permitted_functions: ClassVar[set[Callable[..., Any]]] = set()
    primary_protected_name: ClassVar[str]

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)

        if cls is TorchAxisProtected or isabstract(cls):
            return

        axes = cls.protected_axes
        if not axes:
            msg = f"{cls.__name__} must define protected_axes as a non-empty dict[str, int]."
            raise TypeError(msg)

        cls.primary_protected_name = primary_protected_name(axes)

        permitted_functions = getattr(cls, "permitted_functions", set())
        if not isinstance(permitted_functions, set) or not all(callable(func) for func in permitted_functions):
            msg = f"{cls.__name__}.permitted_functions must be a set of callables."
            raise TypeError(msg)
        cls.permitted_functions = set(permitted_functions)

    def protected_value(self) -> TorchProtectedValue:
        """Return the primary protected value."""
        return getattr(self, type(self).primary_protected_name)

    def _torch_protected_value(self) -> TorchLike[Any] | torch.Tensor:
        """Return the first declared torch-like field, including init=False fields."""
        for name in type(self).protected_axes:
            value = getattr(self, name)
            if not isinstance(value, np.ndarray):
                return value

        msg = "No torch-like protected value is available."
        raise TypeError(msg)

    @overload
    def with_protected_values(self, values: dict[str, TorchProtectedValue]) -> Self: ...

    @overload
    def with_protected_values(
        self, values: dict[str, TorchProtectedValue], func: Callable | None
    ) -> OwnerT | OpResultT | ShapeResultT | CompResultT | SetResultT: ...

    def with_protected_values(
        self,
        values: dict[str, TorchProtectedValue],
        func: Callable | None = None,  # noqa: ARG002
    ) -> OwnerT | OpResultT | ShapeResultT | CompResultT | SetResultT:
        """Return a copy with updated protected field values."""
        return replace(self, **values)  # ty:ignore[invalid-argument-type]

    @override
    def __len__(self) -> int:
        if self.ndim == 0:
            msg = "len() of unsized object"
            raise TypeError(msg)
        return self.shape[0]

    @override
    def __iter__(self) -> Iterator[ShapeResultT]:
        for index in range(len(self)):
            yield self[index]

    @override
    def __array_namespace__(self, /, *, api_version: str | None = None) -> ModuleType:
        value = self._torch_protected_value()
        namespace = getattr(value, "__array_namespace__", None)
        if not callable(namespace):
            msg = "Underlying tensor does not support __array_namespace__."
            raise NotImplementedError(msg)
        return namespace(api_version=api_version)

    @override
    @property
    def dtype(self) -> torch.dtype:
        return self._torch_protected_value().dtype

    @override
    @property
    def device(self) -> torch.device:
        return self._torch_protected_value().device

    @property
    def protected_shape(self) -> tuple[int, ...]:
        """Protected trailing shape of the primary field."""
        primary_name = type(self).primary_protected_name
        axes = type(self).protected_axes[primary_name]
        return protected_shape(value_shape(self.protected_value()), axes)

    @override
    @property
    def mT(self) -> ShapeResultT:
        if self.ndim < 2:
            msg = "mT requires at least 2 batch dimensions."
            raise ValueError(msg)
        return self.transpose(-2, -1)

    @override
    @property
    def mH(self) -> ShapeResultT:
        if self.ndim < 2:
            msg = "mH requires at least 2 batch dimensions."
            raise ValueError(msg)
        return cast("ShapeResultT", torch.adjoint(self))  # ty: ignore[invalid-argument-type]

    def _index_with_protected_axes(self, index: ToIndices, protected_axes_count: int) -> tuple[Any, ...]:
        index_tuple = index if isinstance(index, tuple) else (index,)
        return (*index_tuple, *(slice(None),) * protected_axes_count)

    def _prepare_assignment_value(self, value: object) -> object:
        """Prepare a replacement value before field extraction and shape validation.

        Subclasses may convert the value to match destination metadata or raise
        if incompatible. Return the original value or a converted replacement
        without mutating the supplied object. Assignment retains destination metadata.
        """
        return value

    def _coerce_assignment_value(self, value: object) -> Mapping[str, object]:
        field_names = tuple(self._effective_protected_axes.keys())

        if isinstance(value, type(self)):
            candidate_values: Mapping[str, object] = value.protected_values()
        elif isinstance(value, tuple):
            if len(value) != len(field_names):
                msg = f"Expected tuple with {len(field_names)} values for assignment."
                raise TypeError(msg)
            candidate_values = dict(zip(field_names, value, strict=True))
        elif len(field_names) == 1:
            candidate_values = {field_names[0]: value}
        else:
            msg = "Assignment to multi-field protected object requires matching instance or value tuple."
            raise TypeError(msg)

        current_values = self.protected_values()
        for name, axes in self._effective_protected_axes.items():
            candidate = candidate_values[name]
            if axes == 0:
                continue

            candidate_ndim = value_ndim(candidate)
            candidate_shape = value_shape(candidate)
            _validate_field_ndim(name, candidate_ndim, axes)

            current_shape = value_shape(current_values[name])
            if protected_shape(candidate_shape, axes) != protected_shape(current_shape, axes):
                msg = f"Assigned value for field {name!r} modifies protected trailing axes."
                raise ValueError(msg)

        return candidate_values

    @override
    def __getitem__(self, index: ToIndices, /) -> ShapeResultT:
        require_effective_fields(self._effective_protected_axes)
        values = self.protected_values()
        indexed: dict[str, TorchProtectedValue] = {}

        for name, axes in self._effective_protected_axes.items():
            full_index = self._index_with_protected_axes(index, axes)
            value = values[name]
            result = value[full_index]

            if axes == 0 and not hasattr(result, "ndim"):
                result = (
                    np.asarray(result, dtype=value.dtype) if isinstance(value, np.ndarray) else torch.as_tensor(result)
                )

            result_ndim = value_ndim(result)
            result_shape = value_shape(result)
            _validate_field_ndim(name, result_ndim, axes)

            original_shape = value_shape(values[name])
            if protected_shape(result_shape, axes) != protected_shape(original_shape, axes):
                msg = f"Indexing field {name!r} modified protected trailing axes."
                raise IndexError(msg)

            indexed[name] = result

        return cast("ShapeResultT", self.with_protected_values(indexed))

    @override
    def __setitem__(self, index: ToIndices, value: Self | SetValueT, /) -> None:
        require_effective_fields(self._effective_protected_axes)
        prepared = self._prepare_assignment_value(value)
        values = self.protected_values()
        candidate_values = self._coerce_assignment_value(prepared)

        for name, axes in self._effective_protected_axes.items():
            full_index = self._index_with_protected_axes(index, axes)
            cast("Any", values[name])[full_index] = candidate_values[name]

    @override
    def to(self, *args: Any, **kwargs: Any) -> Self:
        values = self.protected_values()
        updates: dict[str, TorchProtectedValue] = {}
        changed = False

        for name, value in values.items():
            if isinstance(value, np.ndarray):
                updates[name] = value
                continue

            converted = value.to(*args, **kwargs)
            updates[name] = converted
            changed = changed or converted is not value

        if not changed:
            return self

        return self.with_protected_values(updates)

    def __torch_like__(
        self,
        dtype: torch.dtype | None = None,
        /,
        *,
        device: torch.device | str | None = None,
        copy: bool = False,
    ) -> Self:
        return self.to(dtype=dtype, device=device, copy=copy)

    def _postprocess_elementwise_result(
        self,
        values: dict[str, TorchProtectedValue],
        *,
        func: Callable,
        operands: tuple[object, ...],
    ) -> dict[str, TorchProtectedValue]:
        """Adjust arithmetic fields before validation and reconstruction.

        Args:
            values: Computed values for every protected field.
            func: The triggering torch function.
            operands: Original operands in expression order.

        Returns:
            The field values to validate and reconstruct. This hook is not
            called for comparisons or explicit output destinations.
        """
        del func, operands
        return values

    @classmethod
    def __torch_function__(
        cls,
        func: Callable,
        types: tuple[type[Any], ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:  # noqa: ANN401
        del cls
        return torch_function(func, types, args, {} if kwargs is None else kwargs)

    @override
    def numpy(self, *, force: bool = False) -> np.ndarray:
        if len(self._effective_protected_axes) != 1:
            msg = "Cannot convert multi-field protected object to a single numpy array."
            raise TypeError(msg)
        value = next(iter(self.protected_values().values()))
        if isinstance(value, np.ndarray):
            return value.copy() if force else value
        return value.numpy(force=force)

    @override
    def detach(self) -> Self:
        values = {
            name: value if isinstance(value, np.ndarray) else value.detach()
            for name, value in self.protected_values().items()
        }
        return self.with_protected_values(values)

    def reshape(self, *shape: int | tuple[int, ...]) -> ShapeResultT:
        """Return a copy with reshaped protected values."""
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]

        return cast("ShapeResultT", torch.reshape(self, shape))  # ty: ignore[invalid-argument-type]

    def gather(self, dim: int, index: torch.Tensor) -> ShapeResultT:
        """Return a copy with gathered protected values along a batch dimension."""
        return cast("ShapeResultT", torch.gather(self, dim, index))  # ty: ignore[invalid-argument-type]
