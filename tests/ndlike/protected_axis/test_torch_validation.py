"""Tests for ``TorchAxisProtected`` ``__init_subclass__`` validation paths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import pytest

pytest.importorskip("torch")
import torch

from ndlike.protected_axis.torch import (
    TorchAxisProtected,
    _validate_field_ndim,
)


def test_validate_field_ndim_raises_when_too_low() -> None:
    """Helper raises ValueError when ndim is less than required protected axes."""
    with pytest.raises(ValueError, match="ndim 1, expected >= 2"):
        _validate_field_ndim("field", 1, 2)


def test_torch_subclass_requires_non_empty_protected_axes() -> None:
    """Torch subclasses must define ``protected_axes`` as a non-empty dict."""
    with pytest.raises(TypeError, match="must define protected_axes"):

        @dataclass(frozen=True, slots=True)
        class _Empty(TorchAxisProtected[Any]):
            tensor: torch.Tensor
            protected_axes: ClassVar[dict[str, int]] = {}


def test_torch_subclass_rejects_non_dict_protected_axes() -> None:
    """Torch subclasses with non-dict ``protected_axes`` raise TypeError."""
    with pytest.raises(TypeError, match="must define protected_axes"):

        @dataclass(frozen=True, slots=True)
        class _NotDict(TorchAxisProtected[Any]):
            tensor: torch.Tensor
            protected_axes: ClassVar[Any] = ["tensor"]


def test_torch_subclass_rejects_empty_string_field_name() -> None:
    """Empty string field names are rejected for torch subclasses."""
    with pytest.raises(TypeError, match="non-empty string keys"):

        @dataclass(frozen=True, slots=True)
        class _BadName(TorchAxisProtected[Any]):
            tensor: torch.Tensor
            protected_axes: ClassVar[dict[str, int]] = {"": 1}


def test_torch_subclass_rejects_negative_axis_count() -> None:
    """Negative axis counts are rejected."""
    with pytest.raises(TypeError, match="must be an int >= 0"):

        @dataclass(frozen=True, slots=True)
        class _Negative(TorchAxisProtected[Any]):
            tensor: torch.Tensor
            protected_axes: ClassVar[dict[str, int]] = {"tensor": -1}


def test_torch_subclass_rejects_non_int_axis_count() -> None:
    """Non-int axis counts are rejected."""
    with pytest.raises(TypeError, match="must be an int >= 0"):

        @dataclass(frozen=True, slots=True)
        class _Fractional(TorchAxisProtected[Any]):
            tensor: torch.Tensor
            protected_axes: ClassVar[Any] = {"tensor": 1.5}


def test_torch_subclass_accepts_unannotated_protected_field() -> None:
    """Protected fields may be provided by the instance without class annotations."""

    class RuntimeField(TorchAxisProtected[torch.Tensor]):
        protected_axes: ClassVar[dict[str, int]] = {"tensor": 1}

        def __init__(self, tensor: torch.Tensor) -> None:
            self.tensor = tensor

    tensor = torch.ones((2, 3))
    value = RuntimeField(tensor)
    assert value.protected_values()["tensor"] is tensor
    assert value.shape == (2,)


@pytest.mark.parametrize("slots", [False, True])
def test_torch_subclass_accepts_inherited_protected_field(slots: bool) -> None:
    """Inherited dataclass fields support indexing without redeclaration."""

    @dataclass(frozen=True, slots=slots)
    class Parent(TorchAxisProtected[torch.Tensor]):
        tensor: torch.Tensor
        protected_axes: ClassVar[dict[str, int]] = {"tensor": 1}

    @dataclass(frozen=True, slots=slots)
    class Child(Parent):
        pass

    value = Child(torch.arange(6).reshape(2, 3))
    indexed = value[0]
    assert isinstance(indexed, Child)
    torch.testing.assert_close(indexed.tensor, value.tensor[0])


def test_torch_subclass_rejects_non_set_permitted_functions() -> None:
    """``permitted_functions`` must be a set of callables."""
    with pytest.raises(TypeError, match="permitted_functions must be a set"):

        @dataclass(frozen=True, slots=True)
        class _BadFuncs(TorchAxisProtected[Any]):
            tensor: torch.Tensor
            protected_axes: ClassVar[dict[str, int]] = {"tensor": 1}
            permitted_functions: ClassVar[Any] = [torch.mean]


def test_torch_subclass_rejects_non_callable_in_permitted_functions() -> None:
    """Members of ``permitted_functions`` must be callable."""
    with pytest.raises(TypeError, match="permitted_functions must be a set"):

        @dataclass(frozen=True, slots=True)
        class _NonCallable(TorchAxisProtected[Any]):
            tensor: torch.Tensor
            protected_axes: ClassVar[dict[str, int]] = {"tensor": 1}
            permitted_functions: ClassVar[set[Any]] = {"not-callable"}
