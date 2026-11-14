"""Tests for ``NumpyAxisProtected`` ``__init_subclass__`` validation paths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
import pytest

from ndlike.protected_axis.numpy import (
    NumpyAxisProtected,
    _validate_field_ndim,
)


def test_validate_field_ndim_raises_when_too_low() -> None:
    """Helper raises ValueError when ndim is less than required protected axes."""
    with pytest.raises(ValueError, match="ndim 1, expected >= 2"):
        _validate_field_ndim("field", 1, 2)


def test_validate_field_ndim_passes_when_sufficient() -> None:
    """Helper returns silently when ndim is sufficient."""
    _validate_field_ndim("field", 2, 2)
    _validate_field_ndim("field", 5, 2)


def test_subclass_requires_non_empty_protected_axes() -> None:
    """Subclasses must define ``protected_axes`` as a non-empty dict."""
    with pytest.raises(TypeError, match="must define protected_axes"):

        @dataclass(frozen=True, slots=True)
        class _Empty(NumpyAxisProtected):
            array: np.ndarray
            protected_axes: ClassVar[dict[str, int]] = {}


def test_subclass_rejects_non_dict_protected_axes() -> None:
    """Subclasses with non-dict ``protected_axes`` raise TypeError."""
    with pytest.raises(TypeError, match="must define protected_axes"):

        @dataclass(frozen=True, slots=True)
        class _BadAxes(NumpyAxisProtected):
            array: np.ndarray
            protected_axes: ClassVar[Any] = ["array"]


def test_subclass_rejects_empty_string_field_name() -> None:
    """Empty string field names are rejected."""
    with pytest.raises(TypeError, match="non-empty string keys"):

        @dataclass(frozen=True, slots=True)
        class _BadName(NumpyAxisProtected):
            array: np.ndarray
            protected_axes: ClassVar[dict[str, int]] = {"": 1}


def test_subclass_rejects_negative_axis_count() -> None:
    """Negative axis counts are rejected."""
    with pytest.raises(TypeError, match="must be a non-negative integer"):

        @dataclass(frozen=True, slots=True)
        class _Negative(NumpyAxisProtected):
            array: np.ndarray
            protected_axes: ClassVar[dict[str, int]] = {"array": -1}


def test_subclass_accepts_unannotated_protected_field() -> None:
    """Protected fields may be provided by the instance without class annotations."""

    class RuntimeField(NumpyAxisProtected):
        protected_axes: ClassVar[dict[str, int]] = {"array": 1}

        def __init__(self, array: np.ndarray) -> None:
            self.array = array

    array = np.ones((2, 3))
    value = RuntimeField(array)
    assert value.protected_values()["array"] is array
    assert value.shape == (2,)


@pytest.mark.parametrize("slots", [False, True])
def test_subclass_accepts_inherited_protected_field(slots: bool) -> None:
    """Inherited dataclass fields support indexing without redeclaration."""

    @dataclass(frozen=True, slots=slots)
    class Parent(NumpyAxisProtected):
        array: np.ndarray
        protected_axes: ClassVar[dict[str, int]] = {"array": 1}

    @dataclass(frozen=True, slots=slots)
    class Child(Parent):
        pass

    value = Child(np.arange(6).reshape(2, 3))
    indexed = value[0]
    assert isinstance(indexed, Child)
    np.testing.assert_array_equal(indexed.array, value.array[0])


def test_subclass_rejects_non_dict_permitted_ufuncs() -> None:
    """``permitted_ufuncs`` must be a dict."""
    with pytest.raises(TypeError, match="permitted_ufuncs must be a dict"):

        @dataclass(frozen=True, slots=True)
        class _BadUfuncs(NumpyAxisProtected):
            array: np.ndarray
            protected_axes: ClassVar[dict[str, int]] = {"array": 1}
            permitted_ufuncs: ClassVar[Any] = [np.add]


def test_subclass_rejects_non_ufunc_key() -> None:
    """Keys of ``permitted_ufuncs`` must be ``np.ufunc`` instances."""
    with pytest.raises(TypeError, match="permitted_ufuncs keys must be numpy ufuncs"):

        @dataclass(frozen=True, slots=True)
        class _NonUfuncKey(NumpyAxisProtected):
            array: np.ndarray
            protected_axes: ClassVar[dict[str, int]] = {"array": 1}
            permitted_ufuncs: ClassVar[Any] = {"not-a-ufunc": ["__call__"]}


def test_subclass_rejects_non_list_methods() -> None:
    """Values of ``permitted_ufuncs`` must be ``list[str]``."""
    with pytest.raises(TypeError, match="must be a list"):

        @dataclass(frozen=True, slots=True)
        class _BadMethods(NumpyAxisProtected):
            array: np.ndarray
            protected_axes: ClassVar[dict[str, int]] = {"array": 1}
            permitted_ufuncs: ClassVar[Any] = {np.add: "__call__"}


def test_subclass_rejects_non_string_methods() -> None:
    """Methods inside ``permitted_ufuncs`` value lists must be strings."""
    with pytest.raises(TypeError, match="must be a list"):

        @dataclass(frozen=True, slots=True)
        class _NumberMethod(NumpyAxisProtected):
            array: np.ndarray
            protected_axes: ClassVar[dict[str, int]] = {"array": 1}
            permitted_ufuncs: ClassVar[Any] = {np.add: [1]}


def test_subclass_rejects_non_set_permitted_functions() -> None:
    """``permitted_functions`` must be a set of callables."""
    with pytest.raises(TypeError, match="permitted_functions must be a set"):

        @dataclass(frozen=True, slots=True)
        class _BadFuncs(NumpyAxisProtected):
            array: np.ndarray
            protected_axes: ClassVar[dict[str, int]] = {"array": 1}
            permitted_functions: ClassVar[Any] = [np.mean]


def test_subclass_rejects_non_callable_in_permitted_functions() -> None:
    """Members of ``permitted_functions`` must be callable."""
    with pytest.raises(TypeError, match="permitted_functions must be a set"):

        @dataclass(frozen=True, slots=True)
        class _NonCallable(NumpyAxisProtected):
            array: np.ndarray
            protected_axes: ClassVar[dict[str, int]] = {"array": 1}
            permitted_functions: ClassVar[set[Any]] = {"not-callable"}
