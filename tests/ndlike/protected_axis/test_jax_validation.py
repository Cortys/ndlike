"""Tests for ``JaxAxisProtected`` ``__init_subclass__`` validation paths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import pytest

pytest.importorskip("jax")
import jax
from jax import numpy as jnp

from ndlike.protected_axis.jax import (
    JaxAtMethod,
    JaxAxisProtected,
    _validate_field_ndim,
)


def test_validate_field_ndim_raises_when_too_low() -> None:
    """Helper raises ValueError when ndim is less than required protected axes."""
    with pytest.raises(ValueError, match="ndim 1, expected >= 2"):
        _validate_field_ndim("field", 1, 2)


def test_jax_subclass_requires_non_empty_protected_axes() -> None:
    """Jax subclass must define ``protected_axes`` as a non-empty dict."""
    with pytest.raises(TypeError, match="must define protected_axes"):

        @dataclass(frozen=True, slots=True)
        class _Empty(JaxAxisProtected[Any]):
            array: jax.Array
            protected_axes: ClassVar[dict[str, int]] = {}


def test_jax_subclass_rejects_non_dict_protected_axes() -> None:
    """Jax subclass with non-dict ``protected_axes`` raises TypeError."""
    with pytest.raises(TypeError, match="must define protected_axes"):

        @dataclass(frozen=True, slots=True)
        class _NotDict(JaxAxisProtected[Any]):
            array: jax.Array
            protected_axes: ClassVar[Any] = ["array"]


def test_jax_subclass_rejects_empty_string_field_name() -> None:
    """Empty string field names are rejected for jax subclasses."""
    with pytest.raises(TypeError, match="non-empty string keys"):

        @dataclass(frozen=True, slots=True)
        class _BadName(JaxAxisProtected[Any]):
            array: jax.Array
            protected_axes: ClassVar[dict[str, int]] = {"": 1}


def test_jax_subclass_rejects_negative_axis_count() -> None:
    """Negative axis counts are rejected."""
    with pytest.raises(TypeError, match="must be a non-negative integer"):

        @dataclass(frozen=True, slots=True)
        class _Negative(JaxAxisProtected[Any]):
            array: jax.Array
            protected_axes: ClassVar[dict[str, int]] = {"array": -1}


def test_jax_subclass_rejects_non_int_axis_count() -> None:
    """Non-int axis counts are rejected."""
    with pytest.raises(TypeError, match="must be a non-negative integer"):

        @dataclass(frozen=True, slots=True)
        class _Fractional(JaxAxisProtected[Any]):
            array: jax.Array
            protected_axes: ClassVar[Any] = {"array": 1.5}


def test_jax_subclass_accepts_unannotated_protected_field() -> None:
    """Protected fields may be provided by the instance without class annotations."""

    class RuntimeField(JaxAxisProtected[jax.Array]):
        protected_axes: ClassVar[dict[str, int]] = {"array": 1}

        def __init__(self, array: jax.Array) -> None:
            self.array = array

    array = jnp.ones((2, 3))
    value = RuntimeField(array)
    assert value.protected_values()["array"] is array
    assert value.shape == (2,)


@pytest.mark.parametrize("slots", [False, True])
def test_jax_subclass_accepts_inherited_protected_field(slots: bool) -> None:
    """Inherited dataclass fields support indexing without redeclaration."""

    @dataclass(frozen=True, slots=slots)
    class Parent(JaxAxisProtected[jax.Array]):
        array: jax.Array
        protected_axes: ClassVar[dict[str, int]] = {"array": 1}

    @dataclass(frozen=True, slots=slots)
    class Child(Parent):
        pass

    value = Child(jnp.arange(6).reshape(2, 3))
    indexed = value[0]
    assert isinstance(indexed, Child)
    assert jnp.array_equal(indexed.array, value.array[0])


def test_jax_subclass_rejects_non_set_permitted_functions() -> None:
    """``permitted_functions`` must be a set of callables."""
    with pytest.raises(TypeError, match="permitted_functions must be a set"):

        @dataclass(frozen=True, slots=True)
        class _BadFuncs(JaxAxisProtected[Any]):
            array: jax.Array
            protected_axes: ClassVar[dict[str, int]] = {"array": 1}
            permitted_functions: ClassVar[Any] = [jnp.mean]


def test_jax_subclass_rejects_non_callable_in_permitted_functions() -> None:
    """Members of ``permitted_functions`` must be callable."""
    with pytest.raises(TypeError, match="permitted_functions must be a set"):

        @dataclass(frozen=True, slots=True)
        class _NonCallable(JaxAxisProtected[Any]):
            array: jax.Array
            protected_axes: ClassVar[dict[str, int]] = {"array": 1}
            permitted_functions: ClassVar[set[Any]] = {"not-callable"}


@pytest.mark.parametrize("methods", [["set"], {"get"}, {"apply"}, {"unknown"}, {1}, None])
def test_jax_subclass_rejects_invalid_permitted_at_methods(methods):
    with pytest.raises(TypeError, match="permitted_at_methods must be a set of supported"):

        @dataclass(frozen=True)
        class Invalid(JaxAxisProtected[jax.Array]):
            array: jax.Array
            protected_axes: ClassVar[dict[str, int]] = {"array": 1}
            permitted_at_methods: ClassVar[Any] = methods


def test_jax_subclass_copies_indexed_update_permissions():
    methods: set[JaxAtMethod] = {"set", "add"}

    @dataclass(frozen=True)
    class Parent(JaxAxisProtected[jax.Array]):
        array: jax.Array
        protected_axes: ClassVar[dict[str, int]] = {"array": 1}
        permitted_at_methods: ClassVar[set[JaxAtMethod]] = methods

    @dataclass(frozen=True)
    class Child(Parent):
        array: jax.Array

    Child.permitted_at_methods.add("multiply")
    assert Child.permitted_at_methods == {"set", "add", "multiply"}
    assert Parent.permitted_at_methods == methods == {"set", "add"}
    assert Parent.permitted_at_methods is not methods
    assert JaxAxisProtected.permitted_at_methods == {"set"}
