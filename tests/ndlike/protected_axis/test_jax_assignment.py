"""Indexed-update permissions and method-aware assignment preparation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, ClassVar

import numpy as np
import pytest

pytest.importorskip("jax")
import jax
import jax.numpy as jnp

from ndlike import jax_functions as jf
from ndlike.jax_like import JaxLike
from ndlike.protected_axis.jax import JaxAtMethod, JaxAxisProtected


@dataclass(frozen=True)
class _Array(JaxAxisProtected[JaxLike[Any], Any, JaxLike[Any], jax.Array | float]):
    array: jax.Array
    protected_axes: ClassVar[dict[str, int]] = {"array": 1}


@dataclass(frozen=True)
class _TemperaturePair(JaxAxisProtected):
    left: jax.Array
    right: jax.Array
    unit: str = "K"
    protected_axes: ClassVar[dict[str, int]] = {"left": 1, "right": 2}
    permitted_at_methods: ClassVar[set[JaxAtMethod]] = {"set", "add"}

    def _prepare_assignment_value(self, value: object, *, method: JaxAtMethod) -> object:
        if not isinstance(value, _TemperaturePair):
            msg = "Assignment requires temperature metadata."
            raise TypeError(msg)
        offsets = {"K": 0.0, "C": 273.15}
        if value.unit not in offsets:
            msg = "Incompatible units."
            raise ValueError(msg)
        offset = offsets[value.unit] - offsets[self.unit] if method == "set" else 0.0
        return replace(value, left=value.left + offset, right=value.right + offset, unit=self.unit)


@pytest.mark.parametrize("method", ["add", "subtract", "multiply", "divide", "power", "min", "max"])
def test_arithmetic_updates_require_explicit_permission(method):
    target = _Array(jnp.zeros((3, 2)))
    with pytest.raises(TypeError, match=f"Indexed update '{method}' is not permitted for _Array"):
        getattr(target.at[0], method)(1)

    result = target.at[0].set(jnp.ones(2))
    assert isinstance(result, _Array)
    np.testing.assert_array_equal(result.array[0], 1)


def test_permissions_are_independent_of_ordinary_arithmetic():
    @dataclass(frozen=True)
    class Ordinary(_Array):
        array: jax.Array
        permitted_functions: ClassVar[set] = {jf.jax_add}

    @dataclass(frozen=True)
    class Indexed(_Array):
        array: jax.Array
        permitted_at_methods: ClassVar[set[JaxAtMethod]] = {"add"}

    ordinary = Ordinary(jnp.ones((3, 2)))
    shifted = ordinary + 2
    assert isinstance(shifted, Ordinary)
    np.testing.assert_array_equal(shifted.array, 3)
    with pytest.raises(TypeError, match="not permitted"):
        ordinary.at[0].add(2)

    indexed = Indexed(ordinary.array)
    result = indexed.at[0].add(2)
    assert isinstance(result, Indexed)
    np.testing.assert_array_equal(result.array[0], 3)
    with pytest.raises(TypeError):
        _ = indexed + 2


@pytest.mark.parametrize("compiled", [False, True])
def test_denied_updates_skip_preparation_but_allow_reads(compiled):
    @dataclass(frozen=True)
    class ReadOnly(_Array):
        array: jax.Array
        permitted_at_methods: ClassVar[set[JaxAtMethod]] = set()

        def _prepare_assignment_value(self, value: object, *, method: JaxAtMethod) -> object:
            del value, method
            pytest.fail("Denied updates must not invoke preparation.")

    target = ReadOnly(jnp.ones((3, 2)))
    result = target.at[0].get()
    assert isinstance(result, ReadOnly)
    np.testing.assert_array_equal(result.array, 1)

    def update(x):
        return x.at[0].set(jnp.zeros(2))

    operation = jax.jit(update) if compiled else update
    with pytest.raises(TypeError, match="not permitted"):
        operation(target)


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("index", [(2, 1), (2, slice(None))])
@pytest.mark.parametrize("method", ["set", "add"])
def test_update_prepares_original_operand_with_method_context(compiled, index, method):
    target = _TemperaturePair(jnp.zeros((3, 4, 2)), jnp.zeros((3, 4, 2, 2)))
    source = _TemperaturePair(jnp.full(2, 10.0), jnp.full((2, 2), 20.0), unit="C")

    def update(x, y):
        return getattr(x.at[index], method)(y)

    operation = jax.jit(update) if compiled else update

    result = operation(target, source)

    offset = 273.15 if method == "set" else 0
    expected_left = np.zeros(target.left.shape)
    expected_right = np.zeros(target.right.shape)
    expected_left[index] = 10 + offset
    expected_right[index] = 20 + offset
    np.testing.assert_allclose(result.left, expected_left, rtol=1e-6)
    np.testing.assert_allclose(result.right, expected_right, rtol=1e-6)
    assert result.unit == "K"
    assert source.unit == "C"
    np.testing.assert_array_equal(source.left, 10)
    np.testing.assert_array_equal(source.right, 20)
    np.testing.assert_array_equal(target.left, 0)
    np.testing.assert_array_equal(target.right, 0)


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("kind", ["metadata", "array", "tuple"])
def test_preparation_can_reject_incompatible_updates(compiled, kind):
    target = _TemperaturePair(jnp.zeros((3, 2)), jnp.zeros((3, 2, 2)))
    source = _TemperaturePair(jnp.ones(2), jnp.ones((2, 2)), unit="m")
    value = {"metadata": source, "array": source.left, "tuple": (source.left, source.right)}[kind]

    def update(x, y):
        return x.at[0].set(y)

    operation = jax.jit(update) if compiled else update

    with pytest.raises((TypeError, ValueError), match=r"metadata|Incompatible units"):
        operation(target, value)

    np.testing.assert_array_equal(target.left, 0)
    np.testing.assert_array_equal(target.right, 0)


def test_converted_fields_are_shape_checked():
    @dataclass(frozen=True)
    class Broken(_TemperaturePair):
        left: jax.Array
        right: jax.Array

        def _prepare_assignment_value(self, value: object, *, method: JaxAtMethod) -> object:
            del value, method
            return jnp.ones(2), jnp.ones((3, 3))

    target = Broken(jnp.zeros((3, 2)), jnp.zeros((3, 2, 2)))
    with pytest.raises(ValueError, match="modifies protected trailing axes"):
        target.at[0].set(object())  # ty: ignore[invalid-argument-type]


def test_opted_in_updates_preserve_repeated_index_semantics():
    @dataclass(frozen=True)
    class Indexed(_Array):
        array: jax.Array
        permitted_at_methods: ClassVar[set[JaxAtMethod]] = {"add"}

    target = Indexed(jnp.zeros((3, 2)))
    result = target.at[jnp.array([1, 1])].add(2)
    assert isinstance(result, Indexed)
    np.testing.assert_array_equal(result.array, [[0, 0], [4, 4], [0, 0]])
