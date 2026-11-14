"""Value and result typing for native and custom JAX indexed updates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, assert_type

import numpy as np
import pytest

pytest.importorskip("jax")
import jax
import jax.numpy as jnp

from ndlike.jax_like import JaxIndexUpdateHelper, JaxIndexUpdateRef, JaxLike, JaxLikeImplementation
from ndlike.protected_axis.jax import JaxAtMethod, JaxAxisProtected

if TYPE_CHECKING:
    type Result = JaxLike[Any]

    class _CustomValue:
        """An extra assignment type without NumPy conversion support."""

    def _check_native_compatibility(
        array: jax.Array,
    ) -> tuple[
        JaxLike[jax.Array],
        JaxIndexUpdateHelper[jax.Array | float, jax.Array],
        JaxIndexUpdateRef[jax.Array | float, jax.Array],
    ]:
        return array, array.at, array.at[0]

    def _check_update_values(array: JaxLike[Result, Any, Result, _CustomValue], value: _CustomValue) -> None:
        assert_type(array.at, JaxIndexUpdateHelper[JaxLike[Result, Any, Result, _CustomValue] | _CustomValue, Result])
        selected = assert_type(
            array.at[0], JaxIndexUpdateRef[JaxLike[Result, Any, Result, _CustomValue] | _CustomValue, Result]
        )
        assert_type(selected.get(), Result)
        for update in (
            selected.set,
            selected.add,
            selected.subtract,
            selected.multiply,
            selected.divide,
            selected.power,
            selected.min,
            selected.max,
        ):
            assert_type(update(array), Result)
            assert_type(update(value, indices_are_sorted=True, unique_indices=True, mode="drop"), Result)
            update(1.0)  # ty: ignore[invalid-argument-type]
        selected.set(value, unknown_option=True)  # ty: ignore[unknown-argument]

    def _check_default_updates(array: JaxLike) -> None:
        assert_type(array.at[0].set(array), Result)
        array.at[0].set(1.0)  # ty: ignore[invalid-argument-type]
        array.at[0].add(1.0)  # ty: ignore[invalid-argument-type]

    def _check_implementation_updates(
        array: JaxLikeImplementation[Result, Any, Result, _CustomValue], value: _CustomValue
    ) -> None:
        assert_type(array.at[0].set(array), Result)
        assert_type(array.at[0].set(value), Result)
        array.at[0].set(1.0)  # ty: ignore[invalid-argument-type]

    def _check_update_variance(
        helper: JaxIndexUpdateHelper[object, jax.Array], ref: JaxIndexUpdateRef[object, jax.Array]
    ) -> tuple[JaxIndexUpdateHelper[_CustomValue, Result], JaxIndexUpdateRef[_CustomValue, Result]]:
        return helper, ref


@dataclass(frozen=True)
class _Vector(JaxAxisProtected["_Vector", "_Vector", "_Vector", float]):
    array: jax.Array
    protected_axes: ClassVar[dict[str, int]] = {"array": 1}
    permitted_at_methods: ClassVar[set[JaxAtMethod]] = {"set", "add"}


def test_native_helper_supports_typed_updates() -> None:
    data = jnp.arange(3.0)
    helper: JaxIndexUpdateHelper[jax.Array | float, jax.Array] = data.at

    result = assert_type(helper[1].set(9.0, unique_indices=True), jax.Array)
    np.testing.assert_array_equal(result, [0, 9, 2])
    np.testing.assert_array_equal(data, [0, 1, 2])
    np.testing.assert_array_equal(helper[1].get(), 1)


def test_protected_helper_supports_typed_updates() -> None:
    data = _Vector(jnp.zeros((3, 2)))
    value = _Vector(jnp.ones(2))
    helper: JaxIndexUpdateHelper[_Vector, _Vector] = data.at

    result = assert_type(helper[1].set(value, indices_are_sorted=True), _Vector)
    assert isinstance(result, _Vector)
    np.testing.assert_array_equal(result.array, [[0, 0], [1, 1], [0, 0]])
    np.testing.assert_array_equal(data.array, np.zeros((3, 2)))


@pytest.mark.parametrize("native", [False, True])
def test_jax_like_protocol_supports_typed_scalar_updates(native: bool) -> None:
    def update(array: JaxLike[Any, Any, Any, float, Any, Any, JaxLike[Any]]) -> JaxLike[Any]:
        return assert_type(array.at[0].add(2.0), JaxLike[Any])

    data = jnp.zeros((3, 2))
    result = update(data if native else _Vector(data))
    np.testing.assert_array_equal(np.asarray(result), [[2, 2], [0, 0], [0, 0]])
    np.testing.assert_array_equal(data, np.zeros((3, 2)))
