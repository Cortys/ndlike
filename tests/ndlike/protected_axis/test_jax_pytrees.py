"""Protected-field schemas and transformation boundaries for JAX values."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
import pytest

pytest.importorskip("jax")
import jax
import jax.numpy as jnp

from ndlike.jax_functions import jax_add, jax_multiply
from ndlike.jax_like import jax_tree_map
from ndlike.protected_axis.jax import JaxAxisProtected


@dataclass(frozen=True)
class _Metadata:
    shape: tuple[int, ...] = (37,)


@dataclass(frozen=True, slots=True, eq=False)
class _Pair(JaxAxisProtected[jax.Array]):
    array: jax.Array
    second: jax.Array
    metadata: Any = _Metadata()
    protected_axes: ClassVar[dict[str, int]] = {"second": 2, "array": 1}
    permitted_functions: ClassVar[set[Any]] = {jax_add, jax_multiply}


@dataclass(frozen=True, slots=True, eq=False)
class _Sidecar(JaxAxisProtected[jax.Array]):
    array: jax.Array
    labels: np.ndarray | jax.Array
    metadata: Any = None
    protected_axes: ClassVar[dict[str, int]] = {"array": 1, "labels": 0}


_TRANSFORMS = [
    pytest.param(jax.jit(lambda value: value), id="jit"),
    pytest.param(jax.grad(lambda value: jnp.sum(value.array)), id="grad"),
    pytest.param(jax.vmap(lambda value: value), id="vmap"),
    pytest.param(jax.vmap(lambda value: value, in_axes=None, axis_size=2), id="unmapped-vmap"),
    pytest.param(lambda value: jax.tree.map(lambda leaf: leaf, value), id="native-tree-map"),
    pytest.param(lambda value: jax_tree_map(lambda leaf: leaf, value, traverse_jaxlikes=True), id="traversing-helper"),
    pytest.param(jax.jit(lambda value: jax_tree_map(lambda leaf: leaf, value)), id="jit-of-opaque-helper"),
]


def _pair():
    return _Pair(jnp.arange(6.0).reshape(2, 3), jnp.arange(8.0).reshape(2, 2, 2))


def test_protected_fields_are_declared_in_layout_order():
    x = _pair()
    assert _Pair.pytree_children == ("second", "array")
    children, _ = x.tree_flatten()
    assert children[0] is x.second
    assert children[1] is x.array
    mapped = jax.tree.map(lambda value: value + 1, x)
    assert mapped.metadata is x.metadata
    np.testing.assert_array_equal(mapped.array, x.array + 1)
    np.testing.assert_array_equal(mapped.second, x.second + 1)


def test_nested_jit_vmap_and_grad_preserve_the_protected_schema():
    x = _pair()
    mapped = jax.jit(jax.vmap(lambda value: value * 2 + 1))(x)
    assert type(mapped) is _Pair
    assert mapped.metadata is x.metadata
    assert mapped.shape == (2,)
    np.testing.assert_array_equal(mapped.array, x.array * 2 + 1)
    np.testing.assert_array_equal(mapped.second, x.second * 2 + 1)

    def loss(value):
        squared = value * value
        return jnp.sum(squared.array) + jnp.sum(squared.second)

    gradient = jax.jit(jax.grad(loss))(x)
    assert type(gradient) is _Pair
    assert gradient.metadata is x.metadata
    np.testing.assert_array_equal(gradient.array, 2 * x.array)
    np.testing.assert_array_equal(gradient.second, 2 * x.second)
    batched_gradient = jax.vmap(jax.grad(loss))(x)
    np.testing.assert_array_equal(batched_gradient.array, gradient.array)
    np.testing.assert_array_equal(batched_gradient.second, gradient.second)


def test_unprotected_arrays_are_not_silently_batched_or_made_static():
    x = _Pair(_pair().array, _pair().second, jnp.ones(37))
    with pytest.raises(TypeError, match=r"_Pair\.metadata must be static, hashable metadata"):
        jax.jit(lambda value: value)(x)
    # Eager representation operations need no valid static pytree metadata.
    reshaped = x.reshape(1, 2)
    assert isinstance(reshaped, _Pair)
    assert reshaped.metadata is x.metadata
    converted = x.astype(jnp.int32)
    assert converted.metadata is x.metadata
    moved = x.to_device(jax.devices()[0])
    assert moved.metadata is x.metadata


def test_array_convenience_methods_use_protected_fields_independently():
    @dataclass(frozen=True, eq=False)
    class _CustomTree(_Pair):
        array: jax.Array
        second: jax.Array

        def tree_flatten(self):
            msg = "convenience methods must not use custom flattening"
            raise AssertionError(msg)

    x = _pair()
    original = _CustomTree(x.array, x.second, np.ones(37))
    detached = original.stop_gradient()
    viewed = original.view(jnp.int32)
    moved = original.to_device(jax.devices()[0])
    for value in (detached, viewed, moved):
        assert type(value) is _CustomTree
        assert value.metadata is original.metadata
    assert original.block_until_ready() is original


def test_numpy_sidecars_remain_supported_by_eager_structural_operations():
    original = _Sidecar(jnp.ones((2, 3)), np.array(["a", "b"], dtype=object))
    reshaped = original.reshape(1, 2)
    assert isinstance(reshaped, _Sidecar)
    np.testing.assert_array_equal(reshaped.labels, [["a", "b"]])
    indexed = original[0]
    assert isinstance(indexed, _Sidecar)
    assert indexed.labels.item() == "a"
    assert original.astype(jnp.float16).labels is original.labels


@pytest.mark.parametrize("transform", _TRANSFORMS)
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, bool, object, str, bytes])
def test_all_numpy_sidecars_are_rejected_before_transformation(transform, dtype):
    sidecar = np.array([0, 1], dtype=dtype)
    original = _Sidecar(jnp.ones((2, 3)), sidecar)
    with pytest.raises(TypeError, match=r"_Sidecar\.labels contains a NumPy array"):
        transform(original)
    assert original.labels is sidecar


@pytest.mark.parametrize("transform", _TRANSFORMS)
def test_numpy_array_metadata_is_also_rejected_by_transformations(transform):
    original = _Pair(_pair().array, _pair().second, np.ones(37))
    with pytest.raises(TypeError, match=r"_Pair\.metadata must be static, hashable metadata"):
        transform(original)


def test_jit_cannot_silently_convert_numpy_sidecars_in_results():
    def mixed_result(array):
        return _Sidecar(array, np.ones(array.shape[0], dtype=np.float32))

    with pytest.raises(TypeError, match=r"_Sidecar\.labels contains a NumPy array"):
        jax.jit(mixed_result)(jnp.ones((2, 3)))


@pytest.mark.parametrize("force_opaque", [False, True])
def test_opaque_eager_mapping_keeps_numpy_sidecars_under_representation_control(force_opaque):
    @dataclass(frozen=True, eq=False)
    class _Configurable(_Sidecar):
        array: jax.Array
        labels: np.ndarray | jax.Array
        tree_map_traverse: ClassVar[bool] = force_opaque

    labels = np.array(["a", "b"], dtype=object)
    metadata = np.ones(37)
    original = _Configurable(jnp.ones((2, 3)), labels, metadata)
    mapped = jax_tree_map(
        lambda value: value.reshape(1, 2),
        {"value": original},
        traverse_jaxlikes=False if force_opaque else None,
    )["value"]
    assert type(mapped) is _Configurable
    assert mapped.metadata is metadata
    assert isinstance(mapped.labels, np.ndarray)
    np.testing.assert_array_equal(mapped.labels, [["a", "b"]])
    assert mapped.shape == (1, 2)
    assert original.labels is labels


def test_leaf_predicate_can_keep_mixed_objects_opaque_when_traversal_is_requested():
    original = _Sidecar(jnp.ones((2, 3)), np.ones(2))
    mapped = jax_tree_map(
        lambda value: value.reshape(1, 2),
        original,
        traverse_jaxlikes=True,
        is_leaf=lambda value: isinstance(value, _Sidecar),
    )
    assert isinstance(mapped, _Sidecar)
    assert isinstance(mapped.labels, np.ndarray)
    assert mapped.shape == (1, 2)


def test_explicit_jax_data_in_the_same_representation_remains_transformable():
    original = _Sidecar(jnp.ones((2, 3)), jnp.arange(2.0))
    mapped = jax.jit(jax.vmap(lambda value: value))(original)
    assert type(mapped) is _Sidecar
    assert isinstance(mapped.labels, jax.Array)
    np.testing.assert_array_equal(mapped.labels, original.labels)
    gradient = jax.jit(jax.grad(lambda value: jnp.sum(value.array) + jnp.sum(value.labels)))(original)
    np.testing.assert_array_equal(gradient.array, np.ones((2, 3)))
    np.testing.assert_array_equal(gradient.labels, np.ones(2))


def test_protected_objects_are_opaque_in_the_mapping_helper():
    x = _pair()
    seen = []

    def visit(value):
        seen.append(value)
        return value + 1

    mapped = jax_tree_map(visit, {"value": x})
    assert len(seen) == 1
    assert seen[0] is x
    np.testing.assert_array_equal(mapped["value"].array, x.array + 1)
