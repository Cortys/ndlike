"""Dynamic tuple members retain structure through JAX transformations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, NamedTuple, Never, assert_type

import numpy as np
import pytest

pytest.importorskip("jax")
import jax
import jax.numpy as jnp

from ndlike import jax_functions as jf
from ndlike.protected_axis.jax import JaxAxisProtected
from ndlike.tuple.jax import JaxBroadcastTuple


def example() -> JaxBroadcastTuple[tuple[jax.Array, ...]]:
    return JaxBroadcastTuple((jnp.arange(2.0).reshape(2, 1), jnp.arange(3.0), jnp.array(7.0)))


def test_named_members_remain_dynamic_through_jit():
    class Pair(NamedTuple):
        first: jax.Array
        second: jax.Array

    x = JaxBroadcastTuple(Pair(jnp.arange(2.0).reshape(2, 1), jnp.arange(3.0)))
    assert_type(x.reshape(6).members, Pair)
    leaves, structure = jax.tree.flatten(x)
    assert len(leaves) == 2
    assert type(jax.tree.unflatten(structure, leaves).members) is Pair
    result = jax.jit(lambda value: value.reshape(6) + 1)(x)
    assert type(result.members) is Pair
    np.testing.assert_array_equal(result.members.first, [1, 1, 1, 2, 2, 2])
    np.testing.assert_array_equal(result.members.second, [1, 2, 3, 1, 2, 3])


def test_tree_round_trip_and_shape_typing():
    x = example()
    assert_type(x.reshape(6), JaxBroadcastTuple[tuple[jax.Array, ...]])
    assert_type(x[0], JaxBroadcastTuple[tuple[jax.Array, ...]])
    leaves, structure = jax.tree.flatten(x)
    assert len(leaves) == 3
    assert all(a is b for a, b in zip(leaves, x.members, strict=True))
    restored = jax.tree.unflatten(structure, leaves)
    assert type(restored) is type(x)
    assert restored.shape == (2, 3)
    assert tuple(member.shape for member in restored.members) == ((2, 1), (3,), ())
    assert jax.tree.map(lambda value: value + 1, x).shape == x.shape


def test_jit_shaping_and_traced_indices():
    x = example()
    y = jax.jit(lambda value: value.reshape(6))(x)
    assert type(y) is JaxBroadcastTuple
    assert y.shape == (6,)
    for actual, member in zip(y.members, x.members, strict=True):
        np.testing.assert_array_equal(actual, jnp.broadcast_to(member, x.shape).reshape(6))
    indexed = jax.jit(lambda value, index: value[index])(x, jnp.array(1))
    assert indexed.shape == (3,)
    assert tuple(member.shape for member in indexed.members) == ((1,), (3,), ())
    assert jax.jit(jf.jax_transpose)(x).shape == (3, 2)


def test_gradients_accumulate_over_broadcasts():
    x = example()
    grad = jax.grad(lambda value: sum(jnp.sum(member) for member in value.reshape(6).members))(x)
    assert type(grad) is JaxBroadcastTuple
    assert tuple(member.shape for member in grad.members) == ((2, 1), (3,), ())
    for actual, expected in zip(
        grad.members, (jnp.full((2, 1), 3.0), jnp.full((3,), 2.0), jnp.array(6.0)), strict=True
    ):
        np.testing.assert_array_equal(actual, expected)


def test_vmap_uses_physical_children_and_preserves_owner():
    x = jf.jax_broadcast_to(example(), (2, 3))
    y = jax.vmap(lambda value: value.reshape(3, 1))(x)
    assert type(y) is JaxBroadcastTuple
    assert y.shape == (2, 3, 1)


def test_transforms_of_callable_protected_members():
    @dataclass(frozen=True)
    class Game(JaxAxisProtected["Game"]):
        array: jax.Array
        protected_axes: ClassVar[dict[str, int]] = {"array": 1}

        def __call__(self) -> Never:
            pytest.fail("Shaping must not evaluate members")

    x = JaxBroadcastTuple((Game(jnp.ones((2, 1, 4))), jnp.ones(3)))
    assert_type(x, JaxBroadcastTuple[tuple[Game, jax.Array]])
    assert_type(x.members, tuple[Game, jax.Array])
    assert_type(x.members[0], Game)
    assert_type(x.members[1], jax.Array)
    assert_type(x.reshape(6).members[0], Game)
    y = jax.jit(lambda value: value.reshape(6))(x)
    assert type(y.members[0]) is Game
    assert y.members[0].array.shape == (6, 4)
    assert len(jax.tree.leaves(x)) == 2
    assert x.at[1].get().shape == (3,)


def test_indexed_updates_under_jit_and_native_get_options():
    x = example()
    assert_type(x.at[0].set(1), JaxBroadcastTuple[tuple[jax.Array, ...]])
    values = JaxBroadcastTuple((jnp.arange(3.0), jnp.arange(3.0) + 1, jnp.arange(3.0) + 2))
    updated = jax.jit(lambda owner, value: owner.at[1].set(value))(x, values)
    assert type(updated) is JaxBroadcastTuple
    for member, original, value in zip(updated.members, x.members, values.members, strict=True):
        np.testing.assert_array_equal(member, jnp.broadcast_to(original, x.shape).at[1].set(value))
    added = jax.jit(lambda owner: owner.at[:, 1].add(2))(x)
    for actual, member in zip(added.members, x.members, strict=True):
        np.testing.assert_array_equal(actual, jnp.broadcast_to(member, x.shape).at[:, 1].add(2))
    assert tuple(member.shape for member in x.members) == ((2, 1), (3,), ())
    clipped = x.at[9].get(mode="clip")
    assert clipped.shape == (3,)
    for actual, member in zip(clipped.members, x.members, strict=True):
        np.testing.assert_array_equal(actual, jnp.broadcast_to(member, x.shape).at[9].get(mode="clip"))
    applied = x.at[:, 1].apply(lambda value: value * 2)
    assert applied.shape == (2, 3)
    with pytest.raises(TypeError, match="immutable"):
        x[0] = values


def test_arithmetic_gradients_and_multiple_results():
    x = example()
    result = jax.jit(lambda value: jf.jax_sum(value * 2, axis=0))(x)
    assert result.shape == (3,)
    quotient, remainder = jax.jit(lambda value: divmod(value, 2))(x)
    assert type(quotient) is type(remainder) is JaxBroadcastTuple
    gradients = jax.grad(lambda value: sum(jnp.sum(member) for member in (value * 2).members))(x)
    for actual, expected in zip(
        gradients.members, (jnp.full((2, 1), 6.0), jnp.full(3, 4.0), jnp.array(12.0)), strict=True
    ):
        np.testing.assert_array_equal(actual, expected)


def test_unregistered_tuple_subclass_is_dynamic_through_jax_transforms():
    class Members(tuple):
        __slots__ = ()

    x = JaxBroadcastTuple(Members((jnp.ones((2, 1)), jnp.ones(3))))
    assert len(jax.tree.leaves(x)) == 2
    transformed = jax.jit(lambda value: value.reshape(6) + 1)
    result = transformed(x)
    assert type(result.members) is Members
    np.testing.assert_array_equal(result.members[0], np.full(6, 2))
    result = transformed(x + 2)
    np.testing.assert_array_equal(result.members[0], np.full(6, 4))
    gradient = jax.grad(lambda value: sum(jnp.sum(v) for v in value.reshape(6).members))(x)
    assert type(gradient.members) is Members
    np.testing.assert_array_equal(gradient.members[0], np.full((2, 1), 3))
    np.testing.assert_array_equal(gradient.members[1], np.full(3, 2))
    mapped = jax.vmap(lambda value: value + 1)(jf.jax_broadcast_to(x, (2, 3)))
    assert type(mapped.members) is Members
    np.testing.assert_array_equal(mapped.members[0], np.full((2, 3), 2))


def test_member_flattening_preserves_additional_owner_fields():
    class Members(tuple):
        __slots__ = ()

    @dataclass(frozen=True)
    class Tagged(JaxBroadcastTuple[tuple[jax.Array, ...]]):
        extra: jax.Array
        label: str = "static metadata"
        pytree_children: ClassVar[tuple[str, ...]] = ("extra", "members")

    x = Tagged(Members((jnp.ones((2, 1)), jnp.ones(3))), jnp.array(7.0))
    leaves, structure = jax.tree.flatten(x)
    assert len(leaves) == 3
    assert leaves[0] is x.extra
    restored = jax.tree.unflatten(structure, leaves)
    assert type(restored) is Tagged
    assert type(restored.members) is Members
    assert restored.label == x.label
    result = jax.jit(lambda value: value.reshape(6))(x)
    assert type(result) is Tagged
    assert type(result.members) is Members
    assert result.label == x.label
    np.testing.assert_array_equal(result.extra, 7)


def test_empty_tuple_transforms():
    class Empty(NamedTuple):
        pass

    empty = JaxBroadcastTuple(Empty())
    assert jax.tree.leaves(empty) == []
    leaves, layout = jax.tree.flatten(empty)
    restored = jax.tree.unflatten(layout, leaves)
    assert type(restored.members) is Empty
    shaped = jax.jit(lambda value, index: jf.jax_broadcast_to(value, (2, 3))[index].reshape(6))(
        empty, jnp.array([7, 8])
    )
    assert type(shaped.members) is Empty
    assert shaped.shape == ()
    gradient = jax.grad(lambda _value: jnp.array(1.0))(empty)
    assert type(gradient.members) is Empty
    mapped = jax.vmap(lambda value, data: (value[data], data + 1), in_axes=(None, 0))(empty, jnp.arange(3))
    assert type(mapped[0].members) is Empty
    assert mapped[0].shape == ()
    np.testing.assert_array_equal(mapped[1], [1, 2, 3])
    quotient, remainder = jax.jit(lambda value: divmod(value, 3))(empty)
    assert quotient.members == remainder.members == ()

    @dataclass(frozen=True)
    class Composite(JaxAxisProtected["Composite"]):
        data: jax.Array
        optional: JaxBroadcastTuple[Empty]
        protected_axes: ClassVar[dict[str, int]] = {"data": 0, "optional": 0}
        broadcasted_batch: ClassVar[set[str]] = {"optional"}

    result = jax.jit(lambda value, index: value[index].reshape(2))(Composite(jnp.arange(4), empty), jnp.array([3, 1]))
    np.testing.assert_array_equal(result.data, [3, 1])
    assert type(result.optional.members) is Empty
    assert result.shape == (2,)
