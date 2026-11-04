"""Inherited protected layouts retain fields, counts, and broadcast semantics."""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
import pytest

pytest.importorskip("jax")
import jax
from jax import numpy as jnp

from ndlike.protected_axis.jax import JaxAxisProtected


@pytest.mark.parametrize("slots", [False, True])
def test_inherited_layout_supports_indexing_reshape_and_jit(slots):
    @dataclass(frozen=True, slots=slots)
    class Parent(JaxAxisProtected[jax.Array]):
        shared: jax.Array
        values: jax.Array
        protected_axes: ClassVar[dict[str, int]] = {"shared": 1, "values": 2}
        broadcasted_batch: ClassVar[set[str]] = {"shared"}

    @dataclass(frozen=True, slots=slots)
    class Child(Parent):
        labels: jax.Array
        weights: jax.Array
        protected_axes: ClassVar[dict[str, int]] = {"labels": 0, "weights": 1}
        broadcasted_batch: ClassVar[set[str]] = {"weights"}

    assert list(Child.protected_axes.items()) == [("shared", 1), ("values", 2), ("labels", 0), ("weights", 1)]
    assert Child.broadcasted_batch == {"shared", "weights"}
    assert Child.primary_protected_name == "values"
    assert Child.pytree_children == ("shared", "values", "labels", "weights")
    x = Child(jnp.arange(3.0), jnp.arange(24.0).reshape(4, 2, 3), jnp.arange(4.0), jnp.ones(2))
    assert x.shape == (4,)
    indexed = x[1]
    assert type(indexed) is Child
    for name in ("shared", "weights"):
        np.testing.assert_array_equal(getattr(indexed, name), getattr(x, name))
    for name in ("values", "labels"):
        np.testing.assert_array_equal(getattr(indexed, name), getattr(x, name)[1])
    reshaped = x.reshape(2, 2)
    assert isinstance(reshaped, Child)
    assert reshaped.values.shape == (2, 2, 2, 3)
    assert reshaped.labels.shape == (2, 2)
    assert reshaped.shared.shape == (3,)
    assert reshaped.weights.shape == (2,)
    assert len(jax.tree.leaves(x)) == 4
    transformed = jax.jit(lambda value: jax.tree.map(lambda field: field + 1, value))(x)
    assert type(transformed) is Child
    for name in Child.protected_axes:
        np.testing.assert_array_equal(getattr(transformed, name), getattr(x, name) + 1)
    assert Parent.protected_axes == {"shared": 1, "values": 2}
    assert Parent.broadcasted_batch == {"shared"}


@pytest.mark.parametrize("declarations", [{}, {"protected_axes": {}, "broadcasted_batch": set()}])
def test_omitted_or_empty_declarations_preserve_independent_layouts(declarations):
    axes = {"values": 1, "shared": 0}
    batch = {"shared"}

    class Parent(JaxAxisProtected[jax.Array]):
        protected_axes: ClassVar[dict[str, int]] = axes
        broadcasted_batch: ClassVar[set[str]] = batch

    child = type("Child", (Parent,), declarations)
    sibling = type("Sibling", (Parent,), {})
    assert child.protected_axes == axes
    assert child.broadcasted_batch == batch
    assert child.protected_axes is not Parent.protected_axes
    assert child.broadcasted_batch is not Parent.broadcasted_batch
    child.protected_axes["extra"] = 2
    child.broadcasted_batch.add("extra")
    assert Parent.protected_axes == sibling.protected_axes == axes
    assert Parent.broadcasted_batch == sibling.broadcasted_batch == batch
    assert Parent.protected_axes is not axes
    assert Parent.broadcasted_batch is not batch


@pytest.mark.parametrize("count", [0, 2])
def test_subclass_cannot_change_inherited_axis_count(count):
    class Parent(JaxAxisProtected[jax.Array]):
        protected_axes: ClassVar[dict[str, int]] = {"values": 1}

    with pytest.raises(TypeError, match=r"Child.protected_axes\['values'\] conflicts.*inherited axis count"):
        type("Child", (Parent,), {"protected_axes": {"values": count}})


def test_matching_redeclarations_keep_base_field_order():
    class Parent(JaxAxisProtected[jax.Array]):
        protected_axes: ClassVar[dict[str, int]] = {"values": 1, "labels": 0}

    class Child(Parent):
        protected_axes: ClassVar[dict[str, int]] = {"new": 2, "labels": 0, "values": 1}

    assert Child.pytree_children == ("values", "labels", "new")


def test_abstract_bases_accumulate_layouts_before_concrete_validation():
    class Base(JaxAxisProtected[jax.Array]):
        @abstractmethod
        def method(self) -> None: ...

    class Shared(Base):
        protected_axes: ClassVar[dict[str, int]] = {"shared": 1}
        broadcasted_batch: ClassVar[set[str]] = {"shared"}

    class MoreShared(Shared):
        protected_axes: ClassVar[dict[str, int]] = {"other": 2}
        broadcasted_batch: ClassVar[set[str]] = {"other"}

    class Concrete(MoreShared):
        protected_axes: ClassVar[dict[str, int]] = {"values": 0}

        def method(self) -> None:
            pass

    assert MoreShared.protected_axes == {"shared": 1, "other": 2}
    assert Concrete.protected_axes == {"shared": 1, "other": 2, "values": 0}
    assert Concrete.broadcasted_batch == {"shared", "other"}
    assert Concrete.primary_protected_name == "values"


def test_multiple_inheritance_merges_diamond_layouts_left_to_right():
    class Root(JaxAxisProtected[jax.Array]):
        protected_axes: ClassVar[dict[str, int]] = {"values": 1}

    class Left(Root):
        protected_axes: ClassVar[dict[str, int]] = {"left": 0}
        broadcasted_batch: ClassVar[set[str]] = {"left"}

    class Right(Root):
        protected_axes: ClassVar[dict[str, int]] = {"right": 2}
        broadcasted_batch: ClassVar[set[str]] = {"right"}

    class Child(Left, Right):
        protected_axes: ClassVar[dict[str, int]] = {"child": 1}

    assert list(Child.protected_axes.items()) == [("values", 1), ("left", 0), ("right", 2), ("child", 1)]
    assert Child.broadcasted_batch == {"left", "right"}


def test_conflicting_base_counts_are_rejected():
    class Left(JaxAxisProtected[jax.Array]):
        protected_axes: ClassVar[dict[str, int]] = {"values": 1}

    class Right(JaxAxisProtected[jax.Array]):
        protected_axes: ClassVar[dict[str, int]] = {"values": 2}

    with pytest.raises(TypeError, match="conflicts with an inherited axis count"):
        type("Child", (Left, Right), {})


@pytest.mark.parametrize(
    "declarations",
    [
        {"protected_axes": None},
        {"protected_axes": {"new": -1}},
        {"broadcasted_batch": []},
        {"broadcasted_batch": {"missing"}},
        {"broadcasted_batch": {"values"}},
    ],
)
def test_local_declarations_are_validated_against_inherited_layout(declarations):
    class Parent(JaxAxisProtected[Any]):
        protected_axes: ClassVar[dict[str, int]] = {"values": 1, "shared": 0}
        broadcasted_batch: ClassVar[set[str]] = {"shared"}

    with pytest.raises(TypeError, match=r"protected_axes|broadcasted_batch"):
        type("Child", (Parent,), declarations)
