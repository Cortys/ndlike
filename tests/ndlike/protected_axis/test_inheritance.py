"""Protected-axis inheritance behaves consistently across backends."""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from importlib import import_module
from typing import Any, ClassVar

import numpy as np
import pytest


@pytest.fixture(params=["numpy", "torch", "jax"])
def backend(request):
    name = request.param
    pytest.importorskip(name)
    arrays = import_module("jax.numpy" if name == "jax" else name)
    module = import_module(f"ndlike.protected_axis.{name}")
    return getattr(module, f"{name.title()}AxisProtected"), arrays


@pytest.mark.parametrize("slots", [False, True])
def test_subclass_additions_preserve_indexing_and_reshape(backend, slots):
    base, arrays = backend

    @dataclass(frozen=True, slots=slots)
    class Parent(base):
        values: Any
        protected_axes: ClassVar[dict[str, int]] = {"values": 2}

    @dataclass(frozen=True, slots=slots)
    class Child(Parent):
        labels: Any
        protected_axes: ClassVar[dict[str, int]] = {"labels": 0}

    @dataclass(frozen=True, slots=slots)
    class Grandchild(Child):
        vectors: Any
        protected_axes: ClassVar[dict[str, int]] = {"vectors": 1}

    x = Grandchild(arrays.arange(24).reshape(4, 2, 3), arrays.arange(4), arrays.arange(8).reshape(4, 2))
    assert list(Grandchild.protected_axes.items()) == [("values", 2), ("labels", 0), ("vectors", 1)]
    assert x.shape == (4,)
    assert x.primary_protected_name == "values"
    indexed = x[1]
    assert type(indexed) is Grandchild
    for name in Grandchild.protected_axes:
        np.testing.assert_array_equal(getattr(indexed, name), getattr(x, name)[1])
    reshaped = x.reshape(2, 2)
    assert type(reshaped) is Grandchild
    assert reshaped.values.shape == (2, 2, 2, 3)
    assert reshaped.labels.shape == (2, 2)
    assert reshaped.vectors.shape == (2, 2, 2)
    assert Parent.protected_axes == {"values": 2}
    assert Child.protected_axes == {"values": 2, "labels": 0}


@pytest.mark.parametrize("declarations", [{}, {"protected_axes": {}}])
def test_inherited_layouts_are_independent_and_cannot_be_cleared(backend, declarations):
    base, _ = backend
    axes = {"values": 1}

    class Parent(base):
        protected_axes: ClassVar[dict[str, int]] = axes

    child = type("Child", (Parent,), declarations)
    sibling = type("Sibling", (Parent,), {})
    assert child.protected_axes == axes
    assert child.protected_axes is not Parent.protected_axes
    assert Parent.protected_axes is not axes
    child.protected_axes["extra"] = 0
    assert Parent.protected_axes == sibling.protected_axes == axes


@pytest.mark.parametrize("count", [0, 2])
def test_conflicting_subclass_counts_are_rejected(backend, count):
    base, _ = backend

    class Parent(base):
        protected_axes: ClassVar[dict[str, int]] = {"values": 1}

    with pytest.raises(TypeError, match=r"Child.protected_axes\['values'\] conflicts.*inherited axis count"):
        type("Child", (Parent,), {"protected_axes": {"values": count}})


def test_matching_redeclarations_preserve_base_order(backend):
    base, _ = backend

    class Parent(base):
        protected_axes: ClassVar[dict[str, int]] = {"values": 1, "labels": 0}

    class Child(Parent):
        protected_axes: ClassVar[dict[str, int]] = {"new": 2, "labels": 0, "values": 1}

    assert list(Child.protected_axes.items()) == [("values", 1), ("labels", 0), ("new", 2)]


def test_abstract_diamond_inheritance_merges_all_layouts(backend):
    base, _ = backend

    class Abstract(base):
        @abstractmethod
        def method(self) -> None: ...

    class Root(Abstract):
        protected_axes: ClassVar[dict[str, int]] = {"values": 1}

    class Left(Root):
        protected_axes: ClassVar[dict[str, int]] = {"left": 0}

    class Right(Root):
        protected_axes: ClassVar[dict[str, int]] = {"right": 2}

    class Concrete(Left, Right):
        protected_axes: ClassVar[dict[str, int]] = {"child": 1}

        def method(self) -> None:
            pass

    assert Left.protected_axes == {"values": 1, "left": 0}
    assert list(Concrete.protected_axes.items()) == [("values", 1), ("left", 0), ("right", 2), ("child", 1)]
    assert Concrete.primary_protected_name == "values"


def test_conflicting_base_counts_are_rejected(backend):
    base, _ = backend

    class Left(base):
        protected_axes: ClassVar[dict[str, int]] = {"values": 1}

    class Right(base):
        protected_axes: ClassVar[dict[str, int]] = {"values": 2}

    with pytest.raises(TypeError, match="conflicts with an inherited axis count"):
        type("Child", (Left, Right), {})


@pytest.mark.parametrize("axes", [None, [], {"new": -1}, {"new": 1.5}, {"": 0}])
def test_invalid_local_declarations_are_rejected_with_valid_bases(backend, axes):
    base, _ = backend

    class Parent(base):
        protected_axes: ClassVar[dict[str, int]] = {"values": 1}

    with pytest.raises(TypeError, match="protected_axes"):
        type("Child", (Parent,), {"protected_axes": axes})
