"""Dataclass equality inheritance and standard hashing across array backends."""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from importlib import import_module
import operator
from typing import Any, ClassVar

import numpy as np
import pytest


@pytest.fixture(params=["numpy", "jax", "torch"])
def backend(request):
    name = request.param
    namespace = pytest.importorskip("jax.numpy" if name == "jax" else name)
    protected_module = import_module(f"ndlike.protected_axis.{name}")
    base = getattr(protected_module, f"{name.title()}AxisProtected")

    class _Array(base):
        array: Any
        protected_axes: ClassVar[dict[str, int]] = {"array": 0}

        def __array_ufunc__(self, ufunc, method, *inputs: object, **kwargs: object) -> object:
            raw = tuple(arg.array if isinstance(arg, _Array) else arg for arg in inputs)
            return getattr(ufunc, method)(*raw, **kwargs)

        @classmethod
        def __jax_function__(cls, func, types, args=(), kwargs=None) -> object:
            del cls, types
            raw = tuple(arg.array if isinstance(arg, _Array) else arg for arg in args)
            return func(*raw, **(kwargs or {}))

        __torch_function__ = __jax_function__

    return name, _Array, namespace.asarray


@pytest.mark.parametrize("frozen", [False, True])
@pytest.mark.parametrize("slots", [False, True])
def test_default_dataclass_preserves_array_equality(backend, frozen, slots):
    _, base, asarray = backend

    @dataclass(frozen=frozen, slots=slots)
    class _Dataclass(base):
        array: Any
        label: str = "metadata"

    left = _Dataclass(asarray([1, 2]), "left")
    right = _Dataclass(asarray([1, 9]), "right")
    assert _Dataclass.__dict__["__eq__"] is base.__eq__
    np.testing.assert_array_equal(left == right, [True, False])
    np.testing.assert_array_equal(left != right, [False, True])
    np.testing.assert_array_equal(operator.eq(left, left), [True, True])


@pytest.mark.parametrize("frozen", [False, True])
@pytest.mark.parametrize("slots", [False, True])
def test_explicit_equality_is_not_overwritten(backend, frozen, slots):
    _, base, asarray = backend

    @dataclass(frozen=frozen, slots=slots)
    class _Explicit(base):  # noqa: PLW1641
        array: Any

        def __eq__(self, other: object) -> bool:
            del other
            return False

    value = _Explicit(asarray([1, 2]))
    assert operator.eq(value, value) is False
    assert _Explicit.__eq__ is not base.__eq__


@pytest.mark.parametrize("frozen", [False, True])
@pytest.mark.parametrize("slots", [False, True])
def test_subclasses_preserve_custom_parent_equality(backend, frozen, slots):
    _, base, asarray = backend

    class _Parent(base):  # noqa: PLW1641
        array: Any

        def __eq__(self, other: object) -> bool:
            return self is other

    @dataclass(frozen=frozen, slots=slots)
    class _Child(_Parent):
        array: Any

    value = _Child(asarray([1, 2]))
    assert _Child.__dict__["__eq__"] is _Parent.__eq__
    assert operator.eq(value, value) is True
    assert (value == _Child(value.array)) is False


@pytest.mark.parametrize("frozen", [False, True])
@pytest.mark.parametrize("slots", [False, True])
def test_hashing_follows_standard_dataclass_rules(backend, frozen, slots):
    name, base, asarray = backend

    @dataclass(frozen=frozen, slots=slots)
    class _Dataclass(base):
        array: Any
        label: str = "metadata"

    value = _Dataclass(asarray([1, 2]))
    assert isinstance(value, Hashable) is frozen
    if frozen and name == "torch":
        assert hash(value) == hash((value.array, value.label))
    else:
        with pytest.raises(TypeError):
            hash(value)


@pytest.mark.parametrize("frozen", [False, True])
@pytest.mark.parametrize("slots", [False, True])
def test_explicit_hash_is_not_overwritten(backend, frozen, slots):
    _, base, asarray = backend

    @dataclass(frozen=frozen, slots=slots)
    class _ExplicitHash(base):
        array: Any

        def __hash__(self) -> int:
            return 23

    assert hash(_ExplicitHash(asarray([1, 2]))) == 23


@pytest.mark.parametrize("frozen", [False, True])
@pytest.mark.parametrize("slots", [False, True])
def test_eq_false_still_preserves_inherited_hashing(backend, frozen, slots):
    _, base, asarray = backend

    @dataclass(eq=False, frozen=frozen, slots=slots)
    class _Dataclass(base):
        array: Any

    assert _Dataclass.__eq__ is base.__eq__
    assert _Dataclass.__hash__ is None
    with pytest.raises(TypeError):
        hash(_Dataclass(asarray([1, 2])))
