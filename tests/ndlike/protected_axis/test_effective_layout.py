"""Dataclass initialization owns excluded fields across all protected backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
from typing import Any, ClassVar
from unittest.mock import Mock

import numpy as np
import pytest

from ndlike.protected_axis import _common as common

# Tests exercise the cached internal layout and extraction contract directly.
# ruff: noqa: SLF001


@pytest.fixture(params=["numpy", "torch", "jax"])
def backend(request):
    name = request.param
    pytest.importorskip(name)
    arrays = import_module("jax.numpy" if name == "jax" else name)
    module = import_module(f"ndlike.protected_axis.{name}")
    functions = import_module("ndlike.jax_functions") if name == "jax" else arrays
    return name, getattr(module, f"{name.title()}AxisProtected"), arrays, functions


def _function(backend, name):
    kind, _, _, functions = backend
    return getattr(functions, f"jax_{name}" if kind == "jax" else name)


@pytest.mark.parametrize("slots", [False, True])
def test_layout_is_resolved_once_after_decoration_and_per_subclass(backend, slots, monkeypatch):
    _, base, arrays, _ = backend
    inspect_fields = Mock(wraps=common.fields)
    monkeypatch.setattr(common, "fields", inspect_fields)

    @dataclass(frozen=True, slots=slots)
    class Parent(base):
        values: Any
        derived: Any
        protected_axes: ClassVar[dict[str, int]] = {"values": 1, "derived": 1}

    assert isinstance(vars(Parent)["_effective_protected_axes"], common.EffectiveProtectedAxes)
    assert inspect_fields.call_count == 0
    assert Parent._effective_protected_axes == Parent.protected_axes
    assert inspect_fields.call_count == 1

    @dataclass(frozen=True, slots=slots)
    class Child(Parent):
        derived: Any = field(init=False, default_factory=lambda: arrays.ones(3))

    @dataclass(frozen=True, slots=slots)
    class Sibling(Parent):
        pass

    assert isinstance(vars(Child)["_effective_protected_axes"], common.EffectiveProtectedAxes)
    assert inspect_fields.call_count == 1
    x = Child(arrays.ones((2, 3)))
    layout = Child._effective_protected_axes
    assert layout == {"values": 1}
    assert inspect_fields.call_count == 2
    assert vars(Child)["_effective_protected_axes"] is layout
    assert x._effective_protected_axes is layout
    copied = x.with_protected_values(x.protected_values())
    assert type(copied) is Child
    assert copied is not x
    assert copied.values is x.values
    np.testing.assert_array_equal(copied.derived, x.derived)
    assert x[0].shape == ()
    assert x.reshape(1, 2).shape == (1, 2)
    assert inspect_fields.call_count == 2
    assert Parent._effective_protected_axes == Sibling._effective_protected_axes == Parent.protected_axes
    assert Child.protected_axes == Parent.protected_axes
    assert Sibling._effective_protected_axes is not Parent._effective_protected_axes

    @dataclass(frozen=True, slots=slots)
    class Restored(Child):
        derived: Any

    assert Restored._effective_protected_axes == Parent.protected_axes
    restored = Restored(x.values, arrays.ones((2, 3)))
    assert restored[0].derived.shape == (3,)


@pytest.mark.parametrize("slots", [False, True])
def test_operations_only_receive_effective_fields_and_recompute_derived_values(backend, slots):
    name, base, arrays, _ = backend
    negative = _function(backend, "negative" if name != "torch" else "clone")
    stack = _function(backend, "stack")
    contexts = []

    @dataclass(frozen=True, slots=slots)
    class Parent(base):
        derived: Any
        values: Any
        protected_axes: ClassVar[dict[str, int]] = {"derived": 1, "values": 1}
        permitted_functions: ClassVar[set] = {negative}
        permitted_ufuncs: ClassVar[dict] = {np.negative: ["__call__"]}

    @dataclass(frozen=True, slots=slots)
    class Child(Parent):
        derived: Any = field(init=False)

        def __post_init__(self) -> None:
            object.__setattr__(self, "derived", self.values * self.values)

        def with_protected_values(self, values, func=None):
            assert set(values) == {"values"}
            contexts.append(func)
            return base.with_protected_values(self, values, func)

    x = Child(arrays.arange(24).reshape(2, 3, 4))
    assert x.primary_protected_name == "derived"
    assert x.protected_value() is x.derived
    assert list(x.protected_values()) == ["values"]
    assert x.shape == (2, 3)
    internals = common.extract_axis_protected_internals(x)
    assert internals is not None
    assert internals.protected_axes == {"values": 1}
    assert list(internals.values) == ["values"]
    assert internals.batch_shape == (2, 3)

    for result, expected in (
        (x.with_protected_values(x.protected_values()), x.values),
        (x[1], x.values[1]),
        (x.reshape(6), x.values.reshape(6, 4)),
        (negative(x), -x.values if name != "torch" else x.values),
        (stack((x, x)), arrays.stack((x.values, x.values))),
    ):
        assert type(result) is Child
        np.testing.assert_array_equal(result.values, expected)
        np.testing.assert_array_equal(result.derived, expected * expected)
    assert contexts

    converted = x.to(dtype=arrays.float32) if name == "torch" else x.astype(arrays.float32)
    assert converted.values.dtype == converted.derived.dtype == arrays.float32
    if name == "jax":
        for result in (x.stop_gradient(), x.to_device(x.values.device), x.view(arrays.float32)):
            np.testing.assert_array_equal(result.derived, result.values * result.values)


def test_default_operations_do_not_even_read_excluded_fields(backend):
    _, base, arrays, _ = backend
    excluded = arrays.ones(4)
    reads = []

    @dataclass
    class Fixed(base):
        values: Any
        fixed: Any = field(init=False, default_factory=lambda: excluded)
        protected_axes: ClassVar[dict[str, int]] = {"values": 1, "fixed": 1}

        def __getattribute__(self, name) -> Any:  # noqa: ANN401
            if name == "fixed":
                reads.append(name)
            return object.__getattribute__(self, name)

    x = Fixed(arrays.ones((2, 3, 4)))
    result = x.reshape(6)[0]
    assert result.shape == ()
    assert reads == []
    assert list(result.protected_values()) == ["values"]
    assert reads == []
    assert result.fixed is excluded
    assert reads == ["fixed"]


@pytest.mark.parametrize("backend", ["torch", "jax"], indirect=True)
@pytest.mark.parametrize("native_effective", [False, True])
@pytest.mark.parametrize("fixed_first", [False, True])
def test_backend_metadata_uses_first_declared_backend_field(backend, native_effective, fixed_first):
    name, base, arrays, _ = backend
    fixed = arrays.ones((2, 3), dtype=arrays.int32)

    @dataclass
    class WithFixed(base):
        values: Any
        fixed: Any = field(init=False, default_factory=lambda: fixed)
        protected_axes: ClassVar[dict[str, int]] = (
            {"fixed": 1, "values": 0} if fixed_first else {"values": 0, "fixed": 1}
        )

    values = arrays.ones(2, dtype=arrays.float32) if native_effective else np.ones(2, dtype=np.float32)
    x = WithFixed(values)
    x.validate()
    assert list(x.protected_values()) == ["values"]
    expected: Any = values if native_effective and not fixed_first else fixed
    assert x.dtype == expected.dtype
    assert x.device == expected.device
    if name == "jax":
        assert x.sharding == expected.sharding
        assert x.devices() == expected.devices()


def test_non_dataclasses_keep_all_fields_and_custom_reconstruction(backend):
    _, base, arrays, _ = backend

    class Plain(base):
        protected_axes: ClassVar[dict[str, int]] = {"values": 1}

        def __init__(self, values) -> None:
            self.values = values

        def with_protected_values(self, values, func=None):
            del func
            return type(self)(values["values"])

    x = Plain(arrays.ones((2, 3)))
    assert Plain._effective_protected_axes == Plain.protected_axes
    copied = x.with_protected_values(x.protected_values())
    assert type(copied) is Plain
    assert copied is not x
    assert copied.values is x.values
    assert x[0].shape == ()
    assert x.reshape(1, 2).shape == (1, 2)


def test_multiple_inheritance_uses_final_dataclass_field_precedence(backend):
    _, base, arrays, _ = backend

    @dataclass
    class Parent(base):
        values: Any
        other: Any = None
        protected_axes: ClassVar[dict[str, int]] = {"values": 1, "other": 1}

    @dataclass
    class Left(Parent):
        other: Any = field(init=False, default_factory=lambda: arrays.ones(3))

    @dataclass
    class Right(Parent):
        other: Any = None

    @dataclass
    class LeftFirst(Left, Right):
        pass

    @dataclass
    class RightFirst(Right, Left):
        pass

    assert LeftFirst._effective_protected_axes == {"values": 1}
    assert RightFirst._effective_protected_axes == Parent.protected_axes


def test_assignment_does_not_write_class_owned_fields(backend):
    name, base, arrays, _ = backend
    fixed = arrays.ones((2, 3))

    @dataclass
    class Fixed(base):
        values: Any
        fixed: Any = field(init=False, default_factory=lambda: fixed)
        protected_axes: ClassVar[dict[str, int]] = {"values": 1, "fixed": 1}

    x = Fixed(arrays.zeros((2, 3)))
    replacement = arrays.full((3,), 4.0)
    if name == "jax":
        result = x.at[0].set(replacement)
    else:
        x[0] = replacement
        result = x
    np.testing.assert_array_equal(result.values[0], replacement)
    np.testing.assert_array_equal(result.fixed, np.ones((2, 3)))


def test_empty_effective_layout_keeps_public_primary_and_reconstructs(backend):
    name, base, arrays, _ = backend
    negative = _function(backend, "negative" if name != "torch" else "clone")
    fixed = arrays.ones(3)

    @dataclass
    class Fixed(base):
        fixed: Any = field(init=False, default_factory=lambda: fixed)
        protected_axes: ClassVar[dict[str, int]] = {"fixed": 1}
        broadcasted_batch: ClassVar[set[str]] = {"fixed"}
        permitted_functions: ClassVar[set] = {negative}
        permitted_ufuncs: ClassVar[dict] = {np.negative: ["__call__"]}

    x = Fixed()
    assert Fixed._effective_protected_axes == {}
    assert x.primary_protected_name == "fixed"
    assert x.protected_value() is fixed
    assert x.protected_values() == {}
    copied = x.with_protected_values(x.protected_values())
    assert type(copied) is Fixed
    assert copied is not x
    assert copied.fixed is fixed
    assert x.shape == ()
    internals = common.extract_axis_protected_internals(x)
    assert internals is not None
    assert internals.values == internals.protected_axes == {}
    assert internals.broadcasted_batch == frozenset()
    assert internals.batch_shape == ()
    result = negative(x)
    assert type(result) is Fixed
    assert result.fixed is fixed
    with pytest.raises((TypeError, ValueError), match="init=False"):
        x.with_protected_values({"fixed": arrays.zeros(3)})
    with pytest.raises(ValueError, match="at least one effective protected field"):
        x.reshape(1)
    with pytest.raises(ValueError, match="at least one effective protected field"):
        _function(backend, "stack")((x, x))


def test_jax_effective_broadcast_subset_does_not_include_excluded_fields():
    pytest.importorskip("jax")
    jnp = import_module("jax.numpy")
    base = import_module("ndlike.protected_axis.jax").JaxAxisProtected
    jax_axis_protected_internals = import_module("ndlike.protected_axis.jax_functions").jax_axis_protected_internals

    @dataclass
    class Fixed(base):
        fixed: Any = field(init=False, default_factory=lambda: jnp.ones(4))
        values: Any
        protected_axes: ClassVar[dict[str, int]] = {"fixed": 1, "values": 1}
        broadcasted_batch: ClassVar[set[str]] = {"fixed", "values"}

    x = Fixed(jnp.ones((2, 3, 4)))
    internals = jax_axis_protected_internals(x)
    assert internals is not None
    assert internals.broadcasted_batch == frozenset({"values"})
    result = x.reshape(6)
    assert isinstance(result, Fixed)
    assert result.values.shape == (6, 4)
    assert result.fixed.shape == (4,)


def test_public_reader_overrides_keep_permission_vetoes(backend):
    _, base, arrays, _ = backend
    function = _function(backend, "sum")

    @dataclass
    class Veto(base):
        values: Any
        fixed: Any = field(init=False, default_factory=lambda: arrays.ones(3))
        protected_axes: ClassVar[dict[str, int]] = {"values": 1, "fixed": 1}
        permitted_functions: ClassVar[set] = {function}

        def protected_values(self, func=None):
            return base.protected_values(self) if func is None else None

    x = Veto(arrays.ones((2, 3)))
    assert common.extract_axis_protected_internals(x, function, check_is_permitted=True) is None
    assert x[0].shape == ()


def test_numpy_ufunc_out_only_updates_effective_fields():
    base = import_module("ndlike.protected_axis.numpy").NumpyAxisProtected
    fixed = np.ones(3)

    @dataclass
    class Fixed(base):
        values: Any
        fixed: Any = field(init=False, default_factory=lambda: fixed)
        protected_axes: ClassVar[dict[str, int]] = {"values": 1, "fixed": 1}
        permitted_ufuncs: ClassVar[dict] = {np.add: ["__call__"]}

    x = Fixed(np.ones((2, 3)))
    out = Fixed(np.zeros((2, 3)))
    assert np.add(x, 2, out=out) is out  # ty:ignore[no-matching-overload]
    np.testing.assert_array_equal(out.values, np.full((2, 3), 3))
    assert out.fixed is fixed
    np.testing.assert_array_equal(fixed, np.ones(3))


def test_jax_comparisons_with_empty_effective_layout():
    pytest.importorskip("jax")
    jnp = import_module("jax.numpy")
    base = import_module("ndlike.protected_axis.jax").JaxAxisProtected
    functions = import_module("ndlike.jax_functions")

    @dataclass(eq=False)
    class Fixed(base):
        fixed: Any = field(init=False, default_factory=lambda: jnp.ones(3))
        protected_axes: ClassVar[dict[str, int]] = {"fixed": 1}
        broadcasted_batch: ClassVar[set[str]] = {"fixed"}
        permitted_functions: ClassVar[set] = {functions.jax_equal, functions.jax_not_equal}

    result = functions.jax_equal(Fixed(), Fixed())
    assert result.shape == ()
    assert bool(result)
    assert not bool(functions.jax_not_equal(Fixed(), Fixed()))


@pytest.mark.parametrize("count", [0, 1])
def test_shape_reads_only_the_shape_carrying_field(backend, count):
    _, base, _, _ = backend
    reads = []

    class Value:
        def __init__(self, name) -> None:
            self.name = name

        @property
        def shape(self):
            reads.append(self.name)
            if self.name != "first":
                pytest.fail("Reading shape must not inspect the other fields.")
            return (2, 3)

        @property
        def ndim(self):
            pytest.fail("Reading values or shape must not validate field ranks.")

    @dataclass
    class Pair(base):
        first: Any
        second: Any
        protected_axes: ClassVar[dict[str, int]] = {"first": count, "second": count}

    x = Pair(Value("first"), Value("second"))
    expected = (2, 3) if count == 0 else (2,)
    assert x.protected_values() == {"first": x.first, "second": x.second}
    assert reads == []
    assert x.shape == expected
    assert reads == ["first"]
    reads.clear()
    internals = common.extract_axis_protected_internals(x)
    assert internals is not None
    assert internals.batch_shape == expected
    assert reads == ["first"]


def test_extraction_honors_value_and_batch_shape_hooks(backend):
    _, base, arrays, _ = backend
    function = _function(backend, "sum")
    calls = []
    exposed = arrays.ones((7, 3))

    @dataclass
    class Custom(base):
        value: Any
        protected_axes: ClassVar[dict[str, int]] = {"value": 1}

        def protected_values(self, func=None):
            calls.append(func)
            return None if func is not None else {"value": exposed}

        def _protected_batch_shape(self, values=None):
            del values
            return (7,)

    x = Custom(arrays.ones((2, 3)))
    internals = common.extract_axis_protected_internals(x)
    assert internals is not None
    assert internals.values["value"] is exposed
    assert internals.batch_shape == (7,)
    assert calls == [None]
    assert common.extract_axis_protected_internals(x, function, check_is_permitted=True) is None
    assert calls == [None, function]


def test_shape_fast_path_does_not_collect_values(backend):
    _, base, arrays, _ = backend

    @dataclass
    class Value(base):
        value: Any
        protected_axes: ClassVar[dict[str, int]] = {"value": 1}

        def protected_values(self, func=None):
            del func
            pytest.fail("Shape-only access must not collect protected values.")

    assert Value(arrays.ones((2, 3))).shape == (2,)


def test_validate_checks_effective_fields_explicitly(backend):
    _, base, arrays, _ = backend

    @dataclass
    class Pair(base):
        values: Any
        weights: Any
        derived: Any = field(init=False, default=None)
        protected_axes: ClassVar[dict[str, int]] = {"values": 1, "weights": 0, "derived": 2}

    x = Pair(arrays.ones((2, 3)), arrays.ones(2))
    assert x.validate() is None
    x.weights = arrays.ones(5)
    assert x.shape == (2,)
    assert x.protected_values()["weights"] is x.weights
    with pytest.raises(ValueError, match="inconsistent batch-shapes"):
        x.validate()
    x.values = arrays.ones(())
    with pytest.raises(ValueError, match="removed protected trailing axes"):
        x.validate()


def test_validate_can_run_during_dataclass_initialization(backend):
    _, base, arrays, _ = backend

    @dataclass
    class Checked(base):
        values: Any
        weights: Any
        protected_axes: ClassVar[dict[str, int]] = {"values": 1, "weights": 0}

        def __post_init__(self) -> None:
            self.validate()

    x = Checked(arrays.ones((2, 3)), arrays.ones(2))
    assert x[0].validate() is None
    with pytest.raises(ValueError, match="inconsistent batch-shapes"):
        Checked(arrays.ones((2, 3)), arrays.ones(5))
    with pytest.raises(ValueError, match="inconsistent batch-shapes"):
        x.with_protected_values({"weights": arrays.ones(5)})


def test_validate_accepts_an_empty_effective_layout(backend):
    _, base, _, _ = backend

    @dataclass
    class Fixed(base):
        values: Any = field(init=False, default=None)
        protected_axes: ClassVar[dict[str, int]] = {"values": 2}

    assert Fixed().validate() is None


def test_dispatch_requires_the_shared_base(backend):
    name, _, arrays, _ = backend

    class Unrelated:
        protected_axes: ClassVar[dict[str, int]] = {"values": 1}
        permitted_functions: ClassVar[set] = set()
        shape = (2,)

        def protected_values(self, func=None):
            del func
            return {"values": arrays.ones((2, 3))}

        def with_protected_values(self, values, func=None):
            del values, func
            return self

    module = import_module(f"ndlike.protected_axis.{name}_functions")
    extract = getattr(module, f"{name}_axis_protected_internals")
    assert extract(Unrelated()) is None


def test_shape_property_override_does_not_change_internal_batch_shape(backend):
    _, base, arrays, _ = backend

    @dataclass
    class Custom(base):
        values: Any
        protected_axes: ClassVar[dict[str, int]] = {"values": 1}

        @property
        def shape(self):
            return (99,)

    x = Custom(arrays.ones((2, 3)))
    assert x.shape == (99,)
    internals = common.extract_axis_protected_internals(x)
    assert internals is not None
    assert internals.batch_shape == (2,)
