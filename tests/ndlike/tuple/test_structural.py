"""The same logical-batch contract across all three tuple backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Never

import numpy as np
import pytest


def example(backend):
    return backend.cls(
        tuple(backend.array(v) for v in (np.array([[1.0], [2.0]]), np.array([10.0, 20.0, 30.0]), np.array(7.0)))
    )


def assert_members(actual, expected):
    assert len(actual.members) == len(expected)
    for value, reference in zip(actual.members, expected, strict=True):
        np.testing.assert_array_equal(np.broadcast_to(np.asarray(value), actual.shape), reference)


def test_shape_metadata_and_member_access(backend):
    x = example(backend)
    assert x.shape == (2, 3)
    assert x.ndim == 2
    assert len(x) == 2
    assert x.dtype == x.members[0].dtype
    assert x.device == x.members[0].device
    assert tuple(member.shape for member in x.members) == ((2, 1), (3,), ())
    assert len(tuple(x)) == 2
    assert all(item.shape == (3,) for item in x)


@pytest.mark.parametrize(
    "index",
    [
        0,
        -1,
        (slice(None), 1),
        slice(None),
        slice(0, 0),
        (slice(None), slice(0, 0)),
        (None, Ellipsis, None),
        (slice(None), None, slice(1, 3)),
        (1, 2),
    ],
)
def test_basic_indexing_preserves_compact_members(backend, index):
    x = example(backend)
    y = x[index]
    expected = tuple(np.broadcast_to(np.asarray(m), x.shape)[index] for m in x.members)
    assert y.shape == expected[0].shape
    assert type(y) is type(x)
    assert_members(y, expected)
    assert all(np.prod(a.shape) <= np.prod(b.shape) for a, b in zip(y.members, x.members, strict=True))


@pytest.mark.parametrize(
    "index", [np.array([1, 0, 1]), np.array([True, False]), (np.array([0, 1]), np.array([2, 0])), True, False]
)
def test_advanced_indexing_uses_logical_broadcast(backend, index):
    x = example(backend)
    y = x[index]
    expected = tuple(np.broadcast_to(np.asarray(m), x.shape)[index] for m in x.members)
    assert y.shape == expected[0].shape
    assert_members(y, expected)
    assert all(tuple(member.shape) == y.shape for member in y.members)


@pytest.mark.parametrize("target", [(6,), (-1, 2), (1, 2, 3)])
def test_reshape_uses_common_batch(backend, target):
    x = example(backend)
    y = x.reshape(target)
    assert type(y) is type(x)
    assert_members(y, tuple(np.broadcast_to(np.asarray(m), x.shape).reshape(target) for m in x.members))


def test_shape_dispatch(backend):
    x = example(backend)
    if backend.name == "torch":
        transposed = backend.func("permute")(x, (1, 0))
        expanded = backend.func("unsqueeze")(x, 0)
        moved = backend.func("movedim")(x, 0, 1)
    else:
        transposed = backend.func("transpose")(x, (1, 0))
        expanded = backend.func("expand_dims")(x, 0)
        moved = backend.func("moveaxis")(x, 0, 1)
    assert transposed.shape == moved.shape == (3, 2)
    expected = tuple(np.broadcast_to(np.asarray(m), x.shape).T for m in x.members)
    assert_members(transposed, expected)
    assert_members(moved, expected)
    assert expanded.shape == (1, 2, 3)
    assert backend.func("squeeze")(expanded).shape == x.shape
    assert backend.func("broadcast_to")(x, (4, 2, 3)).shape == (4, 2, 3)
    assert backend.func("stack")((x, x)).shape == (2, 2, 3)
    assert backend.func("concatenate")((x, x)).shape == (4, 3)


def test_validation_and_unsupported_operations(backend):
    with pytest.raises(TypeError, match="must be a tuple"):
        backend.cls([backend.array(1)])
    with pytest.raises(TypeError, match="shape and ndim"):
        backend.cls((lambda: backend.array(1),))
    with pytest.raises(ValueError, match="shape mismatch"):
        backend.cls((backend.array(np.ones(2)), backend.array(np.ones(3))))
    x = example(backend)
    with pytest.raises((ValueError, TypeError, RuntimeError)):
        x.reshape(5)
    with pytest.raises(TypeError, match="cannot be coerced"):
        np.asarray(x)
    with pytest.raises(TypeError, match="unsized"):
        len(backend.cls((backend.array(1),)))
    with pytest.raises(ValueError, match="same number"):
        backend.func("stack")((x, backend.cls((backend.array(np.ones((2, 3))),))))


def test_subclass_reconstruction(backend):
    @dataclass(frozen=True, eq=False)
    class Named(backend.cls):
        label: str = "kept"

    x = Named(example(backend).members)
    assert type(x.reshape(6)) is Named
    assert type(x[0]) is Named
    assert x.reshape(6).label == "kept"


def test_opaque_callable_representations_and_protected_composition(backend):
    @dataclass(frozen=True)
    class Vector(backend.base):
        data: Any
        protected_axes: ClassVar[dict[str, int]] = {"data": 1}

        def __call__(self, *args: object, **kwargs: object) -> Never:
            del args, kwargs
            pytest.fail("Shaping must not evaluate callable members")

        def __array__(self, *args: object, **kwargs: object) -> Never:
            del args, kwargs
            pytest.fail("Shaping must not coerce representations")

    @dataclass(frozen=True)
    class Matrix(backend.base):
        data: Any
        protected_axes: ClassVar[dict[str, int]] = {"data": 2}

    a = Vector(backend.array(np.arange(8.0).reshape(2, 1, 4)))
    b = Matrix(backend.array(np.arange(30.0).reshape(3, 2, 5)))
    x = backend.cls((a, b, backend.array(7.0)))
    y = x.reshape(6)
    assert y.shape == (6,)
    assert type(y.members[0]) is Vector
    assert type(y.members[1]) is Matrix
    assert y.members[0].data.shape == (6, 4)
    assert y.members[1].data.shape == (6, 2, 5)
    np.testing.assert_array_equal(y.members[0].data, np.broadcast_to(a.data, (2, 3, 4)).reshape(6, 4))
    assert x[1].members[0].data.shape == (1, 4)

    @dataclass(frozen=True)
    class Composite(backend.base):
        parts: Any
        protected_axes: ClassVar[dict[str, int]] = {"parts": 0}

    owner = Composite(x)
    assert owner.shape == (2, 3)
    assert type(owner.reshape(6)) is Composite
    assert owner.reshape(6).parts.shape == (6,)
    assert owner[1].parts.shape == (3,)


def test_protected_broadcast_parity(backend):
    @dataclass(frozen=True)
    class Shared(backend.base):
        compact: Any
        data: Any
        protected_axes: ClassVar[dict[str, int]] = {"compact": 1, "data": 1}
        broadcasted_batch: ClassVar[set[str]] = {"compact"}

    x = Shared(backend.array(np.ones(4)), backend.array(np.arange(30.0).reshape(2, 3, 5)))
    x.validate()
    assert x.shape == (2, 3)
    y = x.reshape(6)
    assert y.shape == (6,)
    assert y.compact.shape == (4,)
    assert x[1].compact.shape == (4,)
    assert x[:0].shape == (0, 3)
    assert backend.func("broadcast_to")(x, (4, 2, 3)).compact.shape == (4,)

    @dataclass(frozen=True)
    class AllShared(Shared):
        broadcasted_batch: ClassVar[set[str]] = {"data"}

    both = AllShared(backend.array(np.ones((2, 1, 4))), backend.array(np.ones((3, 5))))
    both.validate()
    assert both.shape == (2, 3)
    assert both.reshape(6).shape == (6,)
    assert both[1].shape == (3,)
    assert both.broadcasted_batch == {"compact", "data"}
    assert Shared.broadcasted_batch == {"compact"}


def test_protected_broadcast_arithmetic_remains_opt_in(backend):
    @dataclass(frozen=True)
    class Shared(backend.base):
        compact: Any
        data: Any
        protected_axes: ClassVar[dict[str, int]] = {"compact": 1, "data": 1}
        broadcasted_batch: ClassVar[set[str]] = {"compact"}
        permitted_functions: ClassVar[set] = {backend.func("sum"), backend.func("add"), backend.func("equal")}
        permitted_ufuncs: ClassVar[dict] = {np.add: ["__call__"], np.equal: ["__call__"]}

    x = Shared(backend.array(np.ones(4)), backend.array(np.ones((2, 3, 5))))
    doubled = x + x
    assert doubled.shape == (2, 3)
    assert doubled.compact.shape == (2, 3, 4)
    np.testing.assert_array_equal(backend.func("equal")(x, x), np.ones((2, 3), dtype=bool))
    reduced = backend.func("sum")(x, axis=0)
    assert reduced.shape == (3,)
    np.testing.assert_array_equal(reduced.compact, np.full((3, 4), 2.0))
    if backend.name != "jax":
        with pytest.raises(ValueError, match="fully expanded"):
            x[0] = x[0]
        np.testing.assert_array_equal(x.data, np.ones((2, 3, 5)))


def test_members_require_native_or_custom_dispatch(backend):
    class Coercible:
        shape = ()
        ndim = 0

        def __array__(self, *args: object, **kwargs: object) -> Never:
            del args, kwargs
            pytest.fail("Members must not be implicitly coerced")

    with pytest.raises(TypeError, match="implement"):
        backend.cls((Coercible(),))


def test_index_errors_and_zero_sized_shapes(backend):
    x = example(backend)
    for index in ((0, 0, 0), (Ellipsis, Ellipsis)):
        with pytest.raises(IndexError, match=r"indices|ellipses"):
            _ = x[index]
    empty = backend.cls((backend.array(np.empty((0, 1))), backend.array(np.ones((3,))), backend.array(2.0)))
    assert empty.shape == (0, 3)
    assert empty.reshape(0).shape == (0,)
    scalar = backend.cls((backend.array(2.0),))
    assert scalar.reshape(1, 1).shape == (1, 1)
    assert backend.func("broadcast_to")(scalar, (2, 3)).shape == (2, 3)


def test_metadata_functions_report_logical_shape(backend):
    x = example(backend)
    assert x.shape == (2, 3)
    assert x.ndim == 2
    if backend.name == "torch":
        assert x.size() == (2, 3)
        assert x.size(-1) == 3
        assert backend.xp.numel(x) == 6
    else:
        shape = backend.dispatch("a")(backend.xp.shape) if backend.name == "jax" else backend.xp.shape
        ndim = backend.dispatch("a")(backend.xp.ndim) if backend.name == "jax" else backend.xp.ndim
        size = backend.dispatch("a")(backend.xp.size) if backend.name == "jax" else backend.xp.size
        assert shape(x) == (2, 3)
        assert ndim(x) == 2
        assert size(x) == 6
        assert size(x, axis=-1) == 3
        assert size(x, 0) == 2
        scalar = backend.cls((backend.array(1), backend.array(2)))
        assert shape(scalar) == ()
        assert ndim(scalar) == 0
        assert size(scalar) == 1
        assert isinstance(size(scalar), int)
