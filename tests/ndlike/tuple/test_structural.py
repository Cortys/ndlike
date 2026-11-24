"""The same logical-batch contract across all three tuple backends."""

from __future__ import annotations

from collections import deque
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


@pytest.mark.parametrize("target", [(6,), (-1, 2), (1, 2, 3), [6], [-1, 2], [1, 2, 3]])
def test_reshape_uses_common_batch(backend, target):
    x = example(backend)
    y = x.reshape(target)
    assert type(y) is type(x)
    assert_members(y, tuple(np.broadcast_to(np.asarray(m), x.shape).reshape(target) for m in x.members))
    if isinstance(target, tuple):
        y_varargs = x.reshape(*target)
        assert type(y_varargs) is type(x)
        assert_members(y_varargs, tuple(np.broadcast_to(np.asarray(m), x.shape).reshape(target) for m in x.members))


def test_reshape_can_replace_a_custom_member_with_a_native_array(backend):
    class Custom:
        shape = (2,)
        ndim = 1

        def __array_function__(self, func, types, args, kwargs) -> object:
            return func(backend.array([1, 2]), *args[1:], **kwargs)

        @classmethod
        def __torch_function__(cls, func, types, args, kwargs=None) -> object:
            return func(backend.array([1, 2]), *args[1:], **(kwargs or {}))

        __jax_function__ = __torch_function__

    custom = Custom()
    for value, shape in (
        (backend.cls((custom,)), (2, 1)),
        (backend.cls((custom, backend.array(np.ones((3, 2))))), (3, 2)),
    ):
        result = value.reshape(shape)
        assert isinstance(result, backend.cls)
        assert isinstance(result.members[0], type(backend.array([1, 2])))
        np.testing.assert_array_equal(result.members[0], np.resize([1, 2], shape))
        assert value.members[0] is custom


def test_reshape_packs_custom_outputs(backend):
    class Custom:
        shape = (2,)
        ndim = 1

        def __init__(self, start: int) -> None:
            self.start = start

        def outputs(self):
            values = backend.array([[self.start, self.start + 1]])
            return values, -values

        def __array_function__(self, func, types, args, kwargs) -> tuple:
            return self.outputs()

        @classmethod
        def __torch_function__(cls, func, types, args, kwargs) -> tuple:
            return args[0].outputs()

        __jax_function__ = __torch_function__

    result = backend.cls((Custom(1), Custom(3))).reshape(1, 2)
    assert isinstance(result, tuple)
    assert len(result) == 2
    for output, sign in zip(result, (1, -1), strict=True):
        assert isinstance(output, backend.cls)
        np.testing.assert_array_equal(output.members[0], sign * np.array([[1, 2]]))
        np.testing.assert_array_equal(output.members[1], sign * np.array([[3, 4]]))


def test_numpy_reshape_copy_option(backend):
    if backend.name != "numpy":
        pytest.skip("copy= option on reshape is NumPy-specific")
    x = backend.cls((np.arange(6.0).reshape(2, 3), np.arange(6.0).reshape(2, 3) + 10))
    copied = x.reshape([3, 2], copy=True)
    assert type(copied) is type(x)
    assert all(not np.shares_memory(a, b) for a, b in zip(copied.members, x.members, strict=True))
    viewed = x.reshape((3, 2), copy=False)
    assert all(np.shares_memory(a, b) for a, b in zip(viewed.members, x.members, strict=True))
    default_copy = x.reshape(3, 2, copy=None)
    assert_members(default_copy, tuple(m.reshape(3, 2) for m in x.members))
    with pytest.raises(ValueError, match="order"):
        x.reshape((3, 2), order="Z")


def test_reshape_backend_shape_inputs(backend):
    class _Dim:
        def __index__(self) -> int:
            return 6

    x = example(backend)
    expected_1d = tuple(np.broadcast_to(np.asarray(m), x.shape).reshape(6) for m in x.members)
    expected_2d = tuple(np.broadcast_to(np.asarray(m), x.shape).reshape(3, 2) for m in x.members)

    if backend.name == "torch":
        for target in (backend.xp.Size((3, 2)), (3, 2), [3, 2]):
            y = x.reshape(target)
            assert type(y) is type(x)
            assert_members(y, expected_2d)
        for scalar in (np.int64(6), backend.array(6), _Dim()):
            y = x.reshape(scalar)
            assert type(y) is type(x)
            assert_members(y, expected_1d)
        for unsupported in (range(3, 1, -1), deque([3, 2])):
            with pytest.raises(TypeError):
                x.reshape(unsupported)
    else:
        for target in (range(3, 1, -1), np.array([3, 2])):
            y = x.reshape(target)
            assert type(y) is type(x)
            assert_members(y, expected_2d)
        if backend.name == "jax":
            y_vec = x.reshape(backend.array(np.array([3, 2])))
            assert type(y_vec) is type(x)
            assert_members(y_vec, expected_2d)
        scalars: list[Any] = [np.int64(6), np.array(6), _Dim()]
        if backend.name == "jax":
            scalars.append(backend.array(6))
        for scalar in scalars:
            y = x.reshape(scalar)
            assert type(y) is type(x)
            assert_members(y, expected_1d)


def test_shaping_methods_are_centralized_at_runtime(backend):
    if backend.name in ("numpy", "jax"):
        centralized = ("reshape", "squeeze", "swapaxes")
    else:
        centralized = ("reshape", "squeeze", "unsqueeze", "movedim", "broadcast_to")
    for name in centralized:
        assert name not in backend.cls.__dict__
        assert name not in backend.base.__dict__


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


def test_shape_method_conveniences(backend):
    x = example(backend)
    expected_t = tuple(np.broadcast_to(np.asarray(m), x.shape).T for m in x.members)
    if backend.name == "torch":
        unsqueezed = x.unsqueeze(dim=0)
        assert type(unsqueezed) is type(x)
        assert unsqueezed.shape == (1, 2, 3)
        assert unsqueezed.squeeze().shape == x.shape
        assert unsqueezed.squeeze(None).shape == x.shape
        moved = x.movedim(source=0, destination=1)
        assert moved.shape == (3, 2)
        assert_members(moved, expected_t)
        assert unsqueezed.movedim((0, 2), (2, 0)).shape == (3, 2, 1)
        for broadcasted in (x.broadcast_to((4, 2, 3)), x.broadcast_to(size=(4, 2, 3)), x.broadcast_to(4, 2, 3)):
            assert broadcasted.shape == (4, 2, 3)
            assert_members(broadcasted, tuple(np.broadcast_to(np.asarray(m), (4, 2, 3)) for m in x.members))
        with pytest.raises(TypeError):
            x.broadcast_to(shape=(4, 2, 3))
    else:
        swapped = x.swapaxes(0, 1)
        assert type(swapped) is type(x)
        assert swapped.shape == (3, 2)
        assert_members(swapped, expected_t)
        if backend.name == "numpy":
            with pytest.raises(TypeError):
                x.swapaxes(axis1=0, axis2=1)
        else:
            swapped_kw = x.swapaxes(axis1=0, axis2=1)
            assert_members(swapped_kw, expected_t)


def test_gather_method_conveniences(backend):
    if backend.name == "numpy":
        pytest.skip("gather/take_along_axis convenience methods are JAX/Torch-specific")
    x = example(backend)
    idx_raw = np.array([[2, 0], [1, 2]], dtype=np.int64)
    idx = backend.array(idx_raw)
    idx_axis0_raw = np.array([[1, 0, 1]], dtype=np.int64)
    idx_axis0 = backend.array(idx_axis0_raw)

    if backend.name == "jax":
        gathered_default = x.take_along_axis(idx)
        gathered_kw = x.take_along_axis(indices=idx_axis0, axis=0)
        expected_default = tuple(
            np.take_along_axis(np.broadcast_to(np.asarray(m), x.shape), idx_raw, axis=-1) for m in x.members
        )
        expected_axis0 = tuple(
            np.take_along_axis(np.broadcast_to(np.asarray(m), x.shape), idx_axis0_raw, axis=0) for m in x.members
        )
        assert type(gathered_default) is type(x)
        assert gathered_default.shape == (2, 2)
        assert_members(gathered_default, expected_default)
        assert gathered_kw.shape == (1, 3)
        assert_members(gathered_kw, expected_axis0)

        per_member_idx = backend.cls((idx, backend.array(np.array([[0, 1], [2, 0]], dtype=np.int64)), idx))
        gathered_tuple_idx = x.take_along_axis(per_member_idx, axis=1)
        assert type(gathered_tuple_idx) is type(x)
        expected_per_member = tuple(
            np.take_along_axis(np.broadcast_to(np.asarray(m), x.shape), np.asarray(i), axis=1)
            for m, i in zip(x.members, per_member_idx.members, strict=True)
        )
        assert_members(gathered_tuple_idx, expected_per_member)
    else:
        gathered = x.gather(1, idx)
        gathered_kw = x.gather(dim=0, index=idx_axis0)
        expected = tuple(
            np.take_along_axis(np.broadcast_to(np.asarray(m), x.shape), idx_raw, axis=1) for m in x.members
        )
        expected_axis0 = tuple(
            np.take_along_axis(np.broadcast_to(np.asarray(m), x.shape), idx_axis0_raw, axis=0) for m in x.members
        )
        assert type(gathered) is type(x)
        assert gathered.shape == (2, 2)
        assert_members(gathered, expected)
        assert gathered_kw.shape == (1, 3)
        assert_members(gathered_kw, expected_axis0)

        per_member_idx = backend.cls((idx, backend.array(np.array([[0, 1], [2, 0]], dtype=np.int64)), idx))
        gathered_tuple_idx = x.gather(1, per_member_idx)
        assert type(gathered_tuple_idx) is type(x)
        expected_per_member = tuple(
            np.take_along_axis(np.broadcast_to(np.asarray(m), x.shape), np.asarray(i), axis=1)
            for m, i in zip(x.members, per_member_idx.members, strict=True)
        )
        assert_members(gathered_tuple_idx, expected_per_member)

    @dataclass(frozen=True)
    class Vector(backend.base):
        data: Any
        protected_axes: ClassVar[dict[str, int]] = {"data": 1}

    vec = Vector(backend.array(np.arange(24.0).reshape(2, 3, 4)))
    composed = backend.cls((vec, backend.array(np.arange(6.0).reshape(2, 3))))
    gathered_composed = composed.take_along_axis(idx, axis=1) if backend.name == "jax" else composed.gather(1, idx)
    assert type(gathered_composed) is type(composed)
    assert type(gathered_composed.members[0]) is Vector
    assert gathered_composed.shape == (2, 2)
    assert tuple(gathered_composed.members[0].data.shape) == (2, 2, 4)
    np.testing.assert_array_equal(
        gathered_composed.members[0].data,
        np.take_along_axis(np.asarray(vec.data), idx_raw[..., None], axis=1),
    )


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
        x[0] = x[0]
        np.testing.assert_array_equal(x.compact, np.ones((2, 3, 4)))
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
        assert x.dim() == x.ndimension() == 2
        assert x.numel() == x.nelement() == 6
        assert backend.xp.numel(x) == 6
        scalar = backend.cls((backend.array(1.0), backend.array(2.0)))
        assert scalar.dim() == scalar.ndimension() == 0
        assert scalar.numel() == scalar.nelement() == 1
        zero = backend.cls((backend.array(np.empty((2, 0))), backend.array(np.ones(1))))
        assert zero.dim() == zero.ndimension() == 2
        assert zero.numel() == zero.nelement() == 0
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
