"""Sequence result types, metadata, and validated indexed updates across backends."""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any, ClassVar

import numpy as np
import pytest


@pytest.fixture(params=["numpy", "jax", "torch"])
def backend(request):
    name = request.param
    pytest.importorskip(name)
    arrays = import_module("jax.numpy" if name == "jax" else name)
    module = import_module(f"ndlike.protected_axis.{name}")
    functions = import_module("ndlike.jax_functions") if name == "jax" else arrays
    prefix = "jax_" if name == "jax" else ""
    return (
        getattr(module, f"{name.title()}AxisProtected"),
        arrays.asarray,
        getattr(functions, prefix + "stack"),
        getattr(functions, prefix + "concatenate"),
    )


def _set(dst, index, rhs):
    if hasattr(dst, "at"):
        return dst.at[index].set(rhs)
    dst[index] = rhs
    return dst


@pytest.mark.parametrize("operation", ["stack", "concatenate"])
def test_common_ancestor_and_metadata(backend, operation):
    base, array, stack, concatenate = backend
    combine = stack if operation == "stack" else concatenate

    @dataclass(frozen=True, slots=True)
    class Abstract(base):
        ids: Any
        order: int
        protected_axes: ClassVar[dict[str, int]] = {"ids": 0}

        @abstractmethod
        def basis(self) -> str: ...

    class Concrete(Abstract):
        def basis(self) -> str:
            return "first"

    @dataclass(frozen=True, slots=True)
    class Refinement(Concrete):
        description: str = "extra metadata"

    class AnotherRefinement(Concrete):
        pass

    class Sibling(Abstract):
        def basis(self) -> str:
            return "second"

    x, y = Concrete(array([1, 2]), 3), Refinement(array([3, 4]), 3)
    for items, expected_cls in (
        ((x, y), Concrete),
        ((y, x), Concrete),
        ((y, y), Refinement),
        ((y, AnotherRefinement(x.ids, x.order)), Concrete),
    ):
        result = combine(items)
        assert type(result) is expected_cls
        assert result.order == 3
        np.testing.assert_array_equal(result.ids, combine([i.ids for i in items]))
    with pytest.raises(TypeError, match="abstract"):
        combine((x, Sibling(y.ids, y.order)))
    with pytest.raises(ValueError, match="order"):
        combine((y, Concrete(x.ids, 5)))
    with pytest.raises(ValueError, match="description"):
        combine((y, Refinement(y.ids, 3, "other")))

    @dataclass(frozen=True, slots=True)
    class Merged(Concrete):
        catalog: str = "players"

        @classmethod
        def _combine_sequence_metadata(cls, items, *, func) -> dict[str, int]:
            assert cls is Merged
            assert func is combine
            return {"order": max(item.order for item in items)}

    class RefinedMerge(Merged):
        pass

    left, right = Merged(x.ids, 2), RefinedMerge(y.ids, 7)
    for inputs in ((left, right), (right, left)):
        result = combine(inputs)
        assert type(result) is Merged
        assert result.order == 7
        assert left.order == 2
    with pytest.raises(ValueError, match="catalog"):
        combine((right, Merged(x.ids, 3, catalog="other")))


def test_compact_shared_broadcasts(backend):
    base, array, stack, concatenate = backend

    @dataclass(frozen=True)
    class Batch(base):
        tag: Any
        ids: Any
        protected_axes: ClassVar[dict[str, int]] = {"tag": 0, "ids": 0}
        broadcasted_batch: ClassVar[set[str]] = {"tag"}

    shared = array(3)
    x = Batch(shared, array([1, 2]))
    y = Batch(shared, array([3, 4]))
    for combine, reference in ((concatenate, np.concatenate), (stack, np.stack)):
        assert combine((x, y)).tag is shared
        for other_tag in (array(3), array(5), array([3, 5])):
            other = Batch(other_tag, y.ids)
            result = combine((x, other))
            assert result.tag.shape == result.shape
            np.testing.assert_array_equal(
                result.tag,
                reference((np.full(2, 3), np.broadcast_to(other_tag, (2,)))),
            )

    result = _set(x, 0, Batch(shared, array(9)))
    assert result.tag is shared
    np.testing.assert_array_equal(result.ids, [9, 2])
    result = _set(x, 0, Batch(array(7), array(9)))
    np.testing.assert_array_equal(result.tag, [7, 3])
    assert shared.shape == ()


def test_all_scalar_broadcasts_carry_result_shape(backend):
    base, array, stack, _ = backend

    @dataclass
    class Batch(base):
        x: Any
        y: Any
        protected_axes: ClassVar[dict[str, int]] = {"x": 0, "y": 0}
        broadcasted_batch: ClassVar[set[str]] = {"x", "y"}

    item = Batch(array(1), array(2))
    result = stack((item, item))
    assert result.shape == (2,)
    assert result.y is item.y
    np.testing.assert_array_equal(result.x, [1, 1])


def test_validated_and_raw_updates(backend):
    base, array, stack, concatenate = backend
    inits = []

    @dataclass
    class Batch(base):
        ids: Any
        protected_axes: ClassVar[dict[str, int]] = {"ids": 0}

        def __post_init__(self) -> None:
            inits.append(type(self))
            if bool((self.ids < 0).any()):
                msg = "negative id"
                raise ValueError(msg)

    x, y = Batch(array([1, 2])), Batch(array(3))
    assert _set(x, 0, y).shape == (2,)
    assert len(inits) == 2
    before = np.asarray(x.ids).copy()
    for rhs in (array(-1), (array(-1),)):
        with pytest.raises(ValueError, match="negative id"):
            _set(x, 0, rhs)
        np.testing.assert_array_equal(x.ids, before)
    for combine in (stack, concatenate):
        invalid = array([-1, 0])
        for inputs in ((invalid, x), (x, invalid)):
            with pytest.raises(ValueError, match="negative id"):
                combine(inputs)

    @dataclass
    class Derived(Batch):
        squared: Any = field(init=False)
        protected_axes: ClassVar[dict[str, int]] = {"squared": 0}

        def __post_init__(self) -> None:
            super().__post_init__()
            self.squared = self.ids**2

    d = Derived(array([1, 2]))
    changed = _set(d, 0, Derived(array(5)))
    np.testing.assert_array_equal(changed.squared, [25, 4])
    combined = stack((d, d))
    np.testing.assert_array_equal(combined.squared, combined.ids**2)

    class Strict(Batch):
        revalidate_on_batch_change: ClassVar[bool] = True

    strict, rhs = Strict(array([1, 2])), Strict(array(7))
    before = len(inits)
    _set(strict, 0, rhs)
    concatenate((strict, strict))
    assert len(inits) == before + 2


def test_jit_validated_sequence_and_set():
    jax = pytest.importorskip("jax")
    jnp = import_module("jax.numpy")
    from ndlike.jax_functions import jax_concatenate, jax_stack  # noqa: PLC0415
    from ndlike.protected_axis.jax import JaxAxisProtected  # noqa: PLC0415

    @dataclass(frozen=True)
    class Batch(JaxAxisProtected["Batch"]):
        ids: Any
        tag: Any
        protected_axes: ClassVar[dict[str, int]] = {"ids": 0, "tag": 0}
        broadcasted_batch: ClassVar[set[str]] = {"tag"}
        permitted_at_methods: ClassVar[set] = {"set", "add"}

        def __post_init__(self) -> None:
            if bool(jnp.any(self.ids < 0)):
                msg = "negative id"
                raise ValueError(msg)

    x = Batch(jnp.array([1, 2]), jnp.array(4))
    y = Batch(jnp.array([5, 6]), x.tag)
    index = jnp.array(1)
    combine = jax.jit(lambda a, b, i: (jax_stack((a, b)), jax_concatenate((a, b)), a.at[i].set(b[0])))
    stacked, joined, changed = combine(x, y, index)
    np.testing.assert_array_equal(stacked.ids, [[1, 2], [5, 6]])
    np.testing.assert_array_equal(joined.ids, [1, 2, 5, 6])
    np.testing.assert_array_equal(changed.ids, [1, 5])

    # The same traced field reference occurs twice inside this compiled function.
    compact = jax.jit(lambda a: (jax_stack((a, a)), jax_concatenate((a, a)), a.at[:].set(a)))(x)
    assert all(result.tag.shape == () for result in compact)
    with pytest.raises(ValueError, match="negative id"):
        x.at[0].add((jnp.array(-5), jnp.array(0)))  # ty:ignore[invalid-argument-type]


def test_failed_mutable_assignment_preserves_existing_buffers():
    from ndlike.protected_axis.numpy import NumpyAxisProtected  # noqa: PLC0415

    @dataclass(frozen=True)
    class Pair(NumpyAxisProtected):
        left: Any
        right: Any
        protected_axes: ClassVar[dict[str, int]] = {"left": 0, "right": 0}

    left, right = np.arange(3), np.arange(3)
    right.flags.writeable = False
    dst = Pair(left, right)
    with pytest.raises(ValueError, match="read-only"):
        dst[1] = Pair(np.array(8), np.array(9))
    assert dst.left is left
    np.testing.assert_array_equal(left, np.arange(3))


def test_assignment_checks_destination_metadata_and_cast(backend):
    base, array, _, _ = backend
    calls = []

    @dataclass
    class Batch(base):
        ids: Any
        limit: int
        protected_axes: ClassVar[dict[str, int]] = {"ids": 0}

        def __post_init__(self) -> None:
            calls.append(self.limit)
            if int(self.ids.min()) < 0 or int(self.ids.max()) > self.limit:
                msg = "ID outside the destination limits."
                raise ValueError(msg)

    dst = Batch(array(np.array([1], dtype=np.int16)), 5)
    rhs = Batch(array(np.int16(90)), 100)
    with pytest.raises(ValueError, match="destination limits"):
        _set(dst, 0, rhs)
    np.testing.assert_array_equal(dst.ids, [1])
    assert calls[-1] == 5

    allowed = Batch(array(np.int16(4)), 100)
    before = len(calls)
    result = _set(dst, 0, allowed)
    np.testing.assert_array_equal(result.ids, [4])
    assert result.limit == 5
    assert len(calls) == before + 1

    # Widening int8 -> int16 is lossless and needs no extra constructor call.
    narrow_rhs = Batch(array(np.int8(3)), 5)
    before = len(calls)
    np.testing.assert_array_equal(_set(dst, 0, narrow_rhs).ids, [3])
    assert len(calls) == before

    # Use equal static metadata here to isolate narrowing from the metadata check.
    narrow_dst = Batch(array(np.array([1], dtype=np.int8)), 250)
    wide_rhs = Batch(array(np.int16(200)), 250)
    with pytest.raises((ValueError, TypeError), match=r"destination limits|cast"):
        _set(narrow_dst, 0, wide_rhs)
    np.testing.assert_array_equal(narrow_dst.ids, [1])


@pytest.mark.parametrize("kind", ["numpy", "torch"])
def test_constructor_normalization_reaches_existing_buffer(kind):
    arrays = pytest.importorskip(kind)
    base = getattr(import_module(f"ndlike.protected_axis.{kind}"), f"{kind.title()}AxisProtected")

    @dataclass
    class Batch(base):
        ids: Any
        protected_axes: ClassVar[dict[str, int]] = {"ids": 0}

        def __post_init__(self) -> None:
            self.ids = abs(self.ids)

    dst = Batch(arrays.asarray([1, 2]))
    original = dst.ids
    dst[0] = -5
    assert dst.ids is original
    np.testing.assert_array_equal(original, [5, 2])


def test_small_validated_write_keeps_batch_buffer():
    from ndlike.protected_axis.numpy import NumpyAxisProtected  # noqa: PLC0415

    class Buffer(np.ndarray):
        def copy(self, order="C"):
            assert self.size <= 1, "Updating one ID copied the entire batch"
            return super().copy(order=order)

    rebuilds = 0

    @dataclass
    class Batch(NumpyAxisProtected["Batch"]):
        ids: Any
        protected_axes: ClassVar[dict[str, int]] = {"ids": 0}

        def _with_protected_values_structural(self, values, func=None):
            nonlocal rebuilds
            rebuilds += 1
            return super()._with_protected_values_structural(values, func)

    storage = np.arange(1000).view(Buffer)
    dst = Batch(storage)
    rhs = Batch(np.array([42]))[0]
    before = rebuilds
    dst[7] = rhs
    assert rebuilds == before
    assert dst.ids is storage
    assert storage[7] == 42


@pytest.mark.parametrize("kind", ["numpy", "torch"])
def test_slice_view_assignment_updates_parent_buffer(kind):
    arrays = pytest.importorskip(kind)
    base = getattr(import_module(f"ndlike.protected_axis.{kind}"), f"{kind.title()}AxisProtected")

    @dataclass
    class Batch(base):
        ids: Any
        protected_axes: ClassVar[dict[str, int]] = {"ids": 0}

        def __post_init__(self) -> None:
            self.ids = abs(self.ids)

    dst = Batch(arrays.asarray([1, 2, 3, 4]))
    dst[:2][0] = Batch(arrays.asarray(9))
    np.testing.assert_array_equal(dst.ids, [9, 2, 3, 4])
    dst[:2][1] = -7
    np.testing.assert_array_equal(dst.ids, [9, 7, 3, 4])


def test_sequence_array_metadata_checks_shape_and_validated_overrides(backend):
    base, array, stack, _ = backend

    @dataclass(frozen=True)
    class ArrayMeta(base):
        ids: Any
        weights: np.ndarray
        protected_axes: ClassVar[dict[str, int]] = {"ids": 0}

    a = ArrayMeta(array([1, 2]), np.array([1.0]))
    b = ArrayMeta(array([3, 4]), np.array([1.0, 1.0]))
    with pytest.raises(ValueError, match="weights"):
        stack((a, b))

    inits: list[int] = []

    @dataclass(frozen=True)
    class StrictMerged(base):
        ids: Any
        order: int
        protected_axes: ClassVar[dict[str, int]] = {"ids": 0}
        revalidate_on_batch_change: ClassVar[bool] = True

        def __post_init__(self) -> None:
            inits.append(self.order)

        @classmethod
        def _combine_sequence_metadata(cls, items, *, func) -> dict[str, int]:
            del func
            return {"order": max(item.order for item in items)}

    left, right = StrictMerged(array([1, 2]), 2), StrictMerged(array([3, 4]), 7)
    result = stack((left, right))
    assert result.order == 7
    assert left.order == 2
    assert inits == [2, 7, 7]


def test_nested_strict_field_revalidates_outer_assignment(backend):
    base, array, _, _ = backend
    outer_inits = 0

    @dataclass
    class Inner(base):
        ids: Any
        protected_axes: ClassVar[dict[str, int]] = {"ids": 0}
        revalidate_on_batch_change: ClassVar[bool] = True

    @dataclass
    class Outer(base):
        inner: Inner
        protected_axes: ClassVar[dict[str, int]] = {"inner": 0}

        def __post_init__(self) -> None:
            nonlocal outer_inits
            outer_inits += 1

    dst = Outer(Inner(array([1, 2])))
    rhs = Outer(Inner(array(5)))
    before = outer_inits
    _ = _set(dst, 0, rhs)
    assert outer_inits == before + 1


@pytest.mark.parametrize("kind", ["numpy", "torch"])
def test_validated_assignment_isolates_shared_field_and_refreshes_optional_state(kind):
    arrays = pytest.importorskip(kind)
    base = getattr(import_module(f"ndlike.protected_axis.{kind}"), f"{kind.title()}AxisProtected")

    @dataclass(slots=True)
    class Batch(base):
        ids: Any
        tag: Any
        limit: int
        minimum: int = field(init=False)
        protected_axes: ClassVar[dict[str, int]] = {"ids": 0, "tag": 0}
        broadcasted_batch: ClassVar[set[str]] = {"tag"}
        revalidate_on_batch_change: ClassVar[bool] = True

        def __post_init__(self) -> None:
            if int(self.ids.max()) > self.limit:
                self.tag[...] = -1
                msg = "out of range"
                raise ValueError(msg)
            if int(self.ids.min()) > 0:
                self.minimum = int(self.ids.min())

    shared = arrays.asarray(7)
    dst = Batch(arrays.asarray([1, 2]), shared, 5)
    assert dst.minimum == 1
    with pytest.raises(ValueError, match="out of range"):
        dst[0] = Batch(arrays.asarray(9), shared, 10)
    assert int(shared) == 7
    assert dst.minimum == 1
    np.testing.assert_array_equal(dst.ids, [1, 2])

    dst[0] = Batch(arrays.asarray(0), shared, 5)
    assert dst.tag is shared
    assert not hasattr(dst, "minimum")
    np.testing.assert_array_equal(dst.ids, [0, 2])
