"""Broadcasted batch fields: shape inference, compact storage, operations and JAX transforms.

Shared operation tests cover both a native batch field and fully broadcasted layouts.
Only the native layout can keep singleton fields compact through shape changes.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from functools import partial
from typing import Any, ClassVar
from unittest.mock import Mock

import numpy as np
import pytest

pytest.importorskip("jax")
import jax
import jax.numpy as jnp

from ndlike import jax_functions as jf
from ndlike.jax_like import JaxLike
from ndlike.protected_axis import jax as protected_jax
from ndlike.protected_axis._common import extract_axis_protected_internals, validate_batch_sync
from ndlike.protected_axis.jax import JaxAtMethod, JaxAxisProtected
from ndlike.protected_axis.jax_functions import jax_axis_protected_internals

_PERMITTED = {
    jf.jax_add,
    jf.jax_multiply,
    jf.jax_matmul,
    jf.jax_equal,
    jf.jax_sum,
    jf.jax_mean,
    jf.jax_average,
    jf.jax_std,
    jf.jax_var,
}


@dataclass(frozen=True)
class _Shared(JaxAxisProtected):
    # Deliberately put the broadcastable field first in the layout.
    shared: Any
    values: Any
    protected_axes: ClassVar[dict[str, int]] = {"shared": 1, "values": 1}
    broadcasted_batch: ClassVar[set[str]] = {"shared"}
    permitted_functions: ClassVar[set] = _PERMITTED
    permitted_at_methods: ClassVar[set[JaxAtMethod]] = {"set", "add", "multiply"}


@dataclass(frozen=True)
class _SharedScalar(JaxAxisProtected):
    shared: Any
    values: Any
    protected_axes: ClassVar[dict[str, int]] = {"shared": 0, "values": 1}
    broadcasted_batch: ClassVar[set[str]] = {"shared"}
    permitted_functions: ClassVar[set] = _PERMITTED
    permitted_at_methods: ClassVar[set[JaxAtMethod]] = {"set", "add"}


@dataclass(frozen=True)
class _SharedMatrix(JaxAxisProtected):
    shared: Any
    values: Any
    protected_axes: ClassVar[dict[str, int]] = {"shared": 2, "values": 1}
    broadcasted_batch: ClassVar[set[str]] = {"shared"}


class _FullBroadcast(_Shared):
    broadcasted_batch: ClassVar[set[str]] = {"shared", "values"}


@dataclass(frozen=True, slots=True)
class _FullBroadcastScalar(_SharedScalar):
    label: str = "metadata"
    broadcasted_batch: ClassVar[set[str]] = {"shared", "values"}


_SHAPE_OPERATIONS = [
    pytest.param(lambda x: jf.jax_reshape(x, (3, 2)), id="reshape"),
    pytest.param(lambda x: x.reshape(-1), id="reshape-inferred"),
    pytest.param(jf.jax_transpose, id="transpose"),
    pytest.param(lambda x: jf.jax_transpose(x, (2, 0, 1)), id="transpose-axes"),
    pytest.param(jf.jax_matrix_transpose, id="matrix-transpose"),
    pytest.param(lambda x: jf.jax_swapaxes(x, 0, -1), id="swapaxes"),
    pytest.param(lambda x: jf.jax_moveaxis(x, (0, 2), (2, 0)), id="moveaxis"),
    pytest.param(lambda x: jf.jax_expand_dims(x, -1), id="expand-dims"),
    pytest.param(lambda x: jf.jax_expand_dims(x, (0, -1)), id="expand-dims-multiple"),
    pytest.param(lambda x: jf.jax_squeeze(x, 1), id="squeeze-axis"),
    pytest.param(jf.jax_squeeze, id="squeeze"),
    pytest.param(lambda x: jf.jax_broadcast_to(x, (5, 2, 1, 3)), id="broadcast-to"),
]


def _example(batch=(2, 3), compact=(1, 3), *, kind=_Shared):
    values_batch = batch
    if kind is _FullBroadcast:
        # Split the batch extents between fields; neither need carry the full shape.
        aligned = (1,) * (len(batch) - len(compact)) + compact
        values_batch = tuple(1 if shared == size else size for shared, size in zip(aligned, batch, strict=True))
    return kind(
        jnp.arange(np.prod((*compact, 4)), dtype=float).reshape((*compact, 4)),
        jnp.arange(np.prod((*values_batch, 4)), dtype=float).reshape((*values_batch, 4)),
    )


@pytest.fixture(params=[_Shared, _FullBroadcast], ids=["native-batch", "all-broadcasted"])
def example(request):
    return partial(_example, kind=request.param)


@pytest.fixture(params=[_SharedScalar, _FullBroadcastScalar], ids=["native-batch", "all-broadcasted"])
def scalar_kind(request):
    return request.param


def _fully_broadcasted(shape=(2, 3)):
    shared_shape = (*shape[:-1], 1) if shape else ()
    values_shape = (*shape[-1:], 4)
    return _FullBroadcastScalar(jnp.full(shared_shape, 2.0), jnp.broadcast_to(jnp.arange(4.0), values_shape))


def _expanded(x):
    values = {}
    for name, axes in x.protected_axes.items():
        value = getattr(x, name)
        suffix = value.shape[-axes:] if axes else ()
        values[name] = jnp.broadcast_to(value, (*x.shape, *suffix))
    return x.with_protected_values(values)


def _assert_same(actual, expected):
    assert type(actual) is type(expected)
    assert actual.shape == expected.shape
    for name, axes in actual.protected_axes.items():
        suffix = getattr(actual, name).shape[-axes:] if axes else ()
        np.testing.assert_allclose(
            jnp.broadcast_to(getattr(actual, name), (*actual.shape, *suffix)),
            jnp.broadcast_to(getattr(expected, name), (*expected.shape, *suffix)),
            rtol=1e-5,
            atol=1e-5,
        )


@pytest.fixture
def index_helpers(monkeypatch):
    helpers = {
        name: Mock(wraps=getattr(protected_jax, name))
        for name in ("_is_basic_index", "_normalize_broadcast_index", "_broadcast_batch")
    }
    for name, helper in helpers.items():
        monkeypatch.setattr(protected_jax, name, helper)
    return helpers


class TestLayoutAndShapeInference:
    """Layout, shape inference and validation."""

    @pytest.mark.parametrize("compact", [(), (1,), (3,), (1, 3), (2, 1), (2, 3)])
    def test_layout_and_extraction(self, compact, example):
        x = example(compact=compact)
        assert x.primary_protected_name == "shared"
        assert _Shared.primary_protected_name == "shared"
        assert x.shape == (2, 3)
        assert x.protected_values()["shared"] is x.shared
        internals = jax_axis_protected_internals(x)
        assert internals is not None
        assert internals.batch_shape == x.shape
        assert internals.broadcasted_batch == frozenset(x.broadcasted_batch)
        if type(x) is _Shared:
            assert internals.values["values"] is x.values
        np.testing.assert_array_equal(internals.values["values"], _expanded(x).values)
        np.testing.assert_array_equal(internals.values["shared"], _expanded(x).shared)
        assert x.shared.shape == (*compact, 4)

    def test_shape_is_the_shared_broadcast_independently_of_primary(self):
        x = _fully_broadcasted()
        assert x.shared.shape == (2, 1)
        assert x.values.shape == (3, 4)
        assert x.shape == (2, 3)
        assert len(x) == 2
        assert x.primary_protected_name == "shared"
        assert x.protected_value() is x.shared
        assert set(x.protected_values()) == {"shared", "values"}
        assert [field.name for field in fields(x)] == ["shared", "values", "label"]
        internals = extract_axis_protected_internals(x)
        assert internals is not None
        assert internals.batch_shape == (2, 3)
        assert internals.values["shared"] is x.shared

    def test_replacement_derives_shape_from_the_resulting_fields(self):
        x = _fully_broadcasted()
        y = x.with_protected_values({"shared": jnp.ones((5, 1))})
        assert y.shape == (5, 3)
        assert y.values is x.values
        assert y.label == x.label
        assert x.shape == (2, 3)
        with pytest.raises(ValueError, match="shape mismatch"):
            jax_axis_protected_internals(_FullBroadcastScalar(jnp.ones((2, 3)), jnp.ones((5, 4))))

    def test_nonbroadcasted_shape_fast_path(self, monkeypatch):
        @dataclass(frozen=True)
        class WithWitness(_FullBroadcastScalar):
            witness: Any = None
            protected_axes: ClassVar[dict[str, int]] = {"witness": 0}

        def unexpected(*_args: object, **_kwargs: object):
            pytest.fail("A non-broadcasted field must avoid broadcast-shape inference.")

        monkeypatch.setattr(np, "broadcast_shapes", unexpected)
        x = WithWitness(jnp.array(2.0), jnp.arange(4.0), witness=jnp.ones((2, 3)))
        assert x.shape == (2, 3)
        assert x.primary_protected_name == "shared"
        internals = extract_axis_protected_internals(x)
        assert internals is not None
        assert internals.batch_shape == (2, 3)
        reshaped = x.reshape(6)
        assert isinstance(reshaped, WithWitness)
        assert reshaped.shape == (6,)
        assert reshaped.shared.shape == ()
        assert reshaped.values.shape == (4,)
        assert x.with_protected_values({"witness": jnp.ones(5)}).shape == (5,)
        # A shape query reads only the witness, even if other fields are invalid.
        assert WithWitness(jnp.ones((5, 3)), jnp.ones(4), witness=jnp.ones((2, 3))).shape == (2, 3)

    @pytest.mark.parametrize("slots", [False, True])
    def test_primary_name_is_precomputed_for_each_subclass(self, slots):
        @dataclass(frozen=True, slots=slots)
        class OtherShared(_Shared):
            other: Any
            protected_axes: ClassVar[dict[str, int]] = {"other": 1}
            broadcasted_batch: ClassVar[set[str]] = {"values"}

        assert OtherShared.primary_protected_name == "shared"
        assert _Shared.primary_protected_name == "shared"
        x = OtherShared(jnp.ones(4), jnp.ones(4), jnp.ones((2, 3, 4)))
        assert x.shape == (2, 3)
        internals = jax_axis_protected_internals(x)
        assert internals is not None

    def test_runtime_operations_reuse_precomputed_primary_name(self, monkeypatch, example):
        def unexpected_selection(*_args: object, **_kwargs: object):
            pytest.fail("Primary-field selection must only run when defining the class.")

        monkeypatch.setattr("ndlike.protected_axis.jax.primary_protected_name", unexpected_selection)
        monkeypatch.setattr("ndlike.protected_axis._common.primary_protected_name", unexpected_selection)
        x = example(compact=())
        assert x.primary_protected_name == "shared"
        assert x.shape == (2, 3)
        assert x[0].shape == (3,)
        assert x.reshape(6).shape == (6,)
        assert (x + 1).shape == x.shape
        assert jf.jax_conj(x).shape == x.shape
        assert jf.jax_sum(x).shape == ()
        assert jf.jax_stack((x, x)).shape == (2, 2, 3)
        assert jf.jax_concatenate((x, x)).shape == (4, 3)

    @pytest.mark.parametrize("compact", [(4,), (3, 1), (1, 1, 1)])
    def test_explicit_validation_rejects_invalid_broadcast_shape(self, compact):
        x = _example(compact=compact)
        with pytest.raises(ValueError, match="cannot broadcast to logical batch shape"):
            validate_batch_sync(x.protected_values(), x.protected_axes, x.broadcasted_batch)

    @pytest.mark.parametrize("declaration", [{"missing"}, ["shared"], {1}])
    def test_invalid_broadcast_declaration(self, declaration):
        with pytest.raises(TypeError, match="broadcasted_batch"):
            type(
                "InvalidShared",
                (JaxAxisProtected,),
                {
                    "__annotations__": {"shared": Any, "values": Any},
                    "protected_axes": {"shared": 0, "values": 0},
                    "broadcasted_batch": declaration,
                },
            )

    def test_unmarked_fields_still_require_identical_shapes(self):
        @dataclass(frozen=True)
        class Pair(JaxAxisProtected):
            shared: Any
            first: Any
            second: Any
            protected_axes: ClassVar[dict[str, int]] = {"shared": 0, "first": 0, "second": 0}
            broadcasted_batch: ClassVar[set[str]] = {"shared"}

        x = Pair(jnp.array(1), jnp.ones(2), jnp.ones(3))
        with pytest.raises(ValueError, match="inconsistent batch-shapes"):
            validate_batch_sync(x.protected_values(), x.protected_axes, x.broadcasted_batch)

    def test_shared_field_cannot_enlarge_native_batch_shape_or_broadcast_protected_axes(self):
        x = _example(batch=(1, 3), compact=(2, 3))
        with pytest.raises(ValueError, match="cannot broadcast to logical batch shape"):
            validate_batch_sync(x.protected_values(), x.protected_axes, x.broadcasted_batch)
        y = _Shared(jnp.array(2.0), jnp.ones((2, 3, 4)))
        with pytest.raises(ValueError, match="removed protected trailing axes"):
            validate_batch_sync(y.protected_values(), y.protected_axes, y.broadcasted_batch)


class TestIndexing:
    """Indexing and indexed updates."""

    @pytest.mark.parametrize("marked", [False, True])
    @pytest.mark.parametrize("index", [(slice(None), 1), (slice(None), np.array([2, 0]))])
    def test_full_fields_skip_broadcast_index_helpers(self, marked, index, index_helpers):
        @dataclass(frozen=True)
        class Pair(JaxAxisProtected[JaxLike[Any], JaxLike[Any], JaxLike[Any], tuple[jax.Array, jax.Array]]):
            shared: Any
            values: Any
            protected_axes: ClassVar[dict[str, int]] = {"shared": 1, "values": 1}
            broadcasted_batch: ClassVar[set[str]] = {"shared"} if marked else set()

        x = Pair(jnp.arange(24.0).reshape(2, 3, 4), jnp.ones((2, 3, 4)))
        result = x[index]
        assert isinstance(result, Pair)
        np.testing.assert_array_equal(result.shared, x.shared[index])
        updated = x.at[index].set((jnp.zeros(4), jnp.zeros(4)))
        assert isinstance(updated, Pair)
        np.testing.assert_array_equal(updated.shared, x.shared.at[index].set(jnp.zeros(4)))
        for helper in index_helpers.values():
            helper.assert_not_called()

    @pytest.mark.parametrize(("index", "basic"), [((slice(None), 1), True), ((slice(None), np.array([2, 0])), False)])
    def test_gather_classifies_once_for_multiple_compact_fields(self, index, basic, index_helpers):
        @dataclass(frozen=True)
        class MultipleShared(JaxAxisProtected):
            first: Any
            second: Any
            full: Any
            values: Any
            protected_axes: ClassVar[dict[str, int]] = {"first": 1, "second": 1, "full": 1, "values": 1}
            broadcasted_batch: ClassVar[set[str]] = {"first", "second", "full"}

        x = MultipleShared(
            jnp.arange(4.0),
            jnp.arange(12.0).reshape(1, 3, 4),
            jnp.arange(24.0).reshape(2, 3, 4),
            jnp.ones((2, 3, 4)),
        )
        for count in (1, 2):
            result = x[index]
            assert isinstance(result, MultipleShared)
            for name in x.protected_axes:
                field = getattr(result, name)
                expected = jnp.broadcast_to(getattr(x, name), (2, 3, 4))[index]
                np.testing.assert_array_equal(jnp.broadcast_to(field, expected.shape), expected)
            classifier = index_helpers["_is_basic_index"]
            assert classifier.call_count == count
            if basic:
                assert index_helpers["_normalize_broadcast_index"].call_count == 2 * count
                index_helpers["_broadcast_batch"].assert_not_called()
            else:
                index_helpers["_normalize_broadcast_index"].assert_not_called()
                assert index_helpers["_broadcast_batch"].call_count == 2 * count

    def test_scatter_materializes_only_compact_destinations(self, index_helpers):
        x = _example(compact=())
        result = x.at[:, 1].add((jnp.ones(4), jnp.ones(4)))
        assert isinstance(result, _Shared)
        np.testing.assert_array_equal(result.shared, _expanded(x).shared.at[:, 1].add(jnp.ones(4)))
        index_helpers["_is_basic_index"].assert_not_called()
        index_helpers["_normalize_broadcast_index"].assert_not_called()
        index_helpers["_broadcast_batch"].assert_called_once()

    @pytest.mark.parametrize("compact", [(), (1,), (3,), (1, 3), (2, 1), (2, 3)])
    @pytest.mark.parametrize(
        "index",
        [
            0,
            -1,
            slice(None),
            slice(1, 10),
            slice(0, 0),
            (slice(None, None, -1), slice(None, None, -1)),
            (slice(None), 2),
            (1, 2),
            (None, slice(None), None, slice(None), None),
            (None, 1, Ellipsis),
            (Ellipsis, None),
            (slice(None), None, 1),
            (slice(None), slice(0, 0)),
            (),
        ],
    )
    def test_basic_indexing_preserves_compact_storage(self, compact, index, example):
        x = example(compact=compact)
        result = x[index]
        assert type(result) is type(x)
        _assert_same(result, _expanded(x)[index])
        assert result.shared.size <= x.shared.size
        assert x.shared.shape == (*compact, 4)

    def test_basic_indexing_respects_logical_singleton_slices(self, example):
        x = example(batch=(1, 3), compact=(1, 3))
        result = x[:0]
        assert type(result) is type(x)
        assert result.shape == (0, 3)
        assert result.shared.shape == (0, 3, 4)

    @pytest.mark.parametrize("index", [(Ellipsis, Ellipsis), (0, 0, 0)])
    def test_basic_indexing_rejects_invalid_batch_indices(self, index, example):
        with pytest.raises(IndexError):
            example()[index]

    @pytest.mark.parametrize("compact", [(), (1,), (3,), (2, 1)])
    @pytest.mark.parametrize(
        "index",
        [
            np.array([1, 0, 1]),
            np.array([True, False]),
            (slice(None), np.array([True, False, True])),
            (np.array([0, 1]), np.array([2, 0])),
            (np.array([0, 1]), None, np.array([2, 0])),
            np.array([[True, False, True], [False, True, False]]),
            True,
            False,
        ],
    )
    def test_advanced_indexing_materializes(self, compact, index, example):
        x = example(compact=compact)
        result = x[index]
        expected = _expanded(x)[index]
        assert type(result) is type(x)
        _assert_same(result, expected)
        assert result.shared.shape == expected.shared.shape

    @pytest.mark.parametrize("method", ["set", "add", "multiply"])
    @pytest.mark.parametrize("index", [(1, 2), (slice(None), 1), np.array([True, False])])
    def test_indexed_updates_materialize_destinations(self, method, index, example):
        x = example(compact=())
        candidate = (jnp.full((4,), 7.0), jnp.full((4,), 8.0))
        result = getattr(x.at[index], method)(candidate)
        expected = getattr(_expanded(x).at[index], method)(candidate)
        _assert_same(result, expected)
        assert result.shared.shape == (2, 3, 4)
        assert x.shared.shape == (4,)


class TestShapeOperations:
    """Shape operations and singleton storage."""

    @pytest.mark.parametrize("compact", [(), (1,), (1, 1)])
    @pytest.mark.parametrize("shape", [(6,), (-1, 2), (1, 2, 3), (None, 6)])
    @pytest.mark.parametrize("order", ["C", "F"])
    def test_reshape_reuses_single_entry_fields(self, compact, shape, order):
        x = _example(compact=compact)
        result = jf.jax_reshape(x, shape, order=order)
        assert isinstance(result, _Shared)
        assert result.shared.shape == (4,)
        _assert_same(result, jf.jax_reshape(_expanded(x), shape, order=order))

    def test_scalar_reshape_and_invalid_target(self):
        x = _SharedScalar(jnp.array(2.0), jnp.ones((2, 3, 4)))
        result = x.reshape(6)
        assert isinstance(result, _SharedScalar)
        assert result.shared.shape == ()
        assert result.shape == (6,)
        with pytest.raises(TypeError, match="reshape"):
            x.reshape(5)

    def test_reshape_materializes_partially_shared_fields(self, example):
        x = example(compact=(1, 3))
        result = x.reshape(6)
        assert type(result) is type(x)
        assert result.shared.shape == (6, 4)
        _assert_same(result, _expanded(x).reshape(6))

    def test_singleton_fields_cannot_hide_result_dimensions(self):
        x = _fully_broadcasted(())
        assert x.shape == ()
        reshaped = x.reshape(1, 1)
        assert reshaped.shape == (1, 1)
        assert jax.jit(lambda x: x)(reshaped).shape == (1, 1)
        broadcasted = jf.jax_broadcast_to(x, (2, 3))
        assert isinstance(broadcasted, _FullBroadcastScalar)
        assert broadcasted.shared.shape == (2, 3)
        assert broadcasted.values.shape == (2, 3, 4)
        assert broadcasted.shape == (2, 3)
        assert jax.jit(lambda x: jf.jax_broadcast_to(x, (2, 3)))(x).shape == (2, 3)

    @pytest.mark.parametrize("function", _SHAPE_OPERATIONS)
    @pytest.mark.parametrize("compact", [(), (1,), (1, 1, 1)])
    @pytest.mark.parametrize(
        ("kind", "suffix"),
        [
            (_SharedScalar, ()),
            (_Shared, (4,)),
            (_SharedMatrix, (2, 3)),
            (_FullBroadcastScalar, ()),
            (_FullBroadcast, (4,)),
        ],
    )
    def test_shape_only_operations_handle_single_entry_fields(self, function, compact, kind, suffix):
        shared = jnp.arange(np.prod((*compact, *suffix)), dtype=float).reshape((*compact, *suffix))
        x = kind(shared, jnp.arange(24.0).reshape(2, 1, 3, 4))
        result = function(x)
        expected_shape = (*result.shape, *suffix) if "values" in x.broadcasted_batch else suffix
        assert result.shared.shape == expected_shape
        _assert_same(result, function(_expanded(x)))
        assert x.shared is shared
        assert x.shared.shape == (*compact, *suffix)

    @pytest.mark.parametrize("function", _SHAPE_OPERATIONS)
    def test_shape_only_operations_handle_single_entry_fields_under_jit(self, function, example):
        x = example(batch=(2, 1, 3), compact=(1, 1, 1))
        result = jax.jit(function)(x)
        expected_shape = (*result.shape, 4) if type(x) is _FullBroadcast else (4,)
        assert result.shared.shape == expected_shape
        _assert_same(result, function(_expanded(x)))

    @pytest.mark.parametrize("function", _SHAPE_OPERATIONS)
    def test_shape_only_operations_materialize_partially_shared_fields(self, function, example):
        x = example(batch=(2, 1, 3), compact=(1, 1, 3))
        result = function(x)
        expected = function(_expanded(x))
        _assert_same(result, expected)
        assert result.shared.shape == expected.shared.shape

    @pytest.mark.parametrize("compact", [(), (1,), (1, 1, 1)])
    @pytest.mark.parametrize("backend", ["jax", "numpy"])
    @pytest.mark.parametrize("copy", [None, False, True])
    def test_reshape_copy_preserves_single_entry_storage(self, compact, backend, copy):
        x = _example(batch=(2, 1, 3), compact=compact)
        if backend == "numpy":
            x = _Shared(np.asarray(x.shared), x.values)
        result = jf.jax_reshape(x, (6,), copy=copy)
        assert isinstance(result, _Shared)
        assert result.shared.shape == (4,)
        np.testing.assert_array_equal(result.shared, x.shared.reshape(4))
        if backend == "numpy":
            assert np.shares_memory(result.shared, x.shared) == (copy is not True)

    @pytest.mark.parametrize(
        "function",
        [
            lambda x: jf.jax_reshape(x, (5,)),
            lambda x: jf.jax_transpose(x, (0, 1)),
            lambda x: jf.jax_transpose(x, (0, 0, 2)),
            lambda x: jf.jax_swapaxes(x, 3, 0),
            lambda x: jf.jax_moveaxis(x, 3, 0),
            lambda x: jf.jax_squeeze(x, 0),
            lambda x: jf.jax_expand_dims(x, 4),
            lambda x: jf.jax_broadcast_to(x, (2, 1)),
        ],
    )
    def test_shape_operations_validate_batch_shape(self, function, example):
        x = example(batch=(2, 1, 3), compact=(1, 1, 1))
        with pytest.raises((TypeError, ValueError)):
            function(x)

    def test_matrix_transpose_requires_two_batch_axes(self, example):
        with pytest.raises(ValueError, match="at least 2 batch dimensions"):
            jf.jax_matrix_transpose(example(batch=(6,), compact=()))

    def test_point_batches(self, example):
        point = example(batch=(), compact=())
        indexed = point[None]
        assert isinstance(indexed, _Shared)
        assert indexed.shape == (1,)
        assert indexed.shared.shape == (1, 4)
        reshaped = point.reshape(1, 1)
        assert isinstance(reshaped, _Shared)
        expected_shape = (1, 1, 4) if type(point) is _FullBroadcast else (4,)
        assert reshaped.shared.shape == expected_shape
        _assert_same(reshaped, _expanded(point).reshape(1, 1))

    @pytest.mark.parametrize("shape", [(0, 3), (1, 0, 1)])
    def test_empty_batches_survive_shape_changes(self, shape, example):
        x = example(batch=shape, compact=(*shape[:-1], 1))
        _assert_same(x.reshape(0), _expanded(x).reshape(0))
        _assert_same(x[:0], _expanded(x)[:0])
        _assert_same(jf.jax_sum(x), jf.jax_sum(_expanded(x)))
        if shape[0] == 0:
            with pytest.raises(IndexError):
                x[0]


class TestValueOperations:
    """Element operations, reductions and sequences."""

    @pytest.mark.parametrize("function", [jf.jax_copy, jf.jax_conj, lambda x: jf.jax_astype(x, jnp.complex64)])
    @pytest.mark.parametrize(
        ("batch", "compact"), [((2, 3), ()), ((2, 3), (3,)), ((2, 3), (2, 1)), ((), ()), ((0, 3), (0, 1))]
    )
    def test_value_only_preserves_compact_shapes(self, function, batch, compact, example):
        x = example(batch=batch, compact=compact)
        x = type(x)(x.shared + 1j, x.values + 2j)
        result = function(x)
        assert result.shared.shape == x.shared.shape
        assert result.values.shape == x.values.shape
        _assert_same(result, function(_expanded(x)))
        assert x.astype(jnp.complex64).shared.shape == x.shared.shape

    def test_array_convenience_methods_preserve_shape(self, example):
        x = example(compact=(2, 1))
        for y in (x.stop_gradient(), x.to_device(jax.devices()[0]), x.view(jnp.int32)):
            assert y.shape == x.shape
            assert y.shared.shape == x.shared.shape
            assert y.values.shape == x.values.shape

    @pytest.mark.parametrize(
        "function",
        [
            lambda x: x + 2,
            lambda x: x * x,
            lambda x: jf.jax_stack((x, x), axis=1),
            lambda x: jf.jax_concatenate((x, x), axis=0),
            lambda x: jf.jax_take_along_axis(x, jnp.array([[2, 0], [1, 2]]), axis=1),
        ],
    )
    def test_other_operations_receive_materialized_values(self, function, example):
        x = example(compact=())
        result = function(x)
        expected = function(_expanded(x))
        _assert_same(result, expected)
        assert result.shared.shape == expected.shared.shape
        assert x.shared.shape == (4,)

    @pytest.mark.parametrize("function", [jf.jax_sum, jf.jax_mean, jf.jax_average, jf.jax_std, jf.jax_var])
    @pytest.mark.parametrize("axis", [None, 0, 1, (0, 1), ()])
    @pytest.mark.parametrize("keepdims", [False, True])
    def test_reductions_use_logical_extents(self, function, axis, keepdims, example):
        x = example(compact=(1, 3))
        result = function(x, axis=axis, keepdims=keepdims)
        _assert_same(result, function(_expanded(x), axis=axis, keepdims=keepdims))

    def test_weighted_reduction_and_comparison(self, example):
        x = example(compact=(2, 1))
        weights = jnp.arange(6.0).reshape(2, 3) + 1
        _assert_same(jf.jax_average(x, weights=weights), jf.jax_average(_expanded(x), weights=weights))
        np.testing.assert_array_equal(jf.jax_equal(x, _expanded(x)), jnp.ones(x.shape, dtype=bool))

    def test_sequence_inputs_expand_to_their_own_logical_shapes(self, example):
        x = example(batch=(2, 3), compact=(1, 3))
        y = example(batch=(5, 3), compact=(5, 1))
        result = jf.jax_concatenate((x, y), axis=0)
        assert isinstance(result, _Shared)
        assert result.shared.shape == (7, 3, 4)
        _assert_same(result, jf.jax_concatenate((_expanded(x), _expanded(y)), axis=0))

        y = example(compact=(2, 1))
        result = jf.jax_stack((x, y), axis=-1)
        _assert_same(result, jf.jax_stack((_expanded(x), _expanded(y)), axis=-1))

    def test_matmul_materializes_shared_fields(self, scalar_kind):
        x = scalar_kind(jnp.array(2.0), jnp.ones((2, 3, 4)))
        y = scalar_kind(jnp.array(3.0), jnp.ones((3, 5, 4)))
        result = x @ y
        assert result.shared.shape == (2, 5)
        np.testing.assert_array_equal(result.shared, jnp.full((2, 5), 18.0))


class TestSidecarsAndNestedFields:
    """NumPy sidecars and nested protected fields."""

    @pytest.mark.parametrize("function", _SHAPE_OPERATIONS)
    @pytest.mark.parametrize("compact", [(), (1,), (1, 1, 1)])
    def test_shape_only_operations_preserve_numpy_sidecars(self, function, compact, example):
        shared = np.full((*compact, 2), "shared", dtype=object)
        x = example(batch=(2, 1, 3), compact=compact)
        x = type(x)(shared, x.values)
        result = function(x)
        assert isinstance(result.shared, np.ndarray)
        expected_shape = (*result.shape, 2) if type(x) is _FullBroadcast else (2,)
        assert result.shared.shape == expected_shape
        np.testing.assert_array_equal(result.shared, np.full(expected_shape, "shared", dtype=object))
        assert x.shared is shared

    def test_numpy_sidecars_remain_numpy_and_compact_when_supported(self, scalar_kind):
        x = scalar_kind(np.array("shared", dtype=object), jnp.ones((2, 3, 4)))
        indexed = x[1]
        reshaped = x.reshape(6)
        assert isinstance(indexed, scalar_kind)
        assert isinstance(reshaped, scalar_kind)
        assert indexed.shared.shape == ()
        assert reshaped.shared.shape == ((6,) if scalar_kind is _FullBroadcastScalar else ())
        expanded = jf.jax_broadcast_to(x, (5, 2, 3))
        assert isinstance(expanded, scalar_kind)
        assert isinstance(expanded.shared, np.ndarray)
        assert expanded.shape == (5, 2, 3)
        if scalar_kind is _SharedScalar:
            assert expanded.shared is x.shared
        else:
            np.testing.assert_array_equal(expanded.shared, np.full((5, 2, 3), "shared", dtype=object))
        result = x.at[0].set((np.array("changed", dtype=object), jnp.zeros(4)))
        assert isinstance(result, scalar_kind)
        np.testing.assert_array_equal(result.shared, [["changed"] * 3, ["shared"] * 3])
        with pytest.raises(TypeError, match="NumPy array"):
            jax.jit(lambda x: x)(x)

    def test_nested_single_entry_fields_survive_shape_only_operations(self):
        inner = _SharedScalar(jnp.array(2.0), jnp.ones(4))
        x = _SharedScalar(inner, jnp.ones((2, 1, 3, 4)))
        for function in (jf.jax_transpose, jf.jax_squeeze, lambda x: jf.jax_broadcast_to(x, (5, 2, 1, 3))):
            result = function(x)
            assert isinstance(result, _SharedScalar)
            assert result.shared is inner
            assert result.shared.shape == ()
        copied = jf.jax_reshape(x, (6,), copy=True)
        assert isinstance(copied, _SharedScalar)
        assert isinstance(copied.shared, _SharedScalar)
        assert copied.shared is not inner

    @pytest.mark.parametrize("compact", [(), (1,), (1, 1)])
    @pytest.mark.parametrize("order", ["C", "F"])
    @pytest.mark.parametrize("copy", [None, False, True])
    def test_reshape_nested_single_entry_numpy_sidecar(self, compact, order, copy):
        @dataclass(frozen=True)
        class Mixed(JaxAxisProtected):
            values: Any
            label: Any
            protected_axes: ClassVar[dict[str, int]] = {"values": 1, "label": 0}

        inner = Mixed(jnp.arange(4.0).reshape((*compact, 4)), np.full(compact, "label", dtype=object))
        x = _SharedScalar(inner, jnp.arange(24.0).reshape(2, 3, 4))
        result = jf.jax_reshape(x, (3, 2), order=order, copy=copy)
        expected = jf.jax_reshape(inner, (), order=order, copy=copy)
        assert isinstance(result, _SharedScalar)
        assert isinstance(result.shared, Mixed)
        assert isinstance(expected, Mixed)
        assert result.shape == (3, 2)
        assert result.shared.shape == ()
        assert isinstance(result.shared.label, np.ndarray)
        np.testing.assert_array_equal(result.shared.values, expected.values)
        np.testing.assert_array_equal(result.shared.label, expected.label)
        np.testing.assert_array_equal(result.values, jnp.reshape(x.values, (3, 2, 4), order=order))
        assert np.shares_memory(result.shared.label, inner.label) == (copy is not True)
        assert inner.shape == compact

    def test_nested_protected_fields_preserve_custom_dispatch(self, scalar_kind):
        inner = scalar_kind(jnp.array(2.0), jnp.ones(4))
        x = scalar_kind(inner, jnp.ones((2, 3, 4)))
        result = x + 1
        assert isinstance(result.shared, scalar_kind)
        assert result.shared.shape == x.shape
        np.testing.assert_array_equal(result.shared.shared, jnp.full(x.shape, 3.0))
        np.testing.assert_array_equal(result.shared.values, jnp.full((2, 3, 4), 2.0))
        indexed = x[1, 2]
        assert isinstance(indexed, scalar_kind)
        assert isinstance(indexed.shared, scalar_kind)
        assert indexed.shared.shape == ()


class TestJaxTransformations:
    """Pytrees and JAX transformations."""

    def test_pytree_and_jit_preserve_actual_storage(self, scalar_kind):
        x = (
            _fully_broadcasted()
            if scalar_kind is _FullBroadcastScalar
            else scalar_kind(jnp.array(2.0), jnp.ones((2, 3, 4)))
        )
        leaves, treedef = jax.tree.flatten(x)
        assert len(leaves) == 2
        assert leaves[0] is x.shared
        assert leaves[1] is x.values
        rebuilt = jax.tree.unflatten(treedef, leaves)
        _assert_same(rebuilt, x)
        if scalar_kind is _FullBroadcastScalar:
            assert rebuilt.label == x.label
        for function in (lambda x: x, jf.jax_conj):
            result = jax.jit(function)(x)
            assert result.shared.shape == x.shared.shape
            assert result.values.shape == x.values.shape
            _assert_same(result, function(_expanded(x)))
        for function in (lambda x: x[:1], lambda x: x.reshape(6), lambda x: x + 1):
            _assert_same(jax.jit(function)(x), function(_expanded(x)))

    def test_jit_indexed_updates_leave_unselected_entries_unchanged(self, scalar_kind):
        x = scalar_kind(jnp.array(2.0), jnp.ones((2, 3, 4)))
        result = jax.jit(lambda x, i: x.at[i].add((3.0, jnp.ones(4))))(x, jnp.array(1))
        np.testing.assert_array_equal(result.shared, [[2.0] * 3, [5.0] * 3])
        assert x.shared.shape == ()

    def test_jit_basic_dynamic_integer_and_advanced_indices(self, example):
        x = example(compact=(2, 1))
        result = jax.jit(lambda x, i: x[:, i])(x, jnp.array(2))
        assert result.shared.shape == (2, 4)
        _assert_same(result, _expanded(x)[:, 2])
        result = jax.jit(lambda x, i: x[:, i])(x, jnp.array([2, 0]))
        assert result.shared.shape == (2, 2, 4)
        _assert_same(result, _expanded(x)[:, np.array([2, 0])])
        with pytest.raises(jax.errors.NonConcreteBooleanIndexError):
            jax.jit(lambda x, mask: x[mask])(x, jnp.array([True, False]))
        with pytest.raises(IndexError, match="static"):
            jax.jit(lambda x, stop: x[:stop])(x, jnp.array(1))

    def test_gradients_accumulate_into_compact_storage(self, scalar_kind):
        x = (
            _fully_broadcasted()
            if scalar_kind is _FullBroadcastScalar
            else scalar_kind(jnp.array(2.0), jnp.ones((2, 3, 4)))
        )

        def loss(x):
            result = jf.jax_sum(x * x)
            assert isinstance(result, scalar_kind)
            return result.shared

        gradient = jax.jit(jax.grad(loss))(x)
        assert gradient.shape == x.shape
        assert gradient.shared.shape == x.shared.shape
        np.testing.assert_array_equal(gradient.shared, jnp.full(x.shared.shape, 24.0 / x.shared.size))
        assert gradient.values.shape == x.values.shape

    def test_native_vmap_uses_explicit_physical_child_axes(self, scalar_kind):
        x = scalar_kind(jnp.array(2.0), jnp.arange(24.0).reshape(2, 3, 4))
        axes = scalar_kind(None, 0)
        result = jax.jit(jax.vmap(lambda row: row + 1, in_axes=(axes,)))(x)
        _assert_same(result, x + 1)
        identity = jax.vmap(lambda row: row, in_axes=(axes,), out_axes=axes)(x)
        assert identity.shared.shape == ()
        _assert_same(identity, x)
        reshaped = jax.vmap(lambda row: row.reshape(3, 1), in_axes=(axes,), out_axes=axes)(x)
        assert reshaped.shape == (2, 3, 1)


def test_all_broadcast_shape_reuses_one_value_read(monkeypatch):
    reads = []

    class Observed(_FullBroadcastScalar):
        def protected_values(self, func=None):
            reads.append(func)
            return super().protected_values(func)

    broadcast_shapes = Mock(wraps=np.broadcast_shapes)
    monkeypatch.setattr(np, "broadcast_shapes", broadcast_shapes)
    x = Observed(jnp.ones((2, 1)), jnp.ones((3, 4)))

    def extracted_shape():
        internals = jax_axis_protected_internals(x)
        assert internals is not None
        return internals.batch_shape

    for read in (lambda: x.shape, extracted_shape):
        reads.clear()
        broadcast_shapes.reset_mock()
        assert read() == (2, 3)
        assert reads == [None]
        assert broadcast_shapes.call_count == 1


def test_all_broadcast_context_preserves_unprocessed_shape():
    class Postprocessed(_FullBroadcastScalar):
        def _postprocess_protected_values(self, values, func):
            del func
            return {name: jnp.ones((5, *value.shape)) for name, value in values.items()}

    x = Postprocessed(jnp.ones((2, 1)), jnp.ones((3, 4)))
    internals = extract_axis_protected_internals(x, jf.jax_sum, check_is_permitted=True)
    assert internals is not None
    assert internals.values["shared"].shape == (5, 2, 1)
    assert internals.batch_shape == x.shape == (2, 3)


def test_all_broadcast_batch_shape_hook_is_honored():
    class Explicit(_FullBroadcastScalar):
        def _protected_batch_shape(self, values=None):
            return (8, *super()._protected_batch_shape(values))

    x = Explicit(jnp.ones((2, 1)), jnp.ones((3, 4)))
    assert x.shape == (8, 2, 3)
    internals = jax_axis_protected_internals(x)
    assert internals is not None
    assert internals.batch_shape == (8, 2, 3)


def test_explicit_validate_checks_broadcast_compatibility(example):
    x = example()
    assert x.validate() is None
    invalid = x.with_protected_values({"shared": jnp.ones((5, 3, 4))})
    with pytest.raises(ValueError, match=r"cannot broadcast|shape mismatch"):
        invalid.validate()


def test_flattening_modes_expand_broadcast_fields(example):
    x = example()
    expanded = _expanded(x)
    joined = jf.jax_concatenate((x, x), axis=None)
    assert joined.shape == (12,)
    for name in x.protected_axes:
        flat = np.asarray(getattr(expanded, name)).reshape(6, 4)
        np.testing.assert_array_equal(getattr(joined, name), np.concatenate((flat, flat)))
    indices = jnp.array([5, 0, 2])
    gathered = jf.jax_take_along_axis(x, indices, axis=None)
    assert gathered.shape == (3,)
    for name in x.protected_axes:
        flat = np.asarray(getattr(expanded, name)).reshape(6, 4)
        np.testing.assert_array_equal(getattr(gathered, name), flat[np.asarray(indices)])
