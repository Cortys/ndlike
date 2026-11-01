"""Shared batch storage belongs to protected fields, not a separate array wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar
from unittest.mock import Mock

import numpy as np
import pytest

pytest.importorskip("jax")
import jax
import jax.numpy as jnp

from ndlike import jax_functions as jf
from ndlike.protected_axis import jax as protected_jax
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
class _Shared(JaxAxisProtected[Any]):
    # Deliberately put the broadcastable field first in the layout.
    shared: Any
    values: Any
    protected_axes: ClassVar[dict[str, int]] = {"shared": 1, "values": 1}
    broadcasted_batch: ClassVar[set[str]] = {"shared"}
    permitted_functions: ClassVar[set] = _PERMITTED
    permitted_at_methods: ClassVar[set[JaxAtMethod]] = {"set", "add", "multiply"}


@dataclass(frozen=True)
class _SharedScalar(JaxAxisProtected[Any]):
    shared: Any
    values: Any
    protected_axes: ClassVar[dict[str, int]] = {"shared": 0, "values": 1}
    broadcasted_batch: ClassVar[set[str]] = {"shared"}
    permitted_functions: ClassVar[set] = _PERMITTED
    permitted_at_methods: ClassVar[set[JaxAtMethod]] = {"set", "add"}


@dataclass(frozen=True)
class _SharedMatrix(JaxAxisProtected[Any]):
    shared: Any
    values: Any
    protected_axes: ClassVar[dict[str, int]] = {"shared": 2, "values": 1}
    broadcasted_batch: ClassVar[set[str]] = {"shared"}


_SHAPE_OPERATIONS = [
    pytest.param(lambda x: jf.jax_reshape(x, (3, 2)), id="reshape"),
    pytest.param(jf.jax_transpose, id="transpose"),
    pytest.param(lambda x: jf.jax_transpose(x, (2, 0, 1)), id="transpose-axes"),
    pytest.param(jf.jax_matrix_transpose, id="matrix-transpose"),
    pytest.param(lambda x: jf.jax_swapaxes(x, 0, -1), id="swapaxes"),
    pytest.param(lambda x: jf.jax_moveaxis(x, (0, 2), (2, 0)), id="moveaxis"),
    pytest.param(lambda x: jf.jax_expand_dims(x, -1), id="expand-dims"),
    pytest.param(lambda x: jf.jax_squeeze(x, 1), id="squeeze-axis"),
    pytest.param(jf.jax_squeeze, id="squeeze"),
    pytest.param(lambda x: jf.jax_broadcast_to(x, (5, 2, 1, 3)), id="broadcast-to"),
]


def _example(batch=(2, 3), compact=(1, 3)) -> _Shared:
    return _Shared(
        jnp.arange(np.prod((*compact, 4)), dtype=float).reshape((*compact, 4)),
        jnp.arange(np.prod((*batch, 4)), dtype=float).reshape((*batch, 4)),
    )


def _expanded(x):
    axes = type(x).protected_axes["shared"]
    suffix = x.shared.shape[-axes:] if axes else ()
    return type(x)(jnp.broadcast_to(x.shared, (*x.shape, *suffix)), x.values)


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


@pytest.mark.parametrize("marked", [False, True])
@pytest.mark.parametrize("index", [(slice(None), 1), (slice(None), np.array([2, 0]))])
def test_full_fields_skip_broadcast_index_helpers(marked, index, index_helpers):
    @dataclass(frozen=True)
    class Pair(JaxAxisProtected[Any]):
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
def test_gather_classifies_once_for_multiple_compact_fields(index, basic, index_helpers):
    @dataclass(frozen=True)
    class MultipleShared(JaxAxisProtected[Any]):
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


def test_scatter_materializes_only_compact_destinations(index_helpers):
    x = _example(compact=())
    result = x.at[:, 1].add((jnp.ones(4), jnp.ones(4)))
    assert isinstance(result, _Shared)
    np.testing.assert_array_equal(result.shared, _expanded(x).shared.at[:, 1].add(jnp.ones(4)))
    index_helpers["_is_basic_index"].assert_not_called()
    index_helpers["_normalize_broadcast_index"].assert_not_called()
    index_helpers["_broadcast_batch"].assert_called_once()


@pytest.mark.parametrize("compact", [(), (1,), (3,), (1, 3), (2, 1), (2, 3)])
def test_layout_and_extraction(compact):
    x = _example(compact=compact)
    assert x.primary_protected_name == "values"
    assert _Shared.primary_protected_name == "values"
    assert x.shape == (2, 3)
    assert x.protected_values()["shared"] is x.shared
    internals = jax_axis_protected_internals(x)
    assert internals is not None
    assert internals.primary_name == "values"
    assert internals.batch_shape == x.shape
    assert internals.broadcasted_batch == frozenset({"shared"})
    assert internals.values["values"] is x.values
    np.testing.assert_array_equal(internals.values["shared"], _expanded(x).shared)
    assert x.shared.shape == (*compact, 4)


@pytest.mark.parametrize("slots", [False, True])
def test_primary_name_is_precomputed_for_each_subclass(slots):
    @dataclass(frozen=True, slots=slots)
    class OtherShared(_Shared):
        shared: Any
        values: Any
        broadcasted_batch: ClassVar[set[str]] = {"values"}

    assert OtherShared.primary_protected_name == "shared"
    assert _Shared.primary_protected_name == "values"
    x = OtherShared(jnp.ones((2, 3, 4)), jnp.ones(4))
    assert x.shape == (2, 3)
    internals = jax_axis_protected_internals(x)
    assert internals is not None
    assert internals.primary_name == "shared"


def test_runtime_operations_reuse_precomputed_primary_name(monkeypatch):
    def unexpected_selection(*_args: object, **_kwargs: object):
        pytest.fail("Primary-field selection must only run when defining the class.")

    monkeypatch.setattr("ndlike.protected_axis.jax.primary_protected_name", unexpected_selection)
    monkeypatch.setattr("ndlike.protected_axis._common_functions.primary_protected_name", unexpected_selection)
    x = _example(compact=())
    assert x.primary_protected_name == "values"
    assert x.shape == (2, 3)
    assert x[0].shape == (3,)
    assert x.reshape(6).shape == (6,)
    assert (x + 1).shape == x.shape
    assert jf.jax_conj(x).shape == x.shape
    assert jf.jax_sum(x).shape == ()
    assert jf.jax_stack((x, x)).shape == (2, 2, 3)
    assert jf.jax_concatenate((x, x)).shape == (4, 3)


@pytest.mark.parametrize("compact", [(4,), (3, 1), (1, 1, 1)])
def test_invalid_broadcast_shape(compact):
    with pytest.raises(ValueError, match="cannot broadcast to primary batch shape"):
        _example(compact=compact).protected_values()


@pytest.mark.parametrize("declaration", [{"missing"}, {"shared", "values"}, ["shared"], {1}])
def test_invalid_broadcast_declaration(declaration):
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


def test_unmarked_fields_still_require_identical_shapes():
    @dataclass(frozen=True)
    class Pair(JaxAxisProtected[Any]):
        shared: Any
        first: Any
        second: Any
        protected_axes: ClassVar[dict[str, int]] = {"shared": 0, "first": 0, "second": 0}
        broadcasted_batch: ClassVar[set[str]] = {"shared"}

    with pytest.raises(ValueError, match="inconsistent batch-shapes"):
        Pair(jnp.array(1), jnp.ones(2), jnp.ones(3)).protected_values()


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
def test_basic_indexing_preserves_compact_storage(compact, index):
    x = _example(compact=compact)
    result = x[index]
    assert isinstance(result, _Shared)
    _assert_same(result, _expanded(x)[index])
    assert result.shared.size <= x.shared.size
    assert x.shared.shape == (*compact, 4)


def test_basic_indexing_respects_logical_singleton_slices():
    x = _example(batch=(1, 3), compact=(1, 3))
    result = x[:0]
    assert isinstance(result, _Shared)
    assert result.shape == (0, 3)
    assert result.shared.shape == (0, 3, 4)


@pytest.mark.parametrize("index", [(Ellipsis, Ellipsis), (0, 0, 0)])
def test_basic_indexing_rejects_invalid_batch_indices(index):
    with pytest.raises(IndexError):
        _example()[index]


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
def test_advanced_indexing_materializes(compact, index):
    x = _example(compact=compact)
    result = x[index]
    expected = _expanded(x)[index]
    assert isinstance(result, _Shared)
    _assert_same(result, expected)
    assert result.shared.shape == expected.shared.shape


@pytest.mark.parametrize("function", [jf.jax_copy, jf.jax_conj, lambda x: jf.jax_astype(x, jnp.complex64)])
@pytest.mark.parametrize("compact", [(), (3,), (2, 1)])
def test_value_only_preserves_compact_shapes(function, compact):
    x = _example(compact=compact)
    x = _Shared(x.shared + 1j, x.values + 2j)
    result = function(x)
    assert result.shared.shape == x.shared.shape
    _assert_same(result, function(_expanded(x)))
    assert x.astype(jnp.complex64).shared.shape == x.shared.shape


@pytest.mark.parametrize("compact", [(), (1,), (1, 1)])
@pytest.mark.parametrize("shape", [(6,), (-1, 2), (1, 2, 3), (None, 6)])
@pytest.mark.parametrize("order", ["C", "F"])
def test_reshape_reuses_single_entry_fields(compact, shape, order):
    x = _example(compact=compact)
    result = jf.jax_reshape(x, shape, order=order)
    assert isinstance(result, _Shared)
    assert result.shared.shape == (4,)
    _assert_same(result, jf.jax_reshape(_expanded(x), shape, order=order))


def test_scalar_reshape_and_invalid_target():
    x = _SharedScalar(jnp.array(2.0), jnp.ones((2, 3, 4)))
    result = x.reshape(6)
    assert isinstance(result, _SharedScalar)
    assert result.shared.shape == ()
    assert result.shape == (6,)
    with pytest.raises(TypeError, match="reshape"):
        x.reshape(5)


def test_reshape_materializes_partially_shared_fields():
    x = _example(compact=(1, 3))
    result = x.reshape(6)
    assert isinstance(result, _Shared)
    assert result.shared.shape == (6, 4)
    _assert_same(result, _expanded(x).reshape(6))


@pytest.mark.parametrize("function", _SHAPE_OPERATIONS)
@pytest.mark.parametrize("compact", [(), (1,), (1, 1, 1)])
@pytest.mark.parametrize(("kind", "suffix"), [(_SharedScalar, ()), (_Shared, (4,)), (_SharedMatrix, (2, 3))])
def test_shape_only_operations_reuse_single_entry_fields(function, compact, kind, suffix):
    shared = jnp.arange(np.prod((*compact, *suffix)), dtype=float).reshape((*compact, *suffix))
    x = kind(shared, jnp.arange(24.0).reshape(2, 1, 3, 4))
    result = function(x)
    assert result.shared.shape == suffix
    _assert_same(result, function(_expanded(x)))
    assert x.shared is shared
    assert x.shared.shape == (*compact, *suffix)


@pytest.mark.parametrize("function", _SHAPE_OPERATIONS)
def test_shape_only_operations_keep_single_entry_fields_compact_under_jit(function):
    x = _example(batch=(2, 1, 3), compact=(1, 1, 1))
    result = jax.jit(function)(x)
    assert result.shared.shape == (4,)
    _assert_same(result, function(_expanded(x)))


@pytest.mark.parametrize("function", _SHAPE_OPERATIONS)
def test_shape_only_operations_materialize_partially_shared_fields(function):
    x = _example(batch=(2, 1, 3), compact=(1, 1, 3))
    result = function(x)
    expected = function(_expanded(x))
    _assert_same(result, expected)
    assert result.shared.shape == expected.shared.shape


@pytest.mark.parametrize("function", _SHAPE_OPERATIONS)
@pytest.mark.parametrize("compact", [(), (1,), (1, 1, 1)])
def test_shape_only_operations_preserve_numpy_sidecars(function, compact):
    shared = np.full((*compact, 2), "shared", dtype=object)
    x = _Shared(shared, jnp.ones((2, 1, 3, 4)))
    result = function(x)
    assert isinstance(result.shared, np.ndarray)
    assert result.shared.shape == (2,)
    np.testing.assert_array_equal(result.shared, ["shared", "shared"])
    assert x.shared is shared


@pytest.mark.parametrize("compact", [(), (1,), (1, 1, 1)])
@pytest.mark.parametrize("backend", ["jax", "numpy"])
@pytest.mark.parametrize("copy", [None, False, True])
def test_reshape_copy_preserves_single_entry_storage(compact, backend, copy):
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
def test_shape_only_shortcuts_still_validate_primary_shape(function):
    x = _example(batch=(2, 1, 3), compact=(1, 1, 1))
    with pytest.raises((TypeError, ValueError)):
        function(x)


def test_matrix_transpose_shortcut_still_requires_two_batch_axes():
    with pytest.raises(ValueError, match="at least 2 batch dimensions"):
        jf.jax_matrix_transpose(_example(batch=(6,), compact=()))


def test_nested_single_entry_fields_survive_shape_only_operations():
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
def test_reshape_nested_single_entry_numpy_sidecar(compact, order, copy):
    @dataclass(frozen=True)
    class Mixed(JaxAxisProtected[Any]):
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
def test_other_operations_receive_materialized_values(function):
    x = _example(compact=())
    result = function(x)
    expected = function(_expanded(x))
    _assert_same(result, expected)
    assert result.shared.shape == expected.shared.shape
    assert x.shared.shape == (4,)


@pytest.mark.parametrize("function", [jf.jax_sum, jf.jax_mean, jf.jax_average, jf.jax_std, jf.jax_var])
@pytest.mark.parametrize("axis", [None, 0, 1, (0, 1), ()])
@pytest.mark.parametrize("keepdims", [False, True])
def test_reductions_use_logical_extents(function, axis, keepdims):
    x = _example(compact=(1, 3))
    result = function(x, axis=axis, keepdims=keepdims)
    _assert_same(result, function(_expanded(x), axis=axis, keepdims=keepdims))


def test_weighted_reduction_and_comparison():
    x = _example(compact=(2, 1))
    weights = jnp.arange(6.0).reshape(2, 3) + 1
    _assert_same(jf.jax_average(x, weights=weights), jf.jax_average(_expanded(x), weights=weights))
    np.testing.assert_array_equal(jf.jax_equal(x, _expanded(x)), jnp.ones(x.shape, dtype=bool))


def test_sequence_inputs_expand_to_their_own_logical_shapes():
    x = _example(batch=(2, 3), compact=(1, 3))
    y = _example(batch=(5, 3), compact=(5, 1))
    result = jf.jax_concatenate((x, y), axis=0)
    assert isinstance(result, _Shared)
    assert result.shared.shape == (7, 3, 4)
    _assert_same(result, jf.jax_concatenate((_expanded(x), _expanded(y)), axis=0))

    y = _example(compact=(2, 1))
    result = jf.jax_stack((x, y), axis=-1)
    _assert_same(result, jf.jax_stack((_expanded(x), _expanded(y)), axis=-1))


def test_shared_field_cannot_enlarge_primary_shape_or_broadcast_protected_axes():
    with pytest.raises(ValueError, match="cannot broadcast to primary batch shape"):
        _example(batch=(1, 3), compact=(2, 3)).protected_values()
    with pytest.raises(ValueError, match="expected >= 1"):
        _Shared(jnp.array(2.0), jnp.ones((2, 3, 4))).protected_values()


def test_point_and_empty_batches():
    point = _example(batch=(), compact=())
    indexed = point[None]
    assert isinstance(indexed, _Shared)
    assert indexed.shape == (1,)
    assert indexed.shared.shape == (1, 4)
    reshaped = point.reshape(1, 1)
    assert isinstance(reshaped, _Shared)
    assert reshaped.shared.shape == (4,)
    _assert_same(reshaped, _expanded(point).reshape(1, 1))

    empty = _example(batch=(0, 3), compact=())
    _assert_same(empty[:], _expanded(empty)[:])
    _assert_same(empty.reshape(0), _expanded(empty).reshape(0))
    _assert_same(jf.jax_sum(empty), jf.jax_sum(_expanded(empty)))


def test_matmul_materializes_shared_fields():
    x = _SharedScalar(jnp.array(2.0), jnp.ones((2, 3, 4)))
    y = _SharedScalar(jnp.array(3.0), jnp.ones((3, 5, 4)))
    result = x @ y
    assert result.shared.shape == (2, 5)
    np.testing.assert_array_equal(result.shared, jnp.full((2, 5), 18.0))


@pytest.mark.parametrize("method", ["set", "add", "multiply"])
@pytest.mark.parametrize("index", [(1, 2), (slice(None), 1), np.array([True, False])])
def test_indexed_updates_materialize_destinations(method, index):
    x = _example(compact=())
    candidate = (jnp.full((4,), 7.0), jnp.full((4,), 8.0))
    result = getattr(x.at[index], method)(candidate)
    expected = getattr(_expanded(x).at[index], method)(candidate)
    _assert_same(result, expected)
    assert result.shared.shape == (2, 3, 4)
    assert x.shared.shape == (4,)


def test_numpy_sidecars_remain_numpy_and_compact_when_supported():
    x = _SharedScalar(np.array("shared", dtype=object), jnp.ones((2, 3, 4)))
    indexed = x[1]
    reshaped = x.reshape(6)
    assert isinstance(indexed, _SharedScalar)
    assert isinstance(reshaped, _SharedScalar)
    assert indexed.shared.shape == ()
    assert reshaped.shared.shape == ()
    expanded = jf.jax_broadcast_to(x, (5, 2, 3))
    assert isinstance(expanded, _SharedScalar)
    assert isinstance(expanded.shared, np.ndarray)
    assert expanded.shape == (5, 2, 3)
    assert expanded.shared is x.shared
    result = x.at[0].set((np.array("changed", dtype=object), jnp.zeros(4)))
    assert isinstance(result, _SharedScalar)
    np.testing.assert_array_equal(result.shared, [["changed"] * 3, ["shared"] * 3])
    with pytest.raises(TypeError, match="NumPy array"):
        jax.jit(lambda x: x)(x)


def test_nested_protected_fields_preserve_custom_dispatch():
    inner = _SharedScalar(jnp.array(2.0), jnp.ones(4))
    x = _SharedScalar(inner, jnp.ones((2, 3, 4)))
    result = x + 1
    assert isinstance(result.shared, _SharedScalar)
    assert result.shared.shape == x.shape
    np.testing.assert_array_equal(result.shared.shared, jnp.full(x.shape, 3.0))
    np.testing.assert_array_equal(result.shared.values, jnp.full((2, 3, 4), 2.0))
    indexed = x[1, 2]
    assert isinstance(indexed, _SharedScalar)
    assert isinstance(indexed.shared, _SharedScalar)
    assert indexed.shared.shape == ()


def test_pytree_and_jit_preserve_actual_storage():
    x = _example(compact=())
    leaves = jax.tree.leaves(x)
    assert leaves[0] is x.shared
    assert leaves[1] is x.values
    assert jax.jit(lambda x: x)(x).shared.shape == (4,)
    assert jax.jit(lambda x: x[:1])(x).shared.shape == (4,)
    assert jax.jit(lambda x: x.reshape(6))(x).shared.shape == (4,)
    assert jax.jit(jf.jax_conj)(x).shared.shape == (4,)
    result = jax.jit(lambda x: x + 1)(x)
    assert result.shared.shape == (2, 3, 4)
    _assert_same(result, x + 1)
    assert x.shared.shape == (4,)


def test_jit_indexed_updates_leave_unselected_entries_unchanged():
    x = _SharedScalar(jnp.array(2.0), jnp.ones((2, 3, 4)))
    result = jax.jit(lambda x, i: x.at[i].add((3.0, jnp.ones(4))))(x, jnp.array(1))
    np.testing.assert_array_equal(result.shared, [[2.0] * 3, [5.0] * 3])
    assert x.shared.shape == ()


def test_jit_basic_dynamic_integer_and_advanced_indices():
    x = _example(compact=(2, 1))
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


def test_gradients_accumulate_into_compact_storage():
    x = _SharedScalar(jnp.array(2.0), jnp.ones((2, 3, 4)))

    def loss(x):
        result = jf.jax_sum(x * x)
        assert isinstance(result, _SharedScalar)
        return result.shared

    gradient = jax.jit(jax.grad(loss))(x)
    assert gradient.shared.shape == ()
    assert float(gradient.shared) == 24.0
    assert gradient.values.shape == x.values.shape


def test_native_vmap_uses_explicit_physical_child_axes():
    x = _SharedScalar(jnp.array(2.0), jnp.arange(24.0).reshape(2, 3, 4))
    axes = _SharedScalar(None, 0)
    result = jax.jit(jax.vmap(lambda row: row + 1, in_axes=(axes,)))(x)
    _assert_same(result, x + 1)
    identity = jax.vmap(lambda row: row, in_axes=(axes,), out_axes=axes)(x)
    assert identity.shared.shape == ()
    _assert_same(identity, x)
