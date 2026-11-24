"""NumPy member typing and opaque object-array indexing."""

from __future__ import annotations

from typing import Any, Never, assert_type, cast

import numpy as np
import pytest

from ndlike.tuple.numpy import NumpyBroadcastTuple, NumpyMember


def test_concrete_shape_result_typing():
    x: NumpyBroadcastTuple[tuple[np.ndarray, ...]] = NumpyBroadcastTuple((np.ones((2, 1)), np.ones(3)))
    assert_type(x.reshape(6), NumpyBroadcastTuple[tuple[NumpyMember, ...]])
    assert_type(x[0], NumpyBroadcastTuple[tuple[NumpyMember, ...]])
    assert_type(x.T, NumpyBroadcastTuple[tuple[NumpyMember, ...]])


def test_scalar_member_becomes_an_array_on_reshape():
    value = NumpyBroadcastTuple((np.float64(2), np.arange(3)))
    result = value.reshape(3, 1)
    assert_type(result, NumpyBroadcastTuple[tuple[NumpyMember, ...]])
    assert isinstance(result.members[0], np.ndarray)
    np.testing.assert_array_equal(result.members[0], np.full((3, 1), 2.0))
    assert isinstance(value.members[0], np.float64)


def test_object_array_scalar_does_not_coerce_or_evaluate():
    class Opaque:
        def __array__(self, *args: object, **kwargs: object) -> Never:
            del args, kwargs
            pytest.fail("Object member must not be converted")

        def __call__(self) -> Never:
            pytest.fail("Object member must not be evaluated")

    token = Opaque()
    member = np.empty((2, 1), dtype=object)
    member.fill(token)
    x = NumpyBroadcastTuple((member, np.arange(3)))
    indexed = x[1, 2]
    assert indexed.shape == ()
    scalar = indexed.members[0]
    assert isinstance(scalar, np.ndarray)
    assert scalar.shape == ()
    assert scalar.item() is token


def test_output_destinations_are_rejected_before_writing():
    x = NumpyBroadcastTuple((np.ones(2), np.zeros(2)))
    out = np.full(4, 9.0)
    with pytest.raises(TypeError, match="Output destinations"):
        np.concatenate((x, x), out=out)
    with pytest.raises(TypeError, match="Output destinations"):
        np.concatenate((x, x), 0, out)
    np.testing.assert_array_equal(out, np.full(4, 9.0))


def test_ufunc_methods_and_multi_output_destinations():
    x = NumpyBroadcastTuple((np.array([[1.0], [2.0]]), np.arange(3.0)))
    reduced = np.add.reduce(x, axis=0)
    assert reduced.shape == (3,)
    np.testing.assert_array_equal(reduced.members[0], np.full(3, 3.0))
    out = NumpyBroadcastTuple((np.zeros(3), np.zeros(3)))
    assert np.add.reduce((x), axis=0, out=cast("Any", out)) is out
    np.testing.assert_array_equal(out.members[0], np.full(3, 3.0))
    assert np.add.accumulate(x, axis=0).shape == (2, 3)
    with pytest.raises(ValueError, match="fully expanded"):
        np.add.at(x, (0, 1), 4.0)
    x = NumpyBroadcastTuple(tuple(np.broadcast_to(m, x.shape).copy() for m in x.members))
    np.add.at(x, (0, 1), 4.0)
    np.testing.assert_array_equal(x.members[0], [[1.0, 5.0, 1.0], [2.0, 2.0, 2.0]])
    fraction, whole = np.modf(x / 2)
    result = np.modf(x / 2, out=(fraction, whole))
    assert result[0] is fraction
    assert result[1] is whole
    mixed = cast("Any", np.modf)(x / 2, out=(None, whole))
    assert isinstance(mixed[0], NumpyBroadcastTuple)
    assert mixed[1] is whole
    # Native positional output arguments remain output destinations.
    destination = NumpyBroadcastTuple((np.zeros((4, 3)), np.zeros((4, 3))))
    assert np.concatenate(((x, x)), 0, cast("Any", destination)) is destination


def test_named_results_and_non_array_results():
    x = NumpyBroadcastTuple((np.eye(2), np.eye(2) * 2))
    result = np.linalg.slogdet(x)
    assert isinstance(result.sign, NumpyBroadcastTuple)
    assert isinstance(result.logabsdet, NumpyBroadcastTuple)
    assert np.array_equal(x, x) == (True, True)
