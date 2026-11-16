"""Writes use original storage and native backend behavior without staging."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
import pytest

from ndlike.tuple.numpy import NumpyBroadcastTuple


@pytest.mark.parametrize("protected", [False, True])
def test_compact_reduction_outputs_use_raw_storage(backend, protected):
    if backend.name == "jax":
        pytest.skip("JAX reductions have no out destination")

    @dataclass(frozen=True)
    class Protected(backend.base):
        full: Any
        compact: Any
        protected_axes: ClassVar = {"full": 0, "compact": 0}
        broadcasted_batch: ClassVar = {"compact"}
        permitted_functions: ClassVar = {backend.func("sum")}

    def construct(first, second):
        return Protected(first, second) if protected else backend.cls((first, second))

    source = construct(backend.array(np.ones((2, 3))), backend.array(2.0))
    full, compact = backend.array(np.zeros(3)), backend.array(0.0)
    destination = construct(full, compact)
    if backend.name == "numpy":
        with pytest.raises(ValueError, match="dimensions"):
            backend.func("sum")(source, axis=0, out=destination)
        np.testing.assert_array_equal(full, [2, 2, 2])
        np.testing.assert_array_equal(compact, 0)
    else:
        with pytest.warns(UserWarning, match="resized"):
            assert backend.func("sum")(source, dim=0, out=destination) is destination
        np.testing.assert_array_equal(full, [2, 2, 2])
        np.testing.assert_array_equal(compact, [4, 4, 4])
    actual = (destination.full, destination.compact) if protected else destination.members
    assert actual[0] is full
    assert actual[1] is compact


def test_numpy_copyto_keeps_compact_storage_without_broadcasting(monkeypatch):
    first, second = np.zeros((2, 1)), np.zeros(3)
    destination: Any = NumpyBroadcastTuple((first, second))

    def no_expansion(self):
        del self
        pytest.fail("A write-only destination must not be expanded")

    monkeypatch.setattr(NumpyBroadcastTuple, "_expanded_members", no_expansion)
    assert np.copyto(dst=destination, src=9) is None
    assert destination.members[0] is first
    assert destination.members[1] is second
    np.testing.assert_array_equal(first, np.full((2, 1), 9))
    np.testing.assert_array_equal(second, np.full(3, 9))


def test_numpy_copyto_allows_partial_writes():
    first, second = np.zeros(3), np.zeros(3, dtype=int)
    destination: Any = NumpyBroadcastTuple((first, second))
    with pytest.raises(TypeError, match="cast"):
        np.copyto(destination, 1.5)
    np.testing.assert_array_equal(first, np.full(3, 1.5))
    np.testing.assert_array_equal(second, np.zeros(3, dtype=int))


@pytest.mark.parametrize(
    "operation",
    [
        lambda x: np.put(x, [0, 2], [7, 8]),
        lambda x: np.putmask(x, [[True, False, False], [False, True, False]], 7),
        lambda x: np.place(x, [[True, False, False], [False, True, False]], [7, 8]),
        lambda x: np.fill_diagonal(x, 7),
        lambda x: np.put_along_axis(x, np.array([[0], [1]]), 7, axis=1),
    ],
)
def test_numpy_indexed_mutators_forward_to_original_members(operation):
    members = (np.zeros((2, 3)), np.ones((2, 3)))
    expected = tuple(value.copy() for value in members)
    for value in expected:
        operation(value)
    destination: Any = NumpyBroadcastTuple(members)
    assert operation(destination) is None
    for original, actual, reference in zip(members, destination.members, expected, strict=True):
        assert actual is original
        np.testing.assert_array_equal(actual, reference)


def test_numpy_put_rejects_compact_indexing_after_earlier_writes():
    first, compact = np.zeros((2, 3)), np.zeros(3)
    destination: Any = NumpyBroadcastTuple((first, compact))
    with pytest.raises(ValueError, match="fully expanded"):
        np.put(destination, [0], 9)
    np.testing.assert_array_equal(first, [[9, 0, 0], [0, 0, 0]])
    np.testing.assert_array_equal(compact, np.zeros(3))
