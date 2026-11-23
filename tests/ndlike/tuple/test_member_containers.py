"""Member containers are fixed-length tuples reconstructed through their constructors."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, NamedTuple, Self

import numpy as np
import pytest

from ndlike.tuple._common import tuple_function


class Pair(NamedTuple):
    first: Any
    second: Any


class OtherPair(NamedTuple):
    left: Any
    right: Any


class SequenceTuple(tuple):
    __slots__ = ()

    def __new__(cls, values: Sequence) -> Self:
        # Require an actual sequence, not just an iterable/generator.
        assert len(values) == 2
        return super().__new__(cls, (values[0], values[1]))


@pytest.fixture(params=[Pair, SequenceTuple])
def source(backend, request):
    values = (backend.array(np.array([[1.0], [2.0]])), backend.array(np.array([3.0, 4.0, 5.0])))
    members = Pair(*values) if request.param is Pair else request.param(values)
    return backend.cls(members)


def test_shaping_indexing_and_conversion_preserve_member_container(backend, source):
    results = [
        source[1],
        source[(slice(None), 1)],
        source[backend.array([1, 0])],
        source.reshape(6),
        source.T,
        backend.func("broadcast_to")(source, (4, 2, 3)),
        -source,
    ]
    hook = "__array_like__" if backend.name == "numpy" else f"__{backend.name}_like__"
    results.append(getattr(source, hook)(copy=True))
    results.append(source.__array_like__())
    for result in results:
        assert type(result.members) is type(source.members)
        assert len(result.members) == 2
    np.testing.assert_array_equal(results[3].members[0], [1, 1, 1, 2, 2, 2])
    np.testing.assert_array_equal(results[3].members[1], [3, 4, 5, 3, 4, 5])
    assert tuple(member.shape for member in results[-1].members) == ((2, 1), (3,))


def test_binary_and_multiple_outputs_preserve_first_member_container(backend, source):
    other = backend.cls(OtherPair(backend.array(10.0), backend.array(20.0)))
    result = source + other
    reverse = other + source
    assert type(result.members) is type(source.members)
    assert type(reverse.members) is OtherPair
    np.testing.assert_array_equal(result.members[0], [[11, 11, 11], [12, 12, 12]])
    np.testing.assert_array_equal(reverse.members.right, [[23, 24, 25], [23, 24, 25]])
    first, second = backend.xp.frexp(source) if backend.name == "torch" else divmod(source, 2)
    assert type(first.members) is type(second.members) is type(source.members)


def test_mutations_preserve_destination_container(backend, source):
    container_type = type(source.members)
    if backend.name == "jax":
        other = backend.cls(OtherPair(backend.array(10.0), backend.array(20.0)))
        result = source.at[1].add(other)
        assert type(result.members) is type(source.members)
        np.testing.assert_array_equal(result.members[0], [[1, 1, 1], [12, 12, 12]])
    else:
        destination = backend.cls(OtherPair(backend.array(np.zeros((2, 3))), backend.array(np.zeros((2, 3)))))
        full = destination.members.right
        assert backend.func("add")(source, 1, out=destination) is destination
        assert type(destination.members) is OtherPair
        assert destination.members.right is full
        np.testing.assert_array_equal(destination.members.left, [[2, 2, 2], [3, 3, 3]])
        with pytest.raises(ValueError, match="fully expanded"):
            source[1] = destination[0]
        assert type(source.members) is type(source[1].members) is container_type
        np.testing.assert_array_equal(source.members[0], [[1], [2]])


def test_namedtuple_arguments_are_rebuilt_positionally(source):
    def add_pair(pair):
        assert type(pair) is OtherPair
        return pair.left + pair.right

    result = tuple_function(type(source), add_pair, (OtherPair(source, source),), {})
    assert type(result.members) is type(source.members)
    np.testing.assert_array_equal(result.members[0], [[2, 2, 2], [4, 4, 4]])
