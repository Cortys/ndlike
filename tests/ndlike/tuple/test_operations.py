"""Broad operation forwarding and logical batch indexed updates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
import pytest

from .test_structural import assert_members, example


@pytest.mark.parametrize(
    "operation", [lambda x: x + 2, lambda x: 3 - x, lambda x: x * x, lambda x: -x, lambda x: x > 1]
)
def test_python_operators(backend, operation):
    x = example(backend)
    result = operation(x)
    expected = tuple(operation(np.broadcast_to(np.asarray(member), x.shape)) for member in x.members)
    assert type(result) is type(x)
    assert_members(result, expected)


def test_reductions_and_unlisted_functions(backend):
    x = example(backend)
    reduced = backend.func("sum")(x, axis=0)
    assert reduced.shape == (3,)
    assert_members(reduced, tuple(np.broadcast_to(np.asarray(member), x.shape).sum(axis=0) for member in x.members))
    squared = backend.func("pow" if backend.name == "torch" else "power")(x, 2)
    assert_members(squared, tuple(np.broadcast_to(np.asarray(member), x.shape) ** 2 for member in x.members))
    # This function was not in the old structural whitelist either.
    repeat = (
        backend.dispatch("a")(backend.xp.repeat)
        if backend.name == "jax"
        else backend.func("repeat_interleave" if backend.name == "torch" else "repeat")
    )
    repeated = repeat(x, 2)
    assert repeated.shape == (12,)


def test_multiple_results(backend):
    x = example(backend)
    quotient, remainder = divmod(x, 3)
    assert type(quotient) is type(remainder) is type(x)
    assert_members(quotient, tuple(np.broadcast_to(np.asarray(member), x.shape) // 3 for member in x.members))
    assert_members(remainder, tuple(np.broadcast_to(np.asarray(member), x.shape) % 3 for member in x.members))


def test_tuple_indices_and_masks(backend):
    x = example(backend)
    indices = backend.cls(tuple(backend.array(np.array([1, 0])) for _ in x.members))
    result = x[indices]
    assert result.shape == (2, 3)
    assert_members(result, tuple(np.broadcast_to(np.asarray(member), x.shape)[[1, 0]] for member in x.members))
    masked = x[x > 0]
    assert masked.shape == (6,)
    assert_members(masked, tuple(np.broadcast_to(np.asarray(member), x.shape).reshape(6) for member in x.members))


def test_assignment_and_output_destinations(backend):
    if backend.name == "jax":
        pytest.skip("JAX uses out-of-place .at updates instead")
    x = backend.func("broadcast_to")(example(backend), (2, 3))
    x = x.with_members(tuple(backend.func("clone" if backend.name == "torch" else "copy")(m) for m in x.members))
    old = tuple(np.asarray(member).copy() for member in x.members)
    source = backend.cls(tuple(backend.array(np.arange(3.0) + index) for index in range(3)))
    x[1] = source
    expected = []
    for member, value in zip(old, source.members, strict=True):
        expanded = np.broadcast_to(member, (2, 3)).copy()
        expanded[1] = np.asarray(value)
        expected.append(expanded)
    assert_members(x, expected)
    assert all(tuple(member.shape) == (2, 3) for member in x.members)
    # Ordinary out= and augmented assignment retain the destination owner.
    original = x
    result = backend.func("add")(x, 2, out=x)
    assert result is x
    x += 3
    assert x is original
    assert_members(x, tuple(value + 5 for value in expected))


def test_assignment_preserves_full_member_aliases(backend):
    if backend.name == "jax":
        pytest.skip("JAX arrays are immutable")
    full = backend.array(np.zeros((2, 3)))
    compact = backend.array(np.ones(3))
    x = backend.cls((full, compact))
    with pytest.raises(ValueError, match="fully expanded"):
        x[:, 1] = 7
    assert x.members[0] is full
    np.testing.assert_array_equal(full, [[0, 7, 0], [0, 7, 0]])
    np.testing.assert_array_equal(compact, np.ones(3))
    assert x.members[1] is compact


def test_member_dispatch_still_controls_permissions(backend):
    @dataclass(frozen=True)
    class Restricted(backend.base):
        data: Any
        protected_axes: ClassVar[dict[str, int]] = {"data": 0}

    x = backend.cls((Restricted(backend.array(np.ones(2))), backend.array(1.0)))
    assert x.reshape(2, 1).shape == (2, 1)
    with pytest.raises(TypeError):
        backend.func("add")(x, 1)
