"""Flattening modes operate on visible batches, preserving protected suffixes."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
import math
from typing import Any, ClassVar

import numpy as np
import pytest


@pytest.fixture(params=["numpy", "jax"])
def backend(request):
    name = request.param
    pytest.importorskip(name)
    arrays = import_module("jax.numpy" if name == "jax" else name)
    module = import_module(f"ndlike.protected_axis.{name}")
    concat = np.concatenate if name == "numpy" else import_module("ndlike.jax_functions").jax_concatenate
    return arrays, getattr(module, f"{name.title()}AxisProtected"), concat


@pytest.mark.parametrize("batches", [((2, 3), (4,)), ((), ()), ((0, 3), (2,)), ((2,), (0,))])
@pytest.mark.parametrize("suffix", [(), (4,), (3, 4), (0, 4)])
def test_concatenate_none_flattens_batch_dimensions(backend, batches, suffix):
    arrays, base, concatenate = backend

    @dataclass
    class Pair(base):
        values: Any
        weights: Any
        protected_axes: ClassVar[dict[str, int]] = {"values": len(suffix), "weights": 0}

    xs = [
        Pair(
            arrays.arange(math.prod((*batch, *suffix))).reshape((*batch, *suffix)),
            arrays.arange(math.prod(batch)).reshape(batch),
        )
        for batch in batches
    ]
    result = concatenate(xs, axis=None)
    size = sum(math.prod(batch) for batch in batches)
    assert result.shape == (size,)
    assert result.values.shape == (size, *suffix)
    assert result.weights.shape == (size,)
    assert result.validate() is None
    expected = np.concatenate(
        [np.asarray(x.values).reshape((math.prod(batch), *suffix)) for x, batch in zip(xs, batches, strict=True)],
        axis=0,
    )
    np.testing.assert_array_equal(result.values, expected)
    np.testing.assert_array_equal(result.weights, np.concatenate([np.asarray(x.weights).reshape(-1) for x in xs]))


def test_numpy_concatenate_none_supports_positional_axis_and_out():
    from ndlike.protected_axis.numpy import NumpyAxisProtected  # noqa: PLC0415

    @dataclass
    class Matrix(NumpyAxisProtected[np.ndarray]):
        values: np.ndarray
        protected_axes: ClassVar[dict[str, int]] = {"values": 2}

    x = Matrix(np.arange(24).reshape(2, 3, 4))
    expected = np.concatenate((x.values, x.values), axis=0)
    out = Matrix(np.empty((4, 3, 4), dtype=np.int64))
    assert np.concatenate((x, x), None, out) is out  # ty:ignore[no-matching-overload]
    np.testing.assert_array_equal(out.values, expected)
    raw_out = np.empty_like(out.values)
    assert np.concatenate((x, x), axis=None, out=raw_out) is raw_out
    np.testing.assert_array_equal(raw_out, expected)
    converted = np.concatenate((x, x), axis=None, dtype=np.float32, casting="unsafe")
    assert isinstance(converted, Matrix)
    assert converted.values.dtype == np.float32
    np.testing.assert_array_equal(converted.values, expected)


def test_jax_flattening_modes_work_under_jit():
    jax = pytest.importorskip("jax")
    from jax import numpy as jnp  # noqa: PLC0415

    from ndlike.jax_functions import jax_concatenate, jax_take_along_axis  # noqa: PLC0415
    from ndlike.protected_axis.jax import JaxAxisProtected  # noqa: PLC0415

    @dataclass
    class Matrix(JaxAxisProtected[Any]):
        values: Any
        protected_axes: ClassVar[dict[str, int]] = {"values": 2}

    x = Matrix(jnp.arange(120).reshape(2, 3, 4, 5))
    y = Matrix(jnp.arange(80).reshape(4, 4, 5))
    joined = jax.jit(lambda a, b: jax_concatenate((a, b), axis=None))(x, y)
    assert joined.shape == (10,)
    np.testing.assert_array_equal(joined.values, jnp.concatenate((x.values.reshape(6, 4, 5), y.values)))
    indices = jnp.array([5, 0, 2])
    gathered = jax.jit(lambda a, i: jax_take_along_axis(a, i, axis=None))(x, indices)
    assert gathered.shape == (3,)
    np.testing.assert_array_equal(gathered.values, x.values.reshape(6, 4, 5)[indices])
    with pytest.raises(ValueError, match="same ndim"):
        jax_take_along_axis(x, indices.reshape(1, 3), axis=None)
