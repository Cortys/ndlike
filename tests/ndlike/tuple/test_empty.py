"""Empty member collections are neutral placeholders, independent of batch shape."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, NamedTuple

import numpy as np
import pytest

from ndlike.array_like import to_numpy_like
from ndlike.tuple._common import tuple_function
from ndlike.tuple.numpy import NumpyBroadcastTuple

from .test_conversion import converter


def assert_empty(value, cls):
    assert type(value) is cls
    assert value.members == ()
    assert value.shape == ()
    assert value.ndim == 0
    assert value.device is None


def test_metadata(backend):
    value = backend.cls(())
    assert_empty(value, backend.cls)
    assert value.__array_namespace__() is backend.xp
    if backend.name == "numpy":
        assert value.dtype == np.dtype(object)
        assert value.flags.writeable
        assert value.flags.c_contiguous
    else:
        assert value.dtype is None
    if backend.name == "jax":
        assert value.sharding is None
        assert value.devices() == set()
    if backend.name == "torch":
        assert value.size() == ()
        assert backend.xp.numel(value) == 1
    else:
        assert value.size == 1
        for name, expected in (("shape", ()), ("ndim", 0), ("size", 1)):
            func = getattr(backend.xp, name)
            if backend.name == "jax":
                func = backend.dispatch("a")(func)
            assert func(value) == expected
    with pytest.raises(TypeError, match="unsized"):
        len(value)


@pytest.mark.parametrize(
    "index",
    [
        42,
        (0, 1, 2),
        slice(None, None, 0),
        (None, Ellipsis, None),
        (Ellipsis, Ellipsis),
        np.array([3, 1]),
        np.array([True, False]),
        (np.array([0, 2]), np.array([1, 3])),
    ],
)
def test_all_indices_are_neutral(backend, index):
    value = backend.cls(())
    assert_empty(value[index], backend.cls)
    # There are no members to pair with an index representation either.
    assert_empty(value[backend.cls((backend.array([1, 3]),))], backend.cls)


def test_shaping_and_operators_are_neutral(backend):
    value = backend.cls(())
    results = [
        value.reshape(2, 3),
        value.T,
        value.mT,
        value.squeeze(),
        value + 1,
        value + value,
        value * backend.array(np.ones((2, 3))),
        backend.func("broadcast_to")(value, (2, 3)),
        backend.func("stack")((value, value)),
        backend.func("concatenate")((value, value)),
        backend.func("sum")(value, axis=3),
    ]
    if backend.name == "torch":
        results.extend((value.mH, backend.func("permute")(value, (2, 0, 1))))
    else:
        results.append(backend.func("transpose")(value, (2, 0, 1)))
    for result in results:
        assert_empty(result, backend.cls)
    with pytest.raises(ValueError, match="same number"):
        _ = value + backend.cls((backend.array(1),))


def test_forwarding_does_not_evaluate_function(backend):
    value = backend.cls(())

    def forbidden(*args: object, **kwargs: object):
        del args, kwargs
        pytest.fail("No operation should run without members")

    with pytest.raises(NotImplementedError, match="known result structure"):
        tuple_function(backend.cls, forbidden, (value,), {})
    assert_empty(tuple_function(backend.cls, forbidden, (value,), {}, nout=1), backend.cls)
    assert tuple_function(backend.cls, forbidden, (value,), {}, mutate=0) is None


def test_unknown_backend_result_structure_is_rejected(backend):
    value = backend.cls(())
    svd = backend.xp.linalg.svd
    if backend.name == "jax":
        svd = backend.dispatch("a")(svd)
    with pytest.raises(NotImplementedError, match="known result structure"):
        svd(value)


def test_container_owner_and_conversion(backend):
    class Empty(NamedTuple):
        pass

    @dataclass(frozen=True, eq=False)
    class Tagged(backend.cls):
        members: tuple[Any, ...]
        label: str = "preserved"

    value = Tagged(Empty())
    for result in (value[42], value.reshape(3, 4), value + 1):
        assert_empty(result, Tagged)
        assert result.label == "preserved"
        assert type(result.members) is Empty
    convert = converter(backend)
    assert convert(value) is value
    assert_empty(convert(value, dtype=backend.xp.float32, copy=True), Tagged)
    converted = convert(NumpyBroadcastTuple(Empty()))
    assert_empty(converted, backend.cls)
    assert type(converted.members) is Empty
    restored = to_numpy_like(converted)
    assert_empty(restored, NumpyBroadcastTuple)
    assert isinstance(restored, NumpyBroadcastTuple)
    assert type(restored.members) is Empty


@pytest.mark.parametrize("all_broadcasted", [False, True])
def test_protected_composition(backend, all_broadcasted):
    @dataclass(frozen=True)
    class Composite(backend.base):
        data: Any
        optional: Any
        protected_axes: ClassVar[dict[str, int]] = {"data": 1, "optional": 0}
        broadcasted_batch: ClassVar[set[str]] = {"data", "optional"} if all_broadcasted else {"optional"}

    data = np.arange(24).reshape(2, 3, 4)
    value = Composite(backend.array(data), backend.cls(()))
    value.validate()
    for index in (
        1,
        (slice(None), 1),
        np.array([1, 0, 1]),
        np.array([True, False]),
        (np.array([0, 1]), np.array([2, 0])),
    ):
        result = value[index]
        result.validate()
        assert result.shape == data[index].shape[:-1]
        np.testing.assert_array_equal(result.data, data[index])
        assert_empty(result.optional, backend.cls)
    reshaped = value.reshape(6)
    broadcasted = backend.func("broadcast_to")(value, (5, 2, 3))
    transposed = (
        backend.func("permute")(value, (1, 0)) if backend.name == "torch" else backend.func("transpose")(value, (1, 0))
    )
    for result, shape in ((reshaped, (6,)), (broadcasted, (5, 2, 3)), (transposed, (3, 2))):
        result.validate()
        assert result.shape == shape
        assert_empty(result.optional, backend.cls)
    assert value.dtype == value.data.dtype
    assert value.device == value.data.device


def test_writes(backend):
    value = backend.cls(())
    if backend.name == "jax":
        index = backend.array([3, 5])
        for result in (
            value.at[index].get(),
            value.at[index].get(mode="clip"),
            value.at[index].add(value),
            value.at[index].set(3),
        ):
            assert_empty(result, backend.cls)
    else:
        value[np.array([3, 5])] = value
        value[123] = 7
        output = backend.cls(())
        assert backend.func("add")(value, 2, out=output) is output
        original = value
        value += 1
        assert value is original
        if backend.name == "numpy":
            assert np.copyto(value, 1) is None
            assert np.put(value, [123], 1) is None
            assert np.add.at(value, [123], 1) is None


def test_numpy_multiple_outputs():
    value = NumpyBroadcastTuple(())
    output = NumpyBroadcastTuple(())
    outputs: Any = (None, output)
    for results in (np.modf(value), np.frexp(value), divmod(value, 3), np.modf(value, out=outputs)):
        assert type(results) is tuple
        assert len(results) == 2
        for result in results:
            assert_empty(result, NumpyBroadcastTuple)
    assert np.modf(value, out=outputs)[1] is output
