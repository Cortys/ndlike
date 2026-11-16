"""Conversion hooks preserve member structure and keep optional backends lazy."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
import subprocess
import sys
from typing import Any, ClassVar

import numpy as np

from ndlike.array_like import to_numpy_like
from ndlike.tuple.numpy import NumpyBroadcastTuple


def converter(backend):
    module = "array_like" if backend.name == "numpy" else f"{backend.name}_like"
    return getattr(import_module(f"ndlike.{module}"), f"to_{backend.name}_like")


def test_conversion_preserves_compact_shapes_and_values(backend):
    source = NumpyBroadcastTuple((np.ones((2, 1), dtype=np.float32), np.arange(3, dtype=np.float32), np.array(7.0)))
    converted = converter(backend)(source, dtype=backend.xp.float32)
    assert type(converted) is backend.cls
    assert converted.shape == (2, 3)
    assert tuple(member.shape for member in converted.members) == ((2, 1), (3,), ())
    assert all(member.dtype == backend.xp.float32 for member in converted.members)
    restored = to_numpy_like(converted)
    assert isinstance(restored, NumpyBroadcastTuple)
    for actual, original in zip(restored.members, source.members, strict=True):
        np.testing.assert_array_equal(actual, original)


def test_same_backend_identity_and_explicit_options(backend):
    @dataclass(frozen=True)
    class Tagged(backend.cls):
        members: tuple[Any, ...]
        tag: str = "preserved"

    source = Tagged((backend.array(np.ones((2, 1), dtype=np.float32)), backend.array(np.arange(3, dtype=np.float32))))
    convert = converter(backend)
    assert convert(source) is source
    assert convert(source, copy=False) is source
    if backend.name == "numpy":
        assert convert(source, copy=None) is source
    options = {} if backend.name == "numpy" else {"device": source.device}
    copied = convert(source, copy=True, **options)
    changed = convert(source, dtype=backend.xp.int32, **options)
    assert type(copied) is type(changed) is Tagged
    assert copied.tag == changed.tag == source.tag
    for old, new, integer in zip(source.members, copied.members, changed.members, strict=True):
        assert new is not old
        assert new.shape == integer.shape == old.shape
        assert new.dtype == old.dtype
        assert integer.dtype == backend.xp.int32
        np.testing.assert_array_equal(new, old)
        if backend.name != "numpy":
            assert new.device == integer.device == old.device


def test_member_conversion_hooks_preserve_custom_representations(backend):
    @dataclass(frozen=True)
    class Wrapped(backend.base):
        data: Any
        protected_axes: ClassVar[dict[str, int]] = {"data": 1}

    calls = []
    wrapped = Wrapped(backend.array(np.ones((2, 1, 4), dtype=np.float32)))

    class Convertible(np.ndarray):
        pass

    def convert_member(self, dtype=None, /, **kwargs: object):
        calls.append((self, dtype, kwargs))
        return wrapped

    hook = "__array_like__" if backend.name == "numpy" else f"__{backend.name}_like__"
    setattr(Convertible, hook, convert_member)
    member = np.ones((2, 1)).view(Convertible)
    source = NumpyBroadcastTuple((member, np.arange(3, dtype=np.float32)))
    options = {} if backend.name == "numpy" else {"device": wrapped.device}
    result = converter(backend)(source, dtype=backend.xp.float32, copy=True, **options)
    assert type(result) is backend.cls
    assert result.shape == (2, 3)
    assert result.members[0] is wrapped
    assert result.members[1].shape == (3,)
    assert len(calls) == 1
    assert calls[0][0] is member
    assert calls[0][1] == backend.xp.float32
    assert calls[0][2] == {"copy": True, **options}


def test_numpy_conversion_does_not_import_optional_backends():
    script = """
import sys
import numpy as np
from ndlike.array_like import to_numpy_like
from ndlike.tuple.numpy import NumpyBroadcastTuple

assert 'jax' not in sys.modules
assert 'torch' not in sys.modules
sys.modules['jax'] = None
sys.modules['torch'] = None
source = NumpyBroadcastTuple((np.ones((2, 1)), np.arange(3)))
result = to_numpy_like(source)
assert isinstance(result, NumpyBroadcastTuple)
assert result.shape == (2, 3)
assert all(a is b for a, b in zip(source.members, result.members))
"""
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)  # noqa: S603
