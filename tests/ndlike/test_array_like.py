"""Tests for the abstract ``NumpyArrayLikeImplementation`` fallbacks."""

from __future__ import annotations

from inspect import isabstract
from typing import TYPE_CHECKING, Any, assert_type
from typing_extensions import get_protocol_members

import numpy as np
import pytest

from ndlike.array_like import NumpyArrayLike, NumpyArrayLikeImplementation

if TYPE_CHECKING:

    def _check_dispatched_method_types(
        protocol: NumpyArrayLike[Any], implementation: NumpyArrayLikeImplementation[np.ndarray]
    ) -> None:
        type Result = NumpyArrayLike[Any] | np.ndarray

        assert_type(protocol.T, Result)
        assert_type(protocol.mT, Result)
        assert_type(protocol.transpose(), Result)
        assert_type(protocol.copy(), Result)
        assert_type(protocol.astype(np.float32), Result)
        assert_type(implementation.T, Result)
        assert_type(implementation.mT, Result)
        assert_type(implementation.transpose(), Result)
        assert_type(implementation.copy(), Result)
        assert_type(implementation.astype(np.float32), Result)


class TestNumpyArrayLikeImplementationFallbacks:
    """Default error paths in NumpyArrayLikeImplementation (the abstract base)."""

    def _make_minimal_subclass(self):
        """Build a minimal concrete subclass that does not override the error paths."""

        class _Minimal(NumpyArrayLikeImplementation[np.ndarray]):
            @property
            def dtype(self):
                return np.dtype(np.float64)

            @property
            def device(self):
                return "cpu"

            @property
            def ndim(self):
                return 1

            @property
            def shape(self):
                return (1,)

            @property
            def size(self):
                return 1

            @property
            def flags(self):
                return np.zeros(1).flags

            def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):  # noqa: ANN002, ANN003, ANN204
                return NotImplemented

            def __array_function__(self, func, types, args, kwargs):  # noqa: ANN204
                return NotImplemented

            def __array__(self, dtype=None, /, *, copy=None):  # noqa: ANN204
                return np.asarray([0.0], dtype=dtype, copy=copy)

            def __array_namespace__(self, /, *, api_version=None):  # noqa: ANN204
                del api_version
                return np

            def to_device(self, device, /, *, stream=None):  # noqa: ARG002
                return self

        return _Minimal()

    @pytest.mark.parametrize("attribute", sorted(NumpyArrayLikeImplementation.__abstractmethods__))
    def test_representation_specific_members_must_be_implemented(self, attribute: str) -> None:
        complete = type(self._make_minimal_subclass())
        member = vars(NumpyArrayLikeImplementation)[attribute]
        incomplete = type("_IncompleteArray", (complete,), {attribute: member})
        assert isabstract(incomplete)
        assert incomplete.__abstractmethods__ == frozenset({attribute})
        with pytest.raises(TypeError, match=attribute):
            incomplete()

    @pytest.mark.parametrize("attribute", sorted(get_protocol_members(NumpyArrayLike)))
    def test_protocol_members_are_implemented_or_explicitly_abstract(self, attribute: str) -> None:
        owner = next(base for base in NumpyArrayLikeImplementation.__mro__ if attribute in vars(base))
        assert not vars(owner).get("_is_protocol", False)

    def test_minimal_subclass_is_concrete(self) -> None:
        minimal = self._make_minimal_subclass()
        assert not isabstract(type(minimal))
        assert isinstance(minimal, NumpyArrayLike)

    @pytest.mark.parametrize("native", [False, True])
    def test_dispatched_methods_can_return_another_wrapper_or_native_array(self, native: bool) -> None:
        complete = type(self._make_minimal_subclass())
        expected = np.ones(1) if native else complete()

        def change_result(self, func, types, args, kwargs):
            del self, func, types, args, kwargs
            return expected

        changing = type("_ChangingArray", (complete,), {"__array_function__": change_result})
        original = changing()
        for result in (
            original.T,
            original.mT,
            original.transpose(),
            original.copy(),
            original.astype(np.float32),
        ):
            assert result is expected

    def test_getitem_raises_not_implemented(self) -> None:
        m = self._make_minimal_subclass()
        with pytest.raises(NotImplementedError, match="indexing"):
            _ = m[0]

    def test_setitem_raises_not_implemented(self) -> None:
        m = self._make_minimal_subclass()
        with pytest.raises(NotImplementedError, match="item assignment"):
            m[0] = 1

    def test_index_raises_not_implemented(self) -> None:
        m = self._make_minimal_subclass()
        with pytest.raises(NotImplementedError, match="conversion to index"):
            m.__index__()

    def test_int_raises_not_implemented(self) -> None:
        m = self._make_minimal_subclass()
        with pytest.raises(NotImplementedError, match="conversion to int"):
            m.__int__()

    def test_bool_raises_not_implemented(self) -> None:
        m = self._make_minimal_subclass()
        with pytest.raises(NotImplementedError, match="conversion to bool"):
            m.__bool__()

    def test_float_raises_not_implemented(self) -> None:
        m = self._make_minimal_subclass()
        with pytest.raises(NotImplementedError, match="conversion to float"):
            m.__float__()

    def test_complex_raises_not_implemented(self) -> None:
        m = self._make_minimal_subclass()
        with pytest.raises(NotImplementedError, match="conversion to complex"):
            m.__complex__()

    def test_len_raises_not_implemented(self) -> None:
        m = self._make_minimal_subclass()
        with pytest.raises(NotImplementedError, match="len"):
            len(m)

    def test_iter_raises_not_implemented(self) -> None:
        m = self._make_minimal_subclass()
        with pytest.raises(NotImplementedError, match="iteration"):
            iter(m)
