"""Tests for NumPy-like conversion and implementation fallbacks."""

from __future__ import annotations

from inspect import isabstract
from typing import TYPE_CHECKING, Any, ClassVar, Never, Self, assert_type, get_args, get_origin, get_protocol_members

import numpy as np
import pytest

from ndlike.array_like import (
    ArrayLike,
    NumpyLike,
    NumpyLikeConvertible,
    NumpyLikeImplementation,
    ToIndices,
    WeakArrayLike,
    to_numpy_like,
)

if TYPE_CHECKING:
    from numpy.typing import ArrayLike as NpArrayLike

    class _CustomValue:
        """An assignment value with no NumPy conversion protocol."""

    class _SelfOnlyArray(ArrayLike):
        def __setitem__(self, index: ToIndices, value: Self, /) -> None: ...

    class _ExtraAssignmentArray(ArrayLike[Any, Any, Any, _CustomValue]):
        def __setitem__(self, index: ToIndices, value: Self | _CustomValue, /) -> None: ...

    def _check_default_native_array_protocols(array: np.ndarray) -> tuple[ArrayLike[Any], WeakArrayLike[Any]]:
        return array, array

    def _check_self_only_assignments(array: ArrayLike, weak: WeakArrayLike) -> None:
        assert_type(array, ArrayLike[ArrayLike[Any]])
        assert_type(weak, WeakArrayLike[WeakArrayLike[Any]])
        array[0] = array
        weak[0] = weak
        array[0] = 1  # ty: ignore[invalid-assignment]
        weak[0] = 1  # ty: ignore[invalid-assignment]

    def _check_concrete_self_assignments(
        array: _SelfOnlyArray, extra: _ExtraAssignmentArray, value: _CustomValue
    ) -> None:
        array[0] = array
        extra[0] = extra
        extra[0] = value
        array[0] = value  # ty: ignore[invalid-assignment]
        extra[0] = 1  # ty: ignore[invalid-assignment]

    def _check_native_array_protocols(
        array: np.ndarray,
    ) -> tuple[ArrayLike[np.ndarray], WeakArrayLike[np.ndarray]]:
        return array, array

    def _check_numpy_assignment_inputs(array: np.ndarray) -> ArrayLike[Any, Any, Any, NpArrayLike]:
        return array

    def _check_custom_assignment_protocols(
        array: NumpyLikeImplementation[NumpyLike[Any], Any, NumpyLike[Any], object],
    ) -> tuple[ArrayLike[Any, Any, Any, _CustomValue], WeakArrayLike[Any, Any, Any, _CustomValue]]:
        return array, array

    def _check_assignment_values(
        array: ArrayLike[Any, Any, Any, _CustomValue],
        weak: WeakArrayLike[Any, Any, Any, _CustomValue],
        value: _CustomValue,
    ) -> None:
        array[0] = array
        weak[0] = weak
        array[0] = value
        weak[0] = value
        array[0] = 1  # ty: ignore[invalid-assignment]
        weak[0] = 1  # ty: ignore[invalid-assignment]

    def _check_assignment_contravariance(
        array: ArrayLike[Any, Any, Any, object], weak: WeakArrayLike[Any, Any, Any, object]
    ) -> tuple[ArrayLike[Any, Any, Any, _CustomValue], WeakArrayLike[Any, Any, Any, _CustomValue]]:
        return array, weak

    def _check_assignment_cannot_be_widened(
        array: ArrayLike[Any, Any, Any, _CustomValue], weak: WeakArrayLike[Any, Any, Any, _CustomValue]
    ) -> None:
        broad: ArrayLike[Any, Any, Any, object] = array  # ty: ignore[invalid-assignment]
        broad_weak: WeakArrayLike[Any, Any, Any, object] = weak  # ty: ignore[invalid-assignment]
        del broad, broad_weak

    def _check_dispatched_method_types(
        protocol: NumpyLike, implementation: NumpyLikeImplementation, order: str
    ) -> None:
        type Result = NumpyLike[Any]

        assert_type(protocol.T, Result)
        assert_type(protocol.mT, Result)
        assert_type(protocol.transpose(), Result)
        assert_type(protocol.copy(), Result)
        assert_type(protocol.astype(np.float32), Result)
        assert_type(implementation.T, Result)
        assert_type(implementation.mT, Result)
        assert_type(implementation.transpose(), Result)
        assert_type(implementation.reshape(2, 3), Result)
        assert_type(implementation.reshape((2, 3)), Result)
        assert_type(implementation.reshape([2, 3], order="F", copy=True), Result)
        assert_type(implementation.reshape((2, 3), order=order), Result)
        assert_type(implementation.squeeze(), Result)
        assert_type(implementation.squeeze(axis=0), Result)
        assert_type(implementation.swapaxes(0, 1), Result)
        assert_type(implementation.copy(), Result)
        assert_type(implementation.astype(np.float32), Result)


class _Convertible(np.ndarray):
    """An array whose conversion hook records whether dispatch selected it."""

    hook_calls: ClassVar[int] = 0

    def __array_like__(self, dtype=None, /, *, copy=None) -> np.ndarray:
        type(self).hook_calls += 1
        return np.asarray(self, dtype=dtype, copy=copy)


def test_converter_protocol_remains_generic() -> None:
    assert get_origin(NumpyLikeConvertible[np.ndarray]) is NumpyLikeConvertible


@pytest.mark.parametrize("protocol", [WeakArrayLike, ArrayLike, NumpyLike, NumpyLikeConvertible])
def test_assignment_type_defaults_to_never(protocol) -> None:
    args = get_args(protocol[()])
    if protocol is NumpyLikeConvertible:
        assert args == (Never,)
    else:
        assert args[3] is Never
        assert args[1] == args[2] == args[4] == args[6] == args[0]


def test_array_converter_uses_its_hook() -> None:
    data = np.arange(3.0).view(_Convertible)
    previous_calls = _Convertible.hook_calls
    assert isinstance(data, NumpyLikeConvertible)

    result = to_numpy_like(data, dtype=np.float32, copy=True)

    assert _Convertible.hook_calls == previous_calls + 1
    assert result.dtype == np.float32
    assert not np.shares_memory(result, data)
    np.testing.assert_array_equal(result, np.arange(3.0, dtype=np.float32))


def test_hook_alone_does_not_satisfy_the_converter_protocol() -> None:
    class _HookOnly:
        def __array_like__(self, dtype=None, /, *, copy=None) -> np.ndarray:
            del dtype, copy
            pytest.fail("A hook-only object must not be selected as an array converter")

    assert not isinstance(_HookOnly(), NumpyLikeConvertible)


def test_conversion_falls_back_to_numpy() -> None:
    result = to_numpy_like([1, 2, 3], dtype=np.float32)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32
    np.testing.assert_array_equal(result, [1, 2, 3])


def test_native_array_is_numpy_like() -> None:
    data = np.arange(3.0)
    assert isinstance(data, NumpyLike)
    assert isinstance(data, NumpyLikeImplementation)
    assert to_numpy_like(data, copy=False) is data


@pytest.mark.parametrize("shape", [(), (3,), (2, 3)])
def test_native_array_satisfies_core_protocols(shape: tuple[int, ...]) -> None:
    data = np.zeros(shape)
    assert isinstance(data, WeakArrayLike)
    assert isinstance(data, ArrayLike)


class TestNumpyLikeImplementationFallbacks:
    """Default error paths in NumpyLikeImplementation (the abstract base)."""

    def _make_minimal_subclass(self):
        """Build a minimal concrete subclass that does not override the error paths."""

        class _Minimal(NumpyLikeImplementation):
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

    @pytest.mark.parametrize("attribute", sorted(NumpyLikeImplementation.__abstractmethods__))
    def test_representation_specific_members_must_be_implemented(self, attribute: str) -> None:
        complete = type(self._make_minimal_subclass())
        member = vars(NumpyLikeImplementation)[attribute]
        incomplete = type("_IncompleteArray", (complete,), {attribute: member})
        assert isabstract(incomplete)
        assert incomplete.__abstractmethods__ == frozenset({attribute})
        with pytest.raises(TypeError, match=attribute):
            incomplete()

    @pytest.mark.parametrize("attribute", sorted(get_protocol_members(NumpyLike)))
    def test_protocol_members_are_implemented_or_explicitly_abstract(self, attribute: str) -> None:
        owner = next(base for base in NumpyLikeImplementation.__mro__ if attribute in vars(base))
        assert not vars(owner).get("_is_protocol", False)

    def test_minimal_subclass_is_concrete(self) -> None:
        minimal = self._make_minimal_subclass()
        assert not isabstract(type(minimal))
        assert isinstance(minimal, NumpyLike)

    def test_numpy_protocol_is_not_widened_with_convenience_methods(self) -> None:
        members = get_protocol_members(NumpyLike)
        for name in ("reshape", "squeeze", "swapaxes"):
            assert name not in members

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
            original.reshape(1),
            original.reshape((1,)),
            original.reshape([1]),
            original.reshape(1, copy=True),
            original.squeeze(),
            original.swapaxes(0, 0),
            original.copy(),
            original.astype(np.float32),
        ):
            assert result is expected

    def test_shaping_methods_dispatch_to_numpy_functions(self) -> None:
        complete = type(self._make_minimal_subclass())
        calls: list[tuple[Any, tuple[Any, ...], dict[str, Any]]] = []
        sentinel = complete()

        def record(self, func, types, args, kwargs):
            del self, types
            calls.append((func, args, kwargs))
            return sentinel

        recording = type("_RecordingArray", (complete,), {"__array_function__": record})
        array = recording()

        assert array.reshape(2, 3) is sentinel
        assert calls[-1] == (np.reshape, (array, (2, 3)), {"order": "C"})

        assert array.reshape((2, 3), copy=None) is sentinel
        assert calls[-1] == (np.reshape, (array, (2, 3)), {"order": "C"})

        assert array.reshape([2, 3], order="F", copy=True) is sentinel
        assert calls[-1] == (np.reshape, (array, [2, 3]), {"order": "F", "copy": True})

        range_shape = range(2, 4)
        assert array.reshape(range_shape, copy=False) is sentinel
        assert calls[-1] == (np.reshape, (array, range_shape), {"order": "C", "copy": False})

        scalar_dim: Any = np.int64(6)
        assert array.reshape(scalar_dim) is sentinel
        assert calls[-1] == (np.reshape, (array, scalar_dim), {"order": "C"})

        scalar_array: Any = np.array(6)
        assert array.reshape(scalar_array) is sentinel
        assert calls[-1] == (np.reshape, (array, scalar_array), {"order": "C"})

        vector_array: Any = np.array([2, 3])
        assert array.reshape(vector_array) is sentinel
        assert calls[-1] == (np.reshape, (array, vector_array), {"order": "C"})

        class _IndexDim:
            def __index__(self) -> int:
                return 6

        custom_dim: Any = _IndexDim()
        assert array.reshape(custom_dim) is sentinel
        assert calls[-1] == (np.reshape, (array, custom_dim), {"order": "C"})

        assert array.reshape(None) is sentinel
        assert calls[-1] == (np.reshape, (array, (None,)), {"order": "C"})

        assert array.squeeze() is sentinel
        assert calls[-1] == (np.squeeze, (array,), {"axis": None})

        assert array.squeeze((0, 2)) is sentinel
        assert calls[-1] == (np.squeeze, (array,), {"axis": (0, 2)})

        assert array.swapaxes(0, 1) is sentinel
        assert calls[-1] == (np.swapaxes, (array, 0, 1), {})

        with pytest.raises(TypeError):
            array.swapaxes(axis1=0, axis2=1)

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
