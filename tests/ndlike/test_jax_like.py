"""Tests for the JAX-like protocols, base class and override machinery."""

from __future__ import annotations

from dataclasses import dataclass, field
from inspect import isabstract
import operator
from typing import Any, ClassVar, Never, Self, assert_type
from typing_extensions import get_protocol_members

import numpy as np
import pytest

pytest.importorskip("jax")

import jax
import jax.numpy as jnp

from ndlike.jax_functions import (
    handle_jax_function,
    has_jax_function,
    jax_add,
    jax_average,
    jax_concatenate,
    jax_copy,
    jax_mean,
    jax_multiply,
    jax_reshape,
    jax_stack,
    jax_std,
    jax_sum,
    jax_var,
)
from ndlike.jax_like import (
    JaxLike,
    JaxLikeConvertible,
    JaxLikeImplementation,
    jax_tree_map,
    to_jax_like,
)
from ndlike.protected_axis.numpy import NumpyAxisProtected


def _unwrap(value: Any) -> Any:  # noqa: ANN401
    if isinstance(value, _Wrapper):
        return value.array
    if isinstance(value, (list, tuple)):
        return type(value)(_unwrap(item) for item in value)
    return value


@dataclass(frozen=True)
class _Wrapper(JaxLikeImplementation[jax.Array]):
    """Minimal concrete subclass that forwards every call to the wrapped array."""

    array: jax.Array
    label: str = "wrapped"
    pytree_children: ClassVar[tuple[str, ...]] = ("array",)

    @classmethod
    def __jax_function__(cls, func, types, args=(), kwargs=None):  # noqa: ANN206
        del types
        kwargs = {} if kwargs is None else kwargs
        result = func(*(_unwrap(arg) for arg in args), **{key: _unwrap(value) for key, value in kwargs.items()})
        return cls(result) if isinstance(result, jax.Array) else result

    @property
    def dtype(self):
        return self.array.dtype

    @property
    def device(self):
        return self.array.device

    @property
    def ndim(self) -> int:
        return self.array.ndim

    @property
    def shape(self) -> tuple[int, ...]:
        return self.array.shape

    @property
    def size(self) -> int:
        return self.array.size

    @property
    def at(self):
        return self.array.at

    @property
    def sharding(self):
        return self.array.sharding

    def devices(self) -> set[jax.Device]:
        return self.array.devices()

    def __getitem__(self, index):  # noqa: ANN204
        return type(self)(self.array[index], self.label)

    def __setitem__(self, index, value) -> None:
        del index, value
        msg = "JAX arrays are immutable."
        raise NotImplementedError(msg)

    def __array_namespace__(self, /, *, api_version=None):  # noqa: ANN204
        del api_version
        return jnp

    def __len__(self) -> int:
        return len(self.array)

    def __iter__(self):  # noqa: ANN204
        return (self[index] for index in range(len(self)))


@dataclass(frozen=True)
class _NumpyArray(NumpyAxisProtected[np.ndarray]):
    array: np.ndarray
    protected_axes: ClassVar[dict[str, int]] = {"array": 0}
    permitted_ufuncs: ClassVar[dict[np.ufunc, list[str]]] = {
        func: ["__call__"] for func in (np.add, np.subtract, np.multiply, np.true_divide, np.power)
    }


@dataclass(frozen=True)
class _Convertible:
    """Minimal ``JaxLikeConvertible`` used to check the ``to_jax_like`` registration."""

    values: list[float]

    def __jax_like__(self, dtype=None, /, *, device=None, copy=False):  # noqa: ANN204
        return jnp.asarray(self.values, dtype=dtype, device=device, copy=copy)

    def __array__(self, dtype=None, /, *, copy=None):  # noqa: ANN204
        return np.asarray(self.values, dtype=dtype, copy=copy)

    def __getitem__(self, index):  # noqa: ANN204
        return self.values[index]

    def __setitem__(self, index, value) -> None:
        self.values[index] = value

    def __array_namespace__(self, /, *, api_version=None):  # noqa: ANN204
        del api_version
        return np

    def to_device(self, device, /, *, stream=None):
        del device, stream
        return self

    def __len__(self) -> int:
        return len(self.values)

    def __iter__(self):  # noqa: ANN204
        return iter(self.values)

    @property
    def ndim(self) -> int:
        return 1

    @property
    def shape(self) -> tuple[int, ...]:
        return (len(self.values),)

    @property
    def dtype(self):
        return np.dtype(np.float64)

    @property
    def size(self) -> int:
        return len(self.values)

    @property
    def device(self) -> str:
        return "cpu"

    @property
    def T(self):  # noqa: N802
        return self

    @property
    def mT(self):  # noqa: N802
        return self


class TestProtocols:
    """The protocols must accept plain JAX arrays."""

    def test_jax_array_is_jax_like(self) -> None:
        assert isinstance(jnp.ones(3), JaxLike)

    def test_jax_array_is_jax_like_2d(self) -> None:
        assert isinstance(jnp.ones((2, 3)), JaxLike)

    def test_jax_array_is_jax_like_implementation(self) -> None:
        assert isinstance(jnp.ones(3), JaxLikeImplementation)

    def test_jax_array_is_not_convertible(self) -> None:
        assert not isinstance(jnp.ones(3), JaxLikeConvertible)

    def test_convertible_is_convertible(self) -> None:
        assert isinstance(_Convertible([1.0, 2.0]), JaxLikeConvertible)

    def test_jax_function_is_not_part_of_the_protocol(self) -> None:
        """``jax.Array`` has no ``__jax_function__``, so the protocol must not require it."""
        assert not hasattr(jnp.ones(3), "__jax_function__")
        assert "__jax_function__" not in get_protocol_members(JaxLike)


class TestToJaxLike:
    """The ``to_jax_like`` dispatch."""

    def test_from_list(self) -> None:
        result = to_jax_like([1.0, 2.0, 3.0])
        assert isinstance(result, jax.Array)
        np.testing.assert_allclose(np.asarray(result), [1.0, 2.0, 3.0])

    def test_from_numpy(self) -> None:
        result = to_jax_like(np.arange(4.0))
        assert isinstance(result, jax.Array)
        np.testing.assert_allclose(np.asarray(result), np.arange(4.0))

    def test_dtype_is_honoured(self) -> None:
        result = to_jax_like([1, 2, 3], jnp.float32)
        assert result.dtype == jnp.float32

    def test_device_is_honoured(self) -> None:
        device = jax.devices()[0]
        result = to_jax_like([1, 2, 3], device=device)
        assert device in result.devices()

    def test_no_copy_returns_same_array(self) -> None:
        array = jnp.ones(3)
        assert to_jax_like(array) is array

    def test_copy_returns_new_array(self) -> None:
        array = jnp.ones(3)
        assert to_jax_like(array, copy=True) is not array

    def test_convertible_uses_dunder(self) -> None:
        result = to_jax_like(_Convertible([1.0, 2.0]), jnp.float32)
        assert isinstance(result, jax.Array)
        assert result.dtype == jnp.float32
        np.testing.assert_allclose(np.asarray(result), [1.0, 2.0])


class TestOverrideMachinery:
    """``has_jax_function`` and ``handle_jax_function``."""

    def test_has_jax_function_false_for_plain_arrays(self) -> None:
        assert not has_jax_function((jnp.ones(3), None, 2))

    def test_has_jax_function_true_for_implementations(self) -> None:
        assert has_jax_function((jnp.ones(3), _Wrapper(jnp.ones(3))))

    def test_handle_jax_function_dispatches(self) -> None:
        wrapper = _Wrapper(jnp.arange(6.0))
        result = handle_jax_function(jax_sum, (wrapper,), wrapper)
        assert isinstance(result, _Wrapper)
        assert float(np.asarray(result.array)) == pytest.approx(15.0)

    def test_handle_jax_function_skips_not_implemented(self) -> None:
        class _NotImplementedWrapper(_Wrapper):
            @classmethod
            def __jax_function__(cls, func, types, args=(), kwargs=None):  # noqa: ANN206
                del cls, func, types, args, kwargs
                return NotImplemented

        first = _NotImplementedWrapper(jnp.arange(3.0))
        second = _Wrapper(jnp.arange(3.0))

        result = handle_jax_function(jax_sum, (first, second), second)
        assert isinstance(result, _Wrapper)

    def test_handle_jax_function_raises_when_unhandled(self) -> None:
        class _AlwaysNotImplemented(_Wrapper):
            @classmethod
            def __jax_function__(cls, func, types, args=(), kwargs=None):  # noqa: ANN206
                del cls, func, types, args, kwargs
                return NotImplemented

        wrapper = _AlwaysNotImplemented(jnp.arange(3.0))
        with pytest.raises(TypeError, match="no implementation found"):
            handle_jax_function(jax_sum, (wrapper,), wrapper)

    def test_handle_jax_function_raises_without_overloads(self) -> None:
        with pytest.raises(TypeError, match="no implementation found"):
            handle_jax_function(jax_sum, (jnp.ones(3),), jnp.ones(3))


class TestWrappers:
    """The ``jnp`` mirroring wrappers fall back to plain ``jax.numpy``."""

    def test_reshape_without_override(self) -> None:
        result = jax_reshape(jnp.arange(6), (2, 3))
        assert result.shape == (2, 3)

    def test_reduction_wrappers_without_override(self) -> None:
        array = jnp.arange(6.0)
        assert float(jax_sum(array)) == pytest.approx(15.0)
        assert float(jax_mean(array)) == pytest.approx(2.5)
        assert float(jax_std(array)) == pytest.approx(float(jnp.std(array)))
        assert float(jax_var(array)) == pytest.approx(float(jnp.var(array)))

    def test_average_with_weights(self) -> None:
        array = jnp.asarray([1.0, 2.0, 3.0])
        weights = jnp.asarray([1.0, 0.0, 1.0])
        averaged = jax_average(array, weights=weights)
        assert isinstance(averaged, jax.Array)
        assert float(averaged) == pytest.approx(2.0)

    def test_concatenate_and_stack_without_override(self) -> None:
        array = jnp.arange(3.0)
        assert jax_concatenate([array, array]).shape == (6,)
        assert jax_stack([array, array], 1).shape == (3, 2)

    def test_concatenate_dispatches_on_sequence_elements(self) -> None:
        wrapper = _Wrapper(jnp.arange(3.0))
        result = jax_concatenate([wrapper, wrapper])
        assert isinstance(result, _Wrapper)
        assert result.array.shape == (6,)

    def test_override_can_change_wrapper_class_or_return_a_native_array(self) -> None:
        class _ChangingWrapper(_Wrapper):
            @classmethod
            def __jax_function__(
                cls, func, types, args: tuple[Any, ...] = (), kwargs: dict[str, Any] | None = None
            ) -> object:
                del cls, types
                original = args[0]
                assert isinstance(original, _ChangingWrapper)
                if func is jax_copy:
                    return _Wrapper(jnp.copy(original.array))
                if func is jax_sum:
                    return jnp.sum(original.array, **({} if kwargs is None else kwargs))
                return NotImplemented

        original = _ChangingWrapper(jnp.arange(3.0))
        changed = assert_type(jax_copy(original), jax.Array | JaxLike[Any])
        assert isinstance(changed, _Wrapper)
        assert not isinstance(changed, _ChangingWrapper)
        np.testing.assert_array_equal(changed.array, original.array)

        unwrapped = assert_type(jax_sum(original), jax.Array | JaxLike[Any])
        assert isinstance(unwrapped, jax.Array)
        assert float(unwrapped) == pytest.approx(3.0)

    @pytest.mark.parametrize("native", [False, True])
    def test_dispatched_methods_do_not_promise_the_original_wrapper_type(self, native: bool) -> None:
        class _ChangingWrapper(_Wrapper):
            @classmethod
            def __jax_function__(
                cls, func, types, args: tuple[Any, ...] = (), kwargs: dict[str, Any] | None = None
            ) -> object:
                del cls, types
                result = func(*(_unwrap(arg) for arg in args), **({} if kwargs is None else kwargs))
                return result if native else _Wrapper(result)

        type Result = JaxLike[Any] | jax.Array

        original = _ChangingWrapper(jnp.arange(6.0).reshape(2, 3))
        results = (
            assert_type(original.T, Result),
            assert_type(original.mT, Result),
            assert_type(original.mH, Result),
            assert_type(original.copy(), Result),
            assert_type(original.reshape(3, 2), Result),
            assert_type(original.transpose(), Result),
            assert_type(original.ravel(), Result),
            assert_type(original.flatten(), Result),
            assert_type(original.squeeze(), Result),
        )
        for result in results:
            if native:
                assert isinstance(result, jax.Array)
            else:
                assert isinstance(result, _Wrapper)
                assert not isinstance(result, _ChangingWrapper)

        # Field mapping and synchronization do not dispatch on the outer wrapper.
        assert isinstance(assert_type(original.astype(jnp.float32), _ChangingWrapper), _ChangingWrapper)
        assert isinstance(assert_type(original.view(jnp.float32), _ChangingWrapper), _ChangingWrapper)
        assert isinstance(assert_type(original.stop_gradient(), _ChangingWrapper), _ChangingWrapper)
        assert isinstance(assert_type(original.to_device(jax.devices()[0]), _ChangingWrapper), _ChangingWrapper)
        assert assert_type(original.block_until_ready(), _ChangingWrapper) is original


class TestImplementationDefaults:
    """The concrete methods that ``JaxLikeImplementation`` provides for free."""

    @staticmethod
    def _wrapper() -> _Wrapper:
        return _Wrapper(jnp.arange(6.0).reshape(2, 3))

    def test_transpose_property(self) -> None:
        assert self._wrapper().T.shape == (3, 2)

    def test_matrix_transpose_property(self) -> None:
        assert self._wrapper().mT.shape == (3, 2)

    def test_adjoint_property(self) -> None:
        result = self._wrapper().mH
        np.testing.assert_allclose(np.asarray(result), np.arange(6.0).reshape(2, 3).T)

    def test_reshape_with_separate_ints(self) -> None:
        assert self._wrapper().reshape(3, 2).shape == (3, 2)

    def test_reshape_with_sequence(self) -> None:
        assert self._wrapper().reshape((3, 2)).shape == (3, 2)

    def test_transpose_method_with_axes(self) -> None:
        assert self._wrapper().transpose(1, 0).shape == (3, 2)

    def test_transpose_method_without_axes(self) -> None:
        assert self._wrapper().transpose().shape == (3, 2)

    def test_ravel(self) -> None:
        assert self._wrapper().ravel().shape == (6,)

    def test_flatten(self) -> None:
        assert self._wrapper().flatten().shape == (6,)

    def test_squeeze(self) -> None:
        wrapper = _Wrapper(jnp.ones((1, 3)))
        assert wrapper.squeeze().shape == (3,)

    def test_squeeze_with_axis(self) -> None:
        wrapper = _Wrapper(jnp.ones((1, 3)))
        assert wrapper.squeeze(0).shape == (3,)

    def test_copy(self) -> None:
        wrapper = self._wrapper()
        copied = wrapper.copy()
        assert isinstance(copied, _Wrapper)
        np.testing.assert_allclose(np.asarray(copied), np.asarray(wrapper))

    def test_item(self) -> None:
        assert _Wrapper(jnp.asarray(3.0)).item() == pytest.approx(3.0)

    def test_tolist(self) -> None:
        assert _Wrapper(jnp.asarray([1.0, 2.0])).tolist() == [1.0, 2.0]

    def test_array_conversion(self) -> None:
        np.testing.assert_allclose(np.asarray(self._wrapper()), np.arange(6.0).reshape(2, 3))

    def test_array_conversion_with_dtype(self) -> None:
        assert np.asarray(self._wrapper(), dtype=np.int32).dtype == np.int32

    def test_astype(self) -> None:
        cast_wrapper = self._wrapper().astype(jnp.int32)
        assert isinstance(cast_wrapper, _Wrapper)
        assert cast_wrapper.array.dtype == jnp.int32

    def test_block_until_ready_returns_self(self) -> None:
        wrapper = self._wrapper()
        assert wrapper.block_until_ready() is wrapper

    def test_to_device(self) -> None:
        moved = self._wrapper().to_device(jax.devices()[0])
        assert isinstance(moved, _Wrapper)
        np.testing.assert_allclose(np.asarray(moved), np.arange(6.0).reshape(2, 3))

    def test_to_device_rejects_stream(self) -> None:
        with pytest.raises(NotImplementedError, match="stream"):
            self._wrapper().to_device(jax.devices()[0], stream=0)

    def test_stop_gradient_returns_same_values(self) -> None:
        detached = self._wrapper().stop_gradient()
        assert isinstance(detached, _Wrapper)
        np.testing.assert_allclose(np.asarray(detached), np.arange(6.0).reshape(2, 3))

    def test_stop_gradient_cuts_the_gradient(self) -> None:
        def loss(wrapper: _Wrapper) -> jax.Array:
            return jnp.sum(wrapper.stop_gradient().array) + jnp.sum(wrapper.array)

        gradient = jax.grad(loss)(self._wrapper())
        np.testing.assert_allclose(np.asarray(gradient.array), np.ones((2, 3)))

    @pytest.mark.parametrize("attribute", sorted(JaxLikeImplementation.__abstractmethods__))
    def test_representation_specific_members_must_be_implemented(self, attribute: str) -> None:
        """Leaving any representation-specific member abstract prevents instantiation."""
        member = vars(JaxLikeImplementation)[attribute]
        assert member.__isabstractmethod__
        incomplete = type("_IncompleteWrapper", (_Wrapper,), {attribute: member})
        assert isabstract(incomplete)
        assert incomplete.__abstractmethods__ == frozenset({attribute})
        with pytest.raises(TypeError, match=attribute):
            incomplete(jnp.ones(3))

    @pytest.mark.parametrize("attribute", sorted(get_protocol_members(JaxLike)))
    def test_protocol_members_are_implemented_or_explicitly_abstract(self, attribute: str) -> None:
        owner = next(base for base in JaxLikeImplementation.__mro__ if attribute in vars(base))
        assert not vars(owner).get("_is_protocol", False)

    def test_complete_wrapper_satisfies_the_protocol(self) -> None:
        assert not isabstract(_Wrapper)
        assert isinstance(self._wrapper(), JaxLike)


class TestPytreeRegistration:
    """Concrete subclasses are usable inside JAX transformations."""

    def test_tree_map_round_trip(self) -> None:
        wrapper = _Wrapper(jnp.arange(3.0))
        mapped = jax.tree.map(lambda leaf: leaf * 2, wrapper)
        assert isinstance(mapped, _Wrapper)
        np.testing.assert_allclose(np.asarray(mapped.array), [0.0, 2.0, 4.0])

    def test_static_fields_survive_flattening(self) -> None:
        wrapper = _Wrapper(jnp.arange(3.0), label="static")
        mapped = jax.tree.map(lambda leaf: leaf + 1, wrapper)
        assert mapped.label == "static"

    def test_jit_round_trip(self) -> None:
        @jax.jit
        def double(value: _Wrapper) -> _Wrapper:
            return _Wrapper(value.array * 2, value.label)

        result = double(_Wrapper(jnp.arange(3.0)))
        assert isinstance(result, _Wrapper)
        np.testing.assert_allclose(np.asarray(result.array), [0.0, 2.0, 4.0])

    def test_none_field_is_a_child(self) -> None:
        @dataclass(frozen=True)
        class _Optional(_Wrapper):
            weights: jax.Array | None = None
            pytree_children: ClassVar[tuple[str, ...]] = ("array", "weights")

        wrapper = _Optional(jnp.arange(3.0))
        children, _ = wrapper.tree_flatten()
        assert children[-1] is None

    @pytest.mark.parametrize(
        ("declaration", "error", "message"),
        [
            (None, TypeError, "must declare pytree_children"),
            (["array"], TypeError, "tuple of non-empty field names"),
            ((1,), TypeError, "tuple of non-empty field names"),
            (("",), TypeError, "tuple of non-empty field names"),
            (("array", "array"), ValueError, "must not repeat"),
            (("missing",), ValueError, "unknown fields: missing"),
        ],
    )
    def test_invalid_child_declarations(self, declaration, error, message):
        cls = type("_InvalidSchema", (_Wrapper,), {"pytree_children": declaration})
        with pytest.raises(error, match=message):
            jax.tree.leaves(cls(jnp.ones(3)))

    def test_missing_declaration_is_not_inferred(self):
        class _Undeclared(JaxLikeImplementation[Any]):
            pass

        with pytest.raises(TypeError, match="must declare pytree_children"):
            _Undeclared._pytree_child_names()  # noqa: SLF001

    def test_non_dataclasses_need_an_explicit_stored_field_layout(self):
        class _Manual(JaxLikeImplementation[Any]):
            pass

        with pytest.raises(TypeError, match="must be a dataclass or override"):
            _Manual._pytree_field_names()  # noqa: SLF001

    def test_child_order_and_placeholder_values_are_stable(self):
        @dataclass(frozen=True, slots=True)
        class _Pair(_Wrapper):
            weights: jax.Array | None = None
            pytree_children: ClassVar[tuple[str, ...]] = ("weights", "array")

        original = _Pair(jnp.ones(3), weights=jnp.zeros(3))
        children, aux = original.tree_flatten()
        assert children[0] is original.weights
        assert children[1] is original.array
        placeholders = (object(), object())
        rebuilt = _Pair.tree_unflatten(aux, placeholders)
        rebuilt_children, rebuilt_aux = rebuilt.tree_flatten()
        assert all(left is right for left, right in zip(rebuilt_children, placeholders, strict=True))
        assert rebuilt_aux == aux

    def test_scalar_children_participate_in_autodiff(self):
        @dataclass(frozen=True)
        class _Scaled(_Wrapper):
            scale: float = 2.0
            pytree_children: ClassVar[tuple[str, ...]] = ("array", "scale")

        x = _Scaled(jnp.arange(3.0))
        gradient = jax.grad(lambda value: jnp.sum(value.array) * value.scale)(x)
        np.testing.assert_array_equal(gradient.array, [2.0, 2.0, 2.0])
        assert float(gradient.scale) == pytest.approx(3.0)

    @pytest.mark.parametrize(
        "container",
        [
            pytest.param(lambda value: value, id="direct"),
            pytest.param(lambda value: [value], id="list"),
            pytest.param(lambda value: {"nested": (jnp.ones(2), value)}, id="nested-dict-tuple"),
        ],
    )
    def test_numpy_arrays_in_dynamic_fields_are_rejected_recursively(self, container):
        @dataclass(frozen=True)
        class _WithPayload(_Wrapper):
            payload: Any = None
            pytree_children: ClassVar[tuple[str, ...]] = ("array", "payload")

        x = _WithPayload(jnp.ones(3), payload=container(np.ones(2)))
        with pytest.raises(TypeError, match=r"_WithPayload\.payload contains a NumPy array"):
            jax.tree.leaves(x)

    def test_nested_implementations_reject_their_own_numpy_fields(self):
        @dataclass(frozen=True)
        class _Nested(_Wrapper):
            array: Any

        array: Any = np.ones(3)
        x = _Nested(_Wrapper(array))
        with pytest.raises(TypeError, match=r"_Wrapper\.array contains a NumPy array"):
            jax.jit(lambda value: value)(x)

    def test_registered_child_containers_cannot_hide_numpy_arrays(self):
        @jax.tree_util.register_pytree_node_class
        @dataclass(frozen=True)
        class _Container:
            value: Any

            def tree_flatten(self):
                return (self.value,), None

            @classmethod
            def tree_unflatten(cls, aux_data, children) -> Self:
                del aux_data
                return cls(children[0])

        @dataclass(frozen=True)
        class _WithPayload(_Wrapper):
            payload: Any = None
            pytree_children: ClassVar[tuple[str, ...]] = ("array", "payload")

        x = _WithPayload(jnp.ones(3), payload=_Container(np.ones(2)))
        with pytest.raises(TypeError, match=r"_WithPayload\.payload contains a NumPy array"):
            jax.tree.leaves(x)

    def test_numpy_scalars_remain_supported_dynamic_numbers(self):
        @dataclass(frozen=True)
        class _Scaled(_Wrapper):
            scale: Any = None
            pytree_children: ClassVar[tuple[str, ...]] = ("array", "scale")

        x = _Scaled(jnp.ones(3), scale=np.float32(2))
        result = jax.jit(lambda value: jnp.sum(value.array) * value.scale)(x)
        assert float(result) == pytest.approx(6.0)

    @pytest.mark.parametrize("compiled", [False, True])
    def test_jax_float0_cotangents_can_be_repackaged_and_traversed(self, compiled):
        gradient_function = jax.grad(lambda value: jnp.sum(value.array.astype(jnp.float32)), allow_int=True)
        if compiled:
            gradient_function = jax.jit(gradient_function)
        gradient = gradient_function(_Wrapper(jnp.arange(3, dtype=jnp.int32), "integer"))
        assert gradient.array.dtype == jax.dtypes.float0
        assert gradient.label == "integer"
        leaves = jax.tree.leaves(gradient)
        assert len(leaves) == 1
        assert leaves[0] is gradient.array

    def test_shaped_and_none_metadata_are_not_children(self):
        @dataclass(frozen=True)
        class _ShapeMetadata:
            shape: tuple[int, ...] = (37,)

        @dataclass(frozen=True, slots=True)
        class _WithMetadata(_Wrapper):
            metadata: _ShapeMetadata = field(default_factory=_ShapeMetadata)
            optional_metadata: None = None

        x = _WithMetadata(jnp.arange(6.0).reshape(2, 3))
        assert len(x.tree_flatten()[0]) == 1
        mapped = jax.jit(jax.vmap(lambda value: value))(x)
        assert mapped.metadata is x.metadata
        assert mapped.optional_metadata is None
        np.testing.assert_array_equal(mapped.array, x.array)

    @pytest.mark.parametrize(
        "metadata",
        [np.ones(2), jnp.ones(2), [], {}, set(), (np.ones(2),)],
    )
    def test_unsupported_static_metadata_reports_the_field(self, metadata):
        @dataclass(frozen=True)
        class _WithMetadata(_Wrapper):
            metadata: Any = None

        x = _WithMetadata(jnp.ones(3), metadata=metadata)
        with pytest.raises(TypeError, match=r"_WithMetadata\.metadata must be static, hashable metadata"):
            jax.tree.leaves(x)

    def test_array_metadata_is_rejected_even_with_an_identity_hash(self):
        class _HashableArray(np.ndarray):
            __hash__ = object.__hash__  # ty:ignore[invalid-assignment]

        @dataclass(frozen=True)
        class _WithMetadata(_Wrapper):
            metadata: Any = None

        x = _WithMetadata(jnp.ones(3), metadata=np.ones(2).view(_HashableArray))
        with pytest.raises(TypeError, match="without JAX/NumPy arrays"):
            x.tree_flatten()

    def test_changing_static_metadata_retraces(self):
        traced_labels = []

        @jax.jit
        def labeled_sum(value):
            traced_labels.append(value.label)
            return jnp.sum(value.array)

        array = jnp.ones(3)
        assert float(labeled_sum(_Wrapper(array, "one"))) == 3.0
        assert float(labeled_sum(_Wrapper(array * 2, "one"))) == 6.0
        assert float(labeled_sum(_Wrapper(array, "two"))) == 3.0
        assert traced_labels == ["one", "two"]

    def test_explicitly_empty_children_are_allowed(self):
        @dataclass(frozen=True)
        class _Static(_Wrapper):
            array: Any = None
            pytree_children: ClassVar[tuple[str, ...]] = ()

        original = _Static(label="static")
        assert jax.tree.leaves(original) == []
        rebuilt = jax.jit(lambda value: value)(original)
        assert rebuilt.array is None
        assert rebuilt.label == "static"

    def test_custom_flattening_does_not_require_the_default_schema(self):
        class _Custom(_Wrapper):
            pytree_children: ClassVar[Any] = None

            def tree_flatten(self):
                return (self.array,), self.label

            @classmethod
            def tree_unflatten(cls, aux_data, children) -> Self:
                return cls(children[0], aux_data)

        original = _Custom(jnp.arange(3.0), "custom")
        rebuilt = jax.jit(lambda value: value)(original)
        assert isinstance(rebuilt, _Custom)
        assert rebuilt.label == "custom"
        np.testing.assert_array_equal(rebuilt.array, original.array)

    def test_dataclass_post_init_is_skipped_for_transforms_and_invalid_gradients(self):
        constructed = []

        @dataclass(frozen=True, slots=True)
        class _Positive(_Wrapper):
            derived: str = field(init=False, default="checked")

            def __post_init__(self) -> None:
                constructed.append("post_init")
                if bool(jnp.any(self.array <= 0)):
                    msg = "primal must be positive"
                    raise ValueError(msg)

        x = _Positive(jnp.ones((2, 3)), "positive")
        identity = jax.jit(jax.vmap(lambda value: value))(x)
        assert identity.derived == "checked"
        gradient = jax.jit(jax.grad(lambda value: -jnp.sum(value.array)))(x)
        assert type(gradient) is _Positive
        assert gradient.label == "positive"
        assert gradient.derived == "checked"
        np.testing.assert_array_equal(gradient.array, -np.ones((2, 3)))
        assert constructed == ["post_init"]
        with pytest.raises(ValueError, match="primal must be positive"):
            _Positive(gradient.array)

    def test_custom_init_is_skipped(self):
        constructed = []

        class _CheckedInit(_Wrapper):
            def __init__(self, array, label="checked") -> None:
                constructed.append("init")
                super().__init__(array, label)

        x = _CheckedInit(jnp.ones(3))
        gradient = jax.grad(lambda value: -jnp.sum(value.array))(x)
        assert isinstance(gradient, _CheckedInit)
        np.testing.assert_array_equal(gradient.array, -np.ones(3))
        assert constructed == ["init"]


class TestNumpyUfuncCoercion:
    def test_implementation_does_not_define_numpy_ufunc_override(self):
        assert not hasattr(JaxLikeImplementation, "__array_ufunc__")

    def test_unary_ufunc_uses_array_conversion_not_representation_conversion(self):
        class _WithHook(_Wrapper):
            def __array_like__(self, dtype=None, /, *, copy=None) -> Never:
                del dtype, copy
                pytest.fail("NumPy coercion must not call the representation-preserving hook")

        x = _WithHook(jnp.arange(3.0))
        result = np.sin(x)
        assert type(result) is np.ndarray
        np.testing.assert_array_equal(result, np.sin(np.asarray(x)))

    @pytest.mark.parametrize("kind", ["scalar", "numpy", "wrapper"])
    @pytest.mark.parametrize("reflected", [False, True])
    def test_binary_ufunc_coercion_preserves_operand_order(self, kind, reflected):
        x = _Wrapper(jnp.arange(3.0))
        other = {"scalar": 2, "numpy": np.array([1.0, 2.0, 3.0]), "wrapper": _Wrapper(jnp.ones(3))}[kind]
        left, right = (other, x) if reflected else (x, other)
        result = np.subtract(left, right)
        assert type(result) is np.ndarray
        np.testing.assert_array_equal(result, np.subtract(np.asarray(left), np.asarray(right)))

    def test_native_numpy_output_buffer_is_respected(self):
        x = _Wrapper(jnp.ones(3))
        out = np.empty(3)
        assert np.add(x, 2, out=out) is out
        np.testing.assert_array_equal(out, 3)
        np.testing.assert_array_equal(np.asarray(x), 1)

    @pytest.mark.parametrize("operation", [lambda x: np.add(x, 1, out=x), lambda x: np.add.at(x, 0, 1)])
    def test_numpy_rejects_wrapper_mutation_targets(self, operation):
        x = _Wrapper(jnp.ones(3))
        with pytest.raises(TypeError):
            operation(x)
        np.testing.assert_array_equal(np.asarray(x), 1)

    def test_tracing_conversion_restrictions_propagate(self):
        with pytest.raises(jax.errors.TracerArrayConversionError):
            jax.jit(lambda array: np.sin(_Wrapper(array)))(jnp.ones(3))

    def test_array_conversion_errors_propagate_unchanged(self):
        error = ValueError("conversion unavailable")

        class _Broken(_Wrapper):
            def __array__(self, dtype=None, /, *, copy=None) -> Never:
                del dtype, copy
                raise error

        with pytest.raises(ValueError, match="conversion unavailable") as caught:
            np.sin(_Broken(jnp.ones(3)))
        assert caught.value is error

    @pytest.mark.parametrize("declines", [False, True])
    def test_explicit_ufunc_override_takes_precedence_over_coercion(self, declines):
        sentinel = object()

        class _Override(_Wrapper):
            def __array_ufunc__(self, ufunc, method, *inputs: object, **kwargs: object) -> object:
                del self, ufunc, method, inputs, kwargs
                return NotImplemented if declines else sentinel

        x = _Override(jnp.ones(3))
        if declines:
            with pytest.raises(TypeError):
                np.sin(x)
        else:
            assert np.sin(x) is sentinel

    def test_subclasses_can_explicitly_disable_numpy_ufuncs(self):
        class _NoNumpy(_Wrapper):
            __array_ufunc__ = None

        with pytest.raises(TypeError):
            np.sin(_NoNumpy(jnp.ones(3)))


class TestGenericOperators:
    @pytest.mark.parametrize("op", [operator.add, operator.sub, operator.mul, operator.truediv, operator.pow])
    @pytest.mark.parametrize("scalar", [False, True])
    def test_numpy_left_operand_is_not_forced_to_defer(self, op, scalar):
        x = _Wrapper(jnp.array([1.0, 2.0]))
        other = np.float64(3) if scalar else np.array([3.0, 4.0])
        result = op(other, x)
        assert isinstance(result, _Wrapper if scalar else np.ndarray)
        np.testing.assert_allclose(np.asarray(result), op(other, np.asarray(x)))

    @pytest.mark.parametrize("other", [2, np.array([2.0, 3.0]), jnp.array([2.0, 3.0])])
    def test_base_class_provides_binary_and_reflected_operators(self, other):
        x = _Wrapper(jnp.array([1.0, 4.0]))
        for index, (result, expected) in enumerate(
            [
                (x + other, x.array + other),
                (other + x, other + x.array),
                (x * other, x.array * other),
                (other * x, other * x.array),
                (other - x, other - x.array),
            ]
        ):
            numpy_left = isinstance(other, np.ndarray) and index in (1, 3, 4)
            assert isinstance(result, np.ndarray if numpy_left else _Wrapper)
            np.testing.assert_array_equal(np.asarray(result), expected)

    def test_generic_operators_compose_with_jit_vmap_and_grad(self):
        x = _Wrapper(jnp.arange(6.0).reshape(2, 3))
        transformed = jax.jit(jax.vmap(lambda value: value * value + 1))(x)
        np.testing.assert_array_equal(transformed.array, x.array**2 + 1)
        gradient = jax.grad(lambda value: jnp.sum((value * value).array))(x)
        np.testing.assert_array_equal(gradient.array, 2 * x.array)

    def test_unsupported_binary_operators_preserve_python_fallback(self):
        class _Denied(_Wrapper):
            @classmethod
            def __jax_function__(cls, func, types, args=(), kwargs=None):  # noqa: ANN206
                del cls, func, types, args, kwargs
                return NotImplemented

        class _Reflected:
            def __radd__(self, other) -> str:
                del other
                return "reflected"

        x = _Denied(jnp.ones(3))
        assert x + _Reflected() == "reflected"
        with pytest.raises(TypeError):
            _ = x * 2

    @pytest.mark.parametrize(
        "method",
        [
            f"__{prefix}{name}__"
            for name in (
                "add",
                "sub",
                "mul",
                "truediv",
                "floordiv",
                "mod",
                "pow",
                "matmul",
                "and",
                "or",
                "xor",
                "lshift",
                "rshift",
                "divmod",
            )
            for prefix in ("", "r")
        ]
        + [f"__{name}__" for name in ("eq", "ne", "lt", "le", "gt", "ge")],
    )
    def test_foreign_representations_are_declined_before_jax_dispatch(self, method):
        x = _Wrapper(jnp.ones(3))
        other = _NumpyArray(np.ones(3))
        assert getattr(x, method)(other) is NotImplemented

    @pytest.mark.parametrize(
        "op", [operator.add, operator.sub, operator.mul, operator.truediv, operator.pow, divmod, operator.matmul]
    )
    def test_foreign_operand_can_handle_the_reflected_operation(self, op):
        method = f"__r{op.__name__}__"
        other = type("Reflected", (), {method: lambda self, left: ("reflected", left, self)})()
        x = _Wrapper(jnp.ones(3))
        result = op(x, other)
        assert result[0] == "reflected"
        assert result[1] is x
        assert result[2] is other

    @pytest.mark.parametrize("op", [operator.add, operator.sub, operator.mul, operator.truediv, operator.pow])
    @pytest.mark.parametrize("reflected", [False, True])
    def test_numpy_representation_handles_mixed_infix_operations(self, op, reflected):
        x = _Wrapper(jnp.array([1, 2, 3]))
        other = _NumpyArray(np.array([2, 3, 4]))
        left, right = (other, x) if reflected else (x, other)
        result = op(left, right)
        assert isinstance(result, _NumpyArray)
        raw = np.asarray(x)
        expected = op(other.array, raw) if reflected else op(raw, other.array)
        np.testing.assert_array_equal(result.array, expected)
        np.testing.assert_array_equal(x.array, [1, 2, 3])
        with pytest.raises(TypeError):
            jax_add(x, other)  # ty: ignore[invalid-argument-type]

    @pytest.mark.parametrize(
        "op", [operator.add, operator.sub, operator.mul, operator.truediv, operator.pow, divmod, operator.matmul]
    )
    def test_implementation_errors_propagate_unchanged(self, op):
        error = TypeError("implementation error")

        class Broken(_Wrapper):
            @classmethod
            def __jax_function__(cls, func, types, args=(), kwargs=None) -> Never:
                del cls, func, types, args, kwargs
                raise error

        with pytest.raises(TypeError, match="implementation error") as caught:
            op(Broken(jnp.ones(3)), 2)
        assert caught.value is error

    def test_traced_scalar_operands_remain_supported(self):
        x = _Wrapper(jnp.ones(3))
        result = jax.jit(lambda value, scale: value + scale)(x, jnp.array(2.0))
        assert isinstance(result, _Wrapper)
        np.testing.assert_array_equal(result.array, 3)

    @pytest.mark.parametrize("scalar", [True, 2, 1.5, 2j, np.bool_(True), np.int64(2), np.float32(2)])
    def test_supported_scalar_families_remain_supported(self, scalar):
        x = _Wrapper(jnp.ones(3))
        result = x + scalar
        assert isinstance(result, _Wrapper)
        np.testing.assert_array_equal(result.array, x.array + scalar)

    def test_operand_check_does_not_coerce_foreign_objects_or_accept_non_callable_overrides(self):
        class Foreign:
            __jax_function__ = None

            def __array__(self, *args: object, **kwargs: object) -> Never:
                del args, kwargs
                pytest.fail("Operand checks must not coerce foreign objects")

        x = _Wrapper(jnp.ones(3))
        assert x.__add__(Foreign()) is NotImplemented
        assert x.__pow__(Foreign()) is NotImplemented


class TestRepresentationFields:
    def test_immutable_copy_hooks_cannot_cause_in_place_updates(self):
        class _Immutable(_Wrapper):
            def __copy__(self) -> Self:
                return self

        x = _Immutable(jnp.ones(3))
        converted = x.astype(jnp.int32)
        assert converted is not x
        assert converted.array.dtype == jnp.int32
        assert x.array.dtype == jnp.float32

    def test_convenience_methods_do_not_use_pytree_overrides_or_validate_metadata(self):
        @dataclass(frozen=True, slots=True)
        class _CustomTree(_Wrapper):
            metadata: Any = None

            def tree_flatten(self):
                msg = "array methods must not flatten the representation"
                raise AssertionError(msg)

            @classmethod
            def tree_unflatten(cls, aux_data, children) -> Self:
                del cls, aux_data, children
                msg = "array methods must not unflatten the representation"
                raise AssertionError(msg)

        metadata = np.arange(5.0)
        x = _CustomTree(jnp.arange(3.0), metadata=metadata)
        for mapped in (x.astype(jnp.int32), x.view(jnp.int32), x.stop_gradient(), x.to_device(jax.devices()[0])):
            assert type(mapped) is _CustomTree
            assert mapped.metadata is metadata
        assert x.block_until_ready() is x

    def test_representation_fields_can_be_selected_independently(self):
        @dataclass(frozen=True)
        class _DifferentFields(_Wrapper):
            other: Any = None

            @classmethod
            def _array_field_names(cls) -> tuple[str, ...]:
                del cls
                return ("other",)

        x = _DifferentFields(jnp.ones(3), other=jnp.ones(5))
        converted = x.astype(jnp.int32)
        assert converted.array is x.array
        assert converted.other.dtype == jnp.int32

    def test_optional_array_fields_are_preserved(self):
        @dataclass(frozen=True)
        class _Optional(_Wrapper):
            weights: jax.Array | None = None
            pytree_children: ClassVar[tuple[str, ...]] = ("array", "weights")

        x = _Optional(jnp.ones(3))
        assert x.astype(jnp.int32).weights is None
        assert x.stop_gradient().weights is None
        assert x.block_until_ready() is x

    def test_nested_convenience_methods_respect_the_inner_representation(self):
        @dataclass(frozen=True, slots=True)
        class _Inner(_Wrapper):
            metadata: Any = None

            def tree_flatten(self):
                msg = "array methods must not flatten nested representations"
                raise AssertionError(msg)

        @dataclass(frozen=True)
        class _Outer(_Wrapper):
            array: Any

        metadata = np.ones(5)
        inner = _Inner(jnp.ones(3), metadata=metadata)
        outer = _Outer(inner)
        for mapped in (outer.stop_gradient(), outer.view(jnp.int32), outer.to_device(jax.devices()[0])):
            assert type(mapped.array) is _Inner
            assert mapped.array.metadata is metadata
        assert outer.block_until_ready() is outer


class _AdditiveMultiply(_Wrapper):
    """Distinguish custom multiplication from structural leaf multiplication."""

    @classmethod
    def __jax_function__(cls, func, types, args=(), kwargs=None):  # noqa: ANN206
        if func is jax_multiply:
            return cls(jnp.add(*(_unwrap(arg) for arg in args)))
        return super().__jax_function__(func, types, args, kwargs)


class TestJaxTreeMap:
    def test_opaque_mapping_uses_custom_operators(self):
        x = _AdditiveMultiply(jnp.array([1.0, 3.0]))
        mapped = jax_tree_map(lambda value: value * 2, {"value": x, "array": jnp.array([2.0])})
        assert isinstance(mapped["value"], _AdditiveMultiply)
        np.testing.assert_array_equal(mapped["value"].array, [3.0, 5.0])
        np.testing.assert_array_equal(mapped["array"], [4.0])
        structural = jax.tree.map(lambda value: value * 2, x)
        np.testing.assert_array_equal(structural.array, [2.0, 6.0])

    def test_class_flag_requests_structural_traversal(self):
        class _Traversable(_AdditiveMultiply):
            tree_map_traverse: ClassVar[bool] = True

        mapped = jax_tree_map(lambda value: value * 2, _Traversable(jnp.array([1.0, 3.0])))
        assert isinstance(mapped, _Traversable)
        np.testing.assert_array_equal(mapped.array, [2.0, 6.0])

    @pytest.mark.parametrize("class_traverse", [False, True])
    @pytest.mark.parametrize("call_traverse", [False, True])
    def test_per_call_traversal_overrides_the_class_flag(self, class_traverse, call_traverse):
        cls = type("_Configurable", (_AdditiveMultiply,), {"tree_map_traverse": class_traverse})
        mapped = jax_tree_map(lambda value: value * 2, cls(jnp.array([1.0, 3.0])), traverse_jaxlikes=call_traverse)
        np.testing.assert_array_equal(mapped.array, [2.0, 6.0] if call_traverse else [3.0, 5.0])

    def test_leaf_predicate_takes_precedence(self):
        x = _AdditiveMultiply(jnp.array([1.0, 3.0]))
        mapped = jax_tree_map(
            lambda value: value * 2,
            x,
            traverse_jaxlikes=True,
            is_leaf=lambda value: isinstance(value, _AdditiveMultiply),
        )
        np.testing.assert_array_equal(mapped.array, [3.0, 5.0])
        assert jax_tree_map(len, [x], is_leaf=lambda value: isinstance(value, list)) == 1

    def test_multiple_trees_can_have_different_metadata_at_opaque_nodes(self):
        left = {"value": _Wrapper(jnp.array([1.0]), "left")}
        right = {"value": _Wrapper(jnp.array([3.0]), "right")}
        mapped = jax_tree_map(lambda x, y: x + y, left, right)
        np.testing.assert_array_equal(mapped["value"].array, [4.0])

    def test_nested_implementations_use_their_own_flags(self):
        @dataclass(frozen=True)
        class _Outer(_Wrapper):
            array: Any
            tree_map_traverse: ClassVar[bool] = True

        inner = _Wrapper(jnp.ones(3))
        seen = []

        def visit(value):
            seen.append(value)
            return value

        mapped = jax_tree_map(visit, _Outer(inner))
        assert len(seen) == 1
        assert seen[0] is inner
        assert mapped.array is inner

    def test_helper_composes_with_jit_and_grad(self):
        def loss(value):
            mapped = jax_tree_map(lambda leaf: leaf * leaf, {"value": value})
            return jnp.sum(mapped["value"].array)

        x = _Wrapper(jnp.arange(3.0))
        gradient = jax.jit(jax.grad(loss))(x)
        np.testing.assert_array_equal(gradient.array, 2 * x.array)
