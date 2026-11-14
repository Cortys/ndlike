"""Static contracts for owner, operand, assignment, and result specialization."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Never, assert_type, get_args

import numpy as np

from ndlike.array_like import ArrayLike, NumpyLike, NumpyLikeImplementation, WeakArrayLike
from ndlike.protected_axis.numpy import NumpyAxisProtected

if TYPE_CHECKING:
    import jax
    from numpy.typing import NDArray
    import torch

    from ndlike import jax_functions as jf
    from ndlike.jax_like import JaxLike, JaxLikeImplementation
    from ndlike.protected_axis.jax import JaxAxisProtected
    from ndlike.protected_axis.torch import TorchAxisProtected
    from ndlike.torch_functions import torch_average
    from ndlike.torch_like import TorchLike, TorchLikeImplementation

    class _Numpy(NumpyLikeImplementation["_Numpy", float, str, bytes, np.ndarray, bool, int]): ...

    class _Jax(JaxLikeImplementation["_Jax", float, str, bytes, np.ndarray, bool, int]): ...

    class _Torch(TorchLikeImplementation["_Torch", float, str, bytes, torch.Tensor, bool, int]): ...

    class _Vector[T: "_Vector"](JaxLikeImplementation[T]): ...

    class _Leaf(_Vector["_Leaf"]): ...

    class _NumpyProtected(NumpyAxisProtected["_NumpyProtected", float, str, bytes, np.ndarray, bool, int]): ...

    class _JaxProtected(JaxAxisProtected["_JaxProtected", float, str, bytes, np.ndarray, bool, int]): ...

    class _TorchProtected(TorchAxisProtected["_TorchProtected", float, str, bytes, torch.Tensor, bool, int]): ...

    class _NumpyOwner(NumpyAxisProtected["_NumpyOwner"]): ...

    class _ProtectedVector[T: "_ProtectedVector"](JaxAxisProtected[T]): ...

    class _JaxOwner(_ProtectedVector["_JaxOwner"]): ...

    class _TorchOwner(TorchAxisProtected["_TorchOwner"]): ...

    def _check_protected_defaults(numpy: _NumpyOwner, jax_array: _JaxOwner, tensor: _TorchOwner) -> None:
        assert_type(numpy + numpy, _NumpyOwner)
        assert_type(numpy < numpy, NDArray[np.bool_])  # noqa: PLR0124
        assert_type(numpy.T, _NumpyOwner)
        assert_type(numpy.reshape(2, 3), _NumpyOwner)
        assert_type(numpy[0], _NumpyOwner)
        assert_type(next(iter(numpy)), _NumpyOwner)
        assert_type(numpy.copy(), _NumpyOwner)
        assert_type(jax_array + jax_array, _JaxOwner)
        assert_type(jax_array < jax_array, jax.Array)  # noqa: PLR0124
        assert_type(jax_array.reshape(2, 3), _JaxOwner)
        assert_type(jax_array[0], _JaxOwner)
        assert_type(next(iter(jax_array)), _JaxOwner)
        assert_type(jax_array.copy(), _JaxOwner)
        assert_type(jax_array.at[0].get(), _JaxOwner)
        assert_type(jax_array.at[0].set(jax_array), _JaxOwner)
        assert_type(tensor + tensor, _TorchOwner)
        assert_type(tensor < tensor, torch.Tensor)  # noqa: PLR0124
        assert_type(tensor.reshape(2, 3), _TorchOwner)
        assert_type(tensor[0], _TorchOwner)
        assert_type(next(iter(tensor)), _TorchOwner)
        assert_type(tensor.clone(), _TorchOwner)
        numpy[0] = numpy
        tensor[0] = tensor
        numpy + 1.0  # ty: ignore[unsupported-operator]
        jax_array + 1.0  # ty: ignore[unsupported-operator]
        tensor + 1.0  # ty: ignore[unsupported-operator]
        numpy[0] = 1.0  # ty: ignore[invalid-assignment]
        tensor[0] = 1.0  # ty: ignore[invalid-assignment]
        jax_array.at[0].set(1.0)  # ty: ignore[invalid-argument-type]

    def _check_protected_results(numpy: _NumpyProtected, jax_array: _JaxProtected, tensor: _TorchProtected) -> None:
        assert_type(numpy + 1.0, str)
        assert_type(numpy < 1.0, bool)
        assert_type(numpy.T, np.ndarray)
        assert_type(numpy.reshape(2, 3), np.ndarray)
        assert_type(numpy.astype(np.float32), str)
        assert_type(numpy[0], np.ndarray)
        assert_type(next(iter(numpy)), np.ndarray)
        assert_type(numpy.copy(), _NumpyProtected)
        assert_type(jf.jax_add(jax_array, 1.0), str)
        assert_type(jf.jax_sum(jax_array), str)
        assert_type(jax_array < 1.0, bool)
        assert_type(jax_array.reshape(2, 3), np.ndarray)
        assert_type(jax_array.mT, np.ndarray)
        assert_type(jax_array.mH, np.ndarray)
        assert_type(jax_array.take_along_axis(jax.numpy.array([0])), np.ndarray)
        assert_type(jax_array[0], np.ndarray)
        assert_type(next(iter(jax_array)), np.ndarray)
        assert_type(jf.jax_copy(jax_array), _JaxProtected)
        assert_type(jax_array.copy(), _JaxProtected)
        assert_type(jax_array.at[0].get(), np.ndarray)
        for update in (
            jax_array.at[0].set,
            jax_array.at[0].add,
            jax_array.at[0].subtract,
            jax_array.at[0].multiply,
            jax_array.at[0].divide,
            jax_array.at[0].power,
            jax_array.at[0].min,
            jax_array.at[0].max,
        ):
            assert_type(update(b"value"), int)
            assert_type(update(jax_array), int)
            update(1.0)  # ty: ignore[invalid-argument-type]
        assert_type(tensor + 1.0, str)
        assert_type(torch_average(tensor), str)
        assert_type(tensor < 1.0, bool)
        assert_type(tensor.mT, torch.Tensor)
        assert_type(tensor.mH, torch.Tensor)
        assert_type(tensor.reshape(2, 3), torch.Tensor)
        assert_type(tensor.gather(0, torch.tensor([0])), torch.Tensor)
        assert_type(tensor[0], torch.Tensor)
        assert_type(next(iter(tensor)), torch.Tensor)
        assert_type(tensor.clone(), _TorchProtected)
        numpy[0] = b"value"
        tensor[0] = b"value"
        numpy[0] = 1.0  # ty: ignore[invalid-assignment]
        tensor[0] = 1.0  # ty: ignore[invalid-assignment]

    def _check_protected_owner_bounds(
        numpy: NumpyAxisProtected[int],  # ty: ignore[invalid-type-arguments]
        jax_owner: JaxAxisProtected[torch.Tensor],  # ty: ignore[invalid-type-arguments]
        torch_owner: TorchAxisProtected[jax.Array],  # ty: ignore[invalid-type-arguments]
    ) -> None:
        del numpy, jax_owner, torch_owner

    def _check_indexing_contracts(
        core: ArrayLike[Any, Any, Any, Never, str],
        weak: WeakArrayLike[Any, Any, Any, Never, str],
        numpy: NumpyLike[np.ndarray, float, int, bytes, str],
        jax_array: JaxLike[jax.Array, float, int, bytes, str],
        torch_tensor: TorchLike[torch.Tensor, float, int, bytes, str],
    ) -> None:
        core + core  # ty: ignore[unsupported-operator]
        assert_type(core[0], str)
        assert_type(next(iter(core)), str)
        assert_type(weak[0], str)
        assert_type(next(iter(weak)), str)
        assert_type(numpy[0], str)
        assert_type(next(iter(numpy)), str)
        assert_type(numpy.copy(), np.ndarray)
        assert_type(jax_array[0], str)
        assert_type(next(iter(jax_array)), str)
        assert_type(jax_array.copy(), jax.Array)
        assert_type(jax_array.at[0].get(), str)
        assert_type(torch_tensor[0], str)
        assert_type(next(iter(torch_tensor)), str)
        assert_type(torch_tensor.clone(), torch.Tensor)

    def _check_owner_defaults(array: NumpyLike, jax_array: JaxLike, tensor: TorchLike, leaf: _Leaf) -> None:
        assert_type(array + array, NumpyLike[Any])
        assert_type(array == array, NDArray[np.bool_])  # noqa: PLR0124 - check the self-operand contract
        assert_type(jax_array + jax_array, JaxLike[Any])
        assert_type(jax_array < jax_array, jax.Array)  # noqa: PLR0124 - check the self-operand contract
        assert_type(tensor + tensor, TorchLike[Any])
        assert_type(tensor != tensor, torch.Tensor)  # noqa: PLR0124 - check the self-operand contract
        assert_type(leaf + leaf, _Leaf)
        assert_type(leaf.reshape(2, 3), _Leaf)
        assert_type(leaf[0], _Leaf)
        assert_type(next(iter(leaf)), _Leaf)
        assert_type(leaf.copy(), _Leaf)
        assert_type(leaf.at[0].set(leaf), _Leaf)
        leaf + 1.0  # ty: ignore[unsupported-operator]

    def _check_numpy(array: _Numpy) -> None:
        assert_type(array + 1.0, str)
        assert_type(1.0 - array, str)
        assert_type(-array, str)
        assert_type(array == 1.0, bool)
        assert_type(array.copy(), _Numpy)
        assert_type(array[0], np.ndarray)
        assert_type(next(iter(array)), np.ndarray)
        assert_type(array.astype(np.float32), str)
        assert_type(array.T, np.ndarray)
        assert_type(divmod(array, 2.0), tuple[str, str])
        assert_type(divmod(2.0, array), tuple[str, str])
        array[0] = array
        array[0] = b"value"
        array[0] = 1.0  # ty: ignore[invalid-assignment]
        array + b"wrong"  # ty: ignore[unsupported-operator]

    def _check_inplace_protocols(
        numpy: NumpyLike[np.ndarray, float, str], tensor: TorchLike[torch.Tensor, float, str]
    ) -> None:
        numpy += 1.0
        tensor += 1.0
        assert_type(numpy, NumpyLike[np.ndarray, float, str])
        assert_type(tensor, TorchLike[torch.Tensor, float, str])
        numpy @= 1.0
        assert_type(numpy, NumpyLike[np.ndarray, float, str])

    def _check_numpy_inplace(array: _Numpy) -> None:
        for update in (
            array.__iadd__,
            array.__isub__,
            array.__imul__,
            array.__itruediv__,
            array.__ifloordiv__,
            array.__imod__,
            array.__ipow__,
            array.__imatmul__,
            array.__iand__,
            array.__ior__,
            array.__ixor__,
            array.__ilshift__,
            array.__irshift__,
        ):
            assert_type(update(1.0), _Numpy)
        array += 1.0
        assert_type(array, _Numpy)
        array *= 2.0
        assert_type(array, _Numpy)
        array += b"wrong"  # ty: ignore[unsupported-operator]

    def _check_torch_inplace(array: _Torch) -> None:
        for update in (
            array.__iadd__,
            array.__isub__,
            array.__imul__,
            array.__itruediv__,
            array.__ifloordiv__,
            array.__imod__,
            array.__ipow__,
            array.__imatmul__,
            array.__iand__,
            array.__ior__,
            array.__ixor__,
            array.__ilshift__,
            array.__irshift__,
        ):
            assert_type(update(1.0), _Torch)
        array += 1.0
        assert_type(array, _Torch)
        array *= 2.0
        assert_type(array, _Torch)
        array += b"wrong"  # ty: ignore[unsupported-operator]

    def _check_jax(array: _Jax, native: jax.Array) -> None:
        assert_type(array + 1.0, str)
        assert_type(array < 1.0, bool)
        assert_type(array.reshape(2, 3), np.ndarray)
        assert_type(array.at[0].set(b"value"), int)
        assert_type(array.at[0].add(array), int)
        assert_type(array.at[0].get(), np.ndarray)
        assert_type(divmod(array, 2.0), tuple[str, str])
        assert_type(divmod(2.0, array), tuple[str, str])
        assert_type(jf.jax_divmod(array, 2.0), tuple[str, str])
        assert_type(jf.jax_divmod(2.0, array), tuple[str, str])
        assert_type(jf.jax_frexp(array), tuple[str, str])
        assert_type(jf.jax_modf(array), tuple[str, str])
        assert_type(jf.jax_add(array, 1.0), str)
        assert_type(jf.jax_add(1.0, array), str)
        assert_type(jf.jax_sum(array), str)
        assert_type(jf.jax_less(array, 1.0), bool)
        assert_type(jf.jax_copy(array), _Jax)
        assert_type(array.copy(), _Jax)
        assert_type(array[0], np.ndarray)
        assert_type(next(iter(array)), np.ndarray)
        assert_type(jf.jax_copy(native), jax.Array)
        assert_type(jf.jax_reshape(array, (2, 3)), np.ndarray)
        assert_type(jf.jax_reshape(array, (None, 3)), np.ndarray)
        assert_type(jf.jax_add(native, 1.0), jax.Array)
        assert_type(jf.jax_sum(native), jax.Array)
        array[0] = b"value"
        array[0] = 1.0  # ty: ignore[invalid-assignment]
        array.at[0].set(1.0)  # ty: ignore[invalid-argument-type]
        jf.jax_add(array, b"wrong")  # ty: ignore[no-matching-overload]

    def _check_torch(array: _Torch, native: torch.Tensor) -> None:
        assert_type(array + 1.0, str)
        assert_type(array > 1.0, bool)
        assert_type(array.clone(), _Torch)
        assert_type(array[0], torch.Tensor)
        assert_type(next(iter(array)), torch.Tensor)
        assert_type(divmod(array, 2.0), tuple[str, str])
        assert_type(divmod(2.0, array), tuple[str, str])
        assert_type(array.resolve_conj(), str)
        assert_type(torch_average(array), str)
        assert_type(torch_average(native), torch.Tensor)
        array[0] = array
        array[0] = b"value"
        array[0] = 1.0  # ty: ignore[invalid-assignment]

    def _check_native_result_contexts(array: jax.Array, numpy: np.ndarray, tensor: torch.Tensor) -> None:
        # A destination annotation must not determine a native operation's result.
        value: str
        pair: tuple[str, str]
        copied: _Jax
        value = jf.jax_positive(array)  # ty: ignore[invalid-assignment]
        value = jf.jax_negative(array)  # ty: ignore[invalid-assignment]
        value = jf.jax_absolute(array)  # ty: ignore[invalid-assignment]
        value = jf.jax_invert(array)  # ty: ignore[invalid-assignment]
        pair = jf.jax_divmod(array, 2)  # ty: ignore[invalid-assignment]
        pair = jf.jax_frexp(array)  # ty: ignore[invalid-assignment]
        pair = jf.jax_modf(array)  # ty: ignore[invalid-assignment]
        value = jf.jax_broadcast_to(array, (2, 3))  # ty: ignore[invalid-assignment]
        value = jf.jax_reshape(array, (2, 3))  # ty: ignore[invalid-assignment]
        value = jf.jax_transpose(array)  # ty: ignore[invalid-assignment]
        value = jf.jax_matrix_transpose(array)  # ty: ignore[invalid-assignment]
        value = jf.jax_moveaxis(array, 0, 1)  # ty: ignore[invalid-assignment]
        value = jf.jax_swapaxes(array, 0, 1)  # ty: ignore[invalid-assignment]
        value = jf.jax_squeeze(array)  # ty: ignore[invalid-assignment]
        value = jf.jax_expand_dims(array, 0)  # ty: ignore[invalid-assignment]
        value = jf.jax_concatenate([array, numpy])  # ty: ignore[invalid-assignment]
        value = jf.jax_stack([array, numpy])  # ty: ignore[invalid-assignment]
        value = jf.jax_take_along_axis(array, array)  # ty: ignore[invalid-assignment]
        value = jf.jax_conj(array)  # ty: ignore[invalid-assignment]
        copied = jf.jax_copy(array)  # ty: ignore[invalid-assignment]
        copied = jf.jax_copy(numpy)  # ty: ignore[invalid-assignment]
        copied = jf.jax_copy(1.0)  # ty: ignore[invalid-assignment]
        value = jf.jax_astype(array, "float32")  # ty: ignore[invalid-assignment]
        value = jf.jax_mean(array)  # ty: ignore[invalid-assignment]
        value = jf.jax_sum(array)  # ty: ignore[invalid-assignment]
        value = jf.jax_sum(numpy)  # ty: ignore[invalid-assignment]
        value = jf.jax_sum(1.0)  # ty: ignore[invalid-assignment]
        value = jf.jax_sum(array, initial=array, where=array)  # ty: ignore[invalid-assignment]
        value = jf.jax_std(array)  # ty: ignore[invalid-assignment]
        value = jf.jax_var(array)  # ty: ignore[invalid-assignment]
        value = jf.jax_average(array)  # ty: ignore[invalid-assignment]
        value = torch_average(tensor)  # ty: ignore[invalid-assignment]
        value = torch_average(tensor, weights=tensor, out=tensor)  # ty: ignore[invalid-assignment]
        del value, pair, copied

    def _check_secondary_result_contracts(
        array: _Jax, native: jax.Array, tensor: _Torch, native_tensor: torch.Tensor
    ) -> None:
        assert_type(jf.jax_mean(native, where=array), str)
        assert_type(jf.jax_sum(native, initial=array), str)
        assert_type(jf.jax_sum(native, where=array), str)
        assert_type(jf.jax_std(native, where=array), str)
        assert_type(jf.jax_var(native, where=array), str)
        assert_type(jf.jax_average(native, weights=array), str | tuple[str, str])
        assert_type(jf.jax_concatenate([native, array]), np.ndarray)
        assert_type(jf.jax_stack([native, array]), np.ndarray)
        assert_type(jf.jax_take_along_axis(native, array), np.ndarray)
        assert_type(torch_average(native_tensor, weights=tensor), str)
        assert_type(torch_average(native_tensor, out=tensor), str)

    def _check_native_owners(
        numpy: np.ndarray, jax_array: jax.Array, torch_tensor: torch.Tensor, boolean: NDArray[np.bool_]
    ) -> tuple[
        NumpyLike[np.ndarray],
        JaxLike[jax.Array],
        TorchLike[torch.Tensor],
        WeakArrayLike[Any],
        ArrayLike[NDArray[np.bool_]],
    ]:
        return numpy, jax_array, torch_tensor, boolean, boolean

    def _check_invalid_owners(
        nonarray: ArrayLike[int],  # ty: ignore[invalid-type-arguments]
        jax_owner: JaxLike[torch.Tensor],  # ty: ignore[invalid-type-arguments]
        torch_owner: TorchLike[jax.Array],  # ty: ignore[invalid-type-arguments]
    ) -> None:
        del nonarray, jax_owner, torch_owner


def test_dependent_defaults_are_substituted_at_runtime() -> None:
    args = get_args(ArrayLike[np.ndarray])
    assert args[0] is np.ndarray
    assert args[1] is args[2] is args[4] is args[6] is np.ndarray
    assert args[3] is Never


def test_generic_subclasses_can_forward_only_the_owner() -> None:
    class Vector[T: "Vector"](ArrayLike[T]):
        pass

    class Leaf(Vector["Leaf"]):
        pass

    assert vars(Vector)["__parameters__"] == Vector.__type_params__
    assert vars(Leaf)["__parameters__"] == ()


def test_protected_generic_subclasses_can_forward_only_the_owner() -> None:
    from abc import abstractmethod  # noqa: PLC0415

    class Vector[T: "Vector"](NumpyAxisProtected[T]):
        @abstractmethod
        def marker(self) -> None: ...

    class Leaf(Vector["Leaf"]):
        pass

    assert vars(Vector)["__parameters__"] == Vector.__type_params__
    assert vars(Leaf)["__parameters__"] == ()
    assert [param.__name__ for param in NumpyAxisProtected.__type_params__] == [
        param.__name__ for param in NumpyLike.__type_params__
    ]
    assert get_args(NumpyAxisProtected[np.ndarray]) == get_args(NumpyLike[np.ndarray])
