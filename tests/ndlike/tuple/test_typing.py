"""Bounded member tuples retain positions, lengths, and array owner types."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, assert_type

from ndlike.tuple.numpy import NumpyBroadcastTuple

if TYPE_CHECKING:
    import jax
    import numpy as np
    import torch

    from ndlike import jax_functions as jf
    from ndlike.array_like import AnyNumpyLike
    from ndlike.jax_like import AnyJaxLike
    from ndlike.torch_like import AnyTorchLike
    from ndlike.tuple.jax import JaxBroadcastTuple
    from ndlike.tuple.torch import TorchBroadcastTuple

    def _check_empty() -> None:
        assert_type(NumpyBroadcastTuple(()), NumpyBroadcastTuple[tuple[()]])
        assert_type(JaxBroadcastTuple(()), JaxBroadcastTuple[tuple[()]])
        assert_type(TorchBroadcastTuple(()), TorchBroadcastTuple[tuple[()]])
        assert_type(NumpyBroadcastTuple(())[42].members, tuple[()])
        assert_type(NumpyBroadcastTuple(()).swapaxes(0, 1).members, tuple[()])
        assert_type(JaxBroadcastTuple(()).reshape(3, 4).members, tuple[()])
        assert_type(JaxBroadcastTuple(()).swapaxes(0, 1).members, tuple[()])
        assert_type(TorchBroadcastTuple(()).T.members, tuple[()])
        assert_type(TorchBroadcastTuple(()).unsqueeze(0).members, tuple[()])
        assert_type(TorchBroadcastTuple(()).movedim(0, 1).members, tuple[()])
        assert_type(TorchBroadcastTuple(()).broadcast_to((2, 3)).members, tuple[()])
        assert_type(TorchBroadcastTuple(()).dim(), int)
        assert_type(TorchBroadcastTuple(()).numel(), int)

    def _check_numpy[T1: AnyNumpyLike | np.ndarray | np.generic, T2: AnyNumpyLike | np.ndarray | np.generic](
        first: T1, second: T2
    ) -> None:
        array = NumpyBroadcastTuple((first, second))
        assert_type(array, NumpyBroadcastTuple[tuple[T1, T2]])
        assert_type(array.members, tuple[T1, T2])
        assert_type(array.members[0], T1)
        assert_type(array.members[1], T2)
        assert_type(array[0], NumpyBroadcastTuple[tuple[T1, T2]])
        assert_type(array.reshape(2, 3), NumpyBroadcastTuple[tuple[T1, T2]])
        assert_type(array.swapaxes(0, 1), NumpyBroadcastTuple[tuple[T1, T2]])
        assert_type(array.squeeze(), NumpyBroadcastTuple[tuple[T1, T2]])
        assert_type(array.T, NumpyBroadcastTuple[tuple[T1, T2]])
        assert_type(array.with_members((first, second)), NumpyBroadcastTuple[tuple[T1, T2]])
        assert_type(next(iter(array)), NumpyBroadcastTuple[tuple[T1, T2]])
        assert_type(array.__array_like__(), NumpyBroadcastTuple[tuple[T1, T2]])
        assert_type(array.__array_like__(copy=True), NumpyBroadcastTuple[tuple[T1, T2]])
        assert_type(NumpyBroadcastTuple((first,)), NumpyBroadcastTuple[tuple[T1]])
        array.with_members((second, first))  # ty: ignore[invalid-argument-type]
        NumpyBroadcastTuple[tuple[T1, T2]]((first,))  # ty: ignore[invalid-argument-type]

    def _check_jax[T1: AnyJaxLike | jax.Array, T2: AnyJaxLike | jax.Array](first: T1, second: T2) -> None:
        array = JaxBroadcastTuple((first, second))
        assert_type(array, JaxBroadcastTuple[tuple[T1, T2]])
        assert_type(array.members, tuple[T1, T2])
        assert_type(array.members[0], T1)
        assert_type(array.members[1], T2)
        assert_type(array[0], JaxBroadcastTuple[tuple[T1, T2]])
        assert_type(array.reshape(2, 3), JaxBroadcastTuple[tuple[T1, T2]])
        assert_type(array.swapaxes(0, 1), JaxBroadcastTuple[tuple[T1, T2]])
        assert_type(array.swapaxes(axis1=0, axis2=1), JaxBroadcastTuple[tuple[T1, T2]])
        assert_type(array.T, JaxBroadcastTuple[tuple[T1, T2]])
        assert_type(jf.jax_reshape(array, (2, 3)), JaxBroadcastTuple[tuple[T1, T2]])
        assert_type(array.at[0].get(), JaxBroadcastTuple[tuple[T1, T2]])
        assert_type(array.at[0].set(1), JaxBroadcastTuple[tuple[T1, T2]])
        assert_type(array.__jax_like__(), JaxBroadcastTuple[tuple[T1, T2]])
        assert_type(array.__jax_like__(copy=True), JaxBroadcastTuple[tuple[T1, T2]])
        assert_type(JaxBroadcastTuple((first,)), JaxBroadcastTuple[tuple[T1]])
        array.with_members((first,))  # ty: ignore[invalid-argument-type]
        JaxBroadcastTuple[tuple[T1, T2]]((second, first))  # ty: ignore[invalid-argument-type]

    def _check_torch[T1: AnyTorchLike | torch.Tensor, T2: AnyTorchLike | torch.Tensor](first: T1, second: T2) -> None:
        array = TorchBroadcastTuple((first, second))
        assert_type(array, TorchBroadcastTuple[tuple[T1, T2]])
        assert_type(array.members, tuple[T1, T2])
        assert_type(array.members[0], T1)
        assert_type(array.members[1], T2)
        assert_type(array[0], TorchBroadcastTuple[tuple[T1, T2]])
        assert_type(array.reshape(2, 3), TorchBroadcastTuple[tuple[T1, T2]])
        assert_type(array.unsqueeze(0), TorchBroadcastTuple[tuple[T1, T2]])
        assert_type(array.movedim(0, 1), TorchBroadcastTuple[tuple[T1, T2]])
        assert_type(array.broadcast_to((2, 3)), TorchBroadcastTuple[tuple[T1, T2]])
        assert_type(array.broadcast_to(size=(2, 3)), TorchBroadcastTuple[tuple[T1, T2]])
        assert_type(array.broadcast_to(2, 3), TorchBroadcastTuple[tuple[T1, T2]])
        assert_type(array.squeeze(), TorchBroadcastTuple[tuple[T1, T2]])
        assert_type(array.dim(), int)
        assert_type(array.ndimension(), int)
        assert_type(array.numel(), int)
        assert_type(array.nelement(), int)
        assert_type(array.T, TorchBroadcastTuple[tuple[T1, T2]])
        assert_type(array.__torch_like__(), TorchBroadcastTuple[tuple[T1, T2]])
        assert_type(array.__torch_like__(copy=True), TorchBroadcastTuple[tuple[T1, T2]])
        assert_type(TorchBroadcastTuple((first,)), TorchBroadcastTuple[tuple[T1]])
        array.with_members((first, second, first))  # ty: ignore[invalid-argument-type]
        TorchBroadcastTuple[tuple[T1, T2]]((first, second, first))  # ty: ignore[invalid-argument-type]

    def _check_variable_numpy[T: AnyNumpyLike | np.ndarray | np.generic](members: tuple[T, ...]) -> None:
        array = NumpyBroadcastTuple(members)
        assert_type(array, NumpyBroadcastTuple[tuple[T, ...]])
        assert_type(array.members[0], T)
        assert_type(array.reshape(2, 3).members, tuple[T, ...])

    def _check_variable_jax[T: AnyJaxLike | jax.Array](members: tuple[T, ...]) -> None:
        array = JaxBroadcastTuple(members)
        assert_type(array, JaxBroadcastTuple[tuple[T, ...]])
        assert_type(array.members[0], T)
        assert_type(array.at[0].get().members, tuple[T, ...])

    def _check_variable_torch[T: AnyTorchLike | torch.Tensor](members: tuple[T, ...]) -> None:
        array = TorchBroadcastTuple(members)
        assert_type(array, TorchBroadcastTuple[tuple[T, ...]])
        assert_type(array.members[0], T)
        assert_type(array.reshape(2, 3).members, tuple[T, ...])

    def _check_forwarding_numpy[Members: tuple[AnyNumpyLike | np.ndarray | np.generic, ...]](members: Members) -> None:
        array = NumpyBroadcastTuple(members)
        assert_type(array, NumpyBroadcastTuple[Members])
        assert_type(array.members, Members)
        assert_type(array.reshape(2, 3).members, Members)

    def _check_forwarding_jax[Members: tuple[AnyJaxLike | jax.Array, ...]](members: Members) -> None:
        array = JaxBroadcastTuple(members)
        assert_type(array, JaxBroadcastTuple[Members])
        assert_type(array.members, Members)
        assert_type(array.at[0].set(1).members, Members)

    def _check_forwarding_torch[Members: tuple[AnyTorchLike | torch.Tensor, ...]](members: Members) -> None:
        array = TorchBroadcastTuple(members)
        assert_type(array, TorchBroadcastTuple[Members])
        assert_type(array.members, Members)
        assert_type(array.reshape(2, 3).members, Members)

    def _check_rejected_members(numpy: np.ndarray, jax_array: jax.Array, tensor: torch.Tensor) -> None:
        NumpyBroadcastTuple([numpy])  # ty: ignore[invalid-argument-type]
        JaxBroadcastTuple([jax_array])  # ty: ignore[invalid-argument-type]
        TorchBroadcastTuple([tensor])  # ty: ignore[invalid-argument-type]
        NumpyBroadcastTuple((1,))  # ty: ignore[invalid-argument-type]
        JaxBroadcastTuple((1,))  # ty: ignore[invalid-argument-type]
        TorchBroadcastTuple((1,))  # ty: ignore[invalid-argument-type]
        NumpyBroadcastTuple((numpy, "invalid"))  # ty: ignore[invalid-argument-type]
        JaxBroadcastTuple((jax_array, "invalid"))  # ty: ignore[invalid-argument-type]
        TorchBroadcastTuple((tensor, "invalid"))  # ty: ignore[invalid-argument-type]
        NumpyBroadcastTuple((tensor,))  # ty: ignore[invalid-argument-type]
        JaxBroadcastTuple((numpy,))  # ty: ignore[invalid-argument-type]
        TorchBroadcastTuple((jax_array,))  # ty: ignore[invalid-argument-type]
        _invalid_numpy: NumpyBroadcastTuple[tuple[int]]  # ty: ignore[invalid-type-arguments]
        _invalid_jax: JaxBroadcastTuple[tuple[int]]  # ty: ignore[invalid-type-arguments]
        _invalid_torch: TorchBroadcastTuple[tuple[int]]  # ty: ignore[invalid-type-arguments]

    class _NumpyMembers(NamedTuple):
        first: np.ndarray
        second: np.float64

    class _JaxMembers(NamedTuple):
        first: jax.Array
        second: AnyJaxLike

    class _TorchMembers(NamedTuple):
        first: torch.Tensor
        second: AnyTorchLike

    def _check_named_members(numpy: _NumpyMembers, jax_array: _JaxMembers, tensor: _TorchMembers) -> None:
        n = NumpyBroadcastTuple(numpy)
        assert_type(n, NumpyBroadcastTuple[_NumpyMembers])
        assert_type(n.reshape(6).members, _NumpyMembers)
        assert_type(n[0].members.second, np.float64)
        j = JaxBroadcastTuple(jax_array)
        assert_type(j, JaxBroadcastTuple[_JaxMembers])
        assert_type(j.reshape(6).members, _JaxMembers)
        assert_type(j.at[0].set(1).members.first, jax.Array)
        t = TorchBroadcastTuple(tensor)
        assert_type(t, TorchBroadcastTuple[_TorchMembers])
        assert_type(t.reshape(6).members, _TorchMembers)
        assert_type(t.__torch_like__(copy=True).members.first, torch.Tensor)
