"""Bounded member tuples retain positions, lengths, and array owner types."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, NamedTuple, assert_type

from ndlike.tuple.numpy import NumpyBroadcastTuple, NumpyMember

if TYPE_CHECKING:
    import jax
    import numpy as np
    import torch

    from ndlike import jax_functions as jf
    from ndlike.jax_like import AnyJaxLike
    from ndlike.protected_axis.jax import JaxAxisProtected
    from ndlike.protected_axis.numpy import NumpyAxisProtected
    from ndlike.protected_axis.torch import TorchAxisProtected
    from ndlike.torch_like import AnyTorchLike
    from ndlike.tuple.jax import JaxBroadcastTuple, JaxMember
    from ndlike.tuple.torch import TorchBroadcastTuple, TorchMember

    type NumpyFallback = NumpyBroadcastTuple[tuple[NumpyMember, ...]]
    type JaxFallback = JaxBroadcastTuple[tuple[JaxMember, ...]]
    type TorchFallback = TorchBroadcastTuple[tuple[TorchMember, ...]]

    def _check_empty(jax_idx: jax.Array, torch_idx: torch.Tensor) -> None:
        assert_type(NumpyBroadcastTuple(()), NumpyBroadcastTuple[tuple[()]])
        assert_type(JaxBroadcastTuple(()), JaxBroadcastTuple[tuple[()]])
        assert_type(TorchBroadcastTuple(()), TorchBroadcastTuple[tuple[()]])
        assert_type(NumpyBroadcastTuple(())[42].members, tuple[()])
        assert_type(NumpyBroadcastTuple(()).reshape([2, 3], copy=True).members, tuple[()])
        assert_type(NumpyBroadcastTuple(()).swapaxes(0, 1).members, tuple[()])
        assert_type(JaxBroadcastTuple(()).reshape(3, 4).members, tuple[()])
        assert_type(JaxBroadcastTuple(()).swapaxes(0, 1).members, tuple[()])
        assert_type(JaxBroadcastTuple(()).take_along_axis(jax_idx).members, tuple[()])
        assert_type(TorchBroadcastTuple(()).T.members, tuple[TorchMember, ...])
        assert_type(TorchBroadcastTuple(()).reshape([2, 3]).members, tuple[()])
        assert_type(TorchBroadcastTuple(()).unsqueeze(0).members, tuple[()])
        assert_type(TorchBroadcastTuple(()).movedim(0, 1).members, tuple[()])
        assert_type(TorchBroadcastTuple(()).broadcast_to((2, 3)).members, tuple[()])
        assert_type(TorchBroadcastTuple(()).gather(0, torch_idx).members, tuple[()])
        assert_type(TorchBroadcastTuple(()).dim(), int)
        assert_type(TorchBroadcastTuple(()).numel(), int)

    def _check_numpy[T1: NumpyMember, T2: NumpyMember](first: T1, second: T2) -> None:
        array = NumpyBroadcastTuple((first, second))
        assert_type(array, NumpyBroadcastTuple[tuple[T1, T2]])
        assert_type(array.members, tuple[T1, T2])
        assert_type(array.members[0], T1)
        assert_type(array.members[1], T2)
        assert_type(array[0], NumpyFallback)
        assert_type(array.reshape(2, 3), NumpyFallback)
        assert_type(array.reshape([2, 3], copy=True), NumpyFallback)
        assert_type(array.reshape((2, 3), copy=False), NumpyFallback)
        assert_type(array.swapaxes(0, 1), NumpyFallback)
        assert_type(array.squeeze(), NumpyFallback)
        assert_type(array.T, NumpyFallback)
        assert_type(array.with_members((first, second)), NumpyBroadcastTuple[tuple[T1, T2]])
        assert_type(next(iter(array)), NumpyFallback)
        assert_type(array.__array_like__(), NumpyBroadcastTuple[tuple[T1, T2]])
        assert_type(array.__array_like__(copy=True), NumpyBroadcastTuple[tuple[T1, T2]])
        assert_type(NumpyBroadcastTuple((first,)), NumpyBroadcastTuple[tuple[T1]])
        array.with_members((second, first))  # ty: ignore[invalid-argument-type]
        NumpyBroadcastTuple[tuple[T1, T2]]((first,))  # ty: ignore[invalid-argument-type]

    def _check_jax[T1: JaxMember, T2: JaxMember](first: T1, second: T2, indices: jax.Array) -> None:
        array = JaxBroadcastTuple((first, second))
        assert_type(array, JaxBroadcastTuple[tuple[T1, T2]])
        assert_type(array.members, tuple[T1, T2])
        assert_type(array.members[0], T1)
        assert_type(array.members[1], T2)
        assert_type(array[0], JaxFallback)
        assert_type(array.reshape(2, 3), JaxFallback)
        assert_type(array.swapaxes(0, 1), JaxFallback)
        assert_type(array.swapaxes(axis1=0, axis2=1), JaxFallback)
        assert_type(array.take_along_axis(indices), JaxFallback)
        assert_type(array.take_along_axis(indices=array, axis=0), JaxFallback)
        assert_type(array.T, JaxFallback)
        assert_type(jf.jax_reshape(array, (2, 3)), JaxFallback)
        assert_type(array.at[0].get(), JaxFallback)
        assert_type(array.at[0].set(1), JaxBroadcastTuple[tuple[T1, T2]])
        assert_type(array.__jax_like__(), JaxBroadcastTuple[tuple[T1, T2]])
        assert_type(array.__jax_like__(copy=True), JaxBroadcastTuple[tuple[T1, T2]])
        assert_type(JaxBroadcastTuple((first,)), JaxBroadcastTuple[tuple[T1]])
        array.with_members((first,))  # ty: ignore[invalid-argument-type]
        JaxBroadcastTuple[tuple[T1, T2]]((second, first))  # ty: ignore[invalid-argument-type]

    def _check_torch[T1: TorchMember, T2: TorchMember](first: T1, second: T2, index: torch.Tensor) -> None:
        array = TorchBroadcastTuple((first, second))
        assert_type(array, TorchBroadcastTuple[tuple[T1, T2]])
        assert_type(array.members, tuple[T1, T2])
        assert_type(array.members[0], T1)
        assert_type(array.members[1], T2)
        assert_type(array[0], TorchFallback)
        assert_type(array.reshape(2, 3), TorchFallback)
        assert_type(array.reshape([2, 3]), TorchFallback)
        assert_type(array.unsqueeze(0), TorchFallback)
        assert_type(array.movedim(0, 1), TorchFallback)
        assert_type(array.broadcast_to((2, 3)), TorchFallback)
        assert_type(array.broadcast_to(size=(2, 3)), TorchFallback)
        assert_type(array.broadcast_to(2, 3), TorchFallback)
        assert_type(array.squeeze(), TorchFallback)
        assert_type(array.gather(0, index), TorchFallback)
        assert_type(array.gather(dim=1, index=array), TorchFallback)
        assert_type(array.dim(), int)
        assert_type(array.ndimension(), int)
        assert_type(array.numel(), int)
        assert_type(array.nelement(), int)
        assert_type(array.T, TorchFallback)
        assert_type(array.__torch_like__(), TorchBroadcastTuple[tuple[T1, T2]])
        assert_type(array.__torch_like__(copy=True), TorchBroadcastTuple[tuple[T1, T2]])
        assert_type(TorchBroadcastTuple((first,)), TorchBroadcastTuple[tuple[T1]])
        array.with_members((first, second, first))  # ty: ignore[invalid-argument-type]
        TorchBroadcastTuple[tuple[T1, T2]]((first, second, first))  # ty: ignore[invalid-argument-type]

    def _check_variable_numpy[T: NumpyMember](members: tuple[T, ...]) -> None:
        array = NumpyBroadcastTuple(members)
        assert_type(array, NumpyBroadcastTuple[tuple[T, ...]])
        assert_type(array.members[0], T)
        assert_type(array.reshape(2, 3).members, tuple[NumpyMember, ...])

    def _check_variable_jax[T: JaxMember](members: tuple[T, ...]) -> None:
        array = JaxBroadcastTuple(members)
        assert_type(array, JaxBroadcastTuple[tuple[T, ...]])
        assert_type(array.members[0], T)
        assert_type(array.at[0].get().members, tuple[JaxMember, ...])

    def _check_variable_torch[T: TorchMember](members: tuple[T, ...]) -> None:
        array = TorchBroadcastTuple(members)
        assert_type(array, TorchBroadcastTuple[tuple[T, ...]])
        assert_type(array.members[0], T)
        assert_type(array.reshape(2, 3).members, tuple[TorchMember, ...])

    def _check_forwarding_numpy[Members: tuple[NumpyMember, ...]](members: Members) -> None:
        array = NumpyBroadcastTuple(members)
        assert_type(array, NumpyBroadcastTuple[Members])
        assert_type(array.members, Members)
        assert_type(array.reshape(2, 3).members, tuple[NumpyMember, ...])

    def _check_forwarding_jax[Members: tuple[JaxMember, ...]](members: Members) -> None:
        array = JaxBroadcastTuple(members)
        assert_type(array, JaxBroadcastTuple[Members])
        assert_type(array.members, Members)
        assert_type(array.at[0].set(1).members, Members)

    def _check_forwarding_torch[Members: tuple[TorchMember, ...]](members: Members) -> None:
        array = TorchBroadcastTuple(members)
        assert_type(array, TorchBroadcastTuple[Members])
        assert_type(array.members, Members)
        assert_type(array.reshape(2, 3).members, tuple[TorchMember, ...])

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

    def _check_named_members(
        numpy: _NumpyMembers, jax_array: _JaxMembers, tensor: _TorchMembers, jax_idx: jax.Array, torch_idx: torch.Tensor
    ) -> None:
        n = NumpyBroadcastTuple(numpy)
        assert_type(n, NumpyBroadcastTuple[_NumpyMembers])
        assert_type(n.reshape(6).members, tuple[NumpyMember, ...])
        assert_type(n[0].members, tuple[NumpyMember, ...])
        j = JaxBroadcastTuple(jax_array)
        assert_type(j, JaxBroadcastTuple[_JaxMembers])
        assert_type(j.reshape(6).members, tuple[JaxMember, ...])
        assert_type(j.take_along_axis(jax_idx).members, tuple[JaxMember, ...])
        assert_type(j.at[0].set(1).members.first, jax.Array)
        t = TorchBroadcastTuple(tensor)
        assert_type(t, TorchBroadcastTuple[_TorchMembers])
        assert_type(t.reshape(6).members, tuple[TorchMember, ...])
        assert_type(t.gather(0, torch_idx).members, tuple[TorchMember, ...])
        assert_type(t.__torch_like__(copy=True).members.first, torch.Tensor)

    class _NumpyPosition(NumpyAxisProtected["_NumpyPosition"]): ...

    class _NumpyVelocity(NumpyAxisProtected["_NumpyVelocity"]): ...

    class _NumpyView(NumpyAxisProtected["_NumpyView", Any, Any, Any, np.ndarray]): ...

    class _NumpySpecialPosition(_NumpyPosition): ...

    class _NumpyPair(NamedTuple):
        position: _NumpyPosition
        velocity: _NumpyVelocity

    def _check_numpy_shape_contract(
        position: _NumpyPosition,
        native: np.ndarray,
        view: _NumpyView,
        child: _NumpySpecialPosition,
        pair: _NumpyPair,
        positions: tuple[_NumpyPosition, ...],
    ) -> None:
        fixed = NumpyBroadcastTuple((position, native))
        assert_type(fixed.reshape(6), NumpyFallback)
        assert_type(NumpyBroadcastTuple(pair).reshape(6).members, _NumpyPair)
        assert_type(NumpyBroadcastTuple(pair).T, NumpyFallback)
        assert_type(NumpyBroadcastTuple(pair).mT, NumpyFallback)
        assert_type(NumpyBroadcastTuple(pair).swapaxes(0, 1).members.position, _NumpyPosition)
        assert_type(NumpyBroadcastTuple(positions).reshape(6).members, tuple[_NumpyPosition, ...])
        assert_type(NumpyBroadcastTuple((position, view)).reshape(6), NumpyFallback)
        assert_type(NumpyBroadcastTuple((child,)).reshape(6), NumpyFallback)
        assert_type(NumpyBroadcastTuple((view,))[0], NumpyFallback)
        assert_type(next(iter(NumpyBroadcastTuple((view,)))), NumpyFallback)
        assert_type(
            NumpyBroadcastTuple((NumpyBroadcastTuple(pair),)).reshape(6).members, tuple[NumpyBroadcastTuple[_NumpyPair]]
        )
        assert_type(NumpyBroadcastTuple((NumpyBroadcastTuple((view,)),)).reshape(6), NumpyFallback)
        assert_type(NumpyBroadcastTuple((fixed,))[0], NumpyFallback)
        assert_type(NumpyBroadcastTuple(pair)[0].members, _NumpyPair)

    class _JaxPosition(JaxAxisProtected["_JaxPosition"]): ...

    class _JaxVelocity(JaxAxisProtected["_JaxVelocity"]): ...

    class _JaxView(JaxAxisProtected["_JaxView", Any, Any, Any, jax.Array]): ...

    class _JaxSpecialPosition(_JaxPosition): ...

    class _JaxPair(NamedTuple):
        position: _JaxPosition
        velocity: _JaxVelocity

    def _check_jax_shape_contract(
        position: _JaxPosition,
        native: jax.Array,
        view: _JaxView,
        child: _JaxSpecialPosition,
        pair: _JaxPair,
        positions: tuple[_JaxPosition, ...],
    ) -> None:
        fixed = JaxBroadcastTuple((position, native))
        assert_type(fixed.reshape(6), JaxFallback)
        assert_type(JaxBroadcastTuple(pair).reshape(6).members, _JaxPair)
        assert_type(JaxBroadcastTuple(pair).T, JaxFallback)
        assert_type(JaxBroadcastTuple(pair).mT, JaxFallback)
        assert_type(JaxBroadcastTuple(pair).swapaxes(0, 1).members.position, _JaxPosition)
        assert_type(JaxBroadcastTuple(positions).reshape(6).members, tuple[_JaxPosition, ...])
        assert_type(JaxBroadcastTuple((position, view)).reshape(6), JaxFallback)
        assert_type(JaxBroadcastTuple((child,)).reshape(6), JaxFallback)
        assert_type(JaxBroadcastTuple((view,))[0], JaxFallback)
        assert_type(next(iter(JaxBroadcastTuple((view,)))), JaxFallback)
        nested = JaxBroadcastTuple((JaxBroadcastTuple(pair),))
        nested_view = JaxBroadcastTuple((JaxBroadcastTuple((view,)),))
        assert_type(nested.reshape(6).members, tuple[JaxBroadcastTuple[_JaxPair]])
        assert_type(jf.jax_reshape(nested, (6,)).members, tuple[JaxBroadcastTuple[_JaxPair]])
        assert_type(nested.at[0].get().members, tuple[JaxBroadcastTuple[_JaxPair]])
        assert_type(nested_view.reshape(6), JaxFallback)
        assert_type(jf.jax_reshape(nested_view, (6,)), JaxFallback)
        assert_type(nested_view.at[0].get(), JaxFallback)
        assert_type(jf.jax_reshape(fixed, (2, 3)), JaxFallback)
        assert_type(jf.jax_reshape(JaxBroadcastTuple(pair), (2, 3)).members, _JaxPair)
        assert_type(jf.jax_reshape(JaxBroadcastTuple((view,)), (2, 3)), JaxFallback)
        assert_type(jf.jax_transpose(JaxBroadcastTuple(pair)).members, _JaxPair)
        assert_type(JaxBroadcastTuple(pair).at[0].get().members, _JaxPair)
        assert_type(jf.jax_take_along_axis(JaxBroadcastTuple(pair), native).members, _JaxPair)
        assert_type(fixed.take_along_axis(JaxBroadcastTuple(pair)), JaxFallback)
        assert_type(jf.jax_take_along_axis(fixed, JaxBroadcastTuple(pair)), JaxFallback)
        assert_type(jf.jax_take_along_axis(native, JaxBroadcastTuple(pair)), JaxFallback)
        assert_type(jf.jax_stack([JaxBroadcastTuple(pair), fixed]), JaxFallback)
        assert_type(jf.jax_concatenate([JaxBroadcastTuple(pair), fixed]), JaxFallback)
        assert_type(jf.jax_stack([JaxBroadcastTuple((view,))]), JaxFallback)
        left, right = JaxBroadcastTuple((native,)), JaxBroadcastTuple((position,))
        assert_type(jf.jax_stack((left, right)), JaxFallback)

    class _TorchPosition(TorchAxisProtected["_TorchPosition"]): ...

    class _TorchVelocity(TorchAxisProtected["_TorchVelocity"]): ...

    class _TorchView(TorchAxisProtected["_TorchView", Any, Any, Any, torch.Tensor]): ...

    class _TorchSpecialPosition(_TorchPosition): ...

    class _TorchPair(NamedTuple):
        position: _TorchPosition
        velocity: _TorchVelocity

    def _check_torch_shape_contract(
        position: _TorchPosition,
        native: torch.Tensor,
        view: _TorchView,
        child: _TorchSpecialPosition,
        pair: _TorchPair,
        positions: tuple[_TorchPosition, ...],
    ) -> None:
        fixed = TorchBroadcastTuple((position, native))
        assert_type(fixed.reshape(6), TorchFallback)
        assert_type(TorchBroadcastTuple(pair).reshape(6).members, _TorchPair)
        assert_type(TorchBroadcastTuple(pair).T, TorchFallback)
        assert_type(TorchBroadcastTuple(pair).mT, TorchFallback)
        assert_type(TorchBroadcastTuple(pair).mH, TorchFallback)
        assert_type(TorchBroadcastTuple(pair).unsqueeze(0).members.position, _TorchPosition)
        assert_type(TorchBroadcastTuple(positions).reshape(6).members, tuple[_TorchPosition, ...])
        assert_type(TorchBroadcastTuple((position, view)).reshape(6), TorchFallback)
        assert_type(TorchBroadcastTuple((child,)).reshape(6), TorchFallback)
        assert_type(TorchBroadcastTuple((view,))[0], TorchFallback)
        assert_type(next(iter(TorchBroadcastTuple((view,)))), TorchFallback)
        assert_type(
            TorchBroadcastTuple((TorchBroadcastTuple(pair),)).reshape(6).members, tuple[TorchBroadcastTuple[_TorchPair]]
        )
        assert_type(TorchBroadcastTuple((TorchBroadcastTuple((view,)),)).reshape(6), TorchFallback)

        assert_type(fixed.gather(0, TorchBroadcastTuple(pair)), TorchFallback)

    class _NumpySubclass(np.ndarray): ...

    class _TensorSubclass(torch.Tensor): ...

    class _JaxTupleSubclass(JaxBroadcastTuple[tuple[_JaxPosition]]): ...

    def _check_native_subclasses_and_shapes(
        subarray: _NumpySubclass,
        tensor: _TensorSubclass,
        known_shape: np.ndarray[tuple[Literal[2]], np.dtype[np.float64]],
        known_dtype: np.ndarray[tuple[int, ...], np.dtype[np.float64]],
        tagged: _JaxTupleSubclass,
    ) -> None:
        assert_type(NumpyBroadcastTuple((subarray,)).reshape(6), NumpyFallback)
        assert_type(TorchBroadcastTuple((tensor,)).reshape(6), TorchFallback)
        assert_type(NumpyBroadcastTuple((known_shape,)).reshape(1, 2), NumpyFallback)
        assert_type(NumpyBroadcastTuple((known_dtype,)).reshape(1, 2), NumpyFallback)
        assert_type(tagged.reshape(6).members, tuple[_JaxPosition])
        assert_type(tagged.at[0].set(0), _JaxTupleSubclass)
