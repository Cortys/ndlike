"""Tests for the Torch-like conversion protocol and dispatch."""

from __future__ import annotations

from dataclasses import dataclass
from inspect import isabstract
from typing import TYPE_CHECKING, Any, ClassVar, assert_type, get_origin
from typing_extensions import get_protocol_members

import numpy as np
import pytest

pytest.importorskip("torch")
import torch

from ndlike.protected_axis.torch import TorchAxisProtected
from ndlike.torch_like import TorchLike, TorchLikeConvertible, TorchLikeImplementation, to_torch_like


@dataclass(frozen=True, slots=True)
class _Tensor(TorchAxisProtected[torch.Tensor]):
    tensor: torch.Tensor
    protected_axes: ClassVar[dict[str, int]] = {"tensor": 1}


if TYPE_CHECKING:

    def _check_dispatched_method_types(protocol: TorchLike[Any], implementation: _Tensor) -> None:
        type Result = TorchLike[Any] | torch.Tensor

        assert_type(protocol.T, Result)
        assert_type(protocol.mT, Result)
        assert_type(protocol.mH, Result)
        assert_type(implementation.T, Result)
        assert_type(implementation.mT, Result)
        assert_type(implementation.mH, Result)
        assert_type(implementation.clone(), Result)
        assert_type(implementation.transpose(0, 1), Result)
        assert_type(implementation.permute(1, 0), Result)
        assert_type(implementation.resolve_conj(), Result)
        assert_type(implementation.resolve_neg(), Result)
        assert_type(implementation.reshape(2, 3), Result)
        assert_type(implementation.gather(0, torch.zeros((2, 3), dtype=torch.long)), Result)
        assert_type(implementation.to(dtype=torch.float32), _Tensor)
        assert_type(implementation.detach(), _Tensor)
        assert_type(implementation.cpu(), _Tensor)
        assert_type(implementation.to_device("cpu"), _Tensor)


@pytest.mark.parametrize("native", [False, True])
def test_dispatched_methods_can_return_another_wrapper_or_native_tensor(native: bool) -> None:
    tensor = torch.ones((2, 3, 4))
    expected = tensor if native else _Tensor(tensor)

    def change_result(cls, func, types, args=(), kwargs=None) -> object:
        del cls, func, types, args, kwargs
        return expected

    changing = type(
        "_ChangingTensor",
        (_Tensor,),
        {"__annotations__": {"tensor": torch.Tensor}, "__torch_function__": classmethod(change_result)},
    )
    original = changing(tensor)
    for result in (
        original.T,
        original.mT,
        original.mH,
        original.clone(),
        original.transpose(0, 1),
        original.permute(1, 0),
        original.resolve_conj(),
        original.resolve_neg(),
        original.reshape(2, 3),
        original.gather(0, torch.zeros((2, 3), dtype=torch.long)),
    ):
        assert result is expected


@pytest.mark.parametrize("attribute", sorted(TorchLikeImplementation.__abstractmethods__))
def test_representation_specific_members_must_be_implemented(attribute: str) -> None:
    member = vars(TorchLikeImplementation)[attribute]
    incomplete = type("_IncompleteTensor", (_Tensor,), {attribute: member})
    assert isabstract(incomplete)
    assert incomplete.__abstractmethods__ == frozenset({attribute})
    with pytest.raises(TypeError, match=attribute):
        incomplete(torch.ones((2, 3)))


@pytest.mark.parametrize("attribute", sorted(get_protocol_members(TorchLike)))
def test_protocol_members_are_implemented_or_explicitly_abstract(attribute: str) -> None:
    owner = next(base for base in TorchLikeImplementation.__mro__ if attribute in vars(base))
    assert not vars(owner).get("_is_protocol", False)


def test_protected_tensor_fulfills_the_abstract_contract() -> None:
    assert not isabstract(_Tensor)
    assert isinstance(_Tensor(torch.ones((2, 3))), TorchLike)


class _Convertible(np.ndarray):
    """An array whose conversion hook records whether dispatch selected it."""

    hook_calls: ClassVar[int] = 0

    def __torch_like__(
        self,
        dtype: torch.dtype | None = None,
        /,
        *,
        device: torch.device | str | None = None,
        copy: bool = False,
    ) -> torch.Tensor:
        type(self).hook_calls += 1
        tensor = torch.as_tensor(np.asarray(self), dtype=dtype, device=device)
        return tensor.clone() if copy else tensor


def test_converter_protocol_remains_generic() -> None:
    assert get_origin(TorchLikeConvertible[np.ndarray]) is TorchLikeConvertible


def test_array_converter_uses_its_hook() -> None:
    data = np.arange(3.0).view(_Convertible)
    previous_calls = _Convertible.hook_calls
    assert isinstance(data, TorchLikeConvertible)

    result = to_torch_like(data, torch.float64, device="cpu", copy=True)

    assert _Convertible.hook_calls == previous_calls + 1
    assert isinstance(result, torch.Tensor)
    assert result.dtype == torch.float64
    torch.testing.assert_close(result, torch.arange(3.0, dtype=torch.float64))


def test_hook_alone_does_not_satisfy_the_converter_protocol() -> None:
    class _HookOnly:
        def __torch_like__(self, dtype=None, /, *, device=None, copy=False) -> torch.Tensor:
            del dtype, device, copy
            pytest.fail("A hook-only object must not be selected as an array converter")

    assert not isinstance(_HookOnly(), TorchLikeConvertible)
