"""Broadcast tuples preserve native Torch autograd and concrete result typing."""

from __future__ import annotations

from typing import Any, assert_type, cast

import pytest

pytest.importorskip("torch")
import torch

from ndlike.tuple.torch import TorchBroadcastTuple


def test_gradients_and_detach():
    a = torch.ones((2, 1), requires_grad=True)
    b = torch.ones(3, requires_grad=True)
    c = torch.ones((), requires_grad=True)
    x = TorchBroadcastTuple((a, b, c))
    assert_type(x.reshape(6), TorchBroadcastTuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor]])
    assert_type(x[0], TorchBroadcastTuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor]])
    torch.stack([member.sum() for member in x.reshape(6).members]).sum().backward()
    torch.testing.assert_close(a.grad, torch.full((2, 1), 3.0))
    torch.testing.assert_close(b.grad, torch.full((3,), 2.0))
    torch.testing.assert_close(c.grad, torch.tensor(6.0))
    detached = x.detach()
    assert type(detached) is TorchBroadcastTuple
    assert all(not member.requires_grad for member in detached.members)
    assert tuple(member.shape for member in detached.members) == ((2, 1), (3,), ())


def test_named_results_and_arithmetic_gradients():
    a = torch.ones((2, 1), requires_grad=True)
    b = torch.arange(3.0, requires_grad=True)
    x = TorchBroadcastTuple((a, b))
    assert_type(x + 2, TorchBroadcastTuple[tuple[torch.Tensor, torch.Tensor]])
    assert_type(x > 0, TorchBroadcastTuple[tuple[Any, ...]])
    result = torch.max(cast("Any", x), dim=1)
    assert isinstance(result, torch.return_types.max)
    assert isinstance(result.values, TorchBroadcastTuple)
    assert isinstance(result.indices, TorchBroadcastTuple)
    assert torch.equal(cast("Any", x), cast("Any", x)) == (True, True)
    loss = torch.sum(x * 2)
    torch.stack(list(loss.members)).sum().backward()
    torch.testing.assert_close(a.grad, torch.full((2, 1), 6.0))
    torch.testing.assert_close(b.grad, torch.full((3,), 4.0))
    assert x.mH.shape == (3, 2)
