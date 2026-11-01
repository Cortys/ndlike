"""Assignment preparation preserves representation metadata and validates conversions."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import ClassVar

import pytest

pytest.importorskip("torch")
import torch

from ndlike.protected_axis.torch import TorchAxisProtected


@dataclass(frozen=True)
class _LengthPair(TorchAxisProtected[torch.Tensor]):
    left: torch.Tensor
    right: torch.Tensor
    unit: str = "m"
    protected_axes: ClassVar[dict[str, int]] = {"left": 1, "right": 2}

    def _prepare_assignment_value(self, value: object) -> object:
        if not isinstance(value, _LengthPair):
            msg = "Assignment requires length metadata."
            raise TypeError(msg)
        scales = {"m": 1.0, "cm": 0.01}
        if value.unit not in scales:
            msg = "Incompatible units."
            raise ValueError(msg)
        factor = scales[value.unit] / scales[self.unit]
        return replace(value, left=value.left * factor, right=value.right * factor, unit=self.unit)


@pytest.mark.parametrize("index", [(2, 1), (2, slice(None))])
def test_assignment_converts_metadata_before_extracting_fields(index):
    target = _LengthPair(torch.zeros((3, 4, 2)), torch.zeros((3, 4, 2, 2)))
    source = _LengthPair(torch.full((2,), 100.0), torch.full((2, 2), 200.0), unit="cm")

    target[index] = source

    expected_left = torch.zeros_like(target.left)
    expected_right = torch.zeros_like(target.right)
    expected_left[index] = 1
    expected_right[index] = 2
    torch.testing.assert_close(target.left, expected_left)
    torch.testing.assert_close(target.right, expected_right)
    assert target.unit == "m"
    assert source.unit == "cm"
    torch.testing.assert_close(source.left, torch.full((2,), 100.0))
    torch.testing.assert_close(source.right, torch.full((2, 2), 200.0))


@pytest.mark.parametrize("kind", ["metadata", "array", "tuple"])
def test_rejected_assignment_leaves_all_fields_untouched(kind):
    target = _LengthPair(torch.zeros((3, 2)), torch.zeros((3, 2, 2)))
    source = _LengthPair(torch.ones(2), torch.ones((2, 2)), unit="s")
    value = {"metadata": source, "array": source.left, "tuple": (source.left, source.right)}[kind]

    with pytest.raises((TypeError, ValueError), match=r"metadata|Incompatible units"):
        target[0] = value

    assert torch.count_nonzero(target.left) == 0
    assert torch.count_nonzero(target.right) == 0


def test_converted_fields_are_shape_checked_before_assignment():
    @dataclass(frozen=True)
    class Broken(_LengthPair):
        left: torch.Tensor
        right: torch.Tensor

        def _prepare_assignment_value(self, value: object) -> object:
            del value
            return torch.ones(2), torch.ones((3, 3))

    target = Broken(torch.zeros((3, 2)), torch.zeros((3, 2, 2)))
    with pytest.raises(ValueError, match="modifies protected trailing axes"):
        target[0] = object()

    assert torch.count_nonzero(target.left) == 0
    assert torch.count_nonzero(target.right) == 0
