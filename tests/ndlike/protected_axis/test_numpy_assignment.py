"""Assignment preparation preserves representation metadata and validates conversions."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import ClassVar

import numpy as np
import pytest

from ndlike.protected_axis.numpy import NumpyAxisProtected


@dataclass(frozen=True)
class _LengthPair(NumpyAxisProtected[np.ndarray]):
    left: np.ndarray
    right: np.ndarray
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
    target = _LengthPair(np.zeros((3, 4, 2)), np.zeros((3, 4, 2, 2)))
    source = _LengthPair(np.full(2, 100.0), np.full((2, 2), 200.0), unit="cm")

    target[index] = source

    expected_left = np.zeros_like(target.left)
    expected_right = np.zeros_like(target.right)
    expected_left[index] = 1
    expected_right[index] = 2
    np.testing.assert_array_equal(target.left, expected_left)
    np.testing.assert_array_equal(target.right, expected_right)
    assert target.unit == "m"
    assert source.unit == "cm"
    np.testing.assert_array_equal(source.left, 100)
    np.testing.assert_array_equal(source.right, 200)


@pytest.mark.parametrize("kind", ["metadata", "array", "tuple"])
def test_rejected_assignment_leaves_all_fields_untouched(kind):
    target = _LengthPair(np.zeros((3, 2)), np.zeros((3, 2, 2)))
    source = _LengthPair(np.ones(2), np.ones((2, 2)), unit="s")
    value = {"metadata": source, "array": source.left, "tuple": (source.left, source.right)}[kind]

    with pytest.raises((TypeError, ValueError), match=r"metadata|Incompatible units"):
        target[0] = value

    np.testing.assert_array_equal(target.left, 0)
    np.testing.assert_array_equal(target.right, 0)


def test_converted_fields_are_shape_checked_before_assignment():
    @dataclass(frozen=True)
    class Broken(_LengthPair):
        left: np.ndarray
        right: np.ndarray

        def _prepare_assignment_value(self, value: object) -> object:
            del value
            return np.ones(2), np.ones((3, 3))

    target = Broken(np.zeros((3, 2)), np.zeros((3, 2, 2)))
    with pytest.raises(ValueError, match="modifies protected trailing axes"):
        target[0] = object()

    np.testing.assert_array_equal(target.left, 0)
    np.testing.assert_array_equal(target.right, 0)
