"""Signature preservation and argument selection for JAX function dispatch."""

from __future__ import annotations

from collections.abc import Callable
from inspect import signature
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, overload

import numpy as np
import pytest

pytest.importorskip("jax")
import jax
import jax.numpy as jnp

from ndlike import jax_functions as jf


class RecordingOverride:
    @classmethod
    def __jax_function__(cls, func, types, args=(), kwargs=None) -> object:
        """Record the exact public function and arguments seen by dispatch."""
        del cls
        return func, types, args, kwargs


@overload
def _overloaded_identity(x: int, /) -> int: ...


@overload
def _overloaded_identity(x: str, /) -> str: ...


@jf.jax_function_dispatch("x")
def _overloaded_identity(x: int | str, /) -> int | str:
    return x


_TYPING_CHECK = """
from typing import TYPE_CHECKING, Any, assert_type, overload
import jax
import numpy as np
from ndlike import jax_functions as jf
from ndlike.jax_like import JaxLike
from ndlike.protected_axis.jax import JaxAxisProtected

type Result = jax.Array | JaxLike[Any]

@overload
def _overloaded_identity(x: int, /) -> int: ...
@overload
def _overloaded_identity(x: str, /) -> str: ...
@jf.jax_function_dispatch("x")
def _overloaded_identity(x: int | str, /) -> int | str:
    return x

if TYPE_CHECKING:
    def _check_operator_types(array: jax.Array, custom: JaxLike[Any], numpy: np.ndarray) -> None:
        assert_type(jf.jax_add(array, 1), jax.Array)
        assert_type(jf.jax_subtract(1, array), jax.Array)
        assert_type(jf.jax_multiply(array, array), jax.Array)
        assert_type(jf.jax_true_divide(array, 2), jax.Array)
        assert_type(jf.jax_floor_divide(array, 2), jax.Array)
        assert_type(jf.jax_remainder(array, 2), jax.Array)
        assert_type(jf.jax_divmod(array, 2), tuple[jax.Array, jax.Array])
        assert_type(jf.jax_frexp(array), tuple[jax.Array, jax.Array])
        assert_type(jf.jax_modf(array), tuple[jax.Array, jax.Array])
        assert_type(jf.jax_power(array, 2), jax.Array)
        assert_type(jf.jax_matmul(array, array), jax.Array)
        assert_type(jf.jax_bitwise_and(array, 1), jax.Array)
        assert_type(jf.jax_bitwise_or(array, 1), jax.Array)
        assert_type(jf.jax_bitwise_xor(array, 1), jax.Array)
        assert_type(jf.jax_left_shift(array, 1), jax.Array)
        assert_type(jf.jax_right_shift(array, 1), jax.Array)
        assert_type(jf.jax_equal(array, 1), jax.Array)
        assert_type(jf.jax_not_equal(array, 1), jax.Array)
        assert_type(jf.jax_less(array, 1), jax.Array)
        assert_type(jf.jax_less_equal(array, 1), jax.Array)
        assert_type(jf.jax_greater(array, 1), jax.Array)
        assert_type(jf.jax_greater_equal(array, 1), jax.Array)
        assert_type(jf.jax_positive(array), jax.Array)
        assert_type(jf.jax_negative(array), jax.Array)
        assert_type(jf.jax_absolute(array), jax.Array)
        assert_type(jf.jax_invert(array), jax.Array)
        assert_type(jf.jax_add(custom, 1), Result)
        assert_type(jf.jax_add(1, custom), Result)
        assert_type(jf.jax_equal(custom, array), Result)
        assert_type(jf.jax_divmod(custom, array), tuple[Result, Result])
        assert_type(jf.jax_frexp(custom), tuple[Result, Result])
        assert_type(jf.jax_modf(custom), tuple[Result, Result])
        assert_type(jf.jax_negative(custom), Result)
        assert_type(jf.jax_reshape(array, (2, 3)), jax.Array)
        assert_type(jf.jax_transpose(array), jax.Array)
        assert_type(jf.jax_matrix_transpose(array), jax.Array)
        assert_type(jf.jax_moveaxis(array, 0, 1), jax.Array)
        assert_type(jf.jax_swapaxes(array, 0, 1), jax.Array)
        assert_type(jf.jax_squeeze(array), jax.Array)
        assert_type(jf.jax_expand_dims(array, 0), jax.Array)
        assert_type(jf.jax_concatenate([array, array]), jax.Array)
        assert_type(jf.jax_take_along_axis(array, array), jax.Array)
        assert_type(jf.jax_conj(array), jax.Array)
        assert_type(jf.jax_copy(array), jax.Array)
        assert_type(jf.jax_astype(array, "float32"), jax.Array)
        assert_type(jf.jax_mean(array), jax.Array)
        assert_type(jf.jax_sum(array, axis=0), jax.Array)
        assert_type(jf.jax_std(array), jax.Array)
        assert_type(jf.jax_var(array), jax.Array)
        assert_type(jf.jax_average(array), jax.Array | tuple[jax.Array, jax.Array])
        assert_type(jf.jax_stack([array, array], axis=0), jax.Array)
        assert_type(jf.jax_reshape(custom, (2, 3)), Result)
        assert_type(jf.jax_reshape(custom, (None, 6)), Result)
        assert_type(jf.jax_transpose(custom), Result)
        assert_type(jf.jax_matrix_transpose(custom), Result)
        assert_type(jf.jax_moveaxis(custom, 0, 1), Result)
        assert_type(jf.jax_swapaxes(custom, 0, 1), Result)
        assert_type(jf.jax_squeeze(custom), Result)
        assert_type(jf.jax_expand_dims(custom, 0), Result)
        assert_type(jf.jax_concatenate([array, custom]), Result)
        assert_type(jf.jax_stack([array, custom]), Result)
        assert_type(jf.jax_take_along_axis(custom, array), Result)
        assert_type(jf.jax_take_along_axis(array, custom), Result)
        assert_type(jf.jax_conj(custom), Result)
        assert_type(jf.jax_copy(custom), Result)
        assert_type(jf.jax_astype(custom, "float32"), Result)
        assert_type(jf.jax_mean(custom), Result)
        assert_type(jf.jax_mean(array, where=custom), Result)
        assert_type(jf.jax_sum(custom), Result)
        assert_type(jf.jax_sum(array, initial=custom), Result)
        assert_type(jf.jax_sum(array, where=custom), Result)
        assert_type(jf.jax_std(custom), Result)
        assert_type(jf.jax_std(array, where=custom), Result)
        assert_type(jf.jax_var(custom), Result)
        assert_type(jf.jax_var(array, where=custom), Result)
        assert_type(jf.jax_average(custom), Result | tuple[Result, Result])
        assert_type(jf.jax_average(array, weights=custom), Result | tuple[Result, Result])
        assert_type(jf.jax_add(1, 2), jax.Array)
        assert_type(jf.jax_add(numpy, numpy), jax.Array)
        assert_type(jf.jax_frexp(numpy), tuple[jax.Array, jax.Array])
        assert_type(jf.jax_modf(numpy), tuple[jax.Array, jax.Array])
        assert_type(jf.jax_frexp(1.5), tuple[jax.Array, jax.Array])
        assert_type(jf.jax_modf(1.5), tuple[jax.Array, jax.Array])
        assert_type(jf.jax_copy(numpy), jax.Array)
        assert_type(jf.jax_copy(1), jax.Array)
        assert_type(jf.jax_stack([numpy, numpy]), jax.Array)
        assert_type(_overloaded_identity(1), int)
        assert_type(_overloaded_identity("value"), str)

    def _check_method_types(custom: JaxLike[Any], protected: JaxAxisProtected[jax.Array]) -> None:
        assert_type(custom.T, Result)
        assert_type(custom.mT, Result)
        assert_type(custom.copy(), Result)
        assert_type(custom.reshape(2, 3), Result)
        assert_type(custom.transpose(), Result)
        assert_type(custom.ravel(), Result)
        assert_type(custom.flatten(), Result)
        assert_type(custom.squeeze(), Result)
        assert_type(protected.mT, Result)
        assert_type(protected.mH, Result)
        assert_type(protected.reshape(2, 3), Result)
        assert_type(protected.take_along_axis(jax.numpy.zeros((2, 3), dtype=int)), Result)
        assert_type(protected.astype(None), JaxAxisProtected[jax.Array])
"""

_INVALID_TYPING_CHECK = """
import jax
from ndlike import jax_functions as jf
from ndlike.jax_like import JaxLike
from typing import Any

def invalid_calls(array: jax.Array, custom: JaxLike[Any]) -> None:
    jf.jax_add("invalid", array)
    jf.jax_add(array)
    jf.jax_add(x1=array, x2=array)
    jf.jax_negative("invalid")
    jf.jax_reshape(array, unknown=(2, 3))
    jf.jax_reshape(array, (None, 6))
    jf.jax_stack([custom], axis="0")
"""


@pytest.mark.parametrize(
    ("source_text", "expected_errors"), [(_TYPING_CHECK, 0), (_INVALID_TYPING_CHECK, 8)], ids=("valid", "invalid")
)
def test_function_signatures_with_ty(tmp_path: Path, source_text: str, expected_errors: int):
    executable = shutil.which("ty")
    if executable is None:
        pytest.skip("ty is required for the static signature regression checks")
    source = tmp_path / "check_jax_functions.py"
    source.write_text(source_text)
    # The generated stubs are a partial tree. Check source signatures without
    # letting that tree shadow imports and turn them into Unknown.
    source_root = Path(__file__).resolve().parents[2] / "src"
    result = subprocess.run(  # noqa: S603
        [
            executable,
            "check",
            str(source),
            "--project",
            str(tmp_path),
            "--python",
            sys.prefix,
            "--extra-search-path",
            str(source_root),
            "--error-on-warning",
            "--output-format",
            "concise",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == bool(expected_errors), result.stdout + result.stderr
    if expected_errors:
        assert result.stdout.count(": error[") == expected_errors, result.stdout
        assert f"Found {expected_errors} diagnostics" in result.stdout


def test_decorator_preserves_signature_metadata_and_overloads():
    def native(x: int, /, *, offset: int = 2) -> int:
        """Add an offset to an integer."""
        return x + offset

    decorated = jf.jax_function_dispatch("x")(native)
    assert signature(decorated) == signature(native)
    assert decorated.__name__ == native.__name__
    assert decorated.__qualname__ == native.__qualname__
    assert decorated.__doc__ == native.__doc__
    assert decorated.__annotations__ == native.__annotations__
    assert decorated(3) == 5
    assert decorated(3, offset=4) == 7
    assert _overloaded_identity(1) == 1
    assert _overloaded_identity("value") == "value"


def test_keyword_operands_and_default_operands_participate_in_dispatch():
    sentinel = RecordingOverride()

    @jf.jax_function_dispatch("x", "other")
    def operation(x: object, *, other: object = sentinel) -> object:
        del x, other
        pytest.fail("A selected custom operand must be dispatched before native execution")

    assert operation(x=3) == (operation, (RecordingOverride,), (), {"x": 3})
    assert operation(3, other=sentinel) == (operation, (RecordingOverride,), (3,), {"other": sentinel})


@pytest.mark.parametrize(
    ("func", "kwargs"),
    [
        (jf.jax_mean, {"where": RecordingOverride()}),
        (jf.jax_sum, {"where": RecordingOverride()}),
        (jf.jax_sum, {"initial": RecordingOverride()}),
        (jf.jax_std, {"where": RecordingOverride()}),
        (jf.jax_var, {"where": RecordingOverride()}),
        (jf.jax_average, {"weights": RecordingOverride()}),
        (jf.jax_take_along_axis, {"indices": RecordingOverride()}),
    ],
)
def test_optional_and_secondary_array_operands_dispatch(func, kwargs):
    array = jnp.ones((2, 3))
    public_api, types, args, received_kwargs = func(array, **kwargs)
    assert public_api is func
    assert types == (RecordingOverride,)
    assert len(args) == 1
    assert args[0] is array
    assert received_kwargs == kwargs


def test_sum_initial_dispatches_when_passed_positionally():
    array = jnp.ones(3)
    initial = RecordingOverride()
    result = jf.jax_sum(array, None, None, None, False, initial)  # ty: ignore[invalid-argument-type]
    assert isinstance(result, tuple)
    assert result[0] is jf.jax_sum
    assert result[1] == (RecordingOverride,)
    assert result[2][-1] is initial
    assert result[3] == {}


def test_sequence_selection_does_not_dispatch_on_configuration():
    sentinel = RecordingOverride()

    @jf.jax_function_dispatch(unpack=("arrays",))
    def operation(arrays: tuple[object, ...], *, label: object = None) -> object:
        return arrays, label

    assert operation((1, 2), label=sentinel) == ((1, 2), sentinel)
    arrays = (1, sentinel, sentinel)
    assert operation(arrays) == (operation, (RecordingOverride,), (arrays,), {})


@pytest.mark.parametrize("func", [jf.jax_stack, jf.jax_concatenate])
def test_sequence_wrappers_keep_public_identity(func):
    arrays = [jnp.ones(3), RecordingOverride()]
    public_api, types, args, kwargs = func(arrays, axis=0)
    assert public_api is func
    assert types == (RecordingOverride,)
    assert args[0] is arrays
    assert kwargs == {"axis": 0}


@pytest.mark.parametrize("func", [jf.jax_frexp, jf.jax_modf])
def test_multi_output_unary_wrappers_dispatch(func):
    value = RecordingOverride()
    assert func(value) == (func, (RecordingOverride,), (value,), {})


def test_subclass_precedence_and_not_implemented_fallback():
    calls = []

    class Base(RecordingOverride):
        @classmethod
        def __jax_function__(cls, func, types, args=(), kwargs=None) -> object:
            calls.append(cls)
            return super().__jax_function__(func, types, args, kwargs)

    class Subclass(Base):
        @classmethod
        def __jax_function__(cls, func, types, args=(), kwargs=None) -> object:
            del func, types, args, kwargs
            calls.append(cls)
            return NotImplemented

    left, right = Base(), Subclass()
    result = jf.jax_add(left, right)  # ty: ignore[invalid-argument-type]
    assert calls == [Subclass, Base]
    assert result == (jf.jax_add, (Subclass, Base), (left, right), {})


def test_declining_override_never_falls_back_to_native_implementation():
    class Declines(RecordingOverride):
        @classmethod
        def __jax_function__(cls, func, types, args=(), kwargs=None) -> object:
            del cls, func, types, args, kwargs
            return NotImplemented

    @jf.jax_function_dispatch("x")
    def operation(x: object) -> object:
        del x
        pytest.fail("Declining overrides must not be coerced by native execution")

    with pytest.raises(TypeError, match="no implementation found for operation"):
        operation(Declines())


@pytest.mark.parametrize(
    ("names", "unpack"),
    [((), ()), (("x", "x"), ()), (("x",), ("x",)), (("missing",), ()), (("args",), ()), ((), ("kwargs",))],
)
def test_invalid_dispatch_selectors_fail_at_decoration(names, unpack):
    def operation(x: object, *args: object, **kwargs: object) -> object:
        del args, kwargs
        return x

    with pytest.raises(ValueError, match="Dispatch selector"):
        jf.jax_function_dispatch(*names, unpack=unpack)(operation)


def test_call_validation_happens_before_override_dispatch():
    with pytest.raises(TypeError, match="missing a required argument"):
        jf.jax_add(RecordingOverride())  # ty: ignore[invalid-argument-type, missing-argument]
    with pytest.raises(TypeError, match=r"positional.only"):
        jf.jax_add(x1=RecordingOverride(), x2=1)  # ty: ignore[positional-only-parameter-as-kwarg]


_PUBLIC_WRAPPERS = [
    getattr(jf, name) for name in jf.__all__ if name.startswith("jax_") and name != "jax_function_dispatch"
]


@pytest.mark.parametrize("func", _PUBLIC_WRAPPERS, ids=lambda func: func.__name__)
def test_public_wrappers_have_preserved_docstrings_and_signatures(func: Callable[..., Any]):
    native = getattr(func, "__wrapped__", None)
    assert native is not None
    assert signature(func) == signature(native)
    assert func.__doc__ == native.__doc__
    assert func.__doc__ is not None
    assert "Args:" in func.__doc__
    assert "Returns:" in func.__doc__


def test_decorated_native_calls_remain_traceable():
    array = jnp.arange(6.0).reshape(2, 3)
    result = jax.jit(lambda x: jf.jax_add(jf.jax_sum(x, axis=1), 1))(array)
    np.testing.assert_array_equal(result, jnp.sum(array, axis=1) + 1)
