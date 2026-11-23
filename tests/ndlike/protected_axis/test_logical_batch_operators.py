"""Logical-batch alignment contract for elementwise binary operators with raw operands.

Raw native arrays align right across logical batch axes and broadcast uniformly across
each field's protected trailing axes (appending ``protected_axes[field]`` singleton
axes), recursing through nested protected fields. Scalars and 0D arrays remain
unchanged, output-only dispatch retains native physical alignment, and matrix
multiplication retains its separate visible-dimension semantics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cache
import operator
from typing import Any, ClassVar

import numpy as np
import pytest

from ndlike.protected_axis._common import expand_for_protected_axes
from ndlike.protected_axis.numpy import NumpyAxisProtected

_NUMPY_PERMITTED: dict[np.ufunc, list[str]] = {
    ufunc: ["__call__"]
    for ufunc in (
        np.add,
        np.subtract,
        np.multiply,
        np.true_divide,
        np.floor_divide,
        np.remainder,
        np.divmod,
        np.equal,
        np.not_equal,
        np.less,
        np.greater,
        np.matmul,
        np.maximum,
    )
}


@dataclass(frozen=True)
class _NumpyMixed(NumpyAxisProtected["_NumpyMixed", Any]):
    scalar: np.ndarray
    vec: np.ndarray
    mat: np.ndarray
    protected_axes: ClassVar[dict[str, int]] = {"scalar": 0, "vec": 1, "mat": 2}
    permitted_ufuncs: ClassVar[dict[np.ufunc, list[str]]] = _NUMPY_PERMITTED


@dataclass(frozen=True)
class _NumpyInner(NumpyAxisProtected["_NumpyInner", Any]):
    leaf: np.ndarray
    protected_axes: ClassVar[dict[str, int]] = {"leaf": 1}
    permitted_ufuncs: ClassVar[dict[np.ufunc, list[str]]] = _NUMPY_PERMITTED


@dataclass(frozen=True)
class _NumpyOuter(NumpyAxisProtected["_NumpyOuter", Any]):
    inner: _NumpyInner
    aux: np.ndarray
    protected_axes: ClassVar[dict[str, int]] = {"inner": 1, "aux": 2}
    permitted_ufuncs: ClassVar[dict[np.ufunc, list[str]]] = _NUMPY_PERMITTED


@cache
def _jax_classes():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from ndlike import jax_functions as jf  # noqa: PLC0415
    from ndlike.protected_axis.jax import JaxAxisProtected  # noqa: PLC0415

    permitted = {
        jf.jax_add,
        jf.jax_subtract,
        jf.jax_multiply,
        jf.jax_true_divide,
        jf.jax_floor_divide,
        jf.jax_remainder,
        jf.jax_divmod,
        jf.jax_equal,
        jf.jax_not_equal,
        jf.jax_less,
        jf.jax_greater,
        jf.jax_matmul,
    }

    @dataclass(frozen=True)
    class _JaxMixed(JaxAxisProtected["_JaxMixed", Any]):
        scalar: Any
        vec: Any
        mat: Any
        protected_axes: ClassVar[dict[str, int]] = {"scalar": 0, "vec": 1, "mat": 2}
        permitted_functions: ClassVar[set] = permitted

    @dataclass(frozen=True)
    class _JaxInner(JaxAxisProtected["_JaxInner", Any]):
        leaf: Any
        protected_axes: ClassVar[dict[str, int]] = {"leaf": 1}
        permitted_functions: ClassVar[set] = permitted

    @dataclass(frozen=True)
    class _JaxOuter(JaxAxisProtected["_JaxOuter", Any]):
        inner: _JaxInner
        aux: Any
        protected_axes: ClassVar[dict[str, int]] = {"inner": 1, "aux": 2}
        permitted_functions: ClassVar[set] = permitted

    return jax, jnp, jf, JaxAxisProtected, _JaxMixed, _JaxInner, _JaxOuter


@cache
def _torch_classes():
    torch = pytest.importorskip("torch")
    from ndlike.protected_axis.torch import TorchAxisProtected  # noqa: PLC0415

    permitted = {
        torch.add,
        torch.sub,
        torch.mul,
        torch.true_divide,
        torch.floor_divide,
        torch.remainder,
        torch.eq,
        torch.ne,
        torch.lt,
        torch.gt,
        torch.matmul,
    }

    @dataclass(frozen=True)
    class _TorchMixed(TorchAxisProtected["_TorchMixed", Any]):
        scalar: Any
        vec: Any
        mat: Any
        protected_axes: ClassVar[dict[str, int]] = {"scalar": 0, "vec": 1, "mat": 2}
        permitted_functions: ClassVar[set] = permitted

    @dataclass(frozen=True)
    class _TorchInner(TorchAxisProtected["_TorchInner", Any]):
        leaf: Any
        protected_axes: ClassVar[dict[str, int]] = {"leaf": 1}
        permitted_functions: ClassVar[set] = permitted

    @dataclass(frozen=True)
    class _TorchOuter(TorchAxisProtected["_TorchOuter", Any]):
        inner: _TorchInner
        aux: Any
        protected_axes: ClassVar[dict[str, int]] = {"inner": 1, "aux": 2}
        permitted_functions: ClassVar[set] = permitted

    return torch, TorchAxisProtected, _TorchMixed, _TorchInner, _TorchOuter


def _make_mixed(backend: str, batch_shape: tuple[int, ...]):
    s = np.arange(1, int(np.prod(batch_shape)) + 1, dtype=np.float32).reshape(batch_shape)
    v = np.arange(1, int(np.prod((*batch_shape, 4))) + 1, dtype=np.float32).reshape((*batch_shape, 4))
    m = np.arange(1, int(np.prod((*batch_shape, 2, 3))) + 1, dtype=np.float32).reshape((*batch_shape, 2, 3))
    if backend == "numpy":
        return _NumpyMixed(s, v, m)
    if backend == "jax":
        _, jnp, _, _, mixed_cls, _, _ = _jax_classes()
        return mixed_cls(jnp.asarray(s), jnp.asarray(v), jnp.asarray(m))
    torch, _, mixed_cls, _, _ = _torch_classes()
    return mixed_cls(torch.tensor(s), torch.tensor(v), torch.tensor(m))


def _make_raw(backend: str, shape: tuple[int, ...]):
    arr = np.arange(2, int(np.prod(shape)) + 2, dtype=np.float32).reshape(shape)
    if backend == "numpy":
        return arr
    if backend == "jax":
        _, jnp, *_ = _jax_classes()
        return jnp.asarray(arr)
    torch, *_ = _torch_classes()
    return torch.tensor(arr)


def test_expand_for_protected_axes_contract() -> None:
    for scalar in (3, 2.5, True, 1 + 2j, np.float32(2.0), np.bool_(True)):
        assert expand_for_protected_axes(scalar, 2) is scalar

    np_0d = np.array(5.0)
    assert expand_for_protected_axes(np_0d, 2) is np_0d

    np_1d = np.array([1.0, 2.0])
    assert expand_for_protected_axes(np_1d, 0) is np_1d
    expanded_np = expand_for_protected_axes(np_1d, 2)
    assert expanded_np.shape == (2, 1, 1)
    assert np.shares_memory(expanded_np, np_1d)


@pytest.mark.parametrize("backend", ["numpy", "jax", "torch"])
@pytest.mark.parametrize(
    ("batch_shape", "raw_shape", "expected_batch"),
    [
        ((2, 3), (2, 3), (2, 3)),
        ((2, 3), (3,), (2, 3)),
        ((2, 3), (2, 1), (2, 3)),
        ((3,), (2, 3), (2, 3)),
        ((), (2, 3), (2, 3)),
        ((2, 3), (), (2, 3)),
        ((0, 3), (3,), (0, 3)),
    ],
)
@pytest.mark.parametrize("reflected", [False, True])
def test_mixed_rank_right_aligned_broadcasting(
    backend: str,
    batch_shape: tuple[int, ...],
    raw_shape: tuple[int, ...],
    expected_batch: tuple[int, ...],
    reflected: bool,
) -> None:
    x = _make_mixed(backend, batch_shape)
    raw = _make_raw(backend, raw_shape)
    raw_np = np.asarray(raw)

    result = operator.sub(raw, x) if reflected else operator.sub(x, raw)
    assert type(result) is type(x)
    assert result.shape == expected_batch

    for name, rank in x.protected_axes.items():
        field_np = np.asarray(getattr(x, name))
        aligned_raw = raw_np[(..., *(None,) * rank)] if raw_np.ndim > 0 and rank > 0 else raw_np
        expected = (aligned_raw - field_np) if reflected else (field_np - aligned_raw)
        np.testing.assert_allclose(np.asarray(getattr(result, name)), expected)


@pytest.mark.parametrize("backend", ["numpy", "jax", "torch"])
@pytest.mark.parametrize("reflected", [False, True])
def test_mixed_rank_comparisons_and_floor_remainder(backend: str, reflected: bool) -> None:
    x = _make_mixed(backend, (2, 3))
    raw = _make_raw(backend, (3,))
    raw_np = np.asarray(raw)

    comp = operator.lt(raw, x) if reflected else operator.lt(x, raw)
    assert tuple(comp.shape) == (2, 3)
    expected = np.ones((2, 3), dtype=bool)
    for name, rank in x.protected_axes.items():
        field_np = np.asarray(getattr(x, name))
        aligned = raw_np[(..., *(None,) * rank)] if rank > 0 else raw_np
        field_comp = (aligned < field_np) if reflected else (field_np < aligned)
        expected &= np.all(field_comp, axis=tuple(range(-rank, 0))) if rank > 0 else field_comp
    np.testing.assert_array_equal(np.asarray(comp), expected)

    q = operator.floordiv(raw, x) if reflected else operator.floordiv(x, raw)
    r = operator.mod(raw, x) if reflected else operator.mod(x, raw)
    assert q.shape == r.shape == (2, 3)
    for name, rank in x.protected_axes.items():
        field_np = np.asarray(getattr(x, name))
        aligned = raw_np[(..., *(None,) * rank)] if rank > 0 else raw_np
        eq = np.floor_divide(aligned, field_np) if reflected else np.floor_divide(field_np, aligned)
        er = np.remainder(aligned, field_np) if reflected else np.remainder(field_np, aligned)
        np.testing.assert_allclose(np.asarray(getattr(q, name)), eq)
        np.testing.assert_allclose(np.asarray(getattr(r, name)), er)


@pytest.mark.parametrize("backend", ["numpy", "jax"])
@pytest.mark.parametrize("reflected", [False, True])
def test_mixed_rank_divmod(backend: str, reflected: bool) -> None:
    x = _make_mixed(backend, (2, 3))
    raw = _make_raw(backend, (3,))
    raw_np = np.asarray(raw)

    q, r = divmod(raw, x) if reflected else divmod(x, raw)
    assert q.shape == r.shape == (2, 3)
    for name, rank in x.protected_axes.items():
        field_np = np.asarray(getattr(x, name))
        aligned = raw_np[(..., *(None,) * rank)] if rank > 0 else raw_np
        eq, er = np.divmod(aligned, field_np) if reflected else np.divmod(field_np, aligned)
        np.testing.assert_allclose(np.asarray(getattr(q, name)), eq)
        np.testing.assert_allclose(np.asarray(getattr(r, name)), er)


@pytest.mark.parametrize("reflected", [False, True])
def test_torch_mixed_rank_divmod_is_unsupported(reflected: bool) -> None:
    x = _make_mixed("torch", (2, 3))
    raw = _make_raw("torch", (3,))
    with pytest.raises(TypeError):
        divmod(raw, x) if reflected else divmod(x, raw)


@pytest.mark.parametrize("backend", ["numpy", "jax", "torch"])
def test_backend_entry_points_zero_protected_and_matmul(backend: str) -> None:
    x = _make_mixed(backend, (2, 3))
    raw = _make_raw(backend, (3,))
    raw_np = np.asarray(raw)

    if backend == "numpy":
        res = np.maximum(raw, x)
        np.testing.assert_allclose(res.mat, np.maximum(raw_np[None, :, None, None], x.mat))
        empty_x: Any = _NumpyInner(np.empty((2, 0), dtype=np.float32))
        empty_w: Any = np.array([1.0, 2.0], dtype=np.float32)
    elif backend == "jax":
        _, jnp, jf, _, _, inner_cls, _ = _jax_classes()
        res = jf.jax_subtract(raw_np, x)
        np.testing.assert_allclose(np.asarray(res.mat), raw_np[None, :, None, None] - np.asarray(x.mat))
        empty_x = inner_cls(jnp.empty((2, 0), dtype=jnp.float32))
        empty_w = jnp.array([1.0, 2.0], dtype=jnp.float32)
    else:
        torch, _, _, inner_cls, _ = _torch_classes()
        res = torch.add(input=x, other=raw, alpha=3)
        np.testing.assert_allclose(res.mat.numpy(), x.mat.numpy() + 3 * raw_np[None, :, None, None])
        empty_x = inner_cls(torch.empty((2, 0), dtype=torch.float32))
        empty_w = torch.tensor([1.0, 2.0], dtype=torch.float32)

    empty_res = empty_x + empty_w
    assert empty_res.shape == (2,)
    assert tuple(empty_res.leaf.shape) == (2, 0)

    # Protected matmul retains separate visible-dimension matrix semantics.
    right = _make_raw(backend, (3, 5))
    left = _make_raw(backend, (4, 2))
    res_r, res_l = x @ right, left @ x
    assert res_r.shape == (2, 5)
    assert res_l.shape == (4, 3)
    expected_vec = np.stack([np.asarray(x.vec)[..., i] @ np.asarray(right) for i in range(4)], axis=-1)
    np.testing.assert_allclose(np.asarray(res_r.vec), expected_vec, rtol=1e-5)


@pytest.mark.parametrize("backend", ["numpy", "jax", "torch"])
@pytest.mark.parametrize("raw_shape", [(3,), (5, 2, 3)])
@pytest.mark.parametrize("reflected", [False, True])
def test_nested_protected_fields_recurse_raw_alignment(
    backend: str, raw_shape: tuple[int, ...], reflected: bool
) -> None:
    leaf_np = np.arange(2 * 3 * 4 * 5, dtype=np.float32).reshape(2, 3, 4, 5)
    aux_np = np.arange(2 * 3 * 2 * 6, dtype=np.float32).reshape(2, 3, 2, 6)
    raw_np = np.arange(1, int(np.prod(raw_shape)) + 1, dtype=np.float32).reshape(raw_shape)

    if backend == "numpy":
        outer: Any = _NumpyOuter(_NumpyInner(leaf_np.copy()), aux_np.copy())
        raw: Any = raw_np
    elif backend == "jax":
        _, jnp, _, _, _, inner_cls, outer_cls = _jax_classes()
        outer = outer_cls(inner_cls(jnp.asarray(leaf_np)), jnp.asarray(aux_np))
        raw = jnp.asarray(raw_np)
    else:
        torch, _, _, inner_cls, outer_cls = _torch_classes()
        outer = outer_cls(inner_cls(torch.tensor(leaf_np)), torch.tensor(aux_np))
        raw = torch.tensor(raw_np)

    result = operator.sub(raw, outer) if reflected else operator.sub(outer, raw)
    expected_batch = np.broadcast_shapes((2, 3), raw_shape)
    assert type(result) is type(outer)
    assert type(result.inner) is type(outer.inner)
    assert result.shape == expected_batch
    assert result.inner.shape == (*expected_batch, 4)

    expected_leaf = (raw_np[..., None, None] - leaf_np) if reflected else (leaf_np - raw_np[..., None, None])
    expected_aux = (raw_np[..., None, None] - aux_np) if reflected else (aux_np - raw_np[..., None, None])
    np.testing.assert_allclose(np.asarray(result.inner.leaf), expected_leaf)
    np.testing.assert_allclose(np.asarray(result.aux), expected_aux)

    comp = operator.gt(raw, outer) if reflected else operator.gt(outer, raw)
    expected_comp = np.all(
        (raw_np[..., None, None] > leaf_np) if reflected else (leaf_np > raw_np[..., None, None]),
        axis=(-2, -1),
    ) & np.all(
        (raw_np[..., None, None] > aux_np) if reflected else (aux_np > raw_np[..., None, None]),
        axis=(-2, -1),
    )
    assert tuple(comp.shape) == expected_batch
    np.testing.assert_array_equal(np.asarray(comp), expected_comp)


@pytest.mark.parametrize("backend", ["numpy", "jax", "torch"])
def test_broadcasted_batch_and_init_false_fields_with_raw_operands(backend: str) -> None:
    if backend == "numpy":

        @dataclass(frozen=True)
        class NumpySharedBase(NumpyAxisProtected["NumpySharedBase", Any]):
            shared: np.ndarray
            values: np.ndarray
            protected_axes: ClassVar[dict[str, int]] = {"shared": 1, "values": 2}
            broadcasted_batch: ClassVar[set[str]] = {"shared"}
            permitted_ufuncs: ClassVar[dict[np.ufunc, list[str]]] = {np.multiply: ["__call__"]}

        @dataclass(frozen=True)
        class NumpyDerived(NumpySharedBase):
            shared: np.ndarray = field(init=False, default_factory=lambda: np.ones((1, 4)))

        x: Any = NumpySharedBase(np.ones((4,)), np.full((2, 3, 2, 4), 2.0))
        d: Any = NumpyDerived(np.full((2, 3, 2, 4), 2.0))
        w: Any = np.array([1.0, 2.0, 3.0])
    elif backend == "jax":
        _, jnp, jf, jax_protected_base, *_ = _jax_classes()

        @dataclass(frozen=True)
        class JaxSharedBase(jax_protected_base):
            shared: Any
            values: Any
            protected_axes: ClassVar[dict[str, int]] = {"shared": 1, "values": 2}
            broadcasted_batch: ClassVar[set[str]] = {"shared"}
            permitted_functions: ClassVar[set] = {jf.jax_multiply}

        @dataclass(frozen=True)
        class JaxDerived(JaxSharedBase):
            shared: Any = field(init=False, default_factory=lambda: jnp.ones((1, 4)))

        x = JaxSharedBase(jnp.ones((4,)), jnp.full((2, 3, 2, 4), 2.0))
        d = JaxDerived(jnp.full((2, 3, 2, 4), 2.0))
        w = jnp.array([1.0, 2.0, 3.0])
    else:
        torch, torch_protected_base, *_ = _torch_classes()

        @dataclass(frozen=True)
        class TorchSharedBase(torch_protected_base):
            shared: Any
            values: Any
            protected_axes: ClassVar[dict[str, int]] = {"shared": 1, "values": 2}
            broadcasted_batch: ClassVar[set[str]] = {"shared"}
            permitted_functions: ClassVar[set] = {torch.mul}

        @dataclass(frozen=True)
        class TorchDerived(TorchSharedBase):
            shared: Any = field(init=False, default_factory=lambda: torch.ones((1, 4)))

        x = TorchSharedBase(torch.ones((4,)), torch.full((2, 3, 2, 4), 2.0))
        d = TorchDerived(torch.full((2, 3, 2, 4), 2.0))
        w = torch.tensor([1.0, 2.0, 3.0])

    w_np = np.array([1.0, 2.0, 3.0])
    res = x * w
    assert res.shape == (2, 3)
    np.testing.assert_allclose(np.asarray(res.shared), np.broadcast_to(w_np[None, :, None], (2, 3, 4)))
    np.testing.assert_allclose(np.asarray(res.values), np.broadcast_to(2.0 * w_np[None, :, None, None], (2, 3, 2, 4)))

    res_d = d * w
    assert type(res_d) is type(d)
    assert res_d.shape == (2, 3)
    np.testing.assert_allclose(np.asarray(res_d.values), np.broadcast_to(2.0 * w_np[None, :, None, None], (2, 3, 2, 4)))


def test_numpy_sequences_inplace_output_only_and_postprocess() -> None:
    x = _make_mixed("numpy", (2,))
    weights = [10.0, 20.0]
    w = np.asarray(weights)
    for res in (x + weights, weights + x, np.add(x, tuple(weights))):
        assert isinstance(res, _NumpyMixed)
        np.testing.assert_allclose(res.vec, x.vec + w[:, None])
        np.testing.assert_allclose(res.mat, x.mat + w[:, None, None])

    # Nested in-place addition aligns raw operand recursively.
    nested = _NumpyOuter(_NumpyInner(np.ones((2, 3, 4))), np.ones((2, 2, 5)))
    nested += w
    np.testing.assert_allclose(nested.inner.leaf, np.broadcast_to((w + 1)[:, None, None], (2, 3, 4)))
    np.testing.assert_allclose(nested.aux, np.broadcast_to((w + 1)[:, None, None], (2, 2, 5)))

    # Output-only dispatch (no protected input) retains native physical raw operand alignment.
    dst = _NumpyInner(np.zeros((2, 3)))
    raw_phys = np.arange(6.0).reshape(2, 3)
    assert np.add(raw_phys, 1.0, out=dst) is dst  # ty: ignore[no-matching-overload]
    np.testing.assert_allclose(dst.leaf, raw_phys + 1.0)

    # NumPy ufunc where= keeps its native physical mask behavior alongside logical-batch raw inputs.
    single = _NumpyInner(np.ones((2, 3)))
    masked = _NumpyInner(np.full((2, 3), -1.0))
    assert np.add(single, w, out=masked, where=np.array([[True], [False]])) is masked  # ty: ignore[no-matching-overload]
    np.testing.assert_allclose(masked.leaf, [[11.0, 11.0, 11.0], [-1.0, -1.0, -1.0]])

    # Postprocess hook runs on elementwise results with raw operands.
    calls: list[tuple[int, ...]] = []

    @dataclass(frozen=True)
    class Hooked(_NumpyInner):
        def _postprocess_ufunc_result(self, result, *, ufunc, method):
            assert ufunc is np.multiply
            assert method == "__call__"
            calls.append(result.shape)
            return result * 2

    hooked = Hooked(np.ones((2, 4))) * np.array([2.0, 3.0])
    assert calls == [(2, 4)]
    np.testing.assert_allclose(hooked.leaf, [[4.0] * 4, [6.0] * 4])


def test_jax_nested_permissions_postprocess_and_transforms() -> None:
    jax, jnp, jf, jax_protected_base, _, inner_cls, outer_cls = _jax_classes()

    @dataclass(frozen=True)
    class DeniedInner(jax_protected_base):
        leaf: Any
        protected_axes: ClassVar[dict[str, int]] = {"leaf": 1}

    @dataclass(frozen=True)
    class PermittingOuter(jax_protected_base):
        inner: DeniedInner
        protected_axes: ClassVar[dict[str, int]] = {"inner": 1}
        permitted_functions: ClassVar[set] = {jf.jax_add}

    with pytest.raises(TypeError, match="no implementation found"):
        _ = PermittingOuter(DeniedInner(jnp.ones((2, 3, 4)))) + jnp.array([1.0, 2.0])

    seen: list[tuple[object, ...]] = []

    @dataclass(frozen=True)
    class Hooked(inner_cls):
        leaf: Any

        def _postprocess_elementwise_result(self, values, *, func, operands):
            assert func is jf.jax_multiply
            seen.append(operands)
            return values

    raw = jnp.array([2.0, 3.0])
    hx = Hooked(jnp.ones((2, 4)))
    _ = hx * raw
    assert seen == [(hx, raw)]

    x = _make_mixed("jax", (2, 3))
    w = jnp.array([2.0, 3.0, 4.0])
    jitted_scaled, jitted_mask = jax.jit(lambda a, weights: (a * weights + weights, a * weights > weights))(x, w)
    np.testing.assert_allclose(np.asarray(jitted_scaled.mat), np.asarray((x * w + w).mat))
    np.testing.assert_array_equal(np.asarray(jitted_mask), np.asarray(x * w > w))

    nested = outer_cls(inner_cls(jnp.ones((2, 3, 4, 5))), jnp.ones((2, 3, 2, 6)))
    nested_res = jax.jit(lambda o, weights: o * weights)(nested, w)
    expected_nested = np.broadcast_to(np.asarray(w)[None, :, None, None], (2, 3, 4, 5))
    np.testing.assert_allclose(np.asarray(nested_res.inner.leaf), expected_nested)

    vmapped = jax.vmap(lambda row, weights: row * weights, in_axes=(0, None))(x, w)
    np.testing.assert_allclose(np.asarray(vmapped.mat), np.asarray((x * w).mat))

    grad_w = jax.jit(jax.grad(lambda weights: jnp.sum((x * weights).vec) + jnp.sum((x * weights).mat)))(w)
    expected_grad = np.sum(np.asarray(x.vec), axis=(0, 2)) + np.sum(np.asarray(x.mat), axis=(0, 2, 3))
    np.testing.assert_allclose(np.asarray(grad_w), expected_grad, rtol=1e-5)


def test_torch_0d_promotion_inplace_output_only_postprocess_and_autograd() -> None:
    torch, _, mixed_cls, inner_cls, outer_cls = _torch_classes()

    # 0D float64 tensor preserves PyTorch scalar-tensor promotion (keeps float32); 1D promotes to float64.
    x = mixed_cls(
        torch.ones((2,), dtype=torch.float32),
        torch.ones((2, 3), dtype=torch.float32),
        torch.ones((2, 2, 3), dtype=torch.float32),
    )
    res_0d = x + torch.tensor(2.0, dtype=torch.float64)
    assert res_0d.scalar.dtype == res_0d.vec.dtype == res_0d.mat.dtype == torch.float32
    res_1d = x + torch.tensor([2.0, 3.0], dtype=torch.float64)
    assert res_1d.scalar.dtype == res_1d.vec.dtype == res_1d.mat.dtype == torch.float64

    # Nested in-place addition aligns raw operand recursively.
    nested = outer_cls(inner_cls(torch.ones((2, 3, 4))), torch.ones((2, 2, 5)))
    w = torch.tensor([10.0, 20.0])
    nested += w
    expected_nested = np.array([11.0, 21.0])[:, None, None]
    np.testing.assert_allclose(nested.inner.leaf.numpy(), np.broadcast_to(expected_nested, (2, 3, 4)))
    np.testing.assert_allclose(nested.aux.numpy(), np.broadcast_to(expected_nested, (2, 2, 5)))

    # Output-only dispatch (no protected input) retains native physical raw operand alignment.
    dst = inner_cls(torch.zeros((2, 3)))
    raw_phys = torch.arange(6.0).reshape(2, 3)
    assert torch.add(raw_phys, 1.0, out=dst) is dst
    np.testing.assert_allclose(dst.leaf.numpy(), (raw_phys + 1.0).numpy())

    # Postprocess hook receives original unexpanded raw operand.
    seen: list[tuple[object, ...]] = []

    @dataclass(frozen=True)
    class Hooked(inner_cls):
        leaf: Any

        def _postprocess_elementwise_result(self, values, *, func, operands):
            assert func is torch.mul
            seen.append(operands)
            return values

    hx = Hooked(torch.ones((2, 4)))
    _ = w * hx
    assert seen == [(w, hx)]

    # Autograd sums raw-weight gradients over protected axes.
    mx = _make_mixed("torch", (2, 3))
    weights = torch.tensor([2.0, 3.0, 4.0], requires_grad=True)
    out = mx * weights
    (out.vec.sum() + out.mat.sum()).backward()
    assert weights.grad is not None
    expected_grad = mx.vec.sum(dim=(0, 2)) + mx.mat.sum(dim=(0, 2, 3))
    np.testing.assert_allclose(weights.grad.numpy(), expected_grad.numpy(), rtol=1e-5)
