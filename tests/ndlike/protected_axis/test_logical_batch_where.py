"""Focused regression tests for three-argument logical-batch ``where`` on protected-axis arrays."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, assert_type

import numpy as np
import pytest

from ndlike.protected_axis.numpy import NumpyAxisProtected

if TYPE_CHECKING:
    import jax

    from ndlike.jax_functions import jax_where as _jax_where_type
    from ndlike.protected_axis.jax import JaxAxisProtected as _JaxAxisProtected

    class _TypedJax(_JaxAxisProtected["_TypedJax"]): ...

    def _check_where_static_types(
        j_cond: jax.Array,
        j_arr: jax.Array,
        j_prot: _TypedJax,
    ) -> None:
        assert_type(_jax_where_type(j_cond, j_arr, j_arr), jax.Array)
        assert_type(_jax_where_type(j_cond, j_prot, j_prot), _TypedJax)
        assert_type(_jax_where_type(j_cond, j_prot, 0.0), _TypedJax)
        assert_type(_jax_where_type(j_cond, j_arr, j_prot), _TypedJax)


@dataclass(frozen=True)
class _BackendContext:
    name: str
    base: type[Any]
    where: Callable[..., Any]
    array: Callable[..., Any]
    bool_array: Callable[..., Any]
    float32: Any
    permitted: set[Callable[..., Any]]

    def to_numpy(self, value: object) -> np.ndarray:
        return np.asarray(value)


def _make_context(backend: str) -> _BackendContext:
    if backend == "numpy":
        return _BackendContext(
            name="numpy",
            base=NumpyAxisProtected,
            where=np.where,
            array=lambda data, dtype=None: np.asarray(data, dtype=np.float64 if dtype is None else dtype),
            bool_array=lambda data: np.asarray(data, dtype=np.bool_),
            float32=np.float32,
            permitted={np.where},
        )
    if backend == "jax":
        pytest.importorskip("jax")
        import jax.numpy as jnp  # noqa: PLC0415

        from ndlike.jax_functions import jax_where  # noqa: PLC0415
        from ndlike.protected_axis.jax import JaxAxisProtected  # noqa: PLC0415

        return _BackendContext(
            name="jax",
            base=JaxAxisProtected,
            where=jax_where,
            array=lambda data, dtype=None: jnp.asarray(data, dtype=jnp.float32 if dtype is None else dtype),
            bool_array=lambda data: jnp.asarray(data, dtype=jnp.bool_),
            float32=jnp.float32,
            permitted={jax_where},
        )
    pytest.importorskip("torch")
    import torch  # noqa: PLC0415

    from ndlike.protected_axis.torch import TorchAxisProtected  # noqa: PLC0415

    return _BackendContext(
        name="torch",
        base=TorchAxisProtected,
        where=torch.where,
        array=lambda data, dtype=None: torch.as_tensor(data, dtype=torch.float32 if dtype is None else dtype),
        bool_array=lambda data: torch.as_tensor(data, dtype=torch.bool),
        float32=torch.float32,
        permitted={torch.where},
    )


@pytest.fixture(params=["numpy", "jax", "torch"])
def ctx(request: pytest.FixtureRequest) -> _BackendContext:
    return _make_context(request.param)


def _make_single_cls(ctx: _BackendContext, axes: int = 1, *, permitted: bool = True) -> type[Any]:
    @dataclass(frozen=True)
    class Single(ctx.base):
        array: Any
        protected_axes: ClassVar[dict[str, int]] = {"array": axes}
        permitted_functions: ClassVar[set[Callable[..., Any]]] = set(ctx.permitted) if permitted else set()

    return Single


def _make_multi_cls(ctx: _BackendContext, *, permitted: bool = True) -> type[Any]:
    @dataclass(frozen=True)
    class Multi(ctx.base):
        scalar_field: Any
        vec_field: Any
        mat_field: Any
        protected_axes: ClassVar[dict[str, int]] = {"scalar_field": 0, "vec_field": 1, "mat_field": 2}
        permitted_functions: ClassVar[set[Callable[..., Any]]] = set(ctx.permitted) if permitted else set()

    return Multi


def _make_nested_classes(
    ctx: _BackendContext, *, outer_permitted: bool = True, inner_permitted: bool = True
) -> tuple[type[Any], type[Any]]:
    @dataclass(frozen=True)
    class Inner(ctx.base):
        vec: Any
        mat: Any
        protected_axes: ClassVar[dict[str, int]] = {"vec": 1, "mat": 2}
        permitted_functions: ClassVar[set[Callable[..., Any]]] = set(ctx.permitted) if inner_permitted else set()

    @dataclass(frozen=True)
    class Outer(ctx.base):
        inner: Any
        aux: Any
        protected_axes: ClassVar[dict[str, int]] = {"inner": 1, "aux": 2}
        permitted_functions: ClassVar[set[Callable[..., Any]]] = set(ctx.permitted) if outer_permitted else set()

    return Outer, Inner


def test_multifield_pair_broadcasting_scalar_cond_and_broadcasted_batch(ctx: _BackendContext) -> None:
    cls = _make_multi_cls(ctx)
    x = cls(
        ctx.array(np.arange(6.0).reshape(2, 3)),
        ctx.array(np.arange(24.0).reshape(2, 3, 4)),
        ctx.array(np.arange(36.0).reshape(2, 3, 2, 3)),
    )
    y = cls(
        ctx.array(np.arange(3.0).reshape(1, 3) + 50.0),
        ctx.array(np.arange(12.0).reshape(1, 3, 4) + 50.0),
        ctx.array(np.arange(18.0).reshape(1, 3, 2, 3) + 50.0),
    )
    cond_raw = np.array([[[True], [False]], [[False], [True]]])  # (2, 2, 1) -> logical batch (2, 2, 3)

    result = ctx.where(ctx.bool_array(cond_raw), x, y)
    assert isinstance(result, cls)
    assert result.shape == (2, 2, 3)
    for name, axes in cls.protected_axes.items():
        expected = np.where(
            cond_raw[(..., *(None,) * axes)],
            ctx.to_numpy(getattr(x, name)),
            ctx.to_numpy(getattr(y, name)),
        )
        np.testing.assert_allclose(ctx.to_numpy(getattr(result, name)), expected)

    # 0D boolean mask (plus Python bool on NumPy/JAX), 0D logical batch, and empty batch/protected axes.
    single = _make_single_cls(ctx, axes=1)
    s_res = ctx.where(ctx.bool_array(False), single(ctx.array([1.0, 2.0])), single(ctx.array([3.0, 4.0])))
    assert s_res.shape == ()
    np.testing.assert_allclose(ctx.to_numpy(s_res.array), [3.0, 4.0])
    if ctx.name != "torch":
        py_res = ctx.where(False, single(ctx.array([1.0, 2.0])), single(ctx.array([3.0, 4.0])))
        assert py_res.shape == ()
        np.testing.assert_allclose(ctx.to_numpy(py_res.array), [3.0, 4.0])
    e_res = ctx.where(
        ctx.bool_array(np.empty((0, 1), dtype=bool)),
        single(ctx.array(np.empty((0, 3, 0), dtype=np.float32))),
        single(ctx.array(np.ones((1, 3, 0), dtype=np.float32))),
    )
    assert e_res.shape == (0, 3)
    assert e_res.array.shape == (0, 3, 0)

    # broadcasted_batch fields materialize cleanly.
    @dataclass(frozen=True)
    class Shared(ctx.base):
        shared: Any
        values: Any
        protected_axes: ClassVar[dict[str, int]] = {"shared": 1, "values": 1}
        broadcasted_batch: ClassVar[set[str]] = {"shared"}
        permitted_functions: ClassVar[set[Callable[..., Any]]] = set(ctx.permitted)

    sh_res = ctx.where(
        ctx.bool_array([True, False]),
        Shared(ctx.array([1.0, 2.0]), ctx.array([[10.0, 20.0], [30.0, 40.0]])),
        Shared(ctx.array([5.0, 6.0]), ctx.array([[50.0, 60.0], [70.0, 80.0]])),
    )
    np.testing.assert_allclose(ctx.to_numpy(sh_res.shared), [[1.0, 2.0], [5.0, 6.0]])
    np.testing.assert_allclose(ctx.to_numpy(sh_res.values), [[10.0, 20.0], [70.0, 80.0]])


@pytest.mark.parametrize("protected_first", [True, False])
@pytest.mark.parametrize("raw_kind", ["scalar", "0d", "batch"])
def test_protected_with_raw_branch_in_either_position(
    ctx: _BackendContext, protected_first: bool, raw_kind: str
) -> None:
    cls = _make_multi_cls(ctx)
    x = cls(
        ctx.array(np.arange(3.0, dtype=np.float32).reshape(1, 3), dtype=ctx.float32),
        ctx.array(np.arange(12.0, dtype=np.float32).reshape(1, 3, 4), dtype=ctx.float32),
        ctx.array(np.arange(18.0, dtype=np.float32).reshape(1, 3, 2, 3), dtype=ctx.float32),
    )
    cond_np = np.array([[True, False, True], [False, True, False]])
    if raw_kind == "scalar":
        raw: object = -5.0
        raw_np: object = -5.0
    elif raw_kind == "0d":
        raw_np = np.array(-5.0, dtype=np.float32)
        raw = ctx.array(raw_np, dtype=ctx.float32)
    else:
        raw_np = np.array([[10.0], [20.0]], dtype=np.float32)
        raw = ctx.array(raw_np, dtype=ctx.float32)

    args = (x, raw) if protected_first else (raw, x)
    result = ctx.where(ctx.bool_array(cond_np), *args)

    assert isinstance(result, cls)
    assert result.shape == (2, 3)
    for name, axes in cls.protected_axes.items():
        field_res = getattr(result, name)
        assert field_res.dtype == ctx.float32
        field_np = ctx.to_numpy(getattr(x, name))
        raw_exp = raw_np[(..., *(None,) * axes)] if isinstance(raw_np, np.ndarray) and raw_np.ndim > 0 else raw_np
        expected = np.where(
            cond_np[(..., *(None,) * axes)],
            field_np if protected_first else raw_exp,
            raw_exp if protected_first else field_np,
        )
        np.testing.assert_allclose(ctx.to_numpy(field_res), expected)


def test_nested_recursion_raw_branch_and_pair_layout_parity(ctx: _BackendContext) -> None:
    outer_cls, inner_cls = _make_nested_classes(ctx)
    x = outer_cls(
        inner_cls(
            ctx.array(np.arange(2 * 3 * 5 * 4, dtype=np.float32).reshape(2, 3, 5, 4)),
            ctx.array(np.arange(2 * 3 * 5 * 2 * 3, dtype=np.float32).reshape(2, 3, 5, 2, 3)),
        ),
        ctx.array(np.arange(2 * 3 * 2 * 4, dtype=np.float32).reshape(2, 3, 2, 4)),
    )
    y = outer_cls(
        inner_cls(
            ctx.array(np.ones((1, 3, 5, 4), dtype=np.float32) * 9.0),
            ctx.array(np.ones((1, 3, 5, 2, 3), dtype=np.float32) * 8.0),
        ),
        ctx.array(np.ones((1, 3, 2, 4), dtype=np.float32) * 7.0),
    )
    cond_np = np.array([[True, False, True], [False, True, False]])
    raw_np = np.array([[10.0], [20.0]], dtype=np.float32)

    # Nested pair + nested with entire raw batch array / scalar in either position.
    pair_res = ctx.where(ctx.bool_array(cond_np), x, y)
    assert isinstance(pair_res, outer_cls)
    assert isinstance(pair_res.inner, inner_cls)
    assert pair_res.shape == (2, 3)
    assert pair_res.inner.shape == (2, 3, 5)
    np.testing.assert_allclose(
        ctx.to_numpy(pair_res.inner.vec),
        np.where(cond_np[..., None, None], ctx.to_numpy(x.inner.vec), ctx.to_numpy(y.inner.vec)),
    )

    for left, right, raw_first, raw_exp in (
        (x, ctx.array(raw_np), False, raw_np[..., None, None]),
        (ctx.array(raw_np), x, True, raw_np[..., None, None]),
        (x, -1.5, False, -1.5),
    ):
        res = ctx.where(ctx.bool_array(cond_np), left, right)
        assert isinstance(res, outer_cls)
        assert isinstance(res.inner, inner_cls)
        expected = np.where(
            cond_np[..., None, None],
            raw_exp if raw_first else ctx.to_numpy(x.inner.vec),
            ctx.to_numpy(x.inner.vec) if raw_first else raw_exp,
        )
        np.testing.assert_allclose(ctx.to_numpy(res.inner.vec), expected)

    # Asymmetric nested field vs native leaf of identical visible shape (2, 3, 5) must be rejected in both orders.
    leaf_outer = outer_cls(ctx.array(np.zeros((2, 3, 5), dtype=np.float32)), ctx.array(np.zeros((2, 3, 2, 4))))
    with pytest.raises(TypeError):
        ctx.where(ctx.bool_array(cond_np), x, leaf_outer)
    with pytest.raises(TypeError):
        ctx.where(ctx.bool_array(cond_np), leaf_outer, x)

    # Inner permission is enforced during recursion.
    outer_only, denied_inner = _make_nested_classes(ctx, outer_permitted=True, inner_permitted=False)
    blocked = outer_only(
        denied_inner(ctx.array(np.ones((2, 3, 4))), ctx.array(np.ones((2, 3, 2, 2)))),
        ctx.array(np.ones((2, 2, 2))),
    )
    with pytest.raises(TypeError):
        ctx.where(ctx.bool_array([True, False]), blocked, blocked)


def test_inf_nan_masking_and_permissions_and_hooks(ctx: _BackendContext) -> None:
    cls = _make_single_cls(ctx, axes=1)
    x = cls(ctx.array(np.array([[1.0, 2.0, 3.0], [np.nan, np.inf, -np.inf]], dtype=np.float32)))
    y = cls(ctx.array(np.array([[np.inf, -np.inf, np.nan], [4.0, 5.0, 6.0]], dtype=np.float32)))
    cond = ctx.bool_array([True, False])

    for args, expected in (
        ((x, y), [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        ((x, 0.0), [[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]),
        ((0.0, y), [[0.0, 0.0, 0.0], [4.0, 5.0, 6.0]]),
    ):
        res = ctx.where(cond, *args)
        assert np.all(np.isfinite(ctx.to_numpy(res.array)))
        np.testing.assert_allclose(ctx.to_numpy(res.array), expected)

    # Opt-in permission & instance veto on either branch.
    denied = _make_single_cls(ctx, axes=1, permitted=False)(ctx.array(np.ones((2, 3))))
    for args in ((denied, denied), (denied, 0.0), (0.0, denied)):
        with pytest.raises(TypeError):
            ctx.where(cond, *args)

    @dataclass(frozen=True)
    class Veto(cls):
        array: Any
        deny: bool = False

        def protected_values(self, func: Callable[..., Any] | None = None):
            if self.deny and func is not None:
                return None
            return super().protected_values() if func is None else super().protected_values(func)

    ok, vetoed = Veto(ctx.array(np.ones((2, 3))), deny=False), Veto(ctx.array(np.ones((2, 3))), deny=True)
    for args in ((ok, vetoed), (vetoed, ok)):
        with pytest.raises(TypeError):
            ctx.where(cond, *args)

    # Input postprocessing and reconstruction context receive the where callable.
    seen: list[tuple[str, Callable[..., Any] | None]] = []

    @dataclass(frozen=True)
    class Hooked(ctx.base):
        array: Any
        protected_axes: ClassVar[dict[str, int]] = {"array": 1}
        permitted_functions: ClassVar[set[Callable[..., Any]]] = set(ctx.permitted)

        def _postprocess_protected_values(self, values: dict[str, Any], func: Callable[..., Any]) -> dict[str, Any]:
            seen.append(("input", func))
            return {"array": values["array"] + 1.0}

        def with_protected_values(self, values: dict[str, Any], func: Callable[..., Any] | None = None) -> Hooked:
            seen.append(("create", func))
            return Hooked(values["array"])

    h_res = ctx.where(cond, Hooked(ctx.array(np.zeros((2, 3)))), 10.0)
    np.testing.assert_allclose(ctx.to_numpy(h_res.array), [[1.0, 1.0, 1.0], [10.0, 10.0, 10.0]])
    assert seen == [("input", ctx.where), ("create", ctx.where)]


def test_validation_rejects_incompatible_shapes_conditions_and_types(ctx: _BackendContext) -> None:
    cls = _make_single_cls(ctx, axes=1)
    x = cls(ctx.array(np.ones((2, 3))))
    cond = ctx.bool_array([True, False])

    with pytest.raises(ValueError, match="identical protected trailing shapes"):
        ctx.where(cond, x, cls(ctx.array(np.ones((2, 1)))))
    with pytest.raises(ValueError, match="shape mismatch"):
        ctx.where(cond, x, cls(ctx.array(np.ones((4, 3)))))
    with pytest.raises(ValueError, match="shape mismatch"):
        ctx.where(ctx.bool_array([True, False, True]), x, x)
    # Storage-shaped raw array (2, 3) is logical batch (2, 3) and must not match protected axis 3.
    with pytest.raises(ValueError, match="shape mismatch"):
        ctx.where(cond, x, ctx.array(np.zeros((2, 3))))

    for bad_cond in (
        ctx.array(np.array([1, 0], dtype=np.int32), dtype=np.int32 if ctx.name == "numpy" else None),
        1,
        x,
    ):
        with pytest.raises(TypeError):
            ctx.where(bad_cond, x, x)
    with pytest.raises(TypeError):
        ctx.where(x, ctx.array(np.ones((2, 3))), ctx.array(np.zeros((2, 3))))

    @dataclass(frozen=True)
    class SubCls(cls):
        array: Any

    with pytest.raises(TypeError):
        ctx.where(cond, x, SubCls(ctx.array(np.ones((2, 3)))))
    with pytest.raises(TypeError):
        ctx.where(cond, x, object())

    multi = _make_multi_cls(ctx)
    unsynced = multi(ctx.array(np.ones(2)), ctx.array(np.ones((5, 4))), ctx.array(np.ones((2, 2, 3))))
    with pytest.raises(ValueError, match="inconsistent batch-shapes"):
        ctx.where(cond, unsynced, unsynced)

    @dataclass(frozen=True)
    class DerivedOnly(ctx.base):
        derived: Any = field(init=False)
        protected_axes: ClassVar[dict[str, int]] = {"derived": 1}
        permitted_functions: ClassVar[set[Callable[..., Any]]] = set(ctx.permitted)

        def __post_init__(self) -> None:
            object.__setattr__(self, "derived", ctx.array(np.ones((2, 3))))

    with pytest.raises(ValueError, match="at least one effective protected field"):
        ctx.where(cond, DerivedOnly(), DerivedOnly())


def test_numpy_foreign_ndarray_subclass_override_and_arities() -> None:
    cls = _make_single_cls(_make_context("numpy"), axes=1)
    x = cls(np.arange(6.0).reshape(2, 3))

    # Plain list condition & raw list branch work; non-bool list rejected.
    res = np.where([True, False], x, [10.0, 20.0])
    assert isinstance(res, cls)
    np.testing.assert_array_equal(res.array, [[0.0, 1.0, 2.0], [20.0, 20.0, 20.0]])
    with pytest.raises(TypeError, match="boolean dtype"):
        np.where([1, 0], x, x)

    # Foreign ndarray subclass defining __array_function__ receives top-level dispatch.
    sentinel = object()

    class CustomSub(np.ndarray):
        def __array_function__(self, func, types, args, kwargs) -> object:
            del types, args, kwargs
            return sentinel if func is np.where else NotImplemented

    sub_cond = np.array([True, False]).view(CustomSub)
    sub_vals = np.array([10.0, 20.0]).view(CustomSub)
    assert np.where(sub_cond, x, 0.0) is sentinel
    assert np.where(np.array([True, False]), x, sub_vals) is sentinel
    assert np.where(np.array([True, False]), sub_vals, x) is sentinel
    # Foreign overrides nested in a Python list are also declined (not silently coerced).
    with pytest.raises(TypeError):
        np.where([sub_cond], x, 0.0)
    with pytest.raises(TypeError):
        np.where(np.array([True, False]), x, [sub_vals])

    with pytest.raises(TypeError):
        np.where(x)
    with pytest.raises(ValueError, match="either both or neither"):
        np.where(np.array([True, False]), x)


def test_jax_jit_grad_postprocess_and_sidecar() -> None:
    pytest.importorskip("jax")
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    from ndlike.jax_functions import jax_where  # noqa: PLC0415
    from ndlike.protected_axis.jax import JaxAxisProtected  # noqa: PLC0415

    ctx = _make_context("jax")
    single_cls = _make_single_cls(ctx, axes=1)
    outer_cls, inner_cls = _make_nested_classes(ctx)
    cond = jnp.array([True, False])

    x = single_cls(jnp.array([[1.0, 2.0, 3.0], [jnp.nan, jnp.inf, -jnp.inf]]))
    y = single_cls(jnp.array([[jnp.inf, jnp.nan, 0.0], [4.0, 5.0, 6.0]]))
    jit_res = jax.jit(jax_where)(cond, x, y)
    np.testing.assert_allclose(jit_res.array, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    np.testing.assert_allclose(jax.vmap(jax_where)(cond, x, y).array, jit_res.array)

    grad_x = jax.grad(lambda a: jnp.sum(jax_where(cond, a, 0.0).array))(x)
    np.testing.assert_allclose(grad_x.array, [[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]])

    nested_x = outer_cls(inner_cls(jnp.ones((2, 3, 4)), jnp.ones((2, 3, 2, 2))), jnp.ones((2, 2, 2)))
    jit_nested = jax.jit(jax_where)(cond, nested_x, jnp.array([10.0, 20.0]))
    np.testing.assert_allclose(jit_nested.inner.vec[0], 1.0)
    np.testing.assert_allclose(jit_nested.inner.vec[1], 20.0)

    # _postprocess_elementwise_result allows compact broadcasted_batch but rejects batch/protected shape changes.
    recorded: list[tuple[Callable[..., Any], tuple[object, ...]]] = []

    @dataclass(frozen=True)
    class Postprocessed(JaxAxisProtected):
        shared: Any
        values: Any
        mode: str = "compact"
        protected_axes: ClassVar[dict[str, int]] = {"shared": 1, "values": 1}
        broadcasted_batch: ClassVar[set[str]] = {"shared"}
        permitted_functions: ClassVar[set[Callable[..., Any]]] = {jax_where}

        def _postprocess_elementwise_result(
            self, values: dict[str, Any], *, func: Callable[..., Any], operands: tuple[object, ...]
        ) -> dict[str, Any]:
            recorded.append((func, operands))
            if self.mode == "compact":
                return {"shared": values["shared"][0], "values": values["values"] * 2.0}
            if self.mode == "bad-batch":
                return {"shared": values["shared"][:1], "values": values["values"][:1]}
            return {"shared": values["shared"], "values": values["values"][..., :1]}

    p = Postprocessed(jnp.ones((3,)), jnp.ones((2, 3)))
    out = jax_where(cond, 5.0, p)
    assert isinstance(out, Postprocessed)
    assert out.shared.shape == (3,)
    np.testing.assert_allclose(out.values, [[10.0, 10.0, 10.0], [2.0, 2.0, 2.0]])
    assert recorded == [(jax_where, (cond, 5.0, p))]

    with pytest.raises(ValueError, match="batch shape"):
        jax_where(cond, Postprocessed(jnp.ones((3,)), jnp.ones((2, 3)), mode="bad-batch"), 0.0)
    with pytest.raises(ValueError, match="modified protected trailing axes"):
        jax_where(cond, Postprocessed(jnp.ones((3,)), jnp.ones((2, 3)), mode="bad-prot"), 0.0)

    @dataclass(frozen=True)
    class Sidecar(JaxAxisProtected):
        array: Any
        sidecar: np.ndarray
        protected_axes: ClassVar[dict[str, int]] = {"array": 1, "sidecar": 0}
        permitted_functions: ClassVar[set[Callable[..., Any]]] = {jax_where}

    with pytest.raises(TypeError):
        jax_where(cond, Sidecar(jnp.ones((2, 3)), np.ones(2)), 0.0)


def test_torch_native_where_device_autograd_and_rejections() -> None:
    pytest.importorskip("torch")
    import torch  # noqa: PLC0415

    from ndlike import torch_functions as tf  # noqa: PLC0415
    from ndlike.protected_axis.torch import TorchAxisProtected  # noqa: PLC0415

    # Torch mirrors native torch.where directly without a custom torch_where wrapper.
    assert not hasattr(tf, "torch_where")
    assert "torch_where" not in tf.__all__
    assert "torch_average" in tf.__all__

    native_where: Any = torch.where

    # Mixed-device (CPU + meta) protected fields work with a 0D boolean tensor mask,
    # while Python / NumPy scalar booleans are rejected like native torch.where.
    @dataclass(frozen=True)
    class MixedDevice(TorchAxisProtected):
        cpu_field: Any
        meta_field: Any
        protected_axes: ClassVar[dict[str, int]] = {"cpu_field": 1, "meta_field": 1}
        permitted_functions: ClassVar[set[Callable[..., Any]]] = {torch.where}

    mixed = MixedDevice(torch.ones((2, 3), device="cpu"), torch.ones((2, 3), device="meta"))
    mixed_res = native_where(torch.tensor(True), mixed, 0.0)
    assert isinstance(mixed_res, MixedDevice)
    assert mixed_res.cpu_field.device.type == "cpu"
    assert mixed_res.meta_field.device.type == "meta"
    torch.testing.assert_close(mixed_res.cpu_field, torch.ones((2, 3)))
    for scalar_bool in (True, False, np.bool_(True)):
        with pytest.raises(TypeError):
            native_where(scalar_bool, mixed, 0.0)

    # Native torch.where (positional, keyword, 1-arg/out/uint8/x-y rejection) + autograd on masked nonfinites.
    @dataclass(frozen=True)
    class Single(TorchAxisProtected):
        array: Any
        protected_axes: ClassVar[dict[str, int]] = {"array": 1}
        permitted_functions: ClassVar[set[Callable[..., Any]]] = {torch.where}

    x_leaf = torch.tensor([[1.0, 2.0, 3.0], [float("nan"), float("inf"), float("-inf")]], requires_grad=True)
    y_leaf = torch.tensor([[float("inf"), float("nan"), 0.0], [4.0, 5.0, 6.0]], requires_grad=True)
    cond = torch.tensor([True, False])
    grad_res = native_where(cond, Single(x_leaf), Single(y_leaf))
    assert isinstance(grad_res, Single)
    grad_res.array.sum().backward()
    assert x_leaf.grad is not None
    assert y_leaf.grad is not None
    torch.testing.assert_close(x_leaf.grad, torch.tensor([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]))
    torch.testing.assert_close(y_leaf.grad, torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]))

    kw_res = native_where(condition=cond, input=Single(torch.ones((2, 3))), other=2.0)
    torch.testing.assert_close(kw_res.array, torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]))
    for options in ({"input": torch.tensor(2.0), "out": None}, {"self": 2.0}):
        kw_rev = native_where(condition=cond, other=Single(torch.ones((2, 3))), **options)
        torch.testing.assert_close(kw_rev.array, torch.tensor([[2.0, 2.0, 2.0], [1.0, 1.0, 1.0]]))
    with pytest.raises(TypeError):
        native_where(condition=cond, x=Single(torch.ones((2, 3))), y=2.0)
    with pytest.raises(TypeError):
        native_where(Single(torch.ones((2, 3))))
    with pytest.raises(TypeError):
        native_where(cond, Single(torch.ones((2, 3))), Single(torch.zeros((2, 3))), out=Single(torch.empty((2, 3))))
    with pytest.raises(TypeError, match="boolean dtype"):
        native_where(torch.tensor([1, 0], dtype=torch.uint8), Single(torch.ones((2, 3))), 0.0)


def test_torch_native_where_postprocess_and_sidecar() -> None:
    pytest.importorskip("torch")
    import torch  # noqa: PLC0415

    from ndlike.protected_axis.torch import TorchAxisProtected  # noqa: PLC0415

    native_where: Any = torch.where
    cond = torch.tensor([True, False])

    # _postprocess_elementwise_result validates expected batch shape while allowing compact broadcasted_batch.
    recorded: list[tuple[Callable[..., Any], tuple[object, ...]]] = []

    @dataclass(frozen=True)
    class Postprocessed(TorchAxisProtected):
        shared: Any
        values: Any
        mode: str = "compact"
        protected_axes: ClassVar[dict[str, int]] = {"shared": 1, "values": 1}
        broadcasted_batch: ClassVar[set[str]] = {"shared"}
        permitted_functions: ClassVar[set[Callable[..., Any]]] = {torch.where}

        def _postprocess_elementwise_result(
            self, values: dict[str, Any], *, func: Callable[..., Any], operands: tuple[object, ...]
        ) -> dict[str, Any]:
            recorded.append((func, operands))
            if self.mode == "compact":
                return {"shared": values["shared"][0], "values": values["values"] * 2.0}
            return {"shared": values["shared"][:1], "values": values["values"][:1]}

    p = Postprocessed(torch.ones((3,)), torch.ones((2, 3)))
    p_res = native_where(cond, 5.0, p)
    assert isinstance(p_res, Postprocessed)
    assert tuple(p_res.shared.shape) == (3,)
    torch.testing.assert_close(p_res.values, torch.tensor([[10.0, 10.0, 10.0], [2.0, 2.0, 2.0]]))
    assert recorded == [(torch.where, (cond, 5.0, p))]
    with pytest.raises(ValueError, match="batch shape"):
        native_where(cond, Postprocessed(torch.ones((3,)), torch.ones((2, 3)), mode="bad-batch"), 0.0)

    @dataclass(frozen=True)
    class Sidecar(TorchAxisProtected):
        array: Any
        sidecar: np.ndarray
        protected_axes: ClassVar[dict[str, int]] = {"array": 1, "sidecar": 0}
        permitted_functions: ClassVar[set[Callable[..., Any]]] = {torch.where}

    with pytest.raises(TypeError):
        native_where(cond, Sidecar(torch.ones((2, 3)), np.ones(2)), 0.0)
