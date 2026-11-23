"""Regression tests for ``zeros_like``, ``ones_like``, and ``full_like`` on protected-axis arrays."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
from typing import TYPE_CHECKING, Any, ClassVar, assert_type

import numpy as np
import pytest

from ndlike.protected_axis.numpy import NumpyAxisProtected

if TYPE_CHECKING:
    import jax

    from ndlike import jax_functions as jf
    from ndlike.protected_axis.jax import JaxAxisProtected

    class _JaxOwner(JaxAxisProtected["_JaxOwner"]): ...

    def _check_static_like_signatures(
        jax_array: jax.Array,
        jax_owner: _JaxOwner,
        numpy_array: np.ndarray,
    ) -> None:
        assert_type(jf.jax_zeros_like(jax_array), jax.Array)
        assert_type(jf.jax_ones_like(jax_array, shape=(2, 3)), jax.Array)
        assert_type(jf.jax_full_like(jax_array, 5.0, dtype="float32", shape=4), jax.Array)
        assert_type(jf.jax_zeros_like(numpy_array), jax.Array)
        assert_type(jf.jax_ones_like(1.0), jax.Array)
        assert_type(jf.jax_zeros_like(jax_owner), _JaxOwner)
        assert_type(jf.jax_ones_like(jax_owner, shape=(2, 3)), _JaxOwner)
        assert_type(jf.jax_full_like(jax_owner, 5.0, shape=()), _JaxOwner)

        # Native results must not be inferred as custom owner types from destination annotations.
        wrong_jax: _JaxOwner
        wrong_jax = jf.jax_zeros_like(jax_array)  # ty: ignore[invalid-assignment]
        wrong_jax = jf.jax_ones_like(jax_array)  # ty: ignore[invalid-assignment]
        wrong_jax = jf.jax_full_like(jax_array, 1.0)  # ty: ignore[invalid-assignment]
        del wrong_jax


@pytest.fixture(params=["numpy", "jax", "torch"])
def backend(request: pytest.FixtureRequest) -> dict[str, Any]:
    name = request.param
    pytest.importorskip(name)
    arrays = import_module("jax.numpy" if name == "jax" else name)
    protected_module = import_module(f"ndlike.protected_axis.{name}")
    base = getattr(protected_module, f"{name.title()}AxisProtected")
    if name == "jax":
        functions = import_module("ndlike.jax_functions")
        prefix = "jax_"
    else:
        functions = arrays
        prefix = ""
    return {
        "name": name,
        "arrays": arrays,
        "base": base,
        "zeros_like": getattr(functions, f"{prefix}zeros_like"),
        "ones_like": getattr(functions, f"{prefix}ones_like"),
        "full_like": getattr(functions, f"{prefix}full_like"),
    }


def _call_like(backend, kind, obj, **kwargs: object):
    if kind == "full_like":
        fill_value = kwargs.pop("fill_value", 7)
        return backend["full_like"](obj, fill_value, **kwargs)
    return backend[kind](obj, **kwargs)


def _expected_scalar(kind, fill_value=7):
    if kind == "zeros_like":
        return 0
    if kind == "ones_like":
        return 1
    return fill_value


def _as_numpy(value):
    return np.asarray(value)


def _dtype(arrays, name):
    return getattr(arrays, name)


# ---------------------------------------------------------------------------
# Cross-backend protected-axis behavior
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["zeros_like", "ones_like", "full_like"])
@pytest.mark.parametrize(
    ("backend", "override", "expected_batch"),
    [
        ("torch", None, (2, 3)),
        *(
            (name, override, expected_batch)
            for name in ("numpy", "jax")
            for override, expected_batch in (
                (None, (2, 3)),
                ((), ()),
                (5, (5,)),
                ((4, 1), (4, 1)),
                ([3, 2], (3, 2)),
                (np.int64(4), (4,)),
                ((np.int64(2), 3), (2, 3)),
                (0, (0,)),
                ((0,), (0,)),
                ((2, 0, 3), (2, 0, 3)),
            )
        ),
    ],
    indirect=["backend"],
)
def test_multi_field_shape_and_values(
    backend,
    kind,
    override,
    expected_batch,
) -> None:
    arrays = backend["arrays"]

    @dataclass(frozen=True)
    class Multi(backend["base"]):
        vec: Any
        mat: Any
        scalar: Any
        label: str = "meta"
        protected_axes: ClassVar[dict[str, int]] = {"vec": 1, "mat": 2, "scalar": 0}

    x = Multi(
        arrays.arange(24, dtype=_dtype(arrays, "float32")).reshape(2, 3, 4),
        arrays.arange(36, dtype=_dtype(arrays, "int32")).reshape(2, 3, 2, 3),
        arrays.ones((2, 3), dtype=_dtype(arrays, "bool")),
        label="kept",
    )
    fill = 3 if kind == "full_like" else None
    kwargs: dict[str, Any] = {} if backend["name"] == "torch" else {"shape": override}
    if fill is not None:
        kwargs["fill_value"] = fill

    result = _call_like(backend, kind, x, **kwargs)
    assert type(result) is Multi
    assert result.label == "kept"
    assert result.shape == expected_batch
    assert tuple(result.vec.shape) == (*expected_batch, 4)
    assert tuple(result.mat.shape) == (*expected_batch, 2, 3)
    assert tuple(result.scalar.shape) == expected_batch

    # dtype=None preserves each field's own dtype.
    assert result.vec.dtype == x.vec.dtype
    assert result.mat.dtype == x.mat.dtype
    assert result.scalar.dtype == x.scalar.dtype

    expected_val = _expected_scalar(kind, fill_value=3)
    np.testing.assert_array_equal(_as_numpy(result.vec), np.full((*expected_batch, 4), expected_val, dtype=np.float32))
    np.testing.assert_array_equal(_as_numpy(result.mat), np.full((*expected_batch, 2, 3), expected_val, dtype=np.int32))
    np.testing.assert_array_equal(_as_numpy(result.scalar), np.full(expected_batch, bool(expected_val), dtype=np.bool_))


@pytest.mark.parametrize("kind", ["zeros_like", "ones_like", "full_like"])
def test_explicit_dtype_including_aggregate_dtype_applies_to_every_field(backend: dict[str, Any], kind: str) -> None:
    arrays = backend["arrays"]

    @dataclass(frozen=True)
    class MixedDtype(backend["base"]):
        primary: Any
        secondary: Any
        protected_axes: ClassVar[dict[str, int]] = {"primary": 1, "secondary": 2}

    f32 = _dtype(arrays, "float32")
    i32 = _dtype(arrays, "int32")
    i16 = _dtype(arrays, "int16")
    x = MixedDtype(
        arrays.ones((2, 3), dtype=f32),
        arrays.ones((2, 4, 5), dtype=i32),
    )
    assert x.dtype == f32

    # Omitting dtype preserves each field's dtype.
    preserved = _call_like(backend, kind, x)
    assert preserved.primary.dtype == f32
    assert preserved.secondary.dtype == i32

    # Explicitly passing x.dtype (aggregate dtype) intentionally casts every field like astype.
    with_aggregate = _call_like(backend, kind, x, dtype=x.dtype)
    assert with_aggregate.primary.dtype == f32
    assert with_aggregate.secondary.dtype == f32
    assert tuple(with_aggregate.primary.shape) == (2, 3)
    assert tuple(with_aggregate.secondary.shape) == (2, 4, 5)

    # Explicitly passing another dtype casts every field.
    with_other = _call_like(backend, kind, x, dtype=i16)
    assert with_other.primary.dtype == i16
    assert with_other.secondary.dtype == i16


@pytest.mark.parametrize("kind", ["zeros_like", "ones_like", "full_like"])
@pytest.mark.parametrize(
    ("backend", "override"),
    [("torch", None), *((name, shape) for name in ("numpy", "jax") for shape in (None, (), 4, (3, 2), (0, 2)))],
    indirect=["backend"],
)
def test_nested_fields_recursive_shape_and_dtype(backend, kind, override) -> None:
    arrays = backend["arrays"]
    f32 = _dtype(arrays, "float32")
    i32 = _dtype(arrays, "int32")

    @dataclass(frozen=True)
    class Leaf(backend["base"]):
        a: Any
        b: Any
        protected_axes: ClassVar[dict[str, int]] = {"a": 1, "b": 2}

    @dataclass(frozen=True)
    class Mid(backend["base"]):
        leaf: Leaf
        tag: Any
        protected_axes: ClassVar[dict[str, int]] = {"leaf": 1, "tag": 0}

    @dataclass(frozen=True)
    class Outer(backend["base"]):
        mid: Mid
        aux: Any
        protected_axes: ClassVar[dict[str, int]] = {"mid": 1, "aux": 1}

    # Outer batch: (2,)
    # Outer.mid protected: (3,) -> Mid batch: (2, 3)
    # Mid.leaf protected: (4,) -> Leaf batch: (2, 3, 4)
    # Leaf.a protected: (5,) -> (2, 3, 4, 5)
    # Leaf.b protected: (6, 7) -> (2, 3, 4, 6, 7)
    leaf = Leaf(
        arrays.ones((2, 3, 4, 5), dtype=f32),
        arrays.ones((2, 3, 4, 6, 7), dtype=i32),
    )
    mid = Mid(leaf=leaf, tag=arrays.ones((2, 3), dtype=i32))
    outer = Outer(mid=mid, aux=arrays.ones((2, 8), dtype=f32))
    assert outer.shape == (2,)
    assert mid.shape == (2, 3)
    assert leaf.shape == (2, 3, 4)

    expected_batch = (2,) if override is None else ((override,) if isinstance(override, int) else tuple(override))
    options = {} if override is None else {"shape": override}
    result = _call_like(backend, kind, outer, **options)
    assert isinstance(result, Outer)
    assert isinstance(result.mid, Mid)
    assert isinstance(result.mid.leaf, Leaf)
    assert result.shape == expected_batch
    assert result.mid.shape == (*expected_batch, 3)
    assert result.mid.leaf.shape == (*expected_batch, 3, 4)
    assert tuple(result.mid.leaf.a.shape) == (*expected_batch, 3, 4, 5)
    assert tuple(result.mid.leaf.b.shape) == (*expected_batch, 3, 4, 6, 7)
    assert tuple(result.mid.tag.shape) == (*expected_batch, 3)
    assert tuple(result.aux.shape) == (*expected_batch, 8)
    assert result.mid.leaf.a.dtype == f32
    assert result.mid.leaf.b.dtype == i32
    assert result.mid.tag.dtype == i32
    assert result.aux.dtype == f32

    casted = _call_like(backend, kind, outer, dtype=outer.dtype, **options)
    assert casted.mid.leaf.a.dtype == f32
    assert casted.mid.leaf.b.dtype == f32
    assert casted.mid.tag.dtype == f32
    assert casted.aux.dtype == f32


@pytest.mark.parametrize("kind", ["zeros_like", "ones_like", "full_like"])
def test_zero_size_templates_and_protected_axes(backend: dict[str, Any], kind: str) -> None:
    arrays = backend["arrays"]

    @dataclass(frozen=True)
    class Single(backend["base"]):
        arr: Any
        protected_axes: ClassVar[dict[str, int]] = {"arr": 1}

    @dataclass(frozen=True)
    class Nested(backend["base"]):
        inner: Single
        protected_axes: ClassVar[dict[str, int]] = {"inner": 1}

    # Zero-size batch template.
    zero_batch = Single(arrays.ones((0, 3, 4)))
    same = _call_like(backend, kind, zero_batch)
    assert same.shape == (0, 3)
    assert tuple(same.arr.shape) == (0, 3, 4)

    # Zero-size protected trailing extent.
    zero_protected = Single(arrays.ones((2, 0)))
    zp_same = _call_like(backend, kind, zero_protected)
    assert zp_same.shape == (2,)
    assert tuple(zp_same.arr.shape) == (2, 0)

    if backend["name"] == "torch":
        return

    grown = _call_like(backend, kind, zero_batch, shape=(2, 5))
    assert grown.shape == (2, 5)
    assert tuple(grown.arr.shape) == (2, 5, 4)
    np.testing.assert_array_equal(_as_numpy(grown.arr), np.full((2, 5, 4), _expected_scalar(kind)))

    zp_resized = _call_like(backend, kind, zero_protected, shape=(4, 5))
    assert zp_resized.shape == (4, 5)
    assert tuple(zp_resized.arr.shape) == (4, 5, 0)

    # Nested zero-size outer protected extent.
    nested_zero = Nested(Single(arrays.ones((2, 0, 4))))
    nz_resized = _call_like(backend, kind, nested_zero, shape=(5,))
    assert nz_resized.shape == (5,)
    assert nz_resized.inner.shape == (5, 0)
    assert tuple(nz_resized.inner.arr.shape) == (5, 0, 4)


@pytest.mark.parametrize("kind", ["zeros_like", "ones_like", "full_like"])
def test_broadcasted_batch_materializes_to_logical_shape_while_keeping_input_compact(
    backend: dict[str, Any], kind: str
) -> None:
    arrays = backend["arrays"]

    @dataclass(frozen=True)
    class Shared(backend["base"]):
        shared: Any
        values: Any
        protected_axes: ClassVar[dict[str, int]] = {"shared": 1, "values": 1}
        broadcasted_batch: ClassVar[set[str]] = {"shared"}

    @dataclass(frozen=True)
    class AllShared(Shared):
        broadcasted_batch: ClassVar[set[str]] = {"shared", "values"}

    expected_val = _expected_scalar(kind)
    x = Shared(arrays.ones((4,)), arrays.ones((2, 3, 4)))
    same = _call_like(backend, kind, x)
    assert same.shape == (2, 3)
    assert tuple(same.shared.shape) == (2, 3, 4)
    assert tuple(same.values.shape) == (2, 3, 4)
    assert tuple(x.shared.shape) == (4,)
    np.testing.assert_array_equal(_as_numpy(same.shared), np.full((2, 3, 4), expected_val))
    np.testing.assert_array_equal(_as_numpy(same.values), np.full((2, 3, 4), expected_val))

    y = AllShared(arrays.ones((2, 1, 4)), arrays.ones((3, 4)))
    same_y = _call_like(backend, kind, y)
    assert same_y.shape == (2, 3)
    assert tuple(same_y.shared.shape) == (2, 3, 4)
    assert tuple(same_y.values.shape) == (2, 3, 4)
    assert tuple(y.shared.shape) == (2, 1, 4)
    assert tuple(y.values.shape) == (3, 4)
    np.testing.assert_array_equal(_as_numpy(same_y.shared), np.full((2, 3, 4), expected_val))

    if backend["name"] == "torch":
        return

    resized = _call_like(backend, kind, x, shape=(5, 6))
    assert resized.shape == (5, 6)
    assert tuple(resized.shared.shape) == (5, 6, 4)
    assert tuple(resized.values.shape) == (5, 6, 4)
    assert tuple(x.shared.shape) == (4,)

    resized_y = _call_like(backend, kind, y, shape=(4,))
    assert resized_y.shape == (4,)
    assert tuple(resized_y.shared.shape) == (4, 4)
    assert tuple(resized_y.values.shape) == (4, 4)


@pytest.mark.parametrize("kind", ["zeros_like", "ones_like", "full_like"])
def test_permissions_reconstruction_and_effective_layout(backend: dict[str, Any], kind: str) -> None:
    arrays = backend["arrays"]
    recorded_funcs: list[Any] = []

    @dataclass(frozen=True)
    class Tracked(backend["base"]):
        values: Any
        derived: Any = field(init=False)
        tag: str = "original"
        protected_axes: ClassVar[dict[str, int]] = {"values": 1, "derived": 1}
        permitted_functions: ClassVar[set[Any]] = set()

        def __post_init__(self) -> None:
            object.__setattr__(self, "derived", self.values + 2)

        def with_protected_values(self, values, func=None):
            assert set(values) == {"values"}
            recorded_funcs.append(func)
            return backend["base"].with_protected_values(self, values, func)

    x = Tracked(arrays.ones((2, 3)), tag="custom")
    result = _call_like(backend, kind, x)
    assert type(result) is Tracked
    assert result.tag == "custom"
    assert result.shape == (2,)
    assert tuple(result.values.shape) == (2, 3)
    assert tuple(result.derived.shape) == (2, 3)
    expected_val = _expected_scalar(kind)
    np.testing.assert_array_equal(_as_numpy(result.values), np.full((2, 3), expected_val))
    np.testing.assert_array_equal(_as_numpy(result.derived), np.full((2, 3), expected_val + 2))
    assert recorded_funcs == [backend[kind]]

    # Empty effective layout (all fields init=False): allowed without shape override, rejected with shape override.
    fixed = arrays.ones((3,))

    @dataclass(frozen=True)
    class EmptyEffective(backend["base"]):
        fixed_val: Any = field(init=False, default_factory=lambda: fixed)
        protected_axes: ClassVar[dict[str, int]] = {"fixed_val": 1}

    empty = EmptyEffective()
    rebuilt = _call_like(backend, kind, empty)
    assert type(rebuilt) is EmptyEffective
    assert rebuilt is not empty
    assert rebuilt.fixed_val is fixed

    if backend["name"] != "torch":
        with pytest.raises(ValueError, match="at least one effective protected field"):
            _call_like(backend, kind, empty, shape=(2,))
    elif kind == "full_like":
        with pytest.raises(TypeError):
            backend[kind](empty, empty)
        with pytest.raises(TypeError):
            backend[kind](input=empty, fill_value=empty)


@pytest.mark.parametrize("backend", ["numpy", "jax"], indirect=True)
@pytest.mark.parametrize("kind", ["zeros_like", "ones_like", "full_like"])
@pytest.mark.parametrize(
    ("bad_shape", "exc_type"),
    [
        (-1, ValueError),
        ((2, -1), ValueError),
        (True, TypeError),
        ((True, 2), TypeError),
        (np.bool_(True), TypeError),
        ("abc", TypeError),
        (1.5, TypeError),
        ((1.5,), TypeError),
        ((None, 2), TypeError),
    ],
)
def test_invalid_shape_override_raises(backend, kind, bad_shape, exc_type) -> None:
    arrays = backend["arrays"]

    @dataclass(frozen=True)
    class Single(backend["base"]):
        arr: Any
        protected_axes: ClassVar[dict[str, int]] = {"arr": 1}

    x = Single(arrays.ones((2, 3)))
    with pytest.raises(exc_type):
        _call_like(backend, kind, x, shape=bad_shape)


@pytest.mark.parametrize("backend", ["numpy", "jax"], indirect=True)
def test_malformed_protected_rank_rejected_on_shape_override(backend: dict[str, Any]) -> None:
    arrays = backend["arrays"]

    @dataclass(frozen=True)
    class NeedsTwo(backend["base"]):
        arr: Any
        protected_axes: ClassVar[dict[str, int]] = {"arr": 2}

    malformed = NeedsTwo(arrays.ones((3,)))
    with pytest.raises(ValueError, match="expected >= 2"):
        backend["zeros_like"](malformed, shape=(2,))


# ---------------------------------------------------------------------------
# NumPy-specific native behavior (order, subok, device)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("func", [np.zeros_like, np.ones_like, np.full_like])
def test_numpy_order_device_and_subok(func) -> None:
    @dataclass(frozen=True)
    class Single(NumpyAxisProtected["Single"]):
        arr: np.ndarray
        protected_axes: ClassVar[dict[str, int]] = {"arr": 1}

    @dataclass(frozen=True)
    class NestedSingle(NumpyAxisProtected["NestedSingle"]):
        inner: Single
        protected_axes: ClassVar[dict[str, int]] = {"inner": 1}

    @dataclass(frozen=True)
    class Pair(NumpyAxisProtected["Pair"]):
        left: np.ndarray
        right: np.ndarray
        protected_axes: ClassVar[dict[str, int]] = {"left": 1, "right": 1}

    def call(obj, **kwargs: object):
        if func is np.full_like:
            return func(obj, 9.0, **kwargs)
        return func(obj, **kwargs)

    x = Single(np.ones((2, 3, 4), dtype=np.float32))
    f_ordered = call(x, order="F", shape=(3, 2), device="cpu")
    assert isinstance(f_ordered, Single)
    assert f_ordered.arr.shape == (3, 2, 4)
    assert f_ordered.arr.flags.f_contiguous

    raw = call(x, subok=False, shape=(5,))
    assert type(raw) is np.ndarray
    assert raw.shape == (5, 4)
    assert raw.dtype == np.float32

    nested = NestedSingle(Single(np.ones((2, 3, 4), dtype=np.float64)))
    raw_nested = call(nested, subok=False, shape=(5,))
    assert type(raw_nested) is np.ndarray
    assert raw_nested.shape == (5, 3, 4)

    pair = Pair(np.ones((2, 3)), np.ones((2, 4)))
    with pytest.raises(TypeError, match="multi-field protected object"):
        call(pair, subok=False)


# ---------------------------------------------------------------------------
# JAX-specific behavior (exports, native dispatch, jit, grad, vmap, sidecars)
# ---------------------------------------------------------------------------


def test_jax_exports_and_native_passthrough(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("jax")
    import jax.numpy as jnp  # noqa: PLC0415

    from ndlike import jax_functions as jf  # noqa: PLC0415

    for name in ("jax_zeros_like", "jax_ones_like", "jax_full_like"):
        assert name in jf.__all__

    arr = jnp.arange(6, dtype=jnp.float32).reshape(2, 3)
    assert jf.jax_zeros_like(arr, shape=(4, 1)).shape == (4, 1)
    assert jf.jax_ones_like(arr, dtype=jnp.int32, shape=()).shape == ()
    assert jf.jax_ones_like(arr, dtype=jnp.int32, shape=()).dtype == jnp.int32
    filled = jf.jax_full_like(arr, 3.5, shape=(0, 2), device=arr.device, out_sharding=None)
    assert filled.shape == (0, 2)
    assert filled.dtype == jnp.float32

    # JAX 0.8.1 jnp.full_like does not accept an out_sharding keyword; verify omission when None.
    orig_full_like = jnp.full_like

    def full_like_081(a, fill_value, dtype=None, shape=None, *, device=None):
        return orig_full_like(a, fill_value, dtype=dtype, shape=shape, device=device)

    monkeypatch.setattr(jnp, "full_like", full_like_081)
    compat = jf.jax_full_like(arr, 2.0, shape=(3,))
    assert compat.shape == (3,)
    np.testing.assert_array_equal(np.asarray(compat), np.full((3,), 2.0, dtype=np.float32))


def test_jax_jit_grad_vmap_and_sidecar_rejection() -> None:
    pytest.importorskip("jax")
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    from ndlike import jax_functions as jf  # noqa: PLC0415
    from ndlike.protected_axis.jax import JaxAxisProtected  # noqa: PLC0415

    @dataclass(frozen=True)
    class Inner(JaxAxisProtected["Inner"]):
        a: jax.Array
        b: jax.Array
        protected_axes: ClassVar[dict[str, int]] = {"a": 1, "b": 2}

    @dataclass(frozen=True)
    class Outer(JaxAxisProtected["Outer"]):
        inner: Inner
        shared: jax.Array
        protected_axes: ClassVar[dict[str, int]] = {"inner": 1, "shared": 1}
        broadcasted_batch: ClassVar[set[str]] = {"shared"}

    @dataclass(frozen=True)
    class MixedSidecar(JaxAxisProtected["MixedSidecar"]):
        values: jax.Array
        sidecar: np.ndarray
        protected_axes: ClassVar[dict[str, int]] = {"values": 1, "sidecar": 0}

    x = Outer(
        Inner(jnp.ones((2, 3, 4), dtype=jnp.float32), jnp.ones((2, 3, 2, 5), dtype=jnp.int32)),
        jnp.ones((6,), dtype=jnp.float32),
    )

    @jax.jit
    def run_jit(obj: Outer, fill_val: jax.Array) -> tuple[Outer, Outer, Outer, Inner]:
        same = jf.jax_zeros_like(obj)
        resized = jf.jax_ones_like(obj, shape=(4, 2))
        empty = jf.jax_full_like(obj, fill_val, shape=(0, 3))
        branch = jax.lax.cond(fill_val > 0, lambda: obj.inner, lambda: jf.jax_zeros_like(obj.inner))
        return same, resized, empty, branch

    same, resized, empty, branch = run_jit(x, jnp.array(2.5, dtype=jnp.float32))
    assert same.shape == (2,)
    assert same.shared.shape == (2, 6)
    assert x.shared.shape == (6,)
    assert same.inner.a.dtype == jnp.float32
    assert same.inner.b.dtype == jnp.int32
    assert resized.shape == (4, 2)
    assert resized.shared.shape == (4, 2, 6)
    assert resized.inner.a.shape == (4, 2, 3, 4)
    assert resized.inner.b.shape == (4, 2, 3, 2, 5)
    assert empty.shape == (0, 3)
    assert empty.inner.a.shape == (0, 3, 3, 4)
    assert branch.a.shape == (2, 3, 4)

    # Gradient flows through dynamic fill_value under jit.
    grad_fn = jax.jit(jax.grad(lambda v: jnp.sum(jf.jax_full_like(x.inner, v, dtype=jnp.float32, shape=(3,)).a)))
    assert float(grad_fn(jnp.array(4.0, dtype=jnp.float32))) == 12.0

    # vmap over leading batch dimension.
    def _vmap_ones(item: Inner) -> Inner:
        return jf.jax_ones_like(item, shape=(5,))

    vmapped = jax.vmap(_vmap_ones)(x.inner)
    assert vmapped.shape == (2, 5)
    assert vmapped.a.shape == (2, 5, 4)
    assert vmapped.b.shape == (2, 5, 2, 5)

    # NumPy sidecar fields return NotImplemented -> TypeError, matching jax_copy.
    mixed = MixedSidecar(jnp.ones((2, 3)), np.array(["a", "b"], dtype=object))
    with pytest.raises(TypeError):
        jf.jax_zeros_like(mixed)


# ---------------------------------------------------------------------------
# Torch-specific native behavior
# ---------------------------------------------------------------------------


def test_torch_protected_native_dispatch_and_rejection() -> None:
    pytest.importorskip("torch")
    import torch  # noqa: PLC0415

    from ndlike import torch_functions as tf  # noqa: PLC0415
    from ndlike.protected_axis.torch import TorchAxisProtected  # noqa: PLC0415

    # Torch mirrors native torch.*_like directly without generic wrapper functions.
    for removed in ("torch_zeros_like", "torch_ones_like", "torch_full_like"):
        assert not hasattr(tf, removed)
        assert removed not in tf.__all__

    @dataclass(frozen=True)
    class Pair(TorchAxisProtected["Pair"]):
        first: torch.Tensor
        second: torch.Tensor
        protected_axes: ClassVar[dict[str, int]] = {"first": 1, "second": 2}

    @dataclass(frozen=True)
    class Nested(TorchAxisProtected["Nested"]):
        inner: Pair
        protected_axes: ClassVar[dict[str, int]] = {"inner": 1}

    @dataclass(frozen=True)
    class ScalarBatch(TorchAxisProtected["ScalarBatch"]):
        vec: torch.Tensor
        scalar: torch.Tensor
        protected_axes: ClassVar[dict[str, int]] = {"vec": 1, "scalar": 0}

    @dataclass(frozen=True)
    class MixedSidecar(TorchAxisProtected["MixedSidecar"]):
        tensor: torch.Tensor
        sidecar: np.ndarray
        protected_axes: ClassVar[dict[str, int]] = {"tensor": 1, "sidecar": 0}

    @dataclass(frozen=True)
    class NumpySingle(NumpyAxisProtected["NumpySingle"]):
        arr: np.ndarray
        protected_axes: ClassVar[dict[str, int]] = {"arr": 1}

    pair = Pair(torch.ones((2, 3), dtype=torch.float32), torch.ones((2, 4, 5), dtype=torch.int32))
    native_zeros: Any = torch.zeros_like
    native_ones: Any = torch.ones_like
    native_full: Any = torch.full_like

    # Native positional and keyword input / fill_value forwarding.
    for zeroed in (native_zeros(pair), native_zeros(input=pair)):
        assert isinstance(zeroed, Pair)
        assert (zeroed.first.dtype, zeroed.second.dtype) == (torch.float32, torch.int32)
        torch.testing.assert_close(zeroed.first, torch.zeros((2, 3)))
        torch.testing.assert_close(zeroed.second, torch.zeros((2, 4, 5), dtype=torch.int32))

    native_o = native_ones(input=pair, dtype=torch.float64)
    assert isinstance(native_o, Pair)
    assert (native_o.first.dtype, native_o.second.dtype) == (torch.float64, torch.float64)

    for filled in (
        native_full(pair, 6),
        native_full(pair, fill_value=6),
        native_full(input=pair, fill_value=6),
        native_full(fill_value=6, input=pair),
        native_full(pair, torch.tensor(6)),
    ):
        assert isinstance(filled, Pair)
        torch.testing.assert_close(filled.first, torch.full((2, 3), 6.0))
        torch.testing.assert_close(filled.second, torch.full((2, 4, 5), 6, dtype=torch.int32))

    # 0D logical batch with 0-axis protected fields.
    scalar_filled = native_full(ScalarBatch(torch.ones((4,), dtype=torch.float32), torch.tensor(True)), 2)
    assert scalar_filled.shape == ()
    assert (tuple(scalar_filled.vec.shape), tuple(scalar_filled.scalar.shape)) == ((4,), ())
    assert scalar_filled.scalar.dtype == torch.bool
    assert bool(scalar_filled.scalar.item()) is True

    # Nested protected suffixes forward positional and keyword input / fill_value.
    nested = Nested(Pair(torch.ones((2, 3, 4)), torch.ones((2, 3, 2, 5), dtype=torch.int64)))
    native_nested = native_zeros(nested)
    assert isinstance(native_nested, Nested)
    assert isinstance(native_nested.inner, Pair)
    assert native_nested.inner.second.dtype == torch.int64

    nested_filled = native_full(input=nested, fill_value=4, dtype=torch.float32)
    assert isinstance(nested_filled, Nested)
    assert (nested_filled.inner.first.dtype, nested_filled.inner.second.dtype) == (torch.float32, torch.float32)
    torch.testing.assert_close(nested_filled.inner.second, torch.full((2, 3, 2, 5), 4.0))

    # Non-native shape=, protected fill_value, duplicate input, non-Torch templates, and NumPy sidecars are rejected.
    for call in (
        lambda: native_zeros(pair, shape=(4,)),
        lambda: native_ones(pair, shape=(4,)),
        lambda: native_full(pair, 6, shape=(4,)),
        lambda: native_full(pair, pair),
        lambda: native_full(torch.ones((2, 3)), pair),
        lambda: Pair.__torch_function__(torch.zeros_like, (Pair,), (pair,), {"input": pair}),
        lambda: native_zeros(MixedSidecar(torch.ones((2, 3)), np.array(["a", "b"], dtype=object))),
    ):
        with pytest.raises(TypeError):
            call()

    assert Pair.__torch_function__(torch.zeros_like, (Pair,), (NumpySingle(np.ones((2, 3))),), {}) is NotImplemented


def test_torch_protected_options_device_and_memory_format() -> None:
    pytest.importorskip("torch")
    import torch  # noqa: PLC0415

    from ndlike.protected_axis.torch import TorchAxisProtected  # noqa: PLC0415

    @dataclass(frozen=True)
    class Pair(TorchAxisProtected["Pair"]):
        first: torch.Tensor
        second: torch.Tensor
        protected_axes: ClassVar[dict[str, int]] = {"first": 1, "second": 2}

    @dataclass(frozen=True)
    class ImageBatch(TorchAxisProtected["ImageBatch"]):
        img: torch.Tensor
        protected_axes: ClassVar[dict[str, int]] = {"img": 3}

    native_zeros: Any = torch.zeros_like
    native_ones: Any = torch.ones_like
    native_full: Any = torch.full_like

    # Device preservation on mixed-device fields and explicit device / layout forwarding.
    mixed_dev = Pair(
        torch.ones((2, 3), dtype=torch.float32, device="cpu"),
        torch.ones((2, 4, 5), dtype=torch.int32, device="meta"),
    )
    preserved_dev = native_zeros(mixed_dev, layout=torch.strided)
    assert (preserved_dev.first.device.type, preserved_dev.second.device.type) == ("cpu", "meta")
    assert (preserved_dev.first.layout, preserved_dev.second.layout) == (torch.strided, torch.strided)

    to_cpu = native_ones(mixed_dev, device="cpu")
    assert (to_cpu.first.device.type, to_cpu.second.device.type) == ("cpu", "cpu")
    assert (to_cpu.first.dtype, to_cpu.second.dtype) == (torch.float32, torch.int32)
    torch.testing.assert_close(to_cpu.second, torch.ones((2, 4, 5), dtype=torch.int32))

    to_meta = native_full(mixed_dev, 3, device=torch.device("meta"))
    assert (to_meta.first.device.type, to_meta.second.device.type) == ("meta", "meta")
    assert (tuple(to_meta.first.shape), tuple(to_meta.second.shape)) == ((2, 3), (2, 4, 5))

    # requires_grad on mixed int+float: fails when dtype=None, succeeds when dtype=torch.float32.
    pair = Pair(torch.ones((2, 3), dtype=torch.float32), torch.ones((2, 4, 5), dtype=torch.int32))
    with pytest.raises(RuntimeError, match="floating point"):
        native_zeros(pair, requires_grad=True)

    grad_pair = native_zeros(pair, dtype=torch.float32, requires_grad=True)
    assert (grad_pair.first.requires_grad, grad_pair.second.requires_grad) == (True, True)
    assert (grad_pair.first.is_leaf, grad_pair.second.is_leaf) == (True, True)
    assert (tuple(grad_pair.first.shape), tuple(grad_pair.second.shape)) == ((2, 3), (2, 4, 5))

    # Non-contiguous stride preservation vs contiguous_format, and channels_last forwarding.
    non_contig = Pair(torch.arange(24.0).reshape(3, 2, 4).transpose(0, 1), torch.ones((2, 3, 2, 5)))
    assert native_zeros(non_contig).first.stride() == non_contig.first.stride()
    assert native_zeros(non_contig, memory_format=torch.contiguous_format).first.is_contiguous()

    imgs = ImageBatch(torch.ones((2, 3, 4, 5)).to(memory_format=torch.channels_last))
    assert native_zeros(imgs).img.is_contiguous(memory_format=torch.channels_last)
    channels_last = native_full(imgs, 2.0, memory_format=torch.channels_last, requires_grad=True)
    assert tuple(channels_last.img.shape) == (2, 3, 4, 5)
    assert channels_last.img.is_contiguous(memory_format=torch.channels_last)
    assert (channels_last.img.is_leaf, channels_last.img.requires_grad) == (True, True)


def test_torch_protected_nested_compact_broadcasted_batch() -> None:
    pytest.importorskip("torch")
    import torch  # noqa: PLC0415

    from ndlike.protected_axis.torch import TorchAxisProtected  # noqa: PLC0415

    @dataclass(frozen=True)
    class CompactInner(TorchAxisProtected["CompactInner"]):
        shared: torch.Tensor
        values: torch.Tensor
        protected_axes: ClassVar[dict[str, int]] = {"shared": 1, "values": 2}
        broadcasted_batch: ClassVar[set[str]] = {"shared"}

    @dataclass(frozen=True)
    class CompactOuter(TorchAxisProtected["CompactOuter"]):
        inner: CompactInner
        weights: torch.Tensor
        protected_axes: ClassVar[dict[str, int]] = {"inner": 1, "weights": 0}
        broadcasted_batch: ClassVar[set[str]] = {"inner"}

    native_zeros: Any = torch.zeros_like
    native_full: Any = torch.full_like

    compact_inner = CompactInner(torch.ones((4,), dtype=torch.float32), torch.ones((3, 2, 5), dtype=torch.int32))
    compact_outer = CompactOuter(compact_inner, torch.ones((2,), dtype=torch.float64))

    compact_res = native_zeros(compact_outer, dtype=torch.float32, requires_grad=True)
    assert compact_res.shape == (2,)
    assert compact_res.inner.shape == (2, 3)
    assert tuple(compact_res.inner.shared.shape) == (2, 3, 4)
    assert tuple(compact_res.inner.values.shape) == (2, 3, 2, 5)
    assert tuple(compact_res.weights.shape) == (2,)
    assert compact_res.inner.shared.is_contiguous()
    assert (compact_res.inner.shared.is_leaf, compact_res.inner.shared.requires_grad) == (True, True)

    # Materialized storage is independent across batch entries, while the input remains compact.
    filled = native_full(input=compact_outer, fill_value=5)
    filled.inner.shared[0, 0, 0] = 99.0
    assert float(filled.inner.shared[1, 0, 0].item()) == 5.0
    assert tuple(compact_outer.inner.shared.shape) == (4,)
    assert tuple(compact_outer.inner.values.shape) == (3, 2, 5)
