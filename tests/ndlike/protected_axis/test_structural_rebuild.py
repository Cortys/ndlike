"""Structural reconstruction without constructor calls on synchronized batch rearrangements."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
from typing import TYPE_CHECKING, Any, ClassVar, Self

import numpy as np
import pytest

from ndlike.protected_axis.numpy import NumpyAxisProtected

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.fixture(params=["numpy", "torch", "jax"])
def backend(request: pytest.FixtureRequest):
    name = request.param
    pytest.importorskip(name)
    arrays = import_module("jax.numpy" if name == "jax" else name)
    module = import_module(f"ndlike.protected_axis.{name}")
    functions = import_module("ndlike.jax_functions") if name == "jax" else arrays
    return {
        "name": name,
        "base": getattr(module, f"{name.title()}AxisProtected"),
        "arrays": arrays,
        "functions": functions,
    }


def _fn(backend, name: str):
    return getattr(backend["functions"], f"jax_{name}" if backend["name"] == "jax" else name)


def _to_numpy(value) -> np.ndarray:
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def _arr(backend, data, *, dtype=None):
    arrays = backend["arrays"]
    if backend["name"] == "torch":
        return arrays.tensor(data, dtype=arrays.float32 if dtype is None else dtype)
    return arrays.asarray(data, dtype=np.float32 if dtype is None else dtype)


def _ints(backend, data):
    arrays = backend["arrays"]
    if backend["name"] == "torch":
        return arrays.tensor(data, dtype=arrays.int64)
    return arrays.asarray(data, dtype=arrays.int32 if backend["name"] == "jax" else np.int64)


def _empty(backend, shape: tuple[int, ...]):
    arrays = backend["arrays"]
    return arrays.empty(shape, dtype=arrays.float32 if backend["name"] == "torch" else np.float32)


def _safe_gather(backend, obj, indices, axis: int | None = 1):
    if backend["name"] == "numpy":
        return np.take_along_axis(obj, indices, axis=axis)
    if backend["name"] == "torch":
        assert axis is not None
        return backend["arrays"].gather(obj, axis, indices)
    return backend["functions"].jax_take_along_axis(obj, indices, axis=axis, mode="clip")


def _make_tracked_cls(
    base,
    *,
    post_init_log: list[str],
    structural_log: list[Callable[..., Any] | None],
    allowed_functions: set[Any] | None = None,
    allowed_ufuncs: dict[np.ufunc, list[str]] | None = None,
    validator: Callable[[Any], None] | None = None,
):
    @dataclass(frozen=True)
    class Tracked(base):
        values: Any
        protected_axes: ClassVar[dict[str, int]] = {"values": 1}
        permitted_functions: ClassVar[set[Any]] = set() if allowed_functions is None else allowed_functions
        permitted_ufuncs: ClassVar[dict[np.ufunc, list[str]]] = {} if allowed_ufuncs is None else allowed_ufuncs

        def __post_init__(self) -> None:
            post_init_log.append(type(self).__name__)
            if validator is not None:
                validator(self)

        def _with_protected_values_structural(self, values, func=None):
            structural_log.append(func)
            return super()._with_protected_values_structural(values, func)

    return Tracked


def _structural_ops(backend) -> list[tuple[str, Callable[[Any], Any]]]:
    name, funcs, arrays = backend["name"], backend["functions"], backend["arrays"]
    idx, adv = _ints(backend, [[2, 0], [1, 1]]), _ints(backend, [1, 0])
    ops: list[tuple[str, Callable[[Any], Any]]] = [
        ("getitem-int", lambda x: x[0]),
        ("getitem-slice", lambda x: x[:, :2]),
        ("getitem-tuple", lambda x: x[1, 2]),
        ("getitem-newaxis", lambda x: x[None, ..., None]),
        ("getitem-advanced", lambda x: x[adv]),
        ("reshape", lambda x: x.reshape(3, 2)),
        ("reshape-1d", lambda x: x.reshape(-1)),
        ("T", lambda x: x.T),
        ("mT", lambda x: x.mT),
        ("squeeze", lambda x: x[:, :1].squeeze(1)),
        ("broadcast_to", lambda x: _fn(backend, "broadcast_to")(x, (4, 2, 3))),
        ("gather", lambda x: _safe_gather(backend, x, idx, axis=1)),
    ]
    if name == "numpy":
        ops += [
            ("reshape-c", lambda x: np.reshape(x, (3, 2), order="C")),
            ("reshape-f", lambda x: np.reshape(x, (3, 2), order="F")),
            ("transpose", lambda x: np.transpose(x, (1, 0))),
            ("matrix_transpose", np.matrix_transpose),
            ("swapaxes", lambda x: x.swapaxes(0, 1)),
            ("moveaxis", lambda x: np.moveaxis(x, 0, 1)),
            ("expand_dims", lambda x: np.expand_dims(x, axis=0)),
            ("copy-method", lambda x: x.copy()),
            ("copy-func", lambda x: np.copy(x, subok=True)),
            ("gather-none", lambda x: np.take_along_axis(x, _ints(backend, [5, 0, 2]), axis=None)),
        ]
    elif name == "jax":
        ops += [
            ("at-get", lambda x: x.at[0].get()),
            ("reshape-f", lambda x: funcs.jax_reshape(x, (3, 2), order="F")),
            ("ravel", lambda x: x.ravel()),
            ("flatten", lambda x: x.flatten()),
            ("transpose", lambda x: funcs.jax_transpose(x, (1, 0))),
            ("matrix_transpose", funcs.jax_matrix_transpose),
            ("swapaxes", lambda x: x.swapaxes(0, 1)),
            ("moveaxis", lambda x: funcs.jax_moveaxis(x, 0, 1)),
            ("expand_dims", lambda x: funcs.jax_expand_dims(x, axis=0)),
            ("copy-method", lambda x: x.copy()),
            ("copy-func", funcs.jax_copy),
            ("gather-none", lambda x: funcs.jax_take_along_axis(x, _ints(backend, [5, 0, 99]), axis=None, mode="clip")),
        ]
    else:
        ops += [
            ("transpose", lambda x: x.transpose(0, 1)),
            ("permute", lambda x: x.permute(1, 0)),
            ("movedim", lambda x: x.movedim(0, 1)),
            ("moveaxis", lambda x: arrays.moveaxis(x, 0, 1)),
            ("unsqueeze", lambda x: x.unsqueeze(0)),
            ("clone-method", lambda x: x.clone()),
            ("clone-func", arrays.clone),
            ("gather-method", lambda x: x.gather(1, idx)),
        ]
    return ops


@pytest.mark.parametrize("slots", [False, True])
@pytest.mark.parametrize("frozen", [False, True])
def test_structural_ops_skip_init_preserve_metadata_and_unset_init_false(backend, slots: bool, frozen: bool) -> None:
    post_init_calls = 0
    getattr_calls: list[str] = []

    @dataclass(frozen=frozen, slots=slots)
    class Paired(backend["base"]):
        vectors: Any
        totals: Any
        meta: Any = field(default_factory=object)
        cached: Any = field(init=False)
        unset_lazy: Any = field(init=False)
        protected_axes: ClassVar[dict[str, int]] = {"vectors": 1, "totals": 0}

        def __post_init__(self) -> None:
            nonlocal post_init_calls
            post_init_calls += 1
            self.validate()
            if not np.allclose(_to_numpy(self.vectors).sum(axis=-1), _to_numpy(self.totals)):
                msg = "totals must match vector sums"
                raise ValueError(msg)
            object.__setattr__(self, "cached", {"tag": "computed"})

        def __getattr__(self, name: str) -> object:
            getattr_calls.append(name)
            raise AttributeError(name)

    raw = np.arange(24.0).reshape(2, 3, 4)
    meta = {"origin": "meta"}
    x = Paired(_arr(backend, raw), _arr(backend, raw.sum(axis=-1)), meta=meta)
    cached = x.cached

    for op_name, op in _structural_ops(backend):
        res = op(x)
        assert type(res) is Paired, op_name
        assert post_init_calls == 1, op_name
        assert res.meta is meta, op_name
        assert res.cached is cached, op_name
        assert getattr_calls == [], f"{op_name} touched unset init=False field"
        np.testing.assert_allclose(_to_numpy(res.vectors).sum(axis=-1), _to_numpy(res.totals))
        with pytest.raises(AttributeError):
            _ = res.unset_lazy
        getattr_calls.clear()

    new_vec = _arr(backend, raw)
    updated = x.with_protected_values({"vectors": new_vec})
    assert post_init_calls == 2
    assert updated.vectors is new_vec
    assert updated.totals is x.totals
    with pytest.raises(ValueError, match="inconsistent batch-shapes"):
        x.with_protected_values({"vectors": _arr(backend, np.ones((4, 3, 4)))})
    assert post_init_calls == 3


def test_indexing_preserves_numpy_scalar_cells(backend) -> None:
    @dataclass
    class Tagged(backend["base"]):
        values: Any
        objects: np.ndarray
        labels: np.ndarray
        protected_axes: ClassVar[dict[str, int]] = {"values": 0, "objects": 0, "labels": 0}

    objects = np.empty(2, dtype=object)
    objects[0], objects[1] = [1, 2], np.arange(3)
    x = Tagged(_arr(backend, [10, 20]), objects, np.array(["a", "b"], dtype="U12"))
    for i in range(2):
        cell = x[i]
        assert cell.objects.shape == cell.labels.shape == cell.shape == ()
        assert cell.objects.item() is objects[i]
        assert cell.labels.dtype == x.labels.dtype


def test_hook_routing_opt_out_and_validating_operations(backend) -> None:
    name, arrays = backend["name"], backend["arrays"]
    add_fn, sum_fn, where_fn = _fn(backend, "add"), _fn(backend, "sum"), _fn(backend, "where")
    stack_fn, reshape_fn = _fn(backend, "stack"), _fn(backend, "reshape")
    concat_fn = _fn(backend, "cat" if name == "torch" else "concatenate")

    inits: list[str] = []
    structural: list[Callable[..., Any] | None] = []
    tracked_cls: Any = _make_tracked_cls(
        backend["base"],
        post_init_log=inits,
        structural_log=structural,
        allowed_functions={add_fn, sum_fn, where_fn},
        allowed_ufuncs={np.add: ["__call__"]},
    )

    @dataclass(frozen=True)
    class Strict(tracked_cls):
        values: Any
        revalidate_on_batch_change: ClassVar[bool] = True

    @dataclass(frozen=True)
    class InheritedStrict(Strict):
        pass

    @dataclass(frozen=True)
    class RelaxedChild(Strict):
        revalidate_on_batch_change: ClassVar[bool] = False

    arr, idx = _arr(backend, np.ones((2, 3, 4))), _ints(backend, [[2, 0], [1, 1]])
    x = tracked_cls(arr)
    _ = x[0], x.reshape(6)
    assert structural == [None, reshape_fn]
    assert len(inits) == 1

    for cls, expected in ((Strict, 5), (InheritedStrict, 5), (RelaxedChild, 1)):
        inits.clear()
        obj = cls(arr)
        _ = obj[0], obj.reshape(6), obj.clone() if name == "torch" else obj.copy(), _safe_gather(backend, obj, idx)
        assert len(inits) == expected

    inits.clear()
    structural.clear()
    cond_data = [[True, False, True], [False, True, False]]
    cond = arrays.tensor(cond_data) if name == "torch" else arrays.asarray(cond_data)
    if name == "numpy":
        assert x.astype(np.float32, copy=False) is x
        extra_ops: list[Callable[[Any], Any]] = [lambda a: a.astype(np.float64), lambda a: np.astype(a, np.float64)]
    elif name == "jax":
        assert x.astype(arrays.float32) is x
        extra_ops = [
            lambda a: a.astype(arrays.float16),
            lambda a: _fn(backend, "astype")(a, arrays.float16),
            _fn(backend, "conj"),
            lambda a: a.at[0].set(_arr(backend, np.zeros((3, 4)))),
        ]
    else:
        assert x.to(dtype=arrays.float32) is x
        extra_ops = [lambda a: a.to(dtype=arrays.float64), lambda a: a.mH]
    assert inits == []

    for op in (
        lambda a: a.with_protected_values({"values": a.values}),
        *extra_ops,
        _fn(backend, "zeros_like"),
        _fn(backend, "ones_like"),
        lambda a: _fn(backend, "full_like")(a, 3.0),
        lambda a: a + a,
        sum_fn,
        lambda a: where_fn(cond, a, a),
        lambda a: stack_fn((a, a), 0),
        lambda a: concat_fn((a, a), 0),
    ):
        before = len(inits)
        _ = op(x)
        assert len(inits) == before + 1
    assert structural == []

    if name in ("numpy", "torch"):
        before = len(inits)
        x[0] = _arr(backend, np.zeros((3, 4)))
        assert len(inits) == before


def test_fallbacks_for_non_plain_classes(backend) -> None:
    base = backend["base"]
    arr = _arr(backend, np.arange(24.0).reshape(2, 3, 4))
    idx, reshape_fn = _ints(backend, [[2, 0], [1, 1]]), _fn(backend, "reshape")
    counts = {"plain": 0, "derived": 0, "pv": 0, "new": 0}
    wpv_contexts: list[Callable | None] = []

    class NonDataclass(base):
        protected_axes: ClassVar[dict[str, int]] = {"values": 1}

        def __init__(self, values) -> None:
            counts["plain"] += 1
            self.values = values

        def with_protected_values(self, values, func=None):
            del func
            return type(self)(values["values"])

    plain = NonDataclass(arr)
    _ = plain[0], plain.reshape(6)
    assert counts["plain"] == 3

    @dataclass(frozen=True)
    class Derived(base):
        values: Any
        doubled: Any = field(init=False)
        protected_axes: ClassVar[dict[str, int]] = {"values": 1, "doubled": 1}

        def __post_init__(self) -> None:
            counts["derived"] += 1
            object.__setattr__(self, "doubled", self.values * 2)

    d = Derived(arr)
    for res in (d[0], d.reshape(6), _safe_gather(backend, d, idx, axis=1)):
        np.testing.assert_allclose(_to_numpy(res.doubled), _to_numpy(res.values) * 2)
    assert counts["derived"] == 4

    @dataclass(frozen=True)
    class CustomWpv(base):
        values: Any
        protected_axes: ClassVar[dict[str, int]] = {"values": 1}

        def with_protected_values(self, values, func=None):
            wpv_contexts.append(func)
            return base.with_protected_values(self, values, func)

    @dataclass(frozen=True)
    class InheritedWpv(CustomWpv):
        pass

    for cls in (CustomWpv, InheritedWpv):
        wpv_contexts.clear()
        obj = cls(arr)
        _ = obj[0], obj.reshape(6)
        assert wpv_contexts == [None, reshape_fn]

    @dataclass(frozen=True)
    class CustomPv(base):
        values: Any
        protected_axes: ClassVar[dict[str, int]] = {"values": 1}

        def __post_init__(self) -> None:
            counts["pv"] += 1

        def protected_values(self, func=None, *args: object, **kwargs: object):
            return base.protected_values(self, func, *args, **kwargs)

    @dataclass(frozen=True)
    class InheritedPv(CustomPv):
        pass

    for cls in (CustomPv, InheritedPv):
        counts["pv"] = 0
        obj = cls(arr)
        _ = obj[0], obj.reshape(6)
        assert counts["pv"] == 3

    @dataclass(frozen=True)
    class CustomNew(base):
        values: Any
        protected_axes: ClassVar[dict[str, int]] = {"values": 1}

        def __new__(cls, *args: object, **kwargs: object) -> Self:
            del args, kwargs
            counts["new"] += 1
            return super().__new__(cls)

    _ = CustomNew(arr)[0]
    assert counts["new"] == 2


def test_fallbacks_for_undeclared_instance_dict_state(backend) -> None:
    base = backend["base"]
    arr = _arr(backend, np.arange(24.0).reshape(2, 3, 4))

    for slots in (False, True):
        inits = 0

        @dataclass(frozen=True, slots=slots)
        class WithUndeclared(base):
            values: Any
            protected_axes: ClassVar[dict[str, int]] = {"values": 1}

            def __post_init__(self) -> None:
                nonlocal inits
                inits += 1
                object.__setattr__(self, "_extra", self.shape)

        obj = WithUndeclared(arr)
        assert obj[0]._extra == (3,)  # noqa: SLF001
        assert obj.reshape(6)._extra == (6,)  # noqa: SLF001
        assert inits == 3

        @dataclass(frozen=True, slots=slots)
        class CleanBase(base):
            values: Any
            protected_axes: ClassVar[dict[str, int]] = {"values": 1}

        sub_inits = 0

        class UndecoratedSub(CleanBase):
            def __init__(self, values) -> None:
                nonlocal sub_inits
                sub_inits += 1
                super().__init__(values)
                object.__setattr__(self, "_tag", values.shape)

        sub = UndecoratedSub(arr)[0]
        assert type(sub) is UndecoratedSub
        assert sub._tag == (3, 4)  # noqa: SLF001
        assert sub_inits == 2


def test_custom_storage_can_opt_out(backend) -> None:
    @dataclass(frozen=True, slots=True)
    class SlottedBase(backend["base"]):
        values: Any
        protected_axes: ClassVar[dict[str, int]] = {"values": 1}

    class ExtraSlotSub(SlottedBase):
        __slots__ = ("cached_shape",)
        revalidate_on_batch_change: ClassVar[bool] = True

        def __init__(self, values) -> None:
            super().__init__(values)
            object.__setattr__(self, "cached_shape", self.shape)

        @property
        def __dict__(self) -> dict[str, Any]:
            pytest.fail("The opt-out must bypass instance dictionary inspection.")

    obj = ExtraSlotSub(_arr(backend, np.ones((2, 3, 4))))
    assert obj.cached_shape == (2, 3)
    assert obj[0].cached_shape == (3,)


def test_empty_source_to_nonempty_result_falls_back_to_ordinary_construction(backend) -> None:
    name, base = backend["name"], backend["base"]
    inits: list[str] = []
    structural: list[Callable[..., Any] | None] = []

    def require_nonempty_suffix(obj) -> None:
        if 0 not in obj.shape and obj.values.shape[-1] == 0:
            msg = "nonempty batch requires nonempty protected suffix"
            raise ValueError(msg)

    cls = _make_tracked_cls(base, post_init_log=inits, structural_log=structural, validator=require_nonempty_suffix)
    empty = cls(_empty(backend, (0, 0)))

    nonempty_ops: list[Callable[[Any], Any]] = [
        lambda a: a.reshape((1,)),
        lambda a: _safe_gather(backend, a, _ints(backend, [8]), axis=0),
    ]
    if name == "jax":
        nonempty_ops += [lambda a: a[_ints(backend, [0])], lambda a: a.at[_ints(backend, [0])].get()]
    for op in nonempty_ops:
        before = len(inits)
        with pytest.raises(ValueError, match="nonempty protected suffix"):
            op(empty)
        assert len(inits) == before + 1
        assert structural == []

    before = len(inits)
    for op in (
        lambda a: a.reshape((0, 2)),
        lambda a: a[:],
        lambda a: a.clone() if name == "torch" else a.copy(),
        lambda a: _safe_gather(backend, a, _ints(backend, []), axis=0),
    ):
        assert 0 in op(empty).shape
    assert len(inits) == before
    assert len(structural) == 4

    # Compact field preceding witness + all-broadcasted empty layouts both follow logical shape.
    compact_inits = 0

    @dataclass(frozen=True)
    class CompactFirst(base):
        scale: Any
        values: Any
        protected_axes: ClassVar[dict[str, int]] = {"scale": 0, "values": 1}
        broadcasted_batch: ClassVar[set[str]] = {"scale"}

        def __post_init__(self) -> None:
            nonlocal compact_inits
            compact_inits += 1
            require_nonempty_suffix(self)

    cf = CompactFirst(_arr(backend, 2.0), _empty(backend, (0, 0)))
    with pytest.raises(ValueError, match="nonempty protected suffix"):
        cf.reshape((1,))
    assert compact_inits == 2
    assert cf.reshape((0, 2)).shape == (0, 2)
    assert cf[:].shape == (0,)
    assert compact_inits == 2

    all_bc_inits = 0

    @dataclass(frozen=True)
    class AllBroadcast(base):
        a: Any
        values: Any
        protected_axes: ClassVar[dict[str, int]] = {"a": 1, "values": 1}
        broadcasted_batch: ClassVar[set[str]] = {"a", "values"}

        def __post_init__(self) -> None:
            nonlocal all_bc_inits
            all_bc_inits += 1
            require_nonempty_suffix(self)

    ab = AllBroadcast(_empty(backend, (1, 0)), _empty(backend, (0, 0)))
    with pytest.raises(ValueError, match="nonempty protected suffix"):
        ab.reshape((1,))
    assert all_bc_inits == 2
    assert ab[:].shape == (0,)
    assert all_bc_inits == 2


def test_compact_broadcast_and_sequence_func_context(backend) -> None:
    name, base = backend["name"], backend["base"]
    post_init_calls = 0

    @dataclass(frozen=True)
    class Shared(base):
        scale: Any
        values: Any
        protected_axes: ClassVar[dict[str, int]] = {"scale": 0, "values": 1}
        broadcasted_batch: ClassVar[set[str]] = {"scale"}

        def __post_init__(self) -> None:
            nonlocal post_init_calls
            post_init_calls += 1
            self.validate()

    x = Shared(_arr(backend, 2.5), _arr(backend, np.arange(24.0).reshape(2, 3, 4)))
    copied = x.clone() if name == "torch" else x.copy()
    assert x[0].scale.shape == x.reshape(6).scale.shape == copied.scale.shape == ()
    assert x[_ints(backend, [1, 0])].scale.shape == (2, 3)
    gathered = _safe_gather(backend, x, _ints(backend, [[2, 0], [1, 1]]), axis=1)
    assert gathered.scale.shape == (2, 2)
    np.testing.assert_allclose(_to_numpy(gathered.scale), np.full((2, 2), 2.5))
    assert post_init_calls == 1

    read_contexts: list[Callable | None] = []
    create_contexts: list[Callable | None] = []

    @dataclass(frozen=True)
    class Recorded(base):
        values: Any
        protected_axes: ClassVar[dict[str, int]] = {"values": 1}
        permitted_functions: ClassVar[set[Any]] = set()

        def protected_values(self, func=None, *args: object, **kwargs: object):
            read_contexts.append(func)
            return base.protected_values(self, func, *args, **kwargs)

        def with_protected_values(self, values, func=None):
            create_contexts.append(func)
            return base.with_protected_values(self, values, func)

    r = Recorded(_arr(backend, np.arange(6.0).reshape(2, 3)))
    stack_fn = _fn(backend, "stack")
    concat_fns = (
        [backend["arrays"].cat, backend["arrays"].concat, backend["arrays"].concatenate]
        if name == "torch"
        else [_fn(backend, "concatenate")]
    )
    assert stack_fn((r, r), 0).shape == (2, 2)
    assert create_contexts[-1] is stack_fn
    for concat_fn in concat_fns:
        assert concat_fn((r, r), 0).shape == (4,)
        assert create_contexts[-1] is concat_fn
    assert all(ctx is None for ctx in read_contexts)


def test_numpy_reshape_order_a_and_take_along_axis() -> None:
    post_init_calls = 0
    structural_calls: list[Callable[..., Any] | None] = []

    @dataclass(frozen=True)
    class MatchedPair(NumpyAxisProtected["MatchedPair"]):
        left: np.ndarray
        right: np.ndarray
        protected_axes: ClassVar[dict[str, int]] = {"left": 0, "right": 0}

        def __post_init__(self) -> None:
            nonlocal post_init_calls
            post_init_calls += 1
            if not np.array_equal(self.left, self.right):
                msg = "paired fields became desynchronized"
                raise ValueError(msg)

        def _with_protected_values_structural(self, values, func=None):
            structural_calls.append(func)
            return super()._with_protected_values_structural(values, func)

    base = np.arange(6.0).reshape(2, 3)
    mixed = MatchedPair(np.asfortranarray(base), np.ascontiguousarray(base))
    for order in ("C", "c", "F", "f"):
        reshaped = mixed.reshape((3, 2), order=order)
        assert np.array_equal(reshaped.left, reshaped.right)
    assert post_init_calls == 1
    assert len(structural_calls) == 4

    structural_calls.clear()
    for order in ("A", "a"):
        before = post_init_calls
        with pytest.raises(ValueError, match="desynchronized"):
            mixed.reshape((3, 2), order=order)
        assert post_init_calls == before + 1
    same_order = MatchedPair(np.ascontiguousarray(base), np.ascontiguousarray(base))
    before = post_init_calls
    assert np.array_equal(same_order.reshape((3, 2), order="A").left, base.reshape(3, 2))
    assert post_init_calls == before + 1
    assert structural_calls == []

    @dataclass(frozen=True, slots=True)
    class Pair(NumpyAxisProtected["Pair"]):
        left: np.ndarray
        right: np.ndarray
        protected_axes: ClassVar[dict[str, int]] = {"left": 1, "right": 0}

    @dataclass(frozen=True, slots=True)
    class Outer(NumpyAxisProtected["Outer"]):
        inner: Pair
        aux: np.ndarray
        protected_axes: ClassVar[dict[str, int]] = {"inner": 1, "aux": 1}

    x = Pair(np.arange(24.0).reshape(2, 3, 4), np.arange(6.0).reshape(2, 3))
    indices = np.array([[2, 0], [1, 1]])
    gathered: Any = np.take_along_axis(x, indices, axis=np.int64(-1))
    np.testing.assert_array_equal(gathered.left, np.take_along_axis(x.left, indices[..., None], axis=1))
    np.testing.assert_array_equal(gathered.right, np.take_along_axis(x.right, indices, axis=1))

    flat: Any = np.take_along_axis(x, [5, 0, 2], axis=None)
    np.testing.assert_array_equal(flat.left, x.left.reshape(6, 4)[[5, 0, 2]])
    np.testing.assert_array_equal(flat.right, x.right.reshape(6)[[5, 0, 2]])

    assert np.take_along_axis(Pair(np.arange(4.0), np.array(7.0)), np.array([0, 0]), axis=None).shape == (2,)
    outer = Outer(x, np.arange(10.0).reshape(2, 5))
    gathered_outer: Any = np.take_along_axis(outer, np.array([1, 0]), axis=0)
    np.testing.assert_array_equal(gathered_outer.inner.left, x.left[[1, 0]])
    np.testing.assert_array_equal(gathered_outer.aux, outer.aux[[1, 0]])

    with pytest.raises(IndexError):
        np.take_along_axis(x, np.array([[9, 0], [1, 1]]), axis=1)
    with pytest.raises(ValueError, match="same ndim"):
        np.take_along_axis(x, np.array([0, 1]), axis=1)
    with pytest.raises(TypeError):
        np.take_along_axis(x, x, axis=1)


def test_torch_gather_with_out_writes_in_place() -> None:
    torch = pytest.importorskip("torch")
    from ndlike.protected_axis.torch import TorchAxisProtected  # noqa: PLC0415

    inits: list[str] = []
    vec_cls = _make_tracked_cls(TorchAxisProtected, post_init_log=inits, structural_log=[])
    x = vec_cls(torch.arange(24.0).reshape(2, 3, 4))
    out, idx = vec_cls(torch.zeros((2, 2, 4))), torch.tensor([[2, 0], [1, 1]])
    assert torch.gather(x, 1, idx, out=out) is out
    torch.testing.assert_close(out.values, torch.gather(x.values, 1, idx.unsqueeze(-1).expand(2, 2, 4)))
    raw_out = torch.zeros((2, 2, 4))
    assert torch.gather(x, 1, idx, out=raw_out) is raw_out
    assert len(inits) == 2


def test_jax_view_and_take_along_axis_modes() -> None:
    jax = pytest.importorskip("jax")
    jnp = import_module("jax.numpy")
    from ndlike.jax_functions import jax_take_along_axis  # noqa: PLC0415
    from ndlike.protected_axis.jax import JaxAxisProtected  # noqa: PLC0415

    inits: list[str] = []
    structural: list[Callable[..., Any] | None] = []

    def require_positive(obj) -> None:
        if bool(jnp.any(jnp.isnan(obj.values))) or bool(jnp.any(obj.values <= 0)):
            msg = "values must be finite and strictly positive"
            raise ValueError(msg)

    pos_cls = _make_tracked_cls(
        JaxAxisProtected, post_init_log=inits, structural_log=structural, validator=require_positive
    )
    x = pos_cls(jnp.arange(1.0, 25.0).reshape(2, 3, 4))

    viewed = x.view(jnp.int32)
    assert isinstance(viewed, pos_cls)
    assert viewed.values.dtype == jnp.int32
    assert len(inits) == 2
    with pytest.raises(ValueError, match="strictly positive"):
        pos_cls(jnp.array([[1.0]], dtype=jnp.float32)).view(jnp.int8)

    oob_idx, valid_idx = jnp.array([[99, 0], [-5, 1]]), jnp.array([[2, 0], [1, 1]])
    before = len(inits)
    for clip_mode in ("clip", jax.lax.GatherScatterMode.CLIP):
        res = jax_take_along_axis(x, oob_idx, axis=1, mode=clip_mode)
        np.testing.assert_array_equal(
            res.values, jnp.take_along_axis(x.values, oob_idx[..., None], axis=1, mode="clip")
        )
    assert len(inits) == before
    assert len(structural) == 2

    structural.clear()
    with pytest.raises(ValueError, match="strictly positive"):
        jax_take_along_axis(x, oob_idx, axis=1)
    for mode, fill_value in ((None, None), ("fill", None), ("fill", -1.0), ("drop", None), ("promise_in_bounds", None)):
        before = len(inits)
        _ = jax_take_along_axis(x, valid_idx, axis=1, mode=mode, fill_value=fill_value)
        assert len(inits) == before + 1
    assert structural == []

    empty = pos_cls(jnp.zeros((2, 0, 4)))
    for indices, axis in ((jnp.zeros((2, 1), dtype=jnp.int32), 1), (jnp.zeros((1,), dtype=jnp.int32), None)):
        before = len(inits)
        with pytest.raises(ValueError, match="strictly positive"):
            jax_take_along_axis(empty, indices, axis=axis, mode="clip")
        assert len(inits) == before + 1
    assert structural == []

    before = len(inits)
    assert jax_take_along_axis(empty, jnp.zeros((2, 0), dtype=jnp.int32), axis=1, mode="clip").shape == (2, 0)
    assert jax_take_along_axis(empty, jnp.zeros((0,), dtype=jnp.int32), axis=None, mode="clip").shape == (0,)
    assert len(inits) == before
    assert len(structural) == 2

    with pytest.raises(TypeError):
        _ = pos_cls(jnp.empty((0, 4)))[jnp.array([0])]


def test_jax_view_respects_custom_array_field_selection() -> None:
    jnp = pytest.importorskip("jax.numpy")
    from ndlike.protected_axis.jax import JaxAxisProtected  # noqa: PLC0415

    @dataclass
    class Pair(JaxAxisProtected):
        left: Any
        right: Any
        protected_axes: ClassVar[dict[str, int]] = {"left": 1, "right": 1}

        @classmethod
        def _array_field_names(cls) -> tuple[str, ...]:
            return ("right",)

    pair = Pair(jnp.ones(2), jnp.zeros(2))
    viewed = pair.view(jnp.int32)
    assert viewed.left is pair.left
    assert viewed.right.dtype == jnp.int32

    class CustomMapper(Pair):
        def _map_array_fields(self, op):
            return type(self)(op(self.left), self.right)

    custom = CustomMapper(pair.left, pair.right).view(jnp.int32)
    assert custom.left.dtype == jnp.int32
    assert custom.right is pair.right


def test_nested_jax_broadcast_tuple_validated_outer_under_jit() -> None:
    jax = pytest.importorskip("jax")
    jnp = import_module("jax.numpy")
    from ndlike import jax_functions as jf  # noqa: PLC0415
    from ndlike.protected_axis.jax import JaxAxisProtected  # noqa: PLC0415
    from ndlike.tuple.jax import JaxBroadcastTuple  # noqa: PLC0415

    counts = {"child": 0, "outer": 0}

    @dataclass(frozen=True, slots=True)
    class Simplex(JaxAxisProtected["Simplex"]):
        probs: Any
        protected_axes: ClassVar[dict[str, int]] = {"probs": 1}

        def __post_init__(self) -> None:
            counts["child"] += 1
            if bool(jnp.any(self.probs < 0)) or not bool(jnp.allclose(self.probs.sum(axis=-1), 1.0)):
                msg = "probs must lie on the probability simplex"
                raise ValueError(msg)

    @dataclass(frozen=True, slots=True)
    class OuterModel(JaxAxisProtected["OuterModel"]):
        bundle: JaxBroadcastTuple[tuple[Simplex, Simplex]]
        weights: Any
        label: str = "model"
        cached_tag: tuple[str, ...] = field(init=False)
        protected_axes: ClassVar[dict[str, int]] = {"bundle": 0, "weights": 0}

        def __post_init__(self) -> None:
            counts["outer"] += 1
            self.validate()
            first, second = self.bundle.members
            if first.probs.shape[-1] != second.probs.shape[-1] or not bool(jnp.allclose(first.probs, second.probs)):
                msg = "bundle members must have compatible and equal probabilities"
                raise ValueError(msg)
            object.__setattr__(self, "cached_tag", (self.label, "ready"))

    row = jnp.array([[0.2, 0.8], [0.5, 0.5], [0.1, 0.9]])
    p1 = jnp.stack([row, row])
    c1, c2 = Simplex(p1), Simplex(row[None, ...])
    weights = jnp.arange(1.0, 7.0).reshape(2, 3)

    with pytest.raises(ValueError, match="compatible and equal"):
        OuterModel(bundle=JaxBroadcastTuple((c1, Simplex(jnp.array([[[0.7, 0.3]]])))), weights=weights)

    counts.update({"child": 0, "outer": 0})
    outer = OuterModel(bundle=JaxBroadcastTuple((c1, c2)), weights=weights, label="nested")
    assert counts == {"child": 0, "outer": 1}
    cached_ref = outer.cached_tag

    @jax.jit
    def run_all(model: OuterModel, scalar_idx, adv_idx, gather_idx):
        expanded = jf.jax_expand_dims(model, axis=0)
        return (
            model[0],
            model[scalar_idx],
            model[adv_idx],
            model.at[scalar_idx].get(),
            model.at[adv_idx].get(),
            model.reshape(3, 2),
            model.T,
            model.swapaxes(0, 1),
            jf.jax_moveaxis(model, 0, 1),
            expanded,
            expanded.squeeze(0),
            jf.jax_broadcast_to(model, (4, 2, 3)),
            model.copy(),
            jf.jax_take_along_axis(model, gather_idx, axis=1, mode="clip"),
        )

    traced_adv, traced_gather = jnp.array([1, 0, 1]), jnp.array([[99, 0], [-3, 2]])
    results = run_all(outer, jnp.array(1), traced_adv, traced_gather)
    assert counts == {"child": 0, "outer": 1}

    for res in results:
        assert type(res) is OuterModel
        assert res.label == "nested"
        assert res.cached_tag is cached_ref
        assert isinstance(res.bundle, JaxBroadcastTuple)
        m0, m1 = res.bundle.members
        assert isinstance(m0, Simplex)
        assert isinstance(m1, Simplex)
        np.testing.assert_allclose(m0.probs, m1.probs)

    _, idx_scalar, idx_adv, at_scalar, at_adv, reshaped, transposed, *_rest, copied, clipped = results
    assert (idx_scalar.shape, idx_adv.shape, reshaped.shape, transposed.shape, copied.shape, clipped.shape) == (
        (3,),
        (3, 3),
        (3, 2),
        (3, 2),
        (2, 3),
        (2, 2),
    )
    np.testing.assert_allclose(idx_scalar.weights, weights[1])
    np.testing.assert_allclose(at_scalar.weights, weights[1])
    np.testing.assert_allclose(idx_adv.weights, weights[traced_adv])
    np.testing.assert_allclose(at_adv.weights, weights[traced_adv])
    np.testing.assert_allclose(idx_adv.bundle.members[0].probs, p1[traced_adv])
    np.testing.assert_allclose(clipped.weights, jnp.take_along_axis(weights, traced_gather, axis=1, mode="clip"))
    np.testing.assert_allclose(
        clipped.bundle.members[0].probs, jnp.take_along_axis(p1, traced_gather[..., None], axis=1, mode="clip")
    )

    with pytest.raises(jax.errors.TracerBoolConversionError):
        jax.jit(lambda m, idx: jf.jax_take_along_axis(m, idx, axis=1))(outer, traced_gather)
