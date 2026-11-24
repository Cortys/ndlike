<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/Cortys/ndlike/master/docs/source/_static/logo/ndlike-logo-horizontal-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/Cortys/ndlike/master/docs/source/_static/logo/ndlike-logo-horizontal-light.svg">
    <img src="https://raw.githubusercontent.com/Cortys/ndlike/master/docs/source/_static/logo/ndlike-logo-horizontal-light.svg" alt="ndlike" width="320" />
  </picture>
</div>

`ndlike` helps you build custom array-like types on top of NumPy, JAX, and
PyTorch—where each element can be a composite value (such as a vector,
interval, or probability distribution) rather than a single number:

- **Protected-axis arrays** (`NumpyAxisProtected`, `JaxAxisProtected`, `TorchAxisProtected`): treat trailing axes as each element's internal structure so indexing, reshaping, stacking, and opt-in math act on the batch dimensions.
- **Broadcast tuples** (`NumpyBroadcastTuple`, `JaxBroadcastTuple`, `TorchBroadcastTuple`): bundle heterogeneous arrays or custom array-likes under one shared broadcast shape.
- **JAX function dispatch & pytrees** (`ndlike.jax_functions`, `jax_function_dispatch`, `jax_tree_map`): bring `__array_function__`-style dispatch and opaque pytree mapping to JAX.
- **Array protocols & conversions** (`WeakArrayLike`, `ArrayLike`, `NumpyLike`, `JaxLike`, `TorchLike`, `to_*_like`): structural protocols, reusable implementation bases, and structure-preserving backend conversions.

## Install

Requires Python 3.13+.

```sh
pip install ndlike
pip install 'ndlike[torch]'
pip install 'ndlike[jax]'
# or with uv
uv add ndlike
```

NumPy and `flextype` are core dependencies. PyTorch and JAX are optional and are
never imported unless you explicitly import their backend modules.

## Protected-Axis Arrays

When you store a batch of six 3D vectors in a `(6, 3)` array, its logical shape
is `(6,)`, not `(6, 3)`. Subclassing `NumpyAxisProtected` (or its JAX/Torch
counterpart) lets you declare trailing *protected axes* so array operations act
only on the batch dimensions while preserving every element's internal shape:

```python
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from ndlike.protected_axis.numpy import NumpyAxisProtected


@dataclass
class Vectors(NumpyAxisProtected["Vectors"]):
    array: np.ndarray
    protected_axes: ClassVar[dict[str, int]] = {"array": 1}


vectors = Vectors(np.ones((6, 3)))
assert vectors.shape == (6,)
assert vectors.protected_shape == (3,)
assert vectors[0].array.shape == (3,)

grid = vectors.reshape(2, 3)
assert grid.shape == (2, 3)
assert grid.array.shape == (2, 3, 3)

stacked = np.stack([vectors, vectors])
assert isinstance(stacked, Vectors)
assert stacked.shape == (2, 6)
```

`protected_axes` maps each field name to the number of trailing axes belonging
to one element. Dataclasses work out of the box; non-dataclasses only need to
override `with_protected_values` to reconstruct results.

### Opt-in math and reductions

Numerical operations are opt-in so domain invariants cannot be broken silently.
Allow specific functions via `permitted_functions` and (for NumPy) ufuncs via
`permitted_ufuncs`. Permitted reductions automatically reduce only the batch
axes, leaving protected trailing axes intact:

```python
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from ndlike.protected_axis.numpy import NumpyAxisProtected


@dataclass(frozen=True)
class Vectors(NumpyAxisProtected["Vectors", "Vectors | float"]):
    array: np.ndarray
    protected_axes: ClassVar[dict[str, int]] = {"array": 1}
    permitted_functions: ClassVar[set[Callable[..., Any]]] = {np.mean, np.sum}
    permitted_ufuncs: ClassVar[dict[np.ufunc, list[str]]] = {
        np.add: ["__call__", "reduce"],
    }


v = Vectors(np.arange(24.0).reshape(2, 3, 4))

shifted = v + 1.0
assert isinstance(shifted, Vectors) and shifted.shape == (2, 3)

avg = np.mean(v)                      # reduces batch axes (0, 1) only
assert isinstance(avg, Vectors)
assert avg.shape == ()
assert avg.protected_shape == (4,)
```

### Multiple fields, broadcasting, and inheritance

A type can protect multiple fields with different internal shapes while sharing
one batch shape. Marking a field in `broadcasted_batch` lets it store a compact,
broadcastable batch shape (for instance, one shared scalar weight across every
vector) that stays compact across basic indexing and expands only when needed:

```python
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

from ndlike.protected_axis.numpy import NumpyAxisProtected


@dataclass(frozen=True)
class Vectors(NumpyAxisProtected):
    vectors: np.ndarray
    protected_axes: ClassVar[dict[str, int]] = {"vectors": 1}


@dataclass(frozen=True)
class WeightedVectors(Vectors):
    weights: np.ndarray
    protected_axes: ClassVar[dict[str, int]] = {"weights": 0}
    broadcasted_batch: ClassVar[set[str]] = {"weights"}


@dataclass(frozen=True)
class UniformVectors(WeightedVectors):
    weights: np.ndarray = field(init=False, default_factory=lambda: np.array(1.0))


wv = WeightedVectors(vectors=np.ones((2, 3, 4)), weights=np.array(0.5))
assert wv.shape == (2, 3)
assert wv[0].weights.shape == ()      # compact broadcast preserved

uv = UniformVectors(vectors=np.ones((2, 3, 4)))
assert uv.reshape(6).shape == (6,)    # init=False field rebuilt on construction
```

- **Layout inheritance:** `protected_axes` and `broadcasted_batch` accumulate
  left-to-right across base classes (so `WeightedVectors.protected_axes` is
  `{"vectors": 1, "weights": 0}`). Inherited fields and broadcast markers cannot
  be removed by omitting them; redeclaring a field locally overrides its axis
  count (and resolves conflicts across multiple bases).
- **All-broadcasted layouts:** Unmarked fields share the object's batch shape;
  if *every* field is in `broadcasted_batch`, their broadcasted batch shapes
  determine the logical shape (for example, `(2, 1)` and `(3,)` give `(2, 3)`).
- **Derived `init=False` fields:** Redeclaring an inherited protected field with
  `field(init=False)` keeps it accessible as an attribute while excluding it
  from operational transforms—operations transform only independent fields and
  reconstruct the subclass through normal dataclass initialization.

### Batch rearrangements and validation

When indexing, reshaping, transposing, broadcasting, or copying a plain
protected-axis dataclass, ndlike builds a new instance from its declared fields.
It stores the rearranged protected values and carries over non-protected fields by
reference, including `init=False` metadata. This works with frozen dataclasses and
`@dataclass(slots=True)` and skips `__init__` and `__post_init__`.

Set `revalidate_on_batch_change: ClassVar[bool] = True` if your constructor
checks batch-wide invariants or your class relies on hand-written slots, a custom
`__dict__`, or descriptors for field storage. This routes batch rearrangements
through `with_protected_values`. Extra attributes in an ordinary instance dictionary
already trigger that fallback. Operations that can introduce values or combine
objects, such as arithmetic and stacking, use ordinary construction when rebuilding.

### PyTorch and JAX backends

Use `TorchAxisProtected` (`ndlike.protected_axis.torch`) and `JaxAxisProtected`
(`ndlike.protected_axis.jax`) in the same way. `JaxAxisProtected` registers as a
JAX pytree automatically and supports functional `.at[index]` updates:

```python
from dataclasses import dataclass
from typing import ClassVar

import jax
import jax.numpy as jnp

from ndlike.jax_functions import jax_stack
from ndlike.protected_axis.jax import JaxAtMethod, JaxAxisProtected


@dataclass(frozen=True)
class JaxVectors(JaxAxisProtected["JaxVectors"]):
    array: jax.Array
    protected_axes: ClassVar[dict[str, int]] = {"array": 1}
    permitted_at_methods: ClassVar[set[JaxAtMethod]] = {"set", "add"}


v = JaxVectors(jnp.zeros((4, 3)))
updated = v.at[0].set(JaxVectors(jnp.ones(3)))
stacked = jax.jit(lambda a, b: jax_stack((a, b)))(v, updated)
assert stacked.shape == (2, 4)
assert stacked.protected_shape == (3,)
```

- **Indexed update permissions (JAX):** `permitted_at_methods` defaults to
  `{"set"}`. Add names such as `"add"` or `"max"` to opt into other scatter
  updates, or set it to `set()` to disable updates. `.at[index].get()` follows
  ordinary indexing.
- **Preparing assignments:** Override `_prepare_assignment_value(self, value)`
  on NumPy/Torch protected types—or `_prepare_assignment_value(self, value, *, method)`
  on JAX—to normalize or validate incoming values (for example, converting a
  value into `self`'s coordinate frame) without mutating the supplied object.
  Destination metadata is preserved.

## Broadcast Tuples

`NumpyBroadcastTuple`, `JaxBroadcastTuple`, and `TorchBroadcastTuple` (in
`ndlike.tuple.numpy`, `ndlike.tuple.jax`, and `ndlike.tuple.torch`) combine
heterogeneous arrays or custom array-likes into a single object with one logical
broadcast shape:

```python
import numpy as np

from ndlike.tuple.numpy import NumpyBroadcastTuple

parts = NumpyBroadcastTuple((np.ones((2, 1)), np.arange(3), np.array(7)))
assert parts.shape == (2, 3)
assert len(parts) == 2

first_row = parts[0]                  # basic indexing preserves compact broadcasts
assert first_row.shape == (3,)
assert first_row.members[2].shape == ()

doubled = parts * 2                   # operations forward to each member
assert doubled.shape == (2, 3)

stacked = np.stack([parts, parts])    # structural ops expand to the common shape
assert stacked.shape == (2, 2, 3)
```

- Use `.members` to access the stored tuple directly; indexing, `len`, and
  iteration operate on the logical batch.
- Members must be native arrays of the chosen backend (including NumPy scalars)
  or implement that backend's function dispatch protocol.
- Backend conversion hooks (`__array_like__`, `__jax_like__`, `__torch_like__`)
  convert members individually without expanding compact broadcasts, loading
  JAX/Torch on demand. Same-backend conversion is a no-op unless `dtype`,
  `device`, or `copy=True` is passed.
- The generic parameter types the full member tuple—e.g.
  `JaxBroadcastTuple[tuple[Game, Value]]` gives `members: tuple[Game, Value]`,
  while `JaxBroadcastTuple[tuple[Game, ...]]` allows arbitrarily many members.

## JAX Function Dispatch & Pytrees

JAX has no native equivalent of NumPy's `__array_function__`. To let custom
objects participate in JAX array operations, `ndlike.jax_functions` provides
thin wrappers (`jax_reshape`, `jax_stack`, `jax_mean`, `jax_add`, …) that
dispatch to `__jax_function__` on custom types and call `jax.numpy` normally for
native arrays.

You can give your own functions the same dispatch behavior with
`@jax_function_dispatch`:

```python
import jax
import jax.numpy as jnp

from ndlike.jax_functions import jax_function_dispatch, jax_multiply


@jax_function_dispatch("x")
def jax_double(x: jax.Array) -> jax.Array:
    return jax_multiply(x, 2)


assert jnp.array_equal(jax_double(jnp.array([1, 2])), jnp.array([2, 4]))
```

### Pytrees and `jax_tree_map`

`JaxLikeImplementation` provides dataclass-based pytree registration for
`jax.jit`, `jax.vmap`, and `jax.grad`. `JaxAxisProtected` declares its
protected fields automatically; general `JaxLikeImplementation` subclasses
declare their dynamic fields explicitly:

```python
from typing import ClassVar

from ndlike.jax_like import JaxLikeImplementation


class MyJaxArray(JaxLikeImplementation["MyJaxArray"]):
    pytree_children: ClassVar[tuple[str, ...]] = ("array",)
```

- `pytree_children` is inherited in order; extend it when a subclass adds
  dynamic fields. Selection is purely structural and never depends on field
  values or `None`.
- Default pytree traversal requires JAX-backed dynamic fields and rejects NumPy
  arrays (including nested ones) as well as array-valued static metadata,
  preventing silent conversion or freezing. Mixed JAX/NumPy representations
  remain usable eagerly, outside `jit`/`grad`/`vmap`.
- Native `jax.tree.map` traverses registered pytrees into their raw fields,
  bypassing wrapper operators and invariants. Use `jax_tree_map` when the mapped
  function should receive whole custom objects:

```python
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar

import jax
import jax.numpy as jnp

from ndlike.jax_functions import jax_multiply
from ndlike.jax_like import jax_tree_map
from ndlike.protected_axis.jax import JaxAxisProtected


@dataclass(frozen=True)
class Scaleable(JaxAxisProtected["Scaleable", "Scaleable | float"]):
    array: jax.Array
    protected_axes: ClassVar[dict[str, int]] = {"array": 1}
    permitted_functions: ClassVar[set[Callable[..., Any]]] = {jax_multiply}


x = Scaleable(jnp.ones((2, 3)))
mapped = jax_tree_map(lambda value: value * 2, {"values": x})
assert isinstance(mapped["values"], Scaleable)
```

`JaxLikeImplementation` is opaque to `jax_tree_map` by default; set
`tree_map_traverse: ClassVar[bool] = True` on a class to opt into structural
traversal instead.

## Array Protocols, Base Classes & Conversions

Unlike `numpy.typing.ArrayLike` (which only checks convertibility to an array),
`ndlike`'s protocols describe objects that *behave* like arrays—indexing, shape,
and metadata—without requiring the full numerical Array API:

- `WeakArrayLike` (`ndlike.array_like`): the minimal interface shared by NumPy,
  JAX, and PyTorch (omitting the integer `size` property, since Torch uses a
  `.size()` method).
- `ArrayLike` (`ndlike.array_like`): extends `WeakArrayLike` with `size`,
  `__array_namespace__`, and `to_device`.
- `NumpyLike`, `TorchLike`, `JaxLike`: backend-specific protocols adding
  operators and backend methods.

| Backend | Module | Protocol | Implementation base | Conversion helper | Conversion hook |
| --- | --- | --- | --- | --- | --- |
| NumPy | `ndlike.array_like` | `NumpyLike` | `NumpyLikeImplementation` | `to_numpy_like` | `__array_like__` |
| PyTorch | `ndlike.torch_like` | `TorchLike` | `TorchLikeImplementation` | `to_torch_like` | `__torch_like__` |
| JAX | `ndlike.jax_like` | `JaxLike` | `JaxLikeImplementation` | `to_jax_like` | `__jax_like__` |

Each module also exports a matching `*Convertible` protocol
(`NumpyLikeConvertible`, `TorchLikeConvertible`, `JaxLikeConvertible`).
Subclassing an `*Implementation` base gives you convenience methods and
operator mixins while you supply representation metadata and backend dispatch
(`__array_function__`/`__array_ufunc__` for NumPy, `__torch_function__` for
Torch, `__jax_function__` for JAX).

Use the `to_*_like` helpers to accept either custom array-likes or plain data
while keeping custom structure intact:

```python
import numpy as np

from ndlike.array_like import to_numpy_like
from ndlike.jax_like import to_jax_like
from ndlike.tuple.jax import JaxBroadcastTuple
from ndlike.tuple.numpy import NumpyBroadcastTuple

raw = to_numpy_like([1, 2, 3])            # falls back to np.asanyarray
assert isinstance(raw, np.ndarray)

parts = NumpyBroadcastTuple((np.ones((2, 1)), np.arange(3)))
assert to_numpy_like(parts) is parts      # structure preserved via __array_like__

jax_parts = to_jax_like(parts)            # converts members via __jax_like__
assert isinstance(jax_parts, JaxBroadcastTuple)
assert jax_parts.shape == (2, 3)
assert jax_parts.members[0].shape == (2, 1)
```

When an input satisfies the matching `*Convertible` protocol (the core array
interface *plus* the conversion hook, not the hook alone), `to_*_like` calls
its hook so custom types preserve their structure instead of collapsing into raw
arrays. Otherwise it falls back to `numpy.asanyarray`, `torch.as_tensor`, or
`jax.numpy.asarray`. NumPy uses `__array_like__` for consistency with
`__array__`.

### Generic type parameters

The array protocols, `*Implementation` bases, and `*AxisProtected` bases share
seven positional type parameters (trailing defaults may be omitted):

| Position | Parameter | Default | Contract |
| --- | --- | --- | --- |
| 1 | `OwnerT` | Broad corresponding protocol | Owner type (copies return `OwnerT`) |
| 2 | `OpValueT` | `OwnerT` | Accepted operation operands |
| 3 | `OpResultT` | `OwnerT` | Arithmetic and reduction results |
| 4 | `SetValueT` | `Never` | Assignment / `.at` values in addition to `Self` |
| 5 | `ShapeResultT` | `OwnerT` | Shape, indexing, and iteration-element results |
| 6 | `CompResultT` | Backend boolean array type (`WeakArrayLike` for neutral protocols) | Comparison results |
| 7 | `SetResultT` | `OwnerT` | Functional update (`.at`) results |

Common patterns:

```python
# Native array contracts
type NativeNumpy = NumpyLike[np.ndarray]
type NativeJax = JaxLike[jax.Array]
type NativeTorch = TorchLike[torch.Tensor]

# Custom type accepting scalars in arithmetic while preserving its own type
class Vector(JaxAxisProtected["Vector", "Vector | float"]): ...

# Also accepting floats for __setitem__ / .at updates (4th slot)
class MutableVector(JaxAxisProtected["MutableVector", "MutableVector", "MutableVector", float]): ...

# Extensible generic base forwarding its owner to concrete subclasses
class BaseVector[T: "BaseVector"](JaxLikeImplementation[T]): ...
class ConcreteVector(BaseVector["ConcreteVector"]): ...
```

- Explicit result arguments replace the default rather than unioning with
  `OwnerT`.
- In JAX, `.at[index].get()` returns `ShapeResultT`, while `.at[index].set(...)`
  and arithmetic update methods accept `Self | SetValueT` and return `SetResultT`
  (exposed via the final protocols `JaxIndexUpdateHelper[ValueT, ResultT, ShapeResultT = ResultT]`
  and `JaxIndexUpdateRef[ValueT, ResultT, ShapeResultT = ResultT]`).
- Conversion protocols (`*Convertible`) take only a single `SetValueT` parameter.

## Development

```sh
uv sync --all-extras
uv run ruff check
uv run ty check
uv run pytest
```

Pre-commit hooks are managed with `prek`:

```sh
uv run prek install
uv run prek run --all-files
```

`scripts/check_backend_naming.py` enforces backend prefixes on definitions
(`BKN001`) and restricts runtime Torch/JAX imports to backend-prefixed modules
(`BKN002`); use `# noqa: BKN001` or `# noqa: BKN002` for intentional
exceptions.

## License

This project is licensed under the MIT License.
