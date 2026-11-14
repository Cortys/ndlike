# ndlike

NumPy arrays, JAX arrays, and PyTorch tensors provide similar primitives for
indexing, reshaping, stacking, and numerical operations. `ndlike` provides tools
to build wrappers around these types that behave like the underlying arrays,
but whose elements represent custom composite values—such as vectors or
distributions—instead of individual numbers.

Protocols and reusable base classes handle array interfaces and backend function
dispatch. Protected axes distinguish collection dimensions from each element's
internal dimensions, so indexing, reshaping, or stacking a collection preserves
the structure of its elements.

## Installation

Requires Python 3.13 or newer.

```bash
pip install ndlike
pip install 'ndlike[torch]'
pip install 'ndlike[jax]'
```

NumPy and flextype are core dependencies. Torch and JAX are optional and are
never imported by the backend-independent modules. Import backend modules explicitly.

## Array protocols and base classes

`ndlike.array_like.ArrayLike` describes indexing, shape, and metadata rather than
just convertibility to an array (as in `numpy.typing.ArrayLike`). It does not
require the full numerical Array API. `WeakArrayLike` covers the shared
interface without NumPy/JAX's integer `size` property, since Torch uses a method.

The array protocols and implementation bases use these positional type parameters:

| Parameter | Default | Contract |
| --- | --- | --- |
| `OwnerT` | Broad corresponding array protocol | Owner, bounded to the corresponding protocol |
| `OpValueT` | `OwnerT` | Accepted operation operands |
| `OpResultT` | `OwnerT` | Arithmetic and reduction results |
| `SetValueT` | `Never` | Assignment values in addition to `Self` |
| `ShapeResultT` | `OwnerT` | Shape, indexing, and iteration-element results |
| `CompResultT` | Backend boolean array type | Comparison results |
| `SetResultT` | `OwnerT` | Functional update results |

Explicit result arguments replace the default; they are not unioned with the owner.
Copies (`copy`, `clone`, and `jax_copy`) return `OwnerT`. Indexing returns
`ShapeResultT`, and iteration yields elements of that type.
The backend-neutral comparison default is `WeakArrayLike`. JAX and Torch array
annotations do not encode boolean dtype. `WeakArrayLike` and `ArrayLike` require
indexing, shape, and metadata interfaces without numerical operators, so they also
cover boolean arrays that do not support all numerical operators. Operator requirements belong to `NumpyLike`, `JaxLike`,
and `TorchLike`.

For example, `JaxLike[MyArray, MyArray | float]` accepts its own type and floats
as operation operands while returning `MyArray` by default. Additional assignment
inputs occupy the fourth slot. `JaxLike[MyArray, MyArray, MyArray, float]` accepts
`Self | float` for assignment and `.at` updates. `__setitem__` returns `None`;
`.at[index].set(...)` and the arithmetic update methods return `SetResultT`, while
`.at[index].get()` returns `ShapeResultT`. The helper protocols expose these as
`JaxIndexUpdateHelper[ValueT, ResultT, ShapeResultT = ResultT]` and
`JaxIndexUpdateRef[ValueT, ResultT, ShapeResultT = ResultT]`.
Both helper protocols are final: custom helpers implement their interfaces structurally.

An extensible base can forward its owner with
`class Vector[T: "Vector"](JaxLikeImplementation[T]): ...`; a leaf specializes it
with `class ConcreteVector(Vector["ConcreteVector"]): ...`. Implementations must
fulfill the declared result contracts. The owner parameter does not automatically
make `self` assignable to `T` inside the generic base.

Arguments are positional; trailing defaults may be omitted. Existing single type
arguments now mean the owner, not an extra assignment value or storage type.
Conversion protocols retain their single additional-assignment parameter.
The `NumpyAxisProtected`, `JaxAxisProtected`, and `TorchAxisProtected` bases use
the same seven parameters and defaults as their corresponding `*Like` types;
the former storage parameter is removed. For example,
`class Vector(JaxAxisProtected["Vector", "Vector | float"]): ...` preserves
`Vector` results while accepting scalar operands. Native contracts can be
written as `NumpyLike[np.ndarray]`, `JaxLike[jax.Array]`, or
`TorchLike[torch.Tensor]`. Unspecialized backend protocols use broad owner and
operand defaults, so native arrays need not match that particular specialization.

Backend-specific APIs build on these protocols:

| Module | Protocol | Conversion helper | Conversion hook |
| --- | --- | --- | --- |
| `ndlike.array_like` | `NumpyLike` | `to_numpy_like` | `__array_like__` |
| `ndlike.torch_like` | `TorchLike` | `to_torch_like` | `__torch_like__` |
| `ndlike.jax_like` | `JaxLike` | `to_jax_like` | `__jax_like__` |

Each module also provides matching `Implementation` and `Convertible` classes,
such as `NumpyLikeImplementation` and `NumpyLikeConvertible`.

Use the array protocols to type inputs, or subclass an implementation base to
reuse convenience methods while supplying your representation's metadata and
operation-dispatch hooks. NumPy uses `__array_function__` / `__array_ufunc__`,
Torch uses `__torch_function__`, and ndlike supplies `__jax_function__` for JAX.

Conversion helpers call the conversion hook when an input satisfies the matching
convertible protocol: the core array interface plus the hook, not the hook alone.
This lets custom types preserve their structure instead of becoming raw arrays.
Otherwise, the helpers fall back to `numpy.asanyarray`, `torch.as_tensor`, and
`jax.numpy.asarray`, respectively. NumPy retains `__array_like__` for backward
compatibility and consistency with NumPy's `__array__`.

## Protected axes

Protected-axis base classes give structured objects array-like behavior over
their batch dimensions while preserving trailing internal dimensions. For
example, a batch of six 3D vectors should have logical shape `(6,)`, not `(6, 3)`:

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

`protected_axes` maps field names to the number of trailing axes to preserve.
Multiple fields can have different internal shapes while sharing the same batch
shape; indexing and supported shape operations update them together. Dataclasses
work out of the box; other classes must override `with_protected_values` to
reconstruct results.

Numerical operations are opt-in: use `permitted_functions` (for example,
`{np.mean}`) and, for NumPy ufuncs, `permitted_ufuncs` (for example,
`{np.add: ["__call__", "reduce"]}`). Supported reductions operate on batch axes,
leaving internal axes intact.

The corresponding backend classes are `TorchAxisProtected` in
`ndlike.protected_axis.torch` and `JaxAxisProtected` in
`ndlike.protected_axis.jax`.

For all three backends, `protected_axes` accumulates across base classes.
Subclasses only need to declare additions. JAX's `broadcasted_batch` also
accumulates; for example:

```python
import jax
from ndlike.protected_axis.jax import JaxAxisProtected


@dataclass(frozen=True)
class Vectors(JaxAxisProtected):
    vectors: jax.Array
    protected_axes: ClassVar[dict[str, int]] = {"vectors": 1}


@dataclass(frozen=True)
class WeightedVectors(Vectors):
    weights: jax.Array
    protected_axes: ClassVar[dict[str, int]] = {"weights": 0}
    broadcasted_batch: ClassVar[set[str]] = {"weights"}
```

`WeightedVectors.protected_axes` is `{"vectors": 1, "weights": 0}`. Its weights
may have a batch shape broadcastable to the vectors' batch shape. Further
subclasses inherit both declarations, including through abstract bases. Layouts
merge from left to right across bases.

Inherited fields and broadcast markers cannot be removed by omitting them or
declaring empty containers. A subclass can override an inherited axis count by
declaring that field locally in `protected_axes`. Conflicting counts from
multiple bases must also be resolved with a local declaration.
Overrides change the batch/protected-axis split, so the subclass must ensure its
fields and inherited methods remain compatible with the new layout.

### Fields owned by initialization

A dataclass subclass can redeclare an inherited protected field with
`field(init=False)`. The field remains in `protected_axes`, but
`protected_values()` and operations in NumPy, Torch, and ndlike's JAX wrappers
use only the effective layout, excluding these fields. They transform only independent fields;
`with_protected_values` receives those results and normal dataclass initialization
recreates the excluded fields through defaults, factories, or `__post_init__`.
Consequently, `x.with_protected_values(x.protected_values())` reconstructs a copy
without passing any `init=False` fields to the constructor. Excluded fields remain
accessible as attributes and through `protected_value()` when selected as primary.
For example, a specialized weighted representation can fix its weights:

```python
from dataclasses import field
import jax.numpy as jnp


@dataclass(frozen=True)
class UniformVectors(WeightedVectors):
    weights: jax.Array = field(init=False, default_factory=lambda: jnp.array(1.0))
```

The private `_effective_protected_axes` dictionary is resolved on first use from
the concrete class's completed dataclass fields and cached on that class. This
also handles frozen/slotted subclasses and overrides that restore `init=True`.
Non-dataclasses retain every declared protected field. Treat class layouts and
dataclass field definitions as fixed after first use.

### Preparing assignments

NumPy and Torch protected types can override
`_prepare_assignment_value(self, value)` to convert an incoming value to the
destination's metadata convention or raise if incompatible. The default returns
the value unchanged. Preparation runs before field extraction and protected-shape
validation; return the original value or a converted replacement without mutating
the supplied object. Destination metadata is retained. For example, a type with
coordinate-frame metadata can return `value.to_frame(self.frame)`.

JAX uses `_prepare_assignment_value(self, value, *, method)` for `.at[index]`
updates. Import `JaxAtMethod` from `ndlike.protected_axis.jax` to annotate
`permitted_at_methods: ClassVar[set[JaxAtMethod]] = {"set"}`. This enables replacement
by default; add names such as `"add"` or `"max"` to opt into other indexed updates,
or use an empty set to disable updates. These permissions are independent of
`permitted_functions` and are checked before preparation. The hook receives the
update method, and the prepared operand is applied using native fieldwise scatter
semantics. `.at[index].get()` follows ordinary indexing.

## JAX function dispatch

JAX has no native equivalent of NumPy's `__array_function__`. For custom objects,
use wrappers from `ndlike.jax_functions`, such as `jax_reshape`, `jax_stack`, and
`jax_mean`, instead of the corresponding `jax.numpy` functions. They dispatch to
`__jax_function__` on custom types and call JAX normally for native arrays.
`jax_function_dispatch` lets you add the same dispatch behavior to your own
functions. `JaxLikeImplementation` also provides dataclass-based pytree support
for JAX transformations such as `jax.jit`, `jax.vmap`, and `jax.grad`, and
arithmetic operators that dispatch through the same wrappers.

### Explicit JAX representation fields

General `JaxLikeImplementation` subclasses must declare their dynamic fields:

```python
pytree_children: ClassVar[tuple[str, ...]] = ("array",)
```

The declaration is inherited and its order is preserved. Extend it explicitly
when a subclass adds dynamic fields. Selection never depends on a field's value,
shape, or whether it currently contains `None`. `JaxAxisProtected` declares its
protected fields automatically, using `protected_axes` order.

Default pytree traversal requires JAX-backed dynamic array fields. NumPy arrays
are rejected, including numeric arrays and arrays nested in child containers;
array-valued static metadata is also unsupported. This prevents silent backend
conversion or freezing. Mixed representations support eager operations, but
cannot be traversed by `jit`, `grad`, `vmap`, or native tree utilities with this
default layout.

### Mapping arrays versus their components

Native `jax.tree.map` traverses registered representations structurally. It does
not invoke the outer wrapper's operators or preserve its semantic invariants.
Use `jax_tree_map` when the function should receive whole custom arrays:

```python
from ndlike.jax_like import jax_tree_map

# x is a JaxLikeImplementation with multiplication support.
mapped = jax_tree_map(lambda value: value * 2, {"values": x})
```

`JaxLikeImplementation` is opaque to this helper by default. A class can set
`tree_map_traverse: ClassVar[bool] = True` to request structural traversal.

## Development

```bash
uv sync --all-extras
uv run pytest
uv run prek install
uv run prek run --all-files
uv build
```

Dependency groups follow flextype: `lint` contains Ruff and ty, `test` contains
pytest, and `dev` includes both groups plus prek. Development Torch wheels use
the CPU index on Linux and Windows. Type checking requires both backend extras.

`scripts/check_backend_naming.py` enforces backend prefixes on definitions
(BKN001) and restricts runtime Torch/JAX imports to their respective
backend-prefixed modules (BKN002). Type-only imports are exempt. Intentional
exceptions use an explicit `# noqa: BKN001` or `# noqa: BKN002` on the opening line.
