# ndlike

A toolkit for defining custom array types on top of NumPy, PyTorch, and JAX.

Use `ndlike` when your objects contain arrays but represent more than raw numbers:
vectors, distributions, or other structured values. It provides protocols and
reusable base classes so these objects can expose familiar indexing, shape
operations, and backend function dispatch without reimplementing the plumbing.
Protected axes separate batch dimensions from each value's internal dimensions,
letting you reshape or stack collections without changing what an individual
value represents.

## Installation

Requires Python 3.12 or newer.

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
require the full numerical Array API. `ArrayLikeWithoutSize` covers the shared
interface without NumPy/JAX's integer `size` property, since Torch uses a method.

Backend-specific APIs build on these protocols:

| Module | Protocol | Implementation base | Conversion |
| --- | --- | --- | --- |
| `ndlike.array_like` | `NumpyArrayLike` | `NumpyArrayLikeImplementation` | `to_numpy_array_like` |
| `ndlike.torch_like` | `TorchLike` | `TorchLikeImplementation` | `to_torch_like` |
| `ndlike.jax_like` | `JaxLike` | `JaxLikeImplementation` | `to_jax_like` |

Use protocols to type array-like inputs, or subclass an implementation base to
reuse convenience methods while supplying your representation's metadata and
dispatch hooks. NumPy uses `__array_function__` / `__array_ufunc__`, Torch uses
`__torch_function__`, and ndlike supplies `__jax_function__` for JAX.

The conversion helpers honor custom `__array_like__`, `__torch_like__`, or
`__jax_like__` hooks before falling back to native backend conversion. This lets
custom types preserve their structure instead of always becoming raw arrays.

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
class Vectors(NumpyAxisProtected[np.ndarray]):
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
work out of the box; other classes can override `with_protected_values` to
reconstruct results.

Numerical operations are opt-in: use `permitted_functions` (for example,
`{np.mean}`) and, for NumPy ufuncs, `permitted_ufuncs` (for example,
`{np.add: ["__call__", "reduce"]}`). Supported reductions operate on batch axes,
leaving internal axes intact.

The corresponding backend classes are `TorchAxisProtected` in
`ndlike.protected_axis.torch` and `JaxAxisProtected` in
`ndlike.protected_axis.jax`.

## JAX function dispatch

JAX has no native equivalent of NumPy's `__array_function__`. For custom objects,
use wrappers from `ndlike.jax_functions`, such as `jax_reshape`, `jax_stack`, and
`jax_mean`, instead of the corresponding `jax.numpy` functions. They dispatch to
`__jax_function__` on custom types and call JAX normally for native arrays.
`jax_function_dispatch` lets you add the same dispatch behavior to your own
functions. `JaxLikeImplementation` also provides dataclass-based pytree support
for JAX transformations such as `jax.jit` and `jax.vmap`.

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
