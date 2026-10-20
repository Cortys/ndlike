# ndlike

A toolkit for defining custom array types on top of NumPy, PyTorch, and JAX.

## Installation

```bash
pip install ndlike
pip install 'ndlike[torch]'
pip install 'ndlike[jax]'
```

NumPy and flextype are core dependencies. Torch and JAX are optional and are
never imported by the backend-independent modules. Import backend modules explicitly.

## Protected axes

Protected-axis base classes give structured objects array-like behavior over
their batch dimensions while preserving trailing internal dimensions.

```python
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from ndlike.protected_axis.numpy import NumpyAxisProtected


@dataclass
class Vectors(NumpyAxisProtected[np.ndarray]):
    array: np.ndarray
    protected_axes: ClassVar[dict[str, int]] = {"array": 1}


vectors = Vectors(np.ones((5, 3)))
assert vectors.shape == (5,)
assert vectors[0].array.shape == (3,)
```

The corresponding backend classes are `TorchAxisProtected` in
`ndlike.protected_axis.torch` and `JaxAxisProtected` in
`ndlike.protected_axis.jax`. The package also contains array-like protocols,
implementation base classes, conversions, and JAX function override wrappers.

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
