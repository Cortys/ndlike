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

The extraction intentionally preserves implementation code. `pyproject.toml`
records scoped Ruff/ty migration exceptions for previously private docstrings,
runtime override/negative tests, and two Torch protocol implementations. These
are follow-up typing/documentation work, not a change to runtime behavior.

## Origin and releases

Extracted from [probly](https://github.com/pwhofman/probly) using git-filter-repo,
preserving the selected files' commit histories, author attribution, and MIT license.
Probly-specific distribution and quantification tests remain in probly.

The extraction starts from probly commit
`40bee0c32befead3e68b96e2932c69ceff2173df` (the test-separation PR). Historical
paths are rewritten throughout the retained history, including the earlier
`array.py`/`array_functions.py` names. Import updates and standalone project
configuration are applied afterward; old snapshots are not standalone builds.

Releases follow flextype: update the version in `pyproject.toml`, tag that version,
and publish a GitHub release. The release workflow builds with uv and publishes
using PyPI trusted publishing. Configure a pending PyPI publisher for repository
`Cortys/ndlike`, workflow `publish-release.yml`, environment `release` before the
first release. No token is stored in the repository.
