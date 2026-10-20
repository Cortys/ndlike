"""Backend-independent imports must work even when optional backends are blocked."""

from __future__ import annotations

import subprocess
import sys


def test_core_imports_do_not_load_optional_backends() -> None:
    source = """
import importlib.abc
import sys

class BlockBackends(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'jax', 'jaxlib', 'probly'}:
            raise AssertionError(f'Unexpected optional import: {fullname}')

sys.meta_path.insert(0, BlockBackends())
import ndlike
import ndlike.array_like
import ndlike.protected_axis
import ndlike.protected_axis._common_functions
import ndlike.protected_axis.numpy
import ndlike.protected_axis.numpy_functions
assert not {'torch', 'jax', 'jaxlib', 'probly'} & sys.modules.keys()
"""
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", source],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
