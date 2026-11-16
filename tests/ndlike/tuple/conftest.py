"""Backend adapters for the shared tuple tests."""

from __future__ import annotations

from importlib import import_module
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture(params=["numpy", "jax", "torch"])
def backend(request):
    name = request.param
    xp = pytest.importorskip("jax.numpy" if name == "jax" else name)
    module = import_module(f"ndlike.tuple.{name}")
    cls = getattr(module, f"{name.capitalize()}BroadcastTuple")
    base = getattr(import_module(f"ndlike.protected_axis.{name}"), f"{name.capitalize()}AxisProtected")
    funcs = import_module("ndlike.jax_functions") if name == "jax" else xp
    prefix = "jax_" if name == "jax" else ""
    return SimpleNamespace(
        name=name,
        xp=xp,
        cls=cls,
        dispatch=getattr(funcs, "jax_function_dispatch", None),
        base=base,
        func=lambda op: getattr(funcs, prefix + ("eq" if name == "torch" and op == "equal" else op)),
        array=lambda x: xp.asarray(np.asarray(x)),
    )
