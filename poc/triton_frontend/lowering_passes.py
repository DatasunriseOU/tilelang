"""Python helpers for invoking MLIR conversion passes lifted from
facebookincubator/triton-shared.

This module wraps the C++ pybind shim (``_triton_frontend_cxx``) with a
``run_structured_to_memref`` entry point. The pass converts ``tts.*`` ops
(structured block-pointers introduced by ``PtrAnalysis``) into ``memref.*``
ops, allowing TileLang/TVM (which do not speak ``tts.*``) to consume the
rewritten module.

Pipeline position
-----------------
``ttir-text  -->  run_ptr_analysis (tt.* -> tts.*)  -->  run_structured_to_memref
                                                              (tts.* -> memref.*)``

After this helper runs, the resulting TTIR contains only ``tt.*`` ops on
top of ``memref.*`` storage (no leftover ``tts.*`` make_tptr/load/store/
make_gather_scatter_tptr nodes), which the existing ``op_emitters/`` walkers
can already lower to ``tvm.tir.PrimFunc`` via the standard memref load/store
emission path.
"""
from __future__ import annotations

import importlib
from typing import Any

from .ptr_analysis import (  # re-use the same lazy-import + build-dir machinery
    SHIM_MODULE_NAME,
    _load_shim,
    dialects_available,
    shim_available,
)

__all__ = [
    "run_structured_to_memref",
    "structured_to_memref_available",
]


def structured_to_memref_available() -> bool:
    """Return True when ``run_structured_to_memref`` can actually execute.

    Requires the shim to be built AND the dialect-bearing build (i.e. the shim
    was compiled against TRITON_INSTALL_DIR + the vendored TritonStructured
    dialect headers). When False, ``run_structured_to_memref`` raises
    ``NotImplementedError``.
    """
    if not shim_available():
        return False
    if not dialects_available():
        return False
    mod = importlib.import_module(SHIM_MODULE_NAME)
    return hasattr(mod, "run_structured_to_memref")


def run_structured_to_memref(ttir_text: str) -> str:
    """Run the StructuredToMemref conversion pass on ``ttir_text``.

    Parameters
    ----------
    ttir_text:
        MLIR text of a ``builtin.module`` containing ``tt.*`` and ``tts.*``
        ops -- typically the output of
        ``_triton_frontend_cxx.run_ptr_analysis(...)``.

    Returns
    -------
    str
        The rewritten MLIR text. ``tts.make_tptr`` / ``tts.load`` /
        ``tts.store`` ops are replaced with ``memref.subview`` /
        ``memref.copy`` / ``memref.load`` / ``memref.store`` plus the
        accompanying arith/scf/linalg book-keeping. ``tt.*`` ops on tensor
        values are unchanged.

    Raises
    ------
    NotImplementedError
        If the shim is not built, or was built without TritonStructured
        dialect support (``dialects_available() is False``).
    RuntimeError
        Propagated from the C++ side if parse / verify / partial conversion
        fails. The message includes the upstream MLIR diagnostic string.
    """
    if not shim_available():
        raise NotImplementedError(
            "C++ shim '_triton_frontend_cxx' is not built. "
            "Run `python -m poc.triton_frontend.build_cxx --build`."
        )
    if not dialects_available():
        raise NotImplementedError(
            "C++ shim was built without TritonStructured dialect support "
            "(stub mode). Re-configure with -DTRITON_INSTALL_DIR=<...> "
            "and rebuild."
        )
    shim: Any = _load_shim()
    fn = getattr(shim, "run_structured_to_memref", None)
    if fn is None:
        raise NotImplementedError(
            "Shim binary does not export `run_structured_to_memref`. "
            "Rebuild the shim from the current source tree."
        )
    return fn(ttir_text)
