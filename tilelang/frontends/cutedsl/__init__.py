"""CuTeDSL static-source frontend for TileLang.

Implements RFC §7 Phase 4.2: lower a ``@cute.kernel`` Python source string
into a TileLang ``tvm.tir.PrimFunc`` so the kernel can be fused with the
rest of the TileLang IR graph (Triton-imported, FX-imported, native
``T.Kernel`` blocks, ``tl.extern_intrinsic`` bodies, etc.) instead of being
called as an opaque ``extern_intrinsic`` body.

The lowering is **static**: it parses the source string via :mod:`ast` and
maps a small, recognised subset of CuTeDSL ops onto TileLang TIR builders.
No runtime ``cutlass.cute`` import is required (and the local Mac dev box
does not have CuTeDSL installed), which matches the RFC requirement that
the pivot IR stay portable across backends without dragging in NV-only
runtime libraries.

Supported CuTeDSL surface today (the minimum to unblock the
GEMM-fuses-with-Triton-softmax conformance test):

* ``cute.make_tensor("name", shape=(M, N), dtype="float16", scope="shared")``
  -> ``T.alloc_shared((M, N), "float16")`` (or ``T.alloc_fragment`` for
  ``scope="register"``).
* ``cute.gemm(A, B, C)``  -> ``T.gemm(A, B, C)``.
* ``cute.copy(src, dst)`` -> ``T.copy(src, dst)``.
* ``cute.fill(buf, value)`` -> ``T.fill(buf, value)``.
* ``cute.arange(start, stop)`` -> ``T.serial(start, stop)`` loop iv.

Anything else raises a :class:`CuTeDSLLoweringError` so the contract is
explicit and failures are loud, not silently dropped.
"""

from __future__ import annotations

from .lowering import (
    CuTeDSLLoweringError,
    CuTeKernelSignature,
    compile_cute_source,
    from_cute_source,
)

__all__ = [
    "CuTeDSLLoweringError",
    "CuTeKernelSignature",
    "from_cute_source",
    "compile_cute_source",
]
