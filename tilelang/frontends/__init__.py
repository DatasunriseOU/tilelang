"""TileLang language frontends.

Each subpackage adapts a source IR / surface (Triton TTIR, torch.fx, raw
``__device__`` + tile contract, etc.) into a TileLang ``PrimFunc`` so the
standard TileLang transform / codegen pipeline can lower it to one of the
supported targets.

The Triton frontend (:mod:`tilelang.frontends.triton`) is the canonical
entry point used by :func:`tilelang.compile` when it receives a Triton
TTIR string, an ``mlir.ir.Module`` carrying ``tt.func`` ops, or an
already-built :class:`tvm.tir.PrimFunc` produced by ``from_triton_kernel``.
"""

from __future__ import annotations

__all__ = ["triton"]

from . import triton  # noqa: F401  (registers Triton compile dispatch)
