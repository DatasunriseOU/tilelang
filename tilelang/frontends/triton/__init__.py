"""Triton TTIR frontend for TileLang (production package).

This is the canonical, supported entry point for lowering Triton kernels
into TileLang TIR. It exposes:

* :func:`from_ttir` -- lower TTIR text or an ``mlir.ir.Module`` to a
  :class:`tvm.tir.PrimFunc`.
* :func:`from_triton_kernel` -- compile a ``@triton.jit`` Python kernel
  to TTIR and then lower with :func:`from_ttir`.
* :data:`OP_TABLE`, :class:`WalkerCtx`, :class:`LazyTileExpr` -- the
  walker building blocks, re-exported for tests / advanced callers.
* :func:`compile_ttir` -- end-to-end TTIR -> ``JITKernel`` glue used by
  :func:`tilelang.compile` when it receives a TTIR-shaped argument.

The underlying implementation currently lives in :mod:`poc.triton_frontend`
(historical location dating back to the original RFC §5 proof-of-concept).
This package is a thin re-export so callers can rely on a stable import
path under ``tilelang.frontends.triton`` and so :func:`tilelang.compile`
can dispatch on TTIR inputs through :func:`compile_ttir`. The shared
state (``OP_TABLE``, ``MLIR_WALKER_AVAILABLE``, MLIR alias bootstrap)
lives in submodules of :mod:`poc.triton_frontend`, so re-exporting names
does not duplicate it.
"""

from __future__ import annotations

from typing import Any

# Importing the underlying package executes its module-level side
# effects exactly once (registers MLIR alias, builds OP_TABLE, ...).
# Subsequent ``import poc.triton_frontend`` is cached by Python and
# returns the same module object.
from poc.triton_frontend import (  # noqa: F401
    MLIR_WALKER_AVAILABLE,
    LazyTileExpr,
    OP_TABLE,
    TileLangPrimFunc,
    WalkerCtx,
    from_triton_kernel,
    from_ttir,
)


def compile_ttir(
    ttir: Any,
    *,
    out_idx: Any | None = None,
    target: Any | None = None,
    name: str = "main",
    grid: tuple[int, ...] | None = None,
    arg_buffer_shapes: Any | None = None,
    num_warps: int | None = None,
    num_stages: int | None = None,
    execution_backend: str | None = None,
    target_host: Any | None = None,
    verbose: bool | None = None,
    pass_configs: dict | None = None,
    compile_flags: Any | None = None,
    **from_ttir_kwargs: Any,
) -> Any:
    """End-to-end: Triton TTIR -> TileLang ``PrimFunc`` -> ``JITKernel``.

    Accepts the same input shapes :func:`from_ttir` does (TTIR text or an
    ``mlir.ir.Module``) and returns the same kind of object
    :func:`tilelang.compile` returns (a ``JITKernel``). Parameters not
    consumed by :func:`from_ttir` are forwarded to :func:`tilelang.compile`.

    This is the single function :func:`tilelang.compile` calls when it
    detects a TTIR-shaped input.
    """
    prim = from_ttir(
        ttir,
        target=target,
        name=name,
        grid=grid,
        arg_buffer_shapes=arg_buffer_shapes,
        num_warps=num_warps,
        num_stages=num_stages,
        **from_ttir_kwargs,
    )
    if prim is None:
        raise RuntimeError(
            "tilelang.frontends.triton.compile_ttir: from_ttir returned None "
            "(walker context did not produce a PrimFunc); check TTIR for "
            "unsupported ops or malformed module"
        )

    # Local import to defer the heavy ``tilelang.jit`` import chain.
    from tilelang.jit import compile as _tl_compile

    return _tl_compile(
        func=prim,
        out_idx=out_idx,
        execution_backend=execution_backend,
        target=target,
        target_host=target_host,
        verbose=verbose,
        pass_configs=pass_configs,
        compile_flags=compile_flags,
    )


def is_ttir_input(obj: Any) -> bool:
    """Return True iff ``obj`` looks like a TTIR module or text the
    Triton frontend should handle (rather than a TileLang PrimFunc that
    :func:`tilelang.compile` already accepts directly).

    Heuristics:
    * Strings starting with ``module {`` or containing ``tt.func`` are
      treated as TTIR text.
    * An ``mlir.ir.Module`` (any object whose top-level body region
      contains a ``tt.func`` op) is treated as TTIR.
    """
    if isinstance(obj, str):
        head = obj.lstrip()
        if not head:
            return False
        if head.startswith("module") and "tt.func" in obj:
            return True
        return "tt.func" in obj and ("tt.return" in obj or "tt.load" in obj or "tt.store" in obj)

    # Duck-type an mlir.ir.Module: it has a top-level ``body`` block whose
    # operations include ``tt.func``. Some bindings expose ``.operation``.
    body = getattr(obj, "body", None)
    if body is None:
        body = getattr(getattr(obj, "operation", None), "regions", None)
    try:
        from poc.triton_frontend.mlir_walker import wrap_module_for_walker

        wrapped = wrap_module_for_walker(obj)
    except Exception:
        wrapped = None
    if wrapped is None:
        return False
    candidate = getattr(wrapped, "body", None) or getattr(wrapped, "operation", wrapped)
    try:
        for region in getattr(candidate, "regions", ()) or ():
            for block in getattr(region, "blocks", ()) or ():
                for op in getattr(block, "operations", ()) or ():
                    name = getattr(op, "name", "")
                    if name == "tt.func" or (isinstance(name, str) and name.endswith(".func") and name.startswith("tt.")):
                        return True
    except Exception:
        return False
    return False


__all__ = [
    "MLIR_WALKER_AVAILABLE",
    "LazyTileExpr",
    "OP_TABLE",
    "TileLangPrimFunc",
    "WalkerCtx",
    "compile_ttir",
    "from_triton_kernel",
    "from_ttir",
    "is_ttir_input",
]
