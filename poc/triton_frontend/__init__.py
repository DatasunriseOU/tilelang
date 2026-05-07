"""POC: Triton -> TileLang TIR frontend (initial implementation).

This package implements the design described in
``RFC_unified_fused_kernel.md`` (sections 5 and 6). The frontend hooks
Triton at the **TTIR** layer (post-AST, pre-layout-assignment) and emits
TileLang TIR ``PrimFunc`` objects that feed the standard TileLang
transform pipeline.

Reference RFC sections:
- Section 2: pivot rationale (TileLang TIR vs TTGIR).
- Section 5.1: op-by-op map (see :mod:`op_mapping`).
- Section 5.2: layout policy -- TTGIR encodings deliberately not ingested
  (see :mod:`layout`).
- Section 5.5: conformance suite (see :mod:`conformance`).
- Section 6: cross-source extern intrinsic mechanism (future work).

Public API:
    from_triton_kernel(fn, **kwargs) -> TileLangPrimFunc
    from_ttir(ttir_module)           -> TileLangPrimFunc

Layout::

    poc/triton_frontend/
    +-- __init__.py        # this file -- public API + walker driver
    +-- ptr_analysis.py    # wrapper over vendored microsoft/triton-shared
    +-- op_mapping.py      # tt.* -> TileLang op dispatch table
    +-- layout.py          # placeholder for #blocked/#shared/#mma
    +-- pipeline.py        # ordered TileLang TIR transform passes
    +-- tests/             # pytest unit tests for the lowering surface
    +-- conformance/       # RFC section 5.5 reference kernels
    +-- vendored/          # populated by sibling agent
        +-- triton_shared/ # microsoft/triton-shared PtrAnalysis (Apache-2.0)
"""
from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from .op_mapping import OP_TABLE, WalkerCtx

__all__ = [
    "from_triton_kernel",
    "from_ttir",
    "TileLangPrimFunc",
]


# Sentinel type alias. The real return type is ``tvm.tir.PrimFunc``;
# we keep an opaque alias here so callers can type-hint without dragging
# in TVM during scaffold review.
TileLangPrimFunc = Any


# ---------------------------------------------------------------------------
# Triton TTIR acquisition
# ---------------------------------------------------------------------------


def _compile_to_ttir(
    fn: Callable[..., Any],
    *,
    grid: Optional[Tuple[int, ...]],
    constexprs: Optional[Dict[str, Any]],
    target: Optional[str],
) -> Any:
    """Drive ``triton.compiler`` far enough to obtain a TTIR ``mlir.Module``.

    Triton compiles in stages: AST -> TTIR -> TTGIR -> LLVM. We stop
    after the first stage. The exact API has evolved (see Triton 2.x vs
    3.x) so we try a couple of entry points.
    """
    import triton  # noqa: WPS433  (intentional lazy import)
    from triton.compiler import compile as triton_compile  # noqa: WPS433

    # Triton 3.x: ``triton.compiler.compile(src, options={"stage": "ttir"})``.
    # Triton 2.x: ``triton.compile(fn, ..., output="ttir")``.
    # Both spellings appear in the wild; we attempt the 3.x form first
    # and fall back to a kwargs-tolerant call.
    try:
        # Newer-style: src is an ASTSource.
        from triton.compiler.compiler import ASTSource  # noqa: WPS433
        signature = constexprs or {}
        src = ASTSource(fn=fn, signature=signature, constants=constexprs or {})
        compiled = triton_compile(src, target=target, options={"stage": "ttir"})
        return compiled.asm.get("ttir") if hasattr(compiled, "asm") else compiled
    except Exception:  # pragma: no cover -- fall through to legacy path
        try:
            return triton_compile(fn, signature=constexprs, output="ttir")
        except TypeError as exc:
            raise RuntimeError(
                "Could not stop Triton compilation at the TTIR stage; "
                "this Triton version may need a custom hook."
            ) from exc


# ---------------------------------------------------------------------------
# Naive TTIR text-form parser (MVP path -- elementwise kernels only)
#
# triton.compiler returns the TTIR as either:
#   (a) a textual MLIR string (``compiled.asm["ttir"]``), or
#   (b) an ``mlir.ir.Module`` object.
#
# For the MVP we accept both: if (b) we can use ``mlir.ir`` to walk;
# if (a) we run a tiny regex tokenizer that extracts ``tt.<op>`` lines
# in order. The full walker will replace this once the MLIR Python
# bindings are confirmed to be importable in our Triton build.
# ---------------------------------------------------------------------------


_OP_LINE = re.compile(
    r"""
    ^\s*                                  # leading ws
    (?:%[\w\d_]+(?:,\s*%[\w\d_]+)*\s*=\s*)?   # optional result list
    (?P<op>tt\.[\w_]+|async_copy|mbarrier)  # op name
    """,
    re.VERBOSE,
)


def _walk_text_ttir(ttir_text: str, ctx: WalkerCtx) -> List[str]:
    """Naive line-by-line walk over textual TTIR; returns op names visited.

    This intentionally does *not* parse operands or types -- it is the
    minimum surface needed to (a) confirm dispatch coverage in tests and
    (b) serve as a stand-in until MLIR Python bindings are wired up.
    Real lowering uses :func:`from_ttir` with a full ``mlir.ir.Module``.
    """
    visited: List[str] = []
    for line in ttir_text.splitlines():
        m = _OP_LINE.match(line)
        if not m:
            continue
        op_name = m.group("op")
        visited.append(op_name)
        if op_name not in OP_TABLE:
            raise NotImplementedError(
                f"triton_frontend: TTIR op '{op_name}' is not in OP_TABLE."
            )
    return visited


def _walk_mlir_module(module: Any, ctx: WalkerCtx) -> List[str]:
    """Walk a real ``mlir.ir.Module`` and dispatch each op via OP_TABLE."""
    visited: List[str] = []
    # Triton TTIR stores tt.func ops at the module top level.
    # We recurse into all regions and dispatch by op name.

    def _recurse(op: Any) -> None:
        op_name = getattr(op, "name", None) or getattr(op, "operation", None)
        op_name_str = str(op_name) if op_name is not None else ""
        if op_name_str in OP_TABLE:
            visited.append(op_name_str)
            OP_TABLE[op_name_str](op, ctx)
        # Recurse into regions/blocks.
        for region in getattr(op, "regions", ()) or ():
            for block in getattr(region, "blocks", ()) or ():
                for child in getattr(block, "operations", ()) or ():
                    _recurse(child)

    body = getattr(module, "body", None) or getattr(module, "operation", module)
    _recurse(body)
    return visited


# ---------------------------------------------------------------------------
# PrimFunc assembly
# ---------------------------------------------------------------------------


def _make_prim_func(ctx: WalkerCtx, name: str = "main") -> Any:
    """Wrap the walker's emitted statements into a ``tvm.tir.PrimFunc``.

    Buffers come from ``ctx.buffers`` (one per kernel argument); body is
    ``SeqStmt(ctx.stmts)``. The PrimFunc is annotated with ``"tir.noalias"``
    and ``"global_symbol"`` to match what TileLang's pipeline expects.
    """
    import tvm  # noqa: WPS433
    from tvm import tir  # noqa: WPS433

    buffer_map: Dict[Any, Any] = {}
    params: List[Any] = []
    for buf_name, buf in ctx.buffers.items():
        var = tir.Var(buf_name, "handle")
        params.append(var)
        buffer_map[var] = buf

    if not ctx.stmts:
        body = tir.Evaluate(tir.const(0, "int32"))
    elif len(ctx.stmts) == 1:
        body = ctx.stmts[0]
    else:
        body = tir.SeqStmt(ctx.stmts)

    func = tir.PrimFunc(params=params, body=body, buffer_map=buffer_map)
    func = func.with_attr("tir.noalias", True)
    func = func.with_attr("global_symbol", name)
    return func


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def from_triton_kernel(
    fn: Callable[..., Any],
    *,
    grid: Optional[Tuple[int, ...]] = None,
    constexprs: Optional[Dict[str, Any]] = None,
    target: Optional[str] = None,
    **kwargs: Any,
) -> TileLangPrimFunc:
    """Lower a ``@triton.jit`` Python kernel to a TileLang ``PrimFunc``.

    Pipeline (RFC section 5):

      1. Run Triton's frontend to obtain a TTIR module/text.
      2. Delegate to :func:`from_ttir` for the TTIR -> TileLang TIR step.

    Currently supports **elementwise-only** kernels (Tier 1 -- vector_add
    level). Ops outside :data:`op_mapping.OP_TABLE` raise
    ``NotImplementedError``; ops in the table but with stubbed emitters
    raise their own ``NotImplementedError`` with the recipe in the
    docstring.

    Parameters
    ----------
    fn:
        A Python function decorated with ``@triton.jit``.
    grid:
        Optional launch grid. If absent, lifted from kernel metadata.
    constexprs:
        Triton ``constexpr`` bindings.
    target:
        TileLang target string (e.g. ``"cuda"``, ``"hip"``, ``"metal"``).

    Returns
    -------
    TileLangPrimFunc
        A TileLang ``PrimFunc`` ready for :mod:`pipeline` lowering.
    """
    ttir_module = _compile_to_ttir(
        fn, grid=grid, constexprs=constexprs, target=target
    )
    return from_ttir(ttir_module, target=target, name=getattr(fn, "__name__", "main"))


def from_ttir(
    ttir_module: Any,
    *,
    target: Optional[str] = None,
    name: str = "main",
    **kwargs: Any,
) -> TileLangPrimFunc:
    """Lower a Triton TTIR module to a TileLang ``PrimFunc``.

    Accepts either:
      * a textual TTIR string (``str``), or
      * an ``mlir.ir.Module`` object with ``regions``/``blocks``.

    Parameters
    ----------
    ttir_module:
        TTIR text or MLIR module containing one or more ``tt.func`` ops.
    target:
        TileLang target string.
    name:
        Symbol name to assign to the resulting PrimFunc.

    Returns
    -------
    TileLangPrimFunc
        A TileLang ``PrimFunc`` ready for :mod:`pipeline` lowering.
    """
    ctx = WalkerCtx()
    if isinstance(ttir_module, str):
        _walk_text_ttir(ttir_module, ctx)
    else:
        _walk_mlir_module(ttir_module, ctx)
    return _make_prim_func(ctx, name=name)
