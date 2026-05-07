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
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple

from .op_mapping import OP_TABLE, WalkerCtx
from .ptr_analysis import PtrAnalysis, shim_available

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
    except (ImportError, AttributeError, TypeError) as primary_err:
        # Triton 2.x or a slimmer build that lacks ASTSource. Fall back to
        # the legacy positional API. We narrow to import / shape-mismatch
        # errors so genuine compile failures (RuntimeError, ValueError)
        # propagate immediately rather than being silently swallowed.
        try:
            return triton_compile(fn, signature=constexprs, output="ttir")
        except TypeError as exc:
            raise RuntimeError(
                "Could not stop Triton compilation at the TTIR stage; "
                "this Triton version may need a custom hook."
            ) from exc
        except Exception as fallback_err:  # pragma: no cover -- legacy path
            raise RuntimeError(
                f"Triton TTIR extraction failed via both modern and legacy paths: "
                f"primary={primary_err!r}, legacy={fallback_err!r}"
            ) from fallback_err


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

    def _op_name(op: Any) -> str:
        """Extract the dotted MLIR op name across binding shapes."""
        # Real mlir.ir.Operation: ``op.name`` is the dotted op name; some
        # builds expose it via ``op.operation.name``. We try both, then
        # fall back to ``str(op.operation.opview)`` and dict-shaped fakes.
        name = getattr(op, "name", None)
        if not name:
            inner = getattr(op, "operation", None)
            name = getattr(inner, "name", None) if inner is not None else None
        if not name and isinstance(op, dict):
            name = op.get("name")
        return str(name) if name else ""

    def _recurse(op: Any) -> None:
        op_name_str = _op_name(op)
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
    # ``_compile_to_ttir`` may return either an ``mlir.ir.Module`` or a
    # textual MLIR string (depending on Triton version). Force the MLIR
    # object path: re-parse the text through ``mlir.ir`` so the walker
    # sees real ops. Fall back to the explicit text-walker only when the
    # MLIR Python bindings are unavailable.
    if isinstance(ttir_module, str):
        try:
            from mlir import ir as _mlir_ir  # type: ignore
            ctx = _mlir_ir.Context()
            ctx.allow_unregistered_dialects = True
            ttir_module = _mlir_ir.Module.parse(ttir_module, ctx)
        except Exception as exc:  # pragma: no cover -- mlir bindings absent
            warnings.warn(
                "triton_frontend: mlir.ir bindings unavailable; using "
                f"text-TTIR coverage walker. cause={exc!r}",
                RuntimeWarning,
                stacklevel=2,
            )
            return from_ttir(
                ttir_module,
                target=target,
                name=getattr(fn, "__name__", "main"),
                _allow_text_ttir=True,
            )
    return from_ttir(ttir_module, target=target, name=getattr(fn, "__name__", "main"))


def from_ttir(
    ttir_module: Any,
    *,
    target: Optional[str] = None,
    name: str = "main",
    _allow_text_ttir: bool = False,
    **kwargs: Any,
) -> TileLangPrimFunc:
    """Lower a Triton TTIR module to a TileLang ``PrimFunc``.

    Expects an ``mlir.ir.Module`` object with ``regions``/``blocks``. The
    text-TTIR path is **opt-in** (``_allow_text_ttir=True``); it is a
    coverage-only walker that does not populate ``ctx.value_map`` /
    ``ctx.buffers`` and therefore cannot produce a real lowered PrimFunc
    (see ``_walk_text_ttir`` docstring).

    Parameters
    ----------
    ttir_module:
        MLIR module containing one or more ``tt.func`` ops. A textual
        TTIR string is only accepted when ``_allow_text_ttir=True``.
    target:
        TileLang target string.
    name:
        Symbol name to assign to the resulting PrimFunc.
    _allow_text_ttir:
        Internal escape hatch for unit tests that want to exercise the
        regex-based op-name walker without an MLIR module. Production
        callers should always pass an ``mlir.ir.Module``.

    Returns
    -------
    TileLangPrimFunc
        A TileLang ``PrimFunc`` ready for :mod:`pipeline` lowering.
    """
    ctx = WalkerCtx()
    if isinstance(ttir_module, str):
        if not _allow_text_ttir:
            raise TypeError(
                "from_ttir: textual TTIR is no longer the default path; "
                "pass an mlir.ir.Module (recommended) or set "
                "_allow_text_ttir=True for the coverage-only text walker."
            )
        _walk_text_ttir(ttir_module, ctx)
    else:
        # Pre-pass: run microsoft/triton-shared PtrAnalysis to rewrite
        # tt.* pointer arithmetic into ``tts.make_tptr`` ops and seed
        # ctx.value_map with the recovered ``(buffer, indices)`` tuples.
        # Skipped silently when the C++ shim is unavailable so the walker
        # falls back to the MVP scalar path (op_mapping seeds placeholder
        # buffers in that case).
        if shim_available():
            try:
                pa = PtrAnalysis(ttir_module)
                pa.rewrite()
                for state in pa.extract_states():
                    if state.source is None:
                        continue
                    # Surface the full ``PtrState`` keyed by the printed
                    # source so emitters in ``op_mapping`` can either:
                    #   * synthesize ``T.copy(global[region], frag)`` when
                    #     the state describes a multi-element tile (sizes
                    #     non-trivial), or
                    #   * fall back to the scalar BufferLoad/Store path
                    #     when only an offset is available.
                    # Stored as a tagged dict so the legacy 2-tuple
                    # ``(buf, indices)`` shape stays unambiguous.
                    ctx.value_map[state.source] = {
                        "_ptrstate": state,
                        "source": state.source,
                        "offsets": list(state.offsets),
                        "sizes": list(state.sizes),
                        "strides": list(state.strides),
                    }
            except Exception as exc:  # pragma: no cover -- shim build issues
                warnings.warn(
                    f"triton_frontend: PtrAnalysis pre-pass failed; "
                    f"falling back to MVP scalar path. cause={exc!r}",
                    RuntimeWarning,
                    stacklevel=2,
                )
        _walk_mlir_module(ttir_module, ctx)
    return _make_prim_func(ctx, name=name)
