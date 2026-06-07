"""Post-walk fragment-layout re-registration (FRAMEFIX, approach b).

The MLIR->TIR walker assembles a *flat* ``tvm.tir.PrimFunc`` (``launch_thread``
+ ``AllocBuffer`` stmts) BEFORE any TileLang ``T.Kernel`` / ``KernelLaunchFrame``
exists. The tensor-core C accumulator (``dot_c_frag``) is therefore a raw
``decl_buffer`` in ``local.fragment`` scope with NO entry in TileLang's
layout map. When LayoutInference runs:

* ``T.gemm``'s ``InferLayout`` computes the correct MMA store layout for the C
  fragment, but
* the subsequent ``T.copy(fragment -> shared)`` epilogue is a ParallelOp whose
  own SIMT (identity) fragment layout is allowed to OVERRIDE the gemm layout
  (``layout_inference.cc`` lines ~282-317: a non-strict fragment layout that
  ``ProveFragmentContains`` the existing one replaces it).

The result is a layout-blind copy: the fragment is allocated full-size
(4096/thread instead of the MMA-distributed 32/thread) and the copy reads it at
flat ``[i*512 + tid*4]`` offsets, so only the per-thread-resident slots carry
meaning -> 32/4096 correct elements per output tile.

A native ``T.alloc_fragment`` + ``T.gemm`` inside a ``T.Kernel`` frame avoids
this by emitting an ``SBlock`` whose ``annotations["layout_map"]`` STRICTLY pins
the C fragment to its ``make_mma_store_layout``. ``layout_inference.cc``'s
``VisitStmt_(SBlockNode*)`` (line ~959) seeds ``annotated_layout_map_`` from that
annotation, and step-1 strict inference (line ~451) copies it into
``strict_layout_map``, after which the copy can no longer override it.

This module reproduces exactly that annotation on our flat PrimFunc: it wraps
the body in a single ``SBlock`` carrying every ``AllocBuffer`` buffer as
``alloc_buffers`` and a ``layout_map`` annotation mapping each recorded MMA-C
fragment's data ``Var`` to its ``make_mma_store_layout`` Fragment. The
``AllocBuffer`` stmts are replaced by no-ops (the allocation now lives on the
block).

RULE #1: this is a real correctness route. It RAISES (never silently degrades)
if the fragment buffer cannot be found in the body or the layout cannot be
built. Gated by the caller to the CUDA grid-scaled MMA-C case so the Metal /
non-MMA paths (and the fla_dot_exp2 test) are untouched.
"""

from __future__ import annotations

from typing import Any, Dict, List


def _build_c_fragment_layout(tir_buffer: Any, M: int, N: int, K: int,
                             trans_A: bool, trans_B: bool) -> Any:
    """Return the ``make_mma_store_layout`` Fragment for a C accumulator.

    Mirrors ``tilelang.cuda.op.gemm.gemm_mma.GemmMMA._make_mma_emitter``: the
    block warp partition is derived from M/N and a 128-thread block. We
    reproduce the same ``TensorCoreIntrinEmitter`` parameters and call
    ``make_mma_store_layout`` -- the identical layout the gemm op's
    ``InferLayout`` would produce for C.
    """
    from tilelang.cuda.intrinsics.macro.mma_macro_generator import (
        TensorCoreIntrinEmitter,
    )

    # 128-thread block (num_warps*32 = 4*32). The warp partition that
    # GemmWarpPolicy.compute_warp_partition selects for a square M==N tile is
    # the balanced 2x2 (4 warps); derive it from M/N so non-square tiles get a
    # row/col-major partition consistent with the gemm op.
    num_warps = 4  # ctx.num_warps default; threads_per_block = 128
    # Balanced partition: prefer square, fall back to full-row/full-col.
    import math

    def _partition(m: int, n: int, warps: int):
        best = (1, warps)
        best_score = None
        for mw in range(1, warps + 1):
            if warps % mw:
                continue
            nw = warps // mw
            if m % mw or n % nw:
                continue
            # Prefer the partition whose per-warp tile is closest to square.
            wr, wc = m // mw, n // nw
            score = abs(wr - wc)
            if best_score is None or score < best_score:
                best_score = score
                best = (mw, nw)
        return best

    m_warp, n_warp = _partition(M, N, num_warps)
    warp_row_tiles = M // m_warp
    warp_col_tiles = N // n_warp

    emitter = TensorCoreIntrinEmitter(
        a_dtype=str(getattr(tir_buffer, "dtype", "float32")),
        b_dtype=str(getattr(tir_buffer, "dtype", "float32")),
        accum_dtype=str(getattr(tir_buffer, "dtype", "float32")),
        a_transposed=bool(trans_A),
        b_transposed=bool(trans_B),
        block_row_warps=m_warp,
        block_col_warps=n_warp,
        warp_row_tiles=warp_row_tiles,
        warp_col_tiles=warp_col_tiles,
        chunk=int(K),
        num_elems_per_byte=1,
    )
    return emitter.make_mma_store_layout(tir_buffer)


def register_mma_fragment_layouts(prim_func: Any, fragments: List[Dict[str, Any]],
                                  pin_c_layout: bool = True) -> Any:
    """Wrap ``prim_func``'s body in an SBlock that STRICTLY pins each recorded
    MMA-C fragment to its ``make_mma_store_layout`` (FRAMEFIX, approach b).

    Parameters
    ----------
    prim_func : tvm.tir.PrimFunc
        The flat frontend-emitted PrimFunc.
    fragments : list of dict
        ``ctx.mma_c_fragments`` -- one entry per CUDA MMA-C fragment with
        ``buffer``/``M``/``N``/``K``/``trans_A``/``trans_B``.

    Returns
    -------
    tvm.tir.PrimFunc
        A PrimFunc whose body is an ``SBlockRealize(SBlock(...))`` carrying the
        ``layout_map`` annotation. RAISES if a recorded fragment is absent from
        the body (a real lowering bug, never silently skipped).
    """
    if not fragments:
        return prim_func

    import tvm  # noqa: WPS433
    from tvm import tir, tirx  # noqa: WPS433

    # Collect every AllocBuffer buffer in the body. These become the SBlock's
    # alloc_buffers; the original AllocBuffer stmts are replaced by no-ops.
    alloc_bufs: List[Any] = []
    frag_data_to_buf: Dict[Any, Any] = {}

    def _collect(node: Any) -> None:
        if type(node).__name__ == "AllocBuffer":
            buf = node.buffer
            alloc_bufs.append(buf)
            frag_data_to_buf[buf.data] = buf

    tir.stmt_functor.post_order_visit(prim_func.body, _collect)

    # Build the layout_map annotation, keyed by the fragment's data Var. Match
    # the recorded fragment buffers to the live AllocBuffer buffers by data Var
    # (same object) or by name.
    by_name = {b.name: b for b in alloc_bufs}
    layout_map: Dict[Any, Any] = {}
    for entry in fragments:
        rec = entry["buffer"]
        live = frag_data_to_buf.get(getattr(rec, "data", None))
        if live is None:
            live = by_name.get(getattr(rec, "name", None))
        if live is None:
            raise RuntimeError(
                "frame_register.register_mma_fragment_layouts: recorded MMA-C "
                f"fragment {getattr(rec, 'name', rec)!r} is not present among "
                f"the {len(alloc_bufs)} AllocBuffer buffers of the PrimFunc "
                "body. The walker must allocate the gemm C accumulator via "
                "_alloc_tile_buffer (which appends to ctx.local_buffers and "
                "emits an AllocBuffer at body head). RULE #1: refusing to skip "
                "the layout re-registration -- a missing fragment means the "
                "tensor-core store layout would stay unregistered and the "
                "epilogue copy would materialise a layout-blind partial tile."
            )
        if not pin_c_layout:
            # BUG 2 FIX: keep the SBlock wrapping (alloc_buffers) so the flat
            # tile buffers allocate + the epilogue store materialises, but DO
            # NOT pin C -- native LayoutInference owns it deterministically now
            # that the operands are cooperatively-staged fragments. We still
            # VALIDATE every recorded fragment is live (the RAISE above), so a
            # missing fragment is never silently skipped.
            continue
        lay = _build_c_fragment_layout(
            live,
            int(entry["M"]),
            int(entry["N"]),
            int(entry["K"]),
            bool(entry.get("trans_A", False)),
            bool(entry.get("trans_B", False)),
        )
        layout_map[live.data] = lay

    # Replace AllocBuffer stmts with no-ops (the alloc now lives on the block).
    def _strip(node: Any) -> Any:
        if type(node).__name__ == "AllocBuffer":
            return tir.Evaluate(tir.const(0, "int32"))
        return None

    from tvm.tir.stmt_functor import ir_transform  # noqa: WPS433

    stripped = ir_transform(prim_func.body, None, _strip, None)

    # GLOBAL-BUFFER-LIFT FIX (RULE #1 -- correctness, not a workaround). The
    # walked body is wrapped OUTERMOST by the kernel-launch ``thread_extent``
    # AttrStmts (threadIdx.x, blockIdx.x/y/z -- emitted by ``_make_prim_func``
    # in control.py). If we host the alloc_buffers on an SBlock that wraps the
    # WHOLE body (i.e. ABOVE the launch attrs), every local/shared tile buffer
    # is declared at FUNCTION-ROOT scope. ``PlanAndUpdateBufferAllocationLocation``
    # then cannot sink it below the (non-block) thread-extent attrs, so
    # SplitHostDevice LIFTS each one as a GLOBAL kernel parameter backed by a
    # host workspace allocation. That global scratch is SHARED across every
    # grid block -> the per-thread tiles RACE: all blocks overwrite one global
    # ``carry_tile``/``tile_binop`` and the stored result is whichever block
    # won (non-deterministic; dstates MAXDIFF ~1e2, value flips run-to-run).
    #
    # Fix: place the SBlock (and its alloc_buffers) INSIDE the innermost
    # thread-launch attr, wrapping ONLY the compute body. The alloc_buffers
    # then live in DEVICE/kernel scope -> per-block ``__shared__`` and
    # per-thread ``local`` arrays, never lifted to global params, no race.
    # Peel the leading thread_extent AttrStmts, wrap the inner body, re-apply.
    launch_attrs: List[Any] = []
    inner = stripped
    while (
        type(inner).__name__ == "AttrStmt"
        and getattr(inner, "attr_key", None) == "thread_extent"
    ):
        launch_attrs.append(inner)
        inner = inner.body

    block = tirx.SBlock(
        iter_vars=[],
        reads=[],
        writes=[],
        name_hint="root",
        body=inner,
        alloc_buffers=alloc_bufs,
        annotations={"layout_map": layout_map},
    )
    realize = tirx.SBlockRealize([], tvm.tir.const(True, "bool"), block)

    # Re-wrap the launch attrs OUTSIDE the SBlock, innermost-first.
    new_body = realize
    for attr in reversed(launch_attrs):
        new_body = tir.AttrStmt(
            attr.node, attr.attr_key, attr.value, new_body
        )
    return prim_func.with_body(new_body)
