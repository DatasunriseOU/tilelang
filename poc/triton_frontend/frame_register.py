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


def _square_warp_partition(M: int, N: int, num_warps: int,
                           k_n_per_warp: int = 8) -> tuple:
    """Reproduce ``ComputeDefaultWarpPartition`` (Square policy) from
    ``src/backend/cuda/op/gemm.cc``.

    The gemm op's default ``GemmWarpPolicy.Square`` selects ``(m_warp,
    n_warp)`` with ``m_warp*n_warp == num_warps`` whose per-warp tile aspect
    ratio best matches ``M/N``. The C-fragment ``make_mma_store_layout``
    depends on this partition; the FRAMEFIX re-registration MUST use the
    identical partition or the pinned layout conflicts with the gemm's own.

    ``k_n_per_warp`` is the N granularity per warp (the mma n-tile, 8 for the
    standard m16n8k* tensor-core path). RULE #1: RAISE on an impossible
    partition rather than silently returning a skewed one.
    """
    k_m_per_warp = 16  # kMPerWarp in gemm.cc
    if M % k_m_per_warp != 0:
        raise RuntimeError(
            "frame_register._square_warp_partition: M=%d not divisible by %d"
            % (M, k_m_per_warp)
        )
    max_m_warps = M // k_m_per_warp
    ideal_ratio = (float(M) / float(N)) if N > 0 else 1.0
    best_m, best_n = 1, 1
    best_balance = None
    for m in range(1, min(max_m_warps, num_warps) + 1):
        if num_warps % m:
            continue
        n = num_warps // m
        m_per_warp = float(M) / (m * k_m_per_warp)
        n_per_warp = float(N) / (n * k_n_per_warp)
        if m_per_warp < 1 or n_per_warp < 1:
            continue
        balance = abs(m_per_warp / n_per_warp - ideal_ratio)
        if best_balance is None or balance < best_balance:
            best_balance = balance
            best_m, best_n = m, n
    if best_m * best_n != num_warps:
        raise RuntimeError(
            "frame_register._square_warp_partition: no valid (m_warp,n_warp) "
            "for M=%d N=%d num_warps=%d (best=%d*%d)"
            % (M, N, num_warps, best_m, best_n)
        )
    return best_m, best_n


def _build_c_fragment_layout(tir_buffer: Any, M: int, N: int, K: int,
                             trans_A: bool, trans_B: bool,
                             num_warps: int = 4) -> Any:
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

    # Reproduce the gemm op's EXACT warp partition so the re-derived C-store
    # layout equals the one ``T.gemm``'s InferLayout produces (otherwise the
    # pinned layout skews and conflicts -- the bug the prior hardcoded
    # ``num_warps=4`` / nearest-square heuristic caused). This mirrors
    # ``ComputeDefaultWarpPartition`` (isSquare branch) in
    # ``src/backend/cuda/op/gemm.cc``: for the default Square policy it picks
    # the (m_warp, n_warp) whose per-warp aspect ratio best matches M/N.
    m_warp, n_warp = _square_warp_partition(int(M), int(N), int(num_warps))
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
                                  pin_c_layout: bool = True,
                                  swizzle_loads: List[Any] = None) -> Any:
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
    swizzle_loads = list(swizzle_loads or [])
    if not fragments and not swizzle_loads:
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
        # DIRECT-STORE PIN GATE: only the full-block (>=8 warp / 256-thread)
        # MMA-C accumulator needs the strict layout pin. Its direct
        # fragment->global store otherwise free-infers a flat replicated layout
        # that collides with the gemm MMA layout. Sub-block tiles (gemm clamps
        # the warp count, e.g. 64x64 -> 128 threads) adopt the gemm C layout
        # naturally via the direct-store copy and are PROVEN bit-correct
        # unpinned; pinning them with a full-block layout would conflict on
        # thread_range. RULE #1: this gate selects the correct path per tile, it
        # is NOT a degraded fallback (both paths produce bit-correct stores).
        if int(entry.get("num_warps", 4)) < 8:
            continue
        lay = _build_c_fragment_layout(
            live,
            int(entry["M"]),
            int(entry["N"]),
            int(entry["K"]),
            bool(entry.get("trans_A", False)),
            bool(entry.get("trans_B", False)),
            int(entry.get("num_warps", 4)),
        )
        layout_map[live.data] = lay

    # BANKSWIZZLE: pin each recorded raw cp.async-staged SHARED load tile (the
    # register-A source, e.g. the dstates ``dout`` tile) to TileLang's
    # ``make_swizzled_layout`` -- the SAME ldmatrix XOR swizzle T.gemm assigns a
    # B operand. LayoutInference seeds ``annotated_layout_map_`` from this
    # SBlock ``layout_map`` (the identical mechanism used for the C fragment),
    # so BOTH the cp.async store CopyNode and the register-A fragment fill read
    # the buffer through the swizzle -> per-lane LDS hits distinct banks. RULE
    # #1: RAISE if a recorded buffer is absent from the body (a real lowering
    # bug, never silently skipped) -- mirrors the fragment RAISE above.
    if swizzle_loads:
        from tilelang.layout import make_swizzled_layout  # noqa: WPS433
        for rec in swizzle_loads:
            live = frag_data_to_buf.get(getattr(rec, "data", None))
            if live is None:
                live = by_name.get(getattr(rec, "name", None))
            if live is None:
                raise RuntimeError(
                    "frame_register.register_mma_fragment_layouts: recorded "
                    "BANKSWIZZLE shared load tile %r is not present among the "
                    "%d AllocBuffer buffers of the PrimFunc body. The walker "
                    "must allocate it via _alloc_tile_buffer (scope='shared'). "
                    "RULE #1: refusing to skip the swizzle re-registration."
                    % (getattr(rec, "name", rec), len(alloc_bufs))
                )
            # TL_BANK_SWIZZLE_MODE selects the conflict-removal layout:
            #   "pad" (default) -> +1-column linear pad (cheap stride i*(K+1)+j)
            #   "xor"            -> make_swizzled_layout (full ldmatrix XOR)
            # A pad changes ONLY the physical stride (one mul/add per access),
            # far cheaper integer math than the multi-term XOR; it removes the
            # 2-way bank conflicts on a power-of-two inner stride at near-zero
            # address cost. MEASURED same-session A/B on this fp32 SIMT dstates
            # kernel (issue/latency bound, NOT shared-bandwidth bound): the pad
            # is a small reproducible WIN (16.72 -> 16.51 ms CUDA-event,
            # bank-conflict LD 14.69M -> 7.34M), while the full XOR swizzle
            # REGRESSES 2x (16.72 -> 26.06 ms) -- its per-access XOR address math
            # outweighs the (larger 87%) conflict savings because conflicts are
            # NOT the binding bottleneck here. Pad is therefore the default; XOR
            # is retained (env-selectable) for bandwidth-bound shapes where the
            # deeper conflict removal pays off. RULE #1: both modes are proven
            # bit-EXACT (MAXDIFF 0.0 small + large b2/nh128/s16384/nc256);
            # neither is a degraded fallback.
            import os as _os  # noqa: WPS433
            _mode = _os.environ.get("TL_BANK_SWIZZLE_MODE", "pad")
            if _mode == "pad":
                import tvm  # noqa: WPS433
                from tilelang.layout import Layout  # noqa: WPS433
                _shape = [int(d) for d in live.shape]
                if len(_shape) != 2:
                    raise RuntimeError(
                        "BANKSWIZZLE pad mode requires a rank-2 tile, got shape "
                        "%r for %r (RULE#1 RAISE)" % (_shape, getattr(live, "name", live))
                    )
                _pad = int(_os.environ.get("TL_BANK_SWIZZLE_PAD", "1"))
                _stride = _shape[1] + _pad
                def _forward(i, j, _st=_stride):
                    return [i * _st + j]
                layout_map[live.data] = Layout(_shape, _forward)
            else:
                layout_map[live.data] = make_swizzled_layout(live)

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
