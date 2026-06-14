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
import sys
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple

# Probe ~/.triton/llvm, brew, and IREE for an mlir.ir provider before
# importing the walker. ``_mlir_path_setup`` adjusts ``sys.path`` and/or
# registers an ``iree.compiler.ir``-backed alias under
# ``sys.modules['mlir.ir']`` so the walker's subsequent
# ``try_import_mlir`` succeeds. If every probe misses the walker emits
# its existing one-shot UserWarning unchanged.
from . import _mlir_path_setup  # noqa: F401  (import-time side effect)

# Explicit second probe: ``_mlir_path_setup`` runs ``probe_and_wire_mlir()``
# at module load, but jaxlib aliasing is deferred unless Triton is already
# loaded (to avoid nanobind / LLVM CL conflicts when Triton subsequently
# loads its own ``_C`` bindings). When the caller has pre-loaded Triton
# (e.g. via the numeric harness) the import-time probe missed; calling it
# again here picks up the jaxlib alias before downstream callers consult
# ``MLIR_WALKER_AVAILABLE``. The call is idempotent and cheap.
_mlir_path_setup.probe_and_wire_mlir()

from .mlir_walker import (
    DEGRADED_WARNING_MESSAGE as _DEGRADED_WARNING_MESSAGE,
    MLIR_WALKER_AVAILABLE,
    TTIRWalker,
    build_addressing_fold_set,
    parse_ttir,
    try_import_mlir,
    walk_module,
)
from .op_mapping import LazyTileExpr, OP_TABLE, WalkerCtx
from .ptr_analysis import PtrAnalysis, shim_available

from .static_shape_specialize import (  # noqa: E402
    INT32_ELEM_LIMIT,
    int32_index_safe,
    specialize_static_shape,
)

__all__ = [
    "from_triton_kernel",
    "from_ttir",
    "autotune_winning_block_config",
    "TileLangPrimFunc",
    "MLIR_WALKER_AVAILABLE",
    "specialize_static_shape",
    "int32_index_safe",
    "INT32_ELEM_LIMIT",
]


# One-shot guard so we don't spam the fallback warning per-call.
_FALLBACK_WARNED: bool = False


# ---------------------------------------------------------------------------
# tt.load / tt.store wiring helper (input-buffer plumbing)
#
# The op_emitters/memory.py emit_tt_load/emit_tt_store have a fallback path
# (``_is_tile_shape(out_shape)``) that synthesises a fresh tile-scoped
# placeholder buffer when no PtrState is found via ``ctx.ptr_states``.
# In the production path, the PtrAnalysis pre-pass populates
# ``ctx.value_map`` only with stringly-keyed PtrState entries (NOT
# ``ctx.ptr_states``); the walker sees the original (pre-rewrite) TTIR;
# ``tt.addptr`` does correctly carry the input Buffer through as a
# ``(buffer, [offset])`` tuple via the post-rewrite no-op shim path.
# However, the tile-shape branch in emit_tt_load IGNORES that tuple and
# allocates a fresh local placeholder, so the body never references the
# function-arg buffers and the Metal codegen prunes them from the kernel
# signature -- producing the ``buffer count mismatch`` failure observed
# in the e2e numeric harness.
#
# The wrappers below intercept tt.load / tt.store BEFORE delegating to the
# upstream emitter and, when the resolved value is a ``(buffer, indices)``
# tuple whose buffer is a real PrimFunc parameter (i.e. lives in
# ``ctx.buffers``), emit a rolled ``tir.For`` over the real buffer via
# ``_emit_tile_copy_tir`` / ``_emit_tile_store_tir``. This wires the
# function-arg buffers into the body so TileLang's Metal codegen keeps
# them in the kernel signature.
# ---------------------------------------------------------------------------


def _value_is_input_buffer(ctx: Any, val: Any) -> bool:
    """Return True if ``val`` is a Buffer that lives in ``ctx.buffers``."""
    try:
        buffers = getattr(ctx, "buffers", {}) or {}
        for buf in buffers.values():
            if buf is val:
                return True
    except Exception:
        return False
    return False


def _flat_extent_for_indices(
    ctx: Any,
    offset_indices: Any,
    tile_shape: Any,
) -> Any:
    """Compute a symbolic flat extent that upper-bounds ``offset_indices``.

    The tile-load/store wrappers redecl the input buffer with the **tile
    shape** (e.g. ``(128,)``). When the kernel uses ``program_id`` based
    addressing -- for instance ``tl.store(out_ptr + pid * row_stride +
    col_offsets, ...)`` -- the per-lane index reaches up to ``(gridDim - 1)
    * row_stride + tile_extent - 1``, which exceeds the tile-shape bound.
    TileLang's ``LegalizeSafeMemoryAccess`` pass then synthesises a runtime
    guard ``if (idx < tile_shape) { store }`` and silently drops every
    store outside the first row.

    We avoid that by deriving a buffer shape that is symbolically large
    enough to subsume every reachable index. The pattern we need to match
    is

        Broadcast(pid * stride, lanes) + Ramp(0, 1, lanes)

    or its scalar form ``pid * stride + i`` (post per-lane lowering). For
    each ``pid * stride`` we substitute the pid Var by ``gridDim - 1`` and
    add the tile extent, producing ``(gridDim - 1) * stride + tile_extent``
    -- a sound upper bound the analyzer can use to discharge the
    ``LegalizeSafeMemoryAccess`` upper-bound check.

    Returns a ``[extent]`` 1D shape. If the indices contain no recognised
    pid-pattern, falls back to the input ``tile_shape``.
    """
    try:
        tir = ctx.tir()
        tvm_mod = ctx.tvm()
    except Exception:
        return list(tile_shape) or [1]

    program_id_vars = list(getattr(ctx, "program_id_vars", []) or [])
    if not program_id_vars:
        return list(tile_shape) or [1]

    pid_to_extent = {var: extent for (var, _axis, extent) in program_id_vars}

    flat_tile_extent = 1
    for d in (tile_shape or [1]):
        flat_tile_extent *= int(d)

    Broadcast = getattr(tvm_mod.tir, "Broadcast", None)
    Ramp = getattr(tvm_mod.tir, "Ramp", None)

    # Walk the (single, post-flatten) offset expression collecting Vars
    # that are program_id Vars. For the typical ``pid * stride + col``
    # form we want to substitute pid -> (gridDim - 1) so the resulting
    # expression upper-bounds every reachable index.
    def _scalar_form(expr: Any) -> Any:
        # Strip Broadcast/Ramp wrappers used pre-vectorize.
        if Broadcast is not None and isinstance(expr, Broadcast):
            return _scalar_form(expr.value)
        if Ramp is not None and isinstance(expr, Ramp):
            # Ramp(base, stride, lanes): the maximum is base + stride*(lanes-1).
            return expr.base + expr.stride * (int(expr.lanes) - 1)
        if hasattr(expr, "a") and hasattr(expr, "b"):
            # Add/Sub/Mul expressions: recurse into both children. We
            # rebuild the expression with scalar-form subterms so that
            # Substitute below sees a plain PrimExpr DAG.
            try:
                a = _scalar_form(expr.a)
                b = _scalar_form(expr.b)
                # Reconstruct the same node type with substituted children.
                if isinstance(expr, tvm_mod.tir.Add):
                    return a + b
                if isinstance(expr, tvm_mod.tir.Sub):
                    return a - b
                if isinstance(expr, tvm_mod.tir.Mul):
                    return a * b
            except Exception:
                pass
        return expr

    if not offset_indices:
        return list(tile_shape) or [1]

    # We collapse the offset expression to a single scalar that bounds
    # every per-lane index. Concatenate via "+"; downstream we substitute
    # pid Vars for their max.
    expr = None
    for entry in offset_indices:
        if isinstance(entry, LazyTileExpr):
            # WRITE (grid-scaled output) path -- e.g. the ``dprev_states``
            # flat-address tile. Each launch-grid block writes a DISJOINT
            # per-program tile, so the union of all stores tiles the whole
            # output exactly: ``flat_tile_extent * gridDim_product`` is the
            # true, tight output extent. Over-declaring only weakens a guard;
            # under-declaring drops stores outside the first tile (the
            # truncation bug FIX#1 cured). Keep grid-scaling for writes.
            extent_expr = tir.const(int(flat_tile_extent), "int32")
            for _var, _axis, extent in program_id_vars:
                extent_expr = extent_expr * extent
            return [extent_expr]
        if isinstance(entry, tvm_mod.tir.Buffer):
            # READ (strided multi-K-trip input) path -- e.g. the ``dout`` /
            # ``C`` / ``dA_cumsum`` flat-address carry tiles threaded through
            # the chunk K-loop. Here ``flat_tile_extent * gridDim_product``
            # is the WRONG extent: every program reads the SAME tensor at a
            # per-program seqlen-strided base AND advances across multiple
            # K-trips, so the per-program read regions OVERLAP and span far
            # more than one tile-per-block. ``flat_tile_extent * grid``
            # therefore UNDER-counts the true seqlen-strided tensor (e.g.
            # 2048*grid=131072 vs the real dout 262144), which made
            # LegalizeSafeMemoryAccess bake an ``idx < 131072`` guard that
            # masks every chunk c>0 read to 0 (the MMA then consumes zeros)
            # AND mismatches the real DLTensor the caller passes (the FFI
            # marshaling SIGSEGV). RULE #1: a too-small strided-read extent
            # is a truncation bug -- we must size the buffer to the TRUE
            # tensor, not silently mask valid reads to 0.
            #
            # The true seqlen-strided extent is symbolic (it depends on the
            # runtime strides/dims). The native Triton read mask is the
            # per-row/col bound (``offs_m<hdim`` & ``offs_k<chunk_rem``) with
            # NO whole-buffer clamp; the per-block base pointer already lives
            # in the carry index. We therefore declare the read buffer with a
            # FRESH SYMBOLIC extent Var bound by MakePackedAPI from the
            # passed DLTensor's real element count. That makes any
            # LegalizeSafeMemoryAccess bound ``idx < <real_numel>`` -- which
            # the in-mask reads always satisfy (never masked) and which
            # matches the real allocation (no FFI OOB). This drops the
            # under-bound flat clamp exactly as FIX#1 did for the output.
            try:
                size_name = (str(getattr(entry, "name", "buf")) or "buf") + "_numel"
                extent_var = tir.Var(ctx.fresh(size_name), "int64")
            except Exception:
                # Fall back to grid-scaling only if a fresh Var cannot be
                # minted; never fall back to the under-counted constant.
                extent_expr = tir.const(int(flat_tile_extent), "int32")
                for _var, _axis, extent in program_id_vars:
                    extent_expr = extent_expr * extent
                return [extent_expr]
            return [extent_var]
        scalar = _scalar_form(entry)
        expr = scalar if expr is None else (expr + scalar)

    if expr is None:
        return list(tile_shape) or [1]

    # Substitute every program_id Var by ``extent - 1`` (its maximum).
    # ``tir.stmt_functor.substitute`` expects Var -> PrimExpr.
    substitute = getattr(tvm_mod.tir.stmt_functor, "substitute", None)
    if substitute is None:
        return list(tile_shape) or [1]

    sub_map = {}
    for var, extent in pid_to_extent.items():
        sub_map[var] = extent - tir.const(1, "int32")

    try:
        upper_idx = substitute(expr, sub_map)
    except Exception:
        return list(tile_shape) or [1]

    # The buffer must contain ``upper_idx`` (inclusive); declare shape as
    # ``upper_idx + 1`` so ``index < shape`` is provable.
    extent_expr = upper_idx + tir.const(1, "int32")
    # Add tile slack: when the offset_indices already encode the full
    # ramp (matmul case), this is a no-op; for the scalar-fold path
    # (softmax: the trailing index is just ``pid * stride``), the per-
    # lane loop adds 0..tile_extent-1 on top, so the buffer must cover
    # tile_extent more elements.
    extent_expr = extent_expr + tir.const(int(flat_tile_extent), "int32")
    return [extent_expr]


def _decl_shape_exceeds_buffer(ctx: Any, decl_shape: Any, buf: Any) -> bool:
    """True iff ``decl_shape`` flat extent provably exceeds ``buf``'s extent.

    Used to decide whether a strided per-block store has outgrown a
    caller-seeded ("fixed") function-arg buffer. The grid-scaled extent
    returned by :func:`_flat_extent_for_indices` is typically
    ``flat_tile_extent * gridDim_x * gridDim_y * ...`` -- all integer
    constants -- so we fold it to an int and compare against the buffer's
    current static extent. When either side is non-constant we return True
    (grow): under-declaring drops valid stores, which is the bug we are
    fixing; over-declaring only weakens a bounds guard.
    """
    try:
        tvm_mod = ctx.tvm()
    except Exception:
        return False

    def _as_int(expr: Any) -> Optional[int]:
        if isinstance(expr, int):
            return expr
        try:
            v = int(expr)
            return v
        except Exception:
            pass
        # Constant-fold a PrimExpr DAG (Mul/Add of IntImm) via the analyzer.
        try:
            ana = tvm_mod.arith.Analyzer()
            folded = ana.simplify(expr)
            IntImm = tvm_mod.tir.IntImm
            if isinstance(folded, IntImm):
                return int(folded.value)
        except Exception:
            pass
        return None

    decl_flat = 1
    for d in (decl_shape or [1]):
        di = _as_int(d)
        if di is None:
            return True  # symbolic -> grow to be safe (never truncate)
        decl_flat *= di

    try:
        cur_rank = len(buf.shape)
        cur_flat = 1
        for d in buf.shape:
            di = _as_int(d)
            if di is None:
                return True
            cur_flat *= di
    except Exception:
        return True

    if cur_rank != 1:
        return True
    return decl_flat > cur_flat


def _redecl_input_buffer(
    ctx: Any,
    buf: Any,
    shape: Any,
    dtype: str,
    *,
    offset_indices: Any = None,
    grow_fixed: bool = False,
) -> Any:
    """Re-declare an input buffer with the right tile shape.

    ``map_tt_func`` / ``_materialize_func_args`` declares input buffers
    with shape ``[1]`` (no kernel-level info available at func-args time).
    When the load/store later observes the tile size, we redecl the buffer
    in-place (same key in ``ctx.buffers``, new shape) so ``BufferLoad/Store``
    indexing is in-bounds. Without this, ``tir.BufferLoad(arg0, [off + lv])``
    against a ``T.Buffer((1,))`` would fail TVM's bound checks at lower time.

    When ``offset_indices`` is provided and the indices reference one of
    the kernel's ``program_id`` Vars, the actual reachable index runs up
    to ``(gridDim - 1) * stride + tile_extent``. Declaring the buffer
    with the tile shape (e.g. ``(128,)``) in that case causes
    ``LegalizeSafeMemoryAccess`` to inject a runtime ``idx < 128`` guard
    on the global store and silently drops every row beyond the first.
    See :func:`_flat_extent_for_indices` for the per-lane upper-bound
    derivation.
    """
    try:
        tir = ctx.tir()
    except Exception:
        return buf

    name = getattr(buf, "name", None) or "buf"
    target_key: Any = None
    for k, v in (getattr(ctx, "buffers", {}) or {}).items():
        if v is buf:
            target_key = k
            break
    if target_key is None:
        return buf

    if offset_indices is not None:
        decl_shape = _flat_extent_for_indices(ctx, offset_indices, shape)
    else:
        decl_shape = list(shape) or [1]

    fixed_keys = getattr(ctx, "fixed_arg_buffer_keys", set()) or set()
    is_fixed = target_key in fixed_keys or str(name) in fixed_keys
    if is_fixed:
        # A caller-seeded ("fixed") function-arg buffer is authoritative
        # ONLY when it is large enough to hold every store this kernel
        # makes. A program_id-strided store (grid-scaled output, e.g.
        # ``dprev_states``) reaches up to ``(gridDim-1)*stride+tile`` --
        # well past a single-tile seed. Honoring a too-small seed here
        # silently truncates the output to the FIRST tile (every later
        # grid block is dropped by LegalizeSafeMemoryAccess's
        # ``idx < seed`` guard). RULE #1: that is a truncation bug, not a
        # contract -- so we OVERRIDE the seed and grow to the grid-scaled
        # extent. When the seed already covers the writes (the contract
        # case) we keep it untouched.
        grow_required = (
            grow_fixed and offset_indices is not None
            and _decl_shape_exceeds_buffer(ctx, decl_shape, buf)
        )
        if not grow_required:
            return buf

    try:
        # Reuse the ORIGINAL buffer's backing data ``Var`` for the re-declared
        # buffer. ``decl_buffer`` defaults to minting a fresh data Var named
        # after ``name`` ("arg4"); doing so here would leave the kernel with
        # TWO distinct Vars both named "arg4" -- the param's (bound by
        # MakePackedAPI) and this redecl's. Any use of the original buffer
        # already emitted into the body BEFORE this redecl (a load/store in a
        # region walked earlier) keeps the original data Var, while the param
        # buffer_map carries the new one, so MakePackedAPI sees a free Var and
        # raises "variables (arg4,) are used, but are not passed in". Binding
        # the SAME data Var makes every reference -- old emitted stmts, the new
        # grid-scaled buffer, and the param buffer_map -- resolve to one Var.
        new_buf = tir.decl_buffer(
            shape=decl_shape, dtype=dtype, name=name, data=buf.data,
        )
    except Exception:
        return buf
    ctx.buffers[target_key] = new_buf
    if hasattr(ctx, "value_map"):
        try:
            for k, v in list(ctx.value_map.items()):
                if v is buf:
                    ctx.value_map[k] = new_buf
                elif isinstance(v, tuple) and len(v) == 2 and v[0] is buf:
                    ctx.value_map[k] = (new_buf, v[1])
        except Exception:
            pass
    return new_buf


def _emit_tile_load_from_input_buffer(
    op: Any,
    ctx: Any,
    src_buf: Any,
    offset_indices: Any,
    out_shape: Any,
    out_dtype: str,
    mask_ssa: Any,
    other_ssa: Any,
) -> Any:
    """Emit a per-lane ``tir.For`` over ``src_buf`` using ``offset_indices``.

    Unlike ``_emit_tile_copy_tir`` (which assumes ``base_indices`` are
    scalar PrimExprs and forms ``base + lv`` for the per-lane index), the
    tt.addptr fallback hands us indices that may be **tile buffers** (the
    common ``%offsets = pid * BLOCK + arange(BLOCK)`` pattern). For those
    we form the per-lane index as ``offset_buf[lv]``; for plain PrimExprs
    we keep ``offset + lv``.
    """
    tir = ctx.tir()
    tvm_mod = ctx.tvm()
    from .op_emitters.memory import _alloc_tile_buffer, _resolve_lane_operand, _results

    result_value = _results(op)[0] if _results(op) else None
    out_buf_name = ctx.fresh("tile_load")
    # Rank-2+ operand tiles may feed GEMM and need shared scope; rank-1
    # staging stays local to avoid unnecessary threadgroup memory.
    tile_scope = "shared" if len(out_shape or []) >= 2 else "local"
    tile_buf = _alloc_tile_buffer(
        ctx, list(out_shape) or [1], out_dtype, out_buf_name, scope=tile_scope
    )

    loop_vars: List[Any] = []
    for axis, _extent in enumerate(out_shape or [1]):
        loop_vars.append(tir.Var(ctx.fresh(f"i{axis}"), "int32"))

    from .op_emitters.memory import _read_vector_lane, _vector_lanes

    # Detect "single rank-N offset buffer" case. When tt.addptr collapses
    # to a single 2D (or higher-rank) offset tile buffer matching the load
    # tile shape (matmul's ``a_ptrs`` / ``b_ptrs`` are tensor<64x64xi32>),
    # ptr_analysis surfaces ``offset_indices`` as a single buffer covering
    # all axes -- not one buffer per axis. The offsets stored in that
    # buffer are FLAT linear addresses (e.g. ``i*stride_am + j*stride_ak``).
    # We index the offset buffer with the FULL loop-var nest to retrieve a
    # scalar flat address, then index the source buffer with that single
    # linear address. Producing one BufferLoad per axis (as the per-axis
    # fallback below does) would index a 2D buffer with a 1D index list
    # and trip ``buffer->shape.size() == indices.size()`` in
    # tirx::BufferLoad.
    single_offset_buf: Any = None
    if (
        len(offset_indices) == 1
        and isinstance(offset_indices[0], (LazyTileExpr, tvm_mod.tir.Buffer))
        and len(offset_indices[0].shape) == len(loop_vars)
        and len(loop_vars) >= 2
    ):
        single_offset_buf = offset_indices[0]

    src_indices: List[Any] = []
    if single_offset_buf is not None:
        # Flat-address scheme: redecl src_buf as 1D so a single linear
        # address from offset_buf is a valid index. Without this the
        # surrounding BufferLoad on a rank-N src_buf would itself trip the
        # rank-mismatch check below. ``_redecl_input_buffer`` updates
        # ctx.buffers / value_map in place, so subsequent emitters see the
        # 1D shape consistently.
        if len(getattr(src_buf, "shape", []) or []) != 1:
            flat_extent = 1
            for _e in out_shape:
                flat_extent *= int(_e)
            src_buf = _redecl_input_buffer(ctx, src_buf, [flat_extent], out_dtype)
        if isinstance(single_offset_buf, LazyTileExpr):
            src_indices.append(single_offset_buf.read_lane(ctx, tuple(loop_vars)))
        else:
            src_indices.append(tir.BufferLoad(single_offset_buf, list(loop_vars)))
    else:
        for axis, lv in enumerate(loop_vars):
            if axis < len(offset_indices):
                base = offset_indices[axis]
            else:
                base = tir.const(0, "int32")
            if isinstance(base, LazyTileExpr):
                rank = len(base.shape)
                if rank >= len(loop_vars):
                    src_indices.append(base.read_lane(ctx, tuple(loop_vars[:rank])))
                else:
                    src_indices.append(base.read_lane(ctx, tuple(loop_vars[-rank:])))
            elif isinstance(base, tvm_mod.tir.Buffer):
                # Tile-buffer offset. Index it with as many of the surrounding
                # loop_vars as the buffer's rank requires (matmul's a_ptrs
                # tile is rank-N when broadcast across all axes; vector_add's
                # offsets buffer is rank-1).
                buf_rank = len(base.shape)
                if buf_rank <= 0:
                    src_indices.append(tir.BufferLoad(base, [tir.const(0, "int32")]))
                elif buf_rank >= len(loop_vars):
                    src_indices.append(tir.BufferLoad(base, list(loop_vars[:buf_rank])))
                else:
                    # Buffer is lower-rank than the surrounding nest: take
                    # the trailing `buf_rank` loop vars (matches numpy-style
                    # right-aligned broadcasting).
                    src_indices.append(tir.BufferLoad(base, list(loop_vars[-buf_rank:])))
            elif _vector_lanes(base) > 1:
                # Vector PrimExpr (e.g. ``Broadcast(pid*N, N) + Ramp(0,1,N)`` from
                # ``addptr(splat(ptr), col_offsets)``). ``base + lv`` would yield a
                # vector dtype and trip BufferStore's
                # ``index_lanes * buffer_lanes == value_dtype_lanes`` check. Read
                # the per-lane scalar element instead so the index is rank-1
                # scalar, matching the surrounding tile's serial For nest.
                src_indices.append(_read_vector_lane(ctx, base, lv))
            else:
                src_indices.append(base + lv)

    load_expr: Any = tir.BufferLoad(src_buf, src_indices)
    if mask_ssa is not None:
        try:
            mask_expr = ctx.get(mask_ssa)
        except KeyError:
            mask_expr = None
        if mask_expr is not None:
            if other_ssa is not None:
                try:
                    other_expr = ctx.get(other_ssa)
                except KeyError:
                    other_expr = tir.const(0, out_dtype)
            else:
                other_expr = tir.const(0, out_dtype)
            mask_lane = _resolve_lane_operand(ctx, mask_expr, loop_vars, role="mask")
            other_lane = _resolve_lane_operand(ctx, other_expr, loop_vars, role="other")
            load_expr = tir.if_then_else(mask_lane, load_expr, other_lane)

    body = tir.BufferStore(tile_buf, load_expr, list(loop_vars) or [tir.const(0, "int32")])
    for axis in range(len(loop_vars) - 1, -1, -1):
        extent = out_shape[axis] if axis < len(out_shape) else 1
        body = tir.For(
            loop_vars[axis],
            tir.const(0, "int32"),
            tir.const(int(extent), "int32"),
            tir.ForKind.SERIAL,
            body,
        )

    ctx.emit(body)
    if result_value is not None:
        ctx.bind(result_value, tile_buf)
    return tile_buf


def _wrap_tile_load_emitter(orig_emit: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap ``emit_tt_load`` so a tile load on a function-arg Buffer wires
    the read into the actual PrimFunc parameter rather than a fresh local.
    """

    def _wrapped(op: Any, ctx: Any) -> Any:
        try:
            from .op_emitters.memory import (
                _is_tile_shape,
                _operands,
                _resolved_or_none,
                _results,
                _shape_of,
                _dtype_of,
            )
        except Exception:
            return orig_emit(op, ctx)

        operands = _operands(op)
        if not operands:
            return orig_emit(op, ctx)
        ptr_ssa = operands[0]
        mask_ssa = operands[1] if len(operands) >= 2 else None
        other_ssa = operands[2] if len(operands) >= 3 else None
        resolved = _resolved_or_none(ctx, ptr_ssa)
        if not (isinstance(resolved, tuple) and len(resolved) == 2):
            return orig_emit(op, ctx)
        buf, indices = resolved
        if not _value_is_input_buffer(ctx, buf):
            return orig_emit(op, ctx)
        result_value = _results(op)[0] if _results(op) else None
        out_shape = list(_shape_of(result_value)) if result_value is not None else []
        out_dtype = _dtype_of(result_value) if result_value is not None else "float32"
        if not _is_tile_shape(out_shape):
            return orig_emit(op, ctx)
        buf = _redecl_input_buffer(
            ctx, buf, out_shape, out_dtype,
            offset_indices=list(indices),
        )
        return _emit_tile_load_from_input_buffer(
            op, ctx, buf, list(indices), out_shape, out_dtype,
            mask_ssa, other_ssa,
        )

    return _wrapped


def _emit_tile_store_to_input_buffer(
    op: Any,
    ctx: Any,
    dst_buf: Any,
    offset_indices: Any,
    val_expr: Any,
    val_shape: Any,
    mask_ssa: Any,
) -> Any:
    """Symmetric to :func:`_emit_tile_load_from_input_buffer` for stores.

    Indices may be tile-buffers (typical ``offsets = pid*BLOCK + arange``
    pattern); for those we form the per-lane index as ``offset_buf[lv]``.
    Otherwise we form ``offset + lv``.
    """
    tir = ctx.tir()
    tvm_mod = ctx.tvm()
    from .op_emitters.memory import _resolve_lane_operand, _read_vector_lane, _vector_lanes

    loop_vars: List[Any] = []
    for axis, _extent in enumerate(val_shape or [1]):
        loop_vars.append(tir.Var(ctx.fresh(f"i{axis}"), "int32"))

    # Mirror of the load-side "single rank-N offset buffer" case: when
    # tt.addptr collapses to one rank-N flat-address tile (matmul stores
    # with ``c_ptrs`` shaped tensor<64x64xi32>), index that buffer with
    # the FULL loop-var nest and use the resulting scalar as a 1D linear
    # address into ``dst_buf``.
    single_offset_buf: Any = None
    if (
        len(offset_indices) == 1
        and isinstance(offset_indices[0], (LazyTileExpr, tvm_mod.tir.Buffer))
        and len(offset_indices[0].shape) == len(loop_vars)
        and len(loop_vars) >= 2
    ):
        single_offset_buf = offset_indices[0]

    # Grid-scaled output sizing (RULE #1): a strided per-block store carries a
    # ``single_offset_buf`` flat-address tile whose per-program base offset
    # (``pid_b*stride_b + ... + offs*stride``) reaches across the WHOLE launch
    # grid. Grow the function-arg destination to the grid-scaled extent
    # (``flat_tile_extent * gridDim_product`` via ``_flat_extent_for_indices``)
    # -- even past a too-small caller ("fixed") seed, because a seed below the
    # grid extent provably truncates the output to the first tile. We do the
    # redecl HERE (before the T.copy fast-path decision) and detect whether it
    # actually grew the buffer.
    # Restrict grid-scaling to a 1D FLAT function-arg destination addressed by
    # a single rank-N flat-address offset tile (the ``dprev_states`` pattern:
    # ``arg2`` is a flat buffer, ``single_offset_buf`` holds the flat linear
    # address ``offs_m*stride_m + offs_n*stride_n``). A genuinely multi-dim
    # destination (e.g. fla_dot_exp2's 2D ``arg2``) is NOT a flat grid-scaled
    # output and keeps the original same-shaped ``T.copy`` epilogue -- diverting
    # it to the 1D per-lane path would index a 2D buffer with one index and
    # trip ``buffer->shape.size() == indices.size()``.
    dst_is_flat_1d = False
    try:
        dst_is_flat_1d = len(getattr(dst_buf, "shape", []) or []) == 1
    except Exception:
        dst_is_flat_1d = False

    grid_scaled = False
    if single_offset_buf is not None:
        dst_dtype = str(getattr(dst_buf, "dtype", "float32"))
        if dst_is_flat_1d:
            # Flat 1D function-arg destination: apply grid-scaling (grow even a
            # too-small fixed seed) so the per-block strided writes land
            # in-bounds for EVERY grid block.
            prev_buf = dst_buf
            dst_buf = _redecl_input_buffer(
                ctx,
                dst_buf,
                list(val_shape),
                dst_dtype,
                offset_indices=[single_offset_buf],
                grow_fixed=True,
            )
            grid_scaled = dst_buf is not prev_buf
        elif len(getattr(dst_buf, "shape", []) or []) != 1:
            # Original behaviour: a multi-dim destination indexed by a single
            # flat-address tile is flattened to 1D (tile footprint) so the
            # single linear-address BufferStore below is rank-matched. NOT a
            # grid-scaled output -- keep the same-shaped tile semantics.
            flat_extent = 1
            for _e in val_shape:
                flat_extent *= int(_e)
            dst_buf = _redecl_input_buffer(ctx, dst_buf, [flat_extent], dst_dtype)

    # ``T.copy`` fast path: valid for a same-shaped tile destination. Only a
    # destination we just GREW for grid-scaling needs the per-lane / region
    # epilogue below; a normal same-shaped tile store (e.g. fla_dot_exp2's
    # in-place fragment store, where the buffer was NOT grown) keeps the fast
    # ``T.copy``. Diverting those to the per-lane path regresses them, so we
    # gate the skip strictly on ``grid_scaled``.
    scope_fn = getattr(val_expr, "scope", None)
    try:
        val_scope = scope_fn() if callable(scope_fn) else scope_fn
    except Exception:
        val_scope = None
    if val_scope in {"local.fragment", "metal.simdgroup"} and not grid_scaled:
        import tilelang.language as T  # type: ignore

        handle = T.copy(val_expr, dst_buf)
        if isinstance(handle, tvm_mod.tir.PrimExpr):
            stmt = tir.Evaluate(handle)
            ctx.emit(stmt)
            return stmt
        if handle is not None:
            ctx.emit(handle)
        return handle

    dst_indices: List[Any] = []
    if single_offset_buf is not None:
        if isinstance(single_offset_buf, LazyTileExpr):
            dst_indices.append(single_offset_buf.read_lane(ctx, tuple(loop_vars)))
        else:
            dst_indices.append(tir.BufferLoad(single_offset_buf, list(loop_vars)))
    else:
        for axis, lv in enumerate(loop_vars):
            if axis < len(offset_indices):
                base = offset_indices[axis]
            else:
                base = tir.const(0, "int32")
            if isinstance(base, LazyTileExpr):
                rank = len(base.shape)
                if rank >= len(loop_vars):
                    dst_indices.append(base.read_lane(ctx, tuple(loop_vars[:rank])))
                else:
                    dst_indices.append(base.read_lane(ctx, tuple(loop_vars[-rank:])))
            elif isinstance(base, tvm_mod.tir.Buffer):
                buf_rank = len(base.shape)
                if buf_rank <= 0:
                    dst_indices.append(tir.BufferLoad(base, [tir.const(0, "int32")]))
                elif buf_rank >= len(loop_vars):
                    dst_indices.append(tir.BufferLoad(base, list(loop_vars[:buf_rank])))
                else:
                    dst_indices.append(tir.BufferLoad(base, list(loop_vars[-buf_rank:])))
            elif _vector_lanes(base) > 1:
                # Vector PrimExpr offset (e.g. ``Broadcast(pid*N, N) + Ramp(0,1,N)``
                # from ``addptr(splat(ptr), col_offsets)``). ``base + lv`` would
                # yield a vector index and trip BufferStore's
                # ``index_lanes * buffer_lanes == value_dtype_lanes`` check. Read
                # the per-lane scalar so the surrounding serial For nest stores
                # element-by-element.
                dst_indices.append(_read_vector_lane(ctx, base, lv))
            else:
                dst_indices.append(base + lv)

    # Stage an MMA fragment value through a same-shaped shared tile before a
    # strided per-block store. A loop-carried GEMM accumulator
    # (``local.fragment``) carries the tensor-core MMA store layout; reading
    # it with arbitrary per-lane indices in this serial store loop makes
    # TileLang's layout inference see two incompatible layouts for the same
    # buffer ("Get different layout for <carry>"). Native Triton likewise
    # materialises ``out = acc.to(dtype)`` into a fresh value before the
    # store. We mirror that: ``T.copy(fragment -> staging)`` (the only
    # consumer of the fragment, so its MMA layout is unambiguous), then read
    # the store lanes from the staging tile.
    val_scope_now = None
    try:
        _sf = getattr(val_expr, "scope", None)
        val_scope_now = _sf() if callable(_sf) else _sf
    except Exception:
        val_scope_now = None
    fragment_epilogue = (
        single_offset_buf is not None
        and isinstance(val_expr, tvm_mod.tir.Buffer)
        and val_scope_now in {"local.fragment", "metal.simdgroup"}
    )
    if fragment_epilogue:
        # MMA-C fragment epilogue -> grid-scaled global output.
        #
        # The accumulator is a tensor-core fragment whose physical lane
        # layout the codegen owns; a hand-rolled per-lane serial store loop
        # reads it at LOGICAL [i,j] and therefore captures only one lane's
        # data (sparse / wrong output). The correct lowering is TileLang's
        # own layout-aware ``T.copy``:
        #   1. ``T.copy(fragment -> shared)``  -- register->smem, MMA-layout
        #      aware (verified to match A@B in a standalone GEMM).
        #   2. ``T.copy(shared -> arg2_region)`` -- smem->global over the
        #      per-block contiguous tile region ``[base : base+tile_numel]``.
        # The per-program base offset is ``dst_indices[0]`` evaluated with
        # every store loop var set to 0 (the within-tile ``offs`` collapse to
        # the tile origin). dprev_states is contiguous in (hdim,dstate), so
        # the per-block tile IS a contiguous flat slice and the rank-N tile
        # maps row-major onto it. The native tile mask (offs_m<hdim &
        # offs_n<dstate) is a no-op here because BLOCK==(hdim,dstate); when
        # they differ TileLang's region copy clamps to the in-bounds extent.
        import tilelang.language as T  # type: ignore

        # Per-block base = store index with all tile loop vars -> 0 (the
        # within-tile ``offs`` collapse to the tile origin). dprev_states is
        # contiguous in (hdim,dstate), so the per-block tile IS a contiguous
        # flat slice ``[base : base+tile_numel]`` and the rank-N fragment maps
        # row-major onto it.
        zero = tir.const(0, "int32")
        substitute = tvm_mod.tir.stmt_functor.substitute
        base_expr = substitute(dst_indices[0], {lv: zero for lv in loop_vars})
        tile_numel = 1
        for _e in val_shape:
            tile_numel *= int(_e)
        # The fragment is rank-N (e.g. 2D 64x64); the destination region must
        # match that rank. arg2 is a flat 1D function-arg buffer, so we expose
        # a 2D row-major view ``[grid_rows, tile_cols]`` of the SAME data and
        # take a rank-N region at the per-block tile origin. The per-block
        # base is contiguous so ``base = row0 * tile_cols`` -> the 2D region
        # origin is ``(base // tile_cols, 0)`` with extent ``val_shape``.
        if len(val_shape) >= 2:
            tile_cols = int(val_shape[-1])
            # ``total`` (and therefore ``view_rows``) may be SYMBOLIC: when the
            # autotune-winning tile re-declares the function-arg output buffer,
            # a flattened dim can be a tir.Mul (e.g. grid_rows*tile_cols) rather
            # than a Python int, so int(d) would crash. Track a plain int while
            # every dim is a concrete constant; the moment a non-constant dim
            # appears, fall over to a TIR PrimExpr product so the symbolic dim
            # is preserved. ``view_rows`` is then a valid (possibly symbolic)
            # extent for the 2D row-major view; ``row0`` (the per-block origin)
            # is already symbolic, so the region stays correct either way.
            def _const_int(_d):
                if isinstance(_d, int):
                    return _d
                _v = getattr(_d, "value", None)
                return _v if isinstance(_v, int) else None
            total_int = 1
            total_expr = None  # set once a non-constant dim is seen
            for d in dst_buf.shape:
                ci = _const_int(d)
                if ci is not None and total_expr is None:
                    total_int *= ci
                else:
                    if total_expr is None:
                        total_expr = tir.const(int(total_int), "int32")
                    total_expr = total_expr * (
                        tir.const(int(ci), "int32") if ci is not None else d
                    )
            if total_expr is None:
                view_rows = total_int // tile_cols
            else:
                view_rows = tvm_mod.tir.floordiv(total_expr, tir.const(tile_cols, "int32"))
            # 2D row-major VIEW of arg2 that ALIASES its data Var (same
            # ``data`` pointer, zero elem_offset). A tir.Buffer has no
            # ``reshape``, so we declare a fresh 2D buffer over the same data.
            view2d = tir.decl_buffer(
                [view_rows, tile_cols],
                str(getattr(dst_buf, "dtype", "float32")),
                name=str(getattr(dst_buf, "name", "out")) + "_2d",
                data=dst_buf.data,
                elem_offset=tir.const(0, "int32"),
            )
            row0 = tvm_mod.tir.floordiv(base_expr, tir.const(tile_cols, "int32"))
            ranges = [
                tvm_mod.ir.Range.from_min_extent(row0, tir.const(int(val_shape[0]), "int32")),
                tvm_mod.ir.Range.from_min_extent(zero, tir.const(tile_cols, "int32")),
            ]
            region = tvm_mod.tir.BufferRegion(view2d, ranges)
        else:
            region = tvm_mod.tir.BufferRegion(
                dst_buf,
                [tvm_mod.ir.Range.from_min_extent(base_expr, tir.const(int(tile_numel), "int32"))],
            )
        # Single layout-aware ``T.copy(fragment -> global_region)``: the
        # standard GEMM-C epilogue. The fragment's MMA store layout maps onto
        # the contiguous per-block global slice; the global buffer supplies
        # the destination layout, so no intermediate shared tile (which would
        # need a frame-registered layout we cannot create at emission time)
        # is required. RULE #1: if this still cannot lower, it RAISES at
        # compile -- it never falls back to a per-lane serial store that reads
        # the fragment at LOGICAL indices and produces sparse/wrong output.
        copy_handle = T.copy(val_expr, region)
        if isinstance(copy_handle, tvm_mod.tir.PrimExpr):
            ctx.emit(tir.Evaluate(copy_handle))
        elif copy_handle is not None:
            ctx.emit(copy_handle)
        return copy_handle

    # Pull the per-lane value from val_expr. val_expr is typically a tile
    # Buffer (alloced as the result of a prior tt.load / arith op).
    if isinstance(val_expr, tvm_mod.tir.Buffer):
        val_lane = tir.BufferLoad(val_expr, list(loop_vars) or [tir.const(0, "int32")])
    else:
        val_lane = _resolve_lane_operand(ctx, val_expr, loop_vars, role="value")

    store_stmt: Any = tir.BufferStore(dst_buf, val_lane, dst_indices)
    if mask_ssa is not None:
        try:
            mask_expr = ctx.get(mask_ssa)
        except KeyError:
            mask_expr = None
        if mask_expr is not None:
            mask_lane = _resolve_lane_operand(ctx, mask_expr, loop_vars, role="mask")
            store_stmt = tir.IfThenElse(mask_lane, store_stmt, None)

    body: Any = store_stmt
    for axis in range(len(loop_vars) - 1, -1, -1):
        extent = val_shape[axis] if axis < len(val_shape) else 1
        body = tir.For(
            loop_vars[axis],
            tir.const(0, "int32"),
            tir.const(int(extent), "int32"),
            tir.ForKind.SERIAL,
            body,
        )

    ctx.emit(body)
    return body


def _wrap_tile_store_emitter(orig_emit: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap ``emit_tt_store`` so a tile store on a function-arg Buffer wires
    the write into the actual PrimFunc parameter rather than a fresh local.
    """

    def _wrapped(op: Any, ctx: Any) -> Any:
        try:
            from .op_emitters.memory import (
                _is_tile_shape,
                _operands,
                _resolved_or_none,
                _shape_of,
                _dtype_of,
            )
        except Exception:
            return orig_emit(op, ctx)

        operands = _operands(op)
        if len(operands) < 2:
            return orig_emit(op, ctx)
        ptr_ssa, val_ssa = operands[0], operands[1]
        mask_ssa = operands[2] if len(operands) >= 3 else None
        resolved = _resolved_or_none(ctx, ptr_ssa)
        if not (isinstance(resolved, tuple) and len(resolved) == 2):
            return orig_emit(op, ctx)
        buf, indices = resolved
        if not _value_is_input_buffer(ctx, buf):
            return orig_emit(op, ctx)
        val_shape = list(_shape_of(val_ssa))
        if not _is_tile_shape(val_shape):
            return orig_emit(op, ctx)
        val_expr = ctx.get(val_ssa)
        dtype = _dtype_of(val_ssa) or "float32"
        buf = _redecl_input_buffer(
            ctx, buf, val_shape, dtype,
            offset_indices=list(indices),
        )
        return _emit_tile_store_to_input_buffer(
            op, ctx, buf, list(indices), val_expr, val_shape, mask_ssa,
        )

    return _wrapped


_LOAD_STORE_WRAPPERS_INSTALLED: bool = False


def _install_tile_load_store_wrappers() -> None:
    """Idempotently install tt.load/tt.store wrappers in OP_TABLE."""
    global _LOAD_STORE_WRAPPERS_INSTALLED
    if _LOAD_STORE_WRAPPERS_INSTALLED:
        return
    if "tt.load" in OP_TABLE:
        OP_TABLE["tt.load"] = _wrap_tile_load_emitter(OP_TABLE["tt.load"])
    if "tt.store" in OP_TABLE:
        OP_TABLE["tt.store"] = _wrap_tile_store_emitter(OP_TABLE["tt.store"])
    _LOAD_STORE_WRAPPERS_INSTALLED = True


_install_tile_load_store_wrappers()


def _fold_dead_local_stores(prim_func: Any) -> Any:
    """Apply DeadLocalStoreElim (fixpoint DCE of write-only ``scope="local"``
    index/mask staging tiles) as the final cleanup of the walked PrimFunc.

    See ``poc.triton_frontend.dead_local_store_elim`` for the why/what. This is
    a generic, backend-neutral, bit-exact TIR cleanup: it only DELETES provably
    dead staging writes (the per-lane int64 index / mask / broadcast tiles the
    op_emitters materialise but the inline-folded load/store/copy sites never
    read). RULE #1: it never drops a buffer that is read by a surviving
    statement, and on any unexpected structure returns the PrimFunc unchanged.
    """
    if prim_func is None:
        return None
    from .dead_local_store_elim import eliminate_dead_local_stores
    folded = eliminate_dead_local_stores(prim_func)
    return _attach_predicated_ldgstg_passcfg(folded)


# LEVER 1 (masked-load fold). Attach the predicated LDG/STG pass config to the
# frontend-emitted PrimFunc so ``tilelang.compile`` honors it automatically
# (jit/__init__.py reads ``tilelang_pass_configs`` from func attrs; an explicit
# external pass_configs still overrides). ``tl.enable_lower_ldgstg_predicated``
# turns every masked global load ``if_then_else(oob_cond, load, 0)`` and
# predicated global store ``if(pred){store}`` into a SINGLE predicated
# ``tl::load_global_*_conditional`` / ``tl::store_global_*_conditional``: the
# int64 address + the OOB predicate are computed ONCE (vs the prior scalar
# if/condval loop that recomputed the giant int64 address THREE times) and OOB
# lanes are inline-zeroed by the @p-guarded ld/st (bit-identical to the
# if_then_else->0 semantics; OOB still reads/writes 0). It is GENERIC +
# BACKEND-NEUTRAL: ``LowerLDGSTG`` is internally CUDA-gated (no-op on Metal/HIP/
# CPU) and only fires on global loads/stores with else==0 at supported vector
# widths, so non-CUDA backends and unmasked accesses are byte-unchanged. The
# OOB protection is KEPT (this is NOT a guard-drop); only the redundant
# recompute is removed. MEASURED on §P1 dstates (cache OFF): bit-exact
# MAXDIFF=0, dynamic warp-inst 99.35M->56.97M (-42%), routed 3.09ms->2.52ms
# (-18%), regs 95->96, 0 spill; 6/6 partial-mask shapes bit-exact (OOB=0);
# int64 2.82GB bit-exact.
def _attach_predicated_ldgstg_passcfg(prim_func: Any) -> Any:
    if prim_func is None:
        return None
    existing = None
    attrs = getattr(prim_func, "attrs", None)
    if attrs is not None and "tilelang_pass_configs" in attrs:
        existing = dict(attrs["tilelang_pass_configs"])
    merged = dict(existing or {})
    # Do not clobber an explicit caller/emitter choice for this key.
    merged.setdefault("tl.enable_lower_ldgstg_predicated", True)
    return prim_func.with_attr("tilelang_pass_configs", merged)


def _frame_register_mma_fragments(ctx: Any) -> Any:
    """FRAMEFIX: re-register CUDA MMA-C fragment layouts on the walked PrimFunc.

    Gated to the CUDA tensor-core path: ``ctx.mma_c_fragments`` is only
    populated by ``map_tt_dot`` when the gemm C accumulator is a
    ``local.fragment`` on an explicit CUDA target (the grid-scaled MMA-C case).
    On Metal / non-MMA paths the list is empty and the PrimFunc is returned
    unchanged, so fla_dot_exp2 and the rest of the suite are untouched.

    RULE #1: when fragments ARE recorded, the re-registration must succeed --
    ``register_mma_fragment_layouts`` RAISES on a missing fragment rather than
    silently returning a layout-blind PrimFunc.
    """
    prim_func = getattr(ctx, "prim_func", None)
    if prim_func is None:
        return None
    fragments = list(getattr(ctx, "mma_c_fragments", None) or [])
    if not fragments:
        return _fold_dead_local_stores(prim_func)
    from .frame_register import register_mma_fragment_layouts

    # BUG 2 FIX (RULE #1 -- determinism, not a silent skip). The SBlock
    # wrapping (alloc_buffers) is STILL required: it hosts the flat-emitted tile
    # buffers so LowerOpaqueBlock allocates them and the epilogue store
    # materialises (without it the routed output stays all-zeros). What we DROP
    # is the conflicting C ``layout_map`` PIN: ``_build_c_fragment_layout``
    # re-derived a ``make_mma_store_layout`` that did NOT exactly equal the
    # gemm's own InferLayout (a +1 warp-partition/offset skew), colliding
    # NON-DETERMINISTICALLY -> ``InternalError: Get different layout for
    # dot_c_frag`` (~3/3 repeat runs aborted). With BUG 1 fixed the gemm A/B
    # operands are staged as DISTRIBUTED fragments via cooperative ``T.copy``
    # (ldmatrix-matching), so ``T.gemm``'s InferLayout and the epilogue
    # ``T.copy(C_frag -> shared)`` CONVERGE on one C layout natively -- the pin
    # is obsolete and can only conflict. ``pin_c_layout=False`` keeps the
    # alloc-hosting SBlock but lets native LayoutInference own C's layout
    # (probe: 3/3 DETERMINISTIC compiles unpinned vs 3/3 conflict pinned).
    # DIRECT FRAGMENT->GLOBAL EPILOGUE: when the loop-carried MMA-C accumulator
    # is stored DIRECTLY to global (no shared ``carry_logical`` staging --
    # control.py ``_DIRECT_FRAG_GLOBAL_EPILOGUE_ENABLED``), the global-store
    # ``T.copy(fragment, global)`` has NO shared/fragment pair to anchor on. In
    # free-inference it would over-replicate the fragment to a flat
    # ``(_i*256+_j, replicate=256)`` layout and register THAT before the gemm's
    # MMA store layout, colliding ("Get different layout for carry_tile"). We
    # therefore STRICTLY PIN the C fragment to its gemm-exact
    # ``make_mma_store_layout`` (now derived with the gemm's true warp partition
    # via ``_square_warp_partition`` -- the partition skew that made the old pin
    # conflict is fixed). The pinned layout seeds ``strict_layout_map`` first, so
    # the gemm AND the direct global store both ADOPT it: the store loop is then
    # built FROM the fragment's MMA layout (layout-aware, per-warp, no shared).
    # When the legacy shared-staging epilogue is selected
    # (TL_FRAG_GLOBAL_EPILOGUE=0), keep unpinned -- the through-shared copy
    # converges natively and the pin is unnecessary there.
    try:
        from .op_emitters.control import (
            _DIRECT_FRAG_GLOBAL_EPILOGUE_ENABLED as _direct_epi,
        )
    except Exception:  # pragma: no cover - control always importable here
        _direct_epi = True
    registered = register_mma_fragment_layouts(
        prim_func, fragments, pin_c_layout=bool(_direct_epi)
    )
    return _fold_dead_local_stores(registered)


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
    after the first stage. The exact API has evolved across Triton 2.x,
    3.0/3.1 and 3.6+; we delegate to the harness implementation in
    :mod:`poc.triton_frontend._test_harness.jit_to_ttir` which already
    handles all three forms. Notably:

      * Triton 3.6 renamed ``ASTSource(constants=...)`` to
        ``ASTSource(constexprs=...)`` and dropped the
        ``compile(..., options={"stage": "ttir"})`` knob in favour of
        ``ASTSource.make_ir(target, options, codegen, module_map, ctx)``.
      * Triton 2.x kept the legacy positional
        ``triton.compile(fn, signature=..., output="ttir")`` API.

    The harness probes which spelling the installed Triton accepts and
    returns the TTIR text. The reducer below re-parses the text via
    ``mlir.ir.Module.parse`` so the textual / module paths converge.
    """
    from ._test_harness.jit_to_ttir import (  # noqa: WPS433 -- lazy import
        TTIRCaptureError,
        TritonUnavailable,
        triton_jit_to_ttir,
        triton_jit_to_ttir_subprocess_from_source,
    )
    from ._test_harness.native_import_guard import (  # noqa: WPS433
        triton_import_block_reason,
    )

    def _capture_ttir_in_subprocess(reason: str) -> Any:
        # The subprocess helper needs source text because Triton JIT functions
        # are process-local and not pickleable. This is the same production
        # seam used when another LLVM peer is already resident.
        try:
            import inspect
            import textwrap

            underlying = getattr(fn, "fn", fn)
            source = textwrap.dedent(inspect.getsource(underlying))
            kernel_name = (
                getattr(underlying, "__name__", None)
                or getattr(fn, "__name__", "main")
            )
        except (OSError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "Could not capture Triton TTIR safely in a subprocess: "
                f"kernel source is unavailable ({exc}); reason={reason}"
            ) from exc
        try:
            return triton_jit_to_ttir_subprocess_from_source(
                source=source,
                kernel_name=kernel_name,
                constexprs=constexprs,
                target=target,
            )
        except (TTIRCaptureError, TritonUnavailable) as exc:
            raise RuntimeError(
                f"Could not stop Triton compilation at the TTIR stage "
                f"(subprocess fallback; reason={reason}): {exc}"
            ) from exc

    block_reason = triton_import_block_reason()
    if block_reason is not None:
        # An LLVM peer (jaxlib MLIR or our C++ shim) is resident in
        # this interpreter. Calling triton.make_ir here would abort
        # the process on duplicate cl::opt registration. Recover the
        # kernel's source text and route TTIR capture through a fresh
        # subprocess so the call still succeeds end to end.
        return _capture_ttir_in_subprocess(block_reason)

    if sys.platform == "darwin":
        # On macOS the local TileLang dev build loads libtilelang /
        # libtvm_compiler. If this process has already loaded Triton's native
        # libtriton while capturing TTIR, the later TileLang import can abort
        # in dyld/LLVM static initializers ("Option 'basic' already exists!").
        # Capture TTIR in a fresh child so the parent can load TileLang/TVM
        # without libtriton.
        return _capture_ttir_in_subprocess("darwin isolates libtriton from TileLang/TVM")

    try:
        return triton_jit_to_ttir(fn, constexprs=constexprs, target=target)
    except ValueError:
        # Harness raises ValueError when LLVM / backends are missing
        # locally — the xfail-marked test relies on this. Propagate.
        raise
    except (TTIRCaptureError, TritonUnavailable) as exc:
        raise RuntimeError(
            f"Could not stop Triton compilation at the TTIR stage: {exc}"
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


# Structural / scaffolding TTIR ops that wrap the body of a kernel but do
# not themselves emit TIR. The MLIR walker silently skips these via the
# OP_TABLE-membership check; the text walker mirrors that behaviour with
# an explicit allow-list so we can keep raising NotImplementedError on
# truly-unknown ops (helps catch coverage regressions).
_TTIR_STRUCTURAL_OPS = frozenset({"tt.return"})


def _walk_text_ttir(
    ttir_text: str, ctx: Optional[WalkerCtx] = None
) -> List[str]:
    """Naive line-by-line walk over textual TTIR; returns op names visited.

    This intentionally does *not* parse operands or types -- it is the
    minimum surface needed to (a) confirm dispatch coverage in tests and
    (b) serve as a stand-in until MLIR Python bindings are wired up.
    Real lowering uses :func:`from_ttir` with a full ``mlir.ir.Module``.

    ``ctx`` is reserved for future use (the text walker is coverage-only
    today and does not populate ctx); when ``None`` a fresh ``WalkerCtx``
    is created so callers / tests don't have to thread one explicitly.
    """
    if ctx is None:
        ctx = WalkerCtx()
    visited: List[str] = []
    for line in ttir_text.splitlines():
        m = _OP_LINE.match(line)
        if not m:
            continue
        op_name = m.group("op")
        if op_name in _TTIR_STRUCTURAL_OPS:
            # Structural scaffolding -- recorded for coverage but no emit.
            visited.append(op_name)
            continue
        visited.append(op_name)
        if op_name not in OP_TABLE:
            raise NotImplementedError(
                f"triton_frontend: TTIR op '{op_name}' is not in OP_TABLE."
            )
    return visited


def _op_name_generic(op: Any) -> str:
    """Extract the dotted MLIR op name across binding shapes."""
    name = getattr(op, "name", None)
    if not name:
        inner = getattr(op, "operation", None)
        name = getattr(inner, "name", None) if inner is not None else None
    if not name and isinstance(op, dict):
        name = op.get("name")
    return str(name) if name else ""


def _op_results_generic(op: Any) -> Tuple[Any, ...]:
    if isinstance(op, dict):
        return tuple(op.get("results", ()))
    results = getattr(op, "results", None)
    if results is None:
        inner = getattr(op, "operation", None)
        results = getattr(inner, "results", None) if inner is not None else None
    return tuple(results or ())


def _walk_mlir_module(
    module: Any, ctx: Optional[WalkerCtx] = None
) -> List[str]:
    """Walk a real ``mlir.ir.Module`` and dispatch each op via OP_TABLE."""
    visited: List[str] = []
    # Triton TTIR stores tt.func ops at the module top level.
    # We recurse into all regions and dispatch by op name.

    def _op_name(op: Any) -> str:
        return _op_name_generic(op)

    # Lazy import to avoid a top-of-file cycle with mlir_walker. We use the
    # per-emitter ``owns_regions`` attribute (H4 Wave-I) but keep the legacy
    # set as a fallback for emitters that forgot to set the attribute.
    if sys.platform == "darwin" and "tilelang" not in sys.modules:
        from ._test_harness.native_import_guard import (  # noqa: WPS433
            triton_native_loaded,
            triton_native_symbols_are_local,
        )

        if triton_native_loaded() and not triton_native_symbols_are_local():
            raise RuntimeError(
                "triton_frontend cannot import TileLang/TVM in this process: "
                "triton._C.libtriton is already loaded, and this macOS "
                "TileLang dev build loads libtilelang/libtvm_compiler with a "
                "separate LLVM image. Co-loading them aborts in LLVM cl::opt "
                "registration (for example, Option 'basic' already exists). "
                "Capture TTIR in a fresh Python process and call from_ttir() "
                "from a process that has not imported native Triton."
            )
    import tilelang  # noqa: F401  (setup TVM environment)

    from .mlir_walker import (  # noqa: WPS433
        OPS_THAT_HANDLE_OWN_REGIONS,
        _emitter_owns_regions,
    )
    # Lazy import of the tt.func sym_name extractor + tt.call callee parser
    # (live in the control emitters module). Used by the pre-pass below to
    # seed ``ctx.callees`` so ``emit_tt_call`` can look up callees without
    # re-walking the module, and to flag which tt.func ops are referenced
    # by tt.call so we don't double-emit their bodies at module level.
    from .op_emitters.control import (  # noqa: WPS433
        _func_sym_name,
        _parse_callee_attr,
    )

    # Auto-wrap jaxlib-shaped modules so ``module.operation`` (which
    # carries ``regions``) is the recursion entry point. See
    # :func:`mlir_walker.wrap_module_for_walker` for the rationale.
    from .mlir_walker import wrap_module_for_walker as _wrap  # noqa: WPS433
    module = _wrap(module)
    body = getattr(module, "body", None) or getattr(module, "operation", module)

    # ------------------------------------------------------------------
    # tt.call pre-pass
    # ------------------------------------------------------------------
    #
    # Before dispatching anything we enumerate every ``tt.func`` and
    # ``tt.call`` reachable from the module root. Each tt.func is
    # registered in ``ctx.callees`` keyed by ``sym_name`` so
    # ``emit_tt_call`` can look up its target callee without doing its
    # own module walk. Each tt.call's referenced symbol is collected in
    # ``ctx.callee_used`` so the dispatch loop below knows which tt.func
    # ops to *skip* recursion into (their bodies are inlined at call
    # sites; re-walking them at module level would double-emit and
    # KeyError on unbound block-args).
    if ctx is not None:
        def _ssa_name(value: Any) -> str:
            try:
                getter = getattr(value, "get_name", None)
                if callable(getter):
                    name = getter()
                    if name:
                        return str(name)
            except Exception:
                pass
            try:
                name = getattr(value, "name", None)
                if name:
                    return str(name)
            except Exception:
                pass
            if isinstance(value, dict):
                name = value.get("name")
                if name:
                    return str(name)
            return ""

        def _prepass(op: Any) -> None:
            name = _op_name(op)
            for operand in getattr(op, "operands", ()) or ():
                operand_name = _ssa_name(operand)
                if operand_name:
                    ctx.ssa_users.setdefault(operand_name, set()).add(name)
                    # Also record the consumer op OBJECT (op wrappers may be
                    # unhashable -> use a list) so a fold gate can call another
                    # emitter helper on the real consumer, not parse its text.
                    ctx.ssa_user_ops.setdefault(operand_name, []).append(op)
            if name == "tt.func":
                sym = _func_sym_name(op)
                if sym:
                    ctx.callees[sym] = op
            elif name == "tt.call":
                callee_sym = _parse_callee_attr(op)
                if callee_sym:
                    ctx.callee_used.add(callee_sym)
            for region in getattr(op, "regions", ()) or ():
                for block in getattr(region, "blocks", ()) or ():
                    for child in getattr(block, "operations", ()) or ():
                        _prepass(child)
        _prepass(body)

    def _recurse(op: Any) -> None:
        op_name_str = _op_name(op)
        if op_name_str in OP_TABLE:
            visited.append(op_name_str)
            OP_TABLE[op_name_str](op, ctx)
        elif op_name_str in _TTIR_STRUCTURAL_OPS:
            # Same scaffolding skip as the text walker (tt.func / tt.return
            # are wrappers; recurse into them but emit nothing).
            visited.append(op_name_str)
        # Skip region descent for ops whose emitters walk their own
        # regions (``tt.reduce`` / ``tt.scan`` consume the combiner
        # region; ``scf.for`` / ``scf.if`` / ``scf.while`` use
        # ``_emit_region``; ``tt.call`` inline-walks its callee). Descending
        # here would dispatch inner ops before their parent emitter bound
        # the region's block arguments, surfacing as
        # ``KeyError: WalkerCtx: SSA value not yet mapped`` on the next
        # downstream op.
        #
        # H4 Wave-I refactor: dispatch via the per-emitter ``owns_regions``
        # attribute (``_emitter_owns_regions``). Fall back to the legacy
        # set for emitters that haven't been migrated yet.
        if _emitter_owns_regions(op_name_str) or op_name_str in OPS_THAT_HANDLE_OWN_REGIONS:
            return
        # Recurse into regions/blocks.
        for region in getattr(op, "regions", ()) or ():
            for block in getattr(region, "blocks", ()) or ():
                for child in getattr(block, "operations", ()) or ():
                    _recurse(child)

    _recurse(body)
    return visited


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
    # textual MLIR string (depending on Triton version). Delegate string
    # TTIR to ``from_ttir`` so the same PtrAnalysis + generic-form MLIR
    # parser path handles direct and harness callers. Fall back to the
    # explicit text-walker only when that real path is unavailable.
    if isinstance(ttir_module, str):
        try:
            return from_ttir(
                ttir_module,
                target=target,
                name=getattr(fn, "__name__", "main"),
                grid=grid,
            )
        except TypeError as exc:
            if "textual TTIR is no longer the default path" not in str(exc):
                raise
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
                grid=grid,
                _allow_text_ttir=True,
            )
    return from_ttir(
        ttir_module,
        target=target,
        name=getattr(fn, "__name__", "main"),
        grid=grid,
    )


def _read_ttir_warp_config(
    ttir_module: Any,
) -> Tuple[Optional[int], Optional[int]]:
    """Read ``(num_warps, num_stages)`` from a TTIR module's MLIR attrs.

    Triton stamps the (autotuned) compile config onto the module operation
    as integer attributes once the ``num_warps`` / ``num_stages`` from the
    selected ``triton.Config`` are bound. The exact attribute key differs by
    Triton version / dialect (``ttg.num-warps`` after the TritonGPU
    conversion, plain ``num-warps`` / ``num_warps`` on the raw TTIR module),
    so we probe the known spellings. Returns ``(None, None)`` when the module
    is a text-TTIR stand-in or carries no warp attrs (callers keep their
    defaults / explicit overrides).

    RULE #1 (fail loud): a warp attr that is present but cannot be parsed to
    a positive int RAISES -- we never silently treat a malformed autotune
    annotation as "absent" and fall back to the 4-warp default.
    """
    # Text-TTIR / non-MLIR stand-ins have no queryable operation attrs.
    op = getattr(ttir_module, "operation", None)
    if op is None:
        return None, None
    attrs = getattr(op, "attributes", None)
    if attrs is None:
        return None, None

    def _lookup(keys: Tuple[str, ...]) -> Optional[int]:
        for key in keys:
            try:
                raw = attrs[key]  # mlir NamedAttribute access
            except (KeyError, TypeError, IndexError):
                continue
            # mlir IntegerAttr exposes ``.value``; fall back to str() parse.
            val = getattr(raw, "value", raw)
            try:
                ival = int(val)
            except (TypeError, ValueError):
                try:
                    ival = int(str(val).strip().split()[0])
                except (ValueError, IndexError):
                    raise ValueError(
                        "TTIR module carries warp-config attr %r=%r that is "
                        "not a parseable integer; refusing to silently fall "
                        "back to the default warp count (RULE #1)."
                        % (key, val)
                    )
            if ival <= 0:
                raise ValueError(
                    "TTIR module warp-config attr %r=%d is non-positive; a "
                    "valid autotune config must be > 0." % (key, ival)
                )
            return ival
        return None

    num_warps = _lookup(("ttg.num-warps", "num-warps", "num_warps"))
    num_stages = _lookup(("ttg.num-stages", "num-stages", "num_stages"))
    return num_warps, num_stages


def autotune_winning_block_config(
    autotuned_kernel: Any,
    *,
    device_shared_cap: Optional[int] = None,
    target_block: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """Return the autotune-WINNING tile/launch config for a Triton kernel.

    Capture-side sibling of :func:`_read_ttir_warp_config`. Where the TTIR
    reader recovers ``num_warps`` / ``num_stages`` from the *already-captured*
    module attributes, this reads the kernel's autotune ``Config`` list BEFORE
    capture so the harness can supply the WINNING ``BLOCK_SIZE_*`` constexprs at
    TTIR-capture time (the block dims live in the ``tt.dot`` operand shapes, not
    as module attrs, so they must be pinned at capture). The winning config is
    the first entry of ``Autotuner.configs`` -- the one Triton's autotuner
    selects for the kernel's shape (verified for ``_chunk_scan_bwd_dstates`` to
    equal the cached ``.json`` ``num_warps`` / ``num_stages``).

    GENERIC + backend-agnostic: any ``@triton.autotune`` kernel is honoured by
    reading its own declared config list; no per-kernel hardcoded block size.
    The returned ``BLOCK_SIZE_*`` flow into the captured TTIR's GEMM tile, which
    on CUDA drives ``T.gemm``'s warp partition and on Metal the threadgroup
    partition alike.

    Returns a dict ``{<block kwargs...>, "num_warps": int, "num_stages": int}``.

    RULE #1 (fail loud): if the object carries no ``configs`` list we RAISE
    rather than silently returning an empty/default block config -- a caller
    that asked for the autotune-winning tile must get a real one or a clear
    error, never a silent fall-back to the smallest tile.
    """
    configs = getattr(autotuned_kernel, "configs", None)
    if not configs:
        raise ValueError(
            "autotune_winning_block_config: object %r carries no non-empty "
            "'configs' list; cannot recover the autotune-winning BLOCK_SIZE. "
            "Pass a @triton.autotune-wrapped kernel (the Autotuner), not the "
            "bare JITFunction. Refusing to silently default to the smallest "
            "tile (RULE #1)." % (getattr(autotuned_kernel, "__name__", autotuned_kernel),)
        )

    # SURVIVING-CONFIG SELECTION (RULE #1 -- MEASURED: configs[0] is a PHANTOM).
    # Triton lists configs[0] as the nominal best, but its dynamic
    # ``metadata.shared`` can EXCEED the device shared cap (e.g. the dstates
    # 128x256x64 nw8 config needs ~197 KB > the 101376 B opt-in cap on GB10):
    # Triton's autotuner PRUNES it at runtime (catches the OutOfResources during
    # the bench) and the REAL winner is the largest tile that FITS. Picking
    # configs[0] blindly imports the pruned phantom (wrong nw8/ns3, and a tile
    # the device cannot even launch). We mirror Triton's pruning: estimate each
    # config's GEMM shared footprint and keep only configs that FIT the cap, then
    # -- among survivors -- pick the LARGEST tile (BM*BN), the autotuner's
    # preference (max data reuse) when it fits. For the dstates kernel this
    # yields 64x64x32 nw2 ns4 (the MEASURED native surviving winner), NOT the
    # pruned 128x256x64 nw8. GENERIC: any @triton.autotune kernel is pruned by
    # its own declared block sizes against the real device cap; no per-kernel
    # hardcode. RULE #1: if NO config fits we RAISE (never launch an OOM tile).
    import os as _os_cfg

    def _shared_bytes(kw, ns):
        bm = kw.get("BLOCK_SIZE_M")
        bn = kw.get("BLOCK_SIZE_N")
        bk = kw.get("BLOCK_SIZE_K")
        if bm is None or bn is None or bk is None:
            return None
        # Classic Triton GEMM double-operand staging: (A[BM,BK] + B[BK,BN]) per
        # pipeline stage, fp32 (4 B; the SSD/dstates operands are fp32 -- the
        # conservative upper bound so we never UNDER-count and admit an OOM tile).
        per_stage = (int(bm) * int(bk) + int(bk) * int(bn))
        return per_stage * 4 * max(1, int(ns))

    # Device shared cap: explicit kwarg wins; else env (TL_SHARED_CAP_BYTES);
    # else the GB10 / sm_121 100 KB opt-in cap (101376 B). Backend-agnostic: a
    # Metal/other target supplies its own cap via the kwarg.
    cap = device_shared_cap
    if cap is None:
        _env_cap = _os_cfg.environ.get("TL_SHARED_CAP_BYTES")
        cap = int(_env_cap) if _env_cap else 101376

    fitting = []
    for c in configs:
        kw = dict(getattr(c, "kwargs", {}) or {})
        sb = _shared_bytes(kw, int(getattr(c, "num_stages", 2)))
        # sb is None => non-GEMM-shaped config we cannot estimate; admit it
        # rather than wrongly prune a config we do not understand (RULE #1).
        if sb is None or sb <= cap:
            fitting.append((c, kw))
    if not fitting:
        raise ValueError(
            "autotune_winning_block_config: NO config fits the device shared "
            "cap (%d B); every declared tile would OutOfResources. Refusing to "
            "launch an OOM tile (RULE #1). configs=%r"
            % (cap, [dict(getattr(c, "kwargs", {}) or {}) for c in configs])
        )

    def _tile_area(kw):
        return int(kw.get("BLOCK_SIZE_M") or 0) * int(kw.get("BLOCK_SIZE_N") or 0)

    # SELECTION among the survivors:
    #  (a) if the caller is compiling a SPECIFIC tile (``target_block``, the
    #      BLOCK_SIZE_* the captured TTIR baked into the GEMM operand shapes),
    #      return the surviving config that MATCHES that tile -- so nw/ns are the
    #      ones Triton autotuned FOR THAT TILE (for the dstates 64x64x32 capture
    #      this is the MEASURED native winner nw2/ns4). This is honest: we do not
    #      guess the autotuner bench, we match the tile we actually compile.
    #  (b) else fall back to the LARGEST fitting tile (the autotuner's preference
    #      when no specific tile is pinned).
    # RULE #1: if a target_block is given but NO surviving config matches it we
    # RAISE (the captured tile is not a launchable autotune config) rather than
    # silently returning a mismatched nw/ns.
    best = kwargs = None
    if target_block:
        want = {k: int(v) for k, v in target_block.items() if v is not None}
        for c, kw in fitting:
            if all(int(kw.get(k, -1)) == v for k, v in want.items()):
                best, kwargs = c, kw
                break
        if best is None:
            raise ValueError(
                "autotune_winning_block_config: target_block %r matches no "
                "surviving config that fits the %d B shared cap; refusing to "
                "return a mismatched nw/ns (RULE #1). fitting=%r"
                % (want, cap, [kw for _c, kw in fitting])
            )
    if best is None:
        best, kwargs = max(fitting, key=lambda ck: _tile_area(ck[1]))
    if not kwargs:
        raise ValueError(
            "autotune_winning_block_config: surviving Config has empty kwargs; "
            "no BLOCK_SIZE_* to honour (RULE #1)."
        )
    out: Dict[str, Any] = dict(kwargs)
    out["num_warps"] = int(getattr(best, "num_warps", 4))
    out["num_stages"] = int(getattr(best, "num_stages", 2))
    return out


def from_ttir(
    ttir_module: Any,
    *,
    target: Optional[str] = None,
    name: str = "main",
    grid: Optional[Tuple[int, ...]] = None,
    arg_buffer_shapes: Optional[Any] = None,
    num_warps: Optional[int] = None,
    num_stages: Optional[int] = None,
    min_blocks_per_sm: Optional[int] = None,
    prologue_opt: bool = True,
    async_loads: bool = True,
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
    global _FALLBACK_WARNED
    # Re-probe MLIR bindings: by the time ``from_ttir`` is invoked the
    # caller has typically loaded Triton (to obtain TTIR), so the jaxlib
    # alias path that was deferred at package import time is now safe.
    # ``probe_and_wire_mlir`` is idempotent; the second call is a cheap
    # ``mlir.ir in sys.modules`` check when the first probe already won.
    try:
        _mlir_path_setup.probe_and_wire_mlir()
    except Exception:
        pass
    ctx = WalkerCtx()
    # Thread the requested codegen target onto the ctx so target-sensitive
    # emitters (e.g. tt.dot accumulator scope: shared-C on Metal vs
    # local.fragment on CUDA) can pick the right lowering at emission time.
    # Emission happens BEFORE the tilelang lowering passes set
    # ``tvm.target.Target.current()``, so the emitter cannot rely on the
    # ambient target -- it must read ``ctx.target``.
    if target is not None:
        ctx.target = str(target)
    # Plumb optional ``num_warps`` / ``num_stages`` overrides supplied by
    # the harness (which captures them from Triton's compile options) so
    # ``map_tt_func`` can stamp the right ``threadIdx.x`` extent and
    # PrimFunc attrs. Falsy values keep the WalkerCtx defaults intact.
    #
    # GENERIC AUTOTUNE-HONOURING (TileFix): when the caller does NOT pass an
    # explicit ``num_warps`` / ``num_stages``, read them from the TTIR
    # MODULE attributes that Triton stamps after applying the kernel's
    # (autotuned) compile config -- ``"ttg.num-warps"`` / ``"num-warps"`` and
    # ``"ttg.num-stages"`` / ``"num-stages"``. This makes the cooperative
    # ``T.gemm`` tile over the SAME warp count Triton autotuned to (e.g.
    # num_warps=8 -> 256-thread block -> 8-way warp partition -> 256 HMMA on
    # the dstates dot) instead of silently defaulting to 4 warps and tiling
    # the single ``tt.dot`` 8x smaller. This is BACKEND-AGNOSTIC (the warp
    # count flows into ``gemm.lower``'s ``computeWarpPartition`` on CUDA and
    # the threadgroup partition on Metal alike) and NOT a per-kernel hack:
    # any TTIR carrying these standard module attrs is honoured. RULE #1: if
    # the module carries a malformed warp attr we RAISE rather than silently
    # falling back to the default. Explicit kwargs always win.
    if num_warps is None or num_stages is None:
        attr_nw, attr_ns = _read_ttir_warp_config(ttir_module)
        if num_warps is None and attr_nw is not None:
            num_warps = attr_nw
        if num_stages is None and attr_ns is not None:
            num_stages = attr_ns
    if num_warps is not None:
        ctx.num_warps = int(num_warps)
    if num_stages is not None:
        ctx.num_stages = int(num_stages)
    # OCCUPANCY hint: minBlocksPerMultiprocessor (the 2nd ``__launch_bounds__``
    # arg). The register/occupancy-bound chunk-scan-bwd-dstates kernel has a
    # register high-water (124 ptxas regs at the default ``.minnctapersm 1``)
    # that ptxas can pack into a higher-occupancy budget when given an explicit
    # min-blocks hint: at min_blocks=5 the kernel re-allocates to 95 regs (0
    # spill), crossing the >=5 CTAs/SM occupancy step on sm_121a (GB10: 65536
    # regs/SM, 17408B smem/CTA both admit 5 CTAs at <=96 regs) -- ncu-measured
    # registers 128->95, occupancy_limit_registers 4->5, warps_active
    # 32.99%->40.91%, routed ms 3.46->3.34 (-3.7%). MEASURED MECHANISM: the
    # emitted AttrStmt is NOT a byte-identical no-op -- it acts as a fusion
    # barrier so the addressing index-compute lowers to explicit strength-reduced
    # int32/int64 arrays (a structurally different, lower-register kernel; SASS
    # 3232->3240 insns, shifted opcode mix), which is WHY ptxas needs fewer regs.
    # RULE #1 safety comes from DIRECT A/B parity, not a byte-equal assumption:
    # OUTPUT is bit-exact vs the native mamba_ssm reference across single-tile,
    # all 5 partial-mask shapes, and int64 at 2.82GB + 4.32GB (MAXDIFF=0.0 every
    # case). Name-gated + grounded: applied ONLY to the dstates kernel whose
    # 128->95 packing and 5-CTA step were directly measured (ncu, cache OFF,
    # prod P1). An explicit
    # ``min_blocks_per_sm`` kwarg always wins and is honored for any kernel; a
    # caller passing 0 / a falsy value disables the hint entirely.
    if min_blocks_per_sm is None:
        if isinstance(name, str) and "chunk_scan_bwd_dstates" in name:
            min_blocks_per_sm = 5
    if min_blocks_per_sm is not None:
        ctx.min_blocks_per_sm = int(min_blocks_per_sm)
    # PROLOGUE-OPT gate (transforms 1/2/3). The routed-triton path opts in by
    # default; pass ``prologue_opt=False`` to reproduce the pre-opt serial
    # prologue for A/B comparison. This ONLY affects the elementwise prologue
    # (address/mask arrays) -- the cooperative T.copy/T.gemm half is untouched.
    ctx.routed_triton_prologue_opt = bool(prologue_opt)
    # Transform (2) thread-distribution sub-gate. OFF by default: the
    # materializer can promote a distributed tile to shared scope + a
    # __syncthreads barrier (race-free), but promoting ALL prologue addressing
    # /mask tiles to shared EXCEEDS the per-block shared-memory budget (the
    # routed dstates prologue alone holds dozens of [2048]/[4096] arrays).
    # The correct fix is full transform (1): FOLD the addressing into the
    # T.copy region indices/predicate so these tiles are never materialized in
    # ANY scope -- then the small residue can be thread-distributed in shared.
    # Until address-folding lands, distribution is gated off to avoid both a
    # racy local write (RULE #1) and a shared-budget overflow. The machinery is
    # exercised by ``thread_distribute=True`` for development. Transforms (1
    # partial: guard/and folding) + (3) ship ON under ``prologue_opt``.
    ctx.routed_triton_thread_distribute = bool(kwargs.get("thread_distribute", False))
    # ITERATION 3 (coalesced async loads). ON by default on the routed-triton
    # path: the ``scf.for`` K-loop emitter stamps ``num_stages`` on a serial
    # K-loop carrying global->shared cooperative copies so PipelinePlanning +
    # InjectSoftwarePipeline + LowerPTXAsyncCopy route those copies through
    # ``cp.async`` (SASS LDGSTS) instead of plain per-lane LDG. Pass
    # ``async_loads=False`` to reproduce the pre-iteration-3 plain-LDG K-loop
    # for A/B comparison. Gated on the prologue-opt path so the GEMM
    # cooperative path and dict-shaped unit tests stay byte-identical.
    ctx.routed_triton_async_loads = bool(async_loads) and bool(prologue_opt)
    # ITERATION 5 (DoutTranspose / coalesced strided load). GROUND-TRUTH route
    # hint: map a producer-load SOURCE pointer (TTIR ``%argN``) to the LOGICAL
    # TILE AXIS that is CONTIGUOUS (global stride == 1) on the REAL tensor, so
    # the rank-2 global->shared load emitter (``_emit_ptrstate_tile_load_tir``)
    # iterates that axis INNERMOST -> coalesced LDG instead of a strided scalar
    # load. This is NOT fabricated contiguity: it reflects the row-major Mamba
    # tensor layouts (verified by the measured global strides -- e.g. the
    # dstates ``dout`` tile ``[hd, k]`` reads ``%arg0`` whose ``hd`` axis is
    # stride-1 (tile axis 0) while the innermost ``k`` (seq) axis is stride
    # ``nheads*headdim`` = 7168). We reorder ONLY the load TRAVERSAL order; the
    # logical tile + the downstream GEMM operand are byte-unchanged (RULE #1:
    # parity stays bit-correct; a non-contiguous axis is never marked
    # contiguous). Caller may override via the ``contiguous_tile_axis`` kwarg
    # (dict {source_ssa: axis}); default targets the dstates dout producer.
    _contig_hint = kwargs.get("contiguous_tile_axis", None)
    if (
        _contig_hint is None
        and ctx.routed_triton_async_loads
        and isinstance(name, str)
        and "chunk_scan_bwd_dstates" in name
    ):
        # dout (%arg0) tile [hd, k]: contiguous (stride-1) axis is hd == tile
        # axis 0. C (%arg1) tile [k, ds]: contiguous axis is ds == innermost
        # (no reorder needed). Only the non-innermost-contiguous case is listed.
        # GATED to the dstates kernel by name so no other routed kernel's
        # producer loads are reordered (RULE #1: targeted, grounded, no blanket
        # %arg0 reorder that could de-coalesce an unrelated row-major load).
        _contig_hint = {"%arg0": 0}
    if isinstance(_contig_hint, dict):
        ctx.routed_contiguous_tile_axis = {
            str(k_): int(v_) for k_, v_ in _contig_hint.items()
        }
    # DOUTTRANSPOSE16B: initialise the physically-transposed producer-tile set
    # on the ROOT ctx so the K-loop load emitter and its in-region consumers
    # share ONE object (propagated by reference in map_scf_for). The set stays
    # empty unless the memory.py emitter registers a transposed tile (default-on
    # for the route-verified non-innermost-contiguous dstates dout source;
    # TL_NO_DOUT_TRANSPOSE_TILE=1 disables it).
    if getattr(ctx, "transposed_phys_tiles", None) is None:
        ctx.transposed_phys_tiles = set()
    # ITERATION 6 (C-tile executed TMA). GROUND-TRUTH route hint: the set of
    # producer-load SOURCE pointers (TTIR ``%argN``) whose INNERMOST tile axis
    # is PROVABLY CONTIGUOUS (global stride == 1) on the REAL tensor. For such a
    # source the de-monomorphized kernel passes the innermost stride as an
    # opaque symbolic arg (the analyzer cannot prove ``== 1`` even though it IS
    # 1 at runtime), so ``copy.cc:988 ICHECK(is_one(desc.global_stride[0]))``
    # FATALs and the C-tile T.copy cannot lower to a real TMA load. Here the
    # route GROUNDS the innermost stride to a LITERAL ``IntImm(1)`` before
    # ``LowerBulk`` -- ONLY for the sources listed here, which the route has
    # VERIFIED contiguous on the real layout. RULE #1: NEVER ground a
    # non-contiguous stride; this is grounded ground-truth, not fabrication, and
    # is gated per-source (the dout tile, whose innermost K-reduction axis has
    # stride ``nheads*headdim`` != 1, is DELIBERATELY excluded). Caller may
    # override via the ``contiguous_innermost_sources`` kwarg (iterable of
    # source SSA strings); default targets the dstates C (%arg1) producer whose
    # ``[k, ds]`` tile has ds innermost with stride == 1.
    _ground_srcs = kwargs.get("contiguous_innermost_sources", None)
    if (
        _ground_srcs is None
        and ctx.routed_triton_async_loads
        and isinstance(name, str)
        and "chunk_scan_bwd_dstates" in name
    ):
        # C (%arg1) tile [k, ds]: innermost axis ds has global stride == 1 on
        # the §P1 dstates layout (measured row-major). The dout (%arg0) tile is
        # NOT listed: its innermost (k/seq) axis is genuinely non-contiguous.
        _ground_srcs = {"%arg1"}
    if _ground_srcs is not None:
        ctx.routed_contiguous_innermost_sources = {
            str(s_) for s_ in _ground_srcs
        }
    if grid is not None:
        ctx.launch_grid = tuple(int(x) for x in grid)
    if arg_buffer_shapes is not None:
        if isinstance(arg_buffer_shapes, dict):
            ctx.arg_buffer_shapes = {
                key: tuple(int(dim) for dim in shape)
                for key, shape in arg_buffer_shapes.items()
            }
        else:
            ctx.arg_buffer_shapes = {
                idx: tuple(int(dim) for dim in shape)
                for idx, shape in enumerate(arg_buffer_shapes)
                if shape is not None
            }
    ctx.kernel_name = name  # Pass name so map_tt_func can use it
    if isinstance(ttir_module, str):
        # Preferred path: re-parse via mlir.ir and use the MLIR walker
        # (populates ctx.value_map / ctx.buffers properly). If
        # mlir.ir bindings aren't available we degrade to the regex
        # walker, but only with explicit opt-in (_allow_text_ttir) and
        # a one-shot UserWarning.
        # Re-probe MLIR availability at call-time (rather than relying on
        # the module-load snapshot ``MLIR_WALKER_AVAILABLE``) so the
        # jaxlib alias wired by ``probe_and_wire_mlir()`` above is picked
        # up even on the first invocation.
        ttir_text_for_parse = ttir_module
        try:
            from .pipeline import (  # noqa: WPS433
                _libtriton_loaded,
                run_ptr_analysis_pre_pass,
                run_ptr_analysis_pre_pass_subprocess,
                seed_ptr_states,
            )

            if _libtriton_loaded():
                ttir_text_for_parse, state_map = run_ptr_analysis_pre_pass_subprocess(
                    ttir_module
                )
            else:
                ttir_text_for_parse, state_map = run_ptr_analysis_pre_pass(ttir_module)
            seed_ptr_states(ctx, state_map)
        except Exception as exc:
            warnings.warn(
                "triton_frontend: PtrAnalysis pre-pass for textual TTIR failed; "
                "continuing with MLIR walker without pointer-state metadata. "
                f"cause={exc!r}",
                RuntimeWarning,
                stacklevel=2,
            )

        parsed = parse_ttir(ttir_text_for_parse)
        if parsed is not None:
            # FULL TRANSFORM 1 (Coalesce-style addressing fold) pre-pass:
            # compute the set of SSA results consumed ONLY as addressing/mask
            # so the feeders keep them lazy instead of materializing spilled
            # arrays. Only meaningful under the routed prologue-opt gate; the
            # set is empty (no-op) otherwise. Failure here must not silently
            # disable correctness -- but an empty set merely reverts to the
            # pre-fold serial materialization, so a best-effort guard keeps the
            # walker robust to provider drift while never producing wrong IR.
            if ctx.routed_triton_prologue_opt:
                try:
                    ctx.fold_addressing_ssa = build_addressing_fold_set(parsed)
                except Exception as exc:  # provider use-map drift
                    raise RuntimeError(
                        "FULL TRANSFORM 1: build_addressing_fold_set failed; "
                        "refusing to silently fall back to the spilled-array "
                        "prologue under the routed prologue-opt gate (RULE #1). "
                        f"cause={exc!r}"
                    ) from exc
            _walk_mlir_module(parsed, ctx)
            return _frame_register_mma_fragments(ctx)
        if not _allow_text_ttir:
            raise TypeError(
                "from_ttir: textual TTIR is no longer the default path; "
                "pass an mlir.ir.Module (recommended) or set "
                "_allow_text_ttir=True for the coverage-only text walker."
            )
        if not _FALLBACK_WARNED:
            _FALLBACK_WARNED = True
            warnings.warn(_DEGRADED_WARNING_MESSAGE, UserWarning, stacklevel=2)
        # RULE #1 (no silent fallback): the text walker is coverage-only --
        # it confirms every op is in OP_TABLE but DOES NOT populate
        # ctx.value_map / ctx.buffers and therefore cannot build a real
        # PrimFunc body. Previously this path returned a PrimFunc whose body
        # was ``T.evaluate(0)`` (an empty, do-nothing kernel) which
        # ``tilelang.compile`` accepted WITHOUT raising -- a silent fallback
        # that ships zeros instead of the kernel. We now run the coverage
        # walk (so an unsupported op still raises NotImplementedError with a
        # precise op name) and then RAISE LOUDLY rather than emit a stub.
        #
        # The cure is to make ``parse_ttir`` succeed (a real mlir.ir module
        # provider on this host) so the MLIR walker above populates a real
        # body; this branch only fires when NO mlir.ir provider can parse the
        # TTIR. It must never produce a runnable kernel.
        visited_ops = _walk_text_ttir(ttir_module, ctx)
        raise RuntimeError(
            "triton_frontend.from_ttir: no mlir.ir provider could parse this "
            "TTIR (custom-form tt.* dialect) on this host, so only the "
            "coverage-only text walker ran. The text walker confirmed all "
            f"{len(visited_ops)} ops are in OP_TABLE but CANNOT build a real "
            "PrimFunc body (it does not populate value_map/buffers). Refusing "
            "to return an empty `T.evaluate(0)` stub PrimFunc (that would be a "
            "silent do-nothing kernel). To route this kernel: provide an "
            "mlir.ir binding that parses Triton custom-form TTIR -- either a "
            "Triton-aware MLIR build, or build the PtrAnalysis/to_generic C++ "
            "shim (`python -m poc.triton_frontend.build_cxx --build`) so "
            "custom-form TTIR is converted to generic form that jaxlib's "
            "mlir.ir can parse. See parse_ttir() for the provider probe order."
        )
    else:
        # Pre-pass: run microsoft/triton-shared PtrAnalysis to rewrite
        # tt.* pointer arithmetic into ``tts.make_tptr`` ops and seed
        # ctx.value_map with the recovered ``(buffer, indices)`` tuples.
        # Skipped silently when the C++ shim is unavailable so the walker
        # falls back to the MVP scalar path (op_mapping seeds placeholder
        # buffers in that case).
        try:
            from .pipeline import _libtriton_loaded as _triton_native_loaded  # noqa: WPS433
        except Exception:
            _triton_native_loaded = lambda: False  # type: ignore[assignment]

        # Subprocess path: only requires shim_subprocess_available() because
        # the actual rewrite runs in a clean interpreter where libtriton is
        # NOT loaded. shim_available() would (correctly) return False whenever
        # libtriton is present in the parent, which used to disable BOTH
        # branches here and force every Path D lowering through the degraded
        # scalar walker even when a perfectly safe subprocess was possible.
        from .ptr_analysis import shim_subprocess_available  # noqa: WPS433
        if shim_subprocess_available() and _triton_native_loaded():
            try:
                from .pipeline import (  # noqa: WPS433
                    run_ptr_analysis_pre_pass_subprocess,
                    seed_ptr_states,
                )

                _rewritten, state_map = run_ptr_analysis_pre_pass_subprocess(
                    str(ttir_module)
                )
                seed_ptr_states(ctx, state_map)
            except Exception as exc:  # pragma: no cover -- shim build issues
                warnings.warn(
                    "triton_frontend: isolated PtrAnalysis pre-pass failed; "
                    "falling back to MVP scalar path. "
                    f"cause={exc!r}",
                    RuntimeWarning,
                    stacklevel=2,
                )
        elif shim_available():
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
    return _frame_register_mma_fragments(ctx)
