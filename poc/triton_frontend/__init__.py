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
    parse_ttir,
    try_import_mlir,
    walk_module,
)
from .op_mapping import OP_TABLE, WalkerCtx
from .ptr_analysis import PtrAnalysis, shim_available

__all__ = [
    "from_triton_kernel",
    "from_ttir",
    "TileLangPrimFunc",
    "MLIR_WALKER_AVAILABLE",
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
    BufferLoad = getattr(tvm_mod.tir, "BufferLoad", None)

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
        if isinstance(entry, tvm_mod.tir.Buffer):
            # A rank-N offset tile (matmul's C pointer tile, and similar
            # block-pointer paths) has already materialised the flat address
            # expression into a local buffer, so we cannot inspect the
            # original pid/stride expression here. The sound conservative
            # bound is the per-program tile footprint multiplied by every
            # launch-grid extent. This is intentionally an upper bound:
            # over-declaring a PrimFunc parameter shape only weakens
            # LegalizeSafeMemoryAccess guards, while under-declaring drops
            # valid stores outside the first tile.
            extent_expr = tir.const(int(flat_tile_extent), "int32")
            for _var, _axis, extent in program_id_vars:
                extent_expr = extent_expr * extent
            return [extent_expr]
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


def _redecl_input_buffer(
    ctx: Any,
    buf: Any,
    shape: Any,
    dtype: str,
    *,
    offset_indices: Any = None,
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

    try:
        new_buf = tir.decl_buffer(shape=decl_shape, dtype=dtype, name=name)
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
    # Use ``scope="shared"`` so the tile buffer satisfies the scope contract
    # of downstream GEMM consumers: Metal GEMM's ``is_gemm_ss()`` check
    # requires both A and B operand tiles in shared scope; the default
    # ``scope="local"`` produces ``"Unsupported gemm combination, A: local,
    # B: local"`` at ``LowerTileOp`` time.
    tile_buf = _alloc_tile_buffer(ctx, list(out_shape) or [1], out_dtype, out_buf_name, scope="shared")

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
        and isinstance(offset_indices[0], tvm_mod.tir.Buffer)
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
        src_indices.append(tir.BufferLoad(single_offset_buf, list(loop_vars)))
    else:
        for axis, lv in enumerate(loop_vars):
            if axis < len(offset_indices):
                base = offset_indices[axis]
            else:
                base = tir.const(0, "int32")
            if isinstance(base, tvm_mod.tir.Buffer):
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

    scope_fn = getattr(val_expr, "scope", None)
    try:
        val_scope = scope_fn() if callable(scope_fn) else scope_fn
    except Exception:
        val_scope = None
    if val_scope in {"local.fragment", "metal.simdgroup"}:
        import tilelang.language as T  # type: ignore

        handle = T.copy(val_expr, dst_buf)
        if isinstance(handle, tvm_mod.tir.PrimExpr):
            stmt = tir.Evaluate(handle)
            ctx.emit(stmt)
            return stmt
        if handle is not None:
            ctx.emit(handle)
        return handle

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
        and isinstance(offset_indices[0], tvm_mod.tir.Buffer)
        and len(offset_indices[0].shape) == len(loop_vars)
        and len(loop_vars) >= 2
    ):
        single_offset_buf = offset_indices[0]

    dst_indices: List[Any] = []
    if single_offset_buf is not None:
        if len(getattr(dst_buf, "shape", []) or []) != 1:
            flat_extent = 1
            for _e in val_shape:
                flat_extent *= int(_e)
            dst_dtype = str(getattr(dst_buf, "dtype", "float32"))
            dst_buf = _redecl_input_buffer(ctx, dst_buf, [flat_extent], dst_dtype)
        dst_indices.append(tir.BufferLoad(single_offset_buf, list(loop_vars)))
    else:
        for axis, lv in enumerate(loop_vars):
            if axis < len(offset_indices):
                base = offset_indices[axis]
            else:
                base = tir.const(0, "int32")
            if isinstance(base, tvm_mod.tir.Buffer):
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
    )

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


def _walk_mlir_module(
    module: Any, ctx: Optional[WalkerCtx] = None
) -> List[str]:
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

    # Lazy import to avoid a top-of-file cycle with mlir_walker. We use the
    # per-emitter ``owns_regions`` attribute (H4 Wave-I) but keep the legacy
    # set as a fallback for emitters that forgot to set the attribute.
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
    num_warps: Optional[int] = None,
    num_stages: Optional[int] = None,
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
    # Plumb optional ``num_warps`` / ``num_stages`` overrides supplied by
    # the harness (which captures them from Triton's compile options) so
    # ``map_tt_func`` can stamp the right ``threadIdx.x`` extent and
    # PrimFunc attrs. Falsy values keep the WalkerCtx defaults intact.
    if num_warps is not None:
        ctx.num_warps = int(num_warps)
    if num_stages is not None:
        ctx.num_stages = int(num_stages)
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
            _walk_mlir_module(parsed, ctx)
            return getattr(ctx, "prim_func", None)
        if not _allow_text_ttir:
            raise TypeError(
                "from_ttir: textual TTIR is no longer the default path; "
                "pass an mlir.ir.Module (recommended) or set "
                "_allow_text_ttir=True for the coverage-only text walker."
            )
        if not _FALLBACK_WARNED:
            _FALLBACK_WARNED = True
            warnings.warn(_DEGRADED_WARNING_MESSAGE, UserWarning, stacklevel=2)
        _walk_text_ttir(ttir_module, ctx)
        # Dummy fallback for coverage-only walker
        import tvm
        from tvm import tir
        func = tir.PrimFunc(params=[], body=tir.Evaluate(0))
        func = func.with_attr("global_symbol", name)
        func = func.with_attr("tir.noalias", True)
        func = func.with_attr("num_warps", num_warps if num_warps is not None else 4)
        func = func.with_attr("num_stages", num_stages if num_stages is not None else 2)
        return func
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

        if shim_available() and _triton_native_loaded():
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
    return getattr(ctx, "prim_func", None)
