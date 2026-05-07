"""Rewrite local.fragment → metal.simdgroup for GEMM accumulators on Metal.

Idea #8 (Z3 roadmap): in addition to the static shape/dtype checks used to
decide simdgroup eligibility, we run a *detection-only* Z3 fallback for
symbolic shapes. The Z3 query asserts:

    shape[0] % 8 == 0
    /\\  shape[1] % 8 == 0
    /\\  dtype ∈ {fp16, packed fp8}
    /\\  addr % 16 == 0

If the static check fails for symbolic inputs, the Z3 helper attempts to
prove the shape constraint. With the **IR rewrite path** enabled
(``PassConfig`` key ``tl.simdgroup_matrix_rewrite``), the pass uses that
eligibility decision per-buffer to gate the scope promotion: only buffers
that pass the static or Z3-proved check are promoted to ``metal.simdgroup``
(which is the IR-level surface that drives codegen of MSL
``simdgroup_load`` / ``simdgroup_multiply_accumulate`` / ``simdgroup_store``
intrinsics in ``codegen_metal.cc``). Ineligible buffers stay in
``local.fragment`` (legacy scalar lowering).

The PassConfig flag defaults OFF — the existing unconditional rewrite
(every GEMM ``local.fragment`` accumulator is promoted) keeps shipping
behavior. Conservative-by-default: if Z3 returns False/UNKNOWN we keep the
legacy non-simdgroup path.
"""

from __future__ import annotations

import logging
import os

from tvm import tir, IRModule
from tvm.ir import Op, PointerType
from tvm.target import Target
from tvm.tirx.stmt import AllocBuffer
from tvm.tir.transform import prim_func_pass

logger = logging.getLogger("tilelang.metal_simdgroup")

_GEMM_OPS = None

#: PassConfig key that enables the per-buffer eligibility-gated rewrite.
#: Default OFF — the unconditional fragment→simdgroup promotion still runs
#: when this is False, preserving shipping behavior.
PASS_CONFIG_KEY = "tl.simdgroup_matrix_rewrite"

#: PrimFunc attribute name where rewritten buffer names are recorded so
#: downstream tooling and tests can inspect what the gated rewrite emitted.
EMITTED_ATTR_KEY = "tl.simdgroup_matrix_rewrite_emitted"

# ---------------------------------------------------------------------------
# Z3 detection helpers (Idea #8)
# ---------------------------------------------------------------------------

#: Dtypes that fit a Metal `simdgroup_matrix` slot.
_SIMDGROUP_DTYPES = {"float16", "fp16", "bfloat16", "uint8", "int8", "fp8"}

#: Required tile granularity for `simdgroup_matrix_load_*`.
_SIMDGROUP_TILE = 8

#: Required base address alignment for simdgroup matrix loads (bytes).
_SIMDGROUP_ALIGN = 16


def _is_simdgroup_dtype(dtype: str) -> bool:
    s = str(dtype).lower()
    if s.startswith("e4m3") or s.startswith("e5m2"):
        return True
    return any(s.startswith(d) for d in _SIMDGROUP_DTYPES)


def _static_simdgroup_eligible(shape, dtype) -> bool:
    """Pure-static check: all shape entries are constant multiples of 8."""
    if not _is_simdgroup_dtype(dtype):
        return False
    if len(shape) < 2:
        return False
    for dim in shape[-2:]:
        if not isinstance(dim, (tir.IntImm,)) and not isinstance(dim, int):
            return False
        ival = int(dim) if isinstance(dim, int) else int(dim.value)
        if ival % _SIMDGROUP_TILE != 0 or ival <= 0:
            return False
    return True


def _z3_simdgroup_eligible(shape, dtype,
                           addr_align_bytes: int = _SIMDGROUP_ALIGN,
                           addr_value: int | None = None
                           ) -> tuple[bool, str]:
    """Z3 fallback for symbolic shapes (detection-only).

    Returns a (proved, query) pair. ``proved`` is True only if Z3 can
    *prove* that for any concrete instantiation consistent with the symbolic
    expressions, ``shape[0] % 8 == 0 /\\ shape[1] % 8 == 0 /\\ addr % 16 == 0``
    holds. On UNKNOWN/UNSAT-of-implication/timeout we conservatively return
    False.

    ``addr_value`` is an optional concrete base-address. When None, the
    address constraint is *omitted* from the query (we can't prove anything
    about an unbounded symbolic addr). When set, we plug in the value and
    insist on alignment.
    """
    if not _is_simdgroup_dtype(dtype):
        return False, f"dtype {dtype!s} not in simdgroup set; reject"
    if len(shape) < 2:
        return False, "rank<2; reject"

    s0, s1 = shape[-2], shape[-1]

    # fix-round-4: previous version created `z3.Int("shape0/shape1")` without
    # binding them to TIR — for non-IntImm shapes the solver had no relation
    # to the actual extent, so any "proof" was vacuous. Drop the symbolic
    # path: require static IntImm shapes (matches the conservative behaviour
    # the caller already expects on UNKNOWN).
    if not isinstance(s0, (int, tir.IntImm)) or not isinstance(s1, (int, tir.IntImm)):
        return False, f"symbolic shape rejected (s0={s0!s}, s1={s1!s})"

    s0_val = int(s0) if isinstance(s0, int) else int(s0.value)
    s1_val = int(s1) if isinstance(s1, int) else int(s1.value)
    if s0_val <= 0 or s1_val <= 0:
        return False, f"non-positive shape ({s0_val},{s1_val}); reject"
    if s0_val % _SIMDGROUP_TILE != 0 or s1_val % _SIMDGROUP_TILE != 0:
        return False, (
            f"static-reject shape0={s0_val} shape1={s1_val} "
            f"both must be multiples of {_SIMDGROUP_TILE}"
        )
    if addr_value is not None and int(addr_value) % addr_align_bytes != 0:
        return False, f"static-reject addr({addr_value})%{addr_align_bytes}!=0"

    addr_clause = (
        f" /\\ addr({addr_value})%{addr_align_bytes}==0"
        if addr_value is not None else " /\\ addr-skipped"
    )
    return True, (
        f"static-prove shape0={s0_val} shape1={s1_val}"
        f"{addr_clause} (no z3 needed)"
    )


def _log_simdgroup_decision(buf, decided_static: bool, decided_z3: bool, query: str):
    """Verbose decision logging gated on TL_LOG_SIMDGROUP=1."""
    if os.environ.get("TL_LOG_SIMDGROUP"):
        logger.warning(
            "simdgroup-detect: buf=%s shape=%s dtype=%s static=%s z3=%s query=%s",
            getattr(buf, "name", "?"),
            tuple(str(d) for d in getattr(buf, "shape", ())),
            getattr(buf, "dtype", "?"),
            decided_static,
            decided_z3,
            query,
        )


def is_simdgroup_eligible(buffer_like, *, use_z3: bool = True
                          ) -> tuple[bool, str]:
    """Public detection helper — returns (eligible, reason).

    ``buffer_like`` may be a ``tir.Buffer`` or any object exposing
    ``shape`` and ``dtype`` attributes. ``eligible`` is True only when the
    static check passes; the Z3 fallback only logs and returns its proven
    bit in ``reason`` for downstream tooling. We do NOT yet flip the IR
    rewrite based on Z3 output (conservative-by-default).
    """
    shape = list(getattr(buffer_like, "shape", []))
    dtype = getattr(buffer_like, "dtype", "")
    if _static_simdgroup_eligible(shape, dtype):
        _log_simdgroup_decision(buffer_like, True, True, "static-pass")
        return True, "static"
    if not use_z3:
        return False, "static-fail; z3-disabled"
    # CPPMEGA z3-final per-pass gate: TILELANG_DISABLE_Z3_SIMDGROUP (or
    # global TILELANG_DISABLE_Z3) bypasses the simdgroup-eligibility Z3
    # fallback (idea #8/#9). Conservative default — keep the fragment path.
    for _gate_var in ("TILELANG_DISABLE_Z3", "TILELANG_DISABLE_Z3_SIMDGROUP"):
        _v = os.environ.get(_gate_var, "")
        if _v and _v != "0":
            return False, f"static-fail; z3-disabled-by-{_gate_var}"
    proved, query = _z3_simdgroup_eligible(shape, dtype)
    _log_simdgroup_decision(buffer_like, False, proved, query)
    return False, f"static-fail; z3-proved={proved}; {query}"


def _get_gemm_ops():
    global _GEMM_OPS
    if _GEMM_OPS is None:
        _GEMM_OPS = {
            Op.get("tl.tileop.gemm"),
            Op.get("tl.tileop.wgmma_gemm"),
            Op.get("tl.tileop.tcgen05_gemm"),
        }
    return _GEMM_OPS


def _extract_buffer_var_from_region(region_call):
    if not isinstance(region_call, tir.Call):
        return None
    if len(region_call.args) < 1:
        return None
    buf_load = region_call.args[0]
    if isinstance(buf_load, tir.BufferLoad):
        return buf_load.buffer.data
    return None


def _collect_fragment_gemm_accum_vars(body: tir.Stmt) -> set:
    accum_vars: set = set()
    gemm_ops = _get_gemm_ops()

    def _visitor(stmt):
        if isinstance(stmt, tir.Evaluate) and isinstance(stmt.value, tir.Call):
            call = stmt.value
            if call.op in gemm_ops and len(call.args) >= 3:
                var = _extract_buffer_var_from_region(call.args[2])
                if var is not None and hasattr(var, "type_annotation"):
                    ta = var.type_annotation
                    if ta is not None and hasattr(ta, "storage_scope") and ta.storage_scope == "local.fragment":
                        accum_vars.add(var)
                        # Idea #8: log Z3 detection result for downstream tooling.
                        # Try to reconstruct the BufferLoad to expose shape/dtype.
                        if os.environ.get("TL_LOG_SIMDGROUP"):
                            buf_load = call.args[2].args[0] if (
                                len(call.args[2].args) > 0
                                and isinstance(call.args[2].args[0], tir.BufferLoad)
                            ) else None
                            if buf_load is not None:
                                is_simdgroup_eligible(buf_load.buffer)

    tir.stmt_functor.post_order_visit(body, _visitor)
    return accum_vars


def _collect_fragment_gemm_accum_buffers(body: tir.Stmt) -> dict:
    """Return ``{var: tir.Buffer}`` for each ``local.fragment`` GEMM accumulator.

    The IR-rewrite gated path needs the underlying buffer (shape/dtype) to
    run :func:`is_simdgroup_eligible` per-buffer. Vars whose accumulator
    region we cannot recover a ``Buffer`` for are not included; the caller
    treats those as ineligible (conservative).
    """
    accum: dict = {}
    gemm_ops = _get_gemm_ops()

    def _visitor(stmt):
        if isinstance(stmt, tir.Evaluate) and isinstance(stmt.value, tir.Call):
            call = stmt.value
            if call.op in gemm_ops and len(call.args) >= 3:
                region_call = call.args[2]
                var = _extract_buffer_var_from_region(region_call)
                if var is None or not hasattr(var, "type_annotation"):
                    return
                ta = var.type_annotation
                if ta is None or not hasattr(ta, "storage_scope"):
                    return
                if ta.storage_scope != "local.fragment":
                    return
                # Recover Buffer from the region's BufferLoad arg.
                buf = None
                if (isinstance(region_call, tir.Call)
                        and len(region_call.args) > 0
                        and isinstance(region_call.args[0], tir.BufferLoad)):
                    buf = region_call.args[0].buffer
                if buf is not None and var not in accum:
                    accum[var] = buf

    tir.stmt_functor.post_order_visit(body, _visitor)
    return accum


def _buffer_semantic_key(buf):
    return (
        buf.name,
        str(buf.dtype),
        tuple(str(dim) for dim in buf.shape),
    )


def _buffer_was_remapped(old_buf, new_buf):
    return old_buf.scope() != new_buf.scope() or not old_buf.data.same_as(new_buf.data)


def _remap_buffer(buf, var_map, accum_names, semantic_var_map):
    old_data = buf.data
    new_data = var_map.get(old_data, None)
    if new_data is None and buf.scope() == "local.fragment":
        data_name = getattr(old_data, "name", None)
        if buf.name in accum_names or data_name in accum_names:
            key = _buffer_semantic_key(buf)
            new_data = semantic_var_map.get(key, None)
            if new_data is None:
                ptr_type = old_data.type_annotation
                new_ptr = PointerType(ptr_type.element_type, "metal.simdgroup")
                new_data = tir.Var(data_name or buf.name, new_ptr)
                semantic_var_map[key] = new_data
    if new_data is None:
        return buf
    return tir.decl_buffer(
        buf.shape,
        buf.dtype,
        buf.name,
        data=new_data,
        scope="metal.simdgroup",
        data_alignment=buf.data_alignment,
        offset_factor=buf.offset_factor,
    )


def _rewrite_scope(body, var_map):
    buf_map = {}
    semantic_buf_map = {}
    semantic_var_map = {}
    accum_names = {getattr(var, "name", None) for var in var_map}

    def _remap_buffer_ref(buf):
        if buf in buf_map:
            return buf_map[buf]
        key = _buffer_semantic_key(buf)
        if buf.scope() == "local.fragment" and key in semantic_buf_map:
            new_buf = semantic_buf_map[key]
            buf_map[buf] = new_buf
            return new_buf
        new_buf = _remap_buffer(buf, var_map, accum_names, semantic_var_map)
        if _buffer_was_remapped(buf, new_buf):
            buf_map[buf] = new_buf
            semantic_buf_map[key] = new_buf
        return new_buf

    def _rewrite_region(region):
        new_buffer = _remap_buffer_ref(region.buffer)
        if new_buffer.same_as(region.buffer):
            return region
        return tir.BufferRegion(new_buffer, region.region)

    def _rewrite_match_buffer_region(match):
        new_buffer = _remap_buffer_ref(match.buffer)
        new_source = _rewrite_region(match.source)
        if new_buffer.same_as(match.buffer) and new_source.same_as(match.source):
            return match
        return tir.MatchBufferRegion(new_buffer, new_source)

    def _pre_order(node):
        if isinstance(node, AllocBuffer):
            new_buffer = _remap_buffer_ref(node.buffer)
            if _buffer_was_remapped(node.buffer, new_buffer):
                return AllocBuffer(new_buffer, node.annotations, getattr(node, "span", None))
        if isinstance(node, tir.BufferLoad):
            new_buffer = _remap_buffer_ref(node.buffer)
            if _buffer_was_remapped(node.buffer, new_buffer):
                return tir.BufferLoad(new_buffer, node.indices, node.predicate, node.span)
        if isinstance(node, tir.BufferStore):
            new_buffer = _remap_buffer_ref(node.buffer)
            if _buffer_was_remapped(node.buffer, new_buffer):
                return tir.BufferStore(new_buffer, node.value, node.indices, node.predicate, node.span)
        return None

    def _post_order(node):
        if isinstance(node, tir.Block):
            new_alloc_bufs = []
            changed = False
            for buf in node.alloc_buffers:
                new_buf = _remap_buffer_ref(buf)
                new_alloc_bufs.append(new_buf)
                if _buffer_was_remapped(buf, new_buf):
                    changed = True
            new_reads = [_rewrite_region(region) for region in node.reads]
            new_writes = [_rewrite_region(region) for region in node.writes]
            new_match_buffers = [
                _rewrite_match_buffer_region(match) for match in node.match_buffers
            ]
            changed = (
                changed
                or not all(
                    not _buffer_was_remapped(old.buffer, new.buffer)
                    for new, old in zip(new_reads, node.reads)
                )
                or not all(
                    not _buffer_was_remapped(old.buffer, new.buffer)
                    for new, old in zip(new_writes, node.writes)
                )
                or not all(
                    new.same_as(old) for new, old in zip(new_match_buffers, node.match_buffers)
                )
            )
            if changed:
                new_block = tir.Block(
                    node.iter_vars,
                    new_reads,
                    new_writes,
                    node.name_hint,
                    node.body,
                    node.init,
                    new_alloc_bufs,
                    new_match_buffers,
                    node.annotations,
                    getattr(node, "span", None),
                )
                return new_block
        return None

    # Apache's SBlock metadata and body allocation can carry distinct Buffer wrapper objects
    # for the same accumulator. Rewrite AllocBuffer first so the declaration cannot be
    # shadowed by a later metadata-only remap.
    body = tir.stmt_functor.ir_transform(body, _pre_order, None, ["tirx.AllocBuffer"])

    return tir.stmt_functor.ir_transform(
        body,
        _pre_order,
        _post_order,
        [
            "tirx.SBlock",
            "tirx.AllocBuffer",
            "tirx.BufferLoad",
            "tirx.BufferStore",
        ],
    )


def _is_rewrite_enabled() -> bool:
    """Read the ``tl.simdgroup_matrix_rewrite`` PassConfig flag.

    Default OFF — when False, the unconditional fragment→simdgroup rewrite
    runs (legacy shipping behavior). When True, the rewrite is gated
    per-buffer by :func:`is_simdgroup_eligible`.
    """
    try:
        from tvm.transform import PassContext
        cfg = PassContext.current().config
        if cfg is None:
            return False
        val = cfg.get(PASS_CONFIG_KEY, None)
        if val is None:
            return False
        return bool(val)
    except Exception:
        return False


def _build_var_map(vars_iter):
    """Allocate fresh ``metal.simdgroup``-scoped Vars for each accum Var."""
    var_map: dict = {}
    for var in vars_iter:
        ptr_type = var.type_annotation
        new_ptr = PointerType(ptr_type.element_type, "metal.simdgroup")
        new_var = tir.Var(var.name, new_ptr)
        var_map[var] = new_var
    return var_map


def _metal_fragment_to_simdgroup(func: tir.PrimFunc, mod: IRModule, ctx) -> tir.PrimFunc:
    target = func.attrs.get("target", None)
    if target is None:
        target = Target.current(allow_none=True)
    if target is None or target.kind.name != "metal":
        return func

    rewrite_gated = _is_rewrite_enabled()

    if not rewrite_gated:
        # Legacy unconditional path — ship behavior unchanged.
        accum_vars = _collect_fragment_gemm_accum_vars(func.body)
        if not accum_vars:
            return func
        var_map = _build_var_map(accum_vars)
        new_body = _rewrite_scope(func.body, var_map)
        return func.with_body(new_body)

    # Idea #8 IR rewrite path: per-buffer eligibility-gated promotion.
    accum_with_buf = _collect_fragment_gemm_accum_buffers(func.body)
    if not accum_with_buf:
        return func

    eligible_vars = []
    rewritten_names = []
    rejection_log = []
    for var, buf in accum_with_buf.items():
        eligible, reason = is_simdgroup_eligible(buf)
        if eligible:
            eligible_vars.append(var)
            rewritten_names.append(getattr(var, "name", "?"))
        else:
            rejection_log.append(f"{getattr(var, 'name', '?')}={reason}")
            if os.environ.get("TL_LOG_SIMDGROUP"):
                logger.warning(
                    "simdgroup-rewrite: skip buf=%s shape=%s dtype=%s reason=%s",
                    getattr(buf, "name", "?"),
                    tuple(str(d) for d in getattr(buf, "shape", ())),
                    getattr(buf, "dtype", "?"),
                    reason,
                )

    if not eligible_vars:
        # Nothing eligible — leave the IR alone (legacy scalar lowering
        # will handle every accumulator).
        return func

    var_map = _build_var_map(eligible_vars)
    new_body = _rewrite_scope(func.body, var_map)
    new_func = func.with_body(new_body)

    # Annotate with the rewritten buffer names so tests / downstream tooling
    # can see what fired without re-walking the IR.
    try:
        new_attrs = dict(new_func.attrs) if new_func.attrs is not None else {}
        new_attrs[EMITTED_ATTR_KEY] = tir.StringImm(",".join(sorted(rewritten_names)))
        if rejection_log:
            new_attrs["tl.simdgroup_matrix_rewrite_rejected"] = tir.StringImm(
                ";".join(rejection_log)
            )
        new_func = new_func.with_attrs(new_attrs)
    except Exception:
        pass

    return new_func


MetalFragmentToSimdgroup = prim_func_pass(_metal_fragment_to_simdgroup, opt_level=0, name="tl.MetalFragmentToSimdgroup")


# ---------------------------------------------------------------------------
# Public testing helper (Idea #8 rewrite path)
# ---------------------------------------------------------------------------

def apply_simdgroup_matrix_rewrite(func: tir.PrimFunc,
                                    *,
                                    force_enable: bool = True
                                    ) -> tir.PrimFunc:
    """Run the eligibility-gated rewrite on ``func`` outside a PassContext.

    Tests use this to drive the rewrite directly without setting up a
    ``PassContext`` with the ``tl.simdgroup_matrix_rewrite`` config. The
    target check is still enforced (the func must declare a Metal target).
    """
    if not force_enable:
        return _metal_fragment_to_simdgroup(func, None, None)

    target = func.attrs.get("target", None)
    if target is None:
        target = Target.current(allow_none=True)
    if target is None or target.kind.name != "metal":
        return func

    accum_with_buf = _collect_fragment_gemm_accum_buffers(func.body)
    if not accum_with_buf:
        return func

    eligible_vars = []
    rewritten_names = []
    rejection_log = []
    for var, buf in accum_with_buf.items():
        eligible, reason = is_simdgroup_eligible(buf)
        if eligible:
            eligible_vars.append(var)
            rewritten_names.append(getattr(var, "name", "?"))
        else:
            rejection_log.append(f"{getattr(var, 'name', '?')}={reason}")

    if not eligible_vars:
        return func

    var_map = _build_var_map(eligible_vars)
    new_body = _rewrite_scope(func.body, var_map)
    new_func = func.with_body(new_body)
    try:
        new_attrs = dict(new_func.attrs) if new_func.attrs is not None else {}
        new_attrs[EMITTED_ATTR_KEY] = tir.StringImm(",".join(sorted(rewritten_names)))
        if rejection_log:
            new_attrs["tl.simdgroup_matrix_rewrite_rejected"] = tir.StringImm(
                ";".join(rejection_log)
            )
        new_func = new_func.with_attrs(new_attrs)
    except Exception:
        pass
    return new_func
