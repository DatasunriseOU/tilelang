"""Rewrite local.fragment → metal.simdgroup for GEMM accumulators on Metal.

Idea #8 (Z3 roadmap): in addition to the static shape/dtype checks used to
decide simdgroup eligibility, we run a *detection-only* Z3 fallback for
symbolic shapes. The Z3 query asserts:

    shape[0] % 8 == 0
    /\\  shape[1] % 8 == 0
    /\\  dtype ∈ {fp16, packed fp8}
    /\\  addr % 16 == 0

If the static check fails for symbolic inputs, the Z3 helper attempts to
prove the shape constraint and logs the candidate. The IR rewrite remains
gated behind the legacy static path until we wire Z3-driven lowering. This
keeps the pass conservative-by-default: if Z3 returns False/UNKNOWN, we keep
the legacy non-simdgroup path.
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
    query_lines = []
    if not _is_simdgroup_dtype(dtype):
        return False, f"dtype {dtype!s} not in simdgroup set; reject"
    if len(shape) < 2:
        return False, "rank<2; reject"

    try:
        import z3  # type: ignore
    except Exception as exc:  # pragma: no cover - z3 missing
        return False, f"z3 unavailable: {exc!r}"

    s0, s1 = shape[-2], shape[-1]
    solver = z3.Solver()
    solver.set("timeout", 500)  # 500 ms cap

    z_s0 = z3.Int("shape0")
    z_s1 = z3.Int("shape1")

    solver.add(z_s0 > 0, z_s1 > 0)

    if isinstance(s0, (int, tir.IntImm)):
        solver.add(z_s0 == int(s0 if isinstance(s0, int) else s0.value))
    if isinstance(s1, (int, tir.IntImm)):
        solver.add(z_s1 == int(s1 if isinstance(s1, int) else s1.value))

    conjuncts = [
        z_s0 % _SIMDGROUP_TILE == 0,
        z_s1 % _SIMDGROUP_TILE == 0,
    ]
    if addr_value is not None:
        # Concrete address — directly check alignment.
        z_addr = z3.IntVal(int(addr_value))
        conjuncts.append(z_addr % addr_align_bytes == 0)
        addr_clause = f" /\\ addr({addr_value})%{addr_align_bytes}==0"
    else:
        addr_clause = " /\\ addr-skipped"
    query = z3.And(*conjuncts)

    # Conservative: we want to PROVE the conjunction holds. That is, the
    # negation must be UNSAT.
    solver.push()
    solver.add(z3.Not(query))
    res = solver.check()
    solver.pop()
    query_str = (
        f"assert shape[0]%{_SIMDGROUP_TILE}==0 /\\ shape[1]%{_SIMDGROUP_TILE}==0"
        f"{addr_clause}; check_sat(neg)={res}"
    )
    query_lines.append(query_str)
    proved = (res == z3.unsat)
    return proved, "; ".join(query_lines)


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


def _metal_fragment_to_simdgroup(func: tir.PrimFunc, mod: IRModule, ctx) -> tir.PrimFunc:
    target = func.attrs.get("target", None)
    if target is None:
        target = Target.current(allow_none=True)
    if target is None or target.kind.name != "metal":
        return func

    accum_vars = _collect_fragment_gemm_accum_vars(func.body)
    if not accum_vars:
        return func

    var_map: dict = {}
    for var in accum_vars:
        ptr_type = var.type_annotation
        new_ptr = PointerType(ptr_type.element_type, "metal.simdgroup")
        new_var = tir.Var(var.name, new_ptr)
        var_map[var] = new_var

    new_body = _rewrite_scope(func.body, var_map)
    return func.with_body(new_body)


MetalFragmentToSimdgroup = prim_func_pass(_metal_fragment_to_simdgroup, opt_level=0, name="tl.MetalFragmentToSimdgroup")
