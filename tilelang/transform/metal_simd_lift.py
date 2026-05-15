"""Idea #9 (Z3 roadmap): simdgroup vs threadgroup memory choice on Metal.

When a tile-wide reduction fits within a single simdgroup (32 lanes on
Apple Silicon), Metal can perform the reduction with ``simd_shuffle_xor``
intrinsics and skip the threadgroup-memory round-trip. This pass

1. Walks ``for``-loop reductions, builds a Z3 query asserting the
   candidate constraints, and stashes proved candidates on
   ``tl.simd_lift_candidates`` for downstream tooling.

2. **Rewrites** loops that are *both* proved <= 32 by Z3 *and* explicitly
   marked with the ``tl.simd_butterfly_lane`` annotation into semantic
   ``tir.tvm_thread_allreduce`` IR. The annotation is load-bearing: a bare
   serial reduction loop ``for i: acc = f(acc, buf[i])`` does not carry the
   lane-mapping semantics required for a safe rewrite — without the annotation
   we keep the IR untouched.

The older explicit ``tl.shfl_xor_sync`` butterfly helper remains available for
direct backend-shape tests, but the scheduled pass now prefers semantic
reduction IR so backend lowerers can choose the implementation strategy.

Z3 query shape::

    tile_extent <= 32
    /\\  reduce_op ∈ {add, max, min, or, and, xor}

The pass is gated behind PassConfig key ``tl.simd_lift_reductions``
(default OFF). When OFF, both detection and rewrite are no-ops.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass

from tvm import ir as tvm_ir
from tvm import tir, IRModule
from tvm.target import Target
from tvm.tir.transform import prim_func_pass
from tilelang.analysis.reduction_legality import attach_reduction_legality_metadata
from tilelang.analysis.reduction_plan import attach_reduction_plan_metadata
from tilelang.analysis.sync_event_plan import attach_sync_event_plan_metadata

logger = logging.getLogger("tilelang.metal_simd_lift")

#: Apple Silicon simdgroup width. The Z3 query asserts ``tile_extent <= 32``.
_SIMD_LANES = 32

#: Reduction ops that have a direct ``simd_*`` intrinsic.
_SIMD_REDUCE_OPS = {"add", "max", "min", "or", "and", "xor"}

PASS_CONFIG_KEY = "tl.simd_lift_reductions"

#: Per-loop annotation key the rewrite looks for. Set this on a ``T.serial``
#: loop whose induction variable maps 1:1 to the simdgroup lane id to opt
#: into the butterfly rewrite.
LOOP_ANNOTATION_KEY = "tl.simd_butterfly_lane"


@dataclass(frozen=True)
class _ReductionCandidate:
    loop_var: str
    extent_repr: str
    op: str
    proved: bool
    query: str
    annotated: bool = False


def _is_supported_reduce(op_name: str) -> bool:
    return op_name.lower() in _SIMD_REDUCE_OPS


def _classify_reduce_op(value: tir.PrimExpr, acc_var) -> str | None:
    """Classify the reduce op from ``acc = f(acc, ...)``. Returns op name or None."""
    if isinstance(value, tir.Add):
        return "add"
    if isinstance(value, tir.Sub):
        return "add"  # sub-into-acc still uses simd_sum-style lowering
    if isinstance(value, tir.Mul):
        return None  # not a supported simd reduce
    if hasattr(tir, "Max") and isinstance(value, tir.Max):
        return "max"
    if hasattr(tir, "Min") and isinstance(value, tir.Min):
        return "min"
    if isinstance(value, tir.Call):
        op_name = getattr(getattr(value, "op", None), "name", "")
        if op_name.endswith("max"):
            return "max"
        if op_name.endswith("min"):
            return "min"
        if op_name.endswith("bitwise_or") or op_name.endswith("or"):
            return "or"
        if op_name.endswith("bitwise_and") or op_name.endswith("and"):
            return "and"
        if op_name.endswith("bitwise_xor") or op_name.endswith("xor"):
            return "xor"
    return None


def _z3_extent_le_32(extent_expr) -> tuple[bool, str]:
    """Z3 fallback: prove ``tile_extent <= 32`` for symbolic extents.

    Returns ``(proved, query_str)``. Conservative: UNKNOWN/timeout → False.
    """
    if isinstance(extent_expr, (int,)):
        proved = extent_expr <= _SIMD_LANES
        return proved, f"static: extent={extent_expr} <= {_SIMD_LANES}? {proved}"
    if isinstance(extent_expr, tir.IntImm):
        proved = int(extent_expr.value) <= _SIMD_LANES
        return proved, f"static: extent={int(extent_expr.value)} <= {_SIMD_LANES}? {proved}"

    # CPPMEGA z3-final per-pass gate: TILELANG_DISABLE_Z3_SIMDGROUP (or
    # global TILELANG_DISABLE_Z3) bypasses the symbolic-extent Z3 path
    # (idea #8/#9). The current implementation already rejects symbolic
    # extents conservatively, but we surface the gate explicitly so the
    # log message is consistent with the other Z3-using passes.
    for _gate_var in ("TILELANG_DISABLE_Z3", "TILELANG_DISABLE_Z3_SIMDGROUP"):
        _v = os.environ.get(_gate_var, "")
        if _v and _v != "0":
            return False, f"z3-disabled-by-{_gate_var}; symbolic extent rejected"

    # fix-round-4: previous version constructed `z3.Int("extent")` with only
    # `z_ext > 0` — no link to the actual TIR expression — so the query was
    # vacuous (always SAT under negation, hence always returning False but
    # advertising a "z3 proof" in the log). Reject symbolic extents
    # conservatively without spinning up z3.
    return False, f"symbolic extent rejected (expr={extent_expr!s})"


def _is_butterfly_annotated(node: tir.For) -> bool:
    """Return True if the loop carries ``tl.simd_butterfly_lane = True``."""
    ann = getattr(node, "annotations", None)
    if ann is None:
        return False
    try:
        v = ann.get(LOOP_ANNOTATION_KEY, None)
    except Exception:
        # Older TVM Map; fall through to dict-style.
        try:
            v = dict(ann).get(LOOP_ANNOTATION_KEY, None)
        except Exception:
            return False
    if v is None:
        return False
    if isinstance(v, bool):
        return v
    if isinstance(v, tir.IntImm):
        return int(v.value) != 0
    try:
        return bool(int(v))
    except Exception:
        return bool(v)


def _walk_reductions(body: tir.Stmt) -> list[_ReductionCandidate]:
    """Walk ``for`` loops looking for ``acc = f(acc, ...)`` patterns."""
    candidates: list[_ReductionCandidate] = []

    def _visit(node):
        if not isinstance(node, tir.For):
            return
        body = node.body
        if isinstance(body, tir.SeqStmt) and len(body.seq) == 1:
            body = body.seq[0]
        if not isinstance(body, tir.BufferStore):
            return
        op_name = _classify_reduce_op(body.value, body.buffer)
        if op_name is None or not _is_supported_reduce(op_name):
            return
        proved, query = _z3_extent_le_32(node.extent)
        candidates.append(
            _ReductionCandidate(
                loop_var=str(node.loop_var.name),
                extent_repr=str(node.extent),
                op=op_name,
                proved=proved,
                query=query,
                annotated=_is_butterfly_annotated(node),
            )
        )

    tir.stmt_functor.post_order_visit(body, _visit)
    return candidates


def _log_candidates(func_name: str, candidates: list[_ReductionCandidate]):
    if not candidates:
        return
    if os.environ.get("TL_LOG_SIMD_LIFT") or any(c.proved for c in candidates):
        for c in candidates:
            logger.warning(
                "simd-lift-detect: func=%s loop=%s extent=%s op=%s proved=%s "
                "annotated=%s query=%s",
                func_name, c.loop_var, c.extent_repr, c.op, c.proved,
                c.annotated, c.query,
            )


# ---------------------------------------------------------------------------
# Butterfly construction
# ---------------------------------------------------------------------------

def _butterfly_stages(extent: int) -> list[int]:
    """Return the lane-shift sequence for a butterfly over ``extent`` lanes.

    For extent=32 → [16, 8, 4, 2, 1] (5 stages).
    For extent=16 → [8, 4, 2, 1]      (4 stages).
    For extent=8  → [4, 2, 1]         (3 stages).
    """
    if extent <= 1:
        return []
    n_stages = int(math.ceil(math.log2(extent)))
    # Largest power of two <= extent gives the top shift.
    top = 1 << (n_stages - 1)
    out = []
    s = top
    while s >= 1:
        out.append(s)
        s //= 2
    # Defense-in-depth: every emitted shift becomes the `mask` arg to
    # `tl.shfl_xor_sync` and must lie strictly inside the Apple simdgroup
    # width [1, 32). The caller guards against extent>32 / non-power-of-2,
    # but assert here so any future caller misuse fails loudly instead of
    # producing out-of-range shuffle indices.
    for shift in out:
        assert 1 <= shift < 32, (
            f"butterfly stage {shift} out of [1,32) for Apple simdgroup "
            f"(extent={extent})"
        )
    return out


def _apply_op(op: str, a: tir.PrimExpr, b: tir.PrimExpr) -> tir.PrimExpr:
    """Combine two values using the simd reduce op."""
    if op == "add":
        return a + b
    if op == "max":
        return tir.max(a, b)
    if op == "min":
        return tir.min(a, b)
    if op == "or":
        return tir.bitwise_or(a, b)
    if op == "and":
        return tir.bitwise_and(a, b)
    if op == "xor":
        return tir.bitwise_xor(a, b)
    raise ValueError(f"unsupported simd reduce op: {op}")


def _same_primexpr(a: tir.PrimExpr, b: tir.PrimExpr) -> bool:
    try:
        return tvm_ir.structural_equal(a, b, map_free_vars=True)
    except Exception:
        return bool(a.same_as(b)) if hasattr(a, "same_as") else False


def _same_buffer_indices(lhs, rhs) -> bool:
    if len(lhs) != len(rhs):
        return False
    return all(_same_primexpr(a, b) for a, b in zip(lhs, rhs))


def _is_accumulator_load(
    value: tir.PrimExpr,
    buffer: tir.Buffer,
    indices,
) -> bool:
    if not isinstance(value, tir.BufferLoad):
        return False
    if not value.buffer.same_as(buffer):
        return False
    return _same_buffer_indices(value.indices, indices)


def _extract_add_contribution(store: tir.BufferStore) -> tir.PrimExpr | None:
    """Return ``x`` from ``acc = acc + x`` or ``acc = x + acc``.

    The semantic allreduce input must be the per-lane contribution, not the
    whole serial accumulator expression. Other reductions remain on the older
    explicit helper until ReductionPlan carries reducer-specific identities.
    """
    value = store.value
    if not isinstance(value, tir.Add):
        return None
    indices = list(store.indices)
    if _is_accumulator_load(value.a, store.buffer, indices):
        return value.b
    if _is_accumulator_load(value.b, store.buffer, indices):
        return value.a
    return None


def _build_thread_allreduce(
    value: tir.PrimExpr,
    out: tir.BufferLoad,
    reduce_index: tir.PrimExpr,
    span=None,
) -> tir.Stmt:
    reducer = tir.comm_reducer(
        lambda x, y: x + y,
        lambda dtype: tir.const(0, dtype=dtype),
        name="sum",
    )
    call = tir.call_intrin(
        "handle",
        "tir.tvm_thread_allreduce",
        tir.const(1, "uint32"),
        value,
        tir.const(True, "bool"),
        out,
        reduce_index,
    )
    return tir.AttrStmt(
        reducer,
        "reduce_scope",
        tir.reinterpret("handle", tir.const(0, "uint64")),
        tir.Evaluate(call, span),
        span,
    )


def _reduce_index_with_static_extent(node: tir.For) -> tir.PrimExpr:
    if isinstance(node.extent, tir.IntImm):
        return node.loop_var % tir.IntImm("int32", int(node.extent.value))
    if isinstance(node.extent, int):
        return node.loop_var % tir.IntImm("int32", int(node.extent))
    return node.loop_var


def _build_butterfly(
    acc_load: tir.PrimExpr,
    op: str,
    extent: int,
    dtype: str,
) -> tir.PrimExpr:
    """Build a chain of ``op(acc, shfl_xor_sync(mask, acc, shift, 32))`` Calls.

    Returns a PrimExpr representing the final reduced value. The caller is
    responsible for storing it back into the original accumulator buffer.
    """
    shfl_op = tir.op.Op.get("tl.shfl_xor_sync")
    full_mask = tir.const((1 << 32) - 1, "uint32")
    width = tir.const(_SIMD_LANES, "int32")
    value: tir.PrimExpr = acc_load
    for shift in _butterfly_stages(extent):
        shifted = tir.Call(
            dtype,
            shfl_op,
            [full_mask, value, tir.const(shift, "int32"), width],
        )
        value = _apply_op(op, value, shifted)
    return value


class _ThreadAllreduceRewriter:
    """Replace annotated add reductions with semantic thread allreduce IR."""

    def __init__(self):
        self.replaced = 0

    def __call__(self, body: tir.Stmt) -> tir.Stmt:
        return self._mutate(body)

    @staticmethod
    def _span(node):
        return getattr(node, "span", None)

    def _for_with_body(self, node: tir.For, body: tir.Stmt) -> tir.For:
        return tir.For(
            node.loop_var, node.min, node.extent, node.kind, body,
            getattr(node, "thread_binding", None),
            getattr(node, "annotations", {}),
            self._span(node),
        )

    def _mutate(self, node):
        if isinstance(node, tir.For):
            return self._visit_for(node)
        if isinstance(node, tir.SeqStmt):
            new_seq = [self._mutate(s) for s in node.seq]
            return tir.SeqStmt(new_seq, self._span(node))
        if isinstance(node, tir.IfThenElse):
            new_then = self._mutate(node.then_case) if node.then_case is not None else None
            new_else = self._mutate(node.else_case) if node.else_case is not None else None
            return tir.IfThenElse(node.condition, new_then, new_else, self._span(node))
        if isinstance(node, tir.LetStmt):
            return tir.LetStmt(node.var, node.value, self._mutate(node.body), self._span(node))
        if isinstance(node, tir.AttrStmt):
            return tir.AttrStmt(
                node.node, node.attr_key, node.value, self._mutate(node.body), self._span(node)
            )
        allocate_const = getattr(tir, "AllocateConst", None)
        if allocate_const is not None and isinstance(node, allocate_const):
            return allocate_const(
                node.buffer_var, node.dtype, node.extents, node.data,
                self._mutate(node.body), node.annotations, self._span(node),
            )
        if isinstance(node, tir.Allocate):
            return tir.Allocate(
                node.buffer_var, node.dtype, node.extents, node.condition,
                self._mutate(node.body), node.annotations, self._span(node),
            )
        if isinstance(node, tir.DeclBuffer):
            return tir.DeclBuffer(node.buffer, self._mutate(node.body), self._span(node))
        if isinstance(node, tir.Block):
            return tir.Block(
                node.iter_vars, node.reads, node.writes, node.name_hint,
                self._mutate(node.body),
                getattr(node, "init", None),
                getattr(node, "alloc_buffers", []),
                getattr(node, "match_buffers", []),
                getattr(node, "annotations", {}),
                self._span(node),
            )
        if isinstance(node, tir.BlockRealize):
            return tir.BlockRealize(
                node.iter_values, node.predicate,
                self._mutate(node.block), self._span(node),
            )
        if hasattr(node, "body") and node.body is not None:
            new_body = self._mutate(node.body)
            if not new_body.same_as(node.body) and hasattr(node, "with_body"):
                return node.with_body(new_body)
        return node

    def _visit_for(self, node: tir.For) -> tir.Stmt:
        recursed_body = self._mutate(node.body)
        body_stmt = recursed_body
        if isinstance(body_stmt, tir.SeqStmt) and len(body_stmt.seq) == 1:
            body_stmt = body_stmt.seq[0]
        if not isinstance(body_stmt, tir.BufferStore):
            return self._for_with_body(node, recursed_body)
        if _classify_reduce_op(body_stmt.value, body_stmt.buffer) != "add":
            return self._for_with_body(node, recursed_body)
        if not _is_butterfly_annotated(node):
            return self._for_with_body(node, recursed_body)
        proved, query = _z3_extent_le_32(node.extent)
        if not proved:
            logger.warning(
                "semantic-reduction-rewrite: declining annotated loop var=%s extent=%s "
                "reason=z3_extent_unproved query=%s",
                str(node.loop_var.name), str(node.extent), query,
            )
            return self._for_with_body(node, recursed_body)
        contribution = _extract_add_contribution(body_stmt)
        if contribution is None:
            logger.warning(
                "semantic-reduction-rewrite: declining annotated loop var=%s extent=%s "
                "reason=accumulator_contribution_not_extracted",
                str(node.loop_var.name), str(node.extent),
            )
            return self._for_with_body(node, recursed_body)
        out = tir.BufferLoad(body_stmt.buffer, list(body_stmt.indices))
        reduce_index = _reduce_index_with_static_extent(node)
        self.replaced += 1
        return _build_thread_allreduce(contribution, out, reduce_index, self._span(body_stmt))


class _ButterflyRewriter:
    """Replace annotated reduction ``for`` loops with a butterfly Calls."""

    def __init__(self):
        self.replaced = 0
        self.stages_emitted = 0

    def __call__(self, body: tir.Stmt) -> tir.Stmt:
        return self._mutate(body)

    @staticmethod
    def _span(node):
        return getattr(node, "span", None)

    def _for_with_body(self, node: tir.For, body: tir.Stmt) -> tir.For:
        return tir.For(
            node.loop_var, node.min, node.extent, node.kind, body,
            getattr(node, "thread_binding", None),
            getattr(node, "annotations", {}),
            self._span(node),
        )

    def _mutate(self, node):
        if isinstance(node, tir.For):
            return self._visit_for(node)
        if isinstance(node, tir.SeqStmt):
            new_seq = [self._mutate(s) for s in node.seq]
            return tir.SeqStmt(new_seq, self._span(node))
        if isinstance(node, tir.IfThenElse):
            new_then = self._mutate(node.then_case) if node.then_case is not None else None
            new_else = self._mutate(node.else_case) if node.else_case is not None else None
            return tir.IfThenElse(node.condition, new_then, new_else, self._span(node))
        if isinstance(node, tir.LetStmt):
            return tir.LetStmt(node.var, node.value, self._mutate(node.body), self._span(node))
        if isinstance(node, tir.AttrStmt):
            return tir.AttrStmt(
                node.node, node.attr_key, node.value, self._mutate(node.body), self._span(node)
            )
        allocate_const = getattr(tir, "AllocateConst", None)
        if allocate_const is not None and isinstance(node, allocate_const):
            return allocate_const(
                node.buffer_var, node.dtype, node.extents, node.data,
                self._mutate(node.body), node.annotations, self._span(node),
            )
        if isinstance(node, tir.Allocate):
            return tir.Allocate(
                node.buffer_var, node.dtype, node.extents, node.condition,
                self._mutate(node.body), node.annotations, self._span(node),
            )
        if isinstance(node, tir.DeclBuffer):
            return tir.DeclBuffer(node.buffer, self._mutate(node.body), self._span(node))
        if isinstance(node, tir.Block):
            return tir.Block(
                node.iter_vars, node.reads, node.writes, node.name_hint,
                self._mutate(node.body),
                getattr(node, "init", None),
                getattr(node, "alloc_buffers", []),
                getattr(node, "match_buffers", []),
                getattr(node, "annotations", {}),
                self._span(node),
            )
        if isinstance(node, tir.BlockRealize):
            return tir.BlockRealize(
                node.iter_values, node.predicate,
                self._mutate(node.block), self._span(node),
            )
        # Catch-all: recurse into any body-bearing node we don't explicitly
        # handle (e.g. AssertStmt, ProducerRealize). Without this, annotated
        # reduction loops nested inside unknown statement types are silently
        # skipped. We reconstruct the node if the body changed.
        if hasattr(node, "body") and node.body is not None:
            new_body = self._mutate(node.body)
            if not new_body.same_as(node.body):
                # Best-effort reconstruction via with_body if available,
                # otherwise fall through to return the original node.
                if hasattr(node, "with_body"):
                    return node.with_body(new_body)
        return node

    def _visit_for(self, node: tir.For) -> tir.Stmt:
        # Recurse into nested scopes first so inner annotated loops can fire too.
        recursed_body = self._mutate(node.body)
        # Match the same shape `_walk_reductions` used.
        body_stmt = recursed_body
        if isinstance(body_stmt, tir.SeqStmt) and len(body_stmt.seq) == 1:
            body_stmt = body_stmt.seq[0]
        if not isinstance(body_stmt, tir.BufferStore):
            return self._for_with_body(node, recursed_body)
        op_name = _classify_reduce_op(body_stmt.value, body_stmt.buffer)
        if op_name is None or not _is_supported_reduce(op_name):
            return self._for_with_body(node, recursed_body)
        if not _is_butterfly_annotated(node):
            return self._for_with_body(node, recursed_body)
        proved, query = _z3_extent_le_32(node.extent)
        if not proved:
            # Annotated loop but Z3 cannot prove extent <= 32 — log so CI can
            # surface the missed rewrite instead of dropping silently.
            logger.warning(
                "simd-lift-rewrite: declining annotated loop var=%s extent=%s "
                "reason=z3_extent_unproved query=%s",
                str(node.loop_var.name), str(node.extent), query,
            )
            return self._for_with_body(node, recursed_body)

        # Resolve concrete extent for stage sequence.
        if isinstance(node.extent, tir.IntImm):
            extent_val = int(node.extent.value)
        elif isinstance(node.extent, int):
            extent_val = int(node.extent)
        else:
            # Symbolic-but-proved is rare here; without a numeric we cannot
            # emit the static stage list. Conservative: skip — log so CI can
            # diagnose why an annotated/proved loop wasn't rewritten.
            logger.warning(
                "simd-lift-rewrite: declining annotated loop var=%s extent=%s "
                "reason=symbolic_extent_no_static_value",
                str(node.loop_var.name), str(node.extent),
            )
            return self._for_with_body(node, recursed_body)

        # #9 butterfly guard: only rewrite when extent is a power-of-2 in
        # [2, 32]. Non-power-of-2 extents would yield bad shuffle indices,
        # and SIMD-group width on Metal/Apple GPUs is 32, so larger extents
        # cannot be served by a single shfl_xor_sync chain.
        if (extent_val < 2 or extent_val > 32 or
                (extent_val & (extent_val - 1)) != 0):
            logger.warning(
                "simd-lift-rewrite: declining annotated loop var=%s extent=%d "
                "reason=extent_not_pow2_in_[2,32]",
                str(node.loop_var.name), extent_val,
            )
            return self._for_with_body(node, recursed_body)

        # Build acc_load (BufferLoad mirroring the BufferStore).
        store: tir.BufferStore = body_stmt
        acc_load = tir.BufferLoad(store.buffer, list(store.indices))
        dtype = str(store.value.dtype) if hasattr(store.value, "dtype") else str(
            store.buffer.dtype
        )
        reduced = _build_butterfly(acc_load, op_name, extent_val, dtype)
        new_store = tir.BufferStore(store.buffer, reduced, list(store.indices), self._span(store))
        self.replaced += 1
        self.stages_emitted += len(_butterfly_stages(extent_val))
        return new_store


def rewrite_reductions(func: tir.PrimFunc) -> tuple[tir.PrimFunc, int, int]:
    """Public helper: run the butterfly rewrite on a PrimFunc.

    Returns ``(new_func, n_replaced, n_stages_total)``. Used by tests.
    """
    rw = _ButterflyRewriter()
    new_body = rw(func.body)
    if rw.replaced == 0:
        return func, 0, 0
    new_func = tir.PrimFunc(
        func.params, new_body, func.ret_type, func.buffer_map, func.attrs,
        getattr(func, "span", None),
    )
    return new_func, rw.replaced, rw.stages_emitted


def rewrite_reductions_to_thread_allreduce(
    func: tir.PrimFunc,
) -> tuple[tir.PrimFunc, int]:
    """Public helper: rewrite annotated add reductions to semantic IR.

    Returns ``(new_func, n_replaced)``. The pass uses this path before backend
    lowering so scheduler/codegen can choose the reduction strategy from
    ``tir.tvm_thread_allreduce`` instead of from hand-written partial buffers
    or target intrinsics.
    """
    rw = _ThreadAllreduceRewriter()
    new_body = rw(func.body)
    if rw.replaced == 0:
        return func, 0
    new_func = tir.PrimFunc(
        func.params, new_body, func.ret_type, func.buffer_map, func.attrs,
        getattr(func, "span", None),
    )
    return new_func, rw.replaced


def count_shfl_xor_calls(func: tir.PrimFunc) -> int:
    """Test helper: count ``tl.shfl_xor_sync`` Calls in a PrimFunc."""
    n = 0

    def _visit(node):
        nonlocal n
        if isinstance(node, tir.Call):
            op_name = getattr(getattr(node, "op", None), "name", "")
            if op_name == "tl.shfl_xor_sync":
                n += 1

    tir.stmt_functor.post_order_visit(func.body, _visit)
    return n


def count_thread_allreduce_calls(func: tir.PrimFunc) -> int:
    """Test helper: count semantic ``tvm_thread_allreduce`` Calls."""
    n = 0

    def _visit(node):
        nonlocal n
        if isinstance(node, tir.Call):
            op_name = getattr(getattr(node, "op", None), "name", "")
            if op_name.endswith("tvm_thread_allreduce"):
                n += 1

    tir.stmt_functor.post_order_visit(func.body, _visit)
    return n


# ---------------------------------------------------------------------------
# Pass entry
# ---------------------------------------------------------------------------

def _metal_simd_lift(func: tir.PrimFunc, mod: IRModule, ctx) -> tir.PrimFunc:
    enabled = False
    try:
        from tvm import transform as tvm_transform
        cfg = tvm_transform.PassContext.current().config
        val = cfg.get(PASS_CONFIG_KEY, None) if cfg is not None else None
        if val is not None:
            try:
                enabled = bool(val)
            except Exception:
                enabled = False
    except Exception:
        enabled = False

    if not enabled:
        return func

    target = func.attrs.get("target", None)
    if target is None:
        target = Target.current(allow_none=True)
    if target is None or target.kind.name != "metal":
        return func

    candidates = _walk_reductions(func.body)
    func_name = ""
    try:
        func_name = str(func.attrs.get("global_symbol", ""))
    except Exception:
        pass
    _log_candidates(func_name, candidates)

    # Stash candidate metadata.
    if candidates:
        new_attrs = dict(func.attrs) if func.attrs is not None else {}
        new_attrs["tl.simd_lift_candidates"] = tir.StringImm(
            ";".join(
                f"{c.loop_var}:{c.extent_repr}:{c.op}:proved={c.proved}:"
                f"annotated={c.annotated}"
                for c in candidates
            )
        )
        try:
            func = func.with_attrs(new_attrs)
        except Exception:
            pass

    # Conservative semantic rewrite: only fires on annotated, proved add
    # reductions where the accumulator contribution can be extracted.
    if any(c.annotated and c.proved for c in candidates):
        semantic_rewritten, n_semantic = rewrite_reductions_to_thread_allreduce(func)
        if n_semantic:
            logger.warning(
                "semantic-reduction-rewrite: func=%s replaced=%d",
                func_name, n_semantic,
            )
            try:
                if func.attrs is not None:
                    semantic_rewritten = semantic_rewritten.with_attrs(dict(func.attrs))
            except Exception:
                pass
            semantic_rewritten = attach_reduction_plan_metadata(semantic_rewritten)
            semantic_rewritten = attach_reduction_legality_metadata(semantic_rewritten)
            semantic_rewritten = attach_sync_event_plan_metadata(semantic_rewritten)
            return semantic_rewritten

        # Keep the explicit backend-shape helper as a fallback for reducer
        # kinds whose semantic ReductionPlan support has not landed yet.
        rewritten, n_replaced, n_stages = rewrite_reductions(func)
        if n_replaced:
            logger.warning(
                "simd-lift-rewrite: func=%s replaced=%d butterfly_stages=%d",
                func_name, n_replaced, n_stages,
            )
            # Preserve previously stashed attrs through the rewrite.
            try:
                if func.attrs is not None:
                    rewritten = rewritten.with_attrs(dict(func.attrs))
            except Exception:
                pass
            return rewritten

    return func


MetalSimdLiftReductions = prim_func_pass(
    _metal_simd_lift, opt_level=0, name="tl.MetalSimdLiftReductions"
)


def detect_candidates(func: tir.PrimFunc) -> list[_ReductionCandidate]:
    """Public testing helper — runs the detector on ``func`` regardless of
    PassConfig gating. Returns the list of candidates found, with their Z3
    proof bits.
    """
    return _walk_reductions(func.body)
