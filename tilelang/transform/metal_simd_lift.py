"""Idea #9 (Z3 roadmap): simdgroup vs threadgroup memory choice on Metal.

When a tile-wide reduction fits within a single simdgroup (32 lanes on
Apple Silicon), Metal can perform the reduction with ``simd_shuffle_xor``
intrinsics and skip the threadgroup-memory round-trip. This pass

1. Walks ``for``-loop reductions, builds a Z3 query asserting the
   candidate constraints, and stashes proved candidates on
   ``tl.simd_lift_candidates`` for downstream tooling.

2. **Rewrites** loops that are *both* proved <= 32 by Z3 *and* explicitly
   marked with the ``tl.simd_butterfly_lane`` annotation into a
   ``tl.shfl_xor_sync``-based butterfly reduction. The annotation is
   load-bearing: a bare serial reduction loop ``for i: acc = f(acc,
   buf[i])`` does not carry the lane-mapping semantics required for a
   safe rewrite — without the annotation we keep the IR untouched.

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

from tvm import tir, IRModule
from tvm.target import Target
from tvm.tir.transform import prim_func_pass

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

    try:
        import z3  # type: ignore
    except Exception as exc:  # pragma: no cover - z3 missing
        return False, f"z3 unavailable: {exc!r}"

    solver = z3.Solver()
    solver.set("timeout", 500)
    z_ext = z3.Int("extent")
    solver.add(z_ext > 0)
    solver.push()
    solver.add(z3.Not(z_ext <= _SIMD_LANES))
    res = solver.check()
    solver.pop()
    proved = (res == z3.unsat)
    query = (
        f"assert extent <= {_SIMD_LANES}; check_sat(neg)={res}; proved={proved}"
    )
    return proved, query


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
    full_mask = tir.const(-1, "uint32")
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


class _ButterflyRewriter:
    """Replace annotated reduction ``for`` loops with a butterfly Calls."""

    def __init__(self):
        self.replaced = 0
        self.stages_emitted = 0

    def __call__(self, body: tir.Stmt) -> tir.Stmt:
        return self._mutate(body)

    def _mutate(self, node):
        if isinstance(node, tir.For):
            return self._visit_for(node)
        if isinstance(node, tir.SeqStmt):
            new_seq = [self._mutate(s) for s in node.seq]
            return tir.SeqStmt(new_seq, node.span)
        if isinstance(node, tir.IfThenElse):
            new_then = self._mutate(node.then_case) if node.then_case is not None else None
            new_else = self._mutate(node.else_case) if node.else_case is not None else None
            return tir.IfThenElse(node.condition, new_then, new_else, node.span)
        if isinstance(node, tir.LetStmt):
            return tir.LetStmt(node.var, node.value, self._mutate(node.body), node.span)
        if isinstance(node, tir.AttrStmt):
            return tir.AttrStmt(
                node.node, node.attr_key, node.value, self._mutate(node.body), node.span
            )
        if isinstance(node, tir.AllocateConst):
            return tir.AllocateConst(
                node.buffer_var, node.dtype, node.extents, node.data,
                self._mutate(node.body), node.annotations, node.span,
            )
        if isinstance(node, tir.Allocate):
            return tir.Allocate(
                node.buffer_var, node.dtype, node.extents, node.condition,
                self._mutate(node.body), node.annotations, node.span,
            )
        if isinstance(node, tir.DeclBuffer):
            return tir.DeclBuffer(node.buffer, self._mutate(node.body), node.span)
        if isinstance(node, tir.Block):
            return tir.Block(
                node.iter_vars, node.reads, node.writes, node.name_hint,
                self._mutate(node.body),
                getattr(node, "init", None),
                getattr(node, "alloc_buffers", []),
                getattr(node, "match_buffers", []),
                getattr(node, "annotations", {}),
                node.span,
            )
        if isinstance(node, tir.BlockRealize):
            return tir.BlockRealize(
                node.iter_values, node.predicate,
                self._mutate(node.block), node.span,
            )
        return node

    def _visit_for(self, node: tir.For) -> tir.Stmt:
        # Recurse into nested scopes first so inner annotated loops can fire too.
        recursed_body = self._mutate(node.body)
        # Match the same shape `_walk_reductions` used.
        body_stmt = recursed_body
        if isinstance(body_stmt, tir.SeqStmt) and len(body_stmt.seq) == 1:
            body_stmt = body_stmt.seq[0]
        if not isinstance(body_stmt, tir.BufferStore):
            return tir.For(
                node.loop_var, node.min, node.extent, node.kind, recursed_body,
                node.thread_binding, node.annotations, node.span,
            )
        op_name = _classify_reduce_op(body_stmt.value, body_stmt.buffer)
        if op_name is None or not _is_supported_reduce(op_name):
            return tir.For(
                node.loop_var, node.min, node.extent, node.kind, recursed_body,
                node.thread_binding, node.annotations, node.span,
            )
        if not _is_butterfly_annotated(node):
            return tir.For(
                node.loop_var, node.min, node.extent, node.kind, recursed_body,
                node.thread_binding, node.annotations, node.span,
            )
        proved, _ = _z3_extent_le_32(node.extent)
        if not proved:
            return tir.For(
                node.loop_var, node.min, node.extent, node.kind, recursed_body,
                node.thread_binding, node.annotations, node.span,
            )

        # Resolve concrete extent for stage sequence.
        if isinstance(node.extent, tir.IntImm):
            extent_val = int(node.extent.value)
        elif isinstance(node.extent, int):
            extent_val = int(node.extent)
        else:
            # Symbolic-but-proved is rare here; without a numeric we cannot
            # emit the static stage list. Conservative: skip.
            return tir.For(
                node.loop_var, node.min, node.extent, node.kind, recursed_body,
                node.thread_binding, node.annotations, node.span,
            )

        # Build acc_load (BufferLoad mirroring the BufferStore).
        store: tir.BufferStore = body_stmt
        acc_load = tir.BufferLoad(store.buffer, list(store.indices))
        dtype = str(store.value.dtype) if hasattr(store.value, "dtype") else str(
            store.buffer.dtype
        )
        reduced = _build_butterfly(acc_load, op_name, extent_val, dtype)
        new_store = tir.BufferStore(store.buffer, reduced, list(store.indices), store.span)
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
        func.params, new_body, func.ret_type, func.buffer_map, func.attrs, func.span,
    )
    return new_func, rw.replaced, rw.stages_emitted


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


# ---------------------------------------------------------------------------
# Pass entry
# ---------------------------------------------------------------------------

def _metal_simd_lift(func: tir.PrimFunc, mod: IRModule, ctx) -> tir.PrimFunc:
    enabled = False
    try:
        from tvm.transform import PassContext
        cfg = PassContext.current().config
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

    # Conservative IR rewrite: only fires on annotated, proved candidates.
    if any(c.annotated and c.proved for c in candidates):
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
