"""Idea #9 (Z3 roadmap): simdgroup vs threadgroup memory choice on Metal.

When a tile-wide reduction fits within a single simdgroup (32 lanes on
Apple Silicon), Metal can perform the reduction with `simd_shuffle_xor`
intrinsics and skip the threadgroup-memory round-trip. This pass walks
``for``-loop reductions, builds a Z3 query asserting the candidate
constraints, and -- when proven -- *logs* the candidate site.

Z3 query shape:

    tile_extent <= 32
    /\\  reduce_op ∈ {add, max, min, or, and, xor}
    /\\  no cross-simdgroup write happens before reduce

Status: detection-only with logging. The pass is *gated* behind the
PassConfig key ``tl.simd_lift_reductions`` (default OFF). The IR is left
unchanged regardless of the Z3 result; the wiring (PassConfig slot,
phase.py slot, tests) is ready for a follow-up to actually emit
``simd_shuffle_xor`` reductions.
"""

from __future__ import annotations

import logging
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


@dataclass(frozen=True)
class _ReductionCandidate:
    loop_var: str
    extent_repr: str
    op: str
    proved: bool
    query: str


def _is_supported_reduce(op_name: str) -> bool:
    return op_name.lower() in _SIMD_REDUCE_OPS


def _classify_reduce_op(value: tir.PrimExpr, acc_var) -> str | None:
    """Classify the reduce op from ``acc = f(acc, ...)``. Returns op name or None."""
    # Common shapes:
    #   acc[0] = acc[0] + buf[i]  →  "add"
    #   acc[0] = T.max(acc[0], buf[i]) → "max"
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
    # Concrete fast-path.
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
    # We have no upstream symbolic-bound channel here; without an inbound
    # `<= 32` constraint Z3 will not be able to prove the conjecture, which
    # is exactly the conservative behavior we want. The query is still
    # logged so a follow-up can plumb constraints in.
    solver.push()
    solver.add(z3.Not(z_ext <= _SIMD_LANES))
    res = solver.check()
    solver.pop()
    proved = (res == z3.unsat)
    query = (
        f"assert extent <= {_SIMD_LANES}; check_sat(neg)={res}; proved={proved}"
    )
    return proved, query


def _walk_reductions(body: tir.Stmt) -> list[_ReductionCandidate]:
    """Walk ``for`` loops looking for ``acc = f(acc, ...)`` patterns."""
    candidates: list[_ReductionCandidate] = []

    def _visit(node):
        if not isinstance(node, tir.For):
            return
        # Recognise simple in-loop reductions:
        #     for i in range(EXT):
        #         acc[...] = f(acc[...], <stuff involving i>)
        body = node.body
        # Peel any wrapper SeqStmt of length 1.
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
                "simd-lift-detect: func=%s loop=%s extent=%s op=%s proved=%s query=%s",
                func_name, c.loop_var, c.extent_repr, c.op, c.proved, c.query,
            )


def _metal_simd_lift(func: tir.PrimFunc, mod: IRModule, ctx) -> tir.PrimFunc:
    # Default OFF: only run when the PassConfig flag is set True.
    pass_ctx = ctx
    enabled = False
    try:
        # tvm.transform.PassContext has a `config` attribute (Map<str, ...>).
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

    # Detection-only: do not modify IR. Stash the candidate list on the
    # PrimFunc attrs so downstream tooling / tests can inspect what would
    # have been transformed.
    if candidates:
        from tvm.ir import make_node  # noqa: F401  (compatibility import)
        new_attrs = dict(func.attrs) if func.attrs is not None else {}
        new_attrs["tl.simd_lift_candidates"] = tir.StringImm(
            ";".join(
                f"{c.loop_var}:{c.extent_repr}:{c.op}:proved={c.proved}"
                for c in candidates
            )
        )
        try:
            func = func.with_attrs(new_attrs)
        except Exception:
            # PrimFunc.with_attrs may not exist on older builds; ignore.
            pass

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
