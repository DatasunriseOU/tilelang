"""Metal merge-round barrier cleanup.

This pass handles a narrow shape produced by ``ThreadSync("shared")`` for
tree-merge kernels:

    if active_lane:
        ... read shared pair slots, write local merge buffers ...
    T.tvm_storage_sync("shared")
    if active_lane:
        ... write local merge buffers back to shared pair slots ...
    T.tvm_storage_sync("shared")

For a proven active-lane tree merge the middle barrier is redundant.  The
writeback can stay in the same active-lane branch as the local merge, while
the final round barrier still protects the next merge round.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tvm import IRModule, tir
from tvm.target import Target
from tvm.tir.transform import prim_func_pass

PASS_CONFIG_KEY = "tl.z3_proof.barrier_minimization"
FUSION_ATTR = "tl.metal_merge_round_barrier_fusions"


def _pass_enabled(ctx) -> bool:
    try:
        val = ctx.config.get(PASS_CONFIG_KEY, None)
    except Exception:
        return False
    if val is None:
        return False
    try:
        return bool(val)
    except Exception:
        return False


def _is_metal_func(func: tir.PrimFunc) -> bool:
    target = func.attrs.get("target", None) if func.attrs is not None else None
    if target is None:
        target = Target.current(allow_none=True)
    return bool(target is not None and target.kind.name == "metal")


def _structural_equal(lhs, rhs) -> bool:
    try:
        import tvm

        return bool(tvm.ir.structural_equal(lhs, rhs))
    except Exception:
        return str(lhs) == str(rhs)


def _is_zero(expr) -> bool:
    return isinstance(expr, tir.IntImm) and int(expr.value) == 0


def _is_mul_by_two(expr) -> bool:
    if not isinstance(expr, tir.Mul):
        return False
    return (
        isinstance(expr.a, tir.IntImm)
        and int(expr.a.value) == 2
        or isinstance(expr.b, tir.IntImm)
        and int(expr.b.value) == 2
    )


def _is_active_lane_mod_condition(expr) -> bool:
    if not isinstance(expr, tir.EQ):
        return False
    lhs, rhs = expr.a, expr.b
    if _is_zero(lhs):
        lhs, rhs = rhs, lhs
    if not _is_zero(rhs):
        return False
    mod_types = tuple(
        t for t in (getattr(tir, "FloorMod", None), getattr(tir, "Mod", None)) if t is not None
    )
    if not mod_types or not isinstance(lhs, mod_types):
        return False
    return _is_mul_by_two(lhs.b)


def _is_storage_sync(stmt: tir.Stmt, scope: str) -> bool:
    if not isinstance(stmt, tir.Evaluate):
        return False
    call = stmt.value
    if not isinstance(call, tir.Call):
        return False
    op_name = getattr(getattr(call, "op", None), "name", "")
    if not str(op_name).endswith("tvm_storage_sync"):
        return False
    if len(call.args) != 1 or not isinstance(call.args[0], tir.StringImm):
        return False
    return call.args[0].value == scope


def _has_else(stmt: tir.IfThenElse) -> bool:
    return getattr(stmt, "else_case", None) is not None


def _buffer_scope(buffer: tir.Buffer) -> str:
    try:
        return str(buffer.scope())
    except Exception:
        pass
    try:
        return str(buffer.data.type_annotation.storage_scope)
    except Exception:
        return ""


def _buffer_key(buffer: tir.Buffer) -> str:
    try:
        return str(buffer.data)
    except Exception:
        return str(buffer)


@dataclass
class _AccessStats:
    shared_loads: int = 0
    shared_stores: int = 0
    local_loads: int = 0
    local_stores: int = 0
    other_stores: int = 0
    syncs: int = 0
    shared_load_buffers: set[str] = field(default_factory=set)
    shared_store_buffers: set[str] = field(default_factory=set)
    local_load_buffers: set[str] = field(default_factory=set)
    local_store_buffers: set[str] = field(default_factory=set)


def _collect_access_stats(stmt: tir.Stmt) -> _AccessStats:
    stats = _AccessStats()

    def visit(node):
        if isinstance(node, tir.BufferLoad):
            scope = _buffer_scope(node.buffer)
            if scope == "shared":
                stats.shared_loads += 1
                stats.shared_load_buffers.add(_buffer_key(node.buffer))
            elif scope.startswith("local"):
                stats.local_loads += 1
                stats.local_load_buffers.add(_buffer_key(node.buffer))
        elif isinstance(node, tir.BufferStore):
            scope = _buffer_scope(node.buffer)
            if scope == "shared":
                stats.shared_stores += 1
                stats.shared_store_buffers.add(_buffer_key(node.buffer))
            elif scope.startswith("local"):
                stats.local_stores += 1
                stats.local_store_buffers.add(_buffer_key(node.buffer))
            else:
                stats.other_stores += 1
        elif _is_storage_sync(node, "shared"):
            stats.syncs += 1

    tir.stmt_functor.post_order_visit(stmt, visit)
    return stats


def _is_merge_local_phase(stmt: tir.Stmt) -> bool:
    stats = _collect_access_stats(stmt)
    return (
        stats.syncs == 0
        and stats.shared_loads > 0
        and stats.shared_stores == 0
        and stats.local_stores > 0
        and stats.other_stores == 0
    )


def _is_shared_writeback_phase(stmt: tir.Stmt, merge_stats: _AccessStats) -> bool:
    stats = _collect_access_stats(stmt)
    return (
        stats.syncs == 0
        and stats.shared_stores > 0
        and stats.shared_loads == 0
        and stats.local_loads > 0
        and stats.other_stores == 0
        and stats.shared_store_buffers.issubset(merge_stats.shared_load_buffers)
        and bool(stats.local_load_buffers & merge_stats.local_store_buffers)
    )


def _can_fuse(first_if: tir.IfThenElse, second_if: tir.IfThenElse) -> bool:
    if _has_else(first_if) or _has_else(second_if):
        return False
    if not _structural_equal(first_if.condition, second_if.condition):
        return False
    if not _is_active_lane_mod_condition(first_if.condition):
        return False
    merge_stats = _collect_access_stats(first_if.then_case)
    if not _is_merge_local_phase(first_if.then_case):
        return False
    return _is_shared_writeback_phase(second_if.then_case, merge_stats)


def _seq(stmts: list[tir.Stmt]) -> tir.Stmt:
    if len(stmts) == 1:
        return stmts[0]
    return tir.SeqStmt(stmts)


def _rewrite_seq(stmt: tir.SeqStmt) -> tuple[tir.Stmt, int]:
    seq = list(stmt.seq)
    out: list[tir.Stmt] = []
    changed = 0
    i = 0
    while i < len(seq):
        if (
            i + 3 < len(seq)
            and isinstance(seq[i], tir.IfThenElse)
            and _is_storage_sync(seq[i + 1], "shared")
            and isinstance(seq[i + 2], tir.IfThenElse)
            and _is_storage_sync(seq[i + 3], "shared")
            and _can_fuse(seq[i], seq[i + 2])
        ):
            fused_then = _seq([seq[i].then_case, seq[i + 2].then_case])
            out.append(tir.IfThenElse(seq[i].condition, fused_then, None))
            out.append(seq[i + 3])
            changed += 1
            i += 4
            continue
        out.append(seq[i])
        i += 1
    if changed == 0:
        return stmt, 0
    return _seq(out), changed


@prim_func_pass(opt_level=0)
class MetalMergeRoundBarrierCleanup:
    """Fuse canonical Metal merge-round writeback branches.

    The pass is default-off and uses the existing barrier proof config key
    ``tl.z3_proof.barrier_minimization``.  It is intentionally narrower than
    a general barrier-elision pass: if the IR no longer has a modulo active
    lane guard, local merge phase, shared writeback phase, and final shared
    barrier, it leaves the function unchanged.
    """

    def transform_function(
        self, func: tir.PrimFunc, mod: IRModule, ctx
    ) -> tir.PrimFunc:
        if not _pass_enabled(ctx) or not _is_metal_func(func):
            return func

        fusions = 0

        def post(node):
            nonlocal fusions
            if not isinstance(node, tir.SeqStmt):
                return None
            rewritten, count = _rewrite_seq(node)
            if count:
                fusions += count
                return rewritten
            return None

        new_body = tir.stmt_functor.ir_transform(func.body, None, post)
        if fusions == 0:
            return func

        new_func = func.with_body(new_body)
        attrs = dict(new_func.attrs) if new_func.attrs is not None else {}
        attrs[FUSION_ATTR] = tir.IntImm("int32", fusions)
        return new_func.with_attrs(attrs)
