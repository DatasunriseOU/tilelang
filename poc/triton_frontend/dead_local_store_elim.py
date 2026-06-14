"""DeadLocalStoreElim -- fixpoint dead-store elimination of register/local
index/mask staging buffers emitted by the Triton frontend op_emitters.

WHY THIS EXISTS (ncu-pinned root cause of the §P1 dstates 97x gap)
------------------------------------------------------------------
``from_ttir`` lowers Triton index/mask/broadcast tensor ops (``tt.expand_dims``,
``tt.broadcast``, ``tt.splat``, integer ``arith`` binops, ``tt.cmp``) into
EAGER per-lane ``scope="local"`` staging tiles -- one ``T.grid(M,N)`` scalar
loop per arith op. But the ACTUAL global load / store / mask sites re-derive
their flat index + predicate INLINE (PtrAnalysis folds the affine index at the
copy/store site), so the eagerly-materialised index/mask tiles are *written but
never read*. ptxas places those dead tiles in the stack frame:

  * §P1 dstates: 8448 B stack frame, 216 registers, SASS STL=128 / LDL=33,
    17.6 GB local-memory traffic -> caps occupancy at 2 blocks/SM and drives
    the dominant LONG_SCOREBOARD (L1TEX local-latency) stall.

TVM's default lowering (Simplify / StorageRewrite / RemoveNoOp) does NOT remove
a store into an allocated buffer that is never loaded -- that is not their job
(RemoveNoOp removes no-op *statements*; StorageRewrite only *reuses* storage).
So the dead tiles survive all the way into the emitted ``.cu``.

WHAT THIS PASS DOES (generic, backend-neutral, fail-loud)
--------------------------------------------------------
A standard backward dead-store-elimination *fixpoint* over the root ``SBlock``'s
top-level statement sequence:

  1. Candidate buffers = ``scope="local"`` alloc buffers (index/mask staging).
     Shared / fragment / global buffers are NEVER touched.
  2. A top-level statement is a "pure local staging write" iff it writes ONLY
     local candidate buffers and carries NO external side effect (no opaque
     ``Call`` / ``T.copy`` / ``T.gemm`` / global or shared store).
  3. Liveness fixpoint: a candidate buffer is NEEDED iff it is read by a KEPT
     statement. Seed kept = every non-candidate statement. A candidate
     statement is then KEPT iff the buffer it writes is NEEDED. Iterate to
     fixpoint -- this propagates liveness backwards through chains of pure
     index/mask tiles, so a whole transitively-dead chain (e.g. the entire
     epilogue ``cmp -> broadcast -> and`` mask tower whose terminal tile is
     never read because the store recomputes the predicate inline) collapses,
     while a tile that IS read by a surviving ``if`` predicate (the load masks)
     is preserved.
  4. Drop the dead statements from the SeqStmt and the dead buffers from the
     ``SBlock``'s ``alloc_buffers``.

This is backend-NEUTRAL (it deletes index/data-movement arithmetic in the TIR,
before any CUDA/Metal codegen) and BIT-EXACT by construction: it only DELETES
provably-dead writes, never rewrites surviving arithmetic. No int64->i32
narrowing is performed here, so there is zero overflow risk for large shapes.

RULE #1 (fail loud, no silent wrong-output): a buffer is removed ONLY when it is
provably never read by any surviving statement. If the expected structure is not
present, the PrimFunc is returned UNCHANGED (this pass is an optimisation, not a
correctness gate) -- but it never silently drops a live buffer.
"""
from __future__ import annotations

from typing import Any, Dict, List, Set, Tuple

__all__ = ["eliminate_dead_local_stores"]


def _scope_of(buf: Any) -> str:
    try:
        return str(buf.scope())
    except Exception:
        return ""


def _is_local_scope(scope: str) -> bool:
    # Only plain register/local staging is removable. "local.fragment",
    # "shared", "shared.dyn", "global", "wmma.*" are NEVER candidates.
    return scope == "local" or scope == ""


# TVM ``TCallEffectKind`` enum (src/tir/op/op_attr_types): ExprAnnotation=0,
# Pure=1, ReadState=2, UpdateState=3, Opaque=4. A Call is side-effect-free iff
# its op is Pure (1) or a pure expression annotation (0). Everything else --
# ReadState/UpdateState/Opaque (e.g. ``tl.tileop.copy``/``gemm``/``fill``,
# atomics, cp.async, mbarrier) -- AND any op carrying NO effect-kind attribute
# is treated as an EXTERNAL EFFECT (fail-closed: unknown => keep, never wrongly
# delete).
_PURE_EFFECT_KINDS = (0, 1)


def _call_is_pure(call: Any, tvm: Any) -> bool:
    op = getattr(call, "op", None)
    name = getattr(op, "name", None)
    if name is None:
        return False
    try:
        ek = tvm.ir.Op.get(name).get_attr("TCallEffectKind")
    except Exception:
        return False
    if ek is None:
        return False
    try:
        return int(ek) in _PURE_EFFECT_KINDS
    except Exception:
        return False


def _collect_buffer_io(stmt: Any, tir: Any, tvm: Any) -> Tuple[Set[Any], Set[Any], bool]:
    """Return (written_data_vars, read_data_vars, has_external_effect).

    ``has_external_effect`` is True when the statement writes a buffer whose
    scope is NOT plain local, or contains a Call to a non-pure op (T.copy /
    T.gemm / T.fill / cp.async / atomics / any op without a Pure effect kind).
    Pure scalar intrinsics (arith / cmp / bitwise / exp / if_then_else / cast)
    do NOT count -- a staging loop that only computes index/mask arithmetic via
    pure intrinsics remains a removable pure-local write.
    """
    writes: Set[Any] = set()
    reads: Set[Any] = set()
    effect = {"v": False}
    write_scopes: List[str] = []

    def visit(node: Any) -> None:
        if isinstance(node, tir.BufferStore):
            writes.add(node.buffer.data)
            write_scopes.append(_scope_of(node.buffer))
        elif isinstance(node, tir.BufferLoad):
            reads.add(node.buffer.data)
        elif isinstance(node, tir.Call):
            if not _call_is_pure(node, tvm):
                effect["v"] = True

    tir.stmt_functor.post_order_visit(stmt, visit)
    for sc in write_scopes:
        if not _is_local_scope(sc):
            effect["v"] = True
            break
    return writes, reads, effect["v"]


def eliminate_dead_local_stores(prim_func: Any) -> Any:
    """Fixpoint DCE of write-only ``scope="local"`` staging buffers."""
    import tvm  # tvm only importable once the tilelang dev root is on sys.path
    from tvm import tir
    import tvm.tirx as tirx

    body = prim_func.body

    # Body layout (frame_register): AttrStmt(thread_extent) x N -> SBlockRealize
    # -> SBlock(root). Peel the launch attrs to reach the realize/block.
    launch_attrs: List[Any] = []
    inner = body
    while (
        type(inner).__name__ == "AttrStmt"
        and getattr(inner, "attr_key", None) == "thread_extent"
    ):
        launch_attrs.append(inner)
        inner = inner.body

    if type(inner).__name__ != "SBlockRealize":
        return prim_func
    realize = inner
    sblk = realize.block
    if type(sblk).__name__ != "SBlock":
        return prim_func

    block_body = sblk.body
    if not isinstance(block_body, tir.SeqStmt):
        return prim_func

    seq = list(block_body.seq)
    n = len(seq)

    # Local-scope alloc buffers are the DCE candidates.
    local_alloc: Dict[Any, Any] = {}
    for ab in sblk.alloc_buffers:
        if _is_local_scope(_scope_of(ab)):
            local_alloc[ab.data] = ab
    if not local_alloc:
        return prim_func

    # Per-statement IO classification.
    stmt_writes: List[Set[Any]] = []
    stmt_reads: List[Set[Any]] = []
    stmt_pure_local: List[bool] = []
    for s in seq:
        w, r, eff = _collect_buffer_io(s, tir, tvm)
        stmt_writes.append(w)
        stmt_reads.append(r)
        is_pure = (not eff) and (len(w) > 0) and all(v in local_alloc for v in w)
        stmt_pure_local.append(is_pure)

    # Fixpoint liveness. kept[i] => statement i survives.
    kept = [not stmt_pure_local[i] for i in range(n)]
    changed = True
    while changed:
        changed = False
        needed: Set[Any] = set()
        for i in range(n):
            if kept[i]:
                for v in stmt_reads[i]:
                    if v in local_alloc:
                        needed.add(v)
        for i in range(n):
            if stmt_pure_local[i] and not kept[i]:
                if any((v in needed) for v in stmt_writes[i]):
                    kept[i] = True
                    changed = True

    live_buffers: Set[Any] = set()
    for i in range(n):
        if kept[i]:
            for v in stmt_writes[i]:
                live_buffers.add(v)

    dead_stmt = [i for i in range(n) if not kept[i]]
    dead_buf = [v for v in local_alloc if v not in live_buffers]
    if not dead_stmt and not dead_buf:
        return prim_func

    new_seq = [seq[i] for i in range(n) if kept[i]]
    new_body = new_seq[0] if len(new_seq) == 1 else tir.SeqStmt(new_seq)

    new_alloc = [
        ab for ab in sblk.alloc_buffers
        if (ab.data in live_buffers) or (not _is_local_scope(_scope_of(ab)))
    ]

    new_block = tirx.SBlock(
        iter_vars=list(sblk.iter_vars),
        reads=list(sblk.reads),
        writes=list(sblk.writes),
        name_hint=sblk.name_hint,
        body=new_body,
        init=sblk.init,
        alloc_buffers=new_alloc,
        match_buffers=list(sblk.match_buffers),
        annotations=sblk.annotations,
    )
    new_realize = tirx.SBlockRealize(
        list(realize.iter_values), realize.predicate, new_block
    )

    new_outer = new_realize
    for attr in reversed(launch_attrs):
        new_outer = tir.AttrStmt(attr.node, attr.attr_key, attr.value, new_outer)

    return prim_func.with_body(new_outer)
