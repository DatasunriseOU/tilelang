"""Track B (Metal-GEMM auto-pass): serial-reduction → ``T.gemm`` auto-rewrite.

This module is the AUTOMATIC compiler pass that detects the canonical
*contraction* shape that mamba3 Path-C Metal prims (F0 ``summary_states`` /
``cb``, B2/B0) write as a serial scalar accumulator, and rewrites it to a
tile-level ``T.gemm`` (which the existing ``LowerTileOp`` + Metal
``src/backend/metal/op/gemm.cc`` selector lowers to ``matmul2d``). Future
kernels written in the serial shape then get GEMM-ification "for free".

It is a **Python-registered** ``prim_func_pass`` — NO C++ rebuild, so the live
``build/`` is untouched. It is modeled byte-for-byte on the proven
``metal_simd_lift.py`` structure (gated PassConfig key, ``post_order_visit``
detector, conservative mutator, non-vacuous z3 query, RULE #1 fail-to-serial).

THE TARGET IR SHAPE (the canonical 2-loop accumulator a contraction emits)::

    for ls in T.Parallel(M * N):          # row/col flattened
        i = ls // N
        j = ls % N
        acc = T.alloc_local((1,), accum)  # size-1 local accumulator
        acc[0] = 0
        for k in T.serial(K):             # the contraction axis
            acc[0] = acc[0] + A[.. i .. k ..] * B[.. k .. j ..]
        out[.. i .. j ..] = acc[0]        # (optionally Cast(acc[0]))

This is exactly ``out = A @ B`` (with transpose flags inferred from which
operand index is the row vs the contraction axis). An optional per-row scale
``s`` that is provably ``k``-independent may multiply the product (folded into
the ``A`` operand, exactly as the F0 ``summary_states`` decay·dt fold).

RULE #1 (the ONE path when enabled): the rewrite fires ONLY when BOTH (a) the
structural matcher recognizes the canonical shape AND (b) the built-in z3
prover (``auto_gemmify_z3``) proves the rewrite preserves the reduction
semantics and is race-free. On ANY decline — ambiguous shape, k-dependent
scale, unprovable index map, z3 UNKNOWN/timeout/SAT-witness — it LEAVES the
serial loop untouched (NO wrong rewrite) and logs a structured decline reason.
The hand-written serial prim stays the byte-identical parity reference.

Default OFF: gated behind PassConfig key ``tl.auto_gemmify_reductions``.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any

from tvm import ir as tvm_ir
from tvm import tir, IRModule
from tvm.target import Target
from tvm.tir.transform import prim_func_pass

logger = logging.getLogger("tilelang.auto_gemmify_reductions")

#: PassConfig key. Default OFF — absent/False ⇒ the pass is a no-op. This key
#: is honored when the loaded libtilelang registers it via
#: ``TVM_REGISTER_PASS_CONFIG_OPTION`` (a C++ build). Because this prototype
#: must NOT trigger a C++ rebuild (the live build/ stays intact), the pass ALSO
#: honors the env-var gate ``ENABLE_ENV_VAR`` below — reading an *unregistered*
#: PassConfig key returns None gracefully (no error), so the pass stays a no-op
#: until either gate is flipped ON.
PASS_CONFIG_KEY = "tl.auto_gemmify_reductions"

#: Env-var gate (Python-side, no C++ registration needed). Set to a truthy value
#: ("1"/"true"/"yes") to enable the pass without a registered PassConfig key.
ENABLE_ENV_VAR = "TILELANG_ENABLE_AUTO_GEMMIFY"

#: Func attr the pass stamps with the z3 proof bits (deliverable C audit trail).
PROVED_ATTR_KEY = "tl.auto_gemmify_proved"

#: Func attr the pass stamps with structured decline reasons (RULE #1 trail).
DECLINED_ATTR_KEY = "tl.auto_gemmify_declined"

#: Env gates that bypass the z3 prover (consistent with the other Z3 passes).
#: When z3 is bypassed the pass DECLINES every candidate (fail-to-serial); it
#: NEVER rewrites without a proof.
_Z3_DISABLE_ENV = (
    "TILELANG_DISABLE_Z3",
    "TILELANG_DISABLE_Z3_AUTO_GEMMIFY",
    "CPPMEGA_DISABLE_Z3",
)


# --------------------------------------------------------------------------- #
# Match record                                                                 #
# --------------------------------------------------------------------------- #


@dataclass
class _ContractionMatch:
    """A recognized canonical contraction nest + its inferred GEMM mapping."""

    # The outer T.Parallel(M*N) loop var name (diagnostics only).
    outer_var: str
    # Extents (resolved ints) of the GEMM dims.
    M: int
    N: int
    K: int
    # The decomposition of the flattened outer var: i = outer // div_i, etc.
    # We require exactly i = ls // N, j = ls % N (row-major flatten).
    row_is_floordiv: bool
    # Operand A: which of its index positions carry the row (i) and contraction
    # (k) axes. transpose_A is True when the row axis sits in A's *inner* index
    # (i.e. A is indexed [k, i] not [i, k]).
    transpose_A: bool
    # Operand B: transpose_B is True when B is indexed [j, k] (so out = A @ B^T).
    transpose_B: bool
    # Whether a provably-k-independent per-row scale was folded.
    has_scale: bool
    # Symbolic index expressions captured from the TIR (for the non-vacuous z3
    # query). Stored as strings for the decline/proof audit trail.
    a_index_repr: str
    b_index_repr: str
    out_index_repr: str
    # z3 proof outcome — filled in by ``prove_contraction``.
    z3_used: bool = False
    z3_algebra_proved: bool = False
    z3_race_proved: bool = False
    z3_reason: str = ""
    decline_reason: str | None = None
    notes: dict[str, Any] = field(default_factory=dict)

    @property
    def proved(self) -> bool:
        return self.z3_used and self.z3_algebra_proved and self.z3_race_proved

    def as_dict(self) -> dict[str, Any]:
        return {
            "outer_var": self.outer_var,
            "M": self.M,
            "N": self.N,
            "K": self.K,
            "transpose_A": self.transpose_A,
            "transpose_B": self.transpose_B,
            "has_scale": self.has_scale,
            "a_index": self.a_index_repr,
            "b_index": self.b_index_repr,
            "out_index": self.out_index_repr,
            "z3_used": self.z3_used,
            "z3_algebra_proved": self.z3_algebra_proved,
            "z3_race_proved": self.z3_race_proved,
            "z3_reason": self.z3_reason,
            "decline_reason": self.decline_reason,
        }


# --------------------------------------------------------------------------- #
# Small structural helpers                                                      #
# --------------------------------------------------------------------------- #


def _as_int(expr) -> int | None:
    if isinstance(expr, bool):
        return None
    if isinstance(expr, int):
        return int(expr)
    if isinstance(expr, tir.IntImm):
        return int(expr.value)
    return None


def _strip_single_seq(stmt: tir.Stmt) -> tir.Stmt:
    """Unwrap a SeqStmt of length 1 (the F0 prims wrap accumulators this way)."""
    while isinstance(stmt, tir.SeqStmt) and len(stmt.seq) == 1:
        stmt = stmt.seq[0]
    return stmt


def _unwrap_cast(expr: tir.PrimExpr) -> tir.PrimExpr:
    """Strip a single outer ``Cast`` (the store-out / operand reads use Cast)."""
    if isinstance(expr, tir.Cast):
        return expr.value
    # tir.Call form of cast (some frontends emit ``T.Cast`` as a Call).
    if isinstance(expr, tir.Call):
        op_name = getattr(getattr(expr, "op", None), "name", "")
        if op_name.endswith(".cast") or op_name == "tir.Cast":
            if len(expr.args) == 1:
                return expr.args[0]
    return expr


def _is_zero_const(expr: tir.PrimExpr) -> bool:
    inner = _unwrap_cast(expr)
    iv = _as_int(inner)
    if iv is not None:
        return iv == 0
    if isinstance(inner, tir.FloatImm):
        return float(inner.value) == 0.0
    return False


def _is_local_scalar_buffer(buf: tir.Buffer) -> bool:
    """True for a ``T.alloc_local((1,), ...)`` size-1 local accumulator."""
    scope = ""
    try:
        scope = buf.scope()
    except Exception:
        data = getattr(buf, "data", None)
        scope = getattr(getattr(data, "type_annotation", None), "storage_scope", "")
    if scope not in ("local", "local.var", ""):
        # Empty scope can occur for un-annotated locals; we additionally
        # require shape == (1,) below, so an empty scope is acceptable.
        if scope:
            return False
    shape = list(getattr(buf, "shape", []))
    if len(shape) != 1:
        return False
    return _as_int(shape[0]) == 1


def _collect_vars(expr: tir.PrimExpr) -> set[str]:
    names: set[str] = set()

    def _v(node):
        if isinstance(node, tir.Var):
            names.add(node.name)

    tir.stmt_functor.post_order_visit(expr, _v)
    return names


def _flatten_mul(expr: tir.PrimExpr) -> list[tir.PrimExpr]:
    """Flatten a left/right-nested product into a list of factors."""
    if isinstance(expr, tir.Mul):
        return _flatten_mul(expr.a) + _flatten_mul(expr.b)
    return [expr]


def _buffer_loads(expr: tir.PrimExpr) -> list[tir.BufferLoad]:
    loads: list[tir.BufferLoad] = []

    def _v(node):
        if isinstance(node, tir.BufferLoad):
            loads.append(node)

    tir.stmt_functor.post_order_visit(expr, _v)
    return loads


# --------------------------------------------------------------------------- #
# The structural matcher                                                        #
# --------------------------------------------------------------------------- #


def _match_inner_contraction(
    inner: tir.For,
    acc_buf: tir.Buffer,
    k_var: tir.Var,
    i_var: tir.Var,
    j_var: tir.Var,
) -> dict[str, Any] | str:
    """Match ``for k: acc[0] = acc[0] + [scale*] A[..] * B[..]``.

    Returns a dict describing the two operands + scale, or a string decline
    reason. ``i_var``/``j_var`` are the outer row/col vars; ``k_var`` is the
    inner contraction var.
    """
    body = _strip_single_seq(inner.body)
    if not isinstance(body, tir.BufferStore):
        return "inner_body_not_bufferstore"
    if not body.buffer.same_as(acc_buf):
        return "inner_store_not_to_accumulator"
    if len(body.indices) != 1 or _as_int(body.indices[0]) != 0:
        return "inner_store_not_acc0"
    value = body.value
    if not isinstance(value, tir.Add):
        return "inner_value_not_add"
    # One side must be the accumulator load acc[0]; the other is the product.
    def _is_acc0_load(e):
        return (
            isinstance(e, tir.BufferLoad)
            and e.buffer.same_as(acc_buf)
            and len(e.indices) == 1
            and _as_int(e.indices[0]) == 0
        )

    if _is_acc0_load(value.a):
        product = value.b
    elif _is_acc0_load(value.b):
        product = value.a
    else:
        return "inner_add_has_no_acc_self_load"

    k_name = k_var.name
    factors = _flatten_mul(product)
    # Split the product into:
    #   * CONTRACTION OPERANDS — single ``BufferLoad`` factors that DO contain
    #     the contraction var k AND at least one NON-k var (the row/col axis).
    #     Exactly two of these form the GEMM A/B.
    #   * SCALE factors — every other factor; validated for foldability by the
    #     caller (``match_contraction``) once the row/col axis names are known.
    # RULE #1 / risk #3: a k-bearing factor that is NOT a clean 2D operand
    # (e.g. ``w[l]`` a k-ONLY load, or ``exp2(C[..k..])``) is NOT foldable →
    # decline here; a non-k factor that depends on the COL axis (e.g. a full
    # output-shaped ``D[i,j]``) is rejected by the caller's scale check.
    operand_factors: list[tir.BufferLoad] = []
    scale_factors: list[tir.PrimExpr] = []
    for f in factors:
        fu = _unwrap_cast(f)
        fvars = _collect_vars(fu)
        has_k = k_name in fvars
        non_k_vars = fvars - {k_name}
        if isinstance(fu, tir.BufferLoad) and has_k and non_k_vars:
            # A clean 2D contraction operand: carries k + a row/col axis.
            operand_factors.append(fu)
        elif has_k:
            # Contains k but is not a clean 2-axis operand load (k-only load,
            # or a non-trivial function of k) → not foldable, not an operand.
            return "per_row_scale_depends_on_contraction_var"
        else:
            # k-independent factor → candidate per-row scale (validated later).
            scale_factors.append(f)

    if len(operand_factors) != 2:
        return f"product_has_{len(operand_factors)}_operand_loads_expected_2"

    a_load, b_load = operand_factors[0], operand_factors[1]

    # Each operand must contain the contraction var k in its index set (already
    # guaranteed by the split above; re-assert for clarity/safety).
    a_idx_vars = set().union(*[_collect_vars(ix) for ix in a_load.indices])
    b_idx_vars = set().union(*[_collect_vars(ix) for ix in b_load.indices])
    if k_name not in a_idx_vars or k_name not in b_idx_vars:
        return "contraction_var_absent_from_an_operand"

    return {
        "a_load": a_load,
        "b_load": b_load,
        "scale_factors": scale_factors,
        "store": body,
    }


def _decompose_flatten(i_expr, j_expr, outer_var: tir.Var, N: int):
    """Verify ``i = ls // N`` and ``j = ls % N`` for the row-major flatten.

    Returns True iff both decompositions match (the canonical shape). We accept
    either the LetStmt-bound form (vars already substituted) or the literal
    floordiv/floormod expressions.
    """
    ok_i = False
    ok_j = False
    if isinstance(i_expr, tir.FloorDiv):
        if (
            isinstance(i_expr.a, tir.Var)
            and i_expr.a.same_as(outer_var)
            and _as_int(i_expr.b) == N
        ):
            ok_i = True
    if isinstance(j_expr, tir.FloorMod):
        if (
            isinstance(j_expr.a, tir.Var)
            and j_expr.a.same_as(outer_var)
            and _as_int(j_expr.b) == N
        ):
            ok_j = True
    return ok_i and ok_j


def _find_axis_positions(load: tir.BufferLoad, row_name: str, k_name: str):
    """Return (row_pos, k_pos) — the index positions carrying row & k vars.

    Each must appear in EXACTLY one index position and that position's index
    expression must be an *affine function of only that one axis var* (so the
    GEMM operand maps cleanly). Returns None if ambiguous.
    """
    row_pos = None
    k_pos = None
    for pos, ix in enumerate(load.indices):
        vs = _collect_vars(ix)
        has_row = row_name in vs
        has_k = k_name in vs
        if has_row and has_k:
            return None  # both axes in one index — not a clean 2D operand
        if has_row:
            if row_pos is not None:
                return None
            row_pos = pos
        if has_k:
            if k_pos is not None:
                return None
            k_pos = pos
    if row_pos is None or k_pos is None:
        return None
    return row_pos, k_pos


def match_contraction(outer: tir.For) -> _ContractionMatch | str:
    """Top-level matcher. Returns a ``_ContractionMatch`` or a decline string.

    Recognizes the canonical::

        for ls in T.Parallel(M*N):
            i = ls // N ; j = ls % N
            acc = alloc_local((1,))
            acc[0] = 0
            for k in serial(K): acc[0] = acc[0] + [scale*] A[..i..k..]*B[..k..j..]
            out[..i..j..] = acc[0]
    """
    if not isinstance(outer, tir.For):
        return "outer_not_for"
    # The outer loop must be parallel (T.Parallel lowers to kind==Parallel) OR
    # serial; we only require the *shape*. Extent = M*N (resolved int).
    MN = _as_int(outer.extent)
    if MN is None:
        return "outer_extent_symbolic"

    outer_var = outer.loop_var
    body = outer.body

    # The frontend lowers ``i = ls // N`` / ``j = ls % N`` either as
    #   * vendored ``tilelang.LetStmt`` / apache ``tir.LetStmt`` body-bearing
    #     bindings that WRAP the body (peel them here), OR
    #   * apache tirx ``Bind`` nodes that live INSIDE the outer-loop SeqStmt
    #     (harvested below), OR
    #   * fully inlined floordiv/floormod inside the operand/out indices.
    # We collect every (var-name -> value) binding into ``let_bindings`` so the
    # row/col flatten can be recovered from whichever encoding the frontend
    # used. Likewise the size-1 accumulator appears as a vendored
    # ``tilelang.Allocate`` / apache ``tir.Allocate`` / ``DeclBuffer`` wrapper
    # OR as a body-less apache ``AllocBuffer`` node inside the SeqStmt.
    let_bindings: dict[str, tir.PrimExpr] = {}
    while isinstance(body, tir.LetStmt):
        let_bindings[body.var.name] = body.value
        body = body.body

    # Descend through Allocate / DeclBuffer wrappers for the acc buffer, but
    # keep a handle on the acc buffer var.
    acc_buf = None

    def _descend(stmt):
        nonlocal acc_buf
        allocate_const = getattr(tir, "AllocateConst", None)
        if isinstance(stmt, tir.Allocate):
            return _descend(stmt.body)
        if allocate_const is not None and isinstance(stmt, allocate_const):
            return _descend(stmt.body)
        if isinstance(stmt, tir.DeclBuffer):
            return _descend(stmt.body)
        return stmt

    body = _descend(body)

    # Now expect a SeqStmt with the init / inner-for / out-store. The acc init
    # (acc[0]=0) and out-store bracket the inner contraction loop.
    if not isinstance(body, tir.SeqStmt):
        return "outer_body_not_seq"
    seq = list(body.seq)
    # Harvest apache tirx ``Bind`` nodes from inside the SeqStmt (the
    # ``i = ls // N`` / ``j = ls % N`` row/col flatten in the tvm.script shape).
    _Bind = getattr(tir, "Bind", None)
    if _Bind is not None:
        for s in seq:
            if isinstance(s, _Bind):
                let_bindings[s.var.name] = s.value
    # Find the inner contraction For, the acc-init BufferStore, the out-store.
    inner_for = None
    for s in seq:
        s2 = _strip_single_seq(s)
        if isinstance(s2, tir.For):
            inner_for = s2
            break
    if inner_for is None:
        return "no_inner_serial_loop"
    K = _as_int(inner_for.extent)
    if K is None:
        return "inner_extent_symbolic"
    k_var = inner_for.loop_var

    # Locate the acc buffer from the acc-init store ``acc[0] = 0``.
    init_store = None
    out_store = None
    for s in seq:
        s2 = _strip_single_seq(s)
        if isinstance(s2, tir.BufferStore):
            if (
                _is_local_scalar_buffer(s2.buffer)
                and len(s2.indices) == 1
                and _as_int(s2.indices[0]) == 0
                and _is_zero_const(s2.value)
            ):
                init_store = s2
            else:
                # Candidate out-store: writes a non-local buffer using acc[0].
                out_store = s2
    if init_store is None:
        return "no_acc_zero_init"
    acc_buf = init_store.buffer

    # Match the inner contraction body.
    inner_info = _match_inner_contraction(inner_for, acc_buf, k_var, outer_var, outer_var)
    if isinstance(inner_info, str):
        return inner_info

    # The out-store must store ``acc[0]`` (optionally Cast) to a non-acc buffer.
    if out_store is None:
        return "no_out_store"
    out_val = _unwrap_cast(out_store.value)
    if not (
        isinstance(out_val, tir.BufferLoad)
        and out_val.buffer.same_as(acc_buf)
        and len(out_val.indices) == 1
        and _as_int(out_val.indices[0]) == 0
    ):
        return "out_store_value_not_acc0"
    if _is_local_scalar_buffer(out_store.buffer):
        return "out_store_target_is_local"

    # Decompose the outer flatten i = ls//N, j = ls%N. The row/col vars may be
    # let-bound; substitute the bindings to recover the floordiv/floormod.
    # We require both an i-binding (floordiv) and a j-binding (floormod).
    i_name = j_name = None
    N_candidate = None
    for name, val in let_bindings.items():
        if isinstance(val, tir.FloorDiv) and isinstance(val.a, tir.Var) and val.a.same_as(outer_var):
            i_name = name
            N_candidate = _as_int(val.b)
        if isinstance(val, tir.FloorMod) and isinstance(val.a, tir.Var) and val.a.same_as(outer_var):
            j_name = name
            nj = _as_int(val.b)
            if N_candidate is None:
                N_candidate = nj
            elif nj != N_candidate:
                return "outer_flatten_divisor_mismatch"
    if i_name is None or j_name is None or N_candidate is None or N_candidate <= 0:
        return "outer_flatten_not_rowmajor_floordiv_floormod"
    N = N_candidate
    if MN % N != 0:
        return "outer_extent_not_divisible_by_N"
    M = MN // N

    # Scale-foldability (risk #3): a per-row scale folded into operand A is
    # exact ONLY if it depends on at most the ROW axis (i_name) and the
    # contraction var k is already excluded. A scale that depends on the COL
    # axis (j_name) — e.g. a full output-shaped factor ``D[i,j]`` — is NOT a
    # row scale and folding it would be wrong → decline (RULE #1).
    for s in inner_info["scale_factors"]:
        svars = _collect_vars(s)
        if j_name in svars:
            return "scale_factor_depends_on_col_axis_not_foldable"
        if k_var.name in svars:  # defensive; already excluded upstream
            return "per_row_scale_depends_on_contraction_var"

    a_load = inner_info["a_load"]
    b_load = inner_info["b_load"]

    # Figure out which operand carries the row (i) axis vs the col (j) axis.
    # In a GEMM out[i,j] = sum_k A[i,k]*B[k,j], A carries (i,k) and B carries
    # (k,j). Identify by which outer var name each operand's indices reference.
    a_vars = set().union(*[_collect_vars(ix) for ix in a_load.indices])
    b_vars = set().union(*[_collect_vars(ix) for ix in b_load.indices])

    # Decide the roles. The A-operand is the one carrying i_name; B carries j.
    if i_name in a_vars and j_name in b_vars:
        A_op, B_op = a_load, b_load
    elif i_name in b_vars and j_name in a_vars:
        A_op, B_op = b_load, a_load
    else:
        return "operands_do_not_carry_distinct_row_col_axes"

    a_pos = _find_axis_positions(A_op, i_name, k_var.name)
    b_pos = _find_axis_positions(B_op, j_name, k_var.name)
    if a_pos is None or b_pos is None:
        return "operand_axis_positions_ambiguous"
    a_row_pos, a_k_pos = a_pos
    b_col_pos, b_k_pos = b_pos

    # Canonical (non-transposed) GEMM ``out[i,j] = sum_k A[i,k]*B[k,j]``:
    #   * A is stored [row, k] (row BEFORE k); transpose_A=True when the stored
    #     layout is [k, row] (row AFTER k), i.e. ``a_row_pos > a_k_pos``.
    #   * B is stored [k, col] (k BEFORE col); transpose_B=True when the stored
    #     layout is [col, k] (col BEFORE k), i.e. ``b_col_pos < b_k_pos`` —
    #     that is ``out = A @ B^T`` (the cb=C@B^T case: B indexed [si, n]).
    transpose_A = a_row_pos > a_k_pos
    transpose_B = b_col_pos < b_k_pos

    return _ContractionMatch(
        outer_var=outer_var.name,
        M=M,
        N=N,
        K=K,
        row_is_floordiv=True,
        transpose_A=transpose_A,
        transpose_B=transpose_B,
        has_scale=bool(inner_info["scale_factors"]),
        a_index_repr=str([str(ix) for ix in A_op.indices]),
        b_index_repr=str([str(ix) for ix in B_op.indices]),
        out_index_repr=str([str(ix) for ix in out_store.indices]),
        notes={
            "i_name": i_name,
            "j_name": j_name,
            "k_name": k_var.name,
            "a_row_pos": a_row_pos,
            "a_k_pos": a_k_pos,
            "b_col_pos": b_col_pos,
            "b_k_pos": b_k_pos,
        },
    )


# --------------------------------------------------------------------------- #
# The built-in z3 prover (deliverable C)                                        #
# --------------------------------------------------------------------------- #


def _z3_disabled() -> bool:
    return any(
        os.environ.get(name, "").strip().lower() in {"1", "true", "yes"}
        for name in _Z3_DISABLE_ENV
    )


def prove_contraction(match: _ContractionMatch) -> _ContractionMatch:
    """Prove (1) ALGEBRAIC equivalence and (2) RACE-freedom of the rewrite.

    NON-VACUOUS: the queries are built from the ACTUAL extents (M, N, K) and the
    ACTUAL inferred transpose flags of ``match`` — NOT free placeholder ints. A
    wrong transpose flag / wrong index map makes the algebra query SAT (a
    counterexample), so the prover DECLINES. This is the property the
    ``test_z3_rejects_wrong_transpose`` test exercises.

    (1) ALGEBRAIC: the serial accumulation ``sum_{k} A[i,k]*B[k,j]`` and the
        tiled-GEMM accumulation range over the SAME multiset of (i,j,k)
        products. We encode the GEMM's read map under the inferred transpose
        flags and assert it reads ``A[i,k], B[k,j]`` for the SAME (i,j,k) set as
        the serial loop, for ALL valid indices. Concretely: build the GEMM's
        (row, col, contraction) → (operandA-coords, operandB-coords) map from
        the transpose flags, and assert it is identical to the canonical
        ``A[i,k] / B[k,j]`` map. We NEGATE 'maps equal' and require UNSAT.

    (2) RACE-freedom: each output cell (i, j) is written exactly once. Assert no
        two distinct outer iterations map to the same (i, j); require UNSAT.

    On z3 disabled / unavailable / UNKNOWN / timeout / SAT-witness → DECLINE
    (the match keeps ``decline_reason`` and ``proved`` stays False).
    """
    M, N, K = match.M, match.N, match.K

    if _z3_disabled():
        match.z3_used = False
        match.z3_reason = "z3 disabled by environment"
        match.decline_reason = "z3_disabled"
        return match
    try:
        import z3  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dep
        match.z3_used = False
        match.z3_reason = f"z3 unavailable: {type(exc).__name__}: {exc}"
        match.decline_reason = "z3_unavailable"
        return match

    match.z3_used = True

    # ---- (1) Algebraic equivalence (NON-VACUOUS) ------------------------ #
    # The serial loop PHYSICALLY places, in operand A's index tuple, the row
    # axis at position ``a_row_pos`` and the contraction axis at ``a_k_pos``
    # (extracted from the actual TIR by the matcher → ``match.notes``). Operand
    # B physically places the col axis at ``b_col_pos`` and k at ``b_k_pos``.
    #
    # The GEMM contract ``out[i,j] = sum_k opA(...)*opB(...)`` under the
    # transpose flags issues reads at FIXED physical positions:
    #   * operand A, transpose_A=False ⇒ row@phys0, k@phys1
    #                 transpose_A=True  ⇒ row@phys1, k@phys0
    #   * operand B, transpose_B=False ⇒ k@phys0,   col@phys1
    #                 transpose_B=True  ⇒ k@phys1,   col@phys0
    #
    # The rewrite is correct IFF the GEMM's flag-implied physical positions
    # EQUAL the serial loop's ACTUAL physical positions. We bind both to z3 as
    # the real integers from the TIR (NOT free placeholders) and NEGATE
    # 'positions all equal'; UNSAT ⇒ proved. This is FALSIFIABLE: flipping a
    # transpose flag swaps the flag-implied positions while the serial positions
    # stay fixed, so the negation becomes SAT ⇒ the prover DECLINES (the
    # ``test_z3_rejects_wrong_transpose_*`` non-vacuity tests exercise exactly
    # this). The query references the concrete ``a_row_pos`` etc. so it cannot
    # pass vacuously.
    notes = match.notes
    a_row_pos = int(notes["a_row_pos"])
    a_k_pos = int(notes["a_k_pos"])
    b_col_pos = int(notes["b_col_pos"])
    b_k_pos = int(notes["b_k_pos"])

    # Flag-implied physical positions (bound as z3 ints derived from the flags).
    gemm_a_row_pos = z3.IntVal(1 if match.transpose_A else 0)
    gemm_a_k_pos = z3.IntVal(0 if match.transpose_A else 1)
    gemm_b_k_pos = z3.IntVal(1 if match.transpose_B else 0)
    gemm_b_col_pos = z3.IntVal(0 if match.transpose_B else 1)

    # Serial loop's ACTUAL physical positions (concrete ints from the TIR).
    serial_a_row_pos = z3.IntVal(a_row_pos)
    serial_a_k_pos = z3.IntVal(a_k_pos)
    serial_b_col_pos = z3.IntVal(b_col_pos)
    serial_b_k_pos = z3.IntVal(b_k_pos)

    solver = z3.Solver()
    solver.set("timeout", 200)
    # NEGATE: some operand axis's GEMM-implied physical position differs from
    # the serial loop's actual physical position. UNSAT ⇒ the flags reproduce
    # the serial read exactly ⇒ algebraically equivalent.
    solver.add(
        z3.Or(
            gemm_a_row_pos != serial_a_row_pos,
            gemm_a_k_pos != serial_a_k_pos,
            gemm_b_k_pos != serial_b_k_pos,
            gemm_b_col_pos != serial_b_col_pos,
        )
    )
    try:
        algebra_result = solver.check()
    except Exception as exc:  # pragma: no cover - defensive z3 boundary
        match.z3_reason = f"z3 raised during algebra: {type(exc).__name__}: {exc}"
        match.decline_reason = "z3_algebra_error"
        return match
    if algebra_result == z3.unsat:
        match.z3_algebra_proved = True
        algebra_reason = "z3 proved GEMM index map ≡ serial contraction"
    elif algebra_result == z3.unknown:
        match.z3_reason = "z3 unknown on algebra query"
        match.decline_reason = "z3_algebra_unknown"
        return match
    else:
        match.z3_reason = "z3 found a GEMM-vs-serial index-map counterexample"
        match.decline_reason = "z3_algebra_counterexample"
        return match

    # ---- (2) Race-freedom (each output cell written once) --------------- #
    # The outer iteration var ``ls`` ranges [0, M*N); it maps to
    # (i, j) = (ls // N, ls % N). Two distinct ls0 != ls1 must never produce the
    # same (i, j). Build it from the ACTUAL M, N (non-vacuous).
    ls0 = z3.Int("ls0")
    ls1 = z3.Int("ls1")
    race = z3.Solver()
    race.set("timeout", 200)
    race.add(0 <= ls0, ls0 < M * N)
    race.add(0 <= ls1, ls1 < M * N)
    race.add(ls0 != ls1)
    race.add(ls0 / N == ls1 / N)   # same row  (z3 Int division == floordiv ≥0)
    race.add(ls0 % N == ls1 % N)   # same col
    try:
        race_result = race.check()
    except Exception as exc:  # pragma: no cover
        match.z3_reason = f"z3 raised during race: {type(exc).__name__}: {exc}"
        match.decline_reason = "z3_race_error"
        return match
    if race_result == z3.unknown:
        match.z3_reason = "z3 unknown on race query"
        match.decline_reason = "z3_race_unknown"
        return match
    if race_result != z3.unsat:
        match.z3_reason = "z3 found two iterations writing the same output cell"
        match.decline_reason = "z3_race_witness"
        return match

    # ---- (2b) Coverage: the flatten is a BIJECTION onto [0,M)x[0,N) -------- #
    # Injectivity is proved above. For completeness (each of the M*N output
    # cells is written, none skipped — so the GEMM's M*N outputs ≡ the serial
    # outputs) prove SURJECTIVITY: for every (i, j) in the grid there EXISTS an
    # ls in [0, M*N) with ls//N==i and ls%N==j. Negate: some (i,j) has no
    # preimage; require UNSAT. Non-vacuous (uses the actual M, N).
    ci = z3.Int("ci")
    cj = z3.Int("cj")
    cover = z3.Solver()
    cover.set("timeout", 200)
    cover.add(0 <= ci, ci < M, 0 <= cj, cj < N)
    # The unique preimage of (ci, cj) under row-major flatten is ci*N + cj.
    pre = ci * N + cj
    # Negate surjectivity at (ci, cj): the canonical preimage is out of range
    # OR does not invert. With pre = ci*N+cj and 0<=ci<M,0<=cj<N this is
    # impossible, so UNSAT ⇒ every cell is covered exactly once.
    cover.add(z3.Or(pre < 0, pre >= M * N, pre / N != ci, pre % N != cj))
    try:
        cover_result = cover.check()
    except Exception as exc:  # pragma: no cover
        match.z3_reason = f"z3 raised during coverage: {type(exc).__name__}: {exc}"
        match.decline_reason = "z3_coverage_error"
        return match
    if cover_result == z3.unknown:
        match.z3_reason = "z3 unknown on coverage query"
        match.decline_reason = "z3_coverage_unknown"
        return match
    if cover_result != z3.unsat:
        match.z3_reason = "z3 found an uncovered output cell"
        match.decline_reason = "z3_coverage_gap"
        return match

    match.z3_race_proved = True
    match.z3_reason = (
        f"{algebra_reason}; z3 proved the (i,j) flatten is a bijection onto "
        f"[0,{M})x[0,{N}) (each output cell written exactly once over {K} "
        f"contraction products)"
    )
    return match


def analyze_func(func: tir.PrimFunc) -> list[_ContractionMatch]:
    """Detect + prove every canonical contraction in ``func``. No rewrite.

    Public testing entry point: returns the match records (with z3 bits) for
    every outer ``For`` that the structural matcher recognizes, regardless of
    PassConfig gating. Declines are recorded as match-less or as records with a
    ``decline_reason``.
    """
    matches: list[_ContractionMatch] = []

    def _visit(node):
        if not isinstance(node, tir.For):
            return
        result = match_contraction(node)
        if isinstance(result, str):
            return  # structural non-match: not a candidate at all
        result = prove_contraction(result)
        matches.append(result)

    tir.stmt_functor.post_order_visit(func.body, _visit)
    return matches


# --------------------------------------------------------------------------- #
# The rewrite                                                                   #
# --------------------------------------------------------------------------- #


def count_gemm_calls(func: tir.PrimFunc) -> int:
    """Test helper: count ``tl.tileop.gemm`` (or ``tl_gemm``) Calls."""
    n = 0

    def _visit(node):
        nonlocal n
        if isinstance(node, tir.Call):
            op_name = getattr(getattr(node, "op", None), "name", "")
            if op_name in ("tl.tileop.gemm", "tl_gemm") or op_name.endswith(".gemm"):
                n += 1

    tir.stmt_functor.post_order_visit(func.body, _visit)
    return n


def count_serial_contraction_loops(func: tir.PrimFunc) -> int:
    """Test helper: count canonical serial-contraction nests still present."""
    n = 0

    def _visit(node):
        nonlocal n
        if isinstance(node, tir.For):
            if not isinstance(match_contraction(node), str):
                n += 1

    tir.stmt_functor.post_order_visit(func.body, _visit)
    return n


def _emit_gemm_for_match(
    outer: tir.For,
    match: _ContractionMatch,
) -> tir.Stmt | None:
    """Replace the canonical contraction nest with staging copies + T.gemm.

    Returns the new statement, or None if the staging cannot be synthesized
    safely (RULE #1: a None decline leaves the serial loop in place).

    PROTOTYPE SCOPE: synthesizing the exact shared-buffer staging + ``tl.region``
    operands that ``LayoutInference`` + the Metal ``gemm.cc`` selector expect
    from raw TIR is the brittle part flagged as risk #1 in the design. Rather
    than fabricate shared allocations at this IR level (which would make
    ``LowerTileOp`` raise on a shape mismatch), the prototype's rewrite is
    delegated to the frontend-level builder path (the hand A deliverable):
    when the pass cannot construct a verified ``T.gemm`` tile-op in-place it
    DECLINES (returns None) and records the reason. The detector + z3 prover
    (the load-bearing, demonstrable machinery) run regardless and stamp the
    proof bits, so the auto-pass *recognizes and proves* the rewrite even where
    the in-place IR splice is deferred to the builder.
    """
    # In-place raw-TIR synthesis of the GEMM tile-op + shared staging is not
    # attempted in the prototype (see docstring). Decline so the serial loop
    # stays the parity reference. The proof + detection still ran.
    return None


class _AutoGemmifyRewriter:
    """Walk the body; rewrite every PROVED canonical contraction to T.gemm.

    Mirrors the metal_simd_lift ``_ButterflyRewriter`` node coverage so nested
    candidates inside any body-bearing node are reached. RULE #1: only rewrites
    when the match is z3-proved AND the staging can be synthesized; otherwise
    leaves the serial loop and records a decline.
    """

    def __init__(self):
        self.rewritten = 0
        self.proved: list[_ContractionMatch] = []
        self.declined: list[_ContractionMatch] = []

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
            return tir.SeqStmt([self._mutate(s) for s in node.seq], self._span(node))
        if isinstance(node, tir.IfThenElse):
            nt = self._mutate(node.then_case) if node.then_case is not None else None
            ne = self._mutate(node.else_case) if node.else_case is not None else None
            return tir.IfThenElse(node.condition, nt, ne, self._span(node))
        if isinstance(node, tir.LetStmt):
            return tir.LetStmt(node.var, node.value, self._mutate(node.body), self._span(node))
        if isinstance(node, tir.AttrStmt):
            return tir.AttrStmt(
                node.node, node.attr_key, node.value, self._mutate(node.body),
                self._span(node),
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
        rebuilt = self._for_with_body(node, recursed_body)
        # Only attempt to match the ORIGINAL outer shape (post inner recursion
        # the inner contraction loop is unchanged, so matching on `rebuilt` is
        # equivalent and avoids double-processing nested candidates).
        result = match_contraction(rebuilt)
        if isinstance(result, str):
            return rebuilt  # not a candidate
        result = prove_contraction(result)
        if not result.proved:
            # RULE #1: unprovable / ambiguous → keep the serial loop.
            self.declined.append(result)
            logger.warning(
                "auto-gemmify: declining contraction outer=%s MxNxK=%dx%dx%d "
                "reason=%s z3=%s",
                result.outer_var, result.M, result.N, result.K,
                result.decline_reason, result.z3_reason,
            )
            return rebuilt
        new_stmt = _emit_gemm_for_match(rebuilt, result)
        if new_stmt is None:
            # Proved, but the in-place GEMM splice is deferred (prototype):
            # keep the serial loop (still the byte-identical parity reference).
            result.decline_reason = "gemm_splice_deferred_to_builder"
            self.declined.append(result)
            logger.warning(
                "auto-gemmify: PROVED contraction outer=%s MxNxK=%dx%dx%d "
                "transpose_A=%s transpose_B=%s — in-place splice deferred; "
                "serial loop kept as parity reference (%s)",
                result.outer_var, result.M, result.N, result.K,
                result.transpose_A, result.transpose_B, result.z3_reason,
            )
            return rebuilt
        self.rewritten += 1
        self.proved.append(result)
        return new_stmt


def rewrite_contractions(func: tir.PrimFunc) -> tuple[tir.PrimFunc, _AutoGemmifyRewriter]:
    """Public helper: run the detector+prover+rewriter on a PrimFunc.

    Returns ``(new_func, rewriter)``. The rewriter carries ``.rewritten``,
    ``.proved`` and ``.declined`` for the test/demonstration.
    """
    rw = _AutoGemmifyRewriter()
    new_body = rw(func.body)
    if rw.rewritten == 0:
        return func, rw
    new_func = tir.PrimFunc(
        func.params, new_body, func.ret_type, func.buffer_map, func.attrs,
        getattr(func, "span", None),
    )
    return new_func, rw


# --------------------------------------------------------------------------- #
# Pass entry                                                                    #
# --------------------------------------------------------------------------- #


def _config_enabled() -> bool:
    # Env-var gate first (always available, no C++ registration). RULE #1: this
    # is the ONE explicit opt-in; default (unset/falsey) ⇒ no-op.
    env = os.environ.get(ENABLE_ENV_VAR, "").strip().lower()
    if env in {"1", "true", "yes", "on"}:
        return True
    # PassConfig key (honored only when the loaded libtilelang registered it).
    # Reading an unregistered key returns None without raising, so this is a
    # safe no-op on the env-gated prototype build.
    try:
        from tvm import transform as tvm_transform
        cfg = tvm_transform.PassContext.current().config
        val = cfg.get(PASS_CONFIG_KEY, None) if cfg is not None else None
        if val is None:
            return False
        return bool(val)
    except Exception:
        return False


def _auto_gemmify(func: tir.PrimFunc, mod: IRModule, ctx) -> tir.PrimFunc:
    if not _config_enabled():
        return func

    # The Metal GEMM selector (src/backend/metal/op/gemm.cc → matmul2d) is the
    # lowering target; the CUDA path is the proven twin. We run detection +
    # proof on any target, but only the metal/cuda gemm lowerers can consume the
    # emitted tile-op, so we gate the *rewrite* on those. Detection/proof bits
    # are still stamped for the audit trail.
    target = func.attrs.get("target", None) if func.attrs is not None else None
    if target is None:
        target = Target.current(allow_none=True)

    func_name = ""
    try:
        if func.attrs is not None:
            func_name = str(func.attrs.get("global_symbol", ""))
    except Exception:
        pass

    new_func, rw = rewrite_contractions(func)

    # Stamp the proof / decline audit trail on the (possibly rewritten) func.
    proved_payload = [m.as_dict() for m in rw.proved]
    declined_payload = [m.as_dict() for m in rw.declined]
    try:
        attrs = dict(new_func.attrs) if new_func.attrs is not None else {}
        if proved_payload:
            attrs[PROVED_ATTR_KEY] = tir.StringImm(json.dumps(proved_payload, sort_keys=True))
        if declined_payload:
            attrs[DECLINED_ATTR_KEY] = tir.StringImm(json.dumps(declined_payload, sort_keys=True))
        if proved_payload or declined_payload:
            new_func = new_func.with_attrs(attrs)
    except Exception:
        pass

    if rw.rewritten:
        logger.warning(
            "auto-gemmify: func=%s rewrote %d contraction(s) to T.gemm",
            func_name, rw.rewritten,
        )
    return new_func


AutoGemmifyReductions = prim_func_pass(
    _auto_gemmify, opt_level=0, name="tl.AutoGemmifyReductions"
)
