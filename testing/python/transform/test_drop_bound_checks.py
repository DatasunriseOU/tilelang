"""Z3 roadmap idea #4 — DropProvableBoundChecks pass.

Tests that the pass collapses `IfThenElse(i < N, body)` guards when the
analyzer (default + Z3 fallback) can conclusively prove the condition, and
that the pass is conservative under both ambiguous (`i < N+1`) and
overflow-prone (`ext = ceildiv(N,K)*K` near INT_MAX) conditions.

The pass is gated by PassConfig `tl.drop_provable_bound_checks` (default
OFF) so the default-OFF test is structurally important: the pass must be a
no-op without the flag.
"""
from tilelang import tvm as tvm
import tilelang as tl
import tilelang.language as T


def _has_if_with_lt(stmt):
    """Return True iff any IfThenElse in `stmt` has a `<`/`<=` condition."""
    found = [False]

    def visit(s):
        if isinstance(s, tvm.tirx.IfThenElse):
            cond = s.condition
            if isinstance(cond, (tvm.tirx.LT, tvm.tirx.LE)):
                found[0] = True
            elif isinstance(cond, tvm.tirx.And):
                # Any conjunct an LT/LE is enough.
                stack = [cond]
                while stack:
                    c = stack.pop()
                    if isinstance(c, tvm.tirx.And):
                        stack.append(c.a)
                        stack.append(c.b)
                    elif isinstance(c, (tvm.tirx.LT, tvm.tirx.LE)):
                        found[0] = True
                        break

    tvm.tirx.stmt_functor.post_order_visit(stmt, visit)
    return found[0]


def _run(prim_func, *, enable: bool):
    mod = tvm.IRModule.from_expr(prim_func.with_attr("global_symbol", "main"))
    config = {"tl.drop_provable_bound_checks": True} if enable else {}
    with tvm.transform.PassContext(config=config):
        mod = tl.transform.DropProvableBoundChecks()(mod)
    return mod["main"]


def test_static_bound_check_dropped():
    """Static `for i in range(64): if i < 64: ...` — default analyzer
    proves; bound check is dropped."""

    @T.prim_func
    def before(A: T.Tensor((64,), T.float32), B: T.Tensor((64,), T.float32)):
        for i in range(64):
            if i < 64:
                B[i] = A[i]

    after = _run(before, enable=True)
    assert not _has_if_with_lt(after.body), (
        "Static `i < 64` guard should have been dropped:\n" + str(after))


def test_symbolic_bound_check_dropped_via_z3():
    """Symbolic `for i in range(N): if i < N: ...` — default analyzer with
    kSymbolicBound (or Z3 fallback) proves; check dropped."""
    N = T.symbolic("N", "int32")

    @T.prim_func
    def before(A: T.handle, B: T.handle, N_: T.int32):
        Ab = T.match_buffer(A, (N_,), dtype="float32")
        Bb = T.match_buffer(B, (N_,), dtype="float32")
        for i in range(N_):
            if i < N_:
                Bb[i] = Ab[i]

    after = _run(before, enable=True)
    assert not _has_if_with_lt(after.body), (
        "Symbolic `i < N` guard should have been dropped:\n" + str(after))


def test_uncertain_bound_keeps_guard():
    """`for i in range(N): if i < N+1: ...` — `i < N+1` *is* always true
    given `0 <= i < N`, so the guard IS provably-redundant. To test the
    "guard kept" path we instead use `if i < N - 1`, which can fail at
    `i = N - 1`. Both analyzers must conservatively keep this."""

    @T.prim_func
    def before(A: T.handle, B: T.handle, N_: T.int32):
        Ab = T.match_buffer(A, (N_,), dtype="float32")
        Bb = T.match_buffer(B, (N_,), dtype="float32")
        for i in range(N_):
            if i < N_ - 1:
                Bb[i] = Ab[i]

    after = _run(before, enable=True)
    assert _has_if_with_lt(after.body), (
        "`i < N - 1` is NOT provable for the last iteration — guard must be"
        " kept:\n" + str(after))


def test_default_off_preserves_behavior():
    """With config OFF (default), DropProvableBoundChecks is a no-op."""

    @T.prim_func
    def before(A: T.Tensor((64,), T.float32), B: T.Tensor((64,), T.float32)):
        for i in range(64):
            if i < 64:
                B[i] = A[i]

    after = _run(before, enable=False)
    assert _has_if_with_lt(after.body), (
        "With pass config OFF, the bound-check guard must be preserved:\n"
        + str(after))


def test_overflow_near_intmax_keeps_guard():
    """Extents like `ceildiv(N,K)*K` near INT_MAX: an unconstrained Int Z3
    might overprove `i < ceildiv(N,K)*K + 1` because in Z3's unbounded
    integer logic that's a tautology. Under the BV32-emulated free-var
    constraints (`0 <= v < 2^31`) the expression is no longer always-true
    near the top-of-range, so the guard MUST stay.

    We model this as: `for i in range(0, ceildiv(N,K)*K)` with the guard
    `i < ceildiv(N,K)*K + 1` — slightly broader than the loop range. This
    is conceptually safe in unbounded ints but unsound near INT_MAX where
    `ceildiv(N,K)*K + 1` may overflow."""

    @T.prim_func
    def before(A: T.handle, B: T.handle, N_: T.int32, K_: T.int32):
        Ab = T.match_buffer(A, (N_,), dtype="float32")
        Bb = T.match_buffer(B, (N_,), dtype="float32")
        # ceildiv(N, K) * K — uses TIR floordiv builtin
        ext = ((N_ + K_ - 1) // K_) * K_
        for i in range(ext):
            # Note: NOT a strict subset relation in BV32 — must be kept.
            if i < ext + 1:
                Bb[i] = Ab[i]

    after = _run(before, enable=True)
    # Conservative: the BV32-emulated Z3 should not prove this tautology
    # because of overflow risk at the upper end of int32. The guard MUST
    # remain in the IR.
    assert _has_if_with_lt(after.body), (
        "Near-INT_MAX overflow risk — guard must be kept:\n" + str(after))


if __name__ == "__main__":
    import tilelang.testing
    tilelang.testing.main()
