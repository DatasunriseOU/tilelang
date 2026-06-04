"""Tests for Track B: serial-reduction → T.gemm auto-rewrite pass.

These tests exercise the AUTOMATIC ``AutoGemmifyReductions`` pass that detects
the canonical *contraction* nest (the shape the mamba3 F0 Metal prim writes for
``cb`` and ``summary_states``) and, gated behind a built-in z3 proof, recognizes
the rewrite to a tile-level ``T.gemm``.

The load-bearing properties under test:

* The structural matcher recognizes the canonical ``for ls in Parallel(M*N):
  acc=0; for k in serial(K): acc += A[..]*B[..]; out=acc`` nest and infers the
  GEMM dims + transpose flags.
* The built-in z3 prover is NON-VACUOUS: a deliberately-wrong transpose flag
  makes the algebra query find a counterexample and the prover DECLINES.
* RULE #1: on an unprovable / ambiguous / non-canonical pattern the pass LEAVES
  the serial loop untouched (no wrong rewrite).
* Default OFF: with the PassConfig absent the module pass is a no-op.

The tests use the public helpers from the pass module and do NOT require a
built libtilelang or a Metal device (they inspect the matched/proved IR).
"""

from __future__ import annotations

import importlib.util
import os
import sys

import pytest

import tvm
from tvm.script import tir as T
from tvm import tir
from tvm.target import Target

# NOTE: the test builds the canonical-contraction PrimFuncs via
# ``tvm.script.tir`` rather than the ``tilelang.language`` eager frontend. The
# ``tilelang.language`` package has an unrelated, pre-existing circular import
# on this merge branch (tilelang.language.eager <-> tilelang.jit) that blocks
# importing it under the matching cpython here; the matcher operates on the
# resulting ``tir`` nodes either way. ``tvm.script.tir`` emits the SAME logical
# shape (Bind row/col flatten, size-1 local AllocBuffer accumulator, serial-K
# contraction, acc-out store) that the tilelang F0 prim lowers to, and the
# matcher is written to accept both encodings.


def _load_module():
    here = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.normpath(
        os.path.join(here, "..", "..", "..", "tilelang", "transform",
                     "auto_gemmify_reductions.py")
    )
    spec = importlib.util.spec_from_file_location(
        "_worktree_auto_gemmify_reductions", candidate
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


agm = _load_module()
match_contraction = agm.match_contraction
prove_contraction = agm.prove_contraction
analyze_func = agm.analyze_func
rewrite_contractions = agm.rewrite_contractions
count_serial_contraction_loops = agm.count_serial_contraction_loops
PASS_CONFIG_KEY = agm.PASS_CONFIG_KEY


# --------------------------------------------------------------------------- #
# Canonical-shape builders mirroring the F0 Metal prim loops                    #
# --------------------------------------------------------------------------- #


def _build_cb_contraction(L: int = 32, Ndim: int = 16):
    """Mirror F0 ``cb = C @ B^T``: out[li,si] = sum_n C[li,n]*B[si,n].

    Operand layout: A=C indexed [li, n] (row,k) → transpose_A=False;
    B indexed [si, n] (col,k) → transpose_B=True (out = A @ B^T).
    """

    @T.prim_func
    def func(
        Cbuf: T.Buffer((L, Ndim), "float32"),
        Bbuf: T.Buffer((L, Ndim), "float32"),
        out: T.Buffer((L, L), "float32"),
    ):
        for ls in range(L * L):
            li = ls // L
            si = ls % L
            acc = T.alloc_buffer((1,), "float32", scope="local")
            acc[0] = T.float32(0)
            for n in range(Ndim):
                acc[0] = acc[0] + Cbuf[li, n] * Bbuf[si, n]
            out[li, si] = acc[0]

    return func


def _build_summary_contraction(P: int = 16, Ndim: int = 16, L: int = 32):
    """Mirror F0 ``summary_states = (decay·dt·x)^T @ B`` (no-mask clean shape).

    out[pp,nn] = sum_l x[l,pp] * B[l,nn]. Here A=x indexed [l, pp] (k,row) →
    transpose_A=True; B indexed [l, nn] (k,col) → transpose_B=False.
    """

    @T.prim_func
    def func(
        xbuf: T.Buffer((L, P), "float32"),
        Bbuf: T.Buffer((L, Ndim), "float32"),
        out: T.Buffer((P, Ndim), "float32"),
    ):
        for pn in range(P * Ndim):
            pp = pn // Ndim
            nn = pn % Ndim
            acc = T.alloc_buffer((1,), "float32", scope="local")
            acc[0] = T.float32(0)
            for l in range(L):
                acc[0] = acc[0] + xbuf[l, pp] * Bbuf[l, nn]
            out[pp, nn] = acc[0]

    return func


def _build_scaled_contraction(P: int = 16, Ndim: int = 16, L: int = 32):
    """summary_states with a k-INDEPENDENT per-row scale s[pp] folded in.

    out[pp,nn] = sum_l (s[pp] * x[l,pp]) * B[l,nn]. The scale s[pp] does NOT
    depend on the contraction var l, so the matcher must accept it.
    """

    @T.prim_func
    def func(
        xbuf: T.Buffer((L, P), "float32"),
        Bbuf: T.Buffer((L, Ndim), "float32"),
        sbuf: T.Buffer((P,), "float32"),
        out: T.Buffer((P, Ndim), "float32"),
    ):
        for pn in range(P * Ndim):
            pp = pn // Ndim
            nn = pn % Ndim
            acc = T.alloc_buffer((1,), "float32", scope="local")
            acc[0] = T.float32(0)
            for l in range(L):
                acc[0] = acc[0] + sbuf[pp] * xbuf[l, pp] * Bbuf[l, nn]
            out[pp, nn] = acc[0]

    return func


def _build_k_dependent_scale(P: int = 16, Ndim: int = 16, L: int = 32):
    """NON-canonical: a scale that depends on the contraction var l.

    out[pp,nn] = sum_l (w[l] * x[l,pp]) * B[l,nn]. ``w[l]`` IS k-dependent, so
    folding it into a single operand is NOT exact — the matcher must DECLINE.
    """

    @T.prim_func
    def func(
        xbuf: T.Buffer((L, P), "float32"),
        Bbuf: T.Buffer((L, Ndim), "float32"),
        wbuf: T.Buffer((L,), "float32"),
        out: T.Buffer((P, Ndim), "float32"),
    ):
        for pn in range(P * Ndim):
            pp = pn // Ndim
            nn = pn % Ndim
            acc = T.alloc_buffer((1,), "float32", scope="local")
            acc[0] = T.float32(0)
            for l in range(L):
                acc[0] = acc[0] + wbuf[l] * xbuf[l, pp] * Bbuf[l, nn]
            out[pp, nn] = acc[0]

    return func


def _build_three_operand(M: int = 8, Ndim: int = 8, K: int = 8):
    """NON-canonical: a 3-operand product (not a 2-operand contraction)."""

    @T.prim_func
    def func(
        Abuf: T.Buffer((M, K), "float32"),
        Bbuf: T.Buffer((K, Ndim), "float32"),
        Dbuf: T.Buffer((M, Ndim), "float32"),
        out: T.Buffer((M, Ndim), "float32"),
    ):
        for ls in range(M * Ndim):
            i = ls // Ndim
            j = ls % Ndim
            acc = T.alloc_buffer((1,), "float32", scope="local")
            acc[0] = T.float32(0)
            for k in range(K):
                acc[0] = acc[0] + Abuf[i, k] * Bbuf[k, j] * Dbuf[i, j]
            out[i, j] = acc[0]

    return func


def _build_plain_reduction(N: int = 64):
    """NON-canonical: a 1D reduction (sum of a vector) — NOT a contraction."""

    @T.prim_func
    def func(buf: T.Buffer((N,), "float32"), acc: T.Buffer((1,), "float32")):
        for i in range(N):
            acc[0] = acc[0] + buf[i]

    return func


# --------------------------------------------------------------------------- #
# Structural matcher + GEMM-dim/transpose inference                            #
# --------------------------------------------------------------------------- #


def _outer_for(func: tir.PrimFunc) -> tir.For:
    found = []

    def _v(node):
        if isinstance(node, tir.For):
            found.append(node)

    tir.stmt_functor.post_order_visit(func.body, _v)
    # The outermost For is the last visited in post-order at depth 0; just pick
    # the one whose match succeeds.
    for f in found:
        if not isinstance(match_contraction(f), str):
            return f
    return found[-1]


def test_cb_contraction_matches_and_infers_transpose_B():
    func = _build_cb_contraction(L=32, Ndim=16)
    m = match_contraction(_outer_for(func))
    assert not isinstance(m, str), f"expected a match, got decline: {m}"
    assert (m.M, m.N, m.K) == (32, 32, 16)
    # cb = C @ B^T : A indexed [li,n] (row,k) ⇒ not transposed; B indexed
    # [si,n] (col,k) ⇒ transpose_B True.
    assert m.transpose_A is False
    assert m.transpose_B is True
    assert m.has_scale is False


def test_summary_contraction_matches_and_infers_transpose_A():
    func = _build_summary_contraction(P=16, Ndim=16, L=32)
    m = match_contraction(_outer_for(func))
    assert not isinstance(m, str), f"expected a match, got decline: {m}"
    assert (m.M, m.N, m.K) == (16, 16, 32)
    # summary = x^T @ B : A=x indexed [l,pp] (k,row) ⇒ transpose_A True;
    # B indexed [l,nn] (k,col) ⇒ not transposed.
    assert m.transpose_A is True
    assert m.transpose_B is False


def test_scaled_contraction_folds_k_independent_scale():
    func = _build_scaled_contraction()
    m = match_contraction(_outer_for(func))
    assert not isinstance(m, str), f"expected a match, got decline: {m}"
    assert m.has_scale is True


# --------------------------------------------------------------------------- #
# RULE #1: non-canonical / unsafe shapes DECLINE (no wrong rewrite)            #
# --------------------------------------------------------------------------- #


def test_k_dependent_scale_declines():
    func = _build_k_dependent_scale()
    m = match_contraction(_outer_for(func))
    assert isinstance(m, str), "k-dependent scale must structurally decline"
    # ``w[l]`` is a k-ONLY extra load — not a clean 2D operand nor a foldable
    # per-row scale → RULE #1 decline (no wrong rewrite).
    assert m == "per_row_scale_depends_on_contraction_var"


def test_three_operand_product_declines():
    func = _build_three_operand()
    m = match_contraction(_outer_for(func))
    assert isinstance(m, str)
    # ``D[i,j]`` is a full output-shaped factor depending on the COL axis — not
    # a foldable per-row scale → RULE #1 decline.
    assert m == "scale_factor_depends_on_col_axis_not_foldable"


def test_plain_1d_reduction_is_not_a_contraction():
    func = _build_plain_reduction()
    # No outer M*N flatten, no two-operand product → not matched at all.
    matches = analyze_func(func)
    assert matches == []


# --------------------------------------------------------------------------- #
# Built-in z3 proof (deliverable C) — and its NON-VACUITY                       #
# --------------------------------------------------------------------------- #


def test_z3_proves_canonical_contraction():
    func = _build_summary_contraction()
    m = match_contraction(_outer_for(func))
    assert not isinstance(m, str)
    m = prove_contraction(m)
    assert m.z3_used is True
    assert m.z3_algebra_proved is True
    assert m.z3_race_proved is True
    assert m.proved is True
    assert m.decline_reason is None


def test_z3_rejects_wrong_transpose_flag_non_vacuity():
    """NON-VACUITY: flip a CORRECT transpose flag → z3 must find a counterexample.

    This is the load-bearing anti-theater test: if the algebra query were
    vacuous it would pass regardless of the flag. We corrupt the inferred
    ``transpose_A`` and assert the prover DECLINES with an algebra
    counterexample (the rewrite would be wrong, so it must not be 'proved').
    """
    func = _build_summary_contraction()
    m = match_contraction(_outer_for(func))
    assert not isinstance(m, str)
    # Sanity: with M != K the swap is observable. summary has M=16, K=32.
    assert m.M != m.K
    # Corrupt: the true transpose_A is True; flip it to False.
    assert m.transpose_A is True
    m.transpose_A = False
    m = prove_contraction(m)
    assert m.z3_used is True
    assert m.z3_algebra_proved is False
    assert m.proved is False
    assert m.decline_reason == "z3_algebra_counterexample"


def test_z3_rejects_wrong_transpose_B_non_vacuity():
    func = _build_cb_contraction(L=32, Ndim=16)
    m = match_contraction(_outer_for(func))
    assert not isinstance(m, str)
    # cb: M=N=32, K=16. true transpose_B is True; flip it.
    assert m.transpose_B is True
    m.transpose_B = False
    m = prove_contraction(m)
    assert m.z3_algebra_proved is False
    assert m.decline_reason == "z3_algebra_counterexample"


def test_z3_disabled_env_declines_never_rewrites(monkeypatch):
    """With z3 disabled the pass DECLINES (fail-to-serial) — never rewrites."""
    monkeypatch.setenv("TILELANG_DISABLE_Z3_AUTO_GEMMIFY", "1")
    func = _build_summary_contraction()
    m = match_contraction(_outer_for(func))
    assert not isinstance(m, str)
    m = prove_contraction(m)
    assert m.proved is False
    assert m.decline_reason == "z3_disabled"


# --------------------------------------------------------------------------- #
# RULE #1 end-to-end: the rewriter keeps the serial loop on a proved-but-       #
# unspliceable match, and the module pass is OFF by default.                    #
# --------------------------------------------------------------------------- #


def test_rewriter_keeps_serial_loop_as_parity_reference():
    """The detector+prover run; the in-place splice is deferred (prototype) so
    the serial loop survives byte-identical as the parity reference (RULE #1).
    """
    func = _build_summary_contraction()
    new_func, rw = rewrite_contractions(func)
    # No in-place splice in the prototype → serial loop preserved.
    assert rw.rewritten == 0
    assert count_serial_contraction_loops(new_func) == 1
    # But the contraction WAS detected and PROVED (the audit trail).
    assert len(rw.declined) == 1
    proved_match = rw.declined[0]
    assert proved_match.proved is True
    assert proved_match.decline_reason == "gemm_splice_deferred_to_builder"


def test_declined_match_for_unprovable_records_reason():
    """A structurally-matched but z3-unprovable contraction is declined with a
    z3 reason — never silently rewritten."""
    func = _build_summary_contraction()
    outer = _outer_for(func)
    # Build a deliberately-corrupted prove path by disabling z3.
    m = match_contraction(outer)
    assert not isinstance(m, str)
    os.environ["TILELANG_DISABLE_Z3_AUTO_GEMMIFY"] = "1"
    try:
        m = prove_contraction(m)
        assert m.proved is False
        assert m.decline_reason == "z3_disabled"
    finally:
        del os.environ["TILELANG_DISABLE_Z3_AUTO_GEMMIFY"]


def test_default_off_module_pass_is_noop():
    func = _build_summary_contraction().with_attr("global_symbol", "main")
    mod = tvm.IRModule.from_expr(func)
    out = agm.AutoGemmifyReductions(mod)
    tvm.ir.assert_structural_equal(out["main"], mod["main"], True)


def test_pass_on_with_env_gate_stamps_proof_attr(monkeypatch):
    """With the env gate ON, the pass stamps the z3 proof audit trail attr even
    when the in-place splice is deferred (so downstream tooling can see the
    recognized+proved rewrite).

    NOTE: the env gate (not the PassConfig key) is the prototype's enable path —
    the PassConfig key requires a ``TVM_REGISTER_PASS_CONFIG_OPTION`` C++ build
    which would clobber the live build/, so the prototype gates via env var.
    """
    monkeypatch.setenv(agm.ENABLE_ENV_VAR, "1")
    func = _build_summary_contraction().with_attr("global_symbol", "main")
    mod = tvm.IRModule.from_expr(func)
    with Target("metal"):
        out = agm.AutoGemmifyReductions(mod)
    attrs = out["main"].attrs
    # Either proved or declined trail must be present and reference the proof.
    assert (agm.DECLINED_ATTR_KEY in attrs) or (agm.PROVED_ATTR_KEY in attrs)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
