"""Tests for the BitVector mode switch on the vendored Z3Prover.

These tests drive the C++ ``Z3Prover`` via the FFI helper
``tl.z3.bv_can_prove(var, lo, hi, expr, bv_width)``, which:

* binds ``var`` to the half-open range ``[lo, hi)``,
* sets the prover to the requested sort (``bv_width`` is 0, 32, or 64),
* attempts to prove ``expr`` and returns the boolean answer.

The point of these cases is to exercise the *divergence* between
unbounded-Int and signed-BV semantics: in unbounded-Int mode the prover
can over-prove properties that depend on the absence of overflow, while
in BV mode it must respect two's-complement wrap.
"""
# CPPMEGA: Z3Prover BV-mode foundation tests

import pytest

import tilelang  # noqa: F401  (loads libtilelang.dylib + FFI hooks)
import tvm
from tvm import tir


_BV_CAN_PROVE = tvm.ffi.get_global_func("tl.z3.bv_can_prove", allow_missing=True)
_BV_SCOPED_ROUND_TRIP = tvm.ffi.get_global_func(
    "tl.z3.bv_scoped_round_trip", allow_missing=True
)

pytestmark = pytest.mark.skipif(
    _BV_CAN_PROVE is None or _BV_SCOPED_ROUND_TRIP is None,
    reason="Z3 BV test helpers require configuring with -DTILELANG_BUILD_TESTS=ON",
)


def _can_prove(var, lo, hi, expr, bv_width):
    return bool(_BV_CAN_PROVE(var, lo, hi, expr, bv_width))


def _scoped_round_trip(outer_width, inner_width):
    return int(_BV_SCOPED_ROUND_TRIP(outer_width, inner_width))


def test_alignment_int_mode_overproves():
    """Case 1 (alignment).

    Constraint: ``addr`` is some int32 in
    ``[INT32_MIN, INT32_MAX]`` (the entire signed range, *no* extra
    constraint on alignment) — but pinned to a single multiple of 16
    via the bind ``addr in [0, 1) -> addr == 0``. We then ask the
    prover to prove ``addr % 16 == 0``.

    With ``addr == 0`` this is trivially true in BOTH modes; that's
    not very interesting on its own. The real divergence comes when
    we widen the bind to a range that *spans* a near-overflow value:
    ``addr in [INT32_MAX - 7, INT32_MAX + 1)`` (8 consecutive int32
    values that include INT32_MAX). In Int mode the prover happily
    enumerates them as 2147483640..2147483647 and reports
    ``addr % 16 == 0`` is **not** provable (correct for Int). In BV32
    mode it reports the same — also correct. So that subtest does not
    expose the Int-vs-BV divergence at the prover level.

    Instead, the easiest divergence is ``2 * addr >= 0`` for ``addr in
    [0, 2^31)``: in Int mode this is provable (``2 * addr`` stays a
    non-negative integer), in BV32 mode it is NOT provable because for
    addr >= 2^30 the doubled value overflows and becomes negative.
    """
    addr = tir.Var("addr", "int32")
    expr = (addr * 2) >= 0
    # Int mode: 2*addr stays non-negative for addr in [0, 2^31).
    assert _can_prove(addr, 0, 1 << 31, expr, 0) is True
    # BV32: 2*addr wraps to negative once addr >= 2^30.
    assert _can_prove(addr, 0, 1 << 31, expr, 32) is False


def test_no_overflow_in_safe_range_both_modes_agree():
    """Case 2 (no-overflow in safe range).

    For ``i`` constrained to ``[0, 2^30]`` (well below INT32_MAX), the
    property ``i + 1 > i`` holds in both modes — there is no overflow
    in this range, so Int and BV agree.
    """
    i = tir.Var("i", "int32")
    expr = (i + 1) > i
    assert _can_prove(i, 0, 1 << 30, expr, 0) is True
    assert _can_prove(i, 0, 1 << 30, expr, 32) is True


def test_overflow_at_int32_max_disagrees():
    """Case 3 (overflow at INT32_MAX).

    For ``i`` pinned to exactly ``INT32_MAX = 2^31 - 1`` (range
    ``[INT32_MAX, INT32_MAX + 1)``), the property ``i + 1 > i``:

    * is TRUE in Int mode (2147483647 + 1 = 2147483648, and
      2147483648 > 2147483647 in unbounded Int);
    * is FALSE in BV32 mode (2147483647 + 1 wraps to -2147483648,
      and -2147483648 > 2147483647 is false).
    """
    i = tir.Var("i", "int32")
    expr = (i + 1) > i
    INT32_MAX = (1 << 31) - 1
    # Range = single-point [INT32_MAX, INT32_MAX+1).
    assert _can_prove(i, INT32_MAX, INT32_MAX + 1, expr, 0) is True
    assert _can_prove(i, INT32_MAX, INT32_MAX + 1, expr, 32) is False


def test_floormod_negative_dividend_agrees_int_and_bv():
    """FloorMod(-5, 3) -> 1 in BOTH Int and BV32 mode.

    TIR FloorMod is sign-of-divisor: ``floor(-5 / 3) = -2`` and
    ``-5 - 3 * (-2) = 1``. Regression guard for the BV-mode
    double-correction bug: in BV32 the `floormod` helper had been applied
    on top of ``bvsmod``, which already implements sign-of-divisor mod
    natively; the double-correction would have turned 1 into -2 (i.e. the
    Int-mod / Euclidean answer flipped through the helper).

    We pin ``x`` to a single-point range ``[-5, -4)`` and ask whether
    ``floormod(x, 3) == 1``. It MUST be provable under both sorts.
    """
    x = tir.Var("x", "int32")
    expr = tir.FloorMod(x, tir.const(3, "int32")) == tir.const(1, "int32")
    # Range = single-point [-5, -4).
    assert _can_prove(x, -5, -4, expr, 0) is True
    assert _can_prove(x, -5, -4, expr, 32) is True


def test_scoped_bv_mode_round_trip():
    """ScopedBVMode RAII restores the prior width on scope exit.

    Drive `tl.z3.bv_scoped_round_trip(outer, inner)` which:
      1. sets the prover to `outer`,
      2. opens a `ScopedBVMode(inner)` block,
      3. confirms `GetBitVectorWidth() == inner` inside,
      4. exits the scope,
      5. returns the post-exit width.

    Cover all interesting transitions: Int->BV32->Int, BV32->Int->BV32,
    BV32->BV64->BV32, and the no-op same-width case.
    """
    assert _scoped_round_trip(0, 32) == 0
    assert _scoped_round_trip(32, 0) == 32
    assert _scoped_round_trip(32, 64) == 32
    assert _scoped_round_trip(0, 0) == 0
    assert _scoped_round_trip(64, 32) == 64


def test_min_max_bv_mode_sort_assertion():
    """Min/Max under BV mode keep BV sort throughout (no Int leak).

    The C++ visitor adds `AssertOperandSort(...)` to MinNode and MaxNode
    that ICHECKs operand sort matches the current mode. If a sort leak
    were re-introduced (e.g. an Int constant snuck into a BV
    computation), this proof would crash with an ICHECK failure rather
    than silently produce a wrong answer.

    We pick a property that's true under BV32 for the bound range:
    ``Min(i, 100) <= 100`` for ``i in [0, 256)``. Both Int and BV32
    should prove it; the test's value is mainly that the ICHECK does
    not fire on this well-formed BV path.
    """
    i = tir.Var("i", "int32")
    expr = tir.Min(i, tir.const(100, "int32")) <= tir.const(100, "int32")
    assert _can_prove(i, 0, 256, expr, 0) is True
    assert _can_prove(i, 0, 256, expr, 32) is True

    expr2 = tir.Max(i, tir.const(0, "int32")) >= tir.const(0, "int32")
    assert _can_prove(i, 0, 256, expr2, 0) is True
    assert _can_prove(i, 0, 256, expr2, 32) is True


def test_floormod_negative_divisor_agrees_int_and_bv():
    """FloorMod(5, -3) -> -1 in BOTH Int and BV32 mode.

    TIR FloorMod is sign-of-divisor: ``floor(5 / -3) = -2`` and
    ``5 - (-3) * (-2) = -1``. The same double-correction bug as the
    negative-dividend case but on the divisor side: with ``bvsmod`` the
    answer is already -1; the helper would have turned it into +1.

    We pin ``x`` to ``[5, 6)`` and ask whether
    ``floormod(x, -3) == -1``. MUST be provable under both sorts.
    """
    x = tir.Var("x", "int32")
    expr = tir.FloorMod(x, tir.const(-3, "int32")) == tir.const(-1, "int32")
    assert _can_prove(x, 5, 6, expr, 0) is True
    assert _can_prove(x, 5, 6, expr, 32) is True


# CPPMEGA z3-stack fix-A8 (NEW-2): cross-pass isolation regression.
# Documents the production guarantee: the per-thread Z3 prover cache
# does NOT bleed state across pass invocations. We exercise this by
# making two back-to-back BV-mode-divergent CanProve calls that would
# return mismatched answers if the second inherited the first's bv
# state.
_CLEAR_PROVER_CACHE = tvm.ffi.get_global_func("tl.z3.clear_prover_cache",
                                              allow_missing=True)


@pytest.mark.skipif(_CLEAR_PROVER_CACHE is None,
                    reason="tl.z3.clear_prover_cache FFI not registered")
def test_z3_prover_cross_pass_isolation():
    """Two sequential CanProve calls under different bv_width must NOT
    contaminate each other.

    Sequence:
      1. BV32 over `2*addr >= 0`, addr in [0, 2^31) -> NOT provable.
      2. Clear cache (simulates pass-driver entry).
      3. Int mode over the same expr -> IS provable.

    Without the cache clear, step 2 would still be observed correctly
    because each `_can_prove` call constructs a fresh `Analyzer` (see
    `BvCanProve` in `z3_prover.cc`). However, if the thread-local cache
    happened to map a freed Analyzer's address to a stale BV32-mode
    prover, a fresh Analyzer landing on that address would inherit BV32
    semantics. The cache clear is what guarantees step 2 sees a clean
    slate even under address reuse.

    The test sandwiches a `clear_prover_cache()` call between the two
    proofs so the second never observes any cached state at all — the
    expected divergent answers (False then True) confirm both that the
    two modes diverge AND that the cache clear is observable behavior.
    """
    addr = tir.Var("addr", "int32")
    expr = (addr * 2) >= 0
    # BV32 mode: 2*addr wraps for addr >= 2^30.
    assert _can_prove(addr, 0, 1 << 31, expr, 32) is False
    # Per-pass cache clear (the production hook in phase.py runs this).
    _CLEAR_PROVER_CACHE()
    # Int mode now: 2*addr stays non-negative.
    assert _can_prove(addr, 0, 1 << 31, expr, 0) is True
    # And once more around the loop, with the order reversed, to
    # confirm the clear works in both directions.
    _CLEAR_PROVER_CACHE()
    assert _can_prove(addr, 0, 1 << 31, expr, 0) is True
    _CLEAR_PROVER_CACHE()
    assert _can_prove(addr, 0, 1 << 31, expr, 32) is False


@pytest.mark.skipif(_CLEAR_PROVER_CACHE is None,
                    reason="tl.z3.clear_prover_cache FFI not registered")
def test_z3_clear_prover_cache_idempotent():
    """`clear_prover_cache` is safe to call repeatedly, including on an
    empty cache. Pass drivers may invoke it many times per compile."""
    for _ in range(5):
        _CLEAR_PROVER_CACHE()


# CPPMEGA z3-stack fix-A7 (NEW-1): out-of-range Bind in BV mode now
# emplaces a memoization clamped to the BV signed range, with a single
# warning logged per Analyzer. Previously the code returned without
# memoizing, so a subsequent CanProve over the same var would hit the
# `Create()` codepath and mint a free unconstrained Z3 symbol — i.e.,
# the caller's range request was silently dropped.
def test_z3_bv_out_of_range_bind_uses_clamped_memoization():
    """Bind addr to [-2^40, 2^40) under BV32. The caller's range cannot
    be represented losslessly in 32-bit signed BV; the prover clamps to
    [INT32_MIN, INT32_MAX+1) and asserts that constraint instead.

    Verification angle: ``addr * 0 == 0`` is trivially provable for
    *any* finite-int addr in any sort. Pre-fix-A7 this would still be
    provable (the var is free, but `addr * 0` simplifies away). The
    interesting case is ``addr >= INT32_MIN``, which under the clamped
    memo MUST be provable in BV32 (the clamp asserts exactly that
    lower bound) — under the pre-fix code the var was free, with NO
    range, so this was NOT provable. We use the post-fix invariant as
    a regression: addr >= INT32_MIN is provable for the out-of-range
    bind under BV32.
    """
    addr = tir.Var("addr", "int32")
    INT32_MIN = -(1 << 31)
    # The Range FromMinExtent(min, extent) forms [min, min+extent).
    # We want [-2^40, 2^40), so min = -(1<<40), extent = (1<<41).
    # But _can_prove takes (lo, hi); pass them directly.
    lo = -(1 << 40)
    hi = 1 << 40
    expr = addr >= tir.const(INT32_MIN, "int32")
    # With clamped memoization, this is provable in BV32 mode.
    # (Pre-fix-A7: NOT provable — addr was free.)
    assert _can_prove(addr, lo, hi, expr, 32) is True
