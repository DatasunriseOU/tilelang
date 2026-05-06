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


_BV_CAN_PROVE = tvm.ffi.get_global_func("tl.z3.bv_can_prove")


def _can_prove(var, lo, hi, expr, bv_width):
    return bool(_BV_CAN_PROVE(var, lo, hi, expr, bv_width))


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
