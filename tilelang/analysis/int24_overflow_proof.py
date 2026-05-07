"""CPPMEGA Z3 roadmap idea #5: int24 overflow proof for FP8 dot4 accumulators.

The Metal packed-FP8 dot4 intrinsic accumulates the dequantised int8 / e4m3
products into a 24-bit signed accumulator before the post-dot fp32 scale
multiply. The ``metal_fp8_e4m3_dot4`` LUT-decoded path on Apple GPUs uses an
int24 hardware lane internally (cross-lane SIMD-group reduction widens to
fp32, but the per-lane partial sum is int24). To make the dot4 fast path
*provably* safe for arbitrary K we therefore need a static / Z3-discharged
proof that, given operands in ``[-x_max, x_max]`` and ``[-y_max, y_max]``::

    sum_{i=0..K-1} x_i * y_i  in  [-(1<<23), (1<<23) - 1]

Idea #5 in the roadmap calls for exactly this: a small linear-arith proof
obligation that wires into the ``_z3_prove_dot4_legal`` fast path so that
``proved == True`` only when BOTH the alignment legality predicate AND the
int24 non-overflow predicate hold. Either failing keeps the caller on the
legacy scalar simd_sum path, which accumulates in fp32 and has no int24
constraint. This is the conservative-by-default rule -- a wrong proof here
would silently emit packed-dot4 against an over-K kernel and produce wrong
numerics on first dispatch.

Implementation
--------------

Two-tier strategy mirroring the alignment prover:

* Static fast path: when ``K`` is a Python int / ``IntImm`` and the per-
  element bounds are known, the worst-case accumulator is just
  ``K * x_max * y_max``. We compare against the int24 limits directly --
  no Z3 boot required. This is the common case for emitted kernels, since
  the parser already has K as a constant by lowering time.
* Z3 fallback: when ``K`` is symbolic we build a small linear-arith
  query asking whether ``K * x_max * y_max`` could possibly meet or
  exceed the int24 limit under reasonable bounds (we cap the search at
  ``K <= 1 << 16`` -- any kernel with K above 65 536 dwarfs Apple-Silicon
  threadgroup memory and is implausible). Timeout is 50 ms; UNKNOWN /
  exceptions / sat all return False (conservative).

The proof ignores correlation between ``x_i`` and ``y_i``: we use the
absolute-value upper bound on each factor and assume the worst-case
``sign(x_i) == sign(y_i)`` for every term, so every term contributes the
same direction. This is sound for any data distribution and is what the
audit recommends.
"""

from __future__ import annotations

import os
from typing import Union

from tvm import tir


def _z3_pass_disabled(name: str) -> bool:
    """CPPMEGA z3-final per-pass gate (Python mirror of `Z3PassGate`).

    Returns True iff Z3 use for *this* pass should be skipped. Honours the
    blanket ``TILELANG_DISABLE_Z3`` (kill-switch) and the per-pass
    ``TILELANG_DISABLE_Z3_<NAME>``. Truthiness convention matches the C++
    side: a value of ``""`` or ``"0"`` means "enabled"; anything else is
    treated as "disabled".
    """
    for var in ("TILELANG_DISABLE_Z3", f"TILELANG_DISABLE_Z3_{name}"):
        v = os.environ.get(var, "")
        if v and v != "0":
            return True
    return False

# ---------------------------------------------------------------------------
# Z3 import is optional. When z3-solver isn't installed, the symbolic branch
# returns conservative-False (matching the dot4-legality prover policy).
# ---------------------------------------------------------------------------
try:
    import z3 as _z3  # type: ignore

    _Z3_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised when z3-solver is absent
    _z3 = None  # type: ignore
    _Z3_AVAILABLE = False


__all__ = ["prove_dot4_int24_safe"]


# Signed int24 range: ``[-(1<<23), (1<<23)-1]`` = ``[-8 388 608, 8 388 607]``.
_INT24_MIN = -(1 << 23)
_INT24_MAX = (1 << 23) - 1

# Z3 query timeout in milliseconds. Matches ``_Z3_DOT4_TIMEOUT_MS`` in
# ``tilelang.language.fp8_op``; a missed timeout maps to conservative-False.
_Z3_INT24_TIMEOUT_MS = 50

# Upper bound on K for the symbolic Z3 query. Real Metal kernels never reach
# this magnitude (K=2^16 is already four orders of magnitude past the
# dot4 fast-path's preferred K=128 .. 4096 sweet spot), but we cap it
# because Z3 needs a finite range on the contracted-K dimension to prune
# trivially-counter-example searches.
_Z3_INT24_K_UPPER_BOUND = 1 << 16


def _coerce_int(value) -> Union[int, None]:
    """Coerce ``value`` to a Python int when it's a constant, else ``None``."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, tir.IntImm):
        return int(value.value)
    if hasattr(value, "value"):
        try:
            return int(value.value)
        except (TypeError, ValueError):
            return None
    return None


def prove_dot4_int24_safe(
    K: Union[int, tir.PrimExpr],
    x_max: int = 127,
    y_max: int = 127,
) -> bool:
    """Return True iff a K-term int8*int8 dot product provably fits in int24.

    The proof obligation discharged is::

        for all i: |x_i| <= x_max && |y_i| <= y_max
        ==> -(1<<23) <= sum_{i=0..K-1} x_i * y_i <= (1<<23) - 1

    We use the worst-case bound ``|sum| <= K * x_max * y_max``; any signed
    arrangement of the partial products is sandwiched inside that bound.

    Args:
        K: Number of accumulated terms. Either a Python ``int`` (or
            ``tir.IntImm``, which is folded to int) for the static fast
            path, or a symbolic ``tir.PrimExpr`` for the Z3 fallback.
        x_max: Upper bound on ``|x_i|``. Default 127 -- the largest int8
            magnitude representable; e4m3 saturates well below this.
        y_max: Upper bound on ``|y_i|``. Same default and reasoning.

    Returns:
        ``True`` if the int24 range is provably never exceeded.
        ``False`` otherwise -- including UNKNOWN / timeout / exception
        and any case where K is symbolic without a tightening
        constraint. Conservative by construction.
    """
    if x_max < 0 or y_max < 0:
        # Bound API misuse must not crash: caller must pass absolute-value
        # upper bounds. Return False so the caller stays on the scalar path.
        return False

    # ----- Static fast path -------------------------------------------------
    # ``isinstance`` accepts both Python ``int`` and TIR ``IntImm`` here so
    # parser-folded constants discharge the proof in plain arithmetic.
    if isinstance(K, (int, tir.IntImm)):
        k_int = _coerce_int(K)
        if k_int is None or k_int < 0:
            return False
        bound = k_int * int(x_max) * int(y_max)
        # We need both ``+bound <= INT24_MAX`` and ``-bound >= INT24_MIN``.
        # ``-bound > -2^23`` is the strict version called out in the roadmap;
        # since the lower limit is exactly ``-2^23`` we accept ``-bound > -2^23``
        # (i.e. ``bound < 2^23``) -- the upper limit then enforces ``bound <
        # 2^23`` as well, so a single ``bound < (1 << 23)`` check is
        # sufficient and exactly what the static fast path returns.
        return bound < (1 << 23) and -bound > -(1 << 23)

    # ----- Symbolic / Z3 fallback -------------------------------------------
    if not _Z3_AVAILABLE:
        # Without z3-solver we cannot prove the symbolic case; staying on the
        # scalar path is the safe outcome.
        return False

    # CPPMEGA z3-final per-pass gate: TILELANG_DISABLE_Z3_INT24 bypasses the
    # int24 overflow proof (idea #5). Conservative default — caller stays on
    # the scalar path when disabled.
    if _z3_pass_disabled("INT24"):
        return False

    try:
        s = _z3.Solver()
        s.set("timeout", int(_Z3_INT24_TIMEOUT_MS))
        k_var = _z3.Int("K")
        # Reasonable bounds. Lower bound is positivity (an empty K is
        # trivially safe but uninteresting); upper bound prunes
        # absurd contracted-K extents that no kernel uses.
        s.add(k_var > 0, k_var <= _Z3_INT24_K_UPPER_BOUND)

        # Worst-case accumulator magnitude. We let Z3 reason in unbounded
        # Int sort -- this is sound; ``acc`` exceeds int24 iff the dot
        # product can. ``x_max`` / ``y_max`` are concrete ints so the
        # multiplication is linear in ``k_var``.
        acc = k_var * int(x_max) * int(y_max)

        # Goal we need to PROVE: ``acc < 2^23 AND -acc > -2^23``.
        # Z3 returns ``unsat`` for the *negation*, which means there is no
        # K in the allowed range that violates the goal. ``sat`` (a witness
        # exists) -> goal does not hold for at least one K -> return False.
        # ``unknown`` -> conservative-False.
        s.add(_z3.Or(acc >= (1 << 23), -acc <= -(1 << 23)))
        result = s.check()
        return result == _z3.unsat
    except Exception:  # pragma: no cover - defensive; solver init failures
        return False
