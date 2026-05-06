"""CPPMEGA Z3 idea #5: tests for ``prove_dot4_int24_safe``.

Covers the static fast path (boundary K transitions) and the symbolic Z3
fallback. The boundary-search test verifies the exact K at which
``K * 127 * 127 < 2^23`` flips from True to False, since that K value gates
the dot4 fast path's auto-promote dispatcher in ``fp8_op.fp8_scaled_matmul``.
"""

from __future__ import annotations

import unittest

import pytest

from tilelang.analysis.int24_overflow_proof import (
    _Z3_AVAILABLE,
    prove_dot4_int24_safe,
)
from tvm import tir


class StaticPathTests(unittest.TestCase):
    """Cases that hit the Python-only ``int / IntImm`` fast path."""

    def test_static_small_K_safe(self):
        # K=64 -> 64 * 127 * 127 = 1,032,256, well under 2^23 = 8,388,608.
        self.assertTrue(prove_dot4_int24_safe(64))
        # ``IntImm`` behaves identically -- the prover folds it.
        self.assertTrue(prove_dot4_int24_safe(tir.IntImm("int32", 64)))

    def test_static_large_K_unsafe(self):
        # K=2000 -> 2000 * 127 * 127 = 32,258,000 which overflows int24.
        self.assertFalse(prove_dot4_int24_safe(2000))

    def test_static_boundary_K_516(self):
        # Verify the documented K transition. Compute by hand:
        #   K=516: 516 * 127 * 127 = 8,322,564  -> safe (< 8,388,608)
        #   K=520: 520 * 127 * 127 = 8,387,080  -> safe (< 8,388,608)
        #   K=521: 521 * 127 * 127 = 8,403,209  -> unsafe (>= 8,388,608)
        # So the transition happens between K=520 (True) and K=521 (False).
        # Floor of (2^23 - 1) / (127 * 127) is 520, hence K=520 is the
        # largest safe K under the worst-case ``|x|=|y|=127`` assumption.
        self.assertEqual(516 * 127 * 127, 8_322_564)
        self.assertTrue(prove_dot4_int24_safe(516))

        self.assertEqual(520 * 127 * 127, 8_387_080)
        self.assertTrue(prove_dot4_int24_safe(520))

        self.assertEqual(521 * 127 * 127, 8_403_209)
        self.assertFalse(prove_dot4_int24_safe(521))

    def test_static_zero_K_is_trivially_safe(self):
        # Empty dot product: ``sum`` is 0 which sits inside int24.
        self.assertTrue(prove_dot4_int24_safe(0))

    def test_static_negative_max_returns_false_no_crash(self):
        # API misuse: caller must pass absolute upper bounds. We must NOT
        # crash; conservative False is the documented contract.
        self.assertFalse(prove_dot4_int24_safe(64, x_max=-1))
        self.assertFalse(prove_dot4_int24_safe(64, y_max=-1))


@pytest.mark.skipif(not _Z3_AVAILABLE, reason="z3-solver not installed")
class SymbolicZ3PathTests(unittest.TestCase):
    """Cases that fall through to the Z3 solver."""

    def test_symbolic_K_with_constraint(self):
        """Pinning ``K`` to a small constant via Z3 still proves safe.

        The roadmap describes "pass tir.Var with ``0 < K <= 64``" -- our
        prover sees a Z3 ``Int("K")`` and adds the ``> 0`` and
        ``<= _Z3_INT24_K_UPPER_BOUND`` bounds itself, but it cannot import
        a tightening constraint from the caller. So the closest analog is:
        manually construct a Z3 query asking the same question with K
        constrained to ``<= 64``. We verify that the prover-internal Z3
        integration logic does not spuriously fail in that range.
        """
        # The simplest "symbolic" exerciser: pass a tir.Add expression which
        # ``isinstance(K, int | IntImm)`` rejects, forcing the Z3 branch.
        # With no tightening constraint Z3 finds K = 1<<16 violates the
        # int24 bound -- so this case is conservative-False, NOT True.
        # To exercise the True path under symbolic K we'd need the C++-side
        # ``analyzer.bind(...)`` bridge. For now we lock the contract:
        # symbolic K without tighter bound -> conservative-False.
        sym_k = tir.Add(tir.IntImm("int32", 32), tir.IntImm("int32", 32))
        # ``Add`` is not ``int`` and not ``IntImm`` -> symbolic path.
        result = prove_dot4_int24_safe(sym_k)
        # The wrapper passes K_var as a free Z3 Int with bound K <= 65536;
        # at K=65536 the worst-case is 1,057,029,376 which overflows int24.
        # So Z3 finds a counter-example and we conservatively return False.
        self.assertFalse(result)

    def test_symbolic_K_unbounded(self):
        """Truly opaque symbolic K -- conservative False is the contract."""
        sym_k = tir.Var("K_unbounded", "int32")
        # Symbolic path: prover bounds K in [1, 65536] internally and asks
        # whether overflow is impossible. It is possible (at K=65536), so
        # the result is False.
        self.assertFalse(prove_dot4_int24_safe(sym_k))


if __name__ == "__main__":
    unittest.main()
