"""Tests for Idea #9 (Z3 roadmap): simdgroup memory lift on Metal.

Z3 query::

    tile_extent <= 32
    /\\  reduce_op ∈ {add, max, min, or, and, xor}
    /\\  no cross-simdgroup write happens before reduce

Implementation status: detection-only with logging. The pass is gated
behind PassConfig key ``tl.simd_lift_reductions`` (default OFF). These
tests exercise the detector helper directly so they do not need a
built libtilelang nor a Metal target.
"""

from __future__ import annotations

import importlib.util
import os
import sys

import pytest

import tilelang.language as T
from tilelang import tvm as tvm
from tvm import tir


def _load_worktree_module():
    """Load the worktree's metal_simd_lift.py directly.

    See note in test_simdgroup_matrix_detection.py — the TileLang editable
    install pins module paths to the parent source tree, so we side-load
    the worktree copy via importlib.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.normpath(os.path.join(here, "..", "..", "..", "tilelang", "transform", "metal_simd_lift.py"))
    spec = importlib.util.spec_from_file_location("_worktree_metal_simd_lift", candidate)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


metal_simd_lift = _load_worktree_module()
_z3_extent_le_32 = metal_simd_lift._z3_extent_le_32
detect_candidates = metal_simd_lift.detect_candidates
PASS_CONFIG_KEY = metal_simd_lift.PASS_CONFIG_KEY


# ---------------------------------------------------------------------------
# Z3 helper
# ---------------------------------------------------------------------------


def test_z3_extent_static_16_proved():
    proved, query = _z3_extent_le_32(16)
    assert proved, query


def test_z3_extent_static_32_proved():
    proved, _ = _z3_extent_le_32(32)
    assert proved


def test_z3_extent_static_64_rejected():
    proved, query = _z3_extent_le_32(64)
    assert not proved, f"extent=64 must not be proved <=32; query={query}"


def test_z3_extent_intimm_proved():
    proved, _ = _z3_extent_le_32(tir.IntImm("int32", 24))
    assert proved


def test_z3_extent_intimm_rejected():
    proved, _ = _z3_extent_le_32(tir.IntImm("int32", 33))
    assert not proved


def test_z3_extent_symbolic_unconstrained_rejected():
    n = tir.Var("N", "int32")
    proved, query = _z3_extent_le_32(n)
    assert not proved, f"unconstrained symbolic extent must NOT be proved; query={query}"


# ---------------------------------------------------------------------------
# Detection on actual TIR
# ---------------------------------------------------------------------------


def _build_reduction_func(extent):
    @T.prim_func
    def func(buf: T.Tensor[(64,), T.float32], acc: T.Tensor[(1,), T.float32]):
        for i in T.serial(extent):
            acc[0] = acc[0] + buf[i]

    return func


def test_static_extent_16_is_candidate():
    func = _build_reduction_func(16)
    cands = detect_candidates(func)
    assert len(cands) == 1
    c = cands[0]
    assert c.op == "add"
    assert c.proved, c.query


def test_static_extent_64_not_candidate():
    func = _build_reduction_func(64)
    cands = detect_candidates(func)
    # Detector still finds the loop, but proved must be False.
    assert all(not c.proved for c in cands), [c.query for c in cands]


def test_static_extent_32_is_candidate():
    func = _build_reduction_func(32)
    cands = detect_candidates(func)
    assert any(c.proved for c in cands)


def test_unsupported_op_not_a_candidate():
    @T.prim_func
    def func(buf: T.Tensor[(8,), T.float32], acc: T.Tensor[(1,), T.float32]):
        for i in T.serial(8):
            acc[0] = acc[0] * buf[i]  # multiply is not in the SIMD set

    cands = detect_candidates(func)
    # Multiply must be filtered out by _classify_reduce_op.
    assert cands == []


def test_default_off_pass_is_noop():
    # Without the PassConfig flag, the pass must leave IR unchanged.
    func = _build_reduction_func(16)
    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))

    # Default PassContext — flag is OFF.
    out = metal_simd_lift.MetalSimdLiftReductions(mod)
    # Identity (structural equality): IR should be unchanged.
    tvm.ir.assert_structural_equal(out["main"], mod["main"], True)


def test_pass_config_key_constant():
    # The public key exposed for engine wiring must be the documented
    # canonical string. (We don't assert against PassConfigKey here
    # because the editable-installed enum may not yet expose the new
    # entry; the worktree's pass_config.py does — that is checked by
    # the next test.)
    assert PASS_CONFIG_KEY == "tl.simd_lift_reductions"


def test_worktree_pass_config_exposes_key():
    here = os.path.dirname(os.path.abspath(__file__))
    pass_config_path = os.path.normpath(os.path.join(here, "..", "..", "..", "tilelang", "transform", "pass_config.py"))
    spec = importlib.util.spec_from_file_location("_worktree_pass_config", pass_config_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    assert mod.PassConfigKey.TL_SIMD_LIFT_REDUCTIONS.value == PASS_CONFIG_KEY


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
