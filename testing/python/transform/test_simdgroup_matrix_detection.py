"""Tests for Idea #8 (Z3 roadmap): simdgroup matrix detection on Metal.

These tests exercise the public detection helper in
``tilelang.transform.metal_fragment_to_simdgroup`` directly, so they do
not require a Metal target nor a built libtilelang. The Z3 query
covered here is::

    shape[0] % 8 == 0
    /\\  shape[1] % 8 == 0
    /\\  dtype ∈ {fp16, packed fp8}
    /\\  addr % 16 == 0

Conservative-by-default: when Z3 returns False/UNKNOWN the helper keeps
the legacy non-simdgroup path.
"""

from __future__ import annotations

import importlib.util
import os
import sys

import pytest

from tilelang import tvm as tvm  # noqa: F401  (forces tilelang lib load)
from tvm import tir


def _load_worktree_module():
    """Load the worktree's metal_fragment_to_simdgroup.py directly.

    The TileLang editable install pins ``tilelang.transform.*`` to a
    specific source tree (the original clone). When the tests run from a
    git worktree the editable mapping points at the *parent* clone, not
    our worktree, so ``import tilelang.transform.metal_fragment_to_simdgroup``
    yields the unmodified file. We bypass that by loading the worktree
    file directly via ``importlib``.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.normpath(os.path.join(here, "..", "..", "..", "tilelang", "transform", "metal_fragment_to_simdgroup.py"))
    if not os.path.exists(candidate):
        # Fallback: standard import.
        from tilelang.transform import metal_fragment_to_simdgroup as m

        return m
    spec = importlib.util.spec_from_file_location("_worktree_metal_fragment_to_simdgroup", candidate)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


_M = _load_worktree_module()
is_simdgroup_eligible = _M.is_simdgroup_eligible
_static_simdgroup_eligible = _M._static_simdgroup_eligible
_z3_simdgroup_eligible = _M._z3_simdgroup_eligible


class _FakeBuf:
    def __init__(self, shape, dtype, name="buf"):
        self.shape = [tir.IntImm("int32", s) if isinstance(s, int) else s for s in shape]
        self.dtype = dtype
        self.name = name


# ---------------------------------------------------------------------------
# Static cases
# ---------------------------------------------------------------------------


def test_static_fp16_8x8_detected():
    buf = _FakeBuf([8, 8], "float16")
    eligible, reason = is_simdgroup_eligible(buf)
    assert eligible, f"fp16 8x8 should be detected, got reason={reason}"


def test_static_fp16_16x16_detected():
    buf = _FakeBuf([16, 16], "float16")
    eligible, _ = is_simdgroup_eligible(buf)
    assert eligible


def test_static_fp32_8x8_not_detected():
    buf = _FakeBuf([8, 8], "float32")
    eligible, reason = is_simdgroup_eligible(buf)
    assert not eligible, f"fp32 must NOT be eligible: {reason}"


def test_static_fp16_7x8_not_detected():
    buf = _FakeBuf([7, 8], "float16")
    eligible, reason = is_simdgroup_eligible(buf)
    assert not eligible, f"shape[0]=7 must NOT be eligible: {reason}"


def test_static_fp16_8x9_not_detected():
    buf = _FakeBuf([8, 9], "float16")
    eligible, _ = is_simdgroup_eligible(buf)
    assert not eligible


# ---------------------------------------------------------------------------
# fp8 packed
# ---------------------------------------------------------------------------


def test_static_fp8_8x8_detected():
    # uint8/int8 stand in for "packed fp8" pre-quantization buffers.
    buf = _FakeBuf([8, 8], "uint8")
    eligible, _ = is_simdgroup_eligible(buf)
    assert eligible


# ---------------------------------------------------------------------------
# Symbolic cases — Z3 fallback
# ---------------------------------------------------------------------------


def test_symbolic_unconstrained_not_detected():
    # Symbolic shape with no upstream constraint → Z3 cannot prove
    # divisibility → conservative reject.
    n = tir.Var("N", "int32")
    buf = _FakeBuf([n, n], "float16")
    eligible, reason = is_simdgroup_eligible(buf)
    assert not eligible, f"unconstrained symbolic shape must NOT be eligible: {reason}"
    assert "z3-proved=False" in reason


def test_symbolic_with_constant_fallback_query_runs():
    # If shape is concretely 8x8 but presented as IntImm via a Var bind,
    # the Z3 fallback should still terminate and return proved=True.
    proved, query = _z3_simdgroup_eligible([8, 8], "float16")
    assert proved, f"Z3 should prove for concrete 8x8 fp16, query={query}"


def test_symbolic_partial_concrete_z3_prove():
    # shape[0]=16 (div 8), shape[1]=symbolic — Z3 cannot prove unless we
    # supply an additional constraint, so the helper should report False.
    n = tir.Var("M", "int32")
    proved, _ = _z3_simdgroup_eligible([16, n], "float16")
    assert not proved


def test_dtype_rejected_before_z3():
    # Wrong dtype short-circuits — no Z3 invocation needed.
    proved, query = _z3_simdgroup_eligible([8, 8], "float32")
    assert not proved
    assert "not in simdgroup" in query


# ---------------------------------------------------------------------------
# Helper smoke tests
# ---------------------------------------------------------------------------


def test_static_helper_direct():
    assert _static_simdgroup_eligible([8, 8], "float16")
    assert _static_simdgroup_eligible([16, 32], "bfloat16")
    assert not _static_simdgroup_eligible([8, 8], "float32")
    assert not _static_simdgroup_eligible([7, 8], "float16")
    assert not _static_simdgroup_eligible([8], "float16")  # rank<2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
