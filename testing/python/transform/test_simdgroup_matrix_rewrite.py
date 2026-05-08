"""Tests for Idea #8 (Z3 roadmap): simdgroup matrix **IR rewrite** on Metal.

These tests cover the gated rewrite path added on top of the
detection-only helpers in ``test_simdgroup_matrix_detection.py``. The
gating is controlled by:

  * PassConfig key ``tl.simdgroup_matrix_rewrite`` (default OFF, preserves
    legacy unconditional behavior).
  * Per-buffer eligibility via :func:`is_simdgroup_eligible`. Eligible
    accumulators get their storage scope promoted from ``local.fragment``
    to ``metal.simdgroup`` — which is the IR-level surface that triggers
    the MSL ``simdgroup_load`` / ``simdgroup_multiply_accumulate`` /
    ``simdgroup_store`` codegen in ``src/target/codegen_metal.cc``.

The codegen-side intrinsics (``builtin::simdgroup_load`` etc.) already
exist; this idea is purely an IR rewrite — it changes which accumulators
the GEMM tile-op + Metal codegen see as ``metal.simdgroup`` (drives
intrinsic emission) vs ``local.fragment`` (legacy scalar lowering). No new
codegen intrinsic registration was needed.

These tests construct the rewrite scenario at the Python helper level
(without invoking the full TileLang lowering pipeline) so they can run in
CI where the full lowering stack may not import cleanly.
"""

from __future__ import annotations

import importlib.util
import os
import sys

import pytest


def _load_worktree_module():
    """Load the worktree's metal_fragment_to_simdgroup.py directly.

    Mirrors the ``test_simdgroup_matrix_detection.py`` pattern: the
    editable install pins ``tilelang.transform.*`` to the parent clone, so
    we bypass that and load the worktree file via importlib.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.normpath(
        os.path.join(here, "..", "..", "..", "tilelang", "transform",
                     "metal_fragment_to_simdgroup.py")
    )
    if not os.path.exists(candidate):
        from tilelang.transform import metal_fragment_to_simdgroup as m
        return m
    spec = importlib.util.spec_from_file_location(
        "_worktree_metal_fragment_to_simdgroup_rewrite", candidate
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# Force tilelang init (the detection module needs `tvm.tir`).
try:
    from tilelang import tvm as tvm  # noqa: F401
    _IMPORT_OK = True
    _IMPORT_ERR = None
except Exception as exc:  # pragma: no cover - env-dependent
    _IMPORT_OK = False
    _IMPORT_ERR = exc

if _IMPORT_OK:
    from tvm import tir
    _M = _load_worktree_module()
    is_simdgroup_eligible = _M.is_simdgroup_eligible
    PASS_CONFIG_KEY = _M.PASS_CONFIG_KEY
    EMITTED_ATTR_KEY = _M.EMITTED_ATTR_KEY


pytestmark = pytest.mark.skipif(
    not _IMPORT_OK,
    reason=f"tilelang import failed (env-blocked): {_IMPORT_ERR!r}",
)


class _FakeBuf:
    """Minimal stand-in for ``tir.Buffer`` exposing ``shape`` / ``dtype``."""

    def __init__(self, shape, dtype, name="buf"):
        self.shape = [tir.IntImm("int32", s) if isinstance(s, int) else s
                      for s in shape]
        self.dtype = dtype
        self.name = name


# ---------------------------------------------------------------------------
# Helper-level rewrite gating tests
# ---------------------------------------------------------------------------

def test_pass_config_key_default_off():
    """``tl.simdgroup_matrix_rewrite`` must be opt-in.

    The pass file exposes the key as a module-level constant. Default
    behavior of ``_is_rewrite_enabled`` (no PassContext key set) is False.
    """
    assert PASS_CONFIG_KEY == "tl.simdgroup_matrix_rewrite"
    assert _M._is_rewrite_enabled() is False


def test_pass_config_key_can_enable():
    """When PassContext sets the flag True, gating fires."""
    from tvm import transform as tvm_transform
    PassContext = tvm_transform.PassContext
    with PassContext(config={PASS_CONFIG_KEY: True}):
        assert _M._is_rewrite_enabled() is True
    # And it returns to False outside the context.
    assert _M._is_rewrite_enabled() is False


def test_emitted_attr_key_constant():
    assert EMITTED_ATTR_KEY == "tl.simdgroup_matrix_rewrite_emitted"


# ---------------------------------------------------------------------------
# Eligibility-driven gating: 8x8 fp16 → eligible; misaligned/wrong dtype → not.
# ---------------------------------------------------------------------------

def test_8x8_fp16_eligible_for_rewrite():
    """An 8x8 fp16 fragment is eligible — the rewrite would promote it."""
    buf = _FakeBuf([8, 8], "float16", name="C_local")
    eligible, reason = is_simdgroup_eligible(buf)
    assert eligible, f"8x8 fp16 must be eligible, reason={reason}"


def test_8x8_bf16_eligible_for_rewrite():
    buf = _FakeBuf([16, 32], "bfloat16", name="C_local")
    eligible, _ = is_simdgroup_eligible(buf)
    assert eligible


def test_misaligned_shape_falls_back_to_legacy():
    """A shape that is not a multiple of 8 → ineligible → legacy path."""
    buf = _FakeBuf([7, 8], "float16", name="C_misaligned")
    eligible, reason = is_simdgroup_eligible(buf)
    assert not eligible, f"shape[0]=7 must NOT be eligible, reason={reason}"


def test_misaligned_addr_falls_back_to_legacy():
    """A concrete addr that is not 16-byte aligned → Z3 cannot prove."""
    proved, query = _M._z3_simdgroup_eligible([8, 8], "float16",
                                              addr_value=4)
    assert not proved, f"addr=4 (not 16-aligned) must NOT prove: {query}"


def test_wrong_dtype_falls_back_to_legacy():
    """fp32 is not in the simdgroup dtype set → ineligible."""
    buf = _FakeBuf([8, 8], "float32", name="C_fp32")
    eligible, reason = is_simdgroup_eligible(buf)
    assert not eligible, f"fp32 must NOT be eligible, reason={reason}"


# ---------------------------------------------------------------------------
# Symbolic-shape Z3 path: when Z3 cannot prove, gate stays closed.
# ---------------------------------------------------------------------------

def test_symbolic_shape_unproved_does_not_rewrite():
    """Unconstrained symbolic shape → Z3 cannot prove → ineligible."""
    n = tir.Var("N", "int32")
    buf = _FakeBuf([n, n], "float16", name="C_sym")
    eligible, reason = is_simdgroup_eligible(buf)
    assert not eligible
    assert "z3-proved=False" in reason


def test_symbolic_shape_concrete_z3_proves():
    """Static-pass concrete 8x8 fp16 → Z3 fallback also reports proved."""
    proved, query = _M._z3_simdgroup_eligible([8, 8], "float16",
                                              addr_value=16)
    assert proved, f"8x8 fp16 + 16-aligned addr must prove, query={query}"


# ---------------------------------------------------------------------------
# Collection helper: per-buffer info preserves shape/dtype for gating.
# ---------------------------------------------------------------------------

def test_collect_fragment_gemm_accum_buffers_helper_exists():
    """Pass exposes ``_collect_fragment_gemm_accum_buffers`` returning {var: buf}."""
    helper = getattr(_M, "_collect_fragment_gemm_accum_buffers", None)
    assert helper is not None and callable(helper), \
        "rewrite path needs the per-buffer collector"


def test_remap_buffer_does_not_introduce_auto_elem_offset_var():
    """Promoting an internal fragment buffer must not create an ABI offset Var."""
    buf = tir.decl_buffer(
        [8, 8],
        "float32",
        name="C_local",
        scope="local.fragment",
    )
    remapped = _M._remap_buffer(
        buf,
        _M._build_var_map([buf.data]),
        {"C_local"},
        {},
    )

    assert isinstance(remapped.elem_offset, tir.IntImm)
    assert int(remapped.elem_offset.value) == 0


# ---------------------------------------------------------------------------
# `apply_simdgroup_matrix_rewrite` returns the func unchanged for non-Metal
# targets (conservative-by-default).
# ---------------------------------------------------------------------------

def test_apply_rewrite_skips_non_metal_target():
    """If the PrimFunc has no Metal target, the helper is a no-op."""
    from tvm.script import tir as T

    @T.prim_func
    def _f(A: T.Buffer((8, 8), "float16")):
        for i, j in T.grid(8, 8):
            A[i, j] = T.float16(0)

    out = _M.apply_simdgroup_matrix_rewrite(_f)
    # Non-Metal target → no rewrite → no emitted attr added.
    attrs = out.attrs or {}
    assert EMITTED_ATTR_KEY not in (attrs or {})


# ---------------------------------------------------------------------------
# Default-OFF preserves shipping behavior: the unconditional pass still runs
# and rewrites everything; the gated path is only taken when the flag is set.
# ---------------------------------------------------------------------------

def test_default_off_preserves_legacy_unconditional_path():
    """When the PassConfig flag is OFF, ``_is_rewrite_enabled()`` is False
    so ``_metal_fragment_to_simdgroup`` takes the legacy unconditional
    branch (every GEMM accumulator gets promoted, no per-buffer gating).

    We assert the gating function returns False — the legacy branch logic
    itself is unchanged from ship and is covered indirectly by the
    existing GEMM Metal tests.
    """
    assert _M._is_rewrite_enabled() is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
