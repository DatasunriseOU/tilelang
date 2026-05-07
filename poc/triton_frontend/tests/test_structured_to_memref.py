"""Smoke test for the StructuredToMemref conversion pass.

The pass is lifted verbatim from facebookincubator/triton-shared
``lib/Conversion/StructuredToMemref/StructuredToMemref.cpp``. This test
chains it after ``run_ptr_analysis`` so we exercise the realistic
pipeline shape (``tt.* -> tts.* -> memref.*``) end-to-end:

    ttir text
       │
       ▼
    run_ptr_analysis          (tt.addptr / tt.load / tt.store -> tts.make_tptr)
       │
       ▼
    run_structured_to_memref  (tts.make_tptr / tts.load / tts.store -> memref.*)
       │
       ▼
    rewritten text containing memref.subview / memref.copy / memref.* loads
    and NO residual tts.make_tptr / tts.load / tts.store ops.

Hand-authoring a post-PtrAnalysis fixture (with the right number of
dynamic operands and ``static_*`` attribute lists) is brittle, so we
instead use ``run_ptr_analysis`` to produce the fixture deterministically
from a small ``tt.*`` source. This still isolates the StructuredToMemref
pass: the assertions only fire on its output, not on the intermediate
``tts.*`` form (which is the contract of the upstream pass driver).
"""
from __future__ import annotations

import re

import pytest

from poc.triton_frontend.lowering_passes import (
    run_structured_to_memref,
    structured_to_memref_available,
)
from poc.triton_frontend.ptr_analysis import dialects_available

# Minimal vector-add tile-load fixture. PtrAnalysis only emits ``tts.make_tptr``
# for *tile* pointers (i.e. ``tensor<Nx!tt.ptr<...>>``); pure scalar pointer
# arithmetic (``!tt.ptr<...>`` + ``i32``) bypasses the structured rewrite path
# and stays as plain ``tt.addptr``. We therefore use a 1D tile-load kernel
# adapted from the canonical ``vector_add`` fixture so PtrAnalysis is forced
# to produce ``tts.make_tptr`` / ``tts.load`` / ``tts.store`` ops, which
# StructuredToMemref then rewrites away.
TILE_LOAD_MLIR = """\
module {
  tt.func public @add_kernel(
    %arg0: !tt.ptr<f32>,
    %arg1: !tt.ptr<f32>,
    %arg2: !tt.ptr<f32>,
    %arg3: i32
  ) {
    %c0_i32 = arith.constant 0 : i32
    %0 = tt.get_program_id x : i32
    %c1024_i32 = arith.constant 1024 : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %3 = tt.splat %1 : i32 -> tensor<1024xi32>
    %4 = arith.addi %3, %2 : tensor<1024xi32>
    %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %6 = tt.addptr %5, %4 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    %7 = tt.load %6 : tensor<1024x!tt.ptr<f32>>
    %8 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %9 = tt.addptr %8, %4 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    %10 = tt.load %9 : tensor<1024x!tt.ptr<f32>>
    %11 = arith.addf %7, %10 : tensor<1024xf32>
    %12 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %13 = tt.addptr %12, %4 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    tt.store %13, %11 : tensor<1024x!tt.ptr<f32>>
    tt.return
  }
}
"""


@pytest.mark.skipif(
    not structured_to_memref_available(),
    reason=(
        "shim built without TritonStructured + StructuredToMemref bindings -- "
        "rebuild with -DTRITON_INSTALL_DIR set (see _cxx/README.md)."
    ),
)
def test_structured_to_memref_eliminates_tts_ops() -> None:
    """StructuredToMemref must rewrite tts.{make_tptr,load,store} away.

    The pre-condition (``tts.*`` ops in the IR) is established by piping the
    source through PtrAnalysis first; that step is the canonical producer of
    those ops and is already covered by ``test_ptr_analysis.py``.
    """
    # Lazy-load the shim through the same path the production pipeline uses.
    import importlib

    shim = importlib.import_module("_triton_frontend_cxx")
    pa_text = shim.run_ptr_analysis(TILE_LOAD_MLIR)

    # Sanity: PtrAnalysis must have produced at least one tts.make_tptr,
    # otherwise StructuredToMemref has nothing to rewrite and the assertion
    # below would pass vacuously.
    assert "tts.make_tptr" in pa_text, (
        f"PtrAnalysis did not emit tts.make_tptr; fixture unsuitable.\n"
        f"---\n{pa_text}"
    )

    out = run_structured_to_memref(pa_text)
    assert isinstance(out, str) and out

    # Hard requirement: memref.* ops must appear (the conversion target).
    assert "memref." in out, (
        f"StructuredToMemref output lacks memref.* ops:\n---\n{out}"
    )

    # Hard requirement: the three ops the pass marks illegal must be gone.
    illegal_pattern = re.compile(
        r"\btts\.(make_tptr|load|store)\b"
    )
    leftovers = illegal_pattern.findall(out)
    assert not leftovers, (
        f"StructuredToMemref left {len(leftovers)} illegal tts.* op(s) in "
        f"output: {leftovers}\n---\n{out}"
    )


@pytest.mark.skipif(
    not dialects_available(),
    reason="shim built without TritonStructured/Triton dialects",
)
def test_structured_to_memref_helper_is_exposed() -> None:
    """Sanity check: the Python facade exports the expected entry point."""
    from poc.triton_frontend import lowering_passes

    assert hasattr(lowering_passes, "run_structured_to_memref")
    assert callable(lowering_passes.run_structured_to_memref)
