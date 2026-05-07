"""Smoke test for the PtrAnalysis facade.

Source MLIR text below is copied verbatim (sans CHECK lines) from
``microsoft/triton-shared/test/Conversion/TritonToStructured/
addptr_scalar_loopback.mlir`` (MIT, Microsoft + Meta).
"""
from __future__ import annotations

import pytest

from poc.triton_frontend.ptr_analysis import (
    PtrAnalysis,
    PtrState,
    dialects_available,
    shim_available,
)

ADDPTR_MLIR = """\
module {
  tt.func @kernel(
  %arg0 : !tt.ptr<bf16>,
  %arg1 : !tt.ptr<bf16>,
  %arg2 : i32
  ) {
    %0 = tt.addptr %arg0, %arg2 : !tt.ptr<bf16>, i32
    %1 = tt.addptr %arg1, %arg2 : !tt.ptr<bf16>, i32
    %10 = tt.load %0 {cache = 1 : i32, evict = 1 : i32, isVolatile = false}: !tt.ptr<bf16>
    tt.store %1, %10 : !tt.ptr<bf16>
    tt.return
  }
}
"""


@pytest.mark.skipif(
    not dialects_available(),
    reason=(
        "shim built without TritonStructured/Triton dialects -- rebuild with "
        "-DTRITON_INSTALL_DIR set (see _cxx/README.md)."
    ),
)
def test_ptr_analysis_rewrites_addptr() -> None:
    pa = PtrAnalysis(ADDPTR_MLIR)
    rewritten = pa.rewrite()
    assert isinstance(rewritten, str) and rewritten
    # The hallmark of a successful PtrAnalysis rewrite is the appearance of
    # tts.make_tptr in place of (or alongside) the original tt.addptr chain.
    assert "tts.make_tptr" in rewritten


@pytest.mark.skipif(
    not dialects_available(),
    reason="shim built without TritonStructured/Triton dialects",
)
def test_ptr_analysis_extract_states_returns_list() -> None:
    states = PtrAnalysis(ADDPTR_MLIR).extract_states()
    assert isinstance(states, list)
    for s in states:
        assert isinstance(s, PtrState)


@pytest.mark.skipif(not shim_available(), reason="C++ shim not built")
def test_shim_present_implies_dialects_query_returns_bool() -> None:
    # Even in stub mode (shim_available but not dialects_available), the
    # query should never raise; it is the canonical way for callers to
    # branch between full-rewrite and parse-only paths.
    assert isinstance(dialects_available(), bool)
