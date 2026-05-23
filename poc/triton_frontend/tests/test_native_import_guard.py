from __future__ import annotations

from poc.triton_frontend._test_harness.native_import_guard import (
    triton_import_block_reason,
)


def test_triton_import_guard_blocks_when_jaxlib_mlir_peer_loaded_first() -> None:
    reason = triton_import_block_reason({"jaxlib.mlir._mlir_libs": object()})

    assert reason is not None
    assert "jaxlib.mlir._mlir_libs" in reason


def test_triton_import_guard_does_not_block_when_only_tilelang_resident() -> None:
    """TileLang/TVM/tvm_ffi must not block Triton -- the cppmega bridge depends on this."""
    reason = triton_import_block_reason(
        {
            "tilelang": object(),
            "tilelang_cython_wrapper": object(),
            "tvm": object(),
            "tvm_ffi": object(),
        }
    )

    assert reason is None


def test_triton_import_guard_blocks_when_libtriton_and_jaxlib_mlir_coresident() -> None:
    """After 2026-05 dual-binding audit: jaxlib MLIR + libtriton always conflicts.

    Original assumption: if ``triton._C.libtriton`` was already in
    ``sys.modules`` we had "paid the LLVM cl::opt cost" and any
    subsequent ``make_ir`` was safe. That broke when jaxlib's
    MLIR/LLVM nanobind extension was loaded between two Triton
    ``make_ir`` calls: the next ``make_ir`` aborted the interpreter
    on duplicate cl::opt registration. The guard now reports the
    jaxlib peer even after libtriton resident, so callers either skip
    live Triton or re-run in a fresh subprocess.
    """
    reason = triton_import_block_reason(
        {
            "triton._C.libtriton": object(),
            "jaxlib.mlir._mlir_libs": object(),
        }
    )

    assert reason is not None
    assert "jaxlib.mlir._mlir_libs" in reason
