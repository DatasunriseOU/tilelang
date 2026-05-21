from __future__ import annotations

from poc.triton_frontend._test_harness.native_import_guard import (
    triton_import_block_reason,
)


def test_triton_import_guard_blocks_when_tilelang_peer_loaded_first() -> None:
    reason = triton_import_block_reason({"tilelang": object()})

    assert reason is not None
    assert "tilelang" in reason


def test_triton_import_guard_allows_when_triton_native_already_loaded() -> None:
    reason = triton_import_block_reason(
        {
            "triton._C.libtriton": object(),
            "tilelang": object(),
            "tvm": object(),
        }
    )

    assert reason is None
