"""Compatibility facade for TileLang's private MLX TVM-FFI adapter bridge.

Runtime code should not import this module directly. Call compiled TileLang
kernels and let ``tilelang.jit.adapter.tvm_ffi`` own the native boundary.
"""

from __future__ import annotations

from tilelang.jit.adapter._mlx_tvm_ffi import (
    MLXTVMFFIBridgeUnavailable,
    debug_state,
    is_available,
    metal_call,
    owner_output_buffer,
    owner_output_buffers,
    reset_debug_state,
)

__all__ = [
    "MLXTVMFFIBridgeUnavailable",
    "debug_state",
    "is_available",
    "metal_call",
    "owner_output_buffer",
    "owner_output_buffers",
    "reset_debug_state",
]
