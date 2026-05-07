# Vendored helper for triton-shared integration with the unified fused-kernel compiler.
# Copyright (c) 2026 Project Contributors.
# Original triton-shared sources Copyright (c) Microsoft Corporation and Meta Platforms, Inc.
# Licensed under the MIT License.
"""Smoke test: verify the vendored `tts` dialect actually loads and parses.

Run via:
    python -m poc.triton_frontend.vendored.triton_shared.verify_dialect_loads

Skipped automatically if the C++ shim (``poc.triton_frontend._cxx``) hasn't
been built yet -- the shim is produced by sibling integration #4.
"""
from __future__ import annotations

import sys


def _main() -> int:
    try:
        from poc.triton_frontend._cxx import register_dialects  # type: ignore
    except ImportError as exc:
        print(f"SKIP: poc.triton_frontend._cxx not built ({exc})")
        return 0

    try:
        from mlir import ir  # type: ignore
    except ImportError as exc:
        print(f"SKIP: mlir python bindings not available ({exc})")
        return 0

    src = """
    module {
      func.func @tts_smoke(%base: !tt.ptr<f32>, %off: index, %st: index) {
        %0 = tts.make_tptr %base to
              sizes: [4],
              strides: [%st],
              offsets: [%off],
              shape: [0],
              order: []
              : <f32> to tensor<4x!tt.ptr<f32>>
        return
      }
    }
    """

    with ir.Context() as ctx:
        register_dialects(ctx)
        ctx.allow_unregistered_dialects = False
        with ir.Location.unknown(ctx):
            module = ir.Module.parse(src)
            assert module is not None, "parse returned None"
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
