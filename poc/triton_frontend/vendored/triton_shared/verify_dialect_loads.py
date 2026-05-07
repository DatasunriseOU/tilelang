# Vendored helper for triton-shared integration with the unified fused-kernel compiler.
# Copyright (c) 2026 Project Contributors.
# Original triton-shared sources Copyright (c) Microsoft Corporation and Meta Platforms, Inc.
# Licensed under the MIT License.
"""Smoke test: verify the vendored `tts` dialect actually loads and parses.

Run via:
    python -m poc.triton_frontend.vendored.triton_shared.verify_dialect_loads

Skipped automatically if neither the dedicated dialect-registration shim
(``poc.triton_frontend._cxx.register_triton_structured``) nor the legacy
``poc.triton_frontend._triton_frontend_cxx`` module has been built yet --
both expose a ``register_dialects(ctx)`` entry point produced by the
``TritonSharedRegister`` static library defined in this directory's
``CMakeLists.txt`` (``RegisterTritonStructured.cc``).
"""
from __future__ import annotations

import sys


def _load_register_dialects():
    # Preferred: dedicated triton-shared registration shim.
    try:
        from poc.triton_frontend._cxx.register_triton_structured import (  # type: ignore
            register_dialects,
        )
        return register_dialects
    except ImportError:
        pass
    # Fallback: combined frontend pybind module (when the project chooses
    # to fold the shim into a single .so).
    try:
        from poc.triton_frontend._triton_frontend_cxx import (  # type: ignore
            register_dialects,
        )
        return register_dialects
    except ImportError:
        return None


def _main() -> int:
    register_dialects = _load_register_dialects()
    if register_dialects is None:
        print("SKIP: poc.triton_frontend._cxx.register_triton_structured "
              "(or _triton_frontend_cxx) not built")
        return 0

    try:
        from mlir import ir  # type: ignore
    except ImportError as exc:
        print(f"SKIP: mlir python bindings not available ({exc})")
        return 0

    # Use the full upstream printer form (`!tt.ptr<f32> to tensor<...>`) so the
    # custom assemblyFormat in TritonStructuredDialect.td:78 parses
    # unambiguously. The previous abbreviated `<f32>` form depends on parser
    # context and is not guaranteed to round-trip.
    src = (
        "module {\n"
        "  func.func @tts_smoke(%base: !tt.ptr<f32>, %off: index, %st: index) {\n"
        "    %0 = tts.make_tptr %base to\n"
        "          sizes: [4],\n"
        "          strides: [%st],\n"
        "          offsets: [%off],\n"
        "          shape: [0],\n"
        "          order: []\n"
        "          : !tt.ptr<f32> to tensor<4x!tt.ptr<f32>>\n"
        "    return\n"
        "  }\n"
        "}\n"
    )

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
