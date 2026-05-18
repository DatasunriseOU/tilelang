"""Native extension load-order guards for the Triton frontend harness."""
from __future__ import annotations

import sys
from typing import Optional


_TRITON_NATIVE_MODULES = (
    "triton._C",
    "triton._C.libtriton",
)

_LLVM_PEER_MODULES = (
    "_triton_frontend_cxx",
    "tilelang_cython_wrapper",
    "tilelang",
    "tvm_ffi.core",
    "tvm_ffi",
    "tvm",
)


def triton_native_loaded() -> bool:
    """Return True when Triton's native libtriton is already resident."""
    return any(name in sys.modules for name in _TRITON_NATIVE_MODULES)


def loaded_llvm_peer_modules() -> list[str]:
    """Return loaded native-side peers known to conflict with libtriton."""
    return [name for name in _LLVM_PEER_MODULES if name in sys.modules]


def triton_import_block_reason() -> Optional[str]:
    """Return a human-readable reason to avoid importing Triton now.

    Triton and the TileLang/TVM/shim side can each statically register LLVM
    command-line options. If the TileLang side is already loaded and
    libtriton is not, importing Triton can abort the process before Python can
    catch an exception. In that state callers should report Triton as
    unavailable and run live Triton checks in a fresh process.
    """
    if triton_native_loaded():
        return None
    peers = loaded_llvm_peer_modules()
    if not peers:
        return None
    return (
        "triton import blocked because "
        + ", ".join(peers)
        + " already loaded in this process; importing triton._C.libtriton "
        "too can abort on duplicate LLVM cl::opt registration. Re-run the "
        "Triton-dependent check in a fresh Python process."
    )
