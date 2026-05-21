"""Native extension load-order guards for the Triton frontend harness."""
from __future__ import annotations

import sys
from typing import Any, Mapping, Optional


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


def _loaded_modules(
    loaded_modules: Optional[Mapping[str, Any]],
) -> Mapping[str, Any]:
    return sys.modules if loaded_modules is None else loaded_modules


def triton_native_loaded(
    loaded_modules: Optional[Mapping[str, Any]] = None,
) -> bool:
    """Return True when Triton's native libtriton is already resident."""
    modules = _loaded_modules(loaded_modules)
    return any(name in modules for name in _TRITON_NATIVE_MODULES)


def loaded_llvm_peer_modules(
    loaded_modules: Optional[Mapping[str, Any]] = None,
) -> list[str]:
    """Return loaded native-side peers known to conflict with libtriton."""
    modules = _loaded_modules(loaded_modules)
    return [name for name in _LLVM_PEER_MODULES if name in modules]


def triton_import_block_reason(
    loaded_modules: Optional[Mapping[str, Any]] = None,
) -> Optional[str]:
    """Return a human-readable reason to avoid importing Triton now.

    Triton and the TileLang/TVM/shim side can each statically register LLVM
    command-line options. If the TileLang side is already loaded and
    libtriton is not, importing Triton can abort the process before Python can
    catch an exception. In that state callers should report Triton as
    unavailable and run live Triton checks in a fresh process.
    """
    if triton_native_loaded(loaded_modules):
        return None
    peers = loaded_llvm_peer_modules(loaded_modules)
    if peers:
        return (
            "triton import blocked because "
            + ", ".join(peers)
            + " already loaded in this process; importing triton._C.libtriton "
            "too can abort on duplicate LLVM cl::opt registration. Re-run the "
            "Triton-dependent check in a fresh Python process."
        )
    return None


def triton_compile_block_reason(
    loaded_modules: Optional[Mapping[str, Any]] = None,
) -> Optional[str]:
    """Return a reason to avoid Triton TTIR generation in this process."""
    import_reason = triton_import_block_reason(loaded_modules)
    if import_reason is not None:
        return import_reason

    peers = loaded_llvm_peer_modules(loaded_modules)
    if triton_native_loaded(loaded_modules) and peers:
        return (
            "triton compile blocked because triton._C.libtriton and "
            + ", ".join(peers)
            + " are already loaded in this process; Triton TTIR generation "
            "can abort on duplicate LLVM cl::opt registration. Re-run the "
            "Triton-dependent check in a fresh Python process."
        )
    return None
