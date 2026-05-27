"""Native extension load-order guards for the Triton frontend harness."""
from __future__ import annotations

import sys
from typing import Any, Mapping, Optional


_TRITON_NATIVE_MODULES = (
    "triton._C",
    "triton._C.libtriton",
)

_TILELANG_LLVM_PEER_MODULES = (
    "tilelang",
    "tilelang_cython_wrapper",
    "tvm",
    "tvm.base",
)

# Peer modules whose native LLVM images empirically conflict with Triton's
# ``triton._C.libtriton``. Loading any of these in the same interpreter as
# native Triton can abort on duplicate LLVM ``cl::opt`` registration.
_LLVM_PEER_MODULES = (
    # PtrAnalysis C++ shim, statically linked against its own LLVM.
    "_triton_frontend_cxx",
    # The local TileLang dev build loads libtilelang/libtvm_compiler, whose
    # compiler library carries an LLVM image. Co-loading it with Triton's
    # LLVM-bearing libtriton can abort on options such as "basic"/"pbqp".
    *_TILELANG_LLVM_PEER_MODULES,
    # jaxlib bundles its own MLIR / LLVM nanobind extension under
    # ``jaxlib.mlir._mlir_libs``. Loading it in the same interpreter
    # that has also loaded ``triton._C.libtriton`` aborts the next
    # ``triton.compiler.compiler.make_ir`` call because both extensions
    # register clashing LLVM cl::opt entries. Importing TTIR via the
    # native walker through jaxlib (``_mlir_path_setup.local_jaxlib_mlir_ir``)
    # is the trigger; once jaxlib's MLIR is resident the only safe way
    # to call Triton is in a fresh subprocess.
    "jaxlib.mlir._mlir_libs",
    "jaxlib.mlir",
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


def triton_native_symbols_are_local(
    loaded_modules: Optional[Mapping[str, Any]] = None,
) -> bool:
    """Return True when Triton's top-level import isolated libtriton symbols."""
    modules = _loaded_modules(loaded_modules)
    triton_module = modules.get("triton")
    return bool(getattr(triton_module, "_NATIVE_DLOPEN_LOCAL", False))


def loaded_llvm_peer_modules(
    loaded_modules: Optional[Mapping[str, Any]] = None,
) -> list[str]:
    """Return loaded native-side peers known to conflict with libtriton."""
    modules = _loaded_modules(loaded_modules)
    return [name for name in _LLVM_PEER_MODULES if name in modules]


def _unsafe_peers_for_triton_import(modules: Mapping[str, Any]) -> list[str]:
    peers = loaded_llvm_peer_modules(modules)
    if not sys.platform == "darwin":
        return peers
    if triton_native_symbols_are_local(modules) or not triton_native_loaded(modules):
        return [name for name in peers if name not in _TILELANG_LLVM_PEER_MODULES]
    return peers


def triton_import_block_reason(
    loaded_modules: Optional[Mapping[str, Any]] = None,
) -> Optional[str]:
    """Return a human-readable reason to avoid importing/calling Triton now.

    Triton and the TileLang/TVM/shim side can each register LLVM command-line
    options. If a non-Triton LLVM peer (TileLang, TVM, the C++ shim, or
    jaxlib's bundled MLIR/LLVM nanobind extension) is resident in the
    interpreter, importing or calling native Triton can abort the process on
    duplicate LLVM cl::opt registration -- even if ``triton._C.libtriton``
    itself has already been imported (the abort happens on the next attempt
    to spin up or load another LLVM image). In that state callers should
    report Triton as unavailable and run live Triton checks in a fresh
    subprocess.
    """
    modules = _loaded_modules(loaded_modules)
    peers = _unsafe_peers_for_triton_import(modules)
    if peers:
        return (
            "triton import blocked / triton compile blocked because "
            + ", ".join(peers)
            + " already loaded in this process; calling triton._C.libtriton "
            "on top can abort on duplicate LLVM cl::opt registration. "
            "Re-run the Triton-dependent check in a fresh Python process."
        )
    return None


def triton_compile_block_reason(
    loaded_modules: Optional[Mapping[str, Any]] = None,
) -> Optional[str]:
    """Return a reason to avoid Triton TTIR generation in this process."""
    import_reason = triton_import_block_reason(loaded_modules)
    if import_reason is not None:
        return import_reason

    modules = _loaded_modules(loaded_modules)
    peers = _unsafe_peers_for_triton_import(modules)
    if triton_native_loaded(modules) and peers:
        return (
            "triton compile blocked because triton._C.libtriton and "
            + ", ".join(peers)
            + " are already loaded in this process; Triton TTIR generation "
            "can abort on duplicate LLVM cl::opt registration. Re-run the "
            "Triton-dependent check in a fresh Python process."
        )
    return None
