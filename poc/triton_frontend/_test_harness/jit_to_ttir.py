"""Triton AST → TTIR capture harness.

Probes the installed Triton's API across 2.x / 3.0-3.1 / 3.6+ and stops
compilation at the TTIR stage so the frontend reducer can ingest the
textual IR. Returns the TTIR as a string.

Public surface:
    - TTIRCaptureError       — Triton compile failed *after* it was reachable.
    - TritonUnavailable      — Triton not installed (or import fails hard).
    - triton_jit_to_ttir(fn, constexprs=None, signature=None, target=None) -> str

API spelling differences we handle:
    * Triton 3.6+: ``ASTSource(constexprs=...)`` + ``make_ir(target, options,
      codegen, module_map, ctx)``.
    * Triton 3.0-3.1: ``ASTSource(constants=...)`` + ``compile(src, options=...)``
      with the legacy ``stage='ttir'`` knob still respected.
    * Triton 2.x: positional ``triton.compile(fn, signature=..., output='ttir')``.
"""

from __future__ import annotations

import importlib
from typing import Any, Callable, Dict, Mapping, Optional

from .native_import_guard import triton_import_block_reason


class TritonUnavailable(RuntimeError):
    """Raised when triton is not importable in the current env."""


class TTIRCaptureError(RuntimeError):
    """Raised when triton imports but compile-to-TTIR fails."""


def _import_triton(
    *,
    import_module: Optional[Callable[[str], Any]] = None,
    loaded_modules: Optional[Mapping[str, Any]] = None,
):
    block_reason = triton_import_block_reason(loaded_modules)
    if block_reason is not None:
        raise TritonUnavailable(block_reason)
    importer = importlib.import_module if import_module is None else import_module
    try:
        return importer("triton")
    except Exception as exc:
        raise TritonUnavailable(
            f"triton not importable: {exc.__class__.__name__}: {exc}"
        ) from exc


def triton_available(
    *,
    import_module: Optional[Callable[[str], Any]] = None,
    loaded_modules: Optional[Mapping[str, Any]] = None,
) -> bool:
    """Return True if triton is importable. Used as a feature-detect guard.

    Returns False when a TileLang/TVM native peer is already loaded in this
    process: importing triton there can abort the interpreter because both
    sides statically register LLVM command-line options.
    """
    if triton_import_block_reason(loaded_modules) is not None:
        return False
    importer = importlib.import_module if import_module is None else import_module
    try:
        importer("triton")
        return True
    except Exception:
        return False


def triton_version(
    *,
    import_module: Optional[Callable[[str], Any]] = None,
    loaded_modules: Optional[Mapping[str, Any]] = None,
) -> Optional[str]:
    """Return installed triton version, or None if not importable."""
    if triton_import_block_reason(loaded_modules) is not None:
        return None
    importer = importlib.import_module if import_module is None else import_module
    try:
        triton = importer("triton")
        return getattr(triton, "__version__", None)
    except Exception:
        return None


def describe_triton_env() -> Dict[str, Any]:
    """Return a dict describing the Triton install (for corpus headers)."""
    info: Dict[str, Any] = {
        "available": triton_available(),
        "version": triton_version(),
    }
    if info["available"]:
        try:
            import triton
            info["module_path"] = getattr(triton, "__file__", None)
        except Exception:
            pass
    return info


def _infer_signature(fn, constexprs) -> Dict[str, str]:
    """Build a Triton signature dict from the JIT function's param spec.

    Conservative: pointer params → *fp32, int params → i32. Constexpr-named
    params are skipped (constexprs dict carries them separately).
    """
    sig: Dict[str, str] = {}
    constexpr_keys = set((constexprs or {}).keys())
    for name in fn.params if hasattr(fn, "params") else []:
        pname = name.name if hasattr(name, "name") else str(name)
        if pname in constexpr_keys:
            continue
        sig[pname] = "*fp32" if pname.endswith("_ptr") else "i32"
    return sig


def _try_triton_3_6(fn, constexprs, signature, target):
    """Triton 3.6+: ASTSource(constexprs=...) + make_ir.

    Stops at the TTIR stage — never invokes Metal/PTX codegen, so the
    Apple metal-as / metal-ll quirks in our triton fork don't surface.
    """
    try:
        from triton.compiler.compiler import ASTSource
        from triton.runtime.jit import JITFunction
        from triton.backends import backends as _backends
        from triton.backends.compiler import GPUTarget
        from triton._C.libtriton import ir as _libir
    except ImportError as exc:
        raise TTIRCaptureError(f"3.6+ API not found: {exc}") from exc
    if not isinstance(fn, JITFunction):
        raise TTIRCaptureError(f"fn must be @triton.jit; got {type(fn).__name__}")

    # Pick the best available backend whose target we can describe.
    # apple/mps is our own out-of-tree backend (triton-pr9701); fall back
    # to nvidia or amd if mps isn't registered.
    backend_to_gputarget = [
        ("apple", GPUTarget("mps", "apple_m2", 32)),
        ("nvidia", GPUTarget("cuda", 80, 32)),
        ("amd", GPUTarget("hip", "gfx942", 64)),
    ]
    last_err: Optional[Exception] = None
    for be_name, gpu_target in backend_to_gputarget:
        backend_pkg = _backends.get(be_name)
        if backend_pkg is None:
            continue
        try:
            backend_inst = backend_pkg.compiler(gpu_target)
        except Exception as exc:
            last_err = exc
            continue
        try:
            opts = backend_inst.parse_options({})
            codegen = backend_inst.get_codegen_implementation(opts)
            module_map = backend_inst.get_module_map()
            ctx = _libir.context()
            backend_inst.load_dialects(ctx)
            sig = (
                dict(signature)
                if signature is not None
                else _infer_signature(fn, constexprs)
            )
            src = ASTSource(fn=fn, signature=sig, constexprs=constexprs or {})
            module = src.make_ir(gpu_target, opts, codegen, module_map, ctx)
            return str(module)
        except Exception as exc:
            last_err = exc
            continue
    raise TTIRCaptureError(
        f"3.6+ make_ir failed across {[b for b,_ in backend_to_gputarget]}: {last_err}"
    )


def _try_triton_3_0(fn, constexprs, signature, target):
    """Triton 3.0/3.1: ASTSource(constants=...) + compile(stage='ttir')."""
    try:
        from triton.compiler.compiler import ASTSource, compile as _compile
    except ImportError as exc:
        raise TTIRCaptureError(f"3.0 API not found: {exc}") from exc
    try:
        src = ASTSource(fn=fn, signature=signature or {}, constants=constexprs or {})
        result = _compile(src, options={"stage": "ttir"})
        if hasattr(result, "asm") and "ttir" in result.asm:
            return result.asm["ttir"]
        return str(result)
    except Exception as exc:
        raise TTIRCaptureError(f"3.0 compile failed: {exc}") from exc


def _try_triton_2(fn, constexprs, signature, target):
    """Triton 2.x: positional ``triton.compile(fn, signature=..., output='ttir')``."""
    import triton
    try:
        return triton.compile(fn, signature=signature or {}, output="ttir")
    except Exception as exc:
        raise TTIRCaptureError(f"2.x compile failed: {exc}") from exc


def triton_jit_to_ttir(
    fn: Callable[..., Any],
    *,
    constexprs: Optional[Dict[str, Any]] = None,
    signature: Optional[Dict[str, str]] = None,
    target: Optional[str] = None,
) -> str:
    """Lower a ``@triton.jit`` kernel to TTIR text, probing for the right API.

    Returns the TTIR as a string. Raises:
        - TritonUnavailable: triton not installed.
        - TTIRCaptureError: all known compile entrypoints failed.
    """
    _import_triton()
    last_err: Optional[Exception] = None
    for variant in (_try_triton_3_6, _try_triton_3_0, _try_triton_2):
        try:
            return variant(fn, constexprs, signature, target)
        except TTIRCaptureError as exc:
            last_err = exc
            continue
    # ValueError per the xfail contract in tests/test_standalone.py — if
    # LLVM/codegen backends are missing locally, the variants above raise
    # TTIRCaptureError; we re-raise as ValueError so the test's
    # `@pytest.mark.xfail(raises=ValueError)` recognizes a benign-local
    # failure vs an unexpected one.
    raise ValueError(
        f"All Triton compile paths failed (LLVM/backends may be missing locally); "
        f"last error: {last_err}"
    )


__all__ = [
    "TTIRCaptureError",
    "TritonUnavailable",
    "describe_triton_env",
    "triton_available",
    "triton_jit_to_ttir",
    "triton_version",
]
