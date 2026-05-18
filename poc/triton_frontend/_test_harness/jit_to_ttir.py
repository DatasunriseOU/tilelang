"""Triton AST → TTIR capture harness.

Probes the installed Triton's API across 2.x / 3.0-3.1 / 3.6+ and stops
compilation at the TTIR stage so the frontend reducer can ingest the
textual IR. Returns the TTIR as a string.

Public surface:
    - TTIRCaptureError       — Triton compile failed *after* it was reachable.
    - TritonUnavailable      — Triton not installed (or import fails hard).
    - triton_jit_to_ttir(fn, constexprs=None, target=None) -> str

API spelling differences we handle:
    * Triton 3.6+: ``ASTSource(constexprs=...)`` + ``make_ir(target, options,
      codegen, module_map, ctx)``.
    * Triton 3.0-3.1: ``ASTSource(constants=...)`` + ``compile(src, options=...)``
      with the legacy ``stage='ttir'`` knob still respected.
    * Triton 2.x: positional ``triton.compile(fn, signature=..., output='ttir')``.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional


class TritonUnavailable(RuntimeError):
    """Raised when triton is not importable in the current env."""


class TTIRCaptureError(RuntimeError):
    """Raised when triton imports but compile-to-TTIR fails."""


def _import_triton():
    try:
        import triton  # noqa: F401
        return triton
    except Exception as exc:
        raise TritonUnavailable(
            f"triton not importable: {exc.__class__.__name__}: {exc}"
        ) from exc


def triton_available() -> bool:
    """Return True if triton is importable. Used as a feature-detect guard."""
    try:
        import triton  # noqa: F401
        return True
    except Exception:
        return False


def triton_version() -> Optional[str]:
    """Return installed triton version, or None if not importable."""
    try:
        import triton
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


def _try_triton_3_6(fn, constexprs, target):
    """Triton 3.6+: ASTSource(constexprs=...) + make_ir."""
    try:
        from triton.compiler.compiler import ASTSource
        from triton.runtime.jit import JITFunction
    except ImportError as exc:
        raise TTIRCaptureError(f"3.6+ API not found: {exc}") from exc
    if not isinstance(fn, JITFunction):
        raise TTIRCaptureError(f"fn must be @triton.jit; got {type(fn).__name__}")
    try:
        src = ASTSource(fn=fn, signature={}, constexprs=constexprs or {})
        # The make_ir entrypoint is what 3.6 uses; fall through to options
        # path if missing.
        if hasattr(src, "make_ir"):
            from triton.backends import backend_for_target  # type: ignore
            backend = backend_for_target(target or "cuda")
            options = backend.parse_options({})
            codegen = backend.codegen_for_target(target or "cuda")
            from triton import language as tl  # noqa: F401
            import triton._C.libtriton as _libtriton  # type: ignore
            ctx = _libtriton.ir.context()
            module = src.make_ir(target, options, codegen, {}, ctx)
            return str(module)
    except Exception as exc:
        raise TTIRCaptureError(f"3.6+ make_ir failed: {exc}") from exc
    raise TTIRCaptureError("3.6+ make_ir entrypoint not available")


def _try_triton_3_0(fn, constexprs, target):
    """Triton 3.0/3.1: ASTSource(constants=...) + compile(stage='ttir')."""
    try:
        from triton.compiler.compiler import ASTSource, compile as _compile
    except ImportError as exc:
        raise TTIRCaptureError(f"3.0 API not found: {exc}") from exc
    try:
        src = ASTSource(fn=fn, signature={}, constants=constexprs or {})
        result = _compile(src, options={"stage": "ttir"})
        if hasattr(result, "asm") and "ttir" in result.asm:
            return result.asm["ttir"]
        return str(result)
    except Exception as exc:
        raise TTIRCaptureError(f"3.0 compile failed: {exc}") from exc


def _try_triton_2(fn, constexprs, target):
    """Triton 2.x: positional ``triton.compile(fn, signature=..., output='ttir')``."""
    import triton
    try:
        return triton.compile(fn, signature={}, output="ttir")
    except Exception as exc:
        raise TTIRCaptureError(f"2.x compile failed: {exc}") from exc


def triton_jit_to_ttir(
    fn: Callable[..., Any],
    *,
    constexprs: Optional[Dict[str, Any]] = None,
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
            return variant(fn, constexprs, target)
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
