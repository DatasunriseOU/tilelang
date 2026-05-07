"""Triton ``@triton.jit`` -> TTIR text conversion.

Thin wrapper that captures the *first-stage* TTIR text from a Python
Triton kernel. The reducer in :mod:`poc.triton_frontend` accepts both
textual TTIR and ``mlir.ir.Module`` objects; this helper produces the
text form so callers don't need to import ``mlir.ir`` themselves.

Why a separate helper?
----------------------
``poc.triton_frontend._compile_to_ttir`` already exists but is module
private and bound into the larger ``from_triton_kernel`` flow. The
harness wants to:

  * call the Triton compiler with explicit option / signature shapes
    so it can be parameterised per kernel in the corpus,
  * record the raw TTIR text for diagnostics in the report,
  * tolerate Triton-version skew across 2.x / 3.x without bringing the
    failure into the reducer's coverage.

We do NOT modify the reducer. If Triton is unavailable in this Python
environment, every call here returns ``None`` and the harness uses
canned TTIR fixtures instead.
"""
from __future__ import annotations

import importlib
import inspect
from typing import Any, Callable, Dict, Optional, Tuple

__all__ = [
    "TritonUnavailable",
    "TTIRCaptureError",
    "triton_available",
    "triton_jit_to_ttir",
    "describe_triton_env",
]


class TritonUnavailable(RuntimeError):
    """Triton (or one of its hard deps such as torch) is not importable."""


class TTIRCaptureError(RuntimeError):
    """Triton imported, but the compile-to-TTIR call raised."""


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------


def triton_available() -> bool:
    """Return True iff ``import triton`` and ``import triton.compiler`` succeed."""
    try:
        importlib.import_module("triton")
        importlib.import_module("triton.compiler")
        return True
    except Exception:
        return False


def describe_triton_env() -> Dict[str, Any]:
    """Return a small dict of triton/torch/mlir version hints for the report.

    Never raises -- missing modules are recorded as ``None``.
    """
    out: Dict[str, Any] = {}
    for mod_name in ("triton", "torch", "tvm", "tilelang"):
        try:
            m = importlib.import_module(mod_name)
            out[mod_name] = getattr(m, "__version__", "unknown")
        except Exception as exc:
            out[mod_name] = f"not importable ({type(exc).__name__})"
    # mlir.ir lives behind an env var and is the make-or-break for the full
    # walker; record its availability separately.
    try:
        importlib.import_module("mlir.ir")
        out["mlir.ir"] = "available"
    except Exception as exc:
        out["mlir.ir"] = f"not importable ({type(exc).__name__})"
    return out


# ---------------------------------------------------------------------------
# Compile-to-TTIR
# ---------------------------------------------------------------------------


def _signature_from_constexprs(
    fn: Callable[..., Any], constexprs: Optional[Dict[str, Any]]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Build (signature, constants) dicts compatible with both Triton 2.x/3.x.

    Triton 3.x's ``ASTSource`` takes:
        signature: {arg_name: triton_dtype_string}
        constants: {arg_name: python_value}
    For pointer arguments we default to ``"*fp32"``; for ``constexpr`` we
    use the user-supplied value. The harness only needs Triton to reach the
    TTIR stage -- correctness of pointer dtype is not material since the
    reducer never executes.
    """
    constexprs = dict(constexprs or {})
    signature: Dict[str, Any] = {}
    constants: Dict[str, Any] = {}
    sig = inspect.signature(fn)
    for name, _param in sig.parameters.items():
        if name in constexprs:
            constants[name] = constexprs[name]
            # Triton 3.6 expects the literal ``"constexpr"`` string in the
            # signature dict alongside the constexpr value in
            # ``constexprs={...}``.
            signature[name] = "constexpr"
            continue
        # Default heuristic: arguments whose name ends with ``_ptr`` /
        # ``_ptr32`` look like pointers; everything else is treated as i32.
        if name.endswith("_ptr") or "ptr" in name.lower() or name.endswith("_ptr32"):
            signature[name] = "*fp32"
        else:
            signature[name] = "i32"
    return signature, constants


def triton_jit_to_ttir(
    fn: Callable[..., Any],
    *,
    constexprs: Optional[Dict[str, Any]] = None,
    target: Optional[str] = "cuda",
) -> str:
    """Compile a ``@triton.jit`` kernel just far enough to capture TTIR text.

    The reducer accepts the returned string via
    :func:`poc.triton_frontend.from_ttir`. Raises
    :class:`TritonUnavailable` when Triton can't be imported and
    :class:`TTIRCaptureError` when the compile call itself fails.

    Notes
    -----
    Triton's compiler API has shifted across versions. We try, in order:

      1. ``triton.compiler.compile(ASTSource(...), options={"stage": "ttir"})``
      2. ``triton.compiler.compile(fn, signature=..., output="ttir")``

    Both forms are documented in the wild; we accept whichever works.
    """
    if not triton_available():
        raise TritonUnavailable(
            "import triton failed; install triton (and torch) or run the "
            "harness with canned TTIR fixtures instead."
        )

    triton = importlib.import_module("triton")  # noqa: F841 -- import probe
    compiler = importlib.import_module("triton.compiler")
    # Triton 3.6 ``ASTSource`` expects the JITFunction (not the unwrapped
    # python function); only fall back to ``fn.fn`` when introspecting the
    # plain signature for ``_signature_from_constexprs``.
    raw_fn = getattr(fn, "fn", fn)
    jit_fn = fn  # may already be a JITFunction
    signature, constants = _signature_from_constexprs(raw_fn, constexprs)

    primary_exc: Optional[BaseException] = None

    # --- Attempt 1: Triton 3.6 make_ir() path (no compile() stage knob). ---
    #
    # Triton 3.6 dropped ``triton.compiler.compile(src, options={"stage":"ttir"})``.
    # The replacement is ``ASTSource.make_ir(target, options, codegen_fns,
    # module_map, ctx)`` which returns an ``ir.module`` whose ``str()`` is
    # the TTIR text. We have to construct the per-target backend ourselves
    # to source ``options``, ``codegen_fns``, and ``module_map``.
    try:
        ast_mod = importlib.import_module("triton.compiler.compiler")
        ASTSource = getattr(ast_mod, "ASTSource", None)
        if ASTSource is None:
            raise RuntimeError("ASTSource not present on triton.compiler.compiler")

        # Probe whether ASTSource is the 3.6 form (constexprs kwarg) or the
        # 3.0/3.1 form (constants kwarg).
        import inspect as _inspect

        ast_init_params = set(
            _inspect.signature(ASTSource.__init__).parameters
        )
        if "constexprs" in ast_init_params:
            src = ASTSource(
                fn=jit_fn, signature=signature, constexprs=constants
            )
        else:
            # Older 3.x form -- accepts ``constants=`` directly.
            src = ASTSource(fn=jit_fn, signature=signature, constants=constants)

        # Has ``make_ir``? -> 3.6+ path. Construct backend + ctx.
        if hasattr(src, "make_ir"):
            from triton._C.libtriton import ir as _ir  # type: ignore
            from triton.backends.compiler import GPUTarget  # type: ignore
            from triton.backends import backends as _backends  # type: ignore

            # Map our string ``target`` to a real GPUTarget. Prefer driver
            # autodetect; fall back to a hand-built one for headless hosts.
            gpu_target = None
            try:
                runtime = importlib.import_module("triton.runtime")
                gpu_target = runtime.driver.active.get_current_target()
            except Exception:
                gpu_target = None
            if gpu_target is None:
                # Heuristic mapping; the harness only needs TTIR which is
                # mostly target-independent.
                if target == "metal" or target == "mps":
                    gpu_target = GPUTarget(
                        backend="mps", arch="apple_m", warp_size=32
                    )
                else:
                    gpu_target = GPUTarget(
                        backend="cuda", arch="sm_80", warp_size=32
                    )

            # The registry key is the *package* name (``apple``) but the
            # GPUTarget.backend identifier is the device class (``mps``).
            # Search by ``supports_target`` to be robust across versions.
            backend_entry = _backends.get(gpu_target.backend)
            if backend_entry is None:
                for _name, _entry in _backends.items():
                    try:
                        if _entry.compiler.supports_target(gpu_target):
                            backend_entry = _entry
                            break
                    except Exception:
                        continue
            if backend_entry is None:
                raise RuntimeError(
                    f"no triton backend registered for "
                    f"{gpu_target.backend!r} (registry keys: "
                    f"{list(_backends.keys())})"
                )
            be = backend_entry.compiler(gpu_target)
            options = be.parse_options({})
            ctx = _ir.context()
            _ir.load_dialects(ctx)
            be.load_dialects(ctx)
            codegen = be.get_codegen_implementation(options)
            module_map = be.get_module_map()
            module = src.make_ir(gpu_target, options, codegen, module_map, ctx)
            return str(module)

        # Otherwise: 3.0/3.1 fallback via compile() with options dict.
        compiled = compiler.compile(src, target=target, options={"stage": "ttir"})
        if hasattr(compiled, "asm"):
            ttir = compiled.asm.get("ttir")
            if isinstance(ttir, str) and ttir:
                return ttir
        if isinstance(compiled, str):
            return compiled
    except Exception as exc:  # noqa: BLE001 -- fall through to legacy path
        primary_exc = exc

    # --- Attempt 2: legacy positional API (Triton 2.x). ---
    try:
        legacy = compiler.compile(raw_fn, signature=constants, output="ttir")
        if isinstance(legacy, str):
            return legacy
        if hasattr(legacy, "asm"):
            ttir = legacy.asm.get("ttir")
            if isinstance(ttir, str):
                return ttir
        raise TTIRCaptureError(
            f"Triton legacy compile returned unexpected type {type(legacy)!r}"
        )
    except TTIRCaptureError:
        raise
    except Exception as exc:
        msg = f"Triton TTIR capture failed; primary={primary_exc!r}, legacy={exc!r}"
        raise TTIRCaptureError(msg) from exc

    # If we reach here both paths returned objects without TTIR text.
    raise TTIRCaptureError(
        f"Triton compile produced no 'ttir' asm field; primary error={primary_exc!r}"
    )
