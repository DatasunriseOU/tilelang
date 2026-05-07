"""End-to-end numeric verification harness.

Runs each kernel in :mod:`numeric_kernels` through the full pipeline:

    @triton.jit kernel
        |
        v
    triton.compiler.compile(...)            -> TTIR text
        |
        v
    poc.triton_frontend.from_ttir(...)      -> tvm.tir.PrimFunc
        |
        v
    tilelang.compile(prim, target="metal")  -> CompiledArtifact
        |
        v
    mx.fast.metal_kernel(src, ...)          -> callable
        |
        v
    np.allclose(output, expected, ...)      -> NUMERIC_PASS / NUMERIC_DIVERGE

Each step has a precise failure label (see :data:`Verdict`). On missing
deps (``triton`` / ``tilelang`` / ``mlx`` / ``tvm`` not importable) the
kernel is reported as ``SKIP: <component> unavailable`` rather than
errored out -- this keeps the harness usable on hosts without the full
toolchain (e.g. CI runners that only have numpy).

We never substitute the numpy reference for the kernel output. If the
GPU path fails for any reason, the verdict reflects that.

Public entry point: :func:`run_all` -- runs every kernel listed in
``KERNEL_MODULES`` and writes a markdown report to
``/tmp/triton_e2e_numeric.md``.
"""
from __future__ import annotations

import dataclasses
import importlib
import io
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from . import numeric_kernels


__all__ = [
    "Verdict",
    "KernelResult",
    "run_one",
    "run_all",
    "REPORT_PATH",
]


REPORT_PATH = Path("/tmp/triton_e2e_numeric.md")


# ---------------------------------------------------------------------------
# Verdict labels
#
# Order matters for aggregate reporting -- earlier labels are "earlier"
# in the pipeline, so when the harness summary lists "first failing step"
# we use the label index.
# ---------------------------------------------------------------------------


class Verdict:
    """Verdict label namespace -- string constants only.

    We use a class instead of a StrEnum so the labels stay simple
    strings (StrEnum requires Python 3.11+ and complicates JSON dumps).
    """

    SKIP = "SKIP"
    TTIR_FAIL = "TTIR_FAIL"
    LOWER_FAIL = "LOWER_FAIL"
    COMPILE_FAIL = "COMPILE_FAIL"
    RUNTIME_FAIL = "RUNTIME_FAIL"
    NUMERIC_DIVERGE = "NUMERIC_DIVERGE"
    NUMERIC_PASS = "NUMERIC_PASS"


@dataclasses.dataclass
class KernelResult:
    """Outcome of running one kernel through the full pipeline."""

    name: str
    verdict: str
    detail: str = ""
    elapsed_s: float = 0.0
    max_abs_err: Optional[float] = None
    max_rel_err: Optional[float] = None
    first_mismatches: Optional[List[Tuple[int, ...]]] = None

    def short(self) -> str:
        """One-line summary for the markdown table."""
        head = f"{self.name}: {self.verdict}"
        if self.detail:
            head += f" -- {self.detail}"
        return head


# ---------------------------------------------------------------------------
# Dep probing
# ---------------------------------------------------------------------------


def _probe_deps() -> Dict[str, Optional[str]]:
    """Return ``{component: error_str_or_None}`` for each pipeline dep.

    A non-None value means the component is unavailable; the harness
    will SKIP each kernel with that error string.
    """
    deps: Dict[str, Optional[str]] = {
        "triton": None,
        "tvm": None,
        "tilelang": None,
        "mlx": None,
    }
    for name in deps:
        try:
            importlib.import_module(name if name != "mlx" else "mlx.core")
        except Exception as exc:  # noqa: BLE001 -- broad-by-design, recorded
            deps[name] = f"{type(exc).__name__}: {exc}"
    return deps


# ---------------------------------------------------------------------------
# Stage 1: TTIR capture
# ---------------------------------------------------------------------------


def _capture_ttir(kernel_mod: Any) -> Tuple[Optional[str], Optional[str]]:
    """Run ``triton.compiler.compile`` to obtain TTIR text.

    Returns ``(ttir_text, error)``. On failure ``ttir_text is None`` and
    ``error`` is the diagnostic.
    """
    try:
        import triton  # type: ignore
        from triton.compiler.compiler import ASTSource  # type: ignore
        from triton.compiler import compile as triton_compile  # type: ignore
    except Exception as exc:  # noqa: BLE001
        return None, f"triton import failed: {type(exc).__name__}: {exc}"

    fn = kernel_mod.TRITON_KERNEL
    if fn is None:
        return None, "TRITON_KERNEL is None (triton failed to import in kernel module)"

    # The signature/constants split changed across Triton minor versions.
    # We try the modern form first (signature dict + constants dict);
    # callers that hit older Triton hosts can fall through to the wider
    # try/except.
    try:
        # Build a minimal signature: every non-constexpr parameter is a
        # raw pointer (``*fp32``) or scalar int. We don't actually run
        # Triton's autotune -- we only need TTIR -- so the signature
        # types only need to type-check at the TTIR layer.
        # NOTE: this is intentionally hand-rolled; if a kernel takes an
        # exotic dtype, override by patching its module to provide
        # ``TTIR_SIGNATURE``.
        signature = getattr(kernel_mod, "TTIR_SIGNATURE", None)
        if signature is None:
            signature = _default_signature(fn, kernel_mod.META_ARGS)
        src = ASTSource(
            fn=fn,
            signature=signature,
            constants=kernel_mod.META_ARGS,
        )
        compiled = triton_compile(src, options={"stage": "ttir"})
        asm = getattr(compiled, "asm", None)
        if asm is None:
            return None, f"triton compile returned no .asm: {compiled!r}"
        ttir = asm.get("ttir")
        if ttir is None:
            return None, f"triton compile produced no 'ttir' key in asm: keys={list(asm)}"
        return str(ttir), None
    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc(limit=4)
        return None, f"triton.compile raised: {type(exc).__name__}: {exc}\n{tb}"


def _default_signature(fn: Any, constants: Dict[str, Any]) -> Dict[str, str]:
    """Build a default Triton signature from ``fn``'s parameter names.

    Convention: any parameter ending in ``_ptr`` is a pointer to fp32;
    any name in ``constants`` is dropped; everything else is i32.
    """
    arg_names = getattr(fn, "arg_names", None)
    if arg_names is None:
        # Fallback: introspect via __wrapped__ or func_signature.
        try:
            import inspect

            sig = inspect.signature(getattr(fn, "fn", fn))
            arg_names = [p.name for p in sig.parameters.values()]
        except Exception:
            arg_names = []
    sig_dict: Dict[str, str] = {}
    for name in arg_names:
        if name in constants:
            continue
        if name.endswith("_ptr"):
            sig_dict[name] = "*fp32"
        else:
            sig_dict[name] = "i32"
    return sig_dict


# ---------------------------------------------------------------------------
# Stage 2: TTIR -> PrimFunc
# ---------------------------------------------------------------------------


def _lower_ttir(ttir_text: str, kernel_name: str) -> Tuple[Any, Optional[str]]:
    """Run ``poc.triton_frontend.from_ttir`` on the captured TTIR.

    The frontend prefers an ``mlir.ir.Module`` but accepts the textual
    coverage path via ``_allow_text_ttir=True``. We try MLIR first; if
    the bindings are missing we fall back to the text path -- the
    harness explicitly notes this in the verdict detail since it limits
    what the lowering can produce.
    """
    try:
        from poc.triton_frontend import from_ttir  # type: ignore
    except Exception as exc:  # noqa: BLE001
        return None, f"poc.triton_frontend import failed: {type(exc).__name__}: {exc}"

    # Try the real MLIR path first.
    try:
        from mlir import ir as _mlir_ir  # type: ignore

        ctx = _mlir_ir.Context()
        ctx.allow_unregistered_dialects = True
        module = _mlir_ir.Module.parse(ttir_text, ctx)
        prim = from_ttir(module, name=kernel_name)
        return prim, None
    except ImportError:
        # Fall through to text path.
        pass
    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc(limit=6)
        return None, (
            f"from_ttir(MLIR Module) raised: {type(exc).__name__}: {exc}\n{tb}"
        )

    # Text-walker fallback (coverage-only -- ctx.value_map / ctx.buffers
    # are NOT populated, so the resulting PrimFunc is a stub and TileLang
    # compile will likely fail at COMPILE_FAIL or produce a trivial empty
    # kernel that does not match the reference). We still attempt it so
    # the harness can report the *next* failing stage.
    try:
        prim = from_ttir(
            ttir_text,
            name=kernel_name,
            _allow_text_ttir=True,
        )
        return prim, "(text-walker fallback; mlir.ir bindings absent)"
    except NotImplementedError as exc:
        # OP_TABLE doesn't cover this op -- this is the LOWER_FAIL signal.
        return None, f"NotImplementedError: {exc}"
    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc(limit=6)
        return None, f"from_ttir(text) raised: {type(exc).__name__}: {exc}\n{tb}"


# ---------------------------------------------------------------------------
# Stage 3: TileLang compile -> Metal source
# ---------------------------------------------------------------------------


def _compile_metal(prim: Any) -> Tuple[Any, Optional[str]]:
    """Compile the PrimFunc to a Metal CompiledArtifact via TileLang."""
    try:
        import tilelang  # type: ignore
        import tvm  # type: ignore  # noqa: F401  -- needed for PassContext below
    except Exception as exc:  # noqa: BLE001
        return None, f"tilelang/tvm import failed: {type(exc).__name__}: {exc}"

    try:
        # Match the call shape used in
        # ``testing/python/metal/test_metal_codegen_linux.py`` so we
        # exercise the exact same path TileLang's own conformance does.
        with tvm.transform.PassContext(), tvm.target.Target("metal"):
            artifact = tilelang.lower(prim, target="metal")
        return artifact, None
    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc(limit=6)
        return None, f"tilelang.lower raised: {type(exc).__name__}: {exc}\n{tb}"


# ---------------------------------------------------------------------------
# Stage 4 + 5: MLX wrap + run
# ---------------------------------------------------------------------------


def _run_mlx(
    artifact: Any,
    kernel_args: List[np.ndarray],
    kernel_mod: Any,
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """Wrap the artifact's metal_source in mx.fast.metal_kernel and run.

    NOTE: TileLang's Metal artifact emits a Metal Shading Language
    function; ``mx.fast.metal_kernel`` expects a fragment that operates
    on ``inputs[i]`` arrays and writes to ``outputs[i]``. The two are
    NOT directly interchangeable -- we have to splice the body. For now
    we attempt a best-effort splice: if it fails, we return RUNTIME_FAIL
    with the diagnostic.
    """
    try:
        import mlx.core as mx  # type: ignore
    except Exception as exc:  # noqa: BLE001
        return None, f"mlx import failed: {type(exc).__name__}: {exc}"

    # Delegate buffer-name renaming + MLX wrapping to
    # ``cppmega_mlx.nn._tilelang._mlx_runtime.wrap_tilelang_metal_kernel``,
    # which parses the TileLang-emitted ``kernel void name(...) { ... }``
    # signature, renames the positional ``A``, ``B``, ... buffer params
    # to MLX's required ``inp0``/``out0`` convention in the body, and
    # builds the ``mx.fast.metal_kernel`` callable. See that module's
    # docstring for the rename strategy.
    try:
        from cppmega_mlx.nn._tilelang._mlx_runtime import (  # type: ignore
            MLXRuntimeError,
            wrap_tilelang_metal_kernel,
        )
    except Exception as exc:  # noqa: BLE001
        return None, (
            f"cppmega_mlx._mlx_runtime import failed: "
            f"{type(exc).__name__}: {exc}"
        )

    # Convention used by all kernels in numeric_kernels/: the LAST array
    # in ``kernel_args`` is the output, everything else is an input.
    *inputs_np, output_np = kernel_args

    try:
        adapter = wrap_tilelang_metal_kernel(
            artifact,
            input_count=len(inputs_np),
            output_count=1,
            name="triton_e2e_kernel",
        )
    except MLXRuntimeError as exc:
        return None, f"wrap_tilelang_metal_kernel: {exc}"
    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc(limit=6)
        return None, (
            f"wrap_tilelang_metal_kernel raised: "
            f"{type(exc).__name__}: {exc}\n{tb}"
        )

    try:
        mx_inputs = [mx.array(a) for a in inputs_np]
        grid = tuple(int(g) for g in kernel_mod.LAUNCH_GRID)
        # Pad grid to 3D as mx.fast.metal_kernel requires.
        while len(grid) < 3:
            grid = grid + (1,)
        threadgroup = (1, 1, 1)
        out_arrays = adapter(
            inputs=mx_inputs,
            output_shapes=[output_np.shape],
            output_dtypes=[_np_to_mx_dtype(mx, output_np.dtype)],
            grid=grid,
            threadgroup=threadgroup,
        )
        mx.eval(out_arrays)
        result_np = np.array(out_arrays[0], copy=False).astype(output_np.dtype)
        return result_np, None
    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc(limit=6)
        return None, f"mx.fast.metal_kernel raised: {type(exc).__name__}: {exc}\n{tb}"


def _np_to_mx_dtype(mx: Any, np_dtype: np.dtype) -> Any:
    """Map numpy dtype to mlx dtype. Only fp32/fp16/i32 are wired."""
    name = np.dtype(np_dtype).name
    table = {
        "float32": mx.float32,
        "float16": mx.float16,
        "int32": mx.int32,
    }
    if name not in table:
        raise ValueError(f"unsupported dtype for mlx wrap: {name}")
    return table[name]


# ---------------------------------------------------------------------------
# Stage 6: numeric compare
# ---------------------------------------------------------------------------


def _compare(
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    atol: float,
    rtol: float,
) -> Tuple[bool, float, float, List[Tuple[int, ...]]]:
    """Return ``(passed, max_abs, max_rel, first_5_mismatch_indices)``."""
    if actual.shape != expected.shape:
        return (
            False,
            float("inf"),
            float("inf"),
            [],
        )
    diff = np.abs(actual - expected)
    rel = diff / (np.abs(expected) + 1e-30)
    max_abs = float(diff.max())
    max_rel = float(rel.max())
    passed = bool(np.allclose(actual, expected, atol=atol, rtol=rtol))
    if passed:
        return True, max_abs, max_rel, []
    bad_mask = ~np.isclose(actual, expected, atol=atol, rtol=rtol)
    bad_idx = np.argwhere(bad_mask)
    first_five = [tuple(int(x) for x in idx) for idx in bad_idx[:5]]
    return False, max_abs, max_rel, first_five


# ---------------------------------------------------------------------------
# Per-kernel orchestration
# ---------------------------------------------------------------------------


def run_one(kernel_module_name: str, deps: Dict[str, Optional[str]]) -> KernelResult:
    """Run a single kernel module end-to-end and return a KernelResult."""
    started = time.monotonic()

    # If any pipeline component is unavailable we SKIP -- this is the
    # honest verdict (we can't even attempt the pipeline).
    for comp in ("triton", "tvm", "tilelang", "mlx"):
        if deps.get(comp):
            return KernelResult(
                name=kernel_module_name,
                verdict=Verdict.SKIP,
                detail=f"{comp} unavailable: {deps[comp]}",
                elapsed_s=time.monotonic() - started,
            )

    try:
        kernel_mod = importlib.import_module(
            f"poc.triton_frontend._test_harness.numeric_kernels.{kernel_module_name}"
        )
    except Exception as exc:  # noqa: BLE001
        return KernelResult(
            name=kernel_module_name,
            verdict=Verdict.SKIP,
            detail=f"kernel module import failed: {type(exc).__name__}: {exc}",
            elapsed_s=time.monotonic() - started,
        )

    # Stage 1: TTIR
    ttir_text, ttir_err = _capture_ttir(kernel_mod)
    if ttir_text is None:
        return KernelResult(
            name=kernel_module_name,
            verdict=Verdict.TTIR_FAIL,
            detail=ttir_err or "<unknown>",
            elapsed_s=time.monotonic() - started,
        )

    # Stage 2: PrimFunc
    prim, lower_err = _lower_ttir(ttir_text, kernel_module_name)
    if prim is None:
        return KernelResult(
            name=kernel_module_name,
            verdict=Verdict.LOWER_FAIL,
            detail=lower_err or "<unknown>",
            elapsed_s=time.monotonic() - started,
        )
    text_fallback_note = lower_err if isinstance(lower_err, str) else ""

    # Stage 3: Metal artifact
    artifact, compile_err = _compile_metal(prim)
    if artifact is None:
        return KernelResult(
            name=kernel_module_name,
            verdict=Verdict.COMPILE_FAIL,
            detail=(text_fallback_note + " " if text_fallback_note else "")
            + (compile_err or "<unknown>"),
            elapsed_s=time.monotonic() - started,
        )

    # Stage 4 + 5: MLX run
    args, expected = kernel_mod.make_inputs()
    actual, runtime_err = _run_mlx(artifact, args, kernel_mod)
    if actual is None:
        return KernelResult(
            name=kernel_module_name,
            verdict=Verdict.RUNTIME_FAIL,
            detail=runtime_err or "<unknown>",
            elapsed_s=time.monotonic() - started,
        )

    # Stage 6: compare
    passed, max_abs, max_rel, mismatches = _compare(
        actual, expected, atol=kernel_mod.ATOL, rtol=kernel_mod.RTOL
    )
    if passed:
        return KernelResult(
            name=kernel_module_name,
            verdict=Verdict.NUMERIC_PASS,
            detail=f"max_abs={max_abs:.3e} max_rel={max_rel:.3e}",
            elapsed_s=time.monotonic() - started,
            max_abs_err=max_abs,
            max_rel_err=max_rel,
        )
    return KernelResult(
        name=kernel_module_name,
        verdict=Verdict.NUMERIC_DIVERGE,
        detail=f"max_abs={max_abs:.3e} max_rel={max_rel:.3e}",
        elapsed_s=time.monotonic() - started,
        max_abs_err=max_abs,
        max_rel_err=max_rel,
        first_mismatches=mismatches,
    )


# ---------------------------------------------------------------------------
# Aggregate runner + report
# ---------------------------------------------------------------------------


def run_all(report_path: Optional[Path] = None) -> List[KernelResult]:
    """Run every kernel module and write a markdown report."""
    deps = _probe_deps()
    results: List[KernelResult] = []
    for mod_name in numeric_kernels.KERNEL_MODULES:
        results.append(run_one(mod_name, deps))

    if report_path is None:
        report_path = REPORT_PATH
    _write_report(results, deps, report_path)
    return results


def _write_report(
    results: List[KernelResult],
    deps: Dict[str, Optional[str]],
    report_path: Path,
) -> None:
    """Write a markdown summary at ``report_path``."""
    counts: Dict[str, int] = {}
    for r in results:
        counts[r.verdict] = counts.get(r.verdict, 0) + 1

    buf = io.StringIO()
    buf.write("# Triton -> TileLang -> Metal -> MLX numeric smoke\n\n")
    buf.write("Generated by `poc.triton_frontend._test_harness.numeric_smoke`.\n\n")

    buf.write("## Dependency probe\n\n")
    buf.write("| Component | Status |\n|---|---|\n")
    for k, v in deps.items():
        status = "OK" if v is None else f"MISSING ({v})"
        buf.write(f"| `{k}` | {status} |\n")
    buf.write("\n")

    buf.write("## Per-kernel verdict\n\n")
    buf.write("| Kernel | Verdict | Elapsed (s) | Detail |\n|---|---|---|---|\n")
    for r in results:
        det = r.detail.replace("\n", " <br> ")
        buf.write(f"| `{r.name}` | **{r.verdict}** | {r.elapsed_s:.3f} | {det} |\n")
    buf.write("\n")

    buf.write("## Aggregate counts\n\n")
    for label in (
        Verdict.NUMERIC_PASS,
        Verdict.NUMERIC_DIVERGE,
        Verdict.RUNTIME_FAIL,
        Verdict.COMPILE_FAIL,
        Verdict.LOWER_FAIL,
        Verdict.TTIR_FAIL,
        Verdict.SKIP,
    ):
        buf.write(f"- {label}: **{counts.get(label, 0)}**\n")
    buf.write("\n")

    # Detailed per-kernel diagnostics (full multi-line details).
    buf.write("## Diagnostics (full)\n\n")
    for r in results:
        buf.write(f"### `{r.name}` -- {r.verdict}\n\n")
        buf.write("```\n")
        buf.write(r.detail or "(no detail)")
        buf.write("\n```\n\n")
        if r.first_mismatches:
            buf.write("First 5 mismatched indices:\n\n")
            for idx in r.first_mismatches:
                buf.write(f"- {idx}\n")
            buf.write("\n")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(buf.getvalue())


if __name__ == "__main__":  # pragma: no cover -- manual invocation
    res = run_all()
    for r in res:
        print(r.short())
    print(f"\nReport: {REPORT_PATH}")
