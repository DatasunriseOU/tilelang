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
deps (``triton`` / ``tilelang`` / ``mlx`` / ``cppmega_mlx`` / ``tvm`` not
importable) the kernel is reported as ``SKIP: <component> unavailable``
rather than errored out -- this keeps the harness usable on hosts without
the full toolchain (e.g. CI runners that only have numpy).

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
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable
from collections.abc import Mapping

import numpy as np


if __package__ in (None, ""):
    _REPO_ROOT = Path(__file__).resolve().parents[3]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    __package__ = "poc.triton_frontend._test_harness"

from . import numeric_kernels  # noqa: E402
from .native_import_guard import triton_compile_block_reason


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
    max_abs_err: float | None = None
    max_rel_err: float | None = None
    first_mismatches: list[tuple[int, ...]] | None = None

    def short(self) -> str:
        """One-line summary for the markdown table."""
        head = f"{self.name}: {self.verdict}"
        if self.detail:
            head += f" -- {self.detail}"
        return head


# ---------------------------------------------------------------------------
# Dep probing
# ---------------------------------------------------------------------------


def _probe_deps(
    *,
    import_module: Callable[[str], Any] | None = None,
    loaded_modules: Mapping[str, Any] | None = None,
) -> dict[str, str | None]:
    """Return ``{component: error_str_or_None}`` for each pipeline dep.

    A non-None value means the component is unavailable; the harness
    will SKIP each kernel with that error string.
    """
    importer = importlib.import_module if import_module is None else import_module
    deps: dict[str, str | None] = {
        "triton": None,
        "tvm": None,
        "tilelang": None,
        "mlx": None,
        "cppmega_mlx": None,
    }
    for name in deps:
        if name == "triton":
            block_reason = triton_compile_block_reason(loaded_modules)
            if block_reason is not None:
                deps[name] = f"RuntimeError: {block_reason}"
                continue
        try:
            module_name = "mlx.core" if name == "mlx" else name
            if name == "cppmega_mlx":
                module_name = "cppmega_mlx.nn._tilelang._mlx_runtime"
            importer(module_name)
        except Exception as exc:  # noqa: BLE001 -- broad-by-design, recorded
            deps[name] = f"{type(exc).__name__}: {exc}"
    return deps


# ---------------------------------------------------------------------------
# Stage 1: TTIR capture
# ---------------------------------------------------------------------------


def _capture_ttir(
    kernel_mod: Any,
) -> tuple[str | None, str | None, dict[str, Any]]:
    """Run ``triton.compiler.compile`` to obtain TTIR text.

    Returns ``(ttir_text, error, options)``. ``options`` is a small dict
    exposing Triton's parsed compile options (``num_warps``, ``num_stages``)
    so the reducer downstream can stamp matching attrs on the resulting
    PrimFunc — TileLang's ``gemm.lower`` reads ``num_warps`` from the
    ``threadIdx.x`` ``thread_extent`` AttrStmt the reducer synthesises
    from these. On failure ``ttir_text is None``, ``error`` is the
    diagnostic, and ``options`` is the Triton-default ``{4, 2}``.
    """
    _default_options: dict[str, Any] = {"num_warps": 4, "num_stages": 2}

    # Peer guard: if the current interpreter has already loaded a
    # native LLVM peer that conflicts with ``triton._C.libtriton``
    # (notably ``jaxlib.mlir._mlir_libs`` -- pulled in by the TTIR
    # walker's ``local_jaxlib_mlir_ir`` resolver), an in-process
    # ``make_ir`` will abort. Route the TTIR capture to a fresh
    # subprocess in that case so the harness keeps working without
    # killing the host process.
    from poc.triton_frontend._test_harness.native_import_guard import (
        triton_import_block_reason,
    )

    if triton_import_block_reason() is not None:
        from poc.triton_frontend._test_harness import jit_to_ttir as _jit

        module_name = getattr(kernel_mod, "__name__", None)
        if not module_name:
            return None, (
                "peer LLVM module resident and kernel module has no "
                "__name__; cannot subprocess-capture TTIR"
            ), _default_options
        meta = getattr(kernel_mod, "META_ARGS", {}) or {}
        sig = getattr(kernel_mod, "TTIR_SIGNATURE", None)
        try:
            ttir_text = _jit.triton_jit_to_ttir_subprocess(
                module_name=module_name,
                kernel_attr="TRITON_KERNEL",
                constexprs=dict(meta),
                signature=dict(sig) if isinstance(sig, dict) else None,
            )
        except _jit.TTIRCaptureError as exc:
            return None, f"subprocess TTIR capture failed: {exc}", _default_options
        opts = {
            "num_warps": int(meta.get("num_warps", _default_options["num_warps"])),
            "num_stages": int(meta.get("num_stages", _default_options["num_stages"])),
        }
        return ttir_text, None, opts

    try:
        import triton  # type: ignore  # noqa: F401  -- import probe
        from triton.compiler.compiler import ASTSource  # type: ignore
    except Exception as exc:  # noqa: BLE001
        return None, f"triton import failed: {type(exc).__name__}: {exc}", _default_options

    fn = kernel_mod.TRITON_KERNEL
    if fn is None:
        return (
            None,
            "TRITON_KERNEL is None (triton failed to import in kernel module)",
            _default_options,
        )

    # Triton 3.6 dropped the ``compile(src, options={"stage":"ttir"})`` knob.
    # The replacement is ``ASTSource.make_ir(target, options, codegen_fns,
    # module_map, ctx)`` which returns an MLIR module whose ``str()`` is
    # the TTIR text. Older 3.x compile-with-stage flow is kept as a
    # fallback so this helper still works on 3.0/3.1 hosts.
    try:
        signature = getattr(kernel_mod, "TTIR_SIGNATURE", None)
        if signature is None:
            signature = _default_signature(fn, kernel_mod.META_ARGS)
            # In Triton 3.6 every signature key (including constexprs) must
            # be present; constexprs use the literal string "constexpr".
            for k in kernel_mod.META_ARGS:
                signature.setdefault(k, "constexpr")

        # Try 3.6 form (constexprs= kwarg); fall back to 3.x form.
        import inspect as _inspect

        ast_init_params = set(
            _inspect.signature(ASTSource.__init__).parameters
        )
        if "constexprs" in ast_init_params:
            src = ASTSource(
                fn=fn, signature=signature, constexprs=kernel_mod.META_ARGS
            )
        else:
            src = ASTSource(
                fn=fn, signature=signature, constants=kernel_mod.META_ARGS
            )

        if hasattr(src, "make_ir"):
            from triton._C.libtriton import ir as _ir  # type: ignore
            from triton.backends.compiler import GPUTarget  # type: ignore
            from triton.backends import backends as _backends  # type: ignore

            gpu_target = None
            try:
                runtime = importlib.import_module("triton.runtime")
                gpu_target = runtime.driver.active.get_current_target()
            except Exception:
                gpu_target = None
            if gpu_target is None:
                gpu_target = GPUTarget(
                    backend="mps", arch="apple_m", warp_size=32
                )

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
                return None, (
                    f"no triton backend registered for {gpu_target.backend!r} "
                    f"(registry keys: {list(_backends.keys())})"
                ), _default_options
            be = backend_entry.compiler(gpu_target)
            options = be.parse_options({})
            ctx = _ir.context()
            _ir.load_dialects(ctx)
            be.load_dialects(ctx)
            codegen = be.get_codegen_implementation(options)
            module_map = be.get_module_map()
            module = src.make_ir(gpu_target, options, codegen, module_map, ctx)
            # Surface Triton's ``num_warps`` / ``num_stages`` so the
            # reducer can wrap the body in a matching ``threadIdx.x``
            # ``thread_extent`` AttrStmt (TileLang's ``gemm.lower`` derives
            # ``num_warps = block_size / warp_size`` from that extent).
            opts_dict: dict[str, Any] = {
                "num_warps": int(getattr(options, "num_warps", 4) or 4),
                "num_stages": int(getattr(options, "num_stages", 2) or 2),
            }
            return str(module), None, opts_dict

        # 3.0/3.1 fallback path.
        from triton.compiler import compile as triton_compile  # type: ignore

        compiled = triton_compile(src, options={"stage": "ttir"})
        asm = getattr(compiled, "asm", None)
        if asm is None:
            return None, f"triton compile returned no .asm: {compiled!r}", _default_options
        ttir = asm.get("ttir")
        if ttir is None:
            return (
                None,
                f"triton compile produced no 'ttir' key in asm: keys={list(asm)}",
                _default_options,
            )
        return str(ttir), None, _default_options
    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc(limit=6)
        return (
            None,
            f"triton.compile raised: {type(exc).__name__}: {exc}\n{tb}",
            _default_options,
        )


def _default_signature(fn: Any, constants: dict[str, Any]) -> dict[str, str]:
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
    sig_dict: dict[str, str] = {}
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


def _ensure_cxx_shim_on_syspath() -> None:
    """Best-effort: locate the prebuilt ``_triton_frontend_cxx`` extension.

    The shim is conventionally built under ``poc/triton_frontend/_cxx/
    build/`` but the orchestrator's CI also puts py3.13 builds under
    ``build-port-313/``. ``ensure_built`` only looks at the canonical
    ``build/``; mirror its behaviour for the sibling dirs so the harness
    finds the shim regardless of where it was placed.
    """
    import sys
    from pathlib import Path

    if importlib.util.find_spec("_triton_frontend_cxx") is not None:
        return

    # Run the canonical locator first (handles the "build/" case).
    try:
        from poc.triton_frontend.build_cxx import ensure_built  # type: ignore

        if ensure_built(build=False, verbose=False):
            return
    except Exception:
        pass

    # Sibling-dir scan: anything under ``_cxx/`` that contains a
    # CPython-suffixed shim. Prefer the suffix matching this interpreter.
    here = Path(__file__).resolve().parent.parent / "_cxx"
    if not here.exists():
        return
    py_suffix = f"cpython-{sys.version_info.major}{sys.version_info.minor}"
    candidates = sorted(here.glob(f"build*/{ '_triton_frontend_cxx' }*.so"))
    # Prefer Python-version-matched builds.
    candidates.sort(key=lambda p: (py_suffix not in p.name, str(p)))
    for cand in candidates:
        path_str = str(cand.parent)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
        importlib.invalidate_caches()
        if importlib.util.find_spec("_triton_frontend_cxx") is not None:
            return
        # Do not import the shim here. Triton's native libtriton and the
        # local C++ shim both register LLVM options process-globally; after
        # Triton has been loaded, importing the shim in-process can abort the
        # interpreter before Python can catch an exception. The lowering path
        # calls the shim through an isolated subprocess when needed.


def _lower_ttir(
    ttir_text: str,
    kernel_name: str,
    options: dict[str, Any] | None = None,
    arg_buffer_shapes: list[tuple[int, ...]] | None = None,
    grid: tuple[int, ...] | None = None,
) -> tuple[Any, str | None]:
    """Run ``poc.triton_frontend.from_ttir`` on the captured TTIR.

    Uses the lifted helpers:

    * :func:`poc.triton_frontend.pipeline.is_custom_form_ttir` +
      :func:`poc.triton_frontend.pipeline.round_trip_through_cxx_shim`
      to convert Triton's custom-form TTIR into generic op form
      (parseable by jaxlib's stripped ``mlir.ir`` bindings).
    * :func:`poc.triton_frontend.mlir_walker.wrap_module_for_walker`
      to wrap jaxlib's body-as-Block ``Module`` so the walker uses
      the ``operation`` branch which carries ``regions``.

    On hosts without jaxlib *and* without ``mlir.ir`` we fall back to
    the text walker; the verdict detail flags it.
    """
    try:
        from poc.triton_frontend import from_ttir  # type: ignore
        from poc.triton_frontend._mlir_path_setup import (  # type: ignore
            local_jaxlib_mlir_ir,
        )
        from poc.triton_frontend.mlir_walker import wrap_module_for_walker  # type: ignore
        from poc.triton_frontend.pipeline import (  # type: ignore
            is_custom_form_ttir,
            round_trip_through_cxx_shim,
        )
    except Exception as exc:  # noqa: BLE001
        return None, f"poc.triton_frontend import failed: {type(exc).__name__}: {exc}"

    # Locate the C++ shim before we try the real MLIR walker. We obtain a
    # LOCAL reference to ``jaxlib.mlir.ir`` instead of publishing the alias
    # in ``sys.modules``. Publishing the alias collides with native
    # Triton's ``triton._C`` nanobind extension on the very next ``make_ir``
    # call and aborts the interpreter mid-suite -- see
    # ``_mlir_path_setup.local_jaxlib_mlir_ir`` docstring.
    _ensure_cxx_shim_on_syspath()
    _mlir_ir = local_jaxlib_mlir_ir()
    if _mlir_ir is None:
        try:
            from mlir import ir as _mlir_ir  # type: ignore
        except Exception:
            _mlir_ir = None  # type: ignore[assignment]

    # Try the real MLIR path first. Convert via the C++ shim's
    # ``to_generic()`` when input is custom-form so a vanilla ``mlir.ir``
    # (without ``tt`` dialect) can still parse the result.
    if _mlir_ir is None:
        raise ImportError("no mlir.ir provider available")
    try:
        ctx = _mlir_ir.Context()
        ctx.allow_unregistered_dialects = True

        parse_text = ttir_text
        if is_custom_form_ttir(ttir_text):
            parse_text = round_trip_through_cxx_shim(ttir_text)

        with ctx, _mlir_ir.Location.unknown(ctx):
            module = _mlir_ir.Module.parse(parse_text, ctx)

        adapter = wrap_module_for_walker(module)
        # Plumb Triton's ``num_warps`` / ``num_stages`` through so
        # ``_make_prim_func`` can wrap the body in a matching
        # ``threadIdx.x`` ``thread_extent`` AttrStmt; without this
        # ``gemm.lower`` defaults block_size to 1 and ``num_warps`` collapses
        # to 0, tripping the ``m_warp * n_warp == num_warps`` ICHECK.
        opts = options or {}
        prim = from_ttir(
            adapter,
            name=kernel_name,
            grid=grid,
            arg_buffer_shapes=arg_buffer_shapes,
            num_warps=opts.get("num_warps"),
            num_stages=opts.get("num_stages"),
        )
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
        opts = options or {}
        prim = from_ttir(
            ttir_text,
            name=kernel_name,
            grid=grid,
            arg_buffer_shapes=arg_buffer_shapes,
            num_warps=opts.get("num_warps"),
            num_stages=opts.get("num_stages"),
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


def _compile_metal(prim: Any) -> tuple[Any, str | None]:
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


def _positive_int_or_none(value: Any) -> int | None:
    """Best-effort conversion for TVM IntImm / Python ints."""
    if value is None:
        return None
    raw = getattr(value, "value", value)
    try:
        out = int(raw)
    except Exception:
        s = str(value)
        if not s.isdigit():
            return None
        out = int(s)
    return out if out > 0 else None


def _threadgroup_from_artifact(artifact: Any, kernel_mod: Any) -> tuple[int, int, int]:
    """Infer the Metal threadgroup shape TileLang encoded on the device func."""
    explicit = getattr(kernel_mod, "THREADGROUP", None)
    if explicit is not None:
        dims = tuple(int(x) for x in explicit)
        while len(dims) < 3:
            dims = dims + (1,)
        return dims[:3]

    dims = [1, 1, 1]
    device_mod = getattr(artifact, "device_mod", None)
    funcs = getattr(device_mod, "functions", {}) if device_mod is not None else {}
    for _gv, func in getattr(funcs, "items", lambda: [])():
        attrs = getattr(func, "attrs", None)
        if attrs is None:
            continue
        try:
            thread_extent = attrs.get("thread_extent")
        except Exception:
            thread_extent = None
        if not thread_extent:
            continue
        for axis, key in enumerate(("threadIdx.x", "threadIdx.y", "threadIdx.z")):
            try:
                raw = thread_extent.get(key)
            except Exception:
                raw = None
                for k, v in getattr(thread_extent, "items", lambda: [])():
                    if str(k) == key:
                        raw = v
                        break
            val = _positive_int_or_none(raw)
            if val is not None:
                dims[axis] = max(dims[axis], val)
    return tuple(dims)


# ---------------------------------------------------------------------------
# Stage 4 + 5: MLX wrap + run
# ---------------------------------------------------------------------------


def _run_mlx(
    artifact: Any,
    kernel_args: list[np.ndarray],
    kernel_mod: Any,
) -> tuple[np.ndarray | None, str | None]:
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

    # Build the args-struct inline mapping. TileLang's Metal emitter packs
    # scalar runtime args into a ``<kernel>_args_t`` struct keyed by
    # PrimFunc param name (``arg<N>`` for the Nth original PrimFunc param,
    # plus auto-injected ``gridDim_<i>`` entries). ``mx.fast.metal_kernel``
    # rebuilds the kernel signature itself and does NOT carry that struct
    # parameter through, so each ``arg.<field>[0]`` access in the body
    # must be substituted with a literal int. Values are derived from the
    # kernel module's runtime metadata:
    #
    # * ``gridDim_<i>`` <- ``kernel_mod.LAUNCH_GRID[i]`` (default 1).
    # * ``arg<N>`` for N >= (input_count + output_count) <- the matching
    #   Triton scalar arg from the kernel module. By convention, scalar
    #   args follow the buffer args in PrimFunc declaration order; the
    #   only such scalar in vector_add is ``n_elements`` whose value is
    #   the length of the first input array.
    args_struct_inline: dict[str, int] = {}
    launch_grid = tuple(int(g) for g in getattr(kernel_mod, "LAUNCH_GRID", (1,)))
    for i in range(3):
        args_struct_inline[f"gridDim_{i}"] = (
            launch_grid[i] if i < len(launch_grid) else 1
        )
    # Best-effort scalar-arg inference. The harness only knows the inputs
    # and the kernel module; for the conformance ladder this is enough
    # because the only scalar arg is ``n_elements`` (= input length). A
    # kernel module may override this by exposing ``KERNEL_SCALAR_ARGS``
    # as ``Dict[str, int]`` with explicit (PrimFunc-name -> value) pairs.
    explicit_scalars = getattr(kernel_mod, "KERNEL_SCALAR_ARGS", None)
    if isinstance(explicit_scalars, dict):
        for k, v in explicit_scalars.items():
            args_struct_inline[str(k)] = int(v)
    else:
        # Fallback: assume PrimFunc names buffer params arg0..arg{B-1}
        # and the next scalar param ``arg<B>`` is ``n_elements`` (the
        # length of the first input). This holds for vector_add and any
        # 1D elementwise kernel; richer kernels need KERNEL_SCALAR_ARGS.
        buffer_count = len(inputs_np) + 1  # +1 for the single output
        if inputs_np:
            args_struct_inline[f"arg{buffer_count}"] = int(inputs_np[0].size)

    try:
        adapter = wrap_tilelang_metal_kernel(
            artifact,
            input_count=len(inputs_np),
            output_count=1,
            name="triton_e2e_kernel",
            args_struct_inline=args_struct_inline,
            allow_mx_fast_metal_kernel=True,
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
        threadgroup = _threadgroup_from_artifact(artifact, kernel_mod)
        # MLX's ``grid`` is a total thread grid, while TileLang's
        # ``LAUNCH_GRID`` metadata is a block/threadgroup grid.
        dispatch_grid = tuple(
            max(1, int(grid[i]) * int(threadgroup[i])) for i in range(3)
        )
        out_arrays = adapter(
            inputs=mx_inputs,
            output_shapes=[output_np.shape],
            output_dtypes=[_np_to_mx_dtype(mx, output_np.dtype)],
            grid=dispatch_grid,
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
) -> tuple[bool, float, float, list[tuple[int, ...]]]:
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


def run_one(kernel_module_name: str, deps: dict[str, str | None]) -> KernelResult:
    """Run a single kernel module end-to-end and return a KernelResult."""
    started = time.monotonic()

    # If any pipeline component is unavailable we SKIP -- this is the
    # honest verdict (we can't even attempt the pipeline).
    for comp in ("triton", "tvm", "tilelang", "mlx", "cppmega_mlx"):
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
    ttir_text, ttir_err, ttir_options = _capture_ttir(kernel_mod)
    if ttir_text is None:
        return KernelResult(
            name=kernel_module_name,
            verdict=Verdict.TTIR_FAIL,
            detail=ttir_err or "<unknown>",
            elapsed_s=time.monotonic() - started,
        )

    args, expected = kernel_mod.make_inputs()
    # TTIR pointer arithmetic is flat addptr indexing.  Keep the PrimFunc ABI
    # buffers rank-1 but size them to the full underlying storage so TileLang
    # does not infer a tile-only extent such as 32x32 -> (1024,).
    arg_buffer_shapes = [(int(arr.size),) for arr in args]
    launch_grid = tuple(int(g) for g in getattr(kernel_mod, "LAUNCH_GRID", (1,)))

    # Stage 2: PrimFunc
    prim, lower_err = _lower_ttir(
        ttir_text,
        kernel_module_name,
        ttir_options,
        arg_buffer_shapes=arg_buffer_shapes,
        grid=launch_grid,
    )
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


def run_all(
    report_path: Path | None = None,
    kernels: list[str] | None = None,
) -> list[KernelResult]:
    """Run kernel modules and write a markdown report.

    ``kernels`` -- if provided, restrict the run to these kernel module
    names (must be members of :data:`numeric_kernels.KERNEL_MODULES`).
    Defaults to every registered kernel.
    """
    deps = _probe_deps()
    if kernels is None:
        kernels = list(numeric_kernels.KERNEL_MODULES)
    else:
        unknown = [k for k in kernels if k not in numeric_kernels.KERNEL_MODULES]
        if unknown:
            raise SystemExit(
                f"unknown kernel(s): {unknown}; "
                f"available: {list(numeric_kernels.KERNEL_MODULES)}"
            )
    results: list[KernelResult] = []
    for mod_name in kernels:
        results.append(run_one(mod_name, deps))

    if report_path is None:
        report_path = REPORT_PATH
    _write_report(results, deps, report_path)
    return results


def _build_arg_parser():
    """Build the CLI argparse parser for ``python -m ... numeric_smoke``."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="poc.triton_frontend._test_harness.numeric_smoke",
        description=(
            "Run the Triton -> TileLang -> Metal -> MLX numeric smoke "
            "harness and write a markdown report."
        ),
    )
    parser.add_argument(
        "--kernel",
        action="append",
        default=None,
        choices=list(numeric_kernels.KERNEL_MODULES),
        help=(
            "Restrict the run to the named kernel module. May be passed "
            "multiple times to run a subset; default is every kernel."
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help=f"Where to write the markdown report (default: {REPORT_PATH}).",
    )
    return parser


def _write_report(
    results: list[KernelResult],
    deps: dict[str, str | None],
    report_path: Path,
) -> None:
    """Write a markdown summary at ``report_path``."""
    counts: dict[str, int] = {}
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
    _args = _build_arg_parser().parse_args()
    res = run_all(report_path=_args.report, kernels=_args.kernel)
    for r in res:
        print(r.short())
    print(f"\nReport: {_args.report or REPORT_PATH}")
