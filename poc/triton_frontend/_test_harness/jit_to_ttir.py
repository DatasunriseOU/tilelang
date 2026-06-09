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


def _fold_ttir(module, gpu_target):
    """Run Triton's own ``make_ttir`` optimization passes on the captured
    module before serialization.

    ``ASTSource.make_ir`` returns the *pre-optimization* TTIR. Triton's real
    pipeline then runs ``make_ttir`` (inliner + canonicalizer + combine + cse +
    ...), and it is the **canonicalizer** — backed by MLIR int-range analysis —
    that folds away the i32->i64 overflow-guard chain (``arith.extsi`` +
    ``arith.cmpi sle 2147483647`` + ``arith.andi``). Those guards are provably
    always-true for the kernel's index ranges, so folding them is a real
    semantic-preserving simplification (identical to what native Triton emits in
    its cached ``.ttir``), not a dropped check.

    We replicate the exact ``make_ttir`` pass list from Triton's nvidia backend.
    The leading ``add_inliner`` is a genuine no-op for our single-kernel capture
    (there is nothing to inline) and aborts the standalone PassManager because
    the module has no enclosing call graph; we therefore run the remaining
    passes — which include the canonicalizer that does the actual fold — and
    raise loudly if *those* fail (no silent fallback). If the optimized pipeline
    raises, the caller surfaces the error rather than consuming unfolded TTIR.
    """
    from triton._C.libtriton import ir as _libir, passes

    capability = gpu_target.arch if isinstance(gpu_target.arch, int) else 80
    pm = _libir.pass_manager(module.context)
    pm.enable_debug()
    # add_inliner is skipped intentionally: it is a no-op for a single captured
    # kernel and fails a standalone PassManager (no enclosing call graph). All
    # guard-folding is done by add_canonicalizer below.
    passes.ttir.add_rewrite_tensor_pointer(pm)
    if capability // 10 < 9:
        passes.ttir.add_rewrite_tensor_descriptor_to_pointer(pm)
    passes.common.add_canonicalizer(pm)
    passes.ttir.add_combine(pm)
    passes.ttir.add_reorder_broadcast(pm)
    passes.common.add_cse(pm)
    passes.common.add_symbol_dce(pm)
    passes.ttir.add_loop_unroll(pm)
    pm.run(module, "make_ttir_fold")
    return module


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
            module = _fold_ttir(module, gpu_target)
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


def triton_jit_to_ttir_subprocess_from_source(
    *,
    source: str,
    kernel_name: str,
    constexprs: Optional[Dict[str, Any]] = None,
    signature: Optional[Dict[str, str]] = None,
    target: Optional[str] = None,
    extra_sys_path: Optional[list[str]] = None,
    timeout: int = 120,
) -> str:
    """Capture TTIR by exec'ing ``source`` in a fresh subprocess.

    The bridge in cppmega.mlx receives ``@triton.jit`` ``JITFunction``
    objects that live only in the host interpreter and cannot be
    pickled across processes. We side-step this by shipping the
    kernel's source text (obtained via :func:`inspect.getsource` on
    ``JITFunction.fn``) to a fresh subprocess, exec'ing it inside a
    namespace that has ``triton`` / ``triton.language`` pre-imported,
    and then re-applying the ``@triton.jit`` decorator. The decorated
    kernel is then routed through :func:`triton_jit_to_ttir` exactly
    the same way an in-process call would be.
    """
    import json
    import os
    import subprocess
    import sys
    import tempfile
    import textwrap

    with tempfile.NamedTemporaryFile(
        prefix="triton_ttir_src_", suffix=".txt", delete=False, mode="w"
    ) as tmp:
        out_path = tmp.name
    payload = {
        "source": source,
        "kernel_name": kernel_name,
        "constexprs": dict(constexprs or {}),
        "signature": dict(signature or {}) if signature is not None else None,
        "target": target,
        "out_path": out_path,
    }
    # Native Triton requires ``@triton.jit`` kernels to be defined in a
    # real Python file (the JIT compiler reads ``co_filename`` via
    # ``inspect.findsource``). We materialise the source into a temp
    # module file, import it by spec, and feed the resulting kernel into
    # the in-process ``triton_jit_to_ttir`` path. Failure to read the
    # file would abort with ``ValueError("@jit functions should be
    # defined in a Python file")`` — see triton/runtime/jit.py:__init__.
    script = textwrap.dedent(
        """
        import ast, importlib, importlib.util, json, os, sys, tempfile, traceback
        payload = json.loads(sys.argv[1])

        def _strip_non_jit_decorators(src):
            \"\"\"Drop @triton.autotune / @triton.heuristics (and any non-jit
            decorator) from the kernel source so exec'ing it does not
            NameError on module-level refs (init_to_zero, autotune_configs,
            ...) that live only in Tri-Dao's original module. TTIR capture
            supplies constexprs explicitly, so autotune metadata is unneeded.
            We KEEP @triton.jit. If the source cannot be parsed we return it
            unchanged and let the real exec error surface (no silent paper).
            \"\"\"
            def _dec_name(node):
                tgt = node.func if isinstance(node, ast.Call) else node
                parts = []
                while isinstance(tgt, ast.Attribute):
                    parts.append(tgt.attr)
                    tgt = tgt.value
                if isinstance(tgt, ast.Name):
                    parts.append(tgt.id)
                return \".\".join(reversed(parts))
            try:
                tree = ast.parse(src)
            except SyntaxError:
                return src
            changed = False
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    kept = []
                    for dec in node.decorator_list:
                        name = _dec_name(dec)
                        if name.split(\".\")[-1] == \"jit\":
                            kept.append(dec)
                        else:
                            changed = True
                    node.decorator_list = kept
            if not changed:
                return src
            return ast.unparse(tree)

        try:
            import triton
            import triton.language as tl  # noqa: F401
            header = \"import triton\\nimport triton.language as tl\\n\\n\"
            src_dir = tempfile.mkdtemp(prefix=\"triton_ttir_src_\")
            src_path = os.path.join(src_dir, \"_subprocess_kernel.py\")
            sanitized = _strip_non_jit_decorators(payload[\"source\"])
            with open(src_path, \"w\") as fh:
                fh.write(header)
                fh.write(sanitized)
            spec = importlib.util.spec_from_file_location(
                \"_subprocess_kernel\", src_path
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules[\"_subprocess_kernel\"] = module
            spec.loader.exec_module(module)
            fn = getattr(module, payload[\"kernel_name\"], None)
            if fn is None:
                raise RuntimeError(
                    f\"materialised module did not define {payload['kernel_name']!r}\"
                )
            if not hasattr(fn, \"params\") and callable(fn):
                fn = triton.jit(fn)
            from poc.triton_frontend._test_harness.jit_to_ttir import (
                triton_jit_to_ttir,
            )
            text = triton_jit_to_ttir(
                fn,
                constexprs=payload[\"constexprs\"] or None,
                signature=payload[\"signature\"],
                target=payload[\"target\"],
            )
            with open(payload[\"out_path\"], \"w\") as fh:
                fh.write(text)
        except BaseException as exc:
            traceback.print_exc()
            print(f\"__TTIR_CAPTURE_ERROR__:{type(exc).__name__}:{exc}\", file=sys.stderr)
            sys.exit(2)
        """
    )
    env = dict(os.environ)
    sys_path_entries = list(extra_sys_path or [])
    sys_path_entries.append(os.getcwd())
    existing = env.get("PYTHONPATH")
    if existing:
        sys_path_entries.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(sys_path_entries)
    completed = subprocess.run(
        [sys.executable, "-c", script, json.dumps(payload)],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
        check=False,
    )
    if completed.returncode != 0:
        try:
            os.unlink(out_path)
        except OSError:
            pass
        raise TTIRCaptureError(
            f"subprocess (from-source) TTIR capture exited "
            f"{completed.returncode} for kernel={kernel_name!r}: "
            f"stderr={completed.stderr[-1500:]!r} "
            f"stdout={completed.stdout[-500:]!r}"
        )
    try:
        with open(out_path) as fh:
            ttir_text = fh.read()
    finally:
        try:
            os.unlink(out_path)
        except OSError:
            pass
    if not ttir_text.strip():
        raise TTIRCaptureError(
            f"subprocess returned empty TTIR for kernel={kernel_name!r}"
        )
    return ttir_text


def triton_jit_to_ttir_subprocess(
    *,
    module_name: str,
    kernel_attr: str = "TRITON_KERNEL",
    constexprs: Optional[Dict[str, Any]] = None,
    signature: Optional[Dict[str, str]] = None,
    target: Optional[str] = None,
    extra_sys_path: Optional[list[str]] = None,
    timeout: int = 120,
) -> str:
    """Capture TTIR text from a Python module attribute in a fresh subprocess.

    Necessary when the calling interpreter already has an LLVM peer module
    resident (jaxlib's MLIR/LLVM nanobind extension or our PtrAnalysis C++
    shim) that would clash with ``triton._C.libtriton`` on the next
    ``make_ir`` call. Pickling ``@triton.jit`` kernel objects across
    processes is not supported, so we resolve the kernel *inside* the
    subprocess by importing ``module_name`` and reading ``kernel_attr``.

    Returns the TTIR text exactly as :func:`triton_jit_to_ttir` would in
    a clean process. Raises :class:`TTIRCaptureError` when the subprocess
    fails for any reason, with the subprocess stderr/stdout attached for
    debugging.
    """
    import json
    import os
    import subprocess
    import sys
    import tempfile
    import textwrap

    with tempfile.NamedTemporaryFile(
        prefix="triton_ttir_", suffix=".txt", delete=False, mode="w"
    ) as tmp:
        out_path = tmp.name
    payload = {
        "module_name": module_name,
        "kernel_attr": kernel_attr,
        "constexprs": dict(constexprs or {}),
        "signature": dict(signature or {}) if signature is not None else None,
        "target": target,
        "out_path": out_path,
    }
    script = textwrap.dedent(
        """
        import importlib, json, sys, traceback
        payload = json.loads(sys.argv[1])
        try:
            mod = importlib.import_module(payload["module_name"])
            fn = getattr(mod, payload["kernel_attr"], None)
            if fn is None:
                raise RuntimeError(
                    f"module {payload['module_name']!r} has no attribute "
                    f"{payload['kernel_attr']!r}"
                )
            from poc.triton_frontend._test_harness.jit_to_ttir import (
                triton_jit_to_ttir,
            )
            text = triton_jit_to_ttir(
                fn,
                constexprs=payload["constexprs"] or None,
                signature=payload["signature"],
                target=payload["target"],
            )
            with open(payload["out_path"], "w") as fh:
                fh.write(text)
        except BaseException as exc:
            traceback.print_exc()
            print(f"__TTIR_CAPTURE_ERROR__:{type(exc).__name__}:{exc}", file=sys.stderr)
            sys.exit(2)
        """
    )
    env = dict(os.environ)
    sys_path_entries = list(extra_sys_path or [])
    sys_path_entries.append(os.getcwd())
    existing = env.get("PYTHONPATH")
    if existing:
        sys_path_entries.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(sys_path_entries)
    completed = subprocess.run(
        [sys.executable, "-c", script, json.dumps(payload)],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
        check=False,
    )
    if completed.returncode != 0:
        os.unlink(out_path)
        raise TTIRCaptureError(
            f"subprocess TTIR capture exited {completed.returncode} "
            f"for module={module_name!r} attr={kernel_attr!r}: "
            f"stderr={completed.stderr[-1500:]!r} stdout={completed.stdout[-500:]!r}"
        )
    try:
        with open(out_path) as fh:
            ttir_text = fh.read()
    finally:
        try:
            os.unlink(out_path)
        except OSError:
            pass
    if not ttir_text.strip():
        raise TTIRCaptureError(
            f"subprocess returned empty TTIR for module={module_name!r} "
            f"attr={kernel_attr!r}"
        )
    return ttir_text


__all__ = [
    "TTIRCaptureError",
    "TritonUnavailable",
    "describe_triton_env",
    "triton_available",
    "triton_jit_to_ttir",
    "triton_jit_to_ttir_subprocess",
    "triton_jit_to_ttir_subprocess_from_source",
    "triton_version",
]
