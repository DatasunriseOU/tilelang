"""Static AST -> TileLang DSL lowering for the CuTeDSL frontend.

The lowering is intentionally narrow: it accepts a Python source string
containing one ``@cute.kernel`` (or aliased) decorated function and
emits an equivalent TileLang Python DSL function string. That source is
then exec'd in a fresh namespace with ``tilelang.language as T`` bound so
``@T.prim_func`` builds the canonical ``tvm.tir.PrimFunc``.

This keeps us inside the TileLang TIR pivot (RFC §2 decision) and lets
the produced PrimFunc participate in fusion the same way any native
``T.prim_func`` would. No runtime ``cutlass.cute`` dependency is needed,
which is critical for the local Mac dev box (no CuTeDSL installed) and
for the non-NV backends (Metal / HIP) where CuTeDSL is irrelevant.
"""

from __future__ import annotations

import ast
import textwrap
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from tvm import tir


class CuTeDSLLoweringError(RuntimeError):
    """Raised when the static CuTe -> TileLang lowering cannot proceed."""


@dataclass(frozen=True)
class CuTeKernelSignature:
    """Describes a parameter of the CuTe kernel for the TileLang frontend.

    The CuTe DSL itself encodes parameter shapes/dtypes via cute.Tensor
    type hints, but our static frontend prefers an explicit caller-side
    signature so the lowering is robust to surface differences between
    CuTeDSL releases.
    """

    name: str
    shape: tuple[int, ...]
    dtype: str
    scope: str = "global"  # one of "global", "shared", "fragment", "register"


@dataclass
class _LoweringCtx:
    indent: int = 1
    lines: list[str] = field(default_factory=list)
    used_names: set[str] = field(default_factory=set)

    def emit(self, line: str) -> None:
        prefix = "    " * self.indent
        self.lines.append(prefix + line)


def _detect_cute_aliases(tree: ast.Module) -> tuple[set[str], set[str]]:
    """Return (cute module aliases, cute.kernel decorator aliases)."""

    module_aliases: set[str] = set()
    kernel_aliases: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "cutlass.cute":
                    module_aliases.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module == "cutlass":
                for alias in node.names:
                    if alias.name == "cute":
                        module_aliases.add(alias.asname or alias.name)
            elif node.module == "cutlass.cute":
                for alias in node.names:
                    if alias.name == "kernel":
                        kernel_aliases.add(alias.asname or alias.name)
    return module_aliases, kernel_aliases


def _is_cute_decorator(
    node: ast.AST,
    module_aliases: set[str],
    kernel_aliases: set[str],
) -> bool:
    if isinstance(node, ast.Call):
        node = node.func
    if isinstance(node, ast.Name) and node.id in kernel_aliases:
        return True
    if isinstance(node, ast.Attribute) and node.attr == "kernel":
        # cute.kernel / cutlass.cute.kernel
        value = node.value
        path: list[str] = []
        while isinstance(value, ast.Attribute):
            path.insert(0, value.attr)
            value = value.value
        if isinstance(value, ast.Name):
            path.insert(0, value.id)
        return ".".join(path) in module_aliases
    return False


def _is_cute_call(call: ast.AST, attr: str, module_aliases: set[str]) -> bool:
    if not isinstance(call, ast.Call):
        return False
    func = call.func
    if isinstance(func, ast.Attribute) and func.attr == attr:
        value = func.value
        path: list[str] = []
        while isinstance(value, ast.Attribute):
            path.insert(0, value.attr)
            value = value.value
        if isinstance(value, ast.Name):
            path.insert(0, value.id)
        return ".".join(path) in module_aliases
    return False


def _arg_as_value(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Tuple):
        return tuple(_arg_as_value(e) for e in node.elts)
    if isinstance(node, ast.List):
        return [_arg_as_value(e) for e in node.elts]
    if isinstance(node, ast.Name):
        return node.id
    raise CuTeDSLLoweringError(
        f"unsupported CuTe argument expression: {ast.dump(node)}"
    )


def _kw_dict(call: ast.Call) -> dict[str, Any]:
    return {kw.arg: _arg_as_value(kw.value) for kw in call.keywords if kw.arg}


def _shape_literal(shape: Sequence[int]) -> str:
    """Return a Python tuple literal for ``shape``.

    ``(16)`` is an integer expression, not a tuple. TVMScript shape
    annotations and TileLang allocation calls need the single-dimensional
    case to be rendered as ``(16,)``.
    """

    dims = tuple(int(x) for x in shape)
    if len(dims) == 1:
        return f"({dims[0]},)"
    return "(" + ", ".join(str(x) for x in dims) + ")"


def _lower_stmt(
    stmt: ast.AST,
    ctx: _LoweringCtx,
    module_aliases: set[str],
) -> None:
    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
        call = stmt.value
        if _is_cute_call(call, "copy", module_aliases):
            src, dst = (_arg_as_value(a) for a in call.args[:2])
            ctx.emit(f"T.copy({src}, {dst})")
            return
        if _is_cute_call(call, "gemm", module_aliases):
            a_buf, b_buf, c_buf = (_arg_as_value(a) for a in call.args[:3])
            ctx.emit(f"T.gemm({a_buf}, {b_buf}, {c_buf})")
            return
        if _is_cute_call(call, "fill", module_aliases):
            buf = _arg_as_value(call.args[0])
            value = _arg_as_value(call.args[1])
            ctx.emit(f"T.fill({buf}, {value!r})")
            return
        raise CuTeDSLLoweringError(
            f"unsupported CuTe call statement at top level: {ast.dump(call)}"
        )

    if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
        target = stmt.targets[0]
        if not isinstance(target, ast.Name):
            raise CuTeDSLLoweringError(
                f"only simple name targets are supported, got {ast.dump(target)}"
            )
        name = target.id
        value = stmt.value
        if isinstance(value, ast.Call) and _is_cute_call(value, "make_tensor", module_aliases):
            kwargs = _kw_dict(value)
            shape = kwargs.get("shape") or _arg_as_value(value.args[1])
            dtype = kwargs.get("dtype") or _arg_as_value(value.args[2])
            scope = kwargs.get("scope", "fragment")
            if not isinstance(shape, tuple):
                raise CuTeDSLLoweringError(
                    f"cute.make_tensor 'shape' must be a tuple of ints, got {shape!r}"
                )
            shape_text = _shape_literal(shape)
            if scope == "shared":
                ctx.emit(f"{name} = T.alloc_shared({shape_text}, \"{dtype}\")")
            elif scope in ("fragment", "register"):
                ctx.emit(f"{name} = T.alloc_fragment({shape_text}, \"{dtype}\")")
            else:
                raise CuTeDSLLoweringError(
                    f"unsupported cute.make_tensor scope {scope!r}"
                )
            ctx.used_names.add(name)
            return
        if isinstance(value, ast.Call) and _is_cute_call(value, "arange", module_aliases):
            # Used only as an iter variable; emit a comment so the
            # source is still readable.
            args = [_arg_as_value(a) for a in value.args]
            if len(args) == 1:
                ctx.emit(f"# {name} = cute.arange({args[0]}) -- mapped to T.serial below")
            elif len(args) == 2:
                ctx.emit(f"# {name} = cute.arange({args[0]}, {args[1]}) -- mapped to T.serial below")
            else:
                raise CuTeDSLLoweringError("cute.arange takes 1 or 2 args")
            ctx.used_names.add(name)
            return

    if isinstance(stmt, ast.For):
        iter_call = stmt.iter
        if not (
            isinstance(iter_call, ast.Call)
            and _is_cute_call(iter_call, "arange", module_aliases)
        ):
            raise CuTeDSLLoweringError(
                f"unsupported loop iterator: {ast.dump(iter_call)}"
            )
        args = [_arg_as_value(a) for a in iter_call.args]
        if len(args) == 1:
            start, stop = 0, args[0]
        elif len(args) == 2:
            start, stop = args
        else:
            raise CuTeDSLLoweringError("cute.arange takes 1 or 2 args")
        iv = stmt.target.id if isinstance(stmt.target, ast.Name) else "iv"
        ctx.emit(f"for {iv} in T.serial({stop} - {start}):")
        ctx.indent += 1
        for sub in stmt.body:
            _lower_stmt(sub, ctx, module_aliases)
        ctx.indent -= 1
        return

    if isinstance(stmt, ast.Pass) or (
        isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant)
    ):
        # Docstrings and pass: pass through as a comment.
        return

    raise CuTeDSLLoweringError(
        f"unsupported CuTe statement at this position: {ast.dump(stmt)}"
    )


def _emit_tilelang_source(
    func_name: str,
    signature: Sequence[CuTeKernelSignature],
    body: Sequence[ast.AST],
    module_aliases: set[str],
    grid: Sequence[int] | None,
) -> str:
    ctx = _LoweringCtx(indent=2)
    if grid is None:
        ctx.emit("with T.Kernel(1, threads=128):")
    else:
        grid_args = ", ".join(str(int(g)) for g in grid)
        ctx.emit(f"with T.Kernel({grid_args}, threads=128):")
    ctx.indent += 1
    for stmt in body:
        _lower_stmt(stmt, ctx, module_aliases)
    ctx.indent -= 1
    if not ctx.lines:
        ctx.emit("pass")

    param_decls = ", ".join(
        (
            f"{p.name}: T.Tensor("
            f"{_shape_literal(p.shape)}, \"{p.dtype}\")"
        )
        for p in signature
    )
    body_text = "\n".join(ctx.lines)
    header = (
        "import tilelang\n"
        "import tilelang.language as T\n\n\n"
        "@T.prim_func\n"
        f"def {func_name}({param_decls}):\n"
    )
    return header + body_text + "\n"


def from_cute_source(
    source: str,
    *,
    signature: Sequence[CuTeKernelSignature],
    func_name: str | None = None,
    grid: Sequence[int] | None = None,
    emit_only: bool = False,
) -> tir.PrimFunc | str:
    """Lower a CuTeDSL Python source string into a TileLang ``tir.PrimFunc``.

    Parameters
    ----------
    source:
        Full Python source containing a single ``@cute.kernel`` (or
        aliased) function definition.
    signature:
        Explicit parameter signatures; the static lowering uses them
        verbatim for the emitted ``T.Tensor`` annotations.
    func_name:
        Optional override; defaults to the original CuTe function name.
    grid:
        Optional kernel grid; defaults to a single CTA. The lowering
        emits ``with T.Kernel(*grid, threads=128):`` so caller can choose
        the launch shape.
    emit_only:
        If True, return the emitted TileLang DSL source string instead of
        the compiled PrimFunc. Useful for tests and debugging.
    """

    tree = ast.parse(textwrap.dedent(source))
    module_aliases, kernel_aliases = _detect_cute_aliases(tree)
    if not module_aliases and not kernel_aliases:
        raise CuTeDSLLoweringError(
            "no recognised CuTe imports (cutlass.cute or cutlass.cute.kernel); "
            "lowering refuses to guess"
        )

    cute_funcs = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and any(
            _is_cute_decorator(dec, module_aliases, kernel_aliases)
            for dec in node.decorator_list
        )
    ]
    if not cute_funcs:
        raise CuTeDSLLoweringError(
            "no @cute.kernel (or aliased) function defined in source"
        )
    if len(cute_funcs) != 1:
        raise CuTeDSLLoweringError(
            f"expected exactly one cute.kernel function, found {len(cute_funcs)}: "
            + ", ".join(f.name for f in cute_funcs)
        )
    cute_func = cute_funcs[0]
    chosen_name = func_name or cute_func.name

    emitted = _emit_tilelang_source(
        func_name=chosen_name,
        signature=signature,
        body=cute_func.body,
        module_aliases=module_aliases,
        grid=grid,
    )
    if emit_only:
        return emitted

    # ``@T.prim_func`` (TVMScript) requires the function body to be
    # retrievable via ``inspect.getsource``. ``exec`` of an inline string
    # leaves no source mapping and triggers ``OSError: could not get source
    # code`` inside TVMScript's parser. Materialise the emitted DSL on
    # disk into a temp module and import it; the import side-effect then
    # runs ``@T.prim_func`` with a real source location.
    import importlib.util
    import os
    import sys
    import tempfile
    import uuid

    module_id = f"tilelang_cutedsl_emitted_{uuid.uuid4().hex}"
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        prefix=f"{module_id}_",
        delete=False,
        encoding="utf-8",
    ) as tmp:
        tmp.write(emitted)
        emitted_path = tmp.name
    spec = importlib.util.spec_from_file_location(module_id, emitted_path)
    if spec is None or spec.loader is None:
        os.unlink(emitted_path)
        raise CuTeDSLLoweringError(
            f"failed to create importlib spec for emitted DSL at {emitted_path!r}"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_id] = module
    try:
        try:
            spec.loader.exec_module(module)
        except Exception as exc:
            raise CuTeDSLLoweringError(
                f"emitted TileLang DSL failed to import: {type(exc).__name__}: {exc}\n"
                f"---emitted source ({emitted_path})---\n{emitted}"
            ) from exc
        prim_func = getattr(module, chosen_name, None)
        if not isinstance(prim_func, tir.PrimFunc):
            raise CuTeDSLLoweringError(
                f"emitted TileLang source did not produce a PrimFunc; got {type(prim_func).__name__}"
            )
        return prim_func
    finally:
        sys.modules.pop(module_id, None)
        try:
            os.unlink(emitted_path)
        except FileNotFoundError:
            pass


def compile_cute_source(
    source: str,
    *,
    signature: Sequence[CuTeKernelSignature],
    func_name: str | None = None,
    grid: Sequence[int] | None = None,
    target: Any | None = None,
    execution_backend: str | None = None,
    out_idx: Any | None = None,
) -> Any:
    """End-to-end: CuTe source -> TileLang ``PrimFunc`` -> ``JITKernel``."""

    import tilelang

    prim_func = from_cute_source(
        source,
        signature=signature,
        func_name=func_name,
        grid=grid,
    )
    kwargs: dict[str, Any] = {}
    if target is not None:
        kwargs["target"] = target
    if execution_backend is not None:
        kwargs["execution_backend"] = execution_backend
    if out_idx is not None:
        kwargs["out_idx"] = out_idx
    return tilelang.compile(prim_func, **kwargs)
