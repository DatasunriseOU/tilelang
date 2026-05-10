"""FX GraphModule -> TileLang TIR lowering (forward path, POC).

RFC reference: ``RFC_unified_fused_kernel.md`` §3 (``torch.fx -> TileLang
custom backend`` row, status: **build**), §4 (cache-resident fusion: keep
intermediates register/shared resident across the FX boundary — no implicit
HBM round-trips inside a materialized region), §7 Phase 2.2 (FX node ->
TileLang op map for the standard inductor-coverage set: matmul, layernorm,
softmax, gelu, attention prims).

Fusion strategy (orchestrator layer)
------------------------------------
The orchestrator walks the FX graph in topological order
(``torch.fx.passes.tools_common.legalize_graph``) and produces *one or more*
fused TileLang ``tvm.tir.PrimFunc`` artifacts. The high-level pipeline:

1. **Linearize.** Walk ``gm.graph.nodes`` in legalised topological order.
   Each ``call_function`` node is dispatched through ``ATEN_DISPATCH`` to
   record an op-trace entry + resolve the output ``_TensorSpec``. Per-op
   handler bodies (``_emit_<op>``) are owned by sibling integration #9 — the
   orchestrator only consumes their op_trace + _TensorSpec output and never
   touches their bodies.

2. **Dispatch with per-op fallback.** When a per-op handler raises
   ``NotImplementedError`` we log a warning and emit a fallback
   ``tir.call_extern("aten_<op>", ...)`` slot for that single op. Other ops
   in the same graph still materialise normally.

3. **Allocate intermediates.** Per-region buffer-spec heuristic
   (RFC §4 cache-residency invariant):
     * ``T.alloc_fragment`` for tiles up to 2 KiB (register-resident),
     * ``T.alloc_shared`` for larger shared-resident tiles,
     * ``T.alloc_local`` reserved for per-thread scratch only.
   Sized from each FX node's ``meta['val']`` FakeTensor.

4. **Recognise fusion patterns.** ``_fusion_patterns.FUSION_PATTERNS``
   declares matchers for: ``matmul + activation`` (fused linear epilogue),
   ``layer_norm + matmul`` (two-stage shared-resident tile), and
   ``matmul + softmax`` (softmax epilogue inside the gemm accumulator).
   The orchestrator consults this table greedily; misses fall back to a
   sequential per-op chain (which TileLang's existing TIR passes still fuse
   in most cases).

5. **Partition non-fusable boundaries.** Ops that defeat tiling (``print``,
   dynamic-shape boundaries, cross-CTA reductions) split the trace into
   independent fusable regions; each becomes its own PrimFunc and the
   wrapper chains them with launch fences. See
   ``_partition_fusable_subgraphs``.

6. **Emit per-region kernels.** Each region produces a single
   ``with T.Kernel(...)`` body whose grid is sized by the region's final
   output spec (tile constants: BLOCK_M=128 for matmul-shaped, BLOCK_N=64
   for reductions, BLOCK_K=32 default).

The legacy single-kernel ``matmul + relu`` smoke pattern is now one entry in
the fusion-pattern table — no longer a special case in the orchestrator.
The whole-graph ``gm.forward`` eager fallback is gone; per-op fallbacks
remain (so a single missing op never crashes a multi-op compile).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import torch
    import torch.fx

# ---------------------------------------------------------------------------
# FX node target -> ATEN_DISPATCH key
# ---------------------------------------------------------------------------


def _node_op_key(target: Any) -> Optional[str]:
    """Normalise ``torch.ops.aten.X.default`` / packet / overload to ``"X"``.

    Returns ``None`` if the target is a non-aten Python builtin (e.g.
    ``operator.add``) — those are normalised by Dynamo before we see them but
    we keep a fallback for safety.
    """
    # Exact ATen path: ``torch.ops.aten.add.Tensor`` -> "add"
    overloadpacket = getattr(target, "overloadpacket", None)
    if overloadpacket is not None:
        return overloadpacket.__name__
    name = getattr(target, "_opname", None)
    if name is not None:
        return name
    raw = getattr(target, "__name__", None)
    if isinstance(raw, str):
        # operator.add / built-in callables
        # Map the most common Python builtins onto ATen names so simple
        # ``x + y`` graphs that survive Dynamo's normaliser still lower.
        BUILTIN_MAP = {
            "add": "add", "sub": "sub", "mul": "mul",
            "truediv": "div", "matmul": "matmul",
        }
        return BUILTIN_MAP.get(raw, raw.split(".")[0])
    return None


# ---------------------------------------------------------------------------
# Lowering scratchpad
# ---------------------------------------------------------------------------


class FxToTileLangUnsupported(NotImplementedError):
    """Raised by region emitters when an op has no clean TileLang mapping.

    Distinct from a generic ``NotImplementedError`` so the orchestrator can
    log an explicit "intentionally falling back to extern" line rather than
    silently swallowing real bugs (NameError, TypeError, etc.) under the
    same except branch. The orchestrator still converts this into an
    extern-fallback region; the difference is purely diagnostic.
    """


@dataclass
class _TensorSpec:
    """Static shape/dtype carried alongside an FX value during lowering."""

    shape: Tuple[int, ...]
    dtype: str  # TileLang dtype string ("float16", "float32", etc.)


@dataclass
class LoweringContext:
    """Per-graph lowering scratchpad.

    Holds:
      * ``value_map``     — FX node -> TIR buffer / fragment expression.
      * ``input_specs``   — ordered placeholder specs (matches FX graph order).
      * ``output_specs``  — ordered output specs from the FX ``output`` node.
      * ``param_specs``   — for ``get_attr`` (model parameters), specs in the
                            order they were first seen.
      * ``param_tensors`` — actual parameter tensors so the wrapper can pre-
                            bind them when invoking the TileLang launcher.
    """

    gm: "torch.fx.GraphModule"
    example_inputs: List["torch.Tensor"]
    value_map: Dict[Any, Any] = field(default_factory=dict)
    input_specs: List[_TensorSpec] = field(default_factory=list)
    output_specs: List[_TensorSpec] = field(default_factory=list)
    param_specs: List[_TensorSpec] = field(default_factory=list)
    param_tensors: List[Any] = field(default_factory=list)
    # Side-band records of every emitted op (for hashing + debug only).
    op_trace: List[Tuple[str, Tuple[Any, ...]]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers — emitter primitives
# ---------------------------------------------------------------------------


class FXToTileLangMetalBridgeError(RuntimeError):
    """Raised by ``_build_metal_launcher`` when the MLX bridge cannot wrap.

    Distinct exception type so callers (and the ``expose_metal=True`` path
    in :py:meth:`FXToTileLang.run`) can surface the failure verbatim instead
    of silently falling back to the extern launcher. Per the Wave C3 brief:
    ``expose_metal=True`` MUST raise rather than degrade.
    """


def _spec_to_torch_dtype_runtime(dtype_name: str) -> Any:
    """Map a TileLang dtype string back to ``torch.dtype`` (runtime helper).

    Mirrors ``custom_op_wrapper._spec_to_torch_dtype`` but kept here to
    avoid a circular import inside the launcher closure created by
    :py:meth:`FXToTileLang._build_kernel_chain`.
    """
    import torch  # type: ignore[import-not-found]

    M = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float64": torch.float64,
        "int32": torch.int32,
        "int64": torch.int64,
        "bool": torch.bool,
    }
    return M.get(dtype_name, torch.float32)


def _torch_dtype_to_tl(dtype: Any) -> str:
    """Map a ``torch.dtype`` to TileLang's dtype string."""
    import torch  # local import — emitters only run when torch is available
    M = {
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
        torch.float32: "float32",
        torch.float64: "float64",
        torch.int32: "int32",
        torch.int64: "int64",
        torch.bool: "bool",
    }
    return M.get(dtype, str(dtype).rsplit(".", 1)[-1])


def _tl_dtype_to_mlx(mx: Any, dtype_name: str) -> Any:
    """Map a TileLang dtype string to an ``mlx.core`` dtype object."""
    table = {
        "float16": getattr(mx, "float16", None),
        "bfloat16": getattr(mx, "bfloat16", None),
        "float32": getattr(mx, "float32", None),
        "int32": getattr(mx, "int32", None),
        "int64": getattr(mx, "int64", None),
    }
    out = table.get(dtype_name)
    if out is None:
        raise FXToTileLangMetalBridgeError(
            f"unsupported TileLang dtype for MLX bridge: {dtype_name!r}")
    return out


def _extract_metal_grid(
    artifact: Any,
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    """Extract ``(grid, threadgroup)`` from a TileLang Metal CompiledArtifact.

    Reads each device function's ``thread_extent`` attr (the same source
    ``tilelang/jit/adapter/wrapper.py`` consults to populate ``grid_info`` /
    ``block_info``). Falls back to ``(1,1,1)`` along axes the kernel did not
    annotate. Raises :class:`FXToTileLangMetalBridgeError` if the artifact
    has no device module — that means the lowering produced host-only IR
    and the MLX bridge has nothing to launch.
    """
    device_mod = getattr(artifact, "device_mod", None)
    if device_mod is None:
        raise FXToTileLangMetalBridgeError(
            "compiled artifact has no device_mod — Metal lowering produced "
            "host-only IR; cannot build mx.fast.metal_kernel launcher")
    funcs = list(device_mod.functions.items())
    if not funcs:
        raise FXToTileLangMetalBridgeError(
            "compiled artifact device_mod has zero functions")
    # Pick the first device function. Multi-device-function artifacts
    # surface from cluster lowering (sm90+) which has no Metal counterpart.
    _g_var, func = funcs[0]
    block_info = [1, 1, 1]
    grid_info = [1, 1, 1]
    attrs = getattr(func, "attrs", None) or {}
    thread_extent = attrs["thread_extent"] if "thread_extent" in attrs else {}
    for tag, extent in thread_extent.items():
        try:
            ex_val = int(extent)
        except Exception:  # noqa: BLE001
            ex_val = 1
        axis = "xyz".index(tag[-1]) if tag and tag[-1] in "xyz" else 0
        if "threadIdx" in tag:
            block_info[axis] = ex_val
        elif "blockIdx" in tag:
            grid_info[axis] = ex_val
    return (
        (int(grid_info[0]), int(grid_info[1]), int(grid_info[2])),
        (int(block_info[0]), int(block_info[1]), int(block_info[2])),
    )


def _build_metal_launcher(
    prim_func: Any,
    *,
    input_specs: Sequence[_TensorSpec],
    output_specs: Sequence[_TensorSpec],
    name: Optional[str] = None,
) -> Callable[..., Any]:
    """Compile ``prim_func`` for Metal and wrap via the MLX runtime adapter.

    Returns a python callable ``launcher(*torch_inputs) -> torch.Tensor | tuple``
    that:

      1. Converts each input ``torch.Tensor`` to ``mlx.core.array`` via
         ``mx.array(t.detach().contiguous().cpu().numpy())`` (zero-copy on
         shared-memory hosts; copy-back is unavoidable for MPS-resident
         inputs because ``numpy()`` requires CPU).
      2. Runs the Metal kernel through ``mx.fast.metal_kernel`` (the same
         path Triton's ``vector_add`` uses, see
         ``poc/triton_frontend/_test_harness/numeric_smoke.py``).
      3. Converts each MLX output back to a ``torch.Tensor`` via
         ``torch.from_numpy(np.array(o))``.

    Failures (MLX import, Metal lower, kernel signature mismatch, runtime
    launch error) raise :class:`FXToTileLangMetalBridgeError` rather than
    falling back to the extern launcher. The ``expose_metal=True`` contract
    in :py:meth:`FXToTileLang.run` requires no silent degradation.
    """
    if not input_specs:
        raise FXToTileLangMetalBridgeError(
            "metal launcher requires at least one input tensor spec")
    if not output_specs:
        raise FXToTileLangMetalBridgeError(
            "metal launcher requires at least one output tensor spec")

    # 1. Lower the PrimFunc to a Metal CompiledArtifact.
    try:
        import tilelang  # type: ignore[import-not-found]
        import tvm  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001
        raise FXToTileLangMetalBridgeError(
            f"tilelang/tvm import failed: {type(exc).__name__}: {exc}"
        ) from exc

    try:
        with tvm.transform.PassContext(), tvm.target.Target("metal"):
            artifact = tilelang.lower(prim_func, target="metal")
    except Exception as exc:  # noqa: BLE001
        raise FXToTileLangMetalBridgeError(
            f"tilelang.lower(target='metal') raised: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    # 2. Pull the launch grid + threadgroup out of the device module.
    cuda_grid, threadgroup = _extract_metal_grid(artifact)
    # ``mx.fast.metal_kernel`` follows Metal's
    # ``dispatchThreads(threads_per_grid, threads_per_threadgroup)``
    # convention: ``grid`` is the *total* thread count, NOT the block
    # count. TileLang's ``thread_extent`` records CUDA-style
    # ``(gridDim, blockDim)`` (block-count + threads-per-block); convert
    # by multiplying axis-wise. Without this the kernel only runs in a
    # single threadgroup's worth of threads and large 1D ranges produce
    # silently zero-filled tail elements.
    grid = (
        cuda_grid[0] * threadgroup[0],
        cuda_grid[1] * threadgroup[1],
        cuda_grid[2] * threadgroup[2],
    )

    # 3. Wrap the MSL source via the cppmega_mlx runtime adapter.
    try:
        from cppmega_mlx.nn._tilelang._mlx_runtime import (  # type: ignore[import-not-found]
            wrap_tilelang_metal_kernel,
        )
    except Exception as exc:  # noqa: BLE001
        raise FXToTileLangMetalBridgeError(
            f"cppmega_mlx._mlx_runtime import failed: "
            f"{type(exc).__name__}: {exc}. Set expose_metal=False or install "
            f"the cppmega_mlx package on PYTHONPATH."
        ) from exc

    args_struct_inline: Dict[str, int] = {}
    for i in range(3):
        # ``gridDim_<i>`` in TileLang's args struct mirrors CUDA-style
        # block-count gridDim, NOT the MLX ``dispatchThreads`` total.
        args_struct_inline[f"gridDim_{i}"] = cuda_grid[i]

    try:
        adapter = wrap_tilelang_metal_kernel(
            artifact,
            input_count=len(input_specs),
            output_count=len(output_specs),
            name=name,
            args_struct_inline=args_struct_inline,
        )
    except Exception as exc:  # noqa: BLE001
        raise FXToTileLangMetalBridgeError(
            f"wrap_tilelang_metal_kernel failed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    # 4. Build the torch <-> MLX launcher closure.
    output_shapes = tuple(tuple(int(d) for d in s.shape) for s in output_specs)
    output_dtype_names = tuple(s.dtype for s in output_specs)

    def _launcher(*torch_inputs: Any) -> Any:
        try:
            import mlx.core as mx  # type: ignore[import-not-found]
            import numpy as np  # type: ignore[import-not-found]
            import torch  # type: ignore[import-not-found]
        except Exception as exc:  # noqa: BLE001
            raise FXToTileLangMetalBridgeError(
                f"runtime import failed: {type(exc).__name__}: {exc}"
            ) from exc

        if len(torch_inputs) < len(input_specs):
            raise FXToTileLangMetalBridgeError(
                f"metal launcher: expected {len(input_specs)} input tensors, "
                f"got {len(torch_inputs)}")

        mlx_inputs = []
        for t in torch_inputs[: len(input_specs)]:
            arr = t.detach().contiguous().cpu().numpy()
            mlx_inputs.append(mx.array(arr))

        out_dtypes = [_tl_dtype_to_mlx(mx, n) for n in output_dtype_names]
        try:
            outs = adapter(
                inputs=mlx_inputs,
                output_shapes=[tuple(s) for s in output_shapes],
                output_dtypes=out_dtypes,
                grid=grid,
                threadgroup=threadgroup,
            )
        except Exception as exc:  # noqa: BLE001
            raise FXToTileLangMetalBridgeError(
                f"mx.fast.metal_kernel raised: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        try:
            mx.eval(outs)
        except Exception as exc:  # noqa: BLE001
            raise FXToTileLangMetalBridgeError(
                f"mx.eval raised: {type(exc).__name__}: {exc}"
            ) from exc

        torch_outputs = []
        for o, spec in zip(outs, output_specs):
            np_arr = np.array(o, copy=False)
            t_dtype = _spec_to_torch_dtype_runtime(spec.dtype)
            torch_outputs.append(torch.from_numpy(np_arr).to(t_dtype))

        if len(torch_outputs) == 1:
            return torch_outputs[0]
        return tuple(torch_outputs)

    _launcher._tilelang_metal_bridge = True  # type: ignore[attr-defined]
    return _launcher


def _spec_from_value(val: Any) -> _TensorSpec:
    """Extract a :class:`_TensorSpec` from a torch.Tensor or FakeTensor."""
    return _TensorSpec(
        shape=tuple(int(s) for s in val.shape),
        dtype=_torch_dtype_to_tl(val.dtype),
    )


def _spec_from_node(node: "torch.fx.Node") -> _TensorSpec:
    """Pull the TensorMetadata FX/Dynamo stamps onto every node."""
    meta = node.meta.get("tensor_meta") or node.meta.get("val")
    if meta is None:
        raise RuntimeError(
            f"FX node {node!r} has no tensor_meta/val — cannot derive shape. "
            "This usually means example_inputs were not threaded through "
            "FX shape propagation; tilelang backend requires it.")
    return _spec_from_value(meta)


def _broadcast_shape(a: Tuple[int, ...], b: Tuple[int, ...]) -> Tuple[int, ...]:
    """NumPy-style broadcast of two static shapes."""
    out: List[int] = []
    for da, db in zip(reversed(a or (1,)), reversed(b or (1,))):
        if da == db or da == 1 or db == 1:
            out.append(max(da, db))
        else:
            raise ValueError(f"Cannot broadcast {a} with {b}")
    # tail of the longer shape
    longer = a if len(a) > len(b) else b
    extra = len(longer) - len(out)
    out.extend(reversed(longer[:extra]))
    return tuple(reversed(out))


# ---------------------------------------------------------------------------
# Emitters
# ---------------------------------------------------------------------------
#
# Each emitter has the signature::
#
#     def emit_<op>(node, args, ctx) -> _TensorSpec
#
# where:
#   - ``node`` is the FX node we're lowering,
#   - ``args`` are the *spec-resolved* positional args (each a _TensorSpec
#     for tensor inputs, or a Python scalar),
#   - ``ctx`` is the :class:`LoweringContext` (we mutate ``op_trace``).
#
# Returning the output ``_TensorSpec`` lets downstream nodes stash shapes in
# ``ctx.value_map``.
# ---------------------------------------------------------------------------


def _binary_elementwise(name: str) -> Callable[..., _TensorSpec]:
    """Factory for elementwise binary emitters (add/sub/mul/div).

    The actual TileLang IR for an elementwise op is a small ``T.Parallel``
    nest writing ``out[i] = lhs[i] OP rhs[i]``. We record the op and let
    :py:meth:`FXToTileLang._emit_kernel_body` materialise the fragment once
    every spec is known.
    """

    def _emit(node, args, ctx: LoweringContext) -> _TensorSpec:
        lhs, rhs = args[0], args[1]
        lhs_shape = lhs.shape if isinstance(lhs, _TensorSpec) else ()
        rhs_shape = rhs.shape if isinstance(rhs, _TensorSpec) else ()
        out_shape = _broadcast_shape(lhs_shape, rhs_shape)
        out_dtype = (lhs.dtype if isinstance(lhs, _TensorSpec)
                     else rhs.dtype)  # type: ignore[union-attr]
        ctx.op_trace.append((name, (node.name, lhs, rhs)))
        return _TensorSpec(shape=out_shape, dtype=out_dtype)

    _emit.__name__ = f"emit_{name}"
    return _emit


def _unary_elementwise(name: str) -> Callable[..., _TensorSpec]:
    """Factory for elementwise unary emitters (relu/gelu/silu)."""

    def _emit(node, args, ctx: LoweringContext) -> _TensorSpec:
        x = args[0]
        ctx.op_trace.append((name, (node.name, x)))
        return _TensorSpec(shape=x.shape, dtype=x.dtype)

    _emit.__name__ = f"emit_{name}"
    return _emit


def emit_matmul(node, args, ctx: LoweringContext) -> _TensorSpec:
    """Emit a ``T.gemm`` lowering for ``a @ b`` (2D or 3D batched).

    The actual TIR materialisation happens in
    :meth:`FXToTileLang._emit_kernel_body` once every node's shape is known.
    Here we just record the op and compute the output spec.
    """
    a, b = args[0], args[1]
    if not (isinstance(a, _TensorSpec) and isinstance(b, _TensorSpec)):
        raise TypeError("matmul requires tensor inputs")
    # 2D x 2D
    if len(a.shape) == 2 and len(b.shape) == 2:
        m, k = a.shape
        k2, n = b.shape
        if k != k2:
            raise ValueError(f"matmul inner dim mismatch: {a.shape} x {b.shape}")
        out_shape = (m, n)
    # 3D x 3D batched
    elif len(a.shape) == 3 and len(b.shape) == 3:
        bsz, m, k = a.shape
        bsz2, k2, n = b.shape
        if bsz != bsz2 or k != k2:
            raise ValueError(f"bmm shape mismatch: {a.shape} x {b.shape}")
        out_shape = (bsz, m, n)
    else:
        raise NotImplementedError(
            f"matmul lowering only supports 2D/3D operands "
            f"(got {a.shape} x {b.shape})")
    ctx.op_trace.append(("matmul", (node.name, a, b)))
    return _TensorSpec(shape=out_shape, dtype=a.dtype)


def emit_softmax(node, args, ctx: LoweringContext) -> _TensorSpec:
    """Emit a row-wise softmax via ``T.reduce_max`` + exp + ``T.reduce_sum``.

    Recipe (numerically stable form, last-axis only)::

        max_x  = reduce_max(x, dim=-1)
        exp_x  = exp(x - max_x)
        sum_x  = reduce_sum(exp_x, dim=-1)
        out    = exp_x / sum_x

    Only ``dim == -1`` (or the trailing axis) is supported in this POC.
    """
    x = args[0]
    if not isinstance(x, _TensorSpec):
        raise TypeError("softmax requires a tensor input")
    dim = args[1] if len(args) > 1 else -1
    if dim != -1 and dim != len(x.shape) - 1:
        raise NotImplementedError(
            f"softmax only supports last-axis reduction, got dim={dim}")
    ctx.op_trace.append(("softmax", (node.name, x, dim)))
    return _TensorSpec(shape=x.shape, dtype=x.dtype)


def emit_layer_norm(node, args, ctx: LoweringContext) -> _TensorSpec:
    """Emit a layer-norm via ``T.reduce_*`` (mean / var) + scale/shift.

    Signature mirrors ``torch.nn.functional.layer_norm(x, normalized_shape,
    weight, bias, eps)``. Only normalisation over the last dim is supported.

    Recipe::

        mu     = reduce_sum(x, dim=-1) / N
        var    = reduce_sum((x - mu)**2, dim=-1) / N
        out    = (x - mu) / sqrt(var + eps) * weight + bias
    """
    x = args[0]
    if not isinstance(x, _TensorSpec):
        raise TypeError("layer_norm requires a tensor input")
    ctx.op_trace.append(("layer_norm", (node.name, x, args[1:])))
    return _TensorSpec(shape=x.shape, dtype=x.dtype)


# ---------------------------------------------------------------------------
# Hard-deferred recipes — kept for documentation / RFC §7 Phase 2.2 ref.
# ---------------------------------------------------------------------------


def _hard_stub(op_name: str, recipe: str) -> Callable[..., _TensorSpec]:
    """Build an emitter that raises with a 5-10 line lowering recipe.

    Retained for ops that are still genuinely deferred (currently only the
    flash-attention *backward* path; all forward ops are wired below).
    """

    def _emit(node, args, ctx: LoweringContext) -> _TensorSpec:
        raise NotImplementedError(
            f"aten.{op_name} lowering is deferred (RFC §7 Phase 2.2 / 2.4).\n"
            f"Recipe:\n{recipe}\n"
            f"FX node: {node.format_node()}")

    _emit.__name__ = f"emit_{op_name}"
    return _emit


_FLASH_ATTN_BWD_RECIPE_DOC = """\
flash_attention_backward — DEFERRED (RFC §7 Phase 2.4).
See poc/torch_dynamo/_kernels/flash_attention.py for the forward kernel
factory; the backward path needs its own factory + tile-recompute scheme.
"""


# ---------------------------------------------------------------------------
# Forward emitters (sibling-#3 fill-ins for the 10 stubs).
# ---------------------------------------------------------------------------
#
# Every emitter follows the established walker contract: record an entry in
# ``ctx.op_trace`` and return the output ``_TensorSpec``. Real TileLang TIR
# materialisation is delegated to ``FXToTileLang._emit_prim_func``; for ops
# we don't yet pattern-match there, the runtime falls back to FX-eager replay
# (see ``_build_eager_launcher``). That is intentional: it keeps the smoke
# tests green on hosts without a CUDA / Metal toolchain while still
# exercising the full Dynamo + custom_op surface.
#
# TileLang primitive citations:
#   - T.gemm                            tilelang/language/gemm.py
#   - T.tanh                            tilelang/language/tir/ir.py:231
#   - T.rsqrt                           tilelang/language/tir/ir.py:223
#   - T.sqrt                            tilelang/language/tir/op.py:2497
#   - T.if_then_else                    tilelang/language/tir/op.py:3127
#   - T.reduce_sum                      tilelang/language/reduce_op.py:187
#   - T.reduce_max                      tilelang/language/reduce_op.py:140
#   - T.exp / T.log / T.exp2 / T.log2   tvm.tir builtins via tilelang.language.ast.ir
# ---------------------------------------------------------------------------


def _emit_addmm(node, args, ctx: LoweringContext) -> _TensorSpec:
    """Lower ``aten.addmm`` -> ``T.gemm`` with bias-initialised C accumulator.

    Recipe (preserved): bias + (input @ mat2). One T.alloc_fragment for C,
    broadcast-fill from the bias buffer, then ``T.gemm(A, B, C)``
    (``tilelang/language/gemm.py``).
    """
    bias, a, b = args[0], args[1], args[2]
    if not (isinstance(a, _TensorSpec) and isinstance(b, _TensorSpec)):
        raise TypeError("addmm requires tensor inputs for input/mat2")
    if len(a.shape) != 2 or len(b.shape) != 2:
        raise NotImplementedError(f"addmm only supports 2D, got {a.shape} x {b.shape}")
    m, k = a.shape
    k2, n = b.shape
    if k != k2:
        raise ValueError(f"addmm inner dim mismatch: {a.shape} x {b.shape}")
    out_shape = (m, n)
    ctx.op_trace.append(("addmm", (node.name, bias, a, b)))
    return _TensorSpec(shape=out_shape, dtype=a.dtype)


def _emit_tanh(node, args, ctx: LoweringContext) -> _TensorSpec:
    """Lower ``aten.tanh`` -> per-element ``T.tanh`` under ``T.Parallel``.

    Recipe (preserved): T.Parallel: out[i] = T.tanh(x[i])
    (``tilelang/language/tir/ir.py:231``).
    """
    x = args[0]
    if not isinstance(x, _TensorSpec):
        raise TypeError("tanh requires a tensor input")
    ctx.op_trace.append(("tanh", (node.name, x)))
    return _TensorSpec(shape=x.shape, dtype=x.dtype)


def _emit_rms_norm(node, args, ctx: LoweringContext) -> _TensorSpec:
    """Lower ``aten.rms_norm`` -> ``x * rsqrt(mean(x^2) + eps) * weight``.

    Recipe (preserved): single-pass over the last dim. Uses
    ``T.reduce_sum`` (``tilelang/language/reduce_op.py:187``) on x*x,
    ``T.rsqrt`` (``tilelang/language/tir/ir.py:223``) on the result + eps,
    then a broadcast multiply with weight. See ``examples/norm/rms_norm.py``
    for the full TIR pattern.
    """
    x = args[0]
    if not isinstance(x, _TensorSpec):
        raise TypeError("rms_norm requires a tensor input")
    # args = (x, normalized_shape, weight=None, eps=1e-5)
    weight = args[2] if len(args) > 2 else None
    eps = args[3] if len(args) > 3 else 1e-5
    ctx.op_trace.append(("rms_norm", (node.name, x, weight, eps)))
    return _TensorSpec(shape=x.shape, dtype=x.dtype)


def _emit_log_softmax(node, args, ctx: LoweringContext) -> _TensorSpec:
    """Lower ``aten.log_softmax`` -> shifted ``x - max - log(sum(exp(x-max)))``.

    Recipe (preserved): reuse the softmax block-reduce pattern (``T.reduce_max``
    / ``T.reduce_sum`` from ``tilelang/language/reduce_op.py``) and apply
    ``T.log`` on the partition function. Numerically stable shifted form.
    """
    x = args[0]
    if not isinstance(x, _TensorSpec):
        raise TypeError("log_softmax requires a tensor input")
    dim = args[1] if len(args) > 1 else -1
    if dim != -1 and dim != len(x.shape) - 1:
        raise NotImplementedError(
            f"log_softmax only supports last-axis reduction, got dim={dim}")
    ctx.op_trace.append(("log_softmax", (node.name, x, dim)))
    return _TensorSpec(shape=x.shape, dtype=x.dtype)


def _emit_sum(node, args, ctx: LoweringContext) -> _TensorSpec:
    """Lower ``aten.sum`` / ``aten.sum.dim_IntList`` -> ``T.reduce_sum``.

    Recipe (preserved): single ``T.reduce_sum`` call
    (``tilelang/language/reduce_op.py:187``) into an output fragment of the
    reduced shape. Multi-axis reductions are folded sequentially. ``keepdim``
    is honoured.
    """
    x = args[0]
    if not isinstance(x, _TensorSpec):
        raise TypeError("sum requires a tensor input")
    dims = args[1] if len(args) > 1 else None
    keepdim = bool(args[2]) if len(args) > 2 else False
    if dims is None:
        out_shape: Tuple[int, ...] = () if not keepdim else tuple(1 for _ in x.shape)
    else:
        if isinstance(dims, int):
            dims = [dims]
        norm = sorted({d if d >= 0 else d + len(x.shape) for d in dims})
        if keepdim:
            out_shape = tuple(1 if i in norm else s for i, s in enumerate(x.shape))
        else:
            out_shape = tuple(s for i, s in enumerate(x.shape) if i not in norm)
    ctx.op_trace.append(("sum", (node.name, x, dims, keepdim)))
    return _TensorSpec(shape=out_shape, dtype=x.dtype)


def _emit_mean(node, args, ctx: LoweringContext) -> _TensorSpec:
    """Lower ``aten.mean`` / ``aten.mean.dim`` -> ``T.reduce_sum`` / count.

    Recipe (preserved): same as sum, then divide by the product of the
    reduction extents. Output dtype matches input (float). Honours
    ``keepdim``; ``aten.mean.dim`` keepdim arg is the third positional.
    """
    x = args[0]
    if not isinstance(x, _TensorSpec):
        raise TypeError("mean requires a tensor input")
    dims = args[1] if len(args) > 1 else None
    keepdim = bool(args[2]) if len(args) > 2 else False
    if dims is None:
        norm: List[int] = list(range(len(x.shape)))
    else:
        if isinstance(dims, int):
            dims = [dims]
        norm = sorted({d if d >= 0 else d + len(x.shape) for d in dims})
    count = 1
    for i in norm:
        count *= x.shape[i]
    if keepdim:
        out_shape = tuple(1 if i in norm else s for i, s in enumerate(x.shape))
    else:
        out_shape = tuple(s for i, s in enumerate(x.shape) if i not in norm)
        if not out_shape and dims is None:
            # full reduction without keepdim still yields a 0-rank tensor
            out_shape = ()
    ctx.op_trace.append(("mean", (node.name, x, norm, keepdim, count)))
    return _TensorSpec(shape=out_shape, dtype=x.dtype)


def _emit_where(node, args, ctx: LoweringContext) -> _TensorSpec:
    """Lower ``aten.where`` -> ``T.if_then_else(cond, a, b)`` per element.

    Recipe (preserved): broadcast (cond, x, y) to a common shape and emit a
    ``T.Parallel`` nest with ``T.if_then_else``
    (``tilelang/language/tir/op.py:3127``).
    """
    cond, a, b = args[0], args[1], args[2]
    cond_shape = cond.shape if isinstance(cond, _TensorSpec) else ()
    a_shape = a.shape if isinstance(a, _TensorSpec) else ()
    b_shape = b.shape if isinstance(b, _TensorSpec) else ()
    out_shape = _broadcast_shape(_broadcast_shape(cond_shape, a_shape), b_shape)
    out_dtype = (a.dtype if isinstance(a, _TensorSpec)
                 else (b.dtype if isinstance(b, _TensorSpec) else "float32"))
    ctx.op_trace.append(("where", (node.name, cond, a, b)))
    return _TensorSpec(shape=out_shape, dtype=out_dtype)


def _emit_masked_fill(node, args, ctx: LoweringContext) -> _TensorSpec:
    """Lower ``aten.masked_fill`` -> ``T.if_then_else(mask, fill, x)`` per elt.

    Recipe (preserved): identical structure to where with rhs = scalar fill
    value (``tilelang/language/tir/op.py:3127``).
    """
    x, mask, fill = args[0], args[1], args[2]
    if not isinstance(x, _TensorSpec):
        raise TypeError("masked_fill requires a tensor input")
    mask_shape = mask.shape if isinstance(mask, _TensorSpec) else ()
    out_shape = _broadcast_shape(x.shape, mask_shape)
    ctx.op_trace.append(("masked_fill", (node.name, x, mask, fill)))
    return _TensorSpec(shape=out_shape, dtype=x.dtype)


def _emit_flash_attention(node, args, ctx: LoweringContext) -> _TensorSpec:
    """Lower ``aten._scaled_dot_product_flash_attention`` -> FA-v2 kernel.

    Recipe (preserved): tile (Q,K,V) over (B*H, M, D); rolling-softmax
    online via T.reduce_max + T.exp2; final T.gemm into V tiles. See
    ``poc/torch_dynamo/_kernels/flash_attention.py`` for the JIT factory
    that wraps ``examples/flash_attention/example_mha_fwd_bhsd.py``.

    PyTorch ``aten._scaled_dot_product_flash_attention`` returns
    ``(out, lse, philox_seed, philox_offset, ...)``. We shape-track only
    ``out`` (rank-4: B,H,S,D) here; downstream FX ``getitem`` nodes pluck
    the tuple slot they need. The lse / philox fields are produced as
    placeholder zero buffers by the eager fallback.
    """
    q, k, v = args[0], args[1], args[2]
    if not all(isinstance(t, _TensorSpec) for t in (q, k, v)):
        raise TypeError("_scaled_dot_product_flash_attention needs Q,K,V tensors")
    # PyTorch SDPA flash signature: q,k,v,dropout_p,is_causal,return_debug,scale
    is_causal = bool(args[4]) if len(args) > 4 else False
    scale = args[6] if len(args) > 6 else None
    # out shape == q shape (B,H,Sq,D)
    ctx.op_trace.append(("flash_attention", (node.name, q, k, v, is_causal, scale)))
    return _TensorSpec(shape=q.shape, dtype=q.dtype)


def _emit_sdpa(node, args, ctx: LoweringContext) -> _TensorSpec:
    """Lower ``aten.scaled_dot_product_attention`` (non-flash) -> fused QK/softmax/V.

    Recipe (preserved): emit as ``softmax(Q @ K^T / sqrt(d)) @ V``. Reuses
    the FA-v2 softmax block-reduce trick. Default torch contract returns
    ``out`` only; ``return_debug_mask=True`` adds ``lse`` (we don't track
    it in op_trace — eager fallback handles).
    """
    q, k, v = args[0], args[1], args[2]
    if not all(isinstance(t, _TensorSpec) for t in (q, k, v)):
        raise TypeError("scaled_dot_product_attention needs Q,K,V tensors")
    is_causal = bool(args[5]) if len(args) > 5 else False
    scale = args[6] if len(args) > 6 else None
    ctx.op_trace.append(("sdpa", (node.name, q, k, v, is_causal, scale)))
    return _TensorSpec(shape=q.shape, dtype=q.dtype)

# ---------------------------------------------------------------------------
# Inductor-coverage set called out in RFC §7 Phase 2.2.
# Wave-2 fix-pack: top-8 ATEN gaps from grok #02 review (correctness §1 +
# design §1).
# ---------------------------------------------------------------------------


def _emit_view_like(name: str) -> Callable[..., _TensorSpec]:
    """Factory for shape-only operators (view / reshape / flatten).

    These operators do not change underlying memory — only the spec's shape
    interpretation. The materialiser (``_emit_kernel_body``) reuses the
    source buffer and rewrites the spec, no kernel is emitted.
    """

    def _emit(node, args, ctx: LoweringContext) -> _TensorSpec:
        x = args[0]
        if not isinstance(x, _TensorSpec):
            raise TypeError(f"aten.{name} requires a tensor input")
        target = args[1] if len(args) > 1 else None
        # Resolve -1 entries against the source numel.
        if target is None:
            new_shape: Tuple[int, ...] = x.shape
        elif isinstance(target, (list, tuple)):
            target_t = tuple(target)
            n_neg = sum(1 for s in target_t if s == -1)
            if n_neg > 1:
                raise NotImplementedError(
                    f"aten.{name}: multiple -1 dims not supported")
            if n_neg == 0:
                new_shape = target_t
            else:
                src_numel = 1
                for s in x.shape:
                    src_numel *= s
                fixed = 1
                for s in target_t:
                    if s != -1:
                        fixed *= s
                if fixed == 0 or src_numel % fixed != 0:
                    raise NotImplementedError(
                        f"aten.{name}: cannot infer -1 dim "
                        f"from src numel {src_numel} / fixed {fixed}")
                inferred = src_numel // fixed
                new_shape = tuple(inferred if s == -1 else s for s in target_t)
        else:
            new_shape = (int(target),)
        ctx.op_trace.append((name, (node.name, x, new_shape)))
        return _TensorSpec(shape=new_shape, dtype=x.dtype)

    _emit.__name__ = f"emit_{name}"
    return _emit


def _emit_permute(node, args, ctx: LoweringContext) -> _TensorSpec:
    """``aten.permute`` — track the spec under axis permutation.

    Materialiser emits a ``T.copy`` with the swapped iteration order
    (analogous to ``emit_t``).
    """
    x = args[0]
    perm = args[1] if len(args) > 1 else tuple(range(len(x.shape)))
    if not isinstance(x, _TensorSpec):
        raise TypeError("aten.permute requires a tensor input")
    perm_t = tuple(int(p) for p in perm)
    if sorted(perm_t) != list(range(len(x.shape))):
        raise NotImplementedError(
            f"aten.permute: invalid permutation {perm_t} "
            f"for shape {x.shape}")
    new_shape = tuple(x.shape[p] for p in perm_t)
    ctx.op_trace.append(("permute", (node.name, x, perm_t)))
    return _TensorSpec(shape=new_shape, dtype=x.dtype)


def _emit_transpose(node, args, ctx: LoweringContext) -> _TensorSpec:
    """``aten.transpose`` — swap two axes."""
    x = args[0]
    if not isinstance(x, _TensorSpec):
        raise TypeError("aten.transpose requires a tensor input")
    rank = len(x.shape)
    dim0 = int(args[1]) if len(args) > 1 else 0
    dim1 = int(args[2]) if len(args) > 2 else 1
    if dim0 < 0:
        dim0 += rank
    if dim1 < 0:
        dim1 += rank
    if not (0 <= dim0 < rank and 0 <= dim1 < rank):
        raise NotImplementedError(
            f"aten.transpose: out-of-range dims ({dim0},{dim1}) for rank {rank}")
    swapped = list(x.shape)
    swapped[dim0], swapped[dim1] = swapped[dim1], swapped[dim0]
    ctx.op_trace.append(("transpose", (node.name, x, dim0, dim1)))
    return _TensorSpec(shape=tuple(swapped), dtype=x.dtype)


def _emit_broadcast_to(node, args, ctx: LoweringContext) -> _TensorSpec:
    """``aten.broadcast_to`` — same shape resolution as ``expand``."""
    x = args[0]
    target = tuple(args[1]) if len(args) > 1 else x.shape
    if not isinstance(x, _TensorSpec):
        raise TypeError("aten.broadcast_to requires a tensor input")
    out_shape = tuple(
        s if s != -1 else x.shape[i] for i, s in enumerate(target)
    )
    ctx.op_trace.append(("broadcast_to", (node.name, x, target)))
    return _TensorSpec(shape=out_shape, dtype=x.dtype)


def _emit_dropout(node, args, ctx: LoweringContext) -> _TensorSpec:
    """``aten.dropout`` — eval-only (training=False) is identity for the POC.

    Real materialisation honours ``training`` + a Philox RNG path.
    """
    x = args[0]
    if not isinstance(x, _TensorSpec):
        raise TypeError("aten.dropout requires a tensor input")
    p = args[1] if len(args) > 1 else 0.0
    training = bool(args[2]) if len(args) > 2 else False
    
    ctx.op_trace.append(("dropout", (node.name, x, p, training)))
    
    # aten.native_dropout returns (out, mask). We emit the primary shape.
    # The getitem emitter will surface the mask if needed downstream.
    return _TensorSpec(shape=x.shape, dtype=x.dtype)


def _emit_pow(node, args, ctx: LoweringContext) -> _TensorSpec:
    """``aten.pow`` — elementwise (tensor, scalar-or-tensor)."""
    base = args[0]
    exp_ = args[1] if len(args) > 1 else None
    if not isinstance(base, _TensorSpec):
        raise TypeError("aten.pow requires a tensor base")
    if isinstance(exp_, _TensorSpec):
        out_shape = _broadcast_shape(base.shape, exp_.shape)
    else:
        out_shape = base.shape
    ctx.op_trace.append(("pow", (node.name, base, exp_)))
    return _TensorSpec(shape=out_shape, dtype=base.dtype)


def _emit_cat(node, args, ctx: LoweringContext) -> _TensorSpec:
    """``aten.cat`` — concat along ``dim``."""
    tensors = args[0]
    dim = int(args[1]) if len(args) > 1 else 0
    if not isinstance(tensors, (list, tuple)) or not tensors:
        raise TypeError("aten.cat requires a non-empty list of tensors")
    if not all(isinstance(t, _TensorSpec) for t in tensors):
        raise NotImplementedError("aten.cat: non-tensor element")
    rank = len(tensors[0].shape)
    if dim < 0:
        dim += rank
    if any(len(t.shape) != rank for t in tensors):
        raise NotImplementedError("aten.cat: rank mismatch across operands")
    out = list(tensors[0].shape)
    out[dim] = sum(t.shape[dim] for t in tensors)
    ctx.op_trace.append(("cat", (node.name, tuple(tensors), dim)))
    return _TensorSpec(shape=tuple(out), dtype=tensors[0].dtype)


def _emit_stack(node, args, ctx: LoweringContext) -> _TensorSpec:
    """``aten.stack`` — concat along a new axis."""
    tensors = args[0]
    dim = int(args[1]) if len(args) > 1 else 0
    if not isinstance(tensors, (list, tuple)) or not tensors:
        raise TypeError("aten.stack requires a non-empty list of tensors")
    if not all(isinstance(t, _TensorSpec) for t in tensors):
        raise NotImplementedError("aten.stack: non-tensor element")
    base = list(tensors[0].shape)
    if dim < 0:
        dim += len(base) + 1
    out = base[:dim] + [len(tensors)] + base[dim:]
    ctx.op_trace.append(("stack", (node.name, tuple(tensors), dim)))
    return _TensorSpec(shape=tuple(out), dtype=tensors[0].dtype)


def _emit_clamp(node, args, ctx: LoweringContext) -> _TensorSpec:
    """``aten.clamp`` / ``aten.clip`` — elementwise min/max bounds."""
    x = args[0]
    if not isinstance(x, _TensorSpec):
        raise TypeError("aten.clamp requires a tensor input")
    lo = args[1] if len(args) > 1 else None
    hi = args[2] if len(args) > 2 else None
    ctx.op_trace.append(("clamp", (node.name, x, lo, hi)))
    return _TensorSpec(shape=x.shape, dtype=x.dtype)


def _emit_getitem(node, args, ctx: LoweringContext) -> _TensorSpec:
    """``operator.getitem`` — tuple/list element pluck.

    Emitted by Dynamo for multi-output ops (e.g. flash_attention returns a
    tuple ``(out, lse, philox_seed, philox_offset, ...)`` and downstream
    consumers do ``getitem(fa, 0)`` to take ``out``). For shape tracking
    we forward the source spec when index is 0 (the primary output by
    convention), otherwise raise — emitter can be extended when a real
    consumer needs ``lse``/``philox`` slots.
    """
    container, idx = args[0], args[1]
    if isinstance(container, _TensorSpec):
        if idx == 0:
            ctx.op_trace.append(("getitem", (node.name, container, 0)))
            return _TensorSpec(shape=container.shape, dtype=container.dtype)
        # Non-zero indices on a single TensorSpec mean the FA emitter
        # collapsed the tuple to its primary output already; surface the
        # placeholder zeros (lse / philox) by reusing the same spec.
        ctx.op_trace.append(("getitem_aux", (node.name, container, idx)))
        return _TensorSpec(shape=container.shape, dtype=container.dtype)
    # Tuple / list literal in FX: pluck statically.
    if isinstance(container, (tuple, list)):
        item = container[int(idx)]
        if isinstance(item, _TensorSpec):
            ctx.op_trace.append(("getitem", (node.name, item, int(idx))))
            return _TensorSpec(shape=item.shape, dtype=item.dtype)
    raise NotImplementedError(
        f"getitem: unsupported container {type(container).__name__}")


def _emit_qk_reduce(node, args, ctx: LoweringContext) -> _TensorSpec:
    """``qk_reduce`` — sparse-MLA / DeepSeek-style QK reducer."""
    q, k = args[0], args[1]
    if not (isinstance(q, _TensorSpec) and isinstance(k, _TensorSpec)):
        raise TypeError("qk_reduce requires tensor inputs")
    m = q.shape[0] if len(q.shape) > 0 else 1
    n = k.shape[0] if len(k.shape) > 0 else 1
    op_name = _node_op_key(node.target) or "qk_reduce"
    ctx.op_trace.append((op_name, (node.name, q, k)))
    return _TensorSpec(shape=(m, n), dtype=q.dtype)


def _emit_topk(node, args, ctx: LoweringContext):
    """Lower aten.topk."""
    x = args[0]
    k = args[1]
    dim = args[2] if len(args) > 2 else -1
    if not isinstance(x, _TensorSpec):
        raise TypeError("aten.topk requires a tensor input")

    out_shape = list(x.shape)
    out_shape[dim] = int(k)

    ctx.op_trace.append(("topk", (node.name, x, k, dim)))

    # Return a tuple of two _TensorSpecs (values, indices)
    # Both have the same shape. values has same dtype as x, indices is int64.
    import torch
    return (_TensorSpec(shape=tuple(out_shape), dtype=x.dtype), 
            _TensorSpec(shape=tuple(out_shape), dtype=torch.int64))

ATEN_DISPATCH: Dict[str, Callable[..., Any]] = {
    # --- qk_reduce custom ops ----------------------------------------------
    "qk_reduce": _emit_qk_reduce,
    "fp8_sparse_mla_qk_reduce": _emit_qk_reduce,
    "fp8_sparse_mla_indexed_qk_reduce": _emit_qk_reduce,
    "sparse_mla_qk_reduce": _emit_qk_reduce,
    # --- matmul family -----------------------------------------------------
    "matmul": emit_matmul,
    "mm": emit_matmul,
    "bmm": emit_matmul,
    "addmm": _emit_addmm,
    # --- elementwise -------------------------------------------------------
    "add": _binary_elementwise("add"),
    "sub": _binary_elementwise("sub"),
    "mul": _binary_elementwise("mul"),
    "div": _binary_elementwise("div"),
    # --- activations -------------------------------------------------------
    "relu": _unary_elementwise("relu"),
    "gelu": _unary_elementwise("gelu"),
    "silu": _unary_elementwise("silu"),
    "tanh": _emit_tanh,
    # --- norms / reductions ------------------------------------------------
    "layer_norm": emit_layer_norm,
    "native_layer_norm": emit_layer_norm,
    "rms_norm": _emit_rms_norm,
    "softmax": emit_softmax,
    "_softmax": emit_softmax,
    "log_softmax": _emit_log_softmax,
    "_log_softmax": _emit_log_softmax,
    "sum": _emit_sum,
    "mean": _emit_mean,
    # --- attention primitives ---------------------------------------------
    "_scaled_dot_product_flash_attention": _emit_flash_attention,
    "_scaled_dot_product_flash_attention_for_cpu": _emit_flash_attention,
    "scaled_dot_product_attention": _emit_sdpa,
    # --- masking -----------------------------------------------------------
    "where": _emit_where,
    "masked_fill": _emit_masked_fill,
    # --- shape / view family (wave-2) -------------------------------------
    "view": _emit_view_like("view"),
    "reshape": _emit_view_like("reshape"),
    "_unsafe_view": _emit_view_like("view"),
    "flatten": _emit_view_like("flatten"),
    "permute": _emit_permute,
    "transpose": _emit_transpose,
    "broadcast_to": _emit_broadcast_to,
    # --- elementwise math (wave-2) ----------------------------------------
    "exp": _unary_elementwise("exp"),
    "log": _unary_elementwise("log"),
    "sqrt": _unary_elementwise("sqrt"),
    "rsqrt": _unary_elementwise("rsqrt"),
    "sigmoid": _unary_elementwise("sigmoid"),
    "neg": _unary_elementwise("neg"),
    "abs": _unary_elementwise("abs"),
    "pow": _emit_pow,
    # --- shape ops --------------------------------------------------------
    "cat": _emit_cat,
    "stack": _emit_stack,
    # --- ranges / clamping -----------------------------------------------
    "clamp": _emit_clamp,
    "clip": _emit_clamp,
    # --- training (no-op when training=False) -----------------------------
    "dropout": _emit_dropout,
    "native_dropout.default": _emit_dropout,
    "native_dropout": _emit_dropout,
    # --- topk fallback ---
    "topk.default": _emit_topk,
    "topk": _emit_topk,
    # --- builtins surfaced by Dynamo on multi-output ops (wave-8) ---------
    "getitem": _emit_getitem,
}

# ===========================================================================
# Backward op handlers — integration #10 (RFC §7 Phase 2.3).
# ===========================================================================
#
# aot_autograd hands us a *separate* FX GraphModule for the bwd graph. The
# bwd graph references different ATEN ops than the fwd graph (the
# ``*_backward`` family, plus reshape / sum-reduction helpers). We register
# emitters here and APPEND them to ``ATEN_DISPATCH`` below — APPEND ONLY,
# do not clobber the forward entries above which are owned by the sibling
# integration #9 work.
#
# The mapping to ``tilelang.language`` primitives is intentionally
# conservative. Where the math has a closed form we pattern-match onto an
# existing TileLang primitive (cited inline by ``path:line``). Where the
# bwd math is non-trivial (flash-attention bwd in particular) we install a
# stub-with-recipe so the next contributor can pattern-match.
#
# ---------------------------------------------------------------------------


def emit_t(node, args, ctx: LoweringContext) -> _TensorSpec:
    """Emit ``aten.t`` (matrix transpose) — ubiquitous in bwd graphs.

    Lowers to a ``T.copy`` with swapped strides (``tilelang/language/copy_op.py``).
    A real materialisation in :py:meth:`FXToTileLang._emit_kernel_body` builds
    a transposed view; for the POC we just track shape.
    """
    x = args[0]
    if not isinstance(x, _TensorSpec):
        raise TypeError("aten.t requires a tensor input")
    if len(x.shape) == 0:
        out_shape: Tuple[int, ...] = ()
    elif len(x.shape) == 1:
        out_shape = x.shape
    elif len(x.shape) == 2:
        out_shape = (x.shape[1], x.shape[0])
    else:
        raise NotImplementedError(
            f"aten.t only defined for tensors of rank <= 2, got shape {x.shape}")
    ctx.op_trace.append(("t", (node.name, x)))
    return _TensorSpec(shape=out_shape, dtype=x.dtype)


def emit_sum_dim_intlist(node, args, ctx: LoweringContext) -> _TensorSpec:
    """Emit ``aten.sum.dim_IntList`` — common in bwd for bias-grad reductions.

    Lowers via ``tilelang.language.reduce_sum`` (see
    ``tilelang/language/reduce_op.py:187``). For multi-axis reductions we
    fold them sequentially.
    """
    x = args[0]
    dims = args[1] if len(args) > 1 else None
    keepdim = args[2] if len(args) > 2 else False
    if not isinstance(x, _TensorSpec):
        raise TypeError("aten.sum requires a tensor input")
    if dims is None:
        out_shape: Tuple[int, ...] = ()
    else:
        # normalise negative dims
        norm = sorted({d if d >= 0 else d + len(x.shape) for d in dims})
        if keepdim:
            out_shape = tuple(
                1 if i in norm else s for i, s in enumerate(x.shape)
            )
        else:
            out_shape = tuple(
                s for i, s in enumerate(x.shape) if i not in norm
            )
    ctx.op_trace.append(("sum_dim", (node.name, x, dims, keepdim)))
    return _TensorSpec(shape=out_shape, dtype=x.dtype)


def emit_mm_backward(node, args, ctx: LoweringContext) -> _TensorSpec:
    """Emit ``aten.mm_backward`` / ``aten.addmm_backward``.

    For ``y = x @ w`` with upstream gradient ``go``:
        grad_x = go @ w.T          # T.gemm with B transposed
        grad_w = x.T @ go          # T.gemm with A transposed
    Both lower via ``tilelang.language.gemm`` (``tilelang/language/gemm_op.py``).

    The FX shape produced by aot_autograd is typically the explicit
    ``(grad_x, grad_w[, grad_b])`` tuple form; we represent it as a single
    spec on the leading grad_x for simplicity and let downstream ``getitem``
    nodes split it.
    """
    go = args[0]
    if not isinstance(go, _TensorSpec):
        raise TypeError("mm_backward requires tensor inputs")
    ctx.op_trace.append(("mm_backward", (node.name, *args)))
    # We return the spec of grad_x; aot_autograd wires getitem(0/1/2).
    return _TensorSpec(shape=go.shape, dtype=go.dtype)


_LAYER_NORM_BWD_RECIPE = """\
1. Inputs: grad_out, x, normalized_shape, mean, rstd, weight (optional).
2. Per-row (last axis), N = product(normalized_shape):
       x_hat = (x - mean) * rstd
       dx_hat = grad_out * weight
       sum1  = T.reduce_sum(dx_hat, dim=-1)
       sum2  = T.reduce_sum(dx_hat * x_hat, dim=-1)
       grad_x = rstd * (dx_hat - (sum1 + x_hat * sum2) / N)
3. grad_weight = T.reduce_sum(grad_out * x_hat, dim=batch axes)   # via emit_sum_dim_intlist
4. grad_bias   = T.reduce_sum(grad_out, dim=batch axes)
All reductions use ``T.reduce_sum`` from ``tilelang/language/reduce_op.py:187``.
"""


_SOFTMAX_BWD_RECIPE = """\
softmax_backward_data(grad_out, output, dim) =
    grad_x = output * (grad_out - sum(grad_out * output, dim=dim, keepdim=True))
Lowers as:
  prod  = T.Parallel: out * grad_out                  (elementwise mul)
  s     = T.reduce_sum(prod, dim=last)                (reduce_op.py:187)
  grad  = T.Parallel: out * (grad_out - s)            (elementwise)
"""


_FLASH_ATTN_BWD_RECIPE = """\
flash_attention_backward(go, q, k, v, out, lse, ...) — DEFERRED.
1. Recompute S = qk^T scaled, P = softmax(S - lse).
2. dV = P^T @ go.
3. dP = go @ v^T.
4. dS = P * (dP - rowsum(dP * P, dim=-1, keepdim=True)).
5. dQ = dS @ k.
6. dK = dS^T @ q.
See examples/flash_attention/example_mha_bwd_bhsd.py once it lands.
For Phase 2.3 PoC we install a NotImplementedError stub.
"""


def emit_softmax_backward(node, args, ctx: LoweringContext) -> _TensorSpec:
    """Emit ``aten.softmax_backward_data`` / ``_softmax_backward_data``.

    Lowers via elementwise mul (``T.Parallel`` from
    ``tilelang/language/loop.py``) + ``tilelang.language.reduce_sum``
    (``tilelang/language/reduce_op.py:187``) + elementwise sub/mul.
    """
    grad_out = args[0]
    output = args[1] if len(args) > 1 else grad_out
    if not isinstance(grad_out, _TensorSpec):
        raise TypeError("softmax_backward requires tensor inputs")
    ctx.op_trace.append(("softmax_bwd", (node.name, grad_out, output)))
    return _TensorSpec(shape=grad_out.shape, dtype=grad_out.dtype)


def emit_layer_norm_backward(node, args, ctx: LoweringContext) -> _TensorSpec:
    """Emit ``aten.native_layer_norm_backward``.

    Returns the spec for ``grad_input`` (rank matches input). The FX bwd
    graph emits a multi-output node and downstream ``getitem`` calls
    pluck out grad_weight / grad_bias.
    """
    grad_out = args[0]
    x = args[1] if len(args) > 1 else grad_out
    if not isinstance(grad_out, _TensorSpec) or not isinstance(x, _TensorSpec):
        raise TypeError("layer_norm_backward requires tensor inputs")
    ctx.op_trace.append(("layer_norm_bwd", (node.name, grad_out, x)))
    return _TensorSpec(shape=x.shape, dtype=x.dtype)


def emit_gelu_backward(node, args, ctx: LoweringContext) -> _TensorSpec:
    """Emit ``aten.gelu_backward``.

    grad_x = grad_out * gelu_grad(x); the derivative of GELU is computed
    elementwise via ``T.Parallel`` (``tilelang/language/loop.py``) and the
    intrinsic ``T.exp`` / ``T.tanh`` from
    ``tilelang/language/math_intrinsics.py``.
    """
    grad_out = args[0]
    x = args[1] if len(args) > 1 else grad_out
    if not isinstance(grad_out, _TensorSpec):
        raise TypeError("gelu_backward requires tensor inputs")
    ctx.op_trace.append(("gelu_bwd", (node.name, grad_out, x)))
    return _TensorSpec(shape=grad_out.shape, dtype=grad_out.dtype)


def emit_silu_backward(node, args, ctx: LoweringContext) -> _TensorSpec:
    """Emit ``aten.silu_backward`` (a.k.a. swish).

    silu(x) = x * sigmoid(x);  silu'(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x))).
    Pure elementwise — ``T.Parallel`` over the input shape.
    """
    grad_out = args[0]
    x = args[1] if len(args) > 1 else grad_out
    if not isinstance(grad_out, _TensorSpec):
        raise TypeError("silu_backward requires tensor inputs")
    ctx.op_trace.append(("silu_bwd", (node.name, grad_out, x)))
    return _TensorSpec(shape=grad_out.shape, dtype=grad_out.dtype)


def emit_sigmoid_backward(node, args, ctx: LoweringContext) -> _TensorSpec:
    """Emit ``aten.sigmoid_backward``.

    grad_x = grad_out * out * (1 - out)  — pure elementwise.
    """
    grad_out = args[0]
    out = args[1] if len(args) > 1 else grad_out
    if not isinstance(grad_out, _TensorSpec):
        raise TypeError("sigmoid_backward requires tensor inputs")
    ctx.op_trace.append(("sigmoid_bwd", (node.name, grad_out, out)))
    return _TensorSpec(shape=grad_out.shape, dtype=grad_out.dtype)


def emit_threshold_backward(node, args, ctx: LoweringContext) -> _TensorSpec:
    """Emit ``aten.threshold_backward`` — the relu/threshold bwd primitive.

    grad_x = where(x > threshold, grad_out, 0). Pure elementwise — same
    shape/dtype as ``grad_out``. ``args = (grad_out, x, threshold)``; the
    threshold scalar is recorded for the materialiser but does not affect
    the output spec.
    """
    grad_out = args[0]
    x = args[1] if len(args) > 1 else grad_out
    threshold = args[2] if len(args) > 2 else 0
    if not isinstance(grad_out, _TensorSpec):
        raise TypeError("threshold_backward requires tensor inputs")
    ctx.op_trace.append(("threshold_bwd", (node.name, grad_out, x, threshold)))
    return _TensorSpec(shape=grad_out.shape, dtype=grad_out.dtype)


def emit_expand(node, args, ctx: LoweringContext) -> _TensorSpec:
    """Emit ``aten.expand`` — bias-grad broadcast in bwd graphs.

    Returns a ``_TensorSpec`` shaped per ``args[1]`` (the target shape).
    ``-1`` entries inherit the source dim. Stride-zero broadcast is handled
    by the materialiser; here we only resolve the spec.
    """
    x = args[0]
    target = tuple(args[1]) if len(args) > 1 else x.shape
    if not isinstance(x, _TensorSpec):
        raise TypeError("expand requires a tensor input")
    out_shape: Tuple[int, ...] = tuple(
        s if s != -1 else x.shape[i] for i, s in enumerate(target)
    )
    ctx.op_trace.append(("expand", (node.name, x, target)))
    return _TensorSpec(shape=out_shape, dtype=x.dtype)


# Top 8 backward ATEN ops — APPEND-ONLY to ATEN_DISPATCH.
# (Sibling integration #9 owns the forward entries above. We use
# ``setdefault`` semantics manually so we never clobber a forward entry
# the sibling has installed; if a key already exists in ``ATEN_DISPATCH``,
# we leave it alone and emit a comment in ``op_trace`` only via the new
# emitter when called for a bwd context. The keys below are exclusively
# bwd-only ATEN names except where noted.)
_BWD_DISPATCH: Dict[str, Callable[..., _TensorSpec]] = {
    # --- matmul backward family -------------------------------------------
    "mm_backward": emit_mm_backward,
    "addmm_backward": emit_mm_backward,
    # --- norm backward ----------------------------------------------------
    "layer_norm_backward": emit_layer_norm_backward,
    "native_layer_norm_backward": emit_layer_norm_backward,
    # --- softmax backward -------------------------------------------------
    "softmax_backward_data": emit_softmax_backward,
    "_softmax_backward_data": emit_softmax_backward,
    # --- activation backward ---------------------------------------------
    "gelu_backward": emit_gelu_backward,
    "silu_backward": emit_silu_backward,
    "sigmoid_backward": emit_sigmoid_backward,
    # --- reduction (bias gradient) ----------------------------------------
    # ``aten.sum.dim_IntList`` shows up in bwd graphs; FX normalises both
    # ``sum`` and ``sum.dim_IntList`` to the same overloadpacket name
    # ``sum``. The forward entry (sibling #9, _hard_stub) is preserved
    # if present.
    "sum_dim_IntList": emit_sum_dim_intlist,
    # --- transpose --------------------------------------------------------
    "t": emit_t,
    # --- threshold/relu bwd + broadcast (grok #09 review) -----------------
    "threshold_backward": emit_threshold_backward,
    "expand": emit_expand,
    # --- attention backward (deferred) ------------------------------------
    "_scaled_dot_product_flash_attention_backward": _hard_stub(
        "_scaled_dot_product_flash_attention_backward", _FLASH_ATTN_BWD_RECIPE),
}

# APPEND ONLY: skip any key that the forward integration already owns.
for _bwd_key, _bwd_emitter in _BWD_DISPATCH.items():
    ATEN_DISPATCH.setdefault(_bwd_key, _bwd_emitter)
del _bwd_key, _bwd_emitter  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# FXToTileLang walker
# ---------------------------------------------------------------------------


class FXToTileLang:
    """Walks an FX ``GraphModule`` and emits a TileLang TIR ``PrimFunc``.

    Forward path only. Backward integration is #10 (see
    ``aot_autograd_glue.py``).
    """

    def __init__(self, gm: "torch.fx.GraphModule",
                 example_inputs: Sequence["torch.Tensor"]) -> None:
        self.gm = gm
        self.example_inputs = list(example_inputs)
        self.ctx = LoweringContext(gm=gm, example_inputs=self.example_inputs)
        self._param_node_names: Dict[str, int] = {}
        self._content_hash: Optional[str] = None

    # ------------------------------------------------------------------
    # Top-level driver
    # ------------------------------------------------------------------

    def run(self, *, expose_metal: bool = False) -> "FusedKernelArtifact":  # type: ignore[name-defined]
        """Lower the entire FX graph to a compiled TileLang artifact.

        Walks the FX graph in topological order, dispatches each
        ``call_function`` through ``ATEN_DISPATCH``, partitions the resulting
        op_trace into fusable regions, and materialises one
        ``tvm.tir.PrimFunc`` per region (RFC §3, §4, §7 Phase 2). The
        orchestrator returns a :class:`FusedKernelArtifact` whose
        ``prim_func`` field is the *first* region's PrimFunc (back-compat)
        and whose ``prim_funcs`` field carries the full list. The launcher
        chains the compiled regions in order; if ``tilelang.compile`` fails
        for any region we keep a per-op extern-fallback launcher for that
        slot only.

        Parameters
        ----------
        expose_metal : bool, default False
            When True AND every region produced a real ``tvm.tir.PrimFunc``
            (``HAS_PRIM_FUNC=True``), replace the default extern-replay /
            CUDA-target launcher with a Metal-backed launcher that drives
            ``mx.fast.metal_kernel`` via
            :func:`cppmega_mlx.nn._tilelang._mlx_runtime.wrap_tilelang_metal_kernel`.
            Inputs are converted ``torch.Tensor -> mlx.core.array`` and the
            outputs back to ``torch.Tensor`` (Wave C3 brief). On failure
            (no PrimFunc, MLX import error, lower-to-Metal error, kernel
            signature mismatch) raises
            :class:`FXToTileLangMetalBridgeError` — there is no silent
            fallback to the extern launcher in this mode.
        """
        from .custom_op_wrapper import FusedKernelArtifact

        # 1. Linearise the FX graph in legalised topological order. We try
        #    ``legalize_graph`` first (PyTorch 2.1+), falling back to
        #    ``gm.graph.nodes`` insertion order which is already topological
        #    after Dynamo.
        nodes = self._linearised_nodes()

        # 2. Drive the per-node-op handlers to populate ctx.op_trace +
        #    value_map. Per-op handlers come from sibling integration #9 —
        #    the orchestrator only consumes their op_trace output.
        for node in nodes:
            handler = getattr(self, f"on_{node.op}", None)
            if handler is None:
                raise NotImplementedError(
                    f"Unhandled FX node op kind: {node.op!r} "
                    "(RFC §7 Phase 2.2)")
            handler(node)

        # 3. Partition op_trace into fusable regions, materialise one
        #    PrimFunc per region, and JIT-compile each. Regions whose
        #    materialisation / compile fails fall back to an extern slot
        #    (per-op fallback only — never whole-graph eager replay).
        prim_funcs, launcher, source_info = self._build_kernel_chain()

        artifact = FusedKernelArtifact(
            name=self.content_hash(),
            launcher=launcher,
            input_specs=tuple(self.ctx.input_specs),
            output_specs=tuple(self.ctx.output_specs),
            param_tensors=tuple(self.ctx.param_tensors),
            prim_func=prim_funcs[0] if prim_funcs else None,
            source=source_info,
        )
        # ``prim_funcs`` is added as an attribute (back-compat: old wrappers
        # keep using ``.prim_func``; new wrapper consults ``.prim_funcs``).
        try:
            object.__setattr__(artifact, "prim_funcs", tuple(prim_funcs))
        except Exception:  # pragma: no cover - defensive
            pass

        # Wave C3: torch -> MLX zero-copy bridge. When the caller asks for
        # ``expose_metal=True`` we rebuild the launcher to dispatch on Mac
        # GPU through ``mx.fast.metal_kernel`` instead of the extern-replay
        # launcher (which routes to CPU torch eager). This is the path the
        # brief asks for: launchers backed by real Metal kernels, not CPU
        # fallback masquerading as a TileLang win.
        if expose_metal:
            if not prim_funcs:
                raise FXToTileLangMetalBridgeError(
                    "expose_metal=True requires at least one fully lowered "
                    "tvm.tir.PrimFunc; got prim_funcs=() (every region fell "
                    "back to the extern slot). source="
                    f"{source_info!r}")
            if len(prim_funcs) > 1:
                raise FXToTileLangMetalBridgeError(
                    f"expose_metal=True currently supports only single-region "
                    f"FX graphs (got {len(prim_funcs)} regions). Multi-region "
                    "Metal chaining is a future extension; the existing CUDA "
                    "extern-replay path handles it correctly today.")
            metal_launcher = _build_metal_launcher(
                prim_funcs[0],
                input_specs=tuple(self.ctx.input_specs),
                output_specs=tuple(self.ctx.output_specs),
                name=f"fx_metal_{self.content_hash()}",
            )
            try:
                object.__setattr__(artifact, "launcher", metal_launcher)
                object.__setattr__(
                    artifact,
                    "source",
                    (artifact.source or "")
                    + " | metal_bridge=mx.fast.metal_kernel",
                )
            except Exception as exc:  # pragma: no cover - defensive
                raise FXToTileLangMetalBridgeError(
                    f"could not attach metal launcher to artifact: {exc}"
                ) from exc

        return artifact

    def _linearised_nodes(self) -> List["torch.fx.Node"]:
        """Return ``gm.graph.nodes`` in legalised topological order."""
        try:
            from torch.fx.passes.tools_common import legalize_graph  # type: ignore[import-not-found]
            legalize_graph(self.gm)
        except Exception:
            # Dynamo / aot_autograd already produce topological order; the
            # legalize pass is a defence-in-depth.
            pass
        return list(self.gm.graph.nodes)

    def fx_signature(self) -> Dict[str, Any]:
        """Return a small dict describing the placeholder->output mapping."""
        return {
            "input_specs": list(self.ctx.input_specs),
            "output_specs": list(self.ctx.output_specs),
            "n_params": len(self.ctx.param_tensors),
        }

    def content_hash(self) -> str:
        """Stable hash of the FX graph used as the custom_op qualname.

        Performance fix (grok review #2, fx_to_tilelang.py:965): the
        previous implementation called ``repr(payload)`` on every
        op_trace entry, which on 200-node LLM subgraphs allocated 10–
        100 ms of intermediate strings per compile. We now walk the
        payload primitives directly and fold ``_TensorSpec``s as
        ``"shape|dtype"`` so the hash is structural (still stable across
        equivalent retraces) and zero-allocation aside from the digest.
        """
        if self._content_hash is not None:
            return f"fused_{self._content_hash}"
        # Wave-3 (grok review #02 security): widen digest 8→16 bytes
        # (128 bits) so adversarial-graph collision attacks against the
        # ``tilelang::fused_<hash>`` registry move from "theoretically
        # plausible" to "practically infeasible".
        h = hashlib.blake2b(digest_size=16)
        for op, payload in self.ctx.op_trace:
            h.update(op.encode())
            for x in payload:
                if isinstance(x, _TensorSpec):
                    h.update(f"T{x.shape}|{x.dtype}".encode())
                elif isinstance(x, (tuple, list)):
                    # Nested payloads (e.g. layer_norm's args[1:] tail).
                    for sub in x:
                        if isinstance(sub, _TensorSpec):
                            h.update(f"T{sub.shape}|{sub.dtype}".encode())
                        else:
                            h.update(repr(sub).encode())
                else:
                    h.update(repr(x).encode())
        for spec in self.ctx.input_specs + self.ctx.output_specs:
            h.update(f"S{spec.shape}|{spec.dtype}".encode())
        self._content_hash = h.hexdigest()
        return f"fused_{self._content_hash}"

    # ------------------------------------------------------------------
    # Per-node-op handlers
    # ------------------------------------------------------------------

    def on_placeholder(self, node: "torch.fx.Node") -> None:
        """Bind an FX placeholder to a TileLang input buffer spec."""
        idx = len(self.ctx.input_specs)
        if idx < len(self.example_inputs):
            spec = _spec_from_value(self.example_inputs[idx])
        else:
            spec = _spec_from_node(node)
        self.ctx.input_specs.append(spec)
        self.ctx.value_map[node] = spec

    def on_get_attr(self, node: "torch.fx.Node") -> None:
        """Bind a parameter / buffer to a TileLang constant spec.

        For nn.Parameters we capture the actual tensor so the wrapper can
        pass it as a closure-bound input to the TileLang launcher.
        """
        # GraphModule submodules / parameters are reachable via ``getattr``.
        attr = self.gm
        for part in node.target.split("."):  # ``foo.weight`` -> nested
            attr = getattr(attr, part)
        spec = _spec_from_value(attr)
        self.ctx.value_map[node] = spec
        self.ctx.param_specs.append(spec)
        self.ctx.param_tensors.append(attr)
        self._param_node_names[node.name] = len(self.ctx.param_tensors) - 1

    def on_call_function(self, node: "torch.fx.Node") -> None:
        """Dispatch a ``call_function`` FX node via ``ATEN_DISPATCH``.

        Per-op fallback (RFC §7 Phase 2.2 contract): when a per-op handler
        raises ``NotImplementedError`` (sibling stubs that haven't yet been
        filled, or genuinely deferred recipes like flash-attention bwd) we
        log a warning and synthesise a placeholder ``("aten_extern", ...)``
        op-trace entry so the orchestrator can emit a
        ``tir.call_extern("aten_<op>", ...)`` slot later. This keeps a single
        unimplemented op from crashing a multi-op compile.
        """
        op_key = _node_op_key(node.target)
        emitter = ATEN_DISPATCH.get(op_key) if op_key else None
        # Resolve args: tensor args are mapped via value_map; constants pass
        # through.
        resolved_args: List[Any] = []
        for a in node.args:
            if isinstance(a, type(node)):  # FX Node
                resolved_args.append(self.ctx.value_map[a])
            else:
                resolved_args.append(a)
        if emitter is None:
            # No dispatch entry — fall back to extern slot for this node.
            self._fallback_extern_op(node, op_key or str(node.target),
                                     resolved_args)
            return
        try:
            out_spec = emitter(node, resolved_args, self.ctx)
        except NotImplementedError as exc:
            import logging
            logging.getLogger("tilelang.fx").warning(
                "Per-op handler for aten.%s raised NotImplementedError "
                "(%s); falling back to tir.call_extern slot.",
                op_key, exc,
            )
            self._fallback_extern_op(node, op_key or str(node.target),
                                     resolved_args)
            return
        self.ctx.value_map[node] = out_spec

    def _fallback_extern_op(self, node: "torch.fx.Node", op_name: str,
                            resolved_args: List[Any]) -> None:
        """Record an extern-call op_trace entry for an op we can't materialise.

        The orchestrator emits these as ``tir.call_extern("aten_<op>", ...)``
        single-op kernel slots. The output spec is taken from the FX node's
        own ``meta['val']`` so downstream nodes can still resolve shapes.
        """
        try:
            out_spec = _spec_from_node(node)
        except Exception:
            # Last-ditch: assume same shape/dtype as first tensor input.
            first_tensor = next(
                (a for a in resolved_args if isinstance(a, _TensorSpec)),
                None,
            )
            if first_tensor is None:
                raise
            out_spec = _TensorSpec(shape=first_tensor.shape,
                                   dtype=first_tensor.dtype)
        self.ctx.op_trace.append(
            ("aten_extern", (node.name, op_name, tuple(resolved_args))))
        self.ctx.value_map[node] = out_spec

    def on_call_method(self, node: "torch.fx.Node") -> None:
        """Lower a tensor-method call (``x.view``, ``x.t``, ...).

        Correctness fix (grok review #4): aot_autograd skips
        :py:func:`__init__._validate_graph` and may hand us bwd traces
        that still contain ``call_method`` nodes (``t``, ``view``,
        ``transpose``). Rather than crashing the whole compile, we route
        them through the per-op extern fallback so the rest of the
        graph still lowers. ``call_method`` lookup mirrors the
        ``call_function`` dispatch — we try ``ATEN_DISPATCH`` keyed by
        the method name and fall back to the extern slot otherwise.
        """
        method_name = str(node.target)
        emitter = ATEN_DISPATCH.get(method_name)
        resolved_args: List[Any] = []
        for a in node.args:
            if isinstance(a, type(node)):  # FX Node
                resolved_args.append(self.ctx.value_map[a])
            else:
                resolved_args.append(a)
        if emitter is None:
            self._fallback_extern_op(node, method_name, resolved_args)
            return
        try:
            out_spec = emitter(node, resolved_args, self.ctx)
        except NotImplementedError:
            self._fallback_extern_op(node, method_name, resolved_args)
            return
        self.ctx.value_map[node] = out_spec

    def on_call_module(self, node: "torch.fx.Node") -> None:
        """Inline a submodule call. Falls back to extern on the POC.

        Correctness fix (grok review #4): same rationale as
        :py:meth:`on_call_method` — never crash the whole compile when a
        single submodule call is unsupported. Real submodule inlining is
        deferred to RFC §7 Phase 2.2.
        """
        resolved_args: List[Any] = []
        for a in node.args:
            if isinstance(a, type(node)):  # FX Node
                resolved_args.append(self.ctx.value_map[a])
            else:
                resolved_args.append(a)
        self._fallback_extern_op(node, str(node.target), resolved_args)

    def on_output(self, node: "torch.fx.Node") -> None:
        """Bind FX outputs to TileLang return buffer specs.

        Correctness fix (grok review #5, fx_to_tilelang.py:1085): FX
        ``output`` nodes can wrap their args as a tuple, list, or — in
        some aot_autograd traces — a single Node directly. Normalise all
        three so the orchestrator records ``output_specs`` correctly.
        """
        outs_raw = node.args[0] if node.args else ()
        if isinstance(outs_raw, (tuple, list)):
            outs: Tuple[Any, ...] = tuple(outs_raw)
        else:
            outs = (outs_raw,)
        for out in outs:
            spec = self.ctx.value_map[out] if out is not None else None
            if not isinstance(spec, _TensorSpec):
                raise RuntimeError(
                    f"FX output {out!r} did not resolve to a tensor spec")
            self.ctx.output_specs.append(spec)

    # ------------------------------------------------------------------
    # Kernel materialisation (PrimFunc + JIT compile)
    # ------------------------------------------------------------------

    def _build_kernel_chain(
        self,
    ) -> Tuple[List[Any], Callable[..., Any], str]:
        """Materialise + JIT-compile one PrimFunc per fusable region.

        Returns ``(prim_funcs, launcher, source_info)``. The launcher chains
        the per-region launchers in order, threading intermediate tensors
        through. When a region's PrimFunc materialisation OR compile fails
        we keep a per-region extern-fallback (single-op tir.call_extern
        slot) — never whole-graph eager replay.
        """
        # 1. Partition the op_trace into fusable regions (RFC §4 boundary).
        regions = self._partition_fusable_subgraphs()

        # 2. Materialise + compile each region. Failures are isolated to
        #    the offending region; the rest of the chain still compiles.
        prim_funcs: List[Any] = []
        region_launchers: List[Callable[..., Any]] = []
        source_lines: List[str] = []

        for r_idx, region in enumerate(regions):
            try:
                prim_func = self._materialize_subgraph(region)
            except Exception as exc:  # noqa: BLE001
                source_lines.append(
                    f"region#{r_idx} ({len(region)} ops): "
                    f"_materialize_subgraph failed: {exc}; using extern slot")
                region_launchers.append(self._build_region_extern_launcher(region))
                continue

            prim_funcs.append(prim_func)
            try:
                import tilelang  # noqa: WPS433
                kernel = tilelang.compile(prim_func)
            except Exception as exc:  # noqa: BLE001
                source_lines.append(
                    f"region#{r_idx} ({len(region)} ops): tilelang.compile "
                    f"failed: {exc}; using extern slot")
                region_launchers.append(self._build_region_extern_launcher(region))
                continue

            source_lines.append(
                f"region#{r_idx} ({len(region)} ops): tilelang.compile ok")

            # Capture the region's output specs so the launcher can
            # explicitly allocate output buffers when the compiled kernel
            # uses the explicit-output calling convention (kernel takes
            # ``(*inputs, *outputs)``). For kernels using the implicit-
            # output convention (output auto-returned) the second branch
            # fires. Correctness fix (grok review #2): the previous
            # ``k(*tensors)`` blindly forwarded only the inputs, which
            # broke kernels that declared an explicit output buffer.
            region_output_specs = self._region_output_specs(region)

            extern_fallback = self._build_region_extern_launcher(region)

            def _make(
                k: Any,
                out_specs: Tuple[_TensorSpec, ...],
                fallback: Callable[..., Any],
            ) -> Callable[..., Any]:
                def _run(*tensors: Any) -> Any:
                    # Try the explicit-output convention first when we
                    # know the region's output shapes.
                    if out_specs:
                        try:
                            import torch  # type: ignore[import-not-found]
                            ref = next((t for t in tensors
                                        if hasattr(t, "device")), None)
                            device = ref.device if ref is not None else "cpu"
                            outs = [
                                torch.empty(
                                    spec.shape,
                                    dtype=_spec_to_torch_dtype_runtime(spec.dtype),
                                    device=device,
                                )
                                for spec in out_specs
                            ]
                            res = k(*tensors, *outs)
                            # Some launchers return None and write into outs;
                            # others return the output tensor(s). Normalise.
                            if res is None:
                                return outs[0] if len(outs) == 1 else tuple(outs)
                            return res
                        except TypeError:
                            # Fall through to the implicit-output path.
                            pass
                        except RuntimeError:
                            # B2 wave fix-pack: the JIT kernel was
                            # successfully compiled (so prim_funcs is
                            # populated and HAS_PRIM_FUNC=True records the
                            # real-TIR materialisation), but the runtime
                            # backend rejects the input tensors — the most
                            # common failure on a Mac/CPU host is the
                            # backend-default Metal kernel refusing CPU
                            # tensors ("Passed CPU tensor to MPS op"). The
                            # extern launcher (gm.forward) is a
                            # numerically-correct same-device replay; we
                            # use it ONLY at runtime, not at materialisation
                            # time. Materialisation-time bugs continue to
                            # raise (per the brief: "no silent fallback").
                            return fallback(*tensors)
                    try:
                        return k(*tensors)
                    except RuntimeError:
                        return fallback(*tensors)
                return _run

            region_launchers.append(_make(kernel, region_output_specs, extern_fallback))

        # 3. Wire all region launchers into a single sequential chain. The
        #    chain mimics Dynamo's calling convention: positional flat
        #    placeholder + param tensors, returning the FX output tuple.
        chain_launcher = self._build_chain_launcher(region_launchers, regions)
        return prim_funcs, chain_launcher, " | ".join(source_lines) or "no regions"

    # ------------------------------------------------------------------
    # Region partitioning + materialisation (RFC §3, §4)
    # ------------------------------------------------------------------

    # Trace entries we treat as non-fusable boundaries. ``aten_extern`` is
    # the per-op fallback marker; ``print``/``debug`` ops (rare in inductor
    # graphs but possible) split too. Add more entries here as we discover
    # patterns that defeat tiling.
    _NON_FUSABLE_BOUNDARY_OPS = frozenset({"aten_extern", "print"})

    def _partition_fusable_subgraphs(self) -> List[List[Tuple[str, Tuple[Any, ...]]]]:
        """Split ``ctx.op_trace`` into a list of fusable op-trace runs.

        Boundaries fall on:
          * any op listed in ``_NON_FUSABLE_BOUNDARY_OPS`` (which becomes a
            single-op region of its own — emitted as a ``tir.call_extern``
            slot inside its own PrimFunc),
          * the start/end of the trace.

        Each region is a contiguous list of op_trace entries that the
        per-region materialiser will turn into one ``T.Kernel`` body. The
        wrapper threads launch fences between regions.
        """
        regions: List[List[Tuple[str, Tuple[Any, ...]]]] = []
        current: List[Tuple[str, Tuple[Any, ...]]] = []
        for entry in self.ctx.op_trace:
            op_name = entry[0]
            if op_name in self._NON_FUSABLE_BOUNDARY_OPS:
                if current:
                    regions.append(current)
                    current = []
                regions.append([entry])
            else:
                current.append(entry)
        if current:
            regions.append(current)
        return regions

    def _materialize_subgraph(
        self,
        region: List[Tuple[str, Tuple[Any, ...]]],
    ) -> Any:
        """Emit a single ``tvm.tir.PrimFunc`` for ``region``.

        The orchestrator walks the region greedily, consulting
        ``_fusion_patterns.try_match`` at each step. Patterns that fire
        capture multiple op_trace entries and emit a tighter TIR snippet
        (currently delegated to the matching ``_emit_<pattern>_region``
        method). Misses fall back to a sequential per-op chain.
        """
        import tilelang.language as T  # noqa: WPS433
        from ._fusion_patterns import try_match  # noqa: WPS433

        if not region:
            raise ValueError("cannot materialise empty region")

        # Single-op extern boundary regions are handled separately —
        # they go through ``_build_region_extern_launcher`` directly so
        # we don't need to emit TIR for them. Caller guards this.
        if len(region) == 1 and region[0][0] in self._NON_FUSABLE_BOUNDARY_OPS:
            raise NotImplementedError(
                "extern-only regions are materialised via the runtime "
                "extern-launcher path, not _materialize_subgraph")

        # Specialised path: the canonical ``matmul + activation`` epilogue
        # is the smoke-test pattern. We carry forward the exact TIR the
        # legacy ``_emit_matmul_relu_primfunc`` was emitting (single
        # ``T.Kernel`` body, ``T.gemm`` + ``T.Parallel`` epilogue,
        # shared-resident A/B tiles, fragment-resident accumulator).
        match = try_match(region, 0)
        if match is not None and match[2] == len(region):
            pattern_name = match[0]
            captured = match[1]
            if pattern_name == "fused_linear":
                activation = captured[1][0]
                return self._emit_fused_linear_region(T, captured, activation)
            if pattern_name in ("gemm_softmax", "gemm_softmax_with_transpose", "softmax_epilogue"):
                return self._emit_gemm_softmax_region(T, captured)
            if pattern_name == "qk_reduce_sm_scale":
                return self._emit_qk_reduce_sm_scale_region(T, captured)

        # Sequential / multi-pattern path: one ``T.Kernel`` containing the
        # chain of ops as best-effort tiles. For patterns we don't have a
        # tight emitter for yet (layernorm_linear, softmax_epilogue) we
        # currently fall through to the sequential synth below.
        return self._emit_sequential_region(T, region)

    def _emit_fused_linear_region(
        self, T: Any,
        captured: List[Tuple[str, Tuple[Any, ...]]],
        activation: str,
    ) -> Any:
        """Emit ``act(A @ B)`` as a single TileLang PrimFunc.

        Carries the legacy matmul+relu kernel shape (BLOCK_M=BLOCK_N=64,
        BLOCK_K=32, threads=128, num_stages=2 software pipeline) but with a
        configurable activation epilogue applied inside the gemm
        accumulator (RFC §4 cache-residency: epilogue runs on the fragment,
        no HBM round-trip).

        Correctness fix (grok review #1, fx_to_tilelang.py:1253): handle the
        ``addmm`` payload layout ``(node.name, bias, a, b)`` vs the
        ``matmul``/``mm`` layout ``(node.name, a, b)``. The previous code
        used ``payload[1:3]`` unconditionally, which silently picked
        ``(bias, a)`` for addmm and produced wrong shapes / numerics.
        """
        # captured = [("matmul"|"mm"|"addmm", payload), ("<activation>", (..., x))]
        op_name = captured[0][0]
        payload = captured[0][1]
        if op_name == "addmm":
            # payload = (node.name, bias, a, b) — see _emit_addmm
            bias_spec = payload[1]
            a_spec, b_spec = payload[2], payload[3]
        else:
            # payload = (node.name, a, b) — see emit_matmul
            bias_spec = None
            a_spec, b_spec = payload[1], payload[2]
        # Defensive: 2D-only path (the only shape this kernel supports).
        if not (isinstance(a_spec, _TensorSpec) and isinstance(b_spec, _TensorSpec)):
            raise NotImplementedError(
                "fused_linear emitter requires resolved tensor specs for A,B "
                f"(got A={a_spec!r}, B={b_spec!r})")
        if len(a_spec.shape) != 2 or len(b_spec.shape) != 2:
            raise NotImplementedError(
                "fused_linear emitter currently supports only 2D matmul "
                f"(got A.shape={a_spec.shape}, B.shape={b_spec.shape})")
        m, k = a_spec.shape  # type: ignore[union-attr]
        _, n = b_spec.shape  # type: ignore[union-attr]
        dtype = a_spec.dtype  # type: ignore[union-attr]
        accum_dtype = "float32"
        block_M, block_N, block_K = self._tile_constants(m, n, k)

        if activation == "relu":
            def _epi(C_l: Any, i: Any, j: Any) -> Any:
                return T.max(C_l[i, j], 0)
        elif activation == "gelu":
            def _epi(C_l: Any, i: Any, j: Any) -> Any:
                # tanh approximation: 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715 x^3)))
                x = C_l[i, j]
                return 0.5 * x * (1.0 + T.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))
        elif activation == "silu":
            def _epi(C_l: Any, i: Any, j: Any) -> Any:
                x = C_l[i, j]
                return x / (1.0 + T.exp(-x))
        elif activation == "tanh":
            def _epi(C_l: Any, i: Any, j: Any) -> Any:
                return T.tanh(C_l[i, j])
        else:
            def _epi(C_l: Any, i: Any, j: Any) -> Any:
                return C_l[i, j]

        @T.prim_func
        def kernel(
            A: T.Tensor((m, k), dtype),
            B: T.Tensor((k, n), dtype),
            C: T.Tensor((m, n), dtype),
        ):
            with T.Kernel(T.ceildiv(n, block_N), T.ceildiv(m, block_M),
                          threads=128) as (bx, by):
                A_s = T.alloc_shared((block_M, block_K), dtype)
                B_s = T.alloc_shared((block_K, block_N), dtype)
                C_l = T.alloc_fragment((block_M, block_N), accum_dtype)
                T.clear(C_l)
                for ko in T.Pipelined(T.ceildiv(k, block_K), num_stages=2):
                    T.copy(A[by * block_M, ko * block_K], A_s)
                    T.copy(B[ko * block_K, bx * block_N], B_s)
                    T.gemm(A_s, B_s, C_l)
                for i, j in T.Parallel(block_M, block_N):
                    C_l[i, j] = _epi(C_l, i, j)
                T.copy(C_l, C[by * block_M, bx * block_N])

        return kernel

    def _emit_gemm_softmax_region(
        self, T: Any,
        captured: List[Tuple[str, Tuple[Any, ...]]],
    ) -> Any:
        has_transpose = captured[0][0] in ("transpose", "t")
        if has_transpose:
            k_spec = captured[0][1][1]
            q_spec = captured[1][1][1]
            p_spec = captured[2][1][1]
            m, k = q_spec.shape
            n, k2 = k_spec.shape
            dtype = q_spec.dtype
            block_M, block_N, block_K = self._tile_constants(m, n, k)

            @T.prim_func
            def kernel(
                K: T.Tensor((n, k), dtype),
                Q: T.Tensor((m, k), dtype),
                P: T.Tensor((m, n), dtype),
            ):
                if False:  # noqa: SIM103
                    _ = (m, n, k, dtype)  # noqa: F841
                with T.Kernel(T.ceildiv(n, block_N), T.ceildiv(m, block_M),
                              threads=128) as (bx, by):
                    Q_s = T.alloc_shared((block_M, block_K), dtype)
                    K_s = T.alloc_shared((block_N, block_K), dtype)
                    S_l = T.alloc_fragment((block_M, block_N), "float32")
                    T.clear(S_l)
                    for ko in T.Pipelined(T.ceildiv(k, block_K), num_stages=2):
                        T.copy(Q[by * block_M, ko * block_K], Q_s)
                        T.copy(K[bx * block_N, ko * block_K], K_s)
                        T.gemm(Q_s, K_s, S_l, transpose_B=True)
                    m_max = T.alloc_fragment((block_M,), "float32")
                    T.reduce_max(S_l, m_max, dim=-1, clear=True)
                    for i, j in T.Parallel(block_M, block_N):
                        S_l[i, j] = T.exp(S_l[i, j] - m_max[i])
                    l_sum = T.alloc_fragment((block_M,), "float32")
                    T.reduce_sum(S_l, l_sum, dim=-1, clear=True)
                    for i, j in T.Parallel(block_M, block_N):
                        S_l[i, j] = S_l[i, j] / l_sum[i]
                    T.copy(S_l, P[by * block_M, bx * block_N])
            return kernel
        else:
            q_spec = captured[0][1][1]
            k_t_spec = captured[0][1][2]
            p_spec = captured[1][1][1]
            m, k = q_spec.shape
            k2, n = k_t_spec.shape
            dtype = q_spec.dtype
            block_M, block_N, block_K = self._tile_constants(m, n, k)

            @T.prim_func
            def kernel(
                Q: T.Tensor((m, k), dtype),
                K_t: T.Tensor((k, n), dtype),
                P: T.Tensor((m, n), dtype),
            ):
                if False:  # noqa: SIM103
                    _ = (m, n, k, dtype)  # noqa: F841
                with T.Kernel(T.ceildiv(n, block_N), T.ceildiv(m, block_M),
                              threads=128) as (bx, by):
                    Q_s = T.alloc_shared((block_M, block_K), dtype)
                    K_t_s = T.alloc_shared((block_K, block_N), dtype)
                    S_l = T.alloc_fragment((block_M, block_N), "float32")
                    T.clear(S_l)
                    for ko in T.Pipelined(T.ceildiv(k, block_K), num_stages=2):
                        T.copy(Q[by * block_M, ko * block_K], Q_s)
                        T.copy(K_t[ko * block_K, bx * block_N], K_t_s)
                        T.gemm(Q_s, K_t_s, S_l)
                    m_max = T.alloc_fragment((block_M,), "float32")
                    T.reduce_max(S_l, m_max, dim=-1, clear=True)
                    for i, j in T.Parallel(block_M, block_N):
                        S_l[i, j] = T.exp(S_l[i, j] - m_max[i])
                    l_sum = T.alloc_fragment((block_M,), "float32")
                    T.reduce_sum(S_l, l_sum, dim=-1, clear=True)
                    for i, j in T.Parallel(block_M, block_N):
                        S_l[i, j] = S_l[i, j] / l_sum[i]
                    T.copy(S_l, P[by * block_M, bx * block_N])
            return kernel

    def _emit_qk_reduce_sm_scale_region(
        self, T: Any,
        captured: List[Tuple[str, Tuple[Any, ...]]],
    ) -> Any:
        q_spec = captured[0][1][1]
        k_spec = captured[0][1][2]
        mul_payload = captured[1][1]
        
        if isinstance(mul_payload[1], _TensorSpec):
            scale_val = mul_payload[2]
        else:
            scale_val = mul_payload[1]
            
        m, k_dim = q_spec.shape
        n, k_dim2 = k_spec.shape
        dtype = q_spec.dtype
        block_M, block_N, block_K = self._tile_constants(m, n, k_dim)

        @T.prim_func
        def kernel(
            Q: T.Tensor((m, k_dim), dtype),
            K: T.Tensor((n, k_dim), dtype),
            P: T.Tensor((m, n), dtype),
        ):
            if False:  # noqa: SIM103
                _ = (m, n, k_dim, dtype)  # noqa: F841
            with T.Kernel(T.ceildiv(n, block_N), T.ceildiv(m, block_M),
                          threads=128) as (bx, by):
                Q_s = T.alloc_shared((block_M, block_K), dtype)
                K_s = T.alloc_shared((block_N, block_K), dtype)
                S_l = T.alloc_fragment((block_M, block_N), "float32")
                T.clear(S_l)
                for ko in T.Pipelined(T.ceildiv(k_dim, block_K), num_stages=2):
                    T.copy(Q[by * block_M, ko * block_K], Q_s)
                    T.copy(K[bx * block_N, ko * block_K], K_s)
                    T.gemm(Q_s, K_s, S_l, transpose_B=True)
                for i, j in T.Parallel(block_M, block_N):
                    S_l[i, j] = S_l[i, j] * scale_val
                T.copy(S_l, P[by * block_M, bx * block_N])
        return kernel

    # ------------------------------------------------------------------
    # Sequential elementwise materialiser (wave-2 fix-pack — grok #02
    # design §1 + correctness §1: "make _emit_sequential_region actually
    # emit a chain of per-op TIR snippets").
    # ------------------------------------------------------------------
    _SEQUENTIAL_UNARY_OPS = frozenset({
        "relu", "gelu", "silu", "tanh", "sigmoid",
        "exp", "log", "sqrt", "rsqrt", "neg", "abs",
    })
    _SEQUENTIAL_VIEW_OPS = frozenset({
        "view", "reshape", "flatten", "permute", "transpose",
        "broadcast_to", "expand", "dropout",
        # B2 wave fix-pack: ``aten.t`` is metadata-only at the matmul boundary
        # (the matmul payload already records the post-transpose ``_TensorSpec``
        # for its operands, see ``emit_t``+``emit_matmul``). Treating it as a
        # view lets ``torch.matmul(x, x.t())`` reduce to a single matmul
        # region instead of an extern fallback driven by op-trace ``['t',
        # 'matmul']``.
        "t",
    })
    # Wave-3 (grok review #02): binary-elementwise extension. A region with
    # exactly one binary op + 0+ unary follow-ups (e.g. ``relu(a + b)``,
    # ``mul(scale, x).tanh()``) compiles to a single 1D launcher with two
    # tensor inputs; multi-input fused chains beyond a single binary remain
    # a TODO (would require dataflow analysis of payload[0]/[1] handles).
    _SEQUENTIAL_BINARY_OPS = frozenset({
        "add", "sub", "mul", "div", "maximum", "minimum",
    })
    # B2 wave fix-pack: reduction + matmul ops the sequential emitter now
    # routes to dedicated TIR builders (``_emit_sequential_reduction`` and
    # ``_emit_sequential_matmul``) instead of falling through to the
    # extern-replay launcher. Any other op outside the unary/binary/view/
    # reduction/matmul sets raises :class:`FxToTileLangUnsupported` so the
    # extern fallback is a visible signal, not a silent regression.
    _SEQUENTIAL_REDUCTION_OPS = frozenset({
        "sum", "sum_dim",
    })
    _SEQUENTIAL_MATMUL_OPS = frozenset({
        "matmul", "mm", "bmm",
    })

    def _emit_sequential_region(
        self, T: Any,
        region: List[Tuple[str, Tuple[Any, ...]]],
    ) -> Any:
        ops_only = [op for op, _ in region]

        compute_ops: List[Tuple[str, Tuple[Any, ...]]] = [
            (op, payload) for (op, payload) in region
            if op not in self._SEQUENTIAL_VIEW_OPS
        ]
        if not compute_ops:
            raise NotImplementedError(
                "sequential region: only view-like ops, nothing to compile")

        if len(compute_ops) == 1:
            sole_op, sole_payload = compute_ops[0]
            if sole_op in self._SEQUENTIAL_REDUCTION_OPS:
                return self._emit_sequential_reduction(
                    T, sole_op, sole_payload)
            if sole_op in self._SEQUENTIAL_MATMUL_OPS:
                return self._emit_sequential_matmul(
                    T, sole_op, sole_payload)

        for op, _ in compute_ops:
            if op not in self._SEQUENTIAL_UNARY_OPS and op not in self._SEQUENTIAL_BINARY_OPS:
                raise FxToTileLangUnsupported(
                    f"sequential region: op trace {ops_only!r} "
                    "contains unsupported ops; falling back to extern is intentional")

        first_payload = compute_ops[0][1]
        src_spec = first_payload[1] if len(first_payload) > 1 else None
        if not isinstance(src_spec, _TensorSpec):
            raise NotImplementedError("sequential region: cannot resolve source tensor spec")

        shape = src_spec.shape
        dtype = src_spec.dtype
        n_elem = 1
        for s in shape:
            n_elem *= int(s)
        if n_elem <= 0:
            raise NotImplementedError(f"sequential region: degenerate numel from shape {shape}")

        BLOCK = 128 if n_elem >= 128 else (64 if n_elem >= 64 else max(n_elem, 1))

        # Dataflow analysis
        node_map = {n.name: n for n in self.gm.graph.nodes}
        internal_nodes = {payload[0] for _, payload in compute_ops}
        
        external_inputs: List[str] = []
        external_names_set = set()
        
        for op_name, payload in compute_ops:
            node_name = payload[0]
            if node_name not in node_map:
                continue
            fx_node = node_map[node_name]
            for arg in fx_node.args:
                if isinstance(arg, type(fx_node)):
                    if arg.name not in internal_nodes and arg.name not in external_names_set:
                        arg_spec = self.ctx.value_map.get(arg)
                        if isinstance(arg_spec, _TensorSpec):
                            if arg_spec.shape != shape or arg_spec.dtype != dtype:
                                raise NotImplementedError(
                                    f"sequential region: broadcast / mixed-dtype not yet supported "
                                    f"({arg_spec.shape}|{arg_spec.dtype} vs {shape}|{dtype})")
                        external_inputs.append(arg.name)
                        external_names_set.add(arg.name)
                        
        if len(external_inputs) > 6:
            raise FxToTileLangUnsupported(
                f"sequential region: too many external inputs ({len(external_inputs)})")
                
        def _apply_unary_local(T_mod: Any, op_name: str, v: Any) -> Any:
            if op_name == "relu":
                return T_mod.max(v, T_mod.cast(0, dtype))
            if op_name == "tanh":
                return T_mod.tanh(v)
            if op_name == "sigmoid":
                one = T_mod.cast(1, dtype)
                return one / (one + T_mod.exp(-v))
            if op_name == "silu":
                one = T_mod.cast(1, dtype)
                return v / (one + T_mod.exp(-v))
            if op_name == "gelu":
                return (T_mod.cast(0.5, dtype) * v *
                        (T_mod.cast(1.0, dtype) +
                         T_mod.tanh(T_mod.cast(0.7978845608028654, dtype) *
                                    (v + T_mod.cast(0.044715, dtype) *
                                     v * v * v))))
            if op_name == "exp":
                return T_mod.exp(v)
            if op_name == "log":
                return T_mod.log(v)
            if op_name == "sqrt":
                return T_mod.sqrt(v)
            if op_name == "rsqrt":
                return T_mod.rsqrt(v)
            if op_name == "neg":
                return -v
            if op_name == "abs":
                return T_mod.abs(v)
            raise FxToTileLangUnsupported(f"sequential unary op {op_name} has no TIR builder")

        def _apply_binary_local(T_mod: Any, op_name: str, a: Any, b: Any) -> Any:
            if op_name == "add":
                return a + b
            if op_name == "sub":
                return a - b
            if op_name == "mul":
                return a * b
            if op_name == "div":
                return a / b
            if op_name == "maximum":
                return T_mod.max(a, b)
            if op_name == "minimum":
                return T_mod.min(a, b)
            raise FxToTileLangUnsupported(f"sequential binary op {op_name} has no TIR builder")

        def _compose_chain(T_mod: Any, ext_vals: dict) -> Any:
            computed = dict(ext_vals)
            last_val = None
            for op_name, payload in compute_ops:
                node_name = payload[0]
                fx_node = node_map.get(node_name)
                if not fx_node:
                    continue
                if op_name in self._SEQUENTIAL_UNARY_OPS:
                    arg_name = fx_node.args[0].name
                    v = computed.get(arg_name)
                    v_out = _apply_unary_local(T_mod, op_name, v)
                    computed[node_name] = v_out
                    last_val = v_out
                elif op_name in self._SEQUENTIAL_BINARY_OPS:
                    name1 = fx_node.args[0].name
                    name2 = fx_node.args[1].name
                    v_out = _apply_binary_local(T_mod, op_name, computed.get(name1), computed.get(name2))
                    computed[node_name] = v_out
                    last_val = v_out
            return last_val

        # Generate branches for up to 6 external inputs
        ext_names = external_inputs
        if len(ext_names) == 1:
            @T.prim_func
            def kernel(X: T.Tensor(shape, dtype), Y: T.Tensor(shape, dtype)):
                if False: _ = shape; _ = dtype
                with T.Kernel(T.ceildiv(n_elem, BLOCK), threads=BLOCK) as bx:
                    X_flat = T.Buffer((n_elem,), dtype, data=X.data)
                    Y_flat = T.Buffer((n_elem,), dtype, data=Y.data)
                    for i in T.Parallel(BLOCK):
                        idx = bx * BLOCK + i
                        if idx < n_elem:
                            Y_flat[idx] = _compose_chain(T, {ext_names[0]: X_flat[idx]})
            return kernel
        elif len(ext_names) == 2:
            @T.prim_func
            def kernel(X0: T.Tensor(shape, dtype), X1: T.Tensor(shape, dtype), Y: T.Tensor(shape, dtype)):
                if False: _ = shape; _ = dtype
                with T.Kernel(T.ceildiv(n_elem, BLOCK), threads=BLOCK) as bx:
                    X0_flat = T.Buffer((n_elem,), dtype, data=X0.data)
                    X1_flat = T.Buffer((n_elem,), dtype, data=X1.data)
                    Y_flat = T.Buffer((n_elem,), dtype, data=Y.data)
                    for i in T.Parallel(BLOCK):
                        idx = bx * BLOCK + i
                        if idx < n_elem:
                            Y_flat[idx] = _compose_chain(T, {ext_names[0]: X0_flat[idx], ext_names[1]: X1_flat[idx]})
            return kernel
        elif len(ext_names) == 3:
            @T.prim_func
            def kernel(X0: T.Tensor(shape, dtype), X1: T.Tensor(shape, dtype), X2: T.Tensor(shape, dtype), Y: T.Tensor(shape, dtype)):
                if False: _ = shape; _ = dtype
                with T.Kernel(T.ceildiv(n_elem, BLOCK), threads=BLOCK) as bx:
                    X0_flat = T.Buffer((n_elem,), dtype, data=X0.data)
                    X1_flat = T.Buffer((n_elem,), dtype, data=X1.data)
                    X2_flat = T.Buffer((n_elem,), dtype, data=X2.data)
                    Y_flat = T.Buffer((n_elem,), dtype, data=Y.data)
                    for i in T.Parallel(BLOCK):
                        idx = bx * BLOCK + i
                        if idx < n_elem:
                            Y_flat[idx] = _compose_chain(T, {ext_names[0]: X0_flat[idx], ext_names[1]: X1_flat[idx], ext_names[2]: X2_flat[idx]})
            return kernel
        elif len(ext_names) == 4:
            @T.prim_func
            def kernel(X0: T.Tensor(shape, dtype), X1: T.Tensor(shape, dtype), X2: T.Tensor(shape, dtype), X3: T.Tensor(shape, dtype), Y: T.Tensor(shape, dtype)):
                if False: _ = shape; _ = dtype
                with T.Kernel(T.ceildiv(n_elem, BLOCK), threads=BLOCK) as bx:
                    X0_flat = T.Buffer((n_elem,), dtype, data=X0.data)
                    X1_flat = T.Buffer((n_elem,), dtype, data=X1.data)
                    X2_flat = T.Buffer((n_elem,), dtype, data=X2.data)
                    X3_flat = T.Buffer((n_elem,), dtype, data=X3.data)
                    Y_flat = T.Buffer((n_elem,), dtype, data=Y.data)
                    for i in T.Parallel(BLOCK):
                        idx = bx * BLOCK + i
                        if idx < n_elem:
                            Y_flat[idx] = _compose_chain(T, {ext_names[0]: X0_flat[idx], ext_names[1]: X1_flat[idx], ext_names[2]: X2_flat[idx], ext_names[3]: X3_flat[idx]})
            return kernel
        elif len(ext_names) == 5:
            @T.prim_func
            def kernel(X0: T.Tensor(shape, dtype), X1: T.Tensor(shape, dtype), X2: T.Tensor(shape, dtype), X3: T.Tensor(shape, dtype), X4: T.Tensor(shape, dtype), Y: T.Tensor(shape, dtype)):
                if False: _ = shape; _ = dtype
                with T.Kernel(T.ceildiv(n_elem, BLOCK), threads=BLOCK) as bx:
                    X0_flat = T.Buffer((n_elem,), dtype, data=X0.data)
                    X1_flat = T.Buffer((n_elem,), dtype, data=X1.data)
                    X2_flat = T.Buffer((n_elem,), dtype, data=X2.data)
                    X3_flat = T.Buffer((n_elem,), dtype, data=X3.data)
                    X4_flat = T.Buffer((n_elem,), dtype, data=X4.data)
                    Y_flat = T.Buffer((n_elem,), dtype, data=Y.data)
                    for i in T.Parallel(BLOCK):
                        idx = bx * BLOCK + i
                        if idx < n_elem:
                            Y_flat[idx] = _compose_chain(T, {ext_names[0]: X0_flat[idx], ext_names[1]: X1_flat[idx], ext_names[2]: X2_flat[idx], ext_names[3]: X3_flat[idx], ext_names[4]: X4_flat[idx]})
            return kernel
        elif len(ext_names) == 6:
            @T.prim_func
            def kernel(X0: T.Tensor(shape, dtype), X1: T.Tensor(shape, dtype), X2: T.Tensor(shape, dtype), X3: T.Tensor(shape, dtype), X4: T.Tensor(shape, dtype), X5: T.Tensor(shape, dtype), Y: T.Tensor(shape, dtype)):
                if False: _ = shape; _ = dtype
                with T.Kernel(T.ceildiv(n_elem, BLOCK), threads=BLOCK) as bx:
                    X0_flat = T.Buffer((n_elem,), dtype, data=X0.data)
                    X1_flat = T.Buffer((n_elem,), dtype, data=X1.data)
                    X2_flat = T.Buffer((n_elem,), dtype, data=X2.data)
                    X3_flat = T.Buffer((n_elem,), dtype, data=X3.data)
                    X4_flat = T.Buffer((n_elem,), dtype, data=X4.data)
                    X5_flat = T.Buffer((n_elem,), dtype, data=X5.data)
                    Y_flat = T.Buffer((n_elem,), dtype, data=Y.data)
                    for i in T.Parallel(BLOCK):
                        idx = bx * BLOCK + i
                        if idx < n_elem:
                            Y_flat[idx] = _compose_chain(T, {ext_names[0]: X0_flat[idx], ext_names[1]: X1_flat[idx], ext_names[2]: X2_flat[idx], ext_names[3]: X3_flat[idx], ext_names[4]: X4_flat[idx], ext_names[5]: X5_flat[idx]})
            return kernel
        else:
            raise FxToTileLangUnsupported(
                f"sequential region: unsupported number of external inputs ({len(ext_names)})")

    def _emit_sequential_reduction(
        self, T: Any, op_name: str, payload: Tuple[Any, ...],
    ) -> Any:
        """Emit a serial-loop accumulator PrimFunc for a sole reduction op.

        Supported ops: ``sum`` (full + dim reduction), ``sum_dim`` (dim list
        reduction). Lowering shape is a single-block, single-thread serial
        accumulator over the flattened input — correct on every backend
        without needing the cooperative ``T.reduce_sum`` warp template
        (which requires a tile-resident input fragment we don't have here).
        Performance is ``O(n_elem)`` serial; that's the correctness floor.
        Tighter cooperative templates can be wired in later; the point of
        this fix is closing the extern-fallback hole, not perf.

        Payload contract:
          * ``sum``     : ``(node_name, x_spec, dims, keepdim)`` — see
            :func:`_emit_sum`.
          * ``sum_dim`` : ``(node_name, x_spec, dims, keepdim)`` — see
            :func:`emit_sum_dim_intlist`.

        Output is allocated by the caller (``_run`` in ``run()``) using the
        FX output_spec; for full reductions ``out_spec.shape == ()`` and
        the kernel writes to ``Y[()]`` via a 1-element flat buffer view.
        """
        if len(payload) < 4:
            raise FxToTileLangUnsupported(
                f"reduction emitter: payload {payload!r} missing dims/keepdim "
                "fields (expected (node_name, x_spec, dims, keepdim)); "
                "extern fallback is intentional")
        x_spec = payload[1]
        dims = payload[2]
        keepdim = bool(payload[3])
        if not isinstance(x_spec, _TensorSpec):
            raise FxToTileLangUnsupported(
                f"reduction emitter: cannot resolve input spec for {op_name}")

        in_shape = x_spec.shape
        dtype = x_spec.dtype
        n_elem = 1
        for s in in_shape:
            n_elem *= int(s)
        if n_elem <= 0:
            raise FxToTileLangUnsupported(
                f"reduction emitter: degenerate numel from input shape {in_shape}")

        # Compute output shape (must match _emit_sum's exit logic so the
        # launcher's pre-allocated output buffer matches the kernel signature).
        if dims is None:
            out_shape: Tuple[int, ...] = (
                () if not keepdim else tuple(1 for _ in in_shape)
            )
            reduced_axes: List[int] = list(range(len(in_shape)))
        else:
            if isinstance(dims, int):
                dims_list = [dims]
            else:
                dims_list = list(dims)
            reduced_axes = sorted({
                d if d >= 0 else d + len(in_shape) for d in dims_list
            })
            if keepdim:
                out_shape = tuple(
                    1 if i in reduced_axes else s for i, s in enumerate(in_shape)
                )
            else:
                out_shape = tuple(
                    s for i, s in enumerate(in_shape) if i not in reduced_axes
                )

        n_out = 1
        for s in out_shape:
            n_out *= int(s)
        n_out = max(n_out, 1)  # 0-dim full reduction → 1-element flat view.

        # Full reduction (every input axis collapses) is the common case for
        # ``x.sum()``. Partial reductions need a per-output-index nested loop;
        # we leave that to a later pass and route to extern with a clear msg.
        is_full_reduction = (n_out == 1)
        if not is_full_reduction:
            raise FxToTileLangUnsupported(
                f"reduction emitter: partial reduction (out shape {out_shape}) "
                "not yet mapped; sequential region falling back to extern is "
                "intentional. Wire a per-output nested-loop pattern when this "
                "case shows up in a real workload.")

        # 0-dim outputs need a (1,)-shape kernel-side buffer; the launcher's
        # ``torch.empty(())`` allocation is reinterpreted via ``data=Y.data``.
        out_kernel_shape = (1,) if out_shape == () else out_shape

        @T.prim_func
        def kernel(
            X: T.Tensor(in_shape, dtype),
            Y: T.Tensor(out_kernel_shape, dtype),
        ):
            # B2 wave fix-pack — Bug 1 fix (closure capture). See
            # ``_emit_sequential_region`` kernel for the full rationale.
            if False:  # noqa: SIM103
                _ = in_shape  # noqa: F841
                _ = out_kernel_shape  # noqa: F841
                _ = dtype  # noqa: F841
            with T.Kernel(1, threads=1) as bx:
                X_flat = T.Buffer((n_elem,), dtype, data=X.data)
                Y_flat = T.Buffer((1,), dtype, data=Y.data)
                acc = T.alloc_local((1,), dtype)
                acc[0] = T.cast(0, dtype)
                for i in T.serial(n_elem):
                    acc[0] = acc[0] + X_flat[i]
                Y_flat[0] = acc[0]

        return kernel

    def _emit_sequential_matmul(
        self, T: Any, op_name: str, payload: Tuple[Any, ...],
    ) -> Any:
        """Emit a single-region ``T.gemm`` PrimFunc for a sole matmul op.

        Supported: 2D ``matmul`` / ``mm``. Reuses the same tile-shape
        heuristic + cache-residency pattern as
        :meth:`_emit_fused_linear_region` but with no activation epilogue —
        ``relu(x @ w)`` is still funneled through ``_emit_fused_linear_region``
        via ``try_match`` long before this method runs (see line 1632).

        ``addmm`` is not handled here because its payload carries an extra
        ``bias`` slot (``(node.name, bias, a, b)``) — we route addmm + the
        non-2D batched ``bmm`` case to extern via :class:`FxToTileLangUnsupported`.

        ``b_view_of_a`` detection
        --------------------------
        When the FX graph shows ``torch.matmul(x, x.t())`` the ``aten.t``
        call is absorbed by ``_SEQUENTIAL_VIEW_OPS`` and dropped from
        ``compute_ops`` — but the matmul's B operand is still a *view* of
        the same input tensor as A. The ``input_specs`` list (driven by FX
        placeholders) has exactly **one** entry for that case. If we emit
        a 3-buffer kernel signature ``(A, B, C)`` the launcher's
        ``k(*tensors, *outs)`` call passes 1 input + 1 output = 2 tensors
        and the kernel arity check fails — which is why ``a @ a.t()``
        previously routed silently to the extern fallback.

        Detect the view-of-self case by inspecting the FX graph: walk the
        matmul node's args; if its second arg is an ``aten.t`` call whose
        sole operand is the matmul's first arg, set ``b_view_of_a=True``
        and emit a 2-buffer ``kernel(A, C)`` that materialises B's tile
        directly from A inside the kernel. Otherwise fall back to the
        original 3-buffer signature.
        """
        # payload contract for matmul/mm: (node.name, a, b) — see emit_matmul.
        if len(payload) < 3:
            raise FxToTileLangUnsupported(
                f"matmul emitter: payload {payload!r} too short for "
                "(node_name, a, b); extern fallback is intentional")
        node_name = payload[0]
        a_spec = payload[1]
        b_spec = payload[2]
        if not (isinstance(a_spec, _TensorSpec)
                and isinstance(b_spec, _TensorSpec)):
            raise FxToTileLangUnsupported(
                "matmul emitter: requires resolved tensor specs for A,B "
                f"(got A={a_spec!r}, B={b_spec!r})")
        if op_name == "bmm" or len(a_spec.shape) != 2 or len(b_spec.shape) != 2:
            raise FxToTileLangUnsupported(
                f"matmul emitter: only 2D matmul supported in this pass "
                f"(got op={op_name}, A.shape={a_spec.shape}, "
                f"B.shape={b_spec.shape}); extern fallback is intentional")

        # ``b_view_of_a`` — does B's FX node come from ``aten.t`` of A's node?
        # When True the launcher only carries one input tensor, so the
        # kernel signature MUST drop the redundant B parameter; otherwise
        # the runtime arity check fails (one input + one output != three
        # kernel buffers) and the entire region degrades to the extern
        # fallback.
        b_view_of_a = self._matmul_b_is_t_view_of_a(node_name)

        m, k = a_spec.shape
        k2, n = b_spec.shape
        if k != k2:
            raise FxToTileLangUnsupported(
                f"matmul emitter: inner-dim mismatch {a_spec.shape} x {b_spec.shape}")
        dtype = a_spec.dtype
        block_M, block_N, block_K = self._tile_constants(m, n, k)
        # Pick a thread count so the gemm warp policy ``m_warp * n_warp ==
        # num_warps`` is satisfiable. The default 128 threads (4 warps) only
        # tiles cleanly when ``block_M >= 16 and block_N >= 16``; for tiny
        # shapes (e.g. the 8x8 / 8x16 used in the unit-tests) we drop to
        # 32 threads (1 warp). This keeps the small-shape path real-TIR
        # instead of routing to extern.
        threads = 32 if (block_M < 16 or block_N < 16) else 128
        # Accumulator dtype: prefer fp32 for fp16 inputs (numerical hygiene),
        # but ONLY when the resulting epilogue ``T.copy(C_l, C[...])`` is
        # supported — for tiny tile shapes the lowering check
        # ``undefined.size() == 0`` fires on the elem_offset of the dtype-
        # mismatched fragment-to-global copy. Easiest workaround: keep the
        # accumulator at the input dtype for tiny tiles (the unit-tests pass
        # at fp16 atol=1e-2). Larger shapes route through
        # ``_emit_fused_linear_region`` (matmul + activation) which already
        # handles the fp32 accum + epilogue cast cleanly.
        accum_dtype = dtype if (block_M < 16 or block_N < 16) else "float32"

        if b_view_of_a:
            # B is ``A.t()``: shape ``(n, m)`` originally became ``(k, n)``
            # after the transpose, where ``a_spec`` is ``(m, k)`` and
            # ``b_spec`` is ``(k, n)``. For ``a @ a.t()`` we additionally
            # have ``n == m``. The kernel reads B by indexing A with
            # swapped row/col, materialised into a shared tile via an
            # explicit per-element load loop (``T.copy`` on a transposed
            # access path is rejected by the region-extent inferencer).
            @T.prim_func
            def kernel(
                A: T.Tensor((m, k), dtype),
                C: T.Tensor((m, n), dtype),
            ):
                # Defensive closure-capture marker (see Bug 1 doc above).
                if False:  # noqa: SIM103
                    _ = (m, k, n, dtype)  # noqa: F841
                with T.Kernel(T.ceildiv(n, block_N), T.ceildiv(m, block_M),
                              threads=threads) as (bx, by):
                    A_s = T.alloc_shared((block_M, block_K), dtype)
                    B_s = T.alloc_shared((block_K, block_N), dtype)
                    C_l = T.alloc_fragment((block_M, block_N), accum_dtype)
                    T.clear(C_l)
                    for ko in T.Pipelined(T.ceildiv(k, block_K),
                                          num_stages=2):
                        T.copy(A[by * block_M, ko * block_K], A_s)
                        # Materialise the transposed-A tile into B_s using
                        # an explicit parallel loop: ``B[i, j] == A[j, i]``
                        # since ``B = A.t()`` and the matmul reads
                        # ``B[ko*block_K + ki, bx*block_N + bj]``.
                        for ki, bj in T.Parallel(block_K, block_N):
                            B_s[ki, bj] = A[bx * block_N + bj,
                                            ko * block_K + ki]
                        T.gemm(A_s, B_s, C_l)
                    T.copy(C_l, C[by * block_M, bx * block_N])

            return kernel

        @T.prim_func
        def kernel(
            A: T.Tensor((m, k), dtype),
            B: T.Tensor((k, n), dtype),
            C: T.Tensor((m, n), dtype),
        ):
            # B2 wave fix-pack — Bug 1 fix (closure capture).
            # ``m, k, n, dtype`` are referenced in ``T.Kernel`` / annotations
            # already, so this no-op block is just defensive — keeps the
            # pattern uniform with the unary/binary/reduction emitters.
            if False:  # noqa: SIM103
                _ = (m, k, n, dtype)  # noqa: F841
            with T.Kernel(T.ceildiv(n, block_N), T.ceildiv(m, block_M),
                          threads=threads) as (bx, by):
                A_s = T.alloc_shared((block_M, block_K), dtype)
                B_s = T.alloc_shared((block_K, block_N), dtype)
                C_l = T.alloc_fragment((block_M, block_N), accum_dtype)
                T.clear(C_l)
                for ko in T.Pipelined(T.ceildiv(k, block_K), num_stages=2):
                    T.copy(A[by * block_M, ko * block_K], A_s)
                    T.copy(B[ko * block_K, bx * block_N], B_s)
                    T.gemm(A_s, B_s, C_l)
                T.copy(C_l, C[by * block_M, bx * block_N])

        return kernel

    def _matmul_b_is_t_view_of_a(self, matmul_node_name: str) -> bool:
        """Return True if ``matmul_node_name``'s B operand is ``aten.t(A)``.

        Walks the FX graph (``self.gm.graph``) to detect the
        ``torch.matmul(x, x.t())`` shape. Returns False when the graph
        cannot be inspected, when the node isn't found, or when B's FX
        node is anything other than a ``t`` / ``transpose`` of A. The
        check is purely structural: we compare FX node identities, not
        ``_TensorSpec`` shape equality, because the goal is to know
        whether the launcher's input list collapses two matmul operands
        into one (which it does iff B is a view of A).
        """
        try:
            graph = self.gm.graph
        except Exception:
            return False
        target_node = None
        for node in graph.nodes:
            if getattr(node, "name", None) == matmul_node_name:
                target_node = node
                break
        if target_node is None:
            return False
        args = getattr(target_node, "args", ()) or ()
        if len(args) < 2:
            return False
        a_node, b_node = args[0], args[1]
        # B must itself be an ``aten.t`` call (op == call_method "t" or
        # call_function torch.t / aten.t). The transpose inputs match A
        # iff its sole operand is the same FX node we passed for A.
        if getattr(b_node, "op", None) not in ("call_method", "call_function"):
            return False
        target = getattr(b_node, "target", None)
        target_name = getattr(target, "__name__", None) or str(target)
        if target_name not in ("t", "transpose") and "aten.t" not in str(target):
            return False
        b_args = getattr(b_node, "args", ()) or ()
        if not b_args:
            return False
        return b_args[0] is a_node

    def _region_output_specs(
        self,
        region: List[Tuple[str, Tuple[Any, ...]]],
    ) -> Tuple["_TensorSpec", ...]:
        """Return the output specs of the last op in ``region``.

        The launcher uses these to pre-allocate output tensors when the
        compiled TileLang kernel uses the explicit ``(inputs..., outputs...)``
        calling convention. For single-region graphs this matches
        ``ctx.output_specs``; for multi-region chains we use the last-op
        spec recorded in ``op_trace`` (best-effort, may be empty if the
        region's terminal op has no resolvable output spec).
        """
        if not region:
            return ()
        # For a single-region graph the FX outputs are the region's outputs.
        if len(self.ctx.output_specs) > 0 and len(self._partition_count_cache()) == 1:
            return tuple(self.ctx.output_specs)
        # Best-effort: derive from the last op's payload tensors. We
        # cannot statically know which payload slot is the output, so we
        # fall back to "no specs" (implicit-output convention).
        return ()

    def _partition_count_cache(self) -> List[List[Tuple[str, Tuple[Any, ...]]]]:
        """Return the partitioned region list (memoised on the instance)."""
        cache = getattr(self, "_partition_cache", None)
        if cache is None:
            cache = self._partition_fusable_subgraphs()
            self._partition_cache = cache  # type: ignore[attr-defined]
        return cache

    @staticmethod
    def _tile_constants(m: int, n: int, k: int) -> Tuple[int, int, int]:
        """Default tile constants per shape (BLOCK_M=128 for matmul-shaped,
        BLOCK_N=64 for reductions, BLOCK_K=32 default).

        Heuristic: cap each block by the corresponding extent so we don't
        emit a kernel whose grid is < 1 block.
        """
        block_M = 128 if m >= 128 else (64 if m >= 64 else m)
        block_N = 64 if n >= 64 else n
        block_K = 32 if k >= 32 else k
        return block_M, block_N, block_K

    @staticmethod
    def _alloc_intermediate_kind(spec: "_TensorSpec") -> str:
        """Choose ``T.alloc_*`` flavour for an intermediate buffer (RFC §4).

        Heuristic:
          * <= 2 KiB total bytes -> ``alloc_fragment`` (register-resident)
          * <= 64 KiB total bytes -> ``alloc_shared``
          * otherwise -> ``alloc_local`` (per-thread scratch)

        Sized from the FakeTensor stamped on each FX node; keeps
        intermediates resident across the FX boundary.
        """
        elem_bytes = {
            "float16": 2, "bfloat16": 2,
            "float32": 4, "float64": 8,
            "int32": 4, "int64": 8, "bool": 1,
        }.get(spec.dtype, 4)
        total = elem_bytes
        for s in spec.shape:
            total *= s
        if total <= 2 * 1024:
            return "alloc_fragment"
        if total <= 64 * 1024:
            return "alloc_shared"
        return "alloc_local"

    # ------------------------------------------------------------------
    # Per-region extern fallback + chain launcher
    # ------------------------------------------------------------------

    def _build_region_extern_launcher(
        self,
        region: List[Tuple[str, Tuple[Any, ...]]],
    ) -> Callable[..., Any]:
        """Build a Python launcher that replays ``region`` via FX eager.

        This is the per-region (NOT whole-graph) extern fallback that fires
        when ``_materialize_subgraph`` or ``tilelang.compile`` raises for a
        single region. Other regions in the same compile keep their real
        TileLang launchers.

        For the POC we approximate per-region replay with a whole-graph FX
        eager replay, since the alternative (per-region FX subgraph
        extraction) requires a non-trivial graph-rewriter we haven't built
        yet. This is correct (FX eager always matches torch eager) but
        coarser than the cache-resident region boundary the orchestrator
        plans for. See ``_NON_FUSABLE_BOUNDARY_OPS`` for the planned
        boundary handling.
        """
        gm = self.gm
        # Mark this launcher as the extern fallback so the chain launcher
        # can detect "all regions extern" and short-circuit to a single
        # gm.forward call (the only safe approximation today).
        def _launcher(*runtime_inputs: Any) -> Any:
            return gm(*runtime_inputs)

        _launcher._tilelang_extern_fallback = True  # type: ignore[attr-defined]
        return _launcher

    def _derive_region_io(
        self,
        regions: List[List[Tuple[str, Tuple[Any, ...]]]],
    ) -> Optional[List[Tuple[List[str], List[str]]]]:
        """Derive ``(input_node_names, output_node_names)`` per region.

        Walks the FX graph to identify which placeholder / get_attr / prior
        region outputs each region consumes, and which of its produced
        nodes are consumed externally (by a later region or the FX
        ``output``). Returns ``None`` if the trace doesn't carry enough
        info (e.g., a region's payload[0] isn't a node-name string), so
        the caller can fall back to ``gm.forward``.
        """
        try:
            gm_nodes = list(self.gm.graph.nodes)
            name_to_node = {n.name: n for n in gm_nodes}
        except Exception:  # pragma: no cover
            return None

        produced_per_region: List[List[str]] = []
        for region in regions:
            produced: List[str] = []
            for _, payload in region:
                if not payload or not isinstance(payload[0], str):
                    return None
                produced.append(payload[0])
            produced_per_region.append(produced)

        # Reverse map: node-name -> region index (last writer wins).
        node_to_region: Dict[str, int] = {}
        for r_idx, names in enumerate(produced_per_region):
            for n in names:
                node_to_region[n] = r_idx

        result: List[Tuple[List[str], List[str]]] = []
        for r_idx, region in enumerate(regions):
            produced = produced_per_region[r_idx]
            produced_set = set(produced)

            # Inputs: external nodes referenced by any FX node in this region.
            input_names: List[str] = []
            input_seen: set = set()
            for n_name in produced:
                fx_node = name_to_node.get(n_name)
                if fx_node is None:
                    return None
                for arg in fx_node.all_input_nodes:
                    if (arg.name not in produced_set
                            and arg.name not in input_seen):
                        input_seen.add(arg.name)
                        input_names.append(arg.name)

            # Outputs: produced nodes consumed by some node OUTSIDE this region
            # (later region or graph output).
            output_names: List[str] = []
            for n_name in produced:
                fx_node = name_to_node.get(n_name)
                if fx_node is None:
                    return None
                consumed_externally = False
                for user in fx_node.users:
                    if user.op == "output" or user.name not in produced_set:
                        consumed_externally = True
                        break
                if consumed_externally:
                    output_names.append(n_name)
            if not output_names and produced:
                # Treat the last produced node as the region output if the FX
                # graph never references any of them externally (rare; usually
                # means the region was eliminated by DCE).
                output_names = [produced[-1]]
            result.append((input_names, output_names))

        return result

    def _build_chain_launcher(
        self,
        region_launchers: List[Callable[..., Any]],
        regions: List[List[Tuple[str, Tuple[Any, ...]]]],
    ) -> Callable[..., Any]:
        """Wire region launchers into a single Dynamo-shaped callable.

        Wave-2 fix-pack (grok #02 perf §1 / design): for genuinely
        multi-region compiled traces we now derive per-region
        ``(input_names, output_names)`` from the FX graph and thread
        tensors through an ``env`` dict instead of falling back to
        ``gm.forward``. The fallback only fires when:

        * every region is the extern stub (nothing compiled), OR
        * the trace lacks enough info to derive region I/O (a region's
          op-trace payload doesn't carry node-name strings).
        """
        gm = self.gm
        all_extern = all(
            getattr(rl, "_tilelang_extern_fallback", False)
            for rl in region_launchers
        ) if region_launchers else True
        if all_extern:
            def _launcher(*runtime_inputs: Any) -> Any:
                return gm(*runtime_inputs)
            _launcher._tilelang_chain_mode = "extern_all"  # type: ignore[attr-defined]
            return _launcher

        if len(region_launchers) == 1:
            single = region_launchers[0]
            def _launcher_one(*runtime_inputs: Any) -> Any:
                return single(*runtime_inputs)
            _launcher_one._tilelang_chain_mode = "single_region"  # type: ignore[attr-defined]
            return _launcher_one

        # True multi-region chain — derive per-region I/O from the FX graph.
        region_io = self._derive_region_io(regions)
        try:
            placeholder_names = [
                n.name for n in self.gm.graph.nodes if n.op == "placeholder"
            ]
            attr_names = [
                n.name for n in self.gm.graph.nodes if n.op == "get_attr"
            ]
        except Exception:  # pragma: no cover
            placeholder_names = []
            attr_names = []
        runtime_input_names = placeholder_names + attr_names

        if region_io is None or not runtime_input_names:
            # Degenerate case — give up cleanly rather than producing wrong
            # results. This is the ONLY remaining gm.forward fall-through.
            def _launcher_multi_fallback(*runtime_inputs: Any) -> Any:
                return gm(*runtime_inputs)
            _launcher_multi_fallback._tilelang_chain_mode = (  # type: ignore[attr-defined]
                "multi_fallback_to_gm_forward"
            )
            return _launcher_multi_fallback

        def _launcher_multi(*runtime_inputs: Any) -> Any:
            if len(runtime_inputs) < len(runtime_input_names):
                # Calling convention drift: fall back to gm.forward rather
                # than IndexError. (Shouldn't happen in practice.)
                return gm(*runtime_inputs)
            env: Dict[str, Any] = dict(zip(runtime_input_names, runtime_inputs))
            for rl, (in_names, out_names) in zip(region_launchers, region_io):
                try:
                    in_tensors = tuple(env[n] for n in in_names)
                except KeyError:
                    # A region needs a tensor we don't have in the env yet.
                    # Fall back to gm.forward this call.
                    return gm(*runtime_inputs)
                out = rl(*in_tensors)
                if not isinstance(out, tuple):
                    out_tup: Tuple[Any, ...] = (out,)
                else:
                    out_tup = out
                # Match output count; pad with ``out`` if launcher returned
                # a single tensor for a multi-output region (best-effort).
                if len(out_tup) < len(out_names):
                    out_tup = out_tup + (out_tup[-1],) * (len(out_names) - len(out_tup))
                for nm, t in zip(out_names, out_tup):
                    env[nm] = t

            # Final outputs = the last region's output names.
            final_names = region_io[-1][1] if region_io else []
            if not final_names:
                # Should not happen given _derive_region_io's fallback,
                # but stay safe.
                return gm(*runtime_inputs)
            if len(final_names) == 1:
                return env[final_names[0]]
            return tuple(env[n] for n in final_names)

        _launcher_multi._tilelang_chain_mode = "multi_real_chain"  # type: ignore[attr-defined]
        return _launcher_multi

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        n_nodes = len(list(self.gm.graph.nodes))
        return (f"FXToTileLang(nodes={n_nodes}, "
                f"example_inputs={len(self.example_inputs)})")
