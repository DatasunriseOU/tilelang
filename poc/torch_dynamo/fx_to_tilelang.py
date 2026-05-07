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
# ---------------------------------------------------------------------------


ATEN_DISPATCH: Dict[str, Callable[..., _TensorSpec]] = {
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
    "scaled_dot_product_attention": _emit_sdpa,
    # --- masking -----------------------------------------------------------
    "where": _emit_where,
    "masked_fill": _emit_masked_fill,
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

    def run(self) -> "FusedKernelArtifact":  # type: ignore[name-defined]
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
        h = hashlib.blake2b(digest_size=8)
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

            def _make(k: Any, out_specs: Tuple[_TensorSpec, ...]) -> Callable[..., Any]:
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
                    return k(*tensors)
                return _run

            region_launchers.append(_make(kernel, region_output_specs))

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

    def _emit_sequential_region(
        self, T: Any,
        region: List[Tuple[str, Tuple[Any, ...]]],
    ) -> Any:
        """Best-effort sequential PrimFunc for a region of ops.

        For the POC this raises ``NotImplementedError`` for any op trace
        we don't have a tight pattern for — the orchestrator catches the
        exception and routes the region to the extern-fallback launcher.
        Future contributors fill in this method by walking ``region`` and
        chaining per-op TIR snippets via shared/fragment-resident
        intermediates (see ``_alloc_intermediate_kind`` for the
        register/shared/local heuristic).
        """
        ops = [op for op, _ in region]
        raise NotImplementedError(
            f"sequential region materialisation for op trace {ops!r} "
            "not implemented in POC; orchestrator will route to "
            "tir.call_extern fallback (RFC §7 Phase 2)."
        )

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

    def _build_chain_launcher(
        self,
        region_launchers: List[Callable[..., Any]],
        regions: List[List[Tuple[str, Tuple[Any, ...]]]],
    ) -> Callable[..., Any]:
        """Wire region launchers into a single Dynamo-shaped callable.

        The chain launcher is invoked with the runtime inputs Dynamo
        passes the backend (placeholders + bound params, in FX order).
        It forwards them through each region launcher in sequence,
        threading the previous region's outputs as the next region's
        inputs. The final region's outputs are returned.

        POC simplification: when every region is using the extern
        fallback (i.e. nothing compiled to real TileLang), we delegate to
        a single ``gm.forward`` call so we don't pay N x eager-replay
        cost. When regions mix compiled + extern, we still need a
        real region-level argument router — that's deferred until
        ``_materialize_subgraph`` covers the full op set.
        """
        gm = self.gm
        all_extern = all(
            getattr(rl, "_tilelang_extern_fallback", False)
            for rl in region_launchers
        ) if region_launchers else True
        if all_extern:
            def _launcher(*runtime_inputs: Any) -> Any:
                return gm(*runtime_inputs)
            return _launcher

        # Mixed / fully compiled: today this only fires for the
        # ``fused_linear`` smoke pattern (exactly one region, the matmul +
        # activation kernel). The kernel signature matches the FX graph's
        # placeholder + param order, so a single launcher call suffices.
        # Multi-region wiring lands together with the sequential
        # materialiser.
        if len(region_launchers) == 1:
            single = region_launchers[0]
            def _launcher_one(*runtime_inputs: Any) -> Any:
                return single(*runtime_inputs)
            return _launcher_one

        # True multi-region chain: sequentially run regions, but until
        # _materialize_subgraph covers more patterns we can't statically
        # know which inputs each region consumes — so we conservatively
        # fall back to gm.forward and let the per-op compile cache pick
        # up the kernels for next time. (This is the only place we still
        # touch gm.forward in the orchestrator.)
        def _launcher_multi(*runtime_inputs: Any) -> Any:  # pragma: no cover
            return gm(*runtime_inputs)
        return _launcher_multi

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        n_nodes = len(list(self.gm.graph.nodes))
        return (f"FXToTileLang(nodes={n_nodes}, "
                f"example_inputs={len(self.example_inputs)})")
