"""Declarative fusion-pattern table for the FX -> TileLang lowerer.

RFC reference: ``RFC_unified_fused_kernel.md`` §3 (FX -> TileLang custom
backend, cache-resident fusion) and §4 (intermediates stay register/shared
resident across the FX boundary). RFC §7 Phase 2 (FX op map) calls out the
inductor-coverage set of standard ops; the patterns below recognise the
three highest-leverage fusion opportunities in that set, plus the two
extension patterns from RFC §7 Phase 2.5 (``gemm_softmax`` for attention
QK^T softmax, ``qk_reduce_sm_scale`` for sparse-MLA / DeepSeek-style
indexed QK reducers with a baked-in scalar scale).

A fusion pattern is a ``(matcher_fn, emitter_fn)`` tuple:

  matcher_fn(nodes: List[OpTraceEntry], start: int)
      Returns ``(matched: bool, captured: List[OpTraceEntry], end: int)``.
      ``start`` is the index inside the op_trace list at which to attempt
      the match. The matcher MUST NOT mutate ``nodes``.

  emitter_fn(captured, T, ctx)
      Hook the orchestrator passes when materialising TIR for the captured
      run. The orchestrator's TIR builder (``T = tilelang.language``) and
      the LoweringContext (``ctx``) are forwarded. Returns the output
      ``_TensorSpec`` of the fused region (orchestrator wires the
      intermediate -> output buffer).

The orchestrator falls back to a sequential per-op chain if no matcher
fires for the current ``start`` index. An empty pattern table = pure
sequential lowering — still correct, just less tightly fused.

Currently registered patterns
-----------------------------
1. ``fused_linear`` — ``matmul`` / ``mm`` / ``addmm`` + activation
   (``relu`` / ``gelu`` / ``silu`` / ``tanh``). Emitted as ``T.gemm`` with
   the elementwise epilogue applied inside the same accumulator fragment
   (``T.alloc_fragment``). This is the canonical TileLang fused-linear
   recipe — see ``tilelang/language/gemm.py`` and the ``T.Parallel``
   epilogue idiom in ``examples/dequantize_gemm/``.

2. ``layernorm_linear`` — ``layer_norm`` followed by ``matmul``. Emitted
   as a two-stage tile: first the row-wise reduce
   (``tilelang/language/reduce_op.py:140``) for mean / var into a shared
   buffer, then ``T.gemm`` consumes the normalised tile. Keeps the
   normalised activations shared-resident across the boundary — RFC §4.

3. ``softmax_epilogue`` — ``softmax`` / ``log_softmax`` directly after
   a ``matmul``. The softmax is folded into the gemm's accumulator: a
   single ``T.reduce_max`` + ``T.exp`` + ``T.reduce_sum`` epilogue runs
   on the fragment without materialising the gemm output to global. This
   is the same recipe used by the flash-attention forward kernel
   (``poc/torch_dynamo/_kernels/flash_attention.py``).

4. ``gemm_softmax`` — attention-shaped ``transpose(K, -1, -2)`` followed
   by ``matmul(Q, K^T)`` followed by ``softmax(..., dim=-1)``. Same
   recipe as ``softmax_epilogue`` but the matcher binds the upstream
   ``transpose`` so the emitter can request the gemm's B operand in
   transposed-load form (``T.gemm(..., transpose_B=True)``) instead of
   materialising the transposed K tile. Avoids the ``(B, H, M, N)``
   matmul-result writeback entirely. See the convergence design in
   ``RFC_unified_fused_kernel.md`` §7 Phase 2.5.

5. ``qk_reduce_sm_scale`` — DeepSeek-V3 / sparse-MLA indexed QK reducer
   followed by a scalar multiply (``* sm_scale``). The scalar is folded
   into the reducer's output store so the multiply never sees HBM. See
   ``cppmega_mlx/nn/_tilelang/sparse_mla_fp8_path_c.py`` for the
   ``fp8_sparse_mla_indexed_qk_reduce`` kernel signature — the
   ``sm_scale_buf`` parameter is the bake-in target.
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, List, Optional, Tuple

# An op_trace entry has the shape ``(op_name, payload_tuple)``. We only
# inspect ``op_name`` in the matchers to keep them payload-agnostic.

OpTrace = Tuple[str, Tuple[Any, ...]]
MatcherResult = Tuple[bool, List[OpTrace], int]
MatcherFn = Callable[[List[OpTrace], int], MatcherResult]
EmitterFn = Callable[..., Any]


_MATMUL_OPS = {"matmul", "mm", "addmm"}
_ACTIVATIONS = {"relu", "gelu", "silu", "tanh"}
_SOFTMAX_OPS = {"softmax", "log_softmax"}


def _match_fused_linear(nodes: List[OpTrace], start: int) -> MatcherResult:
    """Match ``matmul``/``mm``/``addmm`` immediately followed by an activation."""
    if start + 1 >= len(nodes):
        return False, [], start
    op0 = nodes[start][0]
    op1 = nodes[start + 1][0]
    if op0 in _MATMUL_OPS and op1 in _ACTIVATIONS:
        return True, [nodes[start], nodes[start + 1]], start + 2
    return False, [], start


def _emit_fused_linear(captured: List[OpTrace], T: Any, ctx: Any) -> str:
    """Documentation-only emitter — orchestrator handles the actual TIR.

    We return the activation name so the orchestrator can route to the
    correct epilogue function. Concrete TIR materialisation lives in
    ``FXToTileLang._emit_fused_linear_region`` because it needs full
    access to the buffer-allocation context.
    """
    activation = captured[1][0]
    return activation


def _match_layernorm_linear(nodes: List[OpTrace], start: int) -> MatcherResult:
    """Match ``layer_norm`` (or rms_norm) followed by ``matmul``/``mm``/``addmm``."""
    if start + 1 >= len(nodes):
        return False, [], start
    op0 = nodes[start][0]
    op1 = nodes[start + 1][0]
    if op0 in {"layer_norm", "native_layer_norm", "rms_norm"} and op1 in _MATMUL_OPS:
        return True, [nodes[start], nodes[start + 1]], start + 2
    return False, [], start


def _emit_layernorm_linear(captured: List[OpTrace], T: Any, ctx: Any) -> str:
    """Two-stage shared-resident tile (norm reduce -> gemm)."""
    return "layernorm_linear"


def _match_softmax_epilogue(nodes: List[OpTrace], start: int) -> MatcherResult:
    """Match ``matmul`` immediately followed by ``softmax`` / ``log_softmax``."""
    if start + 1 >= len(nodes):
        return False, [], start
    op0 = nodes[start][0]
    op1 = nodes[start + 1][0]
    if op0 in _MATMUL_OPS and op1 in _SOFTMAX_OPS:
        return True, [nodes[start], nodes[start + 1]], start + 2
    return False, [], start


def _emit_softmax_epilogue(captured: List[OpTrace], T: Any, ctx: Any) -> str:
    softmax_kind = captured[1][0]
    return softmax_kind


# ---------------------------------------------------------------------------
# Extension patterns (RFC §7 Phase 2.5).
# ---------------------------------------------------------------------------
# These patterns are declared after the original three so the
# orchestrator's first-match-wins ordering keeps the legacy behaviour
# unchanged. ``softmax_epilogue`` only matches the bare ``matmul +
# softmax`` 2-tuple; ``gemm_softmax`` extends that to the K-transpose
# 3-tuple typical of attention QK^T (the upstream ``transpose`` is then
# absorbed as a transposed-B load instead of being lowered to its own
# tile).
# ---------------------------------------------------------------------------


# QK reducer custom-op trace names. The sparse-MLA path-C kernels in
# ``cppmega_mlx/nn/_tilelang/sparse_mla_fp8_path_c.py`` register their
# QK reducer under one of these qualnames; the FX walker echoes the
# bare op short-name into ``ctx.op_trace`` (see
# ``fx_to_tilelang.py::_emit_custom_op``). Add new names here as new
# sparse-MLA variants land.
_QK_REDUCE_OPS = {
    "qk_reduce",
    "fp8_sparse_mla_qk_reduce",
    "fp8_sparse_mla_indexed_qk_reduce",
    "sparse_mla_qk_reduce",
}


# Hit counter — incremented every time ``try_match`` fires for a given
# pattern. Tests assert against this rather than parsing the produced
# PrimFunc IR, which keeps the smoke-test independent of the (still
# evolving) emitter shapes. Reset is callers' responsibility — the
# canonical idiom is ``_FUSION_HITS.clear()`` at the top of a test.
_FUSION_HITS: "dict[str, int]" = {
    "fused_linear": 0,
    "layernorm_linear": 0,
    "softmax_epilogue": 0,
    "gemm_softmax": 0,
    "qk_reduce_sm_scale": 0,
}


class PendingFusionWarning(UserWarning):
    """Emitted at register-time when a fusion pattern's emitter still
    depends on a TileLang surface that is not yet realised in the
    canonical ``cppmega_mlx.nn._tilelang._engine_dispatch.dispatch_lower``
    path.

    This is *not* an error: the orchestrator will still match the
    pattern and feed it through the sequential fallback, which produces
    correct numerics — just without the tight fused kernel. The warning
    exists so callers grepping warnings during a CI run can see exactly
    which fusion opportunities are currently cold.
    """


def _match_gemm_softmax(nodes: List[OpTrace], start: int) -> MatcherResult:
    """Match attention QK^T softmax: ``transpose? + matmul + softmax``.

    The transpose entry is optional — modern PyTorch graphs often fold
    ``k.transpose(-1, -2)`` into the matmul's strided-load lowering at
    the Inductor level, in which case the fx_to_tilelang walker only
    appends a 2-entry ``matmul + softmax`` slice to ``op_trace``. Both
    the 2-op and 3-op forms are accepted; the emitter inspects the
    captured length to decide whether to set ``transpose_B=True`` on the
    gemm or rely on physical operand layout.
    """
    # 3-op: transpose + matmul + softmax
    if start + 2 < len(nodes):
        op0 = nodes[start][0]
        op1 = nodes[start + 1][0]
        op2 = nodes[start + 2][0]
        if (
            op0 == "transpose"
            and op1 in _MATMUL_OPS
            and op2 in _SOFTMAX_OPS
        ):
            return (
                True,
                [nodes[start], nodes[start + 1], nodes[start + 2]],
                start + 3,
            )
    # 2-op fallback: matmul + softmax. Note this overlaps with the
    # legacy ``softmax_epilogue`` pattern; ``try_match`` returns the
    # first hit, so this branch is only reached when ``softmax_epilogue``
    # is removed from FUSION_PATTERNS or reordered after us. Keeping
    # the branch makes the matcher robust to that future re-order.
    if start + 1 < len(nodes):
        op0 = nodes[start][0]
        op1 = nodes[start + 1][0]
        if op0 in _MATMUL_OPS and op1 in _SOFTMAX_OPS:
            return True, [nodes[start], nodes[start + 1]], start + 2
    return False, [], start


def _emit_gemm_softmax(captured: List[OpTrace], T: Any, ctx: Any) -> str:
    """Emitter contract — orchestrator routes to the actual TIR builder.

    Returns a short tag the orchestrator can dispatch on. The concrete
    PrimFunc emitter is ``FXToTileLang._emit_gemm_softmax_region``
    (TODO: not yet wired into the orchestrator's specialised path; the
    sequential fallback in ``_emit_sequential_region`` handles this
    pattern correctly today, just without the QK^T transposed-B load
    optimisation. See ``PendingFusionWarning`` raised at register time.)

    The recipe the dedicated emitter will materialise (kept here as a
    spec so the next contributor doesn't have to re-derive it)::

        # Q tile (BLOCK_M x BLOCK_K), K tile (BLOCK_N x BLOCK_K) — note
        # K is loaded *un-transposed* and the gemm consumes it via
        # transpose_B=True. This is the canonical FA-v2 / TileLang
        # attention recipe; see tilelang/language/gemm.py:gemm() for
        # the transpose flags and the BLOCK_K-tiled accumulator.
        S = T.alloc_fragment([BLOCK_M, BLOCK_N], "float32")
        T.gemm(Q, K, S, transpose_B=True)
        # Numerically-stable softmax fused into the same fragment.
        m = T.reduce_max(S, dim=-1)
        S = T.exp(S - m)
        l = T.reduce_sum(S, dim=-1)
        T.copy(S / l, P)  # P is the (BLOCK_M, BLOCK_N) global out tile
    """
    has_transpose = captured[0][0] == "transpose"
    return "gemm_softmax_with_transpose" if has_transpose else "gemm_softmax"


def _match_qk_reduce_sm_scale(
    nodes: List[OpTrace], start: int,
) -> MatcherResult:
    """Match a sparse-MLA QK reducer followed by a scalar multiply.

    The reducer entry's op-name is one of ``_QK_REDUCE_OPS`` (echoed by
    ``fx_to_tilelang.py::_emit_custom_op`` from the registered TileLang
    custom op qualname). The multiply is the standard ``"mul"`` entry
    the binary-elementwise emitter appends — we don't try to verify the
    RHS is a 0-d / scalar tensor here because the FX walker has already
    enforced the shape contract upstream (the FakeTensor pass would
    have raised on a non-broadcast-compatible mul).
    """
    if start + 1 >= len(nodes):
        return False, [], start
    op0 = nodes[start][0]
    op1 = nodes[start + 1][0]
    if op0 in _QK_REDUCE_OPS and op1 == "mul":
        return True, [nodes[start], nodes[start + 1]], start + 2
    return False, [], start


def _emit_qk_reduce_sm_scale(captured: List[OpTrace], T: Any, ctx: Any) -> str:
    """Emitter contract — orchestrator routes to the actual TIR builder.

    The dedicated emitter (TODO: not yet wired into the canonical
    ``dispatch_lower`` path) re-uses the
    ``make_fp8_sparse_mla_indexed_qk_reduce_kernel`` factory in
    ``cppmega_mlx/nn/_tilelang/sparse_mla_fp8_path_c.py`` but binds the
    captured ``mul`` RHS into the kernel's ``sm_scale_buf`` parameter
    *at compile time* — the kernel already multiplies by
    ``sm_scale_buf[0]`` at line 1042, so the only change needed is to
    elide the trailing ``T.copy(out * scale)`` epilogue when the scale
    has been baked in. Until that wiring lands, the sequential fallback
    runs the multiply as its own tile (correct, but two kernels). See
    ``PendingFusionWarning`` raised at register time.
    """
    return captured[0][0]


# Public table — orchestrator iterates this in declared order; first match wins.
FUSION_PATTERNS: List[Tuple[str, MatcherFn, EmitterFn]] = [
    ("fused_linear", _match_fused_linear, _emit_fused_linear),
    ("layernorm_linear", _match_layernorm_linear, _emit_layernorm_linear),
    ("softmax_epilogue", _match_softmax_epilogue, _emit_softmax_epilogue),
    # Extension patterns — RFC §7 Phase 2.5. Declared after the legacy
    # three so first-match-wins keeps existing matches stable. The
    # gemm_softmax pattern strictly subsumes softmax_epilogue when a
    # transpose is present upstream; in the bare ``matmul + softmax``
    # case softmax_epilogue still wins (declared first).
    ("gemm_softmax", _match_gemm_softmax, _emit_gemm_softmax),
    ("qk_reduce_sm_scale", _match_qk_reduce_sm_scale, _emit_qk_reduce_sm_scale),
]


def _warn_pending_extension_patterns() -> None:
    """Emit one ``PendingFusionWarning`` per extension pattern that has
    no dedicated emitter wired into ``dispatch_lower`` yet.

    Called once at module import time. We intentionally use
    ``warnings.warn`` (not ``logging``) so callers can promote the
    warning to an error in CI via ``-W error::PendingFusionWarning``.
    """
    pending = ("gemm_softmax", "qk_reduce_sm_scale")
    for name in pending:
        warnings.warn(
            (
                f"fusion pattern {name!r} is registered but its dedicated "
                f"TileLang emitter is not yet wired into "
                f"cppmega_mlx.nn._tilelang._engine_dispatch.dispatch_lower; "
                f"matches currently fall through to the sequential "
                f"fallback. Numerics are correct; perf is not."
            ),
            PendingFusionWarning,
            stacklevel=2,
        )


_warn_pending_extension_patterns()


def try_match(nodes: List[OpTrace], start: int) -> Optional[Tuple[str, List[OpTrace], int]]:
    """Try every registered pattern, return the first hit.

    Returns ``(pattern_name, captured_nodes, end_index)`` on a match, or
    ``None`` if nothing fires (orchestrator drops back to sequential).
    """
    for name, matcher, _emitter in FUSION_PATTERNS:
        matched, captured, end = matcher(nodes, start)
        if matched:
            _FUSION_HITS[name] = _FUSION_HITS.get(name, 0) + 1
            return name, captured, end
    return None
