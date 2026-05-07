"""Declarative fusion-pattern table for the FX -> TileLang lowerer.

RFC reference: ``RFC_unified_fused_kernel.md`` §3 (FX -> TileLang custom
backend, cache-resident fusion) and §4 (intermediates stay register/shared
resident across the FX boundary). RFC §7 Phase 2 (FX op map) calls out the
inductor-coverage set of standard ops; the patterns below recognise the
three highest-leverage fusion opportunities in that set.

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
"""

from __future__ import annotations

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


# Public table — orchestrator iterates this in declared order; first match wins.
FUSION_PATTERNS: List[Tuple[str, MatcherFn, EmitterFn]] = [
    ("fused_linear", _match_fused_linear, _emit_fused_linear),
    ("layernorm_linear", _match_layernorm_linear, _emit_layernorm_linear),
    ("softmax_epilogue", _match_softmax_epilogue, _emit_softmax_epilogue),
]


def try_match(nodes: List[OpTrace], start: int) -> Optional[Tuple[str, List[OpTrace], int]]:
    """Try every registered pattern, return the first hit.

    Returns ``(pattern_name, captured_nodes, end_index)`` on a match, or
    ``None`` if nothing fires (orchestrator drops back to sequential).
    """
    for name, matcher, _emitter in FUSION_PATTERNS:
        matched, captured, end = matcher(nodes, start)
        if matched:
            return name, captured, end
    return None
