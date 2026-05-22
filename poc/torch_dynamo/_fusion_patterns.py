"""Greedy FX op-trace fusion pattern registry for the TileLang Dynamo POC.

The FX walker records a compact op trace in ``fx_to_tilelang.py``.  This module
keeps the recognisers separate from the emitters so the orchestrator can choose
cache-resident multi-op kernels without hard-coding every pattern inline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple


OpTraceEntry = Tuple[str, Tuple[object, ...]]
MatchResult = Tuple[str, List[OpTraceEntry], int]
Matcher = Callable[[Sequence[OpTraceEntry], int], Optional[MatchResult]]

_FUSION_HITS: Dict[str, int] = {}


@dataclass(frozen=True)
class FusionPattern:
    """One named op-trace recogniser."""

    name: str
    matcher: Matcher


_LINEAR_OPS = frozenset({"matmul", "mm", "addmm"})
_LINEAR_EPILOGUES = frozenset({"relu", "gelu", "silu", "tanh"})
_TRANSPOSE_OPS = frozenset({"transpose", "t"})
_QK_REDUCE_OPS = frozenset({
    "qk_reduce",
    "fp8_sparse_mla_qk_reduce",
    "fp8_sparse_mla_indexed_qk_reduce",
    "sparse_mla_qk_reduce",
})


def _is_scalar_like(value: object) -> bool:
    shape = getattr(value, "shape", None)
    if shape is None:
        return True
    try:
        return tuple(shape) in {(), (1,)}
    except TypeError:
        return False


def _scalar_operand(payload: Tuple[object, ...]) -> bool:
    if len(payload) < 3:
        return False
    lhs = payload[1]
    rhs = payload[2]
    return _is_scalar_like(lhs) or _is_scalar_like(rhs)


def _match_fused_linear(
    region: Sequence[OpTraceEntry],
    start: int,
) -> Optional[MatchResult]:
    if start + 1 >= len(region):
        return None
    op0 = region[start][0]
    op1 = region[start + 1][0]
    if op0 in _LINEAR_OPS and op1 in _LINEAR_EPILOGUES:
        end = start + 2
        return ("fused_linear", list(region[start:end]), end)
    return None


def _match_gemm_softmax_with_transpose(
    region: Sequence[OpTraceEntry],
    start: int,
) -> Optional[MatchResult]:
    if start + 2 >= len(region):
        return None
    op0, op1, op2 = region[start][0], region[start + 1][0], region[start + 2][0]
    if op0 in _TRANSPOSE_OPS and op1 in _LINEAR_OPS and op2 == "softmax":
        end = start + 3
        return ("gemm_softmax", list(region[start:end]), end)
    return None


def _match_softmax_epilogue(
    region: Sequence[OpTraceEntry],
    start: int,
) -> Optional[MatchResult]:
    if start + 1 >= len(region):
        return None
    op0 = region[start][0]
    op1 = region[start + 1][0]
    if op0 in _LINEAR_OPS and op1 == "softmax":
        end = start + 2
        return ("softmax_epilogue", list(region[start:end]), end)
    return None


def _match_qk_reduce_sm_scale(
    region: Sequence[OpTraceEntry],
    start: int,
) -> Optional[MatchResult]:
    if start + 1 >= len(region):
        return None
    op0 = region[start][0]
    op1, payload1 = region[start + 1]
    if op0 in _QK_REDUCE_OPS and op1 == "mul" and _scalar_operand(payload1):
        end = start + 2
        return ("qk_reduce_sm_scale", list(region[start:end]), end)
    return None


FUSION_PATTERNS: Tuple[FusionPattern, ...] = (
    FusionPattern("gemm_softmax", _match_gemm_softmax_with_transpose),
    FusionPattern("softmax_epilogue", _match_softmax_epilogue),
    FusionPattern("qk_reduce_sm_scale", _match_qk_reduce_sm_scale),
    FusionPattern("fused_linear", _match_fused_linear),
)


def try_match(
    region: Sequence[OpTraceEntry],
    start: int,
) -> Optional[MatchResult]:
    """Return the first pattern match at ``start`` and record the hit."""

    for pattern in FUSION_PATTERNS:
        match = pattern.matcher(region, start)
        if match is None:
            continue
        name, captured, end = match
        _FUSION_HITS[name] = _FUSION_HITS.get(name, 0) + 1
        return name, captured, end
    return None
