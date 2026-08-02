from __future__ import annotations

from tilelang.tileop.gemm.registry import register_gemm_impl
from tilelang.tileop.gemm.gemm_metal import (
    GEMM_INST_METAL,
    GEMM_INST_METAL_SIMDGROUP,
    GEMM_INST_METAL_COOPERATIVE_TENSOR,
    GemmMetal,
    GemmMetalSimdGroup,
)
from tilelang.utils.target import target_is_metal


def _match_metal(target) -> bool:
    return target_is_metal(target)


# Legacy 8x8 simdgroup-matrix path (M1+).
register_gemm_impl(GEMM_INST_METAL_SIMDGROUP, GEMM_INST_METAL, _match_metal, GemmMetalSimdGroup)
# Metal M5+ mpp::tensor_ops::matmul2d cooperative-tensor path
# (PR tile-ai/tilelang#2252).  Selected automatically by
# src/backend/metal/op/gemm.cc when shape/scope preconditions hold.
register_gemm_impl(GEMM_INST_METAL_COOPERATIVE_TENSOR, GEMM_INST_METAL_COOPERATIVE_TENSOR, _match_metal, GemmMetal)
