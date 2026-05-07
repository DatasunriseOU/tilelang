"""TTGIR layout encodings -> TileLang scope/layout (placeholder, NOT used in MVP).

POC scaffold -- every translator raises NotImplementedError.

==============================================================================
Stance (RFC ``RFC_unified_fused_kernel.md`` section 5.2): we DO NOT INGEST
TTGIR layout encodings (``#blocked``/``#shared``/``#mma``/``#linear``).
==============================================================================

The frontend hooks Triton at the **TTIR** layer, which is *before* layout
assignment. By that contract, no encoding attribute should ever reach
this module during normal operation. TileLang's own
``src/transform/layout_inference.cc`` re-derives layouts per target (CUDA
WMMA/WGMMA, AMD MFMA/WMMA, Metal ``simdgroup_matrix``).

Reasoning:
- TTGIR layout IR has churned heavily (LinearLayout migration, PRs #6609 /
  #7777 -- RFC section 2). Tracking it is a moving target.
- Re-deriving layouts in TileLang gives portable codegen across CUDA /
  HIP / Metal "for free".
- ``microsoft/triton-shared`` made the same call.

Why this file exists anyway
---------------------------
Future-use stubs are kept here for the case where we *do* need to ingest
a TTGIR-stage kernel (e.g. for an FA-v3 reference dump that already had
layout assignment baked in -- RFC section 5.2 second bullet).

If/when that is required, fill in the stubs below. Until then they
remain unreachable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

__all__ = [
    "TileLangScope",
    "BlockedToStrided",
    "SharedToTileLangShared",
    "MmaToTileLangFragment",
    "translate",
]


# TileLang memory scopes (mirrors the values used by ``T.alloc_*``).
TileLangScope = str  # Literal["global", "shared", "shared.dyn", "local"] in real impl.


@dataclass
class BlockedToStrided:
    """Future-use: ``#blocked`` -> strided memref + thread/lane mapping.

    Per RFC section 5.2, ``#blocked`` decomposes into a strided buffer
    (recoverable by :class:`triton_frontend.ptr_analysis.PtrAnalysis`)
    plus a thread/lane mapping that TileLang represents via
    ``T.Kernel(bx, by, bz)`` and warp-level annotations.
    """

    sizes_per_thread: Tuple[int, ...] = ()
    threads_per_warp: Tuple[int, ...] = ()
    warps_per_cta: Tuple[int, ...] = ()
    order: Tuple[int, ...] = ()


@dataclass
class SharedToTileLangShared:
    """Future-use: ``#shared`` -> ``T.alloc_shared`` + swizzle annotation."""

    vec: int = 0
    per_phase: int = 0
    max_phase: int = 0
    order: Tuple[int, ...] = ()


@dataclass
class MmaToTileLangFragment:
    """Future-use: ``#mma`` -> ``T.gemm`` + target-specific fragment layout.

    Resolved per target during TileLang layout inference:
      - CUDA  -> WMMA / WGMMA
      - HIP   -> MFMA / WMMA wavefront
      - Metal -> ``simdgroup_matrix_multiply_accumulate``
    """

    version: Tuple[int, int] = (0, 0)
    warps_per_cta: Tuple[int, ...] = ()
    instr_shape: Tuple[int, ...] = ()


def translate(encoding: Any, target: Optional[str] = None) -> Any:
    """Future-use: dispatch on a TTGIR encoding attribute.

    Should never be called in the TTIR-hook path. If the call site is
    reached, treat it as a bug and raise.
    """
    raise NotImplementedError(
        "RFC section 5.2: TTGIR layout ingestion is intentionally not implemented. "
        "Frontend hooks at TTIR (pre-layout-assignment); TileLang re-derives "
        "layouts per target. If you genuinely need TTGIR ingestion, populate "
        "this module's stubs."
    )
