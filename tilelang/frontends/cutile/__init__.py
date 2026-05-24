"""CUTile static-source frontend for TileLang.

Implements Phase 5 RFC: lower an NVIDIA CUTile IR parser into a TileLang tir.PrimFunc.
"""

from __future__ import annotations

from .lowering import (
    CuTileLoweringError,
    CuTileKernelSignature,
    compile_cutile_source,
    from_cutile_source,
)

__all__ = [
    "CuTileLoweringError",
    "CuTileKernelSignature",
    "from_cutile_source",
    "compile_cutile_source",
]
