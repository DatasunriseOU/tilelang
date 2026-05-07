"""TileLang JIT kernel factories used by the FX -> TileLang lowerer.

RFC reference: ``RFC_unified_fused_kernel.md`` §7 Phase 2.2 (kernel
materialisation for hard-deferred ops). Kernels in this package are
imported lazily from ``fx_to_tilelang.py`` so that the lowerer module
itself stays importable without a TileLang JIT backend on PATH.
"""
