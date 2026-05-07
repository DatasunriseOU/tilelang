"""TileLang TIR transform pipeline driver for the Triton frontend.

Real (initial) implementation of the Tier-1 pass list (vector_add,
softmax, matmul). The list is a strict subset of
``tilelang/transform/__init__.py`` plus the standard TVM lowering tail
used by ``tilelang/engine/phase.py`` -- only passes meaningful for
PrimFuncs that just came out of TTIR lowering are included.

RFC ``RFC_unified_fused_kernel.md`` section 3 categorizes every pass as
**reuse / extend / build**. We mirror that shape here per pass with a
status tag plus rationale; ``status="skip"`` passes are dropped from
the materialized Sequential.

Each :class:`PassEntry` also carries a ``role`` string that maps the
pass to one of the Tier-1 stages from the RFC:

    "load_store"  -- masked load/store lowering (tt.load/tt.store)
    "layout"      -- per-target layout inference and tile-op lowering
    "fusion"      -- pipelining, async-copy, fence/barrier handling
    "codegen"     -- buffer flattening, intrinsic resolution, host wrap
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple

__all__ = [
    "PassStatus",
    "PassRole",
    "PassEntry",
    "PASS_ORDER",
    "build_pipeline",
    "run",
]


PassStatus = str  # Literal["reuse", "extend", "skip"] in real impl.
PassRole = str    # Literal["load_store", "layout", "fusion", "codegen"]


@dataclass(frozen=True)
class PassEntry:
    """A single pipeline entry: factory, status, role, rationale.

    ``factory`` is a zero-arg callable returning a ``tvm.transform.Pass``.
    Wrapping the actual ``tilelang.transform.<Name>`` getter lets us
    defer the import of TileLang transforms until ``build_pipeline`` is
    actually invoked, so the module can be inspected without TVM.
    """

    name: str
    status: PassStatus
    role: PassRole
    note: str
    factory: Optional[Callable[[], Any]] = None


def _lazy(name: str, *args: Any) -> Callable[[], Any]:
    """Return a zero-arg factory that imports ``tilelang.transform.<name>``.

    Imports happen on call so ``import poc.triton_frontend.pipeline``
    succeeds without TVM/TileLang installed.
    """

    def _factory() -> Any:
        import tilelang.transform as tlt  # noqa: WPS433
        fn = getattr(tlt, name)
        return fn(*args) if args else fn()

    return _factory


def _lazy_thread_sync(scope: str) -> Callable[[], Any]:
    """ThreadSync needs a storage_scope; bind it now, import later."""
    return _lazy("ThreadSync", scope)


# ---------------------------------------------------------------------------
# Pass order for Tier-1 kernels
#
# Tier 1 = vector_add, softmax, matmul (RFC 5.5 first three).
# Layout inference and LowerTileOp do most of the heavy lifting; the
# rest is the standard TVM device-lowering tail.
# ---------------------------------------------------------------------------

PASS_ORDER: Tuple[PassEntry, ...] = (
    # ---- High-level scheduling / planning -------------------------------
    PassEntry("ClusterPlanning",          "skip",   "fusion",
              "Hopper cluster API; gate on TMA fallback path (RFC 5.4).",
              factory=_lazy("ClusterPlanning")),
    PassEntry("PipelinePlanning",         "reuse",  "fusion",
              "Software-pipelining annotations carry over from tt.async_copy.",
              factory=_lazy("PipelinePlanning")),
    PassEntry("InstructionAnnotation",    "reuse",  "layout",
              "Tag tile ops with kind labels; no Triton specifics.",
              factory=_lazy("InstructionAnnotation")),
    PassEntry("LayoutInference",          "reuse",  "layout",
              "Re-derives layouts per target -- this is exactly RFC 5.2's plan.",
              factory=_lazy("LayoutInference")),
    PassEntry("LowerTileOp",              "reuse",  "layout",
              "Lowers T.gemm / T.copy / T.reduce_* emitted by op_mapping.",
              factory=_lazy("LowerTileOp")),

    # ---- Pipelining / async / barriers ----------------------------------
    PassEntry("InjectSoftwarePipeline",   "reuse",  "fusion",
              "Honors pipeline_stage annotations from async_copy emitter.",
              factory=_lazy("InjectSoftwarePipeline")),
    PassEntry("LowerHopperIntrin",        "extend", "fusion",
              "WGMMA/TMA mapping; only on NV; gate via target.",
              factory=_lazy("LowerHopperIntrin")),
    PassEntry("ThreadSync(shared)",       "reuse",  "fusion",
              "Standard barrier insertion for shared scope.",
              factory=_lazy_thread_sync("shared")),
    PassEntry("ThreadSync(shared.dyn)",   "reuse",  "fusion",
              "Dynamic shared scope companion to the above.",
              factory=_lazy_thread_sync("shared.dyn")),
    PassEntry("FuseMBarrierArriveExpectTx", "extend", "fusion",
              "Hopper-specific; reuse with TMA fallback gating.",
              factory=_lazy("FuseMBarrierArriveExpectTx")),
    PassEntry("InjectFenceProxy",         "reuse",  "fusion",
              "Memory-fence insertion for async copies.",
              factory=_lazy("InjectFenceProxy")),

    # ---- Loop / control transforms (load/store predicates) --------------
    PassEntry("IfStmtBinding",            "reuse",  "load_store",
              "Bind tt.where-derived predicates to loops.",
              factory=_lazy("IfStmtBinding")),
    PassEntry("MergeIfStmt",              "reuse",  "load_store",
              "Cleanup after masked load/store predicates.",
              factory=_lazy("MergeIfStmt")),
    PassEntry("LoopUnswitching",          "reuse",  "load_store",
              "Hoist mask predicates out of inner loops.",
              factory=_lazy("LoopUnswitching")),
    PassEntry("LegalizeVectorizedLoop",   "reuse",  "load_store",
              "Generic; required by VectorizeLoop downstream.",
              factory=_lazy("LegalizeVectorizedLoop")),
    PassEntry("LegalizeSafeMemoryAccess", "reuse",  "load_store",
              "Check load/store bounds; benefits from PtrAnalysis facts.",
              factory=_lazy("LegalizeSafeMemoryAccess")),
    PassEntry("VectorizeLoop",            "reuse",  "load_store",
              "Standard tail-vectorization.",
              factory=_lazy("VectorizeLoop")),

    # ---- Async-copy lowering (target-specific) --------------------------
    PassEntry("LowerPTXAsyncCopy",        "extend", "fusion",
              "Triton's cp.async lowers here on NV; HIP/Metal fall through.",
              factory=_lazy("LowerPTXAsyncCopy")),

    # ---- Buffer / pointer lowering --------------------------------------
    PassEntry("LowerAccessPtr",           "reuse",  "codegen",
              "Pointer-cast normalization.",
              factory=_lazy("LowerAccessPtr")),
    PassEntry("FlattenBuffer",            "reuse",  "codegen",
              "Required before MakePackedAPI.",
              factory=_lazy("FlattenBuffer")),
    PassEntry("PlanAndUpdateBufferAllocationLocation", "reuse", "codegen",
              "Buffer placement; uses scope from op_mapping.",
              factory=_lazy("PlanAndUpdateBufferAllocationLocation")),
    PassEntry("StorageRewrite",           "reuse",  "codegen",
              "Standard TVM buffer reuse.",
              factory=_lazy("StorageRewrite")),
    PassEntry("LowerOpaqueBlock",         "reuse",  "codegen",
              "TIR block opacity lowering.",
              factory=_lazy("LowerOpaqueBlock")),

    # ---- Intrinsic / device kernel finalization -------------------------
    PassEntry("LowerIntrin",              "reuse",  "codegen",
              "Final intrinsic resolution per target.",
              factory=_lazy("LowerIntrin")),
    PassEntry("LowerThreadAllreduce",     "reuse",  "codegen",
              "tt.reduce path lowers to allreduce on cross-warp axes.",
              factory=_lazy("LowerThreadAllreduce")),
    PassEntry("LowerDeviceKernelLaunch",  "reuse",  "codegen",
              "Generates the kernel-launch wrapper.",
              factory=_lazy("LowerDeviceKernelLaunch")),

    # ---- Host-side wrap-up ----------------------------------------------
    PassEntry("AnnotateDeviceRegions",    "reuse",  "codegen", "Standard.",
              factory=_lazy("AnnotateDeviceRegions")),
    PassEntry("SplitHostDevice",          "reuse",  "codegen", "Standard.",
              factory=_lazy("SplitHostDevice")),
    PassEntry("MakePackedAPI",            "reuse",  "codegen", "Standard.",
              factory=_lazy("MakePackedAPI")),
    PassEntry("CombineContextCall",       "reuse",  "codegen", "Standard.",
              factory=_lazy("CombineContextCall")),
)


# ---------------------------------------------------------------------------
# Target-aware filter
# ---------------------------------------------------------------------------

# Passes that are NV-only (Hopper / cp.async). Skipped for HIP/Metal/CPU.
_NV_ONLY: frozenset = frozenset({
    "LowerHopperIntrin",
    "LowerPTXAsyncCopy",
    "FuseMBarrierArriveExpectTx",
    "ClusterPlanning",
})


# Minimal pass set sufficient to lower Tier-1 conformance kernels
# (vector_add / softmax / matmul). All other entries in PASS_ORDER are
# either NV/Hopper-only, Blackwell/2SM/TCGEN05 noise, or fusion
# nice-to-haves -- harmless but unnecessary for the first three.
_TIER1_SUBSET: frozenset = frozenset({
    "InstructionAnnotation",
    "LayoutInference",
    "LowerTileOp",
    "InjectSoftwarePipeline",
    "ThreadSync(shared)",
    "ThreadSync(shared.dyn)",
    "IfStmtBinding",
    "MergeIfStmt",
    "LoopUnswitching",
    "LegalizeVectorizedLoop",
    "VectorizeLoop",
    "FlattenBuffer",
    "PlanAndUpdateBufferAllocationLocation",
    "StorageRewrite",
    "LowerOpaqueBlock",
    "LowerIntrin",
    "LowerThreadAllreduce",
    "LowerDeviceKernelLaunch",
    "AnnotateDeviceRegions",
    "SplitHostDevice",
    "MakePackedAPI",
    "CombineContextCall",
})


def _is_nv(target: Optional[str]) -> bool:
    """True iff ``target`` is a CUDA target (string match keeps it simple)."""
    if target is None:
        return False
    target = target.lower()
    return target.startswith("cuda") or target.startswith("nvidia") or target.startswith("nvptx")


def build_pipeline(
    target: Optional[str] = None,
    *,
    enable_tma: bool = False,
    enable_warp_specialization: bool = False,
    tier1_only: bool = True,
) -> Any:
    """Materialize a ``tvm.transform.Sequential`` for ``target``.

    Filters :data:`PASS_ORDER` by:

    1. ``status != "skip"``.
    2. NV-only passes dropped when ``target`` is HIP / Metal / CPU.
    3. Hopper passes (``LowerHopperIntrin``, ``FuseMBarrierArriveExpectTx``,
       ``ClusterPlanning``) only enabled when ``enable_tma`` /
       ``enable_warp_specialization`` are set.
    4. When ``tier1_only`` is True (default), passes outside
       :data:`_TIER1_SUBSET` are dropped. The subset is sufficient for
       vector_add / softmax / matmul lowering and skips Blackwell / 2SM /
       TCGEN05 noise that the generic ``tilelang.transform`` registry
       would otherwise pull in.

    Returns a ``tvm.transform.Sequential`` ready to apply via
    ``seq(IRModule)`` or to slot into ``tilelang.engine.phase``.
    """
    import tvm  # noqa: WPS433 (intentional lazy import)
    from tvm.transform import Sequential  # noqa: WPS433

    nv = _is_nv(target)
    passes: List[Any] = []
    for entry in PASS_ORDER:
        if entry.status == "skip":
            # ClusterPlanning is currently always "skip" but we keep the
            # gate explicit for forward compat.
            if entry.name == "ClusterPlanning" and enable_warp_specialization and nv:
                pass  # fall through to enable
            else:
                continue
        if entry.name in _NV_ONLY and not nv:
            continue
        if entry.name == "LowerHopperIntrin" and not enable_tma:
            continue
        if entry.name == "FuseMBarrierArriveExpectTx" and not enable_tma:
            continue
        if tier1_only and entry.name not in _TIER1_SUBSET:
            continue
        if entry.factory is None:  # pragma: no cover -- safety
            continue
        passes.append(entry.factory())
    return Sequential(passes, name="TritonFrontendTier1")


def run(prim_func: Any, target: Optional[str] = None, **kwargs: Any) -> Any:
    """Apply :func:`build_pipeline` to a TileLang ``PrimFunc``.

    Wraps ``prim_func`` in an ``IRModule`` first (Sequential operates on
    modules); the result is the lowered ``IRModule``.
    """
    import tvm  # noqa: WPS433
    from tvm.ir import IRModule  # noqa: WPS433

    seq = build_pipeline(target, **kwargs)
    if isinstance(prim_func, IRModule):
        mod = prim_func
    else:
        mod = IRModule({"main": prim_func})
    return seq(mod)
