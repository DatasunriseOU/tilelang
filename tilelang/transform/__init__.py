"""Wrapping transformations."""
# pylint: disable=invalid-name, unsupported-binary-operation

from . import _ffi_api
from .simplify import Simplify, simplify_prim_func, LetInline  # noqa: F401
from .pass_config import PassConfigKey  # noqa: F401
from tilelang import tvm as tvm  # noqa: F401
from tvm import tir  # noqa: F401
from tvm.ir.transform import PassContext  # noqa: F401
from .add_bufstore_wrapper import AddWrapperForSingleBufStore  # noqa: F401
from .hoist_broadcast_values import HoistBroadcastValues  # noqa: F401
from .decouple_type_cast import DecoupleTypeCast  # noqa: F401
from .lower_extern_intrinsic import LowerExternIntrinsic  # noqa: F401
from .metal_scalar_intrinsics import BindMetalScalarIntrinsics  # noqa: F401
from .metal_merge_round import MetalMergeRoundBarrierCleanup  # noqa: F401
from .metal_simdgroup_guard import MetalSimdgroupSemanticGuard  # noqa: F401


def get_pass_context():
    """Get the current pass context"""
    return PassContext.current()


def LegalizeParallelLoop():
    return _ffi_api.LegalizeParallelLoop()  # type: ignore


def ClusterPlanning():
    """ClusterPlanning

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.ClusterPlanning()  # type: ignore


def PipelinePlanning():
    """infer the fragment/shared memory layout

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.PipelinePlanning()  # type: ignore


def InstructionAnnotation():
    """Annotate tile operations with coarse-grained instruction kind.

    This pass runs before LayoutInference and LowerTileOp.  It adds a
    ``tl_instruction_kind`` annotation to each tile-op Call node indicating
    the instruction category ("tma", "cp_async", "sync", "wgmma", etc.)
    that will be selected during lowering.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.InstructionAnnotation()  # type: ignore


def LayoutInference():
    """LayoutInference

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.LayoutInference()  # type: ignore


def LowerTileOp():
    """LowerTileOp

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.LowerTileOp()  # type: ignore


def InjectSoftwarePipeline():
    """InjectSoftwarePipeline

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.InjectSoftwarePipeline()  # type: ignore


def LegalizeNegativeIndex():
    """Legalize negative indices in buffer loads.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.LegalizeNegativeIndex()  # type: ignore


def InjectAssumes():
    """Inject Assumes for natural shape boundary conditions. And convert Assumes in Evaluate(Call(...)) form
    (tvm builtin assume call) to AttrNode form.

    Returns:
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.InjectAssumes()


def VerifyParallelLoop():
    """VerifyParallelLoop

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.VerifyParallelLoop()  # type: ignore


def LowerHopperIntrin():
    """LowerHopperIntrin

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    if hasattr(_ffi_api, "LowerHopperIntrin"):
        return _ffi_api.LowerHopperIntrin()  # type: ignore
    # BUG-PIPE-2 fix: return a proper no-op pass instead of a bare lambda.
    # A bare `lambda f: f` lacks pass metadata and breaks in pipelines that
    # call `.run()` or inspect pass properties.
    return tir.transform.Apply(lambda f: f)  # type: ignore


def LowerTMAToPtrArith():
    """LowerTMAToPtrArith.

    Decomposes Hopper-style TMA descriptor loads/stores
    (``tl::tma_load`` / ``tl::tma_store`` / ``tl::tma_load_im2col``) into
    explicit pointer-arith copy loops on non-Hopper targets (Apple Metal
    SIMDgroup, AMD HIP, pre-Hopper CUDA, CPU). NV Hopper+ paths are
    passed through unchanged so the existing ``LowerHopperIntrin`` pipeline
    keeps owning the native lowering.

    See ``src/transform/lower_tma_to_ptr_arith.cc`` for the lowering rules.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass.
    """
    if hasattr(_ffi_api, "LowerTMAToPtrArith"):
        return _ffi_api.LowerTMAToPtrArith()  # type: ignore
    # BUG-PIPE-2 fix: proper no-op pass (see LowerHopperIntrin above).
    return tir.transform.Apply(lambda f: f)  # type: ignore


def ThreadSync(storage_scope: str):
    """Insert sync between parallel read/write of shared buffers.

    Parameters
    ----------
    storage_scope: str
        The target storage scope.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.ThreadSync(storage_scope)  # type: ignore


def IfStmtBinding():
    """IfStmtBinding

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.IfStmtBinding()  # type: ignore


def MergeIfStmt():
    """MergeIfStmt

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.MergeIfStmt()  # type: ignore


def LoopUnswitching():
    """LoopUnswitching: Hoist loop-invariant if statements out of loops.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.LoopUnswitching()  # type: ignore


def DropProvableBoundChecks():
    """Drop runtime ``if (i < N) buf[i] = ...`` bound-check guards when the
    default analyzer or the vendored Z3 prover can conclusively prove the
    condition (idea #4 of the Z3 roadmap).

    Conservative-by-default: any prover error / timeout / UNKNOWN keeps the
    guard intact. Pass is gated by the ``tl.drop_provable_bound_checks``
    PassConfig (default ``False``); calling this pass without the config
    enabled is a no-op.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.DropProvableBoundChecks()  # type: ignore


def AutoDoubleBuffer():
    """AutoDoubleBuffer: Auto-detect canonical shared-memory tile-load
    patterns and (when Z3 can prove soundness) ping-pong them.

    Default OFF. Enable by setting the PassConfig
    ``tl.auto_double_buffer = True``.

    Currently a SAFE STUB: when enabled and a candidate is detected, the
    pass logs the detection and the Z3 verdict, but leaves the IR
    unchanged. This lets the wiring (PassConfig, phase slot, FFI binding)
    ship without committing to a specific transformation; a future
    iteration can replace the stub with the real ping-pong rewrite.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.AutoDoubleBuffer()  # type: ignore


def PredicateFusion():
    """PredicateFusion (Z3 idea #7): fuse adjacent guarded ``if`` statements.

    Rewrites ``if(a) { if(b) { body } }`` to ``if(a && b) { body }`` when
    Z3 proves the inner predicate is well-defined unconditionally (i.e.
    every BufferLoad/BufferStore index is in-range without assuming the
    outer guard ``a``). Conservative on UNKNOWN/timeout — keeps nesting.

    Controlled by pass config ``tl.predicate_fusion`` (default OFF).

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass.
    """
    return _ffi_api.PredicateFusion()  # type: ignore


def ProducerConsumerWarpSpecialized():
    """Producer-consumer warp specialization at the tile-op level.

    This pass runs before LayoutInference and LowerTileOp. It rewrites
    eligible pipelined tile-op loops into warp-specialized producer and
    consumer branches with explicit barrier synchronization.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.ProducerConsumerWarpSpecialized()  # type: ignore


def ProducerConsumerWarpSpecializedTiled():
    """Compatibility alias for ``ProducerConsumerWarpSpecialized``.

    The tiled tile-op implementation is now the canonical
    ``ProducerConsumerWarpSpecialized`` pass.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return ProducerConsumerWarpSpecialized()


def AnnotateWarpGroupRegAlloc():
    """Inject set_max_nreg calls into warp-specialized functions.

    This pass analyzes the function to collect register hints from set_max_nreg
    and no_set_max_nreg calls, then injects appropriate set_max_nreg calls into
    producer and consumer branches of warp-specialized code.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.AnnotateWarpGroupRegAlloc()  # type: ignore


def FuseMBarrierArriveExpectTx():
    """Fuse simple expect_tx -> TMA issue -> arrive back into arrive_and_expect_tx."""
    return _ffi_api.FuseMBarrierArriveExpectTx()  # type: ignore


def InjectFenceProxy():
    """InjectFenceProxy

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.InjectFenceProxy()  # type: ignore


def InjectTcgen05Fence():
    """Inject tcgen05.fence::before_thread_sync / after_thread_sync at
    conservative TCGEN05/TMEM synchronization boundaries on Blackwell
    (SM100+) targets.

    The current pass wraps CTA-wide shared-memory syncs and also inserts
    fences around linear mbarrier wait/use and use/arrive handoff patterns.
    It is intentionally conservative and does not try to infer arbitrary
    barrier protocols.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.InjectTcgen05Fence()  # type: ignore


def LegalizeVectorizedLoop():
    """LegalizeLoopVectorize

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.LegalizeVectorizedLoop()  # type: ignore


def LegalizeSafeMemoryAccess():
    """LegalizeLoopVectorize

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.LegalizeSafeMemoryAccess()  # type: ignore


def LowerAccessPtr():
    """Lower TileLang frontend `tl.access_ptr` to `tir.builtin.tvm_access_ptr`.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.LowerAccessPtr()  # type: ignore


def MakePackedAPI():
    """MakePackedAPI

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.MakePackedAPI()  # type: ignore


def AnnotateDeviceRegions():
    """AnnotateDeviceRegions

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.AnnotateDeviceRegions()  # type: ignore


def SplitHostDevice():
    """Split host/device functions even for empty kernels.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.SplitHostDevice()  # type: ignore


def AnnotateReadOnlyParams():
    """Annotate read-only handle parameters for PrimFuncs.

    Adds attribute `tl.readonly_param_indices` listing param indices that are
    never written, enabling CUDA codegen to emit `const` qualifiers to unlock
    read-only cache loads.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.AnnotateReadOnlyParams()  # type: ignore


def VectorizeLoop(enable_vectorize: bool = True):
    """VectorizeLoop

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.VectorizeLoop(enable_vectorize)  # type: ignore


def LowerPTXAsyncCopy():
    """Lower eligible global->shared copies into PTX `cp.async` on CUDA.

    When enabled (pass config `tl.enable_async_copy`, default True), this pass
    may rewrite plain user-written global->shared `BufferStore` patterns (e.g.
    SIMT copies in `T.Parallel`) into `tir.ptx_cp_async`, and insert
    `tir.ptx_commit_group` + `tir.ptx_wait_group(0)` to preserve synchronous
    semantics for normal stores. If explicit commit/wait intrinsics already
    exist, the pass avoids duplicating them (and may insert a missing commit
    immediately before an existing wait to cover injected `cp.async`).

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.LowerPTXAsyncCopy()  # type: ignore


def InjectPTXAsyncCopy():
    """Deprecated alias of `LowerPTXAsyncCopy`."""
    return LowerPTXAsyncCopy()


def LowerDeviceStorageAccessInfo():
    """Lower attached storage access information on device.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass

    Note
    ----
    Run this pass after all storage access analysis finish.
    """
    return _ffi_api.LowerDeviceStorageAccessInfo()  # type: ignore


def LowerTileLangLetStmt():
    """Lower TileLang's vendored ``tilelang.LetStmt`` IR node into the
    tirx-equivalent ``SeqStmt({tirx::Bind(var, value), body})``.

    apache/tvm renamed ``tir::LetStmt`` to ``tirx::Bind`` and removed the
    ``body`` field. TileLang vendors the legacy 3-arg ``LetStmt(var, value,
    body)`` node under the type key ``tilelang.LetStmt`` so the existing
    transform/op code can keep working without per-site rewrites. This pass
    converts those nodes to apache-compatible IR and MUST run before any
    apache/tvm tirx pass touches the IR.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass. No-op if no ``tilelang.LetStmt`` nodes are found.
    """
    return _ffi_api.LowerTileLangLetStmt()  # type: ignore


def LowerTileLangAllocate():
    """Lower TileLang's vendored ``tilelang.Allocate`` IR node into the
    tirx-equivalent ``SeqStmt({AllocBuffer(buffer), body})`` (optionally
    wrapped in ``IfThenElse(condition, ...)`` when the predicate is
    non-trivial).

    apache/tvm replaced the legacy 6-field
    ``Allocate(buffer_var, dtype, extents, condition, body, annotations)``
    stmt with the body-less ``AllocBuffer(Buffer, annotations)``. TileLang
    vendors the legacy node so its many call sites compile unchanged; this
    pass converts those nodes to apache-compatible IR and MUST run before
    any apache/tvm tirx pass touches the IR.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass. No-op if no ``tilelang.Allocate`` nodes are found.
    """
    return _ffi_api.LowerTileLangAllocate()  # type: ignore


def CombineContextCall():
    """Combine context calls in the host module.

    Vendored TileLang implementation registered as
    ``tl.transform.CombineContextCall`` (see
    ``src/transform/combine_context_call.cc``). Replaces the upstream
    ``tir.transform.CombineContextCall`` that was removed in apache/tvm
    after the tirx refactor.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.CombineContextCall()  # type: ignore


def ConfigIndexBitwidth():
    """Config index bitwidth.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    ----
    """
    return _ffi_api.ConfigIndexBitwidth()  # type: ignore


def FlattenBuffer():
    """FlattenBuffer

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.FlattenBuffer()  # type: ignore


def MergeSharedMemoryAllocations(enable_aggressive_merge: bool = False, align_bytes: int = 16, disable_reuse: bool = False):
    """MergeSharedMemoryAllocations

    Parameters
    ----------
    enable_aggressive_merge : bool
        Aggressively merge shared-memory allocations.
    align_bytes : int
        Shared-memory allocation alignment.
    disable_reuse : bool
        Disable shared-memory reuse/aliasing (upstream PR #2228). NOTE: our
        vendored C++ MergeSharedMemoryAllocations pass predates the upstream
        per-epoch-liveness rewrite (#2185/#2281) that introduced the
        disable_reuse plumbing, so this flag is accepted for call-site
        compatibility (the backend-aware CUDA pipeline from #2189 passes it)
        but is currently a no-op — reuse stays enabled, matching pre-#2228
        behavior. Wiring it through requires porting the rewritten pass.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    if disable_reuse:
        import warnings

        warnings.warn(
            "MergeSharedMemoryAllocations(disable_reuse=True) is not yet "
            "honored: the vendored C++ pass predates the upstream "
            "per-epoch-liveness rewrite (#2185/#2228). Shared-memory reuse "
            "remains enabled.",
            RuntimeWarning,
            stacklevel=2,
        )
    return _ffi_api.MergeSharedMemoryAllocations(enable_aggressive_merge, align_bytes)  # type: ignore


def LowerL2Persistent():
    """LowerL2Persistent"""
    return _ffi_api.LowerL2Persistent()  # type: ignore


def PersistThreadblock():
    """PersistThreadblock"""
    return _ffi_api.PersistThreadblock()  # type: ignore


def MarkCudaSyncCalls(have_pdl: bool = False):
    """MarkCudaSyncCalls"""
    return _ffi_api.MarkCudaSyncCalls(have_pdl)  # type: ignore


def LowerSharedBarrier():
    """LowerSharedBarrier"""
    return _ffi_api.LowerSharedBarrier()  # type: ignore


def PlanAndUpdateBufferAllocationLocation():
    """Plan and update buffer allocation locations within PrimFuncs.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.PlanAndUpdateBufferAllocationLocation()  # type: ignore


def HoistGlobalBufferAllocations():
    """Hoist global buffer allocations to the top of the block (host side).

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.HoistGlobalBufferAllocations()  # type: ignore


def HoistNonRestrictParams():
    return _ffi_api.HoistNonRestrictParams()  # type: ignore


def StorageRewrite():
    """StorageRewrite

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.StorageRewrite()  # type: ignore


def LowerOpaqueBlock():
    """LowerOpaqueBlock"""
    return _ffi_api.LowerOpaqueBlock()  # type: ignore


def LowerThreadAllreduce():
    """LowerThreadAllreduce"""
    return _ffi_api.LowerThreadAllreduce()  # type: ignore


def LowerIntrin():
    """LowerIntrin"""
    return _ffi_api.LowerIntrin()  # type: ignore


def LowerDeviceKernelLaunch():
    """
    Create and return a transform pass that lowers device kernel launch constructs to target-specific IR.

    This pass transforms high-level device kernel launch and related intrinsics into lower-level
    IR suitable for backend code generation and device-side lowering.

    Returns:
        tvm.transform.Pass: The transform pass that performs device kernel launch lowering.
    """
    return _ffi_api.LowerDeviceKernelLaunch()  # type: ignore


def LowerSharedTmem():
    """LowerSharedTmem"""
    return _ffi_api.LowerSharedTmem()  # type: ignore


def LayoutReducer():
    """
    Return a TVM transform pass that performs layout reduction/normalization.

    This wrapper delegates to the underlying FFI implementation and returns a pass object suitable for use in a PassContext or pass pipeline. The pass is intended to simplify or reduce tensor/layout-related representations during relay/tile transformations.

    Returns:
        The transform pass object produced by the FFI backend.
    """
    return _ffi_api.LayoutReducer()  # type: ignore


def UnrollLoop():
    """Unroll loops as in Halide pipeline.

    This pass unrolls loops based on configuration options including:
    - auto_max_step: Threshold of number of steps to be automatically unrolled
    - auto_max_depth: Maximum nested level of loops that can be automatically unrolled
    - auto_max_extent: Maximum extent of loop that will be unrolled
    - explicit_unroll: Whether to explicitly unroll instead of setting a pragma
    - unroll_local_access: Whether to always unroll local access

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.UnrollLoop()  # type: ignore


def LowerLDGSTG():
    """Lower Ramp-based global memory load/store to ldg/stg intrinsics.

    This pass transforms vectorized global memory loads and stores (using Ramp indices)
    into explicit ldg32/64/128/256 and stg32/64/128/256 intrinsics for better codegen.

    Key behaviors:
    - Converts Ramp-based global BufferLoad to ldg intrinsics
    - Converts Ramp-based global BufferStore to stg intrinsics
    - Supports predicated loads (if_then_else with else=0)
    - Supports predicated stores (if in then case)
    - Skips loads in async scope (will be lowered to cp.async)
    - Only enabled for CUDA targets

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.LowerLDGSTG()  # type: ignore


def LowerBlackwell2SM():
    """Lower 2SM TCGEN5MMA and related on Blackwell target

    Returns:
        fpass : tvm.transform.Pass
            The result pass
    """
    return _ffi_api.LowerBlackwell2SM()  # type: ignore
