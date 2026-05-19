from __future__ import annotations
import collections
import tvm
from tvm import tir, IRModule
from tvm.target import Target
import tilelang
from tilelang.transform import PassContext
from tilelang.contrib.nvcc import have_tma, have_pdl


# CPPMEGA fix-round-2 (MED perf): per-compile sentinel so the Z3 prover
# cache clear runs once per (LowerAndLegalize + OptimizeForTarget) pair
# instead of twice. Keyed by id(IRModule) — short-lived, uniquely
# identifies the in-flight compile. The deque is bounded by consume-on-read
# in OptimizeForTarget so a stale id (after GC) never spuriously skips.
#
# BUG-PIPE-1 fix: use a bounded deque (max 64 entries) instead of an
# unbounded set. If LowerAndLegalize is called without a matching
# OptimizeForTarget (abnormal termination, exception), old entries are
# evicted FIFO instead of leaking forever.
_Z3_CLEARED_COMPILE_IDS: collections.deque = collections.deque(maxlen=64)


def _mark_z3_cleared_for_compile(mod: IRModule) -> None:
    _Z3_CLEARED_COMPILE_IDS.append(id(mod))


def _consume_z3_cleared_for_compile(mod: IRModule) -> bool:
    """Return True if this mod was already cleared, and drop the marker."""
    key = id(mod)
    try:
        _Z3_CLEARED_COMPILE_IDS.remove(key)
        return True
    except ValueError:
        return False


def allow_warp_specialized(pass_ctx: PassContext | None = None, target: Target | None = None) -> bool:
    # avoid circular import
    from tilelang.jit.adapter.utils import is_cuda_target

    if pass_ctx is None:
        pass_ctx = tilelang.transform.get_pass_context()
    if (not is_cuda_target(target)) or (not have_tma(target)):
        return False
    disable_warp_specialized = pass_ctx.config.get("tl.disable_warp_specialized", False)
    return not disable_warp_specialized


def module_has_tma(mod: IRModule) -> bool:
    """Check if any function in the module was lowered with TMA operations.

    This reads the ``tl.has_tma`` attribute set by ``LowerTileOp`` during
    ``LowerAndLegalize``, which is the source of truth for whether TMA
    copies were actually generated.
    """
    return any(func.attrs and func.attrs.get("tl.has_tma", False) for _, func in mod.functions.items())


def allow_vectorize(pass_ctx: PassContext | None = None) -> bool:
    if pass_ctx is None:
        pass_ctx = tilelang.transform.get_pass_context()
    disable_vectorize = pass_ctx.config.get("tir.disable_vectorize", False)
    return not disable_vectorize


def allow_tir_cse(pass_ctx: PassContext | None = None) -> bool:
    if pass_ctx is None:
        pass_ctx = tilelang.transform.get_pass_context()
    return not bool(pass_ctx.config.get(tilelang.PassConfigKey.TIR_DISABLE_CSE, False))


def _apply_metal_hoist_expression(mod: IRModule, pass_ctx: PassContext | None = None) -> IRModule:
    """Run HoistExpression in Metal cleanup without duplicating large flattened bodies."""

    hoist_config = {}
    if pass_ctx is not None:
        raw_config = pass_ctx.config.get("s_tir.HoistExpression")
        if isinstance(raw_config, dict):
            hoist_config.update(raw_config)
        elif raw_config is not None:
            return tir.transform.HoistExpression()(mod)

    hoist_config.setdefault("max_hoisted_conditionals_per_scope", 0)
    with tvm.transform.PassContext(
        opt_level=3,
        config={"s_tir.HoistExpression": hoist_config},
    ):
        return tir.transform.HoistExpression()(mod)


def apply_metal_scalar_pipeline(
    mod: IRModule, target: Target, pass_ctx: PassContext | None = None
) -> IRModule:
    if target.kind.name != "metal" or not allow_tir_cse(pass_ctx):
        return mod
    mod = tilelang.transform.BindMetalScalarIntrinsics()(mod)
    mod = tir.transform.CommonSubexprElim()(mod)
    mod = tilelang.transform.BindMetalScalarIntrinsics()(mod)
    mod = _apply_metal_hoist_expression(mod, pass_ctx)
    mod = tilelang.transform.BindMetalScalarIntrinsics()(mod)
    mod = tir.transform.CommonSubexprElim()(mod)
    mod = tilelang.transform.BindMetalScalarIntrinsics()(mod)
    return mod


def allow_global_thread_synchronization(pass_ctx: PassContext | None = None) -> bool:
    if pass_ctx is None:
        pass_ctx = tilelang.transform.get_pass_context()
    enable_global_thread_sync = pass_ctx.config.get("tir.detect_global_barrier", False)
    return enable_global_thread_sync


def should_enable_aggressive_merge(pass_ctx: PassContext | None = None, target: Target | None = None) -> bool:
    if pass_ctx is None:
        pass_ctx = tilelang.transform.get_pass_context()
    enable_aggressive_merge = bool(pass_ctx.config.get(tilelang.PassConfigKey.TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE, False))
    if allow_warp_specialized(pass_ctx=pass_ctx, target=target):
        # This is a workaround to avoid the bug in the MergeSharedMemoryAllocations pass
        # when warp specialization is enabled, as different warp threads may access different
        # buffers, but the liveness analysis is hard because we need to do pipeline.
        enable_aggressive_merge = False
    return enable_aggressive_merge


def should_force_let_inline(pass_ctx: PassContext | None = None) -> bool:
    if pass_ctx is None:
        pass_ctx = tilelang.transform.get_pass_context()
    return bool(pass_ctx and pass_ctx.config.get(tilelang.PassConfigKey.TL_FORCE_LET_INLINE, False))


def should_enable_ast_print(pass_ctx: PassContext | None = None) -> bool:
    if pass_ctx is None:
        pass_ctx = tilelang.transform.get_pass_context()
    return bool(pass_ctx and pass_ctx.config.get(tilelang.PassConfigKey.TL_AST_PRINT_ENABLE, False))


def should_enable_layout_visual(pass_ctx: PassContext | None = None) -> bool:
    if pass_ctx is None:
        pass_ctx = tilelang.transform.get_pass_context()
    enabled = pass_ctx.config.get(tilelang.PassConfigKey.TL_LAYOUT_VISUALIZATION_ENABLE, False)
    return enabled


def should_enable_race_check(pass_ctx: PassContext | None = None) -> bool:
    if pass_ctx is None:
        pass_ctx = tilelang.transform.get_pass_context()
    enabled = not pass_ctx.config.get(tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK, False)
    return enabled


def should_enable_prelower_semantic_check(pass_ctx: PassContext | None = None) -> bool:
    if pass_ctx is None:
        pass_ctx = tilelang.transform.get_pass_context()
    enabled = not pass_ctx.config.get(tilelang.PassConfigKey.TL_DISABLE_PRELOWER_SEMANTIC_CHECK, False)
    return enabled


def get_layout_visual_formats(pass_ctx: PassContext | None = None) -> list[str]:
    if pass_ctx is None:
        pass_ctx = tilelang.transform.get_pass_context()
    formats_value = pass_ctx.config.get(tilelang.PassConfigKey.TL_LAYOUT_VISUALIZATION_FORMATS, "")
    if not formats_value:
        return ["txt"]

    formats_str = formats_value.strip().lower()
    valid_formats = ["txt", "png", "pdf", "svg", "all"]

    if formats_str == "all":
        return ["txt", "png", "pdf", "svg"]

    if "," in formats_str:
        formats_list = [f.strip() for f in formats_str.split(",")]
    else:
        formats_list = [formats_str]

    invalid_formats = [f for f in formats_list if f not in valid_formats]
    if invalid_formats:
        raise ValueError(
            f"Invalid formats for TL_LAYOUT_VISUALIZATION_FORMATS: {invalid_formats}. "
            f"Valid formats are: {valid_formats}. "
            f"You can choose one of the valid formats or a comma-separated list of formats.(e.g., 'txt,png,pdf')"
        )
    return formats_list


def LayoutVisual(mod: IRModule) -> None:
    """Apply layout visualization pass if enabled."""
    if should_enable_layout_visual():
        formats = get_layout_visual_formats()
        tilelang.analysis.LayoutVisual(formats=formats)(mod)


def PreLowerSemanticCheck(mod: IRModule) -> None:
    """
    Check whether the module is valid before lowering. If not, raise a user-friendly error
    in Python side instead of letting the error dive into the complicated TVM/C++ stack.
    Note: This is a validation-only pipeline of passes and does not modify or return the module.
    """

    if not should_enable_prelower_semantic_check():
        return

    # Print AST for debugging purpose
    if should_enable_ast_print():
        tilelang.analysis.ASTPrinter()(mod)
    # Check if there are any invalid nested loops.
    tilelang.analysis.NestedLoopChecker()(mod)
    # Check if there are any invalid symbolic T.Parallel + fragment access.
    tilelang.analysis.FragmentLoopChecker()(mod)


def LowerAndLegalize(mod: IRModule, target: Target) -> IRModule:
    # Bind the target device information to the module
    """
    Bind target information and progressively legalize and lower frontend Tile
    IR into a form suitable for downstream optimization and codegen.

    This pass pipeline:
    - Binds the provided target to the module.
    - Legalizes frontend Tile IR into TVM-compatible constructs.
    - Simplifies expressions.
    - Configures reducer layouts and performs layout inference for fragments and shared memory.
    - Lowers high-level tile operations and L2 persistent maps.
    - Legalizes vectorized loops and inserts safety checks for memory accesses.
    - Re-simplifies to remove redundancies introduced by safety checks.
    - Attempts loop vectorization for dynamic-shaped loops.

    Parameters:
        mod (IRModule): The input IR module containing frontend Tile IR.
        target (Target): Target device information to bind into the module.

    Returns:
        IRModule: The transformed module, ready for target-specific optimization passes.
    """
    # CPPMEGA z3-stack fix-A8 (NEW-2): drop any stale per-thread Z3 prover
    # cache entries before this pass pipeline runs. The prover cache is
    # keyed by `Analyzer*`; a freed Analyzer's address can be reused by a
    # fresh Analyzer in this pass, which would otherwise inherit the
    # prior pass's memo / scope / bv-mode state. Cheap, idempotent.
    # CPPMEGA fix-round-2 (MED perf): record the clear so the matching
    # call in OptimizeForTarget is a no-op when both phases run back-to-
    # back (the common compile path).
    _z3_clear = tvm.ffi.get_global_func("tl.z3.clear_prover_cache",
                                        allow_missing=True)
    if _z3_clear is not None:
        _z3_clear()
        _mark_z3_cleared_for_compile(mod)
    # CPPMEGA: Lower TileLang's vendored `tilelang::tl_tir::LetStmt` and
    # `tilelang::tl_tir::Allocate` IR nodes to apache TIR equivalents
    # (Bind+SeqStmt, AllocBuffer+SeqStmt) BEFORE any apache TIR pass runs.
    # Apache StmtVisitor/StmtMutator/StmtFunctor have no entry for the vendored
    # nodes and will crash with "NodeFunctor calls un-registered function".
    # `tir.transform.BindTarget` below is the first apache-side pass in this
    # pipeline, so the converters must precede it. They are idempotent on
    # already-lowered IR, so re-running them later in host/device codegen is
    # safe.
    mod = tilelang.transform.LowerTileLangLetStmt()(mod)
    mod = tilelang.transform.LowerTileLangAllocate()(mod)
    mod = tir.transform.BindTarget(target)(mod)
    from tilelang.transform.fp8_late_lower import Fp8ScaledMatmulLateLower

    mod = Fp8ScaledMatmulLateLower(target)(mod)

    if should_force_let_inline():
        # Force-let inline whenever the pass config requests it.
        mod = tilelang.transform.LetInline()(mod)
    # Add wrapper for single buf store
    mod = tilelang.transform.AddWrapperForSingleBufStore()(mod)
    # Normalize negative indices to canonical non-negative form
    mod = tilelang.transform.LegalizeNegativeIndex()(mod)
    # Legalize parallel loops with dynamic bounds
    mod = tilelang.transform.LegalizeParallelLoop()(mod)
    # Verify parallel loop correctness
    if should_enable_race_check():
        mod = tilelang.transform.VerifyParallelLoop()(mod)
    # Inject assumes to speedup tvm prover
    mod = tilelang.transform.InjectAssumes()(mod)
    # Simplify the IR expressions
    mod = tilelang.transform.Simplify()(mod)
    # Set layouts for reducers
    mod = tilelang.transform.LayoutReducer()(mod)
    # Tile-level warp specialization: runs before layout inference so that
    # producer/consumer split happens at the high-level tile-op IR.
    # The pass classifies copy ops as TMA/cp.async/sync inline (no prior
    # InstructionAnnotation pass needed). Shared buffers are multi-versioned
    # internally only for functions where the WS transformation actually
    # applies.
    if allow_warp_specialized(target=target):
        mod = tilelang.transform.ProducerConsumerWarpSpecialized()(mod)
    # Lower 2SM TCGEN5MMA and related on Blackwell target (must run before
    # LayoutInference so that the use_2cta annotation is visible to infer_layout)
    mod = tilelang.transform.LowerBlackwell2SM()(mod)
    # Run pipeline planning and software-pipeline rewriting before layout
    # inference so inferred layouts see the final pipelined structure directly.
    mod = tilelang.transform.PipelinePlanning()(mod)
    mod = tilelang.transform.InjectSoftwarePipeline()(mod)
    mod = tilelang.transform.Simplify()(mod)
    # On Metal, rewrite local.fragment GEMM accumulators to metal.simdgroup
    # before layout inference (which would otherwise require a layout for them)
    from tilelang.transform.metal_fragment_to_simdgroup import MetalFragmentToSimdgroup

    mod = MetalFragmentToSimdgroup(mod)
    # Idea #9 (Z3 roadmap): detect reductions that fit a single simdgroup
    # and could be lifted off threadgroup memory. Default OFF — gated by
    # PassConfig key ``tl.simd_lift_reductions``. Detection-only for now.
    from tilelang.transform.metal_simd_lift import MetalSimdLiftReductions

    mod = MetalSimdLiftReductions(mod)
    # Infer memory layouts for fragments and shared memory
    mod = tilelang.transform.LayoutInference()(mod)
    # Visualize the layout
    LayoutVisual(mod)
    # Lower high-level tile operations to low-level operations
    mod = tilelang.transform.LowerTileOp()(mod)
    # Lower l2 persistent map
    mod = tilelang.transform.LowerL2Persistent()(mod)
    # CPPMEGA: re-run the vendored-IR converters here. ``LowerTileOp`` and the
    # tile-op chain above can re-introduce ``tilelang::tl_tir::Allocate`` /
    # ``LetStmt`` nodes after the entry-point conversion at the top of
    # ``LowerAndLegalize``. Without a re-run, the python-side
    # ``DecoupleTypeCast`` mutator (a ``tirx::PyStmtExprMutator`` subclass)
    # crashes with "NodeFunctor calls un-registered function on type
    # tilelang.Allocate" because apache's ``StmtFunctor`` vtable does not
    # know about the vendored types. The converters are idempotent on
    # already-lowered IR, so the re-run is cheap when nothing was
    # re-introduced.
    mod = tilelang.transform.LowerTileLangLetStmt()(mod)
    mod = tilelang.transform.LowerTileLangAllocate()(mod)
    # Decouple type cast vectorization constraints before vectorization
    mod = tilelang.transform.DecoupleTypeCast()(mod)
    # Legalize vectorized loops to ensure they are valid
    mod = tilelang.transform.LegalizeVectorizedLoop()(mod)
    # Add safety checks for memory accesses
    mod = tilelang.transform.LegalizeSafeMemoryAccess()(mod)
    # CPPMEGA: Z3 idea #7 — predicate fusion. Runs immediately after the
    # safe-memory-access pass (which materializes the `if(a){if(b){...}}`
    # nesting we target) and before vectorization / async-copy lowering.
    # Default OFF; opt-in via PassConfig `tl.predicate_fusion`.
    mod = tilelang.transform.PredicateFusion()(mod)
    # Lower frontend pointer metadata op to standard tvm_access_ptr
    mod = tilelang.transform.LowerAccessPtr()(mod)
    # Simplify again to clean up any duplicated conditions
    # that may have been introduced by safety checks
    # use an enhanced pass to simplify the dynamic symbolics
    # TODO(lei): return to tir pass when kSymbolicBound simplification
    # is merged into tvm.
    mod = tilelang.transform.Simplify()(mod)
    # Hoist any root-block annotations to PrimFunc attrs if pass is available
    mod = tilelang.transform.HoistNonRestrictParams()(mod)
    return mod


def OptimizeForTarget(mod: IRModule, target: Target) -> IRModule:
    pass_ctx = tilelang.transform.get_pass_context()
    # CPPMEGA z3-stack fix-A8 (NEW-2): also clear here. `LowerAndLegalize`
    # and `OptimizeForTarget` are called as separate phases; either may be
    # invoked in isolation by tools, and the per-thread Z3 prover cache
    # outlives both. Clearing at every phase entry keeps the prover state
    # scoped to the current pass invocation.
    # CPPMEGA fix-round-2 (MED perf): skip the clear if LowerAndLegalize
    # already cleared for this compile (common full-pipeline path). The
    # marker is consumed so a re-entrant compile of the same module id
    # post-GC does not skip its own clear.
    if not _consume_z3_cleared_for_compile(mod):
        _z3_clear = tvm.ffi.get_global_func("tl.z3.clear_prover_cache",
                                            allow_missing=True)
        if _z3_clear is not None:
            _z3_clear()
    # CPPMEGA: Defensive re-run of vendored-IR converters in case any TileLang
    # pass in `LowerAndLegalize` re-introduced `tilelang::tl_tir::LetStmt` or
    # `tilelang::tl_tir::Allocate` nodes. This guarantees the IR contains only
    # apache-compatible nodes before the first apache TIR pass below
    # (`tir.transform.NarrowDataType`, `tir.transform.Simplify`, etc.).
    mod = tilelang.transform.LowerTileLangLetStmt()(mod)
    mod = tilelang.transform.LowerTileLangAllocate()(mod)
    # Lower the shared.tmem into specific initialization slot
    mod = tilelang.transform.LowerSharedTmem()(mod)
    # which may be introduced by the LegalizeSafeMemoryAccess
    mod = tilelang.transform.IfStmtBinding()(mod)
    has_tma = module_has_tma(mod)
    # Pipeline barriers are now created at final expanded size by
    # InjectSoftwarePipeline, so no late MVB barrier fixup is needed.
    # Buffer allocation placement is handled uniformly for both paths.
    mod = tilelang.transform.PlanAndUpdateBufferAllocationLocation()(mod)
    # AutoDoubleBuffer: opt-in (PassConfig `tl.auto_double_buffer`, default
    # OFF). Detects canonical shared-memory tile-load patterns and (when Z3
    # can prove soundness) inserts ping-pong buffers. Currently a safe stub
    # — see src/transform/auto_double_buffer.cc.
    mod = tilelang.transform.AutoDoubleBuffer()(mod)
    mod = tilelang.transform.LowerSharedBarrier()(mod)
    if has_tma:
        mod = tilelang.transform.FuseMBarrierArriveExpectTx()(mod)
    mod = tilelang.transform.HoistGlobalBufferAllocations()(mod)
    mod = tilelang.transform.LowerOpaqueBlock()(mod)
    mod = tilelang.transform.Simplify()(mod)
    mod = tir.transform.NarrowDataType(32)(mod)

    mod = tilelang.transform.FlattenBuffer()(mod)
    # ConfigIndexBitwidth must be applied after FlattenBuffer
    # as it will flatten index computing
    mod = tilelang.transform.ConfigIndexBitwidth()(mod)
    mod = tir.transform.Simplify()(mod)

    # CPPMEGA: Z3 roadmap idea #4 — drop provable buffer-bound guards before
    # vectorization. Gated by `tl.drop_provable_bound_checks` PassConfig
    # (default OFF). See src/transform/drop_provable_bound_checks.cc.
    mod = tilelang.transform.DropProvableBoundChecks()(mod)
    mod = tilelang.transform.VectorizeLoop(enable_vectorize=allow_vectorize(pass_ctx=pass_ctx))(mod)

    mod = tilelang.transform.StorageRewrite()(mod)
    mod = tilelang.transform.LoopUnswitching()(mod)
    mod = tilelang.transform.UnrollLoop()(mod)
    mod = tir.transform.RenormalizeSplitPattern()(mod)
    mod = tir.transform.Simplify()(mod)
    mod = tir.transform.RemoveNoOp()(mod)
    mod = tir.transform.HoistIfThenElse()(mod)

    mod = tir.transform.VerifyMemory()(mod)
    mod = tir.transform.AnnotateEntryFunc()(mod)
    # TODO(lei): This is a hack to make sure the
    # thread level allreduce pass can be applied
    # in TL. As Tl only use one thread dimension
    # the var binding information will be lost
    # in the lowering process with Legalization
    # and Simplify pass.
    # We can find a way better to create var instead
    # of putting the LowerThreadAllreduce before
    # the Legalization.
    mod = tir.transform.InferFragment()(mod)
    mod = tilelang.transform.LowerThreadAllreduce()(mod)
    mod = tilelang.transform.LowerLDGSTG()(mod)
    # RFC §5.4: decompose Hopper TMA descriptor copies into pointer-arith
    # `T.copy` loops on non-Hopper targets (Apple Metal SIMDgroup, AMD HIP,
    # pre-Hopper CUDA, CPU). On NV Hopper+ this is a no-op so the
    # `LowerHopperIntrin` pass below still owns the native lowering. Slot
    # is BEFORE `LowerHopperIntrin` (which is gated on CUDA_MAJOR_VERSION
    # >= 12) and AFTER `LowerTileOp` (which produces the TMA Calls). The
    # software-pipeliner (`InjectSoftwarePipeline`) runs earlier on tile-op
    # `T.copy`, so pipelining is unaffected by this pass.
    mod = tilelang.transform.LowerTMAToPtrArith()(mod)
    mod = tilelang.transform.LowerHopperIntrin()(mod)
    # Global Barrier Synchronization must be applied before
    # SplitHostDevice pass, as the global barrier
    if allow_global_thread_synchronization():
        mod = tilelang.transform.ThreadSync("global")(mod)
    mod = tilelang.transform.AnnotateDeviceRegions()(mod)
    # CPPMEGA: SplitHostDevice uses apache `tirx::StmtMutator` which has no
    # dispatch entry for `tilelang::tl_tir::LetStmt` / `tilelang::tl_tir::Allocate`.
    # Several preceding TileLang passes (e.g. LowerThreadAllreduce, LowerLDGSTG,
    # LowerHopperIntrin, AnnotateDeviceRegions) may re-emit vendored nodes, so
    # we lower them again here to guarantee a clean IR for the apache visitor.
    mod = tilelang.transform.LowerTileLangLetStmt()(mod)
    mod = tilelang.transform.LowerTileLangAllocate()(mod)
    mod = tilelang.transform.SplitHostDevice()(mod)

    # Mark the function contains pdl_sync or pdl_trigger
    mod = tilelang.transform.MarkCudaSyncCalls(have_pdl(target))(mod)

    mod = tilelang.transform.AnnotateReadOnlyParams()(mod)
    # MergeSharedMemoryAllocations must be applied after SplitHostDevice
    # because the merged allocation site is at the beginning of each device function
    enable_aggressive_merge = should_enable_aggressive_merge(pass_ctx=pass_ctx, target=target)
    mod = tilelang.transform.MergeSharedMemoryAllocations(enable_aggressive_merge=enable_aggressive_merge)(mod)
    # InjectFenceProxy is a no-op on targets that lack the TMA / async-proxy
    # programming model; the pass itself checks the PrimFunc's target.
    mod = tilelang.transform.InjectFenceProxy()(mod)
    mod = tilelang.transform.ThreadSync("shared")(mod)
    mod = tilelang.transform.ThreadSync("shared.dyn")(mod)
    mod = tilelang.transform.MetalMergeRoundBarrierCleanup()(mod)
    # Inject conservative tcgen05 fences on Blackwell (SM100+).
    # Must run after ThreadSync so that tvm_storage_sync calls are present.
    # The pass handles shared syncs and simple linear wait/use, use/arrive
    # handoffs, and is a no-op on non-SM100 targets or functions without TMEM.
    mod = tilelang.transform.InjectTcgen05Fence()(mod)
    mod = tilelang.transform.MergeIfStmt()(mod)
    # NOTE: LowerPTXAsyncCopy is applied earlier (before PipelinePlanning).
    if allow_warp_specialized(pass_ctx=pass_ctx, target=target):
        mod = tilelang.transform.AnnotateWarpGroupRegAlloc()(mod)
    # Some late target-specific passes can introduce new internal buffers after
    # the normal flattening point.  Re-flatten before wrapping the host API so
    # local buffer elem_offset symbols do not become phantom API arguments.
    mod = tilelang.transform.FlattenBuffer()(mod)
    # Metal scalar cleanup: CSE feeds HoistExpression, then a second CSE pass
    # restores scalar bindings that HoistExpression may expand. Launch lowering
    # inlines those binds when collecting launch args.
    mod = apply_metal_scalar_pipeline(mod, target, pass_ctx)
    mod = tilelang.transform.MakePackedAPI()(mod)
    mod = tilelang.transform.Simplify()(mod)
    mod = apply_metal_scalar_pipeline(mod, target, pass_ctx)
    mod = tilelang.transform.LowerDeviceKernelLaunch()(mod)

    # Transform threadblock to persistent threadblock
    mod = tilelang.transform.PersistThreadblock()(mod)

    return mod
