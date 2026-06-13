"""Triton TTIR -> TileLang TIR op-by-op dispatch table.

Real implementations for all 16 ops in ``RFC_unified_fused_kernel.md``
section 5.1. Status: 16 of 16 emitters now produce TIR (those not
backed by a complete TileLang primitive raise ``NotImplementedError``
with a ``# TODO: verify`` marker; see individual emitters).

Implemented in this file:
    tt.load, tt.store, tt.atomic_rmw, tt.dot, tt.reduce, tt.where,
    tt.broadcast, tt.splat, tt.expand_dims, tt.reshape, tt.make_range,
    async_copy (and tt.async_commit / tt.async_wait), mbarrier
    (init/arrive/wait), tt.make_tensor_descriptor, tt.descriptor_load /
    _store, tt.experimental_descriptor_load / _store, tt.print.

Each ``map_tt_<name>`` is the emitter for one TTIR op. The dispatch
table :data:`OP_TABLE` is consumed by the TTIR walker invoked from
:func:`triton_frontend.from_ttir`. Emitters return a ``tvm.tir`` PrimExpr
or Stmt, with side effects on ``ctx`` (SSA -> buffer/expression map).

The walker (and the emitters) accept either a real ``mlir.ir.Operation``
or a dict-shaped fake of the form
``{"name": str, "operands": [...], "results": [...], "attrs": {...}}``;
the latter is used in unit tests to avoid an MLIR-bindings dependency.

Adding a new mapping
--------------------
1. Add ``def map_tt_<new_name>(op, ctx) -> Any: ...`` below.
2. Register it in :data:`OP_TABLE` under the exact TTIR op name.
3. Cite the RFC subsection that justifies the lowering.
4. Add a conformance kernel under :mod:`triton_frontend.conformance`
   that exercises the new op end-to-end.
"""

from __future__ import annotations

import os as _os
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# When True, force the log/exp synthesis for tt.reduce(mul) instead of the
# native reduce_prod primitive. Useful for numerical-comparison testing on
# backends that haven't validated their "mul" all-reduce path yet.
_USE_LOGEXP_PROD = False

__all__ = [
    "OP_TABLE",
    "TritonFrontendError",
    "EmitError",
    "EmitFn",
    "WalkerCtx",
    "materialize_lazy_tile",
    "should_fold_addressing",
    # Memory ops
    "map_tt_atomic_rmw",
    # Compute ops
    "map_tt_where",
    # Shape ops
    "map_tt_trans",
    # Async / barrier
    "map_tt_async_copy",
    "map_tt_mbarrier",
    "map_tt_sync_threads_partial",
    # TMA
    "map_tt_make_tensor_descriptor",
    "map_tt_descriptor_load",
    "map_tt_descriptor_store",
    "map_tt_experimental_descriptor_load",
    "map_tt_experimental_descriptor_store",
    # Grid / launch
    "map_tt_program_id",
    # Misc
    "map_tt_print",
]


class TritonFrontendError(RuntimeError):
    """Common base for all frontend-raised exceptions.

    Wave G4 introduced this base so :class:`EmitError` (raised by op
    emitters) and :class:`poc.triton_frontend.pipeline.PipelineError`
    (raised by pre-walker stages) form a coherent hierarchy. A pipeline
    driver that wants to catch *any* deliberate frontend failure -- as
    opposed to a generic ``RuntimeError`` from TVM/TileLang internals --
    can write ``except TritonFrontendError`` exactly once.

    Per the Wave G4 hard constraint, this base must NOT be caught with a
    bare ``except TritonFrontendError: pass``. Catch the specific subclass
    and re-raise / annotate; catching the base is reserved for top-level
    drivers that translate the failure into a structured status code.
    """


class EmitError(TritonFrontendError):
    """Raised when an emitter cannot lower an op for a precise, named reason.

    We use a dedicated subclass (rather than ``ValueError`` /
    ``NotImplementedError``) so the walker / pipeline driver can
    distinguish "user input needs adjustment" from "frontend is missing a
    feature": ``EmitError`` always means the former.

    This is the canonical definition. ``op_emitters/{arith,control,reduction}``
    re-export this class via ``from ..op_mapping import EmitError`` so all
    emitter modules raise the same exception type and ``except EmitError``
    in the pipeline catches every emitter-side failure.

    Wave G4: ``EmitError`` is now a subclass of :class:`TritonFrontendError`
    so it shares a common ancestor with :class:`PipelineError`.
    """


EmitFn = Callable[..., Any]
"""Type alias: ``(op: mlir.ir.Operation, ctx: WalkerCtx) -> tvm.tir.Stmt|Expr``."""


class LazyTileExpr:
    """Lane-indexable tile expression that avoids materializing temp buffers."""

    def __init__(
        self,
        shape: Sequence[int],
        dtype: str,
        reader: Callable[[Any, Tuple[Any, ...]], Any],
        *,
        name: str = "",
        constant_value: Any = None,
    ) -> None:
        self.shape = tuple(int(s) for s in shape) or (1,)
        self.dtype = str(dtype)
        self._reader = reader
        self.name = name
        self.constant_value = constant_value

    def read_lane(self, ctx: Any, indices: Sequence[Any]) -> Any:
        return self._reader(ctx, tuple(indices))


_SSA_ASSIGN_RE = re.compile(r"(%[A-Za-z0-9_]+(?:#\d+)?)\s*=")
_SSA_HEAD_RE = re.compile(r"(%[A-Za-z0-9_]+(?:#\d+)?)\b")
_SSA_RESULT_GROUP_RE = re.compile(r"(%[A-Za-z0-9_]+)(?::(\d+))?\s*=")


def _result_number(ssa_value: Any) -> Optional[int]:
    """Best-effort result index for an MLIR OpResult-like object."""
    for attr in ("result_number", "result_index", "index"):
        value = getattr(ssa_value, attr, None)
        if callable(value):
            try:
                value = value()
            except Exception:
                continue
        if value is None:
            continue
        try:
            return int(value)
        except Exception:
            pass
    owner = getattr(ssa_value, "owner", None)
    results = getattr(owner, "results", None)
    if results is not None:
        try:
            for idx, result in enumerate(results):
                if result is ssa_value or result == ssa_value:
                    return idx
        except Exception:
            pass
    return None


def _ssa_name_from_text(text: str, *, result_number: Optional[int] = None) -> Optional[str]:
    """Extract a printed MLIR SSA name from a value or owner op string."""
    text = text.strip()
    if not text:
        return None
    match = _SSA_ASSIGN_RE.search(text)
    if match is not None:
        return match.group(1)
    match = _SSA_RESULT_GROUP_RE.search(text)
    if match is not None:
        base = match.group(1)
        count = match.group(2)
        if count is not None and result_number is not None:
            return f"{base}#{result_number}"
        return base
    if text.startswith("%"):
        match = _SSA_HEAD_RE.match(text)
        if match is not None:
            return match.group(1)
    return None


def _printed_ssa_name(ssa_value: Any) -> Optional[str]:
    """Best-effort printed SSA name for MLIR values used as PtrState refs."""
    for attr in ("get_name", "name"):
        getter = getattr(ssa_value, attr, None)
        if callable(getter):
            try:
                name = getter()
            except Exception:
                continue
            if name:
                return str(name)
        elif isinstance(getter, str) and getter:
            return getter

    try:
        name = _ssa_name_from_text(str(ssa_value))
        if name:
            return name
    except Exception:
        pass

    owner = getattr(ssa_value, "owner", None)
    if owner is not None:
        try:
            name = _ssa_name_from_text(
                str(owner),
                result_number=_result_number(ssa_value),
            )
            if name:
                return name
        except Exception:
            pass
    return None


class WalkerCtx:
    """State threaded through the TTIR walker.

    The walker keeps a mapping from MLIR SSA values to TVM TIR exprs,
    a list of already-emitted statements, plus references to the buffers
    that the eventual ``tvm.tir.PrimFunc`` will declare.

    The fields here are all ``Any`` because real TVM types are imported
    lazily inside emitters; the dataclass-shaped surface keeps emitters
    test-friendly without importing TVM at module load time.
    """

    def __init__(
        self,
        *,
        ptr_analysis_shim_available: Optional[bool] = None,
    ) -> None:
        # SSA value -> tvm.tir.PrimExpr or tvm.tir.Buffer
        self.value_map: Dict[Any, Any] = {}
        # Ordered list of emitted statements (becomes a SeqStmt body).
        self.stmts: List[Any] = []
        # Param name -> tvm.tir.Buffer (for tt.func arguments).
        self.buffers: Dict[str, Any] = {}
        # Buffer-key -> fresh symbolic int64 extent Var used when a flat
        # function-arg buffer is re-declared for a strided per-block
        # load/store. One Var per arg key so every redecl of the same arg
        # resolves to the SAME extent symbol, which MakePackedAPI then binds
        # from the real DLTensor's element count at launch. This keeps the
        # compiled kernel monomorphization-free: the SAME PrimFunc runs at any
        # grid / seqlen, because the buffer extent is symbolic, not a baked
        # constant. See ``_redeclare_ctx_buffer_1d``.
        self.flat_arg_extent_vars: Dict[Any, Any] = {}
        # Auto-generated temp counter for fresh names.
        self._tmp_counter: int = 0
        # Lazy-loaded TVM modules.
        self._tvm: Any = None
        self._T: Any = None
        # SSA value -> source SSA value for transposed views (tt.trans).
        # Lets map_tt_dot fold an intervening tt.trans into transpose_A/B
        # without materialising the transpose. The pair (i, j) records the
        # axes flipped; for the common 2D matmul case both values are the
        # last two axes (e.g. (-2, -1)).
        self.transposed_views: Dict[Any, Tuple[int, int]] = {}
        # SSA name (string, e.g. "%2") -> PtrState seeded by the
        # ``run_ptr_analysis_pre_pass`` helper in pipeline.py. Emitters in
        # ``op_emitters/memory.py`` look up here to choose between the real
        # T.copy path and the per-element ``# DEGRADED:`` fallback.
        self.ptr_states: Dict[str, Any] = {}
        # Optional explicit override for tests and isolated harnesses that
        # need deterministic PtrAnalysis availability without mutating module
        # globals. ``None`` keeps the production probe.
        self.ptr_analysis_shim_available: Optional[bool] = ptr_analysis_shim_available
        # Symbol name (e.g. "@triton.language.standard.max__...") -> tt.func
        # op. Populated by the module-level pre-pass in
        # ``triton_frontend._walk_mlir_module`` so the ``tt.call`` emitter can
        # inline-expand the callee's body without re-walking the whole module.
        # The leading ``@`` is stripped during lookup.
        self.callees: Dict[str, Any] = {}
        # Set of sym_names referenced by at least one ``tt.call``. The
        # walker uses this to avoid recursing into helper ``tt.func`` ops
        # at module-level (their bodies are emitted only via inline expansion
        # at the tt.call site).
        self.callee_used: set = set()
        # Stack of substitution overlays pushed by ``push_substitution``.
        # Each frame is a ``Dict[Any, Any]`` mapping callee block-arg SSA
        # values (or their printed names) to the caller's TIR-resolved
        # operand. ``ctx.get`` consults this stack before falling back to
        # ``value_map``.
        self._subst_stack: List[Dict[Any, Any]] = []
        # Tile-scoped buffers allocated *inside* the kernel body (e.g. the
        # results of ``tt.make_range`` spills, ``tt.broadcast`` /
        # ``tt.expand_dims`` / ``tt.splat`` materialisations, and per-element
        # ``tt.load`` fallbacks). These are NEVER promoted to PrimFunc
        # parameters; ``_make_prim_func`` emits one ``tir.AllocBuffer`` stmt
        # per entry at the head of the body so the buffer's ``data`` Var is
        # properly scoped. Putting them in ``buffer_map`` would make
        # ``tirx::analysis::VerifyMemory`` flag every BufferLoad/BufferStore
        # as "directly accessed by host memory" because the verifier requires
        # that any buffer-arg's data var be accessed only inside a thread
        # environment (see 3rdparty/tvm/src/tirx/analysis/verify_memory.cc
        # ``HandleLoadStoreToVariable``).
        self.local_buffers: List[Any] = []
        # Ordered list of runtime scalar ``tir.Var`` objects originating from
        # ``tt.func`` non-pointer block args (e.g. ``n_elements``). These must
        # be appended to the ``PrimFunc.params`` list so ``MakePackedAPI``
        # sees them as proper API arguments rather than free Vars in the body.
        # Constexprs (``BLOCK_SIZE`` etc.) are already substituted by Triton
        # at the TTIR stage, so anything that survives as a block arg is a
        # runtime arg.
        self.runtime_args: List[Any] = []
        # ``tt.get_program_id(axis=N)`` Vars that the emitter created when no
        # active TileLang KernelLaunchFrame was available to supply a real
        # block binding. ``_make_prim_func`` wraps the body in a
        # ``tir.AttrStmt(IterVar, "thread_extent", extent, body)`` per entry
        # so MakePackedAPI accepts the Var as a thread-environment binding
        # rather than a free Var. Each entry is a ``(var, axis, extent)``
        # tuple; extent defaults to a symbolic placeholder when grid info is
        # not available at lowering time.
        self.program_id_vars: List[Tuple[Any, int, Any]] = []
        # Per-axis cache of the block-binding Var created for
        # ``tt.get_program_id(axis=N)``. Triton kernels frequently read the
        # SAME ``program_id(0)`` more than once and split it locally, e.g.
        #     pid   = tl.program_id(0)
        #     pid_m = pid // num_pid_n
        #     pid_n = pid %  num_pid_n
        # which lowers to TWO ``tt.get_program_id x`` ops in the TTIR. Each
        # must resolve to the SAME blockIdx.x binding (one launch_thread with
        # one extent), NOT two separate ``blockIdx.x`` env-threads -- emitting
        # two collapses the host grid mapping (a duplicate gridDim_0 param) so
        # only blockIdx.(0,0,0) ever runs. Keyed by axis -> the recorded Var.
        self.program_id_axis_var: Dict[int, Any] = {}
        # TileLang's ``gemm.lower`` derives ``num_warps = block_size /
        # warp_size`` from the ``threadIdx.x`` ``thread_extent`` AttrStmt
        # wrapping the kernel body. Triton TTIR doesn't carry this
        # explicitly (TTIR is a tile-level IR; lowering to threads happens
        # later in Triton's stack), so we surface ``num_warps`` /
        # ``num_stages`` here as ctx-level metadata. ``_make_prim_func``
        # consults these to synthesise a ``threadIdx.x`` wrap with extent
        # ``num_warps * 32`` and stamp matching PrimFunc attrs. Defaults
        # match Triton's own defaults for unspecified kernels (4 warps =
        # 128 threads/block, 2 software-pipeline stages); callers (the
        # harness, ``from_ttir``) override these from Triton's compile
        # options or kernel autotune-config when available.
        self.num_warps: int = 4
        self.num_stages: int = 2
        # PROLOGUE-OPT gate (routed-triton path only). When True, the
        # convergence-point tile materializer (``materialize_lazy_tile``)
        # THREAD-DISTRIBUTES the elementwise prologue across the block's
        # ``num_warps*32`` lanes (transform 2) instead of emitting a SERIAL
        # ``tir.For`` that every lane re-runs redundantly (classic
        # ReduceDataDuplication waste). It also lets ``arith.cmpi`` fold the
        # always-true int32-overflow guards to a constant tile (transform 3)
        # and keeps addressing binops lazy so they fold into consumers
        # (transform 1). Gated so the cooperative ``T.copy``/``T.gemm`` half
        # and the dict-shaped unit-test paths stay byte-identical: only the
        # routed Tri-Dao chunk kernels (which set this in ``from_ttir``) opt
        # in. RULE #1: structural thread-binding or RAISE -- never a silent
        # serial fallback.
        self.routed_triton_prologue_opt: bool = False
        # Sub-gate for transform (2) thread-distribution. OFF by default: it is
        # only correct for SHARED-scope cooperative tiles (see
        # ``materialize_lazy_tile``). Local prologue tiles keep the serial
        # fill. Kept as a distinct flag so the always-correct fold/guard-drop
        # transforms (1)/(3) can ship under ``routed_triton_prologue_opt``
        # without coupling to the conditional thread-distribution.
        self.routed_triton_thread_distribute: bool = False
        # ITERATION 3 (coalesced async loads). When True (set together with
        # ``routed_triton_prologue_opt`` on the routed Tri-Dao chunk path), the
        # ``scf.for`` K-loop emitter stamps a ``num_stages`` annotation on the
        # serial K-loop ``tir.For`` whenever the loop body emitted at least one
        # global->shared cooperative ``T.copy`` (``_emit_load_copy``). That
        # annotation is exactly what TileLang's ``T.Pipelined`` would stamp;
        # ``PipelinePlanning`` + ``InjectSoftwarePipeline`` then schedule the
        # global->shared copies as async producers and ``LowerPTXAsyncCopy``
        # emits ``cp.async`` (SASS LDGSTS) instead of plain per-lane LDG. The
        # copy region itself is already a coalesced cooperative T.copy (the same
        # high-level surface the GEMM operands use); software-pipelining is the
        # missing trigger that routes it through the cp.async path. RULE #1:
        # once a K-loop carries a cp.async-eligible global->shared copy we
        # pipeline it (coalesce) -- we never leave it as a silent uncoalesced
        # serial LDG. The GEMM cooperative path is untouched: it does not run
        # under ``routed_triton_prologue_opt`` and its operand copies already
        # lower to ldmatrix/cp.async via the existing GEMM staging.
        self.routed_triton_async_loads: bool = False
        # Shared single-element counter list: ``_emit_load_copy`` appends a 1 for
        # every global->shared cooperative copy it emits. ``map_scf_for`` reads
        # the delta across its body emission to decide whether the K-loop is
        # cp.async-eligible (has at least one such copy). A list (not an int) so
        # the region-child ctx shares the SAME object by reference, exactly like
        # ``local_buffers`` -- in-loop copies surface to the parent.
        self._gmem_shared_copies: List[int] = []
        # FULL TRANSFORM 1 (Coalesce-style addressing fold). Set of MLIR
        # result ``Value`` objects (by identity/hash) whose tile result is
        # consumed ONLY as addressing/mask for a
        # tt.load/tt.store (directly or transitively through other
        # addressing/mask ops). Populated by ``build_addressing_fold_set`` in
        # the walker pre-pass. When ``routed_triton_prologue_opt`` is on and a
        # feeder (``_emit_tile_binop`` / ``emit_tt_broadcast`` /
        # ``emit_tt_expand_dims``) is about to ``materialize_lazy_tile`` a tile
        # whose result SSA is in this set, it instead binds the lane-indexable
        # ``LazyTileExpr`` directly -- the per-lane index/predicate folds into
        # the load/store loop body (region indices + predicate) and the
        # [64]/[2048]/[4096] array is never materialized in ANY scope (no local
        # spill, no shared overflow, nothing to thread-distribute). The
        # cooperative T.copy/T.gemm operand tiles are NEVER in this set (their
        # results feed tt.dot), so the GEMM path stays byte-identical.
        self.fold_addressing_ssa: set = set()
        # Some scalar fallback emitters implement tile-level semantics with
        # serial read-modify-write loops. When one is used under the synthetic
        # ``threadIdx.x`` wrapper, run the whole block body on lane 0 to avoid
        # intra-block races.
        self.requires_single_thread_body: bool = False
        # FRAMEFIX (post-walk fragment-layout re-registration). The walker
        # builds a flat TIR PrimFunc with raw ``decl_buffer`` fragments BEFORE
        # any ``T.Kernel`` frame exists, so TileLang's LayoutInference never
        # gets a STRICT entry for the tensor-core C accumulator's MMA store
        # layout. A subsequent ``T.copy(fragment -> shared)`` then claims the
        # fragment with its own SIMT (identity) layout and OVERRIDES the gemm
        # layout, producing a layout-blind store that materialises only the
        # per-thread-resident slots (32/4096 per tile). ``map_tt_dot`` records
        # each CUDA grid-scaled MMA-C fragment here as
        # ``{"buffer": Buffer, "M": int, "N": int, "K": int,
        #    "trans_A": bool, "trans_B": bool}`` so the post-walk pass
        # (``register_mma_fragment_layouts``) can re-register the fragment's
        # ``make_mma_store_layout`` as a STRICT block ``layout_map`` annotation
        # before LowerLDSM -- exactly the annotation a native ``T.alloc_fragment``
        # + ``T.gemm`` inside ``T.Kernel`` emits. Empty on Metal / non-MMA
        # paths so the gate (CUDA + non-empty) leaves fla_dot_exp2 untouched.
        self.mma_c_fragments: List[Dict[str, Any]] = []
        # BANKSWIZZLE: raw cp.async-staged SHARED load tiles (the
        # register-A source, e.g. the dstates ``dout`` tile) that the
        # post-walk FRAMEFIX SBlock pins to make_swizzled_layout so the
        # per-lane LDS reads hit distinct banks. Mirrors mma_c_fragments:
        # shared into scf.for child ctxs so an in-loop load surfaces to
        # the parent prim_func the re-registration runs on. Empty on
        # non-CUDA / non-register-A paths (the gate leaves them untouched).
        self.swizzle_shared_loads: List[Any] = []
        # Printed SSA name -> op names that consume it. Seeded by the MLIR
        # module pre-pass when available. Emitters use this only for layout
        # choices where downstream composability matters (for example
        # ``tt.dot`` can keep a direct store result in local.fragment but must
        # use shared scope when a later arith op indexes the dot result).
        self.ssa_users: Dict[str, set] = {}
        # Printed operand SSA name -> the set of consumer op OBJECTS. A
        # robust companion to ``ssa_users`` (which holds only op-NAMES):
        # lets a fold gate call another emitter helper on the actual
        # consumer op (no op-string parsing). Seeded by the same prepass.
        self.ssa_user_ops: Dict[str, set] = {}
        # Optional caller-provided ABI shapes for pointer block args.
        # TTIR pointer types do not carry host tensor extents, but runtimes
        # such as MLX validate the DLTensor size against PrimFunc buffer_map.
        # Callers that know the public launch ABI can seed shapes by block
        # argument index or SSA name; emitters then must not shrink them to a
        # per-tile fallback.
        self.arg_buffer_shapes: Dict[Any, Sequence[int]] = {}
        self.fixed_arg_buffer_keys: set = set()
        # Canonical ``threadIdx.x`` TIR Var shared between body emitters and
        # ``map_tt_func`` PrimFunc assembly. Emitters that need a *local*
        # single-lane guard (scalar atomic-rmw serial loops) wrap only THEIR
        # own statement in ``if threadIdx_x == 0`` using this exact Var, while
        # ``map_tt_func`` reuses the same Var for the outer
        # ``threadIdx.x`` ``thread_extent`` AttrStmt. This keeps the whole-block
        # thread binding (extent ``num_warps*32``) wrapping a real ``T.gemm``
        # intact -- a collective warp-level MMA must see all 128 threads, so it
        # must NOT be nested under a single-lane guard (doing so collapses the
        # gemm's ``thread_bounds`` extent to 0 and trips
        # ``m_warp*n_warp==num_warps`` with ``num_warps=0`` in ``gemm.lower``).
        self._thread_var: Any = None
        # When this ctx is a region child (``_emit_region``), point at the
        # root ctx so ``thread_idx_var()`` resolves to the ONE Var that
        # ``map_tt_func`` will bind for the block ``threadIdx.x`` thread_extent.
        # Without this, a child-emitted lane-0 guard would mint a *second*
        # ``threadIdx_x`` Var that ``MakePackedAPI`` then flags as
        # "used, but not passed in" (it is not the thread-env-bound one).
        self._thread_var_root: Any = None

    # ---- helpers --------------------------------------------------------

    def fresh(self, prefix: str = "v") -> str:
        """Return a unique name suitable for a buffer / variable."""
        self._tmp_counter += 1
        return f"{prefix}_{self._tmp_counter}"

    def thread_idx_var(self) -> Any:
        """Return the shared canonical ``threadIdx.x`` TIR Var (lazily made).

        Used by both per-lane scalar emitters (for a LOCAL lane-0 guard around
        their own statement) and ``map_tt_func`` (for the outer
        ``threadIdx.x`` ``thread_extent`` binding). Sharing one Var means a
        local guard and the block thread binding refer to the same thread
        index without the whole-body single-thread wrap that would gate an
        adjacent ``T.gemm``.
        """
        root = self._thread_var_root or self
        if root._thread_var is None:
            root._thread_var = root.tir().Var("threadIdx_x", "int32")
        return root._thread_var

    def num_threads(self) -> int:
        """Return the block thread extent (``num_warps*32``).

        This is the SAME extent ``_make_prim_func`` binds for the canonical
        ``threadIdx.x`` ``thread_extent`` AttrStmt wrapping the whole body, so
        a prologue loop that strides by ``num_threads()`` and offsets by
        ``thread_idx_var()`` partitions its work over exactly the lanes that
        AttrStmt declares.
        """
        root = self._thread_var_root or self
        return int(getattr(root, "num_warps", 4) or 4) * 32

    def tvm(self) -> Any:
        """Lazy-import ``tvm`` and cache the module handle.

        We import ``tilelang`` FIRST so that ``import tvm`` resolves to
        TileLang's vendored TVM (``3rdparty/tvm/python/tvm``) on hosts (gb10)
        where a bare top-level ``tvm`` is not on ``sys.path``. Without this,
        ``import tvm`` would raise ``ModuleNotFoundError`` whenever the
        frontend is driven before ``import tilelang`` happened to run.
        """
        if self._tvm is None:
            import tilelang  # noqa: F401,WPS433 (wires vendored TVM onto path)
            import tvm  # noqa: WPS433 (intentional lazy import)

            self._tvm = tvm
        return self._tvm

    def tir(self) -> Any:
        """Shortcut to ``tvm.tir`` (via TileLang's vendored TVM)."""
        import tilelang  # noqa: F401,WPS433 (wires vendored TVM onto path)
        from tvm import tir

        return tir

    def get(self, ssa_value: Any) -> Any:
        """Resolve an MLIR SSA value to its TIR equivalent.

        Consults the substitution stack first (top-of-stack wins), so a
        ``push_substitution`` overlay can mask a stale ``value_map`` entry
        for an SSA name that the inlined callee reuses internally.
        """
        # Substitution overlays (pushed by ``push_substitution``) take
        # precedence -- this is how ``tt.call`` rebinds the callee's
        # block-arg SSAs to the caller's resolved operands without
        # mutating ``value_map`` for the duration of the inline walk.
        for frame in reversed(self._subst_stack):
            if ssa_value in frame:
                return frame[ssa_value]
            # SSA values may be unhashable Value objects but their printed
            # name (``%arg0``) is a usable key. Try that fallback.
            try:
                # Best-effort name extraction without importing the helper
                # from op_emitters.control (avoids a circular import).
                getter = getattr(ssa_value, "get_name", None)
                if callable(getter):
                    name = getter()
                    if name and name in frame:
                        return frame[name]
            except Exception:
                pass
        try:
            if ssa_value in self.value_map:
                return self.value_map[ssa_value]
        except TypeError:
            # Some MLIR binding objects are unhashable. ``bind`` also stores
            # a printed SSA-name alias, so fall through to that lookup.
            pass
        try:
            name = _printed_ssa_name(ssa_value)
        except Exception:
            name = None
        if name and name in self.value_map:
            return self.value_map[name]
        # Best-effort context: identify the producing op and (if the SSA
        # value is a BlockArgument) the parent op + region index. This
        # is invaluable when the walker descends into a region it should
        # not have (ops in OPS_THAT_HANDLE_OWN_REGIONS) -- the trace
        # points straight at the offending parent.
        producer = "<unknown>"
        try:
            owner = getattr(ssa_value, "owner", None)
            if owner is not None:
                # OpResult.owner -> Operation
                producer = getattr(owner, "name", None) or producer
                # BlockArgument.owner -> Block; climb to its parent op.
                if not getattr(owner, "name", None):
                    parent_op = getattr(owner, "parent_op", None) or getattr(owner, "owner", None)
                    parent_name = getattr(parent_op, "name", None)
                    if parent_name:
                        producer = f"<block-arg of {parent_name}>"
        except Exception:
            pass
        raise KeyError(
            f"WalkerCtx: SSA value {ssa_value!r} not yet mapped "
            f"(producer={producer}); emitter called out of TTIR program "
            f"order? If this SSA originates inside a region (e.g. a "
            f"tt.reduce combiner or scf.for body), the parent op should "
            f"be in mlir_walker.OPS_THAT_HANDLE_OWN_REGIONS."
        )

    def bind(self, ssa_value: Any, tir_value: Any) -> None:
        """Record the TIR value produced by an emitter for ``ssa_value``."""
        self.value_map[ssa_value] = tir_value
        # PtrAnalysis reports symbolic references as printed SSA names
        # (e.g. "%29"). Keep a string alias so memory emitters can resolve
        # offsets/strides back to the TIR value produced by earlier ops.
        try:
            name = _printed_ssa_name(ssa_value)
            if name:
                self.value_map[str(name)] = tir_value
        except Exception:
            pass

    def emit(self, stmt: Any) -> None:
        """Append a TIR statement to the current function body.

        Safety net: emitters occasionally hand us a ``tir.PrimExpr`` (most
        often a ``tir.Call`` produced by ``tilelang.language.gemm`` or
        ``tir.call_intrin``) instead of a Stmt. Inserting a PrimExpr here
        makes ``tir.SeqStmt(stmts: Array<Stmt>)`` reject the list at the
        walker boundary with::

            TypeError: Mismatched type on argument #0 ... Expected
            Array<tirx.Stmt> but got Array[index N: tirx.Call]

        We auto-wrap in ``tir.Evaluate`` and emit a one-shot
        DeprecationWarning so the offending emitter still gets diagnosed.
        Fix the emitter -- this safety net should never quietly absorb a
        production bug.
        """
        if stmt is None:
            return
        try:
            tir = self.tir()
        except Exception:  # pragma: no cover - tvm not importable yet
            self.stmts.append(stmt)
            return
        if isinstance(stmt, tir.PrimExpr):
            import warnings

            warnings.warn(
                f"WalkerCtx.emit() received a PrimExpr "
                f"({type(stmt).__name__}); auto-wrapping in tir.Evaluate. "
                f"The producing emitter should wrap the call itself "
                f"(`ctx.emit(tir.Evaluate(call))`).",
                DeprecationWarning,
                stacklevel=2,
            )
            stmt = tir.Evaluate(stmt)
        self.stmts.append(stmt)

    # ---- tt.call inline-expansion plumbing ------------------------------

    def lookup_callee(self, name: str) -> Optional[Any]:
        """Return the ``tt.func`` op registered for ``name``, or None.

        ``name`` may be supplied with or without a leading ``@``; both are
        normalised before the lookup. Used by ``op_emitters.control.emit_tt_call``
        to find the callee body to inline-expand.
        """
        if not name:
            return None
        key = name.lstrip("@")
        return self.callees.get(key) or self.callees.get(name)

    class _SubstScope:
        """Context manager popping a substitution frame on exit."""

        __slots__ = ("ctx", "frame")

        def __init__(self, ctx: "WalkerCtx", frame: Dict[Any, Any]) -> None:
            self.ctx = ctx
            self.frame = frame

        def __enter__(self) -> Dict[Any, Any]:
            self.ctx._subst_stack.append(self.frame)
            return self.frame

        def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
            popped = self.ctx._subst_stack.pop()
            assert popped is self.frame, (
                "WalkerCtx._subst_stack desync: pop returned a different frame "
                "than the one push_substitution installed; something nested "
                "out-of-order."
            )

    def push_substitution(self, mapping: Dict[Any, Any]) -> "_SubstScope":
        """Push an SSA-substitution overlay for the duration of a ``with`` block.

        Used by ``emit_tt_call`` to bind a callee's block-argument SSAs to
        the caller's already-resolved TIR operands. The overlay is consulted
        by :meth:`get` BEFORE ``value_map``, so it correctly shadows any
        outer binding of the same SSA name.
        """
        return WalkerCtx._SubstScope(self, dict(mapping))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _operands(op: Any) -> Tuple[Any, ...]:
    """Return ``op`` operands as a tuple, hiding the MLIR vs dict shape diff.

    The walker can call us with either a real ``mlir.ir.Operation`` or a
    test-only dict ``{"operands": [...], "results": [...], "attrs": {...}}``.
    """
    if isinstance(op, dict):
        return tuple(op.get("operands", ()))
    return tuple(op.operands)


def _results(op: Any) -> Tuple[Any, ...]:
    """Return ``op`` SSA results, hiding the MLIR vs dict shape diff."""
    if isinstance(op, dict):
        return tuple(op.get("results", ()))
    return tuple(op.results)


def _result_is_consumed(result: Any, ctx: Any = None) -> bool:
    """Return whether a real MLIR result has downstream users.

    Triton generic-form ``tt.atomic_rmw`` prints a result even when source
    code discards the return value of ``tl.atomic_add``. Requesting
    ``return_prev=True`` for those unused results forces TileLang's
    tile-region atomic path into an unsupported mode that emits an
    ``address_of(BufferLoad)`` whose load is later rewritten, tripping
    ``loop_unswitching``'s ``address_of argument must be a BufferLoad`` ICHECK.

    Resolution order (authoritative-first, RULE #1 -- no silent over-broad
    default that papers over a discarded result):

    1. ``ctx.ssa_users`` -- the module pre-pass in ``_walk_mlir_module``
       records, for every operand SSA name, the set of ops that consume it.
       A result whose printed SSA name is NOT a key there has zero users and
       is definitively unused. This is the use-def source of truth for the
       libtriton textual-TTIR path, whose ``Value.uses`` iterator is ``None``.
    2. ``result.uses`` -- when libtriton DOES populate a uses iterator.
    3. Conservative ``True`` -- only for test-only dict/fake results that
       expose neither a ctx users-map entry nor a real uses iterator.
    """
    # 1. Authoritative use-def map seeded by the pre-pass.
    if ctx is not None:
        ssa_users = getattr(ctx, "ssa_users", None)
        name = _printed_ssa_name(result)
        if ssa_users is not None and name:
            return name in ssa_users
    # 2. Real libtriton uses iterator, when present.
    uses = getattr(result, "uses", None)
    if uses is not None:
        try:
            return any(True for _ in uses)
        except Exception:
            return True
    # 3. Test-only shapes with no use-def data.
    return True


def _has_consumed_result(op: Any, ctx: Any = None) -> bool:
    """Return True when any SSA result of ``op`` is actually consumed."""
    return any(_result_is_consumed(result, ctx) for result in _results(op))


def _attrs(op: Any) -> Dict[str, Any]:
    """Return ``op`` attribute dict, hiding the MLIR vs dict shape diff."""
    if isinstance(op, dict):
        return dict(op.get("attrs", {}))
    # Real MLIR: attributes attribute. Keep stringified for portability.
    return {a.name: a.attr for a in op.attributes} if hasattr(op, "attributes") else {}


# ---------------------------------------------------------------------------
# Generic-form properties parser (shared by memory.py and arith.py emitters)
# ---------------------------------------------------------------------------
#
# Triton 3.6 (and any MLIR op that uses Properties storage) prints its
# inherent attributes inside ``<{...}>`` rather than the legacy ``{...}``
# braces. jaxlib's ``mlir.ir`` Python bindings expose properties only when
# the dialect is registered; with ``allow_unregistered_dialects=True`` --
# the only mode we have on hosts that lack a Triton-aware MLIR build --
# ``op.attributes`` returns an EMPTY map for property-only ops. The same
# bug affected ``tt.make_range`` (fixed in op_emitters/memory.py during
# Wave C2) and now bites ``arith.cmpi`` / ``arith.cmpf`` whose ``predicate``
# attribute is stored as a Property in Triton 3.6.
#
# We recover by lifting the ``<{...}>`` slice out of ``str(op)`` (the
# printed assembly *does* include properties even when the dict-shaped
# accessor is empty) and parsing the small ``key = literal`` grammar
# Triton emits. The helpers live here so every op-emitter module can use
# them without duplicating the regex.

# Match ``<{...}>`` at any nesting level inside the printed op. We only
# need the OUTERMOST one -- properties don't nest.
_PROPERTIES_RE = re.compile(r"<\{(?P<body>[^}]*)\}>")
# A key=value pair with an optional ``: <type>`` annotation. Matches
# ``predicate = 4 : i64``, ``start = 0 : i32``, and ``end = 256 : i32``.
_PROP_PAIR_RE = re.compile(
    r"(?P<key>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*"
    r"(?P<val>-?\d+|true|false|\"[^\"]*\")"
    r"(?:\s*:\s*[A-Za-z_][A-Za-z0-9_]*)?"
)


def _parse_generic_properties_shared(op: Any) -> Dict[str, Any]:
    """Extract Triton 3.6 MLIR Properties from ``str(op)``.

    jaxlib's ``mlir.ir`` Python bindings under
    ``allow_unregistered_dialects=True`` don't expose dialect-registered
    Properties via ``op.attributes`` -- only inherent attrs of *registered*
    dialects come through. This helper pulls the ``<{...}>`` block out of
    the printed op string and parses the ``key = literal : <type>`` grammar
    Triton/arith emit.

    Returns an empty dict when the op has no printable ``<{...}>`` block
    (real dict-shaped fakes or attributes-only ops). Values are coerced
    to Python ``int`` / ``bool`` / ``str`` -- the coverage we need for
    ``tt.make_range`` and ``arith.cmpi``/``arith.cmpf`` predicates.
    """
    if isinstance(op, dict):
        return {}
    try:
        text = str(op)
    except Exception:
        return {}
    m = _PROPERTIES_RE.search(text)
    if not m:
        return {}
    out: Dict[str, Any] = {}
    for pair in _PROP_PAIR_RE.finditer(m.group("body")):
        key = pair.group("key")
        raw = pair.group("val")
        if raw == "true":
            out[key] = True
        elif raw == "false":
            out[key] = False
        elif raw.startswith('"') and raw.endswith('"'):
            out[key] = raw[1:-1]
        else:
            try:
                out[key] = int(raw)
            except ValueError:
                out[key] = raw
    return out


def _attrs_with_properties_shared(op: Any) -> Dict[str, Any]:
    """``_attrs`` plus a fallback to the ``<{...}>`` properties block.

    Existing dict-shaped fakes and ops with classic attribute storage
    take the fast path through the legacy ``_attrs`` helper. When that
    returns an empty dict and we're looking at a real MLIR op whose
    inherent attributes live in Properties storage, we fall back to a
    small textual parser. The parser is intentionally narrow: we only
    accept the ``key = scalar : type`` shape Triton/arith emit.
    """
    base: Dict[str, Any] = {}
    try:
        base = dict(_attrs(op))
    except Exception:
        # ``_attrs`` itself can throw on op shapes whose ``op.attributes``
        # iterator yields strings instead of NamedAttribute records (some
        # jaxlib builds do this for unregistered ops). Treat as empty.
        base = {}
    if base:
        return base
    return _parse_generic_properties_shared(op)


def _shape_of(value: Any) -> Tuple[int, ...]:
    """Best-effort shape extraction for a TTIR SSA value.

    Triton TTIR types are MLIR ``RankedTensorType`` for tile values; for
    scalar values we return ``()``. The walker also feeds us a dict-shaped
    fake during unit tests; we accept that shape too.
    """
    if isinstance(value, dict):
        return tuple(value.get("shape", ()))
    typ = getattr(value, "type", None)
    if typ is None:
        return ()
    shape = getattr(typ, "shape", None)
    return tuple(shape) if shape is not None else ()


def _dtype_of(value: Any) -> str:
    """Best-effort element dtype for a TTIR SSA value (defaults to float32).

    Normalises short MLIR dtype spellings (``f32``, ``i32``, ``bf16``,
    ``i1``) to TVM's canonical names (``float32``, ``int32``,
    ``bfloat16``, ``bool``). The MLIR generic form prints element types
    using the short spelling (a bare ``f32`` rather than ``float32``);
    TVM's ``tir.decl_buffer`` rejects those short forms with
    ``ValueError: unknown dtype 'f32'``. Returning the canonical TVM
    spelling here is the pinch point that keeps every emitter that
    threads dtype through ``_dtype_of`` working with both shapes.
    """
    if isinstance(value, dict):
        return _normalize_mlir_dtype(str(value.get("dtype", "float32")))
    typ = getattr(value, "type", None)
    if typ is None:
        return "float32"
    elt = getattr(typ, "element_type", None)
    if elt is None:
        # ``typ`` may itself be a scalar element type (i32 / f32) when the
        # SSA value is non-tensor. Stringify and normalise.
        return _normalize_mlir_dtype(str(typ))
    return _normalize_mlir_dtype(str(elt))


# Canonical short-form -> TVM dtype map. Covers every spelling we expect
# from Triton 3.6 generic-form TTIR (and a handful of legacy aliases for
# robustness). Anything outside this set raises in :func:`_normalize_mlir_dtype`
# so silent dtype defaulting cannot mask a regression -- the maintainer's
# "no silent fallback" hard constraint.
_MLIR_DTYPE_ALIASES: Dict[str, str] = {
    # Floating point
    "f16": "float16",
    "f32": "float32",
    "f64": "float64",
    "bf16": "bfloat16",
    "float16": "float16",
    "float32": "float32",
    "float64": "float64",
    "bfloat16": "bfloat16",
    # Integer
    "i1": "bool",
    "i8": "int8",
    "i16": "int16",
    "i32": "int32",
    "i64": "int64",
    "int8": "int8",
    "int16": "int16",
    "int32": "int32",
    "int64": "int64",
    "bool": "bool",
    # Unsigned (Triton occasionally surfaces these via ptr<u8> etc.)
    "ui8": "uint8",
    "ui16": "uint16",
    "ui32": "uint32",
    "ui64": "uint64",
    "uint8": "uint8",
    "uint16": "uint16",
    "uint32": "uint32",
    "uint64": "uint64",
    # Index -- treat as 64-bit signed for kernel-args (matches TVM convention).
    "index": "int64",
    # Opaque handle -- pointer-typed block args fall back here when the
    # element type can't be resolved; we keep TVM's spelling intact.
    "handle": "handle",
}


# Pointer-typed values print as ``!tt.ptr<T>`` (or, occasionally,
# ``tt.ptr<T>`` without the leading bang in some legacy spellings).
# ``_normalize_mlir_dtype`` unwraps the inner element type because every
# caller of this helper wants the *storage* dtype: ``emit_tt_load`` pulls
# values out of a buffer (so the element dtype is what ``BufferLoad``
# needs), and ``emit_tt_func`` (Wave C1) calls ``decl_buffer`` with the
# unwrapped element dtype. Callers that need to *distinguish* a pointer
# from a scalar should use :func:`_is_ptr_type` instead. Nested pointers
# (``!tt.ptr<!tt.ptr<f32>>``) -- vanishingly rare in Triton kernels but
# technically representable -- are handled by the recursion.
_PTR_RE = re.compile(r"^!?tt\.ptr<(.+)>$")

# Tensor types print as ``tensor<NxT>`` for 1-D tiles or
# ``tensor<NxMx...xT>`` for higher-rank tiles. The element dtype is the
# trailing token after the final ``x``; preceding tokens are integer
# extents. Used by :func:`_parse_tensor_type` (which also surfaces the
# rank info) and by :func:`_normalize_mlir_dtype` (which projects to the
# element dtype because most callers want the storage dtype).
#
# Tile-typed block arguments surface in kernels like ``layer_norm`` where
# Triton's TTIR threads a ``tensor<128xf32>`` across function boundaries.
# Without this branch ``map_tt_func`` raised
# ``unsupported MLIR dtype: 'tensor<128xf32>'``.
_TENSOR_RE = re.compile(r"^tensor<([^>]+)>$")
_TENSOR_DESC_RE = re.compile(r"!?tt\.tensordesc<(?P<tensor>tensor<[^>]+>)>")


def _parse_tensor_type(s: str) -> Tuple[List[int], str]:
    """Parse ``tensor<NxMxT>`` into ``([N, M], normalized_T)``.

    The MLIR generic form prints tensor extents and element dtype joined
    by ``x``, e.g. ``tensor<128xf32>`` or ``tensor<16x32xf32>``. We split
    on the trailing ``x`` to recover the element dtype, then peel off the
    leading ``NxM...`` extents (each must be a base-10 integer; symbolic
    shapes aren't a thing in TTIR after constant-folding so we don't try
    to handle ``?`` here -- the emitter has no shape info to plug in).

    Returns
    -------
    shape : list[int]
        The integer extents in row-major order.
    elt_dtype : str
        Canonical TVM element dtype (``"float32"`` etc.), normalised via
        :func:`_normalize_mlir_dtype` so callers can plumb it directly
        into ``tir.decl_buffer``.

    Raises
    ------
    ValueError
        If ``s`` doesn't match ``tensor<...>`` or any extent fails to
        parse as a positive int. This is deliberate: silently defaulting
        to ``[1]`` or ``float32`` is exactly the regression that this
        helper exists to surface. Caller emitters re-raise as
        :class:`EmitError` to fold into the frontend's error hierarchy.
    """
    m = _TENSOR_RE.match((s or "").strip())
    if m is None:
        raise ValueError(f"not a tensor type: {s!r}")
    inner = m.group(1)
    # rsplit on the last ``x`` -- everything before it is the shape, the
    # tail is the element dtype. ``tensor<128xf32>`` -> ``("128", "f32")``;
    # ``tensor<16x32xf32>`` -> ``("16x32", "f32")``.
    parts = inner.rsplit("x", 1)
    if len(parts) != 2 or not parts[0]:
        raise ValueError(f"malformed tensor type (missing element dtype): {s!r}")
    shape_str, elt_str = parts
    shape: List[int] = []
    for dim in shape_str.split("x"):
        d = dim.strip()
        if not d.isdigit():
            raise ValueError(f"non-integer extent {d!r} in tensor type: {s!r}")
        shape.append(int(d))
    if not shape:
        raise ValueError(f"empty shape in tensor type: {s!r}")
    elt_dtype = _normalize_mlir_dtype(elt_str)
    return shape, elt_dtype


def _is_tensor_type(dtype: str) -> bool:
    """Return True iff ``dtype`` is an MLIR tensor spelling (``tensor<...>``).

    Sibling to :func:`_is_ptr_type`; callers branch on this when they need
    to allocate a fixed-shape buffer rather than a pointer-backed one.
    """
    return _TENSOR_RE.match((dtype or "").strip()) is not None


def _normalize_mlir_dtype(dtype: str) -> str:
    """Canonicalise an MLIR-printed dtype string to TVM's spelling.

    Pointer types (``!tt.ptr<T>``) are recursively unwrapped to their
    element dtype since every caller of this helper operates on the
    storage dtype, not the pointer itself. Tensor types
    (``tensor<NxT>``) are likewise projected to the element dtype --
    callers that need the rank/shape go through
    :func:`_parse_tensor_type` instead. See module-level note on
    ``_PTR_RE`` / ``_TENSOR_RE`` for the rationale; use
    :func:`_is_ptr_type` / :func:`_is_tensor_type` if a caller actually
    needs to know whether the original spelling was a pointer or tensor.

    Raises ``ValueError`` when the input is genuinely unknown so that a
    coverage gap surfaces immediately rather than silently lowering as
    ``float32`` (the regression that motivated this helper).
    """
    s = (dtype or "").strip()
    if not s:
        # Empty string falls back to handle so pointer block args without
        # a resolvable element dtype keep working; downstream emitters
        # that need a real dtype will already have replaced it.
        return "handle"
    m = _PTR_RE.match(s)
    if m is not None:
        # Recurse on the inner type; nested ``!tt.ptr<!tt.ptr<f32>>``
        # collapses to ``float32`` because the storage dtype is what the
        # caller ultimately operates on.
        return _normalize_mlir_dtype(m.group(1))
    tm = _TENSOR_RE.match(s)
    if tm is not None:
        # Project to the element dtype. Rank is preserved on the parsed
        # form via :func:`_parse_tensor_type`; this branch keeps the
        # storage-dtype contract that all other callers rely on.
        _shape, elt = _parse_tensor_type(s)
        return elt
    if s in _MLIR_DTYPE_ALIASES:
        return _MLIR_DTYPE_ALIASES[s]
    # Already-canonical TVM spellings pass through (the alias map already
    # covers them, but keep this branch defensive against future TVM
    # additions like ``float8_e4m3``).
    if s.startswith(("float", "int", "uint")) or s == "bool":
        return s
    raise ValueError(f"unsupported MLIR dtype: {dtype!r}")


def _is_ptr_type(dtype: str) -> bool:
    """Return True iff ``dtype`` is an MLIR pointer spelling.

    Sibling to :func:`_normalize_mlir_dtype` for callers that need to
    branch on pointer-vs-scalar (e.g. an emitter that decides between
    ``decl_buffer`` and ``Var``). Accepts both ``!tt.ptr<T>`` and the
    bang-less ``tt.ptr<T>`` variant.
    """
    return _PTR_RE.match((dtype or "").strip()) is not None


# ---------------------------------------------------------------------------
# PtrState helpers (Wave-2: enable T.copy(global[region], frag) emission
# when triton-shared PtrAnalysis has surfaced multi-element tile metadata).
# ---------------------------------------------------------------------------


def _ptrstate_is_tile(resolved: Dict[str, Any]) -> bool:
    """Return True iff the PtrState describes a tile larger than one element."""
    sizes = resolved.get("sizes") or []
    if not sizes:
        return False
    for s in sizes:
        try:
            if int(s) > 1:
                return True
        except (TypeError, ValueError):
            # Symbolic size string (e.g. "BLOCK_SIZE_M") -> treat as tile.
            if isinstance(s, str) and s.strip() and s.strip() != "1":
                return True
    return False


def _ptrstate_offsets_or_zero(resolved: Dict[str, Any]) -> List[Any]:
    raw = resolved.get("offsets") or []
    out: List[Any] = []
    for o in raw:
        try:
            out.append(int(o))
        except (TypeError, ValueError):
            out.append(o)
    return out or [0]


def _ptrstate_sizes_int(resolved: Dict[str, Any]) -> List[int]:
    """Project ``sizes`` to concrete ints; symbolic dims fall back to 1024."""
    out: List[int] = []
    for s in resolved.get("sizes") or []:
        try:
            out.append(int(s))
        except (TypeError, ValueError):
            out.append(1024)
    return out


def _ptrstate_buffer(ctx: "WalkerCtx", resolved: Dict[str, Any], dtype: str) -> Any:
    """Return (or create) the global buffer that the PtrState aliases."""
    raw_name = resolved.get("source") or ctx.fresh("ptr")
    name = str(raw_name).lstrip("%") or ctx.fresh("ptr")
    if name not in ctx.buffers:
        tir = ctx.tir()
        ctx.buffers[name] = tir.decl_buffer(
            shape=_ptrstate_sizes_int(resolved) or [1024],
            dtype=dtype,
            name=name,
        )
    return ctx.buffers[name]


def _alloc_tile_buffer(
    ctx: "WalkerCtx",
    shape: Sequence[Any],
    dtype: str,
    name: str,
    scope: str = "local",
) -> Any:
    """Allocate a tile-scoped buffer that is visible only inside the kernel body.

    Use this in op-emitters whenever you need a fresh buffer to hold an
    intermediate tile (``tt.make_range`` spill, ``tt.broadcast`` /
    ``tt.expand_dims`` / ``tt.splat`` materialisation, per-element
    ``tt.load`` fallback, ``arith.constant`` dense tile, etc.).

    Why not ``tir.decl_buffer + ctx.buffers[name] = buf``?
    -----------------------------------------------------
    Promoting a tile buffer to ``ctx.buffers`` makes it a PrimFunc parameter
    (it ends up in ``buffer_map``).  TileLang's ``tirx::analysis::VerifyMemory``
    pass walks every ``BufferLoad`` / ``BufferStore`` and, if the data ``Var``
    is in ``buffer_map`` (i.e. comes from a function argument) **and** the
    access is not inside a thread environment, it raises
    ``"Variable 'X' is directly accessed by host memory"``.  Since our
    intermediate tile buffers are written/read at host scope (the
    surrounding ``T.Kernel`` is added later in the pipeline), keeping them
    out of ``buffer_map`` lets the verifier's "skip locally-allocated
    buffers conservatively" branch handle them correctly.

    Why ``T.alloc_fragment``-style scope?
    -------------------------------------
    The buffer carries ``scope="local"`` (or ``scope="local.fragment"`` /
    ``"shared"`` etc., when callers explicitly request one) so downstream
    LowerOpaqueBlock / FlattenBuffer passes treat it as thread-local
    storage rather than global memory.

    The buffer is registered in ``ctx.local_buffers``; ``_make_prim_func``
    emits a ``tir.AllocBuffer`` stmt at the head of the body so the
    buffer's ``data`` Var is in scope before the first reference.

    Falls back to ``tir.decl_buffer`` + no scope tracking when TVM's
    ``tirx`` namespace lacks ``AllocBuffer`` (legacy / unit-test sandbox
    paths); the helper still returns a usable Buffer in that case.
    """
    tir = ctx.tir()
    shape_list = list(shape) if shape else [1]
    # DYNSHARED routing (env-gated, experiment-only): on Blackwell family
    # targets (sm_121f) the ptxas STATIC __shared__ cap is 48KB, so a >48KB
    # staging tile (the routed dstates C-tile TMA kernel emits ~98KB) is
    # rejected by ptxas under compute_120f. Routing the shared tile to the
    # ``shared.dyn`` scope makes it a DYNAMIC allocation: the
    # MergeDynamicSharedMemoryAllocations pass folds it into the single
    # ``buf_dyn_shmem`` extern __shared__, ptxas no longer counts it against
    # the static cap, and the TVM CUDA runtime raises the dynamic limit via
    # cuFuncSetAttribute(MaxDynamicSharedMemorySize). This is the PROPER
    # dynamic opt-in (no hack): it only flips the storage scope; the
    # downstream merge + codegen + runtime opt-in already handle shared.dyn.
    # Gated behind TL_FORCE_DYN_SHARED=1 so the SAFE default (static shared,
    # global sm_121a) and path_c production are untouched.
    if scope == "shared" and _os.environ.get("TL_FORCE_DYN_SHARED") == "1":
        scope = "shared.dyn"
    try:
        # Force ``elem_offset=0`` so TVM doesn't auto-create a free
        # ``\u003cname\u003e_elem_offset`` Var that MakePackedAPI would flag as
        # undefined. Tile-scoped buffers are always zero-offset.
        buf = tir.decl_buffer(shape_list, dtype, name=name, scope=scope, elem_offset=tir.const(0, "int32"))
    except TypeError:
        # Older decl_buffer signatures don't accept ``scope`` kw -- fall
        # back to the unscoped form. The buffer still bypasses buffer_map
        # via ``ctx.local_buffers`` so VerifyMemory will skip it.
        try:
            buf = tir.decl_buffer(shape_list, dtype, name=name, elem_offset=tir.const(0, "int32"))
        except TypeError:
            buf = tir.decl_buffer(shape_list, dtype, name=name)
    ctx.local_buffers.append(buf)
    return buf


def _result_ssa_name(op: Any) -> Optional[str]:
    """Printed RESULT SSA name of ``op``'s first result (fold lookup key)."""
    results = _results(op)
    if not results:
        return None
    value = results[0]
    for attr in ("get_name", "name"):
        getter = getattr(value, attr, None)
        if callable(getter):
            try:
                out = str(getter())
                if out:
                    return out
            except Exception:
                pass
        elif isinstance(getter, str) and getter:
            return getter
    try:
        s = str(value).strip()
        if s:
            head = s.split()[0]
            if head.startswith("%"):
                return head
    except Exception:
        pass
    return None


def should_fold_addressing(ctx: "WalkerCtx", op: Any) -> bool:
    """Return True iff ``op``'s tile result should be kept lazy (transform 1).

    The result is folded into the consuming tt.load/tt.store region indices +
    predicate (Coalesce-style) instead of being materialized into a spilled
    addressing/mask array. Gated to the routed-triton prologue-opt path; the
    fold set (``build_addressing_fold_set``) holds eligible RESULT SSA names --
    the cooperative GEMM operand tiles (whose results feed tt.dot) are never in
    it, so the GEMM path stays byte-identical.

    Match is by RESULT-position SSA name. The use GRAPH was reconstructed by
    Value identity in the pre-pass (reliable); the lookup KEY is a name because
    in-loop ops reach the emitter via a region-child ``WalkerCtx``
    (``_emit_region``) whose Value wrappers are not identity-stable across the
    child boundary, while result->result name spelling IS consistent.
    """
    if not getattr(ctx, "routed_triton_prologue_opt", False):
        return False
    fold_set = getattr(ctx, "fold_addressing_ssa", None)
    if not fold_set:
        return False
    name = _result_ssa_name(op)
    if name is None:
        return False
    return (
        name in fold_set
        or name.lstrip("%") in fold_set
        or f"%{name.lstrip('%')}" in fold_set
    )


def materialize_lazy_tile(
    ctx: "WalkerCtx",
    expr: LazyTileExpr,
    shape: Optional[Sequence[Any]] = None,
    dtype: Optional[str] = None,
    *,
    name: str,
    scope: str = "local",
    loop_var_prefix: Optional[str] = None,
) -> Any:
    """Materialize a lane-indexable tile expression when a Buffer is required."""

    tir = ctx.tir()
    dst_shape = list(shape or expr.shape or [1])
    dst_dtype = _normalize_mlir_dtype(str(dtype or expr.dtype or "float32"))

    # PROLOGUE-OPT transform (2) decision (made BEFORE allocation so the tile
    # can be promoted to SHARED scope for the cooperative fill). See the long
    # comment below for the correctness contract.
    _total = 1
    for _extent in dst_shape or [1]:
        _total *= int(_extent)
    _nthreads = ctx.num_threads() if ctx else 0
    thread_distribute = (
        bool(getattr(ctx, "routed_triton_prologue_opt", False))
        and bool(getattr(ctx, "routed_triton_thread_distribute", False))
        and _nthreads > 0
        and _total > 1
        and not getattr(ctx, "requires_single_thread_body", False)
    )
    # When distributing, the buffer MUST be shared: the 128 lanes cooperatively
    # fill ONE buffer and then read it back through a ``__syncthreads`` barrier.
    # A thread-local buffer would leave 127/128 slots of each lane's private
    # copy uninitialized. RULE #1: shared+sync or the serial fill, never a
    # silently-wrong distributed write into local memory.
    alloc_scope = "shared" if thread_distribute else scope
    dst = _alloc_tile_buffer(
        ctx,
        dst_shape,
        dst_dtype,
        ctx.fresh(name),
        scope=alloc_scope,
    )
    loop_vars = [
        tir.Var(
            ctx.fresh(f"{loop_var_prefix}{axis}" if loop_var_prefix is not None else f"{name}_i{axis}"),
            "int32",
        )
        for axis, _extent in enumerate(dst_shape or [1])
    ]

    # PROLOGUE-OPT transform (2): THREAD-DISTRIBUTE the elementwise tile.
    # When the routed-triton gate is on, partition the flat lane space over
    # the block's ``num_warps*32`` threads instead of emitting a SERIAL
    # ``tir.For`` that every lane re-runs. ``flat = i_local*nthreads + tid``;
    # the multi-dim store/read indices are recovered from ``flat`` by the
    # row-major divmod of ``dst_shape``. A trailing ``flat < total`` guard
    # covers the remainder when ``total`` is not a multiple of ``nthreads``.
    total = _total
    nthreads = _nthreads

    if thread_distribute:
        tid = ctx.thread_idx_var()
        flat_var = tir.Var(ctx.fresh(f"{name}_flat"), "int32")
        flat = flat_var * tir.const(int(nthreads), "int32") + tid
        # Recover per-axis indices from the flattened thread-strided index.
        dist_loop_vars: list = []
        rem = flat
        strides = []
        acc = 1
        for extent in reversed(dst_shape or [1]):
            strides.append(acc)
            acc *= int(extent)
        strides = list(reversed(strides))  # row-major stride per axis
        shape_axes = dst_shape or [1]
        for axis, extent in enumerate(shape_axes):
            stride_c = tir.const(int(strides[axis]), "int32")
            # Row-major decode: idx[axis] = (flat // stride[axis]) % extent.
            # stride is 1 for the innermost axis, so the innermost reduces to
            # ``flat % extent``. The outermost axis' mod is redundant (flat <
            # total) but harmless; we apply mod on every non-unit axis.
            idx_axis = tir.floordiv(rem, stride_c) if int(strides[axis]) != 1 else rem
            if int(extent) != 1:
                idx_axis = tir.floormod(idx_axis, tir.const(int(extent), "int32"))
            dist_loop_vars.append(idx_axis)

        rank = len(expr.shape)
        if rank:
            if len(dist_loop_vars) >= rank:
                src_indices = list(dist_loop_vars[-rank:])
            else:
                src_indices = [tir.const(0, "int32")] * (rank - len(dist_loop_vars)) + list(dist_loop_vars)
            for axis, extent in enumerate(expr.shape):
                if int(extent) == 1:
                    src_indices[axis] = tir.const(0, "int32")
            value = expr.read_lane(ctx, tuple(src_indices))
        else:
            value = expr.read_lane(ctx, ())

        store = tir.BufferStore(dst, value, list(dist_loop_vars))
        # Per-lane bound: skip the tail lanes when total < n_iter*nthreads.
        if total % int(nthreads) != 0:
            store = tir.IfThenElse(
                flat < tir.const(int(total), "int32"), store, None
            )
        n_iter = (total + int(nthreads) - 1) // int(nthreads)
        body = tir.For(
            flat_var,
            tir.const(0, "int32"),
            tir.const(int(n_iter), "int32"),
            tir.ForKind.SERIAL,
            store,
        )
        ctx.emit(body)
        # Barrier: all lanes must finish their cooperative writes before ANY
        # lane reads the shared tile back. Without this a downstream per-lane
        # read of a slot owned by another lane races. ``tvm_storage_sync`` maps
        # to ``__syncthreads`` on CUDA.
        try:
            sync = tir.call_intrin(
                "int32",
                tir.op.Op.get("tir.tvm_storage_sync"),
                tir.StringImm("shared"),
            )
            ctx.emit(tir.Evaluate(sync))
        except Exception as exc:  # pragma: no cover -- intrin registry drift
            raise RuntimeError(
                "PROLOGUE-OPT transform (2): could not emit tvm_storage_sync "
                "barrier after the cooperative shared fill; refusing to emit a "
                "race-prone distributed tile (RULE #1)."
            ) from exc
        return dst

    rank = len(expr.shape)
    if rank:
        if len(loop_vars) >= rank:
            src_indices = list(loop_vars[-rank:])
        else:
            src_indices = [tir.const(0, "int32")] * (rank - len(loop_vars)) + list(loop_vars)
        for axis, extent in enumerate(expr.shape):
            if int(extent) == 1:
                src_indices[axis] = tir.const(0, "int32")
        value = expr.read_lane(ctx, tuple(src_indices))
    else:
        value = expr.read_lane(ctx, ())

    store = tir.BufferStore(
        dst,
        value,
        list(loop_vars) or [tir.const(0, "int32")],
    )
    body: Any = store
    for var, extent in zip(reversed(loop_vars), reversed(dst_shape or [1])):
        body = tir.For(
            var,
            tir.const(0, "int32"),
            tir.const(int(extent), "int32"),
            tir.ForKind.SERIAL,
            body,
        )
    ctx.emit(body)
    return dst


def _emit_load_copy(op: Any, ctx: "WalkerCtx", resolved: Dict[str, Any], mask_ssa: Any, other_ssa: Any) -> Any:
    """Emit ``T.copy(global[region], frag)`` and bind the result SSA to frag.

    The buffer-region path keeps the frontend on the high-level surface
    (RFC 5.1) so LayoutInference / LowerTileOp can apply uniformly. When
    a mask is present we still emit the copy and wrap a subsequent
    fragment-zeroing pass keyed on ``mask`` for the masked-out lanes -- the
    unconditional copy keeps shape inference simple while the mask predicate
    enforces semantics. ``other`` (Triton's masked-out fill) is honoured by
    pre-clearing the fragment to that value before the copy.
    """
    tir = ctx.tir()
    result = _results(op)[0] if _results(op) else None
    out_shape = _ptrstate_sizes_int(resolved) or list(_shape_of(result)) or [1024]
    out_dtype = _dtype_of(result) if result is not None else "float32"

    try:
        import tilelang.language as T  # type: ignore
    except ImportError:  # pragma: no cover -- dict-walker fallback path
        # No TileLang available: fall back to a single BufferLoad so the
        # walker stays usable in the dict-shaped unit-test environment.
        buf = _ptrstate_buffer(ctx, resolved, out_dtype)
        load_expr = tir.BufferLoad(buf, _ptrstate_offsets_or_zero(resolved))
        if result is not None:
            ctx.bind(result, load_expr)
        return load_expr

    src_buf = _ptrstate_buffer(ctx, resolved, out_dtype)
    # Use ``T.alloc_shared`` so the tile buffer lives in shared memory.
    # Metal GEMM's ``is_gemm_ss()`` check requires both A and B operand
    # tiles in shared scope; ``T.alloc_fragment`` creates ``local.fragment``
    # buffers which trip the ``"Unsupported gemm combination"`` error.
    alloc_fn = getattr(T, "alloc_shared", None) or T.alloc_fragment
    frag = alloc_fn(out_shape, out_dtype)

    # Build a buffer-region slice from offsets+sizes. The C++ side accepts
    # either Buffer or BufferRegion; we produce a tir.BufferRegion here so
    # downstream passes see explicit ranges.
    offs = _ptrstate_offsets_or_zero(resolved)
    region = []
    for i, sz in enumerate(out_shape):
        off = offs[i] if i < len(offs) else 0
        try:
            off_i = int(off)
        except (TypeError, ValueError):
            off_i = 0
        region.append((off_i, off_i + int(sz)))

    if other_ssa is not None and mask_ssa is not None:
        # Pre-clear the fragment to ``other`` so masked-out lanes carry the
        # right fill value after the unconditional copy.
        try:
            other_expr = ctx.get(other_ssa)
        except KeyError:
            other_expr = tir.const(0, out_dtype)
        if hasattr(T, "fill"):
            T.fill(frag, other_expr)

    # Emit the copy. Slicing the global buffer with explicit ranges keeps
    # the lowered IR readable; the helper handles BufferRegion construction.
    src_slice = src_buf
    if hasattr(T, "region"):
        try:
            src_slice = T.region(src_buf, "r", *[r[0] for r in region])
        except Exception:  # pragma: no cover -- API drift
            src_slice = src_buf
    T.copy(src_slice, frag)

    # ITERATION 3 (coalesced async loads). Record that this body emitted a
    # global->shared cooperative copy. ``frag`` lives in shared scope
    # (``alloc_shared`` above) and ``src_slice`` is the global operand buffer,
    # so this is a cp.async-eligible producer. ``map_scf_for`` reads the delta
    # on ``ctx._gmem_shared_copies`` across its body emission to decide whether
    # the enclosing serial K-loop should carry the ``num_stages`` software-
    # pipeline annotation that routes these copies through ``LowerPTXAsyncCopy``
    # (cp.async / SASS LDGSTS). Only counted when the alloc actually landed in
    # shared scope -- the ``alloc_fragment`` fallback (Metal gemm-ss path) is
    # not cp.async-eligible. RULE #1: count the real shared producer or nothing
    # -- never claim a coalesced copy that did not get a shared destination.
    if getattr(ctx, "routed_triton_async_loads", False) and alloc_fn is getattr(T, "alloc_shared", None):
        shared_copies = getattr(ctx, "_gmem_shared_copies", None)
        if isinstance(shared_copies, list):
            shared_copies.append(1)

    # Mask is applied lane-wise after the copy via T.if_then_else when
    # downstream consumers need it; the SSA bound to ``result`` is the
    # fragment buffer so dependent ops (gemm, reduce) see the tile.
    if mask_ssa is not None:
        try:
            mask_expr = ctx.get(mask_ssa)
            if hasattr(T, "if_then_else"):
                # Stash a follow-up predicate so consumers can re-apply if
                # the masked-out fill needs to differ from ``other``. We
                # bind the (frag, mask) pair instead of just the fragment.
                ctx.value_map[result] = (frag, mask_expr)  # noqa: E501
                return frag
        except KeyError:
            pass

    if result is not None:
        ctx.bind(result, frag)
    return frag


def _emit_store_copy(op: Any, ctx: "WalkerCtx", resolved: Dict[str, Any], val_expr: Any, mask_ssa: Any) -> Any:
    """Emit ``T.copy(val_frag, global[region])`` for the buffer-region path."""
    tir = ctx.tir()
    # Pull dtype from the value being stored where possible.
    dtype = "float32"
    try:
        dtype = str(getattr(val_expr, "dtype", None) or "float32")
    except Exception:
        pass
    if dtype in {"", "handle"}:
        dtype = "float32"

    try:
        import tilelang.language as T  # type: ignore
    except ImportError:  # pragma: no cover
        buf = _ptrstate_buffer(ctx, resolved, dtype)
        store_stmt = tir.BufferStore(buf, val_expr, _ptrstate_offsets_or_zero(resolved))
        if mask_ssa is not None:
            try:
                mask_expr = ctx.get(mask_ssa)
                store_stmt = tir.IfThenElse(mask_expr, store_stmt, None)
            except KeyError:
                pass
        ctx.emit(store_stmt)
        return store_stmt

    dst_buf = _ptrstate_buffer(ctx, resolved, dtype)
    handle = T.copy(val_expr, dst_buf)
    if mask_ssa is not None:
        # Predicate guards the entire copy; LowerTileOp will fuse this with
        # the per-lane mask once layouts are inferred.
        try:
            mask_expr = ctx.get(mask_ssa)
            handle = tir.IfThenElse(mask_expr, tir.Evaluate(handle), None)
            ctx.emit(handle)
            return handle
        except KeyError:
            pass
    if isinstance(handle, tir.PrimExpr):
        stmt = tir.Evaluate(handle)
        ctx.emit(stmt)
        return stmt
    ctx.emit(handle)
    return handle


# ---------------------------------------------------------------------------
# Memory ops -- RFC section 5.1, "tt.load" / "tt.store" / "tt.atomic_rmw"
# ---------------------------------------------------------------------------


def _atomic_rmw_kind(op: Any) -> str:
    """Extract the RMW kind from a TTIR ``tt.atomic_rmw`` op.

    Triton's ``RMWOp`` enum has values ``and``, ``or``, ``xor``, ``add``,
    ``fadd``, ``max``, ``min``, ``umax``, ``umin``, ``xchg``. We
    canonicalize to ``add``/``max``/``min``/``xchg``/``and``/``or``/``xor``
    after stripping the ``f``/``u`` integer-vs-float prefix.
    """
    # Wave E3: ``rmw_op`` is a Triton 3.6 inherent (Properties-storage) attr.
    # Use the shared helper so jaxlib's empty op.attributes path falls back
    # to parsing the printed ``<{rmw_op = ...}>`` block instead of silently
    # defaulting to None and tripping the "missing 'rmw_op'" error.
    attrs = _attrs_with_properties_shared(op)
    raw = attrs.get("rmw_op") or attrs.get("atomic_rmw_op") or attrs.get("kind")
    if raw is None:
        raise EmitError("tt.atomic_rmw: missing 'rmw_op' attribute")
    s = str(raw).lower().strip()
    # The custom TTIR printer spells the op as ``fadd``. After the C++ shim
    # converts it to generic form, Triton's I32EnumAttr value is printed as
    # an integer property, e.g. ``atomic_rmw_op = 5`` for ``fadd``.
    numeric_enum = {
        "1": "and",
        "2": "or",
        "3": "xor",
        "4": "add",
        "5": "fadd",
        "6": "max",
        "7": "min",
        "8": "umax",
        "9": "umin",
        "10": "exch",
    }
    s = numeric_enum.get(s, s)
    # Strip MLIR-style prefixes/suffixes that occasionally appear.
    if s.startswith("rmw_op."):
        s = s[len("rmw_op.") :]
    if s.startswith("f"):
        # fadd / fmax / fmin -> add / max / min
        rest = s[1:]
        if rest in {"add", "max", "min"}:
            return rest
    if s.startswith("u") and s[1:] in {"max", "min"}:
        # umax / umin -> max / min (signedness disambiguated by buffer dtype)
        return s[1:]
    if s == "exch":
        return "xchg"
    return s


def _constant_tile_bool(value: Any) -> Optional[bool]:
    """Return bool for a splat boolean tile mask, else ``None``."""
    if not isinstance(value, LazyTileExpr):
        return None
    if value.constant_value is None:
        return None
    if str(value.dtype) != "bool":
        return None
    return bool(value.constant_value)


def _shape_product(shape: Sequence[Any]) -> int:
    total = 1
    for extent in shape:
        total *= int(extent)
    return total


def _atomic_tile_shape(
    val_ssa: Any,
    val_expr: Any,
    indices: Sequence[Any],
    mask_expr: Any,
) -> Tuple[int, ...]:
    """Best-effort lane shape for vector/tile atomic operands."""
    candidates: List[Sequence[Any]] = []
    for value in (val_expr, mask_expr, *indices):
        if isinstance(value, LazyTileExpr):
            candidates.append(value.shape)
        else:
            shape = getattr(value, "shape", None)
            if shape is not None:
                candidates.append(tuple(shape))
    candidates.append(_shape_of(val_ssa))
    for shape in candidates:
        if shape and _shape_product(shape) > 1:
            return tuple(int(s) for s in shape)
    return ()


def _flat_lane_index(tir: Any, shape: Sequence[int], loop_vars: Sequence[Any]) -> Any:
    if not loop_vars:
        return tir.const(0, "int32")
    flat = loop_vars[0]
    for var, extent in zip(loop_vars[1:], shape[1:]):
        flat = flat * tir.const(int(extent), "int32") + var
    return flat


def _lane_indices_for_shape(
    tir: Any,
    value_shape: Sequence[int],
    outer_shape: Sequence[int],
    loop_vars: Sequence[Any],
    flat_lane: Any,
) -> Tuple[Any, ...]:
    if not value_shape:
        return ()
    if len(value_shape) == len(loop_vars):
        return tuple(tir.const(0, "int32") if int(extent) == 1 else loop_vars[i] for i, extent in enumerate(value_shape))
    if len(value_shape) == 1:
        return (flat_lane,)
    indices: List[Any] = []
    rem = flat_lane
    for axis, extent in enumerate(value_shape):
        stride = 1
        for trailing in value_shape[axis + 1 :]:
            stride *= int(trailing)
        if stride == 1:
            idx = rem
        else:
            idx = rem // tir.const(stride, "int32")
            rem = rem - idx * tir.const(stride, "int32")
        indices.append(tir.const(0, "int32") if int(extent) == 1 else idx)
    return tuple(indices)


def _atomic_read_lane(
    ctx: WalkerCtx,
    value: Any,
    outer_shape: Sequence[int],
    loop_vars: Sequence[Any],
    flat_lane: Any,
) -> Any:
    """Read one lane from lazy, buffer, or vector-valued atomic operands."""
    tir = ctx.tir()
    tvm_mod = ctx.tvm()
    if isinstance(value, LazyTileExpr):
        lane_indices = _lane_indices_for_shape(tir, value.shape, outer_shape, loop_vars, flat_lane)
        return value.read_lane(ctx, lane_indices)
    if isinstance(value, tvm_mod.tir.Buffer):
        rank = len(value.shape)
        if rank == 0:
            return tir.BufferLoad(value, [tir.const(0, "int32")])
        if rank == len(loop_vars):
            return tir.BufferLoad(value, list(loop_vars))
        return tir.BufferLoad(value, [flat_lane])

    bcast_cls = getattr(tir, "Broadcast", None)
    if bcast_cls is not None and isinstance(value, bcast_cls):
        return value.value
    ramp_cls = getattr(tir, "Ramp", None)
    if ramp_cls is not None and isinstance(value, ramp_cls):
        return value.base + value.stride * flat_lane

    dt = getattr(value, "dtype", None)
    if dt is not None and "x" not in str(dt):
        return value

    binop_pyops = {
        "Add": lambda a, b: a + b,
        "Sub": lambda a, b: a - b,
        "Mul": lambda a, b: a * b,
        "Div": lambda a, b: a / b,
        "Mod": lambda a, b: a % b,
        "FloorDiv": lambda a, b: a // b,
        "FloorMod": lambda a, b: a % b,
    }
    for cls_name, pyop in binop_pyops.items():
        cls = getattr(tir, cls_name, None)
        if cls is not None and isinstance(value, cls):
            lhs = _atomic_read_lane(ctx, value.a, outer_shape, loop_vars, flat_lane)
            rhs = _atomic_read_lane(ctx, value.b, outer_shape, loop_vars, flat_lane)
            return pyop(lhs, rhs)
    for cls_name, fn_name in (("Min", "min"), ("Max", "max")):
        cls = getattr(tir, cls_name, None)
        if cls is not None and isinstance(value, cls):
            lhs = _atomic_read_lane(ctx, value.a, outer_shape, loop_vars, flat_lane)
            rhs = _atomic_read_lane(ctx, value.b, outer_shape, loop_vars, flat_lane)
            fn = getattr(tir, fn_name, None)
            if fn is not None:
                return fn(lhs, rhs)
            return tir.Select(lhs < rhs, lhs, rhs) if fn_name == "min" else tir.Select(lhs > rhs, lhs, rhs)
    cast_cls = getattr(tir, "Cast", None)
    if cast_cls is not None and isinstance(value, cast_cls):
        scalar_dtype = str(value.dtype).rsplit("x", 1)[0]
        return tir.Cast(scalar_dtype, _atomic_read_lane(ctx, value.value, outer_shape, loop_vars, flat_lane))

    return value


def _atomic_intrin_call(
    tir: Any,
    *,
    kind: str,
    buf: Any,
    indices: Sequence[Any],
    val_expr: Any,
    ret_dtype: str,
) -> Any:
    tl_intrin_names = {
        "add": ("tl.atomic_add_elem_op", "tl.atomic_add_ret_elem_op"),
        "max": ("tl.atomic_max_elem_op", "tl.atomic_max_ret_elem_op"),
        "min": ("tl.atomic_min_elem_op", "tl.atomic_min_ret_elem_op"),
        "xchg": ("tl.atomic_xchg_elem_op", "tl.atomic_xchg_ret_elem_op"),
        "and": ("tl.atomic_and_elem_op", "tl.atomic_and_ret_elem_op"),
        "or": ("tl.atomic_or_elem_op", "tl.atomic_or_ret_elem_op"),
        "xor": ("tl.atomic_xor_elem_op", "tl.atomic_xor_ret_elem_op"),
    }
    try:
        addr = tir.call_intrin(
            "handle",
            tir.op.Op.get("tir.address_of"),
            tir.BufferLoad(buf, list(indices)),
        )
    except Exception:  # pragma: no cover -- older TVM bindings
        addr = buf
    if kind in tl_intrin_names:
        no_ret, with_ret = tl_intrin_names[kind]
        op_name = with_ret if ret_dtype != "handle" else no_ret
        return tir.call_intrin(ret_dtype, tir.op.Op.get(op_name), addr, val_expr)
    intrin_name = f"tir.atomic_{kind}"
    return tir.call_intrin(ret_dtype, intrin_name, addr, val_expr)


def _emit_tile_atomic_rmw(
    op: Any,
    ctx: WalkerCtx,
    *,
    kind: str,
    buf: Any,
    indices: Sequence[Any],
    val_expr: Any,
    val_ssa: Any,
    mask_expr: Any,
    return_prev: bool,
) -> Any:
    """Emit per-lane scalar atomics for vector/tile ``tt.atomic_rmw``."""
    tir = ctx.tir()
    shape = _atomic_tile_shape(val_ssa, val_expr, indices, mask_expr)
    if not shape:
        raise EmitError("tt.atomic_rmw: tile atomic requested without tile shape")

    loop_vars = [tir.Var(ctx.fresh(f"atomic_i{axis}"), "int32") for axis, _extent in enumerate(shape)]
    flat_lane = _flat_lane_index(tir, shape, loop_vars)
    target_indices = [_atomic_read_lane(ctx, idx, shape, loop_vars, flat_lane) for idx in indices] or [tir.const(0, "int32")]
    val_lane = _atomic_read_lane(ctx, val_expr, shape, loop_vars, flat_lane)
    result_dtype = _dtype_of(_results(op)[0]) if _results(op) else _dtype_of(val_ssa)
    ret_dtype = result_dtype if return_prev else "handle"
    atomic_call = _atomic_intrin_call(
        tir,
        kind=kind,
        buf=buf,
        indices=target_indices,
        val_expr=val_lane,
        ret_dtype=ret_dtype,
    )

    body: Any
    result_buf = None
    if return_prev:
        result_buf = _alloc_tile_buffer(
            ctx,
            shape,
            result_dtype,
            ctx.fresh("atomic_prev"),
        )
        body = tir.BufferStore(result_buf, atomic_call, list(loop_vars))
    else:
        body = tir.Evaluate(atomic_call)

    if mask_expr is not None:
        constant_mask = _constant_tile_bool(mask_expr)
        if constant_mask is False:
            body = tir.Evaluate(tir.const(0, "int32"))
        elif constant_mask is not True:
            mask_lane = _atomic_read_lane(ctx, mask_expr, shape, loop_vars, flat_lane)
            body = tir.IfThenElse(mask_lane, body, None)

    for var, extent in zip(reversed(loop_vars), reversed(shape)):
        body = tir.For(
            var,
            tir.const(0, "int32"),
            tir.const(int(extent), "int32"),
            tir.ForKind.SERIAL,
            body,
        )
    # This serial scalar atomic loop iterates EVERY tile element; if all 128
    # lanes ran it we would issue 128x duplicate atomics (intra-block race /
    # over-accumulation). Serialise it onto lane 0 -- but LOCALLY, wrapping
    # only this loop in ``if threadIdx_x == 0`` rather than flipping a
    # ctx-global flag that would nest the ENTIRE PrimFunc body (including any
    # adjacent collective ``T.gemm``) under the guard. Gating a gemm on a
    # single lane collapses its ``thread_bounds`` extent to 0, which trips
    # ``gemm.lower``'s ``m_warp*n_warp==num_warps`` ICHECK with ``num_warps=0``
    # (the exact failure on ``_chunk_scan_bwd_dx`` /
    # ``_chunk_state_bwd_ddAcs_stable``, which carry both an atomic_rmw AND a
    # real TF32 gemm). The shared ``threadIdx_x`` Var matches the one
    # ``map_tt_func`` binds for the outer block thread_extent.
    tid_var = ctx.thread_idx_var()
    body = tir.IfThenElse(
        tir.EQ(tid_var, tir.const(0, "int32")),
        body,
        None,
    )
    ctx.emit(body)
    if return_prev and result_buf is not None:
        ctx.bind(_results(op)[0], result_buf)
    return body


def map_tt_atomic_rmw(op: Any, ctx: WalkerCtx) -> Any:
    """Lower ``tt.atomic_rmw`` to ``T.atomic_*`` / ``call_intrin``.

    Recipe (RFC 5.1):
    * Read the ``rmw_op`` attribute (``add``/``max``/``min``/``xchg``/...).
    * Resolve the pointer via ``ptr_analysis`` to ``(buffer, indices)``.
    * Emit ``tilelang.language.atomic_add`` / ``atomic_max`` / etc. when
      a TileLang intrinsic exists; otherwise fall back to
      ``tir.call_intrin("tir.atomic_<op>", buffer.access_ptr("w"), val)``.
    * Mask handling matches ``tt.store``: wrap in ``if_then_else``.
    * Result SSA (the pre-update value) binds the call's return value.
    """
    operands = _operands(op)
    if len(operands) < 2:
        raise EmitError("tt.atomic_rmw: expected at least (ptr, val) operands")
    ptr_ssa, val_ssa = operands[0], operands[1]
    mask_ssa = operands[2] if len(operands) >= 3 else None

    resolved = ctx.get(ptr_ssa)
    if isinstance(resolved, tuple) and len(resolved) == 2:
        buf, indices = resolved
    else:
        buf, indices = resolved, [0]
    val_expr = ctx.get(val_ssa)
    mask_expr = ctx.get(mask_ssa) if mask_ssa is not None else None
    kind = _atomic_rmw_kind(op)

    tir = ctx.tir()
    # Lazy-import TileLang only when emitting (test path uses dicts but
    # never reaches this branch unless TileLang is available).
    try:
        import tilelang.language as T  # type: ignore
    except ImportError:  # pragma: no cover -- tilelang absent
        T = None  # type: ignore

    # We model the destination as buf[indices] -- the scalar atomic path.
    # ``atomic_add`` / ``atomic_max`` / ``atomic_min`` accept a Buffer; we
    # pass the underlying buffer with the indices encoded in ``val_expr``
    # by way of BufferLoad / BufferStore semantics handled inside TileLang.
    return_prev = _has_consumed_result(op, ctx)

    tile_shape = _atomic_tile_shape(val_ssa, val_expr, list(indices), mask_expr)
    mask_is_effective = mask_expr is not None and _constant_tile_bool(mask_expr) is not True
    if tile_shape and mask_is_effective:
        return _emit_tile_atomic_rmw(
            op,
            ctx,
            kind=kind,
            buf=buf,
            indices=list(indices),
            val_expr=val_expr,
            val_ssa=val_ssa,
            mask_expr=mask_expr,
            return_prev=return_prev,
        )

    if T is not None and kind in {"add", "max", "min", "xchg", "and", "or", "xor"}:
        atomic_fn = {
            "add": T.atomic_add,
            "max": T.atomic_max,
            "min": T.atomic_min,
            "xchg": T.atomic_xchg,
            "and": T.atomic_and,
            "or": T.atomic_or,
            "xor": T.atomic_xor,
        }[kind]
        # The scalar TileLang path expects a Buffer directly. When indices
        # are nonzero the caller (ptr_analysis) is expected to slice the
        # buffer ahead of time; the MVP path uses a flat buffer view.
        #
        # tilelang.language.atomic.{add,max,min} dispatch to a
        # "tile-region" lowering when ``dst`` has any extent (i.e. shape !=
        # None). That path does NOT support ``return_prev`` for fp32 today
        # and raises ``NotImplementedError``. We detect this case ahead of
        # time and downgrade to ``return_prev=False``, binding the result
        # SSA to the call's handle so downstream consumers that only
        # treat the SSA as a statement (the typical Triton pattern --
        # users discard the prev value) keep working. A deprecation
        # warning is emitted so the bypass is visible, NOT silent.
        downgraded_return_prev = False
        if return_prev and kind in {"add", "max", "min"}:
            try:
                from tilelang.language.utils import get_extent  # type: ignore

                dst_has_extent = get_extent(buf) is not None
            except Exception:
                dst_has_extent = False
            res_dtype = _dtype_of(_results(op)[0]) if _results(op) else ""
            if dst_has_extent and res_dtype.startswith("float"):
                import warnings as _warnings

                _warnings.warn(
                    f"map_tt_atomic_rmw: return_prev unsupported on tile-region "
                    f"path for dtype={res_dtype!r} (kind={kind!r}); downgrading "
                    f"to return_prev=False and binding result SSA to the call "
                    f"handle. This is a deterministic dispatch, not a silent "
                    f"fallback -- emit a tir.atomic_{kind} via the generic "
                    f"path if you need the prev value.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                return_prev = False
                downgraded_return_prev = True
        result = atomic_fn(buf, val_expr, return_prev=return_prev)
        # If we downgraded, restore ``return_prev`` for the result-binding
        # branch below so the original ``res_ssa`` still gets bound (to the
        # handle expression) -- otherwise downstream walkers that look up
        # the SSA via ``ctx.get`` would raise KeyError.
        if downgraded_return_prev:
            return_prev = True
    else:
        # Unknown rmw_op: fall back to a generic ``tir.atomic_<op>`` extern.
        intrin_name = f"tir.atomic_{kind}"
        ret_dtype = _dtype_of(_results(op)[0]) if return_prev and _results(op) else "handle"
        # Build access pointer: buf.access_ptr("rw") + offset_indices.
        # For the MVP we let the buffer's flat element 0 be the target;
        # ptr_analysis fills in proper indices for non-trivial cases.
        access = (
            tir.call_intrin(
                "handle",
                tir.op.Op.get("tir.address_of"),
                tir.BufferLoad(buf, list(indices)),
            )
            if hasattr(tir.op.Op, "get")
            else buf
        )
        result = tir.call_intrin(ret_dtype, intrin_name, access, val_expr)

    if mask_ssa is not None:
        constant_mask = _constant_tile_bool(mask_expr)
        if constant_mask is True:
            mask_expr = None
        elif constant_mask is False:
            if return_prev:
                result = tir.const(0, _dtype_of(_results(op)[0]))
                ctx.bind(_results(op)[0], result)
                return result
            return result
    if mask_ssa is not None and mask_expr is not None:
        # Wrap in if_then_else; for non-prev returns we still emit the call
        # unconditionally because TileLang intrinsics treat the call as a
        # statement-level handle.
        if return_prev:
            zero = tir.const(0, _dtype_of(_results(op)[0]))
            result = tir.if_then_else(mask_expr, result, zero)
        else:
            result = tir.IfThenElse(mask_expr, tir.Evaluate(result), None)

    if return_prev:
        if downgraded_return_prev:
            if isinstance(result, tir.PrimExpr):
                ctx.emit(tir.Evaluate(result))
            else:
                ctx.emit(result)
        ctx.bind(_results(op)[0], result)
    else:
        ctx.emit(result)
    return result


# ---------------------------------------------------------------------------
# Compute ops -- RFC section 5.1, "tt.dot" / "tt.reduce" / "tt.where"
# ---------------------------------------------------------------------------


def _reduce_combiner_kind(op: Any) -> str:
    """Inspect a ``tt.reduce`` op's combiner region and return its kind.

    Real MLIR carries the combiner inside a region; the dict-shaped fake
    op simply exposes a top-level ``combiner`` field. Result is one of
    ``"add"``, ``"max"``, ``"min"``, ``"mul"``.
    """
    if isinstance(op, dict):
        c = op.get("combiner")
        if c is not None:
            s = str(c).lower()
            if s in {"add", "sum", "addf", "addi"}:
                return "add"
            if s in {"max", "maxnumf", "maxsi", "maxui"}:
                return "max"
            if s in {"min", "minnumf", "minsi", "minui"}:
                return "min"
            if s in {"mul", "prod", "mulf", "muli"}:
                return "mul"
            return s
    # Real MLIR path: the first op of the first region tells us the kind.
    regions = getattr(op, "regions", None) or ()
    for region in regions:
        for block in getattr(region, "blocks", ()) or ():
            for inner in getattr(block, "operations", ()) or ():
                name = getattr(inner, "name", "")
                low = str(name).lower()
                if "add" in low:
                    return "add"
                if "max" in low:
                    return "max"
                if "min" in low:
                    return "min"
                if "mul" in low:
                    return "mul"
    raise EmitError("tt.reduce: cannot determine combiner kind from op")


def map_tt_where(op: Any, ctx: WalkerCtx) -> Any:
    """Lower ``tt.where(cond, t, f)`` to ``tir.Select`` (lane-wise)."""
    tir = ctx.tir()
    operands = _operands(op)
    if len(operands) != 3:
        raise EmitError(f"tt.where: expected 3 operands (cond, true, false); got {len(operands)}")
    cond, t_val, f_val = (ctx.get(o) for o in operands)
    sel = tir.Select(cond, t_val, f_val)
    if _results(op):
        ctx.bind(_results(op)[0], sel)
    return sel


# ---------------------------------------------------------------------------
# Shape ops -- RFC section 5.1, broadcast/splat/expand_dims/reshape/make_range
# ---------------------------------------------------------------------------


def map_tt_trans(op: Any, ctx: WalkerCtx) -> Any:
    """Lower ``tt.trans`` (logical transpose) to a sidecar bind.

    Triton's ``tt.trans`` flips two axes of a tile (default: the last
    two). At the TIR layer we don't materialise the transpose; instead
    we rebind the result SSA to the *same* TIR value as the source and
    record the flipped-axis pair in ``ctx.transposed_views`` so that a
    downstream ``tt.dot`` consumer can fold it into ``transpose_A`` /
    ``transpose_B`` on the emitted ``T.gemm`` call. This matches the
    Triton pattern used by ``tl.dot(A, B, trans_b=True)`` which the
    frontend lowers as ``%bt = tt.trans %b; tt.dot %a, %bt`` when the
    ``trans_b`` kwarg has been propagated as a separate op.
    """
    operands = _operands(op)
    if not operands:
        raise EmitError("tt.trans: missing source operand")
    src_ssa = operands[0]
    src = ctx.get(src_ssa)
    # Wave E3: ``order`` (Triton's permutation tuple) is an inherent attr in
    # Triton 3.6 stored as a Property. Without the shared helper, jaxlib's
    # op.attributes view is empty and we'd default to the last-two-axes
    # swap even when the kernel asked for a different permutation.
    attrs = _attrs_with_properties_shared(op)
    # Default to flipping the last two axes; honour an explicit ``order``
    # attribute when it carries a 2-element permutation that swaps two
    # axes (Triton's general form, but matmul callers always use a swap).
    order = attrs.get("order")
    if order is not None and len(tuple(order)) >= 2:
        a, b = int(tuple(order)[-2]), int(tuple(order)[-1])
    else:
        a, b = -2, -1
    if _results(op):
        result_ssa = _results(op)[0]
        ctx.bind(result_ssa, src)
        # If the source was itself transposed, double-transpose cancels.
        if src_ssa in ctx.transposed_views:
            ctx.transposed_views.pop(result_ssa, None)
        else:
            ctx.transposed_views[result_ssa] = (a, b)
    return src


# ---------------------------------------------------------------------------
# Async / barrier -- RFC section 5.1, async_copy / mbarrier
# ---------------------------------------------------------------------------


def map_tt_async_copy(op: Any, ctx: WalkerCtx) -> Any:
    """Lower ``async_copy`` / ``tt.async_commit`` / ``tt.async_wait``.

    Recipe (RFC 5.1):
    * For the load form (``tt.async_copy_global_to_local`` /
      ``async_copy``): resolve src and dst via ``ptr_analysis`` and emit
      ``tilelang.language.async_copy(src, dst)`` -- the call carries the
      pipeline-stage annotation consumed by
      ``InjectSoftwarePipeline`` / ``LowerPTXAsyncCopy``.
    * ``tt.async_commit_group`` and ``tt.async_wait`` are TIR-layer
      no-ops: TileLang's ``inject_pipeline`` derives commit/wait from
      the surrounding ``T.Pipelined`` frame, so we do not need to emit
      anything here. The ops are still consumed by the walker so they
      are not flagged as unmapped.
    """
    op_name = op.get("name") if isinstance(op, dict) else getattr(op, "name", "")
    name = str(op_name).lower()

    # Commit / wait are pipeline boundary markers.
    if "commit" in name:
        import tilelang.language as T  # type: ignore  # lazy

        if hasattr(T, "ptx_commit_group"):
            handle = T.ptx_commit_group()
        else:
            tir = ctx.tir()
            handle = tir.call_intrin("handle", tir.op.Op.get("tir.ptx_commit_group"))

        tir = ctx.tir()
        ctx.emit(tir.Evaluate(handle))
        return handle

    if "wait" in name:
        attrs = _attrs_with_properties_shared(op)
        num = int(attrs.get("num", 0))
        import tilelang.language as T  # type: ignore  # lazy

        if hasattr(T, "ptx_wait_group"):
            handle = T.ptx_wait_group(num)
        else:
            tir = ctx.tir()
            handle = tir.call_intrin("handle", tir.op.Op.get("tir.ptx_wait_group"), num)

        tir = ctx.tir()
        ctx.emit(tir.Evaluate(handle))
        return handle

    operands = _operands(op)
    if len(operands) < 2:
        raise EmitError("async_copy: expected (src_ptr, dst_ptr) operands")
    src_ssa, dst_ssa = operands[0], operands[1]
    src_resolved = ctx.get(src_ssa)
    dst_resolved = ctx.get(dst_ssa)

    # Unpack (buffer, indices) tuples produced by ptr_analysis; otherwise
    # treat the resolved value as a buffer/region directly.
    def _materialize(resolved: Any) -> Any:
        tir_mod = ctx.tir()
        if isinstance(resolved, tuple) and len(resolved) == 2:
            buf, indices = resolved
            return tir_mod.BufferLoad(buf, list(indices))
        return resolved

    src = _materialize(src_resolved)
    dst = _materialize(dst_resolved)

    import tilelang.language as T  # type: ignore  # lazy

    handle = T.async_copy(src, dst)
    ctx.emit(handle)
    return handle


def map_tt_mbarrier(op: Any, ctx: WalkerCtx) -> Any:
    """Lower ``mbarrier`` ops to ``T.alloc_barrier`` + sync intrinsics.

    Recipe (RFC 5.1):
    * For ``tt.barrier_init`` / ``mbarrier.init``: emit
      ``tilelang.language.alloc_barrier(N)`` and bind the result SSA to
      the returned barrier buffer.
    * For ``tt.barrier_arrive`` / ``mbarrier.arrive``: emit
      ``T.barrier_arrive(barrier)``.
    * For ``tt.barrier_wait`` / ``mbarrier.wait``: emit
      ``T.barrier_wait(barrier, parity)``.
    * For plain barriers (no operand): emit ``T.call_intrin`` for
      ``tir.tvm_storage_sync("shared")``; the backend codegen picks
      __syncthreads / s_barrier / threadgroup_barrier per target.
    """
    op_name = op.get("name") if isinstance(op, dict) else getattr(op, "name", "")
    name = str(op_name).lower()
    # Wave E3: ``count``/``arrive_count``/``parity`` (i.e. the barrier_kind
    # tunables) are Triton 3.6 inherent attrs stored as Properties. Use the
    # shared helper so jaxlib's empty op.attributes path falls back to
    # parsing ``<{count = N : i32}>`` from the printed op text.
    attrs = _attrs_with_properties_shared(op)

    import tilelang.language as T  # type: ignore  # lazy

    if "init" in name:
        arrive_count = int(attrs.get("count") or attrs.get("arrive_count") or 1)
        bar = T.alloc_barrier(arrive_count)
        if _results(op):
            ctx.bind(_results(op)[0], bar)
        return bar

    operands = _operands(op)
    tir = ctx.tir()
    if "arrive" in name:
        if not operands:
            raise EmitError("mbarrier.arrive: missing barrier operand")
        bar = ctx.get(operands[0])
        # Prefer the high-level T.barrier_arrive (re-exported via
        # tilelang.language.builtin). Fall back to a raw call_intrin so
        # the emitter still works if the symbol isn't re-exported.
        # TODO: verify barrier_arrive remains in tilelang.language.
        if hasattr(T, "barrier_arrive"):
            handle = T.barrier_arrive(bar)
        else:
            handle = tir.call_intrin("handle", tir.op.Op.get("tl.mbarrier_arrive"), bar)
        ctx.emit(handle)
        return handle

    if "wait" in name:
        if not operands:
            raise EmitError("mbarrier.wait: missing barrier operand")
        bar = ctx.get(operands[0])
        parity = int(attrs.get("parity", 0))
        # Prefer the high-level T.barrier_wait; fall back to call_intrin.
        if hasattr(T, "barrier_wait"):
            handle = T.barrier_wait(bar, parity)
        else:
            handle = tir.call_intrin("handle", tir.op.Op.get("tl.mbarrier_wait_parity"), bar, parity)
        ctx.emit(handle)
        return handle

    # Plain ``__syncthreads`` style barrier.
    tir = ctx.tir()
    handle = tir.call_intrin("handle", tir.op.Op.get("tir.tvm_storage_sync"), "shared")
    ctx.emit(handle)
    return handle


def map_tt_sync_threads_partial(op: Any, ctx: WalkerCtx) -> Any:
    """Lower ``tt.sync_threads_partial(mask, n_threads)`` to ``T.sync_threads_partial``.

    Recipe (cppmega.mlx topk_selector → unified pipeline migration, Phase 1):
    Triton-style radix-select kernels emit a partial-warp barrier so only the
    lanes set in ``mask`` rendezvous. Forward straight to the new TileLang
    primitive that lowers to ``__syncwarp(mask)`` on CUDA, a no-op on HIP
    (wavefront is hardware-convergent), and ``simdgroup_barrier`` on Metal.
    Also accepts the alias spellings ``triton.language.partial_barrier`` and
    ``tt.partial_barrier`` so multiple TTIR producers route through one
    emitter.
    """
    operands = _operands(op)
    if len(operands) < 2:
        raise EmitError(f"tt.sync_threads_partial: expected (mask, n_threads); got {len(operands)} operands")
    mask = ctx.get(operands[0])
    n_threads = ctx.get(operands[1])

    import tilelang.language as T  # type: ignore  # lazy

    if hasattr(T, "sync_threads_partial"):
        handle = T.sync_threads_partial(mask, n_threads)
    else:
        tir = ctx.tir()
        handle = tir.call_intrin(
            "handle",
            tir.op.Op.get("tl.sync_threads_partial"),
            mask,
            n_threads,
        )
    ctx.emit(handle)
    return handle


# ---------------------------------------------------------------------------
# TMA -- RFC section 5.1 + 5.4 (Hopper TMA strategy)
# ---------------------------------------------------------------------------


def _is_nv_target() -> bool:
    """Return True iff the current TVM target looks like NVIDIA CUDA.

    Lazy: we read ``tvm.target.Target.current()`` and fall back to False
    when no target is active (e.g. during dict-shaped unit tests).
    """
    try:
        import tilelang  # noqa: F401  (wires vendored TVM onto path)
        import tvm  # type: ignore
    except ImportError:  # pragma: no cover
        return False
    target = tvm.target.Target.current(allow_none=True)
    if target is None:
        return False
    kind = str(getattr(target, "kind", "")).lower()
    return "cuda" in kind or "nvptx" in kind


def _tensor_desc_shape_dtype(value: Any, op: Any) -> Tuple[List[int], str]:
    """Extract ``tensor<...>`` payload from a descriptor/result type.

    Real Triton TTIR prints descriptor types as
    ``!tt.tensordesc<tensor<MxNxf32>>`` while descriptor loads return a
    plain ``tensor<MxNxf32>``. Parse only explicit type spellings; guessing
    descriptor rank or dtype here would make TMA fallback unsafe.
    """
    if isinstance(value, dict) and value.get("shape"):
        return list(value.get("shape") or ()), _normalize_mlir_dtype(str(value.get("dtype", "float32")))

    candidates: List[str] = []
    typ = getattr(value, "type", None)
    if typ is not None:
        candidates.append(str(typ))
    try:
        candidates.append(str(value))
    except Exception:
        pass
    try:
        candidates.append(str(op))
    except Exception:
        pass

    for text in candidates:
        type_text = (text or "").strip()
        if not type_text:
            continue
        desc_match = _TENSOR_DESC_RE.search(type_text)
        if desc_match is not None:
            return _parse_tensor_type(desc_match.group("tensor"))
        tensor_match = _TENSOR_RE.match(type_text)
        if tensor_match is not None:
            return _parse_tensor_type(type_text)

    raise EmitError(f"tt.tensor_descriptor: cannot parse tensor descriptor/result type from {getattr(value, 'type', value)!r}")


def _tir_index(ctx: WalkerCtx, value: Any, *, dtype: str = "int32") -> Any:
    """Return ``value`` as a scalar TIR index expression."""
    tir = ctx.tir()
    if isinstance(value, bool):
        return tir.const(int(value), dtype)
    if isinstance(value, int):
        return tir.const(value, dtype)
    if hasattr(value, "dtype") and str(value.dtype) != dtype:
        return tir.Cast(dtype, value)
    return value


def _int_if_const(value: Any) -> Optional[int]:
    """Best-effort concrete integer extraction for IntImm-like values."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    raw = getattr(value, "value", None)
    if raw is not None:
        try:
            return int(raw)
        except Exception:
            return None
    try:
        return int(value)
    except Exception:
        return None


def _descriptor_flat_extent(
    logical_shape: Sequence[Any],
    strides: Sequence[Any],
    block_shape: Sequence[int],
) -> Optional[int]:
    """Return the minimum flat buffer extent for a strided descriptor."""
    shape_ints: List[int] = []
    for dim in logical_shape:
        parsed = _int_if_const(dim)
        if parsed is None:
            shape_ints = []
            break
        shape_ints.append(parsed)
    if not shape_ints:
        shape_ints = [int(dim) for dim in block_shape]

    stride_ints: List[int] = []
    for stride in strides:
        parsed = _int_if_const(stride)
        if parsed is None:
            return None
        stride_ints.append(parsed)

    if not shape_ints or not stride_ints:
        return None
    extent = 1
    for dim, stride in zip(shape_ints, stride_ints):
        extent += max(int(dim) - 1, 0) * max(int(stride), 0)
    return max(int(extent), 1)


def _descriptor_base_and_offset(ctx: WalkerCtx, base: Any) -> Tuple[Any, Any]:
    """Resolve a descriptor base pointer to ``(buffer, flat_base_offset)``."""
    tir = ctx.tir()
    if isinstance(base, tuple) and len(base) == 2:
        buf, indices = base
        if not hasattr(buf, "shape"):
            raise EmitError("tt.tensor_descriptor: tuple base is not a TIR buffer")
        if not indices:
            return buf, tir.const(0, "int32")
        if len(indices) == 1:
            return buf, _tir_index(ctx, indices[0])
        flat: Any = tir.const(0, "int32")
        for idx in indices:
            flat = flat + _tir_index(ctx, idx)
        return buf, flat

    if hasattr(base, "shape"):
        return base, tir.const(0, "int32")
    raise EmitError(f"tt.tensor_descriptor: descriptor base must resolve to a TIR buffer or (buffer, indices), got {type(base).__name__}")


def _maybe_redeclare_descriptor_buffer(
    ctx: WalkerCtx,
    buf: Any,
    dtype: str,
    min_extent: Optional[int],
) -> Any:
    """Resize placeholder pointer buffers when descriptor shape is known."""
    if min_extent is None:
        return buf
    try:
        rank = len(buf.shape)
        current_extent = int(buf.shape[0]) if rank == 1 else 0
    except Exception:
        rank = 0
        current_extent = 0
    if rank == 1 and current_extent >= int(min_extent):
        return buf

    name = getattr(buf, "name", None) or "buf"
    target_key: Any = None
    for key, value in (getattr(ctx, "buffers", {}) or {}).items():
        if value is buf:
            target_key = key
            break
    if target_key is None:
        return buf
    fixed_keys = getattr(ctx, "fixed_arg_buffer_keys", set()) or set()
    if target_key in fixed_keys or str(name) in fixed_keys:
        return buf

    new_buf = ctx.tir().decl_buffer([max(int(min_extent), 1)], dtype, name=str(name))
    ctx.buffers[target_key] = new_buf
    for key, value in list(getattr(ctx, "value_map", {}).items()):
        if value is buf:
            ctx.value_map[key] = new_buf
        elif isinstance(value, tuple) and len(value) == 2 and value[0] is buf:
            ctx.value_map[key] = (new_buf, value[1])
    return new_buf


def _descriptor_flat_index(
    ctx: WalkerCtx,
    offsets: Sequence[Any],
    strides: Sequence[Any],
    loop_vars: Sequence[Any],
    base_offset: Any,
) -> Any:
    """Build the flat source/destination index for a descriptor lane."""
    tir = ctx.tir()
    flat: Any = _tir_index(ctx, base_offset)
    for axis, lv in enumerate(loop_vars):
        offset = offsets[axis] if axis < len(offsets) else tir.const(0, "int32")
        stride = strides[axis] if axis < len(strides) else tir.const(1, "int32")
        offset = _tir_index(ctx, offset)
        stride = _tir_index(ctx, stride)
        flat = flat + (offset + lv) * stride
    return flat


def map_tt_make_tensor_descriptor(op: Any, ctx: WalkerCtx) -> Any:
    """Capture Triton tensor-descriptor metadata for later TMA fallback.

    RFC 5.4 maps tensor descriptors to native TMA on NVIDIA and to
    pointer-arithmetic tile copies elsewhere. This op itself has no side
    effect; it records the base pointer, logical shape, strides, tile shape
    and dtype so ``tt.descriptor_load/store`` can lower from live TTIR.
    """
    operands = _operands(op)
    results = _results(op)
    if len(operands) < 1 or not results:
        raise EmitError("tt.make_tensor_descriptor: expected a base pointer operand and one descriptor result")

    block_shape, dtype = _tensor_desc_shape_dtype(results[0], op)
    rank = len(block_shape)
    expected = 1 + (rank * 2)
    if len(operands) < expected:
        raise EmitError(
            f"tt.make_tensor_descriptor: expected base plus {rank} shape operands and {rank} stride operands; got {len(operands)} operands"
        )

    base = ctx.get(operands[0])
    logical_shape = [ctx.get(operand) for operand in operands[1 : 1 + rank]]
    strides = [ctx.get(operand) for operand in operands[1 + rank : expected]]
    descriptor = {
        "kind": "tensor_descriptor",
        "base": base,
        "logical_shape": logical_shape,
        "strides": strides,
        "block_shape": tuple(int(dim) for dim in block_shape),
        "dtype": dtype,
    }
    ctx.bind(results[0], descriptor)
    return descriptor


def _descriptor_state(value: Any, op_name: str) -> Dict[str, Any]:
    if not isinstance(value, dict) or value.get("kind") != "tensor_descriptor":
        raise EmitError(f"{op_name}: first operand must resolve to a tensor descriptor; got {type(value).__name__}")
    return value


def map_tt_descriptor_load(op: Any, ctx: WalkerCtx) -> Any:
    """Lower real Triton ``tt.descriptor_load`` into a tile copy fallback."""
    operands = _operands(op)
    results = _results(op)
    if len(operands) < 1 or not results:
        raise EmitError("tt.descriptor_load: expected descriptor operand and result")

    desc = _descriptor_state(ctx.get(operands[0]), "tt.descriptor_load")
    result_shape, result_dtype = _tensor_desc_shape_dtype(results[0], op)
    block_shape = tuple(int(dim) for dim in desc.get("block_shape") or result_shape)
    if tuple(result_shape) != tuple(block_shape):
        raise EmitError(
            f"tt.descriptor_load: result tensor shape does not match descriptor block shape ({tuple(result_shape)} vs {block_shape})"
        )

    rank = len(block_shape)
    if len(operands) - 1 < rank:
        raise EmitError(f"tt.descriptor_load: expected {rank} coordinate operands; got {len(operands) - 1}")

    offsets = [ctx.get(operand) for operand in operands[1 : 1 + rank]]
    strides = list(desc.get("strides") or ())
    if len(strides) < rank:
        raise EmitError(f"tt.descriptor_load: descriptor has {len(strides)} strides for rank-{rank} result")

    dtype = _normalize_mlir_dtype(str(result_dtype or desc.get("dtype", "float32")))
    base_buf, base_offset = _descriptor_base_and_offset(ctx, desc["base"])
    min_extent = _descriptor_flat_extent(desc.get("logical_shape") or (), strides, block_shape)
    base_buf = _maybe_redeclare_descriptor_buffer(ctx, base_buf, dtype, min_extent)

    tir = ctx.tir()
    tile_buf = _alloc_tile_buffer(
        ctx,
        list(block_shape) or [1],
        dtype,
        ctx.fresh("desc_load"),
        scope="shared" if rank >= 2 else "local",
    )
    loop_vars = [tir.Var(ctx.fresh(f"i{axis}"), "int32") for axis in range(rank)]
    flat = _descriptor_flat_index(ctx, offsets, strides, loop_vars, base_offset)
    body: Any = tir.BufferStore(
        tile_buf,
        tir.BufferLoad(base_buf, [flat]),
        list(loop_vars) or [tir.const(0, "int32")],
    )
    for axis in range(rank - 1, -1, -1):
        body = tir.For(
            loop_vars[axis],
            tir.const(0, "int32"),
            tir.const(int(block_shape[axis]), "int32"),
            tir.ForKind.SERIAL,
            body,
        )
    ctx.emit(body)
    ctx.bind(results[0], tile_buf)
    return tile_buf


def map_tt_descriptor_store(op: Any, ctx: WalkerCtx) -> Any:
    """Lower real Triton ``tt.descriptor_store`` into a tile store fallback."""
    operands = _operands(op)
    if len(operands) < 2:
        raise EmitError("tt.descriptor_store: expected descriptor, tile value, and offsets")

    desc = _descriptor_state(ctx.get(operands[0]), "tt.descriptor_store")
    value = ctx.get(operands[1])
    block_shape = tuple(int(dim) for dim in desc.get("block_shape") or ())
    if not block_shape:
        block_shape, _dtype = _tensor_desc_shape_dtype(operands[1], op)
        block_shape = tuple(int(dim) for dim in block_shape)
    rank = len(block_shape)
    if len(operands) - 2 < rank:
        raise EmitError(f"tt.descriptor_store: expected {rank} coordinate operands; got {len(operands) - 2}")

    offsets = [ctx.get(operand) for operand in operands[2 : 2 + rank]]
    strides = list(desc.get("strides") or ())
    if len(strides) < rank:
        raise EmitError(f"tt.descriptor_store: descriptor has {len(strides)} strides for rank-{rank} value")

    dtype = _normalize_mlir_dtype(str(desc.get("dtype", "float32")))
    base_buf, base_offset = _descriptor_base_and_offset(ctx, desc["base"])
    min_extent = _descriptor_flat_extent(desc.get("logical_shape") or (), strides, block_shape)
    base_buf = _maybe_redeclare_descriptor_buffer(ctx, base_buf, dtype, min_extent)

    tir = ctx.tir()
    loop_vars = [tir.Var(ctx.fresh(f"i{axis}"), "int32") for axis in range(rank)]
    flat = _descriptor_flat_index(ctx, offsets, strides, loop_vars, base_offset)
    if isinstance(value, LazyTileExpr):
        rhs = value.read_lane(ctx, tuple(loop_vars))
    elif hasattr(value, "shape"):
        rhs = tir.BufferLoad(value, list(loop_vars))
    else:
        rhs = value
    body: Any = tir.BufferStore(base_buf, rhs, [flat])
    for axis in range(rank - 1, -1, -1):
        body = tir.For(
            loop_vars[axis],
            tir.const(0, "int32"),
            tir.const(int(block_shape[axis]), "int32"),
            tir.ForKind.SERIAL,
            body,
        )
    ctx.emit(body)
    return body


def _emit_descriptor_copy(op: Any, ctx: WalkerCtx, *, is_load: bool) -> Any:
    """Shared body for descriptor load/store: TMA on NV, fallback elsewhere.

    The descriptor SSA is the first operand; the in-shared / out-shared
    tile is the second; remaining operands are coordinate offsets that
    PtrAnalysis has folded into a (buffer, indices) tuple.
    """
    operands = _operands(op)
    if len(operands) < 2:
        raise EmitError(f"{'descriptor_load' if is_load else 'descriptor_store'}: expected (desc, tile, ...offsets) operands")
    desc_ssa, tile_ssa = operands[0], operands[1]
    desc = ctx.get(desc_ssa)
    tile = ctx.get(tile_ssa)

    import tilelang.language as T  # type: ignore

    if _is_nv_target():
        # On NV, ``T.tma_copy`` directly accepts a descriptor and a tile.
        # FuseMBarrierArriveExpectTx fuses the surrounding mbarrier proto.
        if is_load:
            handle = T.tma_copy(desc, tile)
        else:
            handle = T.tma_copy(tile, desc)
        ctx.emit(handle)
        return handle

    # Non-NV fallback (Triton PR #6753 strategy): decompose into a regular
    # T.copy over the descriptor's base buffer + coordinate offsets that
    # PtrAnalysis has resolved.
    tir = ctx.tir()
    desc_buf, desc_idx = desc if isinstance(desc, tuple) else (desc, [0])
    src = tir.BufferLoad(desc_buf, list(desc_idx)) if hasattr(desc_buf, "shape") else desc_buf
    if is_load:
        handle = T.copy(src, tile)
    else:
        handle = T.copy(tile, src)
    ctx.emit(handle)
    return handle


def map_tt_experimental_descriptor_load(op: Any, ctx: WalkerCtx) -> Any:
    """Lower TMA descriptor load to ``T.tma_copy`` (NV) / pointer fallback.

    Recipe (RFC 5.1 + 5.4):
    * On NV target: emit ``tilelang.language.tma_copy(desc, dst, ...)``;
      ``FuseMBarrierArriveExpectTx`` later fuses the surrounding
      mbarrier protocol.
    * Off NV: per Triton PR #6753, decompose into a strided pointer-arith
      ``T.copy`` and route through the regular ``LowerTileOp`` path.
    """
    return _emit_descriptor_copy(op, ctx, is_load=True)


def map_tt_experimental_descriptor_store(op: Any, ctx: WalkerCtx) -> Any:
    """Lower TMA descriptor store to ``T.tma_copy`` (NV) / pointer fallback.

    Same recipe as the load variant but for shared->global movement.
    """
    return _emit_descriptor_copy(op, ctx, is_load=False)


# ---------------------------------------------------------------------------
# Misc -- RFC section 5.1, "tt.print"
# ---------------------------------------------------------------------------


def map_tt_print(op: Any, ctx: WalkerCtx) -> Any:
    """Lower ``tt.print`` to ``T.call_extern("printf", ...)``.

    Recipe (RFC 5.1):
    * Format string lives in the op's ``prefix`` attribute.
    * Operand SSAs become printf args after type-promotion (i32, f32, ...).
    * For the buffer-vs-scalar split we delegate to
      ``tilelang.language.print`` for buffers and a direct
      ``tir.call_extern("printf", ...)`` for scalar lists, which is the
      portable surface across CUDA / HIP / Metal codegens.
    """
    operands = _operands(op)
    attrs = _attrs_with_properties_shared(op)
    prefix = str(attrs.get("prefix", attrs.get("msg", "")))

    tir = ctx.tir()
    args = [ctx.get(o) for o in operands]

    # When a single operand is a buffer, defer to the rich TileLang
    # print() macro which handles per-thread / per-warp gating.
    import tilelang.language as T  # type: ignore  # lazy

    if len(args) == 1 and hasattr(args[0], "scope") and callable(args[0].scope):
        # Buffer-shaped arg.
        T.print(args[0], msg=prefix or "")
        return None

    # Scalar / multi-arg path: emit a direct printf call_extern. On
    # Metal, codegen will route this to os_log; on HIP/CUDA it lowers
    # to printf. The prefix may originate from user-supplied TTIR and
    # could carry malicious format specifiers (%n stack-write, etc.) —
    # sanitize before forwarding to the GPU printf runtime.
    fmt = prefix if prefix else " ".join(["%g"] * len(args)) + "\n"
    fmt = _sanitize_printf_format(fmt)
    handle = tir.call_extern("handle", "printf", fmt, *args)
    ctx.emit(handle)
    return handle


_PRINTF_FORBIDDEN_SPECS = ("%n",)


def _sanitize_printf_format(fmt: str) -> str:
    """Defang malicious printf specifiers that could corrupt GPU runtime state.

    %n writes to memory and is dangerous in any context that accepts
    untrusted format strings. We escape it as a literal "%%n" so the
    runtime sees a printable token, never a write directive. Other
    specifiers (%s/%p/%x) are left intact — they only read; an attacker
    controlling the format would still need a matching argument list to
    leak anything non-trivial, and our caller fixes the arg count.
    """
    import re

    if not fmt:
        return fmt

    # Split the format string by literal % (which is written as %%)
    # This prevents us from accidentally thinking the second % in %%n is a format start.
    parts = fmt.split("%%")
    sanitized_parts = []

    for p in parts:
        # Match % followed by optional flags, width, precision, length modifiers, and 'n'
        # e.g., %n, %lln, %10n, %-10.5lln, %*.*n
        # Replace the matched string with % + matched (e.g., %n -> %%n)
        p = re.sub(r"%(?:[-+ #0\'I]*)(?:[0-9*]*)(?:\.[0-9*]*)?(?:hh|h|ll|l|j|z|t|L)*n", lambda m: "%" + m.group(0), p)
        sanitized_parts.append(p)

    return "%%".join(sanitized_parts)


# ---------------------------------------------------------------------------
# Grid / launch -- RFC section 5.1, ``tt.get_program_id`` (a.k.a. tt.program_id)
# ---------------------------------------------------------------------------


def _program_id_axis(op: Any, attrs: Dict[str, Any]) -> int:
    """Resolve the grid axis of a ``tt.get_program_id`` op.

    Triton's MLIR spells the axis two ways:

    * an ``axis = N : i32`` integer attribute (older / generic form), or
    * a bare ``x`` / ``y`` / ``z`` keyword in the op text
      (``tt.get_program_id x``) -- the form emitted by Triton 3.x, where
      the axis is an enum operand, NOT an attribute. The legacy
      ``attrs.get("axis", 0)`` silently returned 0 for ALL three program
      ids in that case, collapsing every grid extent to ``gridDim[0]`` and
      truncating grid-scaled outputs to the first tile (RULE #1: the wrong
      default is the truncation bug). We parse the keyword explicitly.

    Raises when neither form is present (no silent axis-0 default).
    """
    if "axis" in attrs:
        try:
            return int(attrs["axis"])
        except Exception:
            pass
    try:
        text = str(op)
    except Exception:
        text = ""
    m = re.search(r"tt\.get_program_id\s+([xyz])\b", text)
    if m is None:
        m = re.search(r"tt\.(?:get_)?program_id\s*\(\s*([xyz])\s*\)", text)
    if m is not None:
        return {"x": 0, "y": 1, "z": 2}[m.group(1)]
    # Generic-form integer axis embedded in the printed op
    m = re.search(r"axis\s*=\s*(\d+)", text)
    if m is not None:
        return int(m.group(1))
    raise EmitError(
        "tt.get_program_id: cannot determine grid axis -- no `axis` "
        "attribute and no `x`/`y`/`z` keyword found in op text "
        f"{text.strip()[:120]!r}. Refusing to default to axis 0 (RULE #1: "
        "a wrong axis collapses the grid extent and truncates output)."
    )


def map_tt_program_id(op: Any, ctx: WalkerCtx) -> Any:
    """Lower ``tt.get_program_id(axis=N)`` to a block-binding ``Var``.

    Triton's ``tl.program_id(axis=N)`` selects gridDim.(x|y|z); the
    natural TileLang equivalent is ``T.Kernel(bx, by, bz)``'s block
    binding for axis ``N`` (``KernelLaunchFrame.get_block_binding``).

    Outside an active KernelLaunchFrame (e.g. unit tests with dict-shaped
    fakes) we fall back to allocating a fresh ``int32`` Var so the walker
    keeps going; downstream codegen replaces it with the real binding.
    """
    attrs = _attrs_with_properties_shared(op)
    axis = _program_id_axis(op, attrs)
    if axis < 0 or axis > 2:
        raise EmitError(f"tt.program_id: axis must be in [0, 2]; got {axis}")

    var: Any
    try:
        from tilelang.language.kernel import KernelLaunchFrame  # type: ignore

        frame = KernelLaunchFrame.Current()
    except Exception:  # pragma: no cover -- TileLang absent during tests
        frame = None

    if frame is not None:
        try:
            var = frame.get_block_binding(axis)
        except Exception:  # pragma: no cover -- frame missing axis
            var = None
    else:
        var = None

    if var is None:
        # Reuse the binding already created for this axis: a kernel that reads
        # ``program_id(axis)`` more than once (the canonical pid_m/pid_n split
        # off ``program_id(0)``) must map every read to the SAME blockIdx.<axis>
        # env-thread. Creating a fresh Var + a second launch_thread per read
        # produces a duplicate ``gridDim_<axis>`` and breaks the host grid
        # mapping so only block (0,0,0) executes. RULE #1: one axis -> one
        # binding, or the grid silently degenerates.
        cached = ctx.program_id_axis_var.get(axis)
        if cached is not None:
            if _results(op):
                ctx.bind(_results(op)[0], cached)
            return cached
        tir = ctx.tir()
        var = tir.Var(ctx.fresh(f"pid{axis}"), "int32")
        # Record so ``_make_prim_func`` can wrap the body in a
        # ``tir.AttrStmt(IterVar, "thread_extent", extent, body)`` -- without
        # this binding MakePackedAPI flags the Var as a free variable that
        # was neither a function parameter nor a thread-environment-bound
        # iter Var. Extent comes from a kernel-launch grid hint when one is
        # threaded through ``ctx`` (future work); for now we record a
        # symbolic ``tir.Var`` named ``gridDim_<axis>`` that the host
        # launcher fills in. Using a Var (rather than a numeric IntImm)
        # keeps the IR independent of a hard-coded block count.
        launch_grid = getattr(ctx, "launch_grid", None)
        if launch_grid is not None and axis < len(launch_grid):
            extent = tir.const(int(launch_grid[axis]), "int32")
        else:
            extent = tir.Var(f"gridDim_{axis}", "int32")
        ctx.program_id_vars.append((var, axis, extent))
        ctx.program_id_axis_var[axis] = var

    if _results(op):
        ctx.bind(_results(op)[0], var)
    return var


# ---------------------------------------------------------------------------
# Dispatch table (TTIR op name -> emitter)
# ---------------------------------------------------------------------------

OP_TABLE: Dict[str, EmitFn] = {
    # memory
    "tt.atomic_rmw": map_tt_atomic_rmw,
    # compute
    "tt.where": map_tt_where,
    # shape
    "tt.trans": map_tt_trans,
    # async / barrier (multiple TTIR spellings route through one emitter)
    "async_copy": map_tt_async_copy,
    "tt.async_copy_global_to_local": map_tt_async_copy,
    "tt.async_commit_group": map_tt_async_copy,
    "tt.async_wait": map_tt_async_copy,
    "mbarrier": map_tt_mbarrier,
    "tt.barrier_init": map_tt_mbarrier,
    "tt.barrier_arrive": map_tt_mbarrier,
    "tt.barrier_wait": map_tt_mbarrier,
    # partial-warp / subgroup barrier (cppmega.mlx topk_selector migration)
    "tt.sync_threads_partial": map_tt_sync_threads_partial,
    "tt.partial_barrier": map_tt_sync_threads_partial,
    "triton.language.partial_barrier": map_tt_sync_threads_partial,
    # TMA
    "tt.make_tensor_descriptor": map_tt_make_tensor_descriptor,
    "tt.descriptor_load": map_tt_descriptor_load,
    "tt.descriptor_store": map_tt_descriptor_store,
    "tt.experimental_descriptor_load": map_tt_experimental_descriptor_load,
    "tt.experimental_descriptor_store": map_tt_experimental_descriptor_store,
    # grid / launch (multiple TTIR spellings route through one emitter)
    "tt.program_id": map_tt_program_id,
    "tt.get_program_id": map_tt_program_id,
    # misc
    "tt.print": map_tt_print,
}


# ---------------------------------------------------------------------------
# arith.* / math.* / tt.fma -- sourced from op_emitters.arith to avoid the
# 1400-line table in this file growing further (and to keep merge-conflict
# surface small for parallel work on individual op families).
# ---------------------------------------------------------------------------
from .op_emitters.arith import ARITH_EMITTERS  # noqa: E402

OP_TABLE.update(ARITH_EMITTERS)


# ---------------------------------------------------------------------------
# Reductions / scan / dot / atomics -- sourced from op_emitters.reduction.
# Path C kernel surface: ``tt.reduce`` and ``tt.scan`` lower to explicit
# ``tir.For`` + accumulator (rather than the high-level ``T.reduce_*``
# macros invoked by the legacy ``map_tt_reduce`` above). The legacy
# ``map_tt_reduce`` / ``map_tt_dot`` / ``map_tt_atomic_rmw`` stubs in this
# file are deliberately left in place per ``feedback_no_silent_delete``;
# the ``OP_TABLE.update`` call below overrides them with the explicit
# emitters for the modern ``tt.atomic_<op>`` names. ``tt.atomic_rmw``
# (the legacy single-op spelling carrying an ``rmw_op`` attribute) is
# still handled by ``map_tt_atomic_rmw`` here.
# TODO: once Path C is the only path, fold the explicit emitters back
# into this file and delete the high-level stubs above.
# ---------------------------------------------------------------------------
from .op_emitters.reduction import REDUCTION_EMITTERS  # noqa: E402

OP_TABLE.update(REDUCTION_EMITTERS)


# ---------------------------------------------------------------------------
# Memory / shape -- sourced from op_emitters.memory. ``tt.load`` / ``tt.store``
# / ``tt.make_range`` / ``tt.broadcast`` / ``tt.splat`` / ``tt.expand_dims`` /
# ``tt.view`` / ``tt.reshape`` / ``tt.addptr`` / ``tts.make_tptr``. The legacy
# ``map_tt_*`` stubs above are kept per ``feedback_no_silent_delete``; the
# ``OP_TABLE.update`` below overrides them with the real TIR emitters.
# ---------------------------------------------------------------------------
from .op_emitters.memory import MEMORY_EMITTERS  # noqa: E402

OP_TABLE.update(MEMORY_EMITTERS)


# ---------------------------------------------------------------------------
# Control flow + casts -- sourced from op_emitters.control. ``arith.select`` /
# ``arith.{ext,trunc,fpto*,*tofp,bit}cast`` / ``arith.{ext,trunc}{si,ui}`` /
# ``tt.advance`` / ``scf.{for,if,yield}``.
# ---------------------------------------------------------------------------
from .op_emitters.control import CONTROL_EMITTERS  # noqa: E402

OP_TABLE.update(CONTROL_EMITTERS)
