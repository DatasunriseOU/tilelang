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

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple

from .op_mapping import TritonFrontendError

__all__ = [
    "PassStatus",
    "PassRole",
    "PassEntry",
    "PASS_ORDER",
    "PipelineError",
    "build_pipeline",
    "run",
    "run_ptr_analysis_pre_pass",
    "run_ptr_analysis_pre_pass_subprocess",
    "seed_ptr_states",
    "is_custom_form_ttir",
    "round_trip_through_cxx_shim",
]


class PipelineError(TritonFrontendError):
    """Raised when a pre-walker pipeline stage fails in a way that should NOT
    silently degrade. Carries enough context (op name, SSA, state count) to
    point a maintainer at the missing PtrState wiring without a debug build.

    Wave G4: ``PipelineError`` is now a subclass of
    :class:`poc.triton_frontend.op_mapping.TritonFrontendError` so it shares
    a common ancestor with :class:`EmitError`. This lets a single
    ``except TritonFrontendError`` clause in driver code catch any
    deliberate frontend failure (emitter-side or pipeline-side) without
    swallowing unrelated ``RuntimeError``s from TVM/TileLang internals.
    """


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
    from tvm.ir import IRModule  # noqa: WPS433

    seq = build_pipeline(target, **kwargs)
    if isinstance(prim_func, IRModule):
        mod = prim_func
    else:
        mod = IRModule({"main": prim_func})
    return seq(mod)


# ---------------------------------------------------------------------------
# Pre-walker PtrAnalysis pre-pass
#
# The TileLang ``Sequential`` above operates on ``IRModule`` objects after
# the TTIR->TIR walker has already produced a PrimFunc. The C++ shim's
# PtrAnalysis runs *before* that, on raw TTIR text, to rewrite multi-element
# pointer arithmetic into ``tts.make_tptr`` ops and surface ``PtrState``
# descriptors. The two helpers below are the integration glue between the
# shim (see :mod:`poc.triton_frontend.ptr_analysis`) and the walker's
# :class:`WalkerCtx`.
# ---------------------------------------------------------------------------


def _ptr_states_to_map(states: List[Any]) -> dict:
    state_map: dict = {}
    for s in states:
        if s.result_ssa is not None:
            state_map[s.result_ssa] = s
    return state_map


def run_ptr_analysis_pre_pass(
    ttir_text: str,
    *,
    shim_available_fn: Callable[[], bool] | None = None,
    run_with_states_fn: Callable[[str], Tuple[str, List[Any]]] | None = None,
) -> Tuple[str, dict]:
    """Run PtrAnalysis on TTIR text and return ``(rewritten_ttir, state_map)``.

    ``state_map`` is keyed by ``result_ssa`` (e.g. ``"%2"``) for fast lookup
    inside emitters. When the C++ shim is unavailable this returns
    ``(ttir_text, {})`` unchanged -- callers are expected to fall back to
    the MVP scalar path with a visible ``# DEGRADED:`` AttrStmt.

    Hard-constraint: when the shim *is* available but raises while
    extracting states, surface the exception as a :class:`PipelineError`
    with diagnostics. We never silently degrade in that case.
    """
    from .ptr_analysis import shim_available, run_ptr_analysis_with_states_generic

    is_available = shim_available_fn or shim_available
    run_with_states = run_with_states_fn or run_ptr_analysis_with_states_generic

    if not is_available():
        return ttir_text, {}

    try:
        rewritten, states = run_with_states(ttir_text)
    except BaseException as exc:  # noqa: BLE001
        raise PipelineError(
            f"PtrAnalysis pre-pass failed: {type(exc).__name__}: {exc}. "
            "Build invariant: when the C++ shim is loaded we must not "
            "silently degrade -- check the TTIR input for a parser error "
            "or rebuild poc/triton_frontend/_cxx."
        ) from exc

    return rewritten, _ptr_states_to_map(states)


def run_ptr_analysis_pre_pass_subprocess(ttir_text: str) -> Tuple[str, dict]:
    """Run PtrAnalysis in an isolated Python process.

    Triton's native ``libtriton`` and the local PtrAnalysis shim both touch
    LLVM's process-global option registry. If Triton already loaded
    ``libtriton`` in this interpreter, importing the shim in-process can
    abort before Python sees an exception. The subprocess keeps that native
    state isolated while preserving PtrAnalysis metadata for the walker.
    """
    from .ptr_analysis import PtrState

    payload = json.dumps({"sys_path": sys.path, "ttir": ttir_text})
    code = r"""
import json
import sys

payload = json.loads(sys.stdin.read())
for path in reversed(payload.get("sys_path") or []):
    if path and path not in sys.path:
        sys.path.insert(0, path)

from poc.triton_frontend.ptr_analysis import run_ptr_analysis_with_states_generic

rewritten, states = run_ptr_analysis_with_states_generic(payload["ttir"])
sys.stdout.write(json.dumps({
    "rewritten": rewritten,
    "states": [state.__dict__ for state in states],
}))
"""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            input=payload,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            env=os.environ.copy(),
        )
    except Exception as exc:
        raise PipelineError(
            f"PtrAnalysis subprocess failed: {type(exc).__name__}: {exc}"
        ) from exc
    if proc.returncode != 0:
        raise PipelineError(
            "PtrAnalysis subprocess failed with exit "
            f"{proc.returncode}: {proc.stderr.strip()}"
        )
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise PipelineError(
            f"PtrAnalysis subprocess returned invalid JSON: {proc.stdout[:500]!r}"
        ) from exc

    states = []
    for item in data.get("states") or []:
        if not isinstance(item, dict):
            continue
        states.append(
            PtrState(
                offsets=tuple(item.get("offsets") or ()),
                sizes=tuple(item.get("sizes") or ()),
                strides=tuple(item.get("strides") or ()),
                source=item.get("source"),
                modulos=tuple(item.get("modulos") or ()),
                shape=(
                    tuple(item.get("shape"))
                    if item.get("shape") is not None else None
                ),
                op=item.get("op"),
                result_ssa=item.get("result_ssa"),
            )
        )
    return str(data.get("rewritten") or ttir_text), _ptr_states_to_map(states)


def seed_ptr_states(ctx: Any, state_map: dict) -> int:
    """Seed ``ctx.ptr_states`` (and ``ctx.value_map``) with PtrState entries.

    Populates two result-SSA keyed surfaces so emitters can find state:

    1. ``ctx.ptr_states[ssa_name] = PtrState`` -- the new authoritative
       lookup table; emitters in ``op_emitters/memory.py`` consult it as
       ``ctx.ptr_states.get(_op_result_name(op))``.
    2. ``ctx.value_map[ssa_name] = {"_ptrstate": ..., ...}`` -- the legacy
       tagged-dict shape that ``_emit_load_copy`` / ``_emit_store_copy``
       in op_mapping.py already understand. Keying by name (string) means
       a name-based look-up resolves; the existing code already special-
       cases this dict shape via ``_ptrstate_is_tile`` etc.

    Returns the number of states seeded.
    """
    seeded = 0
    if not hasattr(ctx, "ptr_states"):
        ctx.ptr_states = {}
    for ssa_name, state in state_map.items():
        ctx.ptr_states[ssa_name] = state
        ctx.value_map[ssa_name] = {
            "_ptrstate": state,
            "source": state.source,
            "offsets": list(state.offsets),
            "sizes": list(state.sizes),
            "strides": list(state.strides),
            "shape": list(state.shape) if state.shape is not None else None,
        }
        seeded += 1
    return seeded


# ---------------------------------------------------------------------------
# Custom-form -> generic-form round-trip via the C++ shim
#
# Triton's printer emits TTIR in *custom* form -- e.g. ``tt.func @kernel``,
# ``arith.constant 0 : i32``. jaxlib's bundled mlir.ir bindings can only
# parse the *generic* op form (``"dialect.op"(...) : (...) -> (...)``)
# unless the relevant dialect TableGen has been registered, which
# jaxlib does not do for ``tt.*``. The C++ shim links against Triton's
# real dialect registration and offers ``Module.to_generic()`` to
# round-trip the text. This module hosts the helper so non-harness
# entry points (e.g. the future ``triton_to_tilelang_prim`` API) get
# the same fix without re-implementing it.
# ---------------------------------------------------------------------------


# Heuristic: any of these tokens appearing in TTIR text means the printer
# used custom form. We deliberately use plain string membership rather
# than a regex over the full body -- false positives are harmless because
# the C++ shim's ``to_generic()`` is idempotent on already-generic IR.
# NB: every hint must be a substring that does NOT appear in
# *generic* form. Generic always quotes op names (``"tt.foo"(...)``)
# so a hint like ``tt.func @`` (with the ``@`` symbol marker) is
# unambiguous; bare ``tt.return`` would match ``"tt.return"`` and
# misclassify generic form as custom. We use ``\n`` boundaries +
# ``@`` / ``=`` markers to keep the heuristic specific.
_CUSTOM_FORM_HINTS: Tuple[str, ...] = (
    "tt.func @",          # custom-form func declaration
    "tt.func public @",   # variant with visibility keyword
    "= tt.load ",         # custom-form load result-binding form
    "= tt.make_range ",   # custom-form make_range
    "= tt.splat ",        # custom-form splat
    "= arith.constant ",  # custom-form constant
)


def is_custom_form_ttir(ttir_text: str) -> bool:
    """Heuristic: does ``ttir_text`` look like custom-form MLIR?

    A *generic* MLIR module quotes every op name (``"tt.func"(...)``);
    *custom* form lets the dialect printer emit a more readable surface
    (``tt.func @kernel(...)``). jaxlib's mlir.ir parser accepts the
    former (with ``allow_unregistered_dialects=True``) but not the
    latter unless the ``tt`` dialect is registered, which we cannot
    expect on a host with only jaxlib bindings.

    We check for a small list of hint substrings. The hints are chosen
    to be unambiguous custom-form markers -- ``tt.func @`` would never
    appear in generic form because the op name would be quoted.
    """
    if not isinstance(ttir_text, str):
        return False
    return any(hint in ttir_text for hint in _CUSTOM_FORM_HINTS)


def _libtriton_loaded() -> bool:
    """Whether Triton's native libtriton module is already live."""
    return any(name.startswith("triton._C.libtriton") for name in sys.modules)


def _round_trip_through_cxx_shim_subprocess(ttir_text: str) -> Optional[str]:
    """Run the C++ shim in a clean Python process.

    Triton's libtriton and the local C++ shim both touch LLVM's global
    option registry. Loading both into one process can abort the
    interpreter with a duplicate option registration before Python can
    catch anything. The subprocess keeps that native state isolated.
    """
    payload = json.dumps({"sys_path": sys.path, "ttir": ttir_text})
    code = r"""
import json
import sys

payload = json.loads(sys.stdin.read())
for path in reversed(payload.get("sys_path") or []):
    if path and path not in sys.path:
        sys.path.insert(0, path)

from poc.triton_frontend.ptr_analysis import shim_available

shim_available()
import _triton_frontend_cxx as _cxx

ctx = _cxx.Context()
mod = _cxx.Module(ctx, payload["ttir"])
sys.stdout.write(mod.to_generic())
"""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            input=payload,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            env=os.environ.copy(),
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout


def round_trip_through_cxx_shim(ttir_text: str) -> str:
    """Re-print custom-form TTIR as generic form via the C++ shim.

    Calls ``_triton_frontend_cxx.Module(ctx, ttir_text).to_generic()``
    and returns the generic-form text. If the shim is unavailable or
    the round-trip raises, returns the input unchanged so the caller
    can attempt a direct parse (works on hosts with the full ``tt``
    dialect registered, e.g. brew llvm + Triton).

    Use the heuristic :func:`is_custom_form_ttir` to decide whether to
    invoke this helper.
    """
    if _libtriton_loaded():
        converted = _round_trip_through_cxx_shim_subprocess(ttir_text)
        if converted:
            return converted
        return ttir_text

    try:
        from .ptr_analysis import shim_available

        shim_available()
        import _triton_frontend_cxx as _cxx  # type: ignore  # noqa: WPS433
    except Exception:
        return ttir_text

    try:
        cxx_ctx = _cxx.Context()
        cxx_mod = _cxx.Module(cxx_ctx, ttir_text)
        return cxx_mod.to_generic()
    except Exception:
        return ttir_text
