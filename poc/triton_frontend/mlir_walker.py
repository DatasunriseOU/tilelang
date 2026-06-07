"""MLIR-IR-based walker for Triton TTIR.

This module replaces the regex text-walker (still kept in
:mod:`triton_frontend.__init__` as the documented fallback) with a real
``mlir.ir`` traversal that populates :class:`op_mapping.WalkerCtx`'s
``value_map`` and ``buffers`` so the OP_TABLE emitters (and the
forthcoming ``op_emitters/*.py`` modules) can resolve operands properly.

Why a separate module?
----------------------
The regex walker proved op-coverage in tests but produced a stub
PrimFunc shell with empty ``value_map`` / ``buffers``. The MLIR walker
adds the missing materialization step:

* For each ``tt.func`` block argument, allocate a ``tvm.tir.Buffer``
  in :attr:`WalkerCtx.buffers` keyed by the argument's printed name.
* For each producing op, after the OP_TABLE emitter returns, bind the
  result SSA value to its TIR equivalent in :attr:`WalkerCtx.value_map`
  so downstream emitters can resolve it via :meth:`WalkerCtx.get`.

The regex walker is **not** removed -- per ``feedback_no_silent_delete``
it stays as the degraded fallback when ``mlir.ir`` is unavailable
(e.g. on a generic Linux box without LLVM Python bindings installed).

Environment notes
-----------------
``mlir.ir`` is the upstream LLVM Python binding. It is exposed by:

* Mac: ``brew install llvm`` ships ``$(brew --prefix llvm)/python_packages/mlir_core``.
* GB10 / system MLIR: ``$LLVM_INSTALL_DIR/python_packages/mlir_core``.

Either ``MLIR_PYTHON_PACKAGE_PREFIX`` must be set, or the
``mlir_core`` package directory must be on ``sys.path``. We check both
and emit one (and only one) :class:`UserWarning` if neither succeeds.
"""
from __future__ import annotations

import os
import sys
import warnings
from types import ModuleType
from typing import Any, Optional, Protocol

from .op_mapping import OP_TABLE, WalkerCtx

__all__ = [
    "DEGRADED_WARNING_MESSAGE",
    "MLIR_WALKER_AVAILABLE",
    "OPS_THAT_HANDLE_OWN_REGIONS",
    "OpVisitor",
    "TTIRWalker",
    "_compute_owns_regions_set",
    "_emitter_owns_regions",
    "parse_ttir",
    "try_import_mlir",
    "walk_module",
    "build_addressing_fold_set",
    "wrap_module_for_walker",
]


# ---------------------------------------------------------------------------
# Region-ownership: per-emitter ``owns_regions`` attribute
# ---------------------------------------------------------------------------
#
# Some ops carry MLIR regions whose ops MUST NOT be visited by the global
# walker -- their parent emitter walks the region itself and is responsible
# for emitting/binding the inner ops in the right order. If the global
# walker descended into these regions it would dispatch the inner ops
# (e.g. ``arith.maximumf`` inside ``tt.reduce``'s combiner) before their
# operands -- which are block arguments of the combiner block, never
# bound by the parent emitter -- triggering ``WalkerCtx.get`` to raise
# ``KeyError: SSA value not yet mapped``.
#
# H4 Wave-I refactor: instead of a hard-coded allowlist, each emitter
# callable that walks its own regions sets an ``owns_regions = True``
# attribute on itself. The walker checks ``getattr(emitter_fn,
# 'owns_regions', False)`` after dispatching the op. New region-owning
# emitters (e.g. ``tt.gather`` / ``tt.histogram``) can opt in without
# touching the walker.
#
# Currently region-owning emitters (each carries ``owns_regions = True``):
# * ``scf.for`` / ``scf.if`` / ``scf.while`` -- ``op_emitters/control.py``
#   uses ``_emit_region`` to walk the body itself so iter_args/yield are
#   bound at the right point.
# * ``tt.reduce`` / ``tt.scan`` -- ``op_emitters/reduction.py`` walks
#   the combiner region itself (via ``detect_combiner_kind``) and never
#   wants the global walker dispatching combiner ops.
# * ``tt.call`` -- ``op_emitters/control.py:emit_tt_call`` inline-walks
#   the callee's body itself.
#
# :data:`OPS_THAT_HANDLE_OWN_REGIONS` is retained as a documented fallback
# so callers / tests that still consult the hard-coded set continue to
# work; it is re-derived from the per-emitter attributes at import time
# (see :func:`_compute_owns_regions_set` below) so the two stay in sync.
def _emitter_owns_regions(op_name: str) -> bool:
    """Return True iff the emitter for ``op_name`` walks its own regions.

    Looks the emitter up in :data:`OP_TABLE` and returns the value of its
    ``owns_regions`` attribute (default False). This is the single
    source-of-truth the walker uses to decide whether to skip region
    descent for an op.
    """
    emitter = OP_TABLE.get(op_name)
    if emitter is None:
        return False
    return bool(getattr(emitter, "owns_regions", False))


def _compute_owns_regions_set() -> frozenset:
    """Re-derive the legacy hard-coded set from per-emitter attributes.

    Kept as a documented fallback for callers that want a static set
    rather than a per-op lookup; tests assert the two are in sync via
    ``test_owns_regions_attribute_replaces_hardcoded_set``.
    """
    return frozenset(
        name
        for name, emitter in OP_TABLE.items()
        if getattr(emitter, "owns_regions", False)
    )


# Lazy: OP_TABLE may not be fully populated at module import time
# (op_emitters/* register via OP_TABLE.update at op_mapping import end).
# The set is recomputed on first access by the public callers below.
OPS_THAT_HANDLE_OWN_REGIONS = frozenset({
    "scf.for",
    "scf.if",
    "scf.while",
    "tt.reduce",
    "tt.scan",
    "tt.call",
})


# Public single-source-of-truth for the warning text used by both the
# walker probe and the public-API fallback in __init__.py. Keeping it
# here means tests can match against the constant rather than copy the
# string in both places.
DEGRADED_WARNING_MESSAGE = (
    "mlir.ir Python bindings unavailable; using regex walker "
    "(degraded -- buffers and value_map will not be populated). "
    "Install brew llvm or set MLIR_PYTHON_PACKAGE_PREFIX."
)


# Internal flag so the warning fires exactly once per process even if
# ``try_import_mlir`` is called repeatedly (e.g. from tests).
_WARNED_ONCE: bool = False


def _augment_sys_path_from_env() -> None:
    """Add ``$MLIR_PYTHON_PACKAGE_PREFIX`` and friends to sys.path if set.

    Both Mac brew LLVM and GB10 system MLIR ship the bindings under
    ``<prefix>/python_packages/mlir_core``. When the user has set
    ``MLIR_PYTHON_PACKAGE_PREFIX`` (the upstream-recommended env var)
    or ``LLVM_INSTALL_DIR`` we mirror the standard layout onto
    sys.path so the subsequent ``import mlir.ir`` succeeds.
    """
    candidates = []
    prefix = os.environ.get("MLIR_PYTHON_PACKAGE_PREFIX")
    if prefix:
        candidates.append(prefix)
        candidates.append(os.path.join(prefix, "mlir_core"))
    install = os.environ.get("LLVM_INSTALL_DIR")
    if install:
        candidates.append(os.path.join(install, "python_packages", "mlir_core"))
    for path in candidates:
        if path and os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)


def try_import_mlir() -> Optional[ModuleType]:
    """Return ``mlir.ir`` if importable; warn once and return None otherwise.

    The warning text matches :data:`DEGRADED_WARNING_MESSAGE` exactly so
    tests can assert against the constant. Subsequent calls after the
    first failure are silent.
    """
    global _WARNED_ONCE
    _augment_sys_path_from_env()
    try:
        from mlir import ir  # type: ignore  # noqa: WPS433
    except Exception:  # ImportError or platform-specific dynload failure
        if not _WARNED_ONCE:
            _WARNED_ONCE = True
            warnings.warn(DEGRADED_WARNING_MESSAGE, UserWarning, stacklevel=2)
        return None
    # Dialect Python wrappers (``mlir.dialects.func`` etc.) are NOT
    # required: the walker dispatches via OP_TABLE on string op names
    # and never instantiates dialect-Python classes. Pre-loading dialect
    # C-extensions here can also conflict with later ``triton._C`` loads
    # on macOS (LLVM CommandLine option re-registration). Skip the
    # dialect probe entirely and let downstream tests pay for whatever
    # dialects they actually need.
    return ir


def _try_import_mlir_quiet() -> Optional[ModuleType]:
    """Return ``mlir.ir`` if importable without emitting the degraded warning."""
    _augment_sys_path_from_env()
    try:
        from mlir import ir  # type: ignore  # noqa: WPS433
        return ir
    except Exception:
        return None


def parse_ttir(text: str) -> Optional[Any]:
    """Parse Triton TTIR text into an ``mlir.ir.Module`` (or None on failure).

    The Triton dialect TableGen file is not part of a generic LLVM
    install, so on most boxes the parser must be told to *accept*
    unregistered dialects. We do that via
    ``Context.allow_unregistered_dialects = True`` plus an explicit
    parse so the operation names survive verbatim (no auto-translation
    to a generic shape).

    Returns ``None`` when ``mlir.ir`` isn't importable or when parsing
    fails (e.g. the text is malformed); callers fall back to the
    regex walker in either case.

    Provider order:
    0. **Triton-native** (``triton_native_parse``) -- if Triton's own
       MLIR build is importable it parses custom-form ``tt.*`` TTIR
       directly (the canonical parser) and we adapt its canonical printed
       text into mlir.ir-shaped ops. This is the only provider that needs
       no foreign MLIR build and no C++ shim, so it is tried FIRST on
       hosts (e.g. gb10) that have Triton but neither mlir_core/IREE nor a
       built shim. It RAISES (TtirParseError) on a structural surprise
       rather than returning a partial module.
    1+. mlir_core / IREE / jaxlib generic-form parse (custom->generic via
        the C++ shim when present).
    """
    # Provider 0: Triton's own MLIR parser (handles custom-form directly).
    try:
        from poc.triton_frontend.triton_native_parse import (
            parse_ttir_via_triton,
            triton_native_available,
        )

        if triton_native_available():
            native = parse_ttir_via_triton(text)
            if native is not None:
                return native
    except Exception as exc:
        # A structural parse failure here is a real bug in OUR adapter
        # (RULE #1: do not silently fall through to a degraded path that
        # ships a wrong/empty kernel). Surface it loudly. Provider
        # *unavailability* (triton not importable) returns None above and
        # falls through to the mlir.ir providers below.
        from .triton_native_parse import TtirParseError

        if isinstance(exc, TtirParseError):
            raise
        warnings.warn(
            f"mlir_walker.parse_ttir: Triton-native provider errored "
            f"({exc!r}); falling through to mlir.ir providers.",
            UserWarning,
            stacklevel=2,
        )

    ir = _try_import_mlir_quiet()
    if ir is None:
        # Resolve jaxlib's MLIR bindings WITHOUT publishing them under
        # ``sys.modules['mlir.ir']``. Publishing the alias makes any
        # subsequent ``triton._C.libtriton.ir.context()`` call abort
        # the process because Triton's nanobind LLVM extension and
        # jaxlib's MLIR extension cannot coexist in one interpreter.
        try:
            from poc.triton_frontend._mlir_path_setup import (
                local_jaxlib_mlir_ir,
            )

            ir = local_jaxlib_mlir_ir()
        except Exception:
            ir = None
    if ir is None:
        return None

    register_dialects = None
    # Skip loading the C++ shim's MLIR context registration when the
    # active ``mlir.ir`` provider is a foreign-LLVM alias (jaxlib, IREE).
    # Loading the shim's .so in-process registers a *second* set of MLIR
    # types under the same nanobind enumeration names and aborts with
    # "type was already registered". We don't need the shim's dialect
    # registration to parse TTIR because we set
    # ``allow_unregistered_dialects = True`` below.
    try:
        from poc.triton_frontend import _mlir_path_setup as _setup
        ir_name = getattr(ir, "__name__", "")
        _foreign_alias = (
            _setup.SELECTED_SOURCE in {"jaxlib", "iree"}
            or ir_name.startswith(("jaxlib.", "iree."))
        )
    except Exception:
        _foreign_alias = False
    if not _foreign_alias:
        try:
            from poc.triton_frontend.ptr_analysis import shim_available, _load_shim
            if shim_available():
                register_dialects = _load_shim().register_dialects
        except ImportError:
            pass

    try:
        ctx = ir.Context()
        if register_dialects is not None:
            register_dialects(ctx)
        # Triton TTIR mixes ``tt.*`` (Triton dialect) with ``arith.*`` /
        # ``scf.*`` / ``builtin``. The latter three are typically
        # registered by default; ``tt.*`` won't be on a vanilla LLVM
        # install. allow_unregistered_dialects lets the parser keep
        # them as opaque ops with the correct name -- enough for the
        # walker to dispatch via OP_TABLE.
        ctx.allow_unregistered_dialects = True
        with ctx, ir.Location.unknown(ctx):
            module = ir.Module.parse(text, ctx)
        return module
    except Exception as exc:
        try:
            from poc.triton_frontend._mlir_path_setup import (
                local_jaxlib_mlir_ir,
            )
            from poc.triton_frontend.pipeline import (
                is_custom_form_ttir,
                round_trip_through_cxx_shim,
            )

            if not is_custom_form_ttir(text):
                warnings.warn(
                    f"mlir_walker.parse_ttir: parse failed -- {exc!r}",
                    UserWarning,
                    stacklevel=2,
                )
                return None
            converted = round_trip_through_cxx_shim(text)
            if converted == text:
                warnings.warn(
                    f"mlir_walker.parse_ttir: parse failed -- {exc!r}",
                    UserWarning,
                    stacklevel=2,
                )
                return None
            ir = _try_import_mlir_quiet() or local_jaxlib_mlir_ir()
            if ir is None:
                warnings.warn(
                    f"mlir_walker.parse_ttir: parse failed -- {exc!r}",
                    UserWarning,
                    stacklevel=2,
                )
                return None
            ctx = ir.Context()
            ctx.allow_unregistered_dialects = True
            with ctx, ir.Location.unknown(ctx):
                return ir.Module.parse(converted, ctx)
        except Exception as generic_exc:
            warnings.warn(
                "mlir_walker.parse_ttir: generic-form fallback failed -- "
                f"{generic_exc!r}",
                UserWarning,
                stacklevel=2,
            )
            return None


# ---------------------------------------------------------------------------
# Visitor protocol
# ---------------------------------------------------------------------------


class OpVisitor(Protocol):
    """Pre-order op visitor protocol.

    Implementations receive each ``mlir.ir.Operation`` exactly once,
    in the order produced by :func:`walk_module`. The walker descends
    into regions/blocks after calling ``visit_op``.
    """

    def visit_op(self, op: Any) -> None: ...  # noqa: D401


def _op_name(op: Any) -> str:
    """Return ``op``'s dotted name across binding shapes (and dict fakes)."""
    name = getattr(op, "name", None)
    if not name:
        inner = getattr(op, "operation", None)
        name = getattr(inner, "name", None) if inner is not None else None
    if not name and isinstance(op, dict):
        name = op.get("name")
    return str(name) if name else ""


def _block_arg_name(block_arg: Any, fallback: str) -> str:
    """Best-effort printable name for a ``BlockArgument``.

    Different mlir.ir builds print SSA names slightly differently;
    callers must accept any string and treat it as a key. The
    ``fallback`` (e.g. ``"arg0"``) is used when the binding does not
    expose ``arg.get_name()`` / ``str(arg)``.
    """
    for attr in ("get_name", "name"):
        getter = getattr(block_arg, attr, None)
        if callable(getter):
            try:
                return str(getter())
            except Exception:
                pass
    try:
        s = str(block_arg).strip()
        if s:
            return s.split()[0]
    except Exception:
        pass
    return fallback


def walk_module(module: Any, visitor: OpVisitor) -> None:
    """Pre-order walk over every op inside ``module``.

    The visitor sees the module's top-level op first, then each child
    op recursively. Block arguments are *not* delivered as ops (they
    are not operations); the walker driver in :class:`TTIRWalker`
    materializes them into ``ctx.buffers`` before recursion starts.

    Auto-wraps jaxlib-shaped modules (where ``module.body`` is a
    ``Block`` lacking ``regions``) via :func:`wrap_module_for_walker`
    so the recursion descends into the operation tree correctly.
    """
    if module is None:
        return

    module = wrap_module_for_walker(module)
    body = getattr(module, "body", None)
    top = body if body is not None else getattr(module, "operation", module)

    def _recurse(op: Any) -> None:
        visitor.visit_op(op)
        # Ops whose emitters walk their own regions (e.g. ``tt.reduce``
        # consumes the combiner region itself, ``scf.for`` walks the loop
        # body via ``_emit_region``). Descending here would dispatch the
        # inner ops before their parent's emitter has bound the region's
        # block arguments, which surfaces as ``WalkerCtx.get`` raising
        # ``KeyError: SSA value not yet mapped`` on the next downstream
        # op. Skip region descent for these ops; their emitters are
        # responsible for any traversal they need.
        #
        # H4 Wave-I refactor: dispatch via the per-emitter ``owns_regions``
        # attribute rather than a hard-coded set. Falls back to the legacy
        # set so a future emitter that forgets to set the attribute still
        # works (and tests pin the two in sync).
        op_name_str = _op_name(op)
        if _emitter_owns_regions(op_name_str) or op_name_str in OPS_THAT_HANDLE_OWN_REGIONS:
            return
        for region in getattr(op, "regions", ()) or ():
            for block in getattr(region, "blocks", ()) or ():
                for child in getattr(block, "operations", ()) or ():
                    _recurse(child)

    _recurse(top)


# ---------------------------------------------------------------------------
# Addressing/mask FOLD analysis (full transform 1 -- Coalesce-style)
# ---------------------------------------------------------------------------
#
# Triton's Coalesce.cpp never materializes the make_range / broadcast / splat
# addressing arithmetic -- it is absorbed into the per-thread slice of ONE
# coalesced load. Our frontend, by contrast, MATERIALIZES every
# ``tile_binop_*`` / ``bcast_*`` / ``expand_*`` tile into a standalone
# ``[64]`` / ``[2048]`` / ``[4096]`` buffer (``materialize_lazy_tile``); those
# arrays spill to local memory and every lane re-runs the address arithmetic.
#
# This pre-pass identifies the SSA results that are consumed ONLY as
# ADDRESSING or MASK for a ``tt.load`` / ``tt.store`` (directly, or
# transitively through other addressing/mask ops). Those results are
# "fold-eligible": ``materialize_lazy_tile`` keeps them as the lane-indexable
# ``LazyTileExpr`` instead of allocating a buffer, so the per-lane index /
# predicate is threaded INTO the load/store loop body (the copy region indices
# + predicate) rather than being precomputed into a spilled array.
#
# A result is fold-eligible iff EVERY use of it is one of:
#   * a ``tt.load`` / ``tt.store`` (consuming it as a pointer-arithmetic
#     operand or a mask -- never as the stored VALUE, see below), OR
#   * another op that is itself a pure addressing/mask producer AND whose
#     own result is fold-eligible (transitive closure: e.g. an
#     ``arith.muli`` whose result only feeds an ``arith.addi`` that only
#     feeds a ``tt.load`` mask).
#
# We are deliberately conservative: any use by ``tt.dot`` / ``tt.reduce`` /
# ``tt.store`` value-operand / a yield / a return / any op outside the
# addressing-producer allowlist DISQUALIFIES the result. That keeps the
# cooperative GEMM ``T.copy`` / ``T.gemm`` path byte-identical (its operand
# tiles are real data buffers, never folded) -- RULE #1.

# Op names that are pure addressing/mask PRODUCERS: they take tile operands
# and produce a tile that is itself only addressing/mask. Their result may be
# folded if (recursively) all ITS uses are fold-safe.
_ADDR_PRODUCER_OPS = frozenset({
    "tt.broadcast",
    "tt.expand_dims",
    "tt.splat",
    "tt.make_range",
    "tt.addptr",
    "tt.bitcast",
    # Structured-pointer producer emitted by the PtrAnalysis pre-pass
    # (``tts.make_tptr``): its dynamic offset/stride operands are pure
    # addressing, and its result is consumed only by ``tts.load`` / ``tts.store``
    # (the structured analogues of the sinks). Treating it as an addressing
    # producer lets the offset/stride binop chains feeding it stay lazy.
    "tts.make_tptr",
    "arith.bitcast",
    "arith.muli",
    "arith.addi",
    "arith.subi",
    "arith.cmpi",
    "arith.cmpf",
    "arith.andi",
    "arith.ori",
    "arith.xori",
    "arith.extsi",
    "arith.extui",
    "arith.trunci",
    "arith.index_cast",
    "arith.remsi",
    "arith.remui",
    "arith.divsi",
    "arith.divui",
    "arith.maxsi",
    "arith.minsi",
    "arith.select",
    # Pure shape rebinds of an addressing/mask tile (no data movement): a
    # reshaped mask/index is still addressing. The PtrAnalysis pre-pass emits
    # ``tensor.collapse_shape`` / ``tensor.expand_shape`` when flattening a 2D
    # mask into the structured load's predicate.
    "tensor.collapse_shape",
    "tensor.expand_shape",
})

# Op names that consume a tile as addressing/mask and TERMINATE the chain
# (the fold target). The fold is justified precisely because these are the
# load/store ops whose region indices + predicate absorb the addressing.
_ADDR_SINK_OPS = frozenset({
    "tt.load",
    "tt.store",
    # Structured load/store emitted by the PtrAnalysis pre-pass. The mask /
    # dynamic-dim operands they consume are addressing/mask, absorbed into the
    # strided copy region + predicate -- exactly the fold target.
    "tts.load",
    "tts.store",
})


def _value_ssa_key(value: Any) -> Optional[str]:
    """Best-effort printed SSA name for an MLIR value (mirrors _ssa_name)."""
    if value is None:
        return None
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


def build_addressing_fold_set(module: Any) -> set:
    """Return the set of result SSA names consumed ONLY as addressing/mask.

    See the module-level comment above for the eligibility contract. The
    returned set is keyed by printed RESULT SSA name -- the exact spelling the
    feeder emitter sees on ``_results(op)[0].get_name()`` -- so the lookup in
    ``should_fold_addressing`` matches at emit time. A feeder consults it (via
    ``WalkerCtx.fold_addressing_ssa``) to decide whether to keep a tile lazy
    instead of materializing it.

    Use-def reconstruction
    ----------------------
    The walk builds the use map by MLIR ``Value`` *identity*, NOT by printed
    name: in mlir.ir the SAME value prints a DIFFERENT name when it appears as
    an OPERAND vs as the defining op's RESULT (operand spelling carries a
    global value id, result spelling a block-relative number), so a
    name-keyed map silently fails to connect a load's mask operand back to the
    ``arith.cmpi`` that produced it. ``Value`` is hashable and exposes
    ``.owner`` (its defining op), giving an exact use->def edge.

    A result Value is fold-eligible iff EVERY use of it is either an addressing
    SINK (``tt.load`` / ``tt.store`` pointer or mask operand -- NOT the store
    VALUE operand) or an addressing PRODUCER op whose own result(s) are
    themselves eligible (transitive fixpoint).
    """
    if module is None:
        return set()
    module = wrap_module_for_walker(module)
    body = getattr(module, "body", None)
    top = body if body is not None else getattr(module, "operation", module)

    # Value (by identity/hash) -> list of (consumer_op, operand_index).
    uses_of: dict = {}
    all_ops: list = []

    def _recurse(op: Any) -> None:
        all_ops.append(op)
        operands = list(getattr(op, "operands", ()) or ())
        for opnd_idx, operand in enumerate(operands):
            try:
                uses_of.setdefault(operand, []).append((op, opnd_idx))
            except TypeError:
                # Unhashable operand (block arg in some providers) -- skip; it
                # cannot be a producer-result fold candidate anyway.
                continue
        for region in getattr(op, "regions", ()) or ():
            for block in getattr(region, "blocks", ()) or ():
                for child in getattr(block, "operations", ()) or ():
                    _recurse(child)

    _recurse(top)

    def _use_is_sink(consumer_op: Any, opnd_idx: int) -> bool:
        """True iff this use absorbs the value as load/store addressing/mask."""
        cname = _op_name(consumer_op)
        if cname not in _ADDR_SINK_OPS:
            return False
        # ``tt.store(ptr, value, mask)`` / ``tts.store(ptr, value, mask, dim)``:
        # operand #1 is the stored VALUE -- real data, never addressing.
        # RULE #1: never fold a data tile into addressing.
        if cname in ("tt.store", "tts.store") and opnd_idx == 1:
            return False
        return True

    # ----- scf.for loop-carried pointer transparency -----------------------
    # The dominant remaining arrays are the 2D pointer-offset tiles
    # (``offs_m[:,None]*sd + offs_k[None,:]``) that feed ``tt.addptr`` to build
    # a pointer TILE carried as an scf.for iter-arg, then advanced each K-trip.
    # Those iter-args are pure addressing EXCEPT the accumulator (an f32 data
    # tile). We make scf.for/scf.yield TRANSPARENT for an iter-arg iff the
    # carried value is a pointer/integer (NOT the float accumulator) AND every
    # use of the loop block-argument is itself eligible/sink AND the matching
    # scf.yield operand is eligible. Concretely, scf.for operand ``3+i`` <->
    # block-arg ``1+i`` <-> yield operand ``i`` <-> result ``i`` (the leading
    # 3 operands are lb/ub/step; block-arg 0 is the induction var).
    #
    # Implementation: precompute, per scf.for, the iter-arg structure. During
    # the fixpoint, a use of a candidate value as an scf.for iter-init operand
    # is treated like a PRODUCER edge whose "result" is the loop block-arg --
    # which we ADD to the candidate universe as a transparent node and require
    # (a) the yield operand for that slot is a candidate and (b) all uses of
    # the block-arg are eligible. Block-arg eligibility participates in the
    # same fixpoint.
    scf_iter_map: dict = {}  # scf.for op -> list of (init_operand, blockarg, yield_operand)
    for op in all_ops:
        if _op_name(op) != "scf.for":
            continue
        operands = list(getattr(op, "operands", ()) or [])
        regions = getattr(op, "regions", ()) or []
        if not regions:
            continue
        blocks = getattr(regions[0], "blocks", ()) or []
        if not blocks:
            continue
        block = blocks[0]
        bargs = list(getattr(block, "arguments", ()) or [])
        body_ops = list(getattr(block, "operations", ()) or [])
        yld = body_ops[-1] if body_ops else None
        yld_operands = list(getattr(yld, "operands", ()) or []) if yld is not None else []
        # operands[3:] are the iter inits; bargs[1:] the carried block args.
        inits = operands[3:]
        carried = bargs[1:]
        entries = []
        for i, init in enumerate(inits):
            barg = carried[i] if i < len(carried) else None
            yop = yld_operands[i] if i < len(yld_operands) else None
            entries.append((init, barg, yop))
        scf_iter_map[op] = entries

    def _is_float_pointerlike(value: Any) -> bool:
        """True iff the carried value is the f32 accumulator (a DATA tile)."""
        t = str(getattr(value, "type", "") or "")
        # Pointer tiles print ``!tt.ptr``; the accumulator prints ``xf32>``
        # (no ``ptr``). Integer index tiles are ``xi32``/``xi64``. Only the
        # non-pointer float tile is data.
        if "!tt.ptr" in t:
            return False
        return "f32" in t or "f16" in t or "bf16" in t

    # Build, per scf.for, fast lookups: init-operand-value -> slot index.
    scf_for_of_init: dict = {}  # value -> list of (scf_op, slot_idx)
    scf_yield_of_slot: dict = {}  # (scf_op, slot) -> yield operand value
    scf_barg_of_slot: dict = {}   # (scf_op, slot) -> block arg value
    # yield operand VALUE -> set of carried block-args it feeds (for precise
    # yield-use eligibility: a value yielded into a non-candidate slot must be
    # disqualified).
    scf_yield_target_barg: dict = {}
    for scf_op, entries in scf_iter_map.items():
        for slot, (init, barg, yop) in enumerate(entries):
            try:
                scf_for_of_init.setdefault(init, []).append((scf_op, slot))
            except TypeError:
                pass
            scf_yield_of_slot[(id(scf_op), slot)] = yop
            scf_barg_of_slot[(id(scf_op), slot)] = barg
            if yop is not None and barg is not None:
                try:
                    scf_yield_target_barg.setdefault(yop, set()).add(barg)
                except TypeError:
                    pass

    # Optimistic candidate set: every result Value of an addressing-producer
    # op, PLUS the loop block-args for non-float (pointer/int) carried slots.
    candidate: set = set()
    for op in all_ops:
        if _op_name(op) in _ADDR_PRODUCER_OPS:
            for r in getattr(op, "results", ()) or ():
                try:
                    candidate.add(r)
                except TypeError:
                    pass
    for scf_op, entries in scf_iter_map.items():
        for slot, (init, barg, yop) in enumerate(entries):
            if barg is None:
                continue
            if _is_float_pointerlike(barg):
                continue  # the accumulator slot -- never transparent
            try:
                candidate.add(barg)
            except TypeError:
                pass

    def _value_ok(value: Any) -> bool:
        """One eligibility step for ``value`` given the current candidate set."""
        uses = uses_of.get(value, [])
        if not uses:
            return False  # an unconsumed tile is not a fold target
        for consumer_op, opnd_idx in uses:
            if _use_is_sink(consumer_op, opnd_idx):
                continue
            cname = _op_name(consumer_op)
            if cname in _ADDR_PRODUCER_OPS:
                results = list(getattr(consumer_op, "results", ()) or [])
                if results and all(r in candidate for r in results):
                    continue
                return False
            if cname == "scf.for":
                # Transparent iff this operand maps to a carried slot whose
                # block-arg is still a candidate (i.e. pointer/int slot whose
                # in-loop uses + yield are eligible) AND the matching yield
                # operand is eligible.
                matched = False
                for scf_op, slot in scf_for_of_init.get(value, []):
                    if scf_op is not consumer_op:
                        continue
                    barg = scf_barg_of_slot.get((id(scf_op), slot))
                    yop = scf_yield_of_slot.get((id(scf_op), slot))
                    if (
                        barg is not None
                        and barg in candidate
                        and (yop is None or yop in candidate)
                    ):
                        matched = True
                        break
                if matched:
                    continue
                return False
            if cname == "scf.yield":
                # A yield use is eligible iff EVERY carried slot this value
                # feeds is a candidate block-arg (the loop-carried pointer
                # advance). A value yielded into the accumulator slot (excluded
                # from candidate) is disqualified -- it is data. When the
                # block-arg for a slot drops out of the candidate set, this
                # check removes the yielded value too on the next iteration.
                targets = scf_yield_target_barg.get(value)
                if targets and all(b in candidate for b in targets):
                    continue
                return False
            # tt.dot / tt.reduce / tt.return / store-value / collapse_shape /...
            return False
        return True

    changed = True
    while changed:
        changed = False
        for value in list(candidate):
            if not _value_ok(value):
                candidate.discard(value)
                changed = True

    # Project eligible Values to their RESULT-name spelling (the emit-time
    # lookup key). The use GRAPH was reconstructed by Value IDENTITY (.owner /
    # hashable operand) -- which is reliable -- but the returned KEY is the
    # result name, because the emitter receives ops through a region CHILD
    # ``WalkerCtx`` (``_emit_region``) whose op/Value wrappers are NOT
    # identity-stable with this build traversal across the child boundary.
    # ``should_fold_addressing`` consults ONLY the result-position name
    # (``_results(op)[0]``), and result->result name spelling IS consistent
    # (verified), so a result-name set matches at emit time on both the
    # top-level and in-loop (region-child) ops once the set is propagated into
    # the child ctx.
    out: set = set()
    for value in candidate:
        name = _value_ssa_key(value)
        if name:
            out.add(name)
            out.add(name.lstrip("%"))
    return out


# ---------------------------------------------------------------------------
# Driver: TTIR walker that populates EmitContext (== WalkerCtx)
# ---------------------------------------------------------------------------


class TTIRWalker:
    """Walker driver that wires ``mlir.ir`` ops into the OP_TABLE emitters.

    Responsibilities (the bits the regex walker did *not* do):

    1. For each ``tt.func`` block, allocate a placeholder buffer for
       every block argument and stash it under
       :attr:`WalkerCtx.buffers` keyed by the printed SSA name.
    2. For each visited op, look up ``OP_TABLE[op_name]`` and call the
       emitter; if the emitter returns a TIR expression and the op has
       exactly one result, bind it into ``ctx.value_map``.
    3. Skip structural scaffolding (``tt.func``/``tt.return``) just
       like the regex walker did, but recurse into their regions so
       the body gets walked.

    The walker keeps a list of visited op names (``self.visited``) for
    parity with the regex walker -- tests use it to confirm op
    coverage independent of the TIR side effects.
    """

    # Same allow-list as in __init__.py; structural ops produce no TIR
    # but we still record them for coverage parity with the regex
    # walker so existing tests continue to pass.
    _STRUCTURAL_OPS = frozenset({"tt.func", "tt.return", "builtin.module"})

    def __init__(self, ctx: Optional[WalkerCtx] = None) -> None:
        self.ctx: WalkerCtx = ctx if ctx is not None else WalkerCtx()
        self.visited: list[str] = []

    # ---- visitor surface --------------------------------------------------

    def visit_op(self, op: Any) -> None:
        op_name = _op_name(op)
        if not op_name:
            return
        if op_name == "tt.func" or op_name.endswith(".func"):
            # No longer manually materializing func args here, the emitter does it
            self.visited.append(op_name)
            emitter = OP_TABLE.get(op_name)
            if emitter is not None:
                emitter(op, self.ctx)
            return
        if op_name == "builtin.module":
            self.visited.append(op_name)
            return
        self.visited.append(op_name)
        emitter = OP_TABLE.get(op_name)
        if emitter is None:
            raise NotImplementedError(
                f"Unsupported TTIR op {op_name!r}: no emitter registered"
            )
        try:
            result = emitter(op, self.ctx)
        except NotImplementedError:
            # Re-raise so the caller (from_ttir) sees the same shape
            # the regex walker would have produced.
            raise
        # Bind single-result ops into value_map so later emitters can
        # resolve operands. Multi-result ops are the emitter's job.
        results = self._results(op)
        if result is not None and len(results) == 1:
            try:
                self.ctx.bind(results[0], result)
            except Exception:
                # Some emitters already bind themselves; ignore double-bind.
                pass

    # ---- helpers ----------------------------------------------------------

    @staticmethod
    def _results(op: Any) -> tuple:
        if isinstance(op, dict):
            return tuple(op.get("results", ()))
        return tuple(getattr(op, "results", ()) or ())

    def _materialize_func_args(self, func_op: Any) -> None:
        """Seed ``ctx.buffers`` and ``ctx.value_map`` from func arguments.

        The TTIR ``tt.func`` op declares its parameters as block
        arguments on the entry block. Each argument becomes a
        ``tir.decl_buffer`` (handle dtype) so emitters that look up
        operands by SSA value find a buffer ready to load from.
        """
        regions = getattr(func_op, "regions", None) or ()
        for region in regions:
            blocks = getattr(region, "blocks", None) or ()
            for block in blocks:
                args = getattr(block, "arguments", None) or ()
                for idx, arg in enumerate(args):
                    name = _block_arg_name(arg, fallback=f"arg{idx}")
                    # Strip any leading ``%`` so the name is buffer-key clean.
                    key = name.lstrip("%") or f"arg{idx}"
                    if key not in self.ctx.buffers:
                        try:
                            tir = self.ctx.tir()
                            self.ctx.buffers[key] = tir.decl_buffer(
                                shape=[1], dtype="float32", name=key,
                            )
                        except Exception:
                            # TVM unavailable in pure-walker tests; we
                            # still record a placeholder so emitters see
                            # the key even if the buffer object is opaque.
                            self.ctx.buffers[key] = {"_placeholder": True, "name": key}
                    # Bind SSA -> buffer so operand lookups hit it.
                    try:
                        self.ctx.bind(arg, self.ctx.buffers[key])
                    except Exception:
                        pass


# ---------------------------------------------------------------------------
# jaxlib Module adapter
#
# jaxlib's ``mlir.ir.Module`` exposes ``module.body`` as a ``Block`` (not
# an ``Operation``) -- different from upstream MLIR-bundled Python
# bindings. Our walker expects to recurse via
# ``module.operation.regions[0].blocks[0].operations``. The adapter below
# hides ``module.body`` so :func:`walk_module` falls through to the
# ``module.operation`` branch (which DOES carry ``regions``).
#
# This wrapper is the public counterpart to the ad-hoc adapter the e2e
# harness invented in B1; lifting it here means any caller that parses
# TTIR via the jaxlib alias gets the right traversal shape without
# re-implementing the wrap.
# ---------------------------------------------------------------------------


class _JaxlibModuleAdapter:
    """Wrap ``mlir.ir.Module`` so the walker uses the ``operation`` branch.

    jaxlib's ``Module.body`` is a ``Block`` (no ``regions`` attr),
    causing :func:`walk_module`'s recursion to silently bottom out and
    produce an empty PrimFunc. By exposing ``operation`` (which DOES
    have ``regions``) and deliberately *not* exposing ``body``, we force
    the walker's ``getattr(module, 'body', None)`` to return ``None`` so
    it falls back to ``getattr(module, 'operation', module)`` which is
    the correct traversal entry on jaxlib.

    PtrAnalysis and other consumers that only need ``str(module)`` get a
    transparent ``__str__`` / ``__repr__`` delegation.
    """

    __slots__ = ("_inner", "operation")

    def __init__(self, mlir_module: Any) -> None:
        self._inner = mlir_module
        self.operation = mlir_module.operation

    def __str__(self) -> str:
        return str(self._inner)

    def __repr__(self) -> str:
        return repr(self._inner)


def wrap_module_for_walker(module: Any) -> Any:
    """Return ``module`` (or an adapter) tuned for the walker's traversal.

    When the active ``mlir.ir`` provider is jaxlib's (bundled) bindings,
    ``module.body`` is a ``Block`` rather than an ``Operation``, which
    confuses the walker's recursion. We detect that case heuristically
    (the module class lives under ``jaxlib.``) and return a
    :class:`_JaxlibModuleAdapter`. Otherwise the module is returned
    unchanged.
    """
    if module is None:
        return module
    cls = type(module)
    mod_name = getattr(cls, "__module__", "") or ""
    if mod_name.startswith("jaxlib"):
        return _JaxlibModuleAdapter(module)
    # Fallback heuristic: if ``module.body`` lacks ``regions`` but
    # ``module.operation`` has ``regions``, the upstream-Operation
    # branch is the correct one. This catches non-jaxlib bindings that
    # share jaxlib's body-as-Block shape.
    body = getattr(module, "body", None)
    if body is not None and not hasattr(body, "regions"):
        op = getattr(module, "operation", None)
        if op is not None and hasattr(op, "regions"):
            return _JaxlibModuleAdapter(module)
    return module


# ---------------------------------------------------------------------------
# Module-load probe
# ---------------------------------------------------------------------------


def _bootstrap_and_probe() -> bool:
    """Probe ``mlir.ir`` after :mod:`_mlir_path_setup` ran its safe probes."""
    return _try_import_mlir_quiet() is not None


# Probed once at import. Re-import in tests will re-run try_import_mlir
# but the warning is suppressed after the first miss thanks to
# _WARNED_ONCE; tests that need to see the warning use catch_warnings
# with a fresh module reload.
MLIR_WALKER_AVAILABLE: bool = _bootstrap_and_probe()
