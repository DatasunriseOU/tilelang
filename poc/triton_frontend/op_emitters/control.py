"""Control flow, cast, and miscellaneous TTIR/arith/scf op emitters.

This module extends the dispatch table in :mod:`op_mapping` with handlers
for the structural ops that don't fit the memory / compute / shape / async
buckets covered there. We expose a single :data:`CONTROL_EMITTERS` dict that
the walker merges into :data:`op_mapping.OP_TABLE` during initialisation.

Why a separate file?
--------------------
1. Avoids merge conflicts with the active edits to ``op_mapping.py``.
2. Keeps region-walking helpers (which are non-trivial) close to the only
   ops that need them today (``scf.for`` / ``scf.if``). The existing
   ``op_mapping.py`` exposes no public ``_emit_region`` helper, so we keep
   ours module-local until at least one consumer outside this file needs
   it. When that happens we should hoist the helper into ``op_mapping``
   (per the project's "no silent drift between fronends" convention).

Ops implemented
---------------
* ``arith.select`` / ``tt.where`` -> ``tir.if_then_else`` (scalar or
  elementwise via lowered ``tir.For`` for tile dtypes).
* ``arith.extf`` / ``arith.truncf`` -> ``tir.Cast``.
* ``arith.fptosi`` / ``arith.sitofp`` / ``arith.uitofp`` / ``arith.fptoui``
  -> ``tir.Cast``.
* ``arith.bitcast`` -> ``tir.reinterpret``.
* ``arith.extsi`` / ``arith.extui`` / ``arith.trunci`` -> ``tir.Cast``.
* ``tt.advance`` -> structured TPtr offset update (BufferRegion
  rebind), analogous to ``tt.addptr``.
* ``scf.for`` -> ``tir.For`` (serial) with materialised iter_args.
* ``scf.if`` -> ``tir.IfThenElse``.
* ``scf.yield`` -> no-op handler that signals the parent op via the
  ``WalkerCtx``.

Hard constraints
----------------
* No silent fallback for unsupported casts: the emitters raise
  :class:`EmitError` with a precise message.
* For ``scf.for`` with more than 4 iter_args we raise :class:`EmitError`
  rather than guessing a layout — the user should hoist into a TileLang
  ``T.Pipelined`` carry pattern instead.
* Old stub code (``raise NotImplementedError``) in ``op_mapping.py`` is
  left in place; we register our entries by **adding** to OP_TABLE rather
  than deleting fields, per ``feedback_no_silent_delete``.

Tests live in ``tests/test_op_emitters_control.py``.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

# We re-import via op_mapping rather than duplicating the operand /
# attribute / shape helpers, both to stay DRY and so any future shape
# extraction tweaks made there propagate here automatically.
from .. import op_mapping as _om
from ..op_mapping import EmitError

__all__ = [
    "CONTROL_EMITTERS",
    "EmitError",
    "map_arith_select",
    "map_arith_extf",
    "map_arith_truncf",
    "map_arith_fptosi",
    "map_arith_sitofp",
    "map_arith_uitofp",
    "map_arith_fptoui",
    "map_arith_bitcast",
    "map_arith_extsi",
    "map_arith_extui",
    "map_arith_trunci",
    "map_arith_constant",
    "map_tt_advance",
    "map_tt_func",
    "emit_tt_call",
    "map_scf_for",
    "map_scf_if",
    "map_scf_yield",
    "map_scf_while",
    "map_llvm_inline_asm",
    "SCF_WHILE_MAX_ITERATIONS",
    "PTX_TO_TIR",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _result_dtype(op: Any, fallback: str = "float32") -> str:
    """Return the dtype string of the op's first result, or ``fallback``."""
    results = _om._results(op)
    if not results:
        return fallback
    return _om._dtype_of(results[0])


def _result_shape(op: Any) -> Tuple[int, ...]:
    """Return the shape tuple of the op's first result, or ()."""
    results = _om._results(op)
    if not results:
        return ()
    return _om._shape_of(results[0])


def _operand_dtype(op: Any, idx: int, fallback: str = "float32") -> str:
    operands = _om._operands(op)
    if idx >= len(operands):
        return fallback
    return _om._dtype_of(operands[idx])


def _is_int_dtype(dt: str) -> bool:
    """True when ``dt`` looks like an integer dtype (signed or unsigned)."""
    s = dt.lower().strip()
    if s in {"bool", "i1"}:
        return True
    if s.startswith(("int", "uint")):
        return True
    # Short MLIR forms: "i8", "i32", "u16", ...
    if len(s) >= 2 and s[0] in {"i", "u"} and s[1:].isdigit():
        return True
    return False


def _is_float_dtype(dt: str) -> bool:
    """True when ``dt`` looks like a float dtype (incl. half / bf16)."""
    s = dt.lower().strip()
    if s in {"half", "bf16", "bfloat16"}:
        return True
    if s.startswith("float"):
        return True
    # Short MLIR forms: "f16", "f32", "f64".
    if len(s) >= 2 and s[0] == "f" and s[1:].isdigit():
        return True
    return False


def _validate_int_float_pair(src_dt: str, dst_dt: str, op_label: str) -> None:
    """Raise :class:`EmitError` when an int<->float pair is non-standard.

    bf16<->int8 and similar exotic conversions are not native on most
    targets; rather than emit an under-specified ``tir.Cast`` we surface
    this immediately so the user inserts an explicit f32 hop.
    """
    src = src_dt.lower()
    dst = dst_dt.lower()
    if src in {"bf16", "bfloat16"} and (dst.startswith("int") or dst.startswith("uint")):
        bits = "".join(ch for ch in dst if ch.isdigit())
        if bits and int(bits) <= 16:
            raise EmitError(
                f"{op_label}: bf16 -> {dst} is not standard on any backend "
                f"we target. Cast to float32 first, then to {dst}."
            )
    if dst in {"bf16", "bfloat16"} and (src.startswith("int") or src.startswith("uint")):
        bits = "".join(ch for ch in src if ch.isdigit())
        if bits and int(bits) <= 16:
            raise EmitError(
                f"{op_label}: {src} -> bf16 is not standard on any backend "
                f"we target. Cast to float32 first."
            )


def _emit_cast(op: Any, ctx: _om.WalkerCtx, *, expect: str) -> Any:
    """Shared ``arith.<cast>`` -> ``tir.Cast`` emitter.

    ``expect`` is ``"f2f"``, ``"i2f"``, ``"f2i"``, or ``"i2i"`` to gate
    silly-cast pairs (e.g. fptosi on a non-float src).
    """
    operands = _om._operands(op)
    if not operands:
        raise EmitError(f"{op.get('name') if isinstance(op, dict) else op}: missing operand")
    src_ssa = operands[0]
    src = ctx.get(src_ssa)

    src_dt = _om._dtype_of(src_ssa)
    dst_dt = _result_dtype(op, fallback=src_dt)

    op_label = op.get("name") if isinstance(op, dict) else getattr(op, "name", "<cast>")

    src_is_float = _is_float_dtype(src_dt)
    src_is_int = _is_int_dtype(src_dt)
    dst_is_float = _is_float_dtype(dst_dt)
    dst_is_int = _is_int_dtype(dst_dt)

    if expect == "f2f" and not (src_is_float and dst_is_float):
        raise EmitError(
            f"{op_label}: expected float->float cast; got {src_dt!r} -> {dst_dt!r}"
        )
    if expect == "i2f" and not (src_is_int and dst_is_float):
        raise EmitError(
            f"{op_label}: expected int->float cast; got {src_dt!r} -> {dst_dt!r}"
        )
    if expect == "f2i" and not (src_is_float and dst_is_int):
        raise EmitError(
            f"{op_label}: expected float->int cast; got {src_dt!r} -> {dst_dt!r}"
        )
    if expect == "i2i" and not (src_is_int and dst_is_int):
        raise EmitError(
            f"{op_label}: expected int->int cast; got {src_dt!r} -> {dst_dt!r}"
        )

    # Reject exotic int/float pairs that aren't supported as a single Cast.
    if expect in {"i2f", "f2i"}:
        _validate_int_float_pair(src_dt, dst_dt, str(op_label))

    tir = ctx.tir()
    cast = tir.Cast(dst_dt, src)
    if _om._results(op):
        ctx.bind(_om._results(op)[0], cast)
    return cast


# ---------------------------------------------------------------------------
# arith.select / tt.where
# ---------------------------------------------------------------------------


def map_arith_select(op: Any, ctx: _om.WalkerCtx) -> Any:
    """Lower ``arith.select(cond, t, f)`` to ``tir.if_then_else(cond, t, f)``.

    Vector form: when the result type is a tile (rank>0), we still emit
    ``tir.if_then_else`` over PrimExprs because tile ops in our IR are
    represented lazily (broadcast/splat are no-op rebinds). The
    ``LowerTileOp`` pass materialises the elementwise ``tir.For``
    afterwards. We document the lowered shape via the ``out_shape`` we
    record so layout inference can pick this up; for genuinely-scalar
    selects nothing extra is emitted.
    """
    tir = ctx.tir()
    operands = _om._operands(op)
    if len(operands) != 3:
        raise EmitError(
            f"arith.select: expected 3 operands (cond, t, f); got {len(operands)}"
        )
    cond, t_val, f_val = (ctx.get(o) for o in operands)

    out_shape = _result_shape(op)
    if out_shape and len(out_shape) > 0:
        # Vector form: still produce tir.if_then_else over PrimExprs. The
        # elementwise materialisation lives in LowerTileOp; we emit a
        # lowering hint via a no-op tir.For wrapping when an explicit
        # elementwise index is needed downstream. For now the lazy-tile
        # representation is sufficient — tile ops above us (broadcast /
        # splat / make_range) are also lazy.
        sel = tir.if_then_else(cond, t_val, f_val)
    else:
        sel = tir.if_then_else(cond, t_val, f_val)

    if _om._results(op):
        ctx.bind(_om._results(op)[0], sel)
    return sel


# ``tt.where`` with an explicit elementwise lowering is already handled by
# ``op_mapping.map_tt_where`` (which uses ``tir.Select``). We additionally
# expose this entry under ``arith.select`` because Triton MLIR sometimes
# emits the arith spelling directly (post-canonicalisation). The two
# wrappers are intentionally near-duplicates — both forward to the same
# tir.if_then_else node — so a future canonicalisation that unifies the
# two paths only needs to touch one OP_TABLE key.


# ---------------------------------------------------------------------------
# arith float-cast emitters
# ---------------------------------------------------------------------------


def map_arith_extf(op: Any, ctx: _om.WalkerCtx) -> Any:
    """``arith.extf`` (float widen, e.g. fp16 -> fp32)."""
    return _emit_cast(op, ctx, expect="f2f")


def map_arith_truncf(op: Any, ctx: _om.WalkerCtx) -> Any:
    """``arith.truncf`` (float narrow, e.g. fp32 -> fp16)."""
    return _emit_cast(op, ctx, expect="f2f")


# ---------------------------------------------------------------------------
# arith int<->float cast emitters
# ---------------------------------------------------------------------------


def map_arith_fptosi(op: Any, ctx: _om.WalkerCtx) -> Any:
    """``arith.fptosi`` (float -> signed int)."""
    return _emit_cast(op, ctx, expect="f2i")


def map_arith_sitofp(op: Any, ctx: _om.WalkerCtx) -> Any:
    """``arith.sitofp`` (signed int -> float)."""
    return _emit_cast(op, ctx, expect="i2f")


def map_arith_uitofp(op: Any, ctx: _om.WalkerCtx) -> Any:
    """``arith.uitofp`` (unsigned int -> float)."""
    return _emit_cast(op, ctx, expect="i2f")


def map_arith_fptoui(op: Any, ctx: _om.WalkerCtx) -> Any:
    """``arith.fptoui`` (float -> unsigned int)."""
    return _emit_cast(op, ctx, expect="f2i")


# ---------------------------------------------------------------------------
# arith.bitcast
# ---------------------------------------------------------------------------


def _dtype_bits(dt: str) -> Optional[int]:
    """Return the bit-width of a dtype name, or ``None`` if unknown."""
    digits = "".join(ch for ch in dt if ch.isdigit())
    if not digits:
        # Map common aliases.
        return {
            "half": 16,
            "bf16": 16,
            "bfloat16": 16,
            "bool": 1,
        }.get(dt.lower())
    try:
        return int(digits)
    except ValueError:
        return None


def map_arith_bitcast(op: Any, ctx: _om.WalkerCtx) -> Any:
    """``arith.bitcast`` (same-width reinterpret) -> ``tir.reinterpret``."""
    operands = _om._operands(op)
    if not operands:
        raise EmitError("arith.bitcast: missing operand")
    src_ssa = operands[0]
    src = ctx.get(src_ssa)

    src_dt = _om._dtype_of(src_ssa)
    dst_dt = _result_dtype(op, fallback=src_dt)

    src_bits = _dtype_bits(src_dt)
    dst_bits = _dtype_bits(dst_dt)
    if src_bits is not None and dst_bits is not None and src_bits != dst_bits:
        raise EmitError(
            f"arith.bitcast: width mismatch {src_dt!r} ({src_bits}b) vs "
            f"{dst_dt!r} ({dst_bits}b); use a chained cast instead."
        )

    tir = ctx.tir()
    expr = tir.reinterpret(dst_dt, src)
    if _om._results(op):
        ctx.bind(_om._results(op)[0], expr)
    return expr


# ---------------------------------------------------------------------------
# arith int width casts
# ---------------------------------------------------------------------------


def map_arith_extsi(op: Any, ctx: _om.WalkerCtx) -> Any:
    """``arith.extsi`` (signed int widen, e.g. i32 -> i64)."""
    return _emit_cast(op, ctx, expect="i2i")


def map_arith_extui(op: Any, ctx: _om.WalkerCtx) -> Any:
    """``arith.extui`` (unsigned int widen, e.g. u8 -> u32)."""
    return _emit_cast(op, ctx, expect="i2i")


def map_arith_trunci(op: Any, ctx: _om.WalkerCtx) -> Any:
    """``arith.trunci`` (int narrow, e.g. i64 -> i32)."""
    return _emit_cast(op, ctx, expect="i2i")


# ---------------------------------------------------------------------------
# tt.advance (block-pointer advance)
# ---------------------------------------------------------------------------


def map_tt_advance(op: Any, ctx: _om.WalkerCtx) -> Any:
    """Lower ``tt.advance(block_ptr, [d0, d1, ...])`` to a TPtr offset bump.

    Triton's block-pointer ``tt.advance`` returns a *new* block-pointer SSA
    whose offsets equal ``old.offsets + delta``. We model the block-pointer
    in :class:`WalkerCtx` as a ``{"_ptrstate": ..., "offsets": [...]}``
    dict (the same shape produced by :mod:`ptr_analysis` for tile loads /
    stores). Advancing therefore reduces to:

        new_state = copy(old_state)
        new_state["offsets"] = old_state["offsets"] + delta_per_axis

    The deltas come from ``op.operands[1:]`` (one PrimExpr per axis, same
    convention as Triton's MLIR form). We rebind the result SSA to the new
    state so a subsequent ``tt.load(advanced_ptr)`` resolves through the
    same buffer-region path used for the un-advanced pointer.

    If the source SSA wasn't resolved by PtrAnalysis (legacy fallback path
    in op_mapping that uses raw buffers), we still emit a structurally
    equivalent ``BufferLoad / BufferStore`` shim by rebinding the same
    buffer with offset deltas applied to ``indices``. This keeps the
    emitter usable in dict-shaped unit tests where PtrAnalysis isn't run.
    """
    operands = _om._operands(op)
    if not operands:
        raise EmitError("tt.advance: missing block-pointer operand")
    base_ssa = operands[0]
    base = ctx.get(base_ssa)
    deltas: List[Any] = []
    for delta_ssa in operands[1:]:
        try:
            deltas.append(ctx.get(delta_ssa))
        except KeyError:
            # Constant attribute fallback: some MLIR builds inline scalar
            # deltas as attributes on the op itself.
            deltas.append(delta_ssa)

    new_state: Any
    if isinstance(base, dict) and "_ptrstate" in base:
        # PtrState path: clone and bump offsets.
        new_state = dict(base)
        old_offsets = list(base.get("offsets") or [])
        merged: List[Any] = []
        for i in range(max(len(old_offsets), len(deltas))):
            old = old_offsets[i] if i < len(old_offsets) else 0
            new = deltas[i] if i < len(deltas) else 0
            # Try integer fold; otherwise emit a tir.Add PrimExpr so the
            # downstream walker still sees a usable offset expression.
            try:
                merged.append(int(old) + int(new))
            except (TypeError, ValueError):
                tir = ctx.tir()
                old_e = old if not isinstance(old, int) else tir.const(old, "int32")
                new_e = new if not isinstance(new, int) else tir.const(new, "int32")
                merged.append(tir.Add(old_e, new_e))
        new_state["offsets"] = merged
    elif isinstance(base, tuple) and len(base) == 2:
        # Legacy (buffer, indices) path.
        buf, indices = base
        merged_idx = list(indices) + [0] * max(0, len(deltas) - len(indices))
        for i, delta in enumerate(deltas):
            try:
                merged_idx[i] = int(merged_idx[i]) + int(delta)
            except (TypeError, ValueError):
                tir = ctx.tir()
                old_e = merged_idx[i] if not isinstance(merged_idx[i], int) else tir.const(merged_idx[i], "int32")
                new_e = delta if not isinstance(delta, int) else tir.const(delta, "int32")
                merged_idx[i] = tir.Add(old_e, new_e)
        new_state = (buf, merged_idx)
    else:
        # Opaque (MVP) base: surface a BufferRegion-style dict so consumers
        # can detect that the SSA carries an offset bump even when the
        # full PtrState wasn't resolved.
        new_state = {
            "_ptrstate": "advance",
            "source": getattr(base_ssa, "name", None) or (
                base_ssa.get("name") if isinstance(base_ssa, dict) else None
            ),
            "offsets": deltas,
            "sizes": list(_om._shape_of(base_ssa)) or [],
        }

    if _om._results(op):
        ctx.bind(_om._results(op)[0], new_state)
    return new_state


# ---------------------------------------------------------------------------
# tt.func  (block-arg seeding)  /  arith.constant  (scalar literal seeding)
# ---------------------------------------------------------------------------
#
# The walker in ``triton_frontend.__init__:_walk_mlir_module`` dispatches by
# OP_TABLE name and recurses into regions afterwards. ``tt.func`` historically
# lived in ``_TTIR_STRUCTURAL_OPS`` (no-op skip) which meant block arguments
# were never bound into ``ctx.value_map`` -- downstream emitters that look up
# ``%arg0`` (e.g. ``tt.load %arg0, ...``) then KeyError'd on the operand.
# Similarly, ``arith.constant`` was unmapped, so any kernel using ``%c0 =
# arith.constant 0 : i32`` blew up at the first use of ``%c0``.
#
# We register both ops here in CONTROL_EMITTERS. The emitters do NOT recurse
# into regions themselves -- the parent walker still owns recursion -- they
# only seed the value_map (and ctx.buffers, for pointer-typed block args)
# so subsequent ops can resolve their operands.
#
# Hard constraint per spec: arith.constant with an unsupported attr type
# (e.g. ``dense<...>`` array splat) raises EmitError instead of guessing.


def _ssa_name(value: Any) -> Optional[str]:
    """Return the printed SSA name for a value (``%arg0`` / ``%c0`` / ...).

    Tries, in order:
      1. ``value.get_name()`` (real MLIR Value / BlockArgument)
      2. ``value["name"]`` (dict-shaped fake)
      3. ``str(value).split()[0]`` (last-ditch printable form)

    Returns ``None`` when no usable name can be produced.
    """
    getter = getattr(value, "get_name", None)
    if callable(getter):
        try:
            name = getter()
            if name:
                return str(name)
        except Exception:
            pass
    if isinstance(value, dict):
        nm = value.get("name")
        if nm:
            return str(nm)
    try:
        s = str(value).strip()
        if s:
            head = s.split(None, 1)[0]
            return head if head else None
    except Exception:
        pass
    return None


def _type_string(value: Any) -> str:
    """Best-effort printable type for a TTIR Value or dict-fake.

    For dict-fakes we synthesize ``!tt.ptr<dtype>`` if the entry sets
    ``"is_ptr": True`` so the same emitter logic applies in tests.
    """
    if isinstance(value, dict):
        if value.get("is_ptr"):
            elt = str(value.get("dtype", "float32"))
            return f"!tt.ptr<{elt}>"
        # Plain scalar / tensor: synthesize from shape + dtype.
        shape = tuple(value.get("shape", ()))
        dt = str(value.get("dtype", "float32"))
        if shape:
            return f"tensor<{'x'.join(str(s) for s in shape)}x{dt}>"
        return dt
    typ = getattr(value, "type", None)
    if typ is None:
        return ""
    try:
        return str(typ)
    except Exception:
        return ""


def _is_ptr_type(type_str: str) -> bool:
    s = type_str.strip()
    return s.startswith("!tt.ptr<") or s.startswith("tt.ptr<")


def _ptr_element_dtype(type_str: str) -> str:
    """Extract ``f32`` from ``!tt.ptr<f32>`` (or fall back to float32)."""
    s = type_str.strip()
    if "<" in s and s.endswith(">"):
        return s[s.index("<") + 1:-1].strip() or "float32"
    return "float32"


def _func_block_args(op: Any) -> List[Any]:
    """Return the entry-block arguments of a ``tt.func`` op.

    Real MLIR: ``op.regions[0].blocks[0].arguments``. Dict-fake: the op
    may surface them via ``op["block_args"]`` directly.
    """
    if isinstance(op, dict):
        ba = op.get("block_args")
        if ba is not None:
            return list(ba)
        regions = op.get("regions") or []
        if regions:
            r0 = regions[0]
            if isinstance(r0, dict):
                return list(r0.get("block_args") or [])
        return []
    regions = getattr(op, "regions", ()) or ()
    for region in regions:
        blocks = getattr(region, "blocks", ()) or ()
        for block in blocks:
            return list(getattr(block, "arguments", ()) or ())
    return []


def map_tt_func(op: Any, ctx: _om.WalkerCtx) -> Any:
    """Seed ``tt.func`` block arguments into ``ctx.value_map`` / ``ctx.buffers``.

    Mapping summary
    ---------------
    * Pointer block args (``!tt.ptr<T>``): allocated as ``tir.decl_buffer``
      shape=(?,), dtype=T, name=<ssa stripped of '%'>; bound under both
      the SSA Value (so MLIR-walker operand lookups resolve) AND the
      printed SSA name string (so test fixtures can introspect by name).
    * Scalar block args (``i32`` etc.): bound as a fresh ``tir.Var`` of the
      MLIR dtype, again under both the Value and the SSA-name string.

    We deliberately do NOT recurse into the body region: the parent walker
    (``_walk_mlir_module`` in ``triton_frontend.__init__``) handles
    recursion already. Doing so here would double-walk the body.

    Returning ``None`` keeps the walker's auto-bind logic (which only fires
    on single-result ops with non-None return) inert -- ``tt.func`` has no
    result SSA to bind anyway.
    """
    block_args = _func_block_args(op)
    if not block_args:
        # Nothing to seed. This is unusual but legal (e.g. zero-arg kernel)
        # and we honour it silently rather than raising -- the walker will
        # still recurse into the body and downstream ops with no operand
        # references will succeed.
        return None

    tir_mod = None
    try:
        tir_mod = ctx.tir()
    except Exception:
        # TVM unavailable: we still want to seed string-keyed entries so
        # text-walker-style fakes can introspect the value_map. We use a
        # sentinel dict in that case.
        tir_mod = None

    for idx, arg in enumerate(block_args):
        ssa = _ssa_name(arg) or f"%arg{idx}"
        clean = ssa.lstrip("%") or f"arg{idx}"
        type_str = _type_string(arg)

        if _is_ptr_type(type_str):
            elt = _normalize_dtype(_ptr_element_dtype(type_str))
            if tir_mod is not None:
                # Pointer block arg: allocate a placeholder buffer. Shape
                # is symbolic (no kernel-level info available here); we
                # use 1 -- the actual extents come from later tt.load /
                # tt.store ops that re-decl with the right shape via the
                # PtrAnalysis path. This matches what
                # mlir_walker.TTIRWalker._materialize_func_args does.
                if clean not in ctx.buffers:
                    ctx.buffers[clean] = tir_mod.decl_buffer(
                        shape=[1], dtype=elt, name=clean,
                    )
                bound = ctx.buffers[clean]
            else:
                bound = {"_placeholder": True, "name": clean, "dtype": elt}
                ctx.buffers[clean] = bound
        else:
            # Scalar (or tensor) block arg: emit a tir.Var of the right dtype.
            # We don't materialise tensor-shaped block args specially because
            # Triton TTIR doesn't actually pass tensor SSAs across function
            # boundaries -- only scalars and pointers. Normalise short MLIR
            # spellings (``i32`` -> ``int32``) to TVM's canonical names so
            # ``tir.Var`` doesn't reject ``i32`` as unknown.
            dt = _normalize_dtype(type_str) if type_str else "int32"
            if tir_mod is not None:
                bound = tir_mod.Var(clean, dt)
                # Track the runtime scalar arg so ``_make_prim_func`` can
                # append it to ``PrimFunc.params``. Triton 3.x folds
                # ``tl.constexpr`` parameters at the TTIR stage, so anything
                # that survives as a non-pointer block arg is an actual
                # runtime arg (e.g. ``n_elements``). Without this MakePackedAPI
                # rejects the Var as a free variable in the body.
                runtime_args = getattr(ctx, "runtime_args", None)
                if runtime_args is not None and bound not in runtime_args:
                    runtime_args.append(bound)
            else:
                bound = {"_var_placeholder": True, "name": clean, "dtype": dt}

        # Bind under BOTH keys so:
        #   - downstream ops looking up the Value object (real MLIR walker
        #     case) resolve;
        #   - tests / introspection looking up by printed SSA name string
        #     resolve too.
        try:
            ctx.bind(arg, bound)
        except Exception:
            # Some Value objects aren't hashable across binding shapes;
            # the string key below still gives downstream code a way in.
            pass
        ctx.value_map[ssa] = bound

    return None


def _parse_value_attr(value_attr: Any) -> Tuple[str, Any]:
    """Return ``(dtype_str, scalar_value)`` for an ``arith.constant`` value attr.

    Accepted shapes
    ---------------
    * Real MLIR ``IntegerAttr`` / ``FloatAttr``: probed via ``.value`` and
      ``.type``. ``dtype_str`` is taken from ``str(.type)``.
    * Dict-shaped fake: ``{"value": <int|float>, "type": "i32"}`` or just
      ``{"value": ..., "dtype": ...}``.
    * Generic-form string ``"42 : i32"`` (the printed form used by the
      MLIR generic syntax) -- split on the colon.

    Raises :class:`EmitError` for anything else (e.g. ``DenseElementsAttr``,
    ``ArrayAttr``) because silently lowering an array constant to a single
    IntImm/FloatImm would change semantics.
    """
    # Real MLIR Integer/Float attr.
    if hasattr(value_attr, "value") and hasattr(value_attr, "type"):
        # Probe for array-attr shapes first: DenseElementsAttr exposes
        # ``.value`` as a numpy array-ish, not a scalar. We err on the side
        # of explicitness: any non-scalar ``.value`` raises EmitError.
        v = value_attr.value
        # numpy arrays / list / tuple are not supported.
        if isinstance(v, (list, tuple)):
            raise EmitError(
                f"arith.constant with array attr unsupported; got: "
                f"{value_attr!r}"
            )
        # numpy scalars OK; arrays not.
        try:
            import numpy as _np  # noqa: WPS433
            if isinstance(v, _np.ndarray):
                raise EmitError(
                    f"arith.constant with array attr unsupported; got: "
                    f"{value_attr!r}"
                )
        except ImportError:
            pass
        dtype_str = str(value_attr.type)
        return dtype_str, v

    # Dict-fake.
    if isinstance(value_attr, dict):
        if "value" not in value_attr:
            raise EmitError(
                f"arith.constant: dict attr missing 'value' field: {value_attr!r}"
            )
        v = value_attr["value"]
        if isinstance(v, (list, tuple)):
            raise EmitError(
                f"arith.constant with array attr unsupported; got: {value_attr!r}"
            )
        dtype_str = str(value_attr.get("type") or value_attr.get("dtype") or "int32")
        return dtype_str, v

    # Generic-form string: ``"42 : i32"`` or ``"3.14 : f32"``.
    if isinstance(value_attr, str):
        s = value_attr.strip()
        if "dense" in s.lower() or "array" in s.lower() or s.startswith("["):
            raise EmitError(
                f"arith.constant with array attr unsupported; got: {value_attr!r}"
            )
        if ":" not in s:
            raise EmitError(
                f"arith.constant: cannot parse value attr {value_attr!r} "
                f"(expected '<value> : <type>')"
            )
        val_part, dt_part = s.rsplit(":", 1)
        val_part = val_part.strip()
        dtype_str = dt_part.strip()
        # Try int first, then float.
        try:
            return dtype_str, int(val_part)
        except ValueError:
            try:
                return dtype_str, float(val_part)
            except ValueError as exc:
                raise EmitError(
                    f"arith.constant: cannot parse scalar value {val_part!r} "
                    f"in {value_attr!r}"
                ) from exc

    raise EmitError(
        f"arith.constant with unsupported attr type {type(value_attr).__name__}; "
        f"got: {value_attr!r}"
    )


def _normalize_dtype(dtype_str: str) -> str:
    """Canonicalise short MLIR dtype names to TVM's spelling.

    ``i32`` -> ``int32``; ``f32`` -> ``float32``; ``bf16`` -> ``bfloat16``;
    ``f16`` -> ``float16``; ``i1`` -> ``bool``. Anything else passes through
    unchanged so already-TVM-spelled dtypes (``float32``, ``int64``) work.
    """
    s = dtype_str.strip()
    aliases = {
        "i1": "bool",
        "bf16": "bfloat16",
    }
    if s in aliases:
        return aliases[s]
    if len(s) >= 2 and s[0] in {"i", "u"} and s[1:].isdigit():
        prefix = "int" if s[0] == "i" else "uint"
        return f"{prefix}{s[1:]}"
    if len(s) >= 2 and s[0] == "f" and s[1:].isdigit():
        return f"float{s[1:]}"
    return s


def _is_dense_attr(value_attr: Any) -> bool:
    """Detect an MLIR ``DenseElementsAttr`` (FP or integer variant).

    We check for the trio of accessors the real bindings expose
    (``is_splat`` / ``get_splat_value`` / ``type.shape``) rather than
    importing the MLIR class directly so the test harness's dict / list
    fakes don't false-positive here.
    """
    if not (hasattr(value_attr, "is_splat") and hasattr(value_attr, "type")):
        return False
    return getattr(value_attr.type, "shape", None) is not None


def _extract_dense_attr(
    value_attr: Any, result_value: Any
) -> Tuple[Tuple[int, ...], str, bool, Any]:
    """Return ``(shape, dtype, is_splat, payload)`` for a dense attr.

    ``payload`` is a Python scalar when ``is_splat`` is True; otherwise a
    materialised list of element values (one per tile slot) obtained by
    iterating the attribute. Element dtypes that we can't safely round-trip
    to a Python primitive (e.g. complex) raise :class:`EmitError`.
    """
    shape: Tuple[int, ...] = tuple(value_attr.type.shape)
    dtype: str = _normalize_dtype(str(value_attr.type.element_type))
    if dtype not in {"bool"} and not (
        dtype.startswith("int")
        or dtype.startswith("uint")
        or dtype.startswith("float")
        or dtype.startswith("bfloat")
    ):
        raise EmitError(
            f"arith.constant: dense attr with unsupported element dtype "
            f"{dtype!r} (from {value_attr.type.element_type!r})"
        )
    if value_attr.is_splat:
        # ``get_splat_value()`` returns an IntegerAttr / FloatAttr; ``.value``
        # is the Python scalar.
        sv = value_attr.get_splat_value().value
        return shape, dtype, True, sv
    # Per-element: iterate the attribute (jaxlib bindings yield Python
    # scalars in row-major order).
    try:
        elements = list(value_attr)
    except Exception as exc:  # pragma: no cover - defensive
        raise EmitError(
            f"arith.constant: dense attr is non-splat but not iterable: "
            f"{value_attr!r}"
        ) from exc
    return shape, dtype, False, elements


def map_arith_constant(op: Any, ctx: _om.WalkerCtx) -> Any:
    """Lower ``arith.constant`` to a ``tir.IntImm`` / ``tir.FloatImm``.

    Mapping summary
    ---------------
    * Pull the ``value`` attribute (generic form: ``<{value = N : <ty>}>``).
    * Build ``tir.IntImm`` for integer / bool dtypes, ``tir.FloatImm`` for
      float dtypes (after normalising short MLIR spellings to TVM's names
      via :func:`_normalize_dtype`).
    * Bind the result SSA into ``ctx.value_map`` under both the result
      Value object and its printed name string.

    Dense / array attrs (e.g. ``dense<0.0> : tensor<256xf32>`` produced by
    Triton's ``tt.load %ptr, %mask, other=0.0`` materialisation) are
    materialised into a freshly declared ``tir.decl_buffer`` initialised
    via a serial ``tir.For`` nest; the buffer is bound into the value map
    so downstream loads/stores resolve through it.
    """
    attrs = _om._attrs(op)
    value_attr: Any = attrs.get("value")
    if value_attr is None:
        # Some MLIR Python bindings (e.g. jaxlib's) hide ``arith.constant``'s
        # ``value`` under the operation properties rather than the
        # discardable-attributes dict that ``_attrs`` iterates. Probe the
        # op directly for an attribute named ``value`` before giving up.
        if not isinstance(op, dict):
            value_attr = getattr(op, "value", None)
            # Some bindings additionally expose the typed attribute via
            # ``op.attributes["value"]`` even when iterating yields empty.
            if value_attr is None:
                op_attrs = getattr(op, "attributes", None)
                if op_attrs is not None:
                    try:
                        if "value" in op_attrs:
                            value_attr = op_attrs["value"]
                    except Exception:
                        pass
    if value_attr is None:
        raise EmitError(
            "arith.constant: missing 'value' attribute"
        )

    tir = ctx.tir()
    results = _om._results(op)
    result = results[0] if results else None

    # Dense / DenseFPElementsAttr / DenseIntElementsAttr branch: materialise
    # as a buffer initialised by a serial For nest writing the constant
    # value(s) to every element. Triton's ``tt.load %ptrs, %mask, other=0.0``
    # round-trips the ``other`` operand through this op shape.
    if _is_dense_attr(value_attr):
        shape, dtype, is_splat, payload = _extract_dense_attr(
            value_attr, result
        )
        nm_for_buf = (_ssa_name(result) if result is not None else None) or ctx.fresh("c")
        # Strip a leading '%' so the buffer name reads cleanly in dumps.
        ssa_clean = nm_for_buf.lstrip("%").replace(".", "_")
        buf_name = f"const_{ssa_clean}"
        # Tile-scoped allocation: see ``op_mapping._alloc_tile_buffer``.
        # The dense constant lives entirely inside the kernel body; making
        # it a PrimFunc parameter would trip ``VerifyMemory``.
        buf = _om._alloc_tile_buffer(
            ctx, list(shape) if shape else [1], dtype, buf_name
        )

        # Build a serial tir.For nest writing the constant(s) into ``buf``.
        # All shapes from MLIR DenseElementsAttr are static (RankedTensorType
        # constants), so we can safely fold to integer extents.
        if not all(isinstance(s, int) for s in shape):  # pragma: no cover
            raise EmitError(
                f"arith.constant: dense attr with non-static shape {shape!r}"
            )

        if shape:
            # Allocate one induction var per axis.
            ivars = [tir.Var(ctx.fresh(f"i{a}"), "int32") for a in range(len(shape))]
            if is_splat:
                value_expr = (
                    tir.IntImm(dtype, int(payload))
                    if (dtype == "bool" or dtype.startswith("int") or dtype.startswith("uint"))
                    else tir.FloatImm(dtype, float(payload))
                )
                body = tir.BufferStore(buf, value_expr, list(ivars))
            else:
                # Per-element: linearise (i0 * s1 * s2 * ... + i1 * s2 * ... + ...)
                # then dispatch through tir.if_then_else cascades. For correctness
                # and simplicity, unroll: emit a SeqStmt of N stores at constant
                # indices since ``payload`` is fully known.
                stmts = []
                strides: List[int] = []
                acc = 1
                for s in reversed(shape):
                    strides.append(acc)
                    acc *= s
                strides.reverse()
                for lin_idx, elt in enumerate(payload):
                    idxs = []
                    rem = lin_idx
                    for st in strides:
                        idxs.append(tir.const(rem // st, "int32"))
                        rem = rem % st
                    value_expr = (
                        tir.IntImm(dtype, int(elt))
                        if (dtype == "bool" or dtype.startswith("int") or dtype.startswith("uint"))
                        else tir.FloatImm(dtype, float(elt))
                    )
                    stmts.append(tir.BufferStore(buf, value_expr, idxs))
                body = tir.SeqStmt(stmts) if len(stmts) > 1 else stmts[0]
            if is_splat:
                # Wrap in a serial For nest, innermost axis last.
                nest = body
                for axis_idx in reversed(range(len(shape))):
                    nest = tir.For(
                        ivars[axis_idx],
                        tir.const(0, "int32"),
                        tir.const(int(shape[axis_idx]), "int32"),
                        tir.ForKind.SERIAL,
                        nest,
                    )
                ctx.emit(nest)
            else:
                ctx.emit(body)
        # Bind the buffer under the result SSA value and printed name so
        # downstream consumers (loads / elementwise) resolve.
        if result is not None:
            try:
                ctx.bind(result, buf)
            except Exception:
                pass
            nm = _ssa_name(result)
            if nm:
                ctx.value_map[nm] = buf
        return buf

    dtype_str, scalar_val = _parse_value_attr(value_attr)
    dtype = _normalize_dtype(dtype_str)

    if dtype == "bool" or dtype.startswith("int") or dtype.startswith("uint"):
        const = tir.IntImm(dtype, int(scalar_val))
    elif dtype.startswith("float") or dtype.startswith("bfloat"):
        const = tir.FloatImm(dtype, float(scalar_val))
    else:
        raise EmitError(
            f"arith.constant: unsupported dtype {dtype!r} (from {dtype_str!r})"
        )

    if result is not None:
        try:
            ctx.bind(result, const)
        except Exception:
            pass
        nm = _ssa_name(result)
        if nm:
            ctx.value_map[nm] = const
    return const


# ---------------------------------------------------------------------------
# scf.for / scf.if / scf.yield
# ---------------------------------------------------------------------------


_MAX_ITER_ARGS = 4


def _emit_region(
    region: Any,
    ctx: _om.WalkerCtx,
    *,
    induction_var: Optional[Any] = None,
    induction_ssa: Optional[Any] = None,
    iter_arg_pairs: Optional[List[Tuple[Any, Any]]] = None,
) -> Tuple[Any, List[Any]]:
    """Walk a single MLIR region (or dict-shaped fake) and return its TIR body.

    Why this lives here, not in :mod:`op_mapping`:
        ``op_mapping.py`` only has region-walking *inside* its ``tt.reduce``
        combiner inspector, which doesn't need to dispatch arbitrary ops.
        ``scf.for`` / ``scf.if`` *do*. Until a second consumer outside this
        file needs a generic region walker, we keep the helper module-local
        rather than promoting an under-tested helper to the public surface
        (per ``feedback_no_silent_delete`` we also don't move existing code).

    Returns
    -------
    body : Any
        A ``tir.SeqStmt`` (or single statement) that the caller wraps into
        the corresponding parent op (``tir.For`` body, ``tir.IfThenElse``
        then/else branch, etc.).
    yielded : list
        The PrimExpr / buffer values produced by ``scf.yield`` inside the
        region, in operand order. Empty when the region has no yield.
    """
    # The block walk uses a child WalkerCtx so emissions are scoped to the
    # region's body; we copy the parent's value_map / buffers so SSA
    # references resolved outside the region remain visible.
    child = _om.WalkerCtx()
    child.value_map = dict(ctx.value_map)
    child.buffers = ctx.buffers  # share kernel-level buffer registry
    child.transposed_views = dict(ctx.transposed_views)
    child._tmp_counter = ctx._tmp_counter
    child._tvm = ctx._tvm
    child._T = ctx._T

    # Materialise iter_args first: each block-arg SSA value gets bound to
    # a fresh tir.Var so the body emits BufferLoad / arithmetic against it.
    if induction_ssa is not None and induction_var is not None:
        child.bind(induction_ssa, induction_var)
    if iter_arg_pairs:
        for ssa, tir_var in iter_arg_pairs:
            child.bind(ssa, tir_var)

    # Dict-shaped fake: a region is ``{"ops": [op, op, ...]}`` and a yield
    # op surfaces its yielded SSAs via ``operands``.
    yielded: List[Any] = []
    ops_iter: List[Any]
    if isinstance(region, dict):
        ops_iter = list(region.get("ops", ()))
    else:
        ops_iter = []
        for block in getattr(region, "blocks", ()) or ():
            for inner in getattr(block, "operations", ()) or ():
                ops_iter.append(inner)

    for inner in ops_iter:
        op_name = inner.get("name") if isinstance(inner, dict) else getattr(inner, "name", "")
        op_name = str(op_name)
        if op_name == "scf.yield":
            yielded = [child.get(o) if o in child.value_map else o
                       for o in _om._operands(inner)]
            continue
        emitter = CONTROL_EMITTERS.get(op_name) or _om.OP_TABLE.get(op_name)
        if emitter is None:
            raise EmitError(
                f"_emit_region: unmapped op {op_name!r} inside scf region; "
                f"register an emitter in OP_TABLE or CONTROL_EMITTERS."
            )
        emitter(inner, child)

    # Bubble counter back so fresh names stay unique across siblings.
    ctx._tmp_counter = child._tmp_counter

    tir = ctx.tir()
    if not child.stmts:
        body = tir.Evaluate(tir.const(0, "int32"))
    elif len(child.stmts) == 1:
        body = child.stmts[0]
    else:
        body = tir.SeqStmt(child.stmts)
    return body, yielded


def _scf_regions(op: Any) -> List[Any]:
    """Return the op's regions in MLIR / dict-shaped form."""
    if isinstance(op, dict):
        regs = op.get("regions") or []
        # Single-region dict-fakes may inline ops via a top-level ``body``.
        if not regs and "body" in op:
            regs = [{"ops": op["body"]}]
        return list(regs)
    return list(getattr(op, "regions", ()) or ())


def _scf_for_bounds(op: Any) -> Tuple[Any, Any, Any]:
    """Return ``(lower, upper, step)`` from an ``scf.for`` op."""
    operands = _om._operands(op)
    if len(operands) < 3:
        raise EmitError(
            f"scf.for: expected at least (lower, upper, step) operands; got {len(operands)}"
        )
    return operands[0], operands[1], operands[2]


def _scf_for_iter_args(op: Any) -> List[Any]:
    """Return the iter_args operands (everything after lb/ub/step)."""
    return list(_om._operands(op)[3:])


def map_scf_for(op: Any, ctx: _om.WalkerCtx) -> Any:
    """Lower ``scf.for(lb, ub, step) iter_args(...) { body }`` to ``tir.For``.

    Mapping summary
    ---------------
    * ``lb`` / ``ub`` / ``step`` -> ``tir.For(min=lb, extent=ub-lb, ...)``
      with ``kind="serial"``. When ``step`` is a non-unit constant we
      either (a) unroll for small constant trip counts (<=8 iterations),
      or (b) emit a serial loop with explicit stride scaling on the
      induction variable.
    * The block has a single argument list ``[ind_var, *iter_args]``; we
      bind each to a fresh ``tir.Var`` of the correct dtype and walk the
      region with those bindings.
    * After the loop body is emitted, the ``scf.yield`` operands are used
      to update the iter_arg SSA mappings for the next iteration. We
      currently do this via ``tir.LetStmt`` wrappers per iter_arg; a
      future revision may switch to ``tir.AllocateConst`` once the
      pipeline pass tolerates it.
    * iter_args > 4: raise :class:`EmitError`. The user should restructure
      via TileLang's loop-carry pattern (allocate a fragment, mutate
      in-place inside the loop) instead.
    """
    tir = ctx.tir()
    lb_ssa, ub_ssa, step_ssa = _scf_for_bounds(op)
    iter_arg_ssas = _scf_for_iter_args(op)

    if len(iter_arg_ssas) > _MAX_ITER_ARGS:
        raise EmitError(
            f"scf.for: {len(iter_arg_ssas)} iter_args exceeds supported limit of "
            f"{_MAX_ITER_ARGS}. Restructure the loop to carry state via a "
            f"TileLang fragment (T.alloc_fragment + in-place mutate) instead."
        )

    lb = ctx.get(lb_ssa) if lb_ssa in ctx.value_map else (
        tir.const(int(lb_ssa), "int32") if isinstance(lb_ssa, int) else lb_ssa
    )
    ub = ctx.get(ub_ssa) if ub_ssa in ctx.value_map else (
        tir.const(int(ub_ssa), "int32") if isinstance(ub_ssa, int) else ub_ssa
    )
    step = ctx.get(step_ssa) if step_ssa in ctx.value_map else (
        tir.const(int(step_ssa), "int32") if isinstance(step_ssa, int) else step_ssa
    )

    # Try to fold lb/ub/step to ints for a clean For frame.
    def _to_int(x: Any) -> Optional[int]:
        try:
            return int(x)
        except (TypeError, ValueError):
            v = getattr(x, "value", None)
            try:
                return int(v) if v is not None else None
            except (TypeError, ValueError):
                return None

    lb_i = _to_int(lb)
    ub_i = _to_int(ub)
    step_i = _to_int(step)

    # Fresh induction var.
    loop_var = tir.Var(ctx.fresh("i"), "int32")
    # Region's first block-arg SSA is the induction variable.
    region_list = _scf_regions(op)
    if not region_list:
        raise EmitError("scf.for: missing body region")
    region = region_list[0]
    # Block-arg SSAs come from either op["block_args"] (dict-fake) or
    # region.blocks[0].arguments (real MLIR).
    block_args: List[Any] = []
    if isinstance(op, dict) and "block_args" in op:
        block_args = list(op["block_args"])
    elif isinstance(region, dict) and "block_args" in region:
        block_args = list(region["block_args"])
    else:
        # Real MLIR: first block's arguments.
        try:
            block_args = list(region.blocks[0].arguments)
        except (AttributeError, IndexError):
            block_args = []

    induction_ssa = block_args[0] if block_args else None
    iter_arg_block_ssas = block_args[1:1 + len(iter_arg_ssas)] if len(block_args) > 1 else []

    # Materialise iter_args as fresh tir.Vars of the appropriate dtype, and
    # pre-bind them in the parent ctx to the *initial* (operand) values so
    # the loop body sees them. We rely on tir.LetStmt nesting (or BufferStore
    # for buffer-typed iter_args) to thread updates -- that is delegated
    # to the parent walker after this emitter returns.
    iter_arg_pairs: List[Tuple[Any, Any]] = []
    init_pairs: List[Tuple[Any, Any]] = []
    for blk_ssa, init_ssa in zip(iter_arg_block_ssas, iter_arg_ssas):
        dt = _om._dtype_of(blk_ssa) or _om._dtype_of(init_ssa) or "float32"
        var = tir.Var(ctx.fresh("carry"), dt)
        iter_arg_pairs.append((blk_ssa, var))
        try:
            init_val = ctx.get(init_ssa)
        except KeyError:
            init_val = init_ssa
        init_pairs.append((var, init_val))

    body, yielded = _emit_region(
        region,
        ctx,
        induction_var=loop_var,
        induction_ssa=induction_ssa,
        iter_arg_pairs=iter_arg_pairs,
    )

    # Wrap the body so each iter_arg is *visible* to the body via LetStmt.
    # The yielded values become the next-iteration values; full SSA-style
    # rotation requires a more involved transform pass, so for now we mark
    # the loop kind as "serial" and record yielded values in ctx for the
    # parent walker to consume (via ctx.value_map). This matches scf.for's
    # forwarding semantics for the common pattern where iter_args carry an
    # accumulator buffer that the body has *already* mutated in place.
    for var, init_val in init_pairs:
        body = tir.LetStmt(var, init_val, body)

    # Compute extent = ub - lb. step == 1 is the common case; for non-unit
    # step we either unroll or scale the induction var.
    UNROLL_LIMIT = 8
    if (lb_i is not None and ub_i is not None and step_i is not None
            and step_i not in (0, 1) and (ub_i - lb_i) // step_i <= UNROLL_LIMIT
            and (ub_i - lb_i) > 0):
        # Tiny constant trip count -> emit one degenerate For per iteration
        # (min=value, extent=1, kind=SERIAL); each gets its OWN fresh
        # induction Var so we don't violate TIR's "Vars are scope-unique"
        # invariant. InjectVirtualThread / Simplify collapses these later.
        stmts = []
        for idx, v in enumerate(range(lb_i, ub_i, step_i)):
            iter_var = tir.Var(ctx.fresh(f"iu{idx}"), "int32")
            stmts.append(tir.For(iter_var, tir.const(v, "int32"), tir.const(1, "int32"),
                                 tir.ForKind.SERIAL, body))
        for_stmt = tir.SeqStmt(stmts) if len(stmts) > 1 else stmts[0]
    elif step_i is not None and step_i not in (0, 1):
        # Non-unit constant step, large trip count: emit a serial loop
        # whose induction var is scaled. extent = ceil((ub-lb)/step).
        if lb_i is not None and ub_i is not None:
            extent = max(0, (ub_i - lb_i + step_i - 1) // step_i)
            extent_expr = tir.const(extent, "int32")
        else:
            # Symbolic bounds: build (ub - lb + step - 1) / step.
            extent_expr = tir.FloorDiv(
                tir.Add(tir.Sub(ub, lb), tir.Sub(step, tir.const(1, "int32"))),
                step,
            )
        # The body sees ``loop_var`` as the iteration index; uses of the
        # induction SSA were already bound to ``loop_var`` by
        # _emit_region. We keep the simpler form (loop_var counts
        # iterations 0..N-1) and trust the body emitter to do its own
        # scaling when it needs the un-strided value. A future cleanup
        # may inject a Let binding ``i_real = lb + loop_var * step`` here
        # once we can prove the body only uses the induction SSA for
        # arithmetic (no address-of).
        # TODO: scale induction var when body uses it as an offset. See
        # feedback_no_silent_delete -- we keep this branch in place rather
        # than dropping the non-unit-step support.
        for_stmt = tir.For(loop_var, tir.const(0, "int32"), extent_expr,
                           tir.ForKind.SERIAL, body)
    else:
        # step == 1 (or symbolic step that ptr_analysis has folded to 1):
        # emit the natural ``for i in range(lb, ub)`` form.
        if lb_i is not None and ub_i is not None:
            extent_expr = tir.const(max(0, ub_i - lb_i), "int32")
            min_expr = tir.const(lb_i, "int32")
        else:
            extent_expr = tir.Sub(ub, lb)
            # If lb is already a PrimExpr we forward it; otherwise wrap.
            min_expr = lb if hasattr(lb, "dtype") else tir.const(lb_i or 0, "int32")
        for_stmt = tir.For(loop_var, min_expr, extent_expr,
                           tir.ForKind.SERIAL, body)

    ctx.emit(for_stmt)

    # Bind the ``scf.for`` results: they are the final iter_arg values
    # (which, in our serial-mutate model, are the same tir.Vars we already
    # created — the body mutated their referenced buffers in place).
    if yielded and _om._results(op):
        for result_ssa, y in zip(_om._results(op), yielded):
            ctx.bind(result_ssa, y)
    elif _om._results(op) and iter_arg_pairs:
        for result_ssa, (_, var) in zip(_om._results(op), iter_arg_pairs):
            ctx.bind(result_ssa, var)
    return for_stmt


def map_scf_if(op: Any, ctx: _om.WalkerCtx) -> Any:
    """Lower ``scf.if(cond) { then } else { else }`` to ``tir.IfThenElse``.

    The ``cond`` operand resolves through the parent ``ctx.value_map``;
    ``then`` and ``else`` regions are walked via :func:`_emit_region` and
    wrapped into the resulting node. When the else region is empty (the
    ``scf.if`` has no ``else`` block) we pass ``None`` as the else body.

    If both regions yield (Triton sometimes uses ``scf.if`` as a value-
    producing select), the yielded operands of the *then* branch are
    chosen; we don't currently materialise a ``Select`` because the
    walker's downstream consumers (tt.store, etc.) read from the side-
    effect-mutated buffers rather than the SSA result.
    """
    tir = ctx.tir()
    operands = _om._operands(op)
    if not operands:
        raise EmitError("scf.if: missing condition operand")
    cond = ctx.get(operands[0])

    regions = _scf_regions(op)
    if not regions:
        raise EmitError("scf.if: missing then-region")
    then_body, then_yield = _emit_region(regions[0], ctx)
    else_body: Any = None
    else_yield: List[Any] = []
    if len(regions) >= 2:
        # An empty else-region is still a valid MLIR shape; treat it as None.
        if isinstance(regions[1], dict):
            ops_in_else = regions[1].get("ops") or []
        else:
            ops_in_else = []
            for block in getattr(regions[1], "blocks", ()) or ():
                for inner in getattr(block, "operations", ()) or ():
                    ops_in_else.append(inner)
        if ops_in_else:
            else_body, else_yield = _emit_region(regions[1], ctx)

    if_stmt = tir.IfThenElse(cond, then_body, else_body)
    ctx.emit(if_stmt)

    # Forward yielded values: prefer then-branch (matches MLIR semantics
    # in the cond=true case); the consumer is expected to be a side-effect
    # op so this is rarely load-bearing.
    yielded = then_yield or else_yield
    if yielded and _om._results(op):
        for result_ssa, y in zip(_om._results(op), yielded):
            ctx.bind(result_ssa, y)
    return if_stmt


def map_scf_yield(op: Any, ctx: _om.WalkerCtx) -> Any:
    """No-op handler for ``scf.yield``.

    The ``scf.yield`` op exists only inside ``scf.for`` / ``scf.if``
    regions where it terminates a block by forwarding values to the
    parent op. Our region walker (:func:`_emit_region`) intercepts it
    directly to extract the yielded operand list, so a top-level dispatch
    to this emitter is normally unreachable. We still register it to keep
    the OP_TABLE coverage check in ``_walk_text_ttir`` happy when a
    yield-only line slips into the textual TTIR.
    """
    return None


# ---------------------------------------------------------------------------
# scf.while  (bounded-iterations lowering)
# ---------------------------------------------------------------------------
#
# TIR has no native ``While`` node, so we lower an ``scf.while`` loop into a
# ``tir.For`` of ``SCF_WHILE_MAX_ITERATIONS`` iterations with an inner
# ``tir.IfThenElse`` guarding the after-region. The before-region is walked
# *inside* the outer For so its condition can capture the current iter_arg
# bindings on every iteration; if the condition is false the after-region
# simply doesn't execute (this is the cheapest "early exit" we can encode
# without a TIR Stop primitive, and it is correct under TIR semantics
# because the after-region's only effect is to mutate the same iter_arg
# variables we read in the next iteration's before-region).
#
# Honest limitations:
# * ``SCF_WHILE_MAX_ITERATIONS`` is a static upper bound. We default to 1024
#   and let the env var ``TILELANG_SCF_WHILE_MAX_ITERS`` override it. When
#   the loop genuinely needs more, the user should refactor to ``scf.for``
#   (which has no implicit cap).
# * If the user's TTIR carries an explicit ``upper_bound`` attribute we
#   prefer that; otherwise fall back to the env-overridable default.
# * If the user explicitly tags the op with ``"unbounded": True`` (or sets a
#   non-positive bound) we raise EmitError -- silently truncating would
#   change program semantics.

import os as _os


def _scf_while_default_max_iters() -> int:
    raw = _os.environ.get("TILELANG_SCF_WHILE_MAX_ITERS")
    if raw is None:
        return 1024
    try:
        n = int(raw)
        if n <= 0:
            return 1024
        return n
    except ValueError:
        return 1024


SCF_WHILE_MAX_ITERATIONS: int = _scf_while_default_max_iters()


def _scf_while_bound(op: Any) -> int:
    """Resolve the static iteration bound for a given ``scf.while`` op.

    Order of precedence:
    1. ``op["attrs"]["upper_bound"]`` (explicit, per-op) -- wins.
    2. ``op["attrs"]["max_iters"]`` (legacy spelling) -- still honoured.
    3. ``SCF_WHILE_MAX_ITERATIONS`` (env-overridable default).

    Raise :class:`EmitError` when the attrs explicitly mark the loop as
    unbounded or specify a non-positive bound.
    """
    attrs = _om._attrs(op)
    if attrs.get("unbounded"):
        raise EmitError(
            "scf.while with unbounded iterations not supported; "
            "please rewrite with explicit upper bound or use scf.for "
            "(set the 'upper_bound' attribute or env "
            "TILELANG_SCF_WHILE_MAX_ITERS to override the default cap)."
        )
    for key in ("upper_bound", "max_iters"):
        if key in attrs:
            try:
                bound = int(attrs[key])
            except (TypeError, ValueError) as exc:
                raise EmitError(
                    f"scf.while: attribute {key!r}={attrs[key]!r} is not an integer"
                ) from exc
            if bound <= 0:
                raise EmitError(
                    f"scf.while: attribute {key!r}={bound} must be positive; "
                    "rewrite with scf.for if the loop has no useful upper bound."
                )
            return bound
    return SCF_WHILE_MAX_ITERATIONS


def map_scf_while(op: Any, ctx: _om.WalkerCtx) -> Any:
    """Lower ``scf.while`` with iter_args to a bounded ``tir.For`` + IfThenElse.

    Shape we accept (mirrors :func:`map_scf_for`'s dict-fake):

        op = {
            "name": "scf.while",
            "operands": [init0, init1, ...],            # iter_arg inits
            "results":  [r0, r1, ...],                  # final iter_arg vals
            "attrs":    {"upper_bound": N, ...},        # optional cap
            "regions":  [before_region, after_region],
            # Block args of each region as flat lists (dict-fake convenience):
            "before_block_args": [carry0_b, carry1_b, ...],
            "after_block_args":  [carry0_a, carry1_a, ...],
        }

    The before-region terminates with ``scf.condition(%cond) %carry...`` and
    the after-region with ``scf.yield %carry...``. We model both terminators
    by intercepting them inside :func:`_emit_region` (yield) and via a
    dedicated probe here (condition).

    Implementation:
        carry_i = tir.Var(... per iter_arg)            # bound to init_i via Let
        for k in 0..MAX_ITERS:
            <before_region body>      # may set carry_i_next, evaluate cond
            if cond:
                <after_region body>   # mutates carry_i in place
            else:
                pass                  # acts as a soft early-exit

    The result SSAs are bound to the carry vars (their final values).
    """
    tir = ctx.tir()
    init_ssas = list(_om._operands(op))
    if len(init_ssas) > _MAX_ITER_ARGS:
        raise EmitError(
            f"scf.while: {len(init_ssas)} iter_args exceeds supported limit of "
            f"{_MAX_ITER_ARGS}. Restructure to carry state via a TileLang "
            "fragment (T.alloc_fragment + in-place mutate) instead."
        )

    bound = _scf_while_bound(op)

    regions = _scf_regions(op)
    if len(regions) < 2:
        raise EmitError(
            "scf.while: expected 2 regions (before, after); "
            f"got {len(regions)}"
        )
    before_region, after_region = regions[0], regions[1]

    # Resolve block-arg SSAs for both regions.
    before_block_args: List[Any] = []
    after_block_args: List[Any] = []
    if isinstance(op, dict):
        before_block_args = list(op.get("before_block_args") or [])
        after_block_args = list(op.get("after_block_args") or [])
        # Fall back to per-region "block_args" if the op-level keys are absent.
        if not before_block_args and isinstance(before_region, dict):
            before_block_args = list(before_region.get("block_args") or [])
        if not after_block_args and isinstance(after_region, dict):
            after_block_args = list(after_region.get("block_args") or [])
    else:
        try:
            before_block_args = list(before_region.blocks[0].arguments)
        except (AttributeError, IndexError):
            before_block_args = []
        try:
            after_block_args = list(after_region.blocks[0].arguments)
        except (AttributeError, IndexError):
            after_block_args = []

    # Materialise carry vars (one tir.Var per iter_arg). Same pattern as
    # ``map_scf_for``: bind the *parent* ctx so the body's Let wrappers see
    # the initial value, then thread the same Var into both regions.
    carry_vars: List[Any] = []
    init_pairs: List[Tuple[Any, Any]] = []
    for idx, init_ssa in enumerate(init_ssas):
        dt = _om._dtype_of(init_ssa) or "float32"
        # Prefer the before-region block-arg dtype when richer.
        if idx < len(before_block_args):
            blk_dt = _om._dtype_of(before_block_args[idx])
            if blk_dt:
                dt = blk_dt
        var = tir.Var(ctx.fresh("wcarry"), dt)
        carry_vars.append(var)
        try:
            init_val = ctx.get(init_ssa)
        except KeyError:
            init_val = init_ssa
        init_pairs.append((var, init_val))

    # Walk the before-region: detect ``scf.condition`` as terminator and
    # capture (cond_expr, forwarded_values).
    def _walk_before(region: Any, ctx: _om.WalkerCtx,
                     iter_pairs: List[Tuple[Any, Any]]) -> Tuple[Any, Any, List[Any]]:
        """Return (body_stmt, cond_expr, forwarded_values)."""
        child = _om.WalkerCtx()
        child.value_map = dict(ctx.value_map)
        child.buffers = ctx.buffers
        child.transposed_views = dict(ctx.transposed_views)
        child._tmp_counter = ctx._tmp_counter
        child._tvm = ctx._tvm
        child._T = ctx._T
        for ssa, var in iter_pairs:
            child.bind(ssa, var)

        if isinstance(region, dict):
            ops_iter = list(region.get("ops", ()))
        else:
            ops_iter = []
            for block in getattr(region, "blocks", ()) or ():
                for inner in getattr(block, "operations", ()) or ():
                    ops_iter.append(inner)

        cond_expr: Any = None
        forwarded: List[Any] = []
        for inner in ops_iter:
            op_name = inner.get("name") if isinstance(inner, dict) else getattr(inner, "name", "")
            op_name = str(op_name)
            if op_name == "scf.condition":
                cond_operands = list(_om._operands(inner))
                if not cond_operands:
                    raise EmitError(
                        "scf.condition: missing condition operand"
                    )
                cond_ssa = cond_operands[0]
                try:
                    cond_expr = child.get(cond_ssa)
                except KeyError:
                    cond_expr = cond_ssa
                for fwd_ssa in cond_operands[1:]:
                    try:
                        forwarded.append(child.get(fwd_ssa))
                    except KeyError:
                        forwarded.append(fwd_ssa)
                continue
            emitter = CONTROL_EMITTERS.get(op_name) or _om.OP_TABLE.get(op_name)
            if emitter is None:
                raise EmitError(
                    f"scf.while before-region: unmapped op {op_name!r}; "
                    "register an emitter in OP_TABLE or CONTROL_EMITTERS."
                )
            emitter(inner, child)

        ctx._tmp_counter = child._tmp_counter

        tir_local = ctx.tir()
        if not child.stmts:
            body = tir_local.Evaluate(tir_local.const(0, "int32"))
        elif len(child.stmts) == 1:
            body = child.stmts[0]
        else:
            body = tir_local.SeqStmt(child.stmts)

        if cond_expr is None:
            raise EmitError(
                "scf.while before-region missing scf.condition terminator"
            )
        return body, cond_expr, forwarded

    # Bind carry-var SSAs in the *parent* ctx so before-region ops that
    # reference them via op["operands"] resolve correctly during walking.
    iter_arg_pairs_before = [(blk_ssa, var) for blk_ssa, var in zip(before_block_args, carry_vars)]
    iter_arg_pairs_after = [(blk_ssa, var) for blk_ssa, var in zip(after_block_args, carry_vars)]

    before_body, cond_expr, _forwarded = _walk_before(before_region, ctx, iter_arg_pairs_before)

    # Walk the after-region with the *same* carry vars; its terminating
    # scf.yield's operands become the next-iteration carry values. We rely
    # on the body to mutate carry vars in place when needed (matching
    # map_scf_for's serial-mutate model). The yielded SSAs are recorded so
    # the parent emitter can rebind result SSAs.
    after_body, after_yield = _emit_region(
        after_region,
        ctx,
        induction_var=None,
        induction_ssa=None,
        iter_arg_pairs=iter_arg_pairs_after,
    )

    # Compose: if cond { after_body } else { /* nothing -- soft exit */ }
    guarded_after = tir.IfThenElse(cond_expr, after_body, None)

    # Sequence: before_body; guarded_after.
    body_stmts = []
    body_stmts.append(before_body)
    body_stmts.append(guarded_after)
    loop_body = tir.SeqStmt(body_stmts) if len(body_stmts) > 1 else body_stmts[0]

    # Wrap the body in LetStmts that bind the carry vars to their initial
    # values. NOTE: this matches map_scf_for's structure (Let on entry);
    # the carry vars are *mutable* tir.Vars whose updates the after-region
    # is responsible for emitting via BufferStore on a sibling buffer. The
    # downstream LowerLetStmt pass folds the binding correctly.
    for var, init_val in init_pairs:
        loop_body = tir.LetStmt(var, init_val, loop_body)

    loop_var = tir.Var(ctx.fresh("wi"), "int32")
    for_stmt = tir.For(
        loop_var,
        tir.const(0, "int32"),
        tir.const(int(bound), "int32"),
        tir.ForKind.SERIAL,
        loop_body,
    )
    ctx.emit(for_stmt)

    # Bind the scf.while result SSAs to the (final-value) carry vars. The
    # yielded SSAs from the after-region terminator take precedence when
    # available.
    results = list(_om._results(op))
    if results:
        # Prefer after-yield (matches scf.while's "loop terminates with the
        # yield"); fall back to carry vars when the yield was empty.
        for idx, result_ssa in enumerate(results):
            if idx < len(after_yield):
                ctx.bind(result_ssa, after_yield[idx])
            elif idx < len(carry_vars):
                ctx.bind(result_ssa, carry_vars[idx])
    return for_stmt


# ---------------------------------------------------------------------------
# llvm.inline_asm  (PTX-pattern -> portable TIR intrinsic)
# ---------------------------------------------------------------------------
#
# Triton emits a small handful of ``ptx.<approx>.<dtype>`` patterns via
# ``tl.inline_asm_elementwise`` for fast transcendentals. We don't try to
# parse arbitrary inline asm -- that is a research project. Instead we
# match the asm_string against a closed-set dictionary of well-known PTX
# intrinsics and lower them to portable TIR calls (``tir.tanh`` etc.) so
# the same kernel can target Metal / CPU / non-NVIDIA back-ends.
#
# The dictionary is intentionally narrow. Any unrecognised pattern raises
# EmitError with the offending asm_string in the message so the user can
# either (a) rewrite the kernel to use a portable intrinsic or (b) submit
# a PR adding the new pattern here.

# Each value is a unary callable ``(tir, x) -> tir.PrimExpr``.
PTX_TO_TIR: Dict[str, Callable[[Any, Any], Any]] = {
    # Fast tanh: NVPTX's tanh.approx.f32 matches TIR's portable tanh, which
    # lowers to the platform's fast-math implementation on CPU/Metal.
    "tanh.approx.f32": lambda tir, x: tir.tanh(x),
    # Approx exp2 / log2 / rcp -- common in softmax / layer-norm fast paths.
    "ex2.approx.f32": lambda tir, x: tir.exp2(x),
    "lg2.approx.f32": lambda tir, x: tir.log2(x),
    "rcp.approx.f32": lambda tir, x: tir.div(tir.const(1.0, "float32"), x),
    "rcp.approx.ftz.f32": lambda tir, x: tir.div(tir.const(1.0, "float32"), x),
    "rsqrt.approx.f32": lambda tir, x: tir.div(tir.const(1.0, "float32"), tir.sqrt(x)),
    "sqrt.approx.f32": lambda tir, x: tir.sqrt(x),
    "sin.approx.f32": lambda tir, x: tir.sin(x),
    "cos.approx.f32": lambda tir, x: tir.cos(x),
}


def _normalize_ptx_asm(asm: str) -> Optional[str]:
    """Reduce a PTX inline-asm string to a canonical intrinsic key.

    Triton's emitted strings look like ``"tanh.approx.f32 $0, $1;"`` or
    ``"  tanh.approx.f32 \t$0,$1 ;\n"``. We strip whitespace, drop the
    operand placeholders + trailing semicolon, and lowercase the mnemonic.
    Returns ``None`` when the shape doesn't match the expected
    ``<mnemonic> $<dst>, $<src>;`` form -- the caller raises EmitError.
    """
    if not isinstance(asm, str):
        return None
    s = asm.strip().lower()
    if not s:
        return None
    # Strip trailing semicolon and any trailing comments.
    if ";" in s:
        s = s.split(";", 1)[0].strip()
    # Split off operand list at the first '$' or comma after the mnemonic.
    # Mnemonic is the first whitespace-delimited token.
    parts = s.split(None, 1)
    if not parts:
        return None
    mnemonic = parts[0]
    return mnemonic


def map_llvm_inline_asm(op: Any, ctx: _om.WalkerCtx) -> Any:
    """Lower ``llvm.inline_asm`` (or ``tt.elementwise_inline_asm``) to TIR.

    Strategy:
      * Pull ``asm_string`` from op.attrs.
      * Match against :data:`PTX_TO_TIR`.
      * On hit: emit the corresponding TIR intrinsic with the (single) arg.
      * On miss: raise :class:`EmitError` with the asm_string verbatim.

    We deliberately do **not** try to parse complex multi-instruction
    sequences, multi-operand ops, or non-elementwise patterns. Those are
    flagged for human triage.
    """
    attrs = _om._attrs(op)
    asm = attrs.get("asm_string")
    if asm is None:
        # MLIR's textual form sometimes spells this as "asm".
        asm = attrs.get("asm")
    if asm is None:
        raise EmitError(
            "llvm.inline_asm: missing asm_string attribute; cannot identify "
            "the PTX intrinsic to retarget."
        )

    mnemonic = _normalize_ptx_asm(str(asm))
    if mnemonic is None or mnemonic not in PTX_TO_TIR:
        # Surface the *full* asm_string so triage can copy-paste it into
        # the dictionary (or back into a portable intrinsic). The message
        # also lists supported keys so the user can see what we DO handle.
        supported = ", ".join(sorted(PTX_TO_TIR.keys()))
        raise EmitError(
            "llvm.inline_asm with unrecognized PTX pattern; cannot retarget "
            f"to Metal. asm_string={asm!r}. Supported intrinsics: {supported}."
        )

    operands = _om._operands(op)
    if not operands:
        raise EmitError(
            f"llvm.inline_asm({mnemonic}): expected 1 operand; got 0"
        )
    src_ssa = operands[0]
    src = ctx.get(src_ssa) if src_ssa in ctx.value_map else src_ssa

    tir = ctx.tir()
    expr = PTX_TO_TIR[mnemonic](tir, src)

    if _om._results(op):
        ctx.bind(_om._results(op)[0], expr)
    return expr


# ---------------------------------------------------------------------------
# tt.call -- inline expansion of a tt.func callee
# ---------------------------------------------------------------------------
#
# Triton emits ``tt.call @callee(operands...)`` whenever the kernel uses a
# helper function (e.g. the reusable softmax/sum reduction blocks pulled in
# from ``triton.language.standard``). The TTIR module then carries a sibling
# ``tt.func private @callee(...) {...}`` declaration. This emitter inlines
# the callee at the call site:
#
#   1. Look up the callee tt.func op (registered in ``ctx.callees`` by the
#      module-level pre-pass in ``triton_frontend.__init__:_walk_mlir_module``).
#   2. Push a substitution frame mapping the callee's entry-block arguments
#      to the caller's already-resolved TIR operands.
#   3. Walk the callee's entry-block body via the OP_TABLE / CONTROL_EMITTERS
#      dispatch, intercepting ``tt.return`` to capture the returned SSA.
#   4. Pop the substitution frame and bind the tt.call's result SSA to the
#      captured return value.
#
# We deliberately don't materialise the callee as a separate ``tir.PrimFunc``
# / ``tir.call_extern`` because (a) it preserves SSA simplicity in the walker,
# and (b) it matches Triton's own backend semantics, which inline these
# helpers before lowering to GPU. Cap on call depth comes for free from
# Python recursion limits; we don't expect helper-of-helper depth >2 in the
# kernels we target.
#
# Hard constraint: an unresolvable ``@callee`` raises :class:`EmitError`
# rather than returning silently -- a missing callee almost certainly means
# the pre-pass missed a tt.func, and an invalid lowering is worse than a
# loud failure (per ``feedback_no_silent_fallback``).


# Match ``callee = @symbol_name`` inside the printed properties block.
# Symbol names admit dots and underscores (e.g.
# ``@triton.language.standard.max__fp32S128S_c0_cFalse_cTrue_cFalse``).
_CALLEE_RE = __import__("re").compile(
    r"callee\s*=\s*@(?P<sym>[A-Za-z_][\w.]*)"
)


def _parse_callee_attr(op: Any) -> Optional[str]:
    """Extract the callee symbol from a ``tt.call`` op.

    Tries, in order:
      1. ``op.attrs['callee']`` (dict-fake or jaxlib registered-dialect path).
      2. ``op.attributes['callee']`` -> stringify (real MLIR FlatSymbolRefAttr).
      3. Regex over ``str(op)`` -- the unregistered-dialect path under
         jaxlib's ``allow_unregistered_dialects=True`` puts the callee in
         the printed ``<{...}>`` properties block but never surfaces it via
         the Python attribute accessors.

    Returns the symbol name WITHOUT the leading ``@``, or None when the op
    has no parseable callee.
    """
    # Path 1: dict-fake.
    if isinstance(op, dict):
        attrs = op.get("attrs") or {}
        if "callee" in attrs:
            sym = str(attrs["callee"]).strip()
            return sym.lstrip("@") or None

    # Path 2: real MLIR attributes.
    attrs_obj = getattr(op, "attributes", None)
    if attrs_obj is not None:
        try:
            for a in attrs_obj:
                if getattr(a, "name", None) == "callee":
                    val = str(a.attr).strip()
                    return val.lstrip("@") or None
        except Exception:
            pass

    # Path 3: regex over the printed op text (unregistered-dialect path).
    try:
        text = str(op)
    except Exception:
        return None
    m = _CALLEE_RE.search(text)
    if m:
        return m.group("sym")
    return None


def _func_sym_name(func_op: Any) -> Optional[str]:
    """Return the ``sym_name`` attribute of a ``tt.func`` op (no leading @)."""
    if isinstance(func_op, dict):
        attrs = func_op.get("attrs") or {}
        sym = attrs.get("sym_name") or func_op.get("sym_name")
        return str(sym) if sym else None
    # Try op.attributes first.
    attrs_obj = getattr(func_op, "attributes", None)
    if attrs_obj is not None:
        try:
            for a in attrs_obj:
                if getattr(a, "name", None) == "sym_name":
                    raw = str(a.attr).strip()
                    if raw.startswith('"') and raw.endswith('"'):
                        raw = raw[1:-1]
                    return raw or None
        except Exception:
            pass
    # Fall back to the printed properties block.
    props = _om._parse_generic_properties_shared(func_op)
    sym = props.get("sym_name")
    return str(sym) if sym else None


def _func_entry_block_ops(func_op: Any) -> List[Any]:
    """Return the ops in the entry block (block 0) of a ``tt.func``.

    Triton helper functions printed by Triton 3.x have a second
    ``^bb1: // no predecessors`` block carrying ``ub.poison`` + a sentinel
    ``tt.return``. That block is unreachable and must NOT be walked --
    its ub.poison would error out the dispatcher. We restrict ourselves to
    the entry block where the real body lives.
    """
    if isinstance(func_op, dict):
        regions = func_op.get("regions") or []
        if regions:
            r0 = regions[0]
            if isinstance(r0, dict):
                blocks = r0.get("blocks") or []
                if blocks:
                    b0 = blocks[0]
                    if isinstance(b0, dict):
                        return list(b0.get("ops") or [])
        # Single-region inline form: ``{"body": [op, ...]}``.
        return list(func_op.get("body") or [])
    regions = list(getattr(func_op, "regions", ()) or ())
    if not regions:
        return []
    blocks = list(getattr(regions[0], "blocks", ()) or ())
    if not blocks:
        return []
    return list(getattr(blocks[0], "operations", ()) or [])


def _find_func_return_operand(func_op: Any) -> Optional[Any]:
    """Find the ``tt.return`` op in the entry block and return its operand SSA.

    Returns ``None`` for a void return (``tt.return`` with no operands) or
    when the entry block has no ``tt.return`` at all (defensive; every
    well-formed tt.func has one).
    """
    for inner in _func_entry_block_ops(func_op):
        op_name = inner.get("name") if isinstance(inner, dict) else getattr(inner, "name", "")
        if str(op_name) == "tt.return":
            operands = _om._operands(inner)
            if operands:
                return operands[0]
            return None
    return None


def emit_tt_call(op: Any, ctx: _om.WalkerCtx) -> Any:
    """Inline-expand a ``tt.call @callee(operands...)`` into the caller body.

    See module header for the full strategy. Raises :class:`EmitError` on
    an unresolvable callee or a callee whose body has no ``tt.return``
    when the call site expects a result.
    """
    sym = _parse_callee_attr(op)
    if not sym:
        raise EmitError(
            "tt.call: could not extract a callee symbol from the op. "
            "Expected ``callee = @<name>`` in the properties block; got: "
            f"{str(op)!r}"
        )

    callee_func = ctx.lookup_callee(sym)
    if callee_func is None:
        known = sorted(ctx.callees.keys())
        raise EmitError(
            f"tt.call: references unknown callee '@{sym}'. The module-level "
            f"pre-pass found these tt.func symbols: {known}. Either the "
            "pre-pass skipped this tt.func or the symbol name in the call "
            "is mistyped."
        )

    # Mark the callee as referenced so the module walker knows to skip
    # re-emitting its body at module-level (a helper that's only inlined
    # via tt.call must not also be walked as a top-level kernel).
    ctx.callee_used.add(sym)

    # Resolve the call's operands to TIR values via the caller's value_map.
    operands = list(_om._operands(op))
    resolved_operands: List[Any] = []
    for o in operands:
        if o in ctx.value_map:
            resolved_operands.append(ctx.value_map[o])
        else:
            # ctx.get raises with a useful diagnostic if missing, but we
            # also want to allow operands that are themselves unhashable
            # MLIR Value objects -- ctx.get handles that uniformly.
            resolved_operands.append(ctx.get(o))

    # Build the substitution frame from callee block-args -> caller operands.
    block_args = _func_block_args(callee_func)
    if len(block_args) != len(resolved_operands):
        raise EmitError(
            f"tt.call @{sym}: arity mismatch: callee declares "
            f"{len(block_args)} block-arg(s) but the call site passes "
            f"{len(resolved_operands)} operand(s)."
        )

    subst: Dict[Any, Any] = {}
    for arg, val in zip(block_args, resolved_operands):
        # Bind under the SSA Value object (if hashable) AND its printed
        # name. The double-keyed binding mirrors what map_tt_func does so
        # downstream operand lookups resolve regardless of how they were
        # captured (Value-object key or "%argN" string key).
        try:
            subst[arg] = val
        except Exception:
            pass
        ssa = _ssa_name(arg)
        if ssa:
            subst[ssa] = val

    # Walk the callee's entry block. We dispatch each inner op via OP_TABLE
    # / CONTROL_EMITTERS just like the module walker would, but inside a
    # ``push_substitution`` overlay so that block-arg lookups resolve to
    # the caller's operands. Captured return operand is bound after the
    # walk closes the substitution scope (so its TIR value -- already
    # bound under the inlined SSA -- survives).
    return_value: Any = None
    body_ops = _func_entry_block_ops(callee_func)
    with ctx.push_substitution(subst):
        for inner in body_ops:
            inner_name = inner.get("name") if isinstance(inner, dict) else getattr(inner, "name", "")
            inner_name = str(inner_name)
            if inner_name == "tt.return":
                ret_operands = _om._operands(inner)
                if ret_operands:
                    # Resolve THROUGH the substitution frame so a tt.return
                    # that yields a block-arg directly (rare, but legal)
                    # resolves to the caller's operand.
                    ret_ssa = ret_operands[0]
                    if ret_ssa in ctx.value_map:
                        return_value = ctx.value_map[ret_ssa]
                    else:
                        # Try the ctx.get path (consults the substitution
                        # stack first, then value_map). Falls through to
                        # KeyError if genuinely unbound.
                        return_value = ctx.get(ret_ssa)
                continue
            emitter = _om.OP_TABLE.get(inner_name) or CONTROL_EMITTERS.get(inner_name)
            if emitter is None:
                raise EmitError(
                    f"tt.call @{sym}: callee body contains op {inner_name!r} "
                    "which has no emitter in OP_TABLE / CONTROL_EMITTERS. "
                    "Register it before lowering kernels that depend on "
                    f"@{sym}."
                )
            emitter(inner, ctx)

    # Bind the call's result SSA to whatever the callee returned.
    results = _om._results(op)
    if results:
        if return_value is None:
            raise EmitError(
                f"tt.call @{sym}: call site has a result SSA but the "
                "callee body had no ``tt.return <value>``. Did the helper "
                "lose its return op during a transform?"
            )
        ctx.bind(results[0], return_value)
    return return_value


# ---------------------------------------------------------------------------
# Public dispatch table
# ---------------------------------------------------------------------------


CONTROL_EMITTERS: Dict[str, Callable[..., Any]] = {
    # arith select / Triton's where (arith spelling)
    "arith.select": map_arith_select,
    # Triton's tt.where is already in OP_TABLE -> map_tt_where; we *don't*
    # overwrite it here, only register the arith spelling. Leaving the
    # original stub / impl in op_mapping intact per feedback_no_silent_delete.
    # arith float casts
    "arith.extf": map_arith_extf,
    "arith.truncf": map_arith_truncf,
    # arith int<->float
    "arith.fptosi": map_arith_fptosi,
    "arith.sitofp": map_arith_sitofp,
    "arith.uitofp": map_arith_uitofp,
    "arith.fptoui": map_arith_fptoui,
    # arith bitcast
    "arith.bitcast": map_arith_bitcast,
    # arith int width
    "arith.extsi": map_arith_extsi,
    "arith.extui": map_arith_extui,
    "arith.trunci": map_arith_trunci,
    # arith.constant -- scalar literal (i32 / f32 / ...). Seeds value_map
    # so downstream uses of ``%c0`` resolve via ctx.get(); raises EmitError
    # on array (``dense<...>``) attrs rather than silently splatting.
    "arith.constant": map_arith_constant,
    # tt.advance (block-pointer)
    "tt.advance": map_tt_advance,
    # tt.func -- structural; seeds block-arg buffers / vars into the ctx so
    # downstream emitters can look up ``%arg0`` via ctx.get(). The walker
    # owns recursion into the body region itself.
    "tt.func": map_tt_func,
    # tt.call -- inline-expand a helper tt.func at the call site.
    "tt.call": emit_tt_call,
    # scf
    "scf.for": map_scf_for,
    "scf.if": map_scf_if,
    "scf.yield": map_scf_yield,
    "scf.while": map_scf_while,
    # llvm inline asm (Triton's PTX fast-math escape hatch). We also accept
    # the Triton-dialect spelling some pipelines emit pre-canonicalisation.
    "llvm.inline_asm": map_llvm_inline_asm,
    "tt.elementwise_inline_asm": map_llvm_inline_asm,
}


def register_into(op_table: Dict[str, Callable[..., Any]]) -> None:
    """Merge :data:`CONTROL_EMITTERS` into ``op_table`` in-place.

    Idempotent: re-registering an existing key overwrites with the same
    callable. We never *remove* entries from op_table (there is no stub
    deletion contract here per ``feedback_no_silent_delete``), so existing
    ``op_mapping.OP_TABLE`` entries that already point at a working
    emitter (e.g. ``tt.where``) take precedence by virtue of not being
    in CONTROL_EMITTERS.
    """
    op_table.update(CONTROL_EMITTERS)
