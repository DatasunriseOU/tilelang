"""Static-shape specialization for int32 element-index addressing (LEVER).

WHAT
----
The Triton frontend emits a *symbolic* PrimFunc: every tensor stride / dim /
gridDim is a runtime ``int32`` scalar param, and each flat 1-D buffer carries a
*fresh symbolic ``int64`` numel Var* (bound by MakePackedAPI to the real
DLTensor element count). Because BOTH the strides AND the numels are symbolic,
the bound-aware index legalizer
(``src/transform/config_index_bitwidth.cc`` :: ``IndexLegalizer``) cannot prove
any global byte/element address fits ``int32`` and conservatively promotes EVERY
global address term to ``int64`` (IMAD.WIDE / LEA.HI 2-instruction 64-bit
arithmetic). Native Triton instead indexes global accesses in ``int32``
ELEMENTS (``add.s32`` / IMAD-32) -- safe ONLY because the element count fits
``2^31``.

This module reproduces native's strategy **safely**: for a known concrete shape
(the production flow is a per-shape JIT -- TTIR captured per call, strides passed
as concrete ints) it BAKES the concrete stride/dim/gridDim values + the concrete
buffer element counts into the PrimFunc body and ``buffer_map`` as ``IntImm``
constants. Once concrete, ``analyzer_->const_int_bound`` PROVES every flat index
``< 2^31`` and the EXISTING legalizer keeps ``int32`` -- collapsing the int64
HI-half address arithmetic. NO new narrowing pass, NO C++ change; the
bound-aware legalizer already does the right thing once the bounds are concrete.
``ForceNarrowIndexToInt32`` (the throwing/truncating base narrower) is NOT used.

SAFETY (this is the whole point)
--------------------------------
``int32`` element indexing is bit-exact iff the addressed element count fits a
signed 32-bit int. :func:`int32_index_safe` is the EXACT, conservative guard:
specialize to int32 ONLY when EVERY tensor's flat element count ``< 2^31``.
Beyond that (``>= 2^31`` elements ``== >= 8.59 GB`` f32) the caller MUST keep the
unchanged symbolic ``int64`` kernel, which is always correct (native CRASHES
there with cudaErrorIllegalAddress; we never do). This module performs NO
narrowing of the symbolic kernel and never mutates it -- it only PRODUCES a
second, shape-specialized PrimFunc. RULE #1: the guard either picks int32
(proven safe) or leaves the int64 path untouched; no silent fallback, no clamp,
no precision/shape downgrade.

ABI
---
Specialization is ABI-PRESERVING: all params are retained (the runtime values
the caller passes are EQUAL to the baked constants, so the kernel stays
bit-exact). Only the BODY + ``buffer_map`` shapes are rewritten Var->IntImm.
Because params are kept, the specialized kernel is a drop-in for the same call
site / arg list as the symbolic kernel.

GENERIC / backend-neutral
-------------------------
Nothing here is dstates- or CUDA-specific. The specializer is driven by a
NAME-keyed map of concrete scalar values + a NAME-keyed map of buffer element
counts; it discovers scalar params and flat-buffer numel Vars structurally. The
same int64->int32 collapse happens on any backend whose codegen widens promoted
addresses (CUDA IMAD.WIDE, Metal long arithmetic, ...).
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

# Signed int32 element-count ceiling. int32 ELEMENT indexing is exact iff every
# addressed element index is < 2^31 (and >= -2^31). f32 -> 2^31 elems = 8.59 GB.
INT32_ELEM_LIMIT = 1 << 31  # 2147483648


def int32_index_safe(element_counts: Sequence[int]) -> bool:
    """Return True iff int32 ELEMENT indexing is bit-exact for these tensors.

    ``element_counts`` is the flat element count (``tensor.numel()``) of EVERY
    global tensor the kernel addresses. int32 element indexing is exact iff
    every count fits a signed 32-bit int (``< 2^31``). This is the EXACT,
    conservative dispatch guard: True -> the int32-specialized kernel is
    bit-exact; False -> the caller MUST use the symbolic int64 kernel (which is
    always correct). RULE #1: no approximation -- a single tensor at/over the
    limit forces int64.
    """
    if not element_counts:
        # No tensors to address -> nothing to overflow; int32 trivially safe.
        return True
    for n in element_counts:
        n_int = int(n)
        if n_int < 0:
            raise ValueError(
                "int32_index_safe: negative element count %d (corrupt shape)" % n_int)
        if n_int >= INT32_ELEM_LIMIT:
            return False
    return True


def _tir():
    # Import lazily so the module is importable without a built tvm (the guard
    # above is pure-Python and used by callers that may not have tvm loaded).
    import tvm  # noqa: WPS433
    from tvm import tir  # noqa: WPS433
    from tvm.tir.stmt_functor import substitute  # noqa: WPS433
    from tvm.tir import decl_buffer  # noqa: WPS433
    return tir, substitute, decl_buffer


def _buffer_for_param(pf: Any, param: Any) -> Optional[Any]:
    try:
        return pf.buffer_map.get(param)
    except Exception:
        return None


def specialize_static_shape(
    pf: Any,
    *,
    scalar_values: Mapping[str, int],
    buffer_element_counts: Mapping[str, int],
    strict: bool = True,
) -> Any:
    """Bake concrete scalar params + flat-buffer element counts into ``pf``.

    Parameters
    ----------
    pf:
        The (unlowered) frontend ``PrimFunc`` from :func:`from_ttir`. MUST be
        pre-lowering: specializing a lowered PrimFunc trips the undeclared-2D-
        buffer guard in ``tirx::Specialize`` (the 2D tile views aliasing the
        flat buffer have no declaration point). We avoid ``Specialize`` entirely
        and substitute directly into ``body`` + ``buffer_map`` (no decl guard).
    scalar_values:
        ``{param_name: concrete_int}`` for the scalar (non-buffer, integer)
        params -- strides, dims, gridDim. Every integer scalar param MUST be
        present when ``strict`` (RULE #1: a missing stride would leave a
        symbolic term and silently defeat the bound proof). Names follow the
        frontend convention (``arg5``.., ``gridDim_0``..). Duplicated gridDim
        params (the frontend declares each gridDim axis twice) are all bound to
        the same value by name.
    buffer_element_counts:
        ``{buffer_name: numel}`` for each flat 1-D global buffer (``arg0``..).
        Binds the fresh symbolic ``int64`` numel Var in that buffer's shape to a
        concrete ``IntImm`` so the OOB-guard bound ``idx < numel`` is provable.
    strict:
        When True (default) RAISE if an integer scalar param or a flat-buffer
        numel Var has no concrete value supplied. This is the fail-loud path:
        an un-bound symbolic term defeats the int32 bound proof and would
        silently keep int64 -- exactly the silent-degradation RULE #1 forbids.

    Returns
    -------
    A NEW ``PrimFunc`` with the same params (ABI-preserving) whose body +
    buffer_map carry the concrete constants. The input ``pf`` is NOT mutated.
    """
    tir, substitute, decl_buffer = _tir()

    vmap: Dict[Any, Any] = {}
    missing_scalars = []
    for p in pf.params:
        buf = _buffer_for_param(pf, p)
        if buf is not None:
            continue  # buffer handle param -- not a scalar
        dt = p.dtype
        if not (isinstance(dt, str) and dt.startswith("int")):
            continue  # non-integer scalar (none expected, but be defensive)
        name = p.name
        if name in scalar_values:
            vmap[p] = tir.IntImm(dt, int(scalar_values[name]))
        else:
            missing_scalars.append(name)

    # Bind the fresh symbolic flat-buffer numel Vars.
    missing_numels = []
    for p in pf.params:
        buf = _buffer_for_param(pf, p)
        if buf is None:
            continue
        name = p.name
        # A flat 1-D buffer whose single extent is a free Var is the numel slot.
        if len(buf.shape) == 1 and isinstance(buf.shape[0], tir.Var):
            if name in buffer_element_counts:
                sv = buf.shape[0]
                vmap[sv] = tir.IntImm(sv.dtype, int(buffer_element_counts[name]))
            else:
                missing_numels.append(name)

    if strict and (missing_scalars or missing_numels):
        raise ValueError(
            "specialize_static_shape: missing concrete value(s) -- refusing to "
            "emit a partially-symbolic kernel that would silently keep int64 "
            "addressing (RULE #1). missing scalar params=%r ; missing buffer "
            "numels=%r" % (missing_scalars, missing_numels))

    new_body = substitute(pf.body, vmap)
    new_bmap = {}
    for p in pf.params:
        buf = _buffer_for_param(pf, p)
        if buf is None:
            continue
        new_shape = [
            substitute(s, vmap) if isinstance(s, tir.PrimExpr) else s
            for s in buf.shape
        ]
        new_bmap[p] = decl_buffer(
            new_shape, dtype=buf.dtype, name=buf.name, data=buf.data,
            elem_offset=buf.elem_offset, scope=buf.scope,
            data_alignment=buf.data_alignment, offset_factor=buf.offset_factor,
            buffer_type="auto" if buf.buffer_type == 1 else "default")

    return tir.PrimFunc(
        pf.params, new_body, ret_type=pf.ret_type, buffer_map=new_bmap,
        attrs=pf.attrs)
