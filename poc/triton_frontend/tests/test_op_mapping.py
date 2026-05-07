"""Regression tests for Wave E3: Properties-helper migration in op_mapping.

Wave C/D fixed the inherent-attribute-via-Properties bug for the emitters
that already lived in ``op_emitters/{memory,arith}.py``. Wave E3 finishes
the job by migrating the legacy emitters in ``op_mapping.py`` itself --
``map_tt_atomic_rmw``, ``map_tt_dot``, ``map_tt_reduce``, ``map_tt_trans``,
``map_tt_make_range`` (legacy), and ``map_tt_mbarrier`` -- from the bare
``_attrs(op)`` accessor (which silently returns ``{}`` for property-only
ops on jaxlib's ``allow_unregistered_dialects=True`` MLIR build) to the
``_attrs_with_properties_shared(op)`` helper.

These tests exercise the *attribute-extraction* boundary: each test
constructs a ``_FakeMlirOp`` whose ``op.attributes`` is empty (mirroring
jaxlib's behaviour) and whose ``str(op)`` carries a Triton 3.6-style
``<{key = value : type}>`` Properties block. We assert:

1. The shared helper returns the parsed key/value (the regression).
2. Where it's safe to drive the full emitter end-to-end without a real
   tilelang kernel frame, we do so and verify the side-effects.

The emitter-level smoke is necessarily lightweight for ``map_tt_dot``,
``map_tt_reduce``, and ``map_tt_mbarrier`` because those construct
buffers / barriers and need a TileLang ``KernelLaunchFrame``; we drive
those at the helper level.
"""
from __future__ import annotations

from typing import Any, List

import pytest

tvm = pytest.importorskip("tvm")

from poc.triton_frontend.op_mapping import (  # noqa: E402
    WalkerCtx,
    _atomic_rmw_kind,
    _attrs_with_properties_shared,
    _is_ptr_type,
    _is_tensor_type,
    _normalize_mlir_dtype,
    _parse_tensor_type,
    map_tt_make_range,
    map_tt_trans,
)


# ---------------------------------------------------------------------------
# jaxlib-shape fake (mirrors test_op_emitters_arith.py:_FakeMlirOp)
# ---------------------------------------------------------------------------


# ``_FakeMlirOp`` and ``_HashableSSA`` previously lived inline; they now
# come from the shared fixtures module so all op-emitter tests use the
# same hashing / Properties-shape behaviour.
from ._fixtures import FakeMlirOp as _FakeMlirOp, FakeSSA as _HashableSSA  # noqa: E402


def _ssa(name: str, *, shape=(), dtype: str = "float32") -> _HashableSSA:
    """Hashable SSA stand-in (delegates to :class:`FakeSSA`)."""
    return _HashableSSA(name=name, shape=tuple(shape), dtype=dtype)


# ---------------------------------------------------------------------------
# 1. tt.atomic_rmw -- ``rmw_op`` is a Triton 3.6 inherent (Properties) attr
# ---------------------------------------------------------------------------


def test_atomic_rmw_kind_from_properties_block() -> None:
    """Pre-E3, ``_atomic_rmw_kind`` called bare ``_attrs(op)`` and got ``{}``,
    raising ``missing 'rmw_op' attribute`` on every jaxlib-shape op."""
    printed = (
        '%2 = "tt.atomic_rmw"(%ptr, %val) <{rmw_op = "fadd" : i32}>'
        " : (!tt.ptr<f32>, f32) -> f32"
    )
    op = _FakeMlirOp(
        name="tt.atomic_rmw",
        operands=[_ssa("ptr"), _ssa("val")],
        results=[_ssa("res")],
        printed=printed,
    )
    # Helper returns the value -- the f-prefix gets stripped to ``add``.
    assert _atomic_rmw_kind(op) == "add"


# ---------------------------------------------------------------------------
# 2. tt.dot -- ``transpose_A`` / ``transpose_B`` / ``out_dtype``
# ---------------------------------------------------------------------------


def test_tt_dot_attrs_from_properties_block() -> None:
    """``map_tt_dot`` reads ``transpose_A``/``transpose_B`` -- both are
    Triton inherent attrs on the jaxlib path. Verify the shared helper
    surfaces them."""
    printed = (
        '%c = "tt.dot"(%a, %b) <{transpose_A = true, transpose_B = false}>'
        " : (tensor<16x32xf16>, tensor<32x16xf16>) -> tensor<16x16xf32>"
    )
    op = _FakeMlirOp(
        name="tt.dot",
        operands=[_ssa("a"), _ssa("b")],
        results=[_ssa("c")],
        printed=printed,
    )
    attrs = _attrs_with_properties_shared(op)
    assert attrs.get("transpose_A") is True
    assert attrs.get("transpose_B") is False


# ---------------------------------------------------------------------------
# 3. tt.reduce -- ``axis``
# ---------------------------------------------------------------------------


def test_tt_reduce_axis_from_properties_block() -> None:
    """``map_tt_reduce`` reads ``axis`` -- Triton emits this as a Property
    (``<{axis = N : i32}>``). Pre-E3, the legacy emitter would silently
    default to ``-1`` (last axis) for every reduce on jaxlib hosts.
    """
    printed = (
        '%out = "tt.reduce"(%src) <{axis = 1 : i32}>'
        " ({ ^bb0(%a: f32, %b: f32): tt.reduce.return %a : f32 })"
        " : (tensor<16x32xf32>) -> tensor<16xf32>"
    )
    op = _FakeMlirOp(
        name="tt.reduce",
        operands=[_ssa("src")],
        results=[_ssa("out")],
        printed=printed,
    )
    attrs = _attrs_with_properties_shared(op)
    assert attrs.get("axis") == 1


# ---------------------------------------------------------------------------
# 4. tt.trans -- ``order`` (permutation)
# ---------------------------------------------------------------------------


def test_tt_trans_order_from_properties_block() -> None:
    """``map_tt_trans`` reads ``order``; tuples don't survive the simple
    scalar grammar, but the *presence* of an explicit permutation as an
    inherent attr is what differs from the dict-shape case. We exercise
    the integer-scalar variant (the common matmul-folding path emits
    ``<{order = 1 : i32, ... }>`` per axis when expanded) and, more
    importantly, verify that the emitter no longer crashes on a
    jaxlib-shape op with empty op.attributes -- it falls back to the
    "swap last two axes" default rather than throwing.
    """
    src_ssa = _HashableSSA("src", shape=(16, 32), dtype="float32")
    out_ssa = _HashableSSA("out", shape=(32, 16), dtype="float32")
    ctx = WalkerCtx()
    src_buf = tvm.tir.decl_buffer((16, 32), "float32", name="src")
    ctx.bind(src_ssa, src_buf)

    # No ``<{order = ...}>`` block -- mimics the common matmul-folding
    # case where Triton emits a bare ``tt.trans`` for the last-two-axes
    # swap. The pre-E3 code path would still work here (no Property to
    # read), but post-migration we want to confirm we didn't regress
    # the no-property branch.
    printed = '%out = "tt.trans"(%src) : (tensor<16x32xf32>) -> tensor<32x16xf32>'
    op = _FakeMlirOp(
        name="tt.trans",
        operands=[src_ssa],
        results=[out_ssa],
        printed=printed,
    )
    result = map_tt_trans(op, ctx)
    # tt.trans rebinds the result SSA to the same TIR value as the source.
    assert ctx.value_map[out_ssa] is src_buf
    # And records the (-2, -1) flipped-axis pair for downstream tt.dot.
    assert ctx.transposed_views[out_ssa] == (-2, -1)
    assert result is src_buf


# ---------------------------------------------------------------------------
# 5. tt.make_range (legacy, dead path) -- ``start`` / ``end``
# ---------------------------------------------------------------------------


def test_tt_make_range_legacy_from_properties_block() -> None:
    """The legacy ``map_tt_make_range`` in op_mapping.py is currently
    superseded by ``op_emitters/memory.py:emit_tt_make_range`` via
    ``OP_TABLE.update(MEMORY_EMITTERS)`` at module init. We still keep
    the legacy emitter alive (per ``feedback_no_silent_delete``) and
    migrate it defensively: if the merge order ever shifts, this path
    must NOT silently emit a zero-length Ramp from ``start = end = 0``.
    """
    printed = (
        '%r = "tt.make_range"() <{start = 0 : i32, end = 256 : i32}>'
        " : () -> tensor<256xi32>"
    )
    out_ssa = _HashableSSA("r", shape=(256,), dtype="int32")
    op = _FakeMlirOp(
        name="tt.make_range",
        operands=[],
        results=[out_ssa],
        printed=printed,
    )
    ctx = WalkerCtx()
    ramp = map_tt_make_range(op, ctx)
    # tir.Ramp(start, stride, lanes); lanes=256 is the regression guard --
    # pre-E3, attrs would be empty and `lanes = 0 - 0 = 0` would raise
    # ``invalid range [0, 0)``.
    assert ramp.lanes == 256
    # Bound the result SSA to the Ramp.
    assert ctx.value_map[out_ssa] is ramp


# ---------------------------------------------------------------------------
# 6. tt.barrier_init / mbarrier -- ``count`` / ``parity`` (a.k.a. barrier_kind)
# ---------------------------------------------------------------------------


def test_tt_mbarrier_count_from_properties_block() -> None:
    """``map_tt_mbarrier`` reads ``count`` (init) and ``parity`` (wait).
    Triton emits both as Properties on the jaxlib path. We exercise the
    helper at the attribute-extraction layer because driving the full
    emitter requires a TileLang KernelLaunchFrame for ``T.alloc_barrier``.
    """
    printed = (
        '%bar = "tt.barrier_init"() <{count = 8 : i32}>'
        " : () -> !tt.barrier"
    )
    op = _FakeMlirOp(
        name="tt.barrier_init",
        operands=[],
        results=[_ssa("bar")],
        printed=printed,
    )
    attrs = _attrs_with_properties_shared(op)
    assert attrs.get("count") == 8

    # And the parity-bearing wait variant.
    printed_wait = (
        '"tt.barrier_wait"(%bar) <{parity = 1 : i32}> : (!tt.barrier) -> ()'
    )
    op_wait = _FakeMlirOp(
        name="tt.barrier_wait",
        operands=[_ssa("bar")],
        results=[],
        printed=printed_wait,
    )
    wait_attrs = _attrs_with_properties_shared(op_wait)
    assert wait_attrs.get("parity") == 1


# ---------------------------------------------------------------------------
# Wave C2 follow-up: _normalize_mlir_dtype must unwrap ``!tt.ptr<T>`` so
# matmul-style kernels (whose tt.func block args are pointer-typed) lower
# without raising. Sibling helper _is_ptr_type lets callers distinguish
# pointer-vs-scalar without re-doing the regex.
# ---------------------------------------------------------------------------


def test_normalize_mlir_dtype_unwraps_ptr() -> None:
    """``!tt.ptr<f32>`` collapses to the storage dtype ``float32``.

    This was the matmul lowering blocker: the helper raised
    ``unsupported MLIR dtype: '!tt.ptr<f32>'`` because the alias map only
    knew the bare element spellings. Unwrapping is correct because every
    caller wants the storage dtype (BufferLoad / decl_buffer).
    """
    assert _normalize_mlir_dtype("!tt.ptr<f32>") == "float32"
    assert _normalize_mlir_dtype("!tt.ptr<i32>") == "int32"
    assert _normalize_mlir_dtype("!tt.ptr<bf16>") == "bfloat16"


def test_normalize_mlir_dtype_nested_ptr() -> None:
    """Nested pointers (``!tt.ptr<!tt.ptr<f32>>``) recurse to the leaf."""
    assert _normalize_mlir_dtype("!tt.ptr<!tt.ptr<f32>>") == "float32"


def test_is_ptr_type_detects_ptr() -> None:
    """``!tt.ptr<T>`` is a pointer; bare scalar dtypes are not."""
    assert _is_ptr_type("!tt.ptr<f32>") is True
    assert _is_ptr_type("!tt.ptr<!tt.ptr<f32>>") is True
    assert _is_ptr_type("f32") is False
    assert _is_ptr_type("int32") is False
    assert _is_ptr_type("") is False


def test_normalize_mlir_dtype_still_raises_on_unknown() -> None:
    """Hard constraint: unknown dtypes raise -- no silent float32 default."""
    with pytest.raises(ValueError, match="unsupported MLIR dtype"):
        _normalize_mlir_dtype("totally_made_up")


# ---------------------------------------------------------------------------
# Wave C-followup: tensor-typed block arguments. Kernels like ``layer_norm``
# thread tile-typed values (``tensor<128xf32>``) across function boundaries
# in TTIR. Before this fix, ``_normalize_mlir_dtype`` raised
# ``unsupported MLIR dtype: 'tensor<128xf32>'`` and ``map_tt_func``
# couldn't seed the buffer. ``_parse_tensor_type`` is the new helper that
# preserves the rank info for callers that need to ``decl_buffer`` with
# the actual extents.
# ---------------------------------------------------------------------------


def test_parse_tensor_type_1d() -> None:
    """``tensor<128xf32>`` -> ``([128], "float32")``.

    The 1-D shape is the layer_norm path: each row of the input tile is
    threaded through ``tt.func`` as a ``tensor<128xf32>``.
    """
    shape, dtype = _parse_tensor_type("tensor<128xf32>")
    assert shape == [128]
    assert dtype == "float32"


def test_parse_tensor_type_2d() -> None:
    """``tensor<16x32xf32>`` -> ``([16, 32], "float32")``.

    Higher-rank tiles surface in matmul-style kernels; we preserve the
    extents row-major so ``tir.decl_buffer`` can plug them in directly.
    """
    shape, dtype = _parse_tensor_type("tensor<16x32xf32>")
    assert shape == [16, 32]
    assert dtype == "float32"


def test_parse_tensor_type_unknown_inner_raises() -> None:
    """Hard constraint: unknown element dtype raises -- no silent default.

    ``tensor<128xbogus>`` must fail because the regression we're guarding
    against is silently lowering an unknown element type as ``float32``.
    """
    with pytest.raises(ValueError, match="unsupported MLIR dtype"):
        _parse_tensor_type("tensor<128xbogus>")


def test_normalize_mlir_dtype_unwraps_tensor() -> None:
    """``tensor<NxT>`` collapses to the storage dtype ``T``.

    Mirrors the pointer-unwrap behaviour: every caller of
    :func:`_normalize_mlir_dtype` wants the storage dtype; rank info is
    available via :func:`_parse_tensor_type` for the few callers (e.g.
    ``map_tt_func``) that actually need it.
    """
    assert _normalize_mlir_dtype("tensor<128xf32>") == "float32"
    assert _normalize_mlir_dtype("tensor<16x32xf32>") == "float32"
    assert _normalize_mlir_dtype("tensor<8xi32>") == "int32"


def test_is_tensor_type_detects_tensor() -> None:
    """``tensor<NxT>`` is a tensor; pointer / scalar spellings are not."""
    assert _is_tensor_type("tensor<128xf32>") is True
    assert _is_tensor_type("tensor<16x32xf32>") is True
    assert _is_tensor_type("!tt.ptr<f32>") is False
    assert _is_tensor_type("f32") is False
    assert _is_tensor_type("") is False


# ---------------------------------------------------------------------------
# Wave G4: error-class hierarchy
# ---------------------------------------------------------------------------


def test_emit_error_is_triton_frontend_error_subclass() -> None:
    """Wave G4: ``EmitError`` must subclass :class:`TritonFrontendError`
    so a single ``except TritonFrontendError`` clause in driver code
    catches every deliberate frontend failure.
    """
    from poc.triton_frontend.op_mapping import EmitError, TritonFrontendError

    assert issubclass(EmitError, TritonFrontendError)
    # Sanity: the catch-clause shape advertised in the docstring works.
    try:
        raise EmitError("hierarchy smoke")
    except TritonFrontendError as exc:
        assert "hierarchy smoke" in str(exc)


def test_pipeline_error_is_triton_frontend_error_subclass() -> None:
    """Wave G4: ``PipelineError`` must subclass :class:`TritonFrontendError`
    so it shares an ancestor with :class:`EmitError`.
    """
    from poc.triton_frontend.op_mapping import TritonFrontendError
    from poc.triton_frontend.pipeline import PipelineError

    assert issubclass(PipelineError, TritonFrontendError)
    try:
        raise PipelineError("pipeline hierarchy smoke")
    except TritonFrontendError as exc:
        assert "pipeline hierarchy smoke" in str(exc)


# ---------------------------------------------------------------------------
# H4 Wave-I: OP_TABLE size pin + EmitError migration smoke
# ---------------------------------------------------------------------------


def test_op_table_has_expected_size() -> None:
    """Pin the :data:`OP_TABLE` size to catch accidental drops or
    unintentional adds.

    Update this constant ONLY when you intentionally add or remove an op
    from OP_TABLE (and update the README at the same time -- see
    ``poc/triton_frontend/README.md`` for the three places the count
    appears). H4 Wave-I noted the README claimed 83 while the actual
    table held 84 (G1 added ``tt.call``); this test prevents that drift.
    """
    from poc.triton_frontend.op_mapping import OP_TABLE

    EXPECTED = 84
    assert len(OP_TABLE) == EXPECTED, (
        f"OP_TABLE size changed from {EXPECTED} to {len(OP_TABLE)}; "
        f"if intentional, update this constant + the three README "
        f"occurrences (poc/triton_frontend/README.md). Current keys: "
        f"{sorted(OP_TABLE.keys())}"
    )


def test_op_mapping_emitters_raise_emit_error_not_value_error() -> None:
    """H4 Wave-I: emitter-internal preconditions raise :class:`EmitError`
    (not plain ``ValueError``) so callers can ``except EmitError``
    uniformly.

    We exercise a representative subset (``tt.load`` / ``tt.dot`` /
    ``tt.where`` / ``tt.broadcast``) by feeding each a dict-shaped op with
    missing operands. Each must raise ``EmitError``. ``ValueError`` is
    explicitly rejected (the migration must not have left these as
    ``ValueError`` so generic ``except ValueError`` catches stop swallowing
    real bugs).
    """
    from poc.triton_frontend.op_mapping import (
        EmitError,
        WalkerCtx,
        map_tt_broadcast,
        map_tt_dot,
        map_tt_load,
        map_tt_where,
    )

    ctx = WalkerCtx()

    # tt.load: missing pointer operand
    with pytest.raises(EmitError, match="tt.load: missing pointer operand"):
        map_tt_load({"name": "tt.load", "operands": []}, ctx)

    # tt.dot: needs 2 operands
    with pytest.raises(EmitError, match=r"tt\.dot: expected at least 2"):
        map_tt_dot({"name": "tt.dot", "operands": ["%a"]}, ctx)

    # tt.where: needs 3 operands
    with pytest.raises(EmitError, match=r"tt\.where: expected 3 operands"):
        map_tt_where({"name": "tt.where", "operands": ["%c", "%t"]}, ctx)

    # tt.broadcast: missing source
    with pytest.raises(EmitError, match=r"tt\.broadcast: missing source"):
        map_tt_broadcast({"name": "tt.broadcast", "operands": []}, ctx)


# ---------------------------------------------------------------------------
# H4-followup: dead-but-loaded legacy stub markup audit
# ---------------------------------------------------------------------------
#
# A handful of legacy ``map_tt_*`` emitters in op_mapping.py are superseded
# at module-init time by the per-family overlay dicts
# (``op_emitters/{arith,memory,reduction,control}.py``) via
# ``OP_TABLE.update(<EMITTERS>)``. Per ``feedback_no_silent_delete`` we keep
# the legacy implementations in tree -- but we MUST mark them with a
# uniform comment so future readers know which entries are dead vs.
# canonical. This test enforces that markup discipline.


def test_map_tt_dead_stubs_are_marked() -> None:
    """Every legacy ``map_tt_*`` superseded by an overlay must carry the
    ``DEAD-BUT-LOADED:`` marker comment immediately above its ``def``.

    The audit list below was assembled by cross-referencing each
    ``map_tt_*`` definition in op_mapping.py against the four overlay
    EMITTERS dicts (ARITH / MEMORY / REDUCTION / CONTROL). Live emitters
    (``map_tt_atomic_rmw``, ``map_tt_where``, ``map_tt_trans``,
    ``map_tt_async_copy``, ``map_tt_mbarrier``,
    ``map_tt_sync_threads_partial``,
    ``map_tt_experimental_descriptor_{load,store}``, ``map_tt_print``,
    ``map_tt_program_id``) are *not* listed here -- their op-table keys
    are never overwritten so they remain canonical.
    """
    import re
    from pathlib import Path

    src_path = (
        Path(__file__).resolve().parent.parent / "op_mapping.py"
    )
    src = src_path.read_text()

    # Dead-but-loaded names (these are overridden by the overlay
    # EMITTERS dicts via OP_TABLE.update(...) at import time).
    DEAD_STUBS = [
        "map_tt_load",
        "map_tt_store",
        "map_tt_dot",
        "map_tt_reduce",
        "map_tt_broadcast",
        "map_tt_splat",
        "map_tt_expand_dims",
        "map_tt_reshape",
        "map_tt_make_range",
    ]

    missing: List[str] = []
    for name in DEAD_STUBS:
        # Look for a ``# DEAD-BUT-LOADED:`` comment somewhere in the
        # block of comment lines immediately preceding ``def <name>(``.
        pattern = re.compile(
            r"# DEAD-BUT-LOADED:[^\n]*"  # marker + same-line text
            r"(?:\n#[^\n]*)*"            # optional continuation comment lines
            r"\ndef " + re.escape(name) + r"\(",
            re.MULTILINE,
        )
        if not pattern.search(src):
            missing.append(name)

    assert not missing, (
        f"Dead-but-loaded legacy emitters missing the "
        f"'# DEAD-BUT-LOADED:' marker in op_mapping.py: {missing!r}. "
        f"Add the standard marker block immediately above each ``def`` "
        f"so readers know the function is superseded by the overlay "
        f"EMITTERS dict at module-init."
    )


def test_op_mapping_live_canonical_emitters_unmarked() -> None:
    """Negative control: emitters that remain canonical in OP_TABLE
    must NOT carry the ``DEAD-BUT-LOADED:`` marker (otherwise the marker
    becomes meaningless -- a regression where someone marks a still-live
    emitter would let the overlay-migration audit silently miss real
    dead code).
    """
    import re
    from pathlib import Path

    src_path = (
        Path(__file__).resolve().parent.parent / "op_mapping.py"
    )
    src = src_path.read_text()

    LIVE_STUBS = [
        "map_tt_atomic_rmw",
        "map_tt_where",
        "map_tt_trans",
        "map_tt_async_copy",
        "map_tt_mbarrier",
        "map_tt_sync_threads_partial",
        "map_tt_experimental_descriptor_load",
        "map_tt_experimental_descriptor_store",
        "map_tt_print",
        "map_tt_program_id",
    ]

    falsely_marked: List[str] = []
    for name in LIVE_STUBS:
        pattern = re.compile(
            r"# DEAD-BUT-LOADED:[^\n]*"
            r"(?:\n#[^\n]*)*"
            r"\ndef " + re.escape(name) + r"\(",
            re.MULTILINE,
        )
        if pattern.search(src):
            falsely_marked.append(name)

    assert not falsely_marked, (
        f"These emitters are still canonical (their OP_TABLE entry is "
        f"never overwritten by an overlay) but carry the "
        f"'# DEAD-BUT-LOADED:' marker: {falsely_marked!r}. Remove the "
        f"marker -- a false-positive label hides real dead code."
    )
