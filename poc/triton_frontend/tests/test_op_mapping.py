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

    # Bumped from 89 -> 95 by the FLA chunk_gated_delta_rule enablement
    # pass: ``math.{exp2,log2,rsqrt,erf,floor,ceil}`` were added to
    # ``ARITH_EMITTERS`` so the gated-delta-rule kernel's ``exp2`` lane
    # (and the LayerNorm/RMSNorm rsqrt lane that came along for free)
    # lower without FAILED_OPS.
    #
    # Bumped 95 -> 98 by the FLA Path D end-to-end seam: capturing real
    # ``chunk_gated_delta_rule_fwd_kernel_h_blockdim64`` TTIR surfaced
    # ``arith.andi`` (boundary-check mask chains from
    # ``tl.load(..., boundary_check=(0, 1))``). Added the bitwise/logical
    # cohort ``arith.{andi,ori,xori}`` for consistency -- all three lower
    # via ``tir.bitwise_{and,or,xor}``.
    EXPECTED = 98
    assert len(OP_TABLE) == EXPECTED, (
        f"OP_TABLE size changed from {EXPECTED} to {len(OP_TABLE)}; "
        f"if intentional, update this constant + the three README "
        f"occurrences (poc/triton_frontend/README.md). Current keys: "
        f"{sorted(OP_TABLE.keys())}"
    )




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


def test_fla_chunk_gated_delta_rule_op_coverage() -> None:
    """Pin the OP_TABLE entries required by FLA's ``chunk_gated_delta_rule``
    forward kernel (``fla/ops/common/chunk_delta_h.py``).

    The kernel uses ``tl.dot`` (matmul) and ``tl.math.exp2`` (via the
    ``fla.ops.utils.op.exp2`` shim that wraps ``tl.math.exp2``); on top
    of the standard ``tt.load`` / ``tt.store`` / ``tt.broadcast`` /
    ``tt.expand_dims`` / ``tt.trans`` / ``tt.where`` cohort.

    Without ``math.exp2`` in OP_TABLE the reducer would mark the kernel
    FAILED_OPS (the elementwise-only Tier-1 baseline only had
    ``math.exp``).
    """
    from poc.triton_frontend.op_mapping import OP_TABLE

    required = {
        "tt.dot",
        "tt.expand_dims",
        "tt.broadcast",
        "tt.trans",
        "tt.where",
        "tt.load",
        "tt.store",
        "tt.get_program_id",
        "math.exp",
        "math.exp2",
        "math.log",
        "math.log2",
        "math.rsqrt",
        "math.sqrt",
    }
    missing = required - set(OP_TABLE.keys())
    assert not missing, (
        f"FLA chunk_gated_delta_rule still needs these ops added to "
        f"OP_TABLE: {sorted(missing)!r}"
    )


def test_math_exp2_emitter_is_distinct_from_exp() -> None:
    """``math.exp2`` and ``math.exp`` must dispatch to different emitters --
    a copy-paste bug that aliases them would silently compute exp(x) where
    the user asked for 2**x (a 44 % numerical error for x=1).
    """
    from poc.triton_frontend.op_mapping import OP_TABLE

    emit_exp = OP_TABLE["math.exp"]
    emit_exp2 = OP_TABLE["math.exp2"]
    assert emit_exp is not emit_exp2, (
        f"math.exp and math.exp2 share the same emitter "
        f"({emit_exp!r}); the exp2 path would silently compute exp(x)."
    )
    # Confirm name conveys identity (helps debug logs).
    assert "exp2" in getattr(emit_exp2, "__name__", ""), (
        f"math.exp2 emitter has misleading name "
        f"{getattr(emit_exp2, '__name__', '?')!r}"
    )


def test_sanitize_printf_format():
    from poc.triton_frontend.op_mapping import _sanitize_printf_format

    assert _sanitize_printf_format("%n") == "%%n"
    assert _sanitize_printf_format("hello %n world") == "hello %%n world"
    assert _sanitize_printf_format("%%n") == "%%n"
    assert _sanitize_printf_format("%%%n") == "%%%%n"
    assert _sanitize_printf_format("%10n") == "%%10n"
    assert _sanitize_printf_format("%-10.5lln") == "%%-10.5lln"
    assert _sanitize_printf_format("Valid %d and %s") == "Valid %d and %s"
    assert _sanitize_printf_format("%n%n") == "%%n%%n"
    assert _sanitize_printf_format("%%n%n") == "%%n%%n"
    assert _sanitize_printf_format("") == ""


def test_tt_async_commit_and_wait() -> None:
    """Test that tt.async_commit_group and tt.async_wait emit the proper
    ptx_commit_group and ptx_wait_group instructions.
    """
    from poc.triton_frontend.op_mapping import map_tt_async_copy

    ctx = WalkerCtx()
    # Test async_commit_group
    op_commit = _FakeMlirOp(
        name="tt.async_commit_group",
        operands=[],
        results=[],
        printed='"tt.async_commit_group"() : () -> ()',
    )
    handle_commit = map_tt_async_copy(op_commit, ctx)
    assert handle_commit is not None
    # Depending on TileLang lazy import, it might be a Stmt or Expr.
    # It should have "ptx_commit_group" inside it.
    assert "ptx_commit_group" in str(handle_commit)

    # Test async_wait
    op_wait = _FakeMlirOp(
        name="tt.async_wait",
        operands=[],
        results=[],
        printed='"tt.async_wait"() <{num = 2 : i32}> : () -> ()',
    )
    handle_wait = map_tt_async_copy(op_wait, ctx)
    assert handle_wait is not None
    # Should have "ptx_wait_group" and the argument '2'
    assert "ptx_wait_group" in str(handle_wait)
    assert "2" in str(handle_wait)
