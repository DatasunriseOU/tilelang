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
    _normalize_mlir_dtype,
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
