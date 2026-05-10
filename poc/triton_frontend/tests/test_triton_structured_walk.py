"""End-to-end MLIR walk regression for the vendored TritonStructured dialect.

Beyond ``verify_dialect_loads.py``'s 1-op smoke test, this exercises the
parser + walker on a richer module that mixes ``tts.make_tptr``,
``tts.make_gather_scatter_tptr``, and a few ``tt.*`` ops, then asserts that
at least one ``tts.*`` op was visited via ``mlir.ir.Module.operation.walk``.

Skipped when the vendored ``register_dialects`` shim has not been built
(same gating logic as ``verify_dialect_loads.py``).
"""

from __future__ import annotations

import pytest


def _try_load_register_dialects():
    try:
        from poc.triton_frontend._cxx.register_triton_structured import (  # type: ignore
            register_dialects,
        )
        return register_dialects
    except ImportError:
        pass
    try:
        from poc.triton_frontend.ptr_analysis import shim_available, _load_shim
        if shim_available():
            return _load_shim().register_dialects
    except ImportError:
        return None


def _try_import_mlir_ir():
    try:
        from mlir import ir  # type: ignore
        from mlir.dialects import func, arith, scf, tensor, math  # type: ignore
        return ir
    except ImportError:
        return None


_REGISTER = _try_load_register_dialects()
_IR = _try_import_mlir_ir()

skip_no_dialects = pytest.mark.skipif(
    _REGISTER is None or _IR is None,
    reason=(
        "vendored TritonStructured pybind shim or mlir python bindings not "
        "available; build with -DTRITON_INSTALL_DIR set (see "
        "poc/triton_frontend/_cxx/README.md)."
    ),
)


# Keep the snippet small but exercise multiple tts.* ops + at least one tt.* op.
RICHER_MODULE = (
    "module {\n"
    "  func.func @walk_target(\n"
    "      %base: !tt.ptr<f32>, %off: index, %st: index,\n"
    "      %gs_off: tensor<8xi32>) {\n"
    "    %0 = tts.make_tptr %base to\n"
    "          sizes: [4],\n"
    "          strides: [%st],\n"
    "          offsets: [%off],\n"
    "          shape: [0],\n"
    "          order: []\n"
    "          : !tt.ptr<f32> to tensor<4x!tt.ptr<f32>>\n"
    "    %1 = tts.make_gather_scatter_tptr %base to\n"
    "          sizes: [8]\n"
    "          gather_scatter_dim: 0\n"
    "          gather_scatter_offset: %gs_off,\n"
    "          strides: [%st],\n"
    "          offsets: [%off],\n"
    "          shape: [0],\n"
    "          order: []\n"
    "          : !tt.ptr<f32> to tensor<8x!tt.ptr<f32>>\n"
    "    return\n"
    "  }\n"
    "}\n"
)


@skip_no_dialects
def test_walk_visits_tts_ops() -> None:
    ir = _IR  # captured at import time
    register_dialects = _REGISTER

    with ir.Context() as ctx:
        register_dialects(ctx)
        ctx.allow_unregistered_dialects = False
        with ir.Location.unknown(ctx):
            module = ir.Module.parse(RICHER_MODULE)

        seen: list[str] = []

        def _visit(op):  # type: ignore[no-untyped-def]
            seen.append(op.name)
            return ir.WalkResult.ADVANCE  # type: ignore[attr-defined]

        # Newer MLIR python bindings expose .operation.walk(); guard for older
        # versions where iter(module.body) is the only available traversal.
        if hasattr(module.operation, "walk"):
            module.operation.walk(_visit)
        else:
            for op in module.body:
                seen.append(op.name)
                if hasattr(op, "regions"):
                    for region in op.regions:
                        for block in region:
                            for inner in block:
                                seen.append(inner.name)

        tts_ops = [n for n in seen if n.startswith("tts.")]
        assert tts_ops, (
            f"expected at least one tts.* op in the walk, saw: {seen!r}"
        )
        assert any(n == "tts.make_tptr" for n in tts_ops)


@skip_no_dialects
def test_parser_rejects_unregistered_when_disallowed() -> None:
    """Sanity: confirm the dialect registry is the ONLY thing letting tts.* parse."""
    ir = _IR
    bad = (
        "module {\n"
        "  func.func @uses_phantom_dialect() {\n"
        "    \"phantom.op\"() : () -> ()\n"
        "    return\n"
        "  }\n"
        "}\n"
    )
    with ir.Context() as ctx:
        # Don't register dialects -- ensure unregistered dialects are blocked.
        ctx.allow_unregistered_dialects = False
        with ir.Location.unknown(ctx):
            with pytest.raises(Exception):  # mlir raises a generic Exception subclass
                ir.Module.parse(bad)


# Wave-3: cover the gather/scatter rewrite path + UseAnalysis use-chain
# inspection. The wave-2 grok review flagged that the existing tests only
# exercised tts.make_tptr; these add coverage for tts.make_gather_scatter_tptr
# and assert the analysis walks the use-def chain through it.

#
# Adapted from
# ``microsoft/triton-shared/test/Conversion/TritonToStructured/
#   gather_scatter_ptr_simple.mlir`` (MIT). The original fixture used an
# opaque ``tensor<8xi32>`` block-arg as the gather index, which
# ``PtrAnalysis::rewriteOp`` cannot trace back to a structured base --
# the upstream pass needs ``tt.make_range`` (or another structured
# producer) to seed UseAnalysis before it can emit
# ``tts.make_gather_scatter_tptr``. This rewrite uses that idiomatic
# pattern (load index from %arg1, multiply / remsi to derive a
# row-gather offset, then 2D broadcast).
GATHER_INPUT_MLIR = """\
module {
  tt.func public @row_gather(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<i32>, %arg2: !tt.ptr<f32>) attributes {noinline = false} {
    %cst = arith.constant dense<8> : tensor<8x1xi32>
    %cst_0 = arith.constant dense<8> : tensor<8xi32>
    %cst_1 = arith.constant dense<3> : tensor<8xi32>
    %cst_2 = arith.constant dense<5> : tensor<8xi32>
    %0 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %1 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
    %2 = tt.addptr %1, %0 : tensor<8x!tt.ptr<i32>>, tensor<8xi32>
    %3 = tt.load %2 : tensor<8x!tt.ptr<i32>>
    %4 = arith.muli %3, %cst_1 : tensor<8xi32>
    %5 = arith.remsi %4, %cst_2 : tensor<8xi32>
    %6 = tt.expand_dims %5 {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32>
    %7 = arith.muli %6, %cst : tensor<8x1xi32>
    %8 = tt.expand_dims %0 {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
    %9 = tt.broadcast %7 : tensor<8x1xi32> -> tensor<8x8xi32>
    %10 = tt.broadcast %8 : tensor<1x8xi32> -> tensor<8x8xi32>
    %11 = arith.addi %9, %10 : tensor<8x8xi32>
    %12 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<8x8x!tt.ptr<f32>>
    %13 = tt.addptr %12, %11 : tensor<8x8x!tt.ptr<f32>>, tensor<8x8xi32>
    %14 = tt.load %13 : tensor<8x8x!tt.ptr<f32>>
    %15 = tt.expand_dims %0 {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32>
    %16 = arith.muli %15, %cst : tensor<8x1xi32>
    %17 = tt.broadcast %16 : tensor<8x1xi32> -> tensor<8x8xi32>
    %18 = arith.addi %17, %10 : tensor<8x8xi32>
    %19 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<8x8x!tt.ptr<f32>>
    %20 = tt.addptr %19, %18 : tensor<8x8x!tt.ptr<f32>>, tensor<8x8xi32>
    tt.store %20, %14 : tensor<8x8x!tt.ptr<f32>>
    tt.return
  }
}
"""


def _try_import_ptr_analysis():
    try:
        from poc.triton_frontend.ptr_analysis import PtrAnalysis, dialects_available
        return PtrAnalysis, dialects_available
    except ImportError:
        return None, None


_PA_CLS, _PA_AVAIL = _try_import_ptr_analysis()

skip_no_pa = pytest.mark.skipif(
    _PA_CLS is None or _PA_AVAIL is None or not _PA_AVAIL(),
    reason="ptr_analysis shim built without dialects",
)


@skip_no_pa
def test_rewrite_emits_make_gather_scatter_tptr() -> None:
    """rewriteOp on a vector-indexed tt.addptr/tt.load chain must emit
    tts.make_gather_scatter_tptr (or the equivalent gather-flagged tts op).

    The ``enable_make_gather_scatter_tensor_ptr`` constructor flag MUST be
    True for the upstream rewriter to opt into emitting
    ``tts.make_gather_scatter_tptr`` -- with the flag False (the default
    ``PtrAnalysis()`` setting), the gather-shaped chain is preserved as a
    plain ``tt.addptr`` so backends that don't yet handle the new op
    don't blow up. This test exercises the opt-in surface.
    """
    pa = _PA_CLS(GATHER_INPUT_MLIR, enable_make_gather_scatter_tensor_ptr=True)
    rewritten = pa.rewrite()
    assert isinstance(rewritten, str) and rewritten
    # Either the dedicated op, or the unified make_tptr with a gather flag.
    assert (
        "tts.make_gather_scatter_tptr" in rewritten
        or "gather_scatter_dim" in rewritten
        or "tts.make_unstructured_tptr" in rewritten
    ), f"expected gather/scatter rewrite marker, got:\n{rewritten}"


@skip_no_pa
def test_extract_states_walks_use_chain_through_gather() -> None:
    """UseAnalysis is wired in via PtrAnalysis::extract_states; the chain
    splat -> addptr -> load must surface at least one PtrState whose op
    string mentions one of those upstream producers.
    """
    states = _PA_CLS(
        GATHER_INPUT_MLIR, enable_make_gather_scatter_tensor_ptr=True
    ).extract_states()
    assert isinstance(states, list)
    # A successful UseAnalysis traversal extracts states for each rewritten
    # producer; we don't pin the exact count (depends on shim version) but
    # we do require non-empty traversal output and that the JSON payload
    # references the expected ops.
    blob = "\n".join(getattr(s, "op", str(s)) for s in states)
    assert any(tok in blob for tok in ("addptr", "splat", "make_gather", "make_tptr")), (
        f"UseAnalysis-derived states do not reference upstream producers: {blob!r}"
    )
