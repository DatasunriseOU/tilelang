"""Tests for the lifted ``poc.triton_frontend.pipeline`` helpers.

Covers the two new public helpers extracted from the e2e harness:

* :func:`is_custom_form_ttir` -- heuristic detector for "custom" vs
  "generic" MLIR text. We check both directions (positive + negative).
* :func:`round_trip_through_cxx_shim` -- delegates to the C++ shim's
  ``Module.to_generic()`` and is gated by shim availability. When the
  shim isn't built we just assert pass-through behaviour.
"""
from __future__ import annotations

import importlib

import pytest


# A tiny custom-form TTIR fragment (Triton's printer output shape).
# The ``tt.func @kernel`` declaration alone is enough for the heuristic
# to fire -- generic form would emit ``"tt.func"() ({...}) : () -> ()``.
_CUSTOM_FORM_TTIR = """
module {
  tt.func public @add_kernel(%x_ptr: !tt.ptr<f32>, %y_ptr: !tt.ptr<f32>) {
    tt.return
  }
}
"""

# Equivalent generic form -- every op name is quoted so a
# parser without the ``tt`` dialect registered can still consume it.
_GENERIC_FORM_TTIR = """
module {
  "tt.func"() ({
  ^bb0(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>):
    "tt.return"() : () -> ()
  }) {sym_name = "add_kernel", function_type = (!tt.ptr<f32>, !tt.ptr<f32>) -> ()} : () -> ()
}
"""


def test_is_custom_form_ttir_positive():
    from poc.triton_frontend.pipeline import is_custom_form_ttir

    assert is_custom_form_ttir(_CUSTOM_FORM_TTIR) is True


def test_is_custom_form_ttir_negative_for_generic():
    from poc.triton_frontend.pipeline import is_custom_form_ttir

    # Generic form should NOT be classified as custom-form. Note: the
    # heuristic is a token check; ``"tt.func"`` (quoted) does not
    # contain ``tt.func @`` so this is unambiguous.
    assert is_custom_form_ttir(_GENERIC_FORM_TTIR) is False


def test_is_custom_form_ttir_handles_non_string():
    from poc.triton_frontend.pipeline import is_custom_form_ttir

    # Defensive: callers may accidentally pass a Module object or None.
    assert is_custom_form_ttir(None) is False  # type: ignore[arg-type]
    assert is_custom_form_ttir(123) is False  # type: ignore[arg-type]


def test_round_trip_through_cxx_shim_handles_custom_form():
    """When the shim is available, custom-form TTIR round-trips into
    generic form (every op name quoted). When the shim is unavailable
    we exercise the pass-through fallback.
    """
    from poc.triton_frontend.pipeline import round_trip_through_cxx_shim

    try:
        importlib.import_module("_triton_frontend_cxx")
    except Exception:
        pytest.skip("_triton_frontend_cxx not built; pass-through fallback")

    out = round_trip_through_cxx_shim(_CUSTOM_FORM_TTIR)
    # Generic form quotes the op name -- ``"tt.func"`` should appear
    # somewhere in the round-tripped output.
    assert isinstance(out, str)
    assert '"tt.func"' in out, (
        f"expected generic op-form (quoted op names), got:\n{out[:400]}"
    )


def test_round_trip_through_cxx_shim_passthrough_on_failure():
    """If the shim raises (e.g. malformed TTIR text) we return the
    input unchanged so the caller can attempt a direct parse.
    """
    from poc.triton_frontend.pipeline import round_trip_through_cxx_shim

    junk = "this is not valid mlir at all"
    out = round_trip_through_cxx_shim(junk)
    # Either the shim isn't built (returns input), or it parses and
    # raises (returns input via the except branch). Both end up with
    # the original string.
    assert out == junk


def test_program_id_wrapped_in_thread_extent():
    """``_make_prim_func`` wraps the body with ``AttrStmt(thread_extent)`` for
    every program_id Var the walker recorded.

    Regression test for the ``MakePackedAPI`` failure
    ``variables (pid0_*, ...) are used, but are not passed in as API
    arguments``: the lowering pipeline rejects free Vars in the body unless
    they're either function parameters or thread-environment-bound IterVars.
    Wrapping ``pid`` with ``tir.AttrStmt(IterVar, "thread_extent", extent,
    body)`` puts it in the latter bucket and lets MakePackedAPI through.
    """
    pytest.importorskip("tvm")
    from tvm import tir

    from poc.triton_frontend import _make_prim_func
    from poc.triton_frontend.op_mapping import WalkerCtx

    ctx = WalkerCtx()
    # Synthesize a minimal kernel body: ``pid = ...; tile[pid] = 1.0``.
    pid = tir.Var("pid0_1", "int32")
    extent = tir.Var("gridDim_0", "int32")
    ctx.program_id_vars.append((pid, 0, extent))
    # A buffer-less Evaluate stmt that *uses* pid keeps the test focused on
    # the AttrStmt wrap; verifying the inner body just needs to be non-empty.
    ctx.stmts.append(tir.Evaluate(pid))

    func = _make_prim_func(ctx, name="kernel_with_pid")

    # The body is wrapped with multiple ``AttrStmt(thread_extent)`` layers --
    # the outermost is now ``threadIdx.x`` (added so TileLang's
    # ``CurrentThreadBounds()`` resolves to a positive extent), and the
    # inner ones are the program_id ``blockIdx.{x,y,z}`` bindings. We walk
    # the AttrStmt chain to locate the blockIdx binding for ``pid``.
    assert isinstance(func.body, tir.AttrStmt), (
        f"expected outer AttrStmt(thread_extent), got {type(func.body).__name__}"
    )
    assert func.body.attr_key == "thread_extent", (
        f"expected attr_key='thread_extent', got {func.body.attr_key!r}"
    )

    def _find_blockidx_attrstmt(node):
        """Walk the AttrStmt chain to find the blockIdx binding for pid."""
        cur = node
        while isinstance(cur, tir.AttrStmt):
            if cur.attr_key == "thread_extent":
                iv = cur.node
                tag = str(iv.thread_tag)
                if "blockIdx" in tag and iv.var.same_as(pid):
                    return cur
            cur = cur.body
        return None

    block_attr = _find_blockidx_attrstmt(func.body)
    assert block_attr is not None, (
        "no AttrStmt(thread_extent) with blockIdx tag and the program_id "
        f"Var found anywhere in the body chain; got: {func.body!r}"
    )
    iter_var = block_attr.node
    assert iter_var.var.same_as(pid), (
        "AttrStmt IterVar.var must be the program_id Var"
    )
    assert "blockIdx" in str(iter_var.thread_tag), (
        f"expected blockIdx.* thread_tag, got {iter_var.thread_tag!r}"
    )
    # The extent Var must also appear in params so MakePackedAPI sees it as
    # a packed arg rather than a free Var.
    param_names = {str(getattr(p, "name", p)) for p in func.params}
    assert "gridDim_0" in param_names, (
        f"extent Var must be promoted to a PrimFunc param; got {param_names}"
    )


def test_runtime_scalar_arg_added_to_params():
    """Scalar block args from ``tt.func`` (e.g. ``n_elements``) are appended
    to ``PrimFunc.params``.

    Regression test for the ``arg3`` half of the MakePackedAPI failure.
    Triton 3.x already folds ``tl.constexpr`` parameters at the TTIR stage,
    so anything that survives as a non-pointer ``tt.func`` block arg is a
    runtime arg and must appear in the packed-API signature.
    """
    pytest.importorskip("tvm")
    from tvm import tir

    from poc.triton_frontend import _make_prim_func
    from poc.triton_frontend.op_mapping import WalkerCtx

    ctx = WalkerCtx()
    n_elements = tir.Var("n_elements", "int32")
    ctx.runtime_args.append(n_elements)
    ctx.stmts.append(tir.Evaluate(n_elements))

    func = _make_prim_func(ctx, name="kernel_with_runtime_scalar")
    param_names = {str(getattr(p, "name", p)) for p in func.params}
    assert "n_elements" in param_names, (
        f"runtime scalar arg must be a PrimFunc param; got {param_names}"
    )


def test_make_prim_func_stamps_num_warps_threadidx_extent():
    """``_make_prim_func`` wraps the body in a ``threadIdx.x`` ``thread_extent``
    AttrStmt sized by ``ctx.num_warps * 32`` and stamps matching
    ``num_warps`` / ``num_stages`` PrimFunc attrs.

    Regression for the matmul ``COMPILE_FAIL`` where TileLang's
    ``GemmWarpPolicyNode::computeWarpPartition`` (src/op/gemm.cc:288)
    raised ``Check failed: m_warp * n_warp == num_warps`` because no
    ``threadIdx.x`` thread_extent existed on the body and ``num_warps``
    collapsed to ``block_size (1) / warp_size (32) == 0``.
    """
    pytest.importorskip("tvm")
    from tvm import tir

    from poc.triton_frontend import _make_prim_func
    from poc.triton_frontend.op_mapping import WalkerCtx

    ctx = WalkerCtx()
    ctx.num_warps = 4
    ctx.num_stages = 2
    # Trivial body so the wrap is the only structural feature.
    ctx.stmts.append(tir.Evaluate(tir.const(0, "int32")))

    func = _make_prim_func(ctx, name="kernel_for_warps")

    # Outermost stmt should be the threadIdx.x AttrStmt with extent 128.
    assert isinstance(func.body, tir.AttrStmt)
    assert func.body.attr_key == "thread_extent"
    iv = func.body.node
    assert "threadIdx.x" in str(iv.thread_tag), (
        f"expected threadIdx.x tag, got {iv.thread_tag!r}"
    )
    extent_value = int(func.body.value)
    assert extent_value == 4 * 32, (
        f"expected threadIdx.x extent == num_warps*32 == 128, got {extent_value}"
    )
    # PrimFunc-level attrs reflect ctx.num_warps / ctx.num_stages.
    assert int(func.attrs["num_warps"]) == 4
    assert int(func.attrs["num_stages"]) == 2


def test_make_prim_func_overrides_num_warps_via_ctx():
    """``ctx.num_warps`` overrides the default; threadIdx.x extent and the
    matching PrimFunc attr must move in lockstep.
    """
    pytest.importorskip("tvm")
    from tvm import tir

    from poc.triton_frontend import _make_prim_func
    from poc.triton_frontend.op_mapping import WalkerCtx

    ctx = WalkerCtx()
    ctx.num_warps = 8
    ctx.num_stages = 3
    ctx.stmts.append(tir.Evaluate(tir.const(0, "int32")))

    func = _make_prim_func(ctx, name="kernel_eight_warps")
    extent_value = int(func.body.value)
    assert extent_value == 8 * 32 == 256
    assert int(func.attrs["num_warps"]) == 8
    assert int(func.attrs["num_stages"]) == 3


def test_from_ttir_plumbs_num_warps_kwarg():
    """``from_ttir(ttir_module, num_warps=...)`` overrides ``WalkerCtx``
    defaults so the harness can plumb Triton's compile options through.
    """
    pytest.importorskip("tvm")
    from poc.triton_frontend import from_ttir

    # Minimal text TTIR — empty module body. The text walker just builds
    # an empty ctx; we only care that the kwargs land on the PrimFunc.
    ttir = """
module {
  tt.func public @noop() attributes {noinline = false} {
    tt.return
  }
}
"""
    func = from_ttir(
        ttir,
        name="noop",
        num_warps=2,
        num_stages=1,
        _allow_text_ttir=True,
    )
    assert int(func.attrs["num_warps"]) == 2
    assert int(func.attrs["num_stages"]) == 1
