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

    # The outermost stmt must be an AttrStmt with attr_key="thread_extent"
    # whose IterVar.var is the same Var the body references.
    assert isinstance(func.body, tir.AttrStmt), (
        f"expected outer AttrStmt(thread_extent), got {type(func.body).__name__}"
    )
    assert func.body.attr_key == "thread_extent", (
        f"expected attr_key='thread_extent', got {func.body.attr_key!r}"
    )
    iter_var = func.body.node
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
