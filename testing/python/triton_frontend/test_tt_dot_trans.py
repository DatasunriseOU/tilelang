"""Smoke tests for ``tt.trans`` + ``tt.dot`` lowering (Phase-1 migration).

These exercise the dict-shaped fake-op path so we don't need MLIR
bindings: the walker code in ``poc.triton_frontend.op_mapping`` accepts
either a real ``mlir.ir.Operation`` or a dict shape ``{"operands": [...],
"results": [...], "attrs": {...}}``. The migration plan calls out that
``cppmega.mlx`` kernels (``dsa_splitk_indexer_loss``,
``sparse_mla_path_c``) emit ``%bt = tt.trans %b ; tt.dot %a, %bt`` rather
than a single ``tt.dot`` carrying a ``transpose_B`` attribute, so we lock
down the fold-into-flag behaviour here.
"""
from __future__ import annotations

import pytest


def _ctx():
    try:
        from poc.triton_frontend import op_mapping
    except Exception as exc:
        pytest.skip(f"poc.triton_frontend.op_mapping unavailable: {exc!r}")
    return op_mapping


def test_tt_trans_records_transposed_view_and_rebinds_source():
    om = _ctx()
    ctx = om.WalkerCtx()

    # Pretend %b is some buffer/PrimExpr that an upstream op already bound.
    ctx.bind("%b", "TIR_VALUE_B")

    op = {"operands": ["%b"], "results": ["%bt"], "attrs": {}}
    om.map_tt_trans(op, ctx)

    # The result SSA aliases the source TIR value (no materialisation).
    assert ctx.get("%bt") == "TIR_VALUE_B"
    # And the sidecar records the flipped pair (default last two axes).
    assert ctx.transposed_views.get("%bt") == (-2, -1)


def test_tt_trans_double_transpose_cancels():
    om = _ctx()
    ctx = om.WalkerCtx()
    ctx.bind("%b", "TIR_VALUE_B")

    om.map_tt_trans({"operands": ["%b"], "results": ["%bt"], "attrs": {}}, ctx)
    om.map_tt_trans({"operands": ["%bt"], "results": ["%btt"], "attrs": {}}, ctx)

    # Two transposes cancel: %btt is bound but NOT in transposed_views.
    assert ctx.get("%btt") == "TIR_VALUE_B"
    assert "%btt" not in ctx.transposed_views


def test_tt_trans_explicit_order_attr_is_honoured():
    om = _ctx()
    ctx = om.WalkerCtx()
    ctx.bind("%x", "TIR_VALUE_X")

    op = {
        "operands": ["%x"],
        "results": ["%xt"],
        "attrs": {"order": (0, 2, 1)},  # 3-D: keep dim 0, swap (1, 2)
    }
    om.map_tt_trans(op, ctx)
    assert ctx.transposed_views.get("%xt") == (2, 1)


class _FakeRes:
    """Minimal MLIR-shaped result that satisfies ``_shape_of``/``_dtype_of``.

    A plain dict can't be used because ``WalkerCtx.bind`` stores results as
    keys in ``value_map`` (must be hashable). An object with ``.type.shape``
    and ``.type.element_type`` matches the real-MLIR path in those helpers.
    """

    def __init__(self, shape, dtype):
        self.type = type("Ty", (), {"shape": shape, "element_type": dtype})()


class _FakeSharedBuffer:
    def scope(self):
        return "shared"


def test_dot_after_trans_b_emits_gemm_with_transpose_B_true():
    """End-to-end fake-op walk: ``%bt = tt.trans %b ; tt.dot %a, %bt``."""
    om = _ctx()
    captured = {}

    class _FakeT:
        @staticmethod
        def alloc_fragment(shape, dtype):
            captured["alloc_fragment"] = (tuple(shape), dtype)
            return "FRESH_C"

        @staticmethod
        def gemm(a, b, c, transpose_A=False, transpose_B=False, clear_accum=False):
            captured["gemm"] = {
                "a": a,
                "b": b,
                "c": c,
                "transpose_A": transpose_A,
                "transpose_B": transpose_B,
                "clear_accum": clear_accum,
            }

    ctx = om.WalkerCtx()
    a_fake = _FakeRes((16, 16), "float16")
    b_fake = _FakeRes((16, 16), "float16")
    bt_fake = _FakeRes((16, 16), "float16")
    c_fake = _FakeRes((16, 16), "float32")
    # Bind them in context
    ctx.bind(a_fake, "TIR_A")
    ctx.bind(b_fake, "TIR_B")
    ctx.bind(c_fake, _FakeSharedBuffer())

    # tt.trans %b -> %bt
    om.map_tt_trans(
        {"operands": [b_fake], "results": [bt_fake], "attrs": {}}, ctx
    )

    # tt.dot %a, %bt -> %c. No transpose_B in attrs; the trans should fold.
    dot_op = {
        "operands": [a_fake, bt_fake, c_fake],
        "results": [_FakeRes((16, 16), "float32")],
        "attrs": {},
    }
    from poc.triton_frontend.op_emitters.reduction import map_tt_dot

    map_tt_dot(dot_op, ctx, language_module=_FakeT)
    assert "gemm" in captured, "T.gemm should have been called"
    assert captured["gemm"]["transpose_B"] is True, (
        "trans_b on %bt should fold into transpose_B=True at gemm time"
    )
    assert captured["gemm"]["transpose_A"] is False


def test_dot_with_transpose_B_attr_xors_with_pre_trans_b():
    """If trans_b is *both* on the dot attr and via tt.trans, they cancel."""
    om = _ctx()
    captured = {}

    class _FakeT:
        @staticmethod
        def alloc_fragment(shape, dtype):
            return "FRESH_C"

        @staticmethod
        def gemm(a, b, c, transpose_A=False, transpose_B=False, clear_accum=False):
            captured["transpose_B"] = transpose_B
            return None

    ctx = om.WalkerCtx()
    a_fake = _FakeRes((4, 4), "float16")
    b_fake = _FakeRes((4, 4), "float16")
    bt_fake = _FakeRes((4, 4), "float16")
    c_fake = _FakeRes((4, 4), "float32")
    ctx.bind(a_fake, "A")
    ctx.bind(b_fake, "B")
    ctx.bind(c_fake, _FakeSharedBuffer())
    om.map_tt_trans(
        {"operands": [b_fake], "results": [bt_fake], "attrs": {}}, ctx
    )
    
    dot_op = {
        "operands": [a_fake, bt_fake, c_fake],
        "results": [_FakeRes((4, 4), "float32")],
        "attrs": {"trans_b": True},
    }
    from poc.triton_frontend.op_emitters.reduction import map_tt_dot
    map_tt_dot(dot_op, ctx, language_module=_FakeT)
    # => transpose_B=False reaches T.gemm.
    assert captured["transpose_B"] is False


def test_layout_inference_tt_dot_trans_b():
    """Ensure that layout inference propagates correctly for tt.dot trans_b (float16 and int8)."""
    import tilelang as tl
    import tilelang.language as T
    import tvm
    from tilelang.utils.target import determine_target

    auto_target = tvm.target.Target(determine_target("auto"))

    def make_mod(dtype_a, dtype_b, dtype_c):
        block_M, block_N, block_K = 64, 64, 32

        @T.prim_func
        def main(
            A: T.Tensor((64, 64), dtype_a),
            B_t: T.Tensor((64, 64), dtype_b),
            C: T.Tensor((64, 64), dtype_c),
        ):
            with T.Kernel(1, threads=128) as _:
                A_shared = T.alloc_shared((block_M, block_K), dtype_a)
                B_shared = T.alloc_shared((block_N, block_K), dtype_b)
                C_local = T.alloc_fragment((block_M, block_N), dtype_c)

                T.clear(C_local)
                for k in T.Pipelined(T.ceildiv(64, block_K), num_stages=3):
                    T.copy(A[0:block_M, k * block_K : (k + 1) * block_K], A_shared)
                    T.copy(B_t[0:block_N, k * block_K : (k + 1) * block_K], B_shared)
                    T.gemm(A_shared, B_shared, C_local, transpose_B=True)

                T.copy(C_local, C[0:block_M, 0:block_N])

        return tvm.IRModule({"main": main})

    # Test float16
    mod_f16 = make_mod(T.float16, T.float16, T.float16)
    with tvm.target.Target(auto_target):
        mod_f16 = tvm.tir.transform.BindTarget(auto_target)(mod_f16)
        mod_f16 = tl.transform.LayoutInference()(mod_f16)
        assert mod_f16 is not None

    # Test int8
    mod_i8 = make_mod(T.int8, T.int8, T.int32)
    with tvm.target.Target(auto_target):
        mod_i8 = tvm.tir.transform.BindTarget(auto_target)(mod_i8)
        mod_i8 = tl.transform.LayoutInference()(mod_i8)
        assert mod_i8 is not None
