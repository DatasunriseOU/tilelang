from __future__ import annotations

import pytest

tvm = pytest.importorskip("tvm")

from poc.triton_frontend.op_emitters.control import (
    CONTROL_EMITTERS,
    map_arith_extsi,
    map_arith_index_cast,
    map_arith_truncf,
    map_tt_func,
)
from poc.triton_frontend.op_mapping import WalkerCtx

from ._fixtures import FakeMlirOp, FakeSSA


def test_vector_int_cast_lowers_per_lane() -> None:
    ctx = WalkerCtx()
    src = FakeSSA(name="src", shape=(4,), dtype="i32")
    dst = FakeSSA(name="dst", shape=(4,), dtype="i64")
    ctx.bind(src, tvm.tir.Ramp(tvm.tir.const(0, "int32"), tvm.tir.const(1, "int32"), 4))

    out = map_arith_extsi(FakeMlirOp("arith.extsi", [src], [dst]), ctx)

    assert isinstance(out, tvm.tir.Buffer)
    assert str(out.dtype) == "int64"
    assert ctx.get(dst) is out
    assert ctx.stmts, "vector cast should emit a per-lane loop"


def test_buffer_float_cast_lowers_per_lane() -> None:
    ctx = WalkerCtx()
    src = FakeSSA(name="src", shape=(4,), dtype="f32")
    dst = FakeSSA(name="dst", shape=(4,), dtype="f16")
    buf = tvm.tir.decl_buffer((4,), "float32", name="src_buf")
    ctx.bind(src, buf)

    out = map_arith_truncf(FakeMlirOp("arith.truncf", [src], [dst]), ctx)

    assert isinstance(out, tvm.tir.Buffer)
    assert str(out.dtype) == "float16"
    assert ctx.get(dst) is out
    assert ctx.stmts, "tile cast should emit a per-lane loop"


def test_index_cast_routes_as_integer_cast() -> None:
    ctx = WalkerCtx()
    src = FakeSSA(name="src", dtype="index")
    dst = FakeSSA(name="dst", dtype="i32")
    ctx.bind(src, tvm.tir.Var("idx", "int64"))

    out = map_arith_index_cast(FakeMlirOp("arith.index_cast", [src], [dst]), ctx)

    assert str(out.dtype) == "int32"
    assert ctx.get(dst) is out


def test_cf_branch_terminators_are_registered() -> None:
    """FLA early-return diamonds use cf.* terminators inside regions."""
    ctx = WalkerCtx()

    assert CONTROL_EMITTERS["cf.br"](FakeMlirOp("cf.br", [], []), ctx) is None
    assert CONTROL_EMITTERS["cf.cond_br"](
        FakeMlirOp("cf.cond_br", [FakeSSA(name="cond", dtype="bool")], []),
        ctx,
    ) is None


def test_tt_func_uses_caller_supplied_pointer_abi_shapes() -> None:
    """Runtime-owned DLTensor ABIs can seed exact pointer buffer sizes."""
    ctx = WalkerCtx()
    ctx.arg_buffer_shapes = {0: (4096,), "beta": (64,)}
    k = FakeSSA({"name": "%k", "dtype": "f16", "is_ptr": True})
    beta = FakeSSA({"name": "%beta", "dtype": "f32", "is_ptr": True})
    t = FakeSSA({"name": "%T", "dtype": "i32"})

    # No regions: map_tt_func returns after binding block args, which is
    # enough for this ABI-shape contract.
    result = map_tt_func({"block_args": [k, beta, t]}, ctx)

    assert result is None
    assert [int(x) for x in ctx.buffers["k"].shape] == [4096]
    assert [int(x) for x in ctx.buffers["beta"].shape] == [64]
    assert {"k", "beta"}.issubset(ctx.fixed_arg_buffer_keys)
