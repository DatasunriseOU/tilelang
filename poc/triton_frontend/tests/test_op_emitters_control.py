from __future__ import annotations

import pytest

tvm = pytest.importorskip("tvm")

from poc.triton_frontend.op_emitters.control import (
    map_arith_extsi,
    map_arith_index_cast,
    map_arith_truncf,
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
