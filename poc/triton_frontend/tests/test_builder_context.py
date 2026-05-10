import pytest
tvm = pytest.importorskip("tvm")
import tilelang.language as T
import tilelang

from poc.triton_frontend.op_mapping import WalkerCtx
from poc.triton_frontend.op_emitters.reduction import map_tt_dot
from poc.triton_frontend.tests._fixtures import FakeSSA

def _fake_value(name: str, *, shape, dtype: str = "float32") -> FakeSSA:
    return FakeSSA(name=name, shape=tuple(shape), dtype=dtype)

def _decl_buffer(name: str, shape, dtype: str = "float32"):
    return tvm.tir.decl_buffer(shape, dtype, name=name)

def test_dot():
    ctx = WalkerCtx()
    a_ssa = _fake_value("a_ssa", shape=[16, 32], dtype="float16")
    b_ssa = _fake_value("b_ssa", shape=[32, 16], dtype="float16")
    out_ssa = _fake_value("c_ssa", shape=[16, 16], dtype="float32")

    a_buf = _decl_buffer("A", [16, 32], "float16")
    b_buf = _decl_buffer("B", [32, 16], "float16")
    ctx.bind(a_ssa, a_buf)
    ctx.bind(b_ssa, b_buf)

    op = {
        "name": "tt.dot",
        "operands": [a_ssa, b_ssa],
        "results": [out_ssa],
        "attrs": {},
    }

    handle_out = []

    @T.prim_func
    def _test_func():
        with T.Kernel(1, threads=128):
            handle = map_tt_dot(op, ctx)
            handle_out.append(handle)

    print("Success")

if __name__ == "__main__":
    test_dot()
