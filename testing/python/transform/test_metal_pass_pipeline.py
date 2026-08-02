import re

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm
from tvm.script import ir as I
from tvm.script import tirx as TX


def _call_packed_args(func):
    calls = []

    def visit(node):
        if (
            isinstance(node, tvm.tir.Call)
            and isinstance(node.op, tvm.ir.Op)
            and str(node.op.name) in {"tir.tvm_call_packed", "tirx.tvm_call_packed"}
        ):
            calls.append(node)

    tvm.tir.stmt_functor.post_order_visit(func.body, visit)
    assert len(calls) == 1, func
    return list(calls[0].args)


def _lower_metal_source(func):
    with tvm.transform.PassContext(), tvm.target.Target("metal"):
        artifact = tilelang.lower(func, target="metal")
    assert artifact.kernel_source is not None
    return artifact


def test_lower_device_kernel_launch_inlines_bind_thread_extent():
    @I.ir_module
    class Before:
        @TX.prim_func
        def main(A: TX.Buffer(16, "float32"), n: TX.int32):
            TX.func_attr({"target": TX.target("llvm")})
            Before.kernel(A.data, n)

        @TX.prim_func
        def kernel(A_data: TX.handle("float32"), n: TX.int32):
            TX.func_attr({"target": TX.target("metal"), "global_symbol": "kernel"})
            A = TX.decl_buffer(16, dtype="float32", data=A_data)
            v: TX.int32 = n + 1
            i = TX.launch_thread("threadIdx.x", v)
            A[i] = TX.float32(0)

    after = tilelang.transform.LowerDeviceKernelLaunch()(Before)

    packed_args = _call_packed_args(after["main"])
    assert str(packed_args[-1]) == "n + 1"
    assert "v" not in str(packed_args[-1])

    extent = after["kernel"].attrs["thread_extent"]["threadIdx.x"]
    assert str(extent) == "n + 1"


@tilelang.jit
def _fragment_epilogue_kernel():
    @T.prim_func
    def main(
        x: T.Tensor((8, 8), T.float16),
        y: T.Tensor((8, 8), T.float16),
        out: T.Tensor((8, 8), T.float16),
    ):
        with T.Kernel(1, threads=128):
            acc = T.alloc_fragment((8, 8), T.float32)
            for i, j in T.Parallel(8, 8):
                acc[i, j] = x[i, j].astype(T.float32) + y[i, j].astype(T.float32)
            T.copy(acc, out)

    return main


def test_fragment_buffer_elem_offset_does_not_escape_to_msl_or_api():
    artifact = _lower_metal_source(_fragment_epilogue_kernel.get_tir())

    assert "_elem_offset" not in artifact.kernel_source
    assert "_elem_offset" not in str(artifact.host_mod)
    assert "kernel void" in artifact.kernel_source


def _mamba_like_metal_scalar_kernel():
    @T.prim_func
    def main(
        x: T.Tensor((1024,), T.float32),
        y: T.Tensor((1024,), T.float32),
        n: T.int32,
    ):
        with T.Kernel(T.ceildiv(n, 128), threads=128):
            grid_tid = T.call_intrin("int32", "tir.metal.thread_position_in_grid_x")
            idx = (grid_tid * 17 + n) % 1024
            if grid_tid < n:
                y[grid_tid % 1024] = x[idx] + x[idx]

    return main


def _non_power_lane_decompose_kernel():
    @T.prim_func
    def main(
        x: T.Tensor((8192,), T.float32),
        y: T.Tensor((8192,), T.float32),
    ):
        with T.Kernel(32, threads=256):
            grid_tid = T.call_intrin("int32", "tir.metal.thread_position_in_grid_x")
            lane = grid_tid % 7168
            head = lane // 64
            if grid_tid < 7168:
                y[lane] = x[head] + x[lane]

    return main


def test_mamba_like_metal_source_has_grid_tid_and_scalar_cse():
    src = _lower_metal_source(_mamba_like_metal_scalar_kernel()).kernel_source

    assert "[[thread_position_in_grid]]" in src
    assert "gridThreadIdx" in src
    assert "_elem_offset" not in src

    kernel_body = src.split("kernel void", 1)[1]
    repeated_index_terms = re.findall(r"\(.*? \* 17.*? \+ .*?\)", kernel_body)
    assert len(repeated_index_terms) <= 1, src
    assert not re.search(r"\bcse_v\d+(?:_\d+)?\s*=\s*cse_v\d+(?:_\d+)?;", kernel_body), src
    assert not re.search(r"x\[\(cse_v\d+(?:_\d+)? & 1023\)\]", kernel_body), src
    assert re.search(
        r"x\[(?P<idx>cse_v\d+(?:_\d+)?)\] \+ x\[(?P=idx)\]",
        kernel_body,
    ), src


def test_metal_grid_position_keeps_builtin_in_hot_indices():
    src = _lower_metal_source(_mamba_like_metal_scalar_kernel()).kernel_source
    kernel_body = src.split("kernel void", 1)[1]

    assert re.search(r"\bint grid_tid = gridThreadIdx\.x;", kernel_body), src
    assert re.search(r"\bgridThreadIdx\.x < (?:arg\.)?n(?:\[0\])?", kernel_body), src
    assert re.search(r"y\[[^\]]*gridThreadIdx\.x[^\]]*\]", kernel_body), src


def test_grid_position_non_power_decompose_uses_unsigned_mod_div():
    src = _lower_metal_source(_non_power_lane_decompose_kernel()).kernel_source
    kernel_body = src.split("kernel void", 1)[1]

    assert "[[thread_position_in_grid]]" in src
    assert "gridThreadIdx" in kernel_body
    assert ">> 31" not in kernel_body
    assert ">>31" not in kernel_body
    assert not re.search(r"\b7168\s*&\s*\(", kernel_body), src


def test_metal_scalar_bind_canonicalizer_reuses_cse_address_binds():
    @I.ir_module
    class Before:
        @TX.prim_func
        def main(A: TX.Buffer(1024, "float32"), B: TX.Buffer(1024, "float32"), i: TX.int32):
            TX.func_attr({"target": TX.target("metal")})
            base: TX.int32 = i * 17 + 3
            idx0: TX.int32 = base % 1024  # noqa: F841
            idx1: TX.int32 = base % 1024
            A[i % 1024] = B[base % 1024] + B[idx1]

    after = tilelang.transform.BindMetalScalarIntrinsics()(Before)
    text = str(after["main"])

    assert "idx1" not in text
    assert "B[idx0] + B[idx0]" in text


def test_metal_index_normalizer_keeps_non_immediate_int_operands_same_dtype():
    A = tvm.tir.decl_buffer((4096,), "float32", name="A")
    B = tvm.tir.decl_buffer((4096,), "float32", name="B")
    i = tvm.tir.Var("i", "int32")
    stride64 = tvm.tir.Var("stride64", "int64")
    widened_i = tvm.tir.Cast("int64", i)
    extent = tvm.tir.IntImm("int64", 4096)

    exprs = [
        tvm.tir.Add(widened_i, stride64),
        tvm.tir.Sub(widened_i, stride64),
        tvm.tir.Mul(widened_i, stride64),
        tvm.tir.Div(widened_i, stride64),
        tvm.tir.Mod(widened_i, stride64),
        tvm.tir.FloorDiv(widened_i, stride64),
        tvm.tir.FloorMod(widened_i, stride64),
    ]
    stores = [
        tvm.tir.BufferStore(
            A,
            tvm.tir.BufferLoad(B, [tvm.tir.FloorMod(expr, extent)]),
            [tvm.tir.IntImm("int32", n)],
        )
        for n, expr in enumerate(exprs)
    ]
    func = tvm.tir.PrimFunc([A, B, i, stride64], tvm.tir.SeqStmt(stores)).with_attr("target", tvm.target.Target("metal"))
    before = tvm.IRModule({"main": func})

    after = tilelang.transform.BindMetalScalarIntrinsics()(before)

    binary_nodes = (
        tvm.tir.Add,
        tvm.tir.Sub,
        tvm.tir.Mul,
        tvm.tir.Div,
        tvm.tir.Mod,
        tvm.tir.FloorDiv,
        tvm.tir.FloorMod,
    )
    checked = []

    def is_scalar_int_dtype(dtype):
        dtype_name = str(dtype)
        return dtype.lanes == 1 and (dtype_name.startswith("int") or dtype_name.startswith("uint"))

    def visit(node):
        if isinstance(node, binary_nodes):
            lhs_dtype = node.a.dtype
            rhs_dtype = node.b.dtype
            if is_scalar_int_dtype(lhs_dtype) and is_scalar_int_dtype(rhs_dtype):
                checked.append(type(node).__name__)
                assert lhs_dtype == rhs_dtype, f"{node}: {lhs_dtype} != {rhs_dtype}"

    tvm.tir.stmt_functor.post_order_visit(after["main"].body, visit)
    assert {"Add", "Sub", "Mul", "Div", "Mod", "FloorDiv", "FloorMod"} <= set(checked)


if __name__ == "__main__":
    tilelang.testing.main()
