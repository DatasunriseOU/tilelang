from __future__ import annotations

import pytest

tvm = pytest.importorskip("tvm")

from poc.triton_frontend.op_emitters.control import (
    CONTROL_EMITTERS,
    map_arith_extsi,
    map_arith_index_cast,
    map_tensor_collapse_shape,
    map_scf_for,
    map_arith_truncf,
    map_tt_func,
)
from poc.triton_frontend.op_mapping import LazyTileExpr, WalkerCtx, _alloc_tile_buffer

from ._fixtures import FakeMlirOp, FakeSSA


def test_vector_int_cast_lowers_per_lane() -> None:
    ctx = WalkerCtx()
    src = FakeSSA(name="src", shape=(4,), dtype="i32")
    dst = FakeSSA(name="dst", shape=(4,), dtype="i64")
    ctx.bind(src, tvm.tir.Ramp(tvm.tir.const(0, "int32"), tvm.tir.const(1, "int32"), 4))

    out = map_arith_extsi(FakeMlirOp("arith.extsi", [src], [dst]), ctx)

    assert isinstance(out, LazyTileExpr)
    assert out.dtype == "int64"
    assert ctx.get(dst) is out
    lane0 = out.read_lane(ctx, (tvm.tir.const(0, "int32"),))
    assert str(lane0.dtype) == "int64"


def test_buffer_float_cast_lowers_per_lane() -> None:
    ctx = WalkerCtx()
    src = FakeSSA(name="src", shape=(4,), dtype="f32")
    dst = FakeSSA(name="dst", shape=(4,), dtype="f16")
    buf = tvm.tir.decl_buffer((4,), "float32", name="src_buf")
    ctx.bind(src, buf)

    out = map_arith_truncf(FakeMlirOp("arith.truncf", [src], [dst]), ctx)

    assert isinstance(out, LazyTileExpr)
    assert out.dtype == "float16"
    assert ctx.get(dst) is out
    lane0 = out.read_lane(ctx, (tvm.tir.const(0, "int32"),))
    assert str(lane0.dtype) == "float16"


def test_index_cast_routes_as_integer_cast() -> None:
    ctx = WalkerCtx()
    src = FakeSSA(name="src", dtype="index")
    dst = FakeSSA(name="dst", dtype="i32")
    ctx.bind(src, tvm.tir.Var("idx", "int64"))

    out = map_arith_index_cast(FakeMlirOp("arith.index_cast", [src], [dst]), ctx)

    assert str(out.dtype) == "int32"
    assert ctx.get(dst) is out


def test_tensor_collapse_shape_reshapes_lazy_tile() -> None:
    ctx = WalkerCtx()
    src = FakeSSA(name="src", shape=(1, 4), dtype="i32")
    dst = FakeSSA(name="dst", shape=(4,), dtype="i32")
    source = LazyTileExpr(
        (1, 4),
        "int32",
        lambda _read_ctx, indices: indices[0] * 10 + indices[1],
        name="source",
    )
    ctx.bind(src, source)

    out = map_tensor_collapse_shape(
        FakeMlirOp("tensor.collapse_shape", [src], [dst]),
        ctx,
    )

    assert isinstance(out, LazyTileExpr)
    assert out.shape == (4,)
    assert ctx.get(dst) is out
    lane2 = out.read_lane(ctx, (tvm.tir.const(2, "int32"),))
    assert "2" in str(lane2)


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


def test_scf_for_materializes_lazy_buffer_carry_yield() -> None:
    """Loop-carried LazyTileExpr values must not leak induction vars after scf.for."""
    ctx = WalkerCtx()
    init = FakeSSA(name="init", shape=(4,), dtype="f32")
    result = FakeSSA(name="result", shape=(4,), dtype="f32")
    induction = FakeSSA(name="iv", dtype="index")
    block_carry = FakeSSA(name="block_carry", shape=(4,), dtype="f32")
    lazy_ssa = FakeSSA(name="lazy", shape=(4,), dtype="f32")
    carry = _alloc_tile_buffer(ctx, [4], "float32", "carry", scope="local")
    ctx.bind(init, carry)

    def emit_lazy(inner, child):
        iv = child.get(induction)
        lazy = LazyTileExpr(
            (4,),
            "float32",
            lambda read_ctx, indices: read_ctx.tir().Cast(
                "float32",
                iv + indices[0],
            ),
            name="lazy_with_iv",
        )
        child.bind(lazy_ssa, lazy)
        return lazy

    CONTROL_EMITTERS["test.lazy_yield"] = emit_lazy
    try:
        map_scf_for(
            {
                "name": "scf.for",
                "operands": [0, 2, 1, init],
                "results": [result],
                "regions": [
                    {
                        "block_args": [induction, block_carry],
                        "ops": [
                            {"name": "test.lazy_yield", "operands": [], "results": [lazy_ssa]},
                            {"name": "scf.yield", "operands": [lazy_ssa], "results": []},
                        ],
                    }
                ],
            },
            ctx,
        )
    finally:
        CONTROL_EMITTERS.pop("test.lazy_yield", None)

    assert ctx.get(result) is carry
    assert "carry_copy" in str(ctx.stmts[-1])


def test_scf_for_materializes_lazy_initial_tile_carry() -> None:
    """Tile iter_args initialized from LazyTileExpr need mutable carry storage."""
    ctx = WalkerCtx()
    init = FakeSSA(name="init", shape=(4,), dtype="f32")
    result = FakeSSA(name="result", shape=(4,), dtype="f32")
    induction = FakeSSA(name="iv", dtype="index")
    block_carry = FakeSSA(name="block_carry", shape=(4,), dtype="f32")
    zero = LazyTileExpr(
        (4,),
        "float32",
        lambda read_ctx, _indices: read_ctx.tir().const(0.0, "float32"),
        name="zero_tile",
    )
    ctx.bind(init, zero)

    map_scf_for(
        {
            "name": "scf.for",
            "operands": [0, 1, 1, init],
            "results": [result],
            "regions": [
                {
                    "block_args": [induction, block_carry],
                    "ops": [
                        {"name": "scf.yield", "operands": [block_carry], "results": []},
                    ],
                }
            ],
        },
        ctx,
    )

    out = ctx.get(result)
    assert isinstance(out, tvm.tir.Buffer)
    assert out is not zero


def test_scf_for_materializes_lazy_pointer_index_carry() -> None:
    """Loop-carried pointer descriptors need mutable index storage."""
    ctx = WalkerCtx()
    base = tvm.tir.decl_buffer((64,), "float32", name="A")
    init = FakeSSA(name="init_ptr", shape=(4,), dtype="f32")
    result = FakeSSA(name="result_ptr", shape=(4,), dtype="f32")
    induction = FakeSSA(name="iv", dtype="index")
    block_carry = FakeSSA(name="block_ptr", shape=(4,), dtype="f32")
    index = LazyTileExpr(
        (4,),
        "int32",
        lambda read_ctx, indices: indices[0],
        name="lazy_index",
    )
    ctx.bind(init, (base, [index]))

    map_scf_for(
        {
            "name": "scf.for",
            "operands": [0, 1, 1, init],
            "results": [result],
            "regions": [
                {
                    "block_args": [induction, block_carry],
                    "ops": [
                        {"name": "scf.yield", "operands": [block_carry], "results": []},
                    ],
                }
            ],
        },
        ctx,
    )

    out = ctx.get(result)
    assert isinstance(out, tuple)
    assert isinstance(out[1][0], tvm.tir.Buffer)
    assert out[1][0] is not index


def test_scf_for_snapshots_multiple_tile_carry_yields() -> None:
    """Multiple tile carries must commit after all yielded values are evaluated."""
    ctx = WalkerCtx()
    init_a = FakeSSA(name="init_a", shape=(4,), dtype="f32")
    init_b = FakeSSA(name="init_b", shape=(4,), dtype="f32")
    result_a = FakeSSA(name="result_a", shape=(4,), dtype="f32")
    result_b = FakeSSA(name="result_b", shape=(4,), dtype="f32")
    induction = FakeSSA(name="iv", dtype="index")
    block_a = FakeSSA(name="block_a", shape=(4,), dtype="f32")
    block_b = FakeSSA(name="block_b", shape=(4,), dtype="f32")
    next_a = FakeSSA(name="next_a", shape=(4,), dtype="f32")
    next_b = FakeSSA(name="next_b", shape=(4,), dtype="f32")

    ctx.bind(
        init_a,
        LazyTileExpr(
            (4,),
            "float32",
            lambda read_ctx, _indices: read_ctx.tir().const(0.0, "float32"),
            name="init_a_lazy",
        ),
    )
    ctx.bind(
        init_b,
        LazyTileExpr(
            (4,),
            "float32",
            lambda read_ctx, _indices: read_ctx.tir().const(10.0, "float32"),
            name="init_b_lazy",
        ),
    )

    def emit_nexts(_inner, child):
        tir = child.tir()
        a_buf = child.get(block_a)
        b_buf = child.get(block_b)
        child.bind(
            next_a,
            LazyTileExpr(
                (4,),
                "float32",
                lambda _read_ctx, indices: tir.BufferLoad(a_buf, list(indices))
                + tir.const(1.0, "float32"),
                name="a_next",
            ),
        )
        child.bind(
            next_b,
            LazyTileExpr(
                (4,),
                "float32",
                lambda _read_ctx, indices: tir.BufferLoad(b_buf, list(indices))
                + tir.BufferLoad(a_buf, list(indices)),
                name="b_next_reads_old_a",
            ),
        )
        return None

    CONTROL_EMITTERS["test.next_carries"] = emit_nexts
    try:
        map_scf_for(
            {
                "name": "scf.for",
                "operands": [0, 2, 1, init_a, init_b],
                "results": [result_a, result_b],
                "regions": [
                    {
                        "block_args": [induction, block_a, block_b],
                        "ops": [
                            {"name": "test.next_carries", "operands": [], "results": []},
                            {"name": "scf.yield", "operands": [next_a, next_b], "results": []},
                        ],
                    }
                ],
            },
            ctx,
        )
    finally:
        CONTROL_EMITTERS.pop("test.next_carries", None)

    names = [str(getattr(buf, "name", "")) for buf in ctx.local_buffers]
    assert any(name.startswith("carry_next") for name in names)
