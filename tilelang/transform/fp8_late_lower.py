"""Late lowering for the T.fp8_scaled_matmul IR marker."""

from __future__ import annotations

from dataclasses import dataclass

from tvm import ir, tir
from tvm.tir.stmt_functor import ir_transform
from tvm.tir.transform import prim_func_pass
from tvm.tirx import Bind, SeqStmt

import tilelang.language as T
from tilelang.language.fp8_op import (
    FP8_SCALED_MATMUL_MARKER,
    _const_int_value,
    _dot4_intrinsics_registered,
    _fp8_dot4_auto_disabled,
    _target_thread_warp_size,
    _z3_prove_dot4_legal,
)


def _op_name(call: tir.Call) -> str:
    return str(getattr(call.op, "name", call.op))


def _string_value(value) -> str:
    if hasattr(value, "value"):
        return str(value.value)
    return str(value).strip('"')


def _int_value(value, default: int = 0) -> int:
    got = _const_int_value(value)
    return default if got is None else int(got)


def _bool_value(value) -> bool:
    return bool(_int_value(value))


def _is_marker_call(call: tir.Call) -> bool:
    if "call_extern" not in _op_name(call) or not call.args:
        return False
    return _string_value(call.args[0]) == FP8_SCALED_MATMUL_MARKER


def _decode_region(expr) -> tir.BufferRegion:
    if not isinstance(expr, tir.Call):
        raise TypeError(f"expected tl.region Call, got {type(expr)}")
    if _op_name(expr) != "tl.tileop.region":
        raise TypeError(f"expected tl.tileop.region, got {_op_name(expr)!r}")
    if len(expr.args) < 3 or not isinstance(expr.args[0], tir.BufferLoad):
        raise TypeError("malformed tl.region marker argument")
    load = expr.args[0]
    extents = list(expr.args[2:])
    if len(load.indices) != len(extents):
        raise ValueError("tl.region load rank does not match extent rank")
    ranges = [
        ir.Range.from_min_extent(index, extent)
        for index, extent in zip(load.indices, extents)
    ]
    return tir.BufferRegion(load.buffer, ranges)


def _region_shape(region: tir.BufferRegion) -> list[tir.PrimExpr]:
    return [r.extent for r in region.region]


def _region_min(region: tir.BufferRegion, axis: int):
    return region.region[axis].min


def _region_extent_int(region: tir.BufferRegion, axis: int) -> int:
    return _int_value(region.region[axis].extent, -1)


def _scale_load(scale: tir.BufferRegion, size_axis0: int, offset, idx):
    if size_axis0 == 1:
        return tir.BufferLoad(scale.buffer, [_region_min(scale, 0)])
    return tir.BufferLoad(scale.buffer, [_region_min(scale, 0) + offset + idx])


def _buffer_innermost_stride(buffer: tir.Buffer) -> int | None:
    strides = getattr(buffer, "strides", None)
    if not strides:
        return 1
    return _const_int_value(strides[-1])


def _linear_min_offset(region: tir.BufferRegion) -> int | None:
    buffer = region.buffer
    strides = getattr(buffer, "strides", None)
    if not strides:
        strides = []
        running = 1
        for extent in reversed(buffer.shape):
            strides.insert(0, running)
            got = _const_int_value(extent)
            if got is None:
                return None
            running *= got
    total = _const_int_value(getattr(buffer, "elem_offset", None) or 0)
    if total is None:
        return None
    for rng, stride in zip(region.region, strides):
        mn = _const_int_value(rng.min)
        st = _const_int_value(stride)
        if mn is None or st is None:
            return None
        total += mn * st
    return total


def _seq(stmts: list[tir.Stmt]) -> tir.Stmt:
    stmts = [stmt for stmt in stmts if stmt is not None]
    if not stmts:
        return tir.Evaluate(0)
    if len(stmts) == 1:
        return stmts[0]
    return SeqStmt(stmts)


def _for(var: tir.Var, extent, body: tir.Stmt, kind=tir.ForKind.SERIAL, annotations=None):
    return tir.For(var, 0, extent, kind, body, annotations=annotations)


def _if(cond, then: tir.Stmt) -> tir.Stmt:
    return tir.IfThenElse(cond, then, None)


@dataclass(frozen=True)
class _Marker:
    a: tir.BufferRegion
    a_scale: tir.BufferRegion
    b: tir.BufferRegion
    b_scale: tir.BufferRegion
    c: tir.BufferRegion
    transpose_b: bool
    a_scale_offset: tir.PrimExpr
    b_scale_offset: tir.PrimExpr
    c_row_offset: tir.PrimExpr
    c_col_offset: tir.PrimExpr
    simd_group_width: int
    outputs_per_block: int


def _decode_marker(call: tir.Call) -> _Marker:
    args = list(call.args[1:])
    if len(args) != 12:
        raise ValueError(f"{FP8_SCALED_MATMUL_MARKER} expects 12 args, got {len(args)}")
    return _Marker(
        a=_decode_region(args[0]),
        a_scale=_decode_region(args[1]),
        b=_decode_region(args[2]),
        b_scale=_decode_region(args[3]),
        c=_decode_region(args[4]),
        transpose_b=_bool_value(args[5]),
        a_scale_offset=args[6],
        b_scale_offset=args[7],
        c_row_offset=args[8],
        c_col_offset=args[9],
        simd_group_width=_int_value(args[10]),
        outputs_per_block=_int_value(args[11]),
    )


def _lower_scalar(marker: _Marker) -> tir.Stmt:
    a, b, c = marker.a, marker.b, marker.c
    m_extent = a.region[0].extent
    k_extent = a.region[1].extent
    full_n_extent = b.region[0].extent if marker.transpose_b else b.region[1].extent
    c_is_global = c.buffer.scope() == "global"
    n_extent = (
        tir.IntImm("int32", marker.outputs_per_block)
        if c_is_global and marker.outputs_per_block > 0
        else full_n_extent
    )
    sa_size = _region_extent_int(marker.a_scale, 0)
    sb_size = _region_extent_int(marker.b_scale, 0)

    i = tir.Var("i", "int32")
    j = tir.Var("j", "int32")
    k = tir.Var("k", "int32")
    base_var = tir.Var("base", "float32")
    a_row = marker.c_row_offset + i if a.buffer.scope() == "global" else i
    b_col = marker.c_col_offset + j if b.buffer.scope() == "global" else j
    c_row = marker.c_row_offset + i if c_is_global else i
    c_col = marker.c_col_offset + j if c_is_global else j
    c_idx = [_region_min(c, 0) + c_row, _region_min(c, 1) + c_col]
    c_load = tir.BufferLoad(c.buffer, c_idx)
    a_load = tir.Cast(
        "float32",
        tir.BufferLoad(a.buffer, [_region_min(a, 0) + a_row, _region_min(a, 1) + k]),
    )
    if marker.transpose_b:
        b_load_raw = tir.BufferLoad(
            b.buffer, [_region_min(b, 0) + b_col, _region_min(b, 1) + k]
        )
    else:
        b_load_raw = tir.BufferLoad(
            b.buffer, [_region_min(b, 0) + k, _region_min(b, 1) + b_col]
        )
    b_load = tir.Cast("float32", b_load_raw)
    dot_store = tir.BufferStore(c.buffer, c_load + a_load * b_load, c_idx)
    k_loop = _for(k, k_extent, dot_store, tir.ForKind.UNROLLED, {
        "pragma_unroll_explicit": False,
        "pragma_unroll_factor": 4,
    })
    sa = _scale_load(marker.a_scale, sa_size, marker.a_scale_offset, i)
    sb = _scale_load(marker.b_scale, sb_size, marker.b_scale_offset, j)
    scaled = tir.BufferStore(
        c.buffer,
        base_var + (c_load - base_var) * tir.Cast("float32", sa) * tir.Cast("float32", sb),
        c_idx,
    )
    cell_body = _seq([Bind(base_var, c_load), k_loop, scaled])
    if c_is_global:
        cell_body = _if(c_col < full_n_extent, cell_body)
    return _for(
        i,
        m_extent,
        _for(j, n_extent, cell_body, tir.ForKind.PARALLEL),
        tir.ForKind.PARALLEL,
    )


def _dot4_legal(marker: _Marker) -> bool:
    if not marker.transpose_b or _region_extent_int(marker.a, 0) != 1:
        return False
    if not str(marker.a.buffer.dtype).startswith("float8_e4m3"):
        return False
    if not str(marker.b.buffer.dtype).startswith("float8_e4m3"):
        return False
    k_a = _region_extent_int(marker.a, 1)
    k_b = _region_extent_int(marker.b, 1)
    if k_a <= 0 or k_b <= 0:
        return False
    a_stride = _buffer_innermost_stride(marker.a.buffer)
    b_stride = _buffer_innermost_stride(marker.b.buffer)
    a_offset = _linear_min_offset(marker.a)
    b_offset = _linear_min_offset(marker.b)
    if None in (a_stride, b_stride, a_offset, b_offset):
        return False
    proved, _ = _z3_prove_dot4_legal(k_a, k_b, a_stride, b_stride, a_offset, b_offset)
    return proved


def _access_ptr(buffer: tir.Buffer, indices: list[tir.PrimExpr], extent) -> tir.PrimExpr:
    return T.access_ptr(tir.BufferLoad(buffer, indices), "r", extent=extent)


def _lower_m1_dot4(marker: _Marker, target) -> tir.Stmt:
    a, b, c = marker.a, marker.b, marker.c
    k_extent = a.region[1].extent
    n_extent = b.region[0].extent
    sgw = marker.simd_group_width or _target_thread_warp_size(target)
    outputs = marker.outputs_per_block or _region_extent_int(b, 0)
    sa_size = _region_extent_int(marker.a_scale, 0)
    sb_size = _region_extent_int(marker.b_scale, 0)

    kk = tir.Var("kk", "int32")
    word_i = tir.Var("word_i", "int32")
    grid_tid_var = tir.Var("grid_tid", "int32")
    simd_lane_var = tir.Var("simd_lane", "int32")
    col_var = tir.Var("col", "int32")
    store_lane = tir.Var("fp8_store_lane", "int32")
    base_var = tir.Var("base", "float32")
    reduced_var = tir.Var("reduced", "float32")
    sa_var = tir.Var("sa", "float32")
    sb_var = tir.Var("sb", "float32")
    dot = tir.decl_buffer((1,), "float32", name="dot", scope="local.var")
    grid_tid = tir.call_intrin("int32", "tir.metal.thread_position_in_grid_x")
    simd_lane = tir.call_intrin("int32", "tir.metal.thread_index_in_simdgroup")
    col = col_var
    k_words = k_extent // tir.IntImm("int32", 4)
    dot_load = tir.BufferLoad(dot, [0])
    a_ptr = _access_ptr(a.buffer, [_region_min(a, 0), _region_min(a, 1)], k_extent)
    b_ptr = _access_ptr(b.buffer, [_region_min(b, 0) + col, _region_min(b, 1)], k_extent)
    dot4 = tir.call_intrin(
        "float32",
        "tir.metal.fp8_e4m3_dot4",
        a_ptr,
        b_ptr,
        word_i,
        word_i,
    )
    dot_acc = tir.BufferStore(dot, dot_load + dot4, [0])
    if _int_value(k_words, -1) % sgw != 0:
        dot_acc = _if(word_i < k_words, dot_acc)
    kk_body = SeqStmt([Bind(word_i, kk * tir.IntImm("int32", sgw) + simd_lane_var), dot_acc])
    kk_loop = _for(kk, tir.ceildiv(k_words, tir.IntImm("int32", sgw)), kk_body, tir.ForKind.UNROLLED, {
        "pragma_unroll_explicit": False,
        "pragma_unroll_factor": 4,
    })
    reduced = tir.call_intrin("float32", "tir.metal.simd_sum", dot_load)
    local_col = col - marker.c_col_offset
    c_idx = [
        _region_min(c, 0) + marker.c_row_offset,
        _region_min(c, 1) + col,
    ]
    base = tir.BufferLoad(c.buffer, c_idx)
    sa = _scale_load(marker.a_scale, sa_size, marker.a_scale_offset, tir.IntImm("int32", 0))
    sb = _scale_load(marker.b_scale, sb_size, marker.b_scale_offset, local_col)
    store = tir.BufferStore(c.buffer, base_var + reduced_var * sa_var * sb_var, c_idx)
    body = _if(
        col < n_extent,
        _seq([
            tir.BufferStore(dot, tir.FloatImm("float32", 0.0), [0]),
            Bind(base_var, base),
            kk_loop,
            Bind(reduced_var, reduced),
            Bind(sa_var, tir.Cast("float32", sa)),
            Bind(sb_var, tir.Cast("float32", sb)),
            _if(
                simd_lane_var == tir.IntImm("int32", 0),
                _for(store_lane, tir.IntImm("int32", 1), store, tir.ForKind.PARALLEL),
            ),
        ]),
    )
    if outputs > 0:
        body = _if(
            tir.all(
                local_col >= tir.IntImm("int32", 0),
                local_col < tir.IntImm("int32", outputs),
            ),
            body,
        )
    body = _seq([
        Bind(grid_tid_var, grid_tid),
        Bind(simd_lane_var, simd_lane),
        Bind(col_var, grid_tid_var // tir.IntImm("int32", sgw)),
        body,
    ])
    return tir.Allocate(dot.data, "float32", [1], tir.IntImm("bool", 1), body)


def _lower_marker(marker: _Marker, target) -> tir.Stmt:
    if (
        target is not None
        and target.kind.name == "metal"
        and marker.c.buffer.scope() == "global"
        and not _fp8_dot4_auto_disabled()
        and _dot4_intrinsics_registered()
        and _dot4_legal(marker)
    ):
        return _lower_m1_dot4(marker, target)
    return _lower_scalar(marker)


def Fp8ScaledMatmulLateLower(target=None):
    def pass_fn(func: tir.PrimFunc, mod, ctx):
        active_target = target
        if active_target is None and func.attrs:
            active_target = func.attrs.get("target", None)

        def post_visit(stmt):
            if (
                isinstance(stmt, tir.Evaluate)
                and isinstance(stmt.value, tir.Call)
                and _is_marker_call(stmt.value)
            ):
                return _lower_marker(_decode_marker(stmt.value), active_target)
            return stmt

        return func.with_body(ir_transform(func.body, None, post_visit))

    return prim_func_pass(pass_fn, opt_level=0, name="tl.Fp8ScaledMatmulLateLower")
