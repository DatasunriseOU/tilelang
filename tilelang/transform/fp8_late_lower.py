"""Late lowering for the T.fp8_scaled_matmul IR marker."""

from __future__ import annotations

from dataclasses import dataclass

from tvm import ir, tir
from tvm.tir.stmt_functor import ir_transform, post_order_visit
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


def _cast_for_store(buffer: tir.Buffer, value: tir.PrimExpr) -> tir.PrimExpr:
    dtype = str(buffer.dtype)
    if str(value.dtype) == dtype:
        return value
    return tir.Cast(dtype, value)


def _load_as_float32(buffer: tir.Buffer, indices: list[tir.PrimExpr]) -> tir.PrimExpr:
    value = tir.BufferLoad(buffer, indices)
    if str(value.dtype) == "float32":
        return value
    return tir.Cast("float32", value)


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
    accumulate: bool


def _decode_marker(call: tir.Call) -> _Marker:
    args = list(call.args[1:])
    if len(args) not in (12, 13):
        raise ValueError(
            f"{FP8_SCALED_MATMUL_MARKER} expects 12 or 13 args, got {len(args)}"
        )
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
        accumulate=True if len(args) == 12 else _bool_value(args[12]),
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
    c_load = _load_as_float32(c.buffer, c_idx)
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
    if not marker.accumulate:
        dot = tir.decl_buffer((1,), "float32", name="dot", scope="local.var")
        dot_load = tir.BufferLoad(dot, [0])
        dot_store = tir.BufferStore(dot, dot_load + a_load * b_load, [0])
        k_loop = _for(k, k_extent, dot_store, tir.ForKind.UNROLLED, {
            "pragma_unroll_explicit": False,
            "pragma_unroll_factor": 4,
        })
        sa = _scale_load(marker.a_scale, sa_size, marker.a_scale_offset, i)
        sb = _scale_load(marker.b_scale, sb_size, marker.b_scale_offset, j)
        scaled = tir.BufferStore(
            c.buffer,
            _cast_for_store(
                c.buffer,
                dot_load * tir.Cast("float32", sa) * tir.Cast("float32", sb),
            ),
            c_idx,
        )
        cell_body = _seq([
            tir.BufferStore(dot, tir.FloatImm("float32", 0.0), [0]),
            k_loop,
            scaled,
        ])
        if c_is_global:
            cell_body = _if(c_col < full_n_extent, cell_body)
        body = _for(
            i,
            m_extent,
            _for(j, n_extent, cell_body, tir.ForKind.SERIAL),
            tir.ForKind.SERIAL,
        )
        return tir.Allocate(dot.data, "float32", [1], tir.IntImm("bool", 1), body)
    dot_store = tir.BufferStore(
        c.buffer,
        _cast_for_store(c.buffer, c_load + a_load * b_load),
        c_idx,
    )
    k_loop = _for(k, k_extent, dot_store, tir.ForKind.UNROLLED, {
        "pragma_unroll_explicit": False,
        "pragma_unroll_factor": 4,
    })
    sa = _scale_load(marker.a_scale, sa_size, marker.a_scale_offset, i)
    sb = _scale_load(marker.b_scale, sb_size, marker.b_scale_offset, j)
    scaled = tir.BufferStore(
        c.buffer,
        _cast_for_store(
            c.buffer,
            base_var + (c_load - base_var) * tir.Cast("float32", sa) * tir.Cast("float32", sb),
        ),
        c_idx,
    )
    cell_body = _seq([Bind(base_var, c_load), k_loop, scaled])
    if c_is_global:
        cell_body = _if(c_col < full_n_extent, cell_body)
    # This fallback emits scalar FP8 dequant/accumulate code.  Marking the
    # output loops as PARALLEL lets the generic vectorizer form int32x16 ramp
    # indices, which Metal cannot print as a native vector type.  Keep the
    # generic fallback serial; the explicit M=1 dot4 path below owns SIMD
    # parallelism for the optimized Metal route.
    return _for(
        i,
        m_extent,
        _for(j, n_extent, cell_body, tir.ForKind.SERIAL),
        tir.ForKind.SERIAL,
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
    # The Metal late-lowered dot4 helper expands to LUT-decoded fp32
    # multiply-adds in codegen_metal.cc. It does not use an int24 accumulator,
    # so K>520 is legal here as long as the packed layout/alignment proof holds.
    proved, _ = _z3_prove_dot4_legal(
        k_a,
        k_b,
        a_stride,
        b_stride,
        a_offset,
        b_offset,
        require_int24_safe=False,
    )
    return proved


def _trans_b_dot4_legal(marker: _Marker) -> bool:
    if not marker.transpose_b:
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
    # The Metal late-lowered dot4 helper expands to LUT-decoded fp32
    # multiply-adds in codegen_metal.cc. It does not use an int24 accumulator,
    # so K>520 is legal here as long as the packed layout/alignment proof holds.
    proved, _ = _z3_prove_dot4_legal(
        k_a,
        k_b,
        a_stride,
        b_stride,
        a_offset,
        b_offset,
        require_int24_safe=False,
    )
    return proved


def _access_ptr(buffer: tir.Buffer, indices: list[tir.PrimExpr], extent) -> tir.PrimExpr:
    return T.access_ptr(tir.BufferLoad(buffer, indices), "r", extent=extent)


def _lower_m1_dot4(
    marker: _Marker,
    target,
    thread_vars: dict[str, tuple[tir.Var, int]] | None = None,
) -> tir.Stmt:
    a, b, c = marker.a, marker.b, marker.c
    k_extent = a.region[1].extent
    n_extent = b.region[0].extent
    sgw = marker.simd_group_width or _target_thread_warp_size(target)
    outputs = marker.outputs_per_block or _region_extent_int(b, 0)
    sa_size = _region_extent_int(marker.a_scale, 0)
    sb_size = _region_extent_int(marker.b_scale, 0)

    kk = tir.Var("kk", "int32")
    word_i = tir.Var("word_i", "int32")
    col_var = tir.Var("col", "int32")
    local_col_var = tir.Var("local_col", "int32")
    a_ptr_var = tir.Var("A_fp8_ptr", "handle")
    b_ptr_var = tir.Var("B_fp8_ptr", "handle")
    a_word_var = tir.Var("a_word", "uint32")
    b_word_var = tir.Var("b_word", "uint32")
    base_var = tir.Var("base", "float32")
    reduced_var = tir.Var("reduced", "float32")
    sa_var = tir.Var("sa", "float32")
    sb_var = tir.Var("sb", "float32")
    dot = tir.decl_buffer((1,), "float32", name="dot", scope="local.var")
    direct_col_buf = tir.decl_buffer((1,), "int32", name="row", scope="local.var")
    simd_lane = tir.call_intrin("int32", "tir.metal.thread_index_in_simdgroup")
    tx_info = thread_vars.get("threadIdx.x") if thread_vars is not None else None
    n_int = _region_extent_int(b, 0)
    direct_grid_tile = (
        tx_info is not None
        and a.buffer.scope() == "global"
        and b.buffer.scope() == "global"
        and c.buffer.scope() == "global"
        and n_int > 0
        and _int_value(marker.c_row_offset, 0) == 0
    )
    single_output_tile = (
        tx_info is not None
        and outputs == 1
        and tx_info[1] <= sgw
        and n_int > 0
    )
    col = col_var
    local_col = local_col_var
    b_col = tir.BufferLoad(direct_col_buf, [0]) if direct_grid_tile else col

    k_words = k_extent // tir.IntImm("int32", 4)
    dot_load = tir.BufferLoad(dot, [0])
    a_ptr = _access_ptr(a.buffer, [_region_min(a, 0), _region_min(a, 1)], k_extent)
    b_ptr = _access_ptr(b.buffer, [_region_min(b, 0) + b_col, _region_min(b, 1)], k_extent)
    dot4 = tir.call_intrin(
        "float32",
        "tir.metal.fp8_e4m3_dot4_words",
        a_word_var,
        b_word_var,
    )
    dot_acc = tir.BufferStore(dot, dot_load + dot4, [0])
    load_a_word = tir.call_intrin(
        "uint32",
        "tir.metal.fp8_load_u32",
        a_ptr_var,
        word_i,
    )
    load_b_word = tir.call_intrin(
        "uint32",
        "tir.metal.fp8_load_u32",
        b_ptr_var,
        word_i,
    )
    dot4_body = SeqStmt([
        Bind(a_word_var, load_a_word),
        Bind(b_word_var, load_b_word),
        dot_acc,
    ])
    if _int_value(k_words, -1) % sgw != 0:
        dot4_body = _if(word_i < k_words, dot4_body)
    kk_body = SeqStmt([
        Bind(word_i, kk * tir.IntImm("int32", sgw) + simd_lane),
        dot4_body,
    ])
    kk_loop = _for(
        kk,
        tir.ceildiv(k_words, tir.IntImm("int32", sgw)),
        kk_body,
        tir.ForKind.UNROLLED,
        {
            "pragma_unroll_explicit": False,
            "pragma_unroll_factor": 4,
        },
    )
    reduced = tir.call_intrin("float32", "tir.metal.simd_sum", dot_load)
    c_idx = [
        _region_min(c, 0) + marker.c_row_offset,
        _region_min(c, 1) + col,
    ]
    base = _load_as_float32(c.buffer, c_idx)
    sa = _scale_load(marker.a_scale, sa_size, marker.a_scale_offset, tir.IntImm("int32", 0))
    if direct_grid_tile:
        sb_offset = tir.IntImm("int32", 0)
        sb_index = col
    elif single_output_tile:
        sb_offset = tir.IntImm("int32", 0)
        sb_index = col
    else:
        sb_offset = marker.b_scale_offset
        sb_index = local_col
    sb = _scale_load(
        marker.b_scale,
        sb_size,
        sb_offset,
        sb_index,
    )
    scaled_value = reduced_var * sa_var * sb_var
    store_value = base_var + scaled_value if marker.accumulate else scaled_value
    store = tir.BufferStore(c.buffer, _cast_for_store(c.buffer, store_value), c_idx)
    exact_output_tile = False
    if tx_info is not None and outputs > 0 and n_int > 0:
        _tx, tx_extent = tx_info
        exact_output_tile = (n_int % outputs == 0) and (tx_extent <= outputs * sgw)
    lane0_store_stmts = []
    if marker.accumulate:
        lane0_store_stmts.append(Bind(base_var, base))
    lane0_store_stmts.extend([
        Bind(sa_var, tir.Cast("float32", sa)),
        Bind(sb_var, tir.Cast("float32", sb)),
        store,
    ])
    lane0_store = _seq(lane0_store_stmts)
    body = _seq([
        kk_loop,
        Bind(reduced_var, reduced),
        _if(simd_lane == tir.IntImm("int32", 0), lane0_store),
    ])
    if not exact_output_tile:
        body = _if(col < n_extent, body)
    add_outputs_guard = outputs > 0
    if direct_grid_tile:
        add_outputs_guard = False
    if add_outputs_guard and tx_info is not None:
        _tx, tx_extent = tx_info
        if tx_extent <= outputs * sgw:
            add_outputs_guard = False
    if add_outputs_guard:
        body = _if(
            tir.all(
                local_col >= tir.IntImm("int32", 0),
                local_col < tir.IntImm("int32", outputs),
            ),
            body,
        )
    grid_tid_bind = None
    direct_col_store = None
    if tx_info is not None:
        local_tid = tir.call_intrin("int32", "tir.metal.thread_position_in_threadgroup_x")
        if direct_grid_tile:
            # Global owner-output vecmat matches Path B's execution model:
            # one Metal SIMDgroup owns one output row and the global thread
            # id determines that row.  Do not rebuild the row from
            # blockIdx/threadIdx; doing so leaves repeated block-local index
            # math in the dot4 hot loop and prevents the Metal compiler from
            # seeing the simple row/word pattern.
            grid_tid_var = tir.Var("grid_tid", "int32")
            grid_tid = tir.call_intrin("int32", "tir.metal.thread_position_in_grid_x")
            grid_tid_bind = Bind(grid_tid_var, grid_tid)
            direct_col_store = tir.BufferStore(
                direct_col_buf,
                grid_tid_var // tir.IntImm("int32", sgw),
                [0],
            )
            col_expr = grid_tid_var // tir.IntImm("int32", sgw)
            local_col_expr = None
        elif single_output_tile:
            # One SIMDgroup owns exactly one output column.  Make the scheduler
            # fact explicit so later bound-check and CSE passes do not preserve
            # a redundant `(threadIdx.x >> 5)` term in the hot loop.
            local_col_expr = tir.IntImm("int32", 0)
            col_expr = marker.c_col_offset
        else:
            local_col_expr = local_tid // tir.IntImm("int32", sgw)
            col_expr = marker.c_col_offset + local_col_var
    else:
        grid_tid_var = tir.Var("grid_tid", "int32")
        grid_tid = tir.call_intrin("int32", "tir.metal.thread_position_in_grid_x")
        grid_tid_bind = Bind(grid_tid_var, grid_tid)
        col_expr = grid_tid_var // tir.IntImm("int32", sgw)
        local_col_expr = col_var - marker.c_col_offset
    prefix = [grid_tid_bind]
    if direct_grid_tile:
        prefix.extend([direct_col_store, Bind(col_var, col_expr)])
    else:
        prefix.extend([
            Bind(local_col_var, local_col_expr),
            Bind(col_var, col_expr),
        ])
    prefix.extend([
        Bind(a_ptr_var, a_ptr),
        Bind(b_ptr_var, b_ptr),
    ])
    body = _seq(prefix + [body])
    body = tir.Allocate(dot.data, "float32", [1], tir.IntImm("bool", 1), body)
    if direct_grid_tile:
        body = tir.Allocate(
            direct_col_buf.data,
            "int32",
            [1],
            tir.IntImm("bool", 1),
            body,
        )
    return body


def _lower_trans_b_direct_dot4(
    marker: _Marker,
    thread_vars: dict[str, tuple[tir.Var, int]],
) -> tir.Stmt:
    """Lower direct global matmul with row-major B[N, K] to packed dot4.

    This path covers the full-matmul version of the existing M=1 direct path:
    one Metal thread owns one output cell, and B must already be K-contiguous
    by output column/row.  It deliberately does not transpose or copy B.
    """
    a, b, c = marker.a, marker.b, marker.c
    tx, tx_extent = thread_vars["threadIdx.x"]
    ty_info = thread_vars.get("threadIdx.y")
    ty = ty_info[0] if ty_info is not None else tir.IntImm("int32", 0)
    ty_extent = ty_info[1] if ty_info is not None else 1
    k_extent = a.region[1].extent
    m_extent = a.region[0].extent
    full_n_extent = b.region[0].extent
    outputs = marker.outputs_per_block or _region_extent_int(b, 0)
    if outputs <= 0:
        outputs = _region_extent_int(c, 1)
    if outputs <= 0:
        outputs = tx_extent
    outputs_imm = tir.IntImm("int32", outputs)
    sa_size = _region_extent_int(marker.a_scale, 0)
    sb_size = _region_extent_int(marker.b_scale, 0)

    word_i = tir.Var("word_i", "int32")
    dot = tir.decl_buffer((1,), "float32", name="dot", scope="local.var")
    dot_load = tir.BufferLoad(dot, [0])

    if ty_extent > 1:
        local_row = ty
        local_col = tx
    else:
        local_row = tx // outputs_imm
        local_col = tx % outputs_imm

    a_row = marker.c_row_offset + local_row if a.buffer.scope() == "global" else local_row
    b_col = marker.c_col_offset + local_col if b.buffer.scope() == "global" else local_col
    c_row = marker.c_row_offset + local_row if c.buffer.scope() == "global" else local_row
    c_col = marker.c_col_offset + local_col if c.buffer.scope() == "global" else local_col

    a_ptr = _access_ptr(a.buffer, [_region_min(a, 0) + a_row, _region_min(a, 1)], k_extent)
    b_ptr = _access_ptr(b.buffer, [_region_min(b, 0) + b_col, _region_min(b, 1)], k_extent)
    dot4 = tir.call_intrin(
        "float32",
        "tir.metal.fp8_e4m3_dot4",
        a_ptr,
        b_ptr,
        word_i,
        word_i,
    )
    k_words = k_extent // tir.IntImm("int32", 4)
    dot_loop = _for(
        word_i,
        k_words,
        tir.BufferStore(dot, dot_load + dot4, [0]),
        tir.ForKind.UNROLLED,
        {
            "pragma_unroll_explicit": False,
            "pragma_unroll_factor": 4,
        },
    )
    c_idx = [_region_min(c, 0) + c_row, _region_min(c, 1) + c_col]
    sa = _scale_load(marker.a_scale, sa_size, marker.a_scale_offset, local_row)
    sb = _scale_load(marker.b_scale, sb_size, marker.b_scale_offset, local_col)
    scaled_value = dot_load * tir.Cast("float32", sa) * tir.Cast("float32", sb)
    store_value = (
        _load_as_float32(c.buffer, c_idx) + scaled_value
        if marker.accumulate
        else scaled_value
    )
    store = tir.BufferStore(
        c.buffer,
        _cast_for_store(c.buffer, store_value),
        c_idx,
    )
    body = _seq([
        tir.BufferStore(dot, tir.FloatImm("float32", 0.0), [0]),
        dot_loop,
        store,
    ])
    body = _if(
        tir.all(
            marker.c_row_offset + local_row < m_extent,
            local_col < outputs_imm,
            marker.c_col_offset + local_col < full_n_extent,
        ),
        body,
    )
    return tir.Allocate(dot.data, "float32", [1], tir.IntImm("bool", 1), body)


def _lower_m1_simd_scalar(marker: _Marker, target) -> tir.Stmt:
    """Lower M=1 vecmat to one SIMD reduction per output column.

    This is the safe row-vector fast path for layouts where B is not
    K-contiguous by output row.  The packed dot4 path requires
    ``transpose_B=True`` so B[j, k:k+4] can be read as one uint32.  For the
    normal ``B[K, N]`` shared tile, B[k:k+4, j] is strided, so we keep scalar
    FP8 decode but distribute the K loop across one Metal simdgroup and reduce
    with ``simd_sum``.  The local-fragment consumer copies column ``j`` from
    thread ``j`` for the M=1 layout, so assign columns to the matching
    producer simdgroup.  This keeps the value in the owning thread's private
    fragment slot without adding shared memory, barriers, or redundant
    cross-simdgroup work.
    """
    a, b, c = marker.a, marker.b, marker.c
    k_extent = a.region[1].extent
    full_n_extent = b.region[0].extent if marker.transpose_b else b.region[1].extent
    n_extent = full_n_extent
    sgw = marker.simd_group_width or _target_thread_warp_size(target)
    sa_size = _region_extent_int(marker.a_scale, 0)
    sb_size = _region_extent_int(marker.b_scale, 0)

    j_outer = tir.Var("j_outer", "int32")
    kk = tir.Var("kk", "int32")
    k_var = tir.Var("k", "int32")
    compute_tid = tir.Var("fp8_compute_tid", "int32")
    a_val_var = tir.Var("a_val", "float32")
    sa_var = tir.Var("sa", "float32")
    cols_per_group_i = 4
    cols_per_group = tir.IntImm("int32", cols_per_group_i)
    chunks_per_simdgroup = tir.IntImm("int32", max(1, sgw // cols_per_group_i))
    dot_buffers = [
        tir.decl_buffer((1,), "float32", name=f"dot{idx}", scope="local.var")
        for idx in range(cols_per_group_i)
    ]

    local_tid = tir.call_intrin("int32", "tir.metal.thread_position_in_threadgroup_x")
    simd_lane = tir.call_intrin("int32", "tir.metal.thread_index_in_simdgroup")
    a_row = marker.c_row_offset if a.buffer.scope() == "global" else tir.IntImm("int32", 0)
    c_row = marker.c_row_offset if c.buffer.scope() == "global" else tir.IntImm("int32", 0)
    a_load = tir.Cast(
        "float32",
        tir.BufferLoad(a.buffer, [_region_min(a, 0) + a_row, _region_min(a, 1) + k_var]),
    )
    n_int = _int_value(n_extent, -1)

    def col_expr(r_idx: int):
        return j_outer * cols_per_group + tir.IntImm("int32", r_idx)

    def b_col_expr(r_idx: int):
        col = col_expr(r_idx)
        return marker.c_col_offset + col if b.buffer.scope() == "global" else col

    def c_idx_expr(r_idx: int):
        col = col_expr(r_idx)
        c_col = marker.c_col_offset + col if c.buffer.scope() == "global" else col
        return [_region_min(c, 0) + c_row, _region_min(c, 1) + c_col]

    dot_acc_stmts: list[tir.Stmt] = []
    for r_idx, dot_buf in enumerate(dot_buffers):
        b_col = b_col_expr(r_idx)
        if marker.transpose_b:
            b_raw = tir.BufferLoad(
                b.buffer, [_region_min(b, 0) + b_col, _region_min(b, 1) + k_var]
            )
        else:
            b_raw = tir.BufferLoad(
                b.buffer, [_region_min(b, 0) + k_var, _region_min(b, 1) + b_col]
            )
        dot_load = tir.BufferLoad(dot_buf, [0])
        dot_acc = tir.BufferStore(
            dot_buf,
            dot_load + a_val_var * tir.Cast("float32", b_raw),
            [0],
        )
        if n_int <= 0 or n_int % cols_per_group_i != 0:
            dot_acc = _if(col_expr(r_idx) < n_extent, dot_acc)
        dot_acc_stmts.append(dot_acc)
    dot_acc_body = _seq(dot_acc_stmts)
    if _int_value(k_extent, -1) % sgw != 0:
        dot_acc_body = _if(k_var < k_extent, dot_acc_body)

    kk_body = _seq([
        Bind(k_var, kk * tir.IntImm("int32", sgw) + simd_lane),
        Bind(a_val_var, a_load),
        dot_acc_body,
    ])
    kk_loop = _for(
        kk,
        tir.ceildiv(k_extent, tir.IntImm("int32", sgw)),
        kk_body,
        tir.ForKind.UNROLLED,
        {
            "pragma_unroll_explicit": False,
            "pragma_unroll_factor": 4,
        },
    )

    sa = _scale_load(marker.a_scale, sa_size, marker.a_scale_offset, tir.IntImm("int32", 0))
    init_stmts = [
        tir.BufferStore(dot_buf, tir.FloatImm("float32", 0.0), [0])
        for dot_buf in dot_buffers
    ]
    store_stmts: list[tir.Stmt] = []
    for r_idx, dot_buf in enumerate(dot_buffers):
        base_var = tir.Var(f"base{r_idx}", "float32")
        reduced_var = tir.Var(f"reduced{r_idx}", "float32")
        sb_var = tir.Var(f"sb{r_idx}", "float32")
        idx = c_idx_expr(r_idx)
        sb = _scale_load(marker.b_scale, sb_size, marker.b_scale_offset, col_expr(r_idx))
        scaled_value = reduced_var * sa_var * sb_var
        store_value = base_var + scaled_value if marker.accumulate else scaled_value
        store = tir.BufferStore(
            c.buffer,
            _cast_for_store(c.buffer, store_value),
            idx,
        )
        if c.buffer.scope() == "global":
            store = _if(simd_lane == tir.IntImm("int32", 0), store)
        lane_store_stmts = []
        if marker.accumulate:
            lane_store_stmts.append(Bind(base_var, _load_as_float32(c.buffer, idx)))
        lane_store_stmts.extend([
            Bind(reduced_var, tir.call_intrin(
                "float32", "tir.metal.simd_sum", tir.BufferLoad(dot_buf, [0])
            )),
            Bind(sb_var, tir.Cast("float32", sb)),
            store,
        ])
        store_body = _seq(lane_store_stmts)
        if n_int <= 0 or n_int % cols_per_group_i != 0:
            store_body = _if(col_expr(r_idx) < n_extent, store_body)
        store_stmts.append(store_body)
    j_body = _seq([
        _seq(init_stmts),
        kk_loop,
        Bind(sa_var, tir.Cast("float32", sa)),
        _seq(store_stmts),
    ])
    j_body = _if(
        (j_outer // chunks_per_simdgroup) == (compute_tid // tir.IntImm("int32", sgw)),
        j_body,
    )
    body = _for(
        j_outer,
        tir.ceildiv(n_extent, cols_per_group),
        j_body,
        tir.ForKind.SERIAL,
    )
    body = _seq([Bind(compute_tid, local_tid), body])
    for dot_buf in reversed(dot_buffers):
        body = tir.Allocate(dot_buf.data, "float32", [1], tir.IntImm("bool", 1), body)
    return body


def _m1_simd_scalar_legal(marker: _Marker) -> bool:
    if _region_extent_int(marker.a, 0) != 1:
        return False
    if marker.c.buffer.scope() == "global":
        return False
    if not str(marker.a.buffer.dtype).startswith("float8_"):
        return False
    if not str(marker.b.buffer.dtype).startswith("float8_"):
        return False
    return _region_extent_int(marker.a, 1) > 0


def _lower_marker(
    marker: _Marker,
    target,
    thread_vars: dict[str, tuple[tir.Var, int]],
) -> tir.Stmt:
    if (
        target is not None
        and target.kind.name == "metal"
        and marker.c.buffer.scope() == "global"
        and not _fp8_dot4_auto_disabled()
        and _dot4_intrinsics_registered()
        and _dot4_legal(marker)
    ):
        return _lower_m1_dot4(marker, target, thread_vars)
    if (
        target is not None
        and target.kind.name == "metal"
        and marker.c.buffer.scope() == "global"
        and "threadIdx.x" in thread_vars
        and not _fp8_dot4_auto_disabled()
        and _dot4_intrinsics_registered()
        and _trans_b_dot4_legal(marker)
    ):
        return _lower_trans_b_direct_dot4(marker, thread_vars)
    if (
        target is not None
        and target.kind.name == "metal"
        and _m1_simd_scalar_legal(marker)
    ):
        return _lower_m1_simd_scalar(marker, target)
    return _lower_scalar(marker)


def _collect_thread_vars(body: tir.Stmt) -> dict[str, tuple[tir.Var, int]]:
    thread_vars: dict[str, tuple[tir.Var, int]] = {}

    def visit(stmt):
        if not isinstance(stmt, tir.AttrStmt) or stmt.attr_key != "thread_extent":
            return
        iter_var = stmt.node
        tag = getattr(iter_var, "thread_tag", None)
        var = getattr(iter_var, "var", None)
        if tag is None or var is None:
            return
        thread_vars[str(tag)] = (var, _int_value(stmt.value, 1))

    post_order_visit(body, visit)
    return thread_vars


def Fp8ScaledMatmulLateLower(target=None):
    def pass_fn(func: tir.PrimFunc, mod, ctx):
        active_target = target
        if active_target is None and func.attrs:
            active_target = func.attrs.get("target", None)
        thread_vars = _collect_thread_vars(func.body)

        def post_visit(stmt):
            if (
                isinstance(stmt, tir.Evaluate)
                and isinstance(stmt.value, tir.Call)
                and _is_marker_call(stmt.value)
            ):
                return _lower_marker(_decode_marker(stmt.value), active_target, thread_vars)
            return stmt

        return func.with_body(ir_transform(func.body, None, post_visit))

    return prim_func_pass(pass_fn, opt_level=0, name="tl.Fp8ScaledMatmulLateLower")
