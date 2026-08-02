import tilelang
from tilelang import tvm


def test_metal_index_normalizer_drops_sparse_mla_flat_long_casts():
    kv = tvm.tir.decl_buffer((1024,), "float16", name="kv")
    reduce_buf = tvm.tir.decl_buffer((32,), "float32", name="reduce_buf")
    out = tvm.tir.decl_buffer((1,), "float32", name="out")
    kv_row_base = tvm.tir.Var("kv_row_base", "int32")
    d = tvm.tir.Var("d", "int32")
    tid = tvm.tir.Var("tid", "int32")
    stride = tvm.tir.Var("stride", "int32")

    kv_idx = tvm.tir.Cast("int64", kv_row_base) + tvm.tir.Cast("int64", d)
    reduce_idx = tvm.tir.Cast("int64", tid) + tvm.tir.Cast("int64", stride)
    body = tvm.tir.SeqStmt(
        [
            tvm.tir.BufferStore(
                out,
                tvm.tir.Cast("float32", tvm.tir.BufferLoad(kv, [kv_idx])),
                [0],
            ),
            tvm.tir.BufferStore(out, tvm.tir.BufferLoad(reduce_buf, [reduce_idx]), [0]),
        ]
    )
    func = tvm.tir.PrimFunc([kv, reduce_buf, out, kv_row_base, d, tid, stride], body).with_attr("target", tvm.target.Target("metal"))
    before = tvm.IRModule({"main": func})

    after = tilelang.transform.BindMetalScalarIntrinsics()(before)
    text = str(after["main"])

    assert 'T.Cast("int64", kv_row_base) + T.Cast("int64", d)' not in text
    assert 'T.Cast("int64", tid) + T.Cast("int64", stride)' not in text
    assert "kv_row_base + d" in text
    assert "tid + stride" in text


def test_metal_scalar_binder_covers_threadgroup_tid():
    src = tvm.tir.decl_buffer((1024,), "float32", name="src")
    out = tvm.tir.decl_buffer((1,), "float32", name="out")
    tid0 = tvm.tir.call_intrin("int32", "tir.metal.thread_position_in_threadgroup_x")
    tid1 = tvm.tir.call_intrin("int32", "tir.metal.thread_position_in_threadgroup_x")
    body = tvm.tir.BufferStore(
        out,
        tvm.tir.BufferLoad(src, [tid0]) + tvm.tir.BufferLoad(src, [tid1]),
        [0],
    )
    func = tvm.tir.PrimFunc([src, out], body).with_attr("target", tvm.target.Target("metal"))

    after = tilelang.transform.BindMetalScalarIntrinsics()(tvm.IRModule({"main": func}))
    text = str(after["main"])

    assert text.count("thread_position_in_threadgroup_x") == 1
    assert "threadgroup_tid" in text


if __name__ == "__main__":
    tilelang.testing.main()
