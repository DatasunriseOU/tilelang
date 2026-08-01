import tilelang
from tilelang import tvm


def test_vectorize_bind_remaps_var_and_its_uses():
    a = tvm.tirx.decl_buffer((4,), "float32", name="A")
    b = tvm.tirx.decl_buffer((4,), "float32", name="B")
    i = tvm.tirx.Var("i", "int32")
    x = tvm.tirx.Var("x", "float32")
    alpha = tvm.tirx.Var("alpha", "float32")

    body = tvm.tirx.SeqStmt(
        [
            tvm.tirx.Bind(x, a[i] + (alpha - alpha)),
            tvm.tirx.BufferStore(b, x, [i]),
        ]
    )
    loop = tvm.tirx.For(i, 0, 4, tvm.tirx.ForKind.VECTORIZED, body)
    func = tvm.tirx.PrimFunc(
        [a.data, b.data, alpha],
        loop,
        buffer_map={a.data: a, b.data: b},
    ).with_attr("global_symbol", "main")
    mod = tvm.IRModule({"main": func})

    vectorized = tilelang.transform.VectorizeLoop()(mod)
    vectorized_body = vectorized["main"].body
    assert isinstance(vectorized_body, tvm.tirx.SeqStmt)
    bind, store = vectorized_body.seq
    assert isinstance(bind, tvm.tirx.Bind)
    assert isinstance(store, tvm.tirx.BufferStore)
    assert str(bind.var.dtype) == "float32x4"
    assert bind.value.dtype == bind.var.dtype
    assert store.value.same_as(bind.var)

    tilelang.transform.Simplify()(vectorized)


if __name__ == "__main__":
    tvm.testing.main()
